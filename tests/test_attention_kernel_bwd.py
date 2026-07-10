"""0.2.0 T7 -- flash-attention BACKWARD D-preprocess Metal kernel (spec Section 4.2.2).

`D_i = sum_d dO_i,d * O_i,d`, the flash-attention paper's row-correction term for `dS`.
Small and independently tested on purpose: a wrong D breaks EVERY downstream gradient
(dQ/dK/dV, T8/T9) while forward parity still passes -- it is the "silent all-grads-wrong"
bug site, so it gets its own parity proof AND its own can-fail perturbation proof before
anything is built on top of it.

Mixed-file convention (`test_attention_kernel_fwd.py`'s lanes, same as
`test_kernel_guard.py`): pure source-templating tests stay in the DEFAULT lane (no GPU);
every test that launches the kernel carries a PER-TEST `@pytest.mark.metal`.

This file grows through T8 (dQ) and T9 (dK/dV) -- see the task brief.

T8 adds the dQ one-owner kernel (spec Section 4.2.3): one program owns dQ[i], loops the
causally-allowed keys (kk <= row), recomputes S/P from Q, K and the saved L, forms
`dS = P*(dP - D)` (D from T7's `launch_bwd_D`), and accumulates `dQ_i += scale*dS@K` in
fp32 registers -- written once. Parity is the autodiff dQ of `math_attention` w.r.t. q,
via `mx.vjp` with a seeded random cotangent (the exact oracle -- no readout projection).
The causal-skip inequality is the named bug site: a `flip_causal` build (kk >= row) gets
its own can-fail perturbation proof.
"""
import itertools
import math

import mlx.core as mx
import pytest

from mlx_train_perf.attention import api
from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.kernel.launch import (
    _bwd_dkv_macs_per_row,
    _bwd_dq_macs_per_row,
    launch_bwd_D,
    launch_bwd_dkv,
    launch_bwd_dq,
    plan_dkv_dispatches,
)
from mlx_train_perf.attention.kernel.source import (
    build_bwd_D_source,
    build_bwd_dkv_mma_source,
    build_bwd_dkv_source,
    build_bwd_dq_mma_source,
    build_bwd_dq_source,
)
from mlx_train_perf.attention.reference import flash_attention_reference, math_attention
from mlx_train_perf.errors import AttentionInputError, LaunchBudgetError

# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_bwd_D_source_substitutes_head_dim() -> None:  # noqa: N802 -- D is the paper's name
    for hd in (64, 96, 128):
        s = build_bwd_D_source(hd)
        assert f"i < {hd}" in s
        assert "HEAD_DIM" not in s  # every sentinel substituted (lossless)


def test_build_bwd_D_source_rejects_bad_head_dim() -> None:  # noqa: N802
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_bwd_D_source(hd)


def test_build_bwd_D_source_default_keeps_the_elementwise_product() -> None:  # noqa: N802
    s = build_bwd_D_source(64)
    assert "(float)o[base + i]" in s
    assert "PROD_FACTOR" not in s  # sentinel substituted


def test_build_bwd_D_source_drop_product_perturbation_drops_o() -> None:  # noqa: N802
    """Test-only perturbation arg: replaces the elementwise product's second factor with
    a constant 1.0f, so the generated body computes rowsum(dO) instead of rowsum(dO*O).
    Never used by production code -- see `launch_bwd_D`'s TEST-ONLY `_drop_product`."""
    s = build_bwd_D_source(64, drop_product=True)
    assert "(float)o[base + i]" not in s
    assert "1.0f" in s


# ---------------------------------------------------------------------------------------
# Shape validation (DEFAULT lane -- raised before any Metal kernel is built/dispatched,
# same convention as test_attention_api.py::test_validate_shapes_rejects_*).
# ---------------------------------------------------------------------------------------


def test_launch_bwd_D_rejects_non_4d_dO() -> None:  # noqa: N802
    d_o = mx.random.normal((4, 16, 32))
    o = mx.random.normal((1, 4, 16, 32))
    mx.eval(d_o, o)
    with pytest.raises(AttentionInputError, match="4-D"):
        launch_bwd_D(d_o, o)


def test_launch_bwd_D_rejects_non_4d_O() -> None:  # noqa: N802
    d_o = mx.random.normal((1, 4, 16, 32))
    o = mx.random.normal((16, 32))
    mx.eval(d_o, o)
    with pytest.raises(AttentionInputError, match="4-D"):
        launch_bwd_D(d_o, o)


def test_launch_bwd_D_rejects_shape_mismatch() -> None:  # noqa: N802
    d_o = mx.random.normal((1, 4, 16, 32))
    o = mx.random.normal((1, 4, 16, 64))
    mx.eval(d_o, o)
    with pytest.raises(AttentionInputError, match="shape"):
        launch_bwd_D(d_o, o)


def test_launch_bwd_D_rejects_batch_rank_mismatch() -> None:  # noqa: N802
    d_o = mx.random.normal((2, 4, 16, 32))
    o = mx.random.normal((1, 4, 16, 32))
    mx.eval(d_o, o)
    with pytest.raises(AttentionInputError, match="shape"):
        launch_bwd_D(d_o, o)


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal).
# ---------------------------------------------------------------------------------------

# Measured worsts (mlx 0.32.0, M1 Max, seed=30, whole grid below: batch {1,2} x
# head_dim {64,128} x n {61,257} x dtype {fp32,bf16}). D always outputs fp32 regardless
# of input dtype (never cast down, matching L's convention in the forward kernel) -- both
# dO and O upcast to fp32 in-kernel BEFORE multiplying, exactly like the reference's
# `.astype(mx.float32)`, so bf16 rounding is common-mode to both sides (not doubled) and
# the only error source, for either dtype, is fp32 reduction-order noise between the
# kernel's 32-lane simd_sum and the reference's single `.sum(axis=-1)`.
# fp32 worst 7.62939453125e-06, bf16 worst 3.814697265625e-06 -- pinned at ~2.5x margin
# over the measured worst, same measure-first convention as the forward kernel's pins.
_TOL_D = {mx.float32: 2e-5, mx.bfloat16: 1e-5}


def _rand_do_o(
    *, b: int, hq: int, n: int, d: int, dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    d_o = mx.random.normal((b, hq, n, d)).astype(dtype)
    o = mx.random.normal((b, hq, n, d)).astype(dtype)
    mx.eval(d_o, o)
    return d_o, o


def _reference_D(d_o: mx.array, o: mx.array) -> mx.array:  # noqa: N802 -- D is the paper's name
    return (d_o.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)


@pytest.mark.metal
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("n", [61, 257])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_D_matches_rowsum(  # noqa: N802
    n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    hq = 4
    d_o, o = _rand_do_o(b=batch, hq=hq, n=n, d=head_dim, dtype=dtype, seed=30)

    d_kernel = launch_bwd_D(d_o, o)
    d_ref = _reference_D(d_o, o)
    mx.eval(d_kernel, d_ref)

    diff = mx.abs(d_kernel - d_ref).max().item()
    print(
        f"[D {['fp32', 'bf16'][dtype == mx.bfloat16]} b{batch} n{n} d{head_dim}] "
        f"diff={diff:.6e}"
    )
    assert diff < _TOL_D[dtype], f"D vs rowsum(dO*O) diff {diff}"


@pytest.mark.metal
def test_D_bitwise_deterministic_across_runs() -> None:  # noqa: N802
    """No atomics (each (b, hq, row) triple's D is written by exactly one simdgroup, no
    cross-thread contention) -> bit-identical D across repeated runs. Lock it (mirrors
    test_attention_kernel_fwd.py::test_fwd_bitwise_deterministic_across_runs: 1 baseline
    + 4 repeats = 5 runs total)."""
    d_o, o = _rand_do_o(b=2, hq=4, n=129, d=64, dtype=mx.float32, seed=31)
    d0 = launch_bwd_D(d_o, o)
    mx.eval(d0)
    for _ in range(4):
        d = launch_bwd_D(d_o, o)
        mx.eval(d)
        assert mx.array_equal(d, d0).item()


@pytest.mark.metal
def test_D_drop_product_perturbation_fails_parity() -> None:  # noqa: N802
    """Deliberate perturbation: build the D kernel with the elementwise product dropped
    (computes rowsum(dO) instead of rowsum(dO*O)). Its output must DIVERGE from the
    correct rowsum -- if this ever matched, the parity test above could not detect a real
    D bug, and D is the site where a wrong value silently breaks every downstream gradient
    while forward parity still passes (mirrors
    test_attention_kernel_fwd.py::test_fwd_wrong_mask_perturbation_fails_parity)."""
    d_o, o = _rand_do_o(b=1, hq=4, n=32, d=64, dtype=mx.float32, seed=32)

    d_wrong = launch_bwd_D(d_o, o, _drop_product=True)
    d_ref = _reference_D(d_o, o)
    mx.eval(d_wrong, d_ref)

    diff = mx.abs(d_wrong - d_ref).max().item()
    assert diff > 1e-2, (
        f"drop-product kernel matched the correct rowsum (diff={diff:.3e}) -- "
        "the parity suite cannot detect a D bug"
    )


# =======================================================================================
# T8 -- dQ one-owner backward kernel (spec Section 4.2.3).
# =======================================================================================

# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_bwd_dq_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_bwd_dq_source(hd, causal=True)
        assert f"float qreg[{hd}];" in s      # q row registers, HEAD_DIM wide
        assert f"float doreg[{hd}];" in s      # dO row registers
        assert f"float dq[{hd}];" in s          # fp32 dQ accumulator
        assert f"dd < {hd}" in s
        assert "HEAD_DIM" not in s              # every sentinel substituted (lossless)


def test_build_bwd_dq_source_causal_keep_comparison() -> None:
    s = build_bwd_dq_source(64, causal=True)
    assert "kk <= row" in s
    assert "kk >= row" not in s
    assert "KEEP_CMP" not in s


def test_build_bwd_dq_source_noncausal_keeps_all_keys() -> None:
    s = build_bwd_dq_source(64, causal=False)
    assert "kk <= row" not in s
    assert "kk >= row" not in s
    assert "bool keep = (true);" in s


def test_build_bwd_dq_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-triangle arm: the causal-skip inequality is flipped so the parity
    # run FAILS -- the named bug site of T8 gets its own can-fail proof.
    s = build_bwd_dq_source(64, causal=True, flip_causal=True)
    assert "kk >= row" in s
    assert "kk <= row" not in s


def test_build_bwd_dq_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_bwd_dq_source(hd, causal=True)


def test_build_bwd_dq_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_bwd_dq_source(64, causal=False, flip_causal=True)


# ---------------------------------------------------------------------------------------
# Shape/dtype validation (DEFAULT lane -- raised before any Metal kernel is built).
# ---------------------------------------------------------------------------------------


def _valid_bwd_dq_inputs(
    *, b: int = 1, hq: int = 4, hkv: int = 2, n: int = 16, d: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    q = mx.random.normal((b, hq, n, d)).astype(dtype)
    k = mx.random.normal((b, hkv, n, d)).astype(dtype)
    v = mx.random.normal((b, hkv, n, d)).astype(dtype)
    d_o = mx.random.normal((b, hq, n, d)).astype(dtype)
    lse = mx.random.normal((b, hq, n))
    d_arr = mx.random.normal((b, hq, n))
    mx.eval(q, k, v, d_o, lse, d_arr)
    return q, k, v, d_o, lse, d_arr


def test_launch_bwd_dq_rejects_non_4d_q() -> None:
    q, k, v, d_o, lse, d_arr = _valid_bwd_dq_inputs()
    with pytest.raises(AttentionInputError, match="4-D"):
        launch_bwd_dq(q[0], k, v, d_o, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dq_rejects_dO_shape_mismatch() -> None:  # noqa: N802
    q, k, v, _d_o, lse, d_arr = _valid_bwd_dq_inputs(d=64)
    bad_do = mx.random.normal((1, 4, 16, 128))  # wrong head_dim vs q
    mx.eval(bad_do)
    with pytest.raises(AttentionInputError, match="dO"):
        launch_bwd_dq(q, k, v, bad_do, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dq_rejects_lse_rank() -> None:
    q, k, v, d_o, _lse, d_arr = _valid_bwd_dq_inputs()
    bad_lse = mx.random.normal((4, 16))  # 2-D, not (B, Hq, N)
    mx.eval(bad_lse)
    with pytest.raises(AttentionInputError, match="3-D"):
        launch_bwd_dq(q, k, v, d_o, bad_lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dq_rejects_dtype_mismatch() -> None:
    q, _k, v, d_o, lse, d_arr = _valid_bwd_dq_inputs(dtype=mx.float32)
    k_bf16 = mx.random.normal((1, 2, 16, 64)).astype(mx.bfloat16)
    mx.eval(k_bf16)
    with pytest.raises(AttentionInputError, match="dtype"):
        launch_bwd_dq(q, k_bf16, v, d_o, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dq_rejects_non_fp32_lse() -> None:
    """L and D seed the backward and are FIXED fp32 device buffers the kernel reads untemplated
    (matching the forward's fp32-L convention) -- a bf16 lse/D would be read as raw fp32 bytes
    and silently corrupt every gradient. Closes T8's review completeness nit (the dQ validator
    did not assert this)."""
    q, k, v, d_o, _lse, d_arr = _valid_bwd_dq_inputs()
    bad_lse = mx.random.normal((1, 4, 16)).astype(mx.bfloat16)
    mx.eval(bad_lse)
    with pytest.raises(AttentionInputError, match="fp32"):
        launch_bwd_dq(q, k, v, d_o, bad_lse, d_arr, scale=0.1, causal=True)


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal).
# ---------------------------------------------------------------------------------------

# Measured worsts (mlx 0.32.0, M1 Max, seed=41, over the whole grid below: batch {1,2} x
# head_dim {64,128} x head-config {4/4, 4/2, 32/8@n64} x n {61,257,64} x dtype {fp32,bf16}).
#
# The dQ kernel accumulates fp32 in-register for both input dtypes (QK dot, exp, dO.V dot,
# dS, and the scale*dS*k accumulate all fp32), so an fp32 diff vs the autodiff oracle is
# pure reduction-order noise. For bf16 the kernel reads bf16 q/k/v/dO and writes bf16 dQ,
# and its D is derived from the bf16-rounded forward O (`flash_attention_reference` casts O
# down, exactly as a real flash backward reloads a bf16-saved O) -- so the bf16 worst carries
# that single common-mode rounding, not a doubled one.
#
# MEASURED WORSTS over the whole grid (mlx 0.32.0, M1 Max, seed=41): fp32 2.464958e-06,
# bf16 1.562500e-02. The bf16 worst is exactly one bf16 ULP (2^-6) at a dQ magnitude near
# 2-4 -- quantized rounding, not accumulation drift. Pins: fp32 5e-6 (~2.0x the measured
# worst, same measure-first convention as the D-kernel / forward fp32 pins); bf16 3e-2
# (~1.9x, bounded by 2 bf16 ULP = 3.125e-2 -- the same ULP-aware bound as the forward's bf16
# O pin: a future case landing between the pin and 2 ULP is not a regression, widen toward
# 2 ULP with a note, never past it).
_TOL_DQ = {mx.float32: 5e-6, mx.bfloat16: 3e-2}

# head-config x N cases; the flagship (32/8) pattern only at N=64 to bound cost -- mirrors
# test_attention_kernel_fwd.py::_HEAD_N_CASES exactly.
_DQ_HEAD_N_CASES = [
    (4, 4, 64), (4, 4, 61), (4, 4, 257),   # MHA
    (4, 2, 64), (4, 2, 61), (4, 2, 257),   # GQA
    (32, 8, 64),                            # flagship group_size-4 pattern
]


def _rand_qkv_do(
    *, b: int, hq: int, hkv: int, n: int, d: int, dtype: mx.Dtype, seed: int = 41
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, n, d)).astype(dtype)
    k = mx.random.normal((b, hkv, n, d)).astype(dtype)
    v = mx.random.normal((b, hkv, n, d)).astype(dtype)
    cot = mx.random.normal((b, hq, n, d)).astype(dtype)  # dO cotangent (w.r.t. O)
    mx.eval(q, k, v, cot)
    return q, k, v, cot


def _dq_oracle(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool
) -> mx.array:
    """Exact dQ oracle: the vector-Jacobian product of `math_attention` w.r.t. q ONLY,
    with the same random cotangent `cot` the kernel path consumes as dO. No readout
    projection -- `mx.vjp` gives the exact autodiff dQ."""
    _, vjps = mx.vjp(
        lambda q_: math_attention(q_, k, v, scale=scale, causal=causal), [q], [cot]
    )
    return vjps[0]


def _dq_kernel(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    rate_macs_per_s: float | None = None, flip_causal: bool = False,
    variant: str = "scalar", d_slab: int | None = None,
) -> mx.array:
    """The kernel dQ path: forward reference gives (O, L); T7's `launch_bwd_D` gives D from
    (dO, O); `launch_bwd_dq` consumes q/k/v/dO/L/D. `variant`/`d_slab` default to the scalar
    body (unchanged for every existing caller); `variant="mma"` selects the T9b rung-B1 body."""
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=causal)
    d_arr = launch_bwd_D(cot, o)
    return launch_bwd_dq(
        q, k, v, cot, lse, d_arr, scale=scale, causal=causal,
        rate_macs_per_s=rate_macs_per_s, _flip_causal=flip_causal,
        variant=variant, d_slab=d_slab,
    )


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), _DQ_HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dq_matches_autodiff_oracle(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=True)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    print(
        f"[dQ {['fp32', 'bf16'][dtype == mx.bfloat16]} b{batch} {hq}/{hkv} n{n} "
        f"d{head_dim}] diff={diff:.6e}"
    )
    assert diff < _TOL_DQ[dtype], f"dQ vs autodiff oracle diff {diff}"


@pytest.mark.metal
def test_dq_bitwise_deterministic_across_runs() -> None:
    """One owner per query row (no atomics, no cross-thread accumulation) -> bit-identical
    dQ across repeated runs. Lock it (1 baseline + 4 repeats = 5 runs, mirrors
    test_D_bitwise_deterministic_across_runs)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=42)
    dq0 = _dq_kernel(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq0)
    for _ in range(4):
        dq = _dq_kernel(q, k, v, cot, scale=scale, causal=True)
        mx.eval(dq)
        assert mx.array_equal(dq, dq0).item()


@pytest.mark.metal
def test_dq_causal_skip_perturbation_fails_parity() -> None:
    """The named bug site: build the dQ kernel with the causal-skip inequality flipped to
    the WRONG triangle (kk >= row). Its dQ must DIVERGE from the causal autodiff oracle --
    if a flipped inequality ever matched, the parity grid could not detect an off-by-one in
    the causal skip (mirrors test_fwd_wrong_mask_perturbation_fails_parity)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=16, d=64, dtype=mx.float32, seed=43)

    dq_wrong = _dq_kernel(q, k, v, cot, scale=scale, causal=True, flip_causal=True)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_wrong, dq_ref)

    diff = mx.abs(dq_wrong.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    assert diff > 1e-2, (
        f"flipped causal-skip kernel matched the causal oracle (diff={diff:.3e}) -- "
        "the parity grid cannot detect a causal off-by-one"
    )


@pytest.mark.metal
def test_dq_split_matches_single_dispatch() -> None:
    """Query-range multi-dispatch writes DISJOINT dQ rows (no chaining -- each row's dQ
    depends only on its own absolute position); the reassembled result must be bit-identical
    to a single dispatch. Run at batch>1 and an N that is not a block multiple (mirrors
    test_fwd_split_matches_single_dispatch)."""
    b, hq, hkv, n, d = 2, 4, 2, 257, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=44)

    single = _dq_kernel(q, k, v, cot, scale=scale, causal=True)  # rate=None -> single dispatch
    # Force ~80 rows/dispatch (4 disjoint dispatches over n=257) via a low rate, keeping the
    # projected per-dispatch inside MAX_DISPATCH_SECONDS AND the total inside MAX_TOTAL_SECONDS.
    per_row = _bwd_dq_macs_per_row(n=n, d=d, b=b, hq=hq)
    split = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True, rate_macs_per_s=per_row * 160.0
    )
    mx.eval(single, split)
    assert mx.array_equal(single, split).item()


# =======================================================================================
# T9 -- dK/dV split-partials backward kernel with CHAINED accumulators (spec Section 4.2.4).
#
# Owner = one thread per (batch, kv_head, key) -- the v1 analogue of T8's per-query-row dQ
# owner. Each DISPATCH covers a bounded query-block range [q_lo, q_hi); the fp32 dK/dV
# accumulators are threaded across chained dispatches EXACTLY like the CE forward chains
# lse/tgt (full buffers + in-kernel offsets, `dk_in`/`dk_out`, `dv_in`/`dv_out`, seeded FROM
# the incoming partial, cast to k/v dtype once after the last dispatch). Inside a dispatch the
# owner loops ALL query heads of its GQA group x the range's causally-allowed queries
# (query row >= key row), query-row OUTER / q-head INNER / ascending -- so a range split
# reproduces the single-dispatch accumulation order bit-for-bit.
#
# The math mirrors api.py's pure-MLX `_flash_attention_backward` dK/dV path exactly,
# specialized to one owner key j:
#   s   = scale * (q_i . k_j)                        (recomputed QK^T, causal-masked)
#   p   = exp(s - L_i)                                (L_i is the forward's saved row logsumexp)
#   dp  = dO_i . v_j                                  (the dP = dO @ V^T term, per query)
#   ds  = p * (dp - D_i)                              (D_i from T7's launch_bwd_D)
#   dV_j += p * dO_i ; dK_j += scale * ds * q_i       (accumulated over the group x queries)
# =======================================================================================

# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_bwd_dkv_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_bwd_dkv_source(hd, causal=True)
        assert f"float kreg[{hd}];" in s       # owner key registers, HEAD_DIM wide
        assert f"float vreg[{hd}];" in s        # owner value registers
        assert f"float dk[{hd}];" in s          # fp32 dK accumulator
        assert f"float dv[{hd}];" in s          # fp32 dV accumulator
        assert f"dd < {hd}" in s
        assert "HEAD_DIM" not in s              # every sentinel substituted (lossless)


def test_build_bwd_dkv_source_causal_keep_comparison() -> None:
    # The owner is the KEY and the loop is over query rows i, so the causal keep is i >= key
    # (query row at or below the diagonal), the swapped-roles analogue of dQ's kk <= row.
    s = build_bwd_dkv_source(64, causal=True)
    assert "i >= key" in s
    assert "i <= key" not in s
    assert "KEEP_CMP" not in s


def test_build_bwd_dkv_source_noncausal_keeps_all_queries() -> None:
    s = build_bwd_dkv_source(64, causal=False)
    assert "i >= key" not in s
    assert "i <= key" not in s
    assert "bool keep = (true);" in s


def test_build_bwd_dkv_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-triangle arm: the causal-keep inequality is flipped so a parity run
    # against the causal oracle FAILS -- the named-bug-site perturbation for dK/dV.
    s = build_bwd_dkv_source(64, causal=True, flip_causal=True)
    assert "i <= key" in s
    assert "i >= key" not in s


def test_build_bwd_dkv_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_bwd_dkv_source(hd, causal=True)


def test_build_bwd_dkv_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_bwd_dkv_source(64, causal=False, flip_causal=True)


# ---------------------------------------------------------------------------------------
# Shape/dtype validation (DEFAULT lane -- raised before any Metal kernel is built; the dK/dV
# launcher shares the dQ boundary contract (q/k/v/dO 4-D, lse/D 3-D fp32, GQA divisibility)).
# ---------------------------------------------------------------------------------------


def test_launch_bwd_dkv_rejects_non_4d_q() -> None:
    q, k, v, d_o, lse, d_arr = _valid_bwd_dq_inputs()
    with pytest.raises(AttentionInputError, match="4-D"):
        launch_bwd_dkv(q[0], k, v, d_o, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dkv_rejects_dO_shape_mismatch() -> None:  # noqa: N802
    q, k, v, _d_o, lse, d_arr = _valid_bwd_dq_inputs(d=64)
    bad_do = mx.random.normal((1, 4, 16, 128))  # wrong head_dim vs q
    mx.eval(bad_do)
    with pytest.raises(AttentionInputError, match="dO"):
        launch_bwd_dkv(q, k, v, bad_do, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dkv_rejects_lse_rank() -> None:
    q, k, v, d_o, _lse, d_arr = _valid_bwd_dq_inputs()
    bad_lse = mx.random.normal((4, 16))  # 2-D, not (B, Hq, N)
    mx.eval(bad_lse)
    with pytest.raises(AttentionInputError, match="3-D"):
        launch_bwd_dkv(q, k, v, d_o, bad_lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dkv_rejects_dtype_mismatch() -> None:
    q, _k, v, d_o, lse, d_arr = _valid_bwd_dq_inputs(dtype=mx.float32)
    k_bf16 = mx.random.normal((1, 2, 16, 64)).astype(mx.bfloat16)
    mx.eval(k_bf16)
    with pytest.raises(AttentionInputError, match="dtype"):
        launch_bwd_dkv(q, k_bf16, v, d_o, lse, d_arr, scale=0.1, causal=True)


def test_launch_bwd_dkv_rejects_non_fp32_lse() -> None:
    """Same fp32-residual contract as the dQ launcher: L/D are FIXED fp32 buffers the kernel
    reads untemplated, so a bf16 lse/D would be misread and silently corrupt dK/dV."""
    q, k, v, d_o, _lse, d_arr = _valid_bwd_dq_inputs()
    bad_lse = mx.random.normal((1, 4, 16)).astype(mx.bfloat16)
    mx.eval(bad_lse)
    with pytest.raises(AttentionInputError, match="fp32"):
        launch_bwd_dkv(q, k, v, d_o, bad_lse, d_arr, scale=0.1, causal=True)


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal).
# ---------------------------------------------------------------------------------------

# Measured worsts over the whole grid below (mlx 0.32.0, M1 Max, seed=51: batch {1,2} x
# head_dim {64,128} x head-config {4/4, 4/2, 32/8@n64} x n {61,257,64} x dtype {fp32,bf16}).
#
# The dK/dV kernel accumulates fp32 in-register for both input dtypes (the QK dot, exp, dO.V
# dot, dS, and the scale*ds*q / p*dO accumulates all fp32), so an fp32 diff vs the autodiff
# oracle is pure reduction-order noise. For bf16 the kernel reads bf16 q/k/v/dO and writes bf16
# dK/dV, and its D is derived from the bf16-rounded forward O (`flash_attention_reference` casts
# O down, exactly as a real flash backward reloads a bf16-saved O) -- so the bf16 worst carries
# that single common-mode rounding, not a doubled one, exactly like the dQ pins.
#
# dV is the harsher of the two (dV_j = sum_i P_ij*dO_i sums over EVERY causally-allowed query,
# vs dK's dS^T@q; the 32/8 GQA cases add a 4-head sum on top), so a single per-dtype pin is set
# at the dV worst. MEASURED WORSTS over the whole grid: fp32 dK 5.006790e-06 / dV 1.144409e-05
# (clean fp32 reduction-order values, ~1.5*2^-17); bf16 dK 3.125000e-02 / dV 6.250000e-02
# (6.25e-2 == 2^-4 == exactly ONE bf16 ULP at a 32/8 dV magnitude near 8-16). Pins: fp32 2.5e-5
# (~2.2x the measured worst, same measure-first convention as the dQ 5e-6 / D 2e-5 fp32 pins);
# bf16 1e-1 (~1.6x, bounded BELOW the 2-bf16-ULP value 0.125 -- the same ULP-aware bound as the
# dQ bf16 pin: a future case landing between the pin and 2 ULP is not a regression, widen toward
# 2 ULP with a note, never past it).
_TOL_DKV = {mx.float32: 2.5e-5, mx.bfloat16: 1e-1}


def _dkv_oracle(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool
) -> tuple[mx.array, mx.array]:
    """Exact dK, dV oracle: the vector-Jacobian product of `math_attention` w.r.t. (k, v)
    with the same random cotangent `cot` the kernel path consumes as dO. No readout projection
    -- `mx.vjp` gives the exact autodiff dK/dV, grouped over the GQA q-head groups by autodiff
    (matching the kernel's in-owner whole-group accumulation)."""
    _, vjps = mx.vjp(
        lambda k_, v_: math_attention(q, k_, v_, scale=scale, causal=causal), [k, v], [cot]
    )
    return vjps[0], vjps[1]  # dK, dV


def _dkv_kernel(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    rate_macs_per_s: float | None = None, flip_causal: bool = False,
    variant: str = "scalar", d_slab: int | None = None,
) -> tuple[mx.array, mx.array]:
    """The kernel dK/dV path: forward reference gives (O, L); T7's `launch_bwd_D` gives D from
    (dO, O); `launch_bwd_dkv` consumes q/k/v/dO/L/D and returns the chained (dK, dV).
    `variant`/`d_slab` default to the scalar body (unchanged for every existing caller);
    `variant="mma"` selects the T9b rung-B2 key-major MMA body."""
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=causal)
    d_arr = launch_bwd_D(cot, o)
    return launch_bwd_dkv(
        q, k, v, cot, lse, d_arr, scale=scale, causal=causal,
        rate_macs_per_s=rate_macs_per_s, _flip_causal=flip_causal,
        variant=variant, d_slab=d_slab,
    )


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), _DQ_HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dkv_matches_autodiff_oracle(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype, seed=51)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(
        f"[dKV {['fp32', 'bf16'][dtype == mx.bfloat16]} b{batch} {hq}/{hkv} n{n} "
        f"d{head_dim}] dK={d_dk:.6e} dV={d_dv:.6e}"
    )
    assert d_dk < _TOL_DKV[dtype], f"dK vs autodiff oracle diff {d_dk}"
    assert d_dv < _TOL_DKV[dtype], f"dV vs autodiff oracle diff {d_dv}"


@pytest.mark.metal
def test_chained_dkv_matches_oracle_when_chaining_is_forced() -> None:
    """review-tests High: a chained-vs-single self-comparison cannot catch a systematic carry
    bug present in EVERY split -- so force a >=3-range chained plan (a tiny artificial rate) at
    a small N and run the REAL multi-dispatch code path against the autodiff oracle. The chained
    accumulator, not just its own consistency, must meet the ground-truth gradient."""
    b, hq, hkv, n, d = 1, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=52)

    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    forced_rate = per_row * 60.0  # rows/dispatch = int(0.5*60) = 30 -> ceil(96/30) = 4 ranges
    assert len(plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=forced_rate)) >= 3

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, rate_macs_per_s=forced_rate)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k - dk_ref).max().item()
    d_dv = mx.abs(dv_k - dv_ref).max().item()
    print(f"[dKV forced-chain] dK={d_dk:.6e} dV={d_dv:.6e}")
    assert d_dk < _TOL_DKV[mx.float32], f"chained dK vs oracle diff {d_dk}"
    assert d_dv < _TOL_DKV[mx.float32], f"chained dV vs oracle diff {d_dv}"


@pytest.mark.metal
def test_gqa_dkv_accumulates_whole_group() -> None:
    """The owner key must accumulate over ALL q-heads of its GQA group. Kill-test validity: the
    gap between the kernel (full group) and an oracle fed a dO with ONE q-head zeroed must
    EXCEED the pinned dK/dV tolerance by an explicit margin -- a bare inequality could otherwise
    hide inside a loose pin. Run in fp32 (tight pin) so the margin is unambiguous."""
    b, hq, hkv, n, d = 1, 32, 8, 64, 64   # group_size == 4 (the flagship GQA pattern)
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=53)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True)
    dk_full, dv_full = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    # Oracle MISSING one q-head's contribution: zero q-head 0's dO (it belongs to kv head 0).
    cot_miss = mx.concatenate([mx.zeros((b, 1, n, d)), cot[:, 1:]], axis=1)
    dk_miss, dv_miss = _dkv_oracle(q, k, v, cot_miss, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_full, dv_full, dk_miss, dv_miss)

    d_full = max(
        mx.abs(dk_k - dk_full).max().item(), mx.abs(dv_k - dv_full).max().item()
    )
    gap_miss = max(
        mx.abs(dk_k - dk_miss).max().item(), mx.abs(dv_k - dv_miss).max().item()
    )
    print(f"[dKV GQA] full-group diff={d_full:.6e} zeroed-head gap={gap_miss:.6e}")
    assert d_full < _TOL_DKV[mx.float32], f"kernel vs full-group oracle diff {d_full}"
    # The zeroed-head oracle must diverge by FAR more than the pin -- the kernel really summed
    # that head. Explicit margin: the gap exceeds the tolerance by >= 1000x (measured much
    # larger; a dropped-head bug would land near 0, well under the pin).
    assert gap_miss > 1000.0 * _TOL_DKV[mx.float32], (
        f"zeroed-head oracle gap {gap_miss:.3e} did not exceed the dK/dV pin by the required "
        "margin -- the whole-group accumulation kill-test cannot detect a dropped head"
    )


@pytest.mark.metal
def test_chained_dispatches_equal_single_dispatch() -> None:
    """Chained multi-range dispatches accumulate dK/dV in a FIXED order (query-row outer,
    q-head inner, ascending), each range seeded from the prior's fp32 output -- so a >=3-range
    split must be BIT-identical to a single [0, n) dispatch (fp32->fp32 store/reload is
    lossless). Run at an N that is not a range multiple (mirrors the fwd/dQ split tests)."""
    b, hq, hkv, n, d = 2, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=54)

    single_dk, single_dv = _dkv_kernel(q, k, v, cot, scale=scale, causal=True)  # rate=None
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    forced_rate = per_row * 60.0  # 4 ranges over n=96
    assert len(plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=forced_rate)) >= 3
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, rate_macs_per_s=forced_rate
    )
    mx.eval(single_dk, single_dv, split_dk, split_dv)
    assert mx.array_equal(single_dk, split_dk).item()
    assert mx.array_equal(single_dv, split_dv).item()


@pytest.mark.metal
def test_dkv_bitwise_deterministic_across_runs() -> None:
    """One owner per (batch, kv_head, key) writes disjoint dK/dV output (no atomics, no
    cross-thread accumulation) -> bit-identical across repeated runs. Lock it (1 baseline + 4
    repeats = 5 runs, mirrors test_dq_bitwise_deterministic_across_runs)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=55)
    dk0, dv0 = _dkv_kernel(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk0, dv0)
    for _ in range(4):
        dk, dv = _dkv_kernel(q, k, v, cot, scale=scale, causal=True)
        mx.eval(dk, dv)
        assert mx.array_equal(dk, dk0).item()
        assert mx.array_equal(dv, dv0).item()


@pytest.mark.metal
def test_dkv_causal_skip_perturbation_fails_parity() -> None:
    """The named bug site: build the dK/dV kernel with the causal-keep inequality flipped to
    the WRONG triangle (i <= key). Its dK/dV must DIVERGE from the causal autodiff oracle -- if
    a flipped inequality ever matched, the parity grid could not detect an off-by-one in the
    causal skip (mirrors test_dq_causal_skip_perturbation_fails_parity)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=16, d=64, dtype=mx.float32, seed=56)

    dk_wrong, dv_wrong = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, flip_causal=True)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_wrong, dv_wrong, dk_ref, dv_ref)

    d_dk = mx.abs(dk_wrong - dk_ref).max().item()
    d_dv = mx.abs(dv_wrong - dv_ref).max().item()
    assert d_dk > 1e-2 or d_dv > 1e-2, (
        f"flipped causal-skip kernel matched the causal oracle (dK={d_dk:.3e}, dV={d_dv:.3e}) "
        "-- the parity grid cannot detect a causal off-by-one"
    )


# =======================================================================================
# T9 Step 4 -- api.py vjp completion: the FULLY kernel-backed backward for impl="kernel"
# (kernel D + dQ + chained dK/dV), with the construction-time calibrated backward rate
# closure-captured. impl="reference" keeps the pure-MLX oracle backward.
# =======================================================================================

# Measured worst |grad diff| vs the math_attention autodiff oracle (mlx 0.32.0, M1 Max,
# seed=61/62): eager 9.536743e-07, under compile 2.123415e-06 -- the kernel D/dQ/dK/dV all
# accumulate fp32, feeding the same math the oracle does, so the gap is pure fp32
# reduction-order noise. Pin 1e-5 (~4.7x the compile worst, matching the ~5x margin the
# reference-backward grad tests use: test_attention_api.py
# ::test_flash_attention_grads_match_autodiff_oracle pins 2e-5 over a 3.8e-6 worst).
_TOL_KERNEL_GRAD = 1e-5


@pytest.mark.metal
def test_flash_attention_kernel_grads_match_oracle_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """impl='kernel' now routes the WHOLE backward through Metal kernels (T7 D + T8 dQ + T9
    chained dK/dV). Grads of sum(flash_attention(impl='kernel')) must match the math_attention
    autodiff oracle, AND the dK/dV kernel launcher must actually fire -- a spy on
    `api.launch_bwd_dkv` plus the kernel-backward engagement counter prove the vjp is
    kernel-backed, not the pure-MLX oracle backward (value parity alone can't tell them apart,
    since both compute the same gradient)."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, _cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=61)
    api.VJP_CALLS.clear()

    seen_dkv: list[int] = []
    real_dkv = api.launch_bwd_dkv

    def spy_dkv(*args: object, **kwargs: object) -> object:
        seen_dkv.append(1)
        return real_dkv(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api, "launch_bwd_dkv", spy_dkv)

    def kernel_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="kernel").sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True).sum()

    g_kernel = mx.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_kernel, *g_math)

    worst = max(
        float(mx.abs(gk - gm).max().item())
        for gk, gm in zip(g_kernel, g_math, strict=True)
    )
    print(f"[kernel-bwd end-to-end] worst |grad diff|={worst:.6e}")
    assert worst < _TOL_KERNEL_GRAD, f"worst |grad diff|={worst}"
    assert seen_dkv, "launch_bwd_dkv never fired -- the backward is not kernel-backed"
    assert api.VJP_CALLS.get("flash_attention_kernel_bwd", 0) > 0


@pytest.mark.metal
def test_flash_attention_kernel_grads_match_oracle_under_compile() -> None:
    """The kernel-backed vjp must survive `mx.compile`: the backward calibration is resolved at
    CONSTRUCTION time and closure-captured, so the compiled graph contains only kernel
    dispatches (no host-sync in the loss/vjp path -- host-sync timing is compile-hostile). Grads
    from the compiled grad function must still match the oracle, and the kernel-backward branch
    must fire during the trace (the engagement counter increments once at trace time)."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, _cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=62)
    api.VJP_CALLS.clear()

    def kernel_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="kernel").sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True).sum()

    compiled_grad = mx.compile(mx.grad(kernel_loss, argnums=(0, 1, 2)))
    g_kernel = compiled_grad(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_kernel, *g_math)

    worst = max(
        float(mx.abs(gk - gm).max().item())
        for gk, gm in zip(g_kernel, g_math, strict=True)
    )
    print(f"[kernel-bwd under compile] worst |grad diff|={worst:.6e}")
    assert worst < _TOL_KERNEL_GRAD, f"worst |grad diff|={worst}"
    assert api.VJP_CALLS.get("flash_attention_kernel_bwd", 0) > 0


# =======================================================================================
# T9b rung B1 -- dQ MMA variant (register-resident D-slabbed backward). One 32-lane
# simdgroup per 32-row query block: S = Q@K^T (MMA), P = exp(scale*S - L) from the SAVED
# L (per-row, NO online-softmax rowmax -- each element independent), dP = dO@V^T (second
# MMA), dS = scale*P*(dP - D), dQ_acc += dS@K_slab (GEMM-B, register-resident fp32 C_dq
# D-slabbed). Same (q,k,v,dO,lse,d_arr,qoffs,scale_in)->(dq_out) contract as the scalar dQ
# kernel; the scalar body is the correctness oracle and stays the default everywhere. The
# controller owns the saturation d_slab sweep (Step-2 tuning) -- this rung is CORRECTNESS +
# small-shape parity only; the mma variant is NOT wired into api.py or the calibrated path.
# =======================================================================================

# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_bwd_dq_mma_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_bwd_dq_mma_source(hd, causal=True)   # default slab 32
        assert "slab0 += 32" in s                      # D_SLAB templated
        assert "C_dq[4][4]" in s                        # D_SLAB_TILES == 32 // 8 == 4
        assert f"slab0 < {hd}" in s                     # HEAD_DIM baked into the slab loop bound
        assert f"d0 < {hd}" in s                        # HEAD_DIM baked into the QK/dOV d-loops
        assert "HEAD_DIM" not in s                      # every sentinel substituted (lossless)
        assert "D_SLAB" not in s                         # (also covers the D_SLAB_TILES sentinel)


def test_build_bwd_dq_mma_source_causal_keep_comparison() -> None:
    s = build_bwd_dq_mma_source(64, causal=True)
    assert "kk <= row" in s
    assert "kk >= row" not in s
    assert "KEEP_CMP" not in s
    # causal walks KV blocks only up to each query block's diagonal (the fwd-mma kv_limit).
    assert "metal::min(n, r0 + block_base + 32u)" in s
    assert "KV_LIMIT" not in s


def test_build_bwd_dq_mma_source_noncausal_keeps_all_keys() -> None:
    s = build_bwd_dq_mma_source(64, causal=False)
    assert "kk <= row" not in s
    assert "kk >= row" not in s
    assert "(true)" in s               # KEEP_CMP -> true
    # non-causal scans every KV block: kv_limit is the full key count.
    assert "uint kv_limit = n;" in s


def test_build_bwd_dq_mma_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-triangle arm: the causal-skip inequality is flipped so a parity run
    # against the causal oracle FAILS -- the named bug site of the dQ kernel, mma-body variant.
    s = build_bwd_dq_mma_source(64, causal=True, flip_causal=True)
    assert "kk >= row" in s
    assert "kk <= row" not in s


def test_build_bwd_dq_mma_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_bwd_dq_mma_source(hd, causal=True)


def test_build_bwd_dq_mma_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_bwd_dq_mma_source(64, causal=False, flip_causal=True)


def test_build_bwd_dq_mma_source_rejects_bad_d_slab() -> None:
    # d_slab must be a positive multiple of 8 dividing head_dim (mirrors build_fwd_mma_source).
    for hd, slab in ((64, 48), (128, 12), (64, 0), (96, 64)):
        with pytest.raises(ValueError, match="d_slab"):
            build_bwd_dq_mma_source(hd, causal=True, d_slab=slab)


def test_build_bwd_dq_mma_source_slab_widths_all_template() -> None:
    # The controller sweeps these at saturation; every valid slab width must template cleanly.
    for slab in (16, 32, 64, 128):
        s = build_bwd_dq_mma_source(128, causal=True, d_slab=slab)
        assert f"slab0 += {slab}" in s
        assert f"C_dq[4][{slab // 8}]" in s
        assert "D_SLAB" not in s
        assert "HEAD_DIM" not in s


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal). The mma dQ body has a DIFFERENT fp32 reduction
# order than the scalar body (simdgroup-matrix reassociation + per-slab recompute vs the
# sequential per-key +=), so its parity worsts are measured and pinned SEPARATELY -- never by
# widening the scalar _TOL_DQ.
#
# MEASURED WORSTS over the parity grid (mlx 0.32.0, M1 Max, seed=41, d_slab default 32 AND 64:
# batch {1,2} x head_dim {64,128} x head-config {4/4, 4/2, 32/8@n64} x n {61,257,64} x
# dtype {fp32,bf16} x causal True, plus head_dim 96 slab {16,32}, the head_dim 128 slab
# {16,32,64,128} build cases, and causal=False): fp32 2.464958e-06, bf16 1.562500e-02. The mma
# body accumulates fp32 in-register for both dtypes (the QK^T + dO@V^T MMAs, exp, dS, and the
# dS@K GEMM-B all fp32), so an fp32 diff vs the autodiff oracle is pure reduction-order noise;
# the bf16 worst is exactly one bf16 ULP (2^-6) at a dQ magnitude near 2-4 (quantized rounding,
# not accumulation drift). The mma reduction order (simdgroup reassociation + per-slab recompute)
# lands on the SAME worst class as the scalar dQ body here -- but the pins are set independently
# (measure-first), never inherited from the scalar. Pins: fp32 5e-6 (~2.0x the measured worst,
# the scalar dQ 5e-6 fp32 convention); bf16 3e-2 (~1.9x, bounded below the 2-bf16-ULP value
# 3.125e-2 -- the same ULP-aware bound as the scalar dQ bf16 pin: a future case landing between
# the pin and 2 ULP is not a regression, widen toward 2 ULP with a note, never past it).
_TOL_DQ_MMA = {mx.float32: 5e-6, mx.bfloat16: 3e-2}


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), _DQ_HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dq_mma_matches_autodiff_oracle(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    """The mma dQ body (default slab 32) must match the same autodiff oracle the scalar body
    does, across the T9 parity grid. The controller owns the saturation d_slab sweep; this rung
    is correctness only, so the default register-safe slab is the parity anchor."""
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    print(
        f"[dQ-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} b{batch} {hq}/{hkv} n{n} "
        f"d{head_dim} slab32] diff={diff:.6e}"
    )
    assert diff < _TOL_DQ_MMA[dtype], f"dQ-mma vs autodiff oracle diff {diff}"


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), [(4, 2, 257), (32, 8, 64)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dq_mma_non_default_slab_matches_oracle(
    hq: int, hkv: int, n: int, head_dim: int, dtype: mx.Dtype
) -> None:
    """A NON-DEFAULT slab (64) must be as correct as the default -- the slab choice is a pure
    throughput lever, never a correctness one (mirrors the forward's slab-independent parity).
    slab=64 is single-pass at head_dim=64 and a 2-pass recompute at head_dim=128."""
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=64)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    print(
        f"[dQ-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} {hq}/{hkv} n{n} d{head_dim} "
        f"slab64] diff={diff:.6e}"
    )
    assert diff < _TOL_DQ_MMA[dtype], f"dQ-mma slab64 vs autodiff oracle diff {diff}"


@pytest.mark.metal
@pytest.mark.parametrize("d_slab", [16, 32], ids=["slab16", "slab32"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dq_mma_head_dim_96_matches_oracle(d_slab: int, dtype: mx.Dtype) -> None:
    """head_dim=96 exercises the non-power-of-two slab-COUNT edge (96/32==3 passes, 96/16==6)
    -- it must build and match the oracle. 96 is a supported head dim the T6 fwd ladder never
    ran, so the dQ mma body proves its correctness here independently."""
    hq, hkv, n = 4, 2, 61
    scale = 1.0 / math.sqrt(96)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=96, dtype=dtype)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=d_slab)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    print(f"[dQ-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} d96 slab{d_slab}] diff={diff:.6e}")
    assert diff < _TOL_DQ_MMA[dtype], f"dQ-mma d96 slab{d_slab} vs oracle diff {diff}"


@pytest.mark.metal
def test_dq_mma_noncausal_matches_oracle() -> None:
    """causal=False: the mma body scans every KV block and keeps every key (kv_limit == n,
    KEEP_CMP == true). Parity against the non-causal autodiff oracle at one config."""
    hq, hkv, n, d = 4, 2, 64, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=45)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=False, variant="mma")
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=False)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k - dq_ref).max().item()
    print(f"[dQ-mma non-causal] diff={diff:.6e}")
    assert diff < _TOL_DQ_MMA[mx.float32], f"dQ-mma non-causal vs oracle diff {diff}"


@pytest.mark.metal
@pytest.mark.parametrize("d_slab", [16, 32, 64, 128], ids=["slab16", "slab32", "slab64", "slab128"])
def test_dq_mma_all_slabs_build_and_match_at_head_dim_128(d_slab: int) -> None:
    """Every controller-swept slab width {16,32,64,128} must BUILD and be correct at the flagship
    head_dim=128 (slab128 is the register-heaviest, single-pass full-D C_dq -- the buildability
    bar). One parity case per slab proves the JIT compiles it and the result is right."""
    hq, hkv, n = 4, 2, 96
    scale = 1.0 / math.sqrt(128)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=128, dtype=mx.float32, seed=46)

    dq_k = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=d_slab)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k - dq_ref).max().item()
    print(f"[dQ-mma d128 slab{d_slab} build+parity] diff={diff:.6e}")
    assert diff < _TOL_DQ_MMA[mx.float32], f"dQ-mma d128 slab{d_slab} vs oracle diff {diff}"


@pytest.mark.metal
def test_dq_mma_split_matches_single_dispatch() -> None:
    """Query-range multi-dispatch writes DISJOINT dQ rows (no chaining -- each row's dQ depends
    only on its own absolute position); the reassembled mma result must be bit-identical to a
    single dispatch. Run at batch>1 and an N that is not a 32-row-block multiple (mirrors the
    scalar test_dq_split_matches_single_dispatch)."""
    b, hq, hkv, n, d = 2, 4, 2, 257, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=44)

    single = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")  # rate=None
    per_row = _bwd_dq_macs_per_row(n=n, d=d, b=b, hq=hq)
    # Force ~80 rows/dispatch (several disjoint mma dispatches over n=257) via a low rate, inside
    # MAX_DISPATCH_SECONDS AND MAX_TOTAL_SECONDS.
    split = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", rate_macs_per_s=per_row * 160.0
    )
    mx.eval(single, split)
    assert mx.array_equal(single, split).item()


@pytest.mark.metal
def test_dq_mma_bitwise_deterministic_across_runs() -> None:
    """One 32-lane simdgroup per query block writes disjoint dQ rows (no atomics, no cross-thread
    accumulation) -> bit-identical dQ across repeated runs. Lock it (1 baseline + 4 repeats = 5
    runs, mirrors test_dq_bitwise_deterministic_across_runs)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=42)
    dq0 = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    mx.eval(dq0)
    for _ in range(4):
        dq = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
        mx.eval(dq)
        assert mx.array_equal(dq, dq0).item()


@pytest.mark.metal
def test_dq_mma_causal_skip_perturbation_fails_parity() -> None:
    """The named bug site, mma body: build with the causal-skip inequality flipped to the WRONG
    triangle (kk >= row). Its dQ must DIVERGE from the causal autodiff oracle -- if a flipped
    inequality ever matched, the parity grid could not detect a causal off-by-one in the mma
    body (mirrors test_dq_causal_skip_perturbation_fails_parity)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=16, d=64, dtype=mx.float32, seed=43)

    dq_wrong = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", flip_causal=True)
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_wrong, dq_ref)

    diff = mx.abs(dq_wrong.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    assert diff > 1e-2, (
        f"flipped causal-skip mma kernel matched the causal oracle (diff={diff:.3e}) -- "
        "the parity grid cannot detect a causal off-by-one"
    )
# =======================================================================================
# T9b rung B2 -- dK/dV MMA variant (key-major, register-resident D-slabbed, CHAINED). One
# 32-lane simdgroup per 32-KEY block per (batch, kv_head): S^T = K@Q^T (MMA, keyxquery),
# P^T = exp(scale*S^T - L_col) from the SAVED L (L indexed by the QUERY = fragment column,
# no online-softmax rowmax), dP^T = V@dO^T (second MMA), dS^T = scale*P^T*(dP^T - D_col), then
# dV_acc += P^T@dO_slab and dK_acc += dS^T@Q_slab (register-resident fp32 C_dv/C_dk, D-slabbed,
# SEEDED FROM dk_in/dv_in). Same (q,k,v,dO,lse,d_arr,dk_in,dv_in,qoffs,scale_in)->(dk_out,
# dv_out) chained contract as the scalar dK/dV kernel; the scalar body is the correctness oracle
# and stays the default everywhere. The controller owns the saturation d_slab sweep -- this rung
# is CORRECTNESS + small-shape parity only; the mma variant is NOT wired into api.py or the
# calibrated path.
# =======================================================================================

# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_bwd_dkv_mma_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_bwd_dkv_mma_source(hd, causal=True)   # default slab 32
        assert "slab0 += 32" in s                        # D_SLAB templated
        assert "C_dv[4][4]" in s                  # dV accumulator, D_SLAB_TILES == 32//8 == 4
        assert "C_dk[4][4]" in s                  # dK accumulator
        assert f"slab0 < {hd}" in s               # HEAD_DIM baked into the slab loop bound
        assert f"d0 < {hd}" in s                  # HEAD_DIM baked into the K@Q^T / V@dO^T d-loops
        assert "HEAD_DIM" not in s                       # every sentinel substituted (lossless)
        assert "D_SLAB" not in s                          # (also covers the D_SLAB_TILES sentinel)


def test_build_bwd_dkv_mma_source_causal_keep_comparison() -> None:
    s = build_bwd_dkv_mma_source(64, causal=True)
    # The owner is the KEY block and the loop is over query blocks; the causal keep is the
    # per-element query>=key, the swapped-roles analogue of dQ-mma's kk<=row.
    assert "i >= key" in s
    assert "i <= key" not in s
    assert "KEEP_CMP" not in s
    # causal walks query blocks only from each key block's diagonal upward (the query-start bound).
    assert "metal::max(q_lo, key_base)" in s
    assert "Q_START" not in s


def test_build_bwd_dkv_mma_source_noncausal_keeps_all_queries() -> None:
    s = build_bwd_dkv_mma_source(64, causal=False)
    assert "i >= key" not in s
    assert "i <= key" not in s
    assert "(true)" in s                # KEEP_CMP -> true
    # non-causal scans every query block from q_lo: the query-start bound is the range start.
    assert "uint q_start = q_lo;" in s


def test_build_bwd_dkv_mma_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-triangle arm: the causal-keep inequality is flipped so a parity run
    # against the causal oracle FAILS -- the named-bug-site perturbation for dK/dV, mma body.
    s = build_bwd_dkv_mma_source(64, causal=True, flip_causal=True)
    assert "i <= key" in s
    assert "i >= key" not in s


def test_build_bwd_dkv_mma_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_bwd_dkv_mma_source(hd, causal=True)


def test_build_bwd_dkv_mma_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_bwd_dkv_mma_source(64, causal=False, flip_causal=True)


def test_build_bwd_dkv_mma_source_rejects_bad_d_slab() -> None:
    # d_slab must be a positive multiple of 8 dividing head_dim (mirrors build_bwd_dq_mma_source).
    for hd, slab in ((64, 48), (128, 12), (64, 0), (96, 64)):
        with pytest.raises(ValueError, match="d_slab"):
            build_bwd_dkv_mma_source(hd, causal=True, d_slab=slab)


def test_build_bwd_dkv_mma_source_slab_widths_all_template() -> None:
    # The controller sweeps these at saturation; every valid slab width must template cleanly.
    for slab in (16, 32, 64, 128):
        s = build_bwd_dkv_mma_source(128, causal=True, d_slab=slab)
        assert f"slab0 += {slab}" in s
        assert f"C_dv[4][{slab // 8}]" in s
        assert f"C_dk[4][{slab // 8}]" in s
        assert "D_SLAB" not in s
        assert "HEAD_DIM" not in s


# ---------------------------------------------------------------------------------------
# plan_dkv_dispatches block-alignment (DEFAULT lane, pure integer arithmetic -- no GPU).
# The mma dK/dV variant needs its chained query-range split to land on 32-row query-block
# boundaries (a mid-block split would merge different partial products inside one hardware
# MMA and break chained==single bit-identity); the scalar path (block_align=1, the default)
# is byte-unchanged.
# ---------------------------------------------------------------------------------------


def test_plan_dkv_dispatches_default_alignment_byte_unchanged() -> None:
    n, d, b, hq = 96, 64, 2, 4
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = per_row * 60.0  # rows_per = int(0.5*60) = 30 (un-aligned)
    default = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate)
    explicit = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate, block_align=1)
    assert default == explicit                                   # default IS block_align=1
    assert default == [(0, 30), (30, 60), (60, 90), (90, 96)]    # scalar behaviour, unchanged


def test_plan_dkv_dispatches_block_aligned_to_32() -> None:
    n, d, b, hq = 96, 64, 2, 4
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = per_row * 80.0  # un-aligned rows_per = int(0.5*80)=40 -> aligned DOWN to 32 (1 block)
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate, block_align=32)
    assert ranges == [(0, 32), (32, 64), (64, 96)]
    # every q_lo is a 32-multiple; ranges tile [0, n) exactly (contiguous, cover [0, n)).
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n
    for lo, _hi in ranges:
        assert lo % 32 == 0
    for (_lo0, hi0), (lo1, _hi1) in itertools.pairwise(ranges):
        assert hi0 == lo1


def test_plan_dkv_dispatches_block_align_rounds_down_to_multiple() -> None:
    n, d, b, hq = 200, 64, 1, 4
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = per_row * 140.0  # rows_per = int(0.5*140) = 70 -> aligned DOWN to 64 (2 blocks)
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate, block_align=32)
    assert ranges == [(0, 64), (64, 128), (128, 192), (192, 200)]


def test_plan_dkv_dispatches_block_align_refusal_unchanged() -> None:
    """Refusal semantics unchanged: when even one minimal aligned unit (a single 32-row query
    block) over-budgets the per-dispatch bound, plan_dkv_dispatches still raises rather than
    return an over-budget range for the uncatchable GPU watchdog to hit."""
    n, d, b, hq = 96, 64, 2, 4
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = per_row * 20.0  # un-aligned rows_per=10; floored to 32 -> 32*per_row/rate = 1.6 s > 0.5
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate, block_align=32)


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal). The mma dK/dV body has a DIFFERENT fp32
# reduction order than the scalar body (key-major simdgroup-matrix reassociation + per-slab
# recompute vs the scalar's sequential per-query +=), so its parity worsts are measured and
# pinned SEPARATELY -- never by widening the scalar _TOL_DKV.
#
# MEASURED WORSTS over the parity grid (mlx 0.32.0, M1 Max, seed=51, d_slab default 32 AND 64:
# batch {1,2} x head_dim {64,128} x head-config {4/4, 4/2, 32/8@n64} x n {61,257,64} x
# dtype {fp32,bf16} x causal True, plus head_dim 96 slab {16,32}, the head_dim 128 slab
# {16,32,64,128} build cases, and causal=False): fp32 dK 5.245209e-06 / dV 9.059906e-06; bf16 dK
# 3.125000e-02 / dV 6.250000e-02. The mma body accumulates fp32 in-register for both dtypes (the
# K@Q^T + V@dO^T MMAs, exp, dS, and the P^T@dO / dS^T@Q GEMM-Bs all fp32), so an fp32 diff vs the
# autodiff oracle is pure reduction-order noise; the bf16 dV worst 6.25e-2 == 2^-4 == exactly ONE
# bf16 ULP at a 32/8 dV magnitude near 8-16 (quantized rounding, not accumulation drift). dV is the
# harsher of the two (dV = sum_i P_ij*dO_i sums over EVERY causally-allowed query; the 32/8 GQA
# cases add a 4-head sum), so a single per-dtype pin is set at the dV worst. Pins: fp32 2e-5 (~2.2x
# the measured dV worst, the same margin ratio the scalar dK/dV 2.5e-5 fp32 pin uses over its own
# 1.14e-5 worst); bf16 1e-1 (~1.6x, bounded BELOW the 2-bf16-ULP value 0.125 -- the same ULP-aware
# bound as the scalar dK/dV bf16 pin: a future case landing between the pin and 2 ULP is not a
# regression, widen toward 2 ULP with a note, never past it). Set INDEPENDENTLY (measure-first) --
# the mma reduction order lands on the same worst class as the scalar dK/dV, but the pins are its
# own measurement, never inherited from or widening the scalar _TOL_DKV.
_TOL_DKV_MMA = {mx.float32: 2e-5, mx.bfloat16: 1e-1}


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), _DQ_HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dkv_mma_matches_autodiff_oracle(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    """The mma dK/dV body (default slab 32) must match the same autodiff oracle the scalar body
    does, across the T9 parity grid. The controller owns the saturation d_slab sweep; this rung
    is correctness only, so the default register-safe slab is the parity anchor."""
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype, seed=51)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(
        f"[dKV-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} b{batch} {hq}/{hkv} n{n} "
        f"d{head_dim} slab32] dK={d_dk:.6e} dV={d_dv:.6e}"
    )
    assert d_dk < _TOL_DKV_MMA[dtype], f"dK-mma vs autodiff oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[dtype], f"dV-mma vs autodiff oracle diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), [(4, 2, 257), (32, 8, 64)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dkv_mma_non_default_slab_matches_oracle(
    hq: int, hkv: int, n: int, head_dim: int, dtype: mx.Dtype
) -> None:
    """A NON-DEFAULT slab (64) must be as correct as the default -- the slab choice is a pure
    throughput lever, never a correctness one (mirrors the forward's slab-independent parity).
    slab=64 is single-pass at head_dim=64 and a 2-pass recompute at head_dim=128."""
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype, seed=51)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=64)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(
        f"[dKV-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} {hq}/{hkv} n{n} d{head_dim} "
        f"slab64] dK={d_dk:.6e} dV={d_dv:.6e}"
    )
    assert d_dk < _TOL_DKV_MMA[dtype], f"dK-mma slab64 vs autodiff oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[dtype], f"dV-mma slab64 vs autodiff oracle diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize("d_slab", [16, 32], ids=["slab16", "slab32"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dkv_mma_head_dim_96_matches_oracle(d_slab: int, dtype: mx.Dtype) -> None:
    """head_dim=96 exercises the non-power-of-two slab-COUNT edge (96/32==3 passes, 96/16==6)
    -- it must build and match the oracle. 96 is a supported head dim the T6 fwd ladder never
    ran, so the dK/dV mma body proves its correctness here independently."""
    hq, hkv, n = 4, 2, 61
    scale = 1.0 / math.sqrt(96)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=96, dtype=dtype, seed=51)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=d_slab)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(
        f"[dKV-mma {['fp32', 'bf16'][dtype == mx.bfloat16]} d96 slab{d_slab}] "
        f"dK={d_dk:.6e} dV={d_dv:.6e}"
    )
    assert d_dk < _TOL_DKV_MMA[dtype], f"dK-mma d96 slab{d_slab} vs oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[dtype], f"dV-mma d96 slab{d_slab} vs oracle diff {d_dv}"


@pytest.mark.metal
def test_dkv_mma_noncausal_matches_oracle() -> None:
    """causal=False: the mma body scans every query block from q_lo and keeps every query
    (q_start == q_lo, KEEP_CMP == true). Parity against the non-causal autodiff oracle."""
    hq, hkv, n, d = 4, 2, 64, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=57)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=False, variant="mma")
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=False)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k - dk_ref).max().item()
    d_dv = mx.abs(dv_k - dv_ref).max().item()
    print(f"[dKV-mma non-causal] dK={d_dk:.6e} dV={d_dv:.6e}")
    assert d_dk < _TOL_DKV_MMA[mx.float32], f"dK-mma non-causal vs oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[mx.float32], f"dV-mma non-causal vs oracle diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize("d_slab", [16, 32, 64, 128], ids=["slab16", "slab32", "slab64", "slab128"])
def test_dkv_mma_all_slabs_build_and_match_at_head_dim_128(d_slab: int) -> None:
    """Every controller-swept slab width {16,32,64,128} must BUILD and be correct at the flagship
    head_dim=128 (slab128 is the register-heaviest -- single-pass full-D C_dv + C_dk, the
    buildability bar). One parity case per slab proves the JIT compiles it and the result is
    right."""
    hq, hkv, n = 4, 2, 96
    scale = 1.0 / math.sqrt(128)
    q, k, v, cot = _rand_qkv_do(b=1, hq=hq, hkv=hkv, n=n, d=128, dtype=mx.float32, seed=58)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma", d_slab=d_slab)
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k - dk_ref).max().item()
    d_dv = mx.abs(dv_k - dv_ref).max().item()
    print(f"[dKV-mma d128 slab{d_slab} build+parity] dK={d_dk:.6e} dV={d_dv:.6e}")
    assert d_dk < _TOL_DKV_MMA[mx.float32], f"dK-mma d128 slab{d_slab} vs oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[mx.float32], f"dV-mma d128 slab{d_slab} vs oracle diff {d_dv}"


@pytest.mark.metal
def test_dkv_mma_chained_matches_oracle_when_chaining_is_forced() -> None:
    """review-tests High (the REQUIRED chained proof, constraint 4d): a chained-vs-single
    self-comparison cannot catch a systematic carry bug present in EVERY split -- so force a
    >=3-range chained plan (a tiny artificial rate) at a small N and run the REAL multi-dispatch
    mma code path against the autodiff oracle. The chained fp32 accumulator, not just its own
    consistency, must meet the ground-truth gradient. The forced ranges are verified 32-aligned
    (the mma variant's block-alignment contract)."""
    b, hq, hkv, n, d = 1, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=52)

    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    forced_rate = per_row * 130.0  # rows_per = int(0.5*130) = 65 -> aligned to 64 -> 2 ranges? no:
    # 64 rows/dispatch over n=96 -> [(0,64),(64,96)] = 2 ranges; bump lower for >=3 aligned ranges.
    forced_rate = per_row * 70.0   # rows_per = int(0.5*70)=35 -> aligned to 32 -> 3 ranges over 96
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=forced_rate, block_align=32)
    assert len(ranges) >= 3
    for lo, _hi in ranges:
        assert lo % 32 == 0                       # every range starts on a 32-row query block

    dk_k, dv_k = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", rate_macs_per_s=forced_rate
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k - dk_ref).max().item()
    d_dv = mx.abs(dv_k - dv_ref).max().item()
    print(f"[dKV-mma forced-chain] ranges={ranges} dK={d_dk:.6e} dV={d_dv:.6e}")
    assert d_dk < _TOL_DKV_MMA[mx.float32], f"chained dK-mma vs oracle diff {d_dk}"
    assert d_dv < _TOL_DKV_MMA[mx.float32], f"chained dV-mma vs oracle diff {d_dv}"


@pytest.mark.metal
def test_dkv_mma_chained_dispatches_equal_single_dispatch() -> None:
    """Chained multi-range mma dispatches accumulate dK/dV in a FIXED order (query block outer
    ascending, q-head inner ascending), each 32-aligned range seeded from the prior's fp32 output
    -- so a >=3-range split must be BIT-identical to a single [0, n) dispatch (fp32->fp32
    store/reload is lossless). A mid-block split would merge different partial products inside one
    MMA and break this; the 32-row block alignment restores the scalar order argument at block
    granularity. Run at an N that is not a range multiple (mirrors the scalar dK/dV split test)."""
    b, hq, hkv, n, d = 2, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=54)

    single_dk, single_dv = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    forced_rate = per_row * 70.0  # 32-aligned rows_per = 32 -> 3 ranges over n=96
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=forced_rate, block_align=32)
    assert len(ranges) >= 3
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", rate_macs_per_s=forced_rate
    )
    mx.eval(single_dk, single_dv, split_dk, split_dv)
    assert mx.array_equal(single_dk, split_dk).item()
    assert mx.array_equal(single_dv, split_dv).item()


@pytest.mark.metal
def test_dkv_mma_bitwise_deterministic_across_runs() -> None:
    """One 32-lane simdgroup per (batch, kv_head, 32-key block) seeds from dk_in/dv_in and stores
    disjoint dK/dV key rows (no atomics, no cross-thread accumulation) -> bit-identical across
    repeated runs. Lock it (1 baseline + 4 repeats = 5 runs, mirrors the scalar dK/dV
    determinism test)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=55)
    dk0, dv0 = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    mx.eval(dk0, dv0)
    for _ in range(4):
        dk, dv = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
        mx.eval(dk, dv)
        assert mx.array_equal(dk, dk0).item()
        assert mx.array_equal(dv, dv0).item()


@pytest.mark.metal
def test_dkv_mma_gqa_accumulates_whole_group() -> None:
    """The owner key must accumulate over ALL q-heads of its GQA group (mma body). Kill-test
    validity: the gap between the kernel (full group) and an oracle fed a dO with ONE q-head
    zeroed must EXCEED the pinned dK/dV tolerance by an explicit margin -- a bare inequality could
    otherwise hide inside a loose pin. Run in fp32 (tight pin) so the margin is unambiguous."""
    b, hq, hkv, n, d = 1, 32, 8, 64, 64   # group_size == 4 (the flagship GQA pattern)
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=53)

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant="mma")
    dk_full, dv_full = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    # Oracle MISSING one q-head's contribution: zero q-head 0's dO (it belongs to kv head 0).
    cot_miss = mx.concatenate([mx.zeros((b, 1, n, d)), cot[:, 1:]], axis=1)
    dk_miss, dv_miss = _dkv_oracle(q, k, v, cot_miss, scale=scale, causal=True)
    mx.eval(dk_k, dv_k, dk_full, dv_full, dk_miss, dv_miss)

    d_full = max(
        mx.abs(dk_k - dk_full).max().item(), mx.abs(dv_k - dv_full).max().item()
    )
    gap_miss = max(
        mx.abs(dk_k - dk_miss).max().item(), mx.abs(dv_k - dv_miss).max().item()
    )
    print(f"[dKV-mma GQA] full-group diff={d_full:.6e} zeroed-head gap={gap_miss:.6e}")
    assert d_full < _TOL_DKV_MMA[mx.float32], f"kernel vs full-group oracle diff {d_full}"
    # The zeroed-head oracle must diverge by FAR more than the pin -- the kernel really summed
    # that head. Explicit margin: the gap exceeds the tolerance by >= 1000x.
    assert gap_miss > 1000.0 * _TOL_DKV_MMA[mx.float32], (
        f"zeroed-head oracle gap {gap_miss:.3e} did not exceed the dK/dV-mma pin by the required "
        "margin -- the whole-group accumulation kill-test cannot detect a dropped head"
    )


@pytest.mark.metal
def test_dkv_mma_causal_skip_perturbation_fails_parity() -> None:
    """The named bug site, mma body: build with the causal-keep inequality flipped to the WRONG
    triangle (i <= key). Its dK/dV must DIVERGE from the causal autodiff oracle -- if a flipped
    inequality ever matched, the parity grid could not detect an off-by-one in the causal skip
    (mirrors test_dkv_causal_skip_perturbation_fails_parity)."""
    scale = 1.0 / math.sqrt(64)
    q, k, v, cot = _rand_qkv_do(b=2, hq=4, hkv=2, n=16, d=64, dtype=mx.float32, seed=56)

    dk_wrong, dv_wrong = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", flip_causal=True
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_wrong, dv_wrong, dk_ref, dv_ref)

    d_dk = mx.abs(dk_wrong - dk_ref).max().item()
    d_dv = mx.abs(dv_wrong - dv_ref).max().item()
    assert d_dk > 1e-2 or d_dv > 1e-2, (
        f"flipped causal-skip mma kernel matched the causal oracle (dK={d_dk:.3e}, dV={d_dv:.3e}) "
        "-- the parity grid cannot detect a causal off-by-one"
    )
