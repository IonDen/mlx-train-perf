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
import math

import mlx.core as mx
import pytest

from mlx_train_perf.attention.kernel.launch import (
    _bwd_dq_macs_per_row,
    launch_bwd_D,
    launch_bwd_dq,
)
from mlx_train_perf.attention.kernel.source import build_bwd_D_source, build_bwd_dq_source
from mlx_train_perf.attention.reference import flash_attention_reference, math_attention
from mlx_train_perf.errors import AttentionInputError

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
) -> mx.array:
    """The kernel dQ path: forward reference gives (O, L); T7's `launch_bwd_D` gives D from
    (dO, O); `launch_bwd_dq` consumes q/k/v/dO/L/D."""
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=causal)
    d_arr = launch_bwd_D(cot, o)
    return launch_bwd_dq(
        q, k, v, cot, lse, d_arr, scale=scale, causal=causal,
        rate_macs_per_s=rate_macs_per_s, _flip_causal=flip_causal,
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
