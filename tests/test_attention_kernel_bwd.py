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

from mlx_train_perf.attention import api
from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.kernel import launch as bwd_launch
from mlx_train_perf.attention.kernel.dispatch import select_bwd_tiles
from mlx_train_perf.attention.kernel.launch import (
    TileShape,
    _bwd_dkv_kernel,
    _dkv_kernel_name,
    calibrated_bwd_dkv_rate,
    calibrated_bwd_dq_rate,
    launch_bwd_D,
    launch_bwd_dkv,
    launch_bwd_dq,
)
from mlx_train_perf.attention.kernel.source import (
    build_bwd_D_source,
    build_bwd_dkv_mma_source,
    build_bwd_dkv_source,
    build_bwd_dq_mma_source,
    build_bwd_dq_source,
)
from mlx_train_perf.attention.reference import flash_attention_reference, math_attention
from mlx_train_perf.attention.segments import PackedMask
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


def _packed_layout(seg_lens: list[int], b: int) -> tuple[mx.array, mx.array]:
    """Build the (B, N) int32 seg_id/seg_start buffers for a fixed list of segment lengths,
    shared across every batch row (mirrors test_attention_kernel_fwd.py::_packed_layout).
    seg_id is contiguous ascending ids (0,0,..,1,1,..); seg_start is each position's
    segment-start index (non-decreasing) -- the PackedMask contract."""
    seg_id_row: list[int] = []
    seg_start_row: list[int] = []
    start = 0
    for sid, ln in enumerate(seg_lens):
        seg_id_row += [sid] * ln
        seg_start_row += [start] * ln
        start += ln
    seg_id = mx.array([seg_id_row] * b, dtype=mx.int32)        # (b, n) contiguous
    seg_start = mx.array([seg_start_row] * b, dtype=mx.int32)
    return seg_id, seg_start


def _packed_mask(
    seg_id: mx.array | None, seg_start: mx.array | None
) -> PackedMask | None:
    """PackedMask from an optional both-or-neither seg pair (None -> pure causal)."""
    if seg_id is None or seg_start is None:
        return None
    return PackedMask(seg_id=seg_id, seg_start=seg_start)


def _dq_oracle(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    segments: PackedMask | None = None,
) -> mx.array:
    """Exact dQ oracle: the vector-Jacobian product of `math_attention` w.r.t. q ONLY,
    with the same random cotangent `cot` the kernel path consumes as dO. No readout
    projection -- `mx.vjp` gives the exact autodiff dQ. `segments` (0.4.0) makes it the
    block-diagonal-causal packed dQ oracle."""
    _, vjps = mx.vjp(
        lambda q_: math_attention(q_, k, v, scale=scale, causal=causal, segments=segments),
        [q], [cot],
    )
    return vjps[0]


def _dq_kernel(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    rate_macs_per_s: float | None = None, flip_causal: bool = False,
    variant: str = "scalar", d_slab: int | None = None,
    force_ranges: list[tuple[int, int]] | None = None,
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
) -> mx.array:
    """The kernel dQ path: forward reference gives (O, L); T7's `launch_bwd_D` gives D from
    (dO, O); `launch_bwd_dq` consumes q/k/v/dO/L/D. `variant`/`d_slab` default to the scalar
    body (unchanged for every existing caller); `variant="mma"` selects the T9b rung-B1 body.
    `force_ranges` is the TEST-ONLY split-forcing seam (`_force_ranges`) -- the production
    planner never splits these tiny packed-regime shapes. `seg_id`/`seg_start` (0.4.0, both or
    neither) switch on PACKED block-diagonal-causal attention: the forward reference and the
    kernel both isolate to same-segment causal keys."""
    segments = _packed_mask(seg_id, seg_start)
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=causal, segments=segments)
    d_arr = launch_bwd_D(cot, o)
    return launch_bwd_dq(
        q, k, v, cot, lse, d_arr, scale=scale, causal=causal,
        rate_macs_per_s=rate_macs_per_s, _flip_causal=flip_causal,
        variant=variant, d_slab=d_slab, _force_ranges=force_ranges,
        seg_id=seg_id, seg_start=seg_start,
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
    # Force 4 disjoint dispatches over n=257 via the TEST-ONLY `_force_ranges` seam -- the
    # production planner would emit a single range at this tiny packed-regime shape.
    split = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True,
        force_ranges=[(0, 80), (80, 160), (160, 240), (240, 257)],
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
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    segments: PackedMask | None = None,
) -> tuple[mx.array, mx.array]:
    """Exact dK, dV oracle: the vector-Jacobian product of `math_attention` w.r.t. (k, v)
    with the same random cotangent `cot` the kernel path consumes as dO. No readout projection
    -- `mx.vjp` gives the exact autodiff dK/dV, grouped over the GQA q-head groups by autodiff
    (matching the kernel's in-owner whole-group accumulation). `segments` (0.4.0) makes it the
    block-diagonal-causal packed dK/dV oracle."""
    _, vjps = mx.vjp(
        lambda k_, v_: math_attention(q, k_, v_, scale=scale, causal=causal, segments=segments),
        [k, v], [cot],
    )
    return vjps[0], vjps[1]  # dK, dV


def _dkv_kernel(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float, causal: bool,
    rate_macs_per_s: float | None = None, flip_causal: bool = False,
    variant: str = "scalar", d_slab: int | None = None,
    force_ranges: list[tuple[int, int]] | None = None,
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
    segment_bound: bool = True, break_early: bool = False,
) -> tuple[mx.array, mx.array]:
    """The kernel dK/dV path: forward reference gives (O, L); T7's `launch_bwd_D` gives D from
    (dO, O); `launch_bwd_dkv` consumes q/k/v/dO/L/D and returns the chained (dK, dV).
    `variant`/`d_slab` default to the scalar body (unchanged for every existing caller);
    `variant="mma"` selects the T9b rung-B2 key-major MMA body. `force_ranges` is the
    TEST-ONLY split-forcing seam (`_force_ranges`) -- the production planner never splits
    these tiny packed-regime shapes. `seg_id`/`seg_start` (0.4.0, both or neither) switch on
    PACKED block-diagonal-causal attention: the forward reference and the kernel both isolate
    to same-segment causal (query, key) pairs. `segment_bound`/`break_early` (0.5.0 T3) pass
    straight through to `launch_bwd_dkv`: the mma variant honors them (spec D1/D5), the scalar
    variant ignores them entirely (D3 -- it stays the assumption-free oracle)."""
    segments = _packed_mask(seg_id, seg_start)
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=causal, segments=segments)
    d_arr = launch_bwd_D(cot, o)
    return launch_bwd_dkv(
        q, k, v, cot, lse, d_arr, scale=scale, causal=causal,
        rate_macs_per_s=rate_macs_per_s, _flip_causal=flip_causal,
        variant=variant, d_slab=d_slab, _force_ranges=force_ranges,
        seg_id=seg_id, seg_start=seg_start,
        segment_bound=segment_bound, break_early=break_early,
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

    # A 4-range chained plan via the TEST-ONLY `_force_ranges` seam (the production planner
    # never splits this tiny packed-regime shape -- see test_attention_launch_plan.py).
    forced = [(0, 30), (30, 60), (60, 90), (90, 96)]

    dk_k, dv_k = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, force_ranges=forced)
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
    # A 4-range chained plan via the TEST-ONLY `_force_ranges` seam (the production planner
    # never splits this tiny packed-regime shape).
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True,
        force_ranges=[(0, 30), (30, 60), (60, 90), (90, 96)],
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
    # Force several disjoint mma dispatches over n=257 (boundaries deliberately NOT
    # 32-aligned -- per-row independence) via the TEST-ONLY `_force_ranges` seam.
    split = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        force_ranges=[(0, 80), (80, 160), (160, 240), (240, 257)],
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


# plan_dkv_dispatches semantics (0.3.0 buffer-model planner, incl. the 32-row block
# alignment the mma variant needs) are covered in tests/test_attention_launch_plan.py.


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

    # A 3-range 32-ALIGNED chained plan via the TEST-ONLY `_force_ranges` seam (the
    # production planner never splits this tiny packed-regime shape).
    ranges = [(0, 32), (32, 64), (64, 96)]
    assert len(ranges) >= 3
    for lo, _hi in ranges:
        assert lo % 32 == 0                       # every range starts on a 32-row query block

    dk_k, dv_k = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", force_ranges=ranges
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
    # A 3-range 32-ALIGNED chained plan via the TEST-ONLY `_force_ranges` seam (the
    # production planner never splits this tiny packed-regime shape).
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        force_ranges=[(0, 32), (32, 64), (64, 96)],
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


# =======================================================================================
# T9b Step 3 (GRADUATION) -- wire the measured MMA backward into production: per-kernel
# calibrated rates (probe-what-you-rate PER kernel, the shared-rate design retired because
# the two throughputs differ measurably), a backward dispatch table, and the api vjp routing
# the table-selected variants. The dispatch-table arithmetic is covered in
# test_attention_kernel_dispatch.py; this section covers the RATE split and the api wiring.
# =======================================================================================

# ---------------------------------------------------------------------------------------
# Per-kernel rate calibration (DEFAULT lane -- a fake `_bwd_dq_kernel`/`_bwd_dkv_kernel`
# fabricates zero-cost output arrays instead of touching Metal, exactly as the forward's
# test_calibrated_fwd_rate_probes_the_selected_variant_and_d_slab does).
# ---------------------------------------------------------------------------------------


def _fake_kernel(
    *, inputs: list[mx.array], template: list[tuple[str, mx.Dtype]],  # noqa: ARG001
    grid: tuple[int, int, int], threadgroup: tuple[int, int, int],  # noqa: ARG001
    output_shapes: list[tuple[int, ...]], output_dtypes: list[mx.Dtype],
) -> list[mx.array]:
    """A Metal-free stand-in for a built kernel: returns zeros of the requested output shapes,
    so the calibration ramp runs its dispatch/timing loop without a GPU (DEFAULT lane)."""
    return [
        mx.zeros(shape, dtype=dtype)
        for shape, dtype in zip(output_shapes, output_dtypes, strict=True)
    ]


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_calibrated_bwd_dq_rate_probes_the_selected_variant_and_d_slab(
    monkeypatch: pytest.MonkeyPatch, packed: bool
) -> None:
    """Probe-what-you-rate for dQ: `calibrated_bwd_dq_rate`'s `measure()` must build the SAME
    (variant, d_slab, packed) dQ kernel the launcher will dispatch -- rating one variant while
    dispatching another sizes the query-row split from the wrong rate. Spies on `_bwd_dq_kernel`
    (the construction seam) with a Metal-free fake and asserts the recorded (variant, d_slab,
    packed) matches what the caller selected. The `packed` arm (0.4.0) proves a packed-keyed
    dQ rate builds the PACKED dQ kernel."""
    monkeypatch.setattr(bwd_launch, "_BWD_DQ_RATE_CACHE", {})
    calls: list[tuple[str, int | None, bool]] = []

    def fake_dq_kernel(
        head_dim: int, causal: bool, flip_causal: bool, variant: str,  # noqa: ARG001
        d_slab: int | None, packed: bool = False,
    ) -> object:
        calls.append((variant, d_slab, packed))
        return _fake_kernel

    monkeypatch.setattr(bwd_launch, "_bwd_dq_kernel", fake_dq_kernel)
    bwd_launch.calibrated_bwd_dq_rate(
        head_dim=64, dtype=mx.float32, b=1, hq=4, hkv=4, n=256, causal=True,
        tile=TileShape(variant="mma", d_slab=64), packed=packed,
    )

    assert calls == [("mma", 64, packed)], (
        f"dQ calibration built {calls}, but the caller selected variant='mma' d_slab=64 "
        f"packed={packed}"
    )


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_calibrated_bwd_dkv_rate_probes_the_selected_variant_and_d_slab(
    monkeypatch: pytest.MonkeyPatch, packed: bool
) -> None:
    """Probe-what-you-rate for dK/dV: `calibrated_bwd_dkv_rate`'s `measure()` must build the
    SAME (variant, d_slab, packed) dK/dV kernel the launcher will dispatch (mirrors the dQ
    test). The `packed` arm (0.4.0) proves a packed-keyed dK/dV rate builds the PACKED kernel."""
    monkeypatch.setattr(bwd_launch, "_BWD_DKV_RATE_CACHE", {})
    calls: list[tuple[str, int | None, bool]] = []

    def fake_dkv_kernel(
        head_dim: int, causal: bool, flip_causal: bool, variant: str,  # noqa: ARG001
        d_slab: int | None, packed: bool = False,
    ) -> object:
        calls.append((variant, d_slab, packed))
        return _fake_kernel

    monkeypatch.setattr(bwd_launch, "_bwd_dkv_kernel", fake_dkv_kernel)
    bwd_launch.calibrated_bwd_dkv_rate(
        head_dim=64, dtype=mx.float32, b=1, hq=4, hkv=4, n=256, causal=True,
        tile=TileShape(variant="mma", d_slab=64), packed=packed,
    )

    assert calls == [("mma", 64, packed)], (
        f"dK/dV calibration built {calls}, but the caller selected variant='mma' d_slab=64 "
        f"packed={packed}"
    )


def test_bwd_dq_rate_cache_key_separates_by_variant_and_d_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dQ rate cache key must include (variant, d_slab) (mirrors `_FWD_RATE_CACHE`): two mma
    dQ calls at the same shape but different `d_slab` are DIFFERENT kernels (different rate) and
    must never collapse onto one cache entry."""
    monkeypatch.setattr(bwd_launch, "_BWD_DQ_RATE_CACHE", {})
    monkeypatch.setattr(bwd_launch, "_bwd_dq_kernel", lambda *a, **k: _fake_kernel)  # noqa: ARG005

    args = {
        "head_dim": 64, "dtype": mx.float32, "b": 1, "hq": 4, "hkv": 4,
        "n": 256, "causal": True,
    }
    bwd_launch.calibrated_bwd_dq_rate(**args, tile=TileShape(variant="mma", d_slab=64))
    bwd_launch.calibrated_bwd_dq_rate(**args, tile=TileShape(variant="mma", d_slab=128))

    keys = list(bwd_launch._BWD_DQ_RATE_CACHE)
    assert len(keys) == 2                          # two distinct cache entries, not one collision
    # key tail is (variant, d_slab, packed); both entries are non-packed here.
    assert {k[-3:] for k in keys} == {("mma", 64, False), ("mma", 128, False)}


def test_bwd_dq_and_dkv_rates_use_independent_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dQ and dK/dV rates are calibrated INDEPENDENTLY: each probes its own kernel and writes
    its own cache, so the two never share a number (the retired shared-rate design's whole
    failure mode). Discrimination: a value seeded into the dQ cache must NOT satisfy a dK/dV
    call at the same shape, and each rate probes only its own kernel."""
    monkeypatch.setattr(bwd_launch, "_BWD_DQ_RATE_CACHE", {})
    monkeypatch.setattr(bwd_launch, "_BWD_DKV_RATE_CACHE", {})
    dq_built: list[tuple[str, int | None]] = []
    dkv_built: list[tuple[str, int | None]] = []

    def fake_dq_kernel(
        head_dim: int, causal: bool, flip_causal: bool, variant: str,  # noqa: ARG001
        d_slab: int | None, packed: bool = False,  # noqa: ARG001
    ) -> object:
        dq_built.append((variant, d_slab))
        return _fake_kernel

    def fake_dkv_kernel(
        head_dim: int, causal: bool, flip_causal: bool, variant: str,  # noqa: ARG001
        d_slab: int | None, packed: bool = False,  # noqa: ARG001
    ) -> object:
        dkv_built.append((variant, d_slab))
        return _fake_kernel

    monkeypatch.setattr(bwd_launch, "_bwd_dq_kernel", fake_dq_kernel)
    monkeypatch.setattr(bwd_launch, "_bwd_dkv_kernel", fake_dkv_kernel)
    args = {
        "head_dim": 64, "dtype": mx.float32, "b": 1, "hq": 4, "hkv": 4,
        "n": 256, "causal": True,
    }
    tile = TileShape(variant="mma", d_slab=128)
    dq_rate = calibrated_bwd_dq_rate(**args, tile=tile)
    dkv_rate = calibrated_bwd_dkv_rate(**args, tile=tile)

    assert dq_built == [("mma", 128)]              # dQ rate probed the dQ kernel only
    assert dkv_built == [("mma", 128)]             # dK/dV rate probed the dK/dV kernel only
    assert len(bwd_launch._BWD_DQ_RATE_CACHE) == 1  # separate caches, one entry each
    assert len(bwd_launch._BWD_DKV_RATE_CACHE) == 1
    assert dq_rate > 0.0
    assert dkv_rate > 0.0


# ---------------------------------------------------------------------------------------
# api vjp wiring (PER-TEST @pytest.mark.metal). The kernel-backward vjp must route
# launch_bwd_dq / launch_bwd_dkv with the variant/d_slab/rate the backward dispatch table
# selects, calibrated ONCE at construction time (outside the traced region). Small shapes
# only (n <= 512) -- the controller re-runs the flagship end-to-end after the commit.
# ---------------------------------------------------------------------------------------


@pytest.mark.metal
def test_kernel_backward_routes_the_table_selected_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T9b Step 3 wiring: `api.py`'s kernel vjp must call `select_bwd_tiles(n, head_dim)` and
    route launch_bwd_dq / launch_bwd_dkv with the SELECTED (variant, d_slab), each with its own
    construction-time calibrated rate closure-captured. Spies on both launchers to record the
    variant/d_slab/rate the api passed, and checks them against the table's own selection for
    this shape. Pre-fix the api passed neither variant nor d_slab (scalar default) and one
    shared rate -- post-fix each must equal `select_bwd_tiles`'s own answer with a real rate."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, _cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=63)

    seen_dq: list[tuple[str | None, int | None, float | None]] = []
    seen_dkv: list[tuple[str | None, int | None, float | None]] = []
    real_dq, real_dkv = api.launch_bwd_dq, api.launch_bwd_dkv

    def spy_dq(*args: object, **kwargs: object) -> object:
        seen_dq.append(
            (kwargs.get("variant"), kwargs.get("d_slab"), kwargs.get("rate_macs_per_s"))  # type: ignore[arg-type]
        )
        return real_dq(*args, **kwargs)  # type: ignore[arg-type]

    def spy_dkv(*args: object, **kwargs: object) -> object:
        seen_dkv.append(
            (kwargs.get("variant"), kwargs.get("d_slab"), kwargs.get("rate_macs_per_s"))  # type: ignore[arg-type]
        )
        return real_dkv(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api, "launch_bwd_dq", spy_dq)
    monkeypatch.setattr(api, "launch_bwd_dkv", spy_dkv)

    def loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="kernel").sum()

    mx.eval(*mx.grad(loss, argnums=(0, 1, 2))(q, k, v))

    dq_tile, dkv_tile = select_bwd_tiles(n, d)
    assert seen_dq, "launch_bwd_dq never fired -- the backward is not kernel-backed"
    assert seen_dkv, "launch_bwd_dkv never fired -- the backward is not kernel-backed"
    v_dq, s_dq, r_dq = seen_dq[0]
    assert (v_dq, s_dq) == (dq_tile.variant, dq_tile.d_slab), (
        f"dQ launched variant={v_dq} d_slab={s_dq}, table selected "
        f"variant={dq_tile.variant} d_slab={dq_tile.d_slab}"
    )
    assert r_dq is not None, "dQ rate is None -- not the construction-time calibrated float"
    assert r_dq > 0.0
    v_dkv, s_dkv, r_dkv = seen_dkv[0]
    assert (v_dkv, s_dkv) == (dkv_tile.variant, dkv_tile.d_slab), (
        f"dK/dV launched variant={v_dkv} d_slab={s_dkv}, table selected "
        f"variant={dkv_tile.variant} d_slab={dkv_tile.d_slab}"
    )
    assert r_dkv is not None, "dK/dV rate is None -- not the construction-time calibrated float"
    assert r_dkv > 0.0


# ---------------------------------------------------------------------------------------
# 0022f: drop-diagonal flip-perturbation hardening. The flipped-triangle perturbations
# above are GROSS (wrong half-plane); a diagonal OFF-BY-ONE (`kk < row` for the row-major
# dQ bodies, `i > key` for the key-major dK/dV bodies) is the subtler named-bug-site the
# suite must also provably detect. The perturbed kernels are built directly from the
# source builders (a test-only flag; the production launchers never expose it).
# ---------------------------------------------------------------------------------------


def test_build_bwd_dq_source_drop_diagonal_perturbation() -> None:
    s = build_bwd_dq_source(64, causal=True, drop_diagonal=True)
    assert "kk < row" in s
    assert "kk <= row" not in s
    s_mma = build_bwd_dq_mma_source(64, causal=True, drop_diagonal=True)
    assert "kk < row" in s_mma
    assert "kk <= row" not in s_mma


def test_build_bwd_dkv_source_drop_diagonal_perturbation() -> None:
    # KEY-major bodies: causal keep is `i >= key`, so dropping the diagonal is `i > key`.
    s = build_bwd_dkv_source(64, causal=True, drop_diagonal=True)
    assert "i > key" in s
    assert "i >= key" not in s
    s_mma = build_bwd_dkv_mma_source(64, causal=True, drop_diagonal=True)
    assert "i > key" in s_mma
    assert "i >= key" not in s_mma


def test_build_bwd_sources_reject_drop_diagonal_misuse() -> None:
    for builder in (
        build_bwd_dq_source, build_bwd_dq_mma_source,
        build_bwd_dkv_source, build_bwd_dkv_mma_source,
    ):
        with pytest.raises(ValueError, match="drop_diagonal"):
            builder(64, causal=False, drop_diagonal=True)
        with pytest.raises(ValueError, match="drop_diagonal"):
            builder(64, causal=True, flip_causal=True, drop_diagonal=True)


def _o_lse_darr(
    q: mx.array, k: mx.array, v: mx.array, cot: mx.array, *, scale: float
) -> tuple[mx.array, mx.array]:
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=True)
    return lse, launch_bwd_D(cot, o)


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dq_drop_diagonal_perturbation_fails_parity(variant: str) -> None:
    b, hq, hkv, n, d = 2, 4, 2, 16, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=43)
    lse, darr = _o_lse_darr(q, k, v, cot, scale=scale)

    builder = build_bwd_dq_mma_source if variant == "mma" else build_bwd_dq_source
    kernel = mx.fast.metal_kernel(
        name=f"mtp_test_dropdiag_dq_{variant}",
        input_names=["q", "k", "v", "d_o", "lse", "d_arr", "qoffs", "scale_in"],
        output_names=["dq_out"],
        source=builder(d, causal=True, drop_diagonal=True),
    )
    scale_in = mx.array([scale], dtype=mx.float32)
    dq_bad = bwd_launch._dispatch_bwd_dq_range(
        kernel, q, k, v, cot, lse, darr, scale_in, r0=0, r1=n, variant=variant,
    )
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dq_bad, dq_ref)

    # The backward reuses the forward's precomputed L/D, so dropping the diagonal removes a
    # term from a plain accumulation (no 0/0) -- dQ stays FINITE and must diverge outright.
    assert bool(mx.isfinite(dq_bad).all().item()), "drop-diagonal dQ went non-finite"
    diff = mx.abs(dq_bad.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    assert diff > 1e-2, (
        f"drop-diagonal dQ ({variant}) matched the causal oracle (diff={diff:.3e}) -- "
        "the parity grid cannot detect a diagonal off-by-one"
    )


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dkv_drop_diagonal_perturbation_fails_parity(variant: str) -> None:
    b, hq, hkv, n, d = 2, 4, 2, 16, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=53)
    lse, darr = _o_lse_darr(q, k, v, cot, scale=scale)

    builder = build_bwd_dkv_mma_source if variant == "mma" else build_bwd_dkv_source
    kernel = mx.fast.metal_kernel(
        name=f"mtp_test_dropdiag_dkv_{variant}",
        input_names=[
            "q", "k", "v", "d_o", "lse", "d_arr", "dk_in", "dv_in", "qoffs", "scale_in",
        ],
        output_names=["dk_out", "dv_out"],
        source=builder(d, causal=True, drop_diagonal=True),
    )
    scale_in = mx.array([scale], dtype=mx.float32)
    dk0 = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dv0 = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dk_bad, dv_bad = bwd_launch._dispatch_bwd_dkv_range(
        kernel, q, k, v, cot, lse, darr, dk0, dv0, scale_in, q_lo=0, q_hi=n,
        variant=variant,
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True)
    mx.eval(dk_bad, dv_bad, dk_ref, dv_ref)

    # dK/dV likewise reuse precomputed L/D -- finite, so they must diverge outright.
    dkv_finite = bool(mx.isfinite(dk_bad).all().item()) and bool(
        mx.isfinite(dv_bad).all().item()
    )
    assert dkv_finite, "drop-diagonal dK/dV went non-finite"
    diff = max(
        mx.abs(dk_bad.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item(),
        mx.abs(dv_bad.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item(),
    )
    assert diff > 1e-2, (
        f"drop-diagonal dK/dV ({variant}) matched the causal oracle (diff={diff:.3e}) -- "
        "the parity grid cannot detect a diagonal off-by-one"
    )


# =======================================================================================
# 0.4.0 T3 -- PACKED backward parity (block-diagonal-causal segments). Each packed backward
# body's keep predicate (dQ scalar/mma, dK/dV scalar/mma) must reproduce the autodiff gradient
# of the block-diagonal oracle (`math_attention(..., segments=PackedMask(...))`, the Task-1
# packed gradient oracle). The forward reference the kernel path consumes is the packed
# `flash_attention_reference(..., segments=...)`; D and the launcher take the seg buffers. The
# `packed=False` default is untouched (every causal test above still runs the unchanged body).
# =======================================================================================

# (n, seg_lens): 2-seg and 5-seg layouts with mid-block boundaries (NOT aligned to the mma
# 32-row query-block size), mirroring test_attention_kernel_fwd.py::_PACKED_CASES exactly --
# the case kv_lo (dQ) + the per-element predicate (dK/dV) must isolate segments exactly.
_PACKED_CASES = [
    (256, [100, 156]),                       # 2-seg, boundary at 100 (mid-block)
    (256, [40, 60, 50, 66, 40]),             # 5-seg, several mid-block boundaries
    (1024, [500, 524]),                      # 2-seg at n=1024
    (1024, [200, 312, 130, 198, 184]),       # 5-seg at n=1024
]

# Measured worsts over the packed backward grid (mlx 0.32.0, M1 Max, seed 61 dQ / 62 dK/dV,
# variants scalar AND mma x _PACKED_CASES x head_dim {64,128} x dtype {fp32,bf16}). The packed
# bodies mask MORE (query, key) pairs than the causal bodies but reuse the identical fp32
# in-register reduction structure, so an fp32 diff is pure reduction-order noise; bf16 carries a
# single common-mode store-ULP rounding (the forward O and dK/dV each cast down once). Pins are
# set INDEPENDENTLY (measure-first), never by widening or inheriting the causal _TOL_DQ/_TOL_DKV:
#   dQ fp32:  worst 3.734404e-06 (n256 segs5 d128; scalar == mma at the worst element) -> 8e-6
#             (~2.1x). Slightly HOTTER than the causal dQ fp32 worst 2.46e-6 -- the packed grid
#             runs a larger reduction (n up to 1024, 5-segment layouts) -- so honestly a touch
#             looser than the causal 5e-6 pin, not padded.
#   dQ bf16:  worst 1.562500e-02 == one bf16 ULP (2^-6) at a dQ magnitude near 2-4 (quantized
#             rounding, not accumulation drift) -> 3e-2, the SAME ULP-aware ceiling as the causal
#             dQ bf16 pin (bounded below the 2-ULP value 3.125e-2; widen toward 2 ULP with a note,
#             never past it).
#   dKV fp32: worst dV 1.287460e-05 / dK 7.867813e-06 (dV harsher -- it sums P over EVERY
#             causally-allowed query) -> 3e-5 (~2.3x the dV worst), both variants (the mma dV worst
#             1.2875e-5 matches the scalar's, and both exceed the causal mma worst 9.06e-6, so the
#             packed pin is honestly above the causal mma 2e-5 pin).
#   dKV bf16: worst dV/dK 3.125000e-02 == one bf16 ULP (2^-5) -> 1e-1, the SAME ULP-aware ceiling
#             as the causal dK/dV bf16 pin (bounded below the 2-ULP value 0.125).
_TOL_PACKED_DQ = {
    "scalar": {mx.float32: 8e-6, mx.bfloat16: 3e-2},
    "mma": {mx.float32: 8e-6, mx.bfloat16: 3e-2},
}
_TOL_PACKED_DKV = {
    "scalar": {mx.float32: 3e-5, mx.bfloat16: 1e-1},
    "mma": {mx.float32: 3e-5, mx.bfloat16: 1e-1},
}


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
@pytest.mark.parametrize(("n", "seg_lens"), _PACKED_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dq_packed_parity_vs_block_diagonal_oracle(
    variant: str, n: int, seg_lens: list[int], head_dim: int, dtype: mx.Dtype
) -> None:
    b, hq, hkv = 2, 4, 2                                        # GQA group_size 2
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype, seed=61)
    seg_id, seg_start = _packed_layout(seg_lens, b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    dq_k = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dq_k, dq_ref)

    diff = mx.abs(dq_k.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    print(
        f"[dQ-packed {variant} {['fp32','bf16'][dtype==mx.bfloat16]} n{n} "
        f"segs{len(seg_lens)} d{head_dim}] diff={diff:.6e}"
    )
    assert diff < _TOL_PACKED_DQ[variant][dtype], f"packed dQ ({variant}) diff {diff}"


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
@pytest.mark.parametrize(("n", "seg_lens"), _PACKED_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_dkv_packed_parity_vs_block_diagonal_oracle(
    variant: str, n: int, seg_lens: list[int], head_dim: int, dtype: mx.Dtype
) -> None:
    b, hq, hkv = 2, 4, 2                                        # GQA group_size 2
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype, seed=62)
    seg_id, seg_start = _packed_layout(seg_lens, b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    dk_k, dv_k = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(
        f"[dKV-packed {variant} {['fp32','bf16'][dtype==mx.bfloat16]} n{n} "
        f"segs{len(seg_lens)} d{head_dim}] dK={d_dk:.6e} dV={d_dv:.6e}"
    )
    assert d_dk < _TOL_PACKED_DKV[variant][dtype], f"packed dK ({variant}) diff {d_dk}"
    assert d_dv < _TOL_PACKED_DKV[variant][dtype], f"packed dV ({variant}) diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dq_single_segment_packed_matches_causal_bitwise(variant: str) -> None:
    """One segment spanning the whole row (seg_id/seg_start all 0) makes the same-segment term
    uniformly true and kv_lo == 0, so the packed dQ kernel loops the IDENTICAL key set in the
    IDENTICAL order as the pure-causal kernel -> BIT-IDENTICAL dQ. N=129 exercises a tail block
    that is not 32-aligned (the mma over-hang path)."""
    b, hq, hkv, n, d = 2, 4, 2, 129, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=63)
    seg_id = mx.zeros((b, n), dtype=mx.int32)
    seg_start = mx.zeros((b, n), dtype=mx.int32)

    dq_c = _dq_kernel(q, k, v, cot, scale=scale, causal=True, variant=variant)
    dq_p = _dq_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    mx.eval(dq_c, dq_p)
    assert mx.array_equal(dq_c, dq_p).item(), f"single-segment packed dQ ({variant}) != causal dQ"


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dkv_single_segment_packed_matches_causal_bitwise(variant: str) -> None:
    """One segment spanning the whole row (seg_id/seg_start all 0) makes the same-segment term
    uniformly true, so the packed dK/dV kernel loops the IDENTICAL (query, key) set in the
    IDENTICAL order as the pure-causal kernel -> BIT-IDENTICAL dK/dV. N=129 exercises a tail key
    block that is not 32-aligned (the mma over-hang path on the key axis)."""
    b, hq, hkv, n, d = 2, 4, 2, 129, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=64)
    seg_id = mx.zeros((b, n), dtype=mx.int32)
    seg_start = mx.zeros((b, n), dtype=mx.int32)

    dk_c, dv_c = _dkv_kernel(q, k, v, cot, scale=scale, causal=True, variant=variant)
    dk_p, dv_p = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    mx.eval(dk_c, dv_c, dk_p, dv_p)
    assert mx.array_equal(dk_c, dk_p).item(), f"single-segment packed dK ({variant}) != causal dK"
    assert mx.array_equal(dv_c, dv_p).item(), f"single-segment packed dV ({variant}) != causal dV"


@pytest.mark.metal
def test_dkv_packed_chained_matches_oracle_when_chaining_is_forced() -> None:
    """review-tests High (the REQUIRED chained proof extended to a PACKED layout): a
    chained-vs-single self-comparison cannot catch a systematic carry bug present in EVERY split
    -- so force a >=3-range chained plan (a tiny artificial rate) at a small packed N and run the
    REAL multi-dispatch mma code path against the packed autodiff oracle. The chained fp32
    accumulator, not just its own consistency, must meet the ground-truth block-diagonal gradient.
    The forced ranges are 32-aligned (the mma variant's block-alignment contract), and the segment
    boundary (48) is mid-block so a range split lands inside a segment."""
    b, hq, hkv, n, d = 1, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=65)
    seg_id, seg_start = _packed_layout([48, 48], b)            # 2-seg, mid-block boundary at 48
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    ranges = [(0, 32), (32, 64), (64, 96)]
    assert len(ranges) >= 3
    for lo, _hi in ranges:
        assert lo % 32 == 0                            # every range starts on a 32-row block

    dk_k, dv_k = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", force_ranges=ranges,
        seg_id=seg_id, seg_start=seg_start,
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k - dk_ref).max().item()
    d_dv = mx.abs(dv_k - dv_ref).max().item()
    print(f"[dKV-packed-mma forced-chain] ranges={ranges} dK={d_dk:.6e} dV={d_dv:.6e}")
    assert d_dk < _TOL_PACKED_DKV["mma"][mx.float32], f"chained packed dK-mma vs oracle diff {d_dk}"
    assert d_dv < _TOL_PACKED_DKV["mma"][mx.float32], f"chained packed dV-mma vs oracle diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dkv_packed_chained_dispatches_equal_single_dispatch(variant: str) -> None:
    """Chained multi-range PACKED dispatches accumulate dK/dV in a FIXED order, each range seeded
    from the prior's fp32 output -- so a >=3-range split must be BIT-identical to a single [0, n)
    dispatch (fp32->fp32 store/reload is lossless). The predicate zeros cross-segment
    contributions identically regardless of the range split. Ranges are 32-aligned (the mma block
    contract); the scalar variant is per-key so any split works, but 32-alignment is a valid split
    for it too."""
    b, hq, hkv, n, d = 2, 4, 2, 96, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=66)
    seg_id, seg_start = _packed_layout([40, 56], b)            # 2-seg, mid-block boundary at 40
    single_dk, single_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant=variant,
        force_ranges=[(0, 32), (32, 64), (64, 96)],
        seg_id=seg_id, seg_start=seg_start,
    )
    mx.eval(single_dk, single_dv, split_dk, split_dv)
    assert mx.array_equal(single_dk, split_dk).item(), f"packed dK ({variant}) split != single"
    assert mx.array_equal(single_dv, split_dv).item(), f"packed dV ({variant}) split != single"


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dq_packed_flip_segments_breaks_parity(variant: str) -> None:
    """Deliberate cross-segment contamination: build the packed dQ kernel with the segment
    equality inverted (`flip_segments`, the segment analogue of `flip_causal`). Segment-1 rows
    attend to earlier segment-0 keys under the flipped predicate, so their dQ MUST diverge from
    the block-diagonal oracle -- if this ever matched, the packed parity grid could not detect a
    segment-masking bug. The backward reuses the correct block-diagonal L/D, so dQ stays FINITE
    (masked keys take the p=0 branch, no exp of -inf) and must diverge outright."""
    b, hq, hkv, d = 2, 4, 2, 64
    n1, n2 = 48, 48
    n = n1 + n2
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=67)
    seg_id, seg_start = _packed_layout([n1, n2], b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=True, segments=pm)
    d_arr = launch_bwd_D(cot, o)

    builder = build_bwd_dq_mma_source if variant == "mma" else build_bwd_dq_source
    kernel = mx.fast.metal_kernel(
        name=f"mtp_test_flipseg_dq_{variant}",
        input_names=["q", "k", "v", "d_o", "lse", "d_arr", "qoffs", "scale_in",
                     "seg_id", "seg_start"],
        output_names=["dq_out"],
        source=builder(d, causal=True, packed=True, flip_segments=True),
    )
    scale_in = mx.array([scale], dtype=mx.float32)
    dq_bad = bwd_launch._dispatch_bwd_dq_range(
        kernel, q, k, v, cot, lse, d_arr, scale_in, r0=0, r1=n, variant=variant,
        seg_id=seg_id, seg_start=seg_start,
    )
    dq_ref = _dq_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dq_bad, dq_ref)

    assert bool(mx.isfinite(dq_bad).all().item()), "flip-segments dQ went non-finite"
    diff = mx.abs(dq_bad.astype(mx.float32) - dq_ref.astype(mx.float32)).max().item()
    assert diff > 1e-2, (
        f"flip-segments dQ ({variant}) matched the block-diagonal oracle (diff={diff:.3e}) -- "
        "the packed parity grid cannot detect a segment-masking bug"
    )


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_dkv_packed_flip_segments_breaks_parity(variant: str) -> None:
    """Deliberate cross-segment contamination for dK/dV: build the packed kernel with the segment
    equality inverted (`flip_segments`). Segment-0 keys accumulate from later segment-1 queries
    under the flipped predicate, so BOTH dK AND dV must diverge from the block-diagonal oracle --
    if this ever matched, the packed parity grid could not detect a segment-masking bug. The
    backward reuses the correct block-diagonal L/D, so dK/dV stay FINITE and must diverge each."""
    b, hq, hkv, d = 2, 4, 2, 64
    n1, n2 = 48, 48
    n = n1 + n2
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=68)
    seg_id, seg_start = _packed_layout([n1, n2], b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    o, lse = flash_attention_reference(q, k, v, scale=scale, causal=True, segments=pm)
    d_arr = launch_bwd_D(cot, o)

    builder = build_bwd_dkv_mma_source if variant == "mma" else build_bwd_dkv_source
    kernel = mx.fast.metal_kernel(
        name=f"mtp_test_flipseg_dkv_{variant}",
        input_names=["q", "k", "v", "d_o", "lse", "d_arr", "dk_in", "dv_in", "qoffs", "scale_in",
                     "seg_id", "seg_start"],
        output_names=["dk_out", "dv_out"],
        source=builder(d, causal=True, packed=True, flip_segments=True),
    )
    scale_in = mx.array([scale], dtype=mx.float32)
    dk0 = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dv0 = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dk_bad, dv_bad = bwd_launch._dispatch_bwd_dkv_range(
        kernel, q, k, v, cot, lse, d_arr, dk0, dv0, scale_in, q_lo=0, q_hi=n,
        variant=variant, seg_id=seg_id, seg_start=seg_start,
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dk_bad, dv_bad, dk_ref, dv_ref)

    dkv_finite = bool(mx.isfinite(dk_bad).all().item()) and bool(
        mx.isfinite(dv_bad).all().item()
    )
    assert dkv_finite, "flip-segments dK/dV went non-finite"
    d_dk = mx.abs(dk_bad.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_bad.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    # flip_segments must break dK and dV parity SEPARATELY (each gradient independently
    # diverges -- a shared `or` could hide one gradient's masking bug behind the other).
    msg = (
        f"flip-segments dK/dV ({variant}) matched the block-diagonal oracle "
        f"(dK={d_dk:.3e}, dV={d_dv:.3e}) -- the packed parity grid cannot detect a segment bug"
    )
    assert d_dk > 1e-2, msg
    assert d_dv > 1e-2, msg


# ---------------------------------------------------------------------------------------
# 0.5.0 T2 -- `segment_bound`/`break_early` threaded through the `_bwd_dkv_kernel`
# `functools.cache` key AND the `mx.fast.metal_kernel` name (DEFAULT lane -- distinct
# cached objects need no GPU). mlx caches compiled kernels by NAME: a call with the same
# name but different source silently returns the FIRST compiled binary, so every
# source-varying flag must appear in both the Python cache key and the name string, or a
# later call with a different flag value would get back an earlier, wrong kernel.
#
# This test proves ONLY Python-cache distinctness (three different flag combinations
# produce three different cached `_MetalKernel` objects). It does NOT prove the emitted
# names differ, and it does NOT prove the compiled kernels behave differently on the GPU
# -- the load-bearing D5 guard (bit-identity + a can-fail `break_early` perturbation) is
# Task 3's metal-lane tests.
# ---------------------------------------------------------------------------------------


def test_dkv_kernel_cache_distinguishes_segment_bound_and_break_early() -> None:
    a = _bwd_dkv_kernel(64, True, False, "mma", None, True)                  # bounded default
    b = _bwd_dkv_kernel(64, True, False, "mma", None, True, False)           # unbounded
    c = _bwd_dkv_kernel(64, True, False, "mma", None, True, True, True)     # break_early
    assert a is not b and a is not c and b is not c  # noqa: PT018


# ---------------------------------------------------------------------------------------
# Checkpoint-A fix -- `_dkv_kernel_name` is the `_bwd_dkv_kernel` inline
# `mx.fast.metal_kernel(name=...)` string construction extracted to a module-level pure
# function, so the name<->source correspondence is directly testable (DEFAULT lane, no
# GPU) rather than only provable by dispatching. mlx caches compiled kernels BY NAME: a
# call with an unchanged name but different source silently returns the FIRST compiled
# binary, so a source-varying flag (segment_bound/break_early, mma variant only) MUST
# also vary the name.
# ---------------------------------------------------------------------------------------


def test_dkv_kernel_name_distinguishes_segment_bound_and_break_early() -> None:
    bound_be_false = _dkv_kernel_name(64, True, False, "mma", None, True, True, False)
    unbound_be_false = _dkv_kernel_name(64, True, False, "mma", None, True, False, False)
    bound_be_true = _dkv_kernel_name(64, True, False, "mma", None, True, True, True)

    assert bound_be_false != unbound_be_false
    assert bound_be_false != bound_be_true
    assert unbound_be_false != bound_be_true

    # `_nb` appears exactly when segment_bound=False; `_be` exactly when break_early=True.
    assert "_nb" not in bound_be_false
    assert "_be" not in bound_be_false
    assert "_nb" in unbound_be_false
    assert "_be" not in unbound_be_false
    assert "_nb" not in bound_be_true
    assert "_be" in bound_be_true


def test_dkv_kernel_name_scalar_variant_ignores_both_flags() -> None:
    """The scalar builder never sees `segment_bound`/`break_early` (D3 -- it stays the
    assumption-free oracle among kernel variants), so its name must be identical
    regardless of either flag's value -- distinguishing them would imply a scalar source
    that does not exist."""
    a = _dkv_kernel_name(64, True, False, "scalar", None, True, True, False)
    b = _dkv_kernel_name(64, True, False, "scalar", None, True, False, True)
    assert a == b


def test_bwd_dkv_kernel_rejects_segment_bound_or_break_early_on_scalar() -> None:
    """`segment_bound`/`break_early` are mma-only knobs (D3 -- the scalar body stays the
    assumption-free, predicate-only oracle and never accepts either). A caller passing
    `variant="scalar"` with `segment_bound=False` or `break_early=True` gets a
    `ValueError` rather than a silently-ignored flag."""
    with pytest.raises(ValueError, match="segment_bound/break_early apply to the mma variant only"):
        _bwd_dkv_kernel(64, True, False, "scalar", None, True, False, False)
    with pytest.raises(ValueError, match="segment_bound/break_early apply to the mma variant only"):
        _bwd_dkv_kernel(64, True, False, "scalar", None, True, True, True)


# ---------------------------------------------------------------------------------------
# 0.5.0 T3 -- packed dK/dV segment-end bound: parity, bit-identity, and RED-perturbation
# proofs (metal lane, spec D1/D4/D5). `launch_bwd_dkv`'s `segment_bound`/`break_early`
# (Task 2) gate the packed MMA query-block loop's uniform-break bound; the scalar body
# never sees either flag (D3 -- it stays the assumption-free oracle, predicate-only).
#
# TRAP (review finding): `_dkv_kernel` defaults to `variant="scalar"`, which never sees
# `segment_bound` -- every bounded/unbounded/break_early comparison below MUST pass
# `variant="mma"` explicitly, or both arms compile the IDENTICAL scalar source and the
# bit-identity assertion passes trivially (a false green).
# ---------------------------------------------------------------------------------------

# Own measurement (mlx 0.32.0, M1 Max, seed=7, over every `_BOUND_LAYOUTS` shape): worst
# dV 2.384186e-06 (mid_key_block_boundary and single_token_segments tied), worst dK
# 1.430511e-06. Pinned at ~2.1x the measured worst -- NEVER inherited from
# `_TOL_PACKED_DKV["mma"]` (3e-5, ~12.6x this test's own worst; that pin belongs to the
# packed-vs-block-diagonal-oracle parity grid, a different comparison at a different
# measured worst -- this file's own convention is measure-first, per-comparison pins).
_TOL_BOUNDED_SCALAR_VS_MMA = 5e-6

_BOUND_LAYOUTS = {
    "mid_key_block_boundary": [40, 216],  # 40 not a multiple of 32: boundary inside a key block
    "single_token_segments": [1] * 8 + [248],
    "many_tiny": [8] * 32,
    "row_filling": [256],
    "two_uneven": [100, 156],
}

# Bit-identity-only extra layout (review finding, Fix 6): the spec's own "many tiny
# segments (64x64 within 4096)" example at REAL scale -- n=4096, 64 segments of 64 tokens
# each. Added only to `test_bounded_dkv_bit_identical_to_unbounded` (not the
# scalar-vs-mma cross-check, which would get slow at this N); the smaller `many_tiny`
# case in `_BOUND_LAYOUTS` already covers the same shape at test speed.
_BOUND_LAYOUTS_BIT_IDENTITY = {**_BOUND_LAYOUTS, "spec_scale_64x64": [64] * 64}


def _bound_case(
    lens: list[int], seed: int
) -> tuple[mx.array, mx.array, mx.array, mx.array, float, mx.array, mx.array]:
    """b=1 q/k/v/dO plus the single-row packed layout for `lens` (sums to n). Fixed
    hq=4/hkv=2/d=64/scale=0.125 across every `_BOUND_LAYOUTS` case."""
    n = sum(lens)
    hq, hkv, d, scale = 4, 2, 64, 0.125
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, n, d))
    k = mx.random.normal((1, hkv, n, d))
    v = mx.random.normal((1, hkv, n, d))
    cot = mx.random.normal((1, hq, n, d))
    seg_id, seg_start = _packed_layout(lens, b=1)
    return q, k, v, cot, scale, seg_id, seg_start


@pytest.mark.metal
@pytest.mark.parametrize(("name", "lens"), sorted(_BOUND_LAYOUTS_BIT_IDENTITY.items()))
def test_bounded_dkv_bit_identical_to_unbounded(name: str, lens: list[int]) -> None:
    """The segment-end bound only SKIPS query blocks the unbounded predicate would have
    zeroed anyway (spec D5) -- bounded and unbounded MUST produce bit-identical dK/dV over
    every layout shape in `_BOUND_LAYOUTS_BIT_IDENTITY` (a mid-block boundary,
    all-singleton segments, many tiny segments, one full-row segment, two uneven
    segments, and the spec's own many-tiny-segments example at real scale, n=4096)."""
    q, k, v, cot, scale, seg_id, seg_start = _bound_case(lens, seed=7)
    dk_a, dv_a = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True,
    )
    # Fully evaluate arm A BEFORE constructing/evaluating arm B: verified on mlx 0.32.0,
    # co-evaluating two structurally-different freshly-JIT'd `mx.fast.metal_kernel`s in a
    # single `mx.eval` corrupted BOTH outputs on a cold process (order-dependent; the
    # full suite was only green because earlier tests had already warmed both kernel
    # variants). Separating the evals keeps each kernel's JIT compile + dispatch isolated.
    mx.eval(dk_a, dv_a)
    dk_b, dv_b = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=False,
    )
    mx.eval(dk_b, dv_b)
    assert mx.array_equal(dk_a, dk_b).item(), f"bounded dK != unbounded dK ({name})"
    assert mx.array_equal(dv_a, dv_b).item(), f"bounded dV != unbounded dV ({name})"


@pytest.mark.metal
def test_break_early_perturbation_fails_parity() -> None:
    """The named D4 bug site: `break_early=True` compares against the block's FIRST key
    instead of the LAST, so a segment boundary falling inside a key block (the
    `mid_key_block_boundary` layout, boundary at 40) truncates valid queries early. Bounded
    (correct) and break_early (perturbed) MUST diverge -- if they ever matched, the
    RED-perturbation test could not detect a wrong-key-in-block bug."""
    lens = _BOUND_LAYOUTS["mid_key_block_boundary"]                   # boundary inside key block 1
    q, k, v, cot, scale, seg_id, seg_start = _bound_case(lens, seed=7)
    dk_ok, dv_ok = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True,
    )
    # Fully evaluate arm A BEFORE constructing/evaluating arm B: verified on mlx 0.32.0,
    # co-evaluating two structurally-different freshly-JIT'd `mx.fast.metal_kernel`s in a
    # single `mx.eval` corrupted BOTH outputs on a cold process (order-dependent; the
    # full suite was only green because earlier tests had already warmed both kernel
    # variants). Separating the evals keeps each kernel's JIT compile + dispatch isolated.
    mx.eval(dk_ok, dv_ok)
    dk_bad, dv_bad = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True, break_early=True,
    )
    mx.eval(dk_bad, dv_bad)
    assert not (mx.array_equal(dk_ok, dk_bad).item() and mx.array_equal(dv_ok, dv_bad).item()), (
        "break_early perturbation matched the bounded kernel -- "
        "the RED-perturbation test cannot detect a wrong-key-in-block bug"
    )


@pytest.mark.metal
def test_bounded_chained_split_inside_segment_bit_identical_to_single() -> None:
    """A chained (multi-dispatch) plan under `segment_bound=True` must stay bit-identical
    to a single [0, n) dispatch, exactly like the pre-0.5.0 chained-vs-single proofs -- the
    bound must not disturb the fp32 accumulator carry across dispatches. Split at 64
    (32-aligned, the mma block-alignment contract) lands inside segment 0's span (0..99)."""
    q, k, v, cot, scale, seg_id, seg_start = _bound_case([100, 156], seed=11)
    single_dk, single_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True,
        force_ranges=[(0, 256)],
    )
    split_dk, split_dv = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True,
        force_ranges=[(0, 64), (64, 256)],  # 32-aligned, inside segment 0 (0..99)
    )
    mx.eval(single_dk, single_dv, split_dk, split_dv)
    assert mx.array_equal(single_dk, split_dk).item(), "bounded chained split dK != single"
    assert mx.array_equal(single_dv, split_dv).item(), "bounded chained split dV != single"


def _packed_layout_per_row(rows_seg_lens: list[list[int]]) -> tuple[mx.array, mx.array]:
    """(B, N) int32 seg_id/seg_start with a DIFFERENT segment layout on each batch row --
    unlike `_packed_layout`, which broadcasts one row across the batch. Every row's seg_lens
    must sum to the same N (the sequence axis is uniform). Origin: copied verbatim from
    tests/test_attention_api.py::_packed_layout_per_row (review finding M2 -- the public-API
    per-row test there exposes no `segment_bound` knob, so this file needs its own copy)."""
    seg_id_rows: list[list[int]] = []
    seg_start_rows: list[list[int]] = []
    for seg_lens in rows_seg_lens:
        seg_id_row: list[int] = []
        seg_start_row: list[int] = []
        start = 0
        for sid, ln in enumerate(seg_lens):
            seg_id_row += [sid] * ln
            seg_start_row += [start] * ln
            start += ln
        seg_id_rows.append(seg_id_row)
        seg_start_rows.append(seg_start_row)
    return mx.array(seg_id_rows, dtype=mx.int32), mx.array(seg_start_rows, dtype=mx.int32)


@pytest.mark.metal
def test_bounded_dkv_per_row_varying_layout_matches_oracle() -> None:
    """Different segment layouts per batch row at the SAME n, under `segment_bound=True` --
    uniquely covers the `seg_off = b * n` addressing under the bound (review finding M2): a
    wrong-row segment-buffer read would mask against the OTHER row's boundary and break
    parity against the per-row block-diagonal oracle. Pins reuse this file's existing MMA
    packed parity tolerance (`_TOL_PACKED_DKV["mma"]`, measure-first -- never widened)."""
    b, hq, hkv, n, d = 2, 4, 2, 256, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v, cot = _rand_qkv_do(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=68)
    seg_id, seg_start = _packed_layout_per_row([[40, 216], [100, 156]])
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    dk_k, dv_k = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma", segment_bound=True,
        seg_id=seg_id, seg_start=seg_start,
    )
    dk_ref, dv_ref = _dkv_oracle(q, k, v, cot, scale=scale, causal=True, segments=pm)
    mx.eval(dk_k, dv_k, dk_ref, dv_ref)

    d_dk = mx.abs(dk_k.astype(mx.float32) - dk_ref.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_k.astype(mx.float32) - dv_ref.astype(mx.float32)).max().item()
    print(f"[dKV-bound per-row mma n{n}] dK={d_dk:.6e} dV={d_dv:.6e}")
    tol = _TOL_PACKED_DKV["mma"][mx.float32]
    assert d_dk < tol, f"bounded per-row dK-mma vs oracle diff {d_dk}"
    assert d_dv < tol, f"bounded per-row dV-mma vs oracle diff {d_dv}"


@pytest.mark.metal
@pytest.mark.parametrize(("name", "lens"), sorted(_BOUND_LAYOUTS.items()))
def test_bounded_mma_matches_scalar_across_bound_layouts(name: str, lens: list[int]) -> None:
    """The scalar dK/dV kernel takes NO `segment_bound` at all (D3 -- it stays the
    assumption-free, predicate-only oracle among kernel variants); the bounded mma kernel
    must still agree with it, over every `_BOUND_LAYOUTS` shape, at this comparison's own
    measured-worst pin (`_TOL_BOUNDED_SCALAR_VS_MMA`, measure-first, never widened)."""
    q, k, v, cot, scale, seg_id, seg_start = _bound_case(lens, seed=7)
    dk_s, dv_s = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="scalar",
        seg_id=seg_id, seg_start=seg_start,
    )
    # Fully evaluate arm A BEFORE constructing/evaluating arm B: verified on mlx 0.32.0,
    # co-evaluating two structurally-different freshly-JIT'd `mx.fast.metal_kernel`s in a
    # single `mx.eval` corrupted BOTH outputs on a cold process (order-dependent; the
    # full suite was only green because earlier tests had already warmed both kernel
    # variants). Separating the evals keeps each kernel's JIT compile + dispatch isolated.
    mx.eval(dk_s, dv_s)
    dk_m, dv_m = _dkv_kernel(
        q, k, v, cot, scale=scale, causal=True, variant="mma",
        seg_id=seg_id, seg_start=seg_start, segment_bound=True,
    )
    mx.eval(dk_m, dv_m)
    d_dk = mx.abs(dk_s.astype(mx.float32) - dk_m.astype(mx.float32)).max().item()
    d_dv = mx.abs(dv_s.astype(mx.float32) - dv_m.astype(mx.float32)).max().item()
    print(f"[dKV-bound scalar-vs-mma {name}] dK={d_dk:.6e} dV={d_dv:.6e}")
    tol = _TOL_BOUNDED_SCALAR_VS_MMA
    assert d_dk < tol, f"scalar-vs-bounded-mma dK diff {d_dk} ({name})"
    assert d_dv < tol, f"scalar-vs-bounded-mma dV diff {d_dv} ({name})"
