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
"""
import mlx.core as mx
import pytest

from mlx_train_perf.attention.kernel.launch import launch_bwd_D
from mlx_train_perf.attention.kernel.source import build_bwd_D_source
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
