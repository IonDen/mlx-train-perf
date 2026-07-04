"""Parity for the backward kernels (Task 16b steps 2-3, v0-correct) against
`chunked_backward` — the proven oracle (see core/chunked.py). Both paths consume the
IDENTICAL (hidden, w, targets, lse, cotangent): `lse` comes from the SAME kernel forward
dispatch used in production (never a separately-computed naive lse), so any measured diff
is attributable to the backward derivation/implementation itself, not to a residual
mismatch. d_hidden covers the frozen-head (QLoRA) path (step 2); d_w covers the trainable
head (step 3) — same grid discipline (both dtypes, both row_tiles, nonuniform cotangent,
planted boundary targets), but d_w is additionally NON-DETERMINISTIC at the bit level
(atomics reorder float additions run to run) — see
`test_dw_parity_vs_chunked_backward_oracle_across_repeated_runs` for the repeated-run
tolerance discipline this requires.
"""
from collections.abc import Callable

import mlx.core as mx
import pytest

from mlx_train_perf.core.chunked import chunked_backward
from mlx_train_perf.core.kernel.launch import backward_dhidden, backward_dw, forward

pytestmark = pytest.mark.metal

# Mirrors test_kernel_parity.py's CASES grid — same spike-checker + planted-tail shapes.
CASES = [
    (64, 32, 1000, 250), (64, 32, 1000, 333), (64, 32, 1000, 1000), (64, 32, 1000, 4096),
    (128, 64, 8192, 1024),
    (65, 33, 1027, 256),   # n % 32 != 0, d % 8 != 0, v % tile != 0 — every tail guard live
    (512, 64, 2048, 512), (8192, 32, 1024, 1024),  # dispatch-bucket n values
]
GENEROUS_RATE = 1e13  # parity shapes are microscopic; budget check must never trip here


def _nonuniform_cotangent(n: int) -> mx.array:
    """Deliberately NOT all-ones/all-equal: alternating sign, varying magnitude — stresses
    the coefficient's sign and scale, unlike a real mean-reduction cotangent (uniform
    1/n)."""
    mx.random.seed(17)
    magnitude = mx.random.uniform(0.1, 2.0, (n,)).astype(mx.float32)
    sign = mx.where(mx.arange(n) % 2 == 0, 1.0, -1.0).astype(mx.float32)
    return sign * magnitude


@pytest.mark.parametrize(("n", "d", "v", "tile"), CASES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("row_tiles", [2, 4])
def test_dhidden_parity_vs_chunked_backward_oracle(
    n: int, d: int, v: int, tile: int, dtype: mx.Dtype, row_tiles: int,
) -> None:
    mx.random.seed(3)
    hidden = mx.random.normal((n, d)).astype(dtype)
    w = (mx.random.normal((v, d)) * 0.05).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    targets[0] = 0                          # planted: first column
    targets[1] = v - 1                      # planted: last column
    targets[2] = min(tile, v) - 1           # planted: first-tile boundary
    ct = _nonuniform_cotangent(n)

    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                       rate_macs_per_s=GENEROUS_RATE)

    ours = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                            tile=tile, rate_macs_per_s=GENEROUS_RATE)

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    w_chunk: Callable[[int, int], mx.array] = lambda a, b: w[a:b]  # noqa: E731
    ref, d_w = chunked_backward(
        hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk, targets=targets, lse=lse,
        cotangent=ct, v=v, chunk_size=tile, head_trainable=False,
    )
    assert d_w is None
    # RED gate was the task-specified initial ceiling (2e-5); measured over this FULL grid
    # (both row_tiles, planted boundary targets, nonuniform cotangent) the worst diffs are
    # 8.046627044677734e-07 (fp32, row_tiles=2, n=128,d=64,v=8192,tile=1024) and
    # 0.001953125 (bf16, row_tiles=2, n=64,d=32,v=1000,tile=250) — pin fp32 2e-6 (~2.5x
    # margin) and bf16 5e-3 (~2.6x margin, same convention as test_chunked.py's bf16 pins:
    # the MSL kernel accumulates fp32 in-register for both dtypes, same reduction-order-
    # noise class as the forward parity gates in test_kernel_parity.py).
    tol = 2e-6 if dtype == mx.float32 else 5e-3
    assert mx.abs(ours.astype(mx.float32) - ref.astype(mx.float32)).max().item() < tol


def test_dhidden_output_shape_and_dtype_match_hidden() -> None:
    mx.random.seed(4)
    n, d, v = 33, 16, 128
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    ct = mx.full((n,), 1.0 / n)
    lse, tgt = forward(hidden, w, targets, row_tiles=4, tile=128, rate_macs_per_s=GENEROUS_RATE)
    d_hidden = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=4, tile=128,
                                rate_macs_per_s=GENEROUS_RATE)
    assert d_hidden.shape == hidden.shape
    assert d_hidden.dtype == hidden.dtype
    assert bool(mx.isfinite(d_hidden.astype(mx.float32)).all().item())


# ---------------------------------------------------------------------------------------
# d_w (trainable head, Task 16b step 3): the ONE new mechanism vs d_hidden is cross-
# ROW-BLOCK atomic accumulation (see core/kernel/source.py's derivation comment). This
# makes d_w's output BIT-LEVEL NON-DETERMINISTIC run to run (atomics reorder the floating-
# point additions), unlike d_hidden's per-row accumulation (no contention, deterministic).
# The pinned tolerances below are measured across >=5 REPEATED runs of the full grid, not
# a single run, so a run whose additions happen to reorder unluckily still passes.


@pytest.mark.parametrize(("n", "d", "v", "tile"), CASES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("row_tiles", [2, 4])
def test_dw_parity_vs_chunked_backward_oracle(
    n: int, d: int, v: int, tile: int, dtype: mx.Dtype, row_tiles: int,
) -> None:
    mx.random.seed(3)
    hidden = mx.random.normal((n, d)).astype(dtype)
    w = (mx.random.normal((v, d)) * 0.05).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    targets[0] = 0                          # planted: first column
    targets[1] = v - 1                      # planted: last column
    targets[2] = min(tile, v) - 1           # planted: first-tile boundary
    ct = _nonuniform_cotangent(n)

    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                       rate_macs_per_s=GENEROUS_RATE)

    ours = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                       tile=tile, rate_macs_per_s=GENEROUS_RATE)

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    w_chunk: Callable[[int, int], mx.array] = lambda a, b: w[a:b]  # noqa: E731
    d_hidden_ref, ref = chunked_backward(
        hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk, targets=targets, lse=lse,
        cotangent=ct, v=v, chunk_size=tile, head_trainable=True,
    )
    assert ref is not None
    del d_hidden_ref
    # RED gate was d_hidden's own pins (2e-6 fp32 / 5e-3 bf16) as an initial guess; both
    # were too tight. Measured over this FULL grid (both row_tiles, planted boundary
    # targets, nonuniform cotangent) ACROSS 5 REPEATED RUNS PER CASE (atomics reorder float
    # additions run to run -- see the module-level comment above and the report's recorded
    # spread) the worst diffs are 7.82012939453125e-05 (fp32, n=8192,d=32,v=1024,tile=1024)
    # and 0.125 (bf16, same shape) -- both driven by reduction DEPTH (d_w sums over all n
    # context rows; n=8192 is the deepest reduction in this grid), not by row_tiles or
    # tile width. Pin fp32 2e-4 (~2.6x margin, matching the d_hidden convention) and bf16
    # 0.25 (~2x margin) -- both are legitimate fp/bf16 reduction-order noise (d_w's typical
    # magnitude at this shape is ~2.6 mean-abs, so 0.125 is a small relative error), not an
    # implementation divergence: run-to-run SPREAD at this same shape was only ~1e-5
    # (fp32) / exactly 0 (bf16, whose own rounding already dominates) -- see the report.
    tol = 2e-4 if dtype == mx.float32 else 0.25
    assert mx.abs(ours.astype(mx.float32) - ref.astype(mx.float32)).max().item() < tol


def test_dw_parity_is_stable_across_repeated_runs() -> None:
    """Atomics reorder float additions run to run, so d_w is bit-level non-deterministic
    (see the module-level comment above). This test runs the deepest-reduction case in the
    grid (n=8192 -- the shape that showed measurable run-to-run SPREAD during tolerance
    measurement, unlike the shallower-n shapes which landed bit-identical across repeats)
    5 times and asserts every run stays under the pinned tolerance -- proving the
    tolerance is robust to run-to-run nondeterminism, not just lucky on one run."""
    mx.random.seed(5)
    n, d, v, tile, row_tiles = 8192, 32, 1024, 1024, 2
    hidden = mx.random.normal((n, d)).astype(mx.float32)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.float32)
    targets = mx.random.randint(0, v, (n,))
    ct = _nonuniform_cotangent(n)
    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                       rate_macs_per_s=GENEROUS_RATE)

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    w_chunk: Callable[[int, int], mx.array] = lambda a, b: w[a:b]  # noqa: E731
    _, ref = chunked_backward(
        hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk, targets=targets, lse=lse,
        cotangent=ct, v=v, chunk_size=tile, head_trainable=True,
    )
    assert ref is not None
    ref32 = ref.astype(mx.float32)

    diffs = []
    for _ in range(5):
        ours = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                           tile=tile, rate_macs_per_s=GENEROUS_RATE)
        diff = mx.abs(ours.astype(mx.float32) - ref32).max().item()
        diffs.append(diff)
        assert diff < 2e-4
    # Printed (not asserted) run-to-run spread -- measured ~1.05e-5 here during tolerance
    # discovery (5 runs: 7.25e-5, 6.68e-5, 6.68e-5, 7.72e-5, 6.68e-5) -- see the report.
    print(f"d_w repeated-run diffs (n={n}, row_tiles={row_tiles}): {diffs}")


def test_dw_output_shape_and_dtype_match_w() -> None:
    mx.random.seed(6)
    n, d, v = 33, 16, 128
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    ct = mx.full((n,), 1.0 / n)
    lse, tgt = forward(hidden, w, targets, row_tiles=4, tile=128, rate_macs_per_s=GENEROUS_RATE)
    d_w = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=4, tile=128,
                      rate_macs_per_s=GENEROUS_RATE)
    assert d_w.shape == w.shape
    assert d_w.dtype == w.dtype
    assert bool(mx.isfinite(d_w.astype(mx.float32)).all().item())
