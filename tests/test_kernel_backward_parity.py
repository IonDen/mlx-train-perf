"""Parity for the d_hidden-only backward kernel (Task 16b step 2, v0-correct) against
`chunked_backward` — the proven oracle (see core/chunked.py). Both paths consume the
IDENTICAL (hidden, w, targets, lse, cotangent): `lse` comes from the SAME kernel forward
dispatch used in production (never a separately-computed naive lse), so any measured diff
is attributable to the backward derivation/implementation itself, not to a residual
mismatch. Frozen-head (QLoRA) path only — the flagship per the task brief; d_w (trainable
head) lands in step 3.
"""
from collections.abc import Callable

import mlx.core as mx
import pytest

from mlx_train_perf.core.chunked import chunked_backward
from mlx_train_perf.core.kernel.launch import backward_dhidden, forward

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
