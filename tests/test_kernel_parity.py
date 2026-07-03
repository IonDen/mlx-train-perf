import mlx.core as mx
import pytest

from mlx_train_perf.core.kernel.launch import forward
from mlx_train_perf.core.naive import naive_linear_ce

pytestmark = pytest.mark.metal

CASES = [  # (n, d, v, tile) — spike checker grid + planted tails
    (64, 32, 1000, 250), (64, 32, 1000, 333), (64, 32, 1000, 1000), (64, 32, 1000, 4096),
    (128, 64, 8192, 1024),
    (65, 33, 1027, 256),   # n % 32 != 0, d % 8 != 0, v % tile != 0 — every tail guard live
    (512, 64, 2048, 512), (8192, 32, 1024, 1024),  # dispatch-bucket n values (spec §6:
    # every crossover shape runs the ACTUAL kernel, at reduced V/D to stay in budget)
]
GENEROUS_RATE = 1e13  # parity shapes are microscopic; budget check must never trip here


@pytest.mark.parametrize(("n", "d", "v", "tile"), CASES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("row_tiles", [2, 4])
def test_value_parity_vs_fp32_exact_naive(n: int, d: int, v: int, tile: int,
                                          dtype: mx.Dtype, row_tiles: int) -> None:
    mx.random.seed(3)
    hidden = mx.random.normal((n, d)).astype(dtype)
    w = (mx.random.normal((v, d)) * 0.05).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    targets[0] = 0
    targets[1] = v - 1
    targets[2] = min(tile, v) - 1        # last id of the first vocab tile — boundary
    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                       rate_macs_per_s=GENEROUS_RATE)
    ours = lse - tgt
    ref = naive_linear_ce(hidden.astype(mx.float32), w.astype(mx.float32), targets)
    # ceilings are 1e-5 (fp32) / 1e-3 (bf16); measured worst over this grid (both dtypes,
    # both row_tiles) is 1.9073486328125e-06 at row_tiles=2, n=128,d=64,v=8192,tile=1024 ->
    # pin fp32 4e-6 (~2.1x margin), bf16 5e-6 (~2.6x margin, same convention as
    # test_chunked.py's dense-CE pins). The kernel accumulates fp32 in-register even for
    # bf16 inputs, so both dtypes land in the same reduction-order-noise class.
    # This pin is measured over THIS grid only — the spike's broader grid saw 4.8e-6 fp32.
    # If a future case legitimately lands between this pin and the 1e-5 ceiling, widen
    # toward the ceiling with a note; don't treat it as a regression.
    tol = 4e-6 if dtype == mx.float32 else 5e-6
    assert mx.abs(ours - ref).max().item() < tol


def test_all_rows_written_when_n_not_multiple_of_block() -> None:
    mx.random.seed(4)
    n, d, v = 33, 16, 128
    hidden = mx.random.normal((n, d))
    w = mx.random.normal((v, d)) * 0.05
    targets = mx.random.randint(0, v, (n,))
    lse, tgt = forward(hidden, w, targets, row_tiles=4, tile=128, rate_macs_per_s=GENEROUS_RATE)
    assert bool(mx.isfinite(lse).all().item()) and bool(mx.isfinite(lse - tgt).all().item())  # noqa: PT018
