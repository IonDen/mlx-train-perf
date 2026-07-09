"""0.2.0 T6 rung 3 -- forward-kernel dispatch table (`select_fwd_tile`), pure arithmetic,
DEFAULT lane (mirrors `tests/test_kernel_dispatch.py`'s house pattern -- no GPU needed to
pick a `TileShape`, only to build/launch the kernel it names).
"""
import pytest

from mlx_train_perf.attention.kernel.dispatch import MEASURED, select_fwd_tile
from mlx_train_perf.attention.kernel.launch import TileShape


def test_measured_bucket_returns_mma_slab128_non_provisional() -> None:
    # rung2b_dslab128.json: 1462.74 G MAC/s at the flagship (head_dim=128, n=8192) shape --
    # the ladder's saturation-bucket winner, directly measured.
    choice = select_fwd_tile(8192, 128)
    assert choice == TileShape(variant="mma", d_slab=128, provisional=False)
    # the whole occupancy regime [8192, 16384) shares this same direct measurement
    choice_mid = select_fwd_tile(12000, 128)
    assert choice_mid == TileShape(variant="mma", d_slab=128, provisional=False)


def test_measured_table_constant_equals_the_committed_artifact_number() -> None:
    assert MEASURED[128][8192] == ("mma", 128, 1462.74)


def test_head_dim_128_below_the_measured_bucket_is_provisional() -> None:
    # same physics (the D-slab restructure has no n-dependence), just unmeasured at this n
    choice = select_fwd_tile(512, 128)
    assert choice.variant == "mma"
    assert choice.d_slab == 128
    assert choice.provisional


def test_head_dim_128_above_the_measured_occupancy_regime_is_provisional() -> None:
    # nearest measured bucket is still 8192, but this n falls outside the [8192, 16384)
    # occupancy window the measurement actually covers -- provisional, mirroring
    # core/kernel/dispatch.py's own occupancy-regime window exactly.
    choice = select_fwd_tile(65536, 128)
    assert choice.variant == "mma"
    assert choice.d_slab == 128
    assert choice.provisional


@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize("n", [16, 61, 512, 8192])
def test_unmeasured_head_dims_always_select_mma_default_slab_provisional(
    head_dim: int, n: int,
) -> None:
    # head_dim in {64, 96} was never run through the T6 ladder at any N -- mma with the
    # source builder's own default slab, always provisional.
    choice = select_fwd_tile(n, head_dim)
    assert choice.variant == "mma"
    assert choice.d_slab is None
    assert choice.provisional


@pytest.mark.parametrize("head_dim", [0, 32, 80, 256])
def test_unknown_head_dim_raises(head_dim: int) -> None:
    with pytest.raises(ValueError, match="head_dim"):
        select_fwd_tile(8192, head_dim)
