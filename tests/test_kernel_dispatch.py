from mlx_train_perf.core.kernel.dispatch import MEASURED, select_variant


def test_measured_buckets_pick_artifact_best() -> None:
    # artifacts: v2d wins 512 (310.7 vs v2c 210.8, v2e 198.7) and 2048 (879.2 vs 644.7);
    # v2e wins 8192 (2423.7). v2c never wins a bucket.
    c512 = select_variant(512)
    assert c512.row_tiles == 2
    assert not c512.provisional
    c2048 = select_variant(2048)
    assert c2048.row_tiles == 2
    assert not c2048.provisional
    c8192 = select_variant(8192)
    assert c8192.row_tiles == 4
    assert not c8192.provisional


def test_table_constants_equal_committed_artifact_numbers() -> None:
    assert MEASURED[512] == (2, 310.7)
    assert MEASURED[2048] == (2, 879.2)
    assert MEASURED[8192] == (4, 2423.7)


def test_unmeasured_buckets_are_provisional_nearest_by_log2() -> None:
    c = select_variant(4096)
    assert c.provisional            # log2-equidistant -> lower bucket wins
    assert c.row_tiles == 2
    big = select_variant(65536)
    assert big.provisional          # nearest measured: 8192
    assert big.row_tiles == 4
    small = select_variant(64)
    assert small.provisional        # nearest measured: 512
    assert small.row_tiles == 2
