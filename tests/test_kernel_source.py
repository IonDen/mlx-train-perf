from mlx_train_perf.core.kernel.source import build_dense_source


def test_rt4_matches_v2e_structure() -> None:
    s = build_dense_source(4)
    assert "simdgroup_float8x8 C[4][4]" in s
    assert "uint r0 = ygroup * 32;" in s
    assert "#pragma clang loop unroll(full)" in s  # steel idiom — required on array loops


def test_rt2_matches_v2d_structure() -> None:
    s = build_dense_source(2)
    assert "simdgroup_float8x8 C[2][4]" in s
    assert "uint r0 = ygroup * 16;" in s


def test_lane_mapping_is_rt_invariant() -> None:
    for rt in (2, 4):
        s = build_dense_source(rt)
        assert "uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);" in s
        assert "uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);" in s
