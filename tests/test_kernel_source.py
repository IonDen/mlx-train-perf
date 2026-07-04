from mlx_train_perf.core.kernel.source import build_dense_source, build_quant_source


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


def test_quant_hoist_reload_branch_present_exactly_once() -> None:
    for rt in (2, 4):
        s = build_quant_source(rt)
        assert s.count("if ((d0 & 63u) == 0u)") == 1


def test_quant_hoist_mtp_dq4_absent_from_full_chunks_present_in_tail() -> None:
    for rt in (2, 4):
        s = build_quant_source(rt)
        full_chunks, tail = s.split("if (dfull < d)", 1)
        assert "mtp_dq4" not in full_chunks
        assert "mtp_dq4" in tail


def test_quant_hoist_row_pointer_indexing_present() -> None:
    for rt in (2, 4):
        s = build_quant_source(rt)
        assert "wq0[ct][d0 >> 3]" in s
        assert "wq1[ct][d0 >> 3]" in s


def test_quant_hoist_block_scope_declarations_present() -> None:
    for rt in (2, 4):
        s = build_quant_source(rt)
        assert "uint sh = 4 * fm;" in s
        assert "float hsc0[4]; float hbi0[4]; float hsc1[4]; float hbi1[4];" in s
        assert "const device uint* wq0[4];" in s
        assert "const device uint* wq1[4];" in s
