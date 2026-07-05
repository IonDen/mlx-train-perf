from mlx_train_perf.core.kernel.source import (
    _BACKWARD_DHIDDEN_MMA_TEMPLATE,
    _DENSE_TEMPLATE,
    _GEMM_A_LOGIT_REGEN,
    build_backward_dhidden_mma_source,
    build_dense_source,
    build_quant_source,
)


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


# ---------------------------------------------------------------------------------------
# Backward d_hidden MMA kernel (Task 16b step 4): fused two-GEMM (logit-regen GEMM-A reused
# verbatim from the forward, then P-formation, then a vocab-contracting GEMM-B into
# d_hidden). The reuse of the forward's logit-regen inner loop is a load-bearing invariant.


def test_gemm_a_logit_regen_is_verbatim_from_the_forward() -> None:
    # GEMM-A (the logit regeneration) is the forward's own MMA inner loop, byte-identical:
    # `_GEMM_A_LOGIT_REGEN` is sliced out of `_DENSE_TEMPLATE`, so it must be a substring of
    # both the forward's template and the backward-MMA template (the reuse the design rests
    # on -- GEMM-A already runs at the forward's measured rate).
    assert _GEMM_A_LOGIT_REGEN in _DENSE_TEMPLATE
    assert _GEMM_A_LOGIT_REGEN in _BACKWARD_DHIDDEN_MMA_TEMPLATE
    # non-trivial: it carries the wp load, the C-tile MMA accumulation, and the guarded tail
    gemm_a_mma = "metal::simdgroup_multiply_accumulate(C[rt][ct], A[rt], B[ct], C[rt][ct]);"
    assert gemm_a_mma in _GEMM_A_LOGIT_REGEN
    assert "if (dfull < d)" in _GEMM_A_LOGIT_REGEN
    # ...but NOT the forward-only online-LSE epilogue (that is where the backward diverges)
    assert "lse_out[row]" not in _GEMM_A_LOGIT_REGEN


def test_backward_dhidden_mma_rt4_structure() -> None:
    s = build_backward_dhidden_mma_source(4)
    assert "simdgroup_float8x8 C[4][4]" in s      # GEMM-A logit tiles (reused)
    assert "uint r0 = ygroup * 32;" in s          # RT=4 row-block sizing
    assert "#pragma clang loop unroll(full)" in s
    # GEMM-B: vocab-contracting accumulate of G @ W_sub into the device d_hidden output
    assert "metal::simdgroup_multiply_accumulate(Db, C[rt][ct], Wb, Db);" in s
    assert "d_hidden_out[base]" in s
    # cross-vsub RMW reads via a per-value ternary (d_hidden_in is const, can't alias out)
    assert "float p0 = (cc == 0) ? d_hidden_in[base] : d_hidden_out[base];" in s


def test_backward_dhidden_mma_rt2_structure() -> None:
    s = build_backward_dhidden_mma_source(2)
    assert "simdgroup_float8x8 C[2][4]" in s
    assert "uint r0 = ygroup * 16;" in s


def test_backward_dhidden_mma_lane_mapping_is_rt_invariant() -> None:
    for rt in (2, 4):
        s = build_backward_dhidden_mma_source(rt)
        assert "uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);" in s
        assert "uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);" in s


def test_backward_dhidden_mma_d_tail_guard_hoisted_out_of_the_output_loop() -> None:
    # The d-output loop runs full 8-column slices unguarded, then one guarded tail slice --
    # the forward's 1.88x tail-guard lesson applied to GEMM-B's output streaming.
    for rt in (2, 4):
        s = build_backward_dhidden_mma_source(rt)
        assert s.count("for (uint d0d = 0; d0d < dfo; d0d += 8)") == 1
        assert s.count("if (dfo < d)") == 1


def test_backward_dhidden_mma_masks_out_of_tile_vocab_columns() -> None:
    # Clamp-duplicated tail columns (out of [0, tcols)) must contribute ZERO to d_hidden --
    # GEMM-B sums G @ W over the vsub, so g is masked to 0 there (0 * w == 0).
    for rt in (2, 4):
        s = build_backward_dhidden_mma_source(rt)
        assert "(col0 < tcols)" in s
        assert "(col1 < tcols)" in s


def test_build_backward_dhidden_mma_source_rejects_bad_row_tiles() -> None:
    for rt in (0, 1, 3, 8):
        try:
            build_backward_dhidden_mma_source(rt)
        except ValueError:
            continue
        raise AssertionError(f"row_tiles={rt} should have been rejected")
