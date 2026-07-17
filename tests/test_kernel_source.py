import pytest

from mlx_train_perf.attention.kernel.source import (
    build_bwd_dkv_mma_source,
    build_bwd_dkv_source,
    build_bwd_dq_mma_source,
    build_bwd_dq_source,
    build_fwd_mma_source,
    build_fwd_source,
)
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


# ---------------------------------------------------------------------------------------
# 0.4.0 T2: the PACKED forward variant (scalar + MMA) -- block-diagonal segment-ID keep
# predicate. `packed=False` (the default) must stay byte-identical to the pre-0.4.0 MSL, so
# every pre-existing caller (and the whole causal parity grid) is untouched. `packed=True`
# wraps the causal keep with a same-segment equality; `flip_segments` inverts that equality
# (the cross-contamination perturbation, the segment analogue of `flip_causal`).


def test_fwd_sources_byte_identical_when_not_packed() -> None:
    # The default (packed off) must equal the explicit packed=False for BOTH fwd builders --
    # the byte-identity contract that keeps the pre-0.4.0 causal kernels bit-for-bit unchanged.
    for build in (build_fwd_source, build_fwd_mma_source):
        assert build(128) == build(128, packed=False)


def test_fwd_sources_have_no_segment_text_when_not_packed() -> None:
    # Stronger byte-identity guard: no packed artifact may leak into the non-packed source
    # (a stray seg_off/kv_lo would perturb the causal MSL even if the string still "worked").
    for build in (build_fwd_source, build_fwd_mma_source):
        s = build(128)
        assert "seg_id" not in s
        assert "seg_start" not in s
        assert "seg_off" not in s
        assert "kv_lo" not in s


def test_fwd_scalar_packed_predicate_wraps_causal_with_segment_equality() -> None:
    s = build_fwd_source(64, packed=True)
    assert "uint seg_off = b * n;" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" in s
    assert "kk <= row" in s  # the base causal predicate is retained inside the wrap


def test_fwd_mma_packed_predicate_and_kv_lower_bound() -> None:
    s = build_fwd_mma_source(64, packed=True)
    assert "uint seg_off = b * n;" in s
    # row is clamped (unlike the scalar builder) -- an over-hang lane's row can be >= n
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" in s
    # the KV-block loop starts at the block's segment-floored lower bound, not 0
    assert "uint kv_lo = " in s
    assert "for (uint kb0 = kv_lo; kb0 < kv_limit; kb0 += 32) {" in s
    assert "for (uint kb0 = 0; kb0 < kv_limit; kb0 += 32) {" not in s


def test_fwd_mma_packed_seg_id_row_read_is_clamped_for_over_hang_lanes() -> None:
    # T14 review finding: a partially-over-hang query block (n not 32-aligned) has lanes with
    # row >= n; the packed predicate's seg_id[seg_off + row] read must clamp row to n-1 (the
    # kv_lo seg_start read already does this) so those lanes read a valid, in-bounds id -- its
    # value is irrelevant since over-hang lanes are discarded before the O/L store.
    s = build_fwd_mma_source(64, packed=True)
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" in s
    assert "seg_id[seg_off + row]" not in s  # no unclamped row read should remain


def test_fwd_scalar_packed_seg_id_row_read_is_not_clamped() -> None:
    # The scalar builder's row (= r0 + local_row; local_row < rows_this, r1 <= n) is always
    # < n by construction, so it is immune to the MMA over-hang bug and needs no clamp.
    s = build_fwd_source(64, packed=True)
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" in s
    assert "metal::min(row, n - 1)" not in s


def test_fwd_scalar_flip_segments_inverts_only_the_equality() -> None:
    s = build_fwd_source(64, packed=True, flip_segments=True)
    assert "seg_id[seg_off + kk] != seg_id[seg_off + row]" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" not in s
    assert "kk <= row" in s  # the causal half is untouched -- only the equality flips


def test_fwd_mma_flip_segments_inverts_only_the_equality() -> None:
    s = build_fwd_mma_source(64, packed=True, flip_segments=True)
    assert "seg_id[seg_off + kk] != seg_id[seg_off + metal::min(row, n - 1)]" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" not in s


def test_fwd_packed_requires_causal() -> None:
    with pytest.raises(ValueError, match="packed"):
        build_fwd_source(128, causal=False, packed=True)
    with pytest.raises(ValueError, match="packed"):
        build_fwd_mma_source(128, causal=False, packed=True)


def test_flip_segments_requires_packed() -> None:
    with pytest.raises(ValueError, match="flip_segments"):
        build_fwd_source(128, flip_segments=True)
    with pytest.raises(ValueError, match="flip_segments"):
        build_fwd_mma_source(128, flip_segments=True)


def test_flip_segments_mutually_exclusive_with_the_causal_perturbations() -> None:
    for build in (build_fwd_source, build_fwd_mma_source):
        with pytest.raises(ValueError, match="flip_segments"):
            build(64, packed=True, flip_segments=True, flip_causal=True)
        with pytest.raises(ValueError, match="flip_segments"):
            build(64, packed=True, flip_segments=True, drop_diagonal=True)


# ---------------------------------------------------------------------------------------
# 0.4.0 T3: the PACKED backward variants (dQ scalar + mma, dK/dV scalar + mma) -- the same
# block-diagonal segment-ID keep predicate as the forward, applied to each backward body.
# `packed=False` (the default) MUST stay byte-identical to the pre-0.4.0 causal backward MSL
# (every existing dQ/dK/dV parity + determinism test is untouched). `packed=True` wraps each
# body's causal keep with a same-segment equality; `flip_segments` inverts that equality (the
# cross-contamination perturbation, the segment analogue of `flip_causal`). The four builders
# take `causal` as a required kwarg, so every call below passes `causal=True` explicitly.

_BWD_BUILDERS = (
    build_bwd_dq_source, build_bwd_dq_mma_source,
    build_bwd_dkv_source, build_bwd_dkv_mma_source,
)


def test_bwd_sources_byte_identical_when_not_packed() -> None:
    # The default (packed off) must equal the explicit packed=False for ALL FOUR backward
    # builders -- the byte-identity contract that keeps the pre-0.4.0 causal kernels
    # bit-for-bit unchanged (mirrors test_fwd_sources_byte_identical_when_not_packed).
    for build in _BWD_BUILDERS:
        assert build(128, causal=True) == build(128, causal=True, packed=False)


def test_bwd_sources_have_no_segment_text_when_not_packed() -> None:
    # Stronger byte-identity guard: no packed artifact may leak into the non-packed source
    # (a stray seg_off/kv_lo would perturb the causal MSL even if the string still "worked").
    for build in _BWD_BUILDERS:
        s = build(128, causal=True)
        assert "seg_id" not in s
        assert "seg_start" not in s
        assert "seg_off" not in s
        assert "kv_lo" not in s


def test_bwd_dq_scalar_packed_predicate_wraps_causal_with_segment_equality() -> None:
    s = build_bwd_dq_source(64, causal=True, packed=True)
    assert "uint seg_off = b * n;" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" in s
    assert "kk <= row" in s  # the base causal predicate is retained inside the wrap


def test_bwd_dq_scalar_packed_seg_id_row_read_is_not_clamped() -> None:
    # The scalar dQ row (= r0 + local_row; local_row < rows_this, r1 <= n) is always < n by
    # construction, so it is immune to the MMA over-hang bug and needs no clamp (mirrors the
    # scalar forward's row analysis).
    s = build_bwd_dq_source(64, causal=True, packed=True)
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" in s
    assert "metal::min(row, n - 1)" not in s


def test_bwd_dq_mma_packed_predicate_and_kv_lower_bound() -> None:
    s = build_bwd_dq_mma_source(64, causal=True, packed=True)
    assert "uint seg_off = b * n;" in s
    # row is clamped (unlike the scalar builder) -- an over-hang lane's row can be >= n
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" in s
    # the KV-block loop starts at the block's segment-floored lower bound, not 0 (mirrors the
    # forward MMA -- dQ is query-major with the same kb0 = 0 KV loop structure)
    assert "uint kv_lo = " in s
    assert "for (uint kb0 = kv_lo; kb0 < kv_limit; kb0 += 32) {" in s
    assert "for (uint kb0 = 0; kb0 < kv_limit; kb0 += 32) {" not in s


def test_bwd_dq_mma_packed_seg_id_row_read_is_clamped_for_over_hang_lanes() -> None:
    # A partially-over-hang query block (n not 32-aligned) has lanes with row >= n; the packed
    # predicate's seg_id[seg_off + row] read must clamp row to n-1 so those lanes read a valid,
    # in-bounds id -- its value is irrelevant since over-hang lanes are discarded before the
    # dQ store (mirrors the forward MMA over-hang clamp).
    s = build_bwd_dq_mma_source(64, causal=True, packed=True)
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" in s
    assert "seg_id[seg_off + row]" not in s  # no unclamped row read should remain


def test_bwd_dkv_scalar_packed_predicate_wraps_causal_with_segment_equality() -> None:
    # KEY-major body: the same-segment term compares the QUERY's seg_id vs the owner KEY's.
    s = build_bwd_dkv_source(64, causal=True, packed=True)
    assert "uint seg_off = b * n;" in s
    assert "seg_id[seg_off + i] == seg_id[seg_off + key]" in s
    assert "i >= key" in s  # the base causal predicate is retained inside the wrap


def test_bwd_dkv_scalar_packed_seg_id_reads_are_not_clamped() -> None:
    # The scalar dK/dV key (< n after the `if (key >= n) return;` guard) and query i
    # (loop-bound i < q_hi <= n) are always < n by construction, so both seg reads are immune
    # to the MMA over-hang bug and need no clamp.
    s = build_bwd_dkv_source(64, causal=True, packed=True)
    assert "seg_id[seg_off + i] == seg_id[seg_off + key]" in s
    assert "metal::min(key, n - 1)" not in s


def test_bwd_dkv_mma_packed_predicate_clamps_key_and_keeps_q_start() -> None:
    s = build_bwd_dkv_mma_source(64, causal=True, packed=True)
    assert "uint seg_off = b * n;" in s
    # the owner KEY index can over-hang on the key axis (a partially-full last key block), so
    # its seg_id read is clamped; the QUERY i is short-circuit-guarded by `i < q_hi` (<= n).
    assert "seg_id[seg_off + i] == seg_id[seg_off + metal::min(key, n - 1)]" in s
    assert "seg_id[seg_off + key]" not in s  # no unclamped key read should remain
    # the query-loop lower bound Q_START stays UNCHANGED (correctness is from the per-query
    # predicate; the segment-end upper-bound optimization is explicitly skipped -- YAGNI), and
    # dK/dV is key-major so there is NO kv_lo lower-bound change (unlike the query-major dQ).
    assert "metal::max(q_lo, key_base)" in s
    assert "kv_lo" not in s


def test_bwd_dq_scalar_flip_segments_inverts_only_the_equality() -> None:
    s = build_bwd_dq_source(64, causal=True, packed=True, flip_segments=True)
    assert "seg_id[seg_off + kk] != seg_id[seg_off + row]" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + row]" not in s
    assert "kk <= row" in s  # the causal half is untouched -- only the equality flips


def test_bwd_dq_mma_flip_segments_inverts_only_the_equality() -> None:
    s = build_bwd_dq_mma_source(64, causal=True, packed=True, flip_segments=True)
    assert "seg_id[seg_off + kk] != seg_id[seg_off + metal::min(row, n - 1)]" in s
    assert "seg_id[seg_off + kk] == seg_id[seg_off + metal::min(row, n - 1)]" not in s


def test_bwd_dkv_scalar_flip_segments_inverts_only_the_equality() -> None:
    s = build_bwd_dkv_source(64, causal=True, packed=True, flip_segments=True)
    assert "seg_id[seg_off + i] != seg_id[seg_off + key]" in s
    assert "seg_id[seg_off + i] == seg_id[seg_off + key]" not in s
    assert "i >= key" in s  # the causal half is untouched -- only the equality flips


def test_bwd_dkv_mma_flip_segments_inverts_only_the_equality() -> None:
    s = build_bwd_dkv_mma_source(64, causal=True, packed=True, flip_segments=True)
    assert "seg_id[seg_off + i] != seg_id[seg_off + metal::min(key, n - 1)]" in s
    assert "seg_id[seg_off + i] == seg_id[seg_off + metal::min(key, n - 1)]" not in s


def test_bwd_packed_requires_causal() -> None:
    for build in _BWD_BUILDERS:
        with pytest.raises(ValueError, match="packed"):
            build(128, causal=False, packed=True)


def test_bwd_flip_segments_requires_packed() -> None:
    for build in _BWD_BUILDERS:
        with pytest.raises(ValueError, match="flip_segments"):
            build(128, causal=True, flip_segments=True)


def test_bwd_flip_segments_mutually_exclusive_with_the_causal_perturbations() -> None:
    for build in _BWD_BUILDERS:
        with pytest.raises(ValueError, match="flip_segments"):
            build(64, causal=True, packed=True, flip_segments=True, flip_causal=True)
        with pytest.raises(ValueError, match="flip_segments"):
            build(64, causal=True, packed=True, flip_segments=True, drop_diagonal=True)
