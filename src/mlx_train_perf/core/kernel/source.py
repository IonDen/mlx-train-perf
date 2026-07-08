"""MSL source builder for the fused-CE kernel family (RT-parameterized): a dense variant
and an int4/gs64 dequant-in-kernel quantized variant.

`_SOURCE_V2E` is the RT=4, 4x4 simdgroup-matrix-tile variant (measured 2423.7 G MAC/s at
production shape). `_DENSE_TEMPLATE` derives from
it by substituting two sentinel tokens at exactly the points that vary with row-tile
count: `RT_COUNT` (row-tile array sizes + `rt`-loop bounds) and `ROWS_PER_BLOCK` (the
per-simdgroup row-block height, `8 * row_tiles`). Everything else — the `fm`/`fn` lane
mapping and the col-tile count (always 4, i.e. a fixed 32-column block) — is INVARIANT
and untouched. `build_dense_source(2)` reconstructs the RT=2 variant (measured 879.2 G
MAC/s at n=2048).

`_QUANT_TEMPLATE` derives from `_DENSE_TEMPLATE` by swapping the B-fragment's `w` reads
for a quantized dequant-in-register load and register-hoisting the per-group scale/bias
(see the comment above `_QUANT_TEMPLATE` for the hoisted-load contract). `build_quant_source`
applies the same two RT sentinels as `build_dense_source`.

Sentinel substitution uses chained `str.replace`, never an f-string or `.format()`: the
MSL body is full of C++ braces, so an f-string would need every literal `{`/`}` escaped
as `{{`/`}}` — one miss yields invalid MSL that these substring tests can't catch, only
a Metal-gated parity run would.
"""

# The RT=4 dense variant: 4x4 simdgroup-matrix tiles, 32 rows per row-block.
_SOURCE_V2E = """
    uint ygroup = thread_position_in_grid.y;     // row-block index (32 rows per block)
    uint lane = thread_position_in_threadgroup.x;
    uint n = hidden_shape[0];
    uint d = hidden_shape[1];
    uint v0 = offs[0];
    uint tcols = offs[1] - v0;
    uint r0 = ygroup * 32;
    if (r0 >= n) return;

    uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);
    uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);

    const device T* h[4];
    uint tg[4];
    float m[4], s[4], tv[4];
    bool ht[4];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < 4; ++rt) {
        uint row = r0 + fm + 8 * rt;
        h[rt] = hidden + (size_t)metal::min(row, n - 1) * d;
        tg[rt] = (row < n) ? (uint)targets[row] : 0xFFFFFFFFu;
        m[rt] = -INFINITY; s[rt] = 0.0f; tv[rt] = 0.0f; ht[rt] = false;
    }

    uint cl = tcols - 1;
    for (uint cc = 0; cc < tcols; cc += 32) {
        const device T* wp0[4];
        const device T* wp1[4];
        #pragma clang loop unroll(full)
        for (uint ct = 0; ct < 4; ++ct) {
            wp0[ct] = w + (size_t)(v0 + metal::min(cc + 8 * ct + fn, cl)) * d;
            wp1[ct] = w + (size_t)(v0 + metal::min(cc + 8 * ct + fn + 1, cl)) * d;
        }
        metal::simdgroup_float8x8 C[4][4];
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                C[rt][ct] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        uint dfull = d & ~7u;
        for (uint d0 = 0; d0 < dfull; d0 += 8) {
            metal::simdgroup_float8x8 A[4];
            metal::simdgroup_float8x8 B[4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                A[rt].thread_elements()[0] = (float)h[rt][d0 + fn];
                A[rt].thread_elements()[1] = (float)h[rt][d0 + fn + 1];
            }
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                B[ct].thread_elements()[0] = (float)wp0[ct][d0 + fm];
                B[ct].thread_elements()[1] = (float)wp1[ct][d0 + fm];
            }
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_multiply_accumulate(C[rt][ct], A[rt], B[ct], C[rt][ct]);
                }
            }
        }
        if (dfull < d) {                          // single guarded tail chunk
            metal::simdgroup_float8x8 A[4];
            metal::simdgroup_float8x8 B[4];
            uint da = dfull + fn;
            uint db = dfull + fm;
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                A[rt].thread_elements()[0] = (da < d) ? (float)h[rt][da] : 0.0f;
                A[rt].thread_elements()[1] = (da + 1 < d) ? (float)h[rt][da + 1] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                B[ct].thread_elements()[0] = (db < d) ? (float)wp0[ct][db] : 0.0f;
                B[ct].thread_elements()[1] = (db < d) ? (float)wp1[ct][db] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_multiply_accumulate(C[rt][ct], A[rt], B[ct], C[rt][ct]);
                }
            }
        }
        // epilogue: per row-tile, this lane holds 8 logits across the 32-column block
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            float bm = -INFINITY;
            float e[8];
            bool inr[8];
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                e[2 * ct] = C[rt][ct].thread_elements()[0];
                e[2 * ct + 1] = C[rt][ct].thread_elements()[1];
                inr[2 * ct] = (cc + 8 * ct + fn) < tcols;
                inr[2 * ct + 1] = (cc + 8 * ct + fn + 1) < tcols;
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                bm = metal::max(bm, inr[j] ? e[j] : -INFINITY);
            }
            bm = metal::max(bm, metal::simd_shuffle_xor(bm, (ushort)8));
            bm = metal::max(bm, metal::simd_shuffle_xor(bm, (ushort)1));
            float bs = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                bs += metal::exp((inr[j] ? e[j] : -INFINITY) - bm);
            }
            bs += metal::simd_shuffle_xor(bs, (ushort)8);
            bs += metal::simd_shuffle_xor(bs, (ushort)1);
            float nm = metal::max(m[rt], bm);
            s[rt] = s[rt] * metal::exp(m[rt] - nm) + bs * metal::exp(bm - nm);
            m[rt] = nm;
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                if (inr[2 * ct] && tg[rt] == v0 + cc + 8 * ct + fn) { tv[rt] = e[2 * ct]; ht[rt] = true; }
                if (inr[2 * ct + 1] && tg[rt] == v0 + cc + 8 * ct + fn + 1) { tv[rt] = e[2 * ct + 1]; ht[rt] = true; }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < 4; ++rt) {
        float tvv = ht[rt] ? tv[rt] : -INFINITY;
        tvv = metal::max(tvv, metal::simd_shuffle_xor(tvv, (ushort)8));
        tvv = metal::max(tvv, metal::simd_shuffle_xor(tvv, (ushort)1));
        float hf = ht[rt] ? 1.0f : 0.0f;
        hf = metal::max(hf, metal::simd_shuffle_xor(hf, (ushort)8));
        hf = metal::max(hf, metal::simd_shuffle_xor(hf, (ushort)1));
        uint row = r0 + fm + 8 * rt;
        if (fn == 0 && row < n) {
            float tile_lse = m[rt] + metal::log(s[rt]);
            float prev = lse_in[row];
            float hi = metal::max(prev, tile_lse);
            float lo = metal::min(prev, tile_lse);
            lse_out[row] = hi + metal::log(1.0f + metal::exp(lo - hi));
            tgt_out[row] = (hf > 0.5f) ? tvv : tgt_in[row];
        }
    }
"""

# Sentinel-token template, derived from _SOURCE_V2E once at import time. RT_COUNT marks
# every row-tile array size and `rt`-loop bound; ROWS_PER_BLOCK marks the per-simdgroup
# row-block height. The col-tile count (always 4 == 32 cols / 8) and the `8 * rt` row
# offset arithmetic (rt is the loop variable, not the count) are left untouched.
_DENSE_TEMPLATE = (
    _SOURCE_V2E.replace("uint r0 = ygroup * 32;", "uint r0 = ygroup * ROWS_PER_BLOCK;")
    .replace("h[4]", "h[RT_COUNT]")
    .replace("tg[4]", "tg[RT_COUNT]")
    .replace("float m[4], s[4], tv[4];", "float m[RT_COUNT], s[RT_COUNT], tv[RT_COUNT];")
    .replace("bool ht[4];", "bool ht[RT_COUNT];")
    .replace("rt < 4", "rt < RT_COUNT")
    .replace("C[4][4]", "C[RT_COUNT][4]")
    .replace("A[4]", "A[RT_COUNT]")
)


def build_dense_source(row_tiles: int) -> str:
    """MSL function body for the dense RT x 4 MMA kernel. RT in {2, 4}."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _DENSE_TEMPLATE.replace("RT_COUNT", str(row_tiles)).replace(
        "ROWS_PER_BLOCK", str(8 * row_tiles)
    )


# Dequantize-in-register helper for MLX's affine int4/gs64 layout (verified against the
# installed mx.quantize output by tests/test_quant_layout.py): `w_q` packs 8 nibbles
# per uint32 word, low-to-high (nibble i of column k lives at bits [4i, 4i+4) of word
# k>>3); `scales`/`biases` hold one value per 64-column group. Goes in metal_kernel's
# `header=` parameter — MSL forbids a nested function definition inside a [[kernel]]
# body, so helpers must be prepended outside it, not spliced into the source string.
QUANT_HELPERS = """
template <typename S>
inline float mtp_dq4(const device uint* wq_row, const device S* sc_row,
                     const device S* bi_row, uint k) {
    uint packed = wq_row[k >> 3];
    uint nib = (packed >> (4 * (k & 7))) & 0xFu;
    uint g = k >> 6;                       // group_size 64
    return (float)sc_row[g] * (float)nib + (float)bi_row[g];
}
"""

# Quant variant, derived from the (still-sentineled) dense template by replacing the
# B-fragment loads: the dense `wp0[ct]`/`wp1[ct]` device-T pointers into `w` become plain
# row indices `c0[ct]`/`c1[ct]` (clamped identically). Everything else (the A/hidden side,
# the lane mapping, the epilogue) is untouched.
#
# The hot-loop B load is REGISTER-HOISTED rather than calling `mtp_dq4` per element: a
# per-group `sc`/`bi` device load costs the same whether the group is about to repeat for
# 8 more MMA iterations or not, so it is fetched once per 64-column group into thread-local
# registers (`hsc0`/`hbi0`/`hsc1`/`hbi1`) at the uniform group-boundary branch
# `(d0 & 63u) == 0u` (correct because `d0` is 8-aligned and `fm < 8`, so every k in
# [d0, d0+8) shares group `d0 >> 6` for 8 consecutive iterations) and the per-column-block
# `wq` row pointers (`wq0`/`wq1`) are hoisted next to `c0`/`c1` so the hot loop does only a
# word load + shift + mask + fma per B element. Measured: per-element `sc`/`bi` device
# re-fetch cost 1363.4 -> 2019.5 G MAC/s at row_tiles=4 (int4/gs64, bf16, production
# shape). `mtp_dq4` stays for the guarded d-tail (cold path; never taken when d % 64 == 0,
# but the tail block must still compile), so `QUANT_HELPERS` stays a required header.
_QUANT_TEMPLATE = (
    _DENSE_TEMPLATE.replace(
        "        const device T* wp0[4];\n"
        "        const device T* wp1[4];\n"
        "        #pragma clang loop unroll(full)\n"
        "        for (uint ct = 0; ct < 4; ++ct) {\n"
        "            wp0[ct] = w + (size_t)(v0 + metal::min(cc + 8 * ct + fn, cl)) * d;\n"
        "            wp1[ct] = w + (size_t)(v0 + metal::min(cc + 8 * ct + fn + 1, cl)) * d;\n"
        "        }",
        "        uint c0[4];\n"
        "        uint c1[4];\n"
        "        #pragma clang loop unroll(full)\n"
        "        for (uint ct = 0; ct < 4; ++ct) {\n"
        "            c0[ct] = v0 + metal::min(cc + 8 * ct + fn, cl);\n"
        "            c1[ct] = v0 + metal::min(cc + 8 * ct + fn + 1, cl);\n"
        "        }",
    )
    .replace(
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] = (float)wp0[ct][d0 + fm];\n"
        "                B[ct].thread_elements()[1] = (float)wp1[ct][d0 + fm];\n"
        "            }",
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] = mtp_dq4(wq + (size_t)c0[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c0[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c0[ct] * (d >> 6), d0 + fm);\n"
        "                B[ct].thread_elements()[1] = mtp_dq4(wq + (size_t)c1[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c1[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c1[ct] * (d >> 6), d0 + fm);\n"
        "            }",
    )
    .replace(
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] = (db < d) ? (float)wp0[ct][db] : 0.0f;\n"
        "                B[ct].thread_elements()[1] = (db < d) ? (float)wp1[ct][db] : 0.0f;\n"
        "            }",
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] = (db < d) ? mtp_dq4(wq + (size_t)c0[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c0[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c0[ct] * (d >> 6), db) : 0.0f;\n"
        "                B[ct].thread_elements()[1] = (db < d) ? mtp_dq4(wq + (size_t)c1[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c1[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c1[ct] * (d >> 6), db) : 0.0f;\n"
        "            }",
    )
    # Register-hoist deltas below: block-scope declarations, hoisted `wq` row pointers,
    # the group-boundary reload, and the hot-loop B-fragment swap. None of these four
    # target strings contains an RT_COUNT/ROWS_PER_BLOCK sentinel, so applying them at the
    # template level (pre-substitution) is equivalent to applying them post-substitution.
    .replace(
        "    uint cl = tcols - 1;",
        "    uint cl = tcols - 1;\n"
        "    uint sh = 4 * fm;\n"
        "    float hsc0[4]; float hbi0[4]; float hsc1[4]; float hbi1[4];\n"
        "    const device uint* wq0[4];\n"
        "    const device uint* wq1[4];",
    )
    .replace(
        "        for (uint ct = 0; ct < 4; ++ct) {\n"
        "            c0[ct] = v0 + metal::min(cc + 8 * ct + fn, cl);\n"
        "            c1[ct] = v0 + metal::min(cc + 8 * ct + fn + 1, cl);\n"
        "        }",
        "        for (uint ct = 0; ct < 4; ++ct) {\n"
        "            c0[ct] = v0 + metal::min(cc + 8 * ct + fn, cl);\n"
        "            c1[ct] = v0 + metal::min(cc + 8 * ct + fn + 1, cl);\n"
        "            wq0[ct] = wq + (size_t)c0[ct] * (d >> 3);\n"
        "            wq1[ct] = wq + (size_t)c1[ct] * (d >> 3);\n"
        "        }",
    )
    .replace(
        "        for (uint d0 = 0; d0 < dfull; d0 += 8) {\n",
        "        for (uint d0 = 0; d0 < dfull; d0 += 8) {\n"
        "            if ((d0 & 63u) == 0u) {\n"
        "                uint g = d0 >> 6;\n"
        "                #pragma clang loop unroll(full)\n"
        "                for (uint ct = 0; ct < 4; ++ct) {\n"
        "                    hsc0[ct] = (float)sc[(size_t)c0[ct] * (d >> 6) + g];\n"
        "                    hbi0[ct] = (float)bi[(size_t)c0[ct] * (d >> 6) + g];\n"
        "                    hsc1[ct] = (float)sc[(size_t)c1[ct] * (d >> 6) + g];\n"
        "                    hbi1[ct] = (float)bi[(size_t)c1[ct] * (d >> 6) + g];\n"
        "                }\n"
        "            }\n",
    )
    .replace(
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] = mtp_dq4(wq + (size_t)c0[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c0[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c0[ct] * (d >> 6), d0 + fm);\n"
        "                B[ct].thread_elements()[1] = mtp_dq4(wq + (size_t)c1[ct] * (d >> 3),\n"
        "                                                     sc + (size_t)c1[ct] * (d >> 6),\n"
        "                                                     bi + (size_t)c1[ct] * (d >> 6), d0 + fm);\n"
        "            }",
        "            #pragma clang loop unroll(full)\n"
        "            for (uint ct = 0; ct < 4; ++ct) {\n"
        "                B[ct].thread_elements()[0] ="
        " (float)((wq0[ct][d0 >> 3] >> sh) & 0xFu) * hsc0[ct] + hbi0[ct];\n"
        "                B[ct].thread_elements()[1] ="
        " (float)((wq1[ct][d0 >> 3] >> sh) & 0xFu) * hsc1[ct] + hbi1[ct];\n"
        "            }",
    )
)


def build_quant_source(row_tiles: int) -> str:
    """MSL function body for the int4/gs64 dequant-in-kernel MMA kernel. RT in {2, 4}."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _QUANT_TEMPLATE.replace("RT_COUNT", str(row_tiles)).replace(
        "ROWS_PER_BLOCK", str(8 * row_tiles)
    )


# ---------------------------------------------------------------------------------------
# Backward (d_hidden-only) kernel -- Task 16b step 2. v0-CORRECT: a simple simdgroup-per-
# row accumulation, no simdgroup_matrix tiling. Speed is explicitly out of scope for this
# rung (perf rungs land in later steps); this body exists to prove the derivation and the
# chained-tile-launch accumulator protocol, not to compete with the forward's MMA rate.
#
# Math (derived from d(nll_i)/d(hidden_i) by hand, then cross-checked against
# core/chunked.py::chunked_backward -- the proven oracle -- BEFORE this body was written):
#   nll_i = lse_i - tgt_i,  tgt_i = logit_i,target_i,  logit_i,j = hidden_i . w_j
#   P_i,j = exp(logit_i,j - lse_i)      (regenerated per vocab tile; lse_i is the SAVED
#                                        residual, deliberately never recomputed here)
#   d(nll_i)/d(hidden_i) = sum_j (P_i,j - onehot(j == target_i)) * w_j
#   d_hidden_i = cotangent_i * d(nll_i)/d(hidden_i)
# accumulated ACROSS chained vocab-tile launches, the SAME feed-forward protocol as the
# forward's lse_in/lse_out chaining (full buffers + in-kernel offsets, never a Python-side
# slice into a chained launch). d_hidden has no cross-ROW-BLOCK accumulation (unlike d_w's
# step 3), so this body needs no atomics / split-K partials.
#
# d_hidden_in/d_hidden_out are FIXED fp32 buffers, never templated on T: chaining the
# accumulator in the hidden/weight dtype (bf16) would re-round on every tile launch --
# exactly the mixed-precision bug chunked_backward's own docstring warns against. The
# launcher casts down to hidden.dtype exactly once, after the last tile launch.
#
# ROWS_PER_BLOCK reuses `build_dense_source`'s row-block sizing convention (8 * row_tiles)
# so this body stays launch-compatible with the SAME grid/threadgroup arithmetic
# `launch.forward` already uses -- a later perf rung can swap the body without touching the
# launcher's dispatch shape. Internally, though, ROWS_PER_BLOCK rows are processed
# SEQUENTIALLY by one simdgroup (all 32 lanes cooperating on one row at a time via
# `simd_sum`), not in parallel across an fm/fn MMA lane split -- v0-correct, deliberately
# not tiled for throughput.
_BACKWARD_DHIDDEN_TEMPLATE = """
    uint ygroup = thread_position_in_grid.y;      // row-block index
    uint lane = thread_position_in_threadgroup.x;  // 0..31 == simd lane
    uint n = hidden_shape[0];
    uint d = hidden_shape[1];
    uint v0 = offs[0];
    uint tcols = offs[1] - v0;
    uint r0 = ygroup * ROWS_PER_BLOCK;
    if (r0 >= n) return;

    for (uint rt = 0; rt < ROWS_PER_BLOCK; ++rt) {
        uint row = r0 + rt;
        if (row >= n) continue;
        uint tgt_col = (uint)targets[row];
        float ct_row = cotangent[row];
        float lse_row = lse[row];
        const device T* hrow = hidden + (size_t)row * d;
        const device float* dh_in_row = d_hidden_in + (size_t)row * d;
        device float* dh_row = d_hidden_out + (size_t)row * d;

        for (uint c = 0; c < tcols; ++c) {
            const device T* wrow = w + (size_t)(v0 + c) * d;
            float part = 0.0f;
            for (uint i = lane; i < d; i += 32) {
                part += (float)hrow[i] * (float)wrow[i];
            }
            float logit = metal::simd_sum(part);
            float onehot = (tgt_col == v0 + c) ? 1.0f : 0.0f;
            float coeff = (metal::exp(logit - lse_row) - onehot) * ct_row;
            for (uint i = lane; i < d; i += 32) {
                float prev = (c == 0) ? dh_in_row[i] : dh_row[i];
                dh_row[i] = prev + coeff * (float)wrow[i];
            }
        }
    }
"""


def build_backward_dhidden_source(row_tiles: int) -> str:
    """MSL function body for the d_hidden-only backward kernel. RT in {2, 4}, same
    row-block-sizing convention as `build_dense_source` (see the comment above this
    function for the full derivation and design rationale)."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _BACKWARD_DHIDDEN_TEMPLATE.replace("ROWS_PER_BLOCK", str(8 * row_tiles))


# ---------------------------------------------------------------------------------------
# Backward (d_hidden-only) MMA kernel -- Task 16b step 4. The frozen/QLoRA-head PERF rung:
# same math and same (hidden, w, targets, lse, cotangent) -> d_hidden contract as the v0
# scalar kernel above, but restructured as a fused pair of simdgroup-matrix GEMMs sharing
# the weight tile, so the dominant cost (regenerating logits) runs at the forward's MMA
# rate instead of v0's zero-reuse scalar rate.
#
#   d_hidden_i = cotangent_i * sum_j (P_i,j - onehot(j == target_i)) * w_j,
#   P_i,j = exp(logit_i,j - lse_i),   logit_i,j = hidden_i . w_j   (lse is the SAVED
#                                                                   forward residual)
#
# factors, per vocab sub-tile (32 columns == vsub), into:
#   GEMM-A  logit_block(32rows x 32vsub) = H_block(32rows x d) @ W_sub(32vsub x d)^T
#           -- EXACTLY the forward's 4x4 `simdgroup_float8x8` accumulation over d; reused
#           verbatim via `_GEMM_A_LOGIT_REGEN` (sliced out of `_DENSE_TEMPLATE`).
#   P-form  g = (exp(logit - lse_row) - onehot) * cot_row, applied in place to the C tiles'
#           per-lane `thread_elements()` (a pure elementwise map -- no data moves between
#           lanes), masking columns past tcols to 0 so the clamp-duplicated tail the logit
#           regen loaded contributes nothing to GEMM-B (0 * w == 0).
#   GEMM-B  d_hidden(32rows x d) += G(32rows x 32vsub) @ W_sub(32vsub x d)
#           -- a SECOND MMA with the vocab as the contraction axis, streaming the (32 x d)
#           output over d in 8-column slices. This is the genuinely new structure.
#
# GEMM-B's output structure -- the design choice the perf hinges on: it is the SIMPLEST
# correct one that gets GEMM-A onto MMA. Each lane owns a fixed, disjoint set of d_hidden
# elements ((fm, fn) bijectively partition the 8x8 output tile -- verified against Apple's
# shipped `steel/gemm/mma.h::BaseMMAFrag<8,8>::get_coord`, which returns {fn, fm} so the
# lane's two elements sit at (row=fm, col=fn) and (row=fm, col=fn+1)). So d_hidden needs NO
# atomics (unlike d_w's cross-row-block contention): each lane read-modify-writes ONLY its
# own elements, in DEVICE memory, serially across vsub blocks -- the accumulator chains
# vsub-to-vsub the way the forward chains lse tile-to-tile (first vsub reads the cross-
# launch input accumulator `d_hidden_in`, later vsubs read back what this same lane just
# wrote to `d_hidden_out` -- same-thread program order, no barrier). The cost is heavy
# device read-modify-write traffic (the full (32 x d) accumulator is re-read/re-written per
# vsub); a d-slice split across simdgroups through threadgroup memory would cut that, but
# whether GEMM-B's traffic actually dominates is a MEASURED question for a later rung, not
# something to pre-optimize here -- correctness of GEMM-A-on-MMA is this rung's deliverable.
#
# d_hidden_in/d_hidden_out stay FIXED fp32 buffers, never templated on T (chaining the
# accumulator in bf16 would re-round every vsub -- the mixed-precision bug the v0 kernel and
# chunked_backward both guard against); the launcher casts to hidden.dtype once at the end.
# Register economics: G lives in the 4x4 C tiles (32 fp32/lane -- the family-independent
# optimum), reused in place from GEMM-A, plus one transient output tile Db + one weight tile
# Wb per (d-slice, row-tile). Measured (devtools/regpressure.py, mlx 0.31.2 / M1 Max): the
# fused kernel compiles at the forward's OWN ceiling -- 448 (RT=4) / 640 (RT=2), no drop --
# because reusing the C tiles in place for G and streaming a single transient output tile
# keeps peak per-lane register state flat. The ceiling is a register-pressure telltale only,
# never a rate verdict (the measured production rate is the verdict) -- see the regpressure
# module docstring.

# GEMM-A reused VERBATIM from the forward: the exact vocab-tile `cc` loop that accumulates
# the 4x4 `simdgroup_float8x8` logit block C[rt][ct] over d (full-chunks + one guarded
# tail). Sliced from `_DENSE_TEMPLATE` by stable anchors -- the pre-`cc`-loop `uint cl`
# declaration through the byte just before the forward's online-LSE epilogue -- so it stays
# byte-identical to the forward and keeps the RT_COUNT sentinels the builder substitutes.
_GEMM_A_LOGIT_REGEN = _DENSE_TEMPLATE[
    _DENSE_TEMPLATE.index("    uint cl = tcols - 1;\n") : _DENSE_TEMPLATE.index("        // epilogue:")
]

# Backward-specific prologue: like the forward's, it caches h[rt] (for GEMM-A) and tg[rt]
# (for the onehot), but drops the forward's online-LSE state (m/s/tv/ht) and adds the
# per-row SAVED residual lse_r[rt] and cotangent cot_r[rt] the P-formation needs. ROWS_PER_
# BLOCK / RT_COUNT are the same sentinels build_dense_source uses.
_BACKWARD_DHIDDEN_MMA_PROLOGUE = """
    uint ygroup = thread_position_in_grid.y;      // row-block index
    uint lane = thread_position_in_threadgroup.x;  // 0..31 == simd lane
    uint n = hidden_shape[0];
    uint d = hidden_shape[1];
    uint v0 = offs[0];
    uint tcols = offs[1] - v0;
    uint r0 = ygroup * ROWS_PER_BLOCK;
    if (r0 >= n) return;

    uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);
    uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);

    const device T* h[RT_COUNT];
    uint tg[RT_COUNT];
    float lse_r[RT_COUNT], cot_r[RT_COUNT];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT_COUNT; ++rt) {
        uint row = r0 + fm + 8 * rt;
        h[rt] = hidden + (size_t)metal::min(row, n - 1) * d;
        tg[rt] = (row < n) ? (uint)targets[row] : 0xFFFFFFFFu;
        lse_r[rt] = (row < n) ? lse[row] : 0.0f;
        cot_r[rt] = (row < n) ? cotangent[row] : 0.0f;
    }
"""

# P-formation + GEMM-B, spliced in where the forward's epilogue was (still inside the `cc`
# loop `_GEMM_A_LOGIT_REGEN` opened; the trailing `}` here closes it). See the block comment
# above build_backward_dhidden_mma_source for the derivation and the fragment layout.
_BACKWARD_DHIDDEN_MMA_GEMMB = """        // P-formation: regenerated logit tile C[rt][ct] -> gradient coefficient g, in place
        // in simdgroup-matrix state. Columns past tcols (clamp-duplicated by the logit
        // regen) are masked to 0 so GEMM-B ignores them.
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < RT_COUNT; ++rt) {
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                uint col0 = cc + 8 * ct + fn;
                uint col1 = cc + 8 * ct + fn + 1;
                float g0 = (col0 < tcols)
                    ? (metal::exp(C[rt][ct].thread_elements()[0] - lse_r[rt])
                       - ((tg[rt] == v0 + col0) ? 1.0f : 0.0f)) * cot_r[rt]
                    : 0.0f;
                float g1 = (col1 < tcols)
                    ? (metal::exp(C[rt][ct].thread_elements()[1] - lse_r[rt])
                       - ((tg[rt] == v0 + col1) ? 1.0f : 0.0f)) * cot_r[rt]
                    : 0.0f;
                C[rt][ct].thread_elements()[0] = g0;
                C[rt][ct].thread_elements()[1] = g1;
            }
        }
        // GEMM-B: d_hidden(rows x d) += G(rows x 32) @ W_sub(32 x d). C now holds G; W_sub's
        // fragment maps fragment-row fm -> vocab-in-chunk, fragment-col fn -> d-column
        // (mirror of GEMM-A's B load, contraction axis swapped from d to vocab). Each lane
        // read-modify-writes ONLY its own output elements; the accumulator chains vsub-to-
        // vsub -- the first vsub reads the cross-launch input accumulator d_hidden_in, later
        // vsubs read back d_hidden_out. The read is a per-value ternary (not a chosen
        // pointer): d_hidden_in is a `const device float*` input, so it cannot alias the
        // mutable d_hidden_out output pointer -- same value-select the v0 kernel uses.
        uint dfo = d & ~7u;
        for (uint d0d = 0; d0d < dfo; d0d += 8) {
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < RT_COUNT; ++rt) {
                metal::simdgroup_float8x8 Db = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_float8x8 Wb;
                    const device T* wr = w + (size_t)(v0 + metal::min(cc + 8 * ct + fm, cl)) * d;
                    Wb.thread_elements()[0] = (float)wr[d0d + fn];
                    Wb.thread_elements()[1] = (float)wr[d0d + fn + 1];
                    metal::simdgroup_multiply_accumulate(Db, C[rt][ct], Wb, Db);
                }
                uint row = r0 + fm + 8 * rt;
                if (row < n) {
                    size_t base = (size_t)row * d + d0d + fn;
                    float p0 = (cc == 0) ? d_hidden_in[base] : d_hidden_out[base];
                    float p1 = (cc == 0) ? d_hidden_in[base + 1] : d_hidden_out[base + 1];
                    d_hidden_out[base] = p0 + Db.thread_elements()[0];
                    d_hidden_out[base + 1] = p1 + Db.thread_elements()[1];
                }
            }
        }
        if (dfo < d) {                                // single guarded d-tail slice
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < RT_COUNT; ++rt) {
                metal::simdgroup_float8x8 Db = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_float8x8 Wb;
                    const device T* wr = w + (size_t)(v0 + metal::min(cc + 8 * ct + fm, cl)) * d;
                    uint ca = dfo + fn;
                    uint cb = dfo + fn + 1;
                    Wb.thread_elements()[0] = (ca < d) ? (float)wr[ca] : 0.0f;
                    Wb.thread_elements()[1] = (cb < d) ? (float)wr[cb] : 0.0f;
                    metal::simdgroup_multiply_accumulate(Db, C[rt][ct], Wb, Db);
                }
                uint row = r0 + fm + 8 * rt;
                if (row < n) {
                    size_t rbase = (size_t)row * d;
                    uint ca = dfo + fn;
                    uint cb = dfo + fn + 1;
                    if (ca < d) {
                        float p = (cc == 0) ? d_hidden_in[rbase + ca] : d_hidden_out[rbase + ca];
                        d_hidden_out[rbase + ca] = p + Db.thread_elements()[0];
                    }
                    if (cb < d) {
                        float p = (cc == 0) ? d_hidden_in[rbase + cb] : d_hidden_out[rbase + cb];
                        d_hidden_out[rbase + cb] = p + Db.thread_elements()[1];
                    }
                }
            }
        }
    }
"""

_BACKWARD_DHIDDEN_MMA_TEMPLATE = (
    _BACKWARD_DHIDDEN_MMA_PROLOGUE + _GEMM_A_LOGIT_REGEN + _BACKWARD_DHIDDEN_MMA_GEMMB
)


def build_backward_dhidden_mma_source(row_tiles: int) -> str:
    """MSL function body for the fused two-GEMM d_hidden-only backward kernel (Task 16b
    step 4). RT in {2, 4}, same row-block-sizing convention as `build_dense_source`; the
    logit-regeneration GEMM is reused verbatim from the forward (see the block comment above
    for the full derivation, the fragment layout, and the GEMM-B output-structure choice)."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _BACKWARD_DHIDDEN_MMA_TEMPLATE.replace("RT_COUNT", str(row_tiles)).replace(
        "ROWS_PER_BLOCK", str(8 * row_tiles)
    )


# ---------------------------------------------------------------------------------------
# Backward (d_w) kernel -- Task 16b step 3. v0-CORRECT, same simdgroup-per-row style as
# `build_backward_dhidden_source`: no simdgroup_matrix tiling, speed explicitly deferred.
#
# Math (derived from d(nll_i)/d(w_j) by hand, then cross-checked against
# core/chunked.py::chunked_backward -- the proven oracle -- BEFORE this body was written):
#   P_i,j = exp(logit_i,j - lse_i)      (regenerated per vocab tile from the SAVED lse
#                                        residual, exactly as build_backward_dhidden_source
#                                        does -- never recomputed from scratch)
#   d(nll_i)/d(w_j)     = (P_i,j - onehot(j == target_i)) * hidden_i
#   d_w_j               = sum_i cotangent_i * d(nll_i)/d(w_j)
# matches chunked_backward's `g32 = (p - onehot) * ct[:, None]` then `g.T @ hidden` exactly
# (d_w[j,:] = sum_i g[i,j] * hidden[i,:]). `tgt` (the raw target-logit residual) never
# appears here either, same as d_hidden's derivation -- only `lse` is needed.
#
# THE ONE NEW MECHANISM vs d_hidden: d_hidden accumulates PER ROW (each context row i is
# owned by exactly one row-block -- no cross-row-block contention). d_w accumulates PER
# VOCAB COLUMN j, and EVERY row-block (a different chunk of context rows) contributes to
# the SAME d_w[j,:] slice -- genuine cross-row-block contention, exactly what Task 16b
# Step 1 ground-truthed (scripts/ground_truth_atomic_outputs.py): `atomic_outputs=True` +
# `atomic_fetch_add_explicit(&out[idx], val, memory_order_relaxed)` on a native Metal
# `device atomic<float>*` output -- plain assignment fails to compile against it.
#
# BUFFER-PERSISTENCE DESIGN (ground-truthed BEFORE writing this body, in a throwaway
# script mirroring scripts/ground_truth_atomic_outputs.py's mechanism, per the task's
# instruction not to guess): `mx.fast.metal_kernel` allocates a genuinely FRESH output
# buffer on every call (re-applying `init_value` every time) -- there is no persistent /
# mutable-across-calls buffer, confirmed by calling the SAME atomic kernel repeatedly with
# different row-block counts and seeing each call's result depend ONLY on its own inputs
# (zero bleed-through). Given that, and given each vocab TILE's d_w rows are structurally
# DISJOINT from every other tile's rows (column j belongs to exactly one tile), the design
# needs no `d_w_in`/`d_w_out` accumulator chain at all (unlike d_hidden's cross-TILE
# chaining, which is real because the SAME row is revisited by every tile): each tile
# launch outputs ONLY its own (tcols, d) fp32 slice (`atomic_outputs=True`,
# `init_value=0.0`), contended ACROSS ROW-BLOCKS within that one launch, and the launcher
# assembles the full (v, d) buffer with a plain `mx.concatenate` across tiles -- the exact
# pattern `chunked_backward`'s own trainable-head branch already uses
# (`d_w_chunks.append(g.T @ hidden)` then `mx.concatenate(d_w_chunks, axis=0)`). This also
# means `v0` is used ONLY to compute the ABSOLUTE column index for the target/onehot
# comparison and for indexing into `w` -- never to offset the OUTPUT address, which stays
# tile-local (`c * d + i`, `c` in `[0, tcols)`).
#
# d_w_out is a FIXED fp32 atomic output, never templated on T -- same fp32-accumulator
# rule as d_hidden_in/d_hidden_out. The launcher casts the concatenated result down to
# `w.dtype` exactly once, after every tile launch.
_BACKWARD_DW_TEMPLATE = """
    uint ygroup = thread_position_in_grid.y;      // row-block index (context rows)
    uint lane = thread_position_in_threadgroup.x;  // 0..31 == simd lane
    uint n = hidden_shape[0];
    uint d = hidden_shape[1];
    uint v0 = offs[0];
    uint tcols = offs[1] - v0;
    uint r0 = ygroup * ROWS_PER_BLOCK;
    if (r0 >= n) return;

    for (uint rt = 0; rt < ROWS_PER_BLOCK; ++rt) {
        uint row = r0 + rt;
        if (row >= n) continue;
        uint tgt_col = (uint)targets[row];
        float ct_row = cotangent[row];
        float lse_row = lse[row];
        const device T* hrow = hidden + (size_t)row * d;

        for (uint c = 0; c < tcols; ++c) {
            const device T* wrow = w + (size_t)(v0 + c) * d;
            float part = 0.0f;
            for (uint i = lane; i < d; i += 32) {
                part += (float)hrow[i] * (float)wrow[i];
            }
            float logit = metal::simd_sum(part);
            float onehot = (tgt_col == v0 + c) ? 1.0f : 0.0f;
            float coeff = (metal::exp(logit - lse_row) - onehot) * ct_row;
            for (uint i = lane; i < d; i += 32) {
                atomic_fetch_add_explicit(&d_w_out[c * d + i], coeff * (float)hrow[i],
                                          metal::memory_order_relaxed);
            }
        }
    }
"""


def build_backward_dw_source(row_tiles: int) -> str:
    """MSL function body for the d_w backward kernel. RT in {2, 4}, same row-block-sizing
    convention as `build_dense_source`/`build_backward_dhidden_source` (see the comment
    above this function for the full derivation, the cross-row-block atomics mechanism,
    and the ground-truthed buffer-persistence design)."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _BACKWARD_DW_TEMPLATE.replace("ROWS_PER_BLOCK", str(8 * row_tiles))
