"""MSL source builder for the dense fused-CE kernel (RT-parameterized).

`_SOURCE_V2E` is a VERBATIM port of `mlx-train-perf-spike/kernel_v2e.py:10-151`'s
`_SOURCE` (the RT=4, 4x4 simdgroup-matrix-tile variant). `_DENSE_TEMPLATE` derives from
it by substituting two sentinel tokens at exactly the points that vary with row-tile
count: `RT_COUNT` (row-tile array sizes + `rt`-loop bounds) and `ROWS_PER_BLOCK` (the
per-simdgroup row-block height, `8 * row_tiles`). Everything else — the `fm`/`fn` lane
mapping and the col-tile count (always 4, i.e. a fixed 32-column block) — is INVARIANT
and untouched. `build_dense_source(2)` reconstructs the RT=2 variant, structurally
equivalent to `mlx-train-perf-spike/kernel_v2d.py`'s `_SOURCE` (measured 879.2 G MAC/s).

Sentinel substitution uses chained `str.replace`, never an f-string or `.format()`: the
MSL body is full of C++ braces, so an f-string would need every literal `{`/`}` escaped
as `{{`/`}}` — one miss yields invalid MSL that these substring tests can't catch, only
a Metal-gated parity run would.
"""

# Port of mlx-train-perf-spike/kernel_v2e.py:10-151 (`_SOURCE`), VERBATIM.
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

# Quant variant, derived from the (still-sentineled) dense template by replacing ONLY the
# B-fragment loads: the dense `wp0[ct]`/`wp1[ct]` device-T pointers into `w` become plain
# row indices `c0[ct]`/`c1[ct]` (clamped identically), and every B load dequantizes via
# `mtp_dq4` instead of a straight cast. Everything else (the A/hidden side, the lane
# mapping, the epilogue) is untouched.
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
)


def build_quant_source(row_tiles: int) -> str:
    """MSL function body for the int4/gs64 dequant-in-kernel MMA kernel. RT in {2, 4}."""
    if row_tiles not in (2, 4):
        raise ValueError(f"row_tiles must be 2 or 4, got {row_tiles}")
    return _QUANT_TEMPLATE.replace("RT_COUNT", str(row_tiles)).replace(
        "ROWS_PER_BLOCK", str(8 * row_tiles)
    )
