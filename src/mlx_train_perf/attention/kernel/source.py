"""MSL source builder for the flash-attention FORWARD kernel v0 (O + L), correctness-only.

Sentinel-token templating, the 0.1.0 `core/kernel/source.py` convention (provably lossless
`str.replace`, never an f-string -- the MSL body is full of C++ braces an f-string would
need doubled): `HEAD_DIM` is substituted with the compile-time head dimension (fixes the
per-thread `qreg`/`acc` array sizes and the D-loop bounds), and `KEEP_CMP` is the per-key
causal keep predicate.

v0 design (deliberately slow, speed is T6's job):
- ONE thread per query row -- no simdgroup cooperation, no `simdgroup_matrix`. Each thread
  runs the full online-softmax recurrence over the keys for its row, holding `m`/`l`/`acc`
  in fp32 registers (`alpha = exp(m_old - m_new)` rescale; `L = m + log(l)` at the end) --
  the sdpa_vector.h / logsumexp.h idiom specialized to Bk=1 (per key), which is the
  simplest form to verify.
- Causal is a per-key keep predicate applied BEFORE the key touches the running max
  (`KEEP_CMP`, normally `kk <= row`), so the diagonal is masked before rowmax and every
  key above the diagonal contributes nothing -- the spec's "diagonal masked before rowmax"
  specialized to Bk=1. (T6 turns this into KV-block loop bounds + in-tile masking.)
- GQA `kv_head = q_head // group_size` is computed in-kernel from the q/k head counts;
  K/V are never expanded.
- fp32 accumulators throughout; inputs read through the `T` template (bf16 or fp32) and
  upcast to fp32; O is written back in the input dtype (single cast on store, matching the
  reference's `o32.astype(q.dtype)`); L is fp32.
- Full buffers + an in-kernel query-row offset (`qoffs`), never a Python-side slice: the
  launcher splits over disjoint query-row ranges and each dispatch writes its own tile-
  local (b, hq, rows, d) O chunk (the CE forward's disjoint-output pattern).

The `flip_causal` arg is TEST-ONLY: it flips the causal comparison to the WRONG triangle
(`kk >= row`) so a parity run FAILS -- the wrong-mask perturbation that proves the suite
can detect a mask bug. Never used by production code.
"""
_KERNEL_HEAD_DIMS = (64, 96, 128)

# The v0 forward body. `qoffs[0]`/`qoffs[1]` are the absolute [r0, r1) query-row range this
# dispatch owns; `local_row` is the tile-local output row. Reads use the ABSOLUTE row into
# the full q/k/v buffers; writes stay tile-local (o_out/l_out are this dispatch's own chunk,
# shape (b, hq, rows_this, HEAD_DIM) / (b, hq, rows_this)).
_FWD_TEMPLATE = """
    uint local_row = thread_position_in_grid.x;   // 0..rows_this-1 (output row, tile-local)
    uint bh = thread_position_in_grid.y;          // 0..(b*hq)-1
    uint r0 = qoffs[0];
    uint r1 = qoffs[1];
    uint rows_this = r1 - r0;
    if (local_row >= rows_this) return;           // defensive (dispatchThreads clamps x)

    uint hq = q_shape[1];
    uint n = q_shape[2];
    uint hkv = k_shape[1];
    uint group_size = hq / hkv;

    uint b = bh / hq;
    uint h = bh % hq;
    uint kvh = h / group_size;
    uint row = r0 + local_row;                     // absolute query position

    float scale = scale_in[0];

    // Row-contiguous base offsets (ensure_row_contiguous=True for v0).
    size_t q_base = ((size_t)(b * hq + h) * n + row) * HEAD_DIM;
    size_t kv_base = (size_t)(b * hkv + kvh) * n * HEAD_DIM;   // + kk * HEAD_DIM per key

    float qreg[HEAD_DIM];
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        qreg[dd] = (float)q[q_base + dd];
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc[HEAD_DIM];
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        acc[dd] = 0.0f;
    }

    for (uint kk = 0; kk < n; ++kk) {
        bool keep = (KEEP_CMP);
        if (!keep) { continue; }
        const device T* krow = k + kv_base + (size_t)kk * HEAD_DIM;
        float dot = 0.0f;
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            dot += qreg[dd] * (float)krow[dd];
        }
        float score = dot * scale;
        float m_new = metal::max(m, score);
        float alpha = metal::exp(m - m_new);       // rescale the running accumulators
        float p = metal::exp(score - m_new);
        l = l * alpha + p;
        const device T* vrow = v + kv_base + (size_t)kk * HEAD_DIM;
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            acc[dd] = acc[dd] * alpha + p * (float)vrow[dd];
        }
        m = m_new;
    }

    float inv = 1.0f / l;                           // causal: l >= 1 (row attends >= key row)
    size_t o_base = ((size_t)(b * hq + h) * rows_this + local_row) * HEAD_DIM;
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        o_out[o_base + dd] = (T)(acc[dd] * inv);
    }
    l_out[(size_t)(b * hq + h) * rows_this + local_row] = m + metal::log(l);
"""


def build_fwd_source(head_dim: int, *, causal: bool = True, flip_causal: bool = False) -> str:
    """MSL function body for the v0 flash-attention forward kernel (O + L).

    `head_dim` in {64, 96, 128} (the kernel's supported head dims) is baked in as a
    compile-time constant. `causal=True` masks each key with `kk <= row` before it enters
    the running max; `causal=False` keeps every key. `flip_causal` is TEST-ONLY -- it flips
    the causal comparison to the wrong triangle (`kk >= row`) so a parity run FAILS.
    """
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
    if not causal:
        keep = "true"
    elif flip_causal:
        keep = "kk >= row"
    else:
        keep = "kk <= row"
    return _FWD_TEMPLATE.replace("HEAD_DIM", str(head_dim)).replace("KEEP_CMP", keep)


# ---------------------------------------------------------------------------------------
# Rung 2: register-resident P@V MMA O-path with D-slabbing -- the throughput restructure of
# rung 1's threadgroup-O body. Correctness contract is IDENTICAL (O + L, same online-softmax
# math, same GQA, same causal mask + KEEP_CMP flip contract, same qoffs/full-buffer offsets);
# what changes is the O path. Rung 1 routed the whole online-softmax epilogue through
# THREADGROUP memory -- a 32x32 S stage, a scalar-per-lane softmax, and a 16 KB O-accumulator
# read-modified-written PER KV block with barriers -- and that threadgroup O-traffic dominated
# the loop. Rung 2 removes ALL threadgroup memory: the score block stays in the C tiles, the
# per-row max/sum are extracted by the CE forward's proven `simd_shuffle_xor(8),(1)` idiom, and
# O is accumulated as register-resident `simdgroup_float8x8` tiles via a SECOND MMA P@V --
# adapting `core/kernel/source.py::_BACKWARD_DHIDDEN_MMA_GEMMB`, which forms a coefficient in
# simdgroup-matrix state in place and MMAs it against a second operand without leaving registers.
#
# Tile geometry (unchanged from rung 1 for the QK^T half):
# - ONE THREADGROUP == ONE simdgroup (32 lanes) owns ONE query block of Bq=32 rows == 4
#   row-tiles of 8 (RT=4), the KV axis tiled in Bk=32-key blocks == 4 col-tiles of 8.
#   S_block(32x32) = scale * Q_block @ K_block^T is the CE inner GEMM verbatim (`fm`/`fn` lane
#   mapping, d-chunk loop over the head dim), inheriting the CE kernel's QK^T parity proof.
#   Head dims {64,96,128} are all %8==0, so the QK^T d-loop needs no guarded tail chunk.
# - CAUSAL BLOCK SKIPPING: the KV loop stops at `kv_limit` = min(n, block_end + 1); blocks
#   fully above the diagonal are never entered, the diagonal block masked IN-TILE per (row,key)
#   before the row max (`KEEP_CMP`, the flip-test perturbation point).
#
# THE O PATH -- register-resident P@V MMA with D-slabbing (the rung-2 restructure):
# - P-FORMATION IN SIMDGROUP STATE: after the QK^T MMA the raw scores are scaled in place, the
#   per-row block max is reduced from the lane's OWNED thread_elements via `simd_shuffle_xor`
#   (each lane holds 8 elements of its 4 fragment rows across the 4 col-tiles; the 8-then-1 XOR
#   masks reduce across the 4 lanes sharing an `fm`), then P = exp(scaled_score - m_new) is
#   written back INTO the C tiles' thread_elements in place (masked to 0 for tail/causal-excluded
#   keys), the `_BACKWARD_DHIDDEN_MMA_GEMMB` P-formation idiom. NO threadgroup S stage.
# - O AS A SECOND MMA: O_block += P @ V_slab accumulated into `C_o[4][D_SLAB/8]`
#   `simdgroup_float8x8` tiles held register-resident across the KV loop. P (= the C tiles, in
#   query-row x key layout) is fed DIRECTLY as the MMA left operand -- exactly as backward
#   GEMM-B feeds G(rows x vocab) -- and V_slab's fragment maps key `fm` -> V row, D-col `fn`
#   (mirror of GEMM-B's W_sub load). Each lane's two C_o thread_elements both live at matrix
#   row 8*rt+fm, so the online-softmax rescale multiplies them by that row's `alpha[rt]` in
#   place (NOT a simdgroup matmul -- a per-thread-element scale, the P-formation access pattern).
# - D-SLABBING (the register-budget lever): a full C_o[4][HEAD_DIM/8] is 64 tiles at d=128 ==
#   128 fp32/lane, and with the 16 live P tiles (32 fp32/lane) that is deep in the spill zone
#   (user-metal-kernels: 32 C-tiles = 64 fp32/lane collapse; the ~32-fp32/lane accumulator
#   optimum is family-independent). So the D dimension is SLABBED: the D-slab is the OUTER loop
#   and the KV loop the INNER, with C_o holding only D_SLAB columns at a time. Live simdgroup
#   state during GEMM-B = 16 P tiles + RT*(D_SLAB/8) C_o tiles; D_SLAB is chosen from the
#   regpressure ceilings so this stays under the ~128-GPR budget (see `_FWD_MMA_D_SLAB`).
#   BOUNDING C_o forces RECOMPUTING the QK^T + softmax per slab (re-reading K): the OUTER-slab
#   structure means the score block is regenerated for each slab pass (m/l are deterministic
#   and recomputed identically). This was chosen over "stage P to threadgroup and re-load per
#   slab" because the register bottleneck is the PERSISTENT C_o accumulator, not P -- staging P
#   frees the transient P tiles but does nothing for a full-D C_o, which can only be bounded by
#   slabbing it, which in turn forces the recompute. The recompute cost is num_slabs x the QK^T
#   MMA (P@V runs once total, split across slabs); the controller measures whether removing the
#   threadgroup O round-trip nets a win against that recompute.
# - fp32 accumulators throughout (C tiles + C_o + m/l all fp32); L = m + log(l) fp32; O cast to
#   the input dtype `T` on the final store -- identical to rung 1's contract.
#
# The reduction ORDER differs from rung 1 (P@V MMA fp32 reassociation + per-slab recompute vs
# the scalar per-key threadgroup accumulate), so the MMA variant's parity worsts are measured
# and pinned SEPARATELY from the scalar pins -- never by widening a scalar pin. Per-row
# independence is preserved (a row's O/L depend only on its own absolute position and keys),
# so the query-range split stays bit-identical to a single dispatch.
#
# `HEAD_DIM` bakes the head dim into the D-slab loop bound and the QK^T d-loop; `D_SLAB` /
# `D_SLAB_TILES` bake the slab width and its col-tile count (D_SLAB/8) into the C_o tile array
# and the D-slab loop step; `KEEP_CMP` is the per-key causal keep predicate; `KV_LIMIT` is the
# KV-block loop bound. Sentinel `str.replace` (never an f-string -- the MSL body is full of C++
# braces), and `D_SLAB_TILES` is substituted BEFORE `D_SLAB` (the longer token first, so the
# shared prefix does not corrupt it) -- the 0.1.0 convention.

# D-slab width for the register-resident C_o accumulator, chosen from the regpressure ceilings
# (tests/test_devtools.py, mlx 0.32.0 / M1 Max) at the ~128-GPR budget: C_o holds RT*(D_SLAB/8)
# simdgroup tiles (2 fp32/lane each) live SIMULTANEOUSLY with the 16 QK/P tiles (32 fp32/lane)
# during GEMM-B. A single value divides all supported head dims {64,96,128} (their GCD is 32),
# keeps the compiled ceiling in the healthy MMA class at every head dim, and bounds the QK^T
# recompute to HEAD_DIM/D_SLAB passes. See the ceiling test for the measured per-slab-width
# comparison that justifies this pick.
_FWD_MMA_D_SLAB = 32

_FWD_MMA_TEMPLATE = """
    uint lane = thread_position_in_threadgroup.x;   // 0..31 == simd lane
    uint block = threadgroup_position_in_grid.x;    // query-block index within this dispatch
    uint bh = thread_position_in_grid.y;            // 0..(b*hq)-1

    uint r0 = qoffs[0];
    uint r1 = qoffs[1];
    uint rows_this = r1 - r0;

    uint hq = q_shape[1];
    uint n = q_shape[2];
    uint hkv = k_shape[1];
    uint group_size = hq / hkv;
    uint b = bh / hq;
    uint h = bh % hq;
    uint kvh = h / group_size;

    float scale = scale_in[0];
    size_t qh_base = (size_t)bh * n * HEAD_DIM;                 // bh == b*hq + h
    size_t kv_base = (size_t)(b * hkv + kvh) * n * HEAD_DIM;

    uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);
    uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);

    uint block_base = block * 32;                              // tile-local row of block start

    // Q row pointers for the 4 row-tiles (within-block rows fm, fm+8, fm+16, fm+24), clamped
    // so an over-hang row (dispatched past n) reads valid memory and is simply never stored.
    const device T* qh[4];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < 4; ++rt) {
        uint qrow = r0 + block_base + fm + 8 * rt;
        qh[rt] = q + qh_base + (size_t)metal::min(qrow, n - 1) * HEAD_DIM;
    }

    uint kv_limit = KV_LIMIT;

    // D-slab OUTER loop: the O accumulator C_o[4][D_SLAB/8] is held register-resident across
    // the KV loop for ONE slab of D columns at a time; QK^T + the online softmax are recomputed
    // per slab (re-reading K) -- the register-bounding structure (see the block comment for why
    // staging P cannot bound the persistent C_o accumulator instead).
    #pragma clang loop unroll(full)
    for (uint slab0 = 0; slab0 < HEAD_DIM; slab0 += D_SLAB) {
        float m[4];
        float l[4];
        metal::simdgroup_float8x8 C_o[4][D_SLAB_TILES];
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            m[rt] = -INFINITY;
            l[rt] = 0.0f;
            #pragma clang loop unroll(full)
            for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                C_o[rt][dt] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }

        for (uint kb0 = 0; kb0 < kv_limit; kb0 += 32) {
            uint kb1 = metal::min(kb0 + 32u, n);
            uint kl = kb1 - 1;                                    // clamp target for tail keys
            const device T* kp0[4];
            const device T* kp1[4];
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                kp0[ct] = k + kv_base + (size_t)metal::min(kb0 + 8 * ct + fn, kl) * HEAD_DIM;
                kp1[ct] = k + kv_base + (size_t)metal::min(kb0 + 8 * ct + fn + 1, kl) * HEAD_DIM;
            }
            // GEMM-A: S_block = Q_block @ K_block^T in fp32 simdgroup-matrix state (CE inner
            // GEMM). Head dims {64,96,128} are all %8==0, so the d-loop needs no guarded tail.
            metal::simdgroup_float8x8 C[4][4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    C[rt][ct] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (uint d0 = 0; d0 < HEAD_DIM; d0 += 8) {
                metal::simdgroup_float8x8 A[4];
                metal::simdgroup_float8x8 B[4];
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < 4; ++rt) {
                    A[rt].thread_elements()[0] = (float)qh[rt][d0 + fn];
                    A[rt].thread_elements()[1] = (float)qh[rt][d0 + fn + 1];
                }
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    B[ct].thread_elements()[0] = (float)kp0[ct][d0 + fm];
                    B[ct].thread_elements()[1] = (float)kp1[ct][d0 + fm];
                }
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < 4; ++rt) {
                    #pragma clang loop unroll(full)
                    for (uint ct = 0; ct < 4; ++ct) {
                        metal::simdgroup_multiply_accumulate(C[rt][ct], A[rt], B[ct], C[rt][ct]);
                    }
                }
            }
            // Scale the raw QK^T scores in place; the online softmax below reads them straight
            // out of the C tiles (no threadgroup S stage).
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    C[rt][ct].thread_elements()[0] *= scale;
                    C[rt][ct].thread_elements()[1] *= scale;
                }
            }
            // Per-row block max over the 32 keys, extracted from the C tiles via the lane's
            // owned elements + simd_shuffle_xor within the fm-row group (the CE forward's
            // epilogue idiom). Masked keys (tail past kb1 or causal-excluded) contribute -INF.
            float m_new[4];
            float alpha[4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                uint row = r0 + block_base + 8 * rt + fm;         // absolute query row (causal)
                float bm = -INFINITY;
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    uint kk = kb0 + 8 * ct + fn;
                    if ((kk < kb1) && (KEEP_CMP)) {
                        bm = metal::max(bm, C[rt][ct].thread_elements()[0]);
                    }
                    kk = kb0 + 8 * ct + fn + 1;
                    if ((kk < kb1) && (KEEP_CMP)) {
                        bm = metal::max(bm, C[rt][ct].thread_elements()[1]);
                    }
                }
                bm = metal::max(bm, metal::simd_shuffle_xor(bm, (ushort)8));
                bm = metal::max(bm, metal::simd_shuffle_xor(bm, (ushort)1));
                m_new[rt] = metal::max(m[rt], bm);
                alpha[rt] = metal::exp(m[rt] - m_new[rt]);
            }
            // Rescale the register-resident O tiles by this row's alpha (both thread elements of
            // C_o[rt][dt] live at matrix row 8*rt+fm, so both scale by alpha[rt] -- a per-thread-
            // element scale, NOT a simdgroup matmul).
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    C_o[rt][dt].thread_elements()[0] *= alpha[rt];
                    C_o[rt][dt].thread_elements()[1] *= alpha[rt];
                }
            }
            // P-formation IN PLACE (exp(scaled_score - m_new), masked to 0) + per-row rowsum.
            float bl[4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                uint row = r0 + block_base + 8 * rt + fm;
                float bs = 0.0f;
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    uint kk = kb0 + 8 * ct + fn;
                    float p0 = ((kk < kb1) && (KEEP_CMP))
                        ? metal::exp(C[rt][ct].thread_elements()[0] - m_new[rt]) : 0.0f;
                    kk = kb0 + 8 * ct + fn + 1;
                    float p1 = ((kk < kb1) && (KEEP_CMP))
                        ? metal::exp(C[rt][ct].thread_elements()[1] - m_new[rt]) : 0.0f;
                    C[rt][ct].thread_elements()[0] = p0;
                    C[rt][ct].thread_elements()[1] = p1;
                    bs += p0 + p1;
                }
                bs += metal::simd_shuffle_xor(bs, (ushort)8);
                bs += metal::simd_shuffle_xor(bs, (ushort)1);
                bl[rt] = bs;
            }
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                l[rt] = l[rt] * alpha[rt] + bl[rt];
                m[rt] = m_new[rt];
            }
            // GEMM-B: C_o += P @ V_slab (contraction over the 32 keys). C[rt][ct] holds P in
            // (query-row fm, key fn) layout -- fed DIRECTLY as the MMA left operand (the proven
            // backward GEMM-B structure). V_slab's fragment maps key fm -> V row, D-col fn; Vt
            // depends only on (ct, dt), so it is loaded once per (dt, ct) and reused across rt.
            #pragma clang loop unroll(full)
            for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_float8x8 Vt;
                    const device T* vr =
                        v + kv_base + (size_t)metal::min(kb0 + 8 * ct + fm, kl) * HEAD_DIM;
                    Vt.thread_elements()[0] = (float)vr[slab0 + 8 * dt + fn];
                    Vt.thread_elements()[1] = (float)vr[slab0 + 8 * dt + fn + 1];
                    #pragma clang loop unroll(full)
                    for (uint rt = 0; rt < 4; ++rt) {
                        metal::simdgroup_multiply_accumulate(
                            C_o[rt][dt], C[rt][ct], Vt, C_o[rt][dt]);
                    }
                }
            }
        }
        // Normalize this slab's D columns (1/l per row) and store; store L once (slab 0). The
        // guard skips over-hang rows; each lane owns disjoint (row, D-col) output elements.
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            uint local_row = block_base + 8 * rt + fm;           // dispatch-local output row
            if (local_row < rows_this) {
                float linv = 1.0f / l[rt];                       // causal: l >= 1 (attends itself)
                size_t o_base = ((size_t)bh * rows_this + local_row) * HEAD_DIM;
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    o_out[o_base + slab0 + 8 * dt + fn] =
                        (T)(C_o[rt][dt].thread_elements()[0] * linv);
                    o_out[o_base + slab0 + 8 * dt + fn + 1] =
                        (T)(C_o[rt][dt].thread_elements()[1] * linv);
                }
                if (slab0 == 0 && fn == 0) {
                    l_out[(size_t)bh * rows_this + local_row] = m[rt] + metal::log(l[rt]);
                }
            }
        }
    }
"""


def build_fwd_mma_source(
    head_dim: int, *, causal: bool = True, flip_causal: bool = False,
    d_slab: int | None = None,
) -> str:
    """MSL function body for the 4x4 simdgroup-matrix (MMA) flash-attention forward (O + L).

    Same correctness contract as `build_fwd_source` (the v0 scalar body) -- O matches the
    attention oracles, L is the fp32 row logsumexp -- with the O path restructured to the
    register-resident P@V MMA with D-slabbing (see the block comment above `_FWD_MMA_TEMPLATE`
    for the tile geometry, the P-formation / P@V idiom, and the D-slab register rationale).

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant. `causal=True` walks
    KV blocks only up to each query block's diagonal and masks the diagonal block per key with
    `kk <= row`; `causal=False` scans every KV block and keeps every key. `flip_causal` is
    TEST-ONLY -- it flips the causal predicate to the wrong triangle (`kk >= row`) so a parity
    run FAILS. `d_slab` (a multiple of 8 dividing `head_dim`) overrides the shipped
    `_FWD_MMA_D_SLAB` -- used by the regpressure probe to sweep candidate slab widths; the
    launcher always uses the default.
    """
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
    slab = _FWD_MMA_D_SLAB if d_slab is None else d_slab
    if slab <= 0 or slab % 8 != 0 or head_dim % slab != 0:
        raise ValueError(
            f"d_slab must be a positive multiple of 8 dividing head_dim={head_dim}, got {slab}"
        )
    if not causal:
        keep = "true"
        kv_limit = "n"
    elif flip_causal:
        keep = "kk >= row"
        kv_limit = "metal::min(n, r0 + block_base + 32u)"
    else:
        keep = "kk <= row"
        kv_limit = "metal::min(n, r0 + block_base + 32u)"
    return (
        _FWD_MMA_TEMPLATE.replace("HEAD_DIM", str(head_dim))
        .replace("D_SLAB_TILES", str(slab // 8))
        .replace("D_SLAB", str(slab))
        .replace("KV_LIMIT", kv_limit)
        .replace("KEEP_CMP", keep)
    )
