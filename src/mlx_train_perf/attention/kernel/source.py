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
# - D-SLABBING (a register-budget KNOB whose measured winner defied the spill arithmetic):
#   a full C_o[4][HEAD_DIM/8] is 64 tiles at d=128 == 128 fp32/lane — nominally deep in the
#   spill zone by the ~32-fp32/lane accumulator heuristic. THE MEASUREMENT SAYS OTHERWISE:
#   at saturation (N=8192 flagship, rung 2b, `_artifacts/attention_fwd_rungs/rung2b_dslab*
#   .json`) the FULL-D single-pass d_slab=128 is the fastest (1462.7 G MAC/s, +57% over
#   slab-32, +7.5% over slab-64) — fewer QK^T recompute passes beat the predicted spill
#   cost, and the compiled-ceiling probe was non-discriminating (flat 384 at every width).
#   The spill heuristic OVER-predicts for simdgroup accumulators at saturation
#   (user-metal-kernels workflow-and-gotchas — the rung-2b entry). So: the dispatch table
#   ships d_slab=128 for the MEASURED head_dim=128 saturation bucket, while `_FWD_MMA_D_SLAB
#   = 32` below stays only as the register-SAFE DEFAULT for the UNMEASURED head dims
#   (64/96) — it is NOT the optimum; sweep at saturation before tuning any new head dim.
#   Mechanics: the D-slab is the OUTER loop and the KV loop the INNER, with C_o holding
#   D_SLAB columns at a time; live simdgroup state during GEMM-B = 16 P tiles +
#   RT*(D_SLAB/8) C_o tiles.
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

# Default D-slab width — the register-SAFE fallback for UNMEASURED head dims (64/96), NOT
# the measured optimum. At the measured head_dim=128 saturation bucket the dispatch table
# overrides to d_slab=128 (single pass), which beat this default by 57% at rung 2b — the
# spill arithmetic that motivated 32 OVER-predicts simdgroup-accumulator cost at saturation
# (artifacts `_artifacts/attention_fwd_rungs/rung2b_dslab*.json`; user-metal-kernels entry).
# 32 divides all supported head dims (GCD) and is the conservative starting point for any
# new head dim's saturation sweep.
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


# ---------------------------------------------------------------------------------------
# Backward: D-preprocess kernel -- T7. `D_i = sum_d dO_i,d * O_i,d`, the flash-attention
# paper's row-correction term for `dS` (spec Section 4.2.2). Small and independently
# tested on purpose: a wrong D silently corrupts EVERY downstream gradient (dQ/dK/dV,
# T8/T9) while forward parity still passes -- see tests/test_attention_kernel_bwd.py's
# module docstring.
#
# ONE SIMDGROUP (32 lanes) PER (batch, q_head, row) TRIPLE -- the CE backward's proven
# `simd_sum` reduction idiom (`core/kernel/source.py::build_backward_dhidden_source`),
# specialized to a single elementwise-then-reduce pass instead of that kernel's
# per-vocab-column GEMM-dot: each lane strides over `HEAD_DIM` at stride 32, accumulating
# a partial `dO[i]*O[i]` sum in an fp32 register, then `simd_sum` collapses the 32 partials
# into the row's D value in one instruction, written by lane 0.
#
# NO query-range splitting, NO chained-launch accumulator (unlike the CE d_hidden kernel's
# vocab-tile chain, or the forward's qoffs-split): a row's D depends ONLY on its own
# (b, hq, row) triple -- fully disjoint output, one dispatch, no LaunchBudgetError guard
# needed (the whole flagship D is ~34 M MACs total; MEASURED 0.638 ms/dispatch at the
# flagship shape b=1/hq=32/n=8192/d=128 -- ~780x under the 0.5 s per-dispatch budget, and
# ~149x under it even at the project's slowest-ever measured rate class; T7 review probe).
#
# fp32 accumulation throughout regardless of the input template `T` (bf16 or fp32): both
# dO and O are upcast `(float)` before multiplying, exactly mirroring the pure-MLX
# reference's `.astype(mx.float32)` -- so bf16 rounding is common-mode to both sides, not
# doubled. D is written as a FIXED fp32 output, never cast down to `T` (matching the
# forward kernel's L convention: the residual that seeds the backward stays fp32 always).
#
# `drop_product` is TEST-ONLY (mirrors the forward kernel's `flip_causal`): it replaces
# the product's second factor with a constant `1.0f`, so the generated body computes
# rowsum(dO) instead of rowsum(dO*O) -- the deliberate wrong-value perturbation that
# proves the parity suite can detect a real D bug. Never used by production code.
_BWD_D_TEMPLATE = """
    uint lane = thread_position_in_threadgroup.x;   // 0..31 == simd lane
    uint row = thread_position_in_grid.y;            // 0..n-1 (absolute query row)
    uint bh = thread_position_in_grid.z;              // 0..(b*hq)-1
    uint n = d_o_shape[2];

    size_t base = ((size_t)bh * n + row) * HEAD_DIM;
    float part = 0.0f;
    for (uint i = lane; i < HEAD_DIM; i += 32) {
        part += (float)d_o[base + i] * PROD_FACTOR;
    }
    float d_val = metal::simd_sum(part);
    if (lane == 0) {
        d_out[(size_t)bh * n + row] = d_val;
    }
"""


def build_bwd_D_source(head_dim: int, *, drop_product: bool = False) -> str:  # noqa: N802 -- D is the paper's name
    """MSL function body for the D-preprocess backward kernel (`D = rowsum(dO * O)`).

    `head_dim` in {64, 96, 128} (the kernel's supported head dims) is baked in as a
    compile-time constant, same contract as the forward's `build_fwd_source`.
    `drop_product` is TEST-ONLY -- it replaces the elementwise product's second factor
    with `1.0f`, so a parity run against the correct rowsum FAILS (see the module-level
    block comment above)."""
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    prod_factor = "1.0f" if drop_product else "(float)o[base + i]"
    return (
        _BWD_D_TEMPLATE.replace("HEAD_DIM", str(head_dim)).replace(
            "PROD_FACTOR", prod_factor
        )
    )


# ---------------------------------------------------------------------------------------
# Backward: dQ kernel v1 (ONE OWNER PER QUERY ROW) -- T8, spec Section 4.2.3. One program
# owns dQ[i]; it loops the causally-allowed keys, recomputes S/P from Q, K and the SAVED L
# (never re-materializing the (N, N) probability matrix), and accumulates the query gradient
# in fp32 registers, writing dQ[i] exactly once. The math mirrors api.py's pure-MLX
# `_flash_attention_backward` dQ path EXACTLY, specialized to Bk=1 (per key):
#   s   = scale * (q_row . k_row)                 (recomputed QK^T, causal-masked)
#   p   = exp(s - L_row)                           (L_row is the forward's saved row logsumexp)
#   dp  = dO_row . v_row                           (the dP = dO @ V^T term, per key)
#   ds  = p * (dp - D_row)                         (D_row from T7's launch_bwd_D -- CONSUMED,
#                                                    never recomputed in-kernel)
#   dQ_row += scale * ds * k_row                   (accumulated over the causally-allowed keys)
#
# ONE-OWNER, NO ATOMICS: each (b, hq, row) triple's dQ is written by exactly one thread over
# disjoint output elements, so the result is bit-identical run to run (no accumulation races)
# and the query-range split stays bit-identical to a single dispatch -- a row's dQ depends
# ONLY on its own absolute position and the keys, never on its query block. This mirrors the
# v0 FORWARD scalar body's per-row structure (`_FWD_TEMPLATE`): register q/dO rows, fp32
# accumulator, full buffers + an in-kernel query-row offset (`qoffs`), tile-local dQ output.
#
# CAUSAL SKIP = the concrete per-key inequality `kk <= row` applied BEFORE a key contributes
# (masked keys have p=0 in the reference, so skipping them is exact) -- the named bug site of
# T8. `flip_causal` is TEST-ONLY: it flips the inequality to the WRONG triangle (`kk >= row`)
# so a parity run against the causal oracle FAILS -- the off-by-one perturbation that proves
# the parity grid can detect a causal-skip bug. Never used by production code.
#
# GQA `kv_head = q_head // group_size` in-kernel (K/V never expanded); fp32 accumulators
# throughout regardless of the input template `T` (bf16 or fp32); L and D are read as fixed
# fp32 device buffers (never templated -- the residuals that seed the backward stay fp32,
# matching the forward's L convention); dQ is cast to the input dtype `T` on the single store.
_BWD_DQ_TEMPLATE = """
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

    // Row-contiguous base offsets (ensure_row_contiguous=True). dO shares q's (B,Hq,N,D)
    // layout; K/V index by kv head; L and D are (B, Hq, N).
    size_t q_base = ((size_t)(b * hq + h) * n + row) * HEAD_DIM;
    size_t kv_base = (size_t)(b * hkv + kvh) * n * HEAD_DIM;   // + kk * HEAD_DIM per key
    size_t row_idx = (size_t)(b * hq + h) * n + row;

    float qreg[HEAD_DIM];
    float doreg[HEAD_DIM];
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        qreg[dd] = (float)q[q_base + dd];
        doreg[dd] = (float)d_o[q_base + dd];
    }

    float l_row = lse[row_idx];
    float d_row = d_arr[row_idx];

    float dq[HEAD_DIM];
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        dq[dd] = 0.0f;
    }

    for (uint kk = 0; kk < n; ++kk) {
        bool keep = (KEEP_CMP);
        if (!keep) { continue; }
        const device T* krow = k + kv_base + (size_t)kk * HEAD_DIM;
        const device T* vrow = v + kv_base + (size_t)kk * HEAD_DIM;
        float s = 0.0f;
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            s += qreg[dd] * (float)krow[dd];
        }
        float p = metal::exp(s * scale - l_row);
        float dp = 0.0f;
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            dp += doreg[dd] * (float)vrow[dd];
        }
        float sds = scale * p * (dp - d_row);
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            dq[dd] += sds * (float)krow[dd];
        }
    }

    size_t dq_base = ((size_t)bh * rows_this + local_row) * HEAD_DIM;
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        dq_out[dq_base + dd] = (T)dq[dd];
    }
"""


def build_bwd_dq_source(head_dim: int, *, causal: bool, flip_causal: bool = False) -> str:
    """MSL function body for the v1 one-owner-per-query-row dQ backward kernel.

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant (fixing the per-thread
    `qreg`/`doreg`/`dq` array sizes and the D-loop bounds). `causal=True` loops only keys
    `kk <= row` (the causal skip); `causal=False` loops every key. `flip_causal` is TEST-ONLY
    -- it flips the causal-skip inequality to the WRONG triangle (`kk >= row`) so a parity run
    against the causal oracle FAILS (the named-bug-site perturbation)."""
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
    return _BWD_DQ_TEMPLATE.replace("HEAD_DIM", str(head_dim)).replace("KEEP_CMP", keep)


# ---------------------------------------------------------------------------------------
# Backward: dQ kernel MMA rung B1 (T9b) -- the 4x4 simdgroup-matrix (register-resident,
# D-slabbed) THROUGHPUT restructure of the v1 scalar one-owner-per-query-row dQ body above.
# Same correctness contract as `build_bwd_dq_source` (dQ matches the api.py autodiff oracle;
# the scalar body is the correctness oracle) with the accumulation restructured to the proven
# FORWARD rung-2 MMA machinery (`_FWD_MMA_TEMPLATE`): ONE 32-lane simdgroup per 32-row query
# block, KV tiled in 32-key blocks, register-resident fp32 dQ accumulator D-slabbed exactly
# like the forward's O accumulator. The v1 scalar kernel measured 35.2 G MAC/s at the flagship
# canary (projected ~11.7s dQ pass vs the 2.0s per-eval budget -> production REFUSES at
# flagship); this is the dQ analogue of the 31.6 -> 1462.7 G forward restructure (T6 rungs
# 1-2b). The controller owns the saturation d_slab sweep + graduation (a later rung); THIS rung
# is CORRECTNESS + small-shape parity, and the mma variant is NOT wired into the API path.
#
# Tile geometry (inherited verbatim from `_FWD_MMA_TEMPLATE`'s QK^T half):
# - ONE THREADGROUP == ONE simdgroup (32 lanes) owns ONE query block of Bq=32 rows == 4
#   row-tiles of 8 (RT=4); the KV axis is tiled in Bk=32-key blocks == 4 col-tiles of 8. The
#   steel lane->(fm, fn) fragment mapping, the d-chunk QK^T GEMM, and the causal block-skip
#   (`kv_limit`) + in-tile predication (`KEEP_CMP`, the flip-test perturbation point) are the
#   forward's, unchanged. Head dims {64,96,128} are all %8==0, so the d-loops need no tail chunk.
#
# THE dQ-SPECIFIC MATH (mirrors the scalar `_BWD_DQ_TEMPLATE` + api.py's `_flash_attention_
# backward` dQ path EXACTLY, per KV block, per D-slab):
#   S   = Q_block @ K_block^T            (GEMM-A #1, unscaled dot in fp32 simdgroup state)
#   P   = exp(scale*S - L_row)           (SAVED L per row -- NO online-softmax rowmax, so P is
#                                         ELEMENTWISE-independent: no simd_shuffle reduction, no
#                                         alpha rescale, unlike the forward's online softmax)
#   dP  = dO_block @ V_block^T           (GEMM-A #2, the dP = dO@V^T term, a second QK^T-shaped MMA)
#   dS  = scale * P * (dP - D_row)       (D_row from T7's launch_bwd_D -- CONSUMED, not recomputed)
#   dQ_block += dS @ K_slab              (GEMM-B, the key contraction -- dS as the MMA left operand
#                                         exactly as forward P@V feeds P; K in place of V)
# P and dP are masked to 0 for tail (kk >= kb1) and causal-excluded (`KEEP_CMP`) keys BEFORE the
# GEMM-B, so a masked key's dS is exactly 0 and contributes nothing to dQ (the reference's p=0).
# dQ has NO softmax normalization (unlike the forward's O, which divides by the row's l) -- the
# accumulated dS@K IS the gradient. L and D are read as FIXED fp32 device buffers (never
# templated -- the residuals that seed the backward stay fp32, the forward's L convention).
#
# D-SLABBING (the register-resident-accumulator knob the controller sweeps): the dQ accumulator
# `C_dq[4][D_SLAB/8]` fp32 simdgroup tiles are held register-resident across the KV loop for ONE
# slab of D columns at a time, with the D-slab as the OUTER loop and the KV loop INNER -- so S,
# P, and dP are RECOMPUTED per slab (re-reading Q/K/V/dO), exactly as the forward recomputes its
# QK^T + softmax per O-slab. Bounding C_dq is what forces the recompute (the persistent
# accumulator is the register bottleneck; the transient S/dP tiles are freed each KV block). The
# default `_BWD_DQ_MMA_D_SLAB = 32` is the register-SAFE fallback (divides all supported head
# dims); the forward's rung-2b measurement showed the register-arithmetic spill heuristic
# OVER-predicts simdgroup-accumulator cost at saturation (full-D single-pass won there), so the
# dQ winner is a MEASURED sweep, not a predicted one -- never assume 32 is the optimum.
#
# The MMA reduction order (fp32 simdgroup reassociation + per-slab recompute) differs from the
# scalar body's sequential per-key `+=`, so the mma variant's parity worsts are measured and
# pinned SEPARATELY from the scalar pins -- never by widening a scalar pin. Per-row independence
# is preserved (a row's dQ depends only on its own absolute position and the keys), so the
# query-range split stays bit-identical to a single dispatch.
#
# `HEAD_DIM` bakes the head dim into the D-slab loop bound and the QK^T/dO@V^T d-loops; `D_SLAB`
# / `D_SLAB_TILES` bake the slab width + its col-tile count (D_SLAB/8) into the C_dq tile array;
# `KEEP_CMP` is the per-key causal keep predicate; `KV_LIMIT` is the KV-block loop bound. Sentinel
# `str.replace` (never an f-string -- the MSL body is full of C++ braces), `D_SLAB_TILES` before
# `D_SLAB` (longer token first, shared-prefix safe) -- the `build_fwd_mma_source` convention.

# Default D-slab width -- the register-SAFE fallback (32 divides all supported head dims
# 64/96/128), NOT a measured optimum; the controller sweeps {16,32,64,128} at saturation.
_BWD_DQ_MMA_D_SLAB = 32

_BWD_DQ_MMA_TEMPLATE = """
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
    size_t qh_base = (size_t)bh * n * HEAD_DIM;                 // bh == b*hq + h (q AND dO layout)
    size_t kv_base = (size_t)(b * hkv + kvh) * n * HEAD_DIM;
    size_t ld_base = (size_t)bh * n;                           // lse/d_arr are (B, Hq, N)

    uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);
    uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);

    uint block_base = block * 32;                              // tile-local row of block start

    // Q and dO row pointers for the 4 row-tiles (within-block rows fm, fm+8, fm+16, fm+24), plus
    // the per-row SAVED L and D -- all clamped to n-1 so an over-hang row (dispatched past n)
    // reads valid memory and is simply never stored. dO shares q's (B,Hq,N,D) layout (qh_base).
    const device T* qh[4];
    const device T* doh[4];
    float l_row[4];
    float d_row[4];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < 4; ++rt) {
        uint qrow = metal::min(r0 + block_base + fm + 8 * rt, n - 1);
        qh[rt] = q + qh_base + (size_t)qrow * HEAD_DIM;
        doh[rt] = d_o + qh_base + (size_t)qrow * HEAD_DIM;
        l_row[rt] = lse[ld_base + qrow];
        d_row[rt] = d_arr[ld_base + qrow];
    }

    uint kv_limit = KV_LIMIT;

    // D-slab OUTER loop: C_dq[4][D_SLAB/8] is held register-resident across the KV loop for ONE
    // slab of D columns at a time; S/P/dP are recomputed per slab (re-reading Q/K/V/dO) -- the
    // register-bounding structure mirroring the forward's O-path (see the block comment).
    #pragma clang loop unroll(full)
    for (uint slab0 = 0; slab0 < HEAD_DIM; slab0 += D_SLAB) {
        metal::simdgroup_float8x8 C_dq[4][D_SLAB_TILES];
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            #pragma clang loop unroll(full)
            for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                C_dq[rt][dt] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }

        for (uint kb0 = 0; kb0 < kv_limit; kb0 += 32) {
            uint kb1 = metal::min(kb0 + 32u, n);
            uint kl = kb1 - 1;                                    // clamp target for tail keys
            const device T* kp0[4];
            const device T* kp1[4];
            const device T* vp0[4];
            const device T* vp1[4];
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                uint c0 = metal::min(kb0 + 8 * ct + fn, kl);
                uint c1 = metal::min(kb0 + 8 * ct + fn + 1, kl);
                kp0[ct] = k + kv_base + (size_t)c0 * HEAD_DIM;
                kp1[ct] = k + kv_base + (size_t)c1 * HEAD_DIM;
                vp0[ct] = v + kv_base + (size_t)c0 * HEAD_DIM;
                vp1[ct] = v + kv_base + (size_t)c1 * HEAD_DIM;
            }

            // GEMM-A #1: S_block = Q_block @ K_block^T in fp32 simdgroup state (the CE inner GEMM,
            // the forward's QK^T verbatim -- unscaled dot; scale is folded into P below).
            metal::simdgroup_float8x8 C_s[4][4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    C_s[rt][ct] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
                        metal::simdgroup_multiply_accumulate(
                            C_s[rt][ct], A[rt], B[ct], C_s[rt][ct]);
                    }
                }
            }

            // P-formation IN PLACE (into C_s): p = exp(scale*s - L_row), masked to 0 for tail /
            // causal-excluded keys. dQ uses the SAVED L directly -- NO online-softmax rowmax, so
            // each element is independent (no simd_shuffle cross-key reduction).
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                uint row = r0 + block_base + 8 * rt + fm;         // absolute query row (causal)
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    uint kk = kb0 + 8 * ct + fn;
                    float p0 = ((kk < kb1) && (KEEP_CMP))
                        ? metal::exp(scale * C_s[rt][ct].thread_elements()[0] - l_row[rt]) : 0.0f;
                    kk = kb0 + 8 * ct + fn + 1;
                    float p1 = ((kk < kb1) && (KEEP_CMP))
                        ? metal::exp(scale * C_s[rt][ct].thread_elements()[1] - l_row[rt]) : 0.0f;
                    C_s[rt][ct].thread_elements()[0] = p0;
                    C_s[rt][ct].thread_elements()[1] = p1;
                }
            }

            // GEMM-A #2: dP_block = dO_block @ V_block^T (same QK^T-shaped MMA, dO for Q, V for K).
            metal::simdgroup_float8x8 C_dp[4][4];
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    C_dp[rt][ct] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (uint d0 = 0; d0 < HEAD_DIM; d0 += 8) {
                metal::simdgroup_float8x8 A[4];
                metal::simdgroup_float8x8 B[4];
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < 4; ++rt) {
                    A[rt].thread_elements()[0] = (float)doh[rt][d0 + fn];
                    A[rt].thread_elements()[1] = (float)doh[rt][d0 + fn + 1];
                }
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    B[ct].thread_elements()[0] = (float)vp0[ct][d0 + fm];
                    B[ct].thread_elements()[1] = (float)vp1[ct][d0 + fm];
                }
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < 4; ++rt) {
                    #pragma clang loop unroll(full)
                    for (uint ct = 0; ct < 4; ++ct) {
                        metal::simdgroup_multiply_accumulate(
                            C_dp[rt][ct], A[rt], B[ct], C_dp[rt][ct]);
                    }
                }
            }

            // dS-formation IN PLACE (into C_s): ds = scale * p * (dp - D_row). C_s holds P,
            // C_dp holds dP; after this C_s holds dS and C_dp is free. Masked keys already have
            // p == 0, so their dS is exactly 0 (they contribute nothing to the GEMM-B below).
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    C_s[rt][ct].thread_elements()[0] =
                        scale * C_s[rt][ct].thread_elements()[0]
                        * (C_dp[rt][ct].thread_elements()[0] - d_row[rt]);
                    C_s[rt][ct].thread_elements()[1] =
                        scale * C_s[rt][ct].thread_elements()[1]
                        * (C_dp[rt][ct].thread_elements()[1] - d_row[rt]);
                }
            }

            // GEMM-B: C_dq += dS @ K_slab (contraction over the 32 keys). C_s holds dS in
            // (query-row fm, key fn) layout -- fed DIRECTLY as the MMA left operand (the proven
            // forward P@V structure with K in place of V). K_slab's fragment maps key fm -> K row,
            // D-col fn; Kt depends only on (ct, dt), loaded once per (dt, ct) and reused across rt.
            #pragma clang loop unroll(full)
            for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_float8x8 Kt;
                    const device T* kr =
                        k + kv_base + (size_t)metal::min(kb0 + 8 * ct + fm, kl) * HEAD_DIM;
                    Kt.thread_elements()[0] = (float)kr[slab0 + 8 * dt + fn];
                    Kt.thread_elements()[1] = (float)kr[slab0 + 8 * dt + fn + 1];
                    #pragma clang loop unroll(full)
                    for (uint rt = 0; rt < 4; ++rt) {
                        metal::simdgroup_multiply_accumulate(
                            C_dq[rt][dt], C_s[rt][ct], Kt, C_dq[rt][dt]);
                    }
                }
            }
        }
        // Store this slab's D columns. The guard skips over-hang rows; each lane owns disjoint
        // (row, D-col) output elements. dQ has NO softmax normalization (unlike the forward's O).
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            uint local_row = block_base + 8 * rt + fm;           // dispatch-local output row
            if (local_row < rows_this) {
                size_t dq_base = ((size_t)bh * rows_this + local_row) * HEAD_DIM;
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    dq_out[dq_base + slab0 + 8 * dt + fn] =
                        (T)(C_dq[rt][dt].thread_elements()[0]);
                    dq_out[dq_base + slab0 + 8 * dt + fn + 1] =
                        (T)(C_dq[rt][dt].thread_elements()[1]);
                }
            }
        }
    }
"""


def build_bwd_dq_mma_source(
    head_dim: int, *, causal: bool, flip_causal: bool = False, d_slab: int | None = None,
) -> str:
    """MSL function body for the 4x4 simdgroup-matrix (MMA) dQ backward kernel -- the
    throughput restructure of the v1 scalar one-owner-per-row dQ body.

    Same correctness contract as `build_bwd_dq_source` (dQ matches the api.py autodiff oracle;
    the scalar body is the correctness oracle) with the accumulation restructured to the proven
    forward rung-2 MMA machinery (see the block comment above `_BWD_DQ_MMA_TEMPLATE`): one
    32-lane simdgroup per 32-row query block, KV tiled in 32-key blocks, S=Q@K^T + dP=dO@V^T
    MMAs, dS=scale*P*(dP-D) with P=exp(scale*S-L) from the saved L, and dQ+=dS@K_slab into a
    register-resident D-slabbed fp32 accumulator.

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant. `causal=True` walks KV
    blocks only up to each query block's diagonal and masks the diagonal block per key with
    `kk <= row`; `causal=False` scans every KV block and keeps every key. `flip_causal` is
    TEST-ONLY -- it flips the causal-skip predicate to the wrong triangle (`kk >= row`) so a
    parity run against the causal oracle FAILS (the named-bug-site perturbation). `d_slab` (a
    positive multiple of 8 dividing `head_dim`) overrides the register-safe `_BWD_DQ_MMA_D_SLAB`
    default (32) -- the controller sweeps {16,32,64,128} at saturation; this rung's launcher
    uses the default and the mma variant is not wired into the API path."""
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
    slab = _BWD_DQ_MMA_D_SLAB if d_slab is None else d_slab
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
        _BWD_DQ_MMA_TEMPLATE.replace("HEAD_DIM", str(head_dim))
        .replace("D_SLAB_TILES", str(slab // 8))
        .replace("D_SLAB", str(slab))
        .replace("KV_LIMIT", kv_limit)
        .replace("KEEP_CMP", keep)
    )


# ---------------------------------------------------------------------------------------
# Backward: dK/dV split-partials kernel v1 (ONE OWNER PER (batch, kv_head, key)) -- T9, spec
# Section 4.2.4. One program owns dK[j] and dV[j] for a single key j; each DISPATCH covers a
# bounded query-block range [q_lo, q_hi) and the owner accumulates that range's contribution
# into fp32 dK/dV registers seeded FROM the incoming chained partial (`dk_in`/`dv_in`). The
# math mirrors api.py's pure-MLX `_flash_attention_backward` dK/dV path EXACTLY, specialized to
# one owner key (the swapped-roles analogue of the dQ v1 body -- there the owner is a query row
# and the loop runs keys; here the owner is a key and the loop runs queries):
#   s   = scale * (q_i . k_j)                          (recomputed QK^T, causal-masked)
#   p   = exp(s - L_i)                                  (L_i is the forward's saved row logsumexp)
#   dp  = dO_i . v_j                                    (the dP = dO @ V^T term, per query)
#   ds  = p * (dp - D_i)                                (D_i from T7's launch_bwd_D -- CONSUMED)
#   dV_j += p * dO_i                                    (P^T @ dO, accumulated over the group)
#   dK_j += scale * ds * q_i                            (scale * dS^T @ q, grouped like dV)
#
# CHAINED == SINGLE, BIT-IDENTICALLY. Unlike the forward/dQ (whose outputs are DISJOINT across
# query blocks, so their launcher concatenates), dK/dV for key j receive a contribution from
# EVERY query row i >= j -- across all query blocks -- so the accumulator must be THREADED
# across chained dispatches like the CE forward chains lse/tgt (full buffers + in-kernel
# offsets, never a Python-side slice -- the CE kernel's 1.22 GB retained-copy lesson). The
# owner (1) seeds dk/dv FROM `dk_in`/`dv_in` FIRST, then (2) accumulates this dispatch's range
# with the query row as the OUTER loop (ascending) and the q-head as the INNER loop (ascending).
# A range split into ascending contiguous [q_lo, q_hi) dispatches, each seeded from the prior's
# fp32 output (fp32->fp32 store/reload is lossless), therefore reproduces the single [0, n)
# dispatch's exact sequential `+=` order -- bit-identical, no atomics, deterministic run to run.
#
# GQA `kv_head = q_head // group_size` (contiguous grouping, T2-pinned): the owner's kv head
# `kvh` owns the CONTIGUOUS q-head group [kvh*group_size, (kvh+1)*group_size), and the inner
# loop walks exactly that group -- K/V are never expanded. Every key is an owner in EVERY
# dispatch (grid.x == n): a key with no causally-allowed query in the range simply copies
# `dk_in`->`dk_out` unchanged, carrying the chained accumulator forward.
#
# CAUSAL SKIP = the concrete per-query inequality `i >= key` (query row at or below the
# diagonal) applied BEFORE a query contributes -- masked queries have p=0 in the reference, so
# skipping them is exact. `flip_causal` is TEST-ONLY: it flips the inequality to the WRONG
# triangle (`i <= key`) so a parity run against the causal oracle FAILS (the named-bug-site
# perturbation, the dK/dV analogue of the dQ kernel's `flip_causal`). Never used by production.
#
# fp32 accumulators throughout regardless of the input template `T` (bf16 or fp32); q/k/v/dO
# are read through `T` and upcast; L and D are fixed fp32 device buffers; dK/dV are written to
# FIXED fp32 output buffers (`dk_out`/`dv_out`) -- the chained partial stays fp32 across the
# whole launch, cast down to k/v dtype exactly once by the launcher after the last dispatch
# (matching the CE d_hidden chain's single final cast, and the forward's fp32-L convention).
_BWD_DKV_TEMPLATE = """
    uint key = thread_position_in_grid.x;         // 0..n-1 (absolute key position == owner)
    uint bkv = thread_position_in_grid.y;         // 0..(b*hkv)-1
    uint q_lo = qoffs[0];
    uint q_hi = qoffs[1];

    uint hq = q_shape[1];
    uint n = q_shape[2];
    uint hkv = k_shape[1];
    uint group_size = hq / hkv;
    if (key >= n) return;                         // defensive (dispatchThreads clamps x)

    uint b = bkv / hkv;
    uint kvh = bkv % hkv;
    uint h0 = kvh * group_size;                   // first q-head of this kv group (contiguous GQA)

    float scale = scale_in[0];

    // Owner's k/v row and its dK/dV accumulator slot -- all share the (B, Hkv, N, D) layout.
    size_t kv_row = ((size_t)(b * hkv + kvh) * n + key) * HEAD_DIM;

    float kreg[HEAD_DIM];
    float vreg[HEAD_DIM];
    float dk[HEAD_DIM];
    float dv[HEAD_DIM];
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        kreg[dd] = (float)k[kv_row + dd];
        vreg[dd] = (float)v[kv_row + dd];
        dk[dd] = dk_in[kv_row + dd];              // seed FROM the incoming chained partial FIRST
        dv[dd] = dv_in[kv_row + dd];
    }

    // Query-row OUTER (ascending), q-head INNER (ascending): a range split reproduces the
    // single-dispatch accumulation order exactly. Queries above the diagonal (i < key under
    // causal) have p=0 in the reference, so skipping them is exact.
    for (uint i = q_lo; i < q_hi; ++i) {
        bool keep = (KEEP_CMP);
        if (!keep) { continue; }
        for (uint h = h0; h < h0 + group_size; ++h) {
            size_t qh_row = ((size_t)(b * hq + h) * n + i) * HEAD_DIM;
            size_t ld_idx = (size_t)(b * hq + h) * n + i;
            float s = 0.0f;
            for (uint dd = 0; dd < HEAD_DIM; ++dd) {
                s += (float)q[qh_row + dd] * kreg[dd];
            }
            float p = metal::exp(s * scale - lse[ld_idx]);
            float dp = 0.0f;
            for (uint dd = 0; dd < HEAD_DIM; ++dd) {
                dp += (float)d_o[qh_row + dd] * vreg[dd];
            }
            float sds = scale * p * (dp - d_arr[ld_idx]);
            for (uint dd = 0; dd < HEAD_DIM; ++dd) {
                dv[dd] += p * (float)d_o[qh_row + dd];
                dk[dd] += sds * (float)q[qh_row + dd];
            }
        }
    }

    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        dk_out[kv_row + dd] = dk[dd];
        dv_out[kv_row + dd] = dv[dd];
    }
"""


def build_bwd_dkv_source(head_dim: int, *, causal: bool, flip_causal: bool = False) -> str:
    """MSL function body for the v1 one-owner-per-key chained dK/dV backward kernel.

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant (fixing the per-thread
    `kreg`/`vreg`/`dk`/`dv` array sizes and the D-loop bounds). `causal=True` keeps only queries
    `i >= key` (the causal skip); `causal=False` keeps every query. `flip_causal` is TEST-ONLY
    -- it flips the causal-keep inequality to the WRONG triangle (`i <= key`) so a parity run
    against the causal oracle FAILS (the named-bug-site perturbation, the dK/dV analogue of the
    dQ kernel's `flip_causal`)."""
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
    if not causal:
        keep = "true"
    elif flip_causal:
        keep = "i <= key"
    else:
        keep = "i >= key"
    return _BWD_DKV_TEMPLATE.replace("HEAD_DIM", str(head_dim)).replace("KEEP_CMP", keep)


# ---------------------------------------------------------------------------------------
# Backward: dK/dV MMA rung B2 (T9b) -- the 4x4 simdgroup-matrix (register-resident, D-slabbed,
# CHAINED) throughput restructure of the v1 scalar one-owner-per-key dK/dV body above. Same
# correctness AND chained-partials contract as `build_bwd_dkv_source` (dK/dV match the api.py
# autodiff oracle; a query-range split is bit-identical to a single dispatch) with the
# accumulation restructured to the proven forward/dQ MMA machinery. The v1 scalar kernel measured
# 82.9 G MAC/s at the flagship canary (projected ~6.6 s dK/dV pass -- now the ENTIRE backward
# pole after rung B1 took dQ to 2027.7 G / 0.20 s); this is the dK/dV analogue of that restructure.
# The controller owns the saturation d_slab sweep + graduation (a later rung); THIS rung is
# CORRECTNESS + small-shape parity, and the mma variant is NOT wired into the API path.
#
# OWNER + TILE GEOMETRY -- the MMA analogue of the scalar per-key owner, ROLES SWAPPED vs the dQ
# MMA (there the owner is a query block and the loop runs KV blocks; here the owner is a KEY block
# and the loop runs QUERY blocks, so dK/dV for a key receive a contribution from EVERY query block
# and the accumulator must be CHAINED across dispatches, unlike dQ's disjoint output):
# - ONE THREADGROUP == ONE simdgroup (32 lanes) owns ONE key block of Bk=32 keys == 4 key-tiles
#   of 8 (KT=4), per (batch, kv_head). The QUERY axis is tiled in Bq=32-query blocks == 4
#   query-tiles of 8 (QT=4), looped ASCENDING over this dispatch's range [q_lo, q_hi); inside each
#   query block the kv-head's contiguous q-head group is looped ASCENDING (EVERY member
#   contributes -- the scalar's whole-group accumulation). The steel lane->(fm, fn) mapping is the
#   forward's, unchanged. Head dims {64,96,128} are all %8==0, so the d-loops need no tail chunk.
#
# KEY-MAJOR ORIENTATION (the controller's recommendation -- chosen so NO fragment transposes are
# needed): build S^T = K@Q^T (keyxquery) fragments directly, so the C tiles' rows are KEYS (the
# owner, fixed) and columns are QUERIES. Then L and D are indexed by the QUERY (the fragment
# COLUMN) -- a lane holds two elements at columns fn and fn+1, so it reads TWO per-query L/D values
# per query-tile (lc0/lc1, dc0/dc1), unlike the dQ MMA where L/D were per-ROW (one value per tile).
# The dK/dV-specific math (mirrors the scalar `_BWD_DKV_TEMPLATE` and api.py's `_flash_attention_
# backward` dK/dV path EXACTLY, per query block, per D-slab, swapped-roles vs dQ):
#   S^T  = K_block @ Q_block^T           (GEMM #1, unscaled dot in fp32 simdgroup state; A=K rows
#                                        (key=fm), B=Q^T (transposed load, query=fn -> S^T column))
#   P^T  = exp(scale*S^T - L_col)         (SAVED L per query -- NO online-softmax rowmax, so P^T is
#                                        ELEMENTWISE-independent: no simd_shuffle, no alpha rescale)
#   dV_block += P^T @ dO_slab            (GEMM-B #1, contraction over the 32 queries -- done BEFORE
#                                        dS^T overwrites P^T; P^T as the MMA left operand, dO in
#                                        place of the forward's V)
#   dP^T = V_block @ dO_block^T          (GEMM #2, the dP=dO@V^T term, a second K@Q^T-shaped MMA)
#   dS^T = scale * P^T * (dP^T - D_col)   (IN PLACE into C_s, overwriting P^T; D_col from T7)
#   dK_block += dS^T @ Q_slab            (GEMM-B #2, the query contraction -- dS^T as the MMA left
#                                        operand, Q in place of the forward's V)
# P^T and dS^T are masked to 0 for tail (query >= q_hi) and causal-excluded (`KEEP_CMP`, per element
# by the query COLUMN) queries BEFORE the GEMM-Bs, so a masked query contributes exactly 0 (P=0 ->
# dS=0). dK/dV have NO softmax normalization (unlike the forward's O ÷ l) -- the accumulated
# GEMM-Bs ARE the gradients. L and D are read as FIXED fp32 device buffers (never templated).
#
# CHAINED-PARTIALS CONTRACT (the load-bearing part -- the scalar's bit-identity must survive):
# (a) SEED C_dv/C_dk (this slab's region) FROM dv_in/dk_in FIRST, before accumulating this
#     dispatch's query blocks -- never accumulate locally and add the seed at the end (that
#     reorders the fp32 addition and breaks chained==single bit-identity). The MMA `C = A*B + C`
#     adds each query block's product on top of the seed.
# (b) Query blocks accumulate in ASCENDING order (q_start..q_hi step 32), q-head inner ascending,
#     query-tile qt 0..3 -- the SAME op sequence whether the range arrived as one dispatch or
#     several. A key with no causally-allowed query in the range stores its seed unchanged (the
#     store epilogue ALWAYS writes the accumulator, guarded only against over-hang keys past n).
# (c) Range splits land on 32-row query-block boundaries (the launcher's `plan_dkv_dispatches`
#     block-alignment) -- a mid-block split would merge different partial products inside one
#     hardware MMA and break bit-identity; alignment restores the scalar order argument at block
#     granularity.
#
# CAUSAL in the key-major orientation: keep = query_i >= key_j. Query blocks entirely below the
# diagonal for this key block are skipped by starting the loop at `q_start` (= max(q_lo, key_base)
# for causal), and the diagonal block is predicated per (query, key) by `KEEP_CMP` (the flip-test
# perturbation point). NO tail-guards in the MMA hot loops; `#pragma clang loop unroll(full)` on
# the fixed-trip thread-local loops. `flip_causal` is TEST-ONLY -- it flips the keep to `i <= key`
# so a parity run against the causal oracle FAILS (the named-bug-site perturbation).
#
# D-SLABBING (the register-resident-accumulator knob the controller sweeps): C_dv[4][D_SLAB/8] AND
# C_dk[4][D_SLAB/8] fp32 simdgroup tiles are held register-resident across the query loop for ONE
# slab of D cols at a time, the D-slab as the OUTER loop -- so S^T/P^T/dP^T/dS^T are RECOMPUTED
# per slab (re-reading Q/K/V/dO), exactly as the forward recomputes its QK^T + softmax per O-slab.
# The default `_BWD_DKV_MMA_D_SLAB = 32` is the register-SAFE fallback (divides all supported head
# dims); the forward's rung-2b measurement showed the register-arithmetic spill heuristic
# OVER-predicts simdgroup-accumulator cost at saturation, so the dK/dV winner is a MEASURED sweep,
# not a predicted one. dK/dV carry TWO accumulators (vs dQ's one), so the per-slab register
# footprint is heavier than dQ's -- correctness is proven for all {16,32,64,128}; the controller
# picks the saturation winner.
#
# The MMA reduction order (fp32 simdgroup reassociation + per-slab recompute) differs from the
# scalar body's sequential per-query `+=`, so the mma variant's parity worsts are measured and
# pinned SEPARATELY from the scalar pins -- never by widening a scalar pin. Per-key independence is
# preserved (a key's dK/dV depend only on its own row and the chained partial), so the query-range
# split stays bit-identical to a single dispatch.
#
# `HEAD_DIM` bakes the head dim into the D-slab loop bound and the K@Q^T/V@dO^T d-loops; `D_SLAB` /
# `D_SLAB_TILES` bake the slab width + its col-tile count (D_SLAB/8) into the C_dv/C_dk tile arrays;
# `KEEP_CMP` is the per-query causal keep predicate; `Q_START` is the query-loop lower bound.
# Sentinel `str.replace` (never an f-string -- the MSL body is full of C++ braces), `D_SLAB_TILES`
# before `D_SLAB` (longer token first, shared-prefix safe) -- the `build_fwd_mma_source` convention.

# Default D-slab width -- the register-SAFE fallback (32 divides all supported head dims
# 64/96/128), NOT a measured optimum; the controller sweeps {16,32,64,128} at saturation.
_BWD_DKV_MMA_D_SLAB = 32

_BWD_DKV_MMA_TEMPLATE = """
    uint lane = thread_position_in_threadgroup.x;   // 0..31 == simd lane
    uint kblock = threadgroup_position_in_grid.x;   // key-block index within this dispatch's grid
    uint bkv = thread_position_in_grid.y;           // 0..(b*hkv)-1

    uint q_lo = qoffs[0];
    uint q_hi = qoffs[1];

    uint hq = q_shape[1];
    uint n = q_shape[2];
    uint hkv = k_shape[1];
    uint group_size = hq / hkv;
    uint b = bkv / hkv;
    uint kvh = bkv % hkv;
    uint h0 = kvh * group_size;                 // first q-head of this kv group (contiguous GQA)

    float scale = scale_in[0];
    size_t kv_base = (size_t)(b * hkv + kvh) * n * HEAD_DIM;   // owner K/V/dK/dV base (per kv head)

    uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);
    uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);

    uint key_base = kblock * 32;                     // owner key block's first key (absolute)

    // Owner K/V row pointers for the 4 key-tiles (within-block keys fm, fm+8, fm+16, fm+24),
    // clamped so an over-hang key (past n) reads valid memory and is simply never stored. K and V
    // are per-kv-head (independent of query block and q-head), so these are hoisted here.
    const device T* kh[4];
    const device T* vh[4];
    #pragma clang loop unroll(full)
    for (uint kt = 0; kt < 4; ++kt) {
        uint krow = metal::min(key_base + 8 * kt + fm, n - 1);
        kh[kt] = k + kv_base + (size_t)krow * HEAD_DIM;
        vh[kt] = v + kv_base + (size_t)krow * HEAD_DIM;
    }

    uint q_start = Q_START;

    // D-slab OUTER loop: C_dv/C_dk[4][D_SLAB/8] are held register-resident across the query loop
    // for ONE slab of D columns at a time, SEEDED FROM dv_in/dk_in FIRST (the chained partial),
    // then this dispatch's query blocks on top. S^T/P^T/dP^T/dS^T recomputed per slab
    // (re-reading K/V/Q/dO) -- the register-bounding structure mirroring the forward's O-path.
    #pragma clang loop unroll(full)
    for (uint slab0 = 0; slab0 < HEAD_DIM; slab0 += D_SLAB) {
        metal::simdgroup_float8x8 C_dv[4][D_SLAB_TILES];
        metal::simdgroup_float8x8 C_dk[4][D_SLAB_TILES];
        // Seed FROM the incoming chained partial (this slab's region) BEFORE accumulating -- never
        // accumulate locally and add the seed at the end (that reorders the fp32 addition and
        // breaks chained==single bit-identity). dk_in/dv_in are fixed fp32 device buffers.
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < 4; ++kt) {
            uint krow = metal::min(key_base + 8 * kt + fm, n - 1);
            size_t seed_base = kv_base + (size_t)krow * HEAD_DIM;
            #pragma clang loop unroll(full)
            for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                C_dv[kt][dt].thread_elements()[0] = dv_in[seed_base + slab0 + 8 * dt + fn];
                C_dv[kt][dt].thread_elements()[1] = dv_in[seed_base + slab0 + 8 * dt + fn + 1];
                C_dk[kt][dt].thread_elements()[0] = dk_in[seed_base + slab0 + 8 * dt + fn];
                C_dk[kt][dt].thread_elements()[1] = dk_in[seed_base + slab0 + 8 * dt + fn + 1];
            }
        }

        for (uint qb0 = q_start; qb0 < q_hi; qb0 += 32) {
            for (uint h = h0; h < h0 + group_size; ++h) {
                size_t qh_base = (size_t)(b * hq + h) * n * HEAD_DIM; // q AND dO layout, per q-head
                size_t ld_base = (size_t)(b * hq + h) * n;             // lse/d_arr, per q-head

                // Transposed query/dO pointers for GEMM #1/#2 (B operand: query row uses fn), and
                // the per-element L/D (indexed by the query = fragment column; a lane owns two
                // columns fn, fn+1 per query-tile). All clamped to n-1 so an over-hang / tail query
                // reads valid memory and is simply masked to 0.
                const device T* qpb0[4];
                const device T* qpb1[4];
                const device T* dopb0[4];
                const device T* dopb1[4];
                float lc0[4]; float lc1[4]; float dc0[4]; float dc1[4];
                #pragma clang loop unroll(full)
                for (uint qt = 0; qt < 4; ++qt) {
                    uint c0 = metal::min(qb0 + 8 * qt + fn, n - 1);
                    uint c1 = metal::min(qb0 + 8 * qt + fn + 1, n - 1);
                    qpb0[qt] = q + qh_base + (size_t)c0 * HEAD_DIM;
                    qpb1[qt] = q + qh_base + (size_t)c1 * HEAD_DIM;
                    dopb0[qt] = d_o + qh_base + (size_t)c0 * HEAD_DIM;
                    dopb1[qt] = d_o + qh_base + (size_t)c1 * HEAD_DIM;
                    lc0[qt] = lse[ld_base + c0];
                    lc1[qt] = lse[ld_base + c1];
                    dc0[qt] = d_arr[ld_base + c0];
                    dc1[qt] = d_arr[ld_base + c1];
                }

                // GEMM #1: S^T = K_block @ Q_block^T in fp32 simdgroup state (keyxquery). A = K
                // (key row=fm, d=fn); B = Q^T (transposed load: query row=fn -> the col of S^T,
                // d=fm). Unscaled dot; scale is folded into P^T below.
                metal::simdgroup_float8x8 C_s[4][4];
                #pragma clang loop unroll(full)
                for (uint kt = 0; kt < 4; ++kt) {
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        C_s[kt][qt] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                }
                for (uint d0 = 0; d0 < HEAD_DIM; d0 += 8) {
                    metal::simdgroup_float8x8 A[4];
                    metal::simdgroup_float8x8 B[4];
                    #pragma clang loop unroll(full)
                    for (uint kt = 0; kt < 4; ++kt) {
                        A[kt].thread_elements()[0] = (float)kh[kt][d0 + fn];
                        A[kt].thread_elements()[1] = (float)kh[kt][d0 + fn + 1];
                    }
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        B[qt].thread_elements()[0] = (float)qpb0[qt][d0 + fm];
                        B[qt].thread_elements()[1] = (float)qpb1[qt][d0 + fm];
                    }
                    #pragma clang loop unroll(full)
                    for (uint kt = 0; kt < 4; ++kt) {
                        #pragma clang loop unroll(full)
                        for (uint qt = 0; qt < 4; ++qt) {
                            metal::simdgroup_multiply_accumulate(
                                C_s[kt][qt], A[kt], B[qt], C_s[kt][qt]);
                        }
                    }
                }

                // P^T-formation IN PLACE: p = exp(scale*S^T - L_col), masked to 0 for tail
                // (query >= q_hi) and causal-excluded (KEEP_CMP) queries. L indexed by the QUERY
                // (fragment column); each of a lane's two elements has its own query.
                #pragma clang loop unroll(full)
                for (uint kt = 0; kt < 4; ++kt) {
                    uint key = key_base + 8 * kt + fm;    // owner key (causal compares vs query)
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        uint i = qb0 + 8 * qt + fn;
                        float p0 = ((i < q_hi) && (KEEP_CMP))
                            ? metal::exp(scale * C_s[kt][qt].thread_elements()[0] - lc0[qt]) : 0.0f;
                        i = qb0 + 8 * qt + fn + 1;
                        float p1 = ((i < q_hi) && (KEEP_CMP))
                            ? metal::exp(scale * C_s[kt][qt].thread_elements()[1] - lc1[qt]) : 0.0f;
                        C_s[kt][qt].thread_elements()[0] = p0;
                        C_s[kt][qt].thread_elements()[1] = p1;
                    }
                }

                // dV GEMM-B: C_dv += P^T @ dO_slab (contract over 32 queries). C_s holds P^T in
                // (key, query) layout -- fed DIRECTLY as the MMA left operand; dO's fragment maps
                // query fm -> dO row, D-col fn. Before dS^T overwrites P^T. Masked queries have
                // P^T == 0, so they contribute nothing regardless of the clamped dO pointer read.
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        metal::simdgroup_float8x8 dOt;
                        const device T* dor =
                            d_o + qh_base + (size_t)metal::min(qb0 + 8 * qt + fm, n - 1) * HEAD_DIM;
                        dOt.thread_elements()[0] = (float)dor[slab0 + 8 * dt + fn];
                        dOt.thread_elements()[1] = (float)dor[slab0 + 8 * dt + fn + 1];
                        #pragma clang loop unroll(full)
                        for (uint kt = 0; kt < 4; ++kt) {
                            metal::simdgroup_multiply_accumulate(
                                C_dv[kt][dt], C_s[kt][qt], dOt, C_dv[kt][dt]);
                        }
                    }
                }

                // GEMM #2: dP^T = V_block @ dO_block^T (keyxquery, same shape as GEMM #1 with V for
                // K, dO for Q).
                metal::simdgroup_float8x8 C_dp[4][4];
                #pragma clang loop unroll(full)
                for (uint kt = 0; kt < 4; ++kt) {
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        C_dp[kt][qt] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                }
                for (uint d0 = 0; d0 < HEAD_DIM; d0 += 8) {
                    metal::simdgroup_float8x8 A[4];
                    metal::simdgroup_float8x8 B[4];
                    #pragma clang loop unroll(full)
                    for (uint kt = 0; kt < 4; ++kt) {
                        A[kt].thread_elements()[0] = (float)vh[kt][d0 + fn];
                        A[kt].thread_elements()[1] = (float)vh[kt][d0 + fn + 1];
                    }
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        B[qt].thread_elements()[0] = (float)dopb0[qt][d0 + fm];
                        B[qt].thread_elements()[1] = (float)dopb1[qt][d0 + fm];
                    }
                    #pragma clang loop unroll(full)
                    for (uint kt = 0; kt < 4; ++kt) {
                        #pragma clang loop unroll(full)
                        for (uint qt = 0; qt < 4; ++qt) {
                            metal::simdgroup_multiply_accumulate(
                                C_dp[kt][qt], A[kt], B[qt], C_dp[kt][qt]);
                        }
                    }
                }

                // dS^T-formation IN PLACE (into C_s): ds = scale * p * (dp - D_col). C_s holds P^T,
                // C_dp holds dP^T; after, C_s holds dS^T, C_dp free. Masked queries already
                // have p == 0, so their dS is exactly 0 (they contribute nothing to dK below).
                #pragma clang loop unroll(full)
                for (uint kt = 0; kt < 4; ++kt) {
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        C_s[kt][qt].thread_elements()[0] =
                            scale * C_s[kt][qt].thread_elements()[0]
                            * (C_dp[kt][qt].thread_elements()[0] - dc0[qt]);
                        C_s[kt][qt].thread_elements()[1] =
                            scale * C_s[kt][qt].thread_elements()[1]
                            * (C_dp[kt][qt].thread_elements()[1] - dc1[qt]);
                    }
                }

                // dK GEMM-B: C_dk += dS^T @ Q_slab (contract over 32 queries). C_s holds dS^T
                // in (key, query) layout; Q's fragment maps query fm -> Q row, D-col fn.
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    #pragma clang loop unroll(full)
                    for (uint qt = 0; qt < 4; ++qt) {
                        metal::simdgroup_float8x8 Qt;
                        const device T* qr =
                            q + qh_base + (size_t)metal::min(qb0 + 8 * qt + fm, n - 1) * HEAD_DIM;
                        Qt.thread_elements()[0] = (float)qr[slab0 + 8 * dt + fn];
                        Qt.thread_elements()[1] = (float)qr[slab0 + 8 * dt + fn + 1];
                        #pragma clang loop unroll(full)
                        for (uint kt = 0; kt < 4; ++kt) {
                            metal::simdgroup_multiply_accumulate(
                                C_dk[kt][dt], C_s[kt][qt], Qt, C_dk[kt][dt]);
                        }
                    }
                }
            }
        }

        // Store this slab's D columns. ALWAYS write the accumulator (a key with no causally-allowed
        // query in the range stores its seed = dk_in/dv_in unchanged, carrying the chained partial
        // forward); the guard only skips over-hang keys (past n). dk_out/dv_out stay fp32 -- the
        // launcher casts to k/v dtype once after the last dispatch. Each lane owns disjoint
        // (key, D-col) output elements.
        #pragma clang loop unroll(full)
        for (uint kt = 0; kt < 4; ++kt) {
            uint key = key_base + 8 * kt + fm;
            if (key < n) {
                size_t out_base = kv_base + (size_t)key * HEAD_DIM;
                #pragma clang loop unroll(full)
                for (uint dt = 0; dt < D_SLAB_TILES; ++dt) {
                    dv_out[out_base + slab0 + 8 * dt + fn] = C_dv[kt][dt].thread_elements()[0];
                    dv_out[out_base + slab0 + 8 * dt + fn + 1] = C_dv[kt][dt].thread_elements()[1];
                    dk_out[out_base + slab0 + 8 * dt + fn] = C_dk[kt][dt].thread_elements()[0];
                    dk_out[out_base + slab0 + 8 * dt + fn + 1] = C_dk[kt][dt].thread_elements()[1];
                }
            }
        }
    }
"""


def build_bwd_dkv_mma_source(
    head_dim: int, *, causal: bool, flip_causal: bool = False, d_slab: int | None = None,
) -> str:
    """MSL function body for the 4x4 simdgroup-matrix (MMA) dK/dV backward kernel -- the key-major,
    register-resident D-slabbed, CHAINED throughput restructure of the v1 scalar one-owner-per-key
    dK/dV body.

    Same correctness AND chained-partials contract as `build_bwd_dkv_source` (dK/dV match the
    api.py autodiff oracle; the scalar body is the correctness oracle; a query-range split seeded
    from the prior dispatch's fp32 output is bit-identical to a single dispatch) with the
    accumulation restructured to the proven forward/dQ rung-2 MMA machinery (see the block comment
    above `_BWD_DKV_MMA_TEMPLATE`): one 32-lane simdgroup per 32-KEY block per (batch, kv_head),
    the QUERY axis tiled in 32-query blocks (ascending) x the kv-head's q-head group (ascending),
    S^T=K@Q^T + dP^T=V@dO^T MMAs, P^T=exp(scale*S^T-L_col), dS^T=scale*P^T*(dP^T-D_col),
    and dV+=P^T@dO_slab / dK+=dS^T@Q_slab into register-resident D-slabbed fp32 accumulators SEEDED
    from dv_in/dk_in.

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant. `causal=True` walks query
    blocks only from each key block's diagonal upward (`q_start = max(q_lo, key_base)`) and masks
    per (query, key) with `i >= key`; `causal=False` scans every query block from q_lo and keeps
    every query. `flip_causal` is TEST-ONLY -- it flips the causal-keep predicate to the wrong
    triangle (`i <= key`) so a parity run against the causal oracle FAILS (the named-bug-site
    perturbation). `d_slab` (a positive multiple of 8 dividing `head_dim`) overrides the
    register-safe `_BWD_DKV_MMA_D_SLAB` default (32) -- the controller sweeps {16,32,64,128} at
    saturation; this rung's launcher uses the default and the mma variant is not wired into the API
    path."""
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
    slab = _BWD_DKV_MMA_D_SLAB if d_slab is None else d_slab
    if slab <= 0 or slab % 8 != 0 or head_dim % slab != 0:
        raise ValueError(
            f"d_slab must be a positive multiple of 8 dividing head_dim={head_dim}, got {slab}"
        )
    if not causal:
        keep = "true"
        q_start = "q_lo"
    elif flip_causal:
        keep = "i <= key"
        q_start = "metal::max(q_lo, key_base)"
    else:
        keep = "i >= key"
        q_start = "metal::max(q_lo, key_base)"
    return (
        _BWD_DKV_MMA_TEMPLATE.replace("HEAD_DIM", str(head_dim))
        .replace("D_SLAB_TILES", str(slab // 8))
        .replace("D_SLAB", str(slab))
        .replace("Q_START", q_start)
        .replace("KEEP_CMP", keep)
    )
