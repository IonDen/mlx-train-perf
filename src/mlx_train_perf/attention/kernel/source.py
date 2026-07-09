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
