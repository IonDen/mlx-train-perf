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
# Rung 1: 4x4 simdgroup-matrix (MMA) forward -- the throughput restructure of the v0 scalar
# body above. Correctness contract is IDENTICAL (O + L, same online-softmax math, same GQA,
# same causal mask, same qoffs/full-buffer offsets); only the score computation moves from a
# scalar per-thread dot to the CE forward's PROVEN 4x4 `simdgroup_float8x8` tiling
# (`core/kernel/source.py::_SOURCE_V2E` -- verified 2423.7 G MAC/s at production shape).
#
# Tile geometry (documented per the rung contract):
# - ONE THREADGROUP == ONE simdgroup (32 lanes) owns ONE query block of Bq=32 rows == 4
#   row-tiles of 8, matching the CE forward's 32-row block (RT=4). 32 is the smallest block
#   the 8x8 fragment divides cleanly that also fills a simdgroup's 32 lanes.
# - The KV axis is tiled in blocks of Bk=32 keys == 4 col-tiles of 8 -- exactly ONE CE
#   `cc`-iteration per KV block. S_block(32x32) = scale * Q_block @ K_block^T is the CE inner
#   GEMM verbatim (hidden->Q rows, w->K keys, contraction over the head dim), so the lane
#   mapping (`fm`/`fn`), the d-chunk loop + guarded d-tail, and the fragment reads are reused
#   unchanged and inherit the CE kernel's parity proof for the QK^T half.
# - CAUSAL BLOCK SKIPPING: the KV loop stops at `kv_limit` = min(n, block_end + 1) (the
#   block's LAST row's diagonal, exclusive) -- KV blocks fully above the diagonal are never
#   entered. The diagonal block is masked IN-TILE per (row, key) before the row max
#   (`KEEP_CMP`, the flip-test perturbation point), so per-row diagonals inside a block are
#   handled while whole above-diagonal blocks are cheaply skipped.
#
# Softmax + O structure (the deliberately-simple, register-safe choice for this rung):
# - S is staged to THREADGROUP memory (`s_tile`, 32x32 fp32 == 4 KB) from the C fragments,
#   then the online-softmax update runs SCALAR-PER-LANE: lane L owns within-block row L,
#   reads its full staged score row, applies the causal keep, updates m/l, and forms P.
# - The O accumulator is a THREADGROUP tile (`o_acc`, 32 x head_dim fp32 == 16 KB at d=128;
#   4+16 KB + m/l fit inside the 32 KB threadgroup limit). O is accumulated LANE-OWNED-ROWS
#   (a scalar P@V per row) rather than as a second MMA -- the rung contract's other blessed
#   option. Crucially this AVOIDS a per-lane `float acc[head_dim]` register array, which at
#   head_dim 128 SPILLS (user-metal-kernels spill-inversion entry: the v0 scalar body's
#   acc[128] spilled and inverted the ceiling signal to 1024). With S/O in threadgroup
#   memory the only large per-lane state is the CE C-tile set (16 fragments == 32 fp32/lane,
#   the family-independent optimum), so the compiled ceiling stays in the CE forward's
#   ~448 class at EVERY head_dim (measured in tests/test_devtools.py).
# - fp32 accumulators throughout (C tiles fp32, m/l/o_acc fp32); L = m + log(l) fp32; O cast
#   to the input dtype `T` on the final store -- identical to the v0 body's contract.
#
# The reduction ORDER differs from v0 (block softmax + MMA fp32 reassociation vs v0's
# per-key scalar recurrence), so the MMA variant's parity worsts are measured and pinned
# SEPARATELY from the scalar pins (tests/test_attention_kernel_fwd.py) -- never by widening a
# scalar pin. Per-row independence is preserved (a row's O/L depend only on its own absolute
# position and keys, never on its block neighbours), so the query-range split stays
# bit-identical to a single dispatch, exactly like v0.
#
# `HEAD_DIM` bakes the head dim into the threadgroup array sizes and the D-wide loop bounds;
# `KEEP_CMP` is the per-key causal keep predicate (same strings as the v0 builder); `KV_LIMIT`
# is the KV-block loop bound expression (causal min-bound vs the full `n`). Sentinel
# `str.replace`, never an f-string (the MSL body is full of C++ braces) -- the 0.1.0
# convention.
_FWD_MMA_TEMPLATE = """
    uint lane = thread_position_in_threadgroup.x;   // 0..31 == the within-block row it owns
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
    uint wbrow = lane;                                        // within-block row this lane owns
    uint local_row = block_base + wbrow;                      // dispatch-local output row
    uint row = r0 + local_row;                                // absolute query row (causal)

    threadgroup float s_tile[32 * 32];
    threadgroup float o_acc[32 * HEAD_DIM];
    threadgroup float m_tg[32];
    threadgroup float l_tg[32];

    // Q row pointers for the 4 row-tiles (within-block rows fm, fm+8, fm+16, fm+24), clamped
    // so an over-hang row (dispatched past n) reads valid memory and is simply never stored.
    const device T* qh[4];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < 4; ++rt) {
        uint qrow = r0 + block_base + fm + 8 * rt;
        qh[rt] = q + qh_base + (size_t)metal::min(qrow, n - 1) * HEAD_DIM;
    }

    // Each lane initializes the online-softmax state of the within-block row it owns.
    m_tg[wbrow] = -INFINITY;
    l_tg[wbrow] = 0.0f;
    #pragma clang loop unroll(full)
    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
        o_acc[wbrow * HEAD_DIM + dd] = 0.0f;
    }

    uint kv_limit = KV_LIMIT;
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
        // S_block = Q_block @ K_block^T in fp32 simdgroup-matrix state (CE inner GEMM).
        metal::simdgroup_float8x8 C[4][4];
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                C[rt][ct] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        uint dfull = HEAD_DIM & ~7u;
        for (uint d0 = 0; d0 < dfull; d0 += 8) {
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
        if (dfull < HEAD_DIM) {                    // dead for head_dim in {64,96,128} (all %8==0);
            metal::simdgroup_float8x8 A[4];        // kept for parity with the CE d-tail idiom
            metal::simdgroup_float8x8 B[4];
            uint da = dfull + fn;
            uint db = dfull + fm;
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                A[rt].thread_elements()[0] = (da < HEAD_DIM) ? (float)qh[rt][da] : 0.0f;
                A[rt].thread_elements()[1] = (da + 1 < HEAD_DIM) ? (float)qh[rt][da + 1] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                B[ct].thread_elements()[0] = (db < HEAD_DIM) ? (float)kp0[ct][db] : 0.0f;
                B[ct].thread_elements()[1] = (db < HEAD_DIM) ? (float)kp1[ct][db] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint rt = 0; rt < 4; ++rt) {
                #pragma clang loop unroll(full)
                for (uint ct = 0; ct < 4; ++ct) {
                    metal::simdgroup_multiply_accumulate(C[rt][ct], A[rt], B[ct], C[rt][ct]);
                }
            }
        }
        // Stage the scaled score block to threadgroup memory (fragment -> [row][col]).
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < 4; ++rt) {
            #pragma clang loop unroll(full)
            for (uint ct = 0; ct < 4; ++ct) {
                uint srow = 8 * rt + fm;
                uint scol = 8 * ct + fn;
                s_tile[srow * 32 + scol] = scale * C[rt][ct].thread_elements()[0];
                s_tile[srow * 32 + scol + 1] = scale * C[rt][ct].thread_elements()[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Scalar online-softmax epilogue: lane `wbrow` owns within-block row `wbrow`. The
        // guard skips over-hang rows WITHOUT returning, so every lane still reaches the
        // barriers uniformly (a divergent barrier would deadlock the threadgroup).
        if (local_row < rows_this) {
            float bm = -INFINITY;
            #pragma clang loop unroll(full)
            for (uint jj = 0; jj < 32; ++jj) {
                uint kk = kb0 + jj;                            // absolute key
                bool keep = (kk < kb1) && (KEEP_CMP);
                if (keep) { bm = metal::max(bm, s_tile[wbrow * 32 + jj]); }
            }
            float m_old = m_tg[wbrow];
            float m_new = metal::max(m_old, bm);
            float alpha = metal::exp(m_old - m_new);           // rescale factor for old m/l/O
            #pragma clang loop unroll(full)
            for (uint dd = 0; dd < HEAD_DIM; ++dd) {
                o_acc[wbrow * HEAD_DIM + dd] *= alpha;
            }
            float block_l = 0.0f;
            for (uint jj = 0; jj < 32; ++jj) {
                uint kk = kb0 + jj;
                bool keep = (kk < kb1) && (KEEP_CMP);
                if (keep) {
                    float p = metal::exp(s_tile[wbrow * 32 + jj] - m_new);
                    block_l += p;
                    const device T* vrow = v + kv_base + (size_t)kk * HEAD_DIM;
                    #pragma clang loop unroll(full)
                    for (uint dd = 0; dd < HEAD_DIM; ++dd) {
                        o_acc[wbrow * HEAD_DIM + dd] += p * (float)vrow[dd];
                    }
                }
            }
            l_tg[wbrow] = l_tg[wbrow] * alpha + block_l;
            m_tg[wbrow] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);       // before s_tile is reused
    }
    // Final normalization + store (tile-local output row, matching the v0 launcher layout).
    if (local_row < rows_this) {
        float linv = 1.0f / l_tg[wbrow];                       // causal: l >= 1 (attends itself)
        size_t o_base = ((size_t)bh * rows_this + local_row) * HEAD_DIM;
        #pragma clang loop unroll(full)
        for (uint dd = 0; dd < HEAD_DIM; ++dd) {
            o_out[o_base + dd] = (T)(o_acc[wbrow * HEAD_DIM + dd] * linv);
        }
        l_out[(size_t)bh * rows_this + local_row] = m_tg[wbrow] + metal::log(l_tg[wbrow]);
    }
"""


def build_fwd_mma_source(
    head_dim: int, *, causal: bool = True, flip_causal: bool = False
) -> str:
    """MSL function body for the 4x4 simdgroup-matrix (MMA) flash-attention forward (O + L).

    Same correctness contract as `build_fwd_source` (the v0 scalar body) -- O matches the
    attention oracles, L is the fp32 row logsumexp -- but the score block is computed with
    the CE forward's proven 4x4 `simdgroup_float8x8` tiling and the KV axis is walked in
    causal-bounded blocks (see the block comment above `_FWD_MMA_TEMPLATE` for the tile
    geometry, the softmax/O structure, and the register-safety rationale).

    `head_dim` in {64, 96, 128} is baked in as a compile-time constant. `causal=True` walks
    KV blocks only up to each query block's diagonal and masks the diagonal block per key
    with `kk <= row`; `causal=False` scans every KV block and keeps every key. `flip_causal`
    is TEST-ONLY -- it flips the causal predicate to the wrong triangle (`kk >= row`) so a
    parity run FAILS (the KV-block loop bound stays the causal one, which at the tiny flip
    shape already covers every key).
    """
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {_KERNEL_HEAD_DIMS}, got {head_dim}"
        )
    if flip_causal and not causal:
        raise ValueError("flip_causal is only meaningful with causal=True")
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
        .replace("KV_LIMIT", kv_limit)
        .replace("KEEP_CMP", keep)
    )
