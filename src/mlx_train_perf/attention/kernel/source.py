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
