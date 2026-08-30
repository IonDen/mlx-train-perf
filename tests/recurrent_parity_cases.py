"""Shared parity cases for the chunked GatedDelta op.

Regimes group cases sharing a tolerance pin: upstream's own suite needed 1e-2
at the collinear+beta->1 corner vs 1e-3/1e-4 elsewhere — one pin for all
would be dishonest.
"""

import mlx.core as mx

SEED = 0
QWEN35_08B_GEOM = dict(Hk=16, Hv=16, Dk=128, Dv=128)  # real 0.8B GDN geometry


def build_case(*, T, chunk_size, B=1, Hk=2, Hv=4, Dk=32, Dv=32,
               repeat_keys=False, with_state=False, g_mode="random",
               beta_high=False, dtype=mx.float32, seed=SEED):
    mx.random.seed(seed)
    q = mx.random.normal(shape=(B, T, Hk, Dk)) * 0.5
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    if repeat_keys:
        k = mx.broadcast_to(k[:, :1], k.shape) * 1.0
    v = mx.random.normal(shape=(B, T, Hv, Dv)) * 0.5
    if g_mode == "zero":
        g = mx.full((B, T, Hv), 1e-30)
    elif g_mode == "one":
        g = mx.full((B, T, Hv), 1.0 - 1e-7)
    else:
        g = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    beta = mx.full((B, T, Hv), 0.999) if beta_high else \
        mx.sigmoid(mx.random.normal(shape=(B, T, Hv)) + 2.0)
    state = mx.random.normal(shape=(B, Hv, Dv, Dk)) * 0.3 if with_state else None
    q, k, v, g, beta = (t.astype(dtype) for t in (q, k, v, g, beta))
    out = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "state": state,
           "chunk_size": chunk_size}
    mx.eval(*(x for x in out.values() if isinstance(x, mx.array)))
    return out


CASES = [
    ("multi_chunk", "benign", dict(T=96, chunk_size=16)),
    ("single_partial_chunk", "benign", dict(T=30, chunk_size=64)),
    ("repeated_keys", "adversarial", dict(T=96, chunk_size=16, repeat_keys=True)),
    ("t_not_multiple", "benign", dict(T=70, chunk_size=16)),
    ("blocked_solve", "benign", dict(T=128, chunk_size=64)),
    ("blocked_repeated", "adversarial", dict(T=128, chunk_size=64, repeat_keys=True)),
    ("carried_state", "benign", dict(T=96, chunk_size=16, with_state=True)),
    ("g_zero", "benign", dict(T=96, chunk_size=16, g_mode="zero")),
    ("g_one", "benign", dict(T=96, chunk_size=16, g_mode="one")),
    ("collinear_beta_high", "blowup_corner",
     dict(T=128, chunk_size=64, repeat_keys=True, g_mode="one", beta_high=True)),
    ("rf1_real_ratio", "benign", dict(T=96, chunk_size=16, Hk=4, Hv=4)),
    ("gqa_rf2", "benign", dict(T=96, chunk_size=16, Hk=2, Hv=4)),
    ("bf16_inputs", "bf16", dict(T=96, chunk_size=16, dtype=mx.bfloat16)),
    ("bf16_g_one", "bf16", dict(T=96, chunk_size=16, g_mode="one", dtype=mx.bfloat16)),
    ("bf16_g_zero", "bf16", dict(T=96, chunk_size=16, g_mode="zero", dtype=mx.bfloat16)),
    ("representative_0p8b", "representative",
     dict(T=512, chunk_size=64, **QWEN35_08B_GEOM)),
    ("bf16_repr_T512", "bf16_representative",
     dict(T=512, chunk_size=64, dtype=mx.bfloat16, **QWEN35_08B_GEOM)),
    # Forward-only at T=2048 (2.08 GiB measured, no oracle backward tape).
    ("bf16_repr_T2048", "bf16_representative",
     dict(T=2048, chunk_size=64, dtype=mx.bfloat16, **QWEN35_08B_GEOM)),
]


def metrics(a: mx.array, b: mx.array) -> tuple[float, float]:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = a32 - b32
    return (float(mx.abs(diff).max()),
            float(mx.linalg.norm(diff.flatten())
                  / mx.maximum(mx.linalg.norm(b32.flatten()), 1e-30)))
