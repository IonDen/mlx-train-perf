"""T9b end-to-end flagship check — the REAL production path after graduation.

Runs the public `flash_attention` (impl="kernel") at the flagship shape under
`mx.value_and_grad` — table-selected forward AND backward variants, per-kernel
calibrated rates captured at construction, the exact path T13's campaign will train
through. Records fwd-only wall, fwd+bwd wall, and the marginal peak memory of the
gradient pass. Runs AFTER the graduation rung's review (controller only).

Run:  uv run python _artifacts/attention_bwd_rungs/t9b_e2e_flagship.py   (~1 min)
"""

import json
import math
import pathlib
import statistics
import subprocess
import time

import mlx.core as mx

from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.core.guards import install_guardrails

ART = pathlib.Path(__file__).resolve().parent / "t9b_e2e_flagship.json"
B, HQ, HKV, N, D = 1, 32, 8, 8192, 128
SCALE = 1.0 / math.sqrt(D)
REPS = 5

install_guardrails()
kq, kk, kv, kw = mx.random.split(mx.random.key(0), 4)
q = mx.random.normal((B, HQ, N, D), dtype=mx.bfloat16, key=kq)
k = mx.random.normal((B, HKV, N, D), dtype=mx.bfloat16, key=kk)
v = mx.random.normal((B, HKV, N, D), dtype=mx.bfloat16, key=kv)
w = mx.random.normal((B, HQ, N, D), dtype=mx.bfloat16, key=kw)  # fixed cotangent
mx.eval(q, k, v, w)


def loss(q_, k_, v_) -> mx.array:
    return (flash_attention(q_, k_, v_, scale=SCALE, causal=True, impl="kernel")
            .astype(mx.float32) * w.astype(mx.float32)).sum()


grad_fn = mx.value_and_grad(loss, argnums=(0, 1, 2))

# Warmup: calibration ramps (per-kernel, construction-time) + JIT, all outside timing.
mx.eval(flash_attention(q, k, v, scale=SCALE, causal=True, impl="kernel"))
lv, gs = grad_fn(q, k, v)
mx.eval(lv, *gs)

fwd_walls = []
for _ in range(REPS):
    t0 = time.perf_counter()
    mx.eval(flash_attention(q, k, v, scale=SCALE, causal=True, impl="kernel"))
    fwd_walls.append(round(time.perf_counter() - t0, 4))

mx.reset_peak_memory()  # note: includes live buffers (gotcha 4) — marginal read below
base_peak = mx.get_peak_memory()
fb_walls = []
for _ in range(REPS):
    t0 = time.perf_counter()
    lv, gs = grad_fn(q, k, v)
    mx.eval(lv, *gs)
    fb_walls.append(round(time.perf_counter() - t0, 4))
grad_peak = mx.get_peak_memory()

result = {
    "rung": "t9b_e2e_flagship",
    "shape": {"b": B, "hq": HQ, "hkv": HKV, "n": N, "d": D, "dtype": "bfloat16",
              "causal": True, "impl": "kernel"},
    "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    "mlx": mx.__version__,
    "fwd_median_s": statistics.median(fwd_walls), "fwd_walls_s": fwd_walls,
    "fwd_bwd_median_s": statistics.median(fb_walls), "fwd_bwd_walls_s": fb_walls,
    "bwd_derived_s": round(statistics.median(fb_walls)
                           - statistics.median(fwd_walls), 4),
    "grad_marginal_peak_gb": round((grad_peak - base_peak) / 1024**3, 3),
    "ladder_reference": {"fwd": 0.1831, "dq_full": 0.2036, "dkv_full": 0.3829,
                         "d_kernel": 0.0007},
}
ART.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
