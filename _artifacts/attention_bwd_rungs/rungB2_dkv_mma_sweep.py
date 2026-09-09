"""T9b rung B2 saturation sweep — dK/dV MMA d_slab ladder at the flagship canary.

Same discipline as rungB1_dq_mma_sweep.py: flagship-shape tail-query-range canary per
slab (one artifact each, resume = skip-if-exists), winner gets a real full flagship
chained dK/dV pass (launcher-split at the measured rate, refusal caught honestly).

Run (controller, after the rung B2 review approves):
  uv run python _artifacts/attention_bwd_rungs/rungB2_dkv_mma_sweep.py   (~1-2 min)
"""

import json
import math
import pathlib
import statistics
import subprocess
import time

import mlx.core as mx

from mlx_train_perf.attention.kernel.dispatch import select_fwd_tile
from mlx_train_perf.attention.kernel.launch import (
    _bwd_dkv_kernel,
    _bwd_dkv_macs_per_row,
    _dispatch_bwd_dkv_range,
    launch_bwd_D,
    launch_bwd_dkv,
    launch_flash_fwd,
)
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.errors import LaunchBudgetError

ART_DIR = pathlib.Path(__file__).resolve().parent
B, HQ, HKV, N, D = 1, 32, 8, 8192, 128
DTYPE = mx.bfloat16
SCALE = 1.0 / math.sqrt(D)
SLABS = (16, 32, 64, 128)
SCALAR_BASELINE_G = 82.89   # step1_bwd_timing.json
GIT_SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

install_guardrails()
kq, kk, kv, kdo = mx.random.split(mx.random.key(0), 4)
q = mx.random.normal((B, HQ, N, D), dtype=DTYPE, key=kq)
k = mx.random.normal((B, HKV, N, D), dtype=DTYPE, key=kk)
v = mx.random.normal((B, HKV, N, D), dtype=DTYPE, key=kv)
d_o = mx.random.normal((B, HQ, N, D), dtype=DTYPE, key=kdo)
mx.eval(q, k, v, d_o)

tile = select_fwd_tile(N, D)
o, lse = launch_flash_fwd(q, k, v, scale=SCALE, causal=True, tile=tile,
                          rate_macs_per_s=1.4e12)
d_arr = launch_bwd_D(d_o, o)
mx.eval(o, lse, d_arr)
scale_in = mx.array([SCALE], dtype=mx.float32)
per_row = _bwd_dkv_macs_per_row(n=N, d=D, b=B, hq=HQ)
dk0 = mx.zeros((B, HKV, N, D), dtype=mx.float32)
dv0 = mx.zeros((B, HKV, N, D), dtype=mx.float32)
mx.eval(dk0, dv0)

# Small-shape cross-variant consistency inputs (parity itself is owned by the tests).
ns = 256
q2, k2, v2, do2 = (a[:, :, :ns, :] for a in (q, k, v, d_o))
o2, l2 = launch_flash_fwd(q2, k2, v2, scale=SCALE, causal=True,
                          tile=select_fwd_tile(ns, D))
da2 = launch_bwd_D(do2, o2)
dk2_s, dv2_s = launch_bwd_dkv(q2, k2, v2, do2, l2, da2, scale=SCALE, causal=True)
mx.eval(q2, k2, v2, do2, l2, da2, dk2_s, dv2_s)

results = {}
for slab in SLABS:
    out = ART_DIR / f"rungB2_dkv_mma_slab{slab}.json"
    if out.exists():
        results[slab] = json.loads(out.read_text())
        print(f"skip (exists): slab{slab} -> {results[slab]['achieved_g_mac_s']} G")
        continue
    kern = _bwd_dkv_kernel(D, True, False, "mma", slab)
    dk2_m, dv2_m = launch_bwd_dkv(q2, k2, v2, do2, l2, da2, scale=SCALE, causal=True,
                                  variant="mma", d_slab=slab)
    xvar = max(
        float(mx.abs(dk2_m.astype(mx.float32) - dk2_s.astype(mx.float32)).max()),
        float(mx.abs(dv2_m.astype(mx.float32) - dv2_s.astype(mx.float32)).max()),
    )
    rows = 64
    mx.eval(*_dispatch_bwd_dkv_range(kern, q, k, v, d_o, lse, d_arr, dk0, dv0,
                                     scale_in, q_lo=N - rows, q_hi=N, variant="mma"))
    t0 = time.perf_counter()
    mx.eval(*_dispatch_bwd_dkv_range(kern, q, k, v, d_o, lse, d_arr, dk0, dv0,
                                     scale_in, q_lo=N - rows, q_hi=N, variant="mma"))
    rate0 = rows * per_row / (time.perf_counter() - t0)
    rows = max(32, min(N, int(0.25 * rate0 / per_row)) // 32 * 32)
    walls = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(*_dispatch_bwd_dkv_range(kern, q, k, v, d_o, lse, d_arr, dk0, dv0,
                                         scale_in, q_lo=N - rows, q_hi=N,
                                         variant="mma"))
        walls.append(round(time.perf_counter() - t0, 4))
    med = statistics.median(walls)
    rec = {
        "rung": f"rungB2_dkv_mma_slab{slab}", "d_slab": slab, "git_sha": GIT_SHA,
        "mlx": mx.__version__, "canary_rows": rows, "median_s": med,
        "walls_s": walls, "achieved_g_mac_s": round(rows * per_row / med / 1e9, 2),
        "xvariant_maxdiff_n256_vs_scalar": xvar,
    }
    out.write_text(json.dumps(rec, indent=2) + "\n")
    results[slab] = rec
    print(f"slab{slab}: {rec['achieved_g_mac_s']} G MAC/s "
          f"(rows {rows}, med {med}s, xvar {xvar:.2e})")

winner = max(results.values(), key=lambda r: r["achieved_g_mac_s"])
print(f"\nwinner: slab{winner['d_slab']} at {winner['achieved_g_mac_s']} G "
      f"(scalar baseline {SCALAR_BASELINE_G} G, "
      f"gain {winner['achieved_g_mac_s'] / SCALAR_BASELINE_G:.1f}x)")

full_out = ART_DIR / "rungB2_dkv_mma_full.json"
if not full_out.exists():
    rate = winner["achieved_g_mac_s"] * 1e9
    rec = {"rung": "rungB2_dkv_mma_full", "d_slab": winner["d_slab"],
           "git_sha": GIT_SHA, "mlx": mx.__version__,
           "rate_passed_g": winner["achieved_g_mac_s"]}
    try:
        mx.eval(*launch_bwd_dkv(q, k, v, d_o, lse, d_arr, scale=SCALE, causal=True,
                                rate_macs_per_s=rate, variant="mma",
                                d_slab=winner["d_slab"]))
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            mx.eval(*launch_bwd_dkv(q, k, v, d_o, lse, d_arr, scale=SCALE,
                                    causal=True, rate_macs_per_s=rate,
                                    variant="mma", d_slab=winner["d_slab"]))
            walls.append(round(time.perf_counter() - t0, 4))
        rec |= {"outcome": "ran", "median_s": statistics.median(walls),
                "walls_s": walls,
                "achieved_g_mac_s_causal_true":
                    round(N * per_row / 2 / statistics.median(walls) / 1e9, 2)}
    except LaunchBudgetError as e:
        rec |= {"outcome": "REFUSES", "error": str(e)}
    full_out.write_text(json.dumps(rec, indent=2) + "\n")
    print("full pass:", json.dumps(rec, indent=2))
else:
    print("full pass artifact exists, skipped")
