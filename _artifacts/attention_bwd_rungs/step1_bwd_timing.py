"""T9b Step 1 — flagship-shape backward timing checkpoint (correctness kernels, un-tuned).

Measures, at the flagship shape (b=1, 32q/8kv, N=8192, D=128, bf16, causal):
  - forward (mma/slab128 via the dispatch table): REAL full-pass median wall
  - D preprocess: REAL full-shape median wall
  - dQ and dK/dV: canary-measured achieved MAC/s (tail query rows vs FULL keys — the
    production-shaped probe under causal; a full pass at v1 scalar rates cannot fit the
    per-eval budget, so passes are PROJECTED from the canary rate, T6-rung-0 style)
  - the production calibrated rates (safety-halved) and the plan_dkv_dispatches /
    dQ-budget verdicts they produce at flagship (split-or-refuse — THE Step 2 gate input)
  - the T9-review shared-rate condition: throughput_dq >= 0.5 * throughput_dkv
  - compiled-ceiling register-pressure telltales for the dq/dkv bodies per head_dim
    (T9 review carry-forward; diagnostic only, never a rate predictor — gotcha 0012)

Artifact: step1_bwd_timing.json (written incrementally per phase).
Run:  uv run python _artifacts/attention_bwd_rungs/step1_bwd_timing.py   (~1-2 min)
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
    MAX_DISPATCH_SECONDS,
    MAX_TOTAL_SECONDS,
    TileShape,
    _bwd_dkv_kernel,
    _bwd_dkv_macs_per_row,
    _bwd_dq_kernel,
    _bwd_dq_macs_per_row,
    _check_launch_budget,
    _dispatch_bwd_dkv_range,
    _dispatch_bwd_dq_range,
    _fwd_macs_per_row,
    _rows_within_dispatch_budget,
    calibrated_bwd_rate,
    calibrated_fwd_rate,
    launch_bwd_D,
    launch_flash_fwd,
    plan_dkv_dispatches,
)
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.errors import LaunchBudgetError

ART = pathlib.Path(__file__).resolve().parent / "step1_bwd_timing.json"
B, HQ, HKV, N, D = 1, 32, 8, 8192, 128
DTYPE = mx.bfloat16
SCALE = 1.0 / math.sqrt(D)
CANARY_TARGET_S = 0.25
REPS = 5

result: dict = {
    "rung": "t9b_step1_bwd_timing",
    "shape": {"b": B, "hq": HQ, "hkv": HKV, "n": N, "d": D, "dtype": "bfloat16",
              "causal": True},
    "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    "mlx": mx.__version__,
    "budgets": {"max_dispatch_s": MAX_DISPATCH_SECONDS, "max_total_s": MAX_TOTAL_SECONDS},
}


def save() -> None:
    ART.write_text(json.dumps(result, indent=2) + "\n")


def median_wall(fn, reps: int = REPS) -> tuple[float, list[float]]:
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        walls.append(round(time.perf_counter() - t0, 4))
    return statistics.median(walls), walls


install_guardrails()
kq, kk, kv, kdo = mx.random.split(mx.random.key(0), 4)
q = mx.random.normal((B, HQ, N, D), dtype=DTYPE, key=kq)
k = mx.random.normal((B, HKV, N, D), dtype=DTYPE, key=kk)
v = mx.random.normal((B, HKV, N, D), dtype=DTYPE, key=kv)
d_o = mx.random.normal((B, HQ, N, D), dtype=DTYPE, key=kdo)
mx.eval(q, k, v, d_o)

# --- Phase 1: forward (table-selected mma variant), real full pass -------------------
tile = select_fwd_tile(N, D)
fwd_rate = calibrated_fwd_rate(
    head_dim=D, dtype=DTYPE, b=B, hq=HQ, hkv=HKV, n=N, causal=True, tile=tile,
)
mx.eval(*launch_flash_fwd(q, k, v, scale=SCALE, causal=True, tile=tile,
                          rate_macs_per_s=fwd_rate))  # JIT + splits warm
fwd_med, fwd_walls = median_wall(
    lambda: launch_flash_fwd(q, k, v, scale=SCALE, causal=True, tile=tile,
                             rate_macs_per_s=fwd_rate)
)
fwd_guard_total = N * _fwd_macs_per_row(n=N, d=D, b=B, hq=HQ)
result["forward"] = {
    "tile": {"variant": tile.variant, "d_slab": tile.d_slab,
             "provisional": tile.provisional},
    "calibrated_rate_g": round(fwd_rate / 1e9, 1),
    "median_s": fwd_med, "walls_s": fwd_walls,
    "achieved_g_mac_s_causal_true": round(fwd_guard_total / 2 / fwd_med / 1e9, 1),
}
save()
o, lse = launch_flash_fwd(q, k, v, scale=SCALE, causal=True, tile=tile,
                          rate_macs_per_s=fwd_rate)
mx.eval(o, lse)

# --- Phase 2: D preprocess, real full shape ------------------------------------------
mx.eval(launch_bwd_D(d_o, o))  # JIT warm
d_med, d_walls = median_wall(lambda: launch_bwd_D(d_o, o))
d_arr = launch_bwd_D(d_o, o)
mx.eval(d_arr)
result["bwd_D"] = {"median_s": d_med, "walls_s": d_walls}
save()

# --- Phase 3: dQ canary (tail rows vs full keys) --------------------------------------
scale_in = mx.array([SCALE], dtype=mx.float32)


def canary(name: str, dispatch_rows, per_row: int) -> dict:
    """Measure achieved MAC/s from a tail-range dispatch sized to ~CANARY_TARGET_S.
    dispatch_rows(rows) must launch ONE dispatch covering query rows [N-rows, N)."""
    rows = 64
    mx.eval(dispatch_rows(rows))  # JIT warm
    t0 = time.perf_counter()
    mx.eval(dispatch_rows(rows))
    t = time.perf_counter() - t0
    rate0 = rows * per_row / t
    rows = max(16, min(N, int(CANARY_TARGET_S * rate0 / per_row)))
    walls = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(dispatch_rows(rows))
        walls.append(round(time.perf_counter() - t0, 4))
    med = statistics.median(walls)
    return {"canary_rows": rows, "median_s": med, "walls_s": walls,
            "achieved_g_mac_s": round(rows * per_row / med / 1e9, 2)}


dq_kernel = _bwd_dq_kernel(D, True, False)
dq_per_row = _bwd_dq_macs_per_row(n=N, d=D, b=B, hq=HQ)
result["bwd_dq"] = canary(
    "dq",
    lambda rows: _dispatch_bwd_dq_range(
        dq_kernel, q, k, v, d_o, lse, d_arr, scale_in, r0=N - rows, r1=N),
    dq_per_row,
)
save()

# --- Phase 4: dK/dV canary (tail rows vs full keys, zero-seeded partials) --------------
dkv_kernel = _bwd_dkv_kernel(D, True, False)
dkv_per_row = _bwd_dkv_macs_per_row(n=N, d=D, b=B, hq=HQ)
dk0 = mx.zeros((B, HKV, N, D), dtype=mx.float32)
dv0 = mx.zeros((B, HKV, N, D), dtype=mx.float32)
mx.eval(dk0, dv0)
result["bwd_dkv"] = canary(
    "dkv",
    lambda rows: _dispatch_bwd_dkv_range(
        dkv_kernel, q, k, v, d_o, lse, d_arr, dk0, dv0, scale_in,
        q_lo=N - rows, q_hi=N),
    dkv_per_row,
)
save()

# --- Phase 5: production calibrated rates + split-or-refuse verdicts ------------------
bwd_rate = calibrated_bwd_rate(
    head_dim=D, dtype=DTYPE, b=B, hq=HQ, hkv=HKV, n=N, causal=True,
)
verdicts: dict = {"calibrated_bwd_rate_g": round(bwd_rate / 1e9, 2),
                  "calibrated_fwd_rate_g": round(fwd_rate / 1e9, 1)}
try:
    ranges = plan_dkv_dispatches(n=N, d=D, b=B, hq=HQ, rate=bwd_rate)
    verdicts["dkv_flagship"] = {
        "verdict": "splits", "num_dispatches": len(ranges),
        "projected_total_s": round(N * dkv_per_row / bwd_rate, 2)}
except LaunchBudgetError as e:
    verdicts["dkv_flagship"] = {"verdict": "REFUSES", "error": str(e)}
try:
    rows_per = _rows_within_dispatch_budget(per_row=dq_per_row, n=N, rate=bwd_rate)
    _check_launch_budget(per_row=dq_per_row, n=N, rows=rows_per, rate=bwd_rate)
    verdicts["dq_flagship"] = {
        "verdict": "splits", "num_dispatches": math.ceil(N / rows_per),
        "projected_total_s": round(N * dq_per_row / bwd_rate, 2)}
except LaunchBudgetError as e:
    verdicts["dq_flagship"] = {"verdict": "REFUSES", "error": str(e)}
result["production_verdicts"] = verdicts
save()

# --- Phase 6: projections + the shared-rate condition ---------------------------------
r_dq = result["bwd_dq"]["achieved_g_mac_s"] * 1e9
r_dkv = result["bwd_dkv"]["achieved_g_mac_s"] * 1e9
proj_dq = N * dq_per_row / 2 / r_dq          # causal-true ~ guard/2
proj_dkv = N * dkv_per_row / 2 / r_dkv
bwd_total = result["bwd_D"]["median_s"] + proj_dq + proj_dkv
result["projections_causal_true"] = {
    "dq_full_pass_s": round(proj_dq, 2),
    "dkv_full_pass_s": round(proj_dkv, 2),
    "backward_total_s": round(bwd_total, 2),
    "forward_measured_s": fwd_med,
    "bwd_over_fwd_wall": round(bwd_total / fwd_med, 1),
    "spec_4_4_reference": "realistic v1 ~0.3 TMAC/s => dkv ~1.8s; bwd ~3x fwd wall",
}
result["shared_rate_condition"] = {
    "throughput_dq_g": round(r_dq / 1e9, 2),
    "half_throughput_dkv_g": round(r_dkv / 2 / 1e9, 2),
    "dq_ge_half_dkv": bool(r_dq >= 0.5 * r_dkv),
}
save()

# --- Phase 7: compiled-ceiling register telltales (diagnostic only) --------------------
try:
    from mlx_train_perf.attention.kernel.source import (
        build_bwd_dkv_source,
        build_bwd_dq_source,
    )
    from mlx_train_perf.devtools.regpressure import compiled_ceiling

    ceilings: dict = {}
    nn, bb = 16, 1
    for hd in (64, 96, 128):
        kq2, kk2, kv2, kdo2 = mx.random.split(mx.random.key(1), 4)
        q2 = mx.random.normal((bb, 4, nn, hd), dtype=mx.bfloat16, key=kq2)
        k2 = mx.random.normal((bb, 2, nn, hd), dtype=mx.bfloat16, key=kk2)
        v2 = mx.random.normal((bb, 2, nn, hd), dtype=mx.bfloat16, key=kv2)
        do2 = mx.random.normal((bb, 4, nn, hd), dtype=mx.bfloat16, key=kdo2)
        l2 = mx.zeros((bb, 4, nn), dtype=mx.float32)
        da2 = mx.zeros((bb, 4, nn), dtype=mx.float32)
        qo2 = mx.array([0, nn], dtype=mx.uint32)
        dk2 = mx.zeros((bb, 2, nn, hd), dtype=mx.float32)
        dv2 = mx.zeros((bb, 2, nn, hd), dtype=mx.float32)
        mx.eval(q2, k2, v2, do2, l2, da2, qo2, dk2, dv2)
        ceilings[f"dq_d{hd}"] = compiled_ceiling(
            build_bwd_dq_source(hd, causal=True),
            input_names=["q", "k", "v", "d_o", "lse", "d_arr", "qoffs", "scale_in"],
            inputs=[q2, k2, v2, do2, l2, da2, qo2, scale_in],
            output_names=["dq_out"], output_shapes=[(bb, 4, nn, hd)],
            output_dtypes=[mx.bfloat16],
        )
        ceilings[f"dkv_d{hd}"] = compiled_ceiling(
            build_bwd_dkv_source(hd, causal=True),
            input_names=["q", "k", "v", "d_o", "lse", "d_arr",
                         "dk_in", "dv_in", "qoffs", "scale_in"],
            inputs=[q2, k2, v2, do2, l2, da2, dk2, dv2, qo2, scale_in],
            output_names=["dk_out", "dv_out"],
            output_shapes=[(bb, 2, nn, hd), (bb, 2, nn, hd)],
            output_dtypes=[mx.float32, mx.float32],
        )
    result["compiled_ceilings"] = ceilings
except Exception as exc:  # pyobjc absent or probe failure — record, don't die
    result["compiled_ceilings"] = {"unavailable": str(exc)[:300]}
save()

print(json.dumps(result, indent=2))
