"""Ground-truth probe: MLX command-buffer packing + the launch-budget kill-unit model.

Evidence instrument for the 0.3.0 launch-budget study. mlx 0.32.0's Metal backend
(`mlx/backend/metal/device.cpp`, read verbatim at tag v0.32.0) commits a command buffer
when `buffer_ops_ > max_ops` OR `(buffer_sizes_ >> 20) > max_mb`, where `buffer_sizes_`
accumulates `array.data_size()` (ELEMENTS, not bytes) once per unique input/output buffer,
and (max_ops, max_mb) = (50, 50) on this machine's `applegpu_g13s` ('s' == max class),
overridable via MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER (both strings present in the
shipped libmlx.dylib). Consequence at flagship attention shapes: every dQ / dK/dV dispatch
exceeds the element threshold on its own, so each runs in ITS OWN command buffer -- the
"chain packs into one kill-unit" reading behind MAX_TOTAL_SECONDS is wrong at those shapes.

The trace store in a .gputrace is opaque without Xcode replay, so this probe validates the
model behaviorally instead:

  probe 1 (ops threshold):  a serial chain of tiny matmuls forces a cross-buffer fence wait
      per boundary; MLX_MAX_OPS_PER_BUFFER=0 (boundary every op) must be measurably slower
      than the default (boundary every 51 ops). Confirms the ops-limit code path is live.
  probe 2 (size threshold): a serial chain of ~13 M-element adds crosses the 50 M-element
      limit every ~4 ops at defaults; MLX_MAX_MB_PER_BUFFER=100000 removes those boundaries.
      A measurable wall gap confirms the size-limit code path is live.
  probe 3 (consequence):    a REAL chained dK/dV pass at a 12288-token flagship-class shape,
      manually range-split so the chain's total projected wall is far ABOVE the 2.0 s
      MAX_TOTAL_SECONDS class the shipping guard refuses -- but every dispatch is its own
      command buffer (inputs ~118 M elements >> 50 M) within the worst-day-proven
      per-dispatch class. The model predicts a clean, kill-free completion.

Each probe runs in a SUBPROCESS (env knobs are read once at Device construction; also the
standard bench isolation discipline). Results land in
`_artifacts/launch_budget_evidence/probe_results.json`. Today's machine state is recorded;
a survival here is TODAY'S-STATE evidence and never licenses raising any pinned budget.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ART_DIR = Path(__file__).resolve().parent.parent / "_artifacts" / "launch_budget_evidence"

_CHILD_TINY_OPS = """
import time, statistics
import mlx.core as mx
a = mx.random.normal((64, 64)); b = mx.random.normal((64, 64)); mx.eval(a, b)
def run():
    c = a
    for _ in range(2000):
        c = c @ b
    mx.eval(c)
run()  # warmup: JIT + clocks
walls = []
for _ in range(5):
    t0 = time.perf_counter(); run(); walls.append(time.perf_counter() - t0)
print(statistics.median(walls))
"""

_CHILD_BIG_SIZES = """
import time, statistics
import mlx.core as mx
mx.set_wired_limit(int(20 * 1024**3))
xs = [mx.random.normal((13_000_000,)) for _ in range(4)]   # ~13 M elements each, fp32
mx.eval(*xs)
def run():
    c = xs[0]
    for i in range(200):
        c = c + xs[i % 4]
    mx.eval(c)
run()
walls = []
for _ in range(5):
    t0 = time.perf_counter(); run(); walls.append(time.perf_counter() - t0)
print(statistics.median(walls))
"""

_CHILD_DKV_CHAIN = """
import time
import mlx.core as mx
from mlx_train_perf.attention.kernel.launch import (
    _bwd_dkv_kernel, _dispatch_bwd_dkv_range,
)
mx.set_wired_limit(int(20 * 1024**3))
mx.set_memory_limit(int(22 * 1024**3))
B, HQ, HKV, N, D = 1, 32, 8, 12288, 128
dt = mx.bfloat16
ks = mx.random.split(mx.random.key(0), 4)
q  = mx.random.normal((B, HQ, N, D), key=ks[0]).astype(dt)
k  = mx.random.normal((B, HKV, N, D), key=ks[1]).astype(dt)
v  = mx.random.normal((B, HKV, N, D), key=ks[2]).astype(dt)
do = mx.random.normal((B, HQ, N, D), key=ks[3]).astype(dt)
lse = mx.zeros((B, HQ, N), dtype=mx.float32)
darr = mx.zeros((B, HQ, N), dtype=mx.float32)
mx.eval(q, k, v, do, lse, darr)
kernel = _bwd_dkv_kernel(D, True, False, "mma", 128)
scale_in = mx.array([1.0 / D ** 0.5], dtype=mx.float32)
# 12 ascending 32-aligned ranges (1024 rows each): per-dispatch projected ~0.35 s at the
# measured 1857.94 G raw rate nominal cost -- inside the shipped per-dispatch class; the
# chain TOTAL projects ~4.3 s nominal (>> the 2.0 s the shipping guard allows).
ranges = [(i * 1024, (i + 1) * 1024) for i in range(12)]
dk = mx.zeros((B, HKV, N, D), dtype=mx.float32)
dv = mx.zeros((B, HKV, N, D), dtype=mx.float32)
# warmup at a tiny range for JIT only
wdk, wdv = _dispatch_bwd_dkv_range(kernel, q, k, v, do, lse, darr, dk, dv, scale_in,
                                   q_lo=0, q_hi=32, variant="mma")
mx.eval(wdk, wdv)
t0 = time.perf_counter()
for q_lo, q_hi in ranges:
    dk, dv = _dispatch_bwd_dkv_range(kernel, q, k, v, do, lse, darr, dk, dv, scale_in,
                                     q_lo=q_lo, q_hi=q_hi, variant="mma")
mx.eval(dk, dv)
wall = time.perf_counter() - t0
ok = bool(mx.isfinite(dk).all().item() and mx.isfinite(dv).all().item())
print(f"{wall} {ok} {mx.get_peak_memory()}")
"""


def _run_child(code: str, env_extra: dict[str, str]) -> str:
    env = dict(os.environ, **env_extra)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600,
    )
    if out.returncode != 0:
        return f"CHILD_FAILED rc={out.returncode} stderr_tail={out.stderr[-800:]}"
    return out.stdout.strip().splitlines()[-1]


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "probe": "command_buffer_packing",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.platform(),
        "model_source": "mlx v0.32.0 mlx/backend/metal/device.cpp (needs_commit, L512-515; "
                        "arch switch L601-623; commit resets L554-557)",
    }

    print("probe 1/3: ops-threshold discriminator (tiny serial matmuls, ~30-60 s) ...")
    results["probe1_tiny_ops"] = {
        "default": _run_child(_CHILD_TINY_OPS, {}),
        "ops_0": _run_child(_CHILD_TINY_OPS, {"MLX_MAX_OPS_PER_BUFFER": "0"}),
        "ops_5000": _run_child(_CHILD_TINY_OPS, {"MLX_MAX_OPS_PER_BUFFER": "5000"}),
        "prediction": "ops_0 markedly slower than default; ops_5000 <= default",
    }
    print("   ", results["probe1_tiny_ops"])

    print("probe 2/3: size-threshold discriminator (13 M-element serial adds, ~30-60 s) ...")
    results["probe2_big_sizes"] = {
        "default": _run_child(_CHILD_BIG_SIZES, {}),
        "mb_huge": _run_child(_CHILD_BIG_SIZES, {"MLX_MAX_MB_PER_BUFFER": "100000"}),
        "mb_1": _run_child(_CHILD_BIG_SIZES, {"MLX_MAX_MB_PER_BUFFER": "1"}),
        "prediction": "mb_1 slower than default (boundary every op); mb_huge fastest or ~default",
    }
    print("   ", results["probe2_big_sizes"])

    print("probe 3/3: real dK/dV chain at 12288, total FAR above the old 2.0 s class (~1-2 min) ...")
    results["probe3_dkv_chain_12288"] = {
        "result": _run_child(_CHILD_DKV_CHAIN, {}),
        "fields": "wall_s finite_ok peak_bytes",
        "note": "12 dispatches, each its own predicted command buffer (inputs ~118 M elements "
                "> 50 M threshold); survival is today's-state evidence only",
    }
    print("   ", results["probe3_dkv_chain_12288"])

    out = ART_DIR / "probe_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
