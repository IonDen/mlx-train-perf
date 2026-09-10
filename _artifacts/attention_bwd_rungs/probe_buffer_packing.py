"""T9b Step 0 — buffer-packing / OS-kill ground-truth probe.

Ground-truths the macOS GPU interactivity-kill mechanics that T6 rung 0 could only
infer from two kills + one survivor (threshold in (2.2s, 8.7s) cumulative per eval;
packing mechanics assumed, not observed). Questions this probe answers directly:

  Q1. Where is the kill boundary for K identical ~0.25s custom-kernel dispatches
      packed under ONE mx.eval?  (sweep K: cumulative ~1s .. ~8s)
  Q2. Does splitting the SAME K dispatches across multiple mx.eval calls raise the
      survivable cumulative total (i.e. is the limit truly per command buffer /
      per eval, or per sustained-saturation window)?
  Q3. Does mx.async_eval between chunks flush command buffers (survivable at a K
      that the single-eval condition kills)?
  Q4. Where is the SINGLE-dispatch kill boundary (0.25s proven safe; rung 0's
      killed dispatch was "projected 1.0s" at an optimistic rate — real length
      ambiguous)? Conditions: one dispatch at ~0.5s and ~1.0s measured.

Design: subprocess-per-condition (a kill can be an uncatchable SIGABRT — exit 134),
each child writes its JSON artifact the instant it finishes; the parent skips
conditions whose artifact already exists (resume) and writes the artifact itself
when the child died before it could (the kill IS the datum).

Known-duration dispatch: one full launch_flash_fwd call at causal=False with
rate_macs_per_s=None == exactly ONE kernel dispatch (no splitting), self-shaped so
every dispatch does identical work. The child measures the real per-dispatch wall
(median of 3, after JIT warmup) and adapts n once toward the 0.25s target, so the
sweep is reported in MEASURED cumulative seconds, not projections.

Run:  uv run python _artifacts/attention_bwd_rungs/probe_buffer_packing.py
      (parent mode; add --child <condition-json> for the child protocol)
"""

import argparse
import json
import math
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ART_DIR = pathlib.Path(__file__).resolve().parent / "buffer_packing_probe"
TARGET_DISPATCH_S = 0.25   # the proven-safe per-dispatch class (T6 rung 0)
BASE_N = 2048              # ~0.27s projected at the rung-0 scalar rate; child adapts once


def _child(cond: dict) -> None:
    import mlx.core as mx

    from mlx_train_perf.attention.kernel.launch import TileShape, launch_flash_fwd
    from mlx_train_perf.core.guards import install_guardrails

    install_guardrails()

    d, hq = 128, 8
    scale = 1.0 / math.sqrt(d)
    tile = TileShape()  # scalar v0 body — the measured, well-understood baseline

    def make_qkv(n: int, seed: int) -> tuple:
        ks = mx.random.split(mx.random.key(seed), 3)
        q = mx.random.normal((1, hq, n, d), dtype=mx.bfloat16, key=ks[0])
        k = mx.random.normal((1, hq, n, d), dtype=mx.bfloat16, key=ks[1])
        v = mx.random.normal((1, hq, n, d), dtype=mx.bfloat16, key=ks[2])
        mx.eval(q, k, v)
        return q, k, v

    # JIT warmup at a small shape (same kernel instance: head_dim/causal/variant match).
    qw, kw, vw = make_qkv(256, 0)
    mx.eval(*launch_flash_fwd(qw, kw, vw, scale=scale, causal=False, tile=tile))

    # Calibrate at the SAFE 0.25s class only (a longer calibration dispatch would itself
    # be the experiment); adapt n once toward that anchor (cost scales ~ n^2), then scale
    # n for the condition's own target from the measured rate.
    target_s = cond.get("target_s", TARGET_DISPATCH_S)
    n = BASE_N
    per_dispatch = 0.0
    for _attempt in range(2):
        q, k, v = make_qkv(n, 1)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            mx.eval(*launch_flash_fwd(q, k, v, scale=scale, causal=False, tile=tile))
            reps.append(time.perf_counter() - t0)
        per_dispatch = statistics.median(reps)
        if 0.66 * TARGET_DISPATCH_S <= per_dispatch <= 1.5 * TARGET_DISPATCH_S:
            break
        n = max(256, 64 * round(n * math.sqrt(TARGET_DISPATCH_S / per_dispatch) / 64))
    if target_s != TARGET_DISPATCH_S:
        n_anchor, t_anchor = n, per_dispatch
        n = max(256, 64 * round(n_anchor * math.sqrt(target_s / t_anchor) / 64))
        per_dispatch = t_anchor * (n / n_anchor) ** 2  # projected, not measured

    K = cond["k"]
    mode = cond["mode"]          # single_eval | eval_every | async_every
    every = cond.get("every", 0)
    # 4 distinct input sets cycled — mirrors production (distinct ranges), defeats any
    # hypothetical common-subexpression merging of identical dispatch nodes.
    inputs = [make_qkv(n, 10 + i) for i in range(4)]

    result = {
        "condition": cond["name"], "mode": mode, "k": K, "every": every,
        "n": n, "per_dispatch_s": round(per_dispatch, 4),
        "projected_cumulative_s": round(K * per_dispatch, 2),
        "mlx": mx.__version__, "macos": platform.mac_ver()[0],
        "script": str(pathlib.Path(__file__).resolve()),
    }
    out_path = ART_DIR / f"{cond['name']}.json"
    outs: list = []
    t0 = time.perf_counter()
    try:
        for i in range(K):
            q, k, v = inputs[i % 4]
            o, lse = launch_flash_fwd(q, k, v, scale=scale, causal=False, tile=tile)
            outs += [o, lse]
            if every and (i + 1) % every == 0:
                if mode == "eval_every":
                    mx.eval(outs)
                elif mode == "async_every":
                    mx.async_eval(outs)
        mx.eval(outs)
        result["outcome"] = "survived"
    except RuntimeError as exc:  # the catchable next-sync form of the kill
        result["outcome"] = "killed_runtime_error"
        result["error"] = str(exc)[:500]
    result["wall_s"] = round(time.perf_counter() - t0, 2)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"::PROBE:: {result['condition']}: {result['outcome']} "
          f"(measured {result['per_dispatch_s']}s/dispatch x {K} = "
          f"{result['projected_cumulative_s']}s projected, wall {result['wall_s']}s)")


def _parent() -> None:
    conditions = [
        {"name": f"single_eval_k{k:02d}", "mode": "single_eval", "k": k}
        for k in (4, 8, 12, 16, 20, 24, 28, 32)
    ] + [
        {"name": "eval_every8_k32", "mode": "eval_every", "k": 32, "every": 8},
        {"name": "async_every4_k32", "mode": "async_every", "k": 32, "every": 4},
        {"name": "async_every8_k32", "mode": "async_every", "k": 32, "every": 8},
        {"name": "single_dispatch_0p5s", "mode": "single_eval", "k": 1, "target_s": 0.5},
        {"name": "single_dispatch_1p0s", "mode": "single_eval", "k": 1, "target_s": 1.0},
    ]
    ART_DIR.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        out_path = ART_DIR / f"{cond['name']}.json"
        if out_path.exists():
            print(f"skip (exists): {cond['name']}")
            continue
        proc = subprocess.run(
            [sys.executable, __file__, "--child", json.dumps(cond)],
            capture_output=True, text=True, timeout=300,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0 and not out_path.exists():
            outcome = (
                "killed_sigabrt" if proc.returncode in (134, -6)
                else f"child_error_rc{proc.returncode}"
            )
            out_path.write_text(json.dumps({
                "condition": cond["name"], "mode": cond["mode"], "k": cond["k"],
                "every": cond.get("every", 0), "outcome": outcome,
                "stderr_tail": proc.stderr[-800:],
                "script": str(pathlib.Path(__file__).resolve()),
            }, indent=2) + "\n")
            print(f"::PROBE:: {cond['name']}: {outcome}")
        time.sleep(2)  # let the window manager recover between conditions

    print("\n=== summary ===")
    for f in sorted(ART_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        print(f"{r['condition']:>22}: {r['outcome']:>22} "
              f"(projected {r.get('projected_cumulative_s', '?')}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.child:
        _child(json.loads(args.child))
    else:
        _parent()
