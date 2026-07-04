"""Production-shape (n=8192, V=151936, D=4096) release-threshold bench for the
dequant-in-kernel quantized forward (Task 10 / backlog 0009).

Conditions (subprocess-per-condition — own MLX allocator state, no cross-condition
buffer retention):
  dense_kernel          — `forward` with a full bf16 head (the existing baseline).
  quant_kernel           — `forward_quantized`: int4/gs64 packed head, dequantized
                           in-register inside the kernel (this task's new path).
  dequant_once_then_dense — `mx.dequantize` the packed head ONCE into a materialized
                           bf16 copy, then run the DENSE kernel on it. Represents the
                           "just dequantize upfront" alternative the quant kernel is
                           meant to avoid; its peak deliberately includes that copy.

Each condition writes its own JSON artifact to `_artifacts/bench_quant_thresholds/`
the instant it finishes, and resume skips a condition whose artifact identity
(mlx version, machine, code hash of the files this bench depends on, shape) is fresh
— an interruption loses at most one condition's run.

Memory protocol (workflow-and-gotchas.md `reset_peak_memory` semantics): each
condition builds hidden/targets AND its OWN head representation (bf16 weights /
packed int4 / packed int4 + a materialized dequantized bf16 copy), evaluates and
warms up (pays Metal JIT + the rate-calibration probe) OUTSIDE the measured window,
THEN takes `active_before = mx.get_active_memory()` and resets peak — so
`active_before_gb` is directly comparable ACROSS conditions (it's where the head-size
delta the release threshold cares about shows up) and `marginal_peak_gb` is the
incremental cost of running the forward passes themselves, not of holding the weights.

Release thresholds this bench feeds (evaluated by the CONTROLLER against the written
artifacts — out of this script's scope, see docs/backlog/mlx-train-perf/
mlx-train-perf-0009.md and the task-10 report):
  - quant_kernel g_mac_per_s >= dense_kernel g_mac_per_s / 1.5
  - (dequant_once_then_dense active_before_gb) - (quant_kernel active_before_gb)
    ~= head size (~1.24 GB at this shape)

Heavy GPU run at production shape — main session only, subprocess-per-condition,
ETA ~5 min total for all three conditions (per the task brief's step 8 budget).
Pre-flight `memory_pressure` before running; never invoke from a subagent.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.core.chunked import QuantSpec
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.core.kernel.dispatch import select_variant
from mlx_train_perf.core.kernel.launch import (
    SAFETY_FACTOR,
    calibrated_rate,
    forward,
    forward_quantized,
    probe_tile_for,
)
from mlx_train_perf.errors import LaunchBudgetError

N, V, D = 8192, 151936, 4096
TILE = 8192              # global-constraints vocab tile default; same-tile comparisons only
GROUP_SIZE, BITS = 64, 4
REPS = 3
CONDITIONS = ("dense_kernel", "quant_kernel", "dequant_once_then_dense")

SCHEMA_VERSION = 1
_SCRIPTS_DIR = Path(__file__).parent
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_quant_thresholds"
_DEPS = [
    _SCRIPTS_DIR / "bench_quant_thresholds.py",
    _SCRIPTS_DIR.parent / "src/mlx_train_perf/core/kernel/source.py",
    _SCRIPTS_DIR.parent / "src/mlx_train_perf/core/kernel/launch.py",
]


def _code_sha() -> str:
    """Fingerprint of the files this bench's result depends on — any edit invalidates
    prior artifacts, deliberately (same convention as mlx-train-perf-spike/common.py)."""
    h = hashlib.sha256()
    for p in _DEPS:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _run_identity(*, condition: str, row_tiles: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mlx_version": _installed_mlx_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "code_sha": _code_sha(),
        "probe": "bench_quant_thresholds",
        "condition": condition,
        "n": N, "v": V, "d": D, "tile": TILE, "row_tiles": row_tiles,
        "group_size": GROUP_SIZE, "bits": BITS,
    }


def _write_result(path: Path, identity: dict[str, object], status: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"identity": identity, "status": status, **fields}, indent=2))
    tmp.rename(path)


def _result_is_fresh(path: Path, identity: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("status") == "ok" and data.get("identity") == identity)


def _quant_calibrated_rate(*, row_tiles: int, n: int, d: int, v: int) -> float:
    """Same probe shape/safety convention as launch.calibrated_rate (dense-only — it
    builds a plain bf16 `w`, so it can't be reused directly for a QuantSpec head)."""
    probe_tile = min(probe_tile_for(n=n, d=d), v)
    mx.random.seed(0)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((probe_tile, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, probe_tile, (n,))
    w_q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=BITS)
    elapsed = 0.0
    for _timed in (False, True):
        t0 = time.perf_counter()
        lse, tgt = forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=probe_tile,
                                     rate_macs_per_s=None)
        mx.eval(lse, tgt)
        elapsed = time.perf_counter() - t0
    return SAFETY_FACTOR * (n * probe_tile * d) / max(elapsed, 1e-9)


def _timed_forward(
    fn: Callable[[], tuple[mx.array, mx.array]], *, reps: int,
) -> tuple[float, list[float], float]:
    """Warmup (pays Metal JIT) OUTSIDE the window, then reset-peak semantics: snapshot
    active memory right before reset so `marginal_peak_gb = peak - active_before` is the
    TRUE incremental allocation, not `active_before + incremental` (workflow-and-gotchas.md).
    Returns (marginal_peak_gb, wall_s_per_rep, active_before_gb)."""
    lse, tgt = fn()
    mx.eval(lse, tgt)
    mx.clear_cache()
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lse, tgt = fn()
        mx.eval(lse, tgt)
        walls.append(time.perf_counter() - t0)
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3
    return marginal_peak_gb, walls, active_before / 1024**3


def run_condition(condition: str) -> None:
    install_guardrails()
    row_tiles = select_variant(N).row_tiles
    ident = _run_identity(condition=condition, row_tiles=row_tiles)
    out = RESULTS / f"bench_{condition}.json"
    if _result_is_fresh(out, ident):
        print(f"skip (fresh): {out.name}")
        return

    mx.random.seed(42)
    hidden = mx.random.normal((N, D)).astype(mx.bfloat16)
    targets = mx.random.randint(0, V, (N,))
    mx.eval(hidden, targets)

    if condition == "dense_kernel":
        w = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
        mx.eval(w)
        rate = calibrated_rate(row_tiles=row_tiles, dtype=mx.bfloat16, n=N, d=D, v=V)

        def fn() -> tuple[mx.array, mx.array]:
            return forward(hidden, w, targets, row_tiles=row_tiles, tile=TILE,
                           rate_macs_per_s=rate)
    else:
        # Transient bf16 source, quantized then dropped — a real deploy loads the packed
        # checkpoint directly and never holds a full bf16 copy at serving time.
        w_src = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
        w_q, scales, biases = mx.quantize(w_src, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(w_q, scales, biases)
        del w_src
        mx.clear_cache()

        if condition == "quant_kernel":
            q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=BITS)
            rate = _quant_calibrated_rate(row_tiles=row_tiles, n=N, d=D, v=V)

            def fn() -> tuple[mx.array, mx.array]:
                return forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=TILE,
                                         rate_macs_per_s=rate)
        else:  # dequant_once_then_dense — peak deliberately includes this materialized copy
            w_dq = mx.dequantize(w_q, scales, biases, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(w_dq)
            del w_q, scales, biases   # one representation resident at a time, like a real deploy
            mx.clear_cache()
            rate = calibrated_rate(row_tiles=row_tiles, dtype=mx.bfloat16, n=N, d=D, v=V)

            def fn() -> tuple[mx.array, mx.array]:
                return forward(hidden, w_dq, targets, row_tiles=row_tiles, tile=TILE,
                               rate_macs_per_s=rate)

    try:
        marginal_peak_gb, walls, active_before_gb = _timed_forward(fn, reps=REPS)
    except LaunchBudgetError as exc:
        # A guard refusal IS a result: it means the variant's measured rate cannot run
        # this tile within the watchdog budget — record it, don't crash the sweep.
        _write_result(out, ident, "refused", error=str(exc),
                      calibrated_rate_macs_per_s=round(rate, 1))
        print(f"{condition}: REFUSED by launch-budget guard "
              f"(calibrated {rate / 1e9:.0f} G MAC/s) — recorded as a result")
        return
    med = statistics.median(walls)
    g_mac_per_s = (N * V * D) / med / 1e9
    _write_result(
        out, ident, "ok",
        wall_s_median=round(med, 4),
        wall_s_all=[round(x, 4) for x in walls],
        g_mac_per_s=round(g_mac_per_s, 1),
        active_before_gb=round(active_before_gb, 3),
        marginal_peak_gb=round(marginal_peak_gb, 3),
        total_peak_gb=round(active_before_gb + marginal_peak_gb, 3),
        calibrated_rate_macs_per_s=round(rate, 1),
    )
    print(f"{condition}: median={med:.3f}s rate={g_mac_per_s:.1f} G MAC/s "
          f"active_before={active_before_gb:.2f} GB marginal_peak={marginal_peak_gb:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS)
    args = ap.parse_args()
    if args.condition:
        run_condition(args.condition)
        return 0
    row_tiles = select_variant(N).row_tiles
    for condition in CONDITIONS:
        ident = _run_identity(condition=condition, row_tiles=row_tiles)
        out = RESULTS / f"bench_{condition}.json"
        if _result_is_fresh(out, ident):
            print(f"skip {condition} (fresh)")
            continue
        subprocess.run([sys.executable, __file__, "--condition", condition], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
