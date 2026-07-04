"""Per-rung bench for the backward ladder (Task 16b, step 4 prerequisite).

Conditions (subprocess-per-condition -- own MLX allocator state, no cross-condition
buffer retention):

  staged_vjp_frozen     -- DenseHead(trainable=False) through `linear_cross_entropy`'s
                           `impl="chunked"` custom_function, `mx.value_and_grad` w.r.t.
                           hidden ONLY (argnums=(0,)). `core.chunked.make_chunked_dense`'s
                           vjp unconditionally computes a d_w cotangent too (it does not
                           branch on `head.trainable`), but since nothing downstream ever
                           references or `mx.eval`s that cotangent under argnums=(0,),
                           MLX's lazy scheduler never executes that branch -- confirmed
                           empirically (a throwaway probe measured a ~3.9x wall-time gap
                           between argnums=(0,) and argnums=(0,1) against an identical
                           vjp body with a deliberately expensive d_w branch). This is the
                           SAME mechanism `mlx-train-perf-spike/gate_trainstep.py`'s own
                           `staged`/`staged_frozen` conditions relied on, routed through
                           the shipped library entry point instead of spike-local
                           custom_functions. Re-measures the gate's staged-frozen baseline
                           same-session.
  staged_vjp_trainable  -- DenseHead(trainable=True), `value_and_grad` w.r.t. (hidden, w)
                           (argnums=(0, 1)) -- the shared vjp's d_w branch is genuinely
                           evaluated this time. Re-measures the gate's staged-trainable
                           baseline same-session.
  kernel_dhidden_v0     -- `backward_dhidden` ALONE. `forward` runs once, UNMEASURED
                           (setup only, outside the timed/peak window) to produce the
                           real (lse, tgt) residual the backward kernel needs; then the
                           backward kernel's OWN rate is ramped (`ramp_tile_and_rate`,
                           below) -- its v0 rate is UNKNOWN and must never be assumed
                           from the forward's measured rate -- and the timed loop runs
                           ONLY the backward dispatch at the tile the ramp converged to.
  kernel_dw_v0          -- `backward_dw` ALONE, same ramp discipline.
  kernel_backward_v0_combined -- d_hidden + d_w run back to back (the real trainable-
                           path cost of the two current, UNFUSED v0 kernels -- each pays
                           its own full logit-regeneration cost; see
                           `macs_for_condition`'s docstring for why this is 4x base MACs,
                           not the ~2x a future fused kernel would achieve).

Each condition writes its own JSON artifact to `_artifacts/bench_backward_ladder/` the
instant it finishes (`mlx_train_perf.bench.artifacts.write_result` -- atomic tmp+rename),
and resume skips a condition whose artifact identity is fresh
(`mlx_train_perf.bench.artifacts.run_identity`/`result_is_fresh` -- `code_sha` there is
computed over `CODE_SHA_DEPS`, which already covers `core/kernel/launch.py` and
`core/kernel/source.py` (BOTH backward builders/launchers: `build_backward_dhidden_source`
+ `build_backward_dw_source`, `backward_dhidden` + `backward_dw`) plus `core/loss.py` and
`core/chunked.py` (the staged-vjp path) -- an interruption loses at most one condition's
run. A `LaunchBudgetError` refusal (either during ramp-driven setup or the final timed
dispatch) IS a result -- recorded, not a crash.

`--tiny` runs the SAME code paths at a cheap synthetic shape (n=256, v=2048, d=256) for
end-to-end verification, writing to `bench_<condition>_tiny.json` -- a DIFFERENT filename
than the production artifacts, so a tiny verification run can never collide with (or be
mistaken for) an actual production measurement.

Heavy GPU run at PRODUCTION shape (n=8192, V=151936, D=4096, bf16, seed 42) -- main
session only, subprocess-per-condition, ETA a few minutes per condition (backward kernels
pay their own ramp calibration on top of the timed dispatches). Pre-flight
`memory_pressure` before running; NEVER invoke a production-shape condition from an agent
session -- `--tiny` is the only mode safe to run outside the main session's heavy-run
protocol.
"""
import argparse
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from mlx_train_perf.bench.artifacts import result_is_fresh, run_identity, write_result
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.core.kernel.dispatch import select_variant
from mlx_train_perf.core.kernel.launch import (
    SAFETY_FACTOR,
    backward_dhidden,
    backward_dw,
    calibrated_rate,
    forward,
    next_probe_tile,
    probe_tile_for,
    sustain_reps,
)
from mlx_train_perf.core.loss import DenseHead, linear_cross_entropy
from mlx_train_perf.errors import LaunchBudgetError

N, V, D = 8192, 151936, 4096
TINY_N, TINY_V, TINY_D = 256, 2048, 256
DTYPE = mx.bfloat16
FORWARD_TILE = 8192          # fixed tile for the (unmeasured, setup-only) forward call
REPS = 3
CONDITIONS = (
    "staged_vjp_frozen", "staged_vjp_trainable",
    "kernel_dhidden_v0", "kernel_dw_v0", "kernel_backward_v0_combined",
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_backward_ladder"


def macs_for_condition(condition: str, *, n: int, v: int, d: int) -> int:
    """Pure MAC-accounting per condition -- total multiply-add-shaped work the reported
    rate (G MAC/s) is computed against. Never used for the launch-budget guard itself
    (each kernel call's own `check_budget` already accounts its own MAC cost internally).

    staged_vjp_frozen: forward (1x n*v*d) + the chunked backward's d_hidden-only cost
    (1x) -- the shared vjp's d_w branch exists but is never EVALUATED (argnums=(0,)
    only; see the module docstring), so it costs nothing here.
    staged_vjp_trainable: forward (1x) + chunked backward's d_hidden (1x) + d_w (1x).
    kernel_dhidden_v0 / kernel_dw_v0: EACH kernel alone regenerates logits tile-wise
    (1x) AND scatter-accumulates its own gradient (1x) -- 2x. No forward MACs counted
    here: the forward that produces (lse, tgt) runs once, unmeasured, outside the timed
    window (isolating the backward kernel's OWN cost is the point of these conditions).
    kernel_backward_v0_combined: both kernels run back to back, NOT yet fused -- there is
    no shared logit regeneration between them (each is its own separate Metal kernel with
    its own internal recompute), so this is genuinely 4x (2x + 2x) base MACs, not the ~2x
    a future FUSED kernel would achieve by sharing the recompute across both
    accumulations.
    """
    base = n * v * d
    if condition == "staged_vjp_frozen":
        return 2 * base
    if condition == "staged_vjp_trainable":
        return 3 * base
    if condition in ("kernel_dhidden_v0", "kernel_dw_v0"):
        return 2 * base
    if condition == "kernel_backward_v0_combined":
        return 4 * base
    raise ValueError(f"unknown condition {condition!r}")


def ramp_tile_and_rate(
    measure: Callable[[int], float], *, n: int, d: int, v: int,
    start_tile: int, max_stages: int = 3,
) -> tuple[int, float]:
    """Same ramp discipline as `launch.calibrate` (probe, climb while the projected tile
    keeps growing, sustain past the ramp to reach ramped GPU clocks, then a median-of-3
    timed measurement) but ALSO returns the tile the ramp converged to. `launch.calibrate`
    itself only returns the rate -- production dispatches for the ALREADY-CALIBRATED
    dense/quantized forward always run at a separately-known-safe fixed tile constant, but
    the backward kernels' v0 rate is UNKNOWN (never assume the forward's), so their
    production dispatch must run at the SAME tile this ramp actually measured, not a
    separately-guessed constant. Mirrors `launch.calibrate`'s internals exactly (same
    `next_probe_tile`/`sustain_reps` calls, same SAFETY_FACTOR application once at the
    end) -- the only difference is tracking and returning `tile` alongside the rate."""
    tile = start_tile
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(tile)
        raw_rate = n * tile * d / max(per_dispatch_s, 1e-9)
        candidate = next_probe_tile(rate_macs_per_s=SAFETY_FACTOR * raw_rate, n=n, d=d, v=v)
        if candidate <= tile:
            break
        tile = candidate
    for _rep in range(sustain_reps(per_dispatch_s=per_dispatch_s)):
        measure(tile)
    timings = [measure(tile) for _ in range(3)]
    median_s = statistics.median(timings)
    rate = n * tile * d / max(median_s, 1e-9)
    return tile, SAFETY_FACTOR * rate


def _timed(
    fn: Callable[[], None], *, reps: int,
) -> tuple[float, list[float], float, float]:
    """Warmup (pays Metal JIT) OUTSIDE the window; reset-peak semantics identical to
    `scripts/bench_quant_thresholds.py._timed_forward` and
    `mlx-train-perf-spike/gate_trainstep.py`'s own convention: snapshot active memory
    right before reset so `marginal_peak_gb` is the TRUE incremental allocation, not
    active+incremental. Returns (median_s, wall_s_all, active_before_gb, marginal_peak_gb).
    """
    fn()
    mx.clear_cache()
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    walls: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t0)
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3
    return statistics.median(walls), walls, active_before / 1024**3, marginal_peak_gb


# --- staged_vjp_* : the shared chunked-vjp path, differing only in value_and_grad's argnums


def _chunked_loss(hidden: mx.array, w: mx.array, targets: mx.array, *, trainable: bool) -> mx.array:
    head = DenseHead(weight=w, trainable=trainable)
    return linear_cross_entropy(hidden, head, targets, impl="chunked", reduction="mean")


def _staged_vjp_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, trainable: bool,
) -> Callable[[], None]:
    if trainable:
        def loss(h: mx.array, ww: mx.array, t: mx.array) -> mx.array:
            return _chunked_loss(h, ww, t, trainable=True)
        vag = mx.value_and_grad(loss, argnums=(0, 1))

        def fn() -> None:
            val, grads = vag(hidden, w, targets)
            mx.eval(val, *grads)
        return fn

    def loss_frozen(h: mx.array, ww: mx.array, t: mx.array) -> mx.array:
        return _chunked_loss(h, ww, t, trainable=False)
    vag_frozen = mx.value_and_grad(loss_frozen, argnums=(0,))

    def fn_frozen() -> None:
        val, g = vag_frozen(hidden, w, targets)
        # argnums=(0,) returns a BARE array, not a 1-tuple (mlx 0.31.2) -- same quirk
        # gate_trainstep.py's own `vag` wrapper handles.
        grads = g if isinstance(g, (tuple, list)) else (g,)
        mx.eval(val, *grads)
    return fn_frozen


# --- kernel_* : the v0 backward kernels, ramp-calibrated (their rate is UNKNOWN)


def _shared_forward_residual(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[mx.array, mx.array]:
    """UNMEASURED setup: the real (lse, tgt) residual the backward kernels need. Run
    once, outside the timed/peak window, exactly like a real training step would produce
    it once and reuse it for both d_hidden and d_w."""
    fwd_rate = calibrated_rate(row_tiles=row_tiles, dtype=hidden.dtype, n=n, d=d, v=v)
    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=FORWARD_TILE,
                       rate_macs_per_s=fwd_rate)
    mx.eval(lse, tgt)
    return lse, tgt


def _dhidden_measure(
    hidden: mx.array, w: mx.array, targets: mx.array, lse: mx.array, tgt: mx.array,
    ct: mx.array, *, row_tiles: int,
) -> Callable[[int], float]:
    warmed: set[int] = set()

    def measure(tile: int) -> float:
        if tile not in warmed:
            out = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                                   tile=tile, rate_macs_per_s=None)
            mx.eval(out)
            warmed.add(tile)
        t0 = time.perf_counter()
        out = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                               tile=tile, rate_macs_per_s=None)
        mx.eval(out)
        return time.perf_counter() - t0
    return measure


def _dw_measure(
    hidden: mx.array, w: mx.array, targets: mx.array, lse: mx.array, tgt: mx.array,
    ct: mx.array, *, row_tiles: int,
) -> Callable[[int], float]:
    warmed: set[int] = set()

    def measure(tile: int) -> float:
        if tile not in warmed:
            out = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                              tile=tile, rate_macs_per_s=None)
            mx.eval(out)
            warmed.add(tile)
        t0 = time.perf_counter()
        out = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                          tile=tile, rate_macs_per_s=None)
        mx.eval(out)
        return time.perf_counter() - t0
    return measure


def _kernel_dhidden_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[Callable[[], None], dict[str, object]]:
    lse, tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)
    start_tile = min(probe_tile_for(n=n, d=d), v)
    tile, rate = ramp_tile_and_rate(
        _dhidden_measure(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
        n=n, d=d, v=v, start_tile=start_tile,
    )

    def fn() -> None:
        out = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                               tile=tile, rate_macs_per_s=rate)
        mx.eval(out)
    return fn, {"tile": tile, "calibrated_rate_macs_per_s": round(rate, 1)}


def _kernel_dw_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[Callable[[], None], dict[str, object]]:
    lse, tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)
    start_tile = min(probe_tile_for(n=n, d=d), v)
    tile, rate = ramp_tile_and_rate(
        _dw_measure(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
        n=n, d=d, v=v, start_tile=start_tile,
    )

    def fn() -> None:
        out = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                          tile=tile, rate_macs_per_s=rate)
        mx.eval(out)
    return fn, {"tile": tile, "calibrated_rate_macs_per_s": round(rate, 1)}


def _kernel_combined_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[Callable[[], None], dict[str, object]]:
    # ONE shared forward residual for both kernels -- the real trainable step computes
    # (lse, tgt) once and reuses it for both accumulations.
    lse, tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)
    start_tile = min(probe_tile_for(n=n, d=d), v)
    dh_tile, dh_rate = ramp_tile_and_rate(
        _dhidden_measure(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
        n=n, d=d, v=v, start_tile=start_tile,
    )
    dw_tile, dw_rate = ramp_tile_and_rate(
        _dw_measure(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
        n=n, d=d, v=v, start_tile=start_tile,
    )

    def fn() -> None:
        # Both dispatches issued before a SINGLE joint eval -- lets MLX's scheduler see
        # both lazy graphs together, rather than artificially serializing via two
        # separate eval calls (which would forbid any overlap the scheduler could find).
        d_hidden = backward_dhidden(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                                    tile=dh_tile, rate_macs_per_s=dh_rate)
        d_w = backward_dw(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                          tile=dw_tile, rate_macs_per_s=dw_rate)
        mx.eval(d_hidden, d_w)
    return fn, {
        "dhidden_tile": dh_tile, "dhidden_rate_macs_per_s": round(dh_rate, 1),
        "dw_tile": dw_tile, "dw_rate_macs_per_s": round(dw_rate, 1),
    }


def run_condition(condition: str, *, tiny: bool) -> None:
    install_guardrails()
    n, v, d = (TINY_N, TINY_V, TINY_D) if tiny else (N, V, D)
    row_tiles = select_variant(n).row_tiles
    ident = run_identity(
        experiment="bench_backward_ladder", condition=condition,
        n=n, v=v, d=d, dtype=str(DTYPE), tiny=tiny,
    )
    out = RESULTS / f"bench_{condition}{'_tiny' if tiny else ''}.json"
    if result_is_fresh(out, ident):
        print(f"skip (fresh): {out.name}")
        return

    mx.random.seed(42)
    hidden = mx.random.normal((n, d)).astype(DTYPE)
    w = (mx.random.normal((v, d)) * 0.02).astype(DTYPE)
    targets = mx.random.randint(0, v, (n,))
    mx.eval(hidden, w, targets)

    try:
        if condition == "staged_vjp_frozen":
            fn, extra = _staged_vjp_fn(hidden, w, targets, trainable=False), {}
        elif condition == "staged_vjp_trainable":
            fn, extra = _staged_vjp_fn(hidden, w, targets, trainable=True), {}
        elif condition == "kernel_dhidden_v0":
            fn, extra = _kernel_dhidden_fn(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
        elif condition == "kernel_dw_v0":
            fn, extra = _kernel_dw_fn(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
        else:
            fn, extra = _kernel_combined_fn(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    except LaunchBudgetError as exc:
        # A guard refusal during ramp-driven SETUP is still a valid, recorded result --
        # it means even the smallest safe tile the ramp could find still refuses.
        write_result(out, ident, "refused", error=str(exc))
        print(f"{condition}: REFUSED during setup/calibration -- recorded as a result")
        return

    try:
        median_s, walls, active_before_gb, marginal_peak_gb = _timed(fn, reps=REPS)
    except LaunchBudgetError as exc:
        write_result(out, ident, "refused", error=str(exc), **extra)
        print(f"{condition}: REFUSED by launch-budget guard -- recorded as a result")
        return

    macs = macs_for_condition(condition, n=n, v=v, d=d)
    g_mac_per_s = macs / median_s / 1e9
    write_result(
        out, ident, "ok",
        wall_s_median=round(median_s, 4),
        wall_s_all=[round(x, 4) for x in walls],
        g_mac_per_s=round(g_mac_per_s, 1),
        active_before_gb=round(active_before_gb, 3),
        marginal_peak_gb=round(marginal_peak_gb, 3),
        total_peak_gb=round(active_before_gb + marginal_peak_gb, 3),
        row_tiles=row_tiles,
        **extra,
    )
    print(f"{condition}: median={median_s:.4f}s rate={g_mac_per_s:.1f} G MAC/s "
          f"marginal_peak={marginal_peak_gb:.3f} GB {extra}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--tiny", action="store_true",
                    help="cheap synthetic shape (n=256, v=2048, d=256) for end-to-end "
                        "verification -- never the production measurement")
    args = ap.parse_args(argv)
    if args.condition:
        run_condition(args.condition, tiny=args.tiny)
        return 0
    n = TINY_N if args.tiny else N
    v = TINY_V if args.tiny else V
    d = TINY_D if args.tiny else D
    for condition in CONDITIONS:
        ident = run_identity(
            experiment="bench_backward_ladder", condition=condition,
            n=n, v=v, d=d, dtype=str(DTYPE), tiny=args.tiny,
        )
        out = RESULTS / f"bench_{condition}{'_tiny' if args.tiny else ''}.json"
        if result_is_fresh(out, ident):
            print(f"skip {condition} (fresh)")
            continue
        cmd = [sys.executable, __file__, "--condition", condition]
        if args.tiny:
            cmd.append("--tiny")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
