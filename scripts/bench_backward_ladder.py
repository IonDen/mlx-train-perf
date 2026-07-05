"""Per-rung bench for the backward ladder (Task 16b, step 4/5 — the STOP-table bench).

Conditions (subprocess-per-condition -- own MLX allocator state, no cross-condition
buffer retention):

  chunked_fallback_frozen / chunked_fallback_trainable
                        -- DenseHead through `linear_cross_entropy`'s `impl="chunked"`
                           custom_function: the FULLY-CHUNKED fallback (chunked forward
                           AND chunked vjp — no fused kernel anywhere in this path).
                           `mx.value_and_grad` w.r.t. hidden ONLY (argnums=(0,)) for
                           frozen, w.r.t. (hidden, w) (argnums=(0, 1)) for trainable --
                           `core.chunked.make_chunked_dense`'s vjp unconditionally
                           computes a d_w cotangent too (it does not branch on
                           `head.trainable`), but since nothing downstream ever
                           references or `mx.eval`s that cotangent under argnums=(0,),
                           MLX's lazy scheduler never executes that branch -- confirmed
                           empirically (a throwaway probe measured a ~3.9x wall-time gap
                           between argnums=(0,) and argnums=(0,1) against an identical
                           vjp body with a deliberately expensive d_w branch).
                           RENAMED from `staged_vjp_frozen`/`staged_vjp_trainable`
                           (mislabel found by the production run: `impl="chunked"` is
                           NOT the shipping staged mechanism -- it is the chunked
                           fallback, whose fully-streamed FORWARD reads 8.5/7.9 GB, not
                           the gate's 2.47 GB, precisely because it never touches the
                           fused kernel forward at all).
  staged_kernel_frozen / staged_kernel_trainable
                        -- the SAME `_staged_fn` wiring, `impl="kernel"` instead: the
                           REAL shipping staged mechanism (fused kernel forward +
                           chunked vjp backward, via `loss.py`'s `_kernel_nll_dense`).
                           This is the actual incumbent the fused MMA backward rung
                           competes against. A dedicated honesty-guard test (see
                           tests/test_bench_backward_ladder.py) proves this resolves to
                           `Resolution.impl == "kernel"`, never silently falling back to
                           chunked the way the ORIGINAL staged_vjp_* mislabel did.
  kernel_dhidden_v0     -- `backward_dhidden` (the v0, zero-reuse-scalar kernel) ALONE.
  kernel_dhidden_mma    -- `backward_dhidden_mma` (the fused two-GEMM MMA kernel,
                           step 4, committed + review-approved) ALONE, same ramp
                           discipline as kernel_dhidden_v0. THIS is the new number the
                           STOP table is built from -- the direct MMA-vs-chunked
                           backward comparison partner is `chunked_dhidden_alone`,
                           below (both share one kernel forward's saved lse/tgt).
  kernel_dw_v0          -- `backward_dw` ALONE, same ramp discipline.
  kernel_backward_v0_combined -- d_hidden + d_w (v0 kernels) run back to back (the real
                           trainable-path cost of the two current, UNFUSED v0 kernels --
                           each pays its own full logit-regeneration cost; see
                           `macs_for_condition`'s docstring for why this is 4x base MACs,
                           not the ~2x a future fused kernel would achieve). Kept as-is:
                           its refusals are the v0 baseline row for the STOP table.
  chunked_dhidden_alone -- `core.chunked.chunked_backward` (the proven chunked backward,
                           `head_trainable=False`) ALONE, on the SAME saved (lse, tgt)
                           residual a kernel forward produced -- the direct wall+memory
                           comparison partner for `kernel_dhidden_mma`. No launch guard
                           (pure MLX, no `check_budget` anywhere on this path).

`kernel_dhidden_v0`/`kernel_dhidden_mma`/`kernel_dw_v0`/`kernel_backward_v0_combined`:
`forward` runs once, UNMEASURED (setup only, outside the timed/peak window) to produce
the real (lse, tgt) residual the backward kernel needs; the backward kernel's OWN rate is
then ramped (`ramp_tile_and_rate`, below) -- its rate is UNKNOWN and must never be
assumed from the forward's -- and the timed loop runs ONLY the backward dispatch at the
tile the ramp converged to. `chunked_dhidden_alone` shares this same "one forward, then
time only the backward" discipline, just with no ramp (pure MLX has no launch budget).

Each condition writes its own JSON artifact to `_artifacts/bench_backward_ladder/` the
instant it finishes (`mlx_train_perf.bench.artifacts.write_result` -- atomic tmp+rename),
and resume skips a condition whose artifact identity is fresh
(`mlx_train_perf.bench.artifacts.run_identity`/`result_is_fresh` -- `code_sha` there is
computed over `CODE_SHA_DEPS`, which already covers `core/kernel/launch.py` and
`core/kernel/source.py` (all THREE backward builders/launchers: `build_backward_dhidden_
source` + `build_backward_dhidden_mma_source` + `build_backward_dw_source`,
`backward_dhidden` + `backward_dhidden_mma` + `backward_dw`) plus `core/loss.py` and
`core/chunked.py` (the staged/chunked-vjp paths). `CODE_SHA_DEPS` deliberately excludes ad
hoc bench scripts under `scripts/` though, so this script's identity ALSO carries its own
`script_sha()` (same convention `ground_truth_atomic_outputs.py` established) -- without
it, an edit to THIS script's OWN measurement logic would not invalidate a stale artifact.
An interruption loses at most one condition's run. A `LaunchBudgetError` refusal (either
during ramp-driven setup or the final timed dispatch) IS a result -- recorded, not a
crash.

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
import hashlib
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from mlx_train_perf.bench.artifacts import result_is_fresh, run_identity, write_result
from mlx_train_perf.core.chunked import chunked_backward
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.core.kernel.dispatch import select_variant
from mlx_train_perf.core.kernel.launch import (
    SAFETY_FACTOR,
    backward_dhidden,
    backward_dhidden_mma,
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
FORWARD_TILE = 8192          # fixed tile for the (unmeasured, setup-only) forward call;
                             # also reused as chunked_backward's chunk_size (pure MLX --
                             # no launch-budget concept, matches loss.py's own default).
REPS = 3

# The two `impl` values `_staged_fn` can be built with -- named constants, not inline
# string literals, specifically so the honesty-guard test in
# tests/test_bench_backward_ladder.py imports and checks THESE (the actual wiring),
# rather than asserting an independent literal that could drift from the real dispatch.
CHUNKED_FALLBACK_IMPL = "chunked"
STAGED_KERNEL_IMPL = "kernel"

CONDITIONS = (
    "chunked_fallback_frozen", "chunked_fallback_trainable",
    "staged_kernel_frozen", "staged_kernel_trainable",
    "kernel_dhidden_v0", "kernel_dhidden_mma", "kernel_dw_v0",
    "kernel_backward_v0_combined", "chunked_dhidden_alone",
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_backward_ladder"


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes. `run_identity`'s `code_sha` (computed over
    `mlx_train_perf.bench.artifacts.CODE_SHA_DEPS`) deliberately excludes ad hoc bench
    scripts under `scripts/` -- same convention `scripts/ground_truth_atomic_outputs.py`
    already established its own `script_sha()` for. Without this, an edit to THIS
    script's OWN measurement/accounting logic (e.g. the `ramp_tile_and_rate` MAC-
    accounting fix) would silently NOT invalidate a previously-written artifact, since
    `code_sha` alone never changes for a script-only edit -- confirmed the hard way: a
    `--tiny` re-verification after fixing `ramp_tile_and_rate` read every prior tiny
    artifact as still \"fresh\" and skipped re-measuring, because this field did not
    exist yet at that point."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def macs_for_condition(condition: str, *, n: int, v: int, d: int) -> int:
    """Pure MAC-accounting per condition -- total multiply-add-shaped work the reported
    rate (G MAC/s) is computed against. Never used for the launch-budget guard itself
    (each kernel call's own `check_budget` already accounts its own MAC cost internally).

    chunked_fallback_frozen / staged_kernel_frozen: forward (1x n*v*d) + the chunked
    backward's d_hidden-only cost (1x) -- the shared vjp's d_w branch exists but is
    never EVALUATED (argnums=(0,) only; see the module docstring), so it costs nothing
    here. The forward IMPLEMENTATION differs between the two (chunked-streamed vs the
    fused kernel), but the MAC COUNT this model charges is the same either way -- both
    still do one N*V*D-shaped forward pass.
    chunked_fallback_trainable / staged_kernel_trainable: forward (1x) + chunked
    backward's d_hidden (1x) + d_w (1x) = 3x.
    kernel_dhidden_v0 / kernel_dhidden_mma / kernel_dw_v0 / chunked_dhidden_alone: EACH
    backward alone regenerates logits tile-wise (1x) AND accumulates its own gradient
    (1x) -- 2x. No forward MACs counted here: the forward that produces (lse, tgt) runs
    once, unmeasured, outside the timed window (isolating the backward's OWN cost is the
    point of these conditions) -- this applies uniformly whether the backward is a v0
    kernel, the MMA kernel, or the pure-MLX chunked backward.
    kernel_backward_v0_combined: both v0 kernels run back to back, NOT yet fused --
    there is no shared logit regeneration between them (each is its own separate Metal
    kernel with its own internal recompute), so this is genuinely 4x (2x + 2x) base
    MACs, not the ~2x a future FUSED kernel would achieve by sharing the recompute
    across both accumulations.
    """
    base = n * v * d
    if condition in ("chunked_fallback_frozen", "staged_kernel_frozen"):
        return 2 * base
    if condition in ("chunked_fallback_trainable", "staged_kernel_trainable"):
        return 3 * base
    if condition in (
        "kernel_dhidden_v0", "kernel_dhidden_mma", "kernel_dw_v0", "chunked_dhidden_alone",
    ):
        return 2 * base
    if condition == "kernel_backward_v0_combined":
        return 4 * base
    raise ValueError(f"unknown condition {condition!r}")


def ramp_tile_and_rate(
    measure: Callable[[int], float], *, n: int, d: int, v: int,
    start_tile: int, max_stages: int = 3,
) -> tuple[int, float]:
    """Same ramp SHAPE as `launch.calibrate` (probe, climb while the projected tile keeps
    growing, sustain past the ramp to reach ramped GPU clocks, then a median-of-3 timed
    measurement) but with DIFFERENT MAC accounting, for a reason that must not be
    flattened into "mirrors calibrate() exactly" -- doing so once produced a real bug
    (fixed here; see below). Also returns the tile the ramp converged to (`calibrate()`
    itself only returns the rate) since the backward kernels' rate is UNKNOWN (never
    assume the forward's), so their production dispatch must run at the SAME tile this
    ramp actually measured, not a separately-guessed constant.

    THE DATA-SIZING DIFFERENCE THAT DRIVES THE ACCOUNTING DIFFERENCE: `calibrated_rate`'s
    own `measure(tile)` closure builds a probe `w`/`targets` pair SIZED TO `tile` (`w`'s
    own vocab dim equals `tile`), so `forward(..., tile=tile)`'s internal
    `for v0 in range(0, v, tile)` loop -- where `v = w.shape[0] == tile` there -- runs
    EXACTLY ONE iteration: `per_dispatch_s` genuinely times ONE dispatch of `n*tile*d`
    MACs, and `n*tile*d/per_dispatch_s` is a correct per-dispatch rate.

    THIS caller's `measure(tile)` closures (see `_dhidden_measure`/`_dw_measure`) reuse
    the FULL PRODUCTION `w`/`targets` (shape `(v, d)` -- deliberately, since these bench
    conditions measure the REAL backward kernel at REAL production scale, not a
    tile-sized synthetic probe). Calling `backward_dhidden`/`backward_dhidden_mma`/
    `backward_dw(..., tile=tile)` against a full-size `w` makes their OWN internal
    `for v0 in range(0, v, tile)` loop run `ceil(v / tile)` iterations -- `measure(tile)`
    times the WHOLE CHAIN across every vocab column, not one dispatch. A prior version of
    this function kept `calibrate()`'s `n*tile*d/per_dispatch_s` formula anyway, which
    under-reports the rate by approximately `v/tile` (at tile=128 against V=151936,
    ~1,187x) -- the ramp collapsed to a "0 G MAC/s" rate and `check_budget` refused every
    kernel condition with a fictional multi-hundred-second projection (confirmed against
    the production run's recorded refusal artifacts). Since every dispatch in the chain
    covers a DIFFERENT, disjoint slice of `v` and the chain's total MAC count is therefore
    ALWAYS `n*v*d` regardless of how many dispatches the chain took or how it was tiled,
    the correct rate is `n*v*d/elapsed` (used for BOTH the ramp-stage rate below and the
    final median-of-3 rate) -- this is genuinely per-dispatch-comparable throughput:
    `check_budget`'s own `n*tile*d/rate` projection then reduces to
    `elapsed * (tile/v)`, i.e. the chain's measured average time per one `tile`-sized
    dispatch, exactly the quantity a launch-budget guard needs.

    `sustain_reps(per_dispatch_s=...)` is still fed the CHAIN's elapsed time (not a
    single dispatch's) here -- that is fine, not a second bug: `sustain_reps`'s whole
    purpose is ensuring enough CONTINUOUS GPU load to reach ramped clocks before the
    timed measurement, and a full chain (especially at small tiles, where it is many
    dispatches back to back) trivially already provides that much continuous load, so
    `ceil(0.75 / chain_elapsed)` reliably floors at 1 extra rep rather than ballooning."""
    tile = start_tile
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(tile)
        raw_rate = n * v * d / max(per_dispatch_s, 1e-9)
        candidate = next_probe_tile(rate_macs_per_s=SAFETY_FACTOR * raw_rate, n=n, d=d, v=v)
        if candidate <= tile:
            break
        tile = candidate
    for _rep in range(sustain_reps(per_dispatch_s=per_dispatch_s)):
        measure(tile)
    timings = [measure(tile) for _ in range(3)]
    median_s = statistics.median(timings)
    rate = n * v * d / max(median_s, 1e-9)
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


# --- chunked_fallback_* / staged_kernel_* : the SAME wiring, differing only in
# `linear_cross_entropy`'s `impl` and in value_and_grad's argnums.


def _staged_loss(
    hidden: mx.array, w: mx.array, targets: mx.array, *, impl: str, trainable: bool,
) -> mx.array:
    head = DenseHead(weight=w, trainable=trainable)
    return linear_cross_entropy(hidden, head, targets, impl=impl, reduction="mean")


def _staged_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, impl: str, trainable: bool,
) -> Callable[[], None]:
    if trainable:
        def loss(h: mx.array, ww: mx.array, t: mx.array) -> mx.array:
            return _staged_loss(h, ww, t, impl=impl, trainable=True)
        vag = mx.value_and_grad(loss, argnums=(0, 1))

        def fn() -> None:
            val, grads = vag(hidden, w, targets)
            mx.eval(val, *grads)
        return fn

    def loss_frozen(h: mx.array, ww: mx.array, t: mx.array) -> mx.array:
        return _staged_loss(h, ww, t, impl=impl, trainable=False)
    vag_frozen = mx.value_and_grad(loss_frozen, argnums=(0,))

    def fn_frozen() -> None:
        val, g = vag_frozen(hidden, w, targets)
        # argnums=(0,) returns a BARE array, not a 1-tuple (mlx 0.31.2) -- same quirk
        # gate_trainstep.py's own `vag` wrapper handles.
        grads = g if isinstance(g, (tuple, list)) else (g,)
        mx.eval(val, *grads)
    return fn_frozen


# --- kernel_* / chunked_dhidden_alone : isolated backward passes, ramp-calibrated where
# a launch guard applies (their rate is UNKNOWN).


def _shared_forward_residual(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[mx.array, mx.array]:
    """UNMEASURED setup: the real (lse, tgt) residual the backward passes need. Run
    once, outside the timed/peak window, exactly like a real training step would produce
    it once and reuse it for both d_hidden and d_w (and, for `chunked_dhidden_alone`, the
    chunked backward)."""
    fwd_rate = calibrated_rate(row_tiles=row_tiles, dtype=hidden.dtype, n=n, d=d, v=v)
    lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=FORWARD_TILE,
                       rate_macs_per_s=fwd_rate)
    mx.eval(lse, tgt)
    return lse, tgt


def _dhidden_measure(
    backward_fn: Callable[..., mx.array],
    hidden: mx.array, w: mx.array, targets: mx.array, lse: mx.array, tgt: mx.array,
    ct: mx.array, *, row_tiles: int,
) -> Callable[[int], float]:
    """`backward_fn` is `backward_dhidden` or `backward_dhidden_mma` -- both share the
    IDENTICAL (hidden, w, targets, lse, tgt, cotangent, *, row_tiles, tile,
    rate_macs_per_s) -> d_hidden contract (`backward_dhidden_mma` is a documented
    drop-in replacement for `backward_dhidden`), so one measure-closure factory serves
    both."""
    warmed: set[int] = set()

    def measure(tile: int) -> float:
        if tile not in warmed:
            out = backward_fn(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
                              tile=tile, rate_macs_per_s=None)
            mx.eval(out)
            warmed.add(tile)
        t0 = time.perf_counter()
        out = backward_fn(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
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


def _kernel_dhidden_generic_fn(
    backward_fn: Callable[..., mx.array],
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[Callable[[], None], dict[str, object]]:
    """Shared builder for `kernel_dhidden_v0` (`backward_fn=backward_dhidden`) and
    `kernel_dhidden_mma` (`backward_fn=backward_dhidden_mma`) -- identical setup/ramp/
    dispatch shape, differing only in which backward kernel is timed."""
    lse, tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)
    start_tile = min(probe_tile_for(n=n, d=d), v)
    tile, rate = ramp_tile_and_rate(
        _dhidden_measure(backward_fn, hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
        n=n, d=d, v=v, start_tile=start_tile,
    )

    def fn() -> None:
        out = backward_fn(hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles,
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
    # (lse, tgt) once and reuses it for both accumulations. Still the v0/v0 pairing
    # (backward_dhidden + backward_dw) -- this condition's refusals are the v0 baseline
    # row for the STOP table, unchanged by the MMA rung.
    lse, tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)
    start_tile = min(probe_tile_for(n=n, d=d), v)
    dh_tile, dh_rate = ramp_tile_and_rate(
        _dhidden_measure(backward_dhidden, hidden, w, targets, lse, tgt, ct, row_tiles=row_tiles),
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


def _chunked_dhidden_alone_fn(
    hidden: mx.array, w: mx.array, targets: mx.array, *, row_tiles: int, n: int, d: int, v: int,
) -> tuple[Callable[[], None], dict[str, object]]:
    """The proven chunked backward (`core.chunked.chunked_backward`, `head_trainable=
    False` -- the frozen/d_hidden-only path), timed ALONE on the SAME saved (lse, tgt)
    residual a kernel forward produced. Direct wall+memory comparison partner for
    `kernel_dhidden_mma`: both share one kernel forward (built OUTSIDE the timed window
    here too), so the two conditions differ ONLY in which backward computes d_hidden.
    No launch guard on this path -- `chunked_backward` is pure MLX, no `check_budget`
    call anywhere in it (matches `_kernel_nll_dense`'s own frozen-path vjp in `loss.py`,
    which calls this exact function with this exact `mm`/`w_chunk` closure shape --
    Python-side `w[v0:v1]` slicing is the ESTABLISHED, already-shipped idiom for the
    pure-MLX chunked path; the "full buffers, no Python-side slices" rule is specific to
    the METAL KERNEL chained launches, not this pure-MLX one)."""
    lse, _tgt = _shared_forward_residual(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
    ct = mx.full((n,), 1.0 / n, dtype=mx.float32)
    mx.eval(ct)

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    def w_chunk(v0: int, v1: int) -> mx.array:
        return w[v0:v1]

    def fn() -> None:
        d_hidden, _ = chunked_backward(
            hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk, targets=targets, lse=lse,
            cotangent=ct, v=v, chunk_size=FORWARD_TILE, head_trainable=False,
        )
        mx.eval(d_hidden)
    return fn, {}


def run_condition(condition: str, *, tiny: bool) -> None:
    install_guardrails()
    n, v, d = (TINY_N, TINY_V, TINY_D) if tiny else (N, V, D)
    row_tiles = select_variant(n).row_tiles
    ident = run_identity(
        experiment="bench_backward_ladder", condition=condition,
        n=n, v=v, d=d, dtype=str(DTYPE), tiny=tiny, script_sha=script_sha(),
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
        if condition == "chunked_fallback_frozen":
            fn, extra = _staged_fn(hidden, w, targets, impl=CHUNKED_FALLBACK_IMPL,
                                   trainable=False), {}
        elif condition == "chunked_fallback_trainable":
            fn, extra = _staged_fn(hidden, w, targets, impl=CHUNKED_FALLBACK_IMPL,
                                   trainable=True), {}
        elif condition == "staged_kernel_frozen":
            fn, extra = _staged_fn(hidden, w, targets, impl=STAGED_KERNEL_IMPL,
                                   trainable=False), {}
        elif condition == "staged_kernel_trainable":
            fn, extra = _staged_fn(hidden, w, targets, impl=STAGED_KERNEL_IMPL,
                                   trainable=True), {}
        elif condition == "kernel_dhidden_v0":
            fn, extra = _kernel_dhidden_generic_fn(backward_dhidden, hidden, w, targets,
                                                   row_tiles=row_tiles, n=n, d=d, v=v)
        elif condition == "kernel_dhidden_mma":
            fn, extra = _kernel_dhidden_generic_fn(backward_dhidden_mma, hidden, w, targets,
                                                   row_tiles=row_tiles, n=n, d=d, v=v)
        elif condition == "kernel_dw_v0":
            fn, extra = _kernel_dw_fn(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
        elif condition == "kernel_backward_v0_combined":
            fn, extra = _kernel_combined_fn(hidden, w, targets, row_tiles=row_tiles, n=n, d=d, v=v)
        else:  # chunked_dhidden_alone
            fn, extra = _chunked_dhidden_alone_fn(hidden, w, targets, row_tiles=row_tiles,
                                                  n=n, d=d, v=v)
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
            n=n, v=v, d=d, dtype=str(DTYPE), tiny=args.tiny, script_sha=script_sha(),
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
