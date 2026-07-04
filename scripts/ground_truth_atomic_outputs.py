"""Ground-truth experiment (mlx-train-perf-0008, Task 16b Step 1): does
`mx.fast.metal_kernel`'s `atomic_outputs=True` float add work correctly under the JIT, and
what does it cost against a split-K-style partials buffer + reduction (steel's pattern),
at a shape representative of the fused backward kernel's `d_w = P^T @ H` cross-row-block
accumulation?

Ground-truthed 2026-07-04 (mlx 0.31.2, M1 Max, macOS 26.5.1): the auto-generated signature
for `atomic_outputs=True` is Metal's own `device atomic<T>*` (confirmed via a `verbose=True`
probe -- NOT MLX's `mlx_atomic<T>` wrapper from `atomic.h`, which is never auto-included for
custom kernels). A plain `out[elem] = val` assignment fails to compile against `atomic<T>`
(Metal deletes the plain assignment operator for `atomic<T>`); the correct accumulate
primitive is `atomic_fetch_add_explicit(&out[elem], val, metal::memory_order_relaxed)`,
which compiles and runs with NO extra header (`<metal_atomic>` is implicitly available).

Both mechanisms share one EXACT reference: contribution(rb, elem) = (rb + 1) * (elem + 1),
so expected[elem] = (elem + 1) * row_blocks*(row_blocks + 1)/2 -- an integer for any shape
this script uses, chosen so a dropped atomic add is detectable by EXACT equality, never
masked by float rounding (`fits_float32_exact` refuses any shape that would exceed the
float32 exact-integer ceiling, 2**24).

Modes:
  --correctness  Tiny shape (default tile=16, d=16, 200 row-blocks -- 51,200 total
                 accumulate ops across both mechanisms combined). Runs both mechanisms,
                 asserts the partials-buffer path (the known-correct oracle) matches the
                 exact reference, and RECORDS (never asserts) whatever `atomic_outputs`
                 produces -- a compile failure or a wrong result is a valid, precisely
                 logged finding, not a bug to fight. Always re-runs (no fresh-skip): the
                 point of this mode is the printed transcript, not resumability.
  --cost         Parameterized shape, representative of the d_w accumulation pattern
                 (default n=8192 context, tile=2048, d=4096, 16 partials/splits). Timed
                 median-of-3 after warmup for EACH mechanism, one artifact per mechanism,
                 resumed by skipping a fresh one. NEVER invoke this mode from an agent
                 session -- the controller runs it on the main thread per the workspace's
                 heavy-run discipline (main-session-only, ETA'd, serialized).

Watchdog-safety estimate at the --cost DEFAULT shape (why every dispatch here is trivially
under the 1 s/dispatch guard `core/kernel/launch.py:check_budget` enforces for the real
kernels): each mechanism moves splits * tile * d = 16 * 2048 * 4096 = 134,217,728
accumulate-shaped operations total (one atomic fetch_add, or one non-atomic partials
write, per (split, element) pair) -- FAR less arithmetic per element than the dense
forward kernel's `d`-deep dot product inner loop. Modeling this pessimistically against
`calibrated_rate`'s own measured ~2400 G MAC/s dense-GEMM-class rate (a rate for work that
does `d`=4096 multiply-adds per output element, not the ~1 op/element this experiment
does) still gives 134.2e6 / 2.4e12 ~= 56 microseconds per mechanism -- four orders of
magnitude under budget even if the true throughput of a scatter/atomic-add dispatch
undershoots that deliberately-heavy proxy rate by 100x.
"""
import argparse
import functools
import hashlib
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx

from mlx_train_perf.bench.artifacts import result_is_fresh, run_identity, write_result
from mlx_train_perf.core.guards import install_guardrails

_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPT_PATH.parent.parent / "_artifacts" / "ground_truth_atomic_outputs"
REPS = 3

# The installed mlx 0.31.2 stub types mx.fast.metal_kernel's return as `object` (nanobind
# gives it no more specific type); it is documented + actually a callable kernel invoker
# (same cast convention as mlx_train_perf.core.kernel.launch._MetalKernel).
_MetalKernel = Callable[..., list[mx.array]]

# Largest N such that every non-negative integer <= N is exactly representable in float32
# (24-bit mantissa) -- the ceiling `fits_float32_exact` guards against.
FLOAT32_EXACT_INT_CEILING = 1 << 24

_CORRECTNESS_DEFAULTS: dict[str, int] = {"tile": 16, "d": 16, "splits": 200}
_COST_DEFAULTS: dict[str, int] = {"tile": 2048, "d": 4096, "splits": 16}

# atomic_outputs=True mechanism: many threadgroups (grid.y = row_blocks) atomically
# fetch-add a per-(row-block, element) contribution into a SHARED (n_elem,) fp32 output --
# the exact cross-row-block contention pattern the fused backward kernel's `d_w`
# accumulation needs. `init_value=0.0` zero-fills the atomic before any thread runs.
_ATOMIC_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint rb = thread_position_in_grid.y;
    T weight = (T)(elem + 1);
    T contribution = (T)(rb + 1) * weight;
    atomic_fetch_add_explicit(&out[elem], contribution, metal::memory_order_relaxed);
"""

# Split-K / partials-buffer mechanism: the SAME contribution, but each (row-block, elem)
# pair writes its own PRIVATE slot of a (row_blocks, n_elem) buffer -- no contention, no
# atomics. `dims[0]` carries n_elem so the kernel can compute the flat row-major offset
# (same house convention as `core/kernel/launch.py`'s `offs` shape-parameter input).
_PARTIALS_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint rb = thread_position_in_grid.y;
    uint n_elem = dims[0];
    T weight = (T)(elem + 1);
    T contribution = (T)(rb + 1) * weight;
    out[rb * n_elem + elem] = contribution;
"""


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes. `run_identity`'s `CODE_SHA_DEPS` is the
    bench harness's own dependency list and deliberately excludes ad hoc experiment
    scripts (per the task brief) -- this is a separate identity field so an edit to this
    script still invalidates its own prior artifacts, without touching that list."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def expected_totals(*, n_elem: int, row_blocks: int) -> list[float]:
    """Exact closed-form reference shared by both mechanisms:
    contribution(rb, elem) = (rb + 1) * (elem + 1), summed over rb in [0, row_blocks) ->
    expected[elem] = (elem + 1) * row_blocks*(row_blocks + 1)/2. Every value is an exact
    integer (as a Python int, then converted to float for comparison against `mx.array`
    contents) -- `fits_float32_exact` is the caller's job to check beforehand."""
    triangular = row_blocks * (row_blocks + 1) // 2
    return [float((elem + 1) * triangular) for elem in range(n_elem)]


def max_expected_value(*, n_elem: int, row_blocks: int) -> int:
    """The largest single value `expected_totals` can produce for this shape (the last
    element, weight == n_elem) -- what `fits_float32_exact` checks against the ceiling."""
    triangular = row_blocks * (row_blocks + 1) // 2
    return n_elem * triangular


def fits_float32_exact(*, n_elem: int, row_blocks: int) -> bool:
    """True iff every expected value for this shape is STRICTLY below the float32
    exact-integer ceiling -- the property that makes a dropped atomic add detectable by
    exact equality rather than maskable by float rounding."""
    return max_expected_value(n_elem=n_elem, row_blocks=row_blocks) < FLOAT32_EXACT_INT_CEILING


def resolve_shape(
    *, mode: str, tile: int | None, d: int | None, splits: int | None,
) -> tuple[int, int, int]:
    """Pure CLI-defaulting logic: `--correctness` defaults to a tiny shape (well under a
    second of GPU time even with full cross-row-block contention on every element);
    `--cost` defaults to the representative d_w accumulation shape from the task brief
    (tile=2048, d=4096, 16 partials/splits). An explicitly passed value always wins over
    the mode's default."""
    defaults = _CORRECTNESS_DEFAULTS if mode == "correctness" else _COST_DEFAULTS
    return (
        tile if tile is not None else defaults["tile"],
        d if d is not None else defaults["d"],
        splits if splits is not None else defaults["splits"],
    )


@functools.cache
def _atomic_kernel() -> _MetalKernel:
    return cast(_MetalKernel, mx.fast.metal_kernel(
        name="gt_atomic_add",
        input_names=["dummy"],
        output_names=["out"],
        source=_ATOMIC_SOURCE,
        atomic_outputs=True,
    ))


@functools.cache
def _partials_kernel() -> _MetalKernel:
    return cast(_MetalKernel, mx.fast.metal_kernel(
        name="gt_partials_write",
        input_names=["dims"],
        output_names=["out"],
        source=_PARTIALS_SOURCE,
    ))


def run_atomic(*, tile: int, d: int, row_blocks: int, threads_per_group: int) -> mx.array:
    """Dispatches the atomic_outputs=True mechanism: grid.y = row_blocks threadgroups (one
    per row-block, the SAME dimension every one of them contends on), grid.x tiled across
    (possibly several) threadgroups of `threads_per_group` lanes each covering the
    (tile, d) output flattened to n_elem."""
    n_elem = tile * d
    tg_x = min(threads_per_group, n_elem)
    kernel = _atomic_kernel()
    dummy = mx.zeros((1,), dtype=mx.float32)
    (out,) = kernel(
        inputs=[dummy],
        template=[("T", mx.float32)],
        grid=(n_elem, row_blocks, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(n_elem,)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )
    return out


def run_partials(*, tile: int, d: int, row_blocks: int, threads_per_group: int) -> mx.array:
    """Dispatches the split-K partials mechanism (same grid shape as `run_atomic`, no
    atomics), then reduces the (row_blocks, n_elem) buffer with `mx.sum(axis=0)` -- the
    reduction step split-K's pattern always needs after the partials-write kernel."""
    n_elem = tile * d
    tg_x = min(threads_per_group, n_elem)
    kernel = _partials_kernel()
    dims = mx.array([n_elem], dtype=mx.uint32)
    (partials,) = kernel(
        inputs=[dims],
        template=[("T", mx.float32)],
        grid=(n_elem, row_blocks, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(row_blocks * n_elem,)],
        output_dtypes=[mx.float32],
    )
    return mx.sum(partials.reshape(row_blocks, n_elem), axis=0)


def _to_float_list(arr: mx.array) -> list[float]:
    """`.tolist()`'s installed stub types a generic recursive union to cover N-d arrays;
    every array this experiment produces is exactly 1-D, so the cast narrows it back to
    the flat `list[float]` it actually is at runtime."""
    return cast(list[float], arr.tolist())


def check_correctness(
    *, tile: int, d: int, row_blocks: int, threads_per_group: int,
) -> dict[str, dict[str, object]]:
    """Runs both mechanisms at the given shape and compares each against the exact
    reference. The partials-buffer path (no atomics -- the known-correct oracle) is
    expected to pass unconditionally; `atomic_outputs` is wrapped in a broad `except`
    because a JIT compile failure or a wrong numeric result IS a valid verdict for THIS
    experiment (the task brief: "don't fight it"), not a bug to propagate as a crash."""
    n_elem = tile * d
    if not fits_float32_exact(n_elem=n_elem, row_blocks=row_blocks):
        raise ValueError(
            f"tile={tile} d={d} row_blocks={row_blocks} gives a max expected value of "
            f"{max_expected_value(n_elem=n_elem, row_blocks=row_blocks)}, at or above the "
            f"float32 exact-integer ceiling ({FLOAT32_EXACT_INT_CEILING}) -- shrink the "
            "shape or row_blocks so a dropped atomic add stays exactly detectable"
        )
    expected = expected_totals(n_elem=n_elem, row_blocks=row_blocks)
    results: dict[str, dict[str, object]] = {}

    try:
        observed = _to_float_list(run_atomic(
            tile=tile, d=d, row_blocks=row_blocks, threads_per_group=threads_per_group,
        ))
        diffs = [abs(o - e) for o, e in zip(observed, expected, strict=True)]
        results["atomic_outputs"] = {
            "pass": observed == expected,
            "max_abs_diff": max(diffs, default=0.0),
        }
    except Exception as exc:
        # A JIT compile failure or a wrong numeric result IS a valid verdict for THIS
        # experiment (the task brief: "don't fight it") -- recorded, not a harness bug.
        results["atomic_outputs"] = {"pass": False, "error": repr(exc)[:500]}

    observed_partials = _to_float_list(run_partials(
        tile=tile, d=d, row_blocks=row_blocks, threads_per_group=threads_per_group,
    ))
    diffs_partials = [abs(o - e) for o, e in zip(observed_partials, expected, strict=True)]
    results["partials_reduction"] = {
        "pass": observed_partials == expected,
        "max_abs_diff": max(diffs_partials, default=0.0),
    }
    return results


def run_correctness(*, tile: int, d: int, row_blocks: int, threads_per_group: int) -> int:
    install_guardrails()
    results = check_correctness(
        tile=tile, d=d, row_blocks=row_blocks, threads_per_group=threads_per_group,
    )
    if not bool(results["partials_reduction"]["pass"]):
        # The partials-buffer path has no atomics/contention -- a mismatch here is a bug
        # in THIS harness (wrong reference, wrong indexing), not a legitimate finding
        # about atomic_outputs. Fail loudly rather than record a misleading verdict.
        raise AssertionError(
            "partials-buffer reduction (the known-correct oracle) mismatched the exact "
            f"reference -- a bug in this harness, not an atomic_outputs finding: "
            f"{results['partials_reduction']}"
        )
    atomic_ok = bool(results["atomic_outputs"]["pass"])
    ident = run_identity(
        experiment="ground_truth_atomic_outputs", mode="correctness", script_sha=script_sha(),
        tile=tile, d=d, row_blocks=row_blocks, threads_per_group=threads_per_group,
    )
    write_result(RESULTS / "correctness.json", ident, "ok", atomic_pass=atomic_ok,
                 mechanisms=results)
    for name, res in results.items():
        print(f"{name}: {'PASS' if res.get('pass') else 'FAIL'} {res}")
    verdict = (
        "atomic_outputs=True is CORRECT under the JIT at this shape (no dropped updates)"
        if atomic_ok else
        "atomic_outputs=True is NOT usable as tested -- see the recorded finding above"
    )
    print(f"VERDICT: {verdict}")
    return 0


def _timed(fn: Callable[[], mx.array], *, reps: int) -> tuple[float, list[float]]:
    """Warmup (pays Metal JIT) outside the measured window, then median-of-`reps`."""
    out = fn()
    mx.eval(out)
    mx.clear_cache()
    walls: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls), walls


def run_cost(
    *, n: int, tile: int, d: int, splits: int, threads_per_group: int, reps: int,
) -> int:
    """NEVER call this from an agent session -- see the module docstring. Kept here so the
    controller can run it directly on the main thread: `python scripts/
    ground_truth_atomic_outputs.py --cost`."""
    install_guardrails()
    mechanisms: dict[str, Callable[[], mx.array]] = {
        "atomic_outputs": lambda: run_atomic(
            tile=tile, d=d, row_blocks=splits, threads_per_group=threads_per_group,
        ),
        "partials_reduction": lambda: run_partials(
            tile=tile, d=d, row_blocks=splits, threads_per_group=threads_per_group,
        ),
    }
    for name, fn in mechanisms.items():
        ident = run_identity(
            experiment="ground_truth_atomic_outputs", mode="cost", script_sha=script_sha(),
            mechanism=name, n=n, tile=tile, d=d, splits=splits,
            threads_per_group=threads_per_group,
        )
        out_path = RESULTS / f"cost_{name}.json"
        if result_is_fresh(out_path, ident):
            print(f"skip (fresh): {out_path.name}")
            continue
        med, walls = _timed(fn, reps=reps)
        write_result(
            out_path, ident, "ok",
            wall_s_median=round(med, 6), wall_s_all=[round(w, 6) for w in walls],
        )
        print(f"{name}: median={med * 1e6:.1f} us (reps={walls})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ground-truth mx.fast.metal_kernel's atomic_outputs=True vs a "
                    "split-K partials buffer for cross-row-block d_w accumulation.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--correctness", action="store_true",
                      help="tiny-shape exact-correctness check (safe to run anywhere)")
    mode.add_argument("--cost", action="store_true",
                      help="production-representative timing -- MAIN SESSION ONLY, never "
                          "from an agent session")
    ap.add_argument("--tile", type=int, default=None,
                    help="output tile rows (default 16 for --correctness, 2048 for --cost)")
    ap.add_argument("--d", type=int, default=None,
                    help="output hidden dim (default 16 for --correctness, 4096 for --cost)")
    ap.add_argument("--splits", type=int, default=None,
                    help="row-block/partials count (default 200 for --correctness, 16 for "
                        "--cost)")
    ap.add_argument("--threads-per-group", type=int, default=64)
    ap.add_argument("--n", type=int, default=8192,
                    help="--cost only: context rows accumulated over (identity/context; "
                        "not used by the accumulation kernels themselves)")
    ap.add_argument("--reps", type=int, default=REPS, help="--cost only: timed reps after warmup")
    args = ap.parse_args(argv)

    mode_name = "correctness" if args.correctness else "cost"
    tile, d, splits = resolve_shape(
        mode=mode_name, tile=args.tile, d=args.d, splits=args.splits,
    )
    threads_per_group = int(args.threads_per_group)
    if args.correctness:
        return run_correctness(
            tile=tile, d=d, row_blocks=splits, threads_per_group=threads_per_group,
        )
    return run_cost(
        n=int(args.n), tile=tile, d=d, splits=splits,
        threads_per_group=threads_per_group, reps=int(args.reps),
    )


if __name__ == "__main__":
    raise SystemExit(main())
