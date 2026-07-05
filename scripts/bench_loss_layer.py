"""Committed acceptance bench: forward/loss-layer rate + peak at production shape
(n in {512, 2048, 8192}, V=151936, D=4096, bf16) x impl in {kernel, chunked, naive}.

Unlike `bench_quant_thresholds.py`/`bench_backward_ladder.py` (which predate or
deliberately bypass `mlx_train_perf.bench.runner` for conditions that need custom
ramp/backward-kernel wiring `worker.py` doesn't support), every condition here is the
EXISTING `loss_layer` kind `bench/worker.py` already implements -- so this script is a
thin driver over `bench.runner.run_conditions` (subprocess-per-condition, instant
artifacts, resume-by-skipping, refusals-as-results are all `run_conditions`'/
`worker.py`'s OWN behavior, not reimplemented here) rather than a bespoke self-
re-invoking dispatcher. `--condition NAME` still targets exactly one condition (by
filtering the list handed to `run_conditions`, not by self-re-invocation the way
`bench_quant_thresholds.py`'s own `--condition` flag works) and `--tiny` still runs
the SAME code paths at a cheap synthetic shape for end-to-end verification, writing to
a SEPARATE directory so a tiny verification run can never collide with (or be mistaken
for) an actual production measurement.

In-script acceptance check: kernel forward wall <= 1.7x naive forward wall at n=8192,
same session (the spec bar; the spike measured 1.63x) -- read directly from the two
n=8192 artifacts rather than through `runner.report`'s aggregate `ratios` dict, which
is keyed only by "slower-impl/faster-impl" and would collide across different `n`
groups that happen to share the same slower/faster pairing.

`condition_identity`'s own `code_sha` (`bench.artifacts.CODE_SHA_DEPS`) already covers
every `src/` file this condition's measurement depends on; it does NOT cover ad hoc
scripts under `scripts/` (the same reasoning `bench_backward_ladder.py`'s own
`script_sha()` documents), so THIS script folds its own byte fingerprint into each
condition's `params` -- an edit to this script's own measurement/acceptance logic
still invalidates a previously-written artifact.

Heavy GPU run at production shape -- main session only, ETA ~10-15 min for the full
3x3 grid (per the task brief's step-1 budget). Pre-flight `memory_pressure` before
running; never invoke a production-shape condition from an agent session -- `--tiny`
is the only mode safe to run outside the main session's heavy-run protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import Condition, report, run_conditions

N_VALUES = (512, 2048, 8192)
IMPLS = ("kernel", "chunked", "naive")
D, V = 4096, 151936
DTYPE = "bfloat16"
REPS = 3
SEED = 0
ACCEPTANCE_RATIO = 1.7   # kernel forward wall must be <= this x naive forward wall
ACCEPTANCE_N = 8192

TINY_N_VALUES = (64,)
TINY_D, TINY_V = 64, 128   # matches bench/worker.py's own tiny-shape precedent --
                           # large enough that g_mac_per_s doesn't round away to 0.000

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_loss_layer"
RESULTS_TINY = RESULTS / "tiny"


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- see the module docstring for why
    `condition_identity`'s `code_sha` alone does not invalidate a stale artifact when
    only this script's own logic changes (same convention as
    `bench_backward_ladder.py`'s own `script_sha()`)."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def build_conditions(*, tiny: bool) -> list[Condition]:
    n_values = TINY_N_VALUES if tiny else N_VALUES
    d = TINY_D if tiny else D
    v = TINY_V if tiny else V
    sha = script_sha()
    conditions = []
    for n in n_values:
        for impl in IMPLS:
            params: dict[str, object] = {
                "n": n, "d": d, "v": v, "dtype": DTYPE, "impl": impl,
                "reps": REPS, "seed": SEED, "script_sha": sha,
            }
            conditions.append(
                Condition(name=f"loss_layer_n{n}_{impl}", kind="loss_layer", params=params)
            )
    return conditions


def _read_result(path: Path) -> tuple[str, float | None, str | None]:
    """Returns `(status, wall_s, session_id)`. A missing/corrupt file or a `status`
    other than `"ok"` reads as `("error"-or-the-recorded-status, None, None)`."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "error", None, None
    status = str(data.get("status", "error"))
    wall = data.get("wall_s")
    identity = data.get("identity", {})
    session_id = identity.get("session_id") if isinstance(identity, dict) else None
    return status, (float(wall) if isinstance(wall, int | float) else None), session_id


def check_acceptance(paths: list[Path]) -> tuple[bool | None, str]:
    """Reads the n=`ACCEPTANCE_N` kernel/naive artifacts directly (never through
    `runner.report`'s aggregate ratio dict, which is keyed only by "slower-impl/
    faster-impl" and would collide across different `n` groups -- see the module
    docstring) and checks kernel_wall <= ACCEPTANCE_RATIO x naive_wall. Returns
    `(None, reason)` -- SKIPPED, not failed -- when either artifact is missing/not-ok
    (e.g. a `--tiny` run, or a refused condition) or when the two artifacts were
    measured under DIFFERENT `session_id`s (e.g. one condition's artifact was resumed
    from an earlier invocation while the other was freshly measured just now) --
    comparing wall-clock numbers across sessions is exactly the unsafe cross-session
    comparison `bench.runner.report`'s own `cross_session_excluded` mechanism refuses
    to make, and this check must refuse it the same way rather than silently reading
    `wall_s` from two potentially different machine sessions."""
    by_name = {p.stem: p for p in paths}
    kernel_path = by_name.get(f"loss_layer_n{ACCEPTANCE_N}_kernel")
    naive_path = by_name.get(f"loss_layer_n{ACCEPTANCE_N}_naive")
    if kernel_path is None or naive_path is None:
        return None, f"no n={ACCEPTANCE_N} kernel/naive conditions in this run"
    kernel_status, kernel_wall, kernel_session = _read_result(kernel_path)
    naive_status, naive_wall, naive_session = _read_result(naive_path)
    if kernel_status != "ok" or naive_status != "ok":
        return None, (
            f"kernel status={kernel_status!r}, naive status={naive_status!r} "
            "(acceptance check needs both 'ok')"
        )
    if kernel_session != naive_session:
        return None, (
            f"kernel and naive were measured under different sessions "
            f"({kernel_session!r} vs {naive_session!r}) -- refusing an unsafe "
            "cross-session comparison; re-run both together in one invocation"
        )
    assert kernel_wall is not None and naive_wall is not None  # noqa: PT018 -- guaranteed by "ok"
    ratio = kernel_wall / naive_wall
    passed = ratio <= ACCEPTANCE_RATIO
    reason = (
        f"kernel/naive wall ratio {ratio:.3f} "
        f"({'<=' if passed else '>'} {ACCEPTANCE_RATIO}) at n={ACCEPTANCE_N} "
        f"(kernel={kernel_wall:.4f}s, naive={naive_wall:.4f}s)"
    )
    return passed, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", help="run only this one named condition")
    ap.add_argument("--tiny", action="store_true",
                    help="cheap synthetic shape (n=64, v=128, d=64) for end-to-end "
                        "verification -- never the production measurement")
    args = ap.parse_args(argv)

    conditions = build_conditions(tiny=args.tiny)
    if args.condition:
        matches = [c for c in conditions if c.name == args.condition]
        if not matches:
            names = ", ".join(c.name for c in conditions)
            raise SystemExit(f"unknown --condition {args.condition!r}; expected one of: {names}")
        conditions = matches

    out_dir = RESULTS_TINY if args.tiny else RESULTS
    paths = run_conditions(conditions, out_dir, session_id=new_session_id())

    summary = report(paths)
    print(json.dumps(summary, indent=2))

    passed, reason = check_acceptance(paths)
    if passed is None:
        print(f"acceptance check: SKIPPED ({reason})")
        return 0
    print(f"acceptance check: {'PASS' if passed else 'FAIL'} -- {reason}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
