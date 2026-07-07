"""North-Star sweep: binary-search the max trainable context length for the 8B QLoRA
recipe (model/revision/dataset recipe pinned in every probe's artifact), for BOTH
stock `mlx_lm.lora` and ours -- the number the project's README states (max trainable
context, 8B QLoRA, 32 GB M1 Max).

Each PROBED context length is one `train_step` condition (the same kind
`bench/worker.py`/`bench_train_step.py` already use, `steps=PROBE_STEPS` -- just
enough to exercise one full forward + chunked-or-compiled backward + optimizer step,
the shape that determines whether this context length fits, not a meaningful loss
curve), dispatched through `bench.runner.run_conditions` -- so a probe that legitimately
OOMs is the standard `run_conditions` crash envelope (`status="error"`), read here as
"this context length does not fit", NOT a crashed sweep. Resume-by-skipping: unlike
`bench_loss_layer.py`/`bench_train_step.py` (which generate a FRESH `session_id` every
invocation deliberately, so a same-session ratio is never silently computed across
two different machine states), THIS script derives a STABLE, deterministic
`session_id` from the recipe itself (`_recipe_session_id`) -- a 1-2 hour sweep must
survive being interrupted and re-run, and this script never computes a same-session
ratio between arms (`ours`/`stock` converge to independent numbers), so the
resume property matters here more than the single-invocation-only property does.

Binary search shape (`find_max_context`): doubles from `--start-seq-len` until a probe
fails (or a generous absolute ceiling is hit), then bisects between the last-fitting
and first-failing bounds down to `--granularity` tokens. `find_max_context` itself is
pure (driven by a caller-supplied `probe` callable) and unit-tested with a fake oracle
in `tests/test_northstar_context_sweep.py` -- the real, Metal-touching probe path is
NOT exercised by this project's own build verification (see that test module's
docstring).

Heavy GPU run -- ETA 1-2 hours (binary search across up to ~15-20 probes per arm, each
a real forward+backward+optimizer step at growing context length against an 8B
quantized model). Main session only, pre-flight `memory_pressure`, get Denis's
EXPLICIT go-ahead before starting a run of this script -- never invoke it from an
agent session, and never invoke it without that explicit go, even at a small
`--start-seq-len`: every probe loads the real flagship model.
"""
import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from mlx_train_perf.bench.runner import Condition, run_conditions

FLAGSHIP_MODEL = "mlx-community/Qwen3-8B-4bit"
PROBE_STEPS = 2                # just enough to exercise one full training step
DEFAULT_START_SEQ_LEN = 1024   # expected to fit comfortably for both arms
DEFAULT_GRANULARITY = 256      # binary-search convergence step, in tokens
MAX_CONTEXT_CEILING = 131_072  # generous absolute stop; never expected to be reached
DATASET_RECIPE = "synthetic-fixed-length-random-tokens (bench/worker.py::_synthetic_train_examples)"

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "northstar_context_sweep"

ARMS = ("ours", "stock")


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- same reasoning as
    `bench_loss_layer.py`/`bench_train_step.py`'s own `script_sha()`."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def _recipe_session_id(
    *, model: str, revision: str | None, batch: int, lora_rank: int, lora_layers: int,
    seed: int, compute_dtype: str = "bfloat16",
) -> str:
    """Deterministic (NOT random) session id -- see the module docstring for why this
    sweep needs stable-across-restarts resume rather than
    `bench.artifacts.new_session_id`'s fresh-per-invocation guarantee. Two invocations
    with the SAME recipe hash to the SAME id (enabling resume); a DIFFERENT recipe
    (model/revision/batch/lora config/seed/compute_dtype, or an edit to this script's
    own logic) hashes to a different one (never silently resuming a stale, incompatible
    sweep)."""
    h = hashlib.sha256()
    for part in (
        model, str(revision), str(batch), str(lora_rank), str(lora_layers), str(seed),
        compute_dtype, script_sha(),
    ):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


def probe_condition_name(*, model: str, seq_len: int, arm: str) -> str:
    return f"northstar_{model_slug(model)}_seq{seq_len}_{arm}"


def build_probe(
    *, model: str, revision: str | None, batch: int, lora_rank: int, lora_layers: int,
    seed: int, arm: str, seq_len: int, compute_dtype: str = "bfloat16",
) -> Condition:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    # `compute_dtype` casts the loaded model's floating params before training (int4
    # weights of the 4-bit flagship stay int4). The `ours` arm's kernel accepts only
    # fp32/bf16 hidden -- the 4-bit flagship computes in fp16 uncast, so `auto` would
    # refuse the kernel and the probe would (falsely) read as "does not fit". Applied to
    # BOTH arms so the max-context comparison holds the trunk dtype constant.
    params: dict[str, object] = {
        "model": model, "revision": revision, "seq_len": seq_len, "batch": batch,
        "steps": PROBE_STEPS, "lora_rank": lora_rank, "lora_layers": lora_layers,
        "impl": "auto", "stock": arm == "stock", "seed": seed,
        "compute_dtype": compute_dtype,
        "script_sha": script_sha(), "dataset_recipe": DATASET_RECIPE,
    }
    return Condition(
        name=probe_condition_name(model=model, seq_len=seq_len, arm=arm),
        kind="train_step", params=params,
    )


def _read_status(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "error"
    return str(data.get("status", "error"))


def make_probe(
    *, model: str, revision: str | None, batch: int, lora_rank: int, lora_layers: int,
    seed: int, arm: str, out_dir: Path, session_id: str, compute_dtype: str = "bfloat16",
) -> Callable[[int], bool]:
    """Builds the `probe(seq_len) -> bool` callable `find_max_context` drives: ONE
    `train_step` condition per call, dispatched through `run_conditions`
    (subprocess-isolated; a legitimate OOM/crash records the standard crash envelope,
    read here as "does not fit" -- see the module docstring). Resume-by-skipping is
    `run_conditions`'s own existing behavior: a context length already probed under
    this EXACT `session_id` is recognized as fresh and never re-launched."""

    def probe(seq_len: int) -> bool:
        condition = build_probe(
            model=model, revision=revision, batch=batch, lora_rank=lora_rank,
            lora_layers=lora_layers, seed=seed, arm=arm, seq_len=seq_len,
            compute_dtype=compute_dtype,
        )
        paths = run_conditions([condition], out_dir, session_id=session_id)
        return _read_status(paths[0]) == "ok"

    return probe


def find_max_context(
    probe: Callable[[int], bool], *, start: int, granularity: int,
    ceiling: int = MAX_CONTEXT_CEILING,
) -> tuple[int, int | None]:
    """Pure driver over a caller-supplied `probe(seq_len) -> bool` ("does this context
    length fit?"): doubles `start` while it keeps fitting, then binary-searches down
    to `granularity` tokens between the last-fitting and first-failing bounds.

    Returns `(max_fitting_seq_len, min_failing_seq_len)`. `min_failing_seq_len` is
    `None` only if `ceiling` was reached WITHOUT ever finding a context that failed
    (every doubling step fit) -- the caller should raise `ceiling` and retry in that
    case, since the search converged on an artificial stop, not the model's real limit.

    If even `start` itself does not fit, returns `(0, start)` -- the caller should
    lower `start` and retry; this function never probes below `start`."""
    if not probe(start):
        return 0, start
    low = start
    high = start * 2
    while high <= ceiling and probe(high):
        low = high
        high *= 2
    if high > ceiling:
        return low, None
    # Invariant entering the loop: probe(low) is True, probe(high) is False.
    while high - low > granularity:
        mid = low + (high - low) // 2
        mid -= mid % granularity  # keep every probe granularity-aligned
        if mid <= low:
            break
        if probe(mid):
            low = mid
        else:
            high = mid
    return low, high


def sweep_arm(
    *, model: str, revision: str | None, batch: int, lora_rank: int, lora_layers: int,
    seed: int, arm: str, start: int, granularity: int, out_dir: Path, session_id: str,
    compute_dtype: str = "bfloat16",
) -> dict[str, object]:
    probe = make_probe(
        model=model, revision=revision, batch=batch, lora_rank=lora_rank,
        lora_layers=lora_layers, seed=seed, arm=arm, out_dir=out_dir,
        session_id=session_id, compute_dtype=compute_dtype,
    )
    max_fitting, min_failing = find_max_context(probe, start=start, granularity=granularity)
    return {
        "arm": arm, "max_fitting_seq_len": max_fitting, "min_failing_seq_len": min_failing,
        "converged": min_failing is not None or max_fitting == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FLAGSHIP_MODEL,
                    help=f"HF repo id or local path (default: {FLAGSHIP_MODEL})")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 == all layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-seq-len", type=int, default=DEFAULT_START_SEQ_LEN)
    ap.add_argument("--granularity", type=int, default=DEFAULT_GRANULARITY)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float32", "float16"],
                    default="bfloat16",
                    help="cast the loaded model's floating params to this dtype before "
                        "each probe (the 4-bit flagship computes in fp16 otherwise, and "
                        "the ours-arm kernel needs bf16). Applied to both arms")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out) if args.out else RESULTS
    session_id = _recipe_session_id(
        model=args.model, revision=args.revision, batch=args.batch,
        lora_rank=args.lora_rank, lora_layers=args.lora_layers, seed=args.seed,
        compute_dtype=args.compute_dtype,
    )
    arms = ARMS if args.arm == "both" else (args.arm,)
    results = [
        sweep_arm(
            model=args.model, revision=args.revision, batch=args.batch,
            lora_rank=args.lora_rank, lora_layers=args.lora_layers, seed=args.seed,
            arm=arm, start=args.start_seq_len, granularity=args.granularity,
            out_dir=out_dir, session_id=session_id, compute_dtype=args.compute_dtype,
        )
        for arm in arms
    ]
    print(json.dumps({
        "model": args.model, "revision": args.revision, "dataset_recipe": DATASET_RECIPE,
        "compute_dtype": args.compute_dtype, "session_id": session_id, "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
