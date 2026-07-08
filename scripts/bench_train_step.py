"""Committed acceptance bench: end-to-end mlx-lm fine-tune steps, ours (via the
`train_step` condition kind, `impl` resolved by `bench/worker.py`'s adapter call --
"auto" resolves to the kernel path on a verified mlx install) vs stock `mlx_lm.lora`
(the SAME `train_step` kind with `stock=True`, which drives the real, compiled
`mlx_lm.tuner.trainer.train()` with its own `default_loss` -- see `bench/worker.py`'s
module docstring for why `ours` and `stock` need different internal step-loop
mechanisms).

Thin driver over `bench.runner.run_conditions`, matching `bench_loss_layer.py`'s own
reasoning for NOT reimplementing subprocess-per-condition/instant-artifacts/resume-by-
skipping/refusal-as-a-result here: `train_step` is an EXISTING `worker.py` condition
kind, so `run_conditions` already provides all of that. Unlike `bench_loss_layer.py`,
`bench.runner.report`'s generic ratio computation does not apply here at all: it groups
conditions by "everything but `impl`", but `ours`/`stock` train_step conditions differ
by their OWN `stock` param (and, for the flagship, are expected to have DIFFERENT
outcomes -- stock may legitimately OOM) -- so this script reads the named `ours`/
`stock` artifact PAIR for each (model, seq_len) directly and computes its own
comparison, refusing (same reasoning as `bench_loss_layer.py`'s `check_acceptance`) an
unsafe cross-session comparison if the two artifacts were not measured together.

Parameterize model/seq/batch/steps: `--model`/`--seq-len` each accept one or more
values (the full cross product x {ours, stock} is built), `--batch`/`--steps`/
`--lora-rank`/`--lora-layers`/`--learning-rate`/`--seed` apply uniformly across the
whole matrix -- the controller picks the production matrix (>= 2 fits-both configs +
the flagship OOM-on-stock config) via these flags. `--smoke` overrides the matrix to
one small REAL model (mlx-community/Llama-3.2-1B-Instruct-4bit) at a short sequence
length for 2 steps, writing to a SEPARATE directory -- for end-to-end verification
only, never the production measurement, and (being a real model load) gated the same
way `tests/test_adapter.py`'s own `@pytest.mark.smoke` test is: see
`tests/test_bench_train_step.py::test_bench_train_step_smoke_end_to_end` (collected,
skipped by default, never executed without `--run-smoke` and a pre-downloaded model).

Heavy GPU run at production shape -- main session only, ETA ~30-60 min total for the
brief's own matrix (20 steps/condition, per the task brief's step-2 budget), serialized
across conditions (subprocess-per-condition already enforces this). Pre-flight
`memory_pressure` before running; never invoke a production-shape or `--smoke`
condition from an agent session -- `--smoke` still loads a real (if small) model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import Condition, run_conditions

# A `llama`-architecture model: the adapter's `split_model` supports llama + qwen3 (the
# 0.1.0 scope), not qwen2 — so the smoke must use a supported family. Llama-3.2-1B-4bit is
# the smallest supported downloaded model, exercising the quantized-head path end to end.
SMOKE_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SMOKE_SEQ_LEN = 512
SMOKE_STEPS = 2

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_train_step"
RESULTS_SMOKE = RESULTS / "smoke"

_ARMS = (("ours", False), ("stock", True))


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- same reasoning as
    `bench_loss_layer.py`'s own `script_sha()`: `condition_identity`'s `code_sha`
    (`bench.artifacts.CODE_SHA_DEPS`) does not cover ad hoc scripts under `scripts/`."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def model_slug(model: str) -> str:
    """Filesystem/identifier-safe stand-in for a repo id or local path (used only in
    the condition NAME -- the real `model` string still lives in `params`, unaltered,
    for the actual `mlx_lm.load` call)."""
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def condition_name(*, model: str, seq_len: int, arm: str) -> str:
    return f"train_step_{model_slug(model)}_seq{seq_len}_{arm}"


def build_conditions(
    *,
    models: list[str],
    seq_lens: list[int],
    batch: int,
    steps: int,
    lora_rank: int,
    lora_layers: int,
    learning_rate: float,
    seed: int,
    revision: str | None,
    impl: str = "auto",
    compute_dtype: str | None = None,
    grad_checkpoint: bool = False,
) -> list[Condition]:
    sha = script_sha()
    conditions: list[Condition] = []
    for model in models:
        for seq_len in seq_lens:
            for arm, stock in _ARMS:
                params: dict[str, object] = {
                    "model": model, "revision": revision, "seq_len": seq_len,
                    "batch": batch, "steps": steps, "lora_rank": lora_rank,
                    "lora_layers": lora_layers, "impl": impl, "stock": stock,
                    "learning_rate": learning_rate, "seed": seed,
                    "compute_dtype": compute_dtype, "grad_checkpoint": grad_checkpoint,
                    "script_sha": sha,
                }
                conditions.append(Condition(
                    name=condition_name(model=model, seq_len=seq_len, arm=arm),
                    kind="train_step", params=params,
                ))
    return conditions


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _session_id_of(result: dict[str, object]) -> str | None:
    identity = result.get("identity")
    if not isinstance(identity, dict):
        return None
    session_id = identity.get("session_id")
    return session_id if isinstance(session_id, str) else None


def compare_ours_vs_stock(
    paths_by_name: dict[str, Path], *, model: str, seq_len: int,
) -> dict[str, object]:
    """One (model, seq_len) row of the comparison report. Never raises -- a missing,
    corrupt, incomplete (either side not `"ok"` -- e.g. the flagship's stock arm
    legitimately OOMing/crashing), or cross-session pair all produce a `status` field
    explaining why the comparison fields are absent, rather than crashing the whole
    report over one condition's expected failure."""
    entry: dict[str, object] = {"model": model, "seq_len": seq_len}
    ours_path = paths_by_name.get(condition_name(model=model, seq_len=seq_len, arm="ours"))
    stock_path = paths_by_name.get(condition_name(model=model, seq_len=seq_len, arm="stock"))
    if ours_path is None or stock_path is None:
        entry["status"] = "missing"
        return entry
    ours = _read_json(ours_path)
    stock = _read_json(stock_path)
    if ours is None or stock is None:
        entry["status"] = "corrupt"
        return entry
    entry["ours_status"] = ours.get("status")
    entry["stock_status"] = stock.get("status")
    if ours.get("status") != "ok" or stock.get("status") != "ok":
        entry["status"] = "incomplete"
        return entry
    if _session_id_of(ours) != _session_id_of(stock):
        entry["status"] = "cross_session"
        return entry
    entry["status"] = "ok"
    ours_tps = ours.get("tokens_per_sec_median")
    stock_tps = stock.get("tokens_per_sec_median")
    entry["ours_tokens_per_sec_median"] = ours_tps
    entry["stock_tokens_per_sec_median"] = stock_tps
    if isinstance(ours_tps, int | float) and isinstance(stock_tps, int | float) and stock_tps:
        # > 1 means ours is FASTER than stock; < 1 means ours is slower.
        entry["ours_tps_over_stock_tps"] = round(ours_tps / stock_tps, 4)
    ours_losses = ours.get("loss_all")
    stock_losses = stock.get("loss_all")
    if (
        isinstance(ours_losses, list) and isinstance(stock_losses, list)
        and ours_losses and len(ours_losses) == len(stock_losses)
    ):
        diffs = [abs(float(a) - float(b))
                for a, b in zip(ours_losses, stock_losses, strict=True)]
        entry["loss_curve_worst_diff"] = round(max(diffs), 6)
    return entry


def build_report(
    paths: list[Path], *, models: list[str], seq_lens: list[int],
) -> list[dict[str, object]]:
    paths_by_name = {p.stem: p for p in paths}
    return [
        compare_ours_vs_stock(paths_by_name, model=model, seq_len=seq_len)
        for model in models for seq_len in seq_lens
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", help="one or more HF repo ids or local paths")
    ap.add_argument("--seq-len", nargs="+", type=int, help="one or more sequence lengths")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 == all layers")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--impl", choices=["auto", "kernel", "chunked", "naive"], default="auto",
                    help="the 'ours' loss impl; kernel needs bf16-compute hidden states")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float32", "float16"],
                    default=None,
                    help="cast the loaded model's floating params to this dtype before "
                        "training (int4 weights of a 4-bit checkpoint stay int4). The "
                        "kernel impl needs bfloat16 on the 4-bit models that otherwise "
                        "compute in fp16; applied to BOTH arms, holding the trunk dtype "
                        "constant so the ours-vs-stock comparison isolates the loss layer")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable gradient checkpointing (recompute activations) on BOTH "
                        "arms -- the realistic long-context QLoRA setup, where ours' "
                        "flat loss-layer memory is visible against a small trunk footprint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None, help="applied to every --model")
    ap.add_argument("--out", default=None, help="output directory (default: this "
                    "script's own _artifacts subdirectory)")
    ap.add_argument("--condition", help="run only this one named condition")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny REAL-model default (Llama-3.2-1B-4bit, seq_len=512, 2 "
                        "steps) for end-to-end verification -- never the production "
                        "matrix; still loads a real model, so this is gated the same "
                        "way tests/test_adapter.py's own smoke test is")
    args = ap.parse_args(argv)

    if args.smoke:
        models, seq_lens, steps = [SMOKE_MODEL], [SMOKE_SEQ_LEN], SMOKE_STEPS
        out_dir = Path(args.out) if args.out else RESULTS_SMOKE
        # SMOKE_MODEL loads as fp16, which the kernel path does not accept — the smoke
        # exercises `ours` end to end through the compiled trainer via the chunked path
        # (the kernel path's compile-compatibility is locked by tests/test_loss_compile.py).
        # No cast: chunked accepts fp16, so the smoke stays on the fp16 checkpoint as-is.
        impl, compute_dtype, grad_checkpoint = "chunked", None, False
    else:
        if not args.model or not args.seq_len:
            raise SystemExit("--model and --seq-len are required unless --smoke is set")
        models, seq_lens, steps = args.model, args.seq_len, args.steps
        out_dir = Path(args.out) if args.out else RESULTS
        impl, compute_dtype, grad_checkpoint = args.impl, args.compute_dtype, args.grad_checkpoint

    conditions = build_conditions(
        models=models, seq_lens=seq_lens, batch=args.batch, steps=steps,
        lora_rank=args.lora_rank, lora_layers=args.lora_layers,
        learning_rate=args.learning_rate, seed=args.seed, revision=args.revision,
        impl=impl, compute_dtype=compute_dtype, grad_checkpoint=grad_checkpoint,
    )
    if args.condition:
        matches = [c for c in conditions if c.name == args.condition]
        if not matches:
            names = ", ".join(c.name for c in conditions)
            raise SystemExit(f"unknown --condition {args.condition!r}; expected one of: {names}")
        conditions = matches

    paths = run_conditions(conditions, out_dir, session_id=new_session_id())
    comparison = build_report(paths, models=models, seq_lens=seq_lens)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
