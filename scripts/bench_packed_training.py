"""Committed throughput bench for 0.4.0 sequence packing: stock batching vs packing, at
one arm per invocation.

Both arms enable flash attention and the fused CE loss (the `packed_train` worker kind in
`bench/worker.py`), so the ONLY variable is the batching strategy -- `stock` (unpacked
`make_loss_fn` + stock `iterate_batches`) vs `packed` (`make_packed_loss_fn` +
`packed_iterate_batches`). Run each arm separately, into a SEPARATE `--out` directory (repo
gotcha 18: two arms sharing one `--out` would risk clobbering each other's artifacts even
though the arm rides the identity); the arm is also part of the condition FILENAME so a
shared dir would still not collide.

Thin driver over `bench.runner.run_conditions`, matching `bench_train_step.py`'s reasoning
for not reimplementing subprocess-per-condition / instant-artifact / resume-by-skipping /
refusal-as-a-result: `packed_train` is a `worker.py` condition kind, so `run_conditions`
already provides all of it. This script builds the ONE condition for the requested arm,
runs it, prints the resulting artifact, and follows the repo bench exit policy: exit 0 only
on a `status="ok"` measurement, nonzero on a refusal / crash / abort (no measurement).

Heavy GPU run at production shape -- main session only, ETA per arm stated before launch,
AC power + `memory_pressure` pre-flight, serialized (never two arms at once). The dataset is
a prep_alpaca jsonl (`scripts/prep_alpaca.py`); build it first.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import Condition, run_conditions

_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPT_PATH.parent.parent / "_artifacts" / "packed_bench"

_ARMS = ("stock", "packed")
DEFAULT_PACK_LEN = 4096


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes (same reasoning as `bench_train_step.py`'s
    `script_sha()`: `condition_identity`'s `code_sha` does not cover ad hoc scripts)."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def dataset_sha(path: Path) -> str:
    """Content digest of the prepped jsonl -- an identity input, so a changed dataset (even
    at the same path) is a different condition and never resume-skips a stale artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def model_slug(model: str) -> str:
    """Filesystem/identifier-safe stand-in for a repo id (used only in the condition NAME;
    the real `model` string still lives in `params`, unaltered, for `mlx_lm.load`)."""
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def condition_name(*, model: str, pack_len: int, arm: str) -> str:
    # The arm is part of the FILENAME, not only the artifact identity (gotcha 18): two arms
    # in one --out dir must never share a filename.
    return f"packed_train_{model_slug(model)}_pack{pack_len}_{arm}"


def build_condition(
    *,
    model: str,
    data: str,
    pack_len: int,
    batch_size: int,
    arm: str,
    steps: int,
    warmup: int,
    lora_rank: int,
    lora_layers: int,
    impl: str,
    learning_rate: float,
    seed: int,
    compute_dtype: str | None,
    grad_checkpoint: bool,
    revision: str | None,
    dataset_sha: str,
) -> Condition:
    """The one `packed_train` condition for the requested arm. `arm`/`pack_len`/
    `dataset_sha` ride `params` (none are reserved identity keys), so they land in the
    artifact identity; `attention_impl` is fixed to "flash" on its dedicated slot -- both
    arms run flash, so the arm distinguishes conditions, not the attention implementation."""
    params: dict[str, object] = {
        "model": model, "revision": revision, "data": data, "pack_len": pack_len,
        "batch": batch_size, "steps": steps, "warmup": warmup, "arm": arm,
        "lora_rank": lora_rank, "lora_layers": lora_layers, "impl": impl,
        "learning_rate": learning_rate, "seed": seed, "compute_dtype": compute_dtype,
        "grad_checkpoint": grad_checkpoint, "dataset_sha": dataset_sha,
        "script_sha": script_sha(),
    }
    return Condition(
        name=condition_name(model=model, pack_len=pack_len, arm=arm),
        kind="packed_train", params=params, attention_impl="flash",
    )


def read_artifact(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--data", required=True, help="prep_alpaca jsonl (from prep_alpaca.py)")
    ap.add_argument("--arm", required=True, choices=_ARMS,
                    help="batching strategy under test (one arm per invocation)")
    ap.add_argument("--pack-len", type=int, default=DEFAULT_PACK_LEN,
                    help="pack length / max sequence length (both arms)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30, help="training steps to time")
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps dropped before the median (compiled trace + calibration)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 == all layers")
    ap.add_argument("--impl", choices=["auto", "kernel", "chunked", "naive"], default="auto",
                    help="the fused-CE impl; kernel needs bf16-compute hidden states")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float32", "float16"],
                    default=None, help="cast the model's floating params before training "
                    "(int4 weights stay int4); the kernel impl needs bfloat16 on 4-bit "
                    "checkpoints that otherwise compute in fp16")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (the realistic long-context QLoRA setup)")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None, help="model checkpoint revision")
    ap.add_argument("--out", default=None, help="output directory (default: this script's "
                    "own _artifacts/packed_bench/<arm> subdirectory -- separate per arm)")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    out_dir = Path(args.out) if args.out else RESULTS / args.arm
    condition = build_condition(
        model=args.model, data=str(data_path), pack_len=args.pack_len,
        batch_size=args.batch_size, arm=args.arm, steps=args.steps, warmup=args.warmup,
        lora_rank=args.lora_rank, lora_layers=args.lora_layers, impl=args.impl,
        learning_rate=args.learning_rate, seed=args.seed, compute_dtype=args.compute_dtype,
        grad_checkpoint=args.grad_checkpoint, revision=args.revision,
        dataset_sha=dataset_sha(data_path),
    )
    paths = run_conditions([condition], out_dir, session_id=new_session_id())
    artifact = read_artifact(paths[0])
    print(json.dumps(artifact, indent=2))
    # Repo bench exit policy: only a real measurement (`status="ok"`) is a success; a
    # refusal / crash / memory-abort all exit nonzero so a wrapping run notices.
    status = artifact.get("status") if artifact else None
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
