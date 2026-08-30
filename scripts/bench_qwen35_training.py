"""End-to-end qwen3_5 fine-tune throughput bench: real mlx-lm LoRA training steps against
a real (`mlx_lm.load`-resolved) qwen3_5 checkpoint, across THREE arms per `(seq_len,
batch)` grid point:

- `stock`     -- the model untouched. Every linear-attention layer keeps mlx-lm's own
                 `GatedDeltaNet`, whose forward routes through mlx-lm's own
                 `gated_delta_update` (its own sequential python-loop ops in train mode,
                 its own fused Metal kernel in eval mode).
- `chunked`   -- `enable_gated_delta_training(model, impl="chunked")`: every
                 linear-attention layer's forward is replaced by a
                 `GatedDeltaTrainingProxy` routed through this project's chunk-parallel
                 op.
- `sequential`-- `enable_gated_delta_training(model, impl="sequential")`: the SAME
                 proxy replacement as `chunked`, but routed through mlx-lm's own
                 sequential op instead of this project's chunked op. This is the
                 attribution control: `chunked` differs from `sequential` in exactly
                 ONE thing (which op the proxy calls), so a throughput gap between them
                 is attributable to the op itself. A throughput gap between `sequential`
                 and `stock`, by contrast, is attributable to the BLOCK REWRITE (holding
                 the original module's submodules under new names, discarding the
                 `use_kernel`/cache-aware branches `GatedDeltaNet.__call__` carries) --
                 conflating that with an op-level win would overstate the chunked op's
                 own contribution.

Every arm casts the loaded checkpoint's floating parameters to bfloat16 before doing
anything else, via a PATH-AWARE walk that honours the model's own `cast_predicate`
(`nn.Module.set_dtype`'s `predicate` kwarg is called with a DTYPE, never a path, so a
bare `model.set_dtype(mx.bfloat16)` cannot invoke qwen3_5's own path-keyed
`cast_predicate` and would downcast `A_log` along with everything else -- corrupting the
gated-decay rates every arm's forward depends on, `chunked`/`sequential` included, since
`GatedDeltaTrainingProxy._pre_norm` refuses outright when `A_log` is not fp32). This
mirrors `mlx_lm/convert.py`'s own `tree_map_with_path` + `cast_predicate` composition,
and asserts `A_log` stayed fp32 afterward on every linear-attention layer so a regression
here fails loudly instead of silently biasing the comparison.

Training data is synthetic (fixed-length random-token rows, one batch group repeated for
`steps` iterations) -- this bench measures per-step wall/tokens-per-second/peak memory,
not convergence behaviour (see `bench_qwen35_convergence.py` for that, on real Alpaca
data). Both `chunked`/`sequential`-enabled arms and the `stock` arm run through the SAME
real, compiled `mlx_lm.tuner.trainer.train()` entry point, so the comparison is
compiled-vs-compiled.

Memory discipline, every subprocess: `install_guardrails()` (wired+soft caps) first,
then `mx.set_cache_limit(...)` to bound the retained pool, then a daemon
`install_memory_watchdog` sampling `active + cache` (dropped buffers move to MLX's
retained cache pool, which an active-only sampler misses) every 50ms against
`effective_memory_ceiling()` -- the two-term ceiling (a static device-relative rule,
consistent across the whole campaign, combined with a DYNAMIC measured-availability
reading taken fresh at THIS subprocess's own start) `bench/worker.py`'s `train_step`
condition also uses; a too-crowded-to-start-safely reading refuses with its own
`refused_environment` status rather than risking a paging storm. `mlx_lm.tuner.trainer.
train()` also raises the wired limit to the device max at its OWN entry -- the loss
callable handed to `train()` re-asserts this project's house cap on its first
invocation, and the measured fields include a POST-run reassert (`WiredCapRegressionError`
if the cap did not hold throughout).

subprocess-per-condition (MLX's lazy allocator otherwise holds buffers across runs
within one process): the top-level invocation builds the full `(seq_len, arm)` grid
(`seq_len` outer, `arm` inner -- so a fresh model load never happens back-to-back for
the same shape across different arms without an intervening different shape) and
self-invokes this script once per stale condition via `subprocess.run([sys.executable,
__file__, ...])` with `--single-condition` pinned to exactly one `(seq_len, arm)` pair --
resume-by-skip identical to `bench.runner.run_conditions`'s own convention. Every arm
dimension (model, arm, seq_len, batch) is part of the artifact FILENAME, not only the
identity, so two conditions differing only in one of those can never share a path.

`--dry-run` builds the condition list and resolves every artifact path + identity
WITHOUT touching MLX or `mlx_lm` at all -- no model load, no GPU allocation, nothing
written.

Heavy-run rules: main session, ETA stated, AC power, `memory_pressure` pre-flight,
serialized against other heavy runs -- never invoked from an agent session.
"""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx

from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    make_watchdog_on_breach,
    new_session_id,
    result_is_fresh,
    write_result,
)
from mlx_train_perf.core.guards import (
    DEFAULT_WALL_BUDGET_S,
    clamped_caps,
    effective_memory_ceiling,
    install_guardrails,
    install_memory_watchdog,
    wired_cap_holds,
)
from mlx_train_perf.errors import (
    MemoryBudgetError,
    MissingDependencyError,
    RecurrentInputError,
    WiredCapRegressionError,
)
from mlx_train_perf.families import text_args, text_model
from mlx_train_perf.recurrent.wrapper import enable_gated_delta_training

_ARMS: tuple[str, ...] = ("stock", "chunked", "sequential")
DEFAULT_SEQ_LENS: tuple[int, ...] = (512, 1024, 2048)
DEFAULT_BATCH = 1
DEFAULT_STEPS = 15

# Bounds the retained allocator cache pool -- part of the memory discipline every
# subprocess installs before doing any real allocation (see the module docstring).
CACHE_LIMIT_BYTES = 4 * 1024**3

_STDERR_TAIL_CHARS = 4000  # enough to see the failing assertion/traceback, not a full dump

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUT_DIR = _SCRIPTS_DIR.parent / "_artifacts" / "bench_qwen35_training"


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- `bench.artifacts.CODE_SHA_DEPS` covers
    the recurrent-path `src/` modules this script calls into, not ad hoc scripts under
    `scripts/`, so without this an edit to this script's own measurement logic would not
    invalidate a previously-written artifact."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def model_slug(model: str) -> str:
    """Filesystem/identifier-safe stand-in for a repo id or local path (used only in the
    condition NAME -- the real `model` string still lives in `params`, unaltered, for
    the actual `mlx_lm.load` call)."""
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def _require_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "the qwen3_5 training bench requires the 'mlx-lm' extra: "
            "pip install 'mlx-train-perf[mlx-lm]'"
        ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainCondition:
    name: str
    model: str
    revision: str | None
    arm: str
    seq_len: int
    batch: int
    steps: int
    lora_rank: int
    lora_layers: int
    learning_rate: float
    seed: int
    grad_checkpoint: bool


def condition_name(*, model: str, arm: str, seq_len: int, batch: int) -> str:
    # Every arm dimension that could otherwise let two conditions collide on one
    # artifact path (model/arm/seq_len/batch) is part of the condition NAME, not just
    # the identity.
    return f"qwen35_train_{model_slug(model)}_{arm}_T{seq_len}_B{batch}"


def build_conditions(
    *, model: str, revision: str | None, arms: list[str], seq_lens: list[int],
    batch: int, steps: int, lora_rank: int, lora_layers: int, learning_rate: float,
    seed: int, grad_checkpoint: bool,
) -> list[TrainCondition]:
    """The full `(seq_len, arm)` grid, pure -- `seq_len` outer, `arm` inner (matches
    `scripts/bench_recurrent_layer.py`'s own grid-construction ordering)."""
    return [
        TrainCondition(
            name=condition_name(model=model, arm=arm, seq_len=seq_len, batch=batch),
            model=model, revision=revision, arm=arm, seq_len=seq_len, batch=batch,
            steps=steps, lora_rank=lora_rank, lora_layers=lora_layers,
            learning_rate=learning_rate, seed=seed, grad_checkpoint=grad_checkpoint,
        )
        for seq_len in seq_lens
        for arm in arms
    ]


def _params_for(condition: TrainCondition) -> dict[str, object]:
    return {
        "model": condition.model, "revision": condition.revision, "arm": condition.arm,
        "seq_len": condition.seq_len, "batch": condition.batch, "steps": condition.steps,
        "lora_rank": condition.lora_rank, "lora_layers": condition.lora_layers,
        "learning_rate": condition.learning_rate, "seed": condition.seed,
        "grad_checkpoint": condition.grad_checkpoint, "compute_dtype": "bfloat16",
        "script_sha": script_sha(),
    }


def _identity_for(condition: TrainCondition, *, session_id: str) -> dict[str, object]:
    return condition_identity(
        kind="qwen35_gated_delta_train", session_id=session_id,
        params=_params_for(condition),
    )


def _out_path(out_dir: Path, condition: TrainCondition) -> Path:
    return out_dir / f"{condition.name}.json"


def _cast_bf16_protecting_a_log(model: Any) -> None:
    """Cast every floating parameter to bf16 IN PLACE, except `A_log`, honouring the
    model's own path-keyed `cast_predicate` -- see the module docstring for why a bare
    `model.set_dtype(mx.bfloat16)` cannot do this. Mirrors `mlx_lm/convert.py`'s own
    `tree_map_with_path` + `cast_predicate` composition. Asserts `A_log` stayed fp32 on
    every linear-attention layer afterward, so a regression here fails loudly instead of
    silently biasing every arm's numbers."""
    from mlx.utils import tree_map_with_path  # noqa: PLC0415

    cast_predicate = model.cast_predicate

    def _cast(path: str, value: mx.array) -> mx.array:
        if cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(_cast, model.parameters()))
    trunk = text_model(model)
    bad = [
        i for i, layer in enumerate(trunk.layers)
        if layer.is_linear and layer.linear_attn.A_log.dtype != mx.float32
    ]
    if bad:
        raise RecurrentInputError(
            f"A_log did not stay float32 after the path-aware bf16 cast at layer(s) "
            f"{bad} -- cast_predicate did not protect it as expected"
        )


def _load_and_prepare_model(condition: TrainCondition) -> Any:
    """Load, cast, (conditionally) enable, freeze, and LoRA-inject -- in the
    load-bearing order `bench/worker.py`'s `run_train_step` also uses: the dtype cast
    lands BEFORE `enable_gated_delta_training` (the supported order -- casting after
    enable would corrupt `A_log` the same way a bare `set_dtype` would, see the module
    docstring), `enable_gated_delta_training` BEFORE `freeze()`/`linear_to_lora_layers`
    (LoRA target discovery walks `named_modules()` by path, so the proxy has to already
    be in the tree at injection time), and a forced `mx.eval(model.parameters())` after
    every lazy setup step so no one-time init cost leaks into the measured window."""
    import mlx_lm  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    model, _tokenizer = cast(
        "tuple[Any, Any]", mlx_lm.load(condition.model, revision=condition.revision)
    )
    _cast_bf16_protecting_a_log(model)
    mx.eval(model.parameters())
    if condition.arm != "stock":
        enable_gated_delta_training(
            model, impl=cast('Literal["chunked", "sequential"]', condition.arm)
        )
    mx.random.seed(condition.seed)  # BEFORE freeze/LoRA-injection: their init draws next
    model.freeze()
    linear_to_lora_layers(
        model, condition.lora_layers,
        {"rank": condition.lora_rank, "dropout": 0.0, "scale": 20.0},
    )
    mx.eval(model.parameters())
    return model


def _synthetic_train_examples(
    *, vocab_size: int, seq_len: int, num_examples: int, seed: int,
) -> list[list[int]]:
    """`num_examples` fixed-length (`seq_len + 1`) random-token rows -- plain
    `random.Random`, independent of MLX's global RNG state. With `num_examples ==
    batch` (this script's only caller), `iterate_batches` ends up with exactly ONE batch
    group, repeated every training step (matches `bench/worker.py`'s own
    `_synthetic_train_examples` convention: this measures throughput, not convergence,
    so a single repeated batch is deliberate, not an oversight)."""
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(seq_len + 1)] for _ in range(num_examples)
    ]


class _SyntheticDataset:
    """`train()` hands its `train_dataset` straight to `iterate_batches` unwrapped, so
    `dataset[idx]` must return the final `(tokens, offset)` pair directly -- `offset=0`
    always, every synthetic row fully unmasked."""

    def __init__(self, examples: list[list[int]]) -> None:
        self._examples = examples

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self._examples[idx], 0

    def __len__(self) -> int:
        return len(self._examples)


class _RecordingCallback:
    """Duck-typed `mlx_lm.tuner.callbacks.TrainingCallback`. Collects every per-step
    report (`steps_per_report=1` means exactly one call per training iteration).
    `on_val_loss_report` is never invoked (`val_dataset=[]` below)."""

    def __init__(self) -> None:
        self.train_info: list[dict[str, object]] = []

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_info.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:  # noqa: ARG002
        return None  # pragma: no cover


def _make_reasserting_loss(
    loss_fn: Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]],
    observed_before: list[int],
) -> Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Wraps a trainer loss callable so its FIRST invocation re-asserts this project's
    house wired cap before doing anything else -- `mlx_lm.tuner.trainer.train()` raises
    the wired limit to the device max at its OWN entry, and the reassert has to live
    INSIDE the compiled step (a caller-side reassert before/after `train()` alone
    protects nothing in between). `mx.compile` only re-runs a traced Python body on a
    recompile, so firing once is sufficient."""

    def wrapped(model: Any, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        if not observed_before:
            observed_before.append(install_guardrails())
        return loss_fn(model, batch, lengths)

    return wrapped


def _run_train_steps(
    model: Any, optimizer: Any,
    loss_fn: Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]],
    examples: list[list[int]], *, batch: int, seq_len: int, steps: int,
    grad_checkpoint: bool,
) -> list[dict[str, object]]:
    """Drives `steps` real fine-tune iterations through the compiled
    `mlx_lm.tuner.trainer.train()` -- used for all three arms identically."""
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415

    callback = _RecordingCallback()
    train_set = _SyntheticDataset(examples)
    with tempfile.TemporaryDirectory(prefix="mlx-train-perf-qwen35-bench-") as tmp_dir:
        args = TrainingArgs(
            batch_size=batch, iters=steps, val_batches=0, steps_per_report=1,
            steps_per_eval=steps + 1, steps_per_save=steps + 1,
            max_seq_length=seq_len + 1, grad_checkpoint=grad_checkpoint,
            adapter_file=str(Path(tmp_dir) / "adapters.safetensors"),
        )
        train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=[],
              args=args, loss=loss_fn, training_callback=cast(Any, callback))
    return callback.train_info


def measure_condition(condition: TrainCondition) -> dict[str, object]:
    """The worker body: load + prepare the model for `condition.arm`, drive `steps`
    real fine-tune iterations, and read back the wired-cap/memory story. Guardrails are
    installed by the CALLER (`_run_single_condition`) before this runs."""
    _require_mlx_lm()
    import mlx.optimizers as optim  # noqa: PLC0415

    model = _load_and_prepare_model(condition)
    base_loss = make_loss_fn(model, impl="auto")
    observed_before: list[int] = []
    loss_fn = _make_reasserting_loss(base_loss, observed_before)

    vocab_size = int(text_args(model).vocab_size)
    examples = _synthetic_train_examples(
        vocab_size=vocab_size, seq_len=condition.seq_len, num_examples=condition.batch,
        seed=condition.seed,
    )

    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    expected_wired, _soft = clamped_caps(dev_max)
    opt = optim.Adam(learning_rate=condition.learning_rate)

    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    step_reports = _run_train_steps(
        model, opt, loss_fn, examples, batch=condition.batch, seq_len=condition.seq_len,
        steps=condition.steps, grad_checkpoint=condition.grad_checkpoint,
    )
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3

    observed_after = install_guardrails()
    if not wired_cap_holds(observed_bytes=observed_after, expected_bytes=expected_wired):
        raise WiredCapRegressionError(
            f"qwen35 train-step condition's wired limit was {observed_after / 1024**3:.2f} "
            f"GB after training, expected the house cap {expected_wired / 1024**3:.2f} GB "
            "-- mlx_lm.tuner.trainer.train()'s entry-time override was not correctly "
            "re-asserted"
        )

    tokens_per_sec_all = [
        float(cast(float, info["tokens_per_second"])) for info in step_reports
    ]
    loss_all = [float(cast(float, info["train_loss"])) for info in step_reports]
    return {
        "arm": condition.arm, "seq_len": condition.seq_len, "batch": condition.batch,
        "tokens_per_sec_median": (
            round(statistics.median(tokens_per_sec_all), 3) if tokens_per_sec_all else 0.0
        ),
        "tokens_per_sec_all": [round(x, 3) for x in tokens_per_sec_all],
        "loss_all": [round(x, 6) for x in loss_all],
        "active_before_gb": round(active_before / 1024**3, 4),
        "marginal_peak_gb": round(marginal_peak_gb, 4),
        "total_peak_gb": round(active_before / 1024**3 + marginal_peak_gb, 4),
        "observed_wired_limit_gb": round(observed_after / 1024**3, 4),
        "house_wired_limit_gb": round(expected_wired / 1024**3, 4),
        "wired_limit_before_reassert_gb": (
            round(observed_before[0] / 1024**3, 4) if observed_before else None
        ),
    }


def _run_single_condition(condition: TrainCondition, *, out_dir: Path, session_id: str) -> Path:
    """Measures exactly ONE condition and writes its artifact -- no freshness check here
    (the orchestrator, `run_grid`, already unlinks a stale artifact before spawning)."""
    out_path = _out_path(out_dir, condition)
    ident = _identity_for(condition, session_id=session_id)
    install_guardrails()  # FIRST -- before any allocation this condition makes
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    try:
        ceiling = effective_memory_ceiling()
    except MemoryBudgetError as exc:
        # Too crowded to START safely is an ENVIRONMENT-transient outcome -- its own
        # status, distinct from a crash. Only "ok" artifacts are fresh on resume, so a
        # later, quieter invocation re-runs this condition automatically.
        write_result(out_path, ident, "refused_environment", error=str(exc))
        return out_path
    warning_field: dict[str, object] = (
        {"memory_warning": ceiling.warning} if ceiling.warning is not None else {}
    )
    watchdog = install_memory_watchdog(
        ceiling_bytes=ceiling.ceiling_bytes,
        sampler=lambda: mx.get_active_memory() + mx.get_cache_memory(),
        interval_s=0.05,
        wall_budget_s=DEFAULT_WALL_BUDGET_S,
        on_breach=make_watchdog_on_breach(out_path, ident, ceiling.ceiling_bytes),
    )
    try:
        fields = measure_condition(condition)
        write_result(out_path, ident, "ok", **fields, **warning_field)
        return out_path
    finally:
        # A breach never reaches here -- `on_breach` already hard-exited the process.
        watchdog.stop()


def _spawn_condition(
    condition: TrainCondition, *, out_dir: Path, session_id: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable, str(_SCRIPT_PATH),
        "--model", condition.model,
        "--arm", condition.arm,
        "--seq-lens", str(condition.seq_len),
        "--batch", str(condition.batch),
        "--steps", str(condition.steps),
        "--lora-rank", str(condition.lora_rank),
        "--lora-layers", str(condition.lora_layers),
        "--learning-rate", str(condition.learning_rate),
        "--seed", str(condition.seed),
        "--out-dir", str(out_dir),
        "--session-id", session_id,
        "--single-condition",
    ]
    if condition.revision is not None:
        cmd += ["--revision", condition.revision]
    if condition.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_grid(
    conditions: list[TrainCondition], *, out_dir: Path, session_id: str,
) -> list[Path]:
    """Subprocess-per-condition orchestration: a fresh artifact is skipped without
    spawning; a stale one is unlinked BEFORE spawning; a nonzero exit or a clean exit
    that wrote nothing is recorded as an `"error"` result on this side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for condition in conditions:
        out_path = _out_path(out_dir, condition)
        ident = _identity_for(condition, session_id=session_id)
        paths.append(out_path)
        if result_is_fresh(out_path, ident):
            continue
        out_path.unlink(missing_ok=True)
        proc = _spawn_condition(condition, out_dir=out_dir, session_id=session_id)
        if proc.returncode != 0 and not out_path.exists():
            stderr_tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
            write_result(
                out_path, ident, "error", error_type="WorkerCrashed",
                error_msg=stderr_tail, returncode=proc.returncode,
            )
        elif proc.returncode == 0 and not out_path.exists():
            write_result(
                out_path, ident, "error", error_type="WorkerExitedWithoutArtifact",
                error_msg="subprocess exited 0 without writing an artifact", returncode=0,
            )
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--revision", default=None, help="model checkpoint revision")
    ap.add_argument("--arm", nargs="+", choices=_ARMS, default=list(_ARMS))
    ap.add_argument("--seq-lens", nargs="+", type=int, default=list(DEFAULT_SEQ_LENS))
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="training steps to time")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 == all layers")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (the realistic long-context QLoRA setup)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the condition list and resolve every artifact path + "
                        "identity WITHOUT touching MLX or mlx_lm -- no model load, no "
                        "GPU allocation, nothing written")
    # Internal self-reinvocation surface (subprocess-per-condition -- see the module
    # docstring): not part of the documented top-level CLI, suppressed from --help.
    ap.add_argument("--session-id", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single-condition", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    session_id = args.session_id or new_session_id()

    conditions = build_conditions(
        model=args.model, revision=args.revision, arms=args.arm, seq_lens=args.seq_lens,
        batch=args.batch, steps=args.steps, lora_rank=args.lora_rank,
        lora_layers=args.lora_layers, learning_rate=args.learning_rate, seed=args.seed,
        grad_checkpoint=args.grad_checkpoint,
    )

    if args.dry_run:
        report = [
            {
                "name": c.name, "out_path": str(_out_path(out_dir, c)),
                "identity": _identity_for(c, session_id=session_id),
            }
            for c in conditions
        ]
        print(json.dumps(report, indent=2))
        return 0

    if args.single_condition:
        if len(args.arm) != 1 or len(args.seq_lens) != 1:
            raise SystemExit(
                "--single-condition requires exactly one --arm and one --seq-lens value"
            )
        condition = conditions[0]
        out_path = _run_single_condition(condition, out_dir=out_dir, session_id=session_id)
        data = json.loads(out_path.read_text())
        print(json.dumps(data, indent=2))
        return 0 if data.get("status") != "error" else 1

    paths = run_grid(conditions, out_dir=out_dir, session_id=session_id)
    print(json.dumps({"paths": [str(p) for p in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
