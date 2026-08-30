"""qwen3_5 LoRA convergence check: real Alpaca fine-tune runs, one arm per invocation --

- `stock`   -- the model untouched (mlx-lm's own `GatedDeltaNet`).
- `enabled` -- `enable_gated_delta_training(model, impl="chunked")`: every
              linear-attention layer routed through this project's chunk-parallel op.

Unlike `bench_qwen35_training.py` (synthetic data, throughput-only), this script trains
on a real prepped Alpaca jsonl (`scripts/prep_alpaca.py`'s `{"tokens": [...], "offset":
N}` records) and tracks the LOSS CURVE: does training with the chunked op converge the
same way stock does, on real data.

Both arms cast the checkpoint's floating parameters to bf16 via the same path-aware,
`A_log`-protecting walk `bench_qwen35_training.py` uses (see that script's module
docstring for why a bare `model.set_dtype` cannot do this), and read training batches
through the SAME seeded `mlx_lm.tuner.trainer.iterate_batches` (`seed=` fixes both the
len-sorted batch grouping's processing order and the validation batch order) -- so two
arms invoked with the same `--seed`/`--data`/`--batch-size`/`--max-seq-length` see
byte-identical training and validation batches in the same order, and a loss-curve
divergence is attributable to the arm, not to different data.

Validation is scored on a HELD-OUT prefix of the same jsonl (the first `--val-examples`
records; deterministic, so "held out" means the same physical examples for both arms)
via `mlx_lm.tuner.trainer.train()`'s OWN built-in validation mechanism
(`val_dataset=`/`steps_per_eval=`): its `evaluate()` helper calls the trainer loss
callable DIRECTLY -- `losses, toks = loss(model, *batch)` -- never through
`nn.value_and_grad`, so every recorded `val_loss` is a plain-forward measurement, never
a gradient-computation byproduct. `train()`'s own scheduling gives "before" (iteration 1,
always measured before any weight update), "every `--val-every` steps", and "after"
(the final iteration) for free.

Because `evaluate()` also calls `model.eval()` before scoring (and `model.train()`
after), the two arms' validation forwards do NOT run the same code path their training
steps did: `GatedDeltaTrainingProxy.__call__` has no separate eval-mode branch (it
always routes through its bound op, `chunked` here, regardless of `model.training`), so
the `enabled` arm's validation is the SAME chunked op its training used. The unwrapped
`GatedDeltaNet.__call__`, by contrast, gates on `use_kernel = not self.training`: in
eval mode it routes through mlx-lm's OWN fused Metal inference kernel, a DIFFERENT code
path from the sequential python-loop op its own training used. So the `stock` arm's val
curve carries an implementation swap the `enabled` arm's val curve does not -- read a
val-loss gap between the two arms with that in mind (`--compare`'s output states this
explicitly, see `_VAL_IMPLEMENTATION_DISCLOSURE` below).

Per-step artifact flush: the training callback writes the FULL accumulated
train/val history to disk on every single `on_train_loss_report`/`on_val_loss_report`
call (`steps_per_report=1` means exactly one call per training iteration) -- an
interrupted run therefore loses at most the step in flight, never an already-completed
one. Intermediate writes carry status `"in_progress"` (not `"ok"`), so
`bench.artifacts.result_is_fresh` never mistakes a partial run for a complete one; the
final `"ok"` write happens only after `train()` returns normally.

`--compare --stock-artifact <path> --enabled-artifact <path>` folds two already-measured
arm artifacts into one comparison report (per-iteration train/val loss for both arms,
plus the validation-implementation disclosure) instead of running a new measurement --
refuses (status `"identity_mismatch"`) rather than comparing two runs whose model/
dataset/seed/batching differ in anything but the arm itself.

Memory discipline and subprocess conventions mirror `bench_qwen35_training.py` (see that
script's module docstring): `install_guardrails()` + `mx.set_cache_limit(...)` +
`effective_memory_ceiling()` + a daemon `install_memory_watchdog` sampling
`active + cache` every 50ms, `train()`'s own entry-time wired-cap override re-asserted
in the loss callable's first call and confirmed after `train()` returns.

`--dry-run` builds the condition, resolves the artifact path/identity, and hashes the
dataset file bytes for the identity (a plain file read, not a GPU op) -- no model load,
no GPU allocation, no dataset row parsed, nothing written.

Heavy-run rules: main session, ETA stated, AC power, `memory_pressure` pre-flight,
serialized against other heavy runs (including the OTHER arm of this same comparison --
subprocess-per-condition here means per-ARM, run one at a time) -- never invoked from an
agent session.
"""
import argparse
import functools
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    make_watchdog_on_breach,
    new_session_id,
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
from mlx_train_perf.families import text_model
from mlx_train_perf.recurrent.wrapper import enable_gated_delta_training

_ARMS: tuple[str, ...] = ("stock", "enabled")
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_STEPS = 200
DEFAULT_VAL_EVERY = 20
DEFAULT_VAL_EXAMPLES = 16

# Bounds the retained allocator cache pool -- part of the memory discipline every
# invocation installs before doing any real allocation (see the module docstring).
CACHE_LIMIT_BYTES = 4 * 1024**3

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
RESULTS = _SCRIPTS_DIR.parent / "_artifacts" / "bench_qwen35_convergence"

_VAL_IMPLEMENTATION_DISCLOSURE = (
    "the 'enabled' arm's validation forward always routes through this project's "
    "chunked GatedDelta op -- the training proxy has no separate eval-mode branch. "
    "The 'stock' arm's eval-mode validation instead routes through mlx-lm's own fused "
    "inference kernel, a DIFFERENT code path from the sequential op stock training "
    "used. The two val-loss curves therefore differ by an implementation detail as "
    "well as by the arm's core recurrence math -- read a val-loss gap between the two "
    "arms with that in mind, not as a pure math-correctness signal."
)


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes (same reasoning as
    `bench_qwen35_training.py`'s own `script_sha()`)."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def dataset_sha(path: Path) -> str:
    """Content digest of the prepped jsonl -- an identity input, so a changed dataset
    (even at the same path) is a different condition and never resume-skips a stale
    artifact. A plain file-bytes read, not a GPU op -- safe under `--dry-run`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _require_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "the qwen3_5 convergence bench requires the 'mlx-lm' extra: "
            "pip install 'mlx-train-perf[mlx-lm]'"
        ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvergenceCondition:
    name: str
    model: str
    revision: str | None
    data: str
    dataset_sha: str
    arm: str
    max_seq_length: int
    batch_size: int
    steps: int
    val_every: int
    val_examples: int
    lora_rank: int
    lora_layers: int
    learning_rate: float
    seed: int
    grad_checkpoint: bool


def condition_name(*, model: str, arm: str, max_seq_length: int, batch_size: int) -> str:
    # Every arm dimension that could otherwise let two conditions collide on one
    # artifact path (model/arm/max_seq_length/batch_size) is part of the condition
    # NAME, not just the identity.
    return f"qwen35_convergence_{model_slug(model)}_{arm}_seq{max_seq_length}_B{batch_size}"


def build_condition(
    *, model: str, revision: str | None, data: str, dataset_sha: str, arm: str,
    max_seq_length: int, batch_size: int, steps: int, val_every: int, val_examples: int,
    lora_rank: int, lora_layers: int, learning_rate: float, seed: int,
    grad_checkpoint: bool,
) -> ConvergenceCondition:
    return ConvergenceCondition(
        name=condition_name(
            model=model, arm=arm, max_seq_length=max_seq_length, batch_size=batch_size,
        ),
        model=model, revision=revision, data=data, dataset_sha=dataset_sha, arm=arm,
        max_seq_length=max_seq_length, batch_size=batch_size, steps=steps,
        val_every=val_every, val_examples=val_examples, lora_rank=lora_rank,
        lora_layers=lora_layers, learning_rate=learning_rate, seed=seed,
        grad_checkpoint=grad_checkpoint,
    )


def _params_for(condition: ConvergenceCondition) -> dict[str, object]:
    return {
        "model": condition.model, "revision": condition.revision, "data": condition.data,
        "dataset_sha": condition.dataset_sha, "arm": condition.arm,
        "max_seq_length": condition.max_seq_length, "batch_size": condition.batch_size,
        "steps": condition.steps, "val_every": condition.val_every,
        "val_examples": condition.val_examples, "lora_rank": condition.lora_rank,
        "lora_layers": condition.lora_layers, "learning_rate": condition.learning_rate,
        "seed": condition.seed, "grad_checkpoint": condition.grad_checkpoint,
        "compute_dtype": "bfloat16", "script_sha": script_sha(),
    }


def _identity_for(condition: ConvergenceCondition, *, session_id: str) -> dict[str, object]:
    return condition_identity(
        kind="qwen35_gated_delta_convergence", session_id=session_id,
        params=_params_for(condition),
    )


def _out_path(out_dir: Path, condition: ConvergenceCondition) -> Path:
    return out_dir / f"{condition.name}.json"


def _cast_bf16_protecting_a_log(model: Any) -> None:
    """Cast every floating parameter to bf16 IN PLACE, except `A_log`, honouring the
    model's own path-keyed `cast_predicate` -- see the module docstring for why a bare
    `model.set_dtype(mx.bfloat16)` cannot do this. Mirrors `mlx_lm/convert.py`'s own
    `tree_map_with_path` + `cast_predicate` composition. Asserts `A_log` stayed fp32 on
    every linear-attention layer afterward, so a regression here fails loudly instead of
    silently biasing the comparison."""
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


def _load_and_prepare_model(condition: ConvergenceCondition) -> Any:
    """Load, cast, (conditionally) enable, freeze, and LoRA-inject -- same load-bearing
    order as `bench_qwen35_training.py`'s `_load_and_prepare_model` (see its docstring):
    cast before enable, enable before freeze/LoRA, `mx.eval` after every lazy setup
    step."""
    import mlx_lm  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    model, _tokenizer = cast(
        "tuple[Any, Any]", mlx_lm.load(condition.model, revision=condition.revision)
    )
    _cast_bf16_protecting_a_log(model)
    mx.eval(model.parameters())
    if condition.arm == "enabled":
        enable_gated_delta_training(model, impl="chunked")
    mx.random.seed(condition.seed)  # BEFORE freeze/LoRA-injection: their init draws next
    model.freeze()
    linear_to_lora_layers(
        model, condition.lora_layers,
        {"rank": condition.lora_rank, "dropout": 0.0, "scale": 20.0},
    )
    mx.eval(model.parameters())
    return model


def _make_reasserting_loss(
    loss_fn: Any, observed_before: list[int],
) -> Any:
    """Wraps a trainer loss callable so its FIRST invocation re-asserts this project's
    house wired cap before doing anything else -- same reasoning as
    `bench_qwen35_training.py`'s own `_make_reasserting_loss`."""

    def wrapped(model: Any, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        if not observed_before:
            observed_before.append(install_guardrails())
        return cast("tuple[mx.array, mx.array]", loss_fn(model, batch, lengths))

    return wrapped


def _load_dataset(path: str) -> list[tuple[list[int], int]]:
    """Read a prep_alpaca jsonl -- one `{"tokens": [...], "offset": N}` object per line
    -- into the `(tokens, offset)` pairs `iterate_batches` consumes."""
    dataset: list[tuple[list[int], int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            dataset.append(([int(t) for t in obj["tokens"]], int(obj["offset"])))
    return dataset


def split_held_out(
    pairs: list[tuple[list[int], int]], *, val_examples: int,
) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    """Deterministic prefix split: the first `val_examples` pairs are held out for
    validation, the rest train. Pure -- the SAME split for any caller handed the same
    `pairs` list, so "held out" means the same physical examples for both arms.
    Returns `(train_pairs, val_pairs)`."""
    val_pairs = pairs[:val_examples]
    train_pairs = pairs[val_examples:]
    return train_pairs, val_pairs


class _PairDataset:
    """`train()`/`evaluate()` hand their dataset argument straight to `iterate_batches`
    unwrapped, so `dataset[idx]` must return the `(tokens, offset)` pair directly."""

    def __init__(self, pairs: list[tuple[list[int], int]]) -> None:
        self._pairs = pairs

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self._pairs[idx]

    def __len__(self) -> int:
        return len(self._pairs)


class _FlushingCallback:
    """Duck-typed `mlx_lm.tuner.callbacks.TrainingCallback` that flushes the FULL
    accumulated train/val history to disk on EVERY call -- see the module docstring for
    why this bounds the loss of an interrupted run to at most the step in flight.
    Intermediate writes carry status `"in_progress"` (not `"ok"`), so
    `bench.artifacts.result_is_fresh` never treats a partial run as complete."""

    def __init__(self, out_path: Path, identity: dict[str, object]) -> None:
        self._out_path = out_path
        self._identity = identity
        self.train_info: list[dict[str, object]] = []
        self.val_info: list[dict[str, object]] = []

    def _flush(self) -> None:
        write_result(
            self._out_path, self._identity, "in_progress",
            train_info=self.train_info, val_info=self.val_info,
        )

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_info.append(dict(train_info))
        self._flush()

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:
        self.val_info.append(dict(val_info))
        self._flush()


def run_convergence(
    condition: ConvergenceCondition, *, out_path: Path, identity: dict[str, object],
) -> dict[str, object]:
    """The worker body: load + prepare the model for `condition.arm`, drive real
    fine-tune iterations on real Alpaca data with `train()`'s own built-in validation
    schedule, and read back the wired-cap/memory story. Guardrails are installed by the
    CALLER (`_run_single_condition`) before this runs."""
    _require_mlx_lm()
    import mlx.optimizers as optim  # noqa: PLC0415
    from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train  # noqa: PLC0415

    pairs = _load_dataset(condition.data)
    train_pairs, val_pairs = split_held_out(pairs, val_examples=condition.val_examples)

    model = _load_and_prepare_model(condition)
    base_loss = make_loss_fn(model, impl="auto")
    observed_before: list[int] = []
    loss_fn = _make_reasserting_loss(base_loss, observed_before)

    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    expected_wired, _soft = clamped_caps(dev_max)
    opt = optim.Adam(learning_rate=condition.learning_rate)

    callback = _FlushingCallback(out_path, identity)
    # Seeds BOTH the len-sorted batch grouping's processing order (training) and the
    # validation batch order -- the SAME seed across two invocations therefore means the
    # SAME example order for both arms (see the module docstring).
    seeded_iterate = functools.partial(iterate_batches, seed=condition.seed)

    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    with tempfile.TemporaryDirectory(prefix="mlx-train-perf-convergence-") as tmp_dir:
        args = TrainingArgs(
            batch_size=condition.batch_size, iters=condition.steps,
            val_batches=-1,  # -1 == the entire held-out set, every validation pass
            steps_per_report=1, steps_per_eval=condition.val_every,
            steps_per_save=condition.steps + 1, max_seq_length=condition.max_seq_length,
            grad_checkpoint=condition.grad_checkpoint,
            adapter_file=str(Path(tmp_dir) / "adapters.safetensors"),
        )
        train(
            model=model, optimizer=opt, train_dataset=_PairDataset(train_pairs),
            val_dataset=_PairDataset(val_pairs), args=args, loss=loss_fn,
            iterate_batches=cast(Any, seeded_iterate), training_callback=cast(Any, callback),
        )
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3

    observed_after = install_guardrails()
    if not wired_cap_holds(observed_bytes=observed_after, expected_bytes=expected_wired):
        raise WiredCapRegressionError(
            f"qwen35 convergence condition's wired limit was {observed_after / 1024**3:.2f} "
            f"GB after training, expected the house cap {expected_wired / 1024**3:.2f} GB "
            "-- mlx_lm.tuner.trainer.train()'s entry-time override was not correctly "
            "re-asserted"
        )

    return {
        "train_info": callback.train_info, "val_info": callback.val_info,
        "num_train_examples": len(train_pairs), "num_val_examples": len(val_pairs),
        "active_before_gb": round(active_before / 1024**3, 4),
        "marginal_peak_gb": round(marginal_peak_gb, 4),
        "total_peak_gb": round(active_before / 1024**3 + marginal_peak_gb, 4),
        "observed_wired_limit_gb": round(observed_after / 1024**3, 4),
        "house_wired_limit_gb": round(expected_wired / 1024**3, 4),
        "wired_limit_before_reassert_gb": (
            round(observed_before[0] / 1024**3, 4) if observed_before else None
        ),
        "validation_implementation_note": _VAL_IMPLEMENTATION_DISCLOSURE,
    }


def _run_single_condition(
    condition: ConvergenceCondition, *, out_path: Path, identity: dict[str, object],
) -> dict[str, object]:
    install_guardrails()  # FIRST -- before any allocation this condition makes
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    try:
        ceiling = effective_memory_ceiling()
    except MemoryBudgetError as exc:
        # Too crowded to START safely is an ENVIRONMENT-transient outcome -- its own
        # status, distinct from a crash. Only "ok" artifacts are fresh on resume, so a
        # later, quieter invocation re-runs this condition automatically.
        write_result(out_path, identity, "refused_environment", error=str(exc))
        return cast("dict[str, object]", json.loads(out_path.read_text()))
    warning_field: dict[str, object] = (
        {"memory_warning": ceiling.warning} if ceiling.warning is not None else {}
    )
    watchdog = install_memory_watchdog(
        ceiling_bytes=ceiling.ceiling_bytes,
        sampler=lambda: mx.get_active_memory() + mx.get_cache_memory(),
        interval_s=0.05,
        wall_budget_s=DEFAULT_WALL_BUDGET_S,
        on_breach=make_watchdog_on_breach(out_path, identity, ceiling.ceiling_bytes),
    )
    try:
        fields = run_convergence(condition, out_path=out_path, identity=identity)
        write_result(out_path, identity, "ok", **fields, **warning_field)
    finally:
        # A breach never reaches here -- `on_breach` already hard-exited the process.
        watchdog.stop()
    return cast("dict[str, object]", json.loads(out_path.read_text()))


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _comparable_identity(identity: dict[str, object]) -> dict[str, object]:
    """Every identity field EXCEPT `arm`/`session_id` -- the axis being compared, and
    the per-invocation token that two independent, per-arm subprocess runs never
    share."""
    return {k: v for k, v in identity.items() if k not in ("arm", "session_id")}


def _series(entries: list[dict[str, object]], *, value_key: str) -> dict[int, float]:
    """`{iteration: value}` extracted from a list of train/val report dicts -- skips
    any entry missing either field rather than raising, so a malformed/partial artifact
    still yields whatever it can."""
    out: dict[int, float] = {}
    for entry in entries:
        it = entry.get("iteration")
        value = entry.get(value_key)
        if isinstance(it, int) and isinstance(value, int | float):
            out[it] = float(value)
    return out


def build_comparison(stock_path: Path, enabled_path: Path) -> dict[str, object]:
    """Fold two already-measured arm artifacts into one comparison report. Never
    raises -- a missing/corrupt/incomplete artifact, or two artifacts whose identity
    disagrees on anything but `arm`/`session_id`, all produce a `status` field
    explaining why the comparison fields are absent, rather than crashing."""
    result: dict[str, object] = {"validation_implementation_note": _VAL_IMPLEMENTATION_DISCLOSURE}
    stock = _read_json(stock_path)
    enabled = _read_json(enabled_path)
    if stock is None or enabled is None:
        result["status"] = "corrupt"
        return result
    if stock.get("status") != "ok" or enabled.get("status") != "ok":
        result["status"] = "incomplete"
        result["stock_status"] = stock.get("status")
        result["enabled_status"] = enabled.get("status")
        return result
    stock_ident = cast("dict[str, object]", stock.get("identity", {}))
    enabled_ident = cast("dict[str, object]", enabled.get("identity", {}))
    stock_comparable = _comparable_identity(stock_ident)
    enabled_comparable = _comparable_identity(enabled_ident)
    if stock_comparable != enabled_comparable:
        differing = sorted(
            k for k in set(stock_comparable) | set(enabled_comparable)
            if stock_comparable.get(k) != enabled_comparable.get(k)
        )
        result["status"] = "identity_mismatch"
        result["differing_identity_fields"] = differing
        return result
    result["status"] = "ok"
    stock_train = cast("list[dict[str, object]]", stock.get("train_info", []))
    enabled_train = cast("list[dict[str, object]]", enabled.get("train_info", []))
    stock_val = cast("list[dict[str, object]]", stock.get("val_info", []))
    enabled_val = cast("list[dict[str, object]]", enabled.get("val_info", []))
    result["train_loss_by_iteration"] = {
        "stock": _series(stock_train, value_key="train_loss"),
        "enabled": _series(enabled_train, value_key="train_loss"),
    }
    result["val_loss_by_iteration"] = {
        "stock": _series(stock_val, value_key="val_loss"),
        "enabled": _series(enabled_val, value_key="val_loss"),
    }
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="HF repo id or local path")
    ap.add_argument("--revision", default=None, help="model checkpoint revision")
    ap.add_argument("--data", default=None, help="prep_alpaca jsonl path")
    ap.add_argument("--arm", choices=_ARMS, default=None,
                    help="one arm per invocation: 'stock' (mlx-lm's own GatedDeltaNet) "
                        "or 'enabled' (this project's chunked op via "
                        "enable_gated_delta_training)")
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="total training iterations")
    ap.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY,
                    help="steps between validation passes (train() ALSO always "
                        "validates before the first step and after the last)")
    ap.add_argument("--val-examples", type=int, default=DEFAULT_VAL_EXAMPLES,
                    help="held-out prefix of --data reserved for validation")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 == all layers")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (the realistic long-context QLoRA setup)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: this script's own _artifacts "
                        "subdirectory, one per arm)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the condition and resolve the artifact path/identity "
                        "WITHOUT touching MLX or mlx_lm -- no model load, no GPU "
                        "allocation, no dataset row parsed, nothing written")
    ap.add_argument("--compare", action="store_true",
                    help="fold two already-measured arm artifacts into one comparison "
                        "report instead of running a new measurement")
    ap.add_argument("--stock-artifact", type=Path, default=None,
                    help="path to the 'stock' arm's artifact JSON (required with --compare)")
    ap.add_argument("--enabled-artifact", type=Path, default=None,
                    help="path to the 'enabled' arm's artifact JSON (required with --compare)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.compare:
        if args.stock_artifact is None or args.enabled_artifact is None:
            raise SystemExit("--compare requires --stock-artifact and --enabled-artifact")
        report = build_comparison(args.stock_artifact, args.enabled_artifact)
        print(json.dumps(report, indent=2))
        return 0 if report.get("status") == "ok" else 1

    if args.model is None or args.data is None or args.arm is None:
        raise SystemExit("--model, --data, and --arm are required unless --compare is set")

    data_path = Path(args.data)
    condition = build_condition(
        model=args.model, revision=args.revision, data=str(data_path),
        dataset_sha=dataset_sha(data_path), arm=args.arm,
        max_seq_length=args.max_seq_length, batch_size=args.batch_size, steps=args.steps,
        val_every=args.val_every, val_examples=args.val_examples, lora_rank=args.lora_rank,
        lora_layers=args.lora_layers, learning_rate=args.learning_rate, seed=args.seed,
        grad_checkpoint=args.grad_checkpoint,
    )
    out_dir = args.out if args.out is not None else RESULTS / condition.arm
    out_path = _out_path(out_dir, condition)
    session_id = new_session_id()
    ident = _identity_for(condition, session_id=session_id)

    if args.dry_run:
        print(json.dumps(
            {"name": condition.name, "out_path": str(out_path), "identity": ident}, indent=2,
        ))
        return 0

    result = _run_single_condition(condition, out_path=out_path, identity=ident)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
