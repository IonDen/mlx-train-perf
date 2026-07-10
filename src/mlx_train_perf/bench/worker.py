"""Bench worker: the subprocess entry point for exactly one `Condition`, invoked by
`runner.run_conditions` as `python -m mlx_train_perf.bench.worker --config <path>`.

Guardrail-reassert note for the `train_step` condition kind: `mlx_lm.tuner.trainer.
train()` raises the wired limit to the device max AT ENTRY (site-packages/mlx_lm/
tuner/trainer.py:229) and then blocks until the training loop finishes. A worker that
calls `install_guardrails()` once before `train()` and again after it returns protects
nothing in between -- the stricter cap is silently overridden for the entire run. The
re-assert has to live INSIDE the loop: the loss callable handed to `train(...)` calls
`install_guardrails()` on its own first invocation (a one-shot flag --
`_make_reasserting_loss` below), and `run_train_step` reads the wired limit back
(`install_guardrails`'s own return value) once more after `train()` returns to CONFIRM
the cap actually held throughout -- `wired_cap_holds`
(`mlx_train_perf.core.guards`) is the pure decision that comparison reduces to, and a
mismatch raises `WiredCapRegressionError` rather than silently reporting numbers
measured under an uncapped run.

`train()` always wraps its per-step function in `mx.compile`, which forbids evaluating
an array during tracing. `make_loss_fn` (the `ours` loss) passes `validate_targets=False`
to `linear_cross_entropy`, skipping the one per-call host sync (`_validate_inputs`'s
out-of-range-target check) that would otherwise break the trace -- safe because the
trainer feeds in-range tokenizer ids. So BOTH arms (`ours` and stock's `default_loss`)
run through the real compiled `train()` step, and the tok/s comparison is compiled vs
compiled, apples to apples.
"""
import argparse
import json
import random
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx

from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.attention.wrapper import enable_flash_attention
from mlx_train_perf.bench.artifacts import condition_identity, write_result
from mlx_train_perf.core.guards import clamped_caps, install_guardrails, wired_cap_holds
from mlx_train_perf.core.loss import DenseHead, HeadRef, QuantizedHead, linear_cross_entropy
from mlx_train_perf.errors import (
    LaunchBudgetError,
    MissingDependencyError,
    MlxTrainPerfError,
    WiredCapRegressionError,
)

_ATTENTION_IMPLS = ("stock", "flash")

_DTYPES: dict[str, mx.Dtype] = {
    "float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16,
}


def _resolve_dtype(name: str) -> mx.Dtype:
    if name not in _DTYPES:
        raise MlxTrainPerfError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def _build_head(*, v: int, d: int, dtype: mx.Dtype, quantized: bool, group_size: int,
                bits: int, seed: int) -> HeadRef:
    mx.random.seed(seed)
    w = (mx.random.normal((v, d)) * 0.02).astype(dtype)
    mx.eval(w)
    if not quantized:
        return DenseHead(weight=w)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(w_q, scales, biases)
    return QuantizedHead(w_q=w_q, scales=scales, biases=biases, group_size=group_size, bits=bits)


def run_loss_layer(params: dict[str, object]) -> dict[str, object]:
    """Times `linear_cross_entropy` at one synthetic grid point. Reset-peak semantics
    (warmup pays Metal JIT OUTSIDE the measured window; `active_before` is snapshotted
    right before the reset so `marginal_peak_gb` is the incremental cost of the forward
    passes themselves) -- the same convention `scripts/bench_quant_thresholds.py` uses."""
    n = int(cast(int, params["n"]))
    d = int(cast(int, params["d"]))
    v = int(cast(int, params["v"]))
    dtype = _resolve_dtype(str(params.get("dtype", "bfloat16")))
    impl = cast(Literal["auto", "kernel", "chunked", "naive"], params.get("impl", "auto"))
    quantized = bool(params.get("quantized", False))
    group_size = int(cast(int, params.get("group_size", 64)))
    bits = int(cast(int, params.get("bits", 4)))
    chunk_size = cast(int | None, params.get("chunk_size"))
    reps = int(cast(int, params.get("reps", 3)))
    seed = int(cast(int, params.get("seed", 0)))

    mx.random.seed(seed)
    hidden = mx.random.normal((n, d)).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    mx.eval(hidden, targets)
    head = _build_head(v=v, d=d, dtype=dtype, quantized=quantized, group_size=group_size,
                       bits=bits, seed=seed + 1)

    def run_once() -> mx.array:
        return linear_cross_entropy(hidden, head, targets, impl=impl, chunk_size=chunk_size,
                                    reduction="mean")

    loss = run_once()
    mx.eval(loss)
    mx.clear_cache()
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    walls: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        loss = run_once()
        mx.eval(loss)
        walls.append(time.perf_counter() - t0)
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3
    med = statistics.median(walls)
    g_mac_per_s = (n * v * d) / med / 1e9
    return {
        "wall_s": round(med, 6),
        "wall_s_all": [round(x, 6) for x in walls],
        "g_mac_per_s": round(g_mac_per_s, 3),
        "active_before_gb": round(active_before / 1024**3, 4),
        "marginal_peak_gb": round(marginal_peak_gb, 4),
        "total_peak_gb": round(active_before / 1024**3 + marginal_peak_gb, 4),
    }


# ---------------------------------------------------------------------------------
# train_step: N real mlx-lm LoRA fine-tune steps end to end -- ours (via the T12
# adapter, `make_loss_fn`) or stock (`mlx_lm.tuner.trainer.default_loss`), selected by
# `params["stock"]`.
# ---------------------------------------------------------------------------------


def _require_mlx_lm() -> None:
    """Mirrors `adapters.mlx_lm._require_mlx_lm`'s check -- duplicated rather than
    imported across modules since it is a tiny, independent fail-fast guard, not
    shared logic. `_load_model` below is the ONLY thing in this function's OWN control
    flow that would otherwise raise a raw `ImportError`; everything else that needs
    `mlx_lm` (the adapter, `mlx_lm.tuner.*`) reaches its own lazy import naturally."""
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "mlx-lm is required for the train_step bench condition; install the "
            "optional 'mlx-lm' extra (pip install 'mlx-train-perf[mlx-lm]')"
        ) from exc


def _load_model(model_id: str, revision: str | None) -> tuple[Any, Any]:
    """The sole call site that touches `mlx_lm.load` (a real repo/checkpoint read) --
    isolated so tests can monkeypatch this one function and hand back a tiny,
    freshly-constructed synthetic model instead of a real downloaded checkpoint. Cast
    to `tuple[Any, Any]` deliberately -- `mlx_lm`'s own real (model, tokenizer) types
    are not imported anywhere in this project (see `adapters/mlx_lm.py`'s identical
    `model: Any` convention); this project verifies model support structurally, not
    by type."""
    import mlx_lm  # noqa: PLC0415
    return cast("tuple[Any, Any]", mlx_lm.load(model_id, revision=revision))


def _synthetic_train_examples(
    *, vocab_size: int, seq_len: int, num_examples: int, seed: int,
) -> list[list[int]]:
    """`num_examples` fixed-length (`seq_len + 1`, matching `train()`'s own
    `inputs=batch[:, :-1]` / `targets=batch[:, 1:]` shift) random-token rows. Plain
    `random.Random`, never `mx.random` -- dataset construction stays independent of
    MLX's global RNG state, which `run_train_step` reseeds separately (`mx.random.
    seed`) right before the LoRA layers' own random-init draw. With `num_examples ==
    batch` (this function's only caller), `iterate_batches` (mlx_lm/tuner/trainer.py)
    ends up with exactly ONE batch group, and `np.random.permutation` of a single-
    element array is deterministic regardless of numpy's global RNG state -- so every
    training step reuses the identical batch, and two conditions built from the same
    `seed` see byte-identical training data at every step. That is what makes the
    per-step loss dump a real parity check against stock's own masking/`ntoks`
    denominator, not an approximate one."""
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(seq_len + 1)] for _ in range(num_examples)
    ]


class _SyntheticDataset:
    """`train()` (unlike `mlx_lm.lora.train_model`, which wraps its dataset argument
    in `CacheDataset`) hands its `train_dataset` STRAIGHT to `iterate_batches`
    unwrapped -- so `isinstance(dataset, CacheDataset)` is false, and
    `iterate_batches`'s own `len_fn = lambda idx: len(dataset[idx][0])` reads
    `dataset[idx]` as the FINAL `(tokens, offset)` pair directly (mlx_lm/tuner/
    trainer.py:113-114). `__getitem__` therefore returns that pair immediately --
    `offset=0` always, every synthetic row fully unmasked -- with no separate
    `process()`/caching indirection (`CacheDataset`'s own job, which does not apply
    here)."""

    def __init__(self, examples: list[list[int]]) -> None:
        self._examples = examples

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self._examples[idx], 0

    def __len__(self) -> int:
        return len(self._examples)


class _RecordingCallback:
    """Duck-typed `mlx_lm.tuner.callbacks.TrainingCallback` -- `train()` only ever
    calls these two methods by name, never `isinstance`-checks, so no inheritance (and
    no module-level `mlx_lm` import) is needed. Collects every per-step report
    (`steps_per_report=1` means exactly one call per training iteration).
    `on_val_loss_report` is never invoked: `run_train_step` always passes an empty
    `val_dataset`, which `train()`'s own `if val_dataset and (...)` guard treats as "no
    validation configured"."""

    def __init__(self) -> None:
        self.train_info: list[dict[str, object]] = []

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_info.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:  # noqa: ARG002
        # `val_info` unused: required for interface parity with `TrainingCallback` --
        # `train()` calls this positionally; never actually invoked (val_dataset=[]).
        return None  # pragma: no cover


def _make_reasserting_loss(
    loss_fn: Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]],
    observed_before: list[int],
) -> Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Wraps a trainer loss callable (ours, via `make_loss_fn`, or stock's own
    `default_loss`) so its FIRST invocation re-asserts this project's house wired cap
    before doing anything else -- see the module docstring for why this must live
    INSIDE the loop rather than around the `train()` call. `mx.compile` only re-runs a
    traced Python body on a recompile, so firing once is sufficient: the reassert
    holds for the rest of the run once it has fired. `observed_before` is a
    caller-owned single-element-once-fired list, doubling as both the one-shot flag
    (empty == not fired yet) and the captured diagnostic value (the wired limit that
    was active immediately before the reassert -- expected to be the device max
    `train()` just set at its own entry)."""

    def wrapped(model: Any, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        if not observed_before:
            observed_before.append(install_guardrails())
        return loss_fn(model, batch, lengths)

    return wrapped


def _run_train_steps(
    model: Any,
    optimizer: Any,
    loss_fn: Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]],
    examples: list[list[int]],
    *,
    batch: int,
    seq_len: int,
    steps: int,
    grad_checkpoint: bool,
) -> list[dict[str, object]]:
    """Drives `steps` real fine-tune iterations through the compiled
    `mlx_lm.tuner.trainer.train()` -- used for BOTH arms. Stock's `default_loss` and
    ours (via `make_loss_fn`, which passes `validate_targets=False`) are both free of the
    per-call host sync `mx.compile` forbids, so both run through the real compiled step
    and the tok/s comparison is apples to apples."""
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415

    callback = _RecordingCallback()
    train_set = _SyntheticDataset(examples)
    with tempfile.TemporaryDirectory(prefix="mlx-train-perf-bench-") as tmp_dir:
        args = TrainingArgs(
            batch_size=batch, iters=steps, val_batches=0, steps_per_report=1,
            steps_per_eval=steps + 1, steps_per_save=steps + 1,
            max_seq_length=seq_len + 1, grad_checkpoint=grad_checkpoint,
            adapter_file=str(Path(tmp_dir) / "adapters.safetensors"),
        )
        # `_RecordingCallback` is duck-typed against `mlx_lm.tuner.callbacks.
        # TrainingCallback` deliberately (see its own docstring) rather than
        # inheriting from it -- avoids a module-level `mlx_lm` import for a class
        # this project only ever hands to `train()`, which never `isinstance`-checks.
        train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=[],
              args=args, loss=loss_fn, training_callback=cast(Any, callback))
    return callback.train_info


def run_train_step(params: dict[str, object]) -> dict[str, object]:
    """Times `steps` real mlx-lm LoRA fine-tune steps end to end against a real
    (`mlx_lm.load`-resolved) model: ours, via the T12 adapter (`make_loss_fn`), or
    stock's own `mlx_lm.tuner.trainer.default_loss` when `params["stock"]` is true.
    Records tokens/sec (median + per-step), per-step loss, and the memory story
    (active-before / marginal-peak / total-peak, matching `run_loss_layer`'s own
    convention) plus the wired-limit contract this module's docstring describes --
    `WiredCapRegressionError` if the observed limit, read back after training, does
    not match this project's house cap.

    Both `ours` (via `make_loss_fn`) and `stock` (`default_loss`) run through the real,
    compiled `mlx_lm.tuner.trainer.train()` -- `make_loss_fn` passes
    `validate_targets=False` so ours is free of the host sync `mx.compile` forbids, so the
    two arms are compared compiled-vs-compiled. Only LoRA fine-tuning is
    modeled (mlx-lm's own `linear_to_lora_layers`, zero-config default target set,
    applied identically to both `ours` and `stock`) -- full fine-tuning
    (`lora_rank=0`) is out of scope for this condition kind.

    `params`: `model` (str, repo id or local path, required), `revision` (str | None,
    default None), `seq_len`/`batch`/`steps` (int, required), `lora_rank` (int, default
    8), `lora_layers` (int, default -1 == all layers), `impl` (default "auto",
    ignored when `stock`), `stock` (bool, default False), `learning_rate` (float,
    default 1e-5), `seed` (int, default 0), `grad_checkpoint` (bool, default False),
    `compute_dtype` (str | None, default None -- when set, e.g. "bfloat16", the loaded
    model's floating params are cast to it before training; the kernel `impl` needs
    this on 4-bit models that otherwise compute in fp16), `attention_impl` ("stock"
    default | "flash" -- "flash" routes every decoder layer's attention through T12's
    `enable_flash_attention`, hinted with THIS run's `seq_len`/`batch` so a compiled
    `train()` traces with warm kernel rate caches; an unknown value raises
    `MlxTrainPerfError`).
    """
    _require_mlx_lm()
    import mlx.optimizers as optim  # noqa: PLC0415
    from mlx_lm.tuner.trainer import default_loss  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    model_id = str(params["model"])
    revision = cast("str | None", params.get("revision"))
    seq_len = int(cast(int, params["seq_len"]))
    batch = int(cast(int, params["batch"]))
    steps = int(cast(int, params["steps"]))
    lora_rank = int(cast(int, params.get("lora_rank", 8)))
    lora_layers = int(cast(int, params.get("lora_layers", -1)))
    impl = cast(Literal["auto", "kernel", "chunked", "naive"], params.get("impl", "auto"))
    stock = bool(params.get("stock", False))
    learning_rate = float(cast(float, params.get("learning_rate", 1e-5)))
    seed = int(cast(int, params.get("seed", 0)))
    grad_checkpoint = bool(params.get("grad_checkpoint", False))
    compute_dtype = cast("str | None", params.get("compute_dtype"))
    attention_impl = str(params.get("attention_impl", "stock"))
    if attention_impl not in _ATTENTION_IMPLS:
        raise MlxTrainPerfError(
            f"unknown attention_impl {attention_impl!r}; expected one of "
            f"{_ATTENTION_IMPLS}"
        )

    model, _tokenizer = _load_model(model_id, revision)
    if compute_dtype is not None:
        # Cast the loaded model's FLOATING params (a 4-bit checkpoint's int4 weights
        # stay int4 -- `set_dtype`'s default predicate skips non-floating params -- while
        # scales/biases/norms move to `compute_dtype`, so the trunk produces
        # `compute_dtype` hidden states). Needed for the kernel `impl`, which accepts
        # only fp32/bf16 hidden, on the 4-bit models that otherwise compute in fp16
        # (they store fp16 scales, overriding a config `torch_dtype=bf16`). Applied
        # BEFORE freeze/LoRA so it does not touch the fp32 LoRA adapters injected next,
        # and identically to both arms when the caller sets it -- holding the trunk
        # dtype constant so the ours-vs-stock tok/s + loss-curve comparison isolates the
        # loss layer. `set_dtype` is a pure `astype` (draws no RNG), so the seed->LoRA-
        # init determinism below is undisturbed.
        model.set_dtype(_resolve_dtype(compute_dtype))
    if attention_impl == "flash":
        # `enable_flash_attention` runs its OWN real calibration forward + `mx.eval`
        # (T12's `_prewarm_rate_caches`, hinted with THIS run's seq_len/batch so a
        # compiled `train()` below traces with warm kernel rate caches -- no in-trace
        # host-sync). The compute_dtype cast above must already be MATERIALIZED
        # (gotcha 14) before that calibration runs, rather than get forced as an
        # incidental side effect of the calibration's own eval. It also MUST run
        # before `model.freeze()`/`linear_to_lora_layers` below: LoRA target
        # discovery walks `named_modules()` by path (`self_attn.q_proj`), so the
        # wrapper has to already be in the tree at injection time for LoRA to land
        # inside it (`FlashAttentionWrapper`'s own docstring, reason 1).
        mx.eval(model.parameters())
        enable_flash_attention(model, seq_len=seq_len, batch_size=batch)
    mx.random.seed(seed)  # BEFORE freeze/LoRA-injection: their random init draws next
    model.freeze()
    linear_to_lora_layers(
        model, lora_layers, {"rank": lora_rank, "dropout": 0.0, "scale": 20.0}
    )
    # Force the lazy setup graphs NOW, before the measurement window below: `set_dtype`
    # is a lazy `apply`/astype and LoRA's random-init is equally lazy, so without this
    # eval the one-time cast + adapter init would execute inside the
    # `reset_peak_memory()` window and leak setup cost into `marginal_peak_gb` (and
    # into step 1's wall time).
    mx.eval(model.parameters())

    base_loss = default_loss if stock else make_loss_fn(model, impl=impl)
    observed_before: list[int] = []
    loss_fn = _make_reasserting_loss(base_loss, observed_before)

    vocab_size = int(model.args.vocab_size)
    examples = _synthetic_train_examples(
        vocab_size=vocab_size, seq_len=seq_len, num_examples=batch, seed=seed,
    )

    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    expected_wired, _soft = clamped_caps(dev_max)

    opt = optim.Adam(learning_rate=learning_rate)

    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    # Both arms run through the real compiled `train()`: stock's `default_loss` and ours
    # (via `make_loss_fn`, `validate_targets=False`) are both free of the host sync
    # `mx.compile` forbids, so the tok/s comparison is compiled-vs-compiled, apples to apples.
    step_reports = _run_train_steps(
        model, opt, loss_fn, examples, batch=batch, seq_len=seq_len, steps=steps,
        grad_checkpoint=grad_checkpoint,
    )
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3

    observed_after = install_guardrails()
    if not wired_cap_holds(observed_bytes=observed_after, expected_bytes=expected_wired):
        raise WiredCapRegressionError(
            f"train_step condition's wired limit was {observed_after / 1024**3:.2f} GB "
            f"after training, expected the house cap {expected_wired / 1024**3:.2f} GB "
            "-- mlx_lm.tuner.trainer.train()'s entry-time override was not correctly "
            "re-asserted (see this module's docstring)"
        )

    tokens_per_sec_all = [
        float(cast(float, info["tokens_per_second"])) for info in step_reports
    ]
    loss_all = [float(cast(float, info["train_loss"])) for info in step_reports]
    return {
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mlx_train_perf.bench.worker")
    ap.add_argument("--config", required=True, help="path to a JSON condition config")
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    kind = str(config["kind"])
    params = cast(dict[str, object], config["params"])
    session_id = str(config["session_id"])
    out = Path(cast(str, config["out"]))

    install_guardrails()  # FIRST -- before any allocation this condition makes

    ident = condition_identity(kind=kind, session_id=session_id, params=params)
    try:
        if kind == "loss_layer":
            fields = run_loss_layer(params)
        elif kind == "train_step":
            fields = run_train_step(params)
        else:
            # Deliberately uncaught: an unsupported kind is a program error (a bad
            # Condition was constructed), not a recorded run outcome -- it crashes this
            # worker process with a nonzero exit, and `runner.run_conditions` records
            # the failure envelope on the CALLER's side instead of this worker writing
            # anything. Referencing the bare `run_loss_layer`/`run_train_step` names
            # above (not a dict bound at import time) also keeps this dispatch
            # monkeypatch-friendly for tests.
            raise MlxTrainPerfError(
                f"unsupported bench condition kind {kind!r}; expected 'loss_layer' or "
                "'train_step'"
            )
    except LaunchBudgetError as exc:
        # A guard refusal IS a result: the calibrated rate cannot serve this shape
        # within the watchdog budget -- record it, don't crash the sweep. A
        # WiredCapRegressionError is a DIFFERENT, more serious failure (a condition
        # that measured under an uncapped run) and is deliberately NOT caught here --
        # it propagates the same way an unsupported kind does.
        write_result(out, ident, "refused", error=str(exc))
        return 0
    write_result(out, ident, "ok", **fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
