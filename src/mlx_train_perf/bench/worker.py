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
import functools
import json
import random
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx

from mlx_train_perf.adapters.mlx_lm import make_loss_fn, make_packed_loss_fn
from mlx_train_perf.attention.wrapper import enable_flash_attention
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    make_watchdog_on_breach,
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
from mlx_train_perf.core.loss import DenseHead, HeadRef, QuantizedHead, linear_cross_entropy
from mlx_train_perf.data.packing import (
    packed_batching_stats,
    packed_iterate_batches,
    stock_batching_stats,
)
from mlx_train_perf.errors import (
    LaunchBudgetError,
    MemoryBudgetError,
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


def run_train_step(
    params: dict[str, object], *, attention_impl: str | None = None,
) -> dict[str, object]:
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
    this on 4-bit models that otherwise compute in fp16).

    `attention_impl` is a DEDICATED keyword (NOT a `params` entry): the same reserved
    identity input `bench.runner.Condition` carries out of `params` and `worker.main`
    forwards here, so identity and execution read one authoritative value. `None`
    (unset -- every 0.1.0-era config) and "stock" both leave attention untouched; "flash"
    routes every decoder layer's attention through T12's `enable_flash_attention`, hinted
    with THIS run's `seq_len`/`batch` so a compiled `train()` traces with warm kernel rate
    caches. Any other value raises `MlxTrainPerfError`.
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
    # Unset (`None`, every 0.1.0-era config) resolves to the stock attention path -- the
    # dedicated keyword is the ONE authoritative source; `params` never carries it (the
    # identity's reserved-key guard rejects that).
    if attention_impl is None:
        attention_impl = "stock"
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


# ---------------------------------------------------------------------------------
# packed_train: the 0.4.0 sequence-packing throughput bench. ONE arm per condition --
# `stock` (unpacked `make_loss_fn` + stock `iterate_batches`) or `packed`
# (`make_packed_loss_fn` + `packed_iterate_batches`). BOTH arms enable flash attention
# and the fused CE loss, so the ONLY variable is the batching strategy. Drives the real,
# compiled `mlx_lm.tuner.trainer.train()` against a REAL prepped dataset (scripts/
# prep_alpaca.py), and reports real (non-pad) tokens/s + samples/hour by combining the
# measured median step wall with deterministic host-side per-step content counts
# (data.packing.{stock,packed}_batching_stats).
# ---------------------------------------------------------------------------------

_ARMS_PACKED = ("stock", "packed")


def median_post_warmup(values: list[float], warmup: int) -> float:
    """Median of `values` after dropping the first `warmup` (the compiled step-1 trace
    plus any first-shape kernel calibration stalls). Falls back to the full list when
    `warmup` would drop everything (a short run), so a non-empty input never yields 0."""
    kept = values[warmup:] if len(values) > warmup else values
    if not kept:
        raise MlxTrainPerfError("no step-wall samples to summarize")
    return statistics.median(kept)


def packed_throughput_fields(
    *, mean_real_tokens_per_step: float, mean_samples_per_step: float,
    median_step_wall_s: float,
) -> dict[str, object]:
    """Compose the measured median step wall with the deterministic per-step content
    counts into real (non-pad) tokens/s and samples/hour -- the throughput headline. A
    non-positive wall (an all-warmup or empty run) yields 0.0 rather than dividing by 0."""
    if median_step_wall_s > 0:
        real_tps = mean_real_tokens_per_step / median_step_wall_s
        samples_hr = mean_samples_per_step / median_step_wall_s * 3600.0
    else:
        real_tps = samples_hr = 0.0
    return {
        "real_tokens_per_second": round(real_tps, 3),
        "samples_per_hour": round(samples_hr, 3),
        "mean_real_tokens_per_step": round(mean_real_tokens_per_step, 4),
        "mean_samples_per_step": round(mean_samples_per_step, 4),
        "median_step_wall_s": round(median_step_wall_s, 6),
    }


def _make_reasserting_loss_varargs(
    loss_fn: Callable[..., tuple[mx.array, mx.array]],
    observed_before: list[int],
) -> Callable[..., tuple[mx.array, mx.array]]:
    """Variadic sibling of `_make_reasserting_loss` for the packed loss's 5-array trainer
    contract (`model, batch, seg_id, seg_start, loss_mask`) -- also reused for the stock
    arm's 3-array `make_loss_fn`. The FIRST call re-asserts this project's house wired cap
    (see the module docstring for why this lives inside the loop); `mx.compile` re-runs the
    traced body only on a recompile, so firing once suffices."""

    def wrapped(model: Any, *arrays: mx.array) -> tuple[mx.array, mx.array]:
        if not observed_before:
            observed_before.append(install_guardrails())
        return loss_fn(model, *arrays)

    return wrapped


def _load_packed_dataset(path: str) -> list[tuple[list[int], int]]:
    """Read the prep_alpaca jsonl -- one `{"tokens": [...], "offset": N}` object per
    line -- into the `(tokens, offset)` pairs both stock and packed `iterate_batches`
    consume. The SOLE file read of this condition kind (isolated for the same reason
    `_load_model` is: tests never touch a real dataset)."""
    dataset: list[tuple[list[int], int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            dataset.append(([int(t) for t in obj["tokens"]], int(obj["offset"])))
    return dataset


def _arm_batching_stats(
    arm: str, lengths: list[int], *, pack_len: int, batch: int, seed: int
) -> tuple[float, float, dict[str, object]]:
    """Deterministic host-side per-step content counts (mean real tokens / mean samples
    per step) plus the arm-specific utilization fields: the stock padding-waste fraction,
    or the packed utilization / separator / tail fractions."""
    if arm == "packed":
        pstats = packed_batching_stats(lengths, pack_len, batch_size=batch, seed=seed)
        return pstats.mean_real_tokens_per_step, pstats.mean_samples_per_step, {
            "num_batches": pstats.num_batches,
            "real_tokens_total": pstats.real_tokens_total,
            "utilization": round(pstats.utilization, 6),
            "separator_fraction": round(pstats.separator_fraction, 6),
            "tail_pad_fraction": round(pstats.tail_pad_fraction, 6),
        }
    sstats = stock_batching_stats(lengths, batch_size=batch, max_seq_length=pack_len)
    return sstats.mean_real_tokens_per_step, sstats.mean_samples_per_step, {
        "num_batches": sstats.num_batches,
        "real_tokens_total": sstats.real_tokens_total,
        "padded_tokens_total": sstats.padded_tokens_total,
        "padding_waste_fraction": round(sstats.padding_waste_fraction, 6),
    }


def _run_packed_train_steps(
    model: Any,
    optimizer: Any,
    loss_fn: Callable[..., tuple[mx.array, mx.array]],
    dataset: list[tuple[list[int], int]],
    *,
    batch: int,
    pack_len: int,
    steps: int,
    grad_checkpoint: bool,
    iterate_batches: Any,
) -> list[dict[str, object]]:
    """Drive `steps` real fine-tune iterations through the compiled `train()`. The packed
    arm passes `iterate_batches=partial(packed_iterate_batches, ...)`; the stock arm passes
    `None`, leaving `train()`'s own stock `iterate_batches` in place. `max_seq_length` is
    the pack length for both -- packed rows are `pack_len + 1` wide (`packed_iterate_batches`
    sizes them), stock pads up to `pack_len`."""
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415

    callback = _RecordingCallback()
    with tempfile.TemporaryDirectory(prefix="mlx-train-perf-packed-") as tmp_dir:
        args = TrainingArgs(
            batch_size=batch, iters=steps, val_batches=0, steps_per_report=1,
            steps_per_eval=steps + 1, steps_per_save=steps + 1,
            max_seq_length=pack_len, grad_checkpoint=grad_checkpoint,
            adapter_file=str(Path(tmp_dir) / "adapters.safetensors"),
        )
        extra: dict[str, Any] = {}
        if iterate_batches is not None:
            extra["iterate_batches"] = iterate_batches
        train(model=model, optimizer=optimizer, train_dataset=dataset, val_dataset=[],
              args=args, loss=loss_fn, training_callback=cast(Any, callback), **extra)
    return callback.train_info


def _setup_packed_model(
    model_id: str, revision: str | None, *, compute_dtype: str | None, pack_len: int,
    batch: int, packed_arm: bool, seed: int, lora_rank: int, lora_layers: int,
) -> Any:
    """Load the model and wire it for a `packed_train` arm, in the load-bearing order
    (mirrors `run_train_step` + `tests/test_packed_smoke.py`): cast to `compute_dtype` if
    set, `mx.eval` the cast (gotcha 14), enable flash attention (`packed=` for the packed
    arm) BEFORE `freeze()`/`linear_to_lora_layers` (LoRA target discovery walks the module
    tree by path), then `mx.eval` the LoRA setup so its lazy graph does not leak into the
    measured window."""
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    model, _tokenizer = _load_model(model_id, revision)
    if compute_dtype is not None:
        model.set_dtype(_resolve_dtype(compute_dtype))
    mx.eval(model.parameters())
    enable_flash_attention(model, seq_len=pack_len, batch_size=batch, packed=packed_arm)
    mx.random.seed(seed)  # BEFORE freeze/LoRA-injection: their random init draws next
    model.freeze()
    linear_to_lora_layers(
        model, lora_layers, {"rank": lora_rank, "dropout": 0.0, "scale": 20.0}
    )
    mx.eval(model.parameters())  # force the lazy setup graphs before the measured window
    return model


def _packed_summary(
    step_reports: list[dict[str, object]], *, warmup: int, mean_real: float,
    mean_samples: float,
) -> dict[str, object]:
    """The measured throughput/loss/peak fields: per-step walls (`1 / iterations_per_second`,
    `steps_per_report=1`), their post-warmup median, the composed real-tokens/s + samples/hour,
    the per-step loss curve, and the callback's peak-memory high-water mark."""
    iters_per_sec = [
        float(cast(float, info["iterations_per_second"])) for info in step_reports
    ]
    walls = [1.0 / rate for rate in iters_per_sec if rate > 0]
    median_wall = median_post_warmup(walls, warmup) if walls else 0.0
    # info["peak_memory"] is the trainer's mx.get_peak_memory()/1e9 (DECIMAL GB); rescale to
    # GiB so callback_peak_memory_gb matches this file's sibling *_gb (GiB) fields.
    peak_all = [
        float(cast(float, info["peak_memory"])) * 1e9 / 1024**3 for info in step_reports
    ]
    loss_all = [float(cast(float, info["train_loss"])) for info in step_reports]
    return {
        **packed_throughput_fields(
            mean_real_tokens_per_step=mean_real, mean_samples_per_step=mean_samples,
            median_step_wall_s=median_wall,
        ),
        "step_walls_s": [round(w, 6) for w in walls],
        "loss_all": [round(x, 6) for x in loss_all],
        "callback_peak_memory_gb": round(max(peak_all), 4) if peak_all else 0.0,
    }


def run_packed_train(params: dict[str, object]) -> dict[str, object]:
    """Time `steps` real mlx-lm LoRA fine-tune steps for ONE batching arm against a real
    (`mlx_lm.load`-resolved) model and a real prepped dataset (`params["data"]`, a
    prep_alpaca jsonl). Both arms enable flash attention and the fused CE loss -- the ONLY
    variable is the batching strategy (`arm="stock"` -> `make_loss_fn` + stock batching;
    `arm="packed"` -> `make_packed_loss_fn` + `packed_iterate_batches`). Reports real
    (non-pad) tokens/s and samples/hour (measured median step wall x deterministic per-step
    content counts), the memory story, the stock padding-waste fraction (stock arm) or the
    packed utilization/separator/tail fractions (packed arm), and this project's wired-limit
    contract (`WiredCapRegressionError` if the cap did not hold through training).

    Wiring order mirrors `run_train_step` and `tests/test_packed_smoke.py`:
    `enable_flash_attention` (with `packed=`) BEFORE `freeze()`/`linear_to_lora_layers` (LoRA
    target discovery walks `named_modules()` by path); `mx.eval(model.parameters())` after
    the cast and after LoRA (gotcha 14); `mx.synchronize()` before the memory snapshot
    (gotcha 15).

    `params`: `model` (str, required), `revision` (str | None), `data` (jsonl path,
    required), `pack_len` (int, required -- the max_seq_length / pack length), `batch`
    (int, required), `steps` (int, required), `warmup` (int, default 5 -- steps dropped
    before the median), `arm` ("stock" | "packed", default "packed"), `lora_rank` (default
    8), `lora_layers` (default -1 == all), `impl` (default "auto"), `learning_rate`
    (default 1e-5), `seed` (default 0), `compute_dtype` (str | None -- the kernel impl needs
    "bfloat16" on 4-bit checkpoints), `grad_checkpoint` (bool, default False)."""
    _require_mlx_lm()
    import mlx.optimizers as optim  # noqa: PLC0415

    model_id = str(params["model"])
    revision = cast("str | None", params.get("revision"))
    data_path = str(params["data"])
    pack_len = int(cast(int, params["pack_len"]))
    batch = int(cast(int, params["batch"]))
    steps = int(cast(int, params["steps"]))
    warmup = int(cast(int, params.get("warmup", 5)))
    arm = str(params.get("arm", "packed"))
    lora_rank = int(cast(int, params.get("lora_rank", 8)))
    lora_layers = int(cast(int, params.get("lora_layers", -1)))
    impl = cast(Literal["auto", "kernel", "chunked", "naive"], params.get("impl", "auto"))
    learning_rate = float(cast(float, params.get("learning_rate", 1e-5)))
    seed = int(cast(int, params.get("seed", 0)))
    compute_dtype = cast("str | None", params.get("compute_dtype"))
    grad_checkpoint = bool(params.get("grad_checkpoint", False))
    if arm not in _ARMS_PACKED:
        raise MlxTrainPerfError(f"unknown arm {arm!r}; expected one of {_ARMS_PACKED}")

    dataset = _load_packed_dataset(data_path)
    lengths = [len(tokens) for tokens, _ in dataset]
    packed_arm = arm == "packed"
    # Deterministic host-side per-step content + arm-specific utilization, before training.
    mean_real, mean_samples, batching_fields = _arm_batching_stats(
        arm, lengths, pack_len=pack_len, batch=batch, seed=seed,
    )

    # Both arms enable flash attention -- the only variable is the batching strategy.
    model = _setup_packed_model(
        model_id, revision, compute_dtype=compute_dtype, pack_len=pack_len, batch=batch,
        packed_arm=packed_arm, seed=seed, lora_rank=lora_rank, lora_layers=lora_layers,
    )

    base_loss: Callable[..., tuple[mx.array, mx.array]]
    iterate: Any
    if packed_arm:
        base_loss = make_packed_loss_fn(model, impl=impl)
        max_pos = cast("int | None", getattr(model.args, "max_position_embeddings", None))
        iterate = functools.partial(
            packed_iterate_batches, seed=seed, max_position_embeddings=max_pos,
        )
    else:
        base_loss = make_loss_fn(model, impl=impl)
        iterate = None
    observed_before: list[int] = []
    loss_fn = _make_reasserting_loss_varargs(base_loss, observed_before)

    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    expected_wired, _soft = clamped_caps(dev_max)
    opt = optim.Adam(learning_rate=learning_rate)

    mx.synchronize()  # gotcha 15: settle pending evals before the memory snapshot
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    step_reports = _run_packed_train_steps(
        model, opt, loss_fn, dataset, batch=batch, pack_len=pack_len, steps=steps,
        grad_checkpoint=grad_checkpoint, iterate_batches=iterate,
    )
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3

    observed_after = install_guardrails()
    if not wired_cap_holds(observed_bytes=observed_after, expected_bytes=expected_wired):
        raise WiredCapRegressionError(
            f"packed_train condition's wired limit was {observed_after / 1024**3:.2f} GB "
            f"after training, expected the house cap {expected_wired / 1024**3:.2f} GB "
            "-- mlx_lm.tuner.trainer.train()'s entry-time override was not correctly "
            "re-asserted (see this module's docstring)"
        )

    return {
        "arm": arm,
        "pack_len": pack_len,
        **batching_fields,
        **_packed_summary(
            step_reports, warmup=warmup, mean_real=mean_real, mean_samples=mean_samples,
        ),
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
    # The dedicated attention identity input `runner.Condition` carries OUT of `params`
    # (a config without the key -- every 0.1.0-era config -- reads `None`). Threaded into
    # BOTH the identity (so this worker rebuilds the SAME identity the runner did) and
    # `run_train_step`, the one authoritative execution source.
    attention_impl = cast("str | None", config.get("attention_impl"))
    # Top-level config field (mirrors the `attention_impl` seam), never inside `params`.
    # Unset (`None`) resolves to the module default. A wall budget is a SAFETY limit, not a
    # measurement dimension, so it is deliberately NOT threaded into `condition_identity`.
    wall_budget_s = cast("float | None", config.get("wall_budget_s"))
    if wall_budget_s is None:
        wall_budget_s = DEFAULT_WALL_BUDGET_S
    out = Path(cast(str, config["out"]))

    install_guardrails()  # FIRST -- before any allocation this condition makes

    ident = condition_identity(
        kind=kind, session_id=session_id, params=params, attention_impl=attention_impl,
    )
    # The active-memory + wall-budget watchdog, installed right after the wired cap:
    # `install_guardrails`'s WIRED cap does NOT bound a PAGEABLE over-allocation, and mlx
    # 0.32.0's `set_memory_limit` is advisory (it pages rather than raising until RAM+swap
    # is exhausted -- see the guards module docstring). That pageable class is what paged
    # for ~3 h into the IOGPUMemory.cpp:550 kernel panic on 2026-07-10. This daemon
    # watchdog fails a runaway condition FAST -- `make_watchdog_on_breach` writes an honest
    # `aborted_*` artifact via the SAME `ident` this worker's ok/refused write uses, then
    # `os._exit(70)`. Stopped on every NORMAL exit path (the `finally`).
    #
    # `effective_memory_ceiling` combines the STATIC device-relative rule with the DYNAMIC
    # measured-availability at start (rank-local `vm_stat`) -- it may REFUSE (typed
    # `MemoryBudgetError`) if this node is too crowded to start safely, and surfaces a
    # degraded-start `memory_warning` we log + record in the artifact.
    try:
        ceiling = effective_memory_ceiling()
    except MemoryBudgetError as exc:
        # 0022d: too crowded to START safely is an ENVIRONMENT-transient outcome -- its own
        # `refused_environment` status, distinct from the condition-intrinsic `refused`
        # (launch budget) and from a crash envelope. Only `"ok"` artifacts are fresh on
        # resume, so a later, quieter invocation re-runs this condition automatically.
        write_result(out, ident, "refused_environment", error=str(exc))
        return 0
    ceiling_bytes = ceiling.ceiling_bytes
    # Omit-when-None (identity convention): a nominal start carries no `memory_warning`.
    warning_field: dict[str, object] = (
        {"memory_warning": ceiling.warning} if ceiling.warning is not None else {}
    )
    watchdog = install_memory_watchdog(
        ceiling_bytes=ceiling_bytes, wall_budget_s=wall_budget_s,
        on_breach=make_watchdog_on_breach(out, ident, ceiling_bytes),
    )
    try:
        try:
            if kind == "loss_layer":
                fields = run_loss_layer(params)
            elif kind == "train_step":
                fields = run_train_step(params, attention_impl=attention_impl)
            elif kind == "packed_train":
                # attention_impl rides the identity (always "flash" for this kind) but is
                # not an execution knob here -- run_packed_train always enables flash on
                # both arms; the batching strategy is the only variable.
                fields = run_packed_train(params)
            else:
                # Deliberately uncaught: an unsupported kind is a program error (a bad
                # Condition was constructed), not a recorded run outcome -- it crashes this
                # worker process with a nonzero exit, and `runner.run_conditions` records
                # the failure envelope on the CALLER's side instead of this worker writing
                # anything. Referencing the bare `run_loss_layer`/`run_train_step` names
                # above (not a dict bound at import time) also keeps this dispatch
                # monkeypatch-friendly for tests.
                raise MlxTrainPerfError(
                    f"unsupported bench condition kind {kind!r}; expected 'loss_layer', "
                    "'train_step', or 'packed_train'"
                )
        except LaunchBudgetError as exc:
            # A guard refusal IS a result: the calibrated rate cannot serve this shape
            # within the kernel launch budget -- record it, don't crash the sweep. A
            # WiredCapRegressionError is a DIFFERENT, more serious failure (a condition
            # that measured under an uncapped run) and is deliberately NOT caught here --
            # it propagates the same way an unsupported kind does.
            write_result(out, ident, "refused", error=str(exc), **warning_field)
            return 0
        write_result(out, ident, "ok", **fields, **warning_field)
        return 0
    finally:
        # Normal completion / refusal / uncaught crash all stop the sampler thread so an
        # in-process caller never leaks it. A breach never reaches here -- `on_breach`
        # already hard-exited the process.
        watchdog.stop()


if __name__ == "__main__":
    raise SystemExit(main())
