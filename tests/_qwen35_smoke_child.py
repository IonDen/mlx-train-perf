"""Subprocess child (run by test_qwen35_smoke.py, never collected by pytest).

Proves the FULL production composition against a REAL qwen3_5 checkpoint, in
ISOLATION from the parent pytest process: mlx-lm's `load()`, this project's
`enable_gated_delta_training`, a LoRA adapter on the GatedDelta projections,
and mlx-lm's own compiled `train()` for a few real steps over padded batches
of at least two distinct lengths (forcing the compiled step to retrace).
Isolation matters because `train()` overrides the process's wired-memory cap
at its own entry (gotcha 8) -- running it in-process would corrupt that cap
for every test that runs afterward in the same session.

Casting to a training compute dtype uses a PATH-AWARE walk over the model's
own `cast_predicate` (mirroring `mlx_lm/convert.py`'s idiom and
`tests/test_recurrent_wrapper.py`'s `test_a_log_fp32_after_set_dtype_then_enable`)
rather than a bare `model.set_dtype(...)`: `Module.set_dtype`'s predicate is
called with a DTYPE, never a path, so it cannot honour qwen3_5's path-keyed
`cast_predicate` and would downcast the GatedDelta gate scalar `A_log` along
with everything else -- the forward guard then refuses (`RecurrentInputError`)
rather than silently training on a corrupted gate.

Writes the LoRA-only trainable parameters to two safetensors files (before and
after `train()`) under the scratch directory given as `sys.argv[1]`, so the
parent test can verify training did real work without re-loading the model in
its own process. Prints per-iteration wall time and the first-vs-later wall
ratio as information (a real-vs-retraced-step cost is recorded here, never
asserted on), then `::SMOKE_OK::` last.
"""
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map_with_path

from mlx_train_perf.core.guards import (
    effective_memory_ceiling,
    install_guardrails,
    install_memory_watchdog,
)
from mlx_train_perf.families import text_args
from mlx_train_perf.recurrent import enable_gated_delta_training

_MODEL_REPO = "mlx-community/Qwen3.5-0.8B-4bit"
_BATCH = 2
_MAX_SEQ_LENGTH = 192
_ITERS = 4
# Comfortably above the observed end-to-end runtime (model load + 4 iters is
# single-digit seconds on a cached checkpoint) but MUST stay strictly below
# the parent test's `subprocess.run(..., timeout=600)` -- if this budget ever
# grew past 600s, the parent's timeout would SIGKILL the child first on a
# genuine hang, discarding the `::SMOKE_MEMORY_BREACH::` marker and the clean
# `os._exit(2)` this watchdog exists to provide in favour of a bare
# `TimeoutExpired` with whatever stdout happened to be captured.
_WALL_BUDGET_S = 240.0
_LORA_CONFIG = {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
    "keys": {"linear_attn.in_proj_qkv", "linear_attn.in_proj_b"},
}


def _ragged_dataset(vocab_size: int) -> list[tuple[list[int], int]]:
    """4 short (8-16 tok) + 4 long (96-128 tok) examples so mlx-lm's own
    length-sorted `iterate_batches` -- which sorts by length before pairing
    into fixed-size batches -- yields two distinct padded batch lengths across
    a batch_size=2, 4-batch epoch (short pairs pad to 33; long pairs pad to
    97-129; `pad_to=32` plus one), forcing the compiled training step to
    retrace at least once."""
    rng = random.Random("qwen35-smoke")
    short = [
        [rng.randrange(4, vocab_size - 4) for _ in range(rng.randint(8, 16))]
        for _ in range(4)
    ]
    long_examples = [
        [rng.randrange(4, vocab_size - 4) for _ in range(rng.randint(96, 128))]
        for _ in range(4)
    ]
    return [(tokens, 0) for tokens in short + long_examples]


def _cast_bf16_preserving_a_log(model: Any) -> None:
    """Casts every floating parameter to bf16 except `A_log`, honouring the
    model's own path-keyed `cast_predicate` -- see the module docstring for
    why a bare `model.set_dtype(...)` cannot do this. Asserts `A_log` stayed
    fp32 so a regression here fails loudly instead of silently biasing the
    gated-delta decay rates."""
    cast_predicate = model.cast_predicate

    def _cast(path: str, value: mx.array) -> mx.array:
        if cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(_cast, model.parameters()))
    mx.eval(model.parameters())

    a_log_dtypes = {
        path: value.dtype
        for path, value in tree_flatten(model.parameters())
        if path.endswith("A_log")
    }
    assert a_log_dtypes, "no A_log parameters found -- model has no linear-attention layers?"
    assert all(dtype == mx.float32 for dtype in a_log_dtypes.values()), a_log_dtypes


def _on_memory_breach(reason: str, details: dict[str, object]) -> None:
    """Fail-CLOSED: `core.guards`'s watchdog thread suppresses any exception
    raised inside this callback, so a raise here would silently disarm the
    guard instead of stopping the run. Non-blocking (no sync, no subprocess)
    and unconditional (the exit lives in `finally`, so a failing print cannot
    skip it)."""
    try:
        print(f"::SMOKE_MEMORY_BREACH:: reason={reason} details={details}", flush=True)
    finally:
        os._exit(2)


class _RecordingCallback:
    """Duck-typed TrainingCallback (the `bench/worker.py::_RecordingCallback` pattern)."""

    def __init__(self) -> None:
        self.train_info: list[dict[str, object]] = []

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_info.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:  # noqa: ARG002
        return None  # pragma: no cover -- val_dataset=[] means never invoked


def main() -> None:
    from mlx.optimizers import Adam  # noqa: PLC0415 -- lazy: smoke-gated heavy deps
    from mlx_lm import load  # noqa: PLC0415
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    from mlx_train_perf.adapters.mlx_lm import make_loss_fn  # noqa: PLC0415

    scratch = Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)
    before_path = scratch / "adapters_before.safetensors"
    after_path = scratch / "adapters_after.safetensors"

    install_guardrails()
    mx.set_cache_limit(2 * 1024**3)
    ceiling = effective_memory_ceiling()
    watchdog = install_memory_watchdog(
        ceiling_bytes=ceiling.ceiling_bytes,
        wall_budget_s=_WALL_BUDGET_S,
        sampler=lambda: mx.get_active_memory() + mx.get_cache_memory(),
        interval_s=0.05,
        on_breach=_on_memory_breach,
    )

    model, _tokenizer = load(_MODEL_REPO)
    mx.eval(model.parameters())
    _cast_bf16_preserving_a_log(model)  # BEFORE enable -- see module docstring
    enable_gated_delta_training(model)

    mx.random.seed(11)  # BEFORE freeze/LoRA-injection: their random init draws next
    model.freeze()
    linear_to_lora_layers(model, -1, _LORA_CONFIG)
    mx.eval(model.parameters())

    before = dict(tree_flatten(model.trainable_parameters()))
    mx.eval(*before.values())
    mx.save_safetensors(str(before_path), before)

    dataset = _ragged_dataset(int(text_args(model).vocab_size))
    loss_fn = make_loss_fn(model)
    callback = _RecordingCallback()

    train(
        model=model,
        optimizer=Adam(learning_rate=1e-5),
        train_dataset=dataset,
        val_dataset=[],
        args=TrainingArgs(
            batch_size=_BATCH,
            iters=_ITERS,
            max_seq_length=_MAX_SEQ_LENGTH,
            steps_per_report=1,
            steps_per_eval=1000,
            steps_per_save=1000,
            adapter_file=str(after_path),
        ),
        loss=loss_fn,
        training_callback=callback,
    )
    install_guardrails()  # train() overwrote the wired cap at entry -- re-assert

    walls = [1.0 / float(info["iterations_per_second"]) for info in callback.train_info]
    for i, wall in enumerate(walls, start=1):
        print(f"iter {i} wall {wall:.3f}s", flush=True)
    if len(walls) > 1:
        later = walls[1:]
        later_mean = sum(later) / len(later)
        ratio = walls[0] / later_mean if later_mean > 0 else math.inf
        print(f"first-vs-later wall ratio {ratio:.3f}", flush=True)

    watchdog.stop()
    print("::SMOKE_OK::", flush=True)


if __name__ == "__main__":
    main()
