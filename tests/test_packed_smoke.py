"""--run-smoke: the packed-training path end-to-end through mlx-lm's REAL compiled `train()`.

One live LoRA fine-tune (4 steps, grad_checkpoint=True) on a real checkpoint with
`packed_iterate_batches` + `make_packed_loss_fn` + `enable_flash_attention(packed=True)` --
the exact wiring the README recipe documents. Model: mlx-community/Qwen2.5-0.5B-Instruct-bf16,
expected pre-downloaded (never fetched); qwen2 exercises the bias-carrying projections the
llama tiny models cannot represent.

Ordering is load-bearing (mirrors `bench/worker.py::run_train_step`):
`enable_flash_attention` BEFORE `freeze()`/`linear_to_lora_layers` -- LoRA target discovery
walks `named_modules()` by path, so the wrapper must already be in the tree for the adapters
to land inside it; `mx.eval(model.parameters())` after LoRA so the lazy adapter init cannot
leak into the training window (gotcha 14).
"""
import math
import random
from pathlib import Path

import mlx.core as mx
import pytest

mlx_lm = pytest.importorskip("mlx_lm")

from mlx_train_perf.adapters.mlx_lm import make_packed_loss_fn  # noqa: E402
from mlx_train_perf.attention.wrapper import enable_flash_attention  # noqa: E402
from mlx_train_perf.data.packing import packed_iterate_batches  # noqa: E402

_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
_B, _L = 2, 512


def _ragged_dataset(vocab_size: int) -> list[tuple[list[int], int]]:
    """24 ragged (tokens, offset) pairs, lengths 40-400 -- several packed rows per epoch."""
    rng = random.Random("packed-smoke")
    return [
        (
            [rng.randrange(4, vocab_size - 4) for _ in range(rng.randint(40, 400))],
            rng.randint(1, 9),
        )
        for _ in range(24)
    ]


class _RecordingCallback:
    """Duck-typed TrainingCallback (the `bench/worker.py::_RecordingCallback` pattern)."""

    def __init__(self) -> None:
        self.train_info: list[dict[str, object]] = []

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        self.train_info.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:  # noqa: ARG002
        return None  # pragma: no cover -- val_dataset=[] means never invoked


@pytest.mark.smoke
def test_packed_training_end_to_end_through_compiled_train(tmp_path: Path) -> None:
    from mlx.optimizers import Adam  # noqa: PLC0415 -- lazy: smoke-gated heavy deps
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    model, _tokenizer = mlx_lm.load(_MODEL)
    mx.eval(model.parameters())
    enable_flash_attention(model, seq_len=_L, batch_size=_B, packed=True)
    mx.random.seed(7)
    model.freeze()
    linear_to_lora_layers(model, 4, {"rank": 8, "dropout": 0.0, "scale": 20.0})
    mx.eval(model.parameters())

    dataset = _ragged_dataset(int(model.args.vocab_size))

    # Host-side expectation check on the exact batches train() will consume: the loss
    # fn's ntoks equals each yielded mask's own sum, every batch supervises >= 1 token,
    # and every eager loss is finite.
    loss_fn = make_packed_loss_fn(model)
    n_batches = 0
    for batch, seg_id, seg_start, loss_mask in packed_iterate_batches(
        dataset=dataset, batch_size=_B, max_seq_length=_L, loop=False, seed=3
    ):
        loss, ntoks = loss_fn(model, batch, seg_id, seg_start, loss_mask)
        mx.eval(loss, ntoks)
        expected = int(loss_mask.sum().item())
        assert expected > 0
        assert int(ntoks.item()) == expected
        assert mx.isfinite(loss).item()
        n_batches += 1
    assert n_batches >= 2, "fixture must produce multiple packed batches"

    callback = _RecordingCallback()
    train(
        model=model,
        optimizer=Adam(learning_rate=1e-5),
        train_dataset=dataset,
        val_dataset=[],
        args=TrainingArgs(
            batch_size=_B,
            iters=4,
            max_seq_length=_L,
            grad_checkpoint=True,
            steps_per_report=1,
            steps_per_eval=1000,
            steps_per_save=1000,
            adapter_file=str(tmp_path / "adapters.safetensors"),
        ),
        loss=make_packed_loss_fn(model),
        iterate_batches=packed_iterate_batches,
        training_callback=callback,
    )

    losses = [float(info["train_loss"]) for info in callback.train_info]  # type: ignore[arg-type]
    assert len(losses) == 4, f"expected 4 per-step reports, got {len(losses)}"
    assert all(math.isfinite(loss) for loss in losses), losses
