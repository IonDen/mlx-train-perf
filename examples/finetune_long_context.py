"""Fine-tune a model on long examples, without truncating them.

A complete, runnable LoRA fine-tune that swaps in two parts from `mlx-train-perf`:
`enable_flash_attention` for an O(N)-memory attention backward, and `make_loss_fn` for a
cross-entropy that never materializes the `(N, V)` logits. Everything else is stock `mlx_lm`.

This is the script `mlx_lm.lora --train` would run for you, written out so the two extra
calls have somewhere to go. Nothing here is elided.

Usage:

    python examples/finetune_long_context.py \
        --model mlx-community/Qwen3-8B-4bit \
        --data path/to/train.jsonl \
        --max-seq-length 8192

`--data` is a JSONL file with one `{"text": "..."}` object per line. Each line becomes one
training example, kept whole up to `--max-seq-length` instead of being cut to 2048.

Model support is Llama, Qwen2 and Qwen3 with full attention; `enable_flash_attention`
raises `UnsupportedAttentionError` up front on anything else.

NOTE: `enable_flash_attention` replaces each layer's attention in place and there is no
undo. Load a fresh copy of the model for inference; a training-configured object raises
`AttentionInputError` the moment a KV cache appears.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.attention import enable_flash_attention


def load_jsonl_dataset(path: Path, tokenizer) -> list[tuple[list[int], int]]:
    """Read a `{"text": ...}` JSONL file into the `(tokens, offset)` pairs the trainer wants.

    `offset` is the prompt length, i.e. how many leading tokens are context rather than
    supervised target. These examples are plain continuations, so the offset is 0 and every
    token is supervised.
    """
    examples: list[tuple[list[int], int]] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            examples.append((tokenizer.encode(text), 0))
    if not examples:
        raise SystemExit(f"no examples found in {path}")
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--data", required=True, type=Path, help="JSONL with a 'text' field")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lora-layers", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--adapter-file", default="adapters.safetensors")
    ap.add_argument(
        "--no-flash", action="store_true", help="skip flash attention (for an A/B run)"
    )
    args = ap.parse_args()

    model, tokenizer = load(args.model)

    # 4-bit checkpoints compute in fp16 at runtime; the kernels need bf16 or fp32. The
    # quantized weights stay int4 -- this casts the floating-point activations only.
    model.set_dtype(mx.bfloat16)

    # Stock mlx-lm LoRA setup: freeze the base model, then swap in adapters.
    model.freeze()
    linear_to_lora_layers(
        model,
        args.lora_layers,
        {"rank": args.lora_rank, "scale": 20.0, "dropout": 0.0},
    )
    # Realize the lazy cast + adapter init before anything is measured or traced.
    mx.eval(model.parameters())

    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"model: {args.model}")
    print(f"trainable parameters: {trainable:,}")

    dataset = load_jsonl_dataset(args.data, tokenizer)
    lengths = [len(tokens) for tokens, _ in dataset]
    print(
        f"dataset: {len(dataset)} examples, "
        f"longest {max(lengths)} tokens, max_seq_length {args.max_seq_length}"
    )
    if max(lengths) > args.max_seq_length:
        print(
            f"  note: {sum(x > args.max_seq_length for x in lengths)} example(s) are still "
            f"longer than --max-seq-length and will be truncated."
        )

    # The two lines this example exists for.
    if not args.no_flash:
        # Hints must match the training shape: the rate caches key on the exact batch size
        # and sequence bucket, so matching them keeps calibration outside the compiled step.
        enable_flash_attention(
            model, seq_len=args.max_seq_length, batch_size=args.batch_size
        )
    loss_fn = make_loss_fn(model, impl="auto")

    train(
        model=model,
        optimizer=optim.AdamW(learning_rate=args.learning_rate),
        train_dataset=dataset,
        args=TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=True,
            adapter_file=args.adapter_file,
        ),
        loss=loss_fn,
    )
    print(f"adapters written to {args.adapter_file}")


if __name__ == "__main__":
    main()
