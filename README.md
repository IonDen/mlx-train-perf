# mlx-train-perf

A fused, logit-free linear-cross-entropy loss for training on Apple Silicon with [MLX](https://github.com/ml-explore/mlx), plus a RAM-fit planner and an honest benchmark harness. It drops into an `mlx-lm` LoRA/QLoRA fine-tune as the loss function.

The idea is the same one behind [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) on the CUDA side, ported to a Metal kernel: compute the cross-entropy loss and its gradient without ever building the full `(N, V)` logits tensor. For a large vocabulary that tensor is the single biggest allocation in the training step, and it is pure waste. You only need the per-token loss and a gradient back into the hidden states.

Released on PyPI as `mlx-train-perf`. Every number below has a committed script under `scripts/` that reproduces it; all were measured on an M1 Max (32 GB), mlx 0.31.2.

## The problem it solves

Standard cross-entropy in a trainer materializes logits of shape `(batch·seq, vocab)`. At Qwen3-8B's vocabulary (151,936) and a 2048-token sequence, that is a 0.6 GB tensor in bf16, plus another for the softmax gradient in the backward pass. The fused kernel never allocates it: the forward regenerates logits in registers tile-by-tile over the vocabulary and returns three `N`-length arrays (the per-token NLL, the log-sum-exp, and the target logit); the backward recomputes the needed tiles instead of reading a stored matrix.

Measured in isolation, at n=8192, V=151936, D=4096, bf16 (`scripts/bench_loss_layer.py`):

| loss layer | peak memory | forward wall |
|---|---|---|
| naive (materialized logits) | 2.318 GB | 1.0× |
| kernel (this project) | 0.0006 GB | 1.64× |

About 3900× less memory for the loss layer, at a 1.64× cost on the forward pass.

## What this does and does not buy you

I want to be precise here, because the honest end-to-end story is narrower than the loss-layer number suggests.

In a real fine-tune step the fused loss frees the memory the logit tensor would have taken. That headroom goes to a larger batch or a longer sequence at the *same* peak. The loss is exact to bf16 tolerance against the stock trainer (per-step loss curves match to about 2e-3), and the throughput cost is small: roughly 8–12% slower per step at bf16 (`scripts/bench_train_step.py`, Qwen3-8B-4bit, LoRA r=8, gradient checkpointing on).

What it does *not* buy you on MLX today is a longer maximum context before you hit an out-of-memory error. We measured this directly (`scripts/northstar_context_sweep.py`): with gradient checkpointing on, ours and the stock trainer hit essentially the same context ceiling for an 8B QLoRA on 32 GB — ours about 8450 tokens, stock about 8700, a gap of one 256-token probe step. The reason is that `mx.fast.scaled_dot_product_attention` has a memory-efficient forward but an O(N²) backward. It materializes the `(N,N)` attention matrix one layer at a time during training, and that term dominates the peak at long context. Once the logits are gone, attention is the bottleneck, not the loss. A memory-efficient attention backward is the piece that would move the context ceiling on MLX, and it is the next thing on the [roadmap](ROADMAP.md).

So this frees real memory at a given context, and it is the right building block, but the flagship "train much longer sequences" win needs the attention backward too.

## Install

```bash
pip install mlx-train-perf            # the loss kernel + planner
pip install "mlx-train-perf[mlx-lm]"  # plus the mlx-lm training adapter
```

Apple Silicon only. Requires mlx (>=0.31.2 recommended; the kernel's JIT contract is verified against it).

## Use it in an mlx-lm fine-tune

The adapter builds a loss callable with the same signature `mlx_lm`'s trainer expects, so you pass it straight to `train(...)`:

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import train
from mlx_train_perf.adapters.mlx_lm import make_loss_fn

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
model.set_dtype(mx.bfloat16)  # 4-bit checkpoints compute in fp16; the kernel needs bf16/fp32
# ... freeze the base model and apply linear_to_lora_layers as in a normal mlx-lm LoRA run ...

loss_fn = make_loss_fn(model, impl="auto")
train(model=model, optimizer=opt, train_dataset=ds, args=args, loss=loss_fn)
```

`make_loss_fn` splits the model into its trunk and its output head and routes the loss through the fused kernel.

## Implementations

`impl` picks how the loss is computed. `"auto"` is the default and the one to use.

- `kernel` — the fused Metal kernel. `"auto"` resolves here when the mlx version is verified and the head/dtype are supported (dense or tied fp32/bf16 head; 4-bit group-size-64 quantized head; hidden states in fp32 or bf16). It never materializes `(N, V)`.
- `chunked` — a pure-MLX fallback that processes the vocabulary in fixed tiles. No Metal kernel, works anywhere MLX does, uses more memory than `kernel` but far less than `naive`. This is also the backward path the kernel forward pairs with today.
- `naive` — materializes the full logits. It is the correctness oracle the other two are tested against, not something to train with.

`"auto"` never silently downgrades. If it cannot use the kernel (unverified mlx, an unsupported head, fp16 hidden states) it raises a typed error naming the reason and the alternatives, so you always know which path ran.

## RAM-fit planner

Before a run, the planner estimates the peak training memory for a config and tells you whether it fits, or suggests a smaller batch or sequence length that would:

```bash
mlx-train-perf plan --config path/to/config.json --batch 1 --seq-len 4096 --lora-rank 8
```

The memory model is fit to measured Qwen3-8B train-step peaks and cross-model validated on Llama-3.2-3B to within about 9%. It accounts for the O(N²) attention backward described above, so it does not under-predict at long context the way a linear model would. It is an estimate, and it errs toward over-predicting, which is the safe direction for a tool whose job is to keep you off the OOM cliff.

## Supported models

- Architectures: Llama and Qwen3. The adapter's model splitter handles these; others raise a typed error.
- Quantization: 4-bit group-size-64 (the mlx-community QLoRA default), or a dense fp32/bf16 head.
- Training: LoRA / QLoRA. Full fine-tuning is estimated by the planner but is not the case this is tuned for.
- Hardware: Apple Silicon.

## Reproducing the numbers

Each claim above has one script. They run on the GPU, take real wall-clock time, and print the artifacts they measured:

```bash
python scripts/bench_loss_layer.py        # the ~3900x loss-layer memory number
python scripts/bench_train_step.py --model mlx-community/Qwen3-8B-4bit --seq-len 1024 2048 \
    --impl kernel --compute-dtype bfloat16 --grad-checkpoint   # end-to-end tok/s vs stock
python scripts/northstar_context_sweep.py # the max-context sweep (1-2 h; heavy)
```

## Community benchmarks

Every number above is from an M1 Max (32 GB), the machine this is developed on. Whether the
kernel and the flash-attention path scale the way the memory model expects on larger
machines is a question only other people's hardware can answer, so there is a one-command
way to measure it and send the numbers back:

```bash
mlx-train-perf contribute --tier quick   # ~10-15 min; --tier full loads a model, ~1-2 h
```

It detects your machine, picks shapes for your RAM, prints a time estimate, runs the
committed benches with the same memory guardrails the project uses, and writes one
provenance-complete file plus a ready-to-paste PR. The three-step submission flow is in
[community-benchmarks/README.md](community-benchmarks/README.md).

Submitted results are folded into the table below (`python scripts/aggregate_community.py`).
Each row is measured on that contributor's own hardware and reported as-is — nothing is
extrapolated to machines no one has run, and a row does not imply a trainable-context
ceiling beyond what that machine measured. The stock-attention baseline comparison is run
on reference hardware by the maintainer, not asked of contributors.

<!-- community-benchmarks:table -->
| Chip | RAM (GB) | mlx | Tier | Loss kernel peak (GB) | Attn flash 2x ratio | Train tok/s (flash) | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| _no submissions yet_ | | | | | | | |
<!-- /community-benchmarks:table -->

## License

MIT. See [LICENSE](LICENSE).
