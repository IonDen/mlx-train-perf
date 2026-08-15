# mlx-train-perf

[![PyPI version](https://img.shields.io/pypi/v/mlx-train-perf.svg)](https://pypi.org/project/mlx-train-perf/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-train-perf.svg)](https://pypi.org/project/mlx-train-perf/)
[![License: MIT](https://img.shields.io/pypi/l/mlx-train-perf.svg)](https://github.com/IonDen/mlx-train-perf/blob/main/LICENSE)

Train on longer sequences, and get through short ones faster, on the Mac you already have.

`mlx-train-perf` is a set of drop-in parts for [MLX](https://github.com/ml-explore/mlx) LoRA and QLoRA fine-tuning: Metal kernels that cut what a single training step allocates, sequence packing for datasets made of short examples, and a planner that answers "will this fit in my RAM?" before you download the weights. It is not a trainer and does not want to be one. You keep `mlx_lm`'s training loop and swap in the pieces you need.

Measured on one M1 Max (32 GB), Qwen3-8B-4bit QLoRA: the longest sequence you can train goes from 7,936 tokens to 23,040, and an Alpaca-shaped instruction dataset trains 3.0× faster per real token (2 to 3× depending on how short your examples are).

### The problem, in mlx-lm's own words

If a fine-tune does not fit in memory, mlx-lm's [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) offers five remedies. The fourth is:

> Longer examples require more memory. If it makes sense for your data, one thing you can do is break your examples into smaller sequences when making the `{train, valid, test}.jsonl` files.

And if you leave `max_seq_length` at its default of 2048 while your data is longer than that, the trainer does it for you:

```
[WARNING] Some sequences are longer than 2048 tokens. The longest sentence 6144 will be
truncated to 2048. Consider pre-splitting your data to save memory.
```

That advice is correct, and it means training on the first 2048 tokens of every contract, transcript, or source file and discarding the rest. The advice exists because MLX's attention has a memory-light forward pass and a backward pass that rebuilds the full `(N, N)` score matrix. An MLX maintainer put it plainly while [closing an out-of-memory report](https://github.com/ml-explore/mlx/issues/3539#issuecomment-4445752643):

> The RAM needed for training grows quadratically as sequence length increases, so I'm afraid the OOM is not something we can simply solve.

It is solvable one layer up. This library replaces that backward pass with one that keeps O(N) state, for the architectures it supports, so context costs what it should. You raise `max_seq_length` instead of cutting up your data.

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-train-perf/main/docs/images/training-step-memory.svg" alt="Training step peak memory at 8192 tokens on Qwen3-8B-4bit: stock attention 25.68 GiB, above what a 32 GB Mac can use; flash attention 12.75 GiB, comfortably under it." width="720">
</p>

## Does this help me?

Including the cases where it does not.

| What you are hitting | Does this help? |
|---|---|
| The truncation warning above, on examples you would rather keep whole | Yes. Raise `max_seq_length` and use the [flash-attention path](https://github.com/IonDen/mlx-train-perf#flash-attention-training-path). |
| You raised `max_seq_length` and the run died or the machine locked up | Yes, if one step was the problem. At 8192 tokens the step's peak drops from 25.68 to 12.75 GiB. |
| An instruction dataset of short examples, and training crawls while the GPU looks idle | Yes. [Packing](https://github.com/IonDen/mlx-train-perf#sequence-packing) moves 2 to 3× more real tokens per second. |
| You want to know whether a config fits before spending an hour finding out | Yes. [`mlx-train-perf plan`](https://github.com/IonDen/mlx-train-perf#ram-fit-planner) answers without loading the model. |
| Everything already fits at the 2048 default | Not for the memory work — below roughly 2,100 tokens stock attention is the faster of the two. Packing can still help if your examples are short. |
| Memory that climbs across iterations at a fixed shape | No. That is a leak somewhere else; these kernels change what one step allocates, not what accumulates between steps. |
| Gemma, Mistral, Phi, or a hybrid model with sliding-window attention | Not yet. Llama, Qwen2 and Qwen3 with full attention, and it refuses the rest up front rather than failing halfway through a run. |
| Inference or serving speed | No. This is training only. |

Every number here has a committed script under `scripts/` that reproduces it, all measured on one M1 Max (32 GB, macOS 26.5). The loss-layer figures were taken on mlx 0.31.2 and reproduce on the pinned 0.32.0; the flash-attention memory figures were taken on 0.32.0 in 0.2.0, and the 0.3.0 context-ceiling figures on 0.32.0.

## Install

```bash
pip install mlx-train-perf            # the loss kernel + planner
pip install "mlx-train-perf[mlx-lm]"  # plus the mlx-lm training adapter
```

Apple Silicon only. Requires mlx >=0.32.0,<0.33, the version the kernels' JIT contract is verified against. The mlx-lm adapter and the flash-attention wrapper need the optional `mlx-lm` extra.

There is no flag to bolt onto `mlx_lm.lora`. These parts attach to a loaded model object, so you drive `mlx_lm`'s `train()` from a short Python script instead of the CLI. That script is the whole difference, and it runs about six lines longer than the one you would have written anyway.

## Three situations, start to finish

### Long examples that keep getting truncated

What you run today:

```bash
mlx_lm.lora --model mlx-community/Qwen3-8B-4bit --train --data ./data \
    --batch-size 1 --grad-checkpoint
```

```
[WARNING] Some sequences are longer than 2048 tokens. The longest sentence 6144 will be
truncated to 2048. Consider pre-splitting your data to save memory.
```

Passing `--max-seq-length 8192` trades the truncation for a crash: the step peaks at 25.68 GiB, and a 32 GB Mac has roughly 24.5 GiB to give it. With the flash path that same step peaks at 12.75 GiB, and the longest sequence you can train moves from 7,936 tokens to 23,040 (`scripts/northstar_context_sweep.py`).

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.attention import enable_flash_attention

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
model.set_dtype(mx.bfloat16)  # 4-bit checkpoints compute in fp16; the kernels need bf16/fp32
# ... freeze the base model and apply linear_to_lora_layers as in a normal mlx-lm LoRA run ...

args = TrainingArgs(batch_size=1, max_seq_length=8192, grad_checkpoint=True, iters=600)
enable_flash_attention(model, seq_len=8192, batch_size=1)

train(model=model, optimizer=opt, train_dataset=ds, args=args,
      loss=make_loss_fn(model, impl="auto"))
```

Two independent levers sit in those last three lines. `enable_flash_attention` swaps each layer's attention for the O(N) path, which is what moves the ceiling. `make_loss_fn` routes the loss through the fused kernel and frees the logit buffer on top of that. Either one works without the other.

The elisions above are the parts of a stock mlx-lm LoRA run that do not change. [`examples/finetune_long_context.py`](https://github.com/IonDen/mlx-train-perf/blob/main/examples/finetune_long_context.py) is the same thing with nothing left out: argument parsing, dataset loading, the freeze and adapter setup, and adapter saving. Run it as is.

One trap worth knowing: `enable_flash_attention` replaces each layer's attention in place and there is no undo. Load a fresh copy of the model for inference, because the training-configured object raises `AttentionInputError` as soon as a KV cache appears.

On a 16 GB machine this shape does not fit. Ask the planner what does, rather than finding out three minutes into a run.

### Thousands of short examples, and a run that crawls

Alpaca averages 84 tokens per example under Qwen3's chat template, and mlx-lm's trainer runs one step per batch of them. On an 8B model at batch 1 that step takes 2.5 s to carry 84 real tokens. Packing fills the row to 4,096 tokens with whole examples instead, and the step then takes 40.4 s to carry about 4,000 — roughly 48 times the tokens for 16 times the wall clock. The difference is fixed per-step cost that a short batch pays in full, and a packed row pays once.

```python
import functools
from mlx_train_perf.adapters.mlx_lm import make_packed_loss_fn
from mlx_train_perf.data.packing import packed_iterate_batches

enable_flash_attention(model, seq_len=4096, batch_size=1, packed=True)
args = TrainingArgs(batch_size=1, max_seq_length=4096, grad_checkpoint=True, iters=600)

train(model=model, optimizer=opt, train_dataset=ds, args=args,
      loss=make_packed_loss_fn(model),
      iterate_batches=functools.partial(
          packed_iterate_batches,
          max_position_embeddings=model.args.max_position_embeddings,
      ))
```

Measured on Qwen3-8B-4bit: 33.1 real tokens per second unpacked against 99.2 packed, a factor of 3.00 (`scripts/bench_packed_training.py`). Dataset items are `(tokens, offset)` pairs, where the offset is the prompt length. The gain comes from amortizing the fixed step cost, so it shrinks as your examples get longer and disappears once they already fill a row. [Sequence packing](#sequence-packing) has the conservative steady-state range and the full recipe.

### You do not know whether any of it will fit

```bash
mlx-train-perf plan --config ./Qwen3-8B-4bit/config.json --batch 1 --lora-rank 8 \
    --attention flash --max-seq
```

The alternative is a 16 GB download and an out-of-memory crash three minutes into training. This loads no weights and spends no GPU time. It reads the config, prices the run against your machine's memory, and hands back the longest sequence that fits. Ask about one specific config with `--seq-len` instead and it answers fits or does not fit, with the peak it predicted. The estimate leans toward over-predicting, which is the safe direction for a tool whose job is keeping you off the cliff.

## If you maintain a trainer

The two kernels are usable without `mlx_lm` and without this project's adapter. `mlx` is the
only runtime dependency; `mlx-lm` is an optional extra that exists solely for the adapter and
the `enable_flash_attention` wrapper.

```python
from mlx_train_perf import linear_cross_entropy, DenseHead, QuantizedHead
from mlx_train_perf.attention import flash_attention

# Loss: hidden states in, scalar out, no (N, V) tensor in between.
loss = linear_cross_entropy(hidden, head, targets, impl="auto", reduction="mean")

# Attention: a drop-in for mx.fast.scaled_dot_product_attention on the training path.
out = flash_attention(q, k, v, scale=scale, causal=True)
```

`head` is a `DenseHead`, a `QuantizedHead`, or a tied embedding via `tied_head(...)`. `q`/`k`/`v`
are `(B, H, N, D)` with `head_dim` in {64, 96, 128} and grouped-query heads mapped contiguously,
matching `mx.fast.scaled_dot_product_attention`'s own convention. Pass `segments=PackedMask(...)`
for block-diagonal packing.

What you are signing up for, stated plainly:

- **In-place mutation.** `enable_flash_attention(model)` swaps attention on a live model object
  and has no undo. `flash_attention` itself is a pure function and mutates nothing, so if you
  own your model code, call it directly and skip the wrapper.
- **Training only.** Both refuse a KV cache. Reload the model for inference.
- **Typed refusals, never silent fallbacks.** An unsupported architecture, head dim, dtype or
  mask raises at enable time or on the first call, naming the reason.
- **The mlx pin is a policy, not neglect.** `mlx>=0.32.0,<0.33` is narrow because the kernels'
  JIT contract is re-verified against each mlx release before the range widens, rather than
  assumed forward-compatible.
- **Calibration is one-time and host-synced.** Warm it at your training shape before a compiled
  step traces, or accept a single in-trace stall on the first call.

If your model family is not in the support list, open an issue and name it. The refusal list is
a statement about what has been verified, not about what the kernels could cover.

## The fused cross-entropy loss

The idea is the same one behind [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) on the CUDA side, ported to a Metal kernel: compute the cross-entropy loss and its gradient without ever building the full `(N, V)` logits tensor. For a large vocabulary that tensor is the single biggest allocation in the training step, and it is pure waste. You only need the per-token loss and a gradient back into the hidden states.

Standard cross-entropy in a trainer materializes logits of shape `(batch·seq, vocab)`. At Qwen3-8B's vocabulary (151,936) and a 2048-token sequence, that is a 0.6 GB tensor in bf16, plus another for the softmax gradient in the backward pass. The fused kernel never allocates it: the forward regenerates logits in registers tile-by-tile over the vocabulary and returns three `N`-length arrays (the per-token NLL, the log-sum-exp, and the target logit); the backward recomputes the needed tiles instead of reading a stored matrix.

Measured in isolation, at n=8192, V=151936, D=4096, bf16 (`scripts/bench_loss_layer.py`):

| loss layer | peak memory | forward wall |
|---|---|---|
| naive (materialized logits) | 2.318 GB | 1.0× |
| kernel (this project) | 0.0006 GB | 1.64× |

About 3900× less memory for the loss layer, at a 1.64× cost on the forward pass.

The fused loss is exact to bf16 tolerance against the stock trainer (per-step loss curves match to about 2e-3), and the throughput cost is small: roughly 8–12% slower per step at bf16 (`scripts/bench_train_step.py`, Qwen3-8B-4bit, LoRA r=8, gradient checkpointing on).

## Flash-attention training path

New in 0.2.0 and opt-in. Removing the logit tensor frees real memory, but on its own it barely moves the training peak at long context. The reason is attention. `mx.fast.scaled_dot_product_attention` has a memory-light forward and an O(N²) backward that rebuilds the `(N, N)` score matrix one layer at a time. Once the logits are gone, that backward is what sets the peak.

0.2.0 adds a flash-attention path with a Metal forward *and* a Metal backward, neither of which materializes the score matrix. It keeps O(N) saved state — the attention output and the log-sum-exp — and recomputes the tiles it needs. You switch it on per model with `enable_flash_attention` and train exactly as before.

On Qwen3-8B-4bit (LoRA rank 8, batch 1, gradient checkpointing on, bf16) the two attention paths are close at a 2048-token sequence. At 8192 they are not: the flash path halves the whole step's peak memory.

| seq 8192, Qwen3-8B-4bit | total peak | marginal peak |
|---|---|---|
| stock attention | 25.68 GiB | 21.31 GiB |
| flash attention | 12.75 GiB | 8.37 GiB |

(`scripts/bench_train_step.py`; M1 Max 32 GB, macOS 26.5, mlx 0.32.0. These memory and throughput figures are the 0.2.0 measurements, carried into 0.3.0 unchanged: 0.3.0 changed how the backward splits its kernel launches, not what it allocates, and the 0.3.0 context sweep below — measured fresh — confirms the flash path's memory still scales linearly in sequence length.)

The 32 GB machine that peaked near its ceiling with stock attention now runs the same step at half the memory. That headroom buys a longer sequence.

The attention op timed alone at the flagship shape (batch 1, 32 query / 8 KV heads, 8192 tokens, head_dim 128) is 0.186 s forward and 0.576 s backward (`scripts/bench_attention_op.py`). Its peak grows 2.00× from 2,048 to 4,096 tokens and 3.06× from 4,096 to 8,192; the second step is above 2× because the chained backward split adds a bounded constant of at most ~0.3 GB, not because the O(N) growth law changed. Stock attention grows 3.76× then 3.05× over the same doublings, from a far higher base. Flash is not universally cheaper: below about 2,100 tokens the stock op's simpler bookkeeping wins, and the two curves cross there. [When the bottleneck moved](https://ineshin.space/papers/when-the-bottleneck-moved/) has the full measurement, including what stock attention does on a 32 GB machine once it starts paging.

### What it costs in throughput

Turning flash attention on is not free. On the stock-loss path at 8192 it costs 5.3% of tokens/sec (74.0 vs 78.1); at 2048 the cost is 5.5% on the fused-loss path (86.4 vs 91.5) and 5.9% on the stock-loss path (92.1 vs 97.8). The fused-loss comparison at 8192 has no stock-attention number to pair with: on this 32 GB machine that baseline condition crosses the memory safety net's ceiling and records an abort instead of a number. That baseline running out of room is the problem flash attention exists to remove. Under flash attention the fused cross-entropy and mlx-lm's stock cross-entropy stay close: 0.94× at 2048 (86.4 vs 92.1 tok/s) and 0.99× at 8192 on Qwen3-8B, 0.92× and 0.97× on Llama-3.2-3B. The loss values match to bf16 tolerance throughout — the worst per-step difference across every measured pair is 2.4e-3. The worst attention-arm throughput ratio measured is 0.94× stock.

### What it changes: the context ceiling on 32 GB

0.2.0 shipped this path with a launch-safety budget that capped context before memory did. That budget turned out to be guarding the wrong unit: the GPU watchdog acts on a single Metal command buffer, not a chain of them, and 0.3.0 measures the margin against the right thing (`scripts/probe_command_buffer_packing.py`; the reasoning is in [How MLX packs Metal command buffers](https://ineshin.space/papers/how-mlx-packs-metal-command-buffers/)).

With that cap gone, the flash path is bound by memory, the same thing that bounds stock attention — and it needs far less of it. Measured the same day with the same search (`scripts/northstar_context_sweep.py`, Qwen3-8B-4bit QLoRA, gradient checkpointing, bf16):

| max trainable context, 32 GB | tokens | peak at the ceiling |
|---|---|---|
| stock attention | 7,936 | 24.5 GiB |
| flash attention | 23,040 | 24.5 GiB |

Both arms stop at the same ~24.5 GiB, the effective memory ceiling on this machine at run time. Under that one budget the flash path trains 2.9× the context, because it holds O(N) saved state where stock holds the O(N²) score matrix. The ratio is the part that travels: raise the available memory and both ceilings rise together (a freshly booted or larger machine lets both climb toward the 28 GiB static ceiling), but the flash path keeps its roughly threefold reach. On the same machine in 0.2.0 this path was launch-capped near 10k tokens; removing that cap is what moved it. (These figures are this release's measurement; the two arms are comparable to each other, taken together, not to 0.2.0's numbers.)

### When it refuses

`enable_flash_attention` is causal-only and training-only, and it refuses anything outside that up front rather than failing mid-run:

| Condition | When | Error |
|---|---|---|
| Model family other than Llama, Qwen2, or Qwen3 | at enable | `UnsupportedAttentionError` |
| Sliding-window or mixed attention (`layer_types` not all `full_attention`) | at enable | `UnsupportedAttentionError` |
| `head_dim` outside {64, 96, 128} | at enable | `UnsupportedAttentionError` |
| Non-zero attention dropout | at enable | `UnsupportedAttentionError` |
| An array attention mask (sliding-window or additive) | first attention call | `AttentionInputError` |
| A KV cache present (inference) | first attention call | `AttentionInputError` |

### Turning it on

```python
from mlx_train_perf.attention import enable_flash_attention

enable_flash_attention(model, seq_len=8192, batch_size=1)
```

Call it in place on a loaded model, after you set the compute dtype and before you build the loss and call `train`. mlx-lm's `train` wraps the step in `mx.compile`, and the kernel calibrates itself with a one-time host-synced timing probe. Passing `seq_len` (and `batch_size`) runs that calibration up front, at your training shape, so the compiled step traces with warm caches. Match them to the shape you actually train: `batch_size` defaults to 1 and must equal your training batch. If a compiled `train` traces at a shape the caches were not warmed for, the calibration runs once inside the traced region instead — the run completes, but the timing probe executes on a machine mid-trace rather than in the controlled up-front window (measured on mlx 0.32.0: a one-time stall, not a crash). Omit the hints and the call still succeeds — eager and `mx.grad` callers calibrate lazily on the first attention call — but a compiled `train` run should always pass them for calibration fidelity.

## Sequence packing

New in 0.4.0 and opt-in. Packing concatenates many short sequences into fixed 4,096-token rows so every step runs at full-context efficiency, as the worked example above describes. A block-diagonal attention mask keeps the sequences independent: a token attends another only when both belong to the same original sequence, enforced inside the flash Metal kernels by a per-token segment id rather than a materialized mask (the mask tensor an `(N, N)` approach would need is exactly the quadratic allocation this library exists to avoid).

Loss masking reproduces mlx-lm's unpacked semantics segment by segment, so the supervised token set is identical to an unpacked run. Three sequences packed into one row produce the same token count and a loss within measured bf16 tolerance of the same three run unpacked: worst difference 5.0e-4 against a 2e-2 pin sized from measured RoPE offset drift (`tests/test_adapter_packed.py`). Cross-sequence contamination is tested by construction: deliberately dropping the segment mask in the test suite moves the loss by 0.11, well past the pin.

Measured on Alpaca (pinned revision, 4,000-example sample, seed 42), LoRA rank 8, batch 1, gradient checkpointing, bf16, pack length 4,096 (`scripts/bench_packed_training.py`):

| real tokens/sec | stock batching | packed | ratio |
|---|---|---|---|
| Qwen3-8B-4bit | 33.1 | 99.2 | 3.00× |
| Llama-3.2-3B-4bit | 76.7 | 226.3 | 2.95× |

Samples per hour move the same way: 1,415 → 4,246 on Qwen3-8B and 2,617 → 7,726 on Llama-3.2-3B. "Real tokens" counts sequence content only, never padding or separators. Both arms of each pair were measured in one session at this release's code state.

Where the win comes from matters for whether you will see it too. At batch size 1, stock batching loses little to padding (17% on this dataset, mostly round-to-32 alignment) — the win comes from amortization. A packed row carries roughly 40–50 Alpaca sequences (47.6 on average under Qwen3's tokenizer, 38.1 under Llama's), so the fixed step cost is paid once per ~4,000 real tokens instead of once per 84, and attention runs at its 4,096-token efficiency instead of a ~100-token shape. A dataset of long sequences packs fewer per row and gains less; one that already fills the context gains nothing. The stock arm's per-step median also includes `mx.compile`'s first trace of each batch width (stock widths vary; packed rows are one constant shape, which is itself part of the win), and a long training run amortizes those traces away. Reading the stock arm at its fastest repeated warm step instead of its median gives a conservative bound of about 2.0–2.3×, so the honest range is 2–2.7× on this dataset. The packed arm's own walls are flat to within 5%.

0.5.0 tightens the packed backward: the dK/dV kernel now bounds its query walk at each key block's segment end instead of masking cross-segment work after computing it. Timed with identical dispatch ranges on both arms (`scripts/bench_packed_dkv.py`), the dK/dV pass on an Alpaca-like row runs 6.2× faster at 4,096 tokens and 8.3× at 8,192; a single-segment row is unchanged. That pass is one slice of the training step, so the win on the full step is smaller: on Qwen3-8B-4bit the packed arm's median step drops from 44.7 s to 40.4 s (+10.7% tokens/sec, with the unpacked arm within half a percent of its prior measurement — the control that pins the gain to the packed backward). The table above is this release's measurement of both models.

### Training packed

The parts drop into the stock trainer the same way the loss does — a batch iterator, a loss function, and the flash-attention switch. Packing requires the flash path (the stock attention cannot express a block-diagonal mask):

```python
import functools
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import train
from mlx_train_perf.adapters.mlx_lm import make_packed_loss_fn
from mlx_train_perf.attention import enable_flash_attention
from mlx_train_perf.data.packing import packed_iterate_batches

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
model.set_dtype(mx.bfloat16)  # 4-bit checkpoints compute in fp16; the kernels need bf16/fp32
enable_flash_attention(model, seq_len=4096, batch_size=1, packed=True)
# ... freeze the base model and apply linear_to_lora_layers as usual ...

train(
    model=model, optimizer=opt,
    train_dataset=dataset,          # items are (tokens, offset) pairs; offset = prompt length
    args=args,                      # args.max_seq_length is the pack length
    loss=make_packed_loss_fn(model),
    iterate_batches=functools.partial(
        packed_iterate_batches,
        max_position_embeddings=model.args.max_position_embeddings,
    ),
)
```

`packed_iterate_batches` re-packs each epoch with a fresh shuffle and hands the trainer fixed-shape batches; `make_packed_loss_fn` walks the model's layers itself to thread the segment mask (the stock model call hardcodes a causal mask) and refuses at construction if `enable_flash_attention` has not run. Pass `packed=True` with `seq_len` equal to your pack length and `batch_size` equal to your training batch: the calibration caches key on the exact batch size and sequence bucket, so matching hints keep the one-time kernel timing probes in the controlled window before `mx.compile` traces the step. The pack length must not exceed the model's trained context — packed sequences keep their relative positions, and the row as a whole runs at absolute positions up to the pack length.

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

Pass `--attention flash` to price the flash-attention path instead of the stock backward:

```bash
mlx-train-perf plan --config path/to/config.json --batch 1 --seq-len 8192 --lora-rank 8 --attention flash
```

The flash model is an analytic saved-state term plus one measured linear coefficient, fit as an envelope over the worst-case measured loss arm so it never under-predicts at a measured anchor. That makes it read more conservatively for the fused loss in particular: up to about 1.4× the measured peak on the fitted model, about 1.6× cross-model. The validated range is 2,048 to 12,288 tokens; past that the fit extrapolates.

Instead of checking one config at a time, ask the planner for the largest sequence length or batch size that fits your budget:

```bash
mlx-train-perf plan --config path/to/config.json --batch 1 --lora-rank 8 --attention flash --max-seq
mlx-train-perf plan --config path/to/config.json --seq-len 8192 --lora-rank 8 --attention flash --max-batch
```

`--max-seq` searches for the largest `--seq-len` and still needs `--batch`; `--max-batch` searches for the largest `--batch` and still needs `--seq-len`. A budget that nothing fits, even at the smallest value searched, is refused with a typed error.

## Supported models

- Architectures: Llama, Qwen2 (the Qwen2.5 family), and Qwen3, for both the loss adapter and the flash-attention wrapper. The adapter's model splitter handles these; others raise a typed error.
- Quantization: 4-bit group-size-64 (the mlx-community QLoRA default), or a dense fp32/bf16 head.
- Training: LoRA / QLoRA. Full fine-tuning is estimated by the planner but is not the case this is tuned for.
- Hardware: Apple Silicon.

## Reproducing the numbers

Each claim above has one script. They run on the GPU, take real wall-clock time, and print the artifacts they measured:

```bash
python scripts/bench_loss_layer.py        # the ~3900x loss-layer memory number
python scripts/bench_attention_op.py      # the single-op flash vs stock memory + timing
# the 12.75 vs 25.68 GiB training table: run each attention arm into its own --out dir
python scripts/bench_train_step.py --model mlx-community/Qwen3-8B-4bit --seq-len 8192 \
    --attention flash --impl kernel --compute-dtype bfloat16 --grad-checkpoint --out _artifacts/flash
python scripts/bench_train_step.py --model mlx-community/Qwen3-8B-4bit --seq-len 8192 \
    --attention stock --impl kernel --compute-dtype bfloat16 --grad-checkpoint --out _artifacts/stock
python scripts/northstar_context_sweep.py # the max-context sweep (1-2 h; heavy)
# the packed dK/dV block-skip ratios (6.2x / 8.3x): one invocation per layout and length
python scripts/bench_packed_dkv.py --n 4096 --layout alpaca --out _artifacts/packed_dkv
python scripts/bench_packed_dkv.py --n 8192 --layout alpaca --out _artifacts/packed_dkv
# the planner's flash-fit anchors and refit (envelope over the committed manifest)
python scripts/fit_calibration.py --manifest _artifacts/calib_050/refit_manifest.json --dry-run
# the packing table (3.00x / 2.95x): prep the dataset once per model, then run each arm
# into its own --out dir (30 timed steps per arm, the script default, matching the
# committed artifacts)
python scripts/prep_alpaca.py --model mlx-community/Qwen3-8B-4bit \
    --out _artifacts/packed_bench/alpaca_qwen3.jsonl --batch-size 1 --pack-len 4096 --max-samples 4000 --seed 42
python scripts/bench_packed_training.py --model mlx-community/Qwen3-8B-4bit \
    --data _artifacts/packed_bench/alpaca_qwen3.jsonl --arm stock --pack-len 4096 --batch-size 1 \
    --grad-checkpoint --compute-dtype bfloat16 --out _artifacts/packed_bench_050
python scripts/bench_packed_training.py --model mlx-community/Qwen3-8B-4bit \
    --data _artifacts/packed_bench/alpaca_qwen3.jsonl --arm packed --pack-len 4096 --batch-size 1 \
    --grad-checkpoint --compute-dtype bfloat16 --out _artifacts/packed_bench_050
```

## Memory safety net

Every benchmark and contribution run is fenced by a device-relative memory guard. A GPU over-allocation on Apple Silicon does not always fail cleanly: `mx.set_memory_limit` is advisory, so an allocation past the soft cap pages instead of raising, and a hard enough paging storm can panic the machine rather than kill the process.

The guard sets an active-memory ceiling from the machine's own RAM. It is anchored at 28 GiB on a 32 GB Mac — above the largest legitimate peak measured here (25.68 GiB) and below physical RAM — and scales from that anchor across the range from 16 GB up to a 1 TB machine. At start it takes the smaller of that static ceiling and what the machine actually has free right now, minus a 2 GiB cushion. A daemon thread samples active memory throughout the run and aborts the moment it reaches the ceiling, writing an honest aborted-status artifact instead of letting the storm build. If the machine is already too loaded to start safely — less than a quarter of RAM effectively available — the run refuses up front with a typed error. Between those two points it proceeds but prints a warning naming how much memory it expected free for the machine's class against how much it measured, so a crowded machine is visible rather than silent.

The guard is rank-local: every input it reads is this node's own RAM, availability, and process memory. On a multi-node `mx.distributed` job each rank sizes its own ceiling and flags its own crowding, and a breach hard-exits that rank — so run distributed training under a launcher (`mpirun` or `mlx.launch`) that propagates a rank failure to the whole job.

The incident that motivated this guard, and what the watchdog does and does not cover, is documented in [When an MLX memory cap is not a safety boundary](https://ineshin.space/papers/when-an-mlx-memory-cap-is-not-a-safety-boundary/).

## Research

Four write-ups cover the work behind this library in more depth than a README can, including the
measurements that went the wrong way. They are published at [ineshin.space](https://ineshin.space)
alongside the rest of my Apple Silicon work, and the source Markdown lives under `docs/papers/`.

- [Fused linear cross-entropy on Apple GPUs](https://ineshin.space/papers/fused-linear-cross-entropy-apple-gpus/)
  explains how vocabulary chunking and a fused Metal kernel avoid materializing logits. It covers
  memory costs, the optimization ladder, failed performance models, and the limits of the evidence.
- [When the bottleneck moved: from fused cross-entropy to FlashAttention on MLX](https://ineshin.space/papers/when-the-bottleneck-moved/)
  explains why removing the logits matrix did not extend context once attention backward set the
  peak. It also covers the command-buffer correction that removed a false launch limit, while
  separating source-reported measurements from claims the available controls cannot support.
- [How MLX packs Metal command buffers](https://ineshin.space/papers/how-mlx-packs-metal-command-buffers/)
  explains the operation and element thresholds that MLX 0.32.0 uses to commit Metal work. It applies
  them to tiled attention, then explains why a whole-chain launch budget rejected valid work. The
  macOS watchdog mechanism remains an inference.
- [When an MLX memory cap is not a safety boundary](https://ineshin.space/papers/when-an-mlx-memory-cap-is-not-a-safety-boundary/)
  reports the kernel-panic incident behind the memory guard: a wired limit that caps residency
  without rejecting allocation, an advisory soft limit, and the active-memory watchdog added as a
  third layer. It separates the observed record from reconstruction and keeps the panic-trigger
  mechanism labeled as an unverified hypothesis.

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
| Apple M1 Max | 32 | 0.32.0 | quick | 0.0006 | 3.06 | — | — |
<!-- /community-benchmarks:table -->

The "Attn flash 2x ratio" column is how the flash forward+backward peak grows per sequence
doubling. The O(N) target is about 2×. On the reference machine the largest measured pair
(4096→8192) reads 3.06× rather than 2×, because the chained backward split adds a small,
budget-bounded constant of at most ~0.3 GB; the growth law is still linear, so a reading a
little above 2× on a given machine is expected, not a regression.

## License

MIT. See [LICENSE](LICENSE).
