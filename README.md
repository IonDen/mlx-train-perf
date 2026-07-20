# mlx-train-perf

[![PyPI version](https://img.shields.io/pypi/v/mlx-train-perf.svg)](https://pypi.org/project/mlx-train-perf/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-train-perf.svg)](https://pypi.org/project/mlx-train-perf/)
[![License: MIT](https://img.shields.io/pypi/l/mlx-train-perf.svg)](https://github.com/IonDen/mlx-train-perf/blob/main/LICENSE)

A fused, logit-free linear-cross-entropy loss for training on Apple Silicon with [MLX](https://github.com/ml-explore/mlx), plus a RAM-fit planner and an honest benchmark harness. It drops into an `mlx-lm` LoRA/QLoRA fine-tune as the loss function.

The idea is the same one behind [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) on the CUDA side, ported to a Metal kernel: compute the cross-entropy loss and its gradient without ever building the full `(N, V)` logits tensor. For a large vocabulary that tensor is the single biggest allocation in the training step, and it is pure waste. You only need the per-token loss and a gradient back into the hidden states.

Released on PyPI as `mlx-train-perf`. Every number below has a committed script under `scripts/` that reproduces it, all measured on one M1 Max (32 GB, macOS 26.5). The loss-layer figures were taken on mlx 0.31.2 and reproduce on the pinned 0.32.0; the flash-attention memory figures were taken on 0.32.0 in 0.2.0, and the 0.3.0 context-ceiling figures on 0.32.0.

## The problem it solves

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

The 32 GB machine that peaked near its ceiling with stock attention now runs the same step at half the memory. That is real headroom: a longer sequence, or a second job on the GPU.

The attention op itself, timed alone at the flagship shape (batch 1, 32 query / 8 KV heads, 8192 tokens, head_dim 128), is 0.186 s on the forward and 0.576 s on the full backward, 3.1× the forward (`scripts/bench_attention_op.py`). As the sequence doubles, the flash op's peak grows about 2.00× (2048→4096) and about 3.06× (4096→8192). The second step is above 2× because the chained backward split adds a small, budget-bounded constant of at most ~0.3 GB, not because the O(N) growth law changed. Stock attention over the same doublings grows 3.76× then 3.05×, and that last figure is flattered by paging: at 8192 the stock op allocates about 32.4 GB on a 32 GB machine and its wall time degrades roughly 41× as it pages, which puts the single flash op about 45× below stock there. Those stock figures are a pre-net measurement of the exact hazard the safety net now prevents. They were taken before the memory watchdog shipped, and on a 32 GB machine `scripts/bench_attention_op.py` now aborts that condition by design (`aborted_memory_ceiling`) rather than paging into it. The flash-side numbers all reproduce. Flash is not universally cheaper, though — below about 2100 tokens the stock op's simpler bookkeeping wins, and the two curves cross there. The win is at real training context, not tiny shapes. (At 16384 the flash op now runs rather than refusing, since the launch guard no longer caps the chain; stock attention still cannot reach that context on 32 GB.)

### What it costs in throughput

Turning flash attention on is not free. On the stock-loss path at 8192 it costs 5.3% of tokens/sec (74.0 vs 78.1); at 2048 the cost is 5.5% on the fused-loss path (86.4 vs 91.5) and 5.9% on the stock-loss path (92.1 vs 97.8). The fused-loss comparison at 8192 has no stock-attention number to pair with: on this 32 GB machine that baseline condition crosses the memory safety net's ceiling and records an abort instead of a number. That baseline running out of room is the problem flash attention exists to remove. Under flash attention the fused cross-entropy and mlx-lm's stock cross-entropy stay close: 0.94× at 2048 (86.4 vs 92.1 tok/s) and 0.99× at 8192 on Qwen3-8B, 0.92× and 0.97× on Llama-3.2-3B. The loss values match to bf16 tolerance throughout — the worst per-step difference across every measured pair is 2.4e-3. The worst attention-arm throughput ratio measured is 0.94× stock; 0.85× is the maintainer's release acceptance bar for that path, provisional and pinned at the PR.

### What it changes: the context ceiling on 32 GB

0.2.0 shipped this path with a caveat — it halved the memory but did not extend the longest sequence you could train, because a launch-safety budget capped it before memory did. That cap turned out to be guarding the wrong thing. Re-reading how mlx schedules Metal work, and re-running the crash that motivated the budget, showed the GPU watchdog acts on a single command buffer, not on a chain of them, and at training shapes each backward dispatch already runs in its own buffer. 0.3.0 replaces the per-chain budget with a per-buffer one that models what the scheduler actually commits. No safety margin moved; what changed is what the margin is measured against. (`scripts/probe_command_buffer_packing.py` reproduces the evidence.)

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

New in 0.4.0 and opt-in. Instruction-tuning datasets are short and ragged: Alpaca under Qwen3's chat template averages 84 tokens per example, and mlx-lm's trainer runs one step per batch of them. At batch size 1 on an 8B model, a compiled training step costs about 2 to 2.5 seconds whether it carries 84 tokens or 4,096 — the fixed per-step cost dominates and the GPU idles. Packing concatenates many sequences into fixed 4,096-token rows so every step runs at full-context efficiency. A block-diagonal attention mask keeps the sequences independent: a token attends another only when both belong to the same original sequence, enforced inside the flash Metal kernels by a per-token segment id rather than a materialized mask (the mask tensor an `(N, N)` approach would need is exactly the quadratic allocation this library exists to avoid).

Loss masking reproduces mlx-lm's unpacked semantics segment by segment, so the supervised token set is identical to an unpacked run. Three sequences packed into one row produce the same token count and a loss within measured bf16 tolerance of the same three run unpacked: worst difference 5.0e-4 against a 2e-2 pin sized from measured RoPE offset drift (`tests/test_adapter_packed.py`). Cross-sequence contamination is tested by construction: deliberately dropping the segment mask in the test suite moves the loss by 0.11, well past the pin.

Measured on Alpaca (pinned revision, 4,000-example sample, seed 42), LoRA rank 8, batch 1, gradient checkpointing, bf16, pack length 4,096 (`scripts/bench_packed_training.py`):

| real tokens/sec | stock batching | packed | ratio |
|---|---|---|---|
| Qwen3-8B-4bit | 32.9 | 89.6 | 2.72× |
| Llama-3.2-3B-4bit | 71.5 | 194.4 | 2.72× |

Samples per hour move the same way: 1,408 → 3,835 on Qwen3-8B and 2,441 → 6,638 on Llama-3.2-3B. "Real tokens" counts sequence content only, never padding or separators.

Where the win comes from matters for whether you will see it too. At batch size 1, stock batching loses little to padding (17% on this dataset, mostly round-to-32 alignment) — the win comes from amortization. A packed row carries roughly 40–50 Alpaca sequences (47.6 on average under Qwen3's tokenizer, 38.1 under Llama's), so the fixed step cost is paid once per ~4,000 real tokens instead of once per 84, and attention runs at its 4,096-token efficiency instead of a ~100-token shape. A dataset of long sequences packs fewer per row and gains less; one that already fills the context gains nothing. The stock arm's per-step median also includes `mx.compile`'s first trace of each batch width (stock widths vary; packed rows are one constant shape, which is itself part of the win), and a long training run amortizes those traces away. Reading the stock arm at its fastest repeated warm step instead of its median gives a conservative bound of about 2.0–2.3×, so the honest range is 2–2.7× on this dataset. The packed arm's own walls are flat to within 5%.

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

## Install

```bash
pip install mlx-train-perf            # the loss kernel + planner
pip install "mlx-train-perf[mlx-lm]"  # plus the mlx-lm training adapter
```

Apple Silicon only. Requires mlx >=0.32.0,<0.33 — the version the kernels' JIT contract is verified against. The mlx-lm adapter and the flash-attention wrapper need the optional `mlx-lm` extra.

## Use it in an mlx-lm fine-tune

The adapter builds a loss callable with the same signature `mlx_lm`'s trainer expects, so you pass it straight to `train(...)`. `enable_flash_attention` is the second, independent lever — turn on either, both, or neither:

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import train
from mlx_train_perf.adapters.mlx_lm import make_loss_fn
from mlx_train_perf.attention import enable_flash_attention

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
model.set_dtype(mx.bfloat16)  # 4-bit checkpoints compute in fp16; the kernels need bf16/fp32
# ... freeze the base model and apply linear_to_lora_layers as in a normal mlx-lm LoRA run ...

enable_flash_attention(model, seq_len=8192, batch_size=1)  # O(N) attention backward
loss_fn = make_loss_fn(model, impl="auto")                 # logit-free cross-entropy
train(model=model, optimizer=opt, train_dataset=ds, args=args, loss=loss_fn)
```

`make_loss_fn` splits the model into its trunk and its output head and routes the loss through the fused kernel; `enable_flash_attention` swaps each layer's attention for the flash path in place.

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

The flash model is an analytic saved-state term plus one measured linear coefficient. Across the four measured anchors up to 8192 tokens it over-predicts the cushioned total by 8% to 20% — again on the safe side.

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
# the 2.72x packing table: prep the dataset once per model, then run each arm into its own --out dir
python scripts/prep_alpaca.py --model mlx-community/Qwen3-8B-4bit \
    --out _artifacts/packed_bench/alpaca_qwen3.jsonl --batch-size 1 --pack-len 4096 --max-samples 4000 --seed 42
python scripts/bench_packed_training.py --model mlx-community/Qwen3-8B-4bit \
    --data _artifacts/packed_bench/alpaca_qwen3.jsonl --arm stock --pack-len 4096 --batch-size 1 \
    --steps 60 --grad-checkpoint --compute-dtype bfloat16 --out _artifacts/packed_bench/qwen3_stock
python scripts/bench_packed_training.py --model mlx-community/Qwen3-8B-4bit \
    --data _artifacts/packed_bench/alpaca_qwen3.jsonl --arm packed --pack-len 4096 --batch-size 1 \
    --steps 30 --grad-checkpoint --compute-dtype bfloat16 --out _artifacts/packed_bench/qwen3_packed
```

## Memory safety net

Every benchmark and contribution run is fenced by a device-relative memory guard. A GPU over-allocation on Apple Silicon does not always fail cleanly: `mx.set_memory_limit` is advisory, so an allocation past the soft cap pages instead of raising, and a hard enough paging storm can panic the machine rather than kill the process.

The guard sets an active-memory ceiling from the machine's own RAM. It is anchored at 28 GiB on a 32 GB Mac — above the largest legitimate peak measured here (25.68 GiB) and below physical RAM — and scales from that anchor across the range from 16 GB up to a 1 TB machine. At start it takes the smaller of that static ceiling and what the machine actually has free right now, minus a 2 GiB cushion. A daemon thread samples active memory throughout the run and aborts the moment it reaches the ceiling, writing an honest aborted-status artifact instead of letting the storm build. If the machine is already too loaded to start safely — less than a quarter of RAM effectively available — the run refuses up front with a typed error. Between those two points it proceeds but prints a warning naming how much memory it expected free for the machine's class against how much it measured, so a crowded machine is visible rather than silent.

The guard is rank-local: every input it reads is this node's own RAM, availability, and process memory. On a multi-node `mx.distributed` job each rank sizes its own ceiling and flags its own crowding, and a breach hard-exits that rank — so run distributed training under a launcher (`mpirun` or `mlx.launch`) that propagates a rank failure to the whole job.

## Research

- [Fused linear cross-entropy on Apple GPUs](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/fused-linear-cross-entropy-apple-gpus.md)
  explains how vocabulary chunking and a fused Metal kernel avoid materializing logits. It covers
  memory costs, the optimization ladder, failed performance models, and the limits of the evidence.
- [When the bottleneck moved: from fused cross-entropy to FlashAttention on MLX](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/when-the-bottleneck-moved.md)
  explains why removing the logits matrix did not extend context once attention backward set the
  peak. It also covers the command-buffer correction that removed a false launch limit, while
  separating source-reported measurements from claims the available controls cannot support.

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
