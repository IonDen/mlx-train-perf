# Roadmap

## Released

### 0.5.0 - 2026-07-23
- Packed dK/dV block skipping. The packed backward kernel now bounds its query walk at
  each key block's segment end instead of masking cross-segment work per element.
  Measured with identical dispatch ranges on both arms, the dK/dV pass on an Alpaca-like
  row runs 6.2× faster at 4,096 tokens and 8.4× at 8,192; a single-segment row is
  unchanged. End to end on Qwen3-8B-4bit, packing's real-token throughput moves from
  2.72× to 3.00× against unpacked batching, and the packed arm's median step drops from
  44.7 s to 40.4 s.
- Planner anchors past 8,192 tokens, with a safer flash fit. New measured anchors at
  10,240 and 12,288 tokens showed the flash memory coefficient under-predicting the
  stock-loss arm at long context. The fit now takes an envelope over the worst measured
  arm, so `plan --attention flash` never under-predicts within its validated range
  (2,048-12,288 tokens), at the cost of reading more conservatively for the fused loss.
- Planner inverse queries. `plan --max-seq` and `--max-batch` return the largest sequence
  length or batch size that fits a memory budget, instead of checking one config at a
  time.

### 0.4.0 - 2026-07-19
- Sequence packing for instruction tuning. Short examples share fixed 4,096-token training
  rows, kept independent by a block-diagonal mask enforced inside the flash kernels with an
  O(N) per-token segment id. Loss masking matches the unpacked trainer segment by segment
  (parity 5.0e-4 on a packed-vs-unpacked batch). Measured on Alpaca at batch 1: real-data
  throughput 2.72× on Qwen3-8B-4bit and on Llama-3.2-3B-4bit alike, a 2.0–2.3× conservative
  steady-state reading — at batch 1 the win is per-step amortization, not padding removal.

### 0.3.1 - 2026-07-15
- Qwen2 (Qwen2.5 family) support in the flash-attention wrapper. `enable_flash_attention` now
  accepts Qwen2 alongside Llama and Qwen3, so a Qwen2.5 fine-tune can use the flash path as
  well as the fused loss adapter it already supported. Forward output matches stock attention
  within 2e-6 in fp32 on both the reference and kernel paths.

### 0.3.0 - 2026-07-14
- Removed the launch-safety cap that held the flash-attention path's trainable context below
  its memory limit. The 2-second per-chain budget guarded a misread of how macOS kills Metal
  work: the watchdog acts on a single command buffer, not a chain, and at training shapes
  each backward dispatch already runs in its own buffer. The guard is now per command buffer,
  at the same 0.5-second worst-day budget. On a 32 GB machine the maximum trainable context
  measured 23,040 tokens with flash attention against 7,936 with stock, both bound by the
  same effective memory ceiling — the flash path reaches 2.9x the context at one memory
  budget, because it keeps O(N) saved state.
- Qwen2 (Qwen2.5 family) support in the mlx-lm loss adapter, with loss parity against the
  stock trainer verified on a real checkpoint.
- Benchmark hygiene: attention-arm artifact filenames, a distinct too-crowded status that
  re-runs on a quieter machine, and a measured quick-tier time estimate for the community kit.

### 0.2.0 - 2026-07-14
- Flash-attention training path: a Metal forward and backward that keep O(N) saved state
  instead of the O(N²) score matrix. At 8192-token context on Qwen3-8B-4bit it roughly halves
  the train-step peak memory. On a 32 GB machine the maximum trainable context is near-tied
  with stock attention, because this path is capped by a kernel launch-safety budget rather
  than by memory. That cap comes from kernel launch throughput and does not depend on RAM:
  extra memory raises stock attention's memory-bound ceiling, while the flash path stays near
  10k until faster dK/dV backward kernels or new launch-budget evidence lift it.
- `enable_flash_attention` wrapper for the Llama (full-attention) and Qwen3 families, with
  typed refusals for the cases a causal training path does not serve.
- Flash-aware RAM-fit planner (`plan --attention flash`).
- Device-relative memory safety net: an active-memory ceiling scaled to the machine's RAM, a
  watchdog that aborts a runaway before it can panic the machine, and a too-crowded refusal,
  all rank-local for `mx.distributed`.
- Community benchmark contribution kit (`mlx-train-perf contribute`) and an aggregated
  community table in the README.

### 0.1.0 - 2026-07-08
- Fused, logit-free linear-cross-entropy Metal kernel (forward), with a chunked pure-MLX
  fallback and a materialized correctness oracle.
- Quantized (4-bit group-size-64) head support.
- mlx-lm training adapter, RAM-fit planner, and benchmark harness.

## Planned

### Faster dK/dV backward kernels
The dK/dV backward is the slowest part of the flash path. A register-scheduling change to its
Metal kernel (interleaving the key-tile work to lower register pressure) is the next lever on
its throughput, which sets tokens/sec at long context. This is a speed item, not a memory or
context-ceiling one — 0.3.0 already made the context ceiling memory-bound.

### Fused backward kernel for the loss
The loss ships a fused forward paired with a proven chunked backward. A fully fused backward
kernel is a further memory reduction for the loss layer itself.

## Not doing
- Trainer UX or a training-loop framework. That is the lane of tools like mlx-lm-lora and
  mlx-tune; this project is a layer they can import.
- Non-Apple-Silicon backends.
