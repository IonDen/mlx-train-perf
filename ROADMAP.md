# Roadmap

## Released

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

### Sequence packing
Pack variable-length examples into fixed blocks to cut padding waste in SFT training. This
needs a block-diagonal (additive) attention mask so examples packed into one block do not
attend across each other — a mask the current causal-only flash path refuses, so supporting
it means extending the flash kernel to a packed-block mask.

### Planner anchors beyond 8192
The flash planner coefficient is fit to anchors up to 8192 tokens. Add measured anchors past
8192 so the flash fit is validated at longer context instead of extrapolated.

### Planner inverse queries
Ask the planner for the largest batch or sequence length that fits a given memory budget,
instead of checking one config at a time.

## Not doing
- Trainer UX or a training-loop framework. That is the lane of tools like mlx-lm-lora and
  mlx-tune; this project is a layer they can import.
- Non-Apple-Silicon backends.
