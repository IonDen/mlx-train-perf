# Roadmap

## Released

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

### Faster backward kernels to raise the context ceiling
On a 32 GB machine the flash-attention path is capped by a per-launch GPU-safety budget, not
by memory, so it does not currently extend the maximum trainable context there. Faster
backward kernels raise that budget cap and let the ceiling grow with the flash path's memory
headroom, on 32 GB and above.

### Fused backward kernel for the loss
The loss ships a fused forward paired with a proven chunked backward. A fully fused backward
kernel is a further memory reduction for the loss layer itself.

### Sequence packing
Pack variable-length examples into fixed blocks to cut padding waste in SFT training. This
needs a block-diagonal (additive) attention mask so examples packed into one block do not
attend across each other — a mask the current causal-only flash path refuses, so supporting
it means extending the flash kernel to a packed-block mask.

### Distinct too-crowded artifact status
When a run refuses because the machine is too loaded to start safely, record that with its own
status in the artifact, separate from a memory-ceiling abort or a clean result, so the two are
easy to tell apart when aggregating community submissions.

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
