# Roadmap

## Released

### 0.1.0 - 2026-07-08
- Fused, logit-free linear-cross-entropy Metal kernel (forward), with a chunked pure-MLX
  fallback and a materialized correctness oracle.
- Quantized (4-bit group-size-64) head support.
- mlx-lm training adapter, RAM-fit planner, and benchmark harness.

## Planned

### Community benchmark contribution kit
Package the benchmark harness into a one-command contribution flow for Apple-Silicon users
on other memory sizes. The goal is a provenance-complete artifact that users can submit
without hand-editing numbers, plus a generated table of community-measured results.

### Memory-efficient attention backward
This is the highest-leverage item. The measured bottleneck for long-context training on
MLX is the attention backward, which materializes the `(N, N)` score matrix one layer at
a time. A tiled, recompute-based attention backward (the FlashAttention backward, written
as a Metal kernel) is what would actually raise the maximum trainable context on Apple
Silicon. The fused loss frees the logit memory; this frees the attention memory, and
together they are what "train much longer sequences" needs.

### Fused backward kernel for the loss
The loss ships with a fused forward and a proven chunked backward. A fully fused backward
kernel is a further memory reduction for the loss layer itself.

### Sequence packing
Pack variable-length examples into fixed blocks to cut padding waste in SFT training.

### Planner inverse queries
Ask the planner for the largest batch or sequence length that fits a given memory budget,
instead of checking one config at a time.

## Not doing
- Trainer UX or a training-loop framework. That is the lane of tools like mlx-lm-lora and
  mlx-tune; this project is a layer they can import.
- Non-Apple-Silicon backends.
