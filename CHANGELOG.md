# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-08

First release. A fused, logit-free linear-cross-entropy loss for MLX training on Apple
Silicon, with an mlx-lm adapter, a RAM-fit planner, and a benchmark harness.

### Added
- Fused Metal cross-entropy kernel that never materializes the `(N, V)` logits: about
  3900x less loss-layer memory than materialized logits in isolation, at a 1.64x forward
  cost (n=8192, V=151936, D=4096, bf16; `scripts/bench_loss_layer.py`). Exact value and
  gradient parity against a materialized reference.
- Three implementations behind one `impl` argument: `kernel` (the fused Metal path),
  `chunked` (a pure-MLX fallback bounded by a fixed vocabulary tile), and `naive` (the
  materialized correctness oracle). `auto` selects the kernel when the mlx version is
  verified and the head and dtype are supported, and otherwise raises a typed error
  naming the reason and the alternatives. It never silently downgrades.
- Quantized-head support: 4-bit group-size-64 heads (the mlx-community QLoRA default) run
  through the kernel.
- mlx-lm training adapter (`make_loss_fn`) that plugs the loss into `mlx_lm`'s compiled
  trainer, with per-step loss curves matching the stock trainer to bf16 tolerance
  (about 2e-3). End-to-end throughput is roughly 8-12% slower per step at bf16
  (`scripts/bench_train_step.py`).
- RAM-fit planner that estimates peak training memory and accounts for the O(N^2)
  attention backward that dominates at long context. Fit to measured Qwen3-8B train-step
  peaks and cross-model validated on Llama-3.2-3B to within about 9%.
- Benchmark harness and committed scripts for every published number.

### Known limits
- The fused loss frees memory at a given context but does not, on its own, extend the
  maximum trainable context on MLX. `mx.fast.scaled_dot_product_attention` has an O(N^2)
  backward that materializes the `(N, N)` attention matrix and becomes the memory
  bottleneck at long context. A memory-efficient attention backward is the next step (see
  `ROADMAP.md`).
- Architectures: Llama and Qwen3 only. Training: LoRA / QLoRA. Apple Silicon only.

[0.1.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.1.0
