# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-14

Adds an opt-in flash-attention training path (Metal forward and backward) that halves the
training-step peak memory at 8192-token context, alongside a device-relative memory safety
net, a one-command community benchmark kit, and a flash-aware planner. The 0.1.0 loss API is
unchanged. Measured on an M1 Max (32 GB, macOS 26.5, mlx 0.32.0), Qwen3-8B-4bit unless noted.

### Added
- Flash-attention Metal kernels (forward + backward) for the training path. At seq 8192
  (LoRA r=8, batch 1, gradient checkpointing on, bf16) the whole train-step peak is
  12.75 GiB with flash attention against 25.68 GiB with stock attention — about half
  (`scripts/bench_train_step.py`). Timed alone at the flagship shape (batch 1, 32 query /
  8 KV heads, 8192 tokens, head_dim 128) the op is 0.186 s forward and 0.576 s on the full
  backward, 3.1x the forward (`scripts/bench_attention_op.py`); at 8192 the single-op flash
  peak sits about 45x below stock. That stock reading is a pre-net measurement of the hazard
  the safety net now prevents: it was measured before the memory watchdog shipped, and on a
  32 GB machine `bench_attention_op.py` now aborts that condition by design
  (`aborted_memory_ceiling`) rather than paging; the flash-side numbers reproduce. Exact
  value and gradient parity against a pure-MLX oracle;
  loss curves match the stock trainer to bf16 tolerance (worst per-step diff 2.4e-3).
- `enable_flash_attention(model, seq_len=..., batch_size=...)`: opt-in mlx-lm wrapper that
  swaps each decoder layer's attention for the flash path in place, for the Llama
  (full-attention) and Qwen3 families. Causal- and training-only: it refuses sliding-window
  or mixed attention, a head_dim outside {64, 96, 128}, attention dropout, an array mask, and
  a KV cache with a typed error at enable time or the first attention call. The
  `seq_len`/`batch_size` hints pre-warm the kernel calibration so a subsequently compiled
  `train()` traces without a host sync inside its compiled region.
- Throughput cost of turning flash attention on: 5.3% of tokens/sec on the stock-loss arm
  at 8192 (74.0 vs 78.1 tok/s); 5.5% (fused-loss) and 5.9% (stock-loss) at 2048
  (`scripts/bench_train_step.py`). The fused-loss arm at 8192 has no stock-attention pair:
  that baseline crosses the memory ceiling on a 32 GB machine and records an abort instead
  of a number. Under flash attention the fused cross-entropy stays close to stock
  cross-entropy (0.94x at 2048, 0.99x at 8192; 0.92x / 0.97x on Llama-3.2-3B).
- Planner `--attention flash`: prices the flash path with an analytic saved-state term plus
  one measured linear coefficient. It over-predicts the cushioned total by 8% to 20% at the
  four measured anchors up to 8192 tokens — on the safe side for a fit tool.
- Memory safety net for every bench and contribution run: a device-relative active-memory
  ceiling (anchored at 28 GiB on a 32 GB Mac, scaling with RAM up to a 1 TB machine), taken
  as the smaller of that ceiling and measured availability at start minus 2 GiB. A daemon
  watchdog aborts a runaway before it can page into a machine-panicking storm and writes an
  aborted-status artifact; a machine with under a quarter of RAM free refuses up front; a
  moderately loaded one warns with expected-versus-measured free memory. Every input is
  rank-local, so it is safe under `mx.distributed`.
- Community benchmark kit: `mlx-train-perf contribute --tier quick|full` detects the machine,
  sizes shapes to its RAM, prints an honest time estimate and any pre-flight warning before
  asking for confirmation, runs the committed benches under the safety net, and writes one
  provenance-complete artifact plus a ready-to-paste PR. Merged submissions aggregate into a
  community table in the README (`scripts/aggregate_community.py`).
- Committed benches `scripts/bench_attention_op.py` (single-op flash vs stock) and
  `scripts/northstar_context_sweep.py` (max-context binary search). On a 32 GB machine the
  longest trainable sequence is 10,240 tokens on the flash-plus-fused-loss path against 9,728
  for stock, measured the same day with the same search.

### Changed
- Minimum mlx is now 0.32.0 (bounded `>=0.32.0,<0.33`); the kernel JIT contract is re-verified
  against it. The `mlx-lm` extra still pins `transformers>=5.0,<5.13`.

### Upgrade notes
0.2.0 is additive. The 0.1.0 loss API (`linear_cross_entropy`, `make_loss_fn`, `impl="auto"`)
is unchanged, and the flash-attention path, the safety net, and the contribution kit are all
opt-in surfaces you reach for by name. The one required change is the mlx floor: 0.2.0 needs
mlx 0.32.0 where 0.1.0 accepted 0.31.2. On a 32 GB machine flash attention roughly halves the
step's peak memory at 8192 but does not extend the maximum trainable context there (it is
capped by a kernel launch-safety budget, not memory); the win to expect is the headroom. That
launch-safety cap comes from kernel launch throughput and does not depend on RAM: extra
memory raises stock attention's memory-bound ceiling, while the flash path stays near 10k
until faster dK/dV backward kernels or new launch-budget evidence lift its cap. The context
figures in this release are not comparable to the numbers 0.1.0 reported.

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

[0.2.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.2.0
[0.1.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.1.0
