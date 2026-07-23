# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-23

Bounds the packed dK/dV backward kernel's query walk at each key block's segment end,
instead of masking cross-segment work after computing it. Measured with identical
dispatch ranges on both arms, interleaved reps (`scripts/bench_packed_dkv.py`), the dK/dV
pass on an Alpaca-like layout (about 45 segments per 4,096-token row) runs 6.2× faster at
4,096 tokens and 8.4× at 8,192; a single-segment row is unchanged (1.00×, 1.02×). End to
end on Qwen3-8B-4bit (LoRA, gradient checkpointing, batch 1, both arms measured fresh at
this release's code, `scripts/bench_packed_training.py`), packed throughput is 99.2 real
tokens/sec against 33.1 unpacked, 3.00× (0.4.0 measured 2.72×), and the packed arm's
median step drops from 44.7 s to 40.4 s. Also widens the planner's flash-attention fit
past 8,192 tokens and adds inverse queries for the largest sequence length or batch a
memory budget allows.

### Added
- Packed dK/dV block skipping: the dK/dV pass is one slice of the training step, so the
  6-8× kernel win (above) lands as about a 10% cut to the whole step. The unpacked arm is
  within 0.4% of its 0.4.0 measurement, which is what pins the end-to-end gain to the
  packed backward rather than to bench noise. This release's runs also set the compute
  dtype to bfloat16 explicitly, where the prior run left it at the model default; the
  unpacked arm's flatness shows that change contributes nothing material. The op bench
  dispatches through fixed 1,024-row chunks to isolate the block-skip from the dispatch
  planner; production dispatches the full row at these shapes, and the skip logic is the
  same either way. Single-segment packed output stays bit-identical to before. A new
  pack-time check rejects non-monotone segment buffers — the packer's own output was
  always monotone, so this closes a gap only for a hand-built `PackedMask` that bypasses
  the packer (`PackedMask.validate()`, opt-in, not called on the packer/loss path).
- Planner anchors to 12,288 and a more conservative flash fit: new measured anchors at
  10,240 and 12,288 tokens (both loss implementations) showed the flash memory coefficient
  under-predicting the stock-loss arm at long context. The fit now takes an envelope over
  the worst measured arm, so `plan --attention flash` never under-predicts at any measured
  anchor, at the cost of reading more conservatively for the fused loss: up to about 1.4×
  the measured peak on the fitted model, about 1.6× cross-model. Flash plans and inverse
  answers come out more conservative than in 0.4.0. The validated range is 2,048-12,288
  tokens; past that the fit extrapolates. Reproducer: `scripts/bench_train_step.py` plus
  `scripts/fit_calibration.py` against the committed manifest.
- Planner inverse queries: `mlx-train-perf plan --max-seq` and `--max-batch` return the
  largest sequence length or batch size that fits the memory budget, instead of checking
  one config at a time. A config that cannot fit at all is refused with a typed error.

## [0.4.0] - 2026-07-19

Adds sequence packing for instruction-tuning data: many short sequences share one
fixed-length training row, kept independent by a block-diagonal attention mask inside the
flash Metal kernels. Measured on Alpaca at batch size 1 with LoRA and gradient
checkpointing, packing moves real-data throughput 2.72× on both Qwen3-8B-4bit (32.9 →
89.6 tokens/sec) and Llama-3.2-3B-4bit (71.5 → 194.4), with a conservative steady-state
reading of 2.0–2.3× once the stock arm's one-time compile traces are discounted. The win
at batch 1 is amortization, not padding: a fixed ~2-second step cost paid once per ~4,000
packed tokens instead of once per ~84.

### Added
- `packed_iterate_batches` (`mlx_train_perf.data.packing`): a drop-in for `train`'s
  `iterate_batches` that packs `(tokens, offset)` sequences into fixed rows with a seeded
  per-epoch shuffle and first-fit placement, and emits per-token segment buffers alongside
  each batch. Fixed shapes every batch, so the compiled step traces once.
- `make_packed_loss_fn` (`mlx_train_perf.adapters.mlx_lm`): the packed counterpart of
  `make_loss_fn`. Reproduces mlx-lm's loss masking segment by segment — the supervised
  token set is identical to an unpacked run, and a packed batch matches the same sequences
  run unpacked to 5.0e-4 (pin 2e-2, sized from measured RoPE offset drift). Refuses at
  construction if flash attention is not enabled.
- `flash_attention(segments=...)` and a `PACKED` variant of the forward, dQ, and dK/dV
  Metal kernels: block-diagonal causal masking via an O(N) per-token segment id, never a
  materialized mask. Single-segment output is bit-identical to the causal kernels; packed
  parity against a block-diagonal oracle holds at the same pinned tolerances as causal,
  through the 8k dispatch bucket. The identical-work packed variant costs at most 11% over
  causal at 8k; on real multi-segment rows the forward and dQ kernels skip cross-segment
  blocks and run about 3× faster than causal at the same length.
- `enable_flash_attention(packed=True)`: pre-warms the packed kernels' calibration caches
  at the training shape so the compiled first step traces with warm caches.
- `scripts/bench_packed_training.py` and `scripts/prep_alpaca.py`: the committed benchmark
  behind the numbers above — one arm per invocation, both arms on flash attention and the
  fused loss so batching strategy is the only variable, real-token accounting that never
  counts padding, and a pinned-revision dataset prep that records the exact padding-waste
  and utilization statistics of the prepared sample.

### Fixed
- `enable_flash_attention`'s documentation overstated what happens when a compiled `train`
  traces at a shape the calibration caches were not warmed for. Measured on mlx 0.32.0:
  the calibration runs once inside the trace and the run completes — a one-time stall, not
  the crash the docs promised. The pre-warm hints remain recommended; the mid-trace
  calibration's measured rate lands within 3% of an up-front one.

## [0.3.1] - 2026-07-15

Adds the Qwen2 (Qwen2.5) family to the flash-attention training path. Qwen2 already worked
with the fused loss adapter; now `enable_flash_attention` accepts it too, so a Qwen2.5
fine-tune can turn on both memory levers.

### Added
- `enable_flash_attention` supports the Qwen2 (Qwen2.5) family, alongside Llama and Qwen3.
  Qwen2's attention has the shape the wrapper already reproduces; its one difference is a bias
  on the query, key, and value projections, which the wrapper applies unchanged because it
  holds those projections directly. Forward output matches stock attention within 2e-6 in
  fp32 on both the reference and kernel paths (`tests/test_attention_wrapper.py`).

## [0.3.0] - 2026-07-14

Removes the launch-safety cap that held the flash-attention path's trainable context below
its memory limit, and adds Qwen2 to the loss adapter. Measured on an M1 Max (32 GB, macOS
26.5, mlx 0.32.0), Qwen3-8B-4bit unless noted.

### Changed
- The attention launch guard is now per command buffer, not per chain. 0.2.0 capped the
  flash path with a 2-second budget on a whole launch chain, on the theory that macOS could
  kill a chain of Metal dispatches that ran too long. Re-reading mlx 0.32.0's scheduler and
  re-testing the crash that motivated the budget showed the watchdog acts on a single
  command buffer that starves display compositing, never a chain or an eval total — and at
  training shapes each backward dispatch already runs in its own buffer
  (`scripts/probe_command_buffer_packing.py` reproduces this). The guard now models that
  composition and projects each buffer against the unchanged 0.5-second worst-day budget,
  using exact causal work counts. No safety budget was raised; what changed is what the
  budget applies to.
- Maximum trainable context on 32 GB (Qwen3-8B-4bit QLoRA, gradient checkpointing, bf16),
  measured the same day with the same search (`scripts/northstar_context_sweep.py`): 23,040
  tokens with flash attention against 7,936 with stock attention, both bound by the same
  ~24.5 GiB effective memory ceiling on this machine. Under one memory budget the flash path
  reaches 2.9x the context, because it keeps O(N) saved state instead of the O(N²) score
  matrix. In 0.2.0 this path was launch-capped near 10k tokens on the same machine; that cap
  is gone. Both ceilings scale with available memory, so the ratio is the portable figure —
  a machine with more free memory lets both climb together.
- The single flash attention op runs at 16,384 tokens instead of refusing on the retired
  launch budget (`scripts/bench_attention_op.py`). Stock attention still cannot reach that
  context on 32 GB; it aborts on the memory ceiling well before it.

### Added
- Qwen2 architecture support in the mlx-lm loss adapter (the Qwen2.5 family, tied and untied
  heads), with loss parity against the stock trainer verified on a real Qwen2.5-0.5B
  checkpoint (worst per-step difference 2.1e-3). `enable_flash_attention` still covers Llama
  and Qwen3; wrapping Qwen2 attention is future work.
- Benchmark hygiene for the community kit and the internal harness: train-step artifact
  filenames carry the attention arm, so two runs that differ only in `--attention` no longer
  overwrite each other in one output directory; a machine too crowded to start a run safely
  records its own status and re-runs on a quieter machine instead of reading as a crash; the
  quick-tier time estimate is re-anchored to a measured run.

### Fixed
- A flaky peak-memory comparison in the test suite on small shared-GPU CI runners: the
  measurement now pins the allocator state at every snapshot boundary.

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

[0.5.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.5.0
[0.4.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.4.0
[0.3.1]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.3.1
[0.3.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.3.0
[0.2.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.2.0
[0.1.0]: https://github.com/IonDen/mlx-train-perf/releases/tag/v0.1.0
