# A fused linear cross-entropy forward kernel for Apple GPUs

*Memory savings, tiling limits, and lessons from failed performance models*

Training a language model often allocates a huge logits matrix just to compute cross-entropy. For `N`
tokens and a vocabulary of size `V`, that matrix has shape `(N, V)`. Yet the loss needs only a
log-sum-exp and one target logit from each row. We built a fused MLX/Metal forward kernel that computes
those values in tiles and never stores the full matrix. At `N=8192`, `V=151936`, and hidden dimension
`D=4096`, the fused forward took 2.105 s versus 1.286 s for materialized MLX. A public memory run also
recorded a much lower peak for the fused path. Unequal warm-up baselines, however, prevent a sound
per-call memory ratio; Section 3 reports the full measurements and explains the limit.

The optimization process produced two less obvious results. First, forcing `mx.eval` after each
pure-MLX chunk increased memory because MLX retained traced intermediates. Second, a load-reuse model
predicted one faster register tile, then failed on the next. Larger tiles became slower. Follow-up
controls ruled out several explanations but could not isolate one cause. The remaining evidence points
to a tradeoff between source-level load reuse and compiled resource use, neither of which we measured
directly at the hardware level.

This paper documents the implementation and those measurements. It does not introduce a new
cross-entropy algorithm. Logit-free linear cross-entropy and streaming softmax reductions are prior
work, including [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009). The optimization ladder is also
a retrospective account, not a reproducible benchmark suite: its historical source revisions and full
launch settings were not frozen. The forward result does not imply the same reduction for a complete
training step.

## 1. Streaming the loss without full logits

Let `H` be an `(N, D)` matrix of hidden states and `W` a `(V, D)` output projection. Ordinary
cross-entropy first computes all logits

```text
Z = H Wᵀ
```

and then, for target `y_i`, computes

```text
loss_i = logsumexp(Z_i) - Z_i,y_i
```

The loss does not require random access to the full row of logits. It requires only two row-wise
statistics: the log-sum-exp and the target logit. This permits a streaming formulation.

The kernel merges each vocabulary tile into a running online log-sum-exp. Let `(m, s)` hold the
current maximum and the exponential sum scaled by that maximum. For a new tile with maximum `m_t`,
the update is

```text
m' = max(m, m_t)
s' = s · exp(m - m') + Σ_j exp(z_j - m')
```

After the last tile, with `(m, s)` as the final running state,
`logsumexp = m + log(s)`. The kernel captures the target logit when its vocabulary index falls inside
the current tile. Only the running state and final per-token values survive. The `(N, V)` matrix never
exists.

In the implementation studied here, Python constructs one lazy Metal dispatch per vocabulary tile.
The kernel is JIT-compiled once and the cached pipeline is reused across the chain. With `V=151936`
and an 8192-column tile, the production shape forms 19 dependent dispatches. Each invocation consumes
the previous invocation's `N`-element accumulators and produces the next ones. A single evaluation
executes and synchronizes the completed chain.

The scope is forward only. The shipping training path saves the forward log-sum-exp and uses a
separate pure-MLX chunked backward that regenerates the required logits tile by tile. It does not
yet use a fused Metal backward, so the isolated measurements in this paper must not be read as
forward-plus-backward results.

## 2. Why pure-MLX chunking still fell short

The first prototype implemented the same vocabulary decomposition with ordinary MLX operations.
It was numerically correct, supported dense and quantized output heads, and demonstrated that a
large loss could be processed without forming one monolithic logits tensor.

It also exposed a counterintuitive evaluation rule. Calling `mx.eval` after every chunk while
MLX traced the gradient retained the traced intermediates instead of releasing each chunk. In an
early forward-plus-backward experiment at `N=8192`, forced per-chunk evaluation reached 55.98 GiB,
versus 9.13 GiB when the chunks remained lazy and the graph was evaluated once at the end. The
eager version used about six times more memory.

Leaving the chunks lazy reduced forward-plus-backward peak memory by roughly two times at the measured
shapes. At `N=4096`, however, it cost 1.31 times the dense path for the dense head and 1.71 times for
the quantized head. These exploratory results motivated kernel fusion. They are not final library
benchmarks and are not directly comparable to the forward-only numbers below.

Chunking expressed the right mathematical decomposition, but it could not control the lifetime and
fusion of every intermediate. A custom Metal kernel could keep tile-local logits and reductions in
registers, making the intended memory bound part of the implementation rather than a scheduling hope.

![Conceptual dataflow comparing materialized MLX, pure-MLX chunking, and the fused Metal forward](https://raw.githubusercontent.com/IonDen/mlx-train-perf/main/docs/papers/diagrams/fused-linear-ce-dataflow.svg)

*Figure 1. The three paths compute the same per-token loss but expose different intermediate state.
This is a conceptual dataflow, not a measured allocation-lifetime trace. The editable
[PlantUML source](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/diagrams/fused-linear-ce-dataflow.puml)
is published with the paper.*

## 3. Experimental setup

All reported measurements came from one Apple M1 Max with 32 GB of unified memory. The exploratory
kernel ladder used MLX 0.31.2 and macOS 26.5.1. The committed loss-layer reproduction used MLX 0.32.0
and macOS 26.5.2. Both used bfloat16 inputs with `V=151936` and `D=4096`.

Each timed condition ran in a fresh process. Before timing, it initialized and evaluated its inputs,
then warmed the JIT compiler and allocator. The table reports the median of three synchronized runs.

Before benchmarking, each kernel variant had to match an fp32 forward reference. The exploratory
suites covered 10 to 19 cases per variant, including tail and alignment cases. The worst saved
absolute per-token loss differences were `4.8e-6` for fp32 inputs and `1.9e-6` for bfloat16 inputs.
These checks cover the forward only, not gradient correctness.

The exploratory parity artifacts are not public. The final library does publish a forward benchmark
with the MLX and OS versions, package/source identity, shape, repetition count, and raw wall times.
That artifact predates the driver's `script_sha` field, so it cannot identify the exact script
revision that produced the rows.

The committed MLX 0.32.0 run produced the following isolated forward result:

| implementation | active before reset | marginal after reset | warmed total peak | median wall | throughput |
|---|---:|---:|---:|---:|---:|
| Materialized MLX logits | 5.8585 GiB | 2.3184 GiB | 8.1768 GiB | 1.285595 s | 3965.577 G MAC/s |
| Fused Metal kernel | 1.2218 GiB | 0.0006 GiB | 1.2224 GiB | 2.104592 s | 2422.382 G MAC/s |

The same-session timing gives a 1.64× forward slowdown. The memory columns need more care. The benchmark
warms each implementation, keeps the warm-up loss alive, clears the allocator cache, and resets the
peak counter. The materialized path therefore starts its measured window with 4.6367 GiB more active
memory than the fused path.

That baseline gap dominates a ratio between the two marginal columns, while the 0.0006 GiB value is
also rounded to four decimal places. The table shows different memory behavior, but it does not provide
a clean per-call reduction ratio. Such a ratio would require raw byte counts and a fresh run from a
common input-only baseline with no live warm-up graph.

## 4. The load-reuse model works once

The initial Metal design assigned one SIMD-group to several rows. Each lane traversed the hidden
dimension, accumulated partial dot products in fp32, and used `simd_sum` to combine the lanes. The
main experimental variable was the register tile: how many rows (`R`) and vocabulary columns (`C`)
one SIMD-group processed together.

An `R × C` tile using four-element bfloat16 vector loads performs `4RC` multiply-accumulates while
requesting `(R+C)` vectors from the source-level inner loop. We defined the nominal load-reuse ratio

```text
load reuse = 4RC / (8(R+C)) MAC per requested byte.
```

This ratio suggested that moving from a `4 × 1` tile to `4 × 4` might raise throughput by about
2.5×. At `N=8192`, measured throughput rose from 294.8 to 814.9 G MAC/s, a 2.76× gain. The direction
and scale fit this first pair. The ratio is not hardware arithmetic intensity: it ignores stores,
online-reduction work, cache-level traffic, accumulator traffic, and the instruction mix. It was
nevertheless tempting to treat the first agreement as a predictive model.

The next tile was slower. Moving from `4 × 4` to `4 × 8` increased nominal load reuse by 33%, yet
reduced throughput at both measured sequence lengths.

The table uses the historical labels for each experimental rung. It does not provide a public,
immutable mapping from every label to its source revision and full launch settings. In particular,
the vocabulary-tile and dispatch geometry are not preserved for each row. The table supports a
retrospective comparison, not an independent rerun of the ladder.

| variant | register tile | nominal load reuse | throughput at `N=8192` | compiled thread ceiling |
|---|---:|---:|---:|---:|
| v1c | `4 × 1` | 0.40 MAC/B | 294.8 G MAC/s | not recorded |
| v1d | `4 × 4` | 1.00 MAC/B | 814.9 G MAC/s | 512 |
| v1f | `4 × 8` | 1.33 MAC/B | 752.3 G MAC/s | 384 |
| v1e | `8 × 8` | 2.00 MAC/B | not safely run at this shape | 384 |

The `4 × 8` tile had 33% more nominal load reuse than `4 × 4`, yet was 8% slower at `N=8192`
and about 20% slower at `N=2048`. The `8 × 8` version collapsed even more severely: in a
same-shape, same-4096-column-tile comparison at `N=2048`, it managed 182.1 G MAC/s against 635.0
for `4 × 8`.

## 5. Four controls narrow the regression

Several explanations looked plausible. The row stride was exactly 8192 bytes. That equals the M1 GPU
L1 data-cache size reported by the reverse-engineered
[metal-benchmarks architecture notes](https://github.com/philipturner/metal-benchmarks), so additional
concurrent streams might have caused cache-set conflicts. Some variants originally used an inline
copy of the benchmark harness. The `8 × 8` experiment also changed its tile shape and source-code
style. Its first small-shape comparison did not give both variants the same opportunity to occupy the
GPU.

Four controls narrowed the field:

1. Changing `D` from 4096 to 4160 changed the row stride from 8192 to 8320 bytes. The `4 × 8`/`4 × 4`
   throughput ratio remained 0.80 at `N=2048`. This rejected the proposed power-of-two cache-set
   mechanism for the `4 × 8` regression.
2. Running both variants back to back through the shared script reproduced the earlier rates within
   1%, eliminating the copied harness as the cause.
3. Rewriting the `4 × 4` kernel with the array-and-unroll style used by the larger tiles reproduced
   the explicit-scalar implementation. The coding idiom was not responsible.
4. Comparing `8 × 8` and `4 × 8` at the same `N=2048` and vocabulary-tile size left a 3.5× residual
   slowdown. This removed the launch-count difference and reduced the small-shape saturation problem,
   but the variants still had different row-block geometry.

We also compiled the generated Metal source through the Metal framework and inspected
`MTLComputePipelineState.maxTotalThreadsPerThreadgroup`. The device maximum was 1024 threads. The
`4 × 4` kernel compiled with a ceiling of 512, while `4 × 8` and `8 × 8` both compiled at 384.
This pipeline-legality limit correlates with heavier compiled resource use. It does not measure
achieved occupancy, resident SIMD groups, or registers. The experiment also did not show that the
lower ceiling changed residency for the actual dispatch.

The controls establish real regressions and correlate them with heavier compiled resource use. They
do not identify a cause. Reduced occupancy is one candidate for `4 × 8`. For `8 × 8`, the estimated
live state exceeded the reverse-engineered 128-by-32-bit GPR model in
[Dougall Johnson's Apple GPU notes](https://dougallj.github.io/applegpu/docs.html). A large slowdown
remained after the matched controls. Register spill is a candidate, but no compiler ISA, spill statistic,
memory-traffic counter, or achieved-occupancy measurement was captured. Neither mechanism is causally
established here.

## 6. Matrix tiles reduce scalar state

The next design used `simdgroup_matrix` operations to increase reuse without growing ordinary scalar
state in the same way. Its early rungs were slower than the best register-array kernel. Larger matrix
tiles eventually closed that gap.

| rung | design | throughput at `N=8192` | slowdown vs materialized MLX path |
|---|---|---:|---:|
| v2a | one `8 × 8` matrix tile | 487.2 G MAC/s | 8.1× |
| v2c | `2 × 2` tiles (`16 × 16`) | 1233.9 G MAC/s | 3.2× |
| v2d | `2 × 4` tiles (`16 × 32`) | 1579.3 G MAC/s | 2.5× |
| v2e | `4 × 4` tiles (`32 × 32`) | 2423.7 G MAC/s | 1.63× |
| v2f | `4 × 8` tiles (`32 × 64`) | 1403.2 G MAC/s | 2.8× |

Performance improved up to the `32 × 32` tile, which used roughly 32 fp32 accumulator elements per
lane. Doubling one tile dimension then cut throughput from 2423.7 to 1403.2 G MAC/s. Across both design
families, more data reuse helped only while the extra per-lane state did not make the compiled kernel
materially heavier.

A compact model for the observed behavior is

```text
throughput may be constrained by both data delivery and compiled resource use.
```

This is a qualitative design heuristic, not a fitted model or a universal Apple GPU law. The experiment
did not measure memory traffic, bandwidth ceilings, register counts, spills, or achieved occupancy, so
it cannot distinguish bandwidth, load-instruction throughput, cache behavior, and latency hiding. Its
conclusion is narrower: better nominal load reuse can still lose when it makes the kernel heavier in
another unmeasured way.

## 7. Forward savings do not predict full-step savings

The fused kernel removes the logits allocation from the forward loss layer by construction. The public
memory run is consistent with a large reduction, but its unequal warmed baselines do not support a clean
ratio. In a complete training step, the transformer, optimizer, model weights, and attention backward
remain. Removing one allocation does not guarantee a proportional increase in maximum context.

The project's 0.1.0 release measurements show why scope matters. With stock attention held across the
loss comparison, the release record reports that fused cross-entropy alone did not materially extend
maximum trainable context. It attributes the long-context peak to stock attention backward. Because
the raw historical sweep is not public, this remains a source-reported result rather than an
independently reconstructable comparison.

The loss kernel therefore removes loss-layer waste without solving training memory as a whole. It
does not automatically increase maximum context.

Speed also changes with scope. The isolated fused forward was 1.64× slower than the highly tuned
materialized MLX path, but the loss layer is only part of a training step. Later end-to-end measurements
found a much smaller step-level penalty. That penalty depends on the model, sequence length, attention
implementation, and backward path. It cannot be inferred from the isolated table alone.

The public
[`bench_train_step.py`](https://github.com/IonDen/mlx-train-perf/blob/v0.3.1/scripts/bench_train_step.py)
documents the later step-level method. The loss-only context conclusion appears under the 0.1.0 known
limits in the [changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.3.1/CHANGELOG.md). The current
[`northstar_context_sweep.py`](https://github.com/IonDen/mlx-train-perf/blob/v0.3.1/scripts/northstar_context_sweep.py)
changes both attention and loss. It therefore documents the later whole-product comparison, not this
loss-only result.

## 8. Lessons from the failed models

For future MLX kernel work:

- Control evaluation boundaries explicitly. In a lazy differentiable system, forcing evaluation inside
  a loop can extend lifetimes rather than shorten them.
- Do not trust a performance model after one successful prediction. The nominal load-reuse ratio fit
  the `4 × 4` pair and failed immediately afterward.
- Compare matched shapes, tile sizes, and opportunities to occupy the GPU. The first `8 × 8` comparison
  mixed a possible resource-pressure effect with idle cores and different launch counts.
- Treat pipeline limits as proxies. A reduced `maxTotalThreadsPerThreadgroup` correlates with a heavier
  compiled kernel, but it does not reveal a register count, residency level, or spill event.
- Keep failed predictions in the record. The stride, harness, and source-idiom hypotheses were
  reasonable, and their rejection explains why the final design changed.
- Inspect the baseline behind a memory delta. Marginal peaks are comparable only when persistent state
  at the reset boundary is also comparable.

## 9. Limitations

This study used one M1 Max, one principal production shape, bfloat16 inputs, and pinned MLX versions.
Absolute rates can change with the machine, OS, MLX release, thermal state, and JIT compiler. Three-run
medians capture large design differences, not fine-grained variance.

The experiments did not confirm the occupancy and register-spill explanations with ISA or counter
evidence. The public memory protocol also uses unequal warm-up baselines, so it cannot support a clean
per-call memory ratio. This article reports that limitation rather than rerunning the experiment. It
also covers only the forward kernel. The shipping backward path and quantized-head variants are outside
its scope.

## 10. Reproducibility and sources

The final loss-layer benchmark is produced by
[`bench_loss_layer.py`](https://github.com/IonDen/mlx-train-perf/blob/v0.3.1/scripts/bench_loss_layer.py).
The [committed JSON artifact](https://github.com/IonDen/mlx-train-perf/blob/v0.3.1/community-benchmarks/apple-m1-max-32gb-2026-07-13.json)
records its condition identities and all three wall-time samples. The exploratory ladder has no equivalent
public bundle, so its results remain source-reported and cannot be rerun independently. The public JSON
identifies the measured package source, but it predates the driver's `script_sha` field. The exact
historical driver revision is therefore unknown.

[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) develops the broader logit-free linear
cross-entropy approach. [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) provides fused training
kernels for other accelerator stacks.

## 11. Conclusion

The fused forward removes the `(N, V)` logits matrix, but it trades memory for time. On the measured
M1 Max shape, it ran 1.64× slower than materialized MLX. The public memory result points to a large
reduction, but its unequal warm-up baselines prevent a reliable per-call ratio.

The optimization ladder matters for a different reason. A load-reuse model correctly predicted the
`4 × 4` improvement and failed at `4 × 8`; the `32 × 32` matrix tile later recovered the best observed
throughput before a larger tile regressed again. Without register or occupancy counters, the cause
remains open. The supported result is narrower and more useful: data reuse helps only while the
compiled resource cost stays under control, and source-level reuse alone cannot predict that boundary.

---

*Prepared 2026-07-14. Last updated 2026-07-19. Denis Ineshin.*
