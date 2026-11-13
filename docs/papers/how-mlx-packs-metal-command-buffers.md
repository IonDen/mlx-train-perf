# How MLX Packs Metal Command Buffers

*A source-based account of operation limits, element accounting, and launch planning on an
M1 Max running macOS 26.5*

> 📄 [Read on the website](https://ineshin.space/papers/how-mlx-packs-metal-command-buffers/) — same paper, formatted for reading.

A chain of Metal kernels is not necessarily one Metal command buffer. That distinction matters when
software estimates whether a long GPU workload is safe to submit.

`mlx-train-perf` initially guarded its tiled attention backward by adding the projected duration of
every dispatch in the chain. Once the sum exceeded two seconds, the launcher refused the shape. The
guard was conservative, but it was built around the wrong scheduling unit. MLX 0.32.0 commits a command
buffer once the operations and unique tensor elements accumulated in it cross configured thresholds. At
the attention shapes that motivated the guard, a single backward dispatch already exceeds the element
threshold. Each such dispatch is therefore commit-triggering, and successive large attention
dispatches cannot share a buffer. The duration of the full chain is not the duration of one packed
buffer.

This article explains that packing rule, applies it to real attention dispatches, and separates three
kinds of evidence: facts visible in MLX source, behavioral observations reported by the project, and
an inference about why macOS terminates some long-running GPU work. Facts about the packing rule come from MLX
source at v0.32.0. The behavioral observations come from the project's own recorded probe and are
given here qualitatively; the probe's method is the committed script. The release README
and changelog are the project's own record of the design that followed.

### Evidence scope

- Project release: `mlx-train-perf` 0.3.0, documented 2026-07-14.
- Runtime: MLX 0.32.0 wheel.
- Machine: Apple M1 Max, 32 GB, reported as `applegpu_g13s`, running macOS 26.5.
- Sources: pinned MLX 0.32.0 source for the packing rule; the project's recorded probe, described
  qualitatively, for the behavioral observations; the public 0.3.0 README and changelog as the
  project's own record of the resulting design. No code or benchmark was rerun.

## 1. Why the scheduling unit matters

Memory-efficient attention avoids materializing the full score matrix by dividing forward and
backward work into tiles. A single logical backward can therefore become a chain of custom-kernel
dispatches. Each dispatch has a projected cost, and a launcher can split the work until no individual
piece exceeds a safety budget.

The first `mlx-train-perf` attention launcher also imposed a total-chain budget:

```text
reject if projected duration of one dispatch > per-dispatch budget
reject if sum of projected durations in the chain > total-chain budget
```

The second rule assumed that consecutive dispatches remained together in one command buffer, or that
macOS judged the cumulative duration of the whole evaluation. Under that model, splitting one unsafe
dispatch into several safe pieces did not help for long contexts: the sum still grew until the chain
was refused.

This created an apparent context ceiling that did not depend on RAM. Attention memory had been reduced, but the
launcher would not use the new headroom. The important question was no longer "how long is the chain?"
It was "what work does MLX actually place in each command buffer?"

## 2. The rule in MLX 0.32.0

The pinned MLX 0.32.0 Metal backend maintains two counters while encoding a command buffer:

- the number of encoded operations;
- the number of elements in unique input and output buffers.

After encoding an operation, MLX checks whether either counter has crossed the limit for the detected
Apple GPU architecture. If so, it commits the command buffer and resets the accounting for the next
one.

For the `applegpu_g13s` architecture reported by the tested M1 Max, the source-recorded configured
limits are 50 operations and 50 shifted element units. The comparison is strict: a commit is needed
when the operation count is greater than 50 or when `floor(elements / 2²⁰)` is greater than 50. The
first triggering values are therefore 51 operations or 51 × 2²⁰ = 53,477,376 elements.

The second counter is easy to misread. MLX names it with `MB`, but the source adds
`array.data_size()`, which is measured in elements rather than bytes. The "50 mega-elements" label is
only shorthand for the shifted-unit rule above; the first triggering count is not 50,000,000. The same
element count also represents different byte totals for bfloat16, float32, or another dtype. It is a
scheduling counter, not a fixed memory-capacity limit.

MLX counts a unique backing-buffer pointer once between commits. This is identity-based bookkeeping,
not semantic tensor-liveness analysis: a short-lived referenced buffer still counts, while aliased
arrays may share one backing-buffer identity. Shared backing buffers are not added again for every
operation in the same command buffer. After a commit, the unique-buffer set and both counters reset,
so buffers referenced by later dispatches are counted again.

These are source facts for MLX 0.32.0. They should not be generalized to later MLX versions or other
architecture classes without checking their corresponding source.

## 3. Applying the rule to tiled attention

The source-reported calculation used the following flagship training shape:

```text
batch                 1
query heads          32
key/value heads       8
sequence length    8192
head dimension      128
compute dtype     bfloat16
```

The source-reported arithmetic is:

| dispatch class | unique input and output elements | shifted units | packing consequence |
|---|---:|---:|---|
| backward `dK/dV` range | about 118 million | 112 | one dispatch crosses the threshold |
| backward `dQ` range | about 85 to 89 million | 81 to 85 | one dispatch crosses the threshold |
| MMA forward range, about 5,400 rows | about 73 million once the first range is encoded | 69 | one dispatch crosses the threshold |
| scalar forward range, 117 rows | 50,331,648 for `q`, `k`, `v`, counted once, plus 482,976 per range | 48 plus about 0.46 per range: 50 after six, 51 at the seventh | seven ranges pack; the commit follows the seventh |

The `dK/dV` row can be reconstructed from the tensor shapes above, assuming the distinct backing
buffers used by that measurement:

```text
q + dO                              2 × (1 × 32 × 8192 × 128) =  67,108,864
k + v + dk_in + dv_in + dk_out + dv_out
                                    6 × (1 ×  8 × 8192 × 128) =  50,331,648
lse + D                             2 × (1 × 32 × 8192)       =     524,288
total                                                               117,964,800 elements
floor(total / 2²⁰)                                                        112 units
```

Because 112 is already over the configured 50-unit limit, MLX commits after encoding that dispatch.
The next range starts a new accounting window and references the large shared buffers again.

The scalar forward row follows from the same shape. Its ranges are 117 query rows long, the range
size the project's planner reported for this shape. Every range reads the same `q`, `k` and `v`,
which count once per window, and writes its own output rows `o` and their `lse` values:

```text
q + k + v           32 × 8192 × 128 + 2 × (8 × 8192 × 128) = 50,331,648 elements = 48 units exactly
o + lse per range   32 × 117 × 128 + 32 × 117              =    482,976 elements ≈ 0.46 units
after six ranges    50,331,648 + 6 × 482,976 = 53,229,504     floor(total / 2²⁰) = 50
after seven ranges  50,331,648 + 7 × 482,976 = 53,712,480     floor(total / 2²⁰) = 51
```

Under the strict rule six ranges leave the counter at 50, and the seventh takes it to 51: seven ranges
pack into one buffer, and the commit follows the seventh. Buffers already counted in the window, such
as framework work encoded before the first range, move the trigger earlier, by about two ranges per
extra unit (one unit is 1,048,576 elements, a little over two ranges of 482,976).

The large backward ranges are the decisive case. Their referenced unique backing buffers already
exceed the element threshold after one encoded operation. Once MLX checks the counters, it commits. Successive
large ranges therefore cannot share a command buffer, although a large range may still follow smaller
framework operations in the buffer it closes. No published Metal capture directly observed those
surrounding boundaries.

Small-footprint work behaves differently. Several scalar forward ranges can stay under both limits
and pack together. A safe planner cannot simply delete all cumulative accounting. It must accumulate
projected work within each modeled command buffer, commit the model when either MLX threshold is
crossed, and then begin a fresh buffer model.

![How the MLX 0.32.0 commit rule packs two dispatch classes at the flagship shape: each large dK/dV backward range triggers a commit after encoding and may follow smaller prefix work, while scalar forward ranges add about 0.46 units each to the 48 units of q, k and v, so seven pack per buffer and the commit follows the seventh](https://raw.githubusercontent.com/IonDen/mlx-train-perf/main/docs/papers/diagrams/command-buffer-packing-sequence.svg)

*Figure 1. Packing behavior implied by the MLX 0.32.0 commit rule at the flagship attention shape.
The structure is the source-derived accounting model and the element counts are source-reported
arithmetic; the figure is not a Metal command-buffer capture. A large range may close a buffer that
already contains smaller framework work. The editable
[PlantUML source](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/diagrams/command-buffer-packing-sequence.puml)
is published with the paper.*

## 4. What the behavioral probes add

Source inspection tells us what MLX intends to count. On 2026-07-14, the project also recorded three
subprocess-per-condition behavioral checks against the shipped 0.32.0 wheel on the M1 Max running
macOS 26.5. These are qualitative consistency checks, not measurements of the exact threshold or an
isolated overhead component. The probe's timings were not published with the release, so this section
reports only what each probe showed. The method is the committed
[probe script](https://github.com/IonDen/mlx-train-perf/blob/v0.3.0/scripts/probe_command_buffer_packing.py).

First, a chain of 2,000 small serial matrix multiplications ran clearly slower when a commit was
forced after every operation than under the default configuration. With the operation threshold
raised to 5,000, which allows fewer commits, the chain ran slower than the default in the recorded
run, against the script's own stated expectation that it would be no slower. The forced-every-operation
contrast is consistent with aggregate overhead from more frequent commits, but the third condition
does not isolate that overhead or establish the exact threshold. The threshold semantics come from
source.

Second, 200 serial additions over roughly 13 million elements ran under the default configuration, a
very high element threshold, and a commit forced after every operation. The direction of the differences
matched the script's stated expectation, within a small spread. With one process per condition, that spread does
not by itself separate the three settings.

Third, a real chained `dK/dV` backward at 12,288 tokens used twelve 1,024-row ranges. The probe script
projects the full chain at roughly 4.3 seconds, beyond the retired total-chain cap. The chain
completed and returned finite outputs. The element arithmetic predicts that every large range
triggers a commit at that shape. One completion shows that such a chain can run; it does not locate
the watchdog's limit, and the probe script treats a survival as evidence about that day's machine
state only.

These probes support the practical consequence of the source rule. They do not directly expose Metal
command-buffer boundaries, nor do they reveal the operating system's internal watchdog decision.

## 5. From a chain budget to a packing-aware planner

The `mlx-train-perf` 0.3.0 planner made two orthogonal corrections. First, it replaced a rectangular
`N × N` work estimate with the causal triangular MAC count for the actual query range. The retired
estimate priced masked work that the kernel never executes and could be roughly twofold pessimistic.
Second, it accumulated those corrected dispatch projections over modeled command-buffer windows rather
than over the whole logical chain.

For every planned dispatch, it estimates:

1. exact causal work for the query range and its projected duration;
2. elements in guaranteed-distinct backing buffers referenced by the current modeled buffer;
3. output buffers only when the caller retains them, which guarantees that they cannot be recycled
   into the same identity during the modeled window;
4. the current buffer's operation count.

MLX itself does not count "live tensors"; it counts backing-buffer identities encountered since the
last commit. The planner uses caller-held references as a conservative way to identify buffers that
are guaranteed to remain distinct. It does not credit short-lived intermediate outputs when allocator
recycling could make their identities uncertain, because falsely predicting an early commit would be
the unsafe direction.

If the next dispatch would make the modeled buffer exceed its time allowance, the planner shortens the
range. When the modeled operation or element count crosses MLX's commit threshold, the next dispatch
starts with fresh buffer counters and a fresh time budget.

The planner therefore no longer rejects work solely because total chain duration exceeds two seconds.
Every modeled buffer must still stay within the conservative allowance, and memory, thermal state,
end-to-end latency, or other runtime limits can still bound the chain. Small dispatches that can pack
together share one allowance; the planner cannot pretend that every custom-kernel call gets its own
buffer.

The change did not raise the project's safety margin. In the 0.3.0 code the measured kernel rate is
multiplied by the same 0.5 calibration safety factor
([`SAFETY_FACTOR = 0.5`](https://github.com/IonDen/mlx-train-perf/blob/v0.3.0/src/mlx_train_perf/attention/kernel/launch.py#L103)),
making projected duration about twice the estimate at the raw measured rate. Each modeled buffer is
then capped at the same 0.5-second projected duration. What changed was the causal work estimate and the window over
which those projections accumulate.

## 6. What is known, and what remains an inference

The evidence supports several claims with different confidence levels.

**Directly established from pinned source:** MLX 0.32.0 checks operation and unique-element counters;
the M1 Max architecture class uses 50/50 thresholds; inputs and outputs contribute elements; commits
reset the accounting.

**Supported by source arithmetic and behavioral observations:** large attention-backward dispatches
cross the element threshold individually, while several smaller forward dispatches may pack; the
forced-boundary timings are qualitatively consistent with aggregate commit overhead; a long chain of
commit-triggering dispatches can execute despite exceeding the retired chain-total budget.

**Inferred rather than directly observed:** macOS interactivity kills are governed by the
non-preemptible occupancy of an individual command buffer, with display contention affecting the
boundary. This model fits an external MLX issue report, but the probes do not instrument the watchdog
itself. It should remain a mechanism hypothesis, not be presented as an Apple-documented law.

## 7. Limits and transferability

The packing constants and dispatch arithmetic are pinned to MLX 0.32.0 and an M1 Max reported as
`applegpu_g13s`. MLX may change its counters, thresholds, architecture mapping, or commit behavior.
Donation, aliasing, and framework operations inserted between custom kernels may also affect which
buffers are counted. Extra intervening work tends to cause earlier commits, but that conservative
direction should still be checked when adapting the planner.

The behavioral probes are single recorded runs on one machine, given here by direction only, not a
statistical performance analysis. One condition of the first probe ran against the script's
expectation. They do not define universal fence costs or watchdog thresholds.

The durable lesson is methodological. For this MLX/macOS configuration, modeled command-buffer windows
are a better operational proxy than whole-chain duration. Pinned runtime source should establish how
work is submitted before cumulative timing rules become product limits; whether that submission unit
also governs an operating-system kill remains a separate empirical question.

## References and source notes

- MLX 0.32.0 `device.cpp`: [backing-buffer identity and size accounting](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L343-L368),
  [operation counting](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L405-L418),
  [the `needs_commit()` rule](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L512-L515),
  and [the architecture-specific thresholds](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L585-L625).
  Section 3 shows the attention-shape arithmetic derived from these rules; the probe observations are
  source-reported, with the public method linked below.
- [`mlx-train-perf` 0.3.0 changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.3.0/CHANGELOG.md)
  and [release README](https://github.com/IonDen/mlx-train-perf/blob/v0.3.0/README.md), the
  project's own release record of the resulting design.
- [`scripts/probe_command_buffer_packing.py`](https://github.com/IonDen/mlx-train-perf/blob/v0.3.0/scripts/probe_command_buffer_packing.py),
  the recorded probe method; this article did not rerun it.
- [ml-explore/mlx issue #3267](https://github.com/ml-explore/mlx/issues/3267), an external report of
  LoRA fine-tuning kills at roughly 1.2-second individual operations with the display active; used
  only for the watchdog inference in section 6.
- [Apple `MTLCommandBufferError`](https://developer.apple.com/documentation/metal/mtlcommandbuffererror),
  documentation for Metal command-buffer error classes; it does not establish this article's inferred
  watchdog boundary.

---

*Prepared 2026-07-15. Last updated 2026-11-13. Denis Ineshin.*
