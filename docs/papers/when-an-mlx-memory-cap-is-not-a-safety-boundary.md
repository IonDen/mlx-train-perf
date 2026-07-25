# When an MLX Memory Cap Is Not a Safety Boundary

*An Apple unified-memory incident report and the process-local watchdog added afterward*

> 📄 [Read on the website](https://ineshin.space/papers/when-an-mlx-memory-cap-is-not-a-safety-boundary/) — same paper, formatted for reading.

A configured memory limit can be real without being a hard stop. That distinction matters on Apple
Silicon, where CPU and GPU workloads share physical memory and some GPU allocations remain pageable.

During an `mlx-train-perf` benchmark campaign, a long-context training condition ran for roughly three
hours and ended in a macOS kernel panic inside `IOGPUMemory`. The project reconstructed the worker as
operating beyond the physical memory of a 32 GB M1 Max: its wired limit remained in force, while the
softer memory limit was advisory and allowed pageable allocation to continue toward RAM plus swap.
The reboot erased the worker's temporary campaign log, so its paging state is an evidence-backed
reconstruction rather than a recovered final memory measurement.

The incident disproved a standing project assumption: a wired limit alone does not turn every GPU
over-allocation into a clean MLX exception. The mitigation was a third layer: a best-effort,
process-local daemon intended to stop sustained active-memory growth before it reaches the observed
storm regime. It is not a hard machine-safety boundary and can miss fast allocations, below-ceiling
thrashing, or pressure created after startup by other processes.

The reconstruction uses dated project records and released documentation; it does not rerun the
workload. The measurements are source-reported and cross-checked against the project README and
changelog.

![The three memory controls on Apple Silicon and where each stops an over-budget GPU allocation: the wired limit caps residency without rejecting the allocation, the soft memory limit is advisory and lets allocation continue into swap, and excess buffers stay pageable, after which the workload either kernel-panics with no process-local stop or is aborted cleanly by the added active-memory watchdog](https://raw.githubusercontent.com/IonDen/mlx-train-perf/main/docs/papers/diagrams/memory-cap-safety-layers.svg)

*Figure 1. How the three controls interact at an over-budget allocation. The wired and soft limits
bound residency and advise the allocator, but neither rejects the allocation; the added watchdog is
the layer that turns a would-be paging storm into an explicit abort. The structure is conceptual and
the kernel-panic trigger is an unverified hypothesis, not a measured causal chain. The editable
[PlantUML source](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/diagrams/memory-cap-safety-layers.puml)
is published with the paper.*

## 1. The safety model before the incident

The original model treated a device-clamped wired limit as if it rejected further allocation. MLX
0.32.0 implements a narrower boundary:

```text
over-budget allocation
        ↓
wired residency set, up to the configured limit
        ↓
additional buffers remain unwired and pageable
        ↓
the OS retains reclaimable headroom, but allocation can continue
```

The wired limit controls how much GPU memory MLX asks the system to keep resident. It can bound the
non-pageable set and preserve reclaimable headroom, but reaching it does not itself reject the next
allocation. Buffers beyond the residency capacity can remain unwired and pageable.

The gap was everything outside that class. The project also used `mx.set_memory_limit`, whose MLX
0.32.0 documentation described its value as a guideline. Exceeding it did not necessarily stop graph
evaluation. Allocation could continue until RAM and available swap were exhausted.

The project had therefore combined two different controls under the word "cap":

| control | memory class | documented behavior |
|---|---|---|
| `mx.set_wired_limit` | resident, non-pageable GPU memory | residency cap; excess buffers may remain unwired and pageable |
| `mx.set_memory_limit` | broader allocator behavior, including pageable memory | advisory guideline; allocation may continue into swap |

The wired limit was not broken. The mistaken assumption was that reaching it would reject allocation
rather than limit residency.

## 2. What happened

The incident occurred during an unattended benchmark campaign on 2026-07-10. The environment was an
M1 Max with 32 GB unified memory, macOS 26.5.1, MLX 0.32.0, `mlx-lm` 0.31.3, and Python 3.13.

The failing worker was running a compiled `mlx-lm` LoRA training step for Qwen3-8B at 8,192 tokens,
with stock attention and stock cross-entropy. This path materialized the full logits tensor with shape
`(8192, 151936)` in addition to the long-context attention workload.

The documentary timeline is:

| condition | reported outcome |
|---|---|
| 2,048-token conditions | completed |
| 8,192 tokens, fused cross-entropy arm | completed at 25.68 GiB total peak |
| 8,192 tokens, stock attention and stock cross-entropy | allocated past physical RAM, paged for roughly three hours, then kernel-panicked |
| later flash-attention condition | never started because the reboot ended the campaign |

The panic log recorded `"completeMemory() prepare count underflow"` at
`IOGPUMemory.cpp:550`. Process identifiers and local log names are intentionally omitted here; they do
not contribute to the technical result.

## 3. The measured chain, and the boundary of the claim

The incident reconstruction rests on four observations from the project records.

The installed MLX documentation explicitly called `mx.set_memory_limit` a guideline and said
that allocation errors occur when no more RAM, including swap, remains. Crossing the configured soft
value was therefore not specified to fail immediately.

The MLX 0.32.0 documentation excerpt recorded with the campaign read:

> "The memory limit is a guideline ... If the memory limit is exceeded and there is no more RAM
> (including swap when available) allocations will result in an exception."

A same-campaign, single-operation attention benchmark had already reported a 32.4 GB marginal
allocator peak on the nominally 32 GB machine. It completed with `status: ok`, but wall time rose from
0.45 seconds at 4,096 tokens to 18.3 seconds at 8,192, about 41×. The project interpreted that slowdown
as paging. The source documents label this value "GB" without resolving decimal versus binary units;
it is a marginal allocator metric and is not directly comparable with the 25.68 GiB total peak below.
It nevertheless showed that crossing the configured MLX soft value did not force a clean exception.

The failing training condition added the stock `(tokens, vocabulary)` logits allocation to a workload
whose fused-loss sibling had peaked at 25.68 GiB total. The project record inferred, rather than
measured from the failed worker, that the stock condition ran roughly 6 GB above that sibling.

The 25.68 GiB sibling completed normally. The machine could run legitimate work close to its
practical ceiling. Severe pageable GPU pressure was the condition associated with the incident, but
the evidence does not isolate it as the cause of the IOGPU assertion.

These observations support the project's incident reconstruction:

```text
wired limit remains active
        ↓
pageable allocation crosses the advisory soft limit
        ↓
allocation continues into system paging
        ↓
sustained GPU memory pressure (reconstructed)
        ↓
kernel panic at the recorded IOGPU assertion
        causal trigger unknown
```

They do not establish the internal driver mechanism. Apple's IOGPU implementation is closed. The
project hypothesized that repeated page-out and fault-in activity disturbed prepare/complete
bookkeeping for GPU-mapped memory, but it could not inspect or prove that explanation. The identical
panic also appears in public reports with much shorter single-allocation triggers, so duration-specific
paging is not required by the available evidence.

The evidence is classified as follows:

| claim | status | documentary source |
|---|---|---|
| three-hour run and recorded panic assertion | directly observed incident record | dated project record |
| 32.4 GB marginal single-op peak and 41× slowdown | separate source-reported measurement | dated project record; release README |
| failed worker operated in the same paging regime | project reconstruction | sibling peak, expected logits, and duration recorded by the project |
| prepare/complete bookkeeping mechanism | unverified hypothesis | project reconstruction and comparison with public reports |
| released watchdog behavior and specific abort paths | source-reported operational behavior | release README and changelog |

## 4. Independent reports narrow the conclusion

The project records link several public reports containing the same assertion and source location:

- [ml-explore/mlx#3346](https://github.com/ml-explore/mlx/issues/3346): an M3 Ultra under high GPU
  memory use across multiple processes.
- [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883): an M3 Ultra with a growing
  KV cache, high wired residency, and almost no free memory.
- [ml-explore/mlx#3186](https://github.com/ml-explore/mlx/issues/3186): an M4 Max with a single very
  large prefill, without a multi-hour paging storm.
- [jundot/omlx#557](https://github.com/jundot/omlx/issues/557): the same M1 Max memory class under
  concurrent inference load.
- [exo-explore/exo#1972](https://github.com/exo-explore/exo/issues/1972): an M4 Max under distributed
  inference load.

The shared assertion across different applications, hardware generations, memory sizes, and workload
durations supports treating this as a broader GPU-memory-pressure failure class. It does not prove
that every report has an identical trigger, that the defect belongs solely to Apple rather than the
MLX/driver interaction, or that sustained paging is necessary. One MLX issue records that maintainers
filed Apple Feedback while stating that the bug might come from either side.

The public evidence therefore changes the wording from "the three-hour paging storm caused a specific
driver bookkeeping bug" to a narrower statement: the local workload entered severe pageable GPU
pressure and ended at the same IOGPU assertion reported by other memory-intensive Apple GPU workloads.

## 5. The mitigation: an active-memory watchdog

The project retained both existing MLX limits and added a separate process-local guard. The resulting
model has three layers:

1. a residency cap for non-pageable GPU memory;
2. the MLX soft memory guideline;
3. a best-effort active-memory watchdog for sustained process-local growth.

The watchdog samples `mx.get_active_memory()` roughly twice per second. Its effective ceiling is the
smaller of a device-relative static value and startup effective availability minus a 2 GiB cushion:

```text
effective ceiling = min(
    static ceiling for the machine's memory class,
    available memory at start - 2 GiB
)
```

For the 32 GB reference machine, the static value is 28.0 GiB. It sits above the project's largest
legitimate measured peak, 25.68 GiB, and below physical memory. The dynamic term matters because the
incident was state-dependent: a schedule that fits after a clean boot may be unsafe when other
processes already occupy unified memory. Here, availability is a one-shot `vm_stat` sum of free,
inactive, and speculative pages, including reclaimable inactive file cache. In the reported release
campaign, this dynamic term set an effective ceiling near 24.5 GiB rather than the 28 GiB static
anchor. Availability is sampled when the watchdog is installed; the resulting ceiling is fixed for
that worker and does not adapt if another process grows later.

When the worker reaches the ceiling, the watchdog attempts to write an `aborted_memory_ceiling` result
and then terminates the process with a hard exit. A wall-clock backstop similarly attempts to record
`aborted_wall_budget`. The harness refuses up front when the computed effective ceiling
`min(static_ceiling, available_at_start - 2 GiB)` is less than one quarter of physical RAM;
intermediate crowding produces a visible warning and a lower fixed ceiling.

The hard-exit design is deliberate. Artifact writing can itself fail under memory pressure, so
termination remains unconditional even if recording the abort raises an exception. In that case the
abort artifact may be absent. A safety callback must not be able to disarm the exit path it is meant
to trigger.

## 6. What the project documents report after the fix

The dated project records report the following post-mitigation campaign observations:

- failing long-context stock conditions subsequently produced `aborted_memory_ceiling` records rather
  than continuing into unbounded paging;
- a crowded-machine campaign lowered its effective ceiling from measured availability and aborted one
  condition cleanly while other conditions completed;
- multiple watchdog aborts and zero further kernel panics were reported across the documented
  post-mitigation campaign.

The release README and changelog independently corroborate the guard design and specific
`aborted_memory_ceiling` paths. They do not independently report every crowded-machine and
campaign-wide observation in the list above.

These are source-reported operational observations, not proof that the watchdog prevents every form of
the IOGPU assertion. They validate the path the mitigation was designed for: sustained growth in MLX
active memory that crosses the configured ceiling slowly enough for the daemon to observe it.

## 7. What remains uncovered

Several residual risks remain.

**Fast allocation.** A single allocation can cross an unsafe region between the watchdog's roughly
half-second samples. The short prefill trigger reported in MLX issue #3186 shows why this class cannot
be dismissed.

**Pressure without active-memory growth.** The watchdog watches allocated active bytes, not swap rate,
compressor churn, page faults, or system-wide GPU memory pressure. MLX's active-memory metric also
excludes cached buffers, so resident footprint can exceed the sampled value. A workload that thrashes
below the ceiling may not trigger it.

**Other processes.** The guard is process-local. The effective-availability check sees crowding at
startup, but it does not impose a system-wide GPU ceiling or control memory growth in concurrent
processes.

**Larger machines.** The 28 GiB anchor is measured on the 32 GB reference system. Scaling the static
formula to 64 GB, 128 GB, or larger machines is a policy extrapolation, not a validated safety boundary.

**Driver resolution.** The collected documents did not identify a confirmed Apple-side fix. Avoiding
the trigger is an application mitigation, not a repair of the underlying IOGPU failure.

## 8. Lessons for MLX tooling

Four practical lessons come out of the incident.

Name the memory class a limit controls. "Hard," "soft," "wired," "active," "pageable," and
"system-wide" are not interchangeable properties, and collapsing them under one word is what hid the
gap here.

Treat an allocator guideline as advice, not a safety boundary. If exceeding it can legally continue
into swap, callers that need failure isolation must add their own observable stop condition.

Size a process ceiling from observed startup state. Unified memory shared with the rest of the system
makes startup availability part of the workload's operating envelope, though a fixed ceiling still
does not protect against later growth by other processes.

Keep the abort path simpler and more reliable than the failing workload. If reporting the breach
fails, the process must still terminate.

## References and source notes

- Dated project records for the incident timeline, reconstructed failed-worker state, and subsequent
  campaign outcomes. Measurements from those records are identified above as source-reported.
- [`mlx-train-perf` README](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/README.md) and
  [changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/CHANGELOG.md), used as independent
  documentary cross-checks for the released memory-safety behavior.
- Public incident links are listed in section 4. They corroborate the assertion class but do not prove
  identical causality.

---

*Prepared 2026-07-17. Last updated 2026-07-24. Denis Ineshin.*
