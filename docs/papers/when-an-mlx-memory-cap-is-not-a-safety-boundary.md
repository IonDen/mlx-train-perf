# When an MLX Memory Cap Is Not a Safety Boundary

*An Apple unified-memory incident report and the process-local watchdog added afterward*

> 📄 [Read on the website](https://ineshin.space/papers/when-an-mlx-memory-cap-is-not-a-safety-boundary/) — same paper, formatted for reading.

## Correction, 2026-11-13

An earlier version of this memo attributed the 25.68 GiB peak to the fused-loss arm and argued that
the stock-loss worker's extra logits pushed the same workload past physical memory. The project's
release record shows otherwise: 25.68 GiB is the stock-loss, stock-attention step completing in the
release benchmark, and the fused-loss arm at 8,192 tokens recorded an abort at the memory ceiling.
Section 3 now explains the incident by machine state, as the project's guard module records it: the
same benchmark schedule ran healthy on a freshly booted machine and panicked on a loaded one.

---

A configured memory limit can be real without being a hard stop. That distinction matters on Apple
Silicon, where CPU and GPU workloads share physical memory and some GPU allocations remain pageable.

During an `mlx-train-perf` benchmark campaign on MLX 0.32.0, a long-context training condition ended
in a macOS kernel panic inside `IOGPUMemory`. The project's
[guard module](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/core/guards.py#L13-L15)
records that the condition allocated past the physical memory of a 32 GB M1 Max and paged for about
three hours first. Its wired limit remained in force, while the softer memory limit was advisory and
allowed pageable allocation to continue toward RAM plus swap. The worker's final paging state is a
reconstruction, not a recovered measurement.

The incident disproved a standing project assumption: a wired limit alone does not turn every GPU
over-allocation into a clean MLX exception. The mitigation was a third layer: a best-effort,
process-local daemon intended to stop sustained active-memory growth before it reaches the observed
storm regime. The project installed it in its own benchmark worker and scripts, not in a user's own
training loop. It is not a hard machine-safety boundary and can miss fast allocations, below-ceiling
thrashing, or pressure created after startup by other processes.

Nothing was rerun for this memo. Its numbers come from the project's own release record: the
[v0.4.0 README](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/README.md) and
[changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/CHANGELOG.md), plus the v0.9.1
changelog for the later attention figures. The incident's date, its paging of about three hours,
the panic assertion and its dependence on machine state come from the
[guard module](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/core/guards.py)
at v0.9.1. That module's opening comment, which says the wired cap turns a wired-class
over-allocation into a clean MLX out-of-memory error, predates the correction in section 1. The
account of the watchdog comes from the same module and the
[abort writer](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/bench/artifacts.py)
at v0.9.1. The incident's raw logs were not published.

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

The incident occurred during a benchmark campaign on 2026-07-10, on an M1 Max with 32 GB of unified
memory and MLX 0.32.0.

The failing worker was a Qwen3-8B training-step condition at 8,192 tokens with stock cross-entropy.
That loss materializes the full logits tensor, shape `(8192, 151936)`, on top of the long-context
attention workload. The guard module names the failing condition's loss arm, not its attention arm.

Three 8,192-token Qwen3-8B outcomes are public:

| condition | where it ran | reported outcome | source |
|---|---|---|---|
| stock cross-entropy | incident campaign, 2026-07-10 | allocated past physical RAM, paged for about three hours, then kernel-panicked | guard module |
| stock cross-entropy, stock attention | 0.2.0 release benchmark | completed at 25.68 GiB total peak | v0.4.0 README and changelog |
| fused cross-entropy, stock attention | 0.2.0 release benchmark | crossed the memory ceiling and recorded an abort | v0.4.0 changelog |

Which other conditions ran on the incident day, and how they ended, is not in the published evidence.

The project's guard module records the panic assertion as
`"completeMemory() prepare count underflow"` at `IOGPUMemory.cpp:550`.

## 3. What the public record explains, and the boundary of the claim

Four observations from public sources bear on the incident.

First, the soft limit was never specified to stop allocation. The installed MLX documentation called
`mx.set_memory_limit` a guideline and said that allocation errors occur when no more RAM, including
swap, remains. The MLX 0.32.0 docstring reads:

> "The memory limit is a guideline ... If the memory limit is exceeded and there is no more RAM
> (including swap when available) allocations will result in an exception."

Second, a different benchmark on the same machine had already paged. The v0.4.0 README reports
that the stock single-operation attention benchmark at 8,192 tokens, run on MLX 0.32.0 before the
watchdog shipped, paged instead of failing cleanly.

Third, the outcome depended on the machine's state. The guard module records that the same schedule
ran healthy on a freshly booted machine and panicked the IOGPU driver on one that was already loaded,
and it calls the 2026-07-10 incident state-driven
([guard module, lines 40 to 42](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/core/guards.py#L40-L42)).
It does not say when the healthy run took place.

Fourth, legitimate work at this length ran close to the machine's practical ceiling. In the 0.2.0
release benchmark the stock-loss, stock-attention step completed at 25.68 GiB. The changelog gives
that arm's throughput at 8,192 tokens and says the lighter fused-loss arm has no stock-attention
number there, because it crossed the memory ceiling and recorded an abort. The release record does
not give that run's ceiling, which depends on the memory free at start (section 5).

No peak for the failed worker itself is available. Severe pageable GPU pressure was the condition
associated with the incident, but the evidence does not isolate it as the cause of the IOGPU
assertion. On the guard module's account, machine state explains why this run paged when the same
schedule had not; it says nothing about how the assertion itself arises.

These observations support the project's incident reconstruction:

```text
machine already loaded when the worker starts
        ↓
wired limit remains active
        ↓
pageable allocation crosses the advisory soft limit
        ↓
allocation continues past physical RAM into paging (about three hours)
        ↓
kernel panic at the recorded IOGPU assertion
        causal trigger unknown
```

They do not establish the internal driver mechanism. Apple's IOGPU implementation is closed. The
project hypothesized that repeated page-out and fault-in activity disturbed prepare/complete
bookkeeping for GPU-mapped memory, but it could not inspect or prove that explanation. The identical
panic also appears in public reports with much shorter single-allocation triggers, so duration-specific
paging is not required by the available evidence.

A later MLX release changed the attention part of this workload. In the library's later attention
benchmark (causal mask, 8,192 tokens), stock attention's forward plus backward peaked at 12.532 GiB
on MLX 0.32.0 and 0.682 GiB on MLX 0.32.4, according to the
[v0.9.1 changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/CHANGELOG.md). MLX 0.32.4
made its backward much lighter on memory. The training step was not re-measured on 0.32.4, and the
library pins `mlx==0.32.0` as of v0.9.1, so the 25.68 GiB figure applies to MLX 0.32.0.

The evidence is classified as follows:

| claim | status | documentary source |
|---|---|---|
| three hours of paging past physical RAM, the panic assertion, and the dependence on machine state | directly observed incident, as the project records it | project guard module at v0.9.1 |
| stock single-op attention paging at 8,192 tokens before the watchdog | separate source-reported measurement on MLX 0.32.0 | v0.4.0 README |
| stock-loss step completing at 25.68 GiB; fused-loss arm aborting at the ceiling | release benchmark on MLX 0.32.0 | v0.4.0 README and changelog |
| prepare/complete bookkeeping mechanism | unverified hypothesis | project reconstruction and comparison with public reports |
| released watchdog behavior and specific abort paths | source-reported operational behavior | v0.4.0 README and changelog; watchdog code at v0.9.1 |

## 4. Reports from other machines narrow the conclusion

Several public reports contain the same assertion and source location:

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
MLX/driver interaction, or that sustained paging is necessary.

What the public evidence supports is narrower: the local workload entered severe pageable GPU
pressure and ended at the same IOGPU assertion reported by other memory-intensive Apple GPU workloads.
It does not show that the three hours of paging caused a specific driver bookkeeping bug.

## 5. The mitigation: an active-memory watchdog

The project retained both existing MLX limits and added a separate process-local guard. The resulting
model has three layers:

1. a residency cap for non-pageable GPU memory;
2. the MLX soft memory guideline;
3. a best-effort active-memory watchdog for sustained process-local growth.

The watchdog is installed in the project's
[benchmark worker](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/bench/worker.py),
which the `mlx-train-perf contribute` kit also drives, and in its benchmark scripts. A fine-tune
that uses the library's loss or attention in its own training loop runs without it.

The watchdog samples `mx.get_active_memory()` roughly twice per second. Its effective ceiling is the
smaller of a device-relative static value and startup effective availability minus a 2 GiB cushion:

```text
effective ceiling = min(
    static ceiling for the machine's memory class,
    available memory at start - 2 GiB
)
```

For the 32 GB reference machine, the static value is 28.0 GiB. It sits above the project's largest
legitimate measured peak, 25.68 GiB, and below physical memory. The dynamic term exists because a
schedule that fits after a clean boot can still be unsafe when other processes already occupy
unified memory. Here, availability is a one-shot `vm_stat` sum of free, inactive, and speculative
pages, including reclaimable inactive file cache. In the 0.3.0 context-ceiling figures reported in
the v0.4.0 README, the effective ceiling sat near 24.5 GiB rather than at the 28 GiB static anchor.
That is below the 25.68 GiB peak, so the watchdog can also stop legitimate work: the
fused-loss, stock-attention arm at 8,192 tokens recorded an abort instead of a number. Availability
is sampled when the watchdog is installed; the resulting ceiling is fixed for that
worker and does not adapt if another process grows later.

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

After the fix, the v0.4.0 README and changelog report that the stock single-operation attention
benchmark at 8,192 tokens stops at the ceiling instead of paging: on a 32 GB machine it now ends with
an `aborted_memory_ceiling` record by design.

Those two documents are the project's own release record. They describe the guard design and the
`aborted_memory_ceiling` path. They do not describe the half-second sampling, the wall-clock backstop
or the unconditional exit; those are in the watchdog code at v0.9.1.

That abort is a source-reported operational observation, not proof that the watchdog prevents every
form of the IOGPU assertion. It exercises the path the mitigation was designed for: sustained growth
in MLX active memory that crosses the configured ceiling slowly enough for the daemon to observe it.

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
processes. Since v0.8.0 the benchmark runner can also run each worker under
[mlx-guard](https://github.com/IonDen/mlx-guard), an external supervisor that measures what macOS
charges the process; this is opt-in, and the in-process watchdog stays on under it
([v0.9.1 changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/CHANGELOG.md)). The
companion paper
[Why the memory limit must live outside the process](https://github.com/IonDen/mlx-guard/blob/main/docs/papers/why-the-memory-limit-must-live-outside-the-process.md)
explains that design.

**Larger machines.** The 28 GiB anchor is measured on the 32 GB reference system. Scaling the static
formula to 64 GB, 128 GB, or larger machines is a policy extrapolation, not a validated safety boundary.

**Driver resolution.** None of the sources cited here reports a confirmed Apple-side fix. Avoiding
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

- [`mlx-train-perf` README](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/README.md) and
  [changelog](https://github.com/IonDen/mlx-train-perf/blob/v0.4.0/CHANGELOG.md) at v0.4.0, the
  project's own release record for the measurements and the released memory-safety behavior.
- [Changelog at v0.9.1](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/CHANGELOG.md), for the
  attention figures on MLX 0.32.4 and 0.32.0 and the opt-in external supervision.
- [Guard module](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/core/guards.py),
  [abort writer](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/bench/artifacts.py)
  and [benchmark worker](https://github.com/IonDen/mlx-train-perf/blob/v0.9.1/src/mlx_train_perf/bench/worker.py)
  at v0.9.1, for the incident date, paging, assertion and dependence on machine state, the
  watchdog's sampling, backstop and exit path, and where it is installed.
- Public incident links are listed in section 4. They corroborate the assertion class but do not prove
  identical causality.

---

*Prepared 2026-07-17. Last updated 2026-11-13. Denis Ineshin.*
