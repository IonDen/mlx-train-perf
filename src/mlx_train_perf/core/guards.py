"""Memory guardrails. TWO independent panic-prevention layers, because a wired cap and a
soft memory limit each only bound ONE failure class:

1. The WIRED cap (`install_guardrails` -> `mx.set_wired_limit`) turns a WIRED-class
   over-allocation into a clean MLX OOM instead of a kernel panic (unified memory; wired
   pages can't swap).

2. The soft memory limit (`mx.set_memory_limit`) is ADVISORY only. Its own mlx 0.32.0
   docstring: the limit "is a guideline ... If the memory limit is exceeded and there is
   no more RAM (including swap when available) allocations will result in an exception."
   In other words a PAGEABLE allocation past the soft limit does NOT raise -- it PAGES,
   and only raises once RAM+swap is fully exhausted. A sustained GPU paging storm before
   that point can panic the IOGPU driver ("completeMemory() prepare count underflow" @
   IOGPUMemory.cpp:550 -- observed 2026-07-10 after a Qwen3-8B stock-CE seq-8192 bench
   condition allocated past physical RAM and paged for ~3 h). Neither the wired cap nor
   the soft limit stops that: the over-allocation is pageable, not wired.

The `install_memory_watchdog` below is the third layer that closes that gap: a daemon
thread samples `mx.get_active_memory()` and fails a runaway condition FAST -- writing an
honest aborted-status artifact and hard-exiting the process -- before the paging storm
reaches the IOGPU-panic path.

The ceiling it fails against has TWO terms (`effective_memory_ceiling` combines them):

  effective = min(static_device_ceiling(memory_size), available_at_start - 2 GiB)

1. STATIC, device-relative, anchored-proportional (`memory_ceiling_bytes`). The 32 GB
   machine class is the CALIBRATION ANCHOR: at exactly 32 GiB physical the ceiling is
   28.0 GiB -- a byte-preservation pin, strictly above the 25.68 GiB legitimate near-cap
   peak (so real work is never falsely aborted) and strictly below 32 GiB physical (so a
   pageable over-allocation is failed before it pages past RAM). Below 32 GiB it keeps
   the original semantics (physical minus 4 GiB, floored at half the device). Above it
   the ceiling grows with 95% of the excess working set (the net is the LAST line above
   the wired caps, so only 5% of the excess is reserved; OS overhead is mostly absolute
   and lives in the 32-anchor). The static rule ALSO doubles as the EXPECTED-FREE model
   for the machine class -- on a 32 GiB box we expect ~28 GiB free at training start (the
   rest is OS + background); a measured availability far below that means other heavy
   processes are resident (`memory_divergence_warning`).

2. DYNAMIC, measured at start (`available_memory_bytes` parsing `vm_stat`): the same
   schedule that ran healthy on a fresh-boot machine panicked IOGPU on a pre-loaded one
   (2026-07-10 was STATE-driven), so the ceiling also caps at what is actually free now
   minus a 2 GiB dynamic headroom. If `vm_stat` is unavailable/unparseable the reader
   returns None and the ceiling degrades to the static term with a logged warning --
   never a crash, never a silent zero. If the effective ceiling falls below the
   safe-start floor (memory_size // 4) the run is REFUSED with a typed `MemoryBudgetError`
   at install time (no silent degradation).

Calibration evidence (cross-checked against external sources, numbers stand): our
availability metric is `vm_stat` free+inactive+speculative, which counts RECLAIMABLE file
cache as available -- NOT the Activity-Monitor "used" semantics behind the widely-cited
"macOS uses 20-30% of RAM idle" figures (those include reclaimable caches). On our metric
the 32 GB baseline measured ~91% free at fresh boot (~29 GiB), consistent with the 28 GiB
expectation. The truly-unreclaimable idle footprint grows SUB-LINEARLY with RAM (a few GB
on small machines; ~8-12 GB on 128 GB+ class), and the ladder's reserves (4 GiB at 32;
5.6 at 64; 8.8 at 128; 28 at 512; 54 at 1024, via 95%-of-excess) track that on the safe
side -- reserving more than the likely unreclaimable footprint. Apple's own Metal
working-set model reserves ~20-25% of total RAM (`max_recommended_working_set_size` is
~75-80% of physical; the dev box reports 78%); that is the conservative sibling the WIRED
caps respect, and this net deliberately sits ABOVE it (25.68 GiB legit work ran healthy
above the 24.96 GiB recommendation) and BELOW physical RAM.

Distributed (mx.distributed / mlx.launch / mpirun with mlx_lm): every input to the
effective ceiling is RANK-LOCAL -- `mx.device_info()` (this node's RAM), `vm_stat` (this
node's availability), `mx.get_active_memory()` (this rank's process). No function here
reads global/cluster state, so on a heterogeneous cluster (e.g. a 32 GB + a 64 GB Mac)
each rank independently gets the right device-relative ceiling and its own divergence
warning -- a crowded NODE flags itself, exactly what you want when one slow node gates a
ring. Both the breach `os._exit(70)` and the too-crowded-at-start refusal fire PER-RANK:
a breach hard-exits its rank, so run distributed training under a launcher that propagates
rank failure (mpirun / mlx.launch monitor children and abort the whole job on a nonzero
rank exit) -- do not orphan ranks with nohup-style launches, or peers blocked in
collectives will stall."""
import contextlib
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from mlx_train_perf.errors import MemoryBudgetError

_GIB = 1024**3
_logger = logging.getLogger(__name__)

# The generous default wall-clock budget a bench condition gets when its config leaves
# `wall_budget_s` unset -- long enough that no legitimate condition trips it, short
# enough that a genuinely stuck run (spinning, not just slow) is eventually failed
# instead of paging indefinitely. Campaign configs set a tighter budget explicitly.
#
# FINDING W (safety review): raised from 3600 -> 7200 -- the known-legit longest campaign
# condition ran 3300 s, so the prior 1 h default cleared it by only 9%, uncomfortably
# close for a value meant to be generous. The wall budget is a secondary backstop, not
# the panic guard: the memory-ceiling watchdog (`install_memory_watchdog`'s
# `ceiling_bytes` check) is what catches a PAGEABLE over-allocation before the IOGPU
# paging-storm panic, and the wired cap (`install_guardrails`) turns a WIRED-class
# over-allocation into a clean MLX OOM. Loosening the wall backstop to 2 h therefore
# costs nothing on the safety axis -- it only gives more legitimate long-running
# conditions headroom before a spurious wall-budget abort.
DEFAULT_WALL_BUDGET_S = 7200.0

# Device-proportional growth above the 32 GB-class reference (0019): the fixed wired_gb
# ceiling was tuned for THIS project's 32 GB baseline (the M1 Max reports ~24.96 GiB, just
# UNDER the reference, so every machine the 0.1.0 evidence was measured on gets
# byte-identical caps by construction). Above the reference the caps grow with 85%/92% of
# the excess working set — a 512 GB Ultra can measure big shapes while always keeping
# real headroom below dev_max (clean MLX OOM, never a wired-memory kernel panic).
_PROPORTIONAL_REF_BYTES = 25 * 1024**3


def clamped_caps(dev_max: int, *, wired_gb: int = 20, soft_gb: int = 22) -> tuple[int, int]:
    excess = max(0, dev_max - _PROPORTIONAL_REF_BYTES)
    wired = min(wired_gb * 1024**3 + int(0.85 * excess), int(0.85 * dev_max))
    soft = min(soft_gb * 1024**3 + int(0.92 * excess), int(0.92 * dev_max))
    if wired >= dev_max:  # pragma: no cover — arithmetic guarantee, kept as a tripwire
        raise AssertionError(f"wired cap {wired} >= device max {dev_max}")
    return wired, soft


def install_guardrails(*, wired_gb: int = 20, soft_gb: int = 22) -> int:
    """Re-asserts this project's device-clamped wired/soft caps. Returns the wired
    limit that was ACTIVE immediately before this call (`mx.set_wired_limit`'s own
    contract: it returns the previous limit, and MLX exposes no separate getter) --
    every caller before this return value existed simply discarded it (a bare
    `install_guardrails()` statement), so adding it is backward compatible. A caller
    that needs to OBSERVE whether a cap held across intervening code that might have
    overridden it (e.g. `bench/worker.py`'s `train_step` condition, re-asserting after
    `mlx_lm.tuner.trainer.train()`'s own entry-time override) reads this return value --
    see `wired_cap_holds` for the decision that reading feeds."""
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, soft = clamped_caps(dev_max, wired_gb=wired_gb, soft_gb=soft_gb)
    previous = mx.set_wired_limit(wired)
    mx.set_memory_limit(soft)
    return previous


def wired_cap_holds(*, observed_bytes: int, expected_bytes: int) -> bool:
    """Pure decision: does an OBSERVED wired limit (an `install_guardrails` return
    value, read again after code that might have overridden the cap) match the
    EXPECTED house cap? Isolated as its own function so the decision is testable
    without any MLX device -- the hazard it exists to catch: `mlx_lm.tuner.trainer.
    train()` raises the wired limit to the device max at its own entry (site-packages/
    mlx_lm/tuner/trainer.py:229), silently overriding a cap installed before it runs."""
    return observed_bytes == expected_bytes


# The 32 GB machine class is the calibration anchor -- see the module docstring. At/below
# it the ceiling keeps the original physical-minus-headroom-floored-at-half rule; above it
# only 5% of the excess working set is reserved (the net is the last line above the wired
# caps, and OS overhead is mostly absolute and already in the 32-anchor).
_CEILING_ANCHOR_BYTES = 32 * _GIB
_CEILING_EXCESS_SHARE = 0.95
# Dynamic-term headroom subtracted from measured-available at start (see the module
# docstring): the fresh-boot ~29 GiB free minus this keeps the effective ceiling (~27 GiB)
# above the 25.68 GiB legitimate near-cap peak.
_DYNAMIC_HEADROOM_BYTES = 2 * _GIB
# Below memory_size // this share the machine is too crowded to start safely -> refusal.
_REFUSAL_FLOOR_DIVISOR = 4
# The static rule doubles as the expected-free model; warn when measured availability is
# below this fraction of it (other heavy processes resident). Kw-only-overridable.
_DIVERGENCE_WARN_FRACTION = 0.75


def memory_ceiling_bytes(memory_size: int, *, headroom_gb: int = 4) -> int:
    """Pure STATIC device-relative active-memory ceiling the watchdog fails against,
    anchored at the 32 GB machine class (T15-style anchored-proportional; see the module
    docstring's two-term model). Off PHYSICAL `memory_size`
    (`mx.device_info()["memory_size"]`), not the recommended working set, because the
    storm class allocates PAST physical RAM.

    - `memory_size` <= 32 GiB: physical minus `headroom_gb` GiB, floored at half the
      device (so a machine smaller than the headroom can't get a zero/negative,
      guard-disabling ceiling). At exactly 32 GiB this is the 28.0 GiB byte-preservation
      pin -- above the 25.68 GiB legit peak, below 32 GiB physical. 16 GiB -> 12 GiB;
      8 GiB -> 4 GiB (subtraction and floor coincide there).
    - `memory_size` > 32 GiB: the 28 GiB anchor (at the default headroom) + 95% of the
      excess working set. A 512 GB Ultra can measure big shapes while always keeping real
      headroom below physical (clean MLX OOM, never a wired-memory kernel panic).

    This rule ALSO doubles as the expected-free model for the machine class -- see
    `memory_divergence_warning`."""
    below_anchor = max(memory_size - headroom_gb * _GIB, memory_size // 2)
    if memory_size <= _CEILING_ANCHOR_BYTES:
        return below_anchor
    anchor_ceiling = _CEILING_ANCHOR_BYTES - headroom_gb * _GIB
    return anchor_ceiling + int(_CEILING_EXCESS_SHARE * (memory_size - _CEILING_ANCHOR_BYTES))


def parse_vm_stat(text: str) -> int:
    """Pure availability parser: (free + inactive + speculative) pages times the page
    size, both read out of `vm_stat`'s own text. Availability = free + reclaimable
    (inactive file cache + speculative read-ahead), the pages the allocator can hand out
    without paging. Raises `ValueError` on empty/malformed input (missing header, missing
    a required line, or a non-integer count) -- the wrapper turns that into a safe None,
    never a silent zero. The page size is read from the header (`page size of N bytes`),
    never hardcoded, so a non-16384-byte-page host is scaled correctly."""
    page_match = re.search(r"page size of (\d+) bytes", text)
    if page_match is None:
        raise ValueError("vm_stat output has no 'page size of N bytes' header")
    page_size = int(page_match.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        row = re.search(rf"^{label}:\s*(\d+)\.?\s*$", text, re.MULTILINE)
        if row is None:
            raise ValueError(f"vm_stat output is missing a required line: {label!r}")
        pages += int(row.group(1))
    return pages * page_size


def _run_vm_stat() -> str:
    """Rank-local subprocess read of `vm_stat` (this node's memory only)."""
    return subprocess.run(
        ["vm_stat"], capture_output=True, text=True, check=True, timeout=10,
    ).stdout


def available_memory_bytes(*, reader: Callable[[], str] = _run_vm_stat) -> int | None:
    """Thin `vm_stat` wrapper over `parse_vm_stat`. Returns available bytes, or None when
    the reader or the parse fails (missing tool, timeout, unparseable output) -- the
    dynamic term must DEGRADE SAFELY to the static ceiling, never crash the run and never
    report a silent zero (which would falsely trip the refusal floor). `reader` is
    injectable so the decision is testable without spawning `vm_stat`."""
    try:
        return parse_vm_stat(reader())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def memory_divergence_warning(
    *,
    available_bytes: int,
    expected_free_bytes: int,
    effective_bytes: int,
    warn_fraction: float = _DIVERGENCE_WARN_FRACTION,
) -> str | None:
    """Pure decision: is the machine far more loaded at training start than its class
    should be? The static rule is the EXPECTED-FREE model (e.g. 28 GiB free on a 32 GiB
    box); measured availability below `warn_fraction` of it means other heavy processes
    are resident. Returns a warning naming all three numbers -- measured available,
    expected-free for this machine class, and the effective ceiling the run will proceed
    under -- or None when availability is within the expected band. Proceeding is still
    allowed in the warning band; only the mem//4 refusal floor stops the run."""
    if available_bytes >= warn_fraction * expected_free_bytes:
        return None
    return (
        f"measured available memory {available_bytes} B is far below the "
        f"{expected_free_bytes} B expected free for this machine class "
        f"(other heavy processes resident?); proceeding under an effective memory "
        f"ceiling of {effective_bytes} B"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveCeiling:
    """Result of `effective_memory_ceiling`: the byte ceiling the watchdog fails against,
    plus an optional degraded-start `warning` (divergence or reader-failure) the caller
    logs and records into its bench artifact as `memory_warning`."""

    ceiling_bytes: int
    warning: str | None


def effective_memory_ceiling(
    *,
    memory_size: int | None = None,
    headroom_gb: int = 4,
    dynamic_headroom_bytes: int = _DYNAMIC_HEADROOM_BYTES,
    warn_fraction: float = _DIVERGENCE_WARN_FRACTION,
    available_reader: Callable[[], int | None] = available_memory_bytes,
) -> EffectiveCeiling:
    """The watchdog's install-time ceiling: `min(static, available_at_start - headroom)`,
    plus the divergence warning and the too-crowded-at-start refusal. All inputs are
    RANK-LOCAL (device_info + vm_stat of THIS node) -- safe under `mx.distributed`.

    `memory_size` defaults to this node's physical RAM (`mx.device_info()["memory_size"]`)
    and `available_reader` to the real `vm_stat` read; both are injectable so the whole
    decision is testable with no GPU and no subprocess. Returns an `EffectiveCeiling`
    (ceiling + optional `memory_warning` text). Raises `MemoryBudgetError` when the
    effective ceiling would fall below the safe-start floor (memory_size // 4)."""
    if memory_size is None:
        memory_size = int(mx.device_info()["memory_size"])
    static = memory_ceiling_bytes(memory_size, headroom_gb=headroom_gb)  # also expected-free
    available = available_reader()
    if available is None:
        # Degrade safely to the static term -- a vm_stat hiccup must never refuse a
        # healthy machine (static >= half the device > the mem//4 floor), but the
        # degraded-observability start is still logged and recorded.
        effective = static
        warning: str | None = (
            f"memory availability could not be measured (vm_stat unavailable); proceeding "
            f"under the static memory ceiling {static} B with no dynamic availability check"
        )
        _logger.warning(warning)
    else:
        effective = min(static, available - dynamic_headroom_bytes)
        warning = memory_divergence_warning(
            available_bytes=available, expected_free_bytes=static,
            effective_bytes=effective, warn_fraction=warn_fraction,
        )
        if warning is not None:
            _logger.warning(warning)
    floor = memory_size // _REFUSAL_FLOOR_DIVISOR
    if effective < floor:
        measured = "unavailable" if available is None else str(available)
        raise MemoryBudgetError(
            f"effective memory ceiling {effective} B is below the safe-start floor "
            f"{floor} B (memory_size // {_REFUSAL_FLOOR_DIVISOR}); measured available "
            f"{measured} B -- the machine is too crowded to start this run safely"
        )
    return EffectiveCeiling(ceiling_bytes=effective, warning=warning)


def breach_reason(
    *, active_bytes: int, ceiling_bytes: int, elapsed_s: float, wall_budget_s: float | None,
) -> str | None:
    """Pure watchdog decision -- the thin thread body reduces to exactly this. Memory is
    checked FIRST (a paging storm is the acute hazard; a wall overrun is the backstop):
    returns `"memory_ceiling"` when active memory has reached the ceiling, else
    `"wall_budget"` when a budget is set and elapsed wall exceeds it, else `None`. A
    `wall_budget_s` of `None` disables the wall check entirely."""
    if active_bytes >= ceiling_bytes:
        return "memory_ceiling"
    if wall_budget_s is not None and elapsed_s > wall_budget_s:
        return "wall_budget"
    return None


def _watchdog_step(
    *,
    sampler: Callable[[], int],
    clock: Callable[[], float],
    start: float,
    ceiling_bytes: int,
    wall_budget_s: float | None,
    on_breach: Callable[[str, dict[str, object]], None],
) -> bool:
    """One sample+decide iteration. Returns True when a breach fired (the loop should
    stop) and False otherwise. NEVER raises -- the thread that drives it must not die
    from a transient `get_active_memory` hiccup nor from a failing `on_breach`:
      - a sampler/clock error is swallowed and read as "no breach this tick" (stay armed);
      - a detected breach ALWAYS signals stop (True), even if `on_breach` itself raises,
        so a failing callback can neither un-guard the thread nor spin it re-firing."""
    try:
        active = sampler()
        elapsed = clock() - start
    except Exception:  # a transient sampling error must not disarm the guard
        return False
    reason = breach_reason(
        active_bytes=active, ceiling_bytes=ceiling_bytes,
        elapsed_s=elapsed, wall_budget_s=wall_budget_s,
    )
    if reason is None:
        return False
    # A detected breach ALWAYS signals stop, even if `on_breach` itself fails -- a failing
    # callback must not un-guard the thread nor spin it re-firing.
    with contextlib.suppress(Exception):
        on_breach(reason, {
            "active_bytes": active, "ceiling_bytes": ceiling_bytes,
            "elapsed_s": elapsed, "wall_budget_s": wall_budget_s,
        })
    return True


class WatchdogHandle:
    """Stop handle for `install_memory_watchdog`. The normal completion path calls
    `stop()` (sets the stop event, joins the daemon thread). A breach path never reaches
    `stop()` -- `on_breach` hard-exits the process."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop_event = stop_event
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def install_memory_watchdog(
    *,
    ceiling_bytes: int,
    wall_budget_s: float | None = None,
    on_breach: Callable[[str, dict[str, object]], None],
    sampler: Callable[[], int] = mx.get_active_memory,
    clock: Callable[[], float] = time.monotonic,
    interval_s: float = 0.5,
) -> WatchdogHandle:
    """Start a DAEMON thread that samples active memory every `interval_s` seconds and
    calls `on_breach(reason, details)` the first time `breach_reason` fires, then stops.

    `mx.eval` runs in C++ and releases the GIL during evaluation, so this Python daemon
    thread genuinely gets to sample while a compiled train/loss step is evaluating --
    `mx.get_active_memory` is a plain counter read (no eval, no allocation, no host
    sync). `sampler`/`clock`/`interval_s` are injectable so the decision is testable with
    a fake sampler and no real GPU allocation. The thread NEVER raises (see
    `_watchdog_step`). Returns a `WatchdogHandle`; the caller stops it on the normal
    completion path (a breach never returns -- `on_breach` is expected to hard-exit)."""
    stop_event = threading.Event()

    def _run() -> None:
        start = clock()
        # `Event.wait` returns True immediately once `stop()` is called, else False after
        # the timeout -- so this both throttles sampling to `interval_s` AND wakes
        # promptly on stop, without a busy loop.
        while not stop_event.wait(interval_s):
            if _watchdog_step(
                sampler=sampler, clock=clock, start=start, ceiling_bytes=ceiling_bytes,
                wall_budget_s=wall_budget_s, on_breach=on_breach,
            ):
                return

    thread = threading.Thread(target=_run, name="mlx-train-perf-memory-watchdog", daemon=True)
    thread.start()
    return WatchdogHandle(stop_event, thread)
