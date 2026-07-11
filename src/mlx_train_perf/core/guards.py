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
reaches the IOGPU-panic path. `memory_ceiling_bytes` is the pure ceiling it fails
against (physical RAM minus headroom): above legitimate near-cap work (~25.7 GiB
measured on the 32 GB reference), below the storm class (past physical RAM)."""
import contextlib
import threading
import time
from collections.abc import Callable

import mlx.core as mx

_GIB = 1024**3

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


def memory_ceiling_bytes(memory_size: int, *, headroom_gb: int = 4) -> int:
    """Pure active-memory ceiling the watchdog fails against: physical `memory_size`
    (`mx.device_info()["memory_size"]`) minus `headroom_gb` GiB. Floored at half the
    device so a machine smaller than the headroom can't produce a zero/negative ceiling
    that would disable the guard -- the floor keeps it meaningful (never reserving MORE
    than half the machine as headroom).

    On the 32 GB reference (`memory_size` == 32 GiB): 28.0 GiB -- strictly ABOVE the
    25.68 GiB legitimate near-cap peak measured on this project's flagship conditions (so
    real work is never falsely aborted) and strictly BELOW 32 GiB physical RAM (so a
    pageable over-allocation is failed before it pages past RAM into the IOGPU panic).
    Unlike `clamped_caps` (tuned to `max_recommended_working_set_size`, ~24.96 GiB), this
    ceiling is off PHYSICAL memory: the storm class allocates PAST physical RAM, so the
    physical size, not the recommended working set, is the right reference."""
    return max(memory_size - headroom_gb * _GIB, memory_size // 2)


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
