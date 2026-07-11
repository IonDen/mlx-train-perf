import threading

import mlx.core as mx

from mlx_train_perf.core import guards
from mlx_train_perf.core.guards import (
    DEFAULT_WALL_BUDGET_S,
    breach_reason,
    clamped_caps,
    install_guardrails,
    install_memory_watchdog,
    memory_ceiling_bytes,
    wired_cap_holds,
)

_GIB = 1024**3


def test_caps_below_requested_on_big_machine() -> None:
    dev_max = 25 * 1024**3  # ~M1 Max 32 GB
    wired, soft = clamped_caps(dev_max)
    assert wired == 20 * 1024**3 and soft == 22 * 1024**3  # noqa: PT018


def test_caps_clamp_on_small_machine() -> None:
    dev_max = 11 * 1024**3  # 16 GB Mac class — hardcoded 20 GB would DISABLE the guard
    wired, soft = clamped_caps(dev_max)
    assert wired == int(0.85 * dev_max) and soft == int(0.92 * dev_max)  # noqa: PT018
    assert wired < dev_max


def test_caps_unchanged_on_the_m1max_baseline() -> None:
    """0019 preservation pin: the REAL M1 Max 32 GB `max_recommended_working_set_size`
    (26,800,603,136 B, just under the 25 GiB proportional reference) must produce the
    EXACT 0.1.0 caps — every shipped calibration constant and bench gate was measured
    under them, so any drift here silently re-bases the project's memory evidence."""
    wired, soft = clamped_caps(26_800_603_136)
    assert wired == 20 * 1024**3
    assert soft == 22 * 1024**3


def test_caps_scale_proportionally_above_the_reference() -> None:
    """0019: the fixed ~20 GiB ceiling was 32 GB-tuned — on big machines it would forbid
    measuring big shapes (the community kit's whole purpose). Above the 25 GiB reference
    the wired cap grows with the device while always keeping headroom below dev_max."""
    for dev_gib in (57.6, 115.2, 460.8):  # 64 / 128 / 512 GB unified-memory classes
        dev_max = int(dev_gib * 1024**3)
        wired, soft = clamped_caps(dev_max)
        assert wired > 20 * 1024**3, f"{dev_gib}: cap stuck at the 32 GB-tuned ceiling"
        assert wired < dev_max  # panic-proof property
        assert wired < soft < dev_max


def test_caps_are_monotone_in_device_size() -> None:
    sizes = [int(g * 1024**3) for g in (8, 11, 24.96, 25, 40, 57.6, 115.2, 230, 460.8)]
    wireds = [clamped_caps(s)[0] for s in sizes]
    assert wireds == sorted(wireds)


def test_install_runs_on_this_machine() -> None:
    install_guardrails()  # must not raise; conftest already installed session caps
    assert mx.device_info()["max_recommended_working_set_size"] > 0


def test_install_guardrails_returns_the_previously_active_wired_limit() -> None:
    """`install_guardrails` now returns whatever wired limit was ACTIVE immediately
    before it re-asserted the house cap (`mx.set_wired_limit`'s own contract — MLX
    exposes no separate getter). `bench/worker.py`'s `train_step` condition reads this
    return value to OBSERVE whether the cap held through an mlx-lm training loop that
    raises it to the device max at its own entry."""
    install_guardrails()  # our own cap is active now
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _soft = clamped_caps(dev_max)
    previous = install_guardrails()  # re-asserting a cap that's already active is a no-op
    assert previous == wired


def test_wired_cap_holds_true_when_observed_matches_expected() -> None:
    assert wired_cap_holds(observed_bytes=100, expected_bytes=100) is True


def test_wired_cap_holds_false_when_observed_differs_from_expected() -> None:
    assert wired_cap_holds(observed_bytes=100, expected_bytes=200) is False


# ---------------------------------------------------------------------------
# memory_ceiling_bytes: the pure active-memory ceiling (physical RAM minus headroom).
# The wired cap only bounds the WIRED class; a PAGEABLE over-allocation still pages
# past physical RAM into the GPU paging storm that panicked IOGPU on 2026-07-10. The
# ceiling is what a runtime watchdog fails against, so it must sit ABOVE legitimate
# near-cap work (25.68 GiB measured) and BELOW the storm class (past physical RAM).
# ---------------------------------------------------------------------------

_M1MAX_32GB_MEMORY_SIZE = 34_359_738_368  # mx.device_info()["memory_size"], exactly 32 GiB


def test_memory_ceiling_subtracts_headroom_gib() -> None:
    assert memory_ceiling_bytes(_M1MAX_32GB_MEMORY_SIZE) == _M1MAX_32GB_MEMORY_SIZE - 4 * _GIB


def test_memory_ceiling_sits_above_legit_peak_and_below_storm_on_this_machine() -> None:
    """The load-bearing property this whole guardrail turns on: 28.0 GiB is strictly
    above the 25.68 GiB legitimate near-cap peak (so real work is never falsely aborted)
    and strictly below 32 GiB physical RAM (so the paging storm is caught before it
    pages past RAM into an IOGPU panic)."""
    ceiling_gib = memory_ceiling_bytes(_M1MAX_32GB_MEMORY_SIZE) / _GIB
    assert ceiling_gib == 28.0
    assert 25.68 < ceiling_gib < 32.0


def test_memory_ceiling_headroom_param_widens_the_reservation() -> None:
    assert memory_ceiling_bytes(_M1MAX_32GB_MEMORY_SIZE, headroom_gb=8) == (
        _M1MAX_32GB_MEMORY_SIZE - 8 * _GIB
    )


def test_memory_ceiling_floors_at_half_the_device_on_a_tiny_machine() -> None:
    """A device smaller than the headroom would make the raw subtraction zero/negative,
    disabling the watchdog. The floor keeps the ceiling at half the device so the guard
    stays meaningful (never reserves MORE than half the machine as headroom)."""
    tiny = 4 * _GIB  # 4 GiB - 4 GiB headroom == 0 -> the floor must engage
    ceiling = memory_ceiling_bytes(tiny)
    assert ceiling == tiny // 2
    assert ceiling > 0


def test_memory_ceiling_floor_does_not_bind_on_a_normal_device() -> None:
    assert memory_ceiling_bytes(_M1MAX_32GB_MEMORY_SIZE) > _M1MAX_32GB_MEMORY_SIZE // 2


# ---------------------------------------------------------------------------
# breach_reason: the pure decision the watchdog thread reduces to (memory checked
# first, then wall). Kept out of the thread so the decision is testable with no thread,
# no clock, no GPU.
# ---------------------------------------------------------------------------


def test_breach_reason_memory_when_active_at_or_above_ceiling() -> None:
    assert breach_reason(
        active_bytes=100, ceiling_bytes=100, elapsed_s=0.0, wall_budget_s=None
    ) == "memory_ceiling"


def test_breach_reason_wall_when_elapsed_exceeds_budget() -> None:
    assert breach_reason(
        active_bytes=0, ceiling_bytes=100, elapsed_s=5.0, wall_budget_s=1.0
    ) == "wall_budget"


def test_breach_reason_none_when_under_both_limits() -> None:
    assert breach_reason(
        active_bytes=50, ceiling_bytes=100, elapsed_s=0.5, wall_budget_s=1.0
    ) is None


def test_breach_reason_memory_takes_precedence_over_wall() -> None:
    assert breach_reason(
        active_bytes=200, ceiling_bytes=100, elapsed_s=5.0, wall_budget_s=1.0
    ) == "memory_ceiling"


def test_breach_reason_no_wall_budget_never_wall_breaches() -> None:
    assert breach_reason(
        active_bytes=0, ceiling_bytes=100, elapsed_s=10**9, wall_budget_s=None
    ) is None


# ---------------------------------------------------------------------------
# _watchdog_step: one sample+decide iteration (the thin thread-body shell). Injecting
# sampler/clock/on_breach makes every branch deterministic without a real thread.
# ---------------------------------------------------------------------------


def test_watchdog_step_returns_false_and_is_quiet_under_ceiling() -> None:
    calls: list[str] = []
    fired = guards._watchdog_step(
        sampler=lambda: 50, clock=lambda: 0.0, start=0.0, ceiling_bytes=100,
        wall_budget_s=None, on_breach=lambda reason, _d: calls.append(reason),
    )
    assert fired is False
    assert calls == []


def test_watchdog_step_reports_memory_breach_and_signals_stop() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    fired = guards._watchdog_step(
        sampler=lambda: 200, clock=lambda: 5.0, start=0.0, ceiling_bytes=100,
        wall_budget_s=None, on_breach=lambda reason, details: calls.append((reason, details)),
    )
    assert fired is True
    assert calls[0][0] == "memory_ceiling"
    assert calls[0][1]["active_bytes"] == 200
    assert calls[0][1]["ceiling_bytes"] == 100


def test_watchdog_step_swallows_a_sampler_error_and_keeps_guarding() -> None:
    """A transient `get_active_memory` hiccup must NOT be read as a breach, and must NOT
    raise out of the thread -- the guard stays armed for the next tick."""
    calls: list[str] = []

    def boom() -> int:
        raise RuntimeError("sampler down")

    fired = guards._watchdog_step(
        sampler=boom, clock=lambda: 0.0, start=0.0, ceiling_bytes=100,
        wall_budget_s=None, on_breach=lambda reason, _d: calls.append(reason),
    )
    assert fired is False
    assert calls == []


def test_watchdog_step_swallows_an_on_breach_error_but_still_stops() -> None:
    """`on_breach` failing must never un-guard the thread nor loop it forever calling a
    failing callback -- a detected breach always signals stop."""
    def boom_breach(_reason: str, _details: dict[str, object]) -> None:
        raise RuntimeError("on_breach down")

    fired = guards._watchdog_step(
        sampler=lambda: 200, clock=lambda: 0.0, start=0.0, ceiling_bytes=100,
        wall_budget_s=None, on_breach=boom_breach,
    )
    assert fired is True


# ---------------------------------------------------------------------------
# install_memory_watchdog: the daemon-thread shell (fire / no-fire / stop / wall).
# Sampler and clock are injected so no real GPU allocation happens; on_breach records
# into a threading.Event so the assertions are race-free.
# ---------------------------------------------------------------------------


def test_watchdog_fires_on_breach_when_sampler_exceeds_ceiling() -> None:
    fired = threading.Event()
    captured: dict[str, object] = {}

    def on_breach(reason: str, details: dict[str, object]) -> None:
        captured["reason"] = reason
        captured["details"] = details
        fired.set()

    handle = install_memory_watchdog(
        ceiling_bytes=1000, on_breach=on_breach, sampler=lambda: 2000, interval_s=0.001,
    )
    assert fired.wait(timeout=2.0)
    handle.stop()
    assert captured["reason"] == "memory_ceiling"
    assert captured["details"]["active_bytes"] == 2000  # type: ignore[index]


def test_watchdog_does_not_fire_when_under_ceiling() -> None:
    fired = threading.Event()
    handle = install_memory_watchdog(
        ceiling_bytes=1000, on_breach=lambda _r, _d: fired.set(),
        sampler=lambda: 500, interval_s=0.001,
    )
    # Many sample ticks in 0.1 s, all under the ceiling -> the callback must never fire.
    assert not fired.wait(timeout=0.1)
    handle.stop()
    assert not fired.is_set()


def test_watchdog_stop_handle_ends_the_thread() -> None:
    handle = install_memory_watchdog(
        ceiling_bytes=10**15, on_breach=lambda _r, _d: None,
        sampler=lambda: 0, interval_s=0.001,
    )
    assert handle.is_alive()
    handle.stop()
    assert not handle.is_alive()


def test_watchdog_fires_on_wall_budget_breach() -> None:
    fired = threading.Event()
    captured: dict[str, object] = {}
    clock_values = iter([0.0, 100.0])  # start, then an elapsed well past the 1 s budget

    def fake_clock() -> float:
        return next(clock_values, 100.0)

    def on_breach(reason: str, _details: dict[str, object]) -> None:
        captured["reason"] = reason
        fired.set()

    handle = install_memory_watchdog(
        ceiling_bytes=10**15, wall_budget_s=1.0, on_breach=on_breach,
        sampler=lambda: 0, clock=fake_clock, interval_s=0.001,
    )
    assert fired.wait(timeout=2.0)
    handle.stop()
    assert captured["reason"] == "wall_budget"


def test_default_wall_budget_is_one_hour() -> None:
    assert DEFAULT_WALL_BUDGET_S == 3600.0
