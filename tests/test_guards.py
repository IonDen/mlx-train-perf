import logging
import threading

import mlx.core as mx
import pytest

from mlx_train_perf.core import guards
from mlx_train_perf.core.guards import (
    DEFAULT_WALL_BUDGET_S,
    EffectiveCeiling,
    available_memory_bytes,
    breach_reason,
    clamped_caps,
    effective_memory_ceiling,
    install_guardrails,
    install_memory_watchdog,
    memory_ceiling_bytes,
    memory_divergence_warning,
    parse_vm_stat,
    wired_cap_holds,
)
from mlx_train_perf.errors import MemoryBudgetError

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
# memory_ceiling_bytes above the 32 GiB anchor: the T15-style anchored-proportional
# ladder (28 GiB at the 32 GB baseline, + 95% of the excess working set above it). The
# 32 GB baseline is the calibration anchor -- the byte-preservation pin must not move.
# ---------------------------------------------------------------------------

_ANCHOR = 32 * _GIB  # the 32 GB-class calibration baseline


def test_memory_ceiling_preserves_the_28gib_pin_at_the_32gib_anchor() -> None:
    """Byte-preservation pin: the validated 0.1.0 baseline (exactly 28.0 GiB at 32 GiB
    physical) must not move -- every fire-path firing in the northstar sweep and every
    no-false-abort at the 25.68 GiB legit peak was measured against it."""
    assert memory_ceiling_bytes(_ANCHOR) == 28 * _GIB


def test_memory_ceiling_ladder_pins_below_the_anchor() -> None:
    """< 32 GiB keeps the current semantics exactly: physical minus 4 GiB, floored at
    half the device (so a small machine can't get a zero/negative, guard-disabling
    ceiling). 16 GiB -> 12 GiB; 8 GiB -> 4 GiB (subtraction and floor coincide there)."""
    assert memory_ceiling_bytes(16 * _GIB) == 12 * _GIB
    assert memory_ceiling_bytes(8 * _GIB) == 4 * _GIB


def test_memory_ceiling_scales_proportionally_above_the_anchor() -> None:
    """> 32 GiB: the 28 GiB anchor + 0.95 x the excess working set. Exact bytes pinned
    (the net is the LAST line above the wired caps, so only 5% of the excess is reserved;
    OS overhead is mostly absolute and lives in the 32-anchor)."""
    for dev in (64 * _GIB, 128 * _GIB, 256 * _GIB, 512 * _GIB, 1024 * _GIB):
        expected = 28 * _GIB + int(0.95 * (dev - _ANCHOR))
        assert memory_ceiling_bytes(dev) == expected
    # ...and the design's pinned GiB ladder rounds to these headline values.
    assert round(memory_ceiling_bytes(64 * _GIB) / _GIB, 1) == 58.4
    assert round(memory_ceiling_bytes(128 * _GIB) / _GIB, 1) == 119.2
    assert round(memory_ceiling_bytes(256 * _GIB) / _GIB, 1) == 240.8
    assert round(memory_ceiling_bytes(512 * _GIB) / _GIB, 1) == 484.0
    assert round(memory_ceiling_bytes(1024 * _GIB) / _GIB, 1) == 970.4


def test_memory_ceiling_is_monotone_across_the_anchor() -> None:
    sizes = [int(g * _GIB) for g in (4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    ceilings = [memory_ceiling_bytes(s) for s in sizes]
    assert ceilings == sorted(ceilings)


# ---------------------------------------------------------------------------
# parse_vm_stat: the pure availability parser -- (free + inactive + speculative) pages
# times the page size read from the header. RED-tested on real captured output plus the
# malformed cases (empty, missing header, missing line, non-integer count).
# ---------------------------------------------------------------------------

# Real `vm_stat` output captured on the dev M1 Max (2026-07-11).
_VM_STAT_FIXTURE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   766265.
Pages active:                                 544692.
Pages inactive:                               390716.
Pages speculative:                            153140.
Pages throttled:                                   0.
Pages wired down:                             136839.
Pages purgeable:                               28259.
"Translation faults":                       93350168.
Pages copy-on-write:                         2562698.
Pages zero filled:                         975018178.
Pages reactivated:                         294163515.
Pages purged:                                3035809.
File-backed pages:                            416431.
Anonymous pages:                              672117.
Pages stored in compressor:                   331346.
Pages occupied by compressor:                  48749.
Decompressions:                             59388000.
Compressions:                               72215397.
Pageins:                                    11434923.
Pageouts:                                      26890.
Swapins:                                     2342756.
Swapouts:                                     2643329.
"""
_VM_STAT_EXPECTED_AVAILABLE = (766265 + 390716 + 153140) * 16384


def test_parse_vm_stat_sums_free_inactive_speculative_times_page_size() -> None:
    assert parse_vm_stat(_VM_STAT_FIXTURE) == _VM_STAT_EXPECTED_AVAILABLE
    assert parse_vm_stat(_VM_STAT_FIXTURE) == 21_465_022_464


def test_parse_vm_stat_reads_the_page_size_from_the_header_not_hardcoded() -> None:
    """A host with a 4096-byte page must not be mis-scaled by the 16384 dev value."""
    text = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free:                    100.\n"
        "Pages inactive:                 50.\n"
        "Pages speculative:              25.\n"
    )
    assert parse_vm_stat(text) == (100 + 50 + 25) * 4096


def test_parse_vm_stat_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="page size"):
        parse_vm_stat("")


def test_parse_vm_stat_raises_when_the_page_size_header_is_missing() -> None:
    text = "Pages free: 100.\nPages inactive: 50.\nPages speculative: 25.\n"
    with pytest.raises(ValueError, match="page size"):
        parse_vm_stat(text)


def test_parse_vm_stat_raises_when_a_required_line_is_missing() -> None:
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                    100.\n"  # no inactive / speculative
    )
    with pytest.raises(ValueError, match="required line"):
        parse_vm_stat(text)


def test_parse_vm_stat_raises_on_a_non_integer_count() -> None:
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                    abc.\n"
        "Pages inactive:                 50.\n"
        "Pages speculative:              25.\n"
    )
    with pytest.raises(ValueError, match="required line"):
        parse_vm_stat(text)


# ---------------------------------------------------------------------------
# available_memory_bytes: the thin `vm_stat` wrapper. Degrades to None (never raises,
# never a silent zero) when the reader or the parse fails, so the ceiling can fall back.
# ---------------------------------------------------------------------------


def test_available_memory_bytes_parses_an_injected_reader() -> None:
    assert available_memory_bytes(reader=lambda: _VM_STAT_FIXTURE) == _VM_STAT_EXPECTED_AVAILABLE


def test_available_memory_bytes_returns_none_when_the_reader_raises() -> None:
    def boom() -> str:
        raise OSError("vm_stat missing")

    assert available_memory_bytes(reader=boom) is None


def test_available_memory_bytes_returns_none_on_unparseable_output() -> None:
    assert available_memory_bytes(reader=lambda: "garbage") is None


def test_available_memory_bytes_reads_real_vm_stat_on_this_machine() -> None:
    """Integration: the real default reader returns a positive byte count on the dev Mac
    (vm_stat is always present on macOS). Rank-local -- reads only this node's memory."""
    value = available_memory_bytes()
    assert value is not None
    assert value > 0


# ---------------------------------------------------------------------------
# memory_divergence_warning: the pure "machine more loaded than expected" decision. The
# static rule doubles as the EXPECTED-FREE model for the machine class (28 GiB free
# expected on a 32 GiB box); a measured availability far below it means other heavy
# processes are resident. Proceeds -- only the mem//4 refusal floor stops the run.
# ---------------------------------------------------------------------------


def test_divergence_warning_fires_far_below_expected_and_names_all_three_numbers() -> None:
    """Denis's example: a 32 GiB machine with only 8 GiB free (other heavy processes
    resident). The warning names measured available, expected-free for this machine
    class, and the effective ceiling the run will use."""
    available, expected, effective = 8 * _GIB, 28 * _GIB, 6 * _GIB
    warning = memory_divergence_warning(
        available_bytes=available, expected_free_bytes=expected, effective_bytes=effective,
    )
    assert warning is not None
    assert str(available) in warning
    assert str(expected) in warning
    assert str(effective) in warning


def test_divergence_warning_silent_on_the_fresh_boot_baseline() -> None:
    """29 GiB available vs a 28 GiB expectation is a healthy fresh-boot start -- no warn."""
    assert memory_divergence_warning(
        available_bytes=29 * _GIB, expected_free_bytes=28 * _GIB, effective_bytes=27 * _GIB,
    ) is None


def test_divergence_warning_fraction_is_overridable() -> None:
    kw = {"available_bytes": 22 * _GIB, "expected_free_bytes": 28 * _GIB,
          "effective_bytes": 20 * _GIB}
    assert memory_divergence_warning(**kw) is None  # 22 GiB >= 0.75 x 28 (21)
    assert memory_divergence_warning(**kw, warn_fraction=0.85) is not None  # 22 < 0.85 x 28


# ---------------------------------------------------------------------------
# effective_memory_ceiling: min(static, available - 2 GiB), the divergence warning, the
# reader-failure fallback, and the mem//4 refusal floor. Injected reader + memory_size,
# no GPU, no real vm_stat.
# ---------------------------------------------------------------------------


def test_effective_ceiling_dynamic_wins_on_a_crowded_machine() -> None:
    """Only 15 GiB free tightens the ceiling to available - 2 GiB (below the 28 GiB
    static) and warns (15 < 0.75 x 28), but still proceeds (13 GiB > the 8 GiB floor)."""
    result = effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: 15 * _GIB)
    assert isinstance(result, EffectiveCeiling)
    assert result.ceiling_bytes == 13 * _GIB  # 15 - 2
    assert result.warning is not None


def test_effective_ceiling_static_wins_on_an_empty_machine() -> None:
    """When available - 2 GiB exceeds the static device rule, the static ceiling caps."""
    result = effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: 31 * _GIB)
    assert result.ceiling_bytes == 28 * _GIB  # min(28, 29)
    assert result.warning is None


def test_effective_ceiling_reproduces_the_fresh_boot_baseline() -> None:
    """Fresh boot ~91% free (=~29 GiB) on the 32 GiB dev Mac: the effective ceiling stays
    ABOVE the 25.68 GiB legitimate near-cap peak, so real work is never falsely aborted.
    The load-bearing no-false-abort reproduction, with an injected reader."""
    result = effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: 29 * _GIB)
    assert result.ceiling_bytes == 27 * _GIB  # min(28, 27)
    assert result.ceiling_bytes > int(25.68 * _GIB)
    assert result.warning is None


def test_effective_ceiling_falls_back_to_static_when_the_reader_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A vm_stat hiccup must NOT refuse a healthy machine -- degrade to the static ceiling
    and log that no availability check was possible (never a silent zero)."""
    with caplog.at_level(logging.WARNING):
        result = effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: None)
    assert result.ceiling_bytes == 28 * _GIB
    assert any("vm_stat" in r.message for r in caplog.records)


def test_effective_ceiling_reader_failure_never_refuses() -> None:
    """The fallback (availability unknown) must never trip the refusal floor -- static is
    always >= half the device > the quarter-device floor."""
    result = effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: None)
    assert result.ceiling_bytes == 28 * _GIB
    assert result.warning is not None  # records the degraded-observability start


def test_effective_ceiling_logs_the_divergence_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: 15 * _GIB)
    assert any(str(13 * _GIB) in r.message for r in caplog.records)


def test_effective_ceiling_defaults_memory_size_to_this_node() -> None:
    """With `memory_size` unset it reads THIS node's physical RAM (rank-local
    `device_info` -- a metadata read, no GPU dispatch). An injected 'plenty free' reader
    keeps the assertion deterministic: the static device rule caps and there is no warn."""
    result = effective_memory_ceiling(available_reader=lambda: 10**15)
    dev = int(mx.device_info()["memory_size"])
    assert result.ceiling_bytes == memory_ceiling_bytes(dev)
    assert result.warning is None


def test_effective_ceiling_refuses_below_the_quarter_device_floor() -> None:
    """Below memory_size // 4 the machine is too crowded to start safely -- a typed,
    package-rooted MemoryBudgetError, not a silent degrade. The message carries the
    measured available and the floor."""
    with pytest.raises(MemoryBudgetError) as exc:
        effective_memory_ceiling(memory_size=_ANCHOR, available_reader=lambda: 5 * _GIB)
    msg = str(exc.value)
    assert str(5 * _GIB) in msg  # measured available
    assert str(_ANCHOR // 4) in msg  # the floor


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


def test_default_wall_budget_is_two_hours() -> None:
    """FINDING W (safety review, reviewer-recommended): the known-legit longest campaign
    condition ran 3300 s -- the prior 3600 s default cleared it by only 9%. The MEMORY
    watchdog is the actual panic guard (a WIRED-class over-allocation goes clean-OOM via
    the wired cap; a PAGEABLE over-allocation is what the memory-ceiling watchdog exists
    to catch fast), so loosening the wall backstop costs nothing on the safety axis."""
    assert DEFAULT_WALL_BUDGET_S == 7200.0
