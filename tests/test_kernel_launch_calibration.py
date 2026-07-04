"""Pure-logic tests for the ramped, sustained-load rate calibration in
`mlx_train_perf.core.kernel.launch`. No GPU here — `calibrate` is exercised through a fake
`measure` closure; the real Metal-backed closures are covered by the metal-lane tests in
`test_kernel_guard.py`."""
import pytest

from mlx_train_perf.core.kernel.launch import calibrate, next_probe_tile, sustain_reps

# Regression shape: n=8192, d=4096, v=151936 — the production dense-CE shape whose
# false-refusal this calibration replaces the micro-probe to fix.
N, D, V = 8192, 4096, 151936


def test_next_probe_tile_rejects_8192_when_projected_over_budget() -> None:
    # halved rate 211e9: n*8192*d/211e9 ~= 1.30s > 1.0s budget -> must shrink to 4096
    # (n*4096*d/211e9 ~= 0.65s, within budget).
    assert next_probe_tile(rate_macs_per_s=211e9, n=N, d=D, v=V) == 4096


def test_next_probe_tile_accepts_8192_when_within_budget() -> None:
    # halved rate 682e9 (half of the measured 1,363.4 G MAC/s production rate):
    # n*8192*d/682e9 ~= 0.40s, within budget -> stays at the 8192 cap.
    assert next_probe_tile(rate_macs_per_s=682e9, n=N, d=D, v=V) == 8192


def test_next_probe_tile_is_a_power_of_two() -> None:
    for rate in (1e9, 50e9, 682e9, 5e12):
        tile = next_probe_tile(rate_macs_per_s=rate, n=N, d=D, v=V)
        assert tile & (tile - 1) == 0


def test_next_probe_tile_floors_at_32_even_when_still_over_budget() -> None:
    # a rate this low projects >1s even at the floor tile — must clamp, not raise or
    # shrink below 32 (check_budget, not this sizing heuristic, is what refuses launches).
    assert next_probe_tile(rate_macs_per_s=10e9, n=1_000_000, d=8192, v=200_000) == 32


def test_next_probe_tile_caps_at_v_when_v_below_8192() -> None:
    # v=1000 < 8192: the cap is min(8192, v), then rounded down to a power of two (512),
    # not 8192, even though the rate would otherwise permit the full 8192 tile.
    assert next_probe_tile(rate_macs_per_s=1e15, n=64, d=64, v=1000) == 512


def test_sustain_reps_ceil_arithmetic() -> None:
    assert sustain_reps(per_dispatch_s=0.2) == 4       # ceil(0.75 / 0.2) == 4


def test_sustain_reps_floors_at_1() -> None:
    assert sustain_reps(per_dispatch_s=2.0) == 1       # ceil(0.75 / 2.0) == 1 already


def test_sustain_reps_caps_at_8() -> None:
    assert sustain_reps(per_dispatch_s=0.001) == 8      # ceil(750) clamped to the cap


def test_sustain_reps_guards_against_a_nonpositive_measurement() -> None:
    # a zero or negative timing (clock-resolution underflow) can't drive ceil division —
    # fall back to the cap rather than raising or looping forever.
    assert sustain_reps(per_dispatch_s=0.0) == 8


def test_calibrate_lands_high_tile_despite_underestimating_micro_probe() -> None:
    """False-refusal regression: a tile-128 micro-probe reading a rate 2.7x below the
    production-tile rate must not anchor the calibration. The ramp has to climb through
    intermediate tiles and land its final (median-of-3) measurement at the tile the
    production dispatch actually runs at — matching the real regression (a lone tile-128
    probe under-reported the rate badly enough to refuse a dispatch that in fact runs at
    ~0.20 s against the 1.0 s budget)."""
    high_rate = 1_363.4e9   # the measured production-tile rate this shape converges to
    low_rate = high_rate / 2.7
    calls: list[int] = []

    def fake_measure(tile: int) -> float:
        calls.append(tile)
        rate = low_rate if tile <= 128 else high_rate
        return N * tile * D / rate

    result = calibrate(measure=fake_measure, n=N, d=D, v=V, start_tile=128, max_stages=3)

    assert result == pytest.approx(high_rate, rel=1e-6)
    assert calls[:3] == [128, 4096, 8192]   # ramp climbs 128 -> 4096 -> 8192 in 3 stages
    # sustain (4 reps at this shape's final per-dispatch time) + median-of-3, all at 8192
    assert calls[3:] == [8192] * (len(calls) - 3)
    assert set(calls[3:]) == {8192}
    assert calls[-1] == 8192


def test_calibrate_stops_ramping_once_the_tile_stops_growing() -> None:
    """A kernel whose rate already projects the same (cap-limited) tile stage over stage
    must not burn every one of max_stages measuring dispatches — it converges in the
    first stage and moves straight to the sustain + median-of-3 phase."""
    n, d, v = 512, 256, 4096
    calls: list[int] = []

    def fake_measure(tile: int) -> float:
        calls.append(tile)
        return n * tile * d / 1e12   # constant rate: the projected tile never grows

    calibrate(measure=fake_measure, n=n, d=d, v=v, start_tile=4096, max_stages=5)

    expected_reps = sustain_reps(per_dispatch_s=n * 4096 * d / 1e12)
    assert calls == [4096] * (1 + expected_reps + 3)   # 1 ramp stage + sustain + median


def test_calibrate_returns_raw_unhalved_rate() -> None:
    """The returned rate is the raw measured rate, not SAFETY_FACTOR-halved — halving is
    the caller's (calibrated_rate's) responsibility, applied once to the final result."""
    rate = 4e11

    def fake_measure(tile: int) -> float:
        return N * tile * D / rate

    result = calibrate(measure=fake_measure, n=N, d=D, v=V, start_tile=8192, max_stages=1)
    assert result == pytest.approx(rate, rel=1e-6)
