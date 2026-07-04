from collections.abc import Callable

import mlx.core as mx
import pytest

from mlx_train_perf.core.kernel import launch
from mlx_train_perf.core.kernel.launch import (
    calibrated_rate,
    check_budget,
    forward,
    probe_tile_for,
)
from mlx_train_perf.errors import LaunchBudgetError


def test_probe_tile_safe_at_floor_production_shape() -> None:
    # n=8192, d=4096, floor 10 G MAC/s, half-budget: tile <= 0.5*10e9/(8192*4096) ~ 149 -> 128
    assert probe_tile_for(n=8192, d=4096) == 128


def test_probe_tile_never_below_32_or_above_8192() -> None:
    assert probe_tile_for(n=64, d=32) == 8192          # tiny shapes: full default tile is safe
    assert probe_tile_for(n=1_000_000, d=8192) == 32   # floor-clamped


def test_check_budget_passes_measured_production_rate() -> None:
    # measured v2e rate 2,423.7 G MAC/s -> 0.11 s/dispatch at production shape: fine
    check_budget(n=8192, d=4096, v=151936, tile=8192, rate_macs_per_s=2.42e12)


def test_check_budget_refuses_spike_default_rate() -> None:
    # the spike's static 14 G MAC/s default projects ~19.6 s/dispatch -> must refuse
    with pytest.raises(LaunchBudgetError):
        check_budget(n=8192, d=4096, v=151936, tile=8192, rate_macs_per_s=14e9)


@pytest.mark.metal
def test_calibrated_rate_is_plausible_and_cached() -> None:
    r1 = calibrated_rate(row_tiles=4, dtype=mx.bfloat16, n=2048, d=256, v=4096)
    r2 = calibrated_rate(row_tiles=4, dtype=mx.bfloat16, n=2048, d=256, v=4096)
    assert r1 == r2                       # cache hit — no second probe dispatch
    assert 1e9 < r1 < 1e14                # a real rate, post safety factor


@pytest.mark.metal
def test_forward_with_calibrated_rate_runs_small_shape() -> None:
    mx.random.seed(5)
    hidden = mx.random.normal((64, 32)).astype(mx.bfloat16)
    w = mx.random.normal((1000, 32)).astype(mx.bfloat16)
    t = mx.random.randint(0, 1000, (64,))
    rate = calibrated_rate(row_tiles=4, dtype=mx.bfloat16, n=64, d=32, v=1000)
    lse, _tgt = forward(hidden, w, t, row_tiles=4, tile=1000, rate_macs_per_s=rate)
    assert bool(mx.isfinite(lse).all().item())


def _count_calibrate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], int]:
    """Monkeypatches launch.calibrate with a call-counting wrapper (same signature) and
    resets the module rate cache so the count reflects only THIS test's calls — a test
    that could pass on a warm cache from another test's key would be unable to fail."""
    monkeypatch.setattr(launch, "_RATE_CACHE", {})
    calls = 0
    real_calibrate = launch.calibrate

    def counting_calibrate(
        *, measure: Callable[[int], float], n: int, d: int, v: int, start_tile: int,
        max_stages: int = 3,
    ) -> float:
        nonlocal calls
        calls += 1
        return real_calibrate(
            measure=measure, n=n, d=d, v=v, start_tile=start_tile, max_stages=max_stages,
        )

    monkeypatch.setattr(launch, "calibrate", counting_calibrate)
    return lambda: calls


@pytest.mark.metal
def test_calibrated_rate_dense_positive_and_cache_hit_skips_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = _count_calibrate_calls(monkeypatch)
    r1 = launch.calibrated_rate(row_tiles=4, dtype=mx.bfloat16, n=64, d=64, v=1024)
    r2 = launch.calibrated_rate(row_tiles=4, dtype=mx.bfloat16, n=64, d=64, v=1024)
    assert r1 > 0
    assert r1 == r2                 # cache hit returns the identical value
    assert call_count() == 1        # ... without a second calibration run


@pytest.mark.metal
def test_calibrated_rate_quantized_positive_and_cache_hit_skips_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = _count_calibrate_calls(monkeypatch)
    r1 = launch.calibrated_rate_quantized(row_tiles=4, dtype=mx.bfloat16, n=64, d=64, v=1024)
    r2 = launch.calibrated_rate_quantized(row_tiles=4, dtype=mx.bfloat16, n=64, d=64, v=1024)
    assert r1 > 0
    assert r1 == r2                 # cache hit returns the identical value
    assert call_count() == 1        # ... without a second calibration run
