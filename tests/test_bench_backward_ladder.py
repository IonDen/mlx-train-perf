"""Pure-logic tests for `scripts/bench_backward_ladder.py` (Task 16b step 4
prerequisite). `scripts/` has no `__init__.py` (matches `bench_quant_thresholds.py`'s
existing convention), so the module is loaded by path rather than via a package import.

Only the GPU-free helpers are covered here: `macs_for_condition` (pure MAC accounting)
and `ramp_tile_and_rate` (the ramp logic, exercised through a FAKE `measure` closure --
same convention as `test_kernel_launch_calibration.py`'s coverage of `launch.calibrate`).
The real Metal-backed condition runners are exercised end-to-end via `--tiny` on the main
session, not unit-tested here.

`ramp_tile_and_rate`'s `measure(tile)` closures in this script (see
`_dhidden_measure`/`_dw_measure`) time a FULL-VOCAB CHAIN against production-sized `w`
buffers -- `ceil(v / tile)` dispatches, not one -- so every `fake_measure` below returns
a CHAIN-elapsed value: the total time to cover all `v` columns, independent of how many
dispatches that took. This is NOT the same convention `test_kernel_launch_calibration.py`
uses for `launch.calibrate` itself, whose OWN `measure` closures (via `calibrated_rate`)
size the probe `w` to `tile` so the "chain" is exactly one dispatch -- see
`ramp_tile_and_rate`'s docstring for the full explanation, including the production
false-refusal this distinction fixes.
"""
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_backward_ladder import (  # noqa: E402 -- import must follow the sys.path insert
    CONDITIONS,
    macs_for_condition,
    ramp_tile_and_rate,
    script_sha,
)

from mlx_train_perf.core.kernel.launch import SAFETY_FACTOR  # noqa: E402

# Same regression shape `test_kernel_launch_calibration.py` uses for `launch.calibrate`.
N, D, V = 8192, 4096, 151936


# ---------------------------------------------------------------------------
# macs_for_condition: pure MAC-accounting table
# ---------------------------------------------------------------------------


def test_macs_for_condition_covers_every_declared_condition() -> None:
    # every name bench_backward_ladder.CONDITIONS actually dispatches must have an entry
    for condition in CONDITIONS:
        assert macs_for_condition(condition, n=8, v=16, d=4) > 0


def test_macs_for_condition_staged_frozen_is_2x_base() -> None:
    assert macs_for_condition("staged_vjp_frozen", n=8, v=16, d=4) == 2 * 8 * 16 * 4


def test_macs_for_condition_staged_trainable_is_3x_base() -> None:
    assert macs_for_condition("staged_vjp_trainable", n=8, v=16, d=4) == 3 * 8 * 16 * 4


def test_macs_for_condition_kernel_dhidden_and_dw_are_2x_base() -> None:
    base = 8 * 16 * 4
    assert macs_for_condition("kernel_dhidden_v0", n=8, v=16, d=4) == 2 * base
    assert macs_for_condition("kernel_dw_v0", n=8, v=16, d=4) == 2 * base


def test_macs_for_condition_combined_is_4x_base_not_2x() -> None:
    # NOT the ~2x a future FUSED kernel would achieve -- the two v0 kernels are separate
    # and each pays its own full logit-regeneration cost (2x + 2x, no shared recompute).
    assert macs_for_condition("kernel_backward_v0_combined", n=8, v=16, d=4) == 4 * 8 * 16 * 4


def test_macs_for_condition_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        macs_for_condition("not_a_real_condition", n=8, v=16, d=4)


# ---------------------------------------------------------------------------
# ramp_tile_and_rate: same ramp SHAPE as launch.calibrate, but CHAIN-MAC accounting
# (n*v*d/elapsed, not n*tile*d/elapsed -- see the module + function docstrings for why).
# Also reports the tile the ramp converged to (calibrate() itself only returns the rate)
# -- the backward kernels' production dispatch must run at the SAME tile the ramp
# measured, since their v0 rate is UNKNOWN and must never be assumed from the forward's.
# ---------------------------------------------------------------------------


def test_ramp_tile_and_rate_lands_the_high_tile_and_safety_margined_rate() -> None:
    # Same false-refusal-regression SHAPE as test_kernel_launch_calibration.py's
    # test_calibrate_lands_high_tile_despite_underestimating_micro_probe: a tile-128
    # micro-probe under-reports the rate by 2.7x; the ramp must still climb to and
    # measure the true high-rate tile (8192), not anchor on the bad early sample.
    # CHAIN semantics: fake_measure returns the elapsed time to cover ALL v columns
    # (n*v*d/effective_rate), independent of tile -- not n*tile*d/effective_rate.
    high_rate = 1_363.4e9
    low_rate = high_rate / 2.7
    calls: list[int] = []

    def fake_measure(tile: int) -> float:
        calls.append(tile)
        rate = low_rate if tile <= 128 else high_rate
        return N * V * D / rate

    tile, rate = ramp_tile_and_rate(fake_measure, n=N, d=D, v=V, start_tile=128)

    assert tile == 8192
    assert rate == pytest.approx(SAFETY_FACTOR * high_rate, rel=1e-6)
    assert calls[:3] == [128, 4096, 8192]


def test_ramp_tile_and_rate_stops_ramping_once_the_tile_stops_growing() -> None:
    n, d, v = 512, 256, 4096
    calls: list[int] = []

    def fake_measure(tile: int) -> float:
        calls.append(tile)
        # constant, TILE-INDEPENDENT chain elapsed: the achievable rate never changes
        # with tile, so the projected tile never grows past the starting one.
        return n * v * d / 1e12

    tile, rate = ramp_tile_and_rate(fake_measure, n=n, d=d, v=v, start_tile=4096)

    assert tile == 4096  # converged tile, NOT the shape's own v/8192 cap
    assert rate == pytest.approx(SAFETY_FACTOR * 1e12, rel=1e-6)
    assert set(calls) == {4096}  # every stage measured the same (already-converged) tile


def test_ramp_tile_and_rate_applies_safety_factor_exactly_once() -> None:
    rate = 4e11

    def fake_measure(tile: int) -> float:  # noqa: ARG001 -- chain elapsed is tile-independent
        return N * V * D / rate

    _, reported_rate = ramp_tile_and_rate(fake_measure, n=N, d=D, v=V, start_tile=8192,
                                          max_stages=1)
    assert reported_rate == pytest.approx(SAFETY_FACTOR * rate, rel=1e-6)


def test_ramp_tile_and_rate_chain_semantics_regression() -> None:
    """Regression for the exact production bug (confirmed by reading the code and
    reproducing against the recorded refusal artifacts): `measure(tile)` closures in
    this script time a FULL-VOCAB CHAIN against production-sized `w` (`ceil(v / tile)`
    dispatches), so the chain's total MAC count is ALWAYS n*v*d regardless of tile --
    dividing by n*tile*d instead (the bug) under-reports the rate by ~v/tile (~1,187x at
    a tile-128 probe against V=151936), collapsing the ramp and producing a "0 G MAC/s"
    fiction that made `check_budget` refuse every kernel condition with a projected
    ~1290 s/dispatch. This fake measure models a kernel whose TRUE, tile-independent
    achievable rate is HIGH_RATE (a full chain covering all V columns takes the same
    total time regardless of how it happened to be tiled)."""
    high_rate = 1_363.4e9
    start_tile = 128

    def fake_measure(tile: int) -> float:  # noqa: ARG001 -- chain elapsed is tile-independent
        return N * V * D / high_rate

    # Document the exact symptom: the OLD (buggy) n*tile*d/elapsed formula at the probe
    # tile would have derived a rate ~V/tile times too low.
    buggy_elapsed = fake_measure(start_tile)
    buggy_rate_at_probe_tile = N * start_tile * D / buggy_elapsed
    assert buggy_rate_at_probe_tile == pytest.approx(high_rate * start_tile / V, rel=1e-9)
    assert high_rate / buggy_rate_at_probe_tile == pytest.approx(V / start_tile, rel=1e-9)

    # The FIXED arithmetic recovers the true, tile-independent rate and lets the ramp
    # climb all the way to the production tile -- it does not collapse/get stuck at the
    # tiny probe tile the way the buggy formula (derived above) would.
    tile, rate = ramp_tile_and_rate(fake_measure, n=N, d=D, v=V, start_tile=start_tile)
    assert tile == 8192
    assert rate == pytest.approx(SAFETY_FACTOR * high_rate, rel=1e-6)


# ---------------------------------------------------------------------------
# script_sha: identity-provenance helper (same convention as
# ground_truth_atomic_outputs.py's own script_sha -- run_identity's code_sha excludes
# ad hoc bench scripts, so this is what actually invalidates a stale artifact when THIS
# script's own logic changes).
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)
