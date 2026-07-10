"""Pure-arithmetic tests for the T9 dK/dV chaining planner and the backward rate ramp reuse
in `mlx_train_perf.attention.kernel.launch`. No GPU here (mirroring
`test_kernel_launch_calibration.py` / the forward ramp-planner tests in
`test_attention_kernel_fwd.py`): `plan_dkv_dispatches` is pure integer arithmetic over the
shared budget helpers, and the ramp is exercised through a FAKE `measure` closure.

`plan_dkv_dispatches` is the dK/dV analogue of the forward/dQ inline query-range split, but
factored out because the chained accumulator makes the split worth testing in isolation: it
must tile [0, n) exactly, keep each range's projected MACs inside the per-dispatch budget, and
raise `LaunchBudgetError` (the 0.1.0 refusal contract) when even one minimal range -- or the
whole pass's total wall -- over-budgets, never launch an over-budget dispatch into the
uncatchable GPU watchdog.
"""
import itertools

import pytest

from mlx_train_perf.attention.kernel.launch import (
    MAX_DISPATCH_SECONDS,
    MAX_TOTAL_SECONDS,
    _bwd_dkv_macs_per_row,
    _calibrate_fwd,
    _fwd_macs_per_row,
    plan_dkv_dispatches,
)
from mlx_train_perf.errors import LaunchBudgetError


def test_bwd_dkv_macs_per_row_counts_four_d_per_pair() -> None:
    """Per query row, each of the n keys costs 4*D MACs -- an s = q.k dot (D), a dp = dO.v dot
    (D), a dV accumulate (D) and a dK accumulate (D) -- across every (batch, q-head). The
    conservative full-key count over-estimates causal (the safe direction: it splits MORE)."""
    assert _bwd_dkv_macs_per_row(n=8192, d=128, b=1, hq=32) == 4 * 128 * 8192 * 1 * 32


def _ranges_tile_exactly(ranges: list[tuple[int, int]], n: int) -> bool:
    """The ranges partition [0, n): start at 0, end at n, contiguous, no gaps or overlaps."""
    if not ranges:
        return False
    if ranges[0][0] != 0 or ranges[-1][1] != n:
        return False
    return all(hi == nxt_lo for (_, hi), (nxt_lo, _) in itertools.pairwise(ranges))


def test_plan_dkv_dispatches_respects_budget() -> None:
    # Pick a rate that forces a multi-range split whose TOTAL stays within MAX_TOTAL_SECONDS
    # (rate >= n*per_row / MAX_TOTAL) yet whose per-dispatch fills the per-dispatch budget.
    n, d, b, hq = 4096, 64, 1, 8
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = n * per_row / 1.5  # total ~1.5 s < MAX_TOTAL_SECONDS
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate)

    assert len(ranges) >= 2                       # the rate genuinely forces a split
    assert _ranges_tile_exactly(ranges, n)
    budget_macs = MAX_DISPATCH_SECONDS * rate
    for q_lo, q_hi in ranges:
        assert (q_hi - q_lo) * per_row <= budget_macs   # no range over the per-dispatch budget


def test_plan_dkv_dispatches_handles_n_smaller_than_one_tile() -> None:
    # A rate generous enough that the whole [0, n) fits one dispatch -> exactly one range.
    n, d, b, hq = 64, 64, 1, 4
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=1e18)
    assert ranges == [(0, n)]


def test_plan_dkv_dispatches_raises_when_minimum_range_over_budgets() -> None:
    # A rate so low that even a single query row projects over the per-dispatch budget: the
    # planner refuses (LaunchBudgetError -- the 0.1.0 refusal contract), never floors-and-ships.
    n, d, b, hq = 16384, 128, 1, 32
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    slow_rate = per_row / 2.0  # 1 row projects to 2 s > the 0.5 s per-dispatch bound
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=slow_rate)


def test_plan_dkv_dispatches_caps_total_projected_time() -> None:
    # Per-dispatch fine (one row ~0.01 s), but the whole pass over all rows blows the total
    # cap -- many individually-safe dispatches must not sum to unbounded wall.
    n, d, b, hq = 16384, 128, 1, 32
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rate = per_row * 100.0  # 1 row ~0.01 s (ok), all 16384 rows ~164 s (> MAX_TOTAL_SECONDS)
    assert n * per_row / rate > MAX_TOTAL_SECONDS
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate)


def test_calibrate_ramp_uses_the_supplied_macs_per_row() -> None:
    """The backward rate reuses the forward's ramp/canary machinery via an additive
    `macs_per_row` parameter (design point 4): passing `_bwd_dkv_macs_per_row` must make the
    ramp derive its rate from the dK/dV 4*D-per-pair cost, NOT the forward's 2*D default. A
    single-stage ramp (start_n == n) skips the canary and returns raw macs/time -- with the
    dkv cost that is 4*d*n*b*hq*n / time, exactly 2x the forward cost, so a param that never
    threaded through would return half this value."""
    n, d, b, hq = 8192, 64, 1, 4
    dispatch_s = 0.25

    def fake_measure(rows: int, keys: int) -> float:  # noqa: ARG001 -- constant-time probe
        return dispatch_s

    result = _calibrate_fwd(
        measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=n, max_stages=1,
        macs_per_row=_bwd_dkv_macs_per_row,
    )
    expected = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq) * n / dispatch_s
    assert result == pytest.approx(expected, rel=1e-9)
    # ... and it is distinguishable from the forward 2*D default (which would be half):
    assert result == pytest.approx(2.0 * _fwd_macs_per_row(n=n, d=d, b=b, hq=hq) * n / dispatch_s,
                                   rel=1e-9)
