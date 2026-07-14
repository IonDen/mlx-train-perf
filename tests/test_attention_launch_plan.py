"""Pure-arithmetic tests for the attention launch planner and the backward rate ramp in
`mlx_train_perf.attention.kernel.launch`. No GPU here (mirroring
`test_kernel_launch_calibration.py`): the planner is pure integer arithmetic and the ramp is
exercised through a FAKE `measure` closure.

0.3.0 semantics (the launch-budget evidence study, backlog 0025): the macOS interactivity
kill applies to an individual COMMAND BUFFER, and mlx 0.32.0 commits a buffer at >50 ops or
>50 M unique input+output ELEMENTS — so the planner models buffer composition and caps each
MODELED BUFFER's summed projected time at `MAX_DISPATCH_SECONDS`. Costs are exact-causal
(triangle) MAC counts, not the old full-rectangle upper bound. There is NO chain-total cap:
a chain whose every dispatch owns its buffer may project unbounded total wall. Chains whose
dispatches could PACK into one buffer (small unique-element footprints, no guaranteed commit)
are capped at one buffer budget summed — tighter than the retired 2.0 s `MAX_TOTAL_SECONDS`,
and honest about the mechanism.
"""
import itertools

import pytest

from mlx_train_perf.attention.kernel.launch import (
    MAX_DISPATCH_SECONDS,
    _bwd_dkv_macs_per_row,
    _calibrate_fwd,
    _fwd_macs_per_row,
    causal_pairs,
    plan_dkv_dispatches,
    range_macs,
)
from mlx_train_perf.errors import LaunchBudgetError

# The flagship-class dK/dV shape: unique input elements (~118 M) exceed the mlx commit
# threshold (51 << 20 ≈ 53.5 M), so every dispatch owns its command buffer.
_OWN = {"n": 8192, "d": 128, "b": 1, "hq": 32, "hkv": 8}
# A small shape whose dK/dV unique input elements (~2.1 M) can never trigger a commit:
# consecutive dispatches must be assumed to pack into ONE buffer.
_PACKED = {"n": 2048, "d": 64, "b": 1, "hq": 4, "hkv": 2}


def _pair(d: int, b: int, hq: int, c: int = 4) -> int:
    return c * d * b * hq


def _dkv_total_macs(n: int, d: int, b: int, hq: int) -> int:
    return causal_pairs(0, n) * _pair(d, b, hq)


def test_bwd_dkv_macs_per_row_counts_four_d_per_pair() -> None:
    """Per query row, each of the n keys costs 4*D MACs -- an s = q.k dot (D), a dp = dO.v dot
    (D), a dV accumulate (D) and a dK accumulate (D) -- across every (batch, q-head). This
    stays the NOMINAL (full-rectangle) per-row cost used by probe sizing; budget projection
    now uses the exact-causal `range_macs` instead."""
    assert _bwd_dkv_macs_per_row(n=8192, d=128, b=1, hq=32) == 4 * 128 * 8192 * 1 * 32


def test_causal_pairs_matches_brute_force() -> None:
    """Closed form sum_{i=r0}^{r1-1}(i+1) == the literal sum, across an exhaustive small grid."""
    for n in (1, 2, 7, 33, 96):
        for r0 in range(n):
            for r1 in range(r0 + 1, n + 1):
                assert causal_pairs(r0, r1) == sum(i + 1 for i in range(r0, r1))


def test_range_macs_causal_charges_the_triangle() -> None:
    pc = _pair(64, 1, 4)
    assert range_macs(r0=0, r1=8, n=32, pair_cost=pc, causal=False) == 8 * 32 * pc
    assert range_macs(r0=0, r1=32, n=32, pair_cost=pc, causal=True) == (32 * 33 // 2) * pc
    # A tail range under causal costs nearly the full rectangle; an early range far less.
    tail = range_macs(r0=24, r1=32, n=32, pair_cost=pc, causal=True)
    head = range_macs(r0=0, r1=8, n=32, pair_cost=pc, causal=True)
    assert head < tail <= 8 * 32 * pc


def _ranges_tile_exactly(ranges: list[tuple[int, int]], n: int) -> bool:
    """The ranges partition [0, n): start at 0, end at n, contiguous, no gaps or overlaps."""
    if not ranges:
        return False
    if ranges[0][0] != 0 or ranges[-1][1] != n:
        return False
    return all(hi == nxt_lo for (_, hi), (nxt_lo, _) in itertools.pairwise(ranges))


def test_plan_dkv_dispatches_respects_the_per_buffer_budget() -> None:
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    pair = _pair(d, b, hq)
    rate = _dkv_total_macs(n, d, b, hq) / 1.5   # whole causal pass ~1.5 s -> a forced split
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate)

    assert len(ranges) >= 2
    assert _ranges_tile_exactly(ranges, n)
    for q_lo, q_hi in ranges:
        t = range_macs(r0=q_lo, r1=q_hi, n=n, pair_cost=pair, causal=True) / rate
        assert t <= MAX_DISPATCH_SECONDS + 1e-9


def test_plan_dkv_dispatches_chain_total_unbounded_when_dispatches_own_their_buffers() -> None:
    """THE 0025 headline behavior: a flagship-class chain whose every dispatch owns its
    command buffer may project an UNBOUNDED total (here ~8 s -- the class the retired
    2.0 s chain cap refused), as long as every dispatch stays inside the per-buffer budget."""
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    pair = _pair(d, b, hq)
    rate = _dkv_total_macs(n, d, b, hq) / 8.0
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate)
    assert _ranges_tile_exactly(ranges, n)
    assert len(ranges) >= 16   # ~8 s of causal work in <= 0.5 s own-buffer dispatches
    for q_lo, q_hi in ranges:
        t = range_macs(r0=q_lo, r1=q_hi, n=n, pair_cost=pair, causal=True) / rate
        assert t <= MAX_DISPATCH_SECONDS + 1e-9


def test_plan_dkv_dispatches_causal_ranges_narrow_toward_the_tail() -> None:
    """Under exact-causal costing, later query rows scan more keys, so each own-buffer range
    gets NARROWER as r0 grows (the final range is a remainder and may be anything smaller)."""
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    rate = _dkv_total_macs(n, d, b, hq) / 8.0
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate)
    widths = [hi - lo for lo, hi in ranges]
    assert all(w1 >= w2 for w1, w2 in itertools.pairwise(widths[:-1]))
    assert widths[0] > widths[-2]


def test_plan_dkv_dispatches_packed_buffer_caps_summed_time() -> None:
    """At a shape whose unique input elements can never trigger an mlx commit, consecutive
    dispatches must be assumed to share ONE command buffer -- so the whole chain is capped at
    one per-buffer budget, and a total that fits emits a SINGLE range (a chain that fits
    0.5 s fits one dispatch)."""
    n, d, b, hq, hkv = (
        _PACKED["n"], _PACKED["d"], _PACKED["b"], _PACKED["hq"], _PACKED["hkv"],
    )
    total = _dkv_total_macs(n, d, b, hq)
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=total / 1.5)
    assert plan_dkv_dispatches(
        n=n, d=d, b=b, hq=hq, hkv=hkv, rate=total / 0.4
    ) == [(0, n)]


def test_plan_dkv_dispatches_handles_n_smaller_than_one_tile() -> None:
    ranges = plan_dkv_dispatches(n=64, d=64, b=1, hq=4, hkv=2, rate=1e18)
    assert ranges == [(0, 64)]


def test_plan_dkv_dispatches_raises_when_minimum_range_over_budgets() -> None:
    """A rate so low that a single TAIL query row projects over the per-buffer budget: the
    planner refuses (LaunchBudgetError -- the 0.1.0 refusal contract), never floors-and-ships."""
    n, d, b, hq, hkv = 16384, 128, 1, 32, 8
    pair = _pair(d, b, hq)
    slow_rate = (n * pair) / 2.0   # the last row alone projects ~2 s > the 0.5 s bound
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=slow_rate)


def test_plan_dkv_dispatches_default_alignment_byte_unchanged() -> None:
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    rate = _dkv_total_macs(n, d, b, hq) / 4.0
    default = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate)
    explicit = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate, block_align=1)
    assert default == explicit


def test_plan_dkv_dispatches_block_aligned_to_32() -> None:
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    rate = _dkv_total_macs(n, d, b, hq) / 4.0
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate, block_align=32)
    assert _ranges_tile_exactly(ranges, n)
    assert len(ranges) >= 4
    for q_lo, q_hi in ranges:
        assert q_lo % 32 == 0
        assert q_hi % 32 == 0 or q_hi == n


def test_plan_dkv_dispatches_block_align_never_widens_past_budget() -> None:
    """Alignment rounds a range DOWN to the block multiple (floored at one block) -- an
    aligned plan's ranges stay within the same per-buffer budget as the unaligned plan's."""
    n, d, b, hq, hkv = _OWN["n"], _OWN["d"], _OWN["b"], _OWN["hq"], _OWN["hkv"]
    pair = _pair(d, b, hq)
    rate = _dkv_total_macs(n, d, b, hq) / 6.0
    ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate, block_align=32)
    for q_lo, q_hi in ranges:
        t = range_macs(r0=q_lo, r1=q_hi, n=n, pair_cost=pair, causal=True) / rate
        assert t <= MAX_DISPATCH_SECONDS + 1e-9


def test_plan_dkv_dispatches_block_align_refusal_unchanged() -> None:
    """When even one minimal block over-budgets at the tail, the aligned planner still raises
    rather than shipping an over-budget range into the uncatchable GPU watchdog."""
    n, d, b, hq, hkv = 16384, 128, 1, 32, 8
    pair = _pair(d, b, hq)
    rate = (32 * n * pair) / 4.0   # one 32-row tail block projects ~4 s >> 0.5 s
    with pytest.raises(LaunchBudgetError):
        plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate, block_align=32)


def test_calibrate_ramp_uses_the_supplied_macs_per_row() -> None:
    """The backward rate reuses the forward's ramp/canary machinery via an additive
    `macs_per_row` parameter: passing `_bwd_dkv_macs_per_row` must make the ramp derive its
    rate from the dK/dV 4*D-per-pair cost, NOT the forward's 2*D default. causal=False keeps
    the full-rectangle credit: a single-stage ramp (start_n == n) returns raw macs/time."""
    n, d, b, hq = 8192, 64, 1, 4
    dispatch_s = 0.25

    def fake_measure(rows: int, keys: int) -> float:  # noqa: ARG001 -- constant-time probe
        return dispatch_s

    result = _calibrate_fwd(
        measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=n, max_stages=1,
        macs_per_row=_bwd_dkv_macs_per_row, causal=False,
    )
    expected = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq) * n / dispatch_s
    assert result == pytest.approx(expected, rel=1e-9)
    assert result == pytest.approx(
        2.0 * _fwd_macs_per_row(n=n, d=d, b=b, hq=hq) * n / dispatch_s, rel=1e-9,
    )


def test_calibrate_ramp_credits_the_causal_triangle() -> None:
    """With causal=True the ramp credits the probe's EXACT causal work (the self-shaped
    [0, n) probe does n(n+1)/2 pairs, not n^2) -- so the returned MAC/s means 'causal-true
    MACs per second', consistent with the planner's projection accounting."""
    n, d, b, hq = 8192, 64, 1, 4
    dispatch_s = 0.25

    def fake_measure(rows: int, keys: int) -> float:  # noqa: ARG001
        return dispatch_s

    result = _calibrate_fwd(
        measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=n, max_stages=1,
        macs_per_row=_bwd_dkv_macs_per_row, causal=True,
    )
    pair = _bwd_dkv_macs_per_row(n=1, d=d, b=b, hq=hq)
    expected = causal_pairs(0, n) * pair / dispatch_s
    assert result == pytest.approx(expected, rel=1e-9)
