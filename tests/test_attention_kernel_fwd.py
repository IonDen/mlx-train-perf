"""0.2.0 T5 -- flash-attention FORWARD Metal kernel v0 (O + L), correctness only.

Two lanes in one file (the `test_kernel_guard.py` mixed-file convention):

- Pure-arithmetic tests (source templating, range planning, budget math) stay in the
  DEFAULT lane -- no `@pytest.mark.metal`, no GPU.
- Every test that launches the kernel carries a PER-TEST `@pytest.mark.metal` -- never a
  module-level mark, so the arithmetic tests above run without `--run-metal`.

Parity is checked against BOTH oracles (T3 `math_attention` and `mx.fast.scaled_dot_
product_attention`) for O, and against the reference `L` (`flash_attention_reference`).
Tolerances are measured-first: the per-case worst is printed, and the committed bound is
the smallest honest value over the grid (see the pin comments).
"""
import math
from typing import cast

import mlx.core as mx
import pytest

from mlx_train_perf.attention import api
from mlx_train_perf.attention.api import flash_attention, resolve_attention_impl
from mlx_train_perf.attention.kernel import launch as fwd_launch
from mlx_train_perf.attention.kernel.dispatch import select_fwd_tile
from mlx_train_perf.attention.kernel.launch import (
    _PROBE_N_HARD_CAP,
    TileShape,
    _calibrate_fwd,
    _fwd_macs_per_row,
    _next_probe_n,
    _start_probe_n,
    calibrated_fwd_rate,
    check_fwd_budget,
    launch_flash_fwd,
)
from mlx_train_perf.attention.kernel.source import (
    _FWD_MMA_D_SLAB,
    build_fwd_mma_source,
    build_fwd_source,
)
from mlx_train_perf.attention.reference import (
    flash_attention_reference,
    kv_head_for,
    math_attention,
)
from mlx_train_perf.errors import LaunchBudgetError

GENEROUS_RATE = 1e13  # parity shapes are microscopic; the budget check must never trip


# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating + budget math (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_fwd_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_fwd_source(hd, causal=True)
        assert f"float qreg[{hd}];" in s
        assert f"float acc[{hd}];" in s
        assert f"dd < {hd}" in s
        assert "HEAD_DIM" not in s  # every sentinel substituted (lossless)


def test_build_fwd_source_causal_keep_comparison() -> None:
    s = build_fwd_source(64, causal=True)
    assert "kk <= row" in s
    assert "kk >= row" not in s


def test_build_fwd_source_noncausal_keeps_all_keys() -> None:
    s = build_fwd_source(64, causal=False)
    assert "kk <= row" not in s
    assert "kk >= row" not in s
    assert "bool keep = (true);" in s


def test_build_fwd_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-mask arm: the causal comparison is flipped to the WRONG
    # triangle so a parity run FAILS -- proving the suite can fail.
    s = build_fwd_source(64, causal=True, flip_causal=True)
    assert "kk >= row" in s
    assert "kk <= row" not in s


def test_build_fwd_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_fwd_source(hd, causal=True)


def test_build_fwd_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_fwd_source(64, causal=False, flip_causal=True)


# ---------------------------------------------------------------------------------------
# Rung 1: 4x4 simdgroup-matrix (MMA) forward source templating (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_fwd_mma_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_fwd_mma_source(hd, causal=True)
        assert f"slab0 < {hd}" in s               # the D-slab OUTER loop spans the full head dim
        assert "C_o[4][" in s                     # register-resident O accumulator tiles (rung 2)
        assert "o_acc" not in s                   # NO threadgroup O accumulator (rung 2)
        assert "threadgroup float" not in s       # rung 2 holds O + softmax in registers, no TG mem
        assert "HEAD_DIM" not in s                # every sentinel substituted (lossless)
        assert "KEEP_CMP" not in s
        assert "KV_LIMIT" not in s
        assert "D_SLAB" not in s


def test_build_fwd_mma_source_slabs_the_d_dimension() -> None:
    # The register-resident C_o accumulator is held for D_SLAB columns at a time; the D-slab
    # OUTER loop steps by D_SLAB and C_o carries RT=4 row-tiles x (D_SLAB/8) col-tiles.
    slab = _FWD_MMA_D_SLAB
    tiles = slab // 8
    for hd in (64, 96, 128):
        assert hd % slab == 0, f"D_SLAB {slab} must divide every supported head dim ({hd})"
        s = build_fwd_mma_source(hd, causal=True)
        assert f"C_o[4][{tiles}]" in s            # RT=4 x (D_SLAB/8) register O tiles
        assert f"slab0 += {slab}" in s            # D-slab OUTER loop steps by D_SLAB


def test_build_fwd_mma_source_d_slab_override_for_the_regpressure_probe() -> None:
    # An explicit d_slab overrides the default -- the regpressure probe sweeps candidate
    # slab widths at each head dim to justify the shipped `_FWD_MMA_D_SLAB`.
    s16 = build_fwd_mma_source(64, causal=True, d_slab=16)
    assert "C_o[4][2]" in s16                     # 16 / 8 == 2 col-tiles
    assert "slab0 += 16" in s16
    s64 = build_fwd_mma_source(128, causal=True, d_slab=64)
    assert "C_o[4][8]" in s64                     # 64 / 8 == 8 col-tiles
    assert "slab0 += 64" in s64


def test_build_fwd_mma_source_rejects_indivisible_d_slab() -> None:
    with pytest.raises(ValueError, match="d_slab"):
        build_fwd_mma_source(96, causal=True, d_slab=64)   # 96 % 64 != 0


def test_build_fwd_mma_source_causal_keep_and_loop_bound() -> None:
    s = build_fwd_mma_source(64, causal=True)
    assert "kk <= row" in s
    assert "kk >= row" not in s
    # causal KV-block loop bound: blocks fully above the diagonal are never entered
    assert "metal::min(n, r0 + block_base + 32u)" in s


def test_build_fwd_mma_source_noncausal_keeps_all_keys_and_scans_all() -> None:
    s = build_fwd_mma_source(64, causal=False)
    assert "kk <= row" not in s
    assert "kk >= row" not in s
    assert "(kk < kb1) && (true)" in s            # boundary check kept, causal predicate open
    assert "uint kv_limit = n;" in s              # non-causal scans every KV block


def test_build_fwd_mma_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-mask arm: the causal predicate is flipped to the WRONG triangle
    # (the KV-block loop bound stays causal -- at the tiny flip-test N it covers all keys).
    s = build_fwd_mma_source(64, causal=True, flip_causal=True)
    assert "kk >= row" in s
    assert "kk <= row" not in s


def test_build_fwd_mma_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_fwd_mma_source(hd, causal=True)


def test_build_fwd_mma_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_fwd_mma_source(64, causal=False, flip_causal=True)


def test_check_fwd_budget_passes_a_cheap_dispatch() -> None:
    # tiny per-row cost at a real rate: nowhere near the 1 s / 60 s bounds
    check_fwd_budget(n=64, d=64, b=1, hq=4, rows=64, rate=1e12)


def test_check_fwd_budget_refuses_when_one_row_over_budgets() -> None:
    # a single query row projected over 1 s at this (deliberately low) rate -> refuse
    per_row = _fwd_macs_per_row(n=16384, d=128, b=1, hq=32)
    slow_rate = per_row / 2.0  # 1 row projects to 2 s > the 1 s per-dispatch bound
    with pytest.raises(LaunchBudgetError):
        check_fwd_budget(n=16384, d=128, b=1, hq=32, rows=1, rate=slow_rate)


def test_check_fwd_budget_refuses_over_total_budget() -> None:
    # per-dispatch fine (1 row), but the whole forward over all rows blows the 60 s total
    per_row = _fwd_macs_per_row(n=16384, d=128, b=1, hq=32)
    rate = per_row * 100.0  # 1 row ~ 0.01 s (ok), all 16384 rows ~ 164 s (> 60 s)
    with pytest.raises(LaunchBudgetError):
        check_fwd_budget(n=16384, d=128, b=1, hq=32, rows=1, rate=rate)


# ---------------------------------------------------------------------------------------
# review-round FINDING 1: N-aware forward calibration ramp planner (DEFAULT lane, no GPU).
# `_start_probe_n` / `_next_probe_n` / `_calibrate_fwd` size the probe key-count instead of
# the pre-fix fixed `_PROBE_N=128` -- exercised here through FAKE `measure` closures, the
# same convention tests/test_kernel_launch_calibration.py uses for the CE kernel's ramp.
# ---------------------------------------------------------------------------------------


def test_start_probe_n_measures_a_small_context_directly() -> None:
    # a real n smaller than the old fixed probe is measured AT n, never padded up to it
    assert _start_probe_n(16) == 16
    assert _start_probe_n(61) == 61


def test_start_probe_n_floors_at_the_old_fixed_probe_for_a_large_context() -> None:
    assert _start_probe_n(128) == 128
    assert _start_probe_n(8192) == 128


def test_next_probe_n_caps_at_n_when_n_is_small() -> None:
    # n=61 < the 128 floor: a generous rate must not push the probe past the real n
    assert _next_probe_n(rate_macs_per_s=1e18, n=61, d=64, b=1, hq=4) == 61


def test_next_probe_n_reaches_n_when_rate_is_generous() -> None:
    # n=8192 is itself a power of two: a generous rate lets the ramp reach it directly
    assert _next_probe_n(rate_macs_per_s=1e15, n=8192, d=64, b=1, hq=4) == 8192


def test_next_probe_n_never_exceeds_the_hard_cap() -> None:
    np_ = _next_probe_n(rate_macs_per_s=1e18, n=100_000, d=64, b=1, hq=4)
    assert np_ <= _PROBE_N_HARD_CAP


def test_next_probe_n_shrinks_when_the_full_cap_over_budgets() -> None:
    # quadratic cost: at n=8192, d=128, b=1, hq=32 the full 8192-key probe projects ~5 s at
    # this rate -- well over the 1 s budget -- so the sizing heuristic must shrink it.
    full_cap_macs = _fwd_macs_per_row(n=8192, d=128, b=1, hq=32) * 8192
    slow_rate = full_cap_macs / 5.0
    np_ = _next_probe_n(rate_macs_per_s=slow_rate, n=8192, d=128, b=1, hq=32)
    assert np_ < 8192


def test_next_probe_n_is_a_power_of_two_above_the_floor() -> None:
    for rate in (1e9, 5e10, 1e12, 1e15):
        np_ = _next_probe_n(rate_macs_per_s=rate, n=8192, d=64, b=1, hq=4)
        assert np_ & (np_ - 1) == 0


def test_calibrate_fwd_ramps_past_the_old_fixed_probe_despite_underestimating_it() -> None:
    """Finding-1 regression: a 128-key micro-probe reading a rate well below the
    production-shape rate must not anchor the calibration -- the ramp climbs through
    intermediate probe sizes and lands its final (median-of-3) measurement at the size a
    flagship dispatch actually runs at, matching the CE kernel's own false-refusal
    regression test (test_kernel_launch_calibration.py
    ::test_calibrate_lands_high_tile_despite_underestimating_micro_probe)."""
    n, d, b, hq = 8192, 64, 1, 4
    high_rate = 200e9
    low_rate = high_rate / 2.7
    calls: list[int] = []

    def fake_measure(rows: int, keys: int) -> float:
        calls.append((rows, keys))
        rate = low_rate if keys <= 128 else high_rate
        macs = _fwd_macs_per_row(n=keys, d=d, b=b, hq=hq) * rows
        return macs / rate

    result = _calibrate_fwd(measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=128, max_stages=3)

    assert calls[0] == (128, 128)                  # the ramp STARTS at the old fixed probe
    assert calls[-1][1] > 128                       # ... but does not STOP there
    assert result == pytest.approx(high_rate, rel=1e-6)


def test_calibrate_fwd_stops_ramping_once_the_probe_stops_growing() -> None:
    """A kernel whose rate already projects the same (cap-limited) probe size stage over
    stage must not burn every one of max_stages measuring dispatches -- it converges in
    the first stage and moves straight to the sustain + median-of-3 phase."""
    n, d, b, hq = 512, 32, 1, 2
    calls: list[tuple[int, int]] = []

    def fake_measure(rows: int, keys: int) -> float:
        calls.append((rows, keys))
        macs = _fwd_macs_per_row(n=keys, d=d, b=b, hq=hq) * rows
        return macs / 1e12   # constant rate: the projected probe size never grows

    _calibrate_fwd(measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=512, max_stages=5)

    per_dispatch_s = (_fwd_macs_per_row(n=512, d=d, b=b, hq=hq) * 512) / 1e12
    # 1 ramp stage + sustain + median -- and NO canary: the ramp already measured the
    # full n x n shape, which is harsher than any production range dispatch.
    expected_reps = fwd_launch._sustain_reps(per_dispatch_s=per_dispatch_s)
    assert calls == [(512, 512)] * (1 + expected_reps + 3)


def test_calibrate_fwd_returns_raw_unhalved_rate() -> None:
    """The returned rate is the raw measured rate, not SAFETY_FACTOR-halved -- halving is
    the caller's (calibrated_fwd_rate's) responsibility, applied once to the final
    result -- matching the CE kernel's own `calibrate` contract."""
    n, d, b, hq = 8192, 64, 1, 4
    rate = 4e11

    def fake_measure(rows: int, keys: int) -> float:
        macs = _fwd_macs_per_row(n=keys, d=d, b=b, hq=hq) * rows
        return macs / rate

    result = _calibrate_fwd(measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=8192, max_stages=1)
    assert result == pytest.approx(rate, rel=1e-6)


def test_calibrate_fwd_final_canary_measures_small_rows_against_full_n_keys() -> None:
    """Interactivity-kill regression (T6 rung 0, macOS 26 / mlx 0.32.0): the ramp's probes
    size THEIR OWN working set (rows == keys == np_), which stays cache-friendlier than a
    production dispatch (few rows x ALL n keys x every head) -- the measured kill: a
    projected-1.0s flagship dispatch sized from the SAFETY-halved ramp rate (106 G MAC/s)
    still ran long enough for macOS to kill the command buffer
    (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). The fix: calibration ends
    with a CANARY -- one small-row-range measurement against FULL-n keys (the true
    DRAM-bound production working set), and the returned rate derives from the canary
    alone. Fake rates here make the canary regime 3x slower than the ramp regime; a
    calibration that anchors on ramp-regime measurements returns ~ramp_rate and fails."""
    # n ABOVE _PROBE_N_HARD_CAP: the ramp legitimately reaches the cap and stops short of
    # n, which is exactly the regime the canary exists for (ramp == n skips it by design).
    n, d, b, hq = 16384, 64, 1, 4
    ramp_rate = 300e9
    dram_rate = ramp_rate / 3.0
    calls: list[tuple[int, int]] = []

    def fake_measure(rows: int, keys: int) -> float:
        calls.append((rows, keys))
        rate = dram_rate if keys == n and rows < keys else ramp_rate
        macs = _fwd_macs_per_row(n=keys, d=d, b=b, hq=hq) * rows
        return macs / rate

    result = _calibrate_fwd(measure=fake_measure, n=n, d=d, b=b, hq=hq, start_n=128, max_stages=3)

    last_rows, last_keys = calls[-1]
    assert last_keys == n            # the canary sees the FULL key working set
    assert last_rows < n             # ... at a small, budget-safe row count
    assert result == pytest.approx(dram_rate, rel=1e-6)  # rate derives from the canary


def test_canary_rows_respects_its_budget() -> None:
    """Pure arithmetic: the canary row count keeps its projected dispatch inside the canary
    budget at the SAFETY-halved ramp rate, floors at 1 row, and never exceeds n."""
    n, d, b, hq = 8192, 128, 1, 32
    raw_ramp_rate = 212e9
    rows = fwd_launch._canary_rows(raw_ramp_rate=raw_ramp_rate, n=n, d=d, b=b, hq=hq)
    per_row = _fwd_macs_per_row(n=n, d=d, b=b, hq=hq)
    projected_s = rows * per_row / (fwd_launch.SAFETY_FACTOR * raw_ramp_rate)
    assert 1 <= rows <= n
    assert projected_s <= fwd_launch._CANARY_BUDGET_S * 1.01
    # and a rate so low that even one row over-budgets still yields the 1-row floor:
    assert fwd_launch._canary_rows(raw_ramp_rate=1e6, n=n, d=d, b=b, hq=hq) == 1


def test_max_dispatch_budget_is_pinned_to_the_interactivity_kill_evidence() -> None:
    """Safety pin: macOS killed a command buffer whose projected time was 1.0s at a
    2x-optimistic rate (~2-4s real) with ErrorImpactingInteractivity -- a SOFTER, earlier
    kill than the assumed 5-10s GPU watchdog. The CE kernel's shipped dispatches run
    ~0.5s real and have never been killed (0.1.0 T13: zero watchdog events). 0.5s
    projected (x SAFETY margin 2 => ~0.25s real) sits in the proven-safe class. Raising
    this constant requires NEW kill-threshold evidence, not convenience."""
    assert fwd_launch.MAX_DISPATCH_SECONDS == 0.5
    # The second half of the same evidence: 35 honest ~0.25s-real dispatches PACKED INTO
    # ONE EVAL (~8.7s cumulative GPU work) were ALSO killed -- the OS kill is per command
    # buffer / cumulative, not per dispatch, and MLX packs consecutive custom dispatches.
    # The CE kernel's ~2.2s evals have never been killed, so ~2s is the proven-safe class
    # for the TOTAL cap too (a flagship v0 scalar forward now refuses honestly instead of
    # dying; MMA-class rates fit a full forward well inside it).
    assert fwd_launch.MAX_TOTAL_SECONDS == 2.0


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal).
# ---------------------------------------------------------------------------------------

# head-config x N cases; the flagship (32/8) pattern only at N=64 to bound cost.
_HEAD_N_CASES = [
    (4, 4, 64), (4, 4, 61), (4, 4, 257),   # MHA
    (4, 2, 64), (4, 2, 61), (4, 2, 257),   # GQA
    (32, 8, 64),                            # flagship group_size-4 pattern
]

# Measured worsts over the whole grid, PER VARIANT (mlx 0.32.0, M1 Max, seed=7).
#
# scalar (the 0.2.0-T5 v0 body -- pins UNCHANGED):
#   O vs math_attention / vs sdpa: fp32 9.537e-07, bf16 7.812e-03
#   L vs reference (always fp32):  fp32 9.537e-07, bf16 1.431e-06
# The kernel accumulates fp32 in-register for both input dtypes, so fp32 O/L diffs are pure
# reduction-order noise (~1e-6). bf16 O is written back in bf16, so its worst is one bf16
# ULP at an O value near 1-2 (2^-7 ~= 7.8e-3) -- the same single rounding the reference's
# o32.astype(bf16) does, differing by at most a ULP from the fp32 accumulation-order gap.
# Pins are the smallest honest bound over THIS grid (fp32 ~2.1x, bf16 O ~1.5x margin, same
# measure-first convention as tests/test_kernel_parity.py). A future case landing between a
# pin and 2 bf16 ULP is not a regression -- widen toward 2 ULP with a note.
#
# mma (rung-2 register-resident P@V MMA O-path with D-slabbing): the score reduction
# reassociates (fp32 QK^T MMA + a second fp32 P@V MMA for O, per-slab recompute, vs scalar's
# per-key recurrence). MEASURED worsts over THIS grid (mlx 0.32.0, M1 Max, seed=7): O fp32
# 9.537e-07, O bf16 7.812e-03, L fp32 9.537e-07, L bf16 9.537e-07 -- UNCHANGED from rung 1;
# the P@V MMA + recompute reassociation stays in the SAME ~1e-6 fp32 / one-bf16-ULP class as
# scalar (L bf16 is actually tighter than scalar's 1.431e-06). So the honest mma pins equal the
# scalar pins here -- measured separately per the rung contract, NOT widened.
_TOL_O = {
    "scalar": {mx.float32: 2e-6, mx.bfloat16: 1.2e-2},
    "mma": {mx.float32: 2e-6, mx.bfloat16: 1.2e-2},
}
_TOL_L = {
    "scalar": {mx.float32: 2e-6, mx.bfloat16: 5e-6},
    "mma": {mx.float32: 2e-6, mx.bfloat16: 5e-6},
}


def _rand_qkv(
    *, b: int, hq: int, hkv: int, n: int, d: int, dtype: mx.Dtype, seed: int = 7
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, n, d)).astype(dtype)
    k = mx.random.normal((b, hkv, n, d)).astype(dtype)
    v = mx.random.normal((b, hkv, n, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
@pytest.mark.parametrize(("hq", "hkv", "n"), _HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_fwd_parity_vs_both_oracles_and_reference_lse(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype, variant: str
) -> None:
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype)

    o_k, l_k = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=GENEROUS_RATE,
    )
    o_math = math_attention(q, k, v, scale=scale, causal=True)
    o_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    # L oracle: logsumexp of the masked scaled scores, fp32 (reference contract)
    _, l_ref = _reference_o_l(q, k, v, scale=scale)
    mx.eval(o_k, l_k, o_math, o_sdpa, l_ref)

    f = mx.float32
    d_math = mx.abs(o_k.astype(f) - o_math.astype(f)).max().item()
    d_sdpa = mx.abs(o_k.astype(f) - o_sdpa.astype(f)).max().item()
    d_l = mx.abs(l_k - l_ref).max().item()
    print(
        f"[{variant} {['fp32','bf16'][dtype==mx.bfloat16]} b{batch} {hq}/{hkv} n{n} "
        f"d{head_dim}] O-math={d_math:.3e} O-sdpa={d_sdpa:.3e} L={d_l:.3e}"
    )

    assert d_math < _TOL_O[variant][dtype], f"O vs math {d_math}"
    assert d_sdpa < _TOL_O[variant][dtype], f"O vs sdpa {d_sdpa}"
    assert d_l < _TOL_L[variant][dtype], f"L vs reference {d_l}"


def _reference_o_l(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float
) -> tuple[mx.array, mx.array]:
    return flash_attention_reference(q, k, v, scale=scale, causal=True)


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_fwd_row0_attends_only_itself(variant: str) -> None:
    """Causal row 0 attends only key 0: O[.,.,0]==V[.,kv,0], L[.,.,0]==scale*(q0.k0)."""
    b, hq, hkv, n, d = 2, 4, 2, 8, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=1)

    o_k, l_k = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=GENEROUS_RATE,
    )
    group = hq // hkv
    mx.eval(o_k, l_k)
    for bb in range(b):
        for h in range(hq):
            kvh = kv_head_for(h, group)
            o0 = o_k[bb, h, 0]
            expect_o = v[bb, kvh, 0]
            expect_l = (q[bb, h, 0].astype(mx.float32) * k[bb, kvh, 0].astype(mx.float32)
                        ).sum().item() * scale
            assert mx.abs(o0 - expect_o).max().item() < 1e-5
            assert abs(l_k[bb, h, 0].item() - expect_l) < 1e-4


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_fwd_bitwise_deterministic_across_runs(variant: str) -> None:
    """No atomics by design (mma holds S/O in registers + simd_shuffle reductions, each lane
    owning disjoint output rows/cols) -> bit-identical O and L across repeated runs. Lock it."""
    q, k, v = _rand_qkv(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=2)
    scale = 1.0 / math.sqrt(64)
    o0, l0 = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=GENEROUS_RATE,
    )
    mx.eval(o0, l0)
    for _ in range(4):
        o, lse = launch_flash_fwd(
            q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
            rate_macs_per_s=GENEROUS_RATE,
        )
        mx.eval(o, lse)
        assert mx.array_equal(o, o0).item()
        assert mx.array_equal(lse, l0).item()


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_fwd_split_matches_single_dispatch(variant: str) -> None:
    """Query-range multi-dispatch writes DISJOINT O/L rows; the reassembled result must
    be bit-identical to a single dispatch. This is the outer-grid offset guard (a wrong
    r0 offset corrupts a chunk) -- run at batch>1 and an N that is not a block multiple.
    For mma, a dispatch boundary that is not 32-aligned (~80 rows) exercises per-row
    independence: a row's O/L depend only on its own absolute position, never its block."""
    b, hq, hkv, n, d = 2, 4, 2, 257, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=3)

    single_o, single_l = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=GENEROUS_RATE,   # one dispatch over all rows
    )
    # Force ~80 rows/dispatch (4 disjoint dispatches over n=257) via a low rate, keeping
    # the projected per-dispatch inside MAX_DISPATCH_SECONDS (0.5 s) AND the projected
    # total (257/160 = 1.6 s) inside the 2.0 s per-eval cap.
    per_row = _fwd_macs_per_row(n=n, d=d, b=b, hq=hq)
    split_rate = per_row * 160.0
    split_o, split_l = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=split_rate,
    )
    mx.eval(single_o, single_l, split_o, split_l)
    assert mx.array_equal(single_o, split_o).item()
    assert mx.array_equal(single_l, split_l).item()


@pytest.mark.metal
@pytest.mark.parametrize("variant", ["scalar", "mma"])
def test_fwd_wrong_mask_perturbation_fails_parity(variant: str) -> None:
    """Deliberate wrong-mask: build the kernel with the causal comparison flipped to the
    WRONG triangle. Its O/L must DIVERGE from the causal reference -- if this ever matched,
    the parity tests above could not detect a real mask bug (the suite would be
    unfalsifiable). For mma the flip perturbs the in-tile KEEP_CMP predicate; the diagonal
    block is masked before the row max, so a flipped predicate genuinely changes the output."""
    b, hq, hkv, n, d = 2, 4, 2, 16, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=4)

    o_wrong, l_wrong = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32, variant=variant),
        rate_macs_per_s=GENEROUS_RATE, _flip_causal=True,
    )
    o_ref, l_ref = _reference_o_l(q, k, v, scale=scale)
    mx.eval(o_wrong, l_wrong, o_ref, l_ref)

    d_o = mx.abs(o_wrong.astype(mx.float32) - o_ref.astype(mx.float32)).max().item()
    d_l = mx.abs(l_wrong - l_ref).max().item()
    assert d_o > 1e-2 or d_l > 1e-2, (
        f"flipped-mask kernel matched the causal reference (O={d_o:.3e}, L={d_l:.3e}) -- "
        "the parity suite cannot detect a mask bug"
    )


# ---------------------------------------------------------------------------------------
# Step 5: impl="kernel" wired into the api -- Metal FORWARD, staged pure-MLX BACKWARD.
# ---------------------------------------------------------------------------------------


@pytest.mark.metal
def test_impl_kernel_resolves_now_that_forward_is_built() -> None:
    """A fully-supported config resolves to 'kernel' (T5), no longer refusing 'not built
    yet'. Metal-marked: resolution needs `mx.metal.is_available()` true."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=64, dtype=mx.float32, seed=6)
    assert resolve_attention_impl(q, k, v, impl="kernel", causal=True) == "kernel"
    assert resolve_attention_impl(q, k, v, impl="auto", causal=True) == "kernel"


@pytest.mark.metal
def test_kernel_forward_reference_backward_grads_match_oracle() -> None:
    """impl='kernel' routes the FORWARD through the Metal kernel; the BACKWARD is the
    staged pure-MLX vjp. Grads of sum(flash_attention(impl='kernel')) must match the
    `math_attention` autodiff oracle, and the hand vjp must actually fire (a dropped
    registration would autodiff through the kernel forward -- but the kernel forward has no
    autodiff, so it would error; the counter proves the vjp path, not just non-error)."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=9)
    api.VJP_CALLS.clear()

    def kernel_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="kernel").sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True).sum()

    g_kernel = mx.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_kernel, *g_math)

    worst = max(
        float(mx.abs(gk - gm).max().item())
        for gk, gm in zip(g_kernel, g_math, strict=True)
    )
    # Kernel forward O/L feed the same pure-MLX backward the oracle uses; worst |grad diff|
    # is fp32 reduction-order noise (measured 3.815e-06, seed=9) -> pin 2e-5 (~5.2x, same
    # convention as test_attention_api.py::test_flash_attention_grads_match_autodiff_oracle).
    assert worst < 2e-5, f"worst |grad diff|={worst}"
    assert api.VJP_CALLS.get("flash_attention", 0) > 0


# ---------------------------------------------------------------------------------------
# review round: calibration-path fixes (Metal integration; the pure ramp-planner logic
# above already RED/GREEN-covers Finding 1's arithmetic).
# ---------------------------------------------------------------------------------------


@pytest.mark.metal
def test_calibrated_fwd_rate_ramps_the_probe_past_the_old_fixed_128(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding-1 integration regression: before the fix, `calibrated_fwd_rate` measured
    EVERY call at a fixed 128-key probe, regardless of the caller's real n. Wraps
    `launch_flash_fwd` to record the key-count of every probe dispatch during a flagship
    (n=8192) calibration -- the largest recorded probe must exceed the old fixed 128, i.e.
    the calibration actually ramps toward the caller's real n instead of staying pinned to
    a cache-resident micro-shape."""
    monkeypatch.setattr(fwd_launch, "_FWD_RATE_CACHE", {})
    seen_n: list[int] = []
    real_dispatch = fwd_launch._dispatch_range

    def recording_dispatch(*args: object, **kwargs: object) -> object:
        k = args[2]
        seen_n.append(k.shape[2])  # type: ignore[union-attr]
        return real_dispatch(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fwd_launch, "_dispatch_range", recording_dispatch)
    calibrated_fwd_rate(
        head_dim=64, dtype=mx.float32, b=1, hq=4, hkv=4, n=8192, causal=True,
        tile=TileShape(),
    )

    assert max(seen_n) > 128, f"probe never grew past the old fixed 128: {sorted(set(seen_n))}"


@pytest.mark.metal
def test_calibrated_fwd_rate_leaves_the_global_rng_stream_untouched() -> None:
    """Finding-2 regression: `calibrated_fwd_rate` must not call `mx.random.seed` -- a
    global-stream mutation would desync any `mx.random` draw a caller makes after the
    first (uncached) kernel invocation, which can fire mid-training. Seeds the GLOBAL
    stream, snapshots the draw immediately AFTER an uncached calibration call, and
    compares it against a CONTROL sequence that never calibrates -- the two must match."""
    fwd_launch._FWD_RATE_CACHE.clear()
    mx.random.seed(123)
    control_before = mx.random.normal((4,))
    control_after = mx.random.normal((4,))
    mx.eval(control_before, control_after)

    fwd_launch._FWD_RATE_CACHE.clear()
    mx.random.seed(123)
    calibrated = mx.random.normal((4,))  # same draw as control_before, pre-calibration
    mx.eval(calibrated)
    calibrated_fwd_rate(
        head_dim=64, dtype=mx.float32, b=1, hq=4, hkv=4, n=16, causal=True,
        tile=TileShape(),
    )
    next_draw = mx.random.normal((4,))   # must match control_after: untouched global stream
    mx.eval(next_draw)

    assert mx.array_equal(calibrated, control_before).item()
    assert mx.array_equal(next_draw, control_after).item(), (
        "calibrated_fwd_rate perturbed the global mx.random stream"
    )


# ---------------------------------------------------------------------------------------
# T6 rung 3: dispatch-table wiring -- `d_slab` cache-key threading, calibration probing the
# SELECTED variant (not a hardcoded scalar), and the api-level dispatch-table integration.
# ---------------------------------------------------------------------------------------


def test_fwd_kernel_cache_key_separates_by_d_slab(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_fwd_kernel`'s cache key must include `d_slab`: two mma kernels built at the same
    (head_dim, causal, flip_causal) but different `d_slab` are DIFFERENT compiled sources
    (different D_SLAB/D_SLAB_TILES baked in) and must never collapse onto one cache entry.
    Monkeypatches `mx.fast.metal_kernel` itself (the real GPU-touching constructor) so this
    stays in the DEFAULT lane -- only `_fwd_kernel`'s own caching/source-building logic is
    under test, not the Metal JIT. Explicitly clears/restores `_fwd_kernel`'s module-level
    `functools.cache` so this never leaks fake kernel objects into a later `--run-metal` test
    reusing the same (head_dim, causal, flip_causal, variant, d_slab) key."""
    fwd_launch._fwd_kernel.cache_clear()
    built: list[str] = []  # compiled source per REAL build (cache misses only)

    def fake_metal_kernel(
        *, name: str, input_names: list[str], output_names: list[str], source: str,  # noqa: ARG001
    ) -> object:
        built.append(source)
        return object()

    monkeypatch.setattr(mx.fast, "metal_kernel", fake_metal_kernel)
    try:
        k_slab128 = fwd_launch._fwd_kernel(128, True, False, "mma", 128)
        k_slab64 = fwd_launch._fwd_kernel(128, True, False, "mma", 64)
        k_slab128_again = fwd_launch._fwd_kernel(128, True, False, "mma", 128)

        assert k_slab128 is not k_slab64
        assert k_slab128 is k_slab128_again          # same key -> cache hit, no rebuild
        assert len(built) == 2                        # only 2 REAL builds happened
        assert built[0] != built[1]                    # the two sources actually differ
        assert "C_o[4][16]" in built[0]                 # 128 / 8 col-tiles
        assert "C_o[4][8]" in built[1]                  # 64 / 8 col-tiles
    finally:
        fwd_launch._fwd_kernel.cache_clear()


def test_calibrated_fwd_rate_probes_the_selected_variant_and_d_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding (T6 rung 3): before this fix, `calibrated_fwd_rate`'s `measure()` always built
    the SCALAR kernel regardless of the caller's `tile` -- rating the scalar kernel while the
    launcher dispatches mma sizes the query-row split from the WRONG rate. Spies on
    `_fwd_kernel` (the kernel-construction seam) with a fake that fabricates zero-cost output
    arrays instead of touching Metal, so this stays in the DEFAULT lane, and asserts the
    recorded (variant, d_slab) matches the `tile` passed in -- probe what you rate."""
    monkeypatch.setattr(fwd_launch, "_FWD_RATE_CACHE", {})
    calls: list[tuple[str, int | None]] = []

    def fake_kernel(
        *, inputs: list[mx.array], template: list[tuple[str, mx.Dtype]],  # noqa: ARG001
        grid: tuple[int, int, int], threadgroup: tuple[int, int, int],  # noqa: ARG001
        output_shapes: list[tuple[int, ...]], output_dtypes: list[mx.Dtype],
    ) -> list[mx.array]:
        return [
            mx.zeros(shape, dtype=dtype)
            for shape, dtype in zip(output_shapes, output_dtypes, strict=True)
        ]

    def fake_fwd_kernel(
        head_dim: int, causal: bool, flip_causal: bool, variant: str,  # noqa: ARG001
        d_slab: int | None,
    ) -> object:
        calls.append((variant, d_slab))
        return fake_kernel

    monkeypatch.setattr(fwd_launch, "_fwd_kernel", fake_fwd_kernel)
    tile = TileShape(variant="mma", d_slab=64)
    fwd_launch.calibrated_fwd_rate(
        head_dim=64, dtype=mx.float32, b=1, hq=4, hkv=4, n=256, causal=True, tile=tile,
    )

    assert calls == [("mma", 64)], (
        f"calibration built {calls}, but the caller selected variant='mma' d_slab=64 -- "
        "measure() must probe the SAME kernel the launcher will dispatch"
    )


@pytest.mark.metal
def test_flash_attention_kernel_path_uses_the_dispatch_table_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T6 rung 3: `api.py`'s kernel path must call `select_fwd_tile(n, head_dim)` instead of
    a hardcoded `TileShape()` -- spies on `launch_flash_fwd` (the seam `api.py` calls into)
    to record the ACTUAL tile the api passed, and checks it against the table's own
    selection for this shape. Pre-fix this reads `TileShape()` (scalar) regardless of shape;
    post-fix it must equal `select_fwd_tile`'s own answer."""
    b, hq, hkv, n, d = 1, 4, 2, 61, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=22)

    seen_tiles: list[TileShape] = []
    real_launch = api.launch_flash_fwd

    def recording_launch(*args: object, **kwargs: object) -> object:
        seen_tiles.append(cast(TileShape, kwargs["tile"]))
        return real_launch(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api, "launch_flash_fwd", recording_launch)
    flash_attention(q, k, v, scale=scale, causal=True, impl="kernel")

    assert seen_tiles == [select_fwd_tile(n, d)]


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), [(4, 4, 61), (4, 2, 257), (32, 8, 64)])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_impl_kernel_matches_oracle_through_dispatch_table_selection(
    hq: int, hkv: int, n: int, head_dim: int,
) -> None:
    """impl='kernel' at small production-shaped configs now routes through
    `select_fwd_tile` (T6 rung 3) instead of a hardcoded scalar `TileShape()`. Looks up the
    ACTUAL table selection for this shape so the test tracks whatever the table picks (mma
    or scalar, measured or provisional) rather than assuming one variant, and confirms it
    still produces a correct forward -- the dispatch-table wiring changing WHICH kernel runs
    must not change correctness."""
    selected = select_fwd_tile(n, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(b=1, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=mx.float32, seed=23)

    o_kernel = flash_attention(q, k, v, scale=scale, causal=True, impl="kernel")
    o_math = math_attention(q, k, v, scale=scale, causal=True)
    mx.eval(o_kernel, o_math)

    diff = mx.abs(o_kernel.astype(mx.float32) - o_math.astype(mx.float32)).max().item()
    assert diff < _TOL_O[selected.variant][mx.float32], (
        f"[{selected.variant} d_slab={selected.d_slab} provisional={selected.provisional}] "
        f"head_dim={head_dim} n={n}: O diff {diff}"
    )
