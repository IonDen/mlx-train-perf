import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm")
from recurrent_parity_cases import CASES, build_case, metrics

from mlx_train_perf.recurrent.ops import chunked_gated_delta
from mlx_train_perf.recurrent.reference import sequential_gated_delta

# Measured in this tree (2026-08-30) by running every CASES entry through both
# chunked_gated_delta and sequential_gated_delta and taking the per-regime,
# per-component (max_abs, rel_fro) worst across all cases sharing that regime
# label -- not tied to a single case. Pinned at measured-worst x ~2. Each block
# below states (a) the in-tree measured worst, (b) the expected noise ceiling
# (fp32 accumulation ~1e-6; bf16 resolution 2^-8 ~= 3.9e-3), and (c) the
# contrast with the brief's Step 7 "expected worsts from the probe" figures.
FWD_PINS: dict[str, dict[str, tuple[float, float]]] = {
    # y: max_abs=3.755093e-06, rel_fro=8.417309e-07 (both blocked_solve).
    #    fp32 accumulation floor ~1e-6 -- matches the brief's 3.755e-6/8.4e-7.
    # state: max_abs=2.145767e-06 (single_partial_chunk), rel_fro=2.359078e-06
    #    (single_partial_chunk) -- also within the ~1e-6 fp32 floor (partial
    #    chunk exercises the pad path, not new numerical behavior).
    "benign": {"y": (8e-6, 2e-6), "state": (5e-6, 5e-6)},
    # y: max_abs=7.152557e-07, rel_fro=2.284231e-07 (blocked_repeated).
    #    fp32 accumulation floor ~1e-6 -- matches the brief's "~7e-7".
    # state: max_abs=2.216548e-07 (blocked_repeated), rel_fro=4.118089e-07
    #    (blocked_repeated) -- same floor; repeated keys don't move it.
    "adversarial": {"y": (1.5e-6, 5e-7), "state": (5e-7, 1e-6)},
    # y: max_abs=2.066731e-03, rel_fro=2.639293e-04 (collinear_beta_high).
    #    Above the flat fp32 floor by design -- this is the ill-conditioned
    #    collinear+beta->1 corner the sub-blocked solve exists to bound, not a
    #    flat 1e-6 regime -- matches the brief's 2.07e-3/2.6e-4.
    # state: max_abs=1.446098e-03, rel_fro=9.676865e-04 (collinear_beta_high).
    "blowup_corner": {"y": (5e-3, 6e-4), "state": (3e-3, 2e-3)},
    # y: max_abs=4.882812e-04, rel_fro=4.481647e-06 (bf16_g_one). bf16
    #    resolution 2^-8 ~= 3.9e-3 -- comfortably under that ceiling, but
    #    HIGHER than the brief's quoted "1.5e-5/5.5e-7": that figure matches
    #    only the bf16_inputs case (measured here too, 1.525879e-05/
    #    5.538143e-07); bf16_g_one is the true worst in this regime because
    #    g=1-1e-7 bf16-rounds to exactly 1.0 (bf16 ULP near 1.0 is 2^-7), so
    #    there is no decay to damp accumulated bf16 rounding noise across
    #    T=96 -- still well inside the bf16 noise ceiling, not surprising.
    # state: max_abs=4.172325e-07 (bf16_inputs), rel_fro=2.329307e-07
    #    (bf16_inputs) -- state stays fp32 across chunks (never cast to bf16
    #    mid-recurrence), so it tracks the fp32 floor, not the bf16 one.
    "bf16": {"y": (1e-3, 1e-5), "state": (1e-6, 5e-7)},
    # y: max_abs=7.152557e-06, rel_fro=7.788288e-07 (representative_0p8b).
    #    fp32 accumulation floor ~1e-6, mildly elevated by T=512 and the real
    #    Hk=Hv=16, Dk=Dv=128 geometry -- matches the brief's 7.2e-6/7.8e-7.
    # state: max_abs=7.301569e-07, rel_fro=1.295902e-06 (representative_0p8b).
    "representative": {"y": (1.5e-5, 2e-6), "state": (1.5e-6, 3e-6)},
    # y: max_abs=3.906250e-03 (one bf16 ULP at this magnitude), rel_fro=
    #    4.653467e-05 (bf16_repr_T512). bf16 resolution 2^-8 ~= 3.9e-3 --
    #    max_abs sits AT the ceiling (a single representable-value step), as
    #    the brief anticipated ("~3.9e-3, one bf16 ulp"/4.7e-5); T=2048
    #    (bf16_repr_T2048) measured the same max_abs and a slightly lower
    #    rel_fro (4.425594e-05), so T512 is the regime worst on both axes.
    # state: max_abs=9.387732e-07 (bf16_repr_T512), rel_fro=1.495402e-06
    #    (bf16_repr_T512) -- fp32-floor again (state never casts to bf16).
    "bf16_representative": {"y": (8e-3, 1e-4), "state": (2e-6, 3e-6)},
}


def assert_under_pin(name, got, pin):
    assert pin[0] >= 0.0, f"{name}: pin not yet measured"
    assert got[0] <= pin[0], f"{name}: max_abs {got[0]:.3e} > pin {pin[0]:.3e}"
    assert got[1] <= pin[1], f"{name}: rel_fro {got[1]:.3e} > pin {pin[1]:.3e}"


@pytest.mark.parametrize(("case_id", "regime", "kw"), CASES,
                         ids=[c[0] for c in CASES])
def test_chunked_fwd_matches_sequential(case_id, regime, kw):
    # Catches per regime: broken blocked-solve branch, mis-clamped log-domain
    # decay, fp32 blow-up handling, wrong GQA broadcast, chunk-padding bugs,
    # dropped state carry.
    d = build_case(**kw)
    y_ref, st_ref = sequential_gated_delta()(
        d["q"], d["k"], d["v"], d["g"], d["beta"], d["state"])
    y, st = chunked_gated_delta(d["q"], d["k"], d["v"], d["g"], d["beta"],
                                d["state"], chunk_size=d["chunk_size"])
    mx.eval(y, st, y_ref, st_ref)
    assert_under_pin(f"{case_id}/y", metrics(y, y_ref), FWD_PINS[regime]["y"])
    assert_under_pin(f"{case_id}/state", metrics(st, st_ref), FWD_PINS[regime]["state"])
