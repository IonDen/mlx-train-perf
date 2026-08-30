import mlx.core as mx
import pytest
from conftest import needs_long_context_room

pytest.importorskip("mlx_lm")
from recurrent_parity_cases import BWD_CASES, build_case, losses_pair, metrics
from test_recurrent_fwd_parity import assert_under_pin

from mlx_train_perf.recurrent.ops import chunked_gated_delta

# Measured in this tree (2026-08-30) by running every BWD_CASES entry through
# mx.grad of losses_pair's loss_seq/loss_chk (argnums=(0,1,2,3,4): q,k,v,g,
# beta), evaluating each arm's grads before building the next, and taking the
# per-regime, per-input (max_abs, rel_fro) worst across all cases sharing that
# regime label. Pinned at measured-worst x ~2. dg is consistently the worst
# component in every regime (~3-5x its siblings) -- backward through the
# log-domain g_cumlog/exp chain is more fp32-sensitive than the linear q/k/v/
# beta paths, matching the "KNOWN divergence" the op's own docstring already
# flags for the g clamp; this was investigated (forward parity for the same
# cases stays at the ~1e-6 floor, and the backward/forward amplification
# ratio is consistent -- ~30-60x -- across every case, not just the g-heavy
# ones), so it is pinned, not treated as a surprise.
BWD_PINS: dict[str, dict[str, tuple[float, float]]] = {
    # T=96 cases (multi_chunk, carried_state) alone: worst max_abs=3.649294e-05
    # (dg, multi_chunk), rel_fro=1.005649e-06 (dg, multi_chunk) -- matches the
    # brief's "7.2e-6..3.6e-5 max_abs .. ~2.5e-7 rel_fro at T=96" for the
    # non-dg components (dq/dk/dv/dbeta rel_fro sit at 2.1e-7..3.2e-7); dg
    # rel_fro running ~4x the brief's floor is the log-gate effect above.
    # blocked_solve (T=128, chunk_size=64) is the regime's true worst on both
    # axes -- same case that is ALSO the forward-parity "benign" worst
    # (FWD_PINS: y max_abs=3.755093e-06) -- more doubling steps in the
    # triangular solve (6 vs 4) propagate more chain-rule terms through the
    # backward pass: dq=3.480911e-05/8.125334e-07, dk=1.907349e-04/
    # 9.085284e-07, dv=1.883507e-05/1.387389e-06, dg=2.174377e-04/
    # 3.187417e-06, dbeta=1.392365e-04/1.185957e-06.
    "benign": {
        "dq": (7e-5, 2e-6), "dk": (4e-4, 2e-6), "dv": (4e-5, 3e-6),
        "dg": (5e-4, 7e-6), "dbeta": (3e-4, 2.5e-6),
    },
    # Single case (bwd_repeated_keys, T=96): dq=7.629395e-06/1.676369e-07,
    # dk=1.525879e-05/1.995048e-07, dv=1.907349e-06/3.460737e-07,
    # dg=1.545623e-05/8.745554e-06, dbeta=1.144409e-05/2.540822e-07.
    # dq/dk/dv/dbeta match the brief's "7.2e-6..3.6e-5 max_abs / ~2.5e-7
    # rel_fro at T=96" almost exactly. dg's rel_fro (8.7e-6) is the one
    # outlier -- ~35x the brief's floor -- but the matching forward case
    # (CASES "repeated_keys") measures y at (4.768372e-07, 2.126548e-07),
    # i.e. the fp32 floor; the ~35-40x forward-to-backward jump for dg here
    # is the SAME log-gate amplification noted above, not specific to
    # repeated keys (the collinear-key stress case for the solve itself is
    # the separate "blowup_corner" forward regime, not exercised here).
    "adversarial": {
        "dq": (1.5e-5, 4e-7), "dk": (3e-5, 4e-7), "dv": (4e-6, 7e-7),
        "dg": (3e-5, 1.8e-5), "dbeta": (2.5e-5, 5e-7),
    },
    # Single case (bwd_representative, T=512, real 0.8B geometry): dq=
    # 1.487732e-04/1.034223e-06, dk=6.408691e-04/1.035051e-06, dv=
    # 3.576279e-05/1.148390e-06, dg=1.022339e-03/3.048340e-06, dbeta=
    # 9.002686e-04/1.019092e-06. Absolute magnitudes scale with T=512 and the
    # Hk=Hv=16/Dk=Dv=128 geometry (bigger sum-of-squares loss => bigger
    # absolute gradient differences at the same fp32 relative precision --
    # rel_fro stays in the same ~1e-6..3e-6 band as the T=96 benign cases).
    "representative": {
        "dq": (3e-4, 2e-6), "dk": (1.3e-3, 2e-6), "dv": (7e-5, 2.5e-6),
        "dg": (2e-3, 6e-6), "dbeta": (1.8e-3, 2e-6),
    },
    # Single case (bwd_bf16_repr_T512): dq=6.250000e-02/6.167679e-05,
    # dk=1.000000e+00/2.758466e-03, dv=7.812500e-03/7.351141e-05,
    # dg=3.125000e-02/3.305451e-05, dbeta=1.562500e-02/6.711047e-06. Matches
    # the brief's expectation almost exactly: dk max_abs lands at ~1.0
    # (scales with the T=512/Hv=16/Dv=128 sum-of-squares loss magnitude --
    # expected, not a bug) and dk rel_fro at ~2.8e-3, just under the bf16
    # resolution ceiling 2^-8 ~= 3.9e-3.
    "bf16_representative": {
        "dq": (0.13, 1.2e-4), "dk": (2.0, 5.6e-3), "dv": (1.6e-2, 1.5e-4),
        "dg": (6.5e-2, 7e-5), "dbeta": (3.2e-2, 1.4e-5),
    },
}


def _bwd_param(case):
    case_id, regime, kw = case
    marks = (needs_long_context_room,) if regime in (
        "representative", "bf16_representative") else ()
    return pytest.param(case_id, regime, kw, marks=marks, id=case_id)


@pytest.mark.parametrize(("case_id", "regime", "kw"), [_bwd_param(c) for c in BWD_CASES])
def test_chunked_bwd_matches_sequential(case_id, regime, kw):
    # Catches per regime: a broken backward through the blocked triangular
    # solve, blown-up gradients at the repeated-keys corner, a wrong
    # repeat_factor broadcast in the backward pass, a mis-handled
    # partial-chunk pad in the vjp, or a dropped state-carry gradient path.
    d = build_case(**kw)
    loss_seq, loss_chk = losses_pair(d["state"], kw["chunk_size"])

    grads_ref = mx.grad(loss_seq, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_ref)
    grads_chk = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_chk)

    pin = BWD_PINS[regime]
    for name, g_ref, g_chk in zip(
        ("dq", "dk", "dv", "dg", "dbeta"), grads_ref, grads_chk, strict=True
    ):
        assert_under_pin(f"{case_id}/{name}", metrics(g_chk, g_ref), pin[name])


def test_every_input_receives_nonzero_gradient():
    # Catches: a differentiable array entering the checkpointed chunk body by
    # closure — mx.checkpoint returns a silent ZERO gradient for it.
    d = build_case(T=96, chunk_size=16, with_state=True)

    def loss(q, k, v, g, beta, state):
        y, s = chunked_gated_delta(q, k, v, g, beta, state, chunk_size=16)
        return (y.astype(mx.float32) ** 2).sum() + (s**2).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        d["q"], d["k"], d["v"], d["g"], d["beta"], d["state"])
    mx.eval(*grads)
    for name, gr in zip(("q", "k", "v", "g", "beta", "state"), grads, strict=True):
        assert bool(mx.isfinite(gr).all()), f"d{name} not finite"
        assert float(mx.abs(gr).max()) > 0.0, f"d{name} is all zero"


def test_bf16_grads_finite_and_nonzero():
    # Catches: fp32-island casts inside the chunk breaking bf16 autodiff
    # (upstream's own bf16 criterion: finite grads, not parity).
    d = build_case(T=96, chunk_size=16, dtype=mx.bfloat16)
    _, loss_chk = losses_pair(None, 16)
    grads = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads)
    for name, gr in zip(("q", "k", "v", "g", "beta"), grads, strict=True):
        assert bool(mx.isfinite(gr).all()), f"d{name} not finite"
        assert float(mx.abs(gr).max()) > 0.0, f"d{name} is all zero"
