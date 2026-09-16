"""Negative controls + named divergences for the chunked GatedDelta backward gate.

Every existing `test_recurrent_bwd_parity.py`/`test_recurrent_fwd_parity.py` case
proves the gate passes on correct code. None of them prove the gate can FAIL --
a parity assertion that never fires (a tautology, a swapped operand, a pin so
loose nothing trips it) would pass every one of those tests too. This file
deliberately breaks the implementation three calibrated ways and asserts the
comparison then exceeds 10x its own regime pin, plus pins two KNOWN, deliberate
forward divergences from the oracle so a future silent change gets noticed.
"""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm")
from recurrent_parity_cases import SEED, build_case, losses_pair, metrics
from test_recurrent_bwd_parity import BWD_PINS
from test_recurrent_fwd_parity import FWD_PINS, assert_under_pin

from mlx_train_perf.recurrent import ops
from mlx_train_perf.recurrent.ops import chunked_gated_delta
from mlx_train_perf.recurrent.reference import sequential_gated_delta

_GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")


def _exceeds_10x(got: tuple[float, float], pin: tuple[float, float]) -> bool:
    return got[0] > 10 * pin[0] or got[1] > 10 * pin[1]


def test_severed_inter_chunk_carry_fails_parity(monkeypatch):
    # Catches: a driver refactor that carries chunk state through a detached
    # buffer (e.g. re-deriving `st` from a cached/copied array instead of the
    # real checkpointed output) -- the gradient path between EVERY chunk pair
    # would silently vanish, and the bwd-parity suite would need to notice.
    # This monkeypatches the module-level `_chunk_ckpt` itself (not merely
    # `mx.stop_gradient` on the outer `state` argument -- that would only
    # sever the FIRST chunk's incoming carry, a strictly weaker property)
    # with a wrapper that stop-gradients every inter-chunk boundary.
    d = build_case(T=96, chunk_size=16, with_state=True)
    loss_seq, loss_chk = losses_pair(d["state"], 16)

    def severed_chunk(state, q, k, v, g, beta, repeat_factor=1):
        y, new_state = ops._gated_delta_chunk(state, q, k, v, g, beta, repeat_factor)
        return y, mx.stop_gradient(new_state)

    monkeypatch.setattr(ops, "_chunk_ckpt", severed_chunk)

    grads_ref = mx.grad(loss_seq, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_ref)
    grads_severed = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_severed)

    pin = BWD_PINS["benign"]
    margins = {
        name: metrics(g_sev, g_ref)
        for name, g_ref, g_sev in zip(_GRAD_NAMES, grads_ref, grads_severed, strict=True)
    }
    assert any(_exceeds_10x(margins[name], pin[name]) for name in _GRAD_NAMES), (
        f"severed inter-chunk carry did not exceed 10x any benign pin: {margins}"
    )


def test_calibrated_grad_scale_fails_parity():
    # Catches: an `assert_under_pin` regression that stops discriminating
    # magnitude at all (e.g. a swapped operand, or comparing an array to
    # itself) -- multiplying the chunked loss by a constant factor scales
    # every one of its gradients by that same factor, so the comparison
    # against the (unscaled) oracle must then diverge. The factor is derived
    # from the benign regime's own worst rel_fro pin (`1 + 20 * pin`), which
    # is self-verifying: a constant-factor scale introduces a relative error
    # of approximately `20 * pin`, i.e. ~20x the 10x-exceedance floor below,
    # regardless of which pin file gets edited later.
    d = build_case(T=96, chunk_size=16)
    loss_seq, loss_chk = losses_pair(d["state"], 16)

    pin = BWD_PINS["benign"]
    worst_rel_fro = max(v[1] for v in pin.values())
    scale = 1.0 + 20.0 * worst_rel_fro

    def loss_chk_scaled(q, k, v, g, beta):
        return loss_chk(q, k, v, g, beta) * scale

    grads_ref = mx.grad(loss_seq, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_ref)
    grads_scaled = mx.grad(loss_chk_scaled, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_scaled)

    margins = {
        name: metrics(g_scl, g_ref)
        for name, g_ref, g_scl in zip(_GRAD_NAMES, grads_ref, grads_scaled, strict=True)
    }
    failing = [name for name in _GRAD_NAMES if _exceeds_10x(margins[name], pin[name])]
    assert failing, f"scale factor {scale} did not push any component past 10x its pin: {margins}"
    # The parity comparison itself -- assert_under_pin -- must actually raise
    # for at least one of those components, proving the gate mechanism fires.
    for name in failing:
        with pytest.raises(AssertionError):
            assert_under_pin(name, margins[name], pin[name])


def test_dropped_decay_fails_forward_parity():
    # Catches: a forward-parity assertion that's vacuously true (e.g.
    # comparing an array to itself, or a pin so loose nothing can trip it) --
    # replacing the true decay gate with all-ones drops decay entirely, which
    # must diverge sharply from the true-g oracle.
    d = build_case(T=96, chunk_size=16)
    y_oracle, state_oracle = sequential_gated_delta()(
        d["q"], d["k"], d["v"], d["g"], d["beta"], d["state"])
    y_dropped, state_dropped = chunked_gated_delta(
        d["q"], d["k"], d["v"], mx.ones_like(d["g"]), d["beta"],
        d["state"], chunk_size=16)
    mx.eval(y_oracle, state_oracle, y_dropped, state_dropped)

    pin = FWD_PINS["benign"]
    y_got = metrics(y_dropped, y_oracle)
    state_got = metrics(state_dropped, state_oracle)
    assert _exceeds_10x(y_got, pin["y"]) or _exceeds_10x(state_got, pin["state"]), (
        f"dropped-decay control did not exceed 10x benign pin: y={y_got}, state={state_got}"
    )


def test_g_clamp_zeroes_dLdg_below_clamp_KNOWN_DIVERGENCE():  # noqa: N802
    # Pins a KNOWN, deliberate forward divergence documented on
    # `ops._gated_delta_chunk`'s own "KNOWN divergence" comment: clamping `g`
    # away from zero before `log` (to keep -inf out of the cumsum) zeroes
    # dL/dg entirely for g <= 1e-12, while the sequential oracle still
    # produces a real, large gradient through the true near-zero decay. This
    # test exists so a future silent change to the forward pass (e.g.
    # loosening or removing the clamp) gets noticed instead of quietly
    # "fixing" a divergence nobody re-measured.
    d = build_case(T=96, chunk_size=16, g_mode="zero")
    loss_seq, loss_chk = losses_pair(d["state"], 16)

    grads_ref = mx.grad(loss_seq, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_ref)
    grads_chk = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_chk)

    dg_chk, dg_ref = grads_chk[3], grads_ref[3]
    assert float(mx.abs(dg_chk).max()) == 0.0, "chunked dL/dg should be exactly zero at the clamp"
    assert float(mx.abs(dg_ref).max()) > 1e-3, "oracle dL/dg should be well above the clamp floor"

    # The other four gradients are unaffected by the clamp and must still
    # meet the ordinary benign-regime pin at this same corner.
    pin = BWD_PINS["benign"]
    for name, idx in (("dq", 0), ("dk", 1), ("dv", 2), ("dbeta", 4)):
        assert_under_pin(name, metrics(grads_chk[idx], grads_ref[idx]), pin[name])


def test_masked_positions_output_unspecified_CONTRACT():  # noqa: N802
    # Documents a deliberate contract: past the mask's valid length, `y` is
    # NOT parity-checked here. A probe measured a 0.66 divergence at masked
    # positions, and upstream itself is three-way inconsistent about what a
    # masked position's output should even be -- the Metal kernel writes 0,
    # the sequential oracle writes a decayed-state output, and this chunked
    # form writes `q @ state_prev`. Only `y[:, :valid]` and the final `state`
    # are asserted; masked-position `y` values are unspecified by contract.
    mx.random.seed(SEED)
    b, t, hk, hv, dk, dv = 1, 8, 2, 2, 8, 8
    q = mx.random.normal(shape=(b, t, hk, dk)) * 0.5
    k = mx.random.normal(shape=(b, t, hk, dk))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal(shape=(b, t, hv, dv)) * 0.5
    g = mx.sigmoid(mx.random.normal(shape=(b, t, hv)))
    beta = mx.sigmoid(mx.random.normal(shape=(b, t, hv)) + 2.0)
    mask = mx.array([[True] * 5 + [False] * 3])
    mx.eval(q, k, v, g, beta, mask)

    y_chk, state_chk = chunked_gated_delta(q, k, v, g, beta, mask=mask, chunk_size=4)
    y_oracle, state_oracle = sequential_gated_delta()(
        q[:, :5], k[:, :5], v[:, :5], g[:, :5], beta[:, :5])
    mx.eval(y_chk, state_chk, y_oracle, state_oracle)

    # Measured in this tree (2026-08-30): y[:, :5] max_abs=1.788143e-07,
    # rel_fro=1.593440e-07; state max_abs=1.192093e-07, rel_fro=9.951312e-08
    # -- both at the fp32 accumulation floor (~1e-6), consistent with the
    # probe's "bitwise-identical oracle paths, state parity 5.96e-8" (probe
    # used a different geometry/seed; same floor). Pinned at measured-worst
    # x ~2.
    assert_under_pin("y[:, :5]", metrics(y_chk[:, :5], y_oracle), (4e-7, 4e-7))
    assert_under_pin("state", metrics(state_chk, state_oracle), (3e-7, 2e-7))
