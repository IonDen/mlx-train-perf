"""Composition gates + dtype-protocol asserts for the chunked GatedDelta op.

Two composition regimes, probe-verified on this tree (mlx 0.32.0) to be BITWISE
identical to the plain (uncomposed) gradient computation:

1. an EXTERNAL ``mx.checkpoint`` wrapped around the op's own loss --
   checkpoint-of-checkpoint, since the op's chunk body already runs each chunk
   under its own internal ``mx.checkpoint``. This is exactly what mlx-lm's
   class-level ``grad_checkpoint`` layers on top in production.
2. ``mx.compile(mx.value_and_grad(...))``, guarded by a trace-count liveness
   sentinel: a Python-side counter inside the traced loss proves the compiled
   arm actually compiled (a pass-through ``mx.compile``, e.g. under
   ``MLX_DISABLE_COMPILE``, would otherwise make this whole arm silently
   vacuous while still reporting bitwise-identical grads for the wrong
   reason -- it never actually exercised the compiled path).

0045 constraint: every comparison below is on GRADIENTS. A ``value_and_grad``
loss VALUE is never asserted on -- there is an open, unexplained anomaly in
this project where that value is insensitive to which implementation ran, so
no gate may rest on it.
"""

import mlx.core as mx
import pytest
from recurrent_parity_cases import build_case, losses_pair, metrics
from test_recurrent_fwd_parity import assert_under_pin

from mlx_train_perf.recurrent.ops import chunked_gated_delta

_GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
# Probe-verified bitwise identity (mlx 0.32.0): both composition arms below
# reproduce the plain path exactly, so the smallest honest pin is exact zero,
# not a cushioned epsilon.
_EXACT_ZERO = (0.0, 0.0)


def test_external_checkpoint_grads_bitwise_match_plain() -> None:
    # Catches: checkpoint-of-checkpoint (an EXTERNAL mx.checkpoint layered on
    # top of the op's own internal per-chunk mx.checkpoint) corrupting a
    # gradient -- e.g. the closure-vs-positional gotcha the op's own docstring
    # flags (a closed-over differentiable array silently receiving a ZERO
    # gradient under mx.checkpoint), or the double-forward-recompute path
    # disagreeing with the single-forward plain path on any component.
    d = build_case(T=96, chunk_size=16)
    _, loss_chk = losses_pair(d["state"], d["chunk_size"])

    grads_plain = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_plain)

    grads_ckpt = mx.grad(mx.checkpoint(loss_chk), argnums=(0, 1, 2, 3, 4))(
        d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_ckpt)

    for name, g_plain, g_ckpt in zip(_GRAD_NAMES, grads_plain, grads_ckpt, strict=True):
        assert_under_pin(f"checkpoint/{name}", metrics(g_ckpt, g_plain), _EXACT_ZERO)


def test_compiled_value_and_grad_bitwise_matches_plain_with_trace_liveness_sentinel() -> None:
    # Catches two independent breaks with one gate: (a) mx.compile(mx.value_and_grad(...))
    # diverging numerically from the plain path -- e.g. compile's op fusion
    # reordering the fp32 accumulation inside the blocked triangular solve;
    # (b) the whole compiled arm going silently VACUOUS if mx.compile becomes
    # a pass-through (MLX_DISABLE_COMPILE) -- which would still report
    # bitwise-identical grads, but for the wrong reason (compile never
    # actually ran). The trace-count assertion below fails loudly in that
    # case instead of letting the grad-parity checks report a false GO.
    d = build_case(T=96, chunk_size=16)
    _, loss_chk = losses_pair(d["state"], d["chunk_size"])

    calls = {"n": 0}

    def traced_loss(
        q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array
    ) -> mx.array:
        calls["n"] += 1  # Python side effect: runs once per REAL trace; a replayed
        # trace (same shapes/dtypes as an already-traced call) skips the Python body
        # entirely and returns the cached compiled graph instead.
        return loss_chk(q, k, v, g, beta)

    vg = mx.value_and_grad(traced_loss, argnums=(0, 1, 2, 3, 4))
    _, grads_plain = vg(d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_plain)  # eager call: +1

    compiled_vg = mx.compile(vg)
    _, grads_c1 = compiled_vg(d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_c1)  # first compiled call: traces for the first time, +1
    _, grads_c2 = compiled_vg(d["q"], d["k"], d["v"], d["g"], d["beta"])
    mx.eval(*grads_c2)  # second compiled call: identical shapes/dtypes -> replays
    # the cached trace, +0

    assert calls["n"] == 2, (
        f"expected exactly 2 Python-body executions (1 eager call + 1 real compile "
        f"trace; the second compiled call should replay the cached trace with zero "
        f"new executions), got {calls['n']} -- if mx.compile is a pass-through here "
        "(e.g. MLX_DISABLE_COMPILE set), this whole compile arm is VACUOUS and the "
        "grad-parity assertions below prove nothing about compiled composition"
    )

    for name, g_plain, g_c in zip(_GRAD_NAMES, grads_plain, grads_c1, strict=True):
        assert_under_pin(f"compile_trace/{name}", metrics(g_c, g_plain), _EXACT_ZERO)
    for name, g_plain, g_c in zip(_GRAD_NAMES, grads_plain, grads_c2, strict=True):
        assert_under_pin(f"compile_replay/{name}", metrics(g_c, g_plain), _EXACT_ZERO)


@pytest.mark.parametrize(("dtype_name", "dtype"), [("fp32", mx.float32), ("bf16", mx.bfloat16)])
def test_output_and_state_dtype_protocol(dtype_name: str, dtype: mx.Dtype) -> None:
    # Catches: an accidental cast dropped or added on the y/state boundary --
    # e.g. y not inheriting the input compute dtype (the op's own `orig_dtype`
    # cast lost or misapplied), or the returned state drifting off its
    # fp32-across-chunks contract for either input dtype.
    d = build_case(T=96, chunk_size=16, dtype=dtype)
    y, st = chunked_gated_delta(
        d["q"], d["k"], d["v"], d["g"], d["beta"], chunk_size=d["chunk_size"])
    mx.eval(y, st)
    assert y.dtype == dtype, f"{dtype_name}: y.dtype {y.dtype} != input dtype {dtype}"
    assert st.dtype == mx.float32, (
        f"{dtype_name}: state dtype {st.dtype} != float32 (state must stay fp32 "
        "regardless of input dtype)"
    )
