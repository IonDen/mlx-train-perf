"""Layer-level parity for `GatedDeltaTrainingProxy`: the op-boundary gate plus the
whole-layer composition gates.

Two same-seeded tiny qwen3_5 models give a stock `GatedDeltaNet` and a
`GatedDeltaTrainingProxy` bit-identical starting weights, so any output
difference below is glue or op, never a weight mismatch. `_pre_norm` is the
numerical gate that matters: the block's own `RMSNormGated` post-norm is
scale-blind (it normalizes `out` before applying the learned `weight`), so it
would absorb a per-(token, head) scale error the op boundary is the only
place that can still see. `test_op_boundary_parity` captures the REAL,
source-pinned `GatedDeltaNet.__call__`'s internal `(out, state)` via a
monkeypatch spy on `gated_delta_update` -- train mode routes that call to
`gated_delta_ops`, the same function `sequential_gated_delta()` returns, so
the capture is an independent oracle built from the actual stock code path,
not a hand-duplicated re-derivation of it. The remaining tests are
composition gates: they prove the post-norm + output-projection wiring in
`__call__` doesn't introduce a NEW divergence on top of whatever `_pre_norm`
already has.

Gradient comparisons use `mx.grad`/`nn.value_and_grad`; no assertion reads a
loss value returned by `value_and_grad`.
"""

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map_with_path

pytest.importorskip("mlx_lm")
import mlx_lm.models.qwen3_5 as qwen3_5_mod
from qwen35_tiny import tiny_qwen35
from recurrent_parity_cases import metrics
from test_recurrent_fwd_parity import assert_under_pin

from mlx_train_perf.errors import RecurrentInputError
from mlx_train_perf.recurrent.wrapper import enable_gated_delta_training

SEED = 0
_B, _S = 2, 96
_HIDDEN_SIZE = 64


def _stock_and_proxy(*, bf16: bool = False, impl: str = "chunked", seed: int = SEED):
    """Two structurally-identical tiny qwen3_5 models built from the same seed -- one
    left stock, one with gated-delta training enabled -- returning each model's first
    linear-attention layer's `linear_attn` submodule. Both arms are explicitly put in
    TRAIN mode: `GatedDeltaNet.__call__` routes `use_kernel=not self.training`, so an
    eval-mode stock arm would run the Metal inference kernel instead of the sequential
    oracle path this comparison relies on."""
    stock_model = tiny_qwen35(seed=seed)
    proxy_model = tiny_qwen35(seed=seed)
    if bf16:

        def cast(path: str, x: mx.array) -> mx.array:
            # A_log stays fp32: the proxy's own guard would otherwise refuse it, and
            # qwen3_5's own cast_predicate makes exactly this exception.
            return x if path.endswith("A_log") else x.astype(mx.bfloat16)

        stock_model.update(tree_map_with_path(cast, stock_model.parameters()))
        proxy_model.update(tree_map_with_path(cast, proxy_model.parameters()))
        mx.eval(stock_model.parameters(), proxy_model.parameters())
    enable_gated_delta_training(proxy_model, impl=impl)

    stock = next(
        layer for layer in stock_model.language_model.model.layers if layer.is_linear
    ).linear_attn
    proxy = next(
        layer for layer in proxy_model.language_model.model.layers if layer.is_linear
    ).linear_attn
    stock.train()
    proxy.train()
    return stock, proxy


def _inputs(*, bf16: bool = False, seed: int = SEED + 1) -> mx.array:
    mx.random.seed(seed)
    x = mx.random.normal((_B, _S, _HIDDEN_SIZE))
    if bf16:
        x = x.astype(mx.bfloat16)
    mx.eval(x)
    return x


def _capture_stock_pre_op(
    monkeypatch: pytest.MonkeyPatch, stock: nn.Module, inputs: mx.array
) -> tuple[mx.array, mx.array]:
    """Runs the real, source-pinned `GatedDeltaNet.__call__` and captures the
    `(out, state)` pair it computes internally, before the post-op gated RMSNorm --
    an independent oracle built from the actual stock code path."""
    captured: dict[str, mx.array] = {}
    real = qwen3_5_mod.gated_delta_update

    def spy(*args: object, **kwargs: object) -> tuple[mx.array, mx.array]:
        out, state = real(*args, **kwargs)
        captured["out"], captured["state"] = out, state
        return out, state

    monkeypatch.setattr(qwen3_5_mod, "gated_delta_update", spy)
    stock(inputs, None, None)
    assert captured, (
        "stock's forward never called gated_delta_update -- the spy never fired, "
        "so this capture is not a real oracle"
    )
    return captured["out"], captured["state"]


def _sum_sq_loss(module: nn.Module, inputs: mx.array) -> mx.array:
    out = module(inputs, None, None)
    return (out.astype(mx.float32) ** 2).sum()


# Measured in this tree (single case: B=2, S=96, tiny qwen3_5 geometry --
# num_v_heads=num_k_heads=2, head_k_dim=head_v_dim=16, hidden_size=64, seed=0
# for both model construction and inputs), comparing GatedDeltaTrainingProxy
# (impl="chunked")._pre_norm's (out, state) against the REAL stock
# GatedDeltaNet.__call__'s internal (out, state) captured via the monkeypatch
# spy above. fp32: out max_abs=2.095476e-09, rel_fro=2.863268e-07; state
# max_abs=5.075708e-08, rel_fro=3.106428e-06 -- both within the fp32
# accumulation floor (~1e-6), no larger than test_recurrent_fwd_parity.py's
# op-level "benign"/"representative" regimes at a comparable T despite this
# comparison additionally exercising the qwen3_5 glue (conv1d, split,
# rms_norm, gate math) the op-level tests never touch. Spot-checked at two
# other seeds (7, 42): state rel_fro ran up to 3.2e-05 there, still inside
# the same fp32-floor order of magnitude -- the pin below covers seed=0 (the
# case this file actually runs) at measured-worst x ~2.
OP_BOUNDARY_PINS: dict[str, dict[str, tuple[float, float]]] = {
    "fp32": {"out": (5e-9, 6e-7), "state": (1.2e-7, 6.5e-6)},
    # bf16: out max_abs=1.862645e-09, rel_fro=4.611970e-08 -- `out` is cast back to
    # bf16 inside the op (`y.astype(orig_dtype)`), which quantizes away nearly all of
    # the fp32-level divergence measured above. state max_abs=7.171184e-07,
    # rel_fro=4.561284e-05 -- state stays fp32 throughout (never cast to bf16 across
    # chunks), so it shows the real chunked-vs-sequential rounding difference under
    # bf16-precision inputs/weights, still comfortably under the bf16 resolution
    # ceiling (2^-8 ~= 3.9e-3). Spot-checked at seeds 7/42: state rel_fro ran 1.4e-05
    # to 4.6e-05, same order of magnitude. Pinned at measured-worst (seed=0) x ~2.
    "bf16": {"out": (4e-9, 1e-7), "state": (1.5e-6, 1e-4)},
}


@pytest.mark.parametrize(
    ("dtype_name", "bf16"), [("fp32", False), ("bf16", True)]
)
def test_op_boundary_parity(
    monkeypatch: pytest.MonkeyPatch, dtype_name: str, bf16: bool
) -> None:
    # Catches: any divergence in the mirrored glue (conv1d, the qkv split, the
    # asymmetric q/k rms_norm scaling, the in-tree gate math) or in the
    # production chunked op itself, that the downstream gated RMSNorm would
    # otherwise mask (it normalizes `out` before applying its learned scale,
    # so a scale error in `out` never survives to the whole-layer output).
    stock, proxy = _stock_and_proxy(bf16=bf16)
    inputs = _inputs(bf16=bf16)

    out_ref, state_ref = _capture_stock_pre_op(monkeypatch, stock, inputs)
    mx.eval(out_ref, state_ref)
    out_chk, state_chk = proxy._pre_norm(inputs, None)
    mx.eval(out_chk, state_chk)

    pin = OP_BOUNDARY_PINS[dtype_name]
    assert_under_pin(f"{dtype_name}/out", metrics(out_chk, out_ref), pin["out"])
    assert_under_pin(f"{dtype_name}/state", metrics(state_chk, state_ref), pin["state"])


# Measured in this tree (same case as OP_BOUNDARY_PINS): full `stock(inputs, None,
# None)` vs full `proxy(inputs, None, None)`, i.e. the op-boundary values above
# additionally passed through the shared gated RMSNorm + out_proj. max_abs=
# 3.734604e-07, rel_fro=3.800941e-07 -- essentially the op-boundary "out" divergence
# unchanged in order of magnitude (the composition step adds no new error, as
# expected: it is the SAME norm/out_proj weights on both arms). Pinned at
# measured-worst x ~2.
WHOLE_LAYER_FWD_PIN: tuple[float, float] = (8e-7, 8e-7)


def test_whole_layer_forward_parity() -> None:
    # Catches: a bug in __call__'s own composition (z's reshape, the norm(out, z)
    # call, the final out.reshape before out_proj) that the op-boundary gate above
    # cannot see, since it never runs __call__ end-to-end.
    stock, proxy = _stock_and_proxy(bf16=False)
    inputs = _inputs(bf16=False)

    out_ref = stock(inputs, None, None)
    out_chk = proxy(inputs, None, None)
    mx.eval(out_ref, out_chk)

    assert_under_pin("forward", metrics(out_chk, out_ref), WHOLE_LAYER_FWD_PIN)


# Measured in this tree (same case): mx.grad of a sum-of-squares loss on
# proxy(inputs) vs stock(inputs), w.r.t. `inputs` alone. max_abs=5.960464e-07,
# rel_fro=6.689028e-07 -- fp32 floor. Pinned at measured-worst x ~2.
GRAD_INPUTS_PIN: tuple[float, float] = (1.2e-6, 1.4e-6)

# Measured in this tree (same case): nn.value_and_grad(module, loss) w.r.t. every
# trainable weight leaf, split into two groups by measured noise profile --
# grouping them under one shared pin would force the tighter group's pin as loose
# as the noisier one, hiding a real regression in six of the nine leaves.
#
# "proj" (conv1d.weight, in_proj_qkv.weight, in_proj_z.weight, in_proj_b.weight,
# norm.weight, out_proj.weight): worst max_abs=3.069639e-05 (conv1d.weight), worst
# rel_fro=3.959619e-07 (in_proj_qkv.weight) -- fp32 floor, unsurprising (these
# leaves sit before or after the op with no exp/log chain in between).
#
# "gate" (in_proj_a.weight, dt_bias, A_log): worst max_abs=1.502138e-04 (A_log),
# worst rel_fro=2.963619e-05 (A_log). These three feed `g` through
# `exp(-exp(A_log) * softplus(a + dt_bias))` -- the same log-domain chain
# test_recurrent_bwd_parity.py's BWD_PINS already documents as the op's most
# fp32-sensitive backward path (there, for dg itself; here, one differentiation
# step further upstream, into the parameters that produce g). Investigated: at
# two other seeds (7, 42) A_log's rel_fro ran up to 1.67e-2 -- a real, if
# unsurprising, further amplification (num_v_heads=2 in this tiny model means
# A_log/dt_bias are 2-element vectors, so their relative-Frobenius-norm is noisy
# by construction on so few elements) rather than a new failure mode; the pin
# below covers seed=0 (the case this file actually runs) at measured-worst x ~2.
GRAD_WEIGHT_PINS: dict[str, tuple[float, float]] = {
    "proj": (6.2e-5, 8e-7),
    "gate": (3.0e-4, 6e-5),
}
_GATE_LEAVES = frozenset({"in_proj_a.weight", "dt_bias", "A_log"})


def test_whole_layer_grad_parity() -> None:
    # Catches: a bug __call__'s composition introduces that only shows up under
    # backward (e.g. a detached or duplicated graph edge through z/norm/out_proj)
    # -- forward parity alone cannot see this.
    stock, proxy = _stock_and_proxy(bf16=False)
    inputs = _inputs(bf16=False)

    grad_inputs_ref = mx.grad(lambda x: _sum_sq_loss(stock, x))(inputs)
    grad_inputs_chk = mx.grad(lambda x: _sum_sq_loss(proxy, x))(inputs)
    mx.eval(grad_inputs_ref, grad_inputs_chk)
    assert_under_pin(
        "grad_inputs", metrics(grad_inputs_chk, grad_inputs_ref), GRAD_INPUTS_PIN
    )

    _, grads_ref = nn.value_and_grad(stock, _sum_sq_loss)(stock, inputs)
    _, grads_chk = nn.value_and_grad(proxy, _sum_sq_loss)(proxy, inputs)
    mx.eval(grads_ref, grads_chk)
    flat_ref = dict(tree_flatten(grads_ref))
    flat_chk = dict(tree_flatten(grads_chk))
    assert flat_ref.keys() == flat_chk.keys()
    for key, g_ref in flat_ref.items():
        group = "gate" if key in _GATE_LEAVES else "proj"
        assert_under_pin(
            f"grad_weight/{key}", metrics(flat_chk[key], g_ref), GRAD_WEIGHT_PINS[group]
        )


def test_impl_sequential_matches_stock_layer() -> None:
    # Catches: any glue bug that only shows up once the CHUNKED-vs-SEQUENTIAL op
    # divergence is removed from the picture -- impl="sequential" routes the
    # proxy through the exact same gated_delta_ops the stock layer's own
    # use_kernel=False dispatch calls, so ANY difference here must be glue, not
    # the production op. Measured in this tree (same case): forward
    # max_abs=0.0, rel_fro=0.0 -- bitwise identical (probe-verified, not merely
    # a loose pin).
    stock, proxy = _stock_and_proxy(bf16=False, impl="sequential")
    inputs = _inputs(bf16=False)

    out_ref = stock(inputs, None, None)
    out_seq = proxy(inputs, None, None)
    mx.eval(out_ref, out_seq)

    assert_under_pin("impl_sequential", metrics(out_seq, out_ref), (0.0, 0.0))


def test_cache_none_positional_works() -> None:
    # Catches: __call__ treating a POSITIONALLY-passed cache=None differently from
    # the keyword form -- DecoderLayer.__call__ always calls
    # `self.linear_attn(self.input_layernorm(x), mask, cache)`, all positional, so
    # this is the actual calling convention production code uses.
    _stock, proxy = _stock_and_proxy(bf16=False)
    inputs = _inputs(bf16=False)

    out = proxy(inputs, None, None)
    mx.eval(out)

    assert out.shape == inputs.shape
    assert bool(mx.isfinite(out).all())


def test_a_log_bf16_guard_refuses_before_the_forward_runs() -> None:
    # Catches: the guard living somewhere that only fires on SOME call paths (e.g.
    # only via __call__, never via a direct _pre_norm call from a future caller) --
    # both entry points must refuse identically, since _pre_norm is itself a public
    # surface the op-boundary gate above calls directly.
    _stock, proxy = _stock_and_proxy(bf16=False)
    proxy.A_log = proxy.A_log.astype(mx.bfloat16)
    mx.eval(proxy.A_log)
    inputs = _inputs(bf16=False)

    with pytest.raises(RecurrentInputError, match="A_log"):
        proxy(inputs, None, None)
    with pytest.raises(RecurrentInputError, match="A_log"):
        proxy._pre_norm(inputs, None)
