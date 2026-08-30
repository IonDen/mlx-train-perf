"""`enable_gated_delta_training` selection + refusal matrix (no forward yet).

Layer selection is structural (`layer.is_linear` + a class check on `layer.linear_attn`,
cross-checked against the raw `text_config.layer_types`, when present) -- see
`recurrent/wrapper.py`'s module docstring for the verified mlx-lm facts this rests on. The
proxy's forward pass lands in a later change; here it only needs to refuse a cache before
raising `NotImplementedError`.
"""
from typing import Any
from unittest import mock

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map_with_path

pytest.importorskip("mlx_lm")

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from qwen35_tiny import tiny_qwen35

from mlx_train_perf.errors import RecurrentInputError, UnsupportedRecurrentError
from mlx_train_perf.recurrent.wrapper import (
    GatedDeltaTrainingProxy,
    enable_gated_delta_training,
)


def test_tiny_model_has_both_flavors():
    # Catches: a tiny config whose every layer is one flavor -- the wrapper and
    # refusal tests would silently cover half the surface.
    layers = tiny_qwen35().language_model.model.layers
    assert any(layer.is_linear for layer in layers)
    assert any(not layer.is_linear for layer in layers)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _tiny_qwen35_moe(*, num_layers: int = 4, full_attention_interval: int = 2) -> Any:
    """A tiny qwen3_5 model with `num_experts > 0` -- the same shape as `tiny_qwen35`
    (`qwen35_tiny.py`), plus the MoE fields `DecoderLayer.__init__` needs to build a
    `SparseMoeBlock` instead of a dense `MLP`. Construction only (no forward), so the tiny
    expert dims below only need to be valid, not realistic."""
    qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")
    text_config = {
        "model_type": "qwen3_5",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "full_attention_interval": full_attention_interval,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
    }
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text_config)
    mx.random.seed(0)
    model = qwen3_5.Model(args)
    mx.eval(model.parameters())
    return model


class _FakeQwen3Next:
    """Impersonates `mlx_lm.models.qwen3_next.Model` by `__module__` only -- the
    qwen3_next refusal fires on the module name alone, before any attribute of the model
    is ever touched, so no further structure is needed (mirrors `test_families.py`'s
    `_FakeLlamaModel` convention)."""


_FakeQwen3Next.__module__ = "mlx_lm.models.qwen3_next"


def _linear_attn_identities(model: Any) -> list[int | None]:
    """`id()` of each linear-attention layer's `linear_attn`, `None` for full-attention
    layers -- `id()` rather than the object itself, because `nn.Module` subclasses `dict`
    and compares by VALUE, which would hide a same-content replacement."""
    return [
        id(layer.linear_attn) if layer.is_linear else None
        for layer in model.language_model.model.layers
    ]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_enable_wraps_only_linear_layers():
    # Catches: enabling on every layer regardless of is_linear (would reach for a
    # nonexistent linear_attn on a full-attention layer, or leave a linear
    # layer's GatedDeltaNet untouched).
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    layers = model.language_model.model.layers
    linear_flags = [layer.is_linear for layer in layers]

    enable_gated_delta_training(model)

    for layer, is_linear in zip(layers, linear_flags, strict=True):
        if is_linear:
            assert isinstance(layer.linear_attn, GatedDeltaTrainingProxy)
        else:
            assert not hasattr(layer, "linear_attn")


def test_parameter_key_set_identical_before_after_enable():
    # Catches: the proxy renaming a submodule (breaking the layers.N.linear_attn.*
    # tree path LoRA/optimizer keys rely on) or retaining `original` as an
    # attribute (doubling every one of its parameter keys). Also compares the
    # MODULE-TREE key set (named_modules()), not just the flattened parameters: mlx's
    # valid_parameter_filter excludes keys starting with "_" from .parameters(), but
    # named_modules() (which linear_to_lora_layers walks for LoRA target discovery)
    # does NOT underscore-filter -- so a regression that retained the original under
    # e.g. `self._original = original` would duplicate every one of its submodules for
    # LoRA discovery while staying completely invisible to the flattened-parameter
    # comparison alone (verified: such a retention leaves keys_after == keys_before
    # but adds 8 extra named_modules() paths under `linear_attn._original.*`).
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    keys_before = set(dict(tree_flatten(model.parameters())).keys())
    module_paths_before = {name for name, _ in model.named_modules()}

    enable_gated_delta_training(model)

    keys_after = set(dict(tree_flatten(model.parameters())).keys())
    assert keys_after == keys_before
    module_paths_after = {name for name, _ in model.named_modules()}
    assert module_paths_after == module_paths_before


def test_cache_not_none_refuses():
    # Catches: the proxy's cache refusal running AFTER (or never running before)
    # the NotImplementedError -- this must be checkable before the forward
    # exists at all.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    enable_gated_delta_training(model)
    linear_layer = next(layer for layer in model.language_model.model.layers if layer.is_linear)
    proxy = linear_layer.linear_attn
    x = mx.zeros((1, 4, 64))

    with pytest.raises(RecurrentInputError, match="cache"):
        proxy(x, None, cache=object())


def test_frozen_keys_carry_onto_proxy():
    # Catches: the proxy silently losing model.freeze()'s {'A_log', 'dt_bias'}
    # (making them trainable again) instead of carrying `_no_grad` over.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    model.freeze()
    enable_gated_delta_training(model)
    linear_layer = next(layer for layer in model.language_model.model.layers if layer.is_linear)
    proxy = linear_layer.linear_attn
    assert proxy._no_grad == {"A_log", "dt_bias"}


def test_proxy_construction_fails_loud_on_uncopied_frozen_key():
    # Catches: an attribute-copy list drifting out of sync with a future
    # nn.Module._no_grad entry -- the proxy must fail loud, not silently
    # un-freeze a parameter the caller explicitly froze.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    model.freeze()
    linear_layer = next(layer for layer in model.language_model.model.layers if layer.is_linear)
    original = linear_layer.linear_attn
    original._no_grad.add("nonexistent_attr")

    with pytest.raises(UnsupportedRecurrentError, match="nonexistent_attr"):
        GatedDeltaTrainingProxy(original, impl="chunked")


# ---------------------------------------------------------------------------
# refusal matrix (8 named cases)
# ---------------------------------------------------------------------------


def test_enable_refuses_wrong_family():
    # Catches: enable_gated_delta_training accepting a non-qwen3_5 model instead
    # of raising through families.text_model's module check.
    with pytest.raises(UnsupportedRecurrentError, match="unsupported model architecture"):
        enable_gated_delta_training(nn.Linear(4, 4))


def test_enable_refuses_moe():
    # Catches: enable_gated_delta_training ignoring num_experts and wrapping a
    # MoE checkpoint's linear layers anyway.
    model = _tiny_qwen35_moe()
    before = _linear_attn_identities(model)

    with pytest.raises(UnsupportedRecurrentError, match="MoE"):
        enable_gated_delta_training(model)

    assert _linear_attn_identities(model) == before


def test_enable_refuses_qwen3_next_by_name():
    # Catches: a qwen3_next model falling through to the generic wrong-family
    # message instead of the dedicated "future work" refusal naming it -- the
    # two families share enough shape (is_linear/linear_attn) that a generic
    # message would be misleading. No layer structure exists on this fake: the
    # refusal fires on __module__ alone, before anything else is touched.
    with pytest.raises(UnsupportedRecurrentError, match="qwen3_next"):
        enable_gated_delta_training(_FakeQwen3Next())


def test_enable_refuses_sharded_layer():
    # Catches: relying on a generic "does this layer have an unexpected
    # attribute" scan -- sharding_group is always present (default None), so
    # such a scan could never distinguish "present" from "set". Requires an
    # explicit `sharding_group is not None` check.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    layers = model.language_model.model.layers
    first_linear = next(i for i, layer in enumerate(layers) if layer.is_linear)
    layers[first_linear].linear_attn.sharding_group = object()
    before = _linear_attn_identities(model)

    with pytest.raises(UnsupportedRecurrentError, match="sharding_group"):
        enable_gated_delta_training(model)

    assert _linear_attn_identities(model) == before


def test_enable_refuses_layer_types_disagreement():
    # Catches: trusting the raw text_config.layer_types list, or the structural
    # is_linear split, alone instead of cross-checking the two against each
    # other. tiny_qwen35(full_attention_interval=2, num_layers=4) is actually
    # [linear, full, linear, full]; this raw list lies and says all-full.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    model.args.text_config["layer_types"] = ["full_attention"] * 4
    before = _linear_attn_identities(model)

    with pytest.raises(UnsupportedRecurrentError, match="layer_types"):
        enable_gated_delta_training(model)

    assert _linear_attn_identities(model) == before


def test_enable_refuses_zero_linear_layers():
    # Catches: enable_gated_delta_training silently succeeding as a no-op when
    # full_attention_interval leaves no layer linear, instead of refusing.
    # full_attention_interval=1 means (idx + 1) % 1 == 0 for every idx, so
    # is_linear is False on every layer.
    model = tiny_qwen35(full_attention_interval=1, num_layers=4)
    before = _linear_attn_identities(model)

    with pytest.raises(UnsupportedRecurrentError, match="no linear-attention layers"):
        enable_gated_delta_training(model)

    assert _linear_attn_identities(model) == before


def test_enable_refuses_double_enable():
    # Catches: enable_gated_delta_training re-wrapping an already-enabled proxy
    # instead of refusing outright. The model IS mutated relative to its
    # pristine state (the first call legitimately succeeded) -- what must NOT
    # change is the state the SECOND, refusing call was given.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    enable_gated_delta_training(model)
    after_first_enable = _linear_attn_identities(model)

    with pytest.raises(UnsupportedRecurrentError, match="already enabled"):
        enable_gated_delta_training(model)

    assert _linear_attn_identities(model) == after_first_enable


def test_enable_refuses_bad_impl():
    # Catches: an unrecognized impl string silently falling back to "chunked"
    # (or "sequential") instead of raising.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    before = _linear_attn_identities(model)

    with pytest.raises(RecurrentInputError, match="banana"):
        enable_gated_delta_training(model, impl="banana")

    assert _linear_attn_identities(model) == before


# ---------------------------------------------------------------------------
# integration gates: routing, freeze, LoRA, eval mode, dtype ordering
# ---------------------------------------------------------------------------

_STOCK_GATED_DELTA_UPDATE = "mlx_lm.models.qwen3_5.gated_delta_update"
"""qwen3_5 does `from .gated_delta import gated_delta_update` at module level, so the
name is resolved in the qwen3_5 module's OWN namespace at call time. Patching
`mlx_lm.models.gated_delta.gated_delta_update` would never bite -- qwen3_5 already
holds its own reference by the time any layer calls it -- and `gated_delta_kernel` is
not imported into the qwen3_5 namespace at all."""


def test_forward_routes_through_the_op_not_stock_gated_delta_update():
    # Catches: enable_gated_delta_training wrapping a layer whose forward still falls
    # through to mlx-lm's own gated_delta_update instead of this project's op (the one
    # thing the proxy exists to change). See the control below for proof this patch
    # target genuinely intercepts the call.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    enable_gated_delta_training(model)
    x = mx.random.randint(0, 128, (1, 4))

    with mock.patch(_STOCK_GATED_DELTA_UPDATE, side_effect=AssertionError("poisoned")):
        out = model(x)
        mx.eval(out)  # must not raise: the enabled layer never reaches the stock call


def test_poison_control_unwrapped_layer_raises():
    # Control for the routing-proof test above. Without this, a poison test that never
    # fires (a typo'd patch target, or a target that resolves to a module the executed
    # code path doesn't actually import from) looks identical to one that correctly
    # passes -- this proves the patch target bites on the stock (un-enabled) layer.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    x = mx.random.randint(0, 128, (1, 4))

    with (
        mock.patch(_STOCK_GATED_DELTA_UPDATE, side_effect=AssertionError("poisoned")),
        pytest.raises(AssertionError, match="poisoned"),
    ):
        model(x)


@pytest.mark.parametrize("freeze_first", [True, False])
def test_freeze_state_carries(freeze_first: bool):
    # Catches: the proxy dropping A_log/dt_bias's frozen state in either construction
    # order -- if either silently became trainable, it would land in a saved adapter
    # file and contaminate a later convergence comparison against the stock model.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    keys_before = set(dict(tree_flatten(model.parameters())).keys())

    if freeze_first:
        model.freeze()
        enable_gated_delta_training(model)
    else:
        enable_gated_delta_training(model)
        model.freeze()

    keys_after = set(dict(tree_flatten(model.parameters())).keys())
    assert keys_after == keys_before
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable == {}


_LORA_CONFIG = {
    "rank": 4,
    "scale": 10.0,
    "dropout": 0.0,
    "keys": {"linear_attn.in_proj_qkv", "linear_attn.in_proj_b"},
}


@pytest.mark.parametrize("enable_first", [True, False])
def test_lora_attach_either_order(enable_first: bool):
    # Catches: the proxy renaming or duplicating linear_attn's submodules in a way that
    # breaks LoRA target discovery (named_modules() by path) in one of the two
    # construction orders -- the proxy holds the original submodules under their
    # original names precisely so linear_attn.in_proj_qkv/in_proj_b survive either way.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)

    if enable_first:
        enable_gated_delta_training(model)
        linear_to_lora_layers(model, num_layers=len(model.layers), config=_LORA_CONFIG)
    else:
        linear_to_lora_layers(model, num_layers=len(model.layers), config=_LORA_CONFIG)
        enable_gated_delta_training(model)

    linear_layer = next(layer for layer in model.language_model.model.layers if layer.is_linear)
    assert isinstance(linear_layer.linear_attn, GatedDeltaTrainingProxy)
    assert isinstance(linear_layer.linear_attn.in_proj_qkv, LoRALinear)
    assert isinstance(linear_layer.linear_attn.in_proj_b, LoRALinear)

    x = mx.random.randint(0, 128, (1, 4))
    out = model(x)
    mx.eval(out)
    assert out.shape[:2] == (1, 4)


def test_eval_mode_forward_works():
    # Catches: a refusal keyed on self.training instead of on cache-is-not-None --
    # mlx-lm's trainer flips the model to eval() for validation at iteration 1 by
    # default and back to train() afterward, so a training-keyed refusal would crash
    # the first validation batch of every fine-tune.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    enable_gated_delta_training(model)
    model.eval()

    x = mx.random.randint(0, 128, (1, 4))
    out = model(x, cache=None)
    mx.eval(out)
    assert out.shape[:2] == (1, 4)


def test_a_log_fp32_after_set_dtype_then_enable():
    # Catches: the forward guard (mis)treating a correctly dtype-protected checkpoint
    # as corrupted. `Module.set_dtype`'s predicate is called with a DTYPE, never a
    # path, so honouring qwen3_5's own path-keyed cast_predicate needs a
    # tree_map_with_path composition -- mirroring mlx_lm.convert's own pattern --
    # rather than passing cast_predicate straight to set_dtype's predicate kwarg
    # (which raises: cast_predicate expects a path string and set_dtype calls its
    # predicate with a dtype). Done BEFORE enable, this is the supported order.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    cast_predicate = model.cast_predicate

    def _cast(path: str, value: mx.array) -> mx.array:
        if cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(_cast, model.parameters()))
    enable_gated_delta_training(model)

    linear_layer = next(layer for layer in model.language_model.model.layers if layer.is_linear)
    assert linear_layer.linear_attn.A_log.dtype == mx.float32

    x = mx.random.randint(0, 128, (1, 4))
    out = model(x)
    mx.eval(out)


def test_a_log_guard_fires_when_set_dtype_runs_after_enable():
    # Catches: silent precision loss on the gating scalar A_log when a caller runs
    # this project's own compute-dtype step (model.set_dtype(mx.bfloat16), the plain
    # default predicate) AFTER enabling gated-delta training. set_dtype's default
    # predicate is dtype-keyed, not path-keyed, so it downcasts A_log along with
    # everything else; the guard must raise loud instead of silently training on a
    # corrupted gate, and its message must name the fix.
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    enable_gated_delta_training(model)
    model.set_dtype(mx.bfloat16)

    x = mx.random.randint(0, 128, (1, 4))
    with pytest.raises(RecurrentInputError, match="BEFORE enable_gated_delta_training"):
        model(x)
