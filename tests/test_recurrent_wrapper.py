"""`enable_gated_delta_training` selection + refusal matrix (no forward yet).

Layer selection is structural (`layer.is_linear` + a class check on `layer.linear_attn`,
cross-checked against the raw `text_config.layer_types`, when present) -- see
`recurrent/wrapper.py`'s module docstring for the verified mlx-lm facts this rests on. The
proxy's forward pass lands in a later change; here it only needs to refuse a cache before
raising `NotImplementedError`.
"""
from typing import Any

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten
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
    # attribute (doubling every one of its parameter keys).
    model = tiny_qwen35(full_attention_interval=2, num_layers=4)
    keys_before = set(dict(tree_flatten(model.parameters())).keys())

    enable_gated_delta_training(model)

    keys_after = set(dict(tree_flatten(model.parameters())).keys())
    assert keys_after == keys_before


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
