"""Builds a tiny, structurally-real `mlx_lm.models.qwen3_5.Model` for tests.

`mlx_lm`'s own `TextModelArgs` defaults are 0.8B-scale (`hidden_size=4096`,
`intermediate_size=14336`, `num_attention_heads=32` against
`num_key_value_heads=8` -- which does not divide evenly and breaks SDPA at
the default `num_attention_heads=2`). This module overrides every field that
matters for construction with a small explicit shape, and exposes only the
knobs later tests need to vary: how many decoder layers, how often the
full-attention layer recurs, and whether the output head is tied.

`Model.__init__` calls `TextModelArgs.from_dict(args.text_config)`, so
`text_config` must be a plain dict -- and `BaseModelArgs.from_dict` silently
drops unknown keys, so a misspelled field becomes a silent default rather
than an error.
"""
from typing import Any

import mlx.core as mx
import pytest

_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_NUM_ATTENTION_HEADS = 2
_NUM_KEY_VALUE_HEADS = 1
_VOCAB_SIZE = 128
_LINEAR_HEAD_COUNT = 2
_LINEAR_HEAD_DIM = 16
_LINEAR_CONV_KERNEL_DIM = 4


def tiny_qwen35(
    *,
    full_attention_interval: int = 2,
    num_layers: int = 4,
    tie_word_embeddings: bool = True,
    seed: int = 0,
) -> Any:
    """A seeded, fully-evaluated tiny qwen3_5 `Model`.

    Construction alone never runs `Qwen3_5TextModel.__call__`, so any
    `num_layers`/`full_attention_interval` pair builds cleanly -- including
    an all-linear shape (`num_layers < full_attention_interval`) callers use
    when they need construction to skip the attention block entirely. A
    *forward* pass is different: `Qwen3_5TextModel.__call__` indexes
    `cache[full_attention_interval - 1]`, so calling this model requires
    `num_layers >= full_attention_interval` or it raises `IndexError`.

    Skips (via `pytest.importorskip`) rather than raising when the optional
    `mlx-lm` extra is not installed.
    """
    qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")
    text_config = {
        "model_type": "qwen3_5",
        "hidden_size": _HIDDEN_SIZE,
        "intermediate_size": _INTERMEDIATE_SIZE,
        "num_hidden_layers": num_layers,
        "num_attention_heads": _NUM_ATTENTION_HEADS,
        "rms_norm_eps": 1e-6,
        "vocab_size": _VOCAB_SIZE,
        "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
        "max_position_embeddings": 64,
        "linear_num_value_heads": _LINEAR_HEAD_COUNT,
        "linear_num_key_heads": _LINEAR_HEAD_COUNT,
        "linear_key_head_dim": _LINEAR_HEAD_DIM,
        "linear_value_head_dim": _LINEAR_HEAD_DIM,
        "linear_conv_kernel_dim": _LINEAR_CONV_KERNEL_DIM,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": False,
        "full_attention_interval": full_attention_interval,
    }
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text_config)
    mx.random.seed(seed)
    model = qwen3_5.Model(args)
    mx.eval(model.parameters())
    return model
