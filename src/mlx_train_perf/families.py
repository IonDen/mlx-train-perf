"""qwen3_5 model-tree accessors.

Qwen3.5 nests its trunk differently from the llama/qwen2/qwen3 families
`mlx_train_perf.adapters.mlx_lm.split_model` already supports. Verified
against the installed mlx-lm==0.31.3 source (`mlx_lm/models/qwen3_5.py`):
`model.args` is a two-field `ModelArgs(model_type, text_config)` wrapper with
none of the usual per-model fields, and the trunk that computes hidden states
lives two levels down, at `model.language_model.model` (a
`Qwen3_5TextModel`), not at `model.model`. The per-model config -- including
`tie_word_embeddings` -- lives at `model.language_model.args` (a
`TextModelArgs`), not `model.args`.

`mlx_lm.models.qwen3_5_moe.Model` SUBCLASSES `mlx_lm.models.qwen3_5.Model`
(same nesting, different `__module__`) but is out of scope here; an
`isinstance`/`issubclass` check would wrongly admit it, so `is_qwen35`
compares `type(model).__module__` exactly instead.

The real `mlx-community/Qwen3.5-0.8B-4bit` checkpoint ships
`tie_word_embeddings=True`, in which case its text model carries no
`lm_head` attribute at all -- tied-ness is read from
`text_args(model).tie_word_embeddings`, never from `hasattr(lm, "lm_head")`.

These are pure `getattr` navigation with typed raises: no `mlx_lm` import at
module level, and every failure -- an unrelated model family or an
unexpectedly-shaped qwen3_5 tree -- raises `UnsupportedRecurrentError`, never
a bare `AttributeError`.
"""
from typing import Any

from mlx import nn

from mlx_train_perf.errors import UnsupportedRecurrentError

_QWEN35_MODULE = "mlx_lm.models.qwen3_5"


def is_qwen35(model: Any) -> bool:
    """True iff `model` is exactly a qwen3_5 `Model`.

    An exact module-name comparison, never `isinstance`/`issubclass` -- see
    the module docstring: the MoE variant subclasses this one and must NOT
    be admitted.
    """
    return type(model).__module__ == _QWEN35_MODULE


def _require_qwen35(model: Any) -> None:
    if not is_qwen35(model):
        raise UnsupportedRecurrentError(
            f"unsupported model architecture (module {type(model).__module__!r}); "
            "expected a qwen3_5 model"
        )


def text_args(model: Any) -> Any:
    """The inner `TextModelArgs` at `model.language_model.args`.

    NOT `model.args` -- on qwen3_5 that is the two-field wrapper without
    `tie_word_embeddings` or any other per-model field.
    """
    _require_qwen35(model)
    try:
        return model.language_model.args
    except AttributeError as exc:
        raise UnsupportedRecurrentError(
            "qwen3_5 model is missing model.language_model.args"
        ) from exc


def text_model(model: Any) -> Any:
    """The hidden-state-computing trunk (`Qwen3_5TextModel`) at
    `model.language_model.model`."""
    _require_qwen35(model)
    try:
        return model.language_model.model
    except AttributeError as exc:
        raise UnsupportedRecurrentError(
            "qwen3_5 model is missing model.language_model.model"
        ) from exc


def model_head(model: Any) -> tuple[nn.Module, bool]:
    """The output-projection module and whether it is tied to the input
    embedding table.

    Returns the module itself (`nn.Linear`/`nn.QuantizedLinear` when untied,
    `nn.Embedding`/`nn.QuantizedEmbedding` when tied) rather than a bound
    callable, matching what `adapters/mlx_lm.py`'s `_head_from_module` and
    `_tied_head_from_embedding` expect. Tied-ness comes from
    `text_args(model).tie_word_embeddings` -- never `hasattr(lm, "lm_head")`,
    which the real tied checkpoint (no `lm_head` attribute at all) would
    read backwards.
    """
    args = text_args(model)
    try:
        tied = bool(args.tie_word_embeddings)
    except AttributeError as exc:
        raise UnsupportedRecurrentError(
            "qwen3_5 model's text args are missing tie_word_embeddings"
        ) from exc
    if tied:
        try:
            return text_model(model).embed_tokens, True
        except AttributeError as exc:
            raise UnsupportedRecurrentError(
                "qwen3_5 model is missing model.language_model.model.embed_tokens"
            ) from exc
    try:
        return model.language_model.lm_head, False
    except AttributeError as exc:
        raise UnsupportedRecurrentError(
            "qwen3_5 model is missing model.language_model.lm_head"
        ) from exc
