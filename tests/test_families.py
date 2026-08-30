"""qwen3_5 model-tree accessor tests.

Most cases use a small fake object tree so the default lane exercises them
without the optional `mlx-lm` extra; only the two tests that need a real
`mlx_lm.models.qwen3_5.Model` are guarded with `pytest.importorskip`.
"""
import pytest

from mlx_train_perf.errors import UnsupportedRecurrentError
from mlx_train_perf.families import is_qwen35, model_head, text_args, text_model


class _FakeTextArgs:
    """Stands in for `TextModelArgs` -- carries `tie_word_embeddings`."""

    def __init__(self, *, tie_word_embeddings: bool) -> None:
        self.tie_word_embeddings = tie_word_embeddings


class _FakeTrunk:
    """Stands in for `Qwen3_5TextModel`, the module at `model.language_model.model`."""

    def __init__(self) -> None:
        self.embed_tokens = "EMBED_TOKENS_SENTINEL"


class _FakeTextModel:
    """Stands in for `TextModel`, the module at `model.language_model`."""

    def __init__(self, *, tie_word_embeddings: bool, with_lm_head: bool = True) -> None:
        self.args = _FakeTextArgs(tie_word_embeddings=tie_word_embeddings)
        self.model = _FakeTrunk()
        if with_lm_head:
            self.lm_head = "LM_HEAD_SENTINEL"


class _FakeQwen35Model:
    def __init__(self, *, tie_word_embeddings: bool = True, with_lm_head: bool = True) -> None:
        self.language_model = _FakeTextModel(
            tie_word_embeddings=tie_word_embeddings, with_lm_head=with_lm_head
        )


_FakeQwen35Model.__module__ = "mlx_lm.models.qwen3_5"


class _FakeQwen35MoeModel(_FakeQwen35Model):
    """Subclasses the qwen3_5 fake exactly like the real `qwen3_5_moe.Model`
    subclasses `qwen3_5.Model` -- same structure, different `__module__`."""


_FakeQwen35MoeModel.__module__ = "mlx_lm.models.qwen3_5_moe"


class _FakeLlamaModel:
    """A structurally different, non-qwen3_5 tree: a flat `.model`/`.args`,
    no `.language_model` at all."""

    def __init__(self) -> None:
        self.model = _FakeTrunk()
        self.args = _FakeTextArgs(tie_word_embeddings=False)


_FakeLlamaModel.__module__ = "mlx_lm.models.llama"


def _tiny_real_qwen35(*, tie_word_embeddings: bool = True):
    qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")
    text_config = {
        "model_type": "qwen3_5",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 48,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": False,
        # Both layers land on the linear (GatedDeltaNet) branch at this
        # interval, so construction never needs the qwen3_next Attention block.
        "full_attention_interval": 4,
    }
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text_config)
    return qwen3_5.Model(args)


# ---------------------------------------------------------------------------
# is_qwen35
# ---------------------------------------------------------------------------

def test_is_qwen35_true_for_qwen3_5_module():
    assert is_qwen35(_FakeQwen35Model()) is True


def test_is_qwen35_false_for_qwen3_5_moe_subclass():
    # Catches: an isinstance/issubclass-based check -- qwen3_5_moe.Model
    # subclasses qwen3_5.Model in the real library, so such a check would
    # wrongly admit the MoE variant this release deliberately refuses.
    assert is_qwen35(_FakeQwen35MoeModel()) is False


def test_is_qwen35_false_for_unrelated_module():
    assert is_qwen35(_FakeLlamaModel()) is False


def test_is_qwen35_true_for_real_qwen3_5_model():
    model = _tiny_real_qwen35()
    assert type(model).__module__ == "mlx_lm.models.qwen3_5"
    assert is_qwen35(model) is True


# ---------------------------------------------------------------------------
# text_args
# ---------------------------------------------------------------------------

def test_text_args_reads_the_nested_text_model_args():
    # Catches: reading `model.args` directly -- on qwen3_5 that is the
    # two-field ModelArgs(model_type, text_config) wrapper with no
    # `tie_word_embeddings` field at all.
    model = _FakeQwen35Model(tie_word_embeddings=True)
    assert text_args(model) is model.language_model.args


def test_text_args_raises_typed_error_for_non_qwen35_model():
    with pytest.raises(UnsupportedRecurrentError):
        text_args(_FakeLlamaModel())


# ---------------------------------------------------------------------------
# text_model
# ---------------------------------------------------------------------------

def test_text_model_reads_the_nested_trunk():
    model = _FakeQwen35Model()
    assert text_model(model) is model.language_model.model


def test_text_model_raises_typed_error_for_non_qwen35_model():
    with pytest.raises(UnsupportedRecurrentError):
        text_model(_FakeLlamaModel())


# ---------------------------------------------------------------------------
# model_head
# ---------------------------------------------------------------------------

def test_model_head_returns_embedding_and_tied_true():
    model = _FakeQwen35Model(tie_word_embeddings=True, with_lm_head=False)
    head, tied = model_head(model)
    assert head is model.language_model.model.embed_tokens
    assert tied is True


def test_model_head_ignores_a_stray_lm_head_when_tied():
    # Catches: branching on `hasattr(lm, "lm_head")` instead of
    # `text_args(model).tie_word_embeddings` -- the real tied checkpoint has
    # no `lm_head` attribute at all, so a hasattr-based branch would
    # coincidentally work on the real model but silently prefer a leftover
    # lm_head here.
    model = _FakeQwen35Model(tie_word_embeddings=True, with_lm_head=True)
    head, tied = model_head(model)
    assert head is model.language_model.model.embed_tokens
    assert tied is True


def test_model_head_returns_lm_head_and_tied_false():
    model = _FakeQwen35Model(tie_word_embeddings=False, with_lm_head=True)
    head, tied = model_head(model)
    assert head is model.language_model.lm_head
    assert tied is False


def test_model_head_raises_typed_error_when_untied_lm_head_missing():
    model = _FakeQwen35Model(tie_word_embeddings=False, with_lm_head=False)
    with pytest.raises(UnsupportedRecurrentError):
        model_head(model)


def test_model_head_raises_typed_error_for_non_qwen35_model():
    with pytest.raises(UnsupportedRecurrentError):
        model_head(_FakeLlamaModel())


def test_model_head_tied_branch_on_real_qwen3_5_model():
    model = _tiny_real_qwen35(tie_word_embeddings=True)
    # Matches the real mlx-community/Qwen3.5-0.8B-4bit checkpoint: tied means
    # no lm_head attribute exists on the text model at all.
    assert not hasattr(model.language_model, "lm_head")
    head, tied = model_head(model)
    assert tied is True
    assert head is model.language_model.model.embed_tokens


# ---------------------------------------------------------------------------
# shared typed-error contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("accessor", [text_args, text_model, model_head])
def test_accessors_raise_typed_not_attributeerror(accessor):
    # Catches: an accessor that skips the module check and reaches straight
    # for `model.language_model.*`, which would surface as a bare
    # AttributeError on a differently-shaped (e.g. llama) model.
    with pytest.raises(UnsupportedRecurrentError):
        accessor(_FakeLlamaModel())
