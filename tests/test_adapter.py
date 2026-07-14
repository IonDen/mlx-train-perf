"""mlx-lm adapter tests.

Requires the optional `mlx-lm` extra; the whole module is skipped (not failed) when it
is absent, so the default lane still passes on a bare `pip install mlx-train-perf`.
"""
import sys

import mlx.core as mx
import pytest
from mlx import nn

pytest.importorskip("mlx_lm")

import mlx_lm
from mlx_lm.models import llama, qwen2, qwen3
from mlx_lm.tuner import trainer as t

from mlx_train_perf.adapters.mlx_lm import (
    _head_from_module,
    _quantized_head,
    _tied_head_from_embedding,
    make_loss_fn,
    split_model,
)
from mlx_train_perf.core.loss import DenseHead, QuantizedHead
from mlx_train_perf.errors import AdapterError, MissingDependencyError


def _tiny_llama(*, tie_word_embeddings: bool = False) -> llama.Model:
    args = llama.ModelArgs(
        model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=tie_word_embeddings,
    )
    return llama.Model(args)


def _tiny_qwen2(*, tie_word_embeddings: bool = False) -> qwen2.Model:
    args = qwen2.ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=tie_word_embeddings,
    )
    return qwen2.Model(args)


def _tiny_qwen3(*, tie_word_embeddings: bool = False) -> qwen3.Model:
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=tie_word_embeddings,
        head_dim=16, max_position_embeddings=64,
    )
    return qwen3.Model(args)


# ---------------------------------------------------------------------------
# split_model
# ---------------------------------------------------------------------------

def test_split_yields_trunk_and_dense_head() -> None:
    model = _tiny_llama()
    trunk, head = split_model(model)
    x = mx.random.randint(0, 256, (2, 8))
    hidden = trunk(x)
    assert hidden.shape == (2, 8, 64)
    assert isinstance(head, DenseHead)
    assert head.weight.shape == (256, 64)
    assert head.trainable  # a fresh nn.Linear is never frozen by default


def test_split_yields_qwen3_trunk_and_head() -> None:
    model = _tiny_qwen3()
    trunk, head = split_model(model)
    x = mx.random.randint(0, 256, (2, 8))
    hidden = trunk(x)
    assert hidden.shape == (2, 8, 64)
    assert isinstance(head, DenseHead)
    assert head.weight.shape == (256, 64)


def test_split_yields_qwen2_trunk_and_head() -> None:
    # 0016: the Qwen2.5 family (qwen2 architecture) -- module layout matches llama's
    # (inner .model trunk; lm_head only when untied).
    model = _tiny_qwen2()
    trunk, head = split_model(model)
    x = mx.random.randint(0, 256, (2, 8))
    hidden = trunk(x)
    assert hidden.shape == (2, 8, 64)
    assert isinstance(head, DenseHead)
    assert head.weight.shape == (256, 64)


def test_split_qwen2_tied_head_reuses_embedding_weight() -> None:
    # Qwen2.5's small checkpoints (0.5B/1.5B) ship tied -- the tied arm is the common one.
    model = _tiny_qwen2(tie_word_embeddings=True)
    _trunk, head = split_model(model)
    assert isinstance(head, DenseHead)
    assert head.weight is model.model.embed_tokens.weight
    assert head.weight.shape == (256, 64)


def test_split_tied_head_reuses_embedding_weight() -> None:
    model = _tiny_llama(tie_word_embeddings=True)
    _trunk, head = split_model(model)
    assert isinstance(head, DenseHead)
    # Tied: the head IS the input embedding table, not a separate lm_head.
    assert head.weight is model.model.embed_tokens.weight
    assert head.weight.shape == (256, 64)
    assert head.trainable  # embed_tokens is unfrozen by default


def test_split_quantized_head_reads_affine_fields() -> None:
    model = _tiny_llama()
    model.lm_head = model.lm_head.to_quantized(group_size=64, bits=4)
    _trunk, head = split_model(model)
    assert isinstance(head, QuantizedHead)
    assert head.group_size == 64
    assert head.bits == 4
    assert head.biases is not None
    # 32 // bits values packed per uint32 word -- must round-trip to hidden_size=64.
    assert head.w_q.shape[-1] * (32 // head.bits) == 64


def test_split_rejects_non_affine_quantized_head() -> None:
    model = _tiny_llama()
    model.lm_head = model.lm_head.to_quantized(mode="mxfp4")
    with pytest.raises(AdapterError) as ei:
        split_model(model)
    assert "affine" in str(ei.value).lower()


def test_split_tied_quantized_head_reads_affine_fields() -> None:
    model = _tiny_llama(tie_word_embeddings=True)
    model.model.embed_tokens = model.model.embed_tokens.to_quantized(
        group_size=64, bits=4
    )
    _trunk, head = split_model(model)
    assert isinstance(head, QuantizedHead)
    assert head.group_size == 64
    assert head.bits == 4


def test_quantized_head_rejects_missing_biases() -> None:
    """Defensive branch: an affine-mode quantized module without biases should never
    occur via `mx.quantize` (affine always returns biases), but `_quantized_head` must
    not silently construct a `QuantizedHead` with a `None` biases field either way."""
    class _AffineNoBiases:
        mode = "affine"
        biases = None
        weight = mx.zeros((4, 4), dtype=mx.uint32)
        scales = mx.ones((4, 1))
        group_size = 64
        bits = 4

    with pytest.raises(AdapterError) as ei:
        _quantized_head(_AffineNoBiases())
    assert "biases" in str(ei.value).lower()


def test_head_from_module_rejects_unsupported_module_type() -> None:
    with pytest.raises(AdapterError) as ei:
        _head_from_module(object())
    assert "unsupported head module type" in str(ei.value).lower()


def test_tied_head_from_embedding_rejects_unsupported_module_type() -> None:
    with pytest.raises(AdapterError) as ei:
        _tied_head_from_embedding(object())
    assert "unsupported embedding module type" in str(ei.value).lower()


def test_unsupported_architecture_is_typed_error() -> None:
    class NotAModel:
        pass

    with pytest.raises(AdapterError) as ei:
        split_model(NotAModel())
    message = str(ei.value).lower()
    assert "llama" in message
    assert "qwen2" in message
    assert "qwen3" in message


def test_missing_mlx_lm_dependency_is_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate an environment without the `mlx-lm` extra installed: `import mlx_lm`
    # raises ImportError when the module is present in sys.modules as None (CPython's
    # documented mechanism for "this import previously failed").
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(MissingDependencyError) as ei:
        split_model(object())
    assert "mlx-lm" in str(ei.value)


# ---------------------------------------------------------------------------
# make_loss_fn: mask/denominator parity against mlx-lm's own default_loss
# ---------------------------------------------------------------------------

def test_loss_and_ntoks_match_stock_on_same_batch() -> None:
    model = _tiny_llama()
    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    # Two-column (offset, length) rows per the installed trainer.py:170 contract --
    # both rows fully unmasked here (offset=0).
    lengths = mx.array([[0, 12], [0, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")  # default lane: no Metal dependency
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)

    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    # Measured exactly 0.0 on this seed/shape (fp32, single chunk); 1e-5 leaves headroom
    # for floating-point noise on other shapes/hardware without padding past "tight".
    assert abs(ours.item() - stock.item()) < 1e-5


def test_loss_and_ntoks_match_stock_on_qwen2() -> None:
    # 0016: the qwen2 masking/loss parity vs stock default_loss (mirrors the llama/qwen3
    # parity tests -- same trainer two-column (offset, length) contract).
    model = _tiny_qwen2()
    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[3, 12], [0, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)

    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    assert abs(ours.item() - stock.item()) < 1e-5


def test_loss_and_ntoks_match_stock_with_prompt_offset() -> None:
    """Nonzero offset (prompt masking): row 0's first 3 steps are masked out, as if
    they were prompt tokens excluded from the loss."""
    model = _tiny_llama()
    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[3, 12], [0, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)

    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    assert abs(ours.item() - stock.item()) < 1e-5


def test_loss_and_ntoks_match_stock_with_fully_masked_row() -> None:
    """One row entirely masked (offset beyond the sequence length) -- the batch-level
    ntoks denominator must still come only from the unmasked row, matching stock, with
    no NaN from the masked row itself."""
    model = _tiny_llama()
    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[0, 12], [20, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)

    assert mx.isfinite(ours).item()
    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    assert abs(ours.item() - stock.item()) < 1e-5


def test_loss_and_ntoks_match_stock_with_all_rows_fully_masked() -> None:
    """Every row masked out (ntoks == 0 for the whole batch): stock `default_loss`
    divides by a zero `ntoks` and produces `nan` -- not an exception. This pins that
    our adapter matches that exact crash-parity behavior (nan loss, ntoks == 0) rather
    than, say, silently returning 0 or raising."""
    model = _tiny_llama()
    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[20, 7], [20, 7]])  # offset=20 > seq len -- every row masked

    loss_fn = make_loss_fn(model, impl="chunked")
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)

    assert int(ntoks_ours.item()) == 0
    assert int(ntoks_stock.item()) == 0
    assert mx.isnan(ours).item()
    assert mx.isnan(stock).item()


def test_loss_and_ntoks_match_stock_with_quantized_head() -> None:
    """Quantized-head parity: our adapter runs `mx.quantized_matmul` directly against
    the packed int4 head (see `make_chunked_quantized`); the stock side uses a plain
    dense `nn.Linear` head whose weight is the DEQUANTIZED reconstruction of the exact
    same quantized weights, sharing the same trunk object so only the head mechanism
    differs. Measured exactly 0.0 on this seed/shape (fp32, single chunk, vocab=256 <<
    the 8192 chunk tile) -- matching core's own test_kernel_quant_parity.py precedent of
    still pinning a small headroom tolerance rather than asserting exact equality even
    when the measured diff is literally 0.0."""
    model = _tiny_llama()
    model.lm_head = model.lm_head.to_quantized(group_size=64, bits=4)

    stock_model = _tiny_llama()
    stock_model.model = model.model  # share the trunk object: isolates the head only
    stock_model.lm_head.weight = mx.dequantize(
        model.lm_head.weight, model.lm_head.scales, model.lm_head.biases,
        group_size=model.lm_head.group_size, bits=model.lm_head.bits,
    )

    mx.random.seed(1)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[0, 12], [0, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(stock_model, batch, lengths)

    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    assert abs(ours.item() - stock.item()) < 1e-5


def test_masked_positions_contribute_zero_gradient() -> None:
    """Masked tokens contribute zero loss AND zero gradient. Deterministic construction:
    token 254 sits at input position 1 (0-based) and token 255 at input position 4;
    under causal attention, only an unmasked step >= 2 (1-based) can see position 1's
    embedding, and only an unmasked step >= 5 can see position 4's. `lengths=[[2, 2]]`
    unmasks ONLY step 2, so 254's embedding-row gradient must be nonzero (it feeds the
    one live loss term) while 255's must be EXACTLY zero (causally excluded, not merely
    masked-and-still-connected)."""
    model = _tiny_llama()
    batch = mx.array([[1, 254, 2, 3, 255, 5, 6, 7, 8, 9]])
    lengths = mx.array([[2, 2]])

    loss_fn = make_loss_fn(model, impl="chunked")

    def scalar_loss(m: nn.Module) -> mx.array:
        loss, _ = loss_fn(m, batch, lengths)
        return loss

    _, grads = nn.value_and_grad(model, scalar_loss)(model)
    embed_grad = grads["model"]["embed_tokens"]["weight"]
    assert mx.abs(embed_grad[255]).max().item() == 0.0     # causally excluded: EXACTLY zero
    assert mx.abs(embed_grad[254]).max().item() > 0.0      # feeds the one unmasked step


def test_loss_fn_reflects_live_weight_updates() -> None:
    """mlx-lm's trainer calls `model.update(params)` immediately before invoking the
    loss (nn.value_and_grad's own source) -- so a loss closure that snapshots the head's
    weight array once at construction would silently keep training against a stale copy
    after the first optimizer step. Prove `make_loss_fn`'s closure re-reads the live
    model instead: mutate `lm_head.weight` directly (no optimizer involved) and confirm
    the loss changes and tracks an independently-computed stock reference over the
    mutated model."""
    model = _tiny_llama()
    mx.random.seed(3)
    batch = mx.random.randint(0, 256, (2, 12))
    lengths = mx.array([[0, 12], [0, 7]])

    loss_fn = make_loss_fn(model, impl="chunked")
    loss_before, _ = loss_fn(model, batch, lengths)

    new_weight = mx.random.normal(model.lm_head.weight.shape) * 0.1
    model.update({"lm_head": {"weight": new_weight}})

    loss_after, _ = loss_fn(model, batch, lengths)
    stock_after, _ = t.default_loss(model, batch, lengths)

    assert abs(loss_after.item() - loss_before.item()) > 1e-6
    assert abs(loss_after.item() - stock_after.item()) < 1e-5


# ---------------------------------------------------------------------------
# Gated smoke test: real (pre-downloaded) quantized model, one live training step.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_real_model_one_train_step() -> None:
    """--run-smoke: one `nn.value_and_grad` step through `make_loss_fn`'s resolved impl
    on a real quantized model -- loss and grads finite, peak memory under the session
    wired cap (`conftest._memory_guard`). Model: mlx-community/Qwen3-8B-4bit, expected to
    already be present in the local Hugging Face cache -- this test does not fetch it."""
    model, _tokenizer = mlx_lm.load("mlx-community/Qwen3-8B-4bit")
    mx.random.seed(7)
    batch = mx.random.randint(0, model.args.vocab_size, (1, 64))
    lengths = mx.array([[0, 64]])
    loss_fn = make_loss_fn(model)  # impl="auto" -> kernel on this (verified) mlx

    def scalar_loss(m: nn.Module) -> mx.array:
        loss, _ = loss_fn(m, batch, lengths)
        return loss

    loss, grads = nn.value_and_grad(model, scalar_loss)(model)
    mx.eval(loss, grads)

    assert mx.isfinite(loss).item()
    assert mx.get_peak_memory() < 20 * 1024**3  # matches the session wired cap


@pytest.mark.smoke
def test_real_qwen2_model_loss_parity() -> None:
    """--run-smoke (0016): masking/loss parity vs stock `default_loss` on a REAL Qwen2.5
    checkpoint (tied embeddings -- the family's common shipping shape). Model:
    mlx-community/Qwen2.5-0.5B-Instruct-bf16, expected pre-downloaded (never fetched)."""
    model, _tokenizer = mlx_lm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    assert type(model).__module__ == "mlx_lm.models.qwen2"
    mx.random.seed(7)
    batch = mx.random.randint(0, model.args.vocab_size, (2, 32))
    lengths = mx.array([[5, 32], [0, 20]])   # prompt offset + ragged length

    loss_fn = make_loss_fn(model)
    ours, ntoks_ours = loss_fn(model, batch, lengths)
    stock, ntoks_stock = t.default_loss(model, batch, lengths)
    mx.eval(ours, ntoks_ours, stock, ntoks_stock)

    assert int(ntoks_ours.item()) == int(ntoks_stock.item())
    # Measured 2.077e-03 on this seed/shape (bf16 checkpoint at untrained-random-token
    # loss ~15.09 -- bf16-ULP class at that magnitude); pinned at ~2x the measured worst,
    # the same headroom convention the attention parity pins use.
    assert abs(ours.item() - stock.item()) < 4e-3
