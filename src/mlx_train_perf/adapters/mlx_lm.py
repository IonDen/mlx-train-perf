"""mlx-lm trainer adapter: `split_model` + `make_loss_fn`.

Verified against the installed mlx-lm==0.31.3 (mlx==0.31.2) source, 2026-07-04:

- **Trainer loss contract** (site-packages/mlx_lm/tuner/trainer.py:86-99, `default_loss`)::

      inputs = batch[:, :-1]
      targets = batch[:, 1:]
      logits = model(inputs)
      steps = mx.arange(1, targets.shape[1] + 1)
      mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
      ce = nn.losses.cross_entropy(logits, targets) * mask
      ntoks = mask.sum()
      ce = ce.astype(mx.float32).sum() / ntoks
      return ce, ntoks

  `lengths` is a TWO-COLUMN `(offset, length)` int array, one row per batch example --
  built by `iterate_batches` (trainer.py:170) as `mx.array(list(zip(offsets, lengths)))`.
  This module reproduces that mask and denominator exactly (mirrors the contract; does
  not import `trainer.py` itself, to keep `mlx_lm` a lazy import here).

- **Injection point**: `mlx_lm.tuner.trainer.train(..., loss: callable = default_loss)`,
  invoked as `nn.value_and_grad(model, loss)(model, *batch)` (trainer.py:218-240,
  240). Per the installed `mlx.nn.value_and_grad` source, its wrapped function calls
  `model.update(params)` **before** calling `loss(model, ...)` -- so by the time our
  loss runs, `model` already carries that step's (gradient-updated, on steps after the
  first) parameters. A loss closure that snapshots a trainable head's weight array once
  at construction would keep computing against that first snapshot forever -- the
  weight *attribute* on `model` gets reassigned by `model.update`, but a frozen
  dataclass field holding the old array object does not follow the reassignment (this
  was confirmed empirically against the installed mlx before relying on it). `loss_fn`
  therefore calls `split_model(model)` fresh on every invocation rather than once at
  `make_loss_fn` construction time.

- **Model structure** (mlx_lm/models/llama.py, mlx_lm/models/qwen2.py,
  mlx_lm/models/qwen3.py -- identical shape in all three families): `Model.__call__`
  computes hidden states via `self.model(x)` (the inner `LlamaModel` / `Qwen2Model` /
  `Qwen3Model`) and projects with `self.lm_head` (`nn.Linear` or `nn.QuantizedLinear`)
  unless `args.tie_word_embeddings` (qwen2's DEFAULT -- the Qwen2.5 0.5B/1.5B
  checkpoints ship tied; `lm_head` only exists untied), in which case it projects via
  `self.model.embed_tokens.as_linear(hidden)` (`nn.Embedding` / `nn.QuantizedEmbedding`).
  Both quantized module types expose `weight, scales, biases, group_size, bits, mode`
  directly; `biases` is `None` and `mode` is not `"affine"` for the non-affine
  quantization modes (`mxfp4`/`mxfp8`/`nvfp4`) that this project's kernel and chunked
  paths do not implement.

Only the Llama, Qwen2 (Qwen2.5 family) and Qwen3 model families are supported (matched
by `type(model).__module__`); anything else raises `AdapterError` naming the support
list.
"""
from collections.abc import Callable
from typing import Any, Literal

import mlx.core as mx
from mlx import nn

from mlx_train_perf.core.loss import (
    DenseHead,
    HeadRef,
    QuantizedHead,
    linear_cross_entropy,
    resolve_impl,
    tied_head,
)
from mlx_train_perf.errors import AdapterError, MissingDependencyError

# Keyed by the exact `type(model).__module__` mlx-lm uses for each family (verified
# against the installed mlx_lm.models.llama / qwen2 / qwen3 above).
_SUPPORTED_FAMILIES: dict[str, str] = {
    "llama": "mlx_lm.models.llama",
    "qwen2": "mlx_lm.models.qwen2",
    "qwen3": "mlx_lm.models.qwen3",
}


def _require_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "mlx-lm is required for mlx_train_perf.adapters.mlx_lm; install the optional "
            "'mlx-lm' extra (pip install 'mlx-train-perf[mlx-lm]')"
        ) from exc


def _quantized_head(module: nn.Module) -> QuantizedHead:
    # `nn` resolves to `Any` here (see the pyproject.toml mypy override for `mlx.nn`),
    # so these attribute reads are unchecked -- correctness is pinned by
    # test_split_quantized_head_reads_affine_fields / test_split_rejects_non_affine_*.
    mode = module.mode
    if mode != "affine":
        raise AdapterError(
            f"quantized head/embedding mode {mode!r} is not supported; only "
            "mode='affine' quantized heads are supported (the only mode this "
            "project's kernel and chunked paths implement)"
        )
    biases = module.biases
    if biases is None:
        raise AdapterError(
            "quantized head/embedding has no biases (mode without biases); "
            "mlx_train_perf's QuantizedHead requires biases"
        )
    return QuantizedHead(
        w_q=module.weight,
        scales=module.scales,
        biases=biases,
        group_size=module.group_size,
        bits=module.bits,
    )


def _head_from_module(module: nn.Module) -> HeadRef:
    """`model.lm_head` case: a dedicated (untied) output projection."""
    if isinstance(module, nn.QuantizedLinear):
        return _quantized_head(module)
    if isinstance(module, nn.Linear):
        # nn.Module.trainable_parameters is itself untyped in mlx's source.
        trainable = "weight" in module.trainable_parameters()  # type: ignore[no-untyped-call]
        return DenseHead(weight=module.weight, trainable=trainable)
    raise AdapterError(
        f"unsupported head module type {type(module).__name__!r}; expected "
        "nn.Linear or nn.QuantizedLinear"
    )


def _tied_head_from_embedding(embedding: nn.Module) -> HeadRef:
    """`model.model.embed_tokens` case, used as the head when weights are tied."""
    if isinstance(embedding, nn.QuantizedEmbedding):
        return _quantized_head(embedding)
    if isinstance(embedding, nn.Embedding):
        # nn.Module.trainable_parameters is itself untyped in mlx's source.
        trainable = "weight" in embedding.trainable_parameters()  # type: ignore[no-untyped-call]
        return tied_head(embedding.weight, trainable=trainable)
    raise AdapterError(
        f"unsupported embedding module type {type(embedding).__name__!r}; expected "
        "nn.Embedding or nn.QuantizedEmbedding"
    )


def split_model(model: Any) -> tuple[Callable[[mx.array], mx.array], HeadRef]:
    """Split an mlx-lm `Model` into a hidden-state trunk and a `HeadRef`.

    `model` is typed `Any` deliberately: importing mlx-lm's model classes here (just
    for a type annotation) would defeat the point of `mlx_lm` being an optional, lazily
    imported dependency. Support is instead verified structurally, at call time.
    """
    _require_mlx_lm()
    module_name = type(model).__module__
    if module_name not in _SUPPORTED_FAMILIES.values():
        supported = ", ".join(sorted(_SUPPORTED_FAMILIES))
        raise AdapterError(
            f"unsupported model architecture (module {module_name!r}); "
            f"mlx_train_perf's mlx-lm adapter supports: {supported}"
        )
    inner = model.model  # the inner LlamaModel / Qwen3Model -- yields hidden states

    def trunk(x: mx.array) -> mx.array:
        return inner(x)  # type: ignore[no-any-return]

    if model.args.tie_word_embeddings:
        head = _tied_head_from_embedding(inner.embed_tokens)
    else:
        head = _head_from_module(model.lm_head)
    return trunk, head


def make_loss_fn(
    model: Any,
    *,
    impl: Literal["auto", "kernel", "chunked", "naive"] = "auto",
    allow_unverified_mlx: bool = False,
) -> Callable[[Any, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Build a loss callable matching mlx-lm's trainer contract:
    `loss(model, batch, lengths) -> (loss, ntoks)` (see the module docstring for the
    exact, version-cited contract this reproduces).

    Fails fast: an unsupported architecture (`AdapterError`) or a missing `mlx-lm`
    install (`MissingDependencyError`) is raised immediately, before any training step
    runs, rather than on the first call.
    """
    # Fail fast only -- the (trunk, head) pair itself is discarded. `loss_fn` below
    # re-derives both from the live `model` argument on every call (see the module
    # docstring for why a construction-time snapshot would go stale).
    split_model(model)

    # The kernel/chunked/naive decision depends on the hidden dtype and row count `n`,
    # neither of which is known until a real batch has flowed through the trunk -- so
    # it is resolved on the first call to `loss_fn` and cached here for every later
    # step (the decision itself does not vary across steps of the same training run).
    resolved_impl: Literal["kernel", "chunked", "naive"] | None = None

    def loss_fn(
        model_arg: Any, batch: mx.array, lengths: mx.array
    ) -> tuple[mx.array, mx.array]:
        nonlocal resolved_impl
        trunk, head = split_model(model_arg)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        hidden = trunk(inputs)
        if resolved_impl is None:
            n = hidden.shape[0] * hidden.shape[1]
            resolved_impl = resolve_impl(
                head=head, dtype=hidden.dtype, n=n, impl=impl,
                allow_unverified_mlx=allow_unverified_mlx,
            ).impl
        # validate_targets=False: mlx_lm's trainer wraps this step in mx.compile, which
        # forbids the range check's host sync; the trainer feeds in-range tokenizer ids, so
        # the check is both unusable here and unnecessary. This is what lets `ours` run
        # through the real compiled train() step, on equal footing with stock.
        nll = linear_cross_entropy(hidden, head, targets, impl=resolved_impl,
                                   reduction="none", validate_targets=False)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
        ntoks = mask.sum()
        loss = (nll * mask).astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


def make_packed_loss_fn(
    model: Any,
    *,
    impl: Literal["auto", "kernel", "chunked", "naive"] = "auto",
    allow_unverified_mlx: bool = False,
) -> Callable[
    [Any, mx.array, mx.array, mx.array, mx.array], tuple[mx.array, mx.array]
]:
    """Build the packed-sequence loss callable for mlx-lm's trainer:
    `loss(model, batch, seg_id, seg_start, loss_mask) -> (loss, ntoks)` -- the 4-array tuple
    `mlx_train_perf.data.packing.packed_iterate_batches` yields (int32 `(B, L+1)` token rows,
    int32 `(B, L)` seg_id, int32 `(B, L)` seg_start, bool `(B, L)` loss_mask).

    Unlike `make_loss_fn`, which calls the model's top-level trunk (whose call hardcodes the
    "causal" mask, `mlx_lm/models/*.py`), this walks the inner model LAYER BY LAYER --
    `embed_tokens -> for layer: layer(h, mask, cache) -> norm` -- so it can thread a
    `PackedMask(seg_id, seg_start)` carrier as each block's `mask`. Verified against the
    installed mlx-lm==0.31.3 (mlx==0.32.0) source (2026-07-18): llama.py:173-197
    (`LlamaModel.__call__`), qwen2.py:137-155 (`Qwen2Model.__call__`), qwen3.py:142-160
    (`Qwen3Model.__call__`) are all exactly that shape, with no extra scaling; the three
    families share `embed_tokens`/`layers`/`norm` (same gate as `make_loss_fn`). The walk
    stays grad-checkpointed: `grad_checkpoint` patches `type(layer).__call__` at class level,
    so direct block calls dispatch through the patched method (spec §3.2).

    Fails fast at CONSTRUCTION (`AdapterError`): an unsupported architecture or missing
    `mlx-lm` (via `split_model`), and -- because only the `FlashAttentionWrapper` can
    interpret the `PackedMask` carrier (stock SDPA would hand it to
    `mx.fast.scaled_dot_product_attention` and die opaquely) -- any layer whose `self_attn`
    is not a wrapper, naming `enable_flash_attention`. As in `make_loss_fn`, the
    construction-time `(trunk, head)` is discarded and re-derived from the live `model`
    argument on every call (see the module docstring for the staleness rationale).
    """
    from mlx_train_perf.attention.segments import PackedMask  # noqa: PLC0415
    from mlx_train_perf.attention.wrapper import FlashAttentionWrapper  # noqa: PLC0415

    split_model(model)  # family/head fail-fast, result discarded (as in make_loss_fn)
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.self_attn, FlashAttentionWrapper):
            raise AdapterError(
                f"make_packed_loss_fn requires enable_flash_attention(model) first; "
                f"layer {i} self_attn is {type(layer.self_attn).__name__!r}"
            )

    resolved_impl: Literal["kernel", "chunked", "naive"] | None = None

    def loss_fn(
        model_arg: Any,
        batch: mx.array,
        seg_id: mx.array,
        seg_start: mx.array,
        loss_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        nonlocal resolved_impl
        _, head = split_model(model_arg)  # fresh per call (module docstring: staleness)
        inner = model_arg.model
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
        h = inner.embed_tokens(inputs)
        for layer in inner.layers:
            h = layer(h, pm, None)
        hidden = inner.norm(h)
        if resolved_impl is None:
            n = hidden.shape[0] * hidden.shape[1]
            resolved_impl = resolve_impl(
                head=head, dtype=hidden.dtype, n=n, impl=impl,
                allow_unverified_mlx=allow_unverified_mlx,
            ).impl
        # validate_targets=False for the same reason as make_loss_fn: the trainer wraps this
        # step in mx.compile (no host syncs), and the ids are in-range by construction.
        nll = linear_cross_entropy(hidden, head, targets, impl=resolved_impl,
                                   reduction="none", validate_targets=False)
        ntoks = loss_mask.sum()
        loss = (nll * loss_mask).astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn
