"""`enable_flash_attention`: the opt-in, per-model-instance mlx-lm integration wrapper.

Replaces each decoder layer's `self_attn` with a `FlashAttentionWrapper` that routes the
post-RoPE attention through `flash_attention` (kernel forward + kernel backward) --
the training-only path this whole 0.2.0 release exists to switch on.

Verified against the installed mlx-lm==0.31.3 (mlx==0.32.0) source (llama/qwen3 2026-07-09,
qwen2 2026-07-15):

- **Stock attention surface** (`mlx_lm/models/llama.py::Attention`,
  `mlx_lm/models/qwen2.py::Attention`, `mlx_lm/models/qwen3.py::Attention` -- identical shape;
  qwen3 adds per-head q/k RMSNorm, qwen2 gives q/k/v a `bias=True` linear):
  `q_proj`/`k_proj`/`v_proj`/`o_proj` (`nn.Linear`), `rope`, the ints `n_heads`/`n_kv_heads`
  and float `scale`, plus (qwen3 only) `q_norm`/`k_norm`. `__call__(x, mask=None, cache=None)`
  projects, reshapes to `(B, H, L, head_dim)`, applies RoPE, calls SDPA, then `o_proj`. The
  wrapper reproduces that EXACTLY, swapping the single SDPA call for `flash_attention` (holding
  the projections as direct submodules means qwen2's q/k/v bias is applied transparently).

- **Two load-bearing reasons the original submodules are held as DIRECT attributes with
  their ORIGINAL names** (`self.q_proj = original.q_proj`, ...), never closed over:
  1. `mlx_lm.tuner.utils.linear_to_lora_layers` discovers LoRA targets by module-tree path
     (`self_attn.q_proj`) via `named_modules()` and replaces them with `update_modules`. The
     names must survive, AND `__call__` must do attribute lookup (`self.q_proj(x)`) at call
     time -- if it closed over the pre-LoRA module, LoRA would inject adapters the forward
     never runs (nonzero grads that never move the loss).
  2. `nn.Module` auto-registration keeps the projections/norms/rope in `parameters()` and
     checkpoint's `trainable_parameters()`. (mlx's `nn.Module` is a `dict` subclass, so
     assigning a submodule stores it as a child; assigning an int/float/str stores a plain
     attribute -- verified against the installed `nn.Module.__setattr__`.)

- **Call-time mask contract**: in the real training path
  `mlx_lm.models.base.create_attention_mask` returns the STRING `"causal"` for `N>1` with no
  cache -- THAT is the supported case (-> `causal=True`); `None` (its `N==1` return) also maps
  to causal. Any `mx.array` mask (sliding-window / additive) or any non-`"causal"` string
  raises `AttentionInputError`; any cache raises `AttentionInputError` (training-only path).

- **Pre-calibration**: `flash_attention`'s kernel
  path runs a host-synced rate probe from its Python body at construction time -- harmless
  under `mx.grad`, but under mlx-lm's compiled `train()` the FIRST trace would host-sync
  inside the compiled region. `enable_flash_attention` therefore PRE-WARMS all three rate
  caches (forward + backward dQ + backward dK/dV) at enable time when `seq_len` is hinted, so
  the compiled step traces with warm caches (the CE adapter's own 0.1.0 precedent:
  calibrate before the compiled `train()`). Callers targeting compiled training MUST pass
  `seq_len` (and `batch_size`) matching their training shape.

Only the Llama, Qwen2 and Qwen3 model families are supported (matched by
`type(model).__module__`, mirroring `adapters/mlx_lm.py`); sliding-window / mixed
`layer_types`, unsupported head_dim, and configured dropout all refuse at enable time
(`UnsupportedAttentionError`).
"""
from typing import Any, Literal, cast

import mlx.core as mx
from mlx import nn

from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.segments import PackedMask
from mlx_train_perf.errors import AttentionInputError, UnsupportedAttentionError

_Impl = Literal["auto", "kernel", "reference"]

# Keyed by the exact `type(model).__module__` mlx-lm uses for each family (mirrors
# `adapters/mlx_lm.py::_SUPPORTED_FAMILIES`).
_SUPPORTED_FAMILIES: tuple[str, ...] = (
    "mlx_lm.models.llama", "mlx_lm.models.qwen2", "mlx_lm.models.qwen3"
)

# The kernel's supported head dims (mirrors `attention/api.py::_KERNEL_HEAD_DIMS`).
_KERNEL_HEAD_DIMS: tuple[int, ...] = (64, 96, 128)


def _resolve_mask(mask: Any) -> bool | PackedMask:
    """Map an mlx-lm attention mask onto `flash_attention`'s call, or refuse.

    `"causal"` (the string `create_attention_mask` returns for the real N>1 training path)
    and `None` (its N==1 return) both mean plain causal -> `True`. A `PackedMask` (0.4.0
    block-diagonal-causal packing) passes through untouched, to be threaded as
    `flash_attention(segments=...)`. An `mx.array` mask is a sliding-window or additive mask
    this causal-only training path does not serve; any other string is unrecognized. Both
    raise `AttentionInputError`."""
    if isinstance(mask, PackedMask):
        return mask
    if isinstance(mask, str):
        if mask == "causal":
            return True
        raise AttentionInputError(
            f"unsupported string mask {mask!r}; the flash-attention training path only "
            "supports the 'causal' mask"
        )
    if mask is None:
        return True
    raise AttentionInputError(
        "array attention masks (sliding-window / additive) are not supported; the "
        "flash-attention training path is causal-only -- enable it only on full-attention "
        "models"
    )


class FlashAttentionWrapper(nn.Module):
    """Drop-in replacement for an mlx-lm `Attention` module that routes SDPA through
    `flash_attention`. Holds the original submodules by their original names (see the module
    docstring for the two load-bearing reasons) and reproduces the stock `__call__` exactly,
    swapping only the single attention call."""

    def __init__(self, original: nn.Module, *, impl: str) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        # Original submodules as DIRECT, ORIGINAL-named attributes (LoRA path discovery +
        # nn.Module auto-registration -- module docstring). nn.Module.__setattr__ stores
        # these (dict-subclass submodules) as children.
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.rope = original.rope
        # qwen3 applies a per-head RMSNorm to q and k before RoPE; llama does not.
        self._has_qk_norm = hasattr(original, "q_norm")
        if self._has_qk_norm:
            self.q_norm = original.q_norm
            self.k_norm = original.k_norm
        # Plain scalars (nn.Module stores non-array/non-module values as regular attributes).
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.scale = original.scale
        self._impl = impl

    def __call__(
        self, x: mx.array, mask: Any = None, cache: Any = None
    ) -> mx.array:
        if cache is not None:
            raise AttentionInputError(
                "FlashAttentionWrapper is a training-only path; a KV cache is not supported"
            )
        b, length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self._has_qk_norm:
            queries = self.q_norm(
                queries.reshape(b, length, self.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(b, length, self.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
        else:
            queries = queries.reshape(b, length, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(b, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(b, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        resolved = _resolve_mask(mask)
        segments = resolved if isinstance(resolved, PackedMask) else None
        output = flash_attention(
            queries, keys, values, scale=self.scale, causal=True, segments=segments,
            impl=cast(_Impl, self._impl),
        )
        output = output.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(output)  # type: ignore[no-any-return]


def _head_dim(model: Any) -> int:
    """Head dim from the model config, family-agnostically: qwen3 sets `args.head_dim`
    directly; llama's is optional and falls back to `hidden_size // num_attention_heads`
    (mirroring `mlx_lm/models/llama.py::Attention.__init__`)."""
    args = model.args
    return int(
        getattr(args, "head_dim", None) or (args.hidden_size // args.num_attention_heads)
    )


def _prewarm_rate_caches(
    model: Any, *, impl: str, seq_len: int, batch_size: int, head_dim: int, packed: bool
) -> None:
    """Warm `flash_attention`'s three kernel rate caches (forward + backward dQ + backward
    dK/dV) at the shape a training forward will hit, so a subsequently compiled `train()`
    traces with warm caches (no host-sync inside the compiled region -- the T5-review
    contract). ONE untraced forward call is sufficient: all three `calibrated_*_rate` probes
    run in `flash_attention`'s Python body at construction time, keyed by
    `(head_dim, dtype, causal, batch, n_heads, n-bucket, variant, d_slab, packed)`, before the
    kernel even dispatches. Inputs are zeros (values are irrelevant to a timing probe and touch
    no global RNG); the dtype is the model's compute dtype, read off a floating trunk norm
    weight (never quantized).

    `packed` (0.4.0): when set, a SECOND forward carrying an all-zeros single-segment
    `PackedMask` at the same shape warms the three PACKED-keyed rate slots (key tail `packed`
    True) -- what a packed training forward (`mask=PackedMask(...)`) hits. The non-packed warm
    stays too: a packed run may still route plain-causal steps (`mask="causal"`)."""
    attn = model.model.layers[0].self_attn  # now a FlashAttentionWrapper
    hq, hkv = attn.n_heads, attn.n_kv_heads
    compute_dtype = model.model.norm.weight.dtype
    q = mx.zeros((batch_size, hq, seq_len, head_dim), dtype=compute_dtype)
    k = mx.zeros((batch_size, hkv, seq_len, head_dim), dtype=compute_dtype)
    v = mx.zeros((batch_size, hkv, seq_len, head_dim), dtype=compute_dtype)
    out = flash_attention(q, k, v, scale=attn.scale, causal=True, impl=cast(_Impl, impl))
    mx.eval(out)
    if packed:
        seg_id = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        seg_start = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        out_packed = flash_attention(
            q, k, v, scale=attn.scale, causal=True, impl=cast(_Impl, impl),
            segments=PackedMask(seg_id=seg_id, seg_start=seg_start),
        )
        mx.eval(out_packed)


def enable_flash_attention(
    model: Any,
    *,
    impl: str = "auto",
    seq_len: int | None = None,
    batch_size: int = 1,
    packed: bool = False,
) -> None:
    """Enable the flash-attention training path on an mlx-lm model IN PLACE.

    Replaces every decoder layer's `self_attn` with a `FlashAttentionWrapper` routing the
    post-RoPE attention through `flash_attention(impl=...)`. Only the Llama, Qwen2 and Qwen3
    families (full attention) are supported; everything else refuses at enable time
    (`UnsupportedAttentionError`) rather than failing mid-training-run.

    `impl`: forwarded to `flash_attention` per call (`"auto"`/`"kernel"` -> the Metal kernel;
    `"reference"` -> the pure-MLX oracle, for parity tests only).

    `seq_len`/`batch_size`/`packed` (compiled-train contract): when `seq_len` is given AND
    `impl` is a kernel impl, pre-warm the kernel rate caches at that shape so a subsequently
    compiled `train()` traces without a host-synced calibration inside the compiled region
    (see `_prewarm_rate_caches`). With `packed=True` the pre-warm ALSO warms the packed-keyed
    rate slots (one extra all-zeros single-segment forward) -- pass it when you will feed the
    wrapper a `PackedMask` mask.

    EXACT-SHAPE CONTRACT: the rate caches key on the EXACT batch `b` and the n-bucket of the
    sequence length (power-of-2 ceiling, floor 512). Pass `batch_size` equal to your runtime
    batch and a `seq_len` that lands in your pack_len's n-bucket. A compiled `train()` traced
    at an un-warmed shape does NOT crash -- it runs a ONE-TIME host-synced calibration inside
    the compiled region (a first-step stall), then caches -- but pass matching hints to avoid
    that stall (the calibration probes with detached arrays, which mlx permits eval'ing inside
    a trace). Defense-in-depth: if such an in-region calibration ever host-syncs on a traced
    array, `flash_attention` upgrades the opaque `[eval]` error to an actionable
    `UnsupportedAttentionError` naming the shape and the pre-warm call. When `seq_len` is
    omitted, enable still succeeds (including with `packed=True`) -- eager / `mx.grad` callers
    calibrate lazily on the first attention call.

    Refuses (`UnsupportedAttentionError`): an unsupported model family; a sliding-window or
    mixed `layer_types` model (`layer_types` entry != "full_attention", or any `use_sliding`
    block); a head_dim outside {64, 96, 128}; configured attention dropout.
    """
    module_name = type(model).__module__
    if module_name not in _SUPPORTED_FAMILIES:
        supported = ", ".join(_SUPPORTED_FAMILIES)
        raise UnsupportedAttentionError(
            f"unsupported model architecture (module {module_name!r}); "
            f"enable_flash_attention supports: {supported}"
        )

    args = model.args
    layer_types = getattr(args, "layer_types", None)
    if layer_types is not None:
        offending = sorted({lt for lt in layer_types if lt != "full_attention"})
        if offending:
            raise UnsupportedAttentionError(
                f"enable_flash_attention supports full-attention models only; model "
                f"declares non-full-attention layer_types {offending} -- sliding-window / "
                "mixed attention is out of scope"
            )
    layers = model.model.layers
    if any(getattr(layer, "use_sliding", False) for layer in layers):
        raise UnsupportedAttentionError(
            "enable_flash_attention supports full-attention models only; model has a "
            "sliding-window attention block -- sliding-window attention is out of scope"
        )

    dropout = getattr(args, "attention_dropout", 0.0)
    if dropout:
        raise UnsupportedAttentionError(
            f"enable_flash_attention does not support attention dropout "
            f"(attention_dropout={dropout}); it must be 0"
        )

    head_dim = _head_dim(model)
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise UnsupportedAttentionError(
            f"enable_flash_attention supports head_dim in {_KERNEL_HEAD_DIMS}; model has "
            f"head_dim={head_dim}"
        )

    for layer in layers:
        layer.self_attn = FlashAttentionWrapper(layer.self_attn, impl=impl)

    if seq_len is not None and impl in ("auto", "kernel"):
        _prewarm_rate_caches(
            model, impl=impl, seq_len=seq_len, batch_size=batch_size, head_dim=head_dim,
            packed=packed,
        )
