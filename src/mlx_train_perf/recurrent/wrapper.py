"""`enable_gated_delta_training`: the opt-in, per-model-instance qwen3_5 integration wrapper.

Replaces each linear-attention decoder layer's `linear_attn` (a `GatedDeltaNet`) with a
`GatedDeltaTrainingProxy` that routes the recurrence through this project's own
chunk-parallel op (`recurrent.ops.chunked_gated_delta`) or the sequential oracle
(`recurrent.reference.sequential_gated_delta`) -- mirroring `attention/wrapper.py`'s
`enable_flash_attention` shape for the recurrent training path.

Verified against the installed mlx-lm==0.31.3 source (`mlx_lm/models/qwen3_5.py`):

- **Layer selection is STRUCTURAL, not config-driven.** `DecoderLayer.__init__` sets
  `self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0` and then
  constructs EITHER `self.linear_attn` (a `GatedDeltaNet`, when `is_linear`) OR
  `self.self_attn` (a full-attention `Attention`), never both. `layer_types` is NOT a
  field on `TextModelArgs` -- `BaseModelArgs.from_dict` silently drops unknown keys -- so
  a `getattr(args, "layer_types", ...)` guard can never fire; it survives only in the RAW
  dict at `model.args.text_config` (the two-field `ModelArgs(model_type, text_config)`
  wrapper `families.text_model`/`families.text_args` deliberately look past). The real
  `mlx-community/Qwen3.5-0.8B-4bit` config DOES carry a `layer_types` list, so this
  wrapper cross-checks the raw dict against the structural split and refuses on
  disagreement, rather than trusting either source alone.

- **`mlx_lm.models.qwen3_5_moe.Model` SUBCLASSES `mlx_lm.models.qwen3_5.Model`** --
  `families.is_qwen35` (an exact `__module__` comparison) already excludes it, same
  reasoning as `attention/wrapper.py`. `mlx_lm.models.qwen3_next` is a SEPARATE,
  differently-structured family (its own `GatedDeltaNet`/`DecoderLayer`) that happens to
  share the same `is_linear`/`linear_attn` shape closely enough that a bare
  "unsupported model architecture" message would be misleading -- it is named explicitly
  as future work below, not folded into the generic wrong-family refusal.

- **`sharding_group` exists on EVERY `GatedDeltaNet` instance, defaulting to `None`** (set
  unconditionally in `GatedDeltaNet.__init__`, only ever changed by `Model.shard()`). A
  generic "does this layer have any unexpected attribute" scan could never flag it --
  the attribute is always present. When set, the stock forward wraps
  `sum_gradients`/`all_sum` distributed calls around the block; the proxy does not
  reproduce those, so a sharded layer refuses at enable time instead of silently
  dropping the collective.

- **`nn.Module.__setattr__` registers any `mx.array`/`dict`/`list`/`tuple`-valued
  attribute as a dict child** (verified against the installed `nn.Module` source, same
  fact `attention/wrapper.py`'s module docstring cites). The proxy therefore copies the
  original `GatedDeltaNet`'s submodules and array leaves under their ORIGINAL names and
  never retains `original` itself -- holding it as an attribute would double every one of
  its parameters in `parameters()`/`trainable_parameters()` and make LoRA target
  discovery (which walks `named_modules()` by path) adapt the same `nn.Linear` twice.

- **`model.freeze()` on a `GatedDeltaNet` populates `_no_grad` with exactly
  `{'A_log', 'dt_bias'}`** (both local, undotted keys -- verified against the installed
  `nn.Module.freeze`/`unfreeze`). The proxy carries `_no_grad` and `_training` over from
  the original explicitly, and fails loud (rather than silently dropping the freeze) if a
  frozen key has no matching attribute on the proxy -- a silent drop would make a
  should-be-frozen parameter trainable, land it in a saved adapter file, and contaminate
  a later convergence comparison against the stock model.

Only exactly `mlx_lm.models.qwen3_5.Model` is supported (via `families.is_qwen35`,
threaded through `families.text_model`/`families.text_args`); `qwen3_5_moe` (any
`num_experts > 0` text_config), `qwen3_next`, any other model family, a distributed
(`sharding_group`-set) layer, a `layer_types` raw-config/structural disagreement, an
already-enabled model, and a model with zero linear-attention layers all refuse at enable
time (`UnsupportedRecurrentError`) rather than failing mid-training-run. `impl` outside
`{"chunked", "sequential"}` refuses with `RecurrentInputError`.
"""
from collections.abc import Callable
from typing import Any, Literal

import mlx.core as mx
from mlx import nn

from mlx_train_perf.errors import RecurrentInputError, UnsupportedRecurrentError
from mlx_train_perf.families import text_args, text_model
from mlx_train_perf.recurrent.ops import chunked_gated_delta
from mlx_train_perf.recurrent.reference import sequential_gated_delta

_Impl = Literal["chunked", "sequential"]
_VALID_IMPLS: tuple[str, ...] = ("chunked", "sequential")

# `mlx_lm.models.qwen3_next` is a distinct family from `mlx_lm.models.qwen3_5` -- see the
# module docstring. Named explicitly so its refusal message is not folded into the
# generic wrong-family one.
_QWEN3_NEXT_MODULE = "mlx_lm.models.qwen3_next"


def _gated_delta_net_class() -> type[Any]:
    """The real `GatedDeltaNet` class, imported lazily -- `mlx-lm` is an optional extra,
    and this wrapper must not require it merely to be imported."""
    import importlib  # noqa: PLC0415 -- lazy: mlx-lm is an optional extra

    return importlib.import_module("mlx_lm.models.qwen3_5").GatedDeltaNet  # type: ignore[no-any-return]


def _raw_layer_types(model: Any, *, expected_len: int) -> list[str] | None:
    """The raw `layer_types` list at `model.args.text_config`, or `None` when the raw
    config carries no such key (the tiny test fixtures; real 0.8B checkpoints do carry
    it). `None` means "nothing to cross-check", never "everything is full attention"."""
    text_config = getattr(getattr(model, "args", None), "text_config", None)
    if not isinstance(text_config, dict):
        return None
    layer_types = text_config.get("layer_types")
    if layer_types is None:
        return None
    if len(layer_types) != expected_len:
        raise UnsupportedRecurrentError(
            f"raw text_config.layer_types has {len(layer_types)} entries, expected "
            f"{expected_len} (one per decoder layer)"
        )
    return [str(lt) for lt in layer_types]


def _check_layer_types_agree(i: int, layer: Any, layer_types: list[str] | None) -> None:
    """Refuse when the raw `text_config.layer_types[i]` disagrees with the structural
    `layer.is_linear` split. A no-op when `layer_types` is `None` (nothing to
    cross-check -- the tiny test fixtures; real 0.8B checkpoints do carry it)."""
    if layer_types is None:
        return
    expected_linear = layer_types[i] != "full_attention"
    if expected_linear != bool(layer.is_linear):
        raise UnsupportedRecurrentError(
            f"raw text_config.layer_types[{i}] ({layer_types[i]!r}) disagrees with the "
            f"structural is_linear split (is_linear={layer.is_linear}) at layer {i}"
        )


def _select_linear_layer(i: int, layer: Any, gated_delta_net_cls: type[Any]) -> bool:
    """True iff layer `i` is a legitimate, not-yet-enabled linear-attention layer this
    function should wrap. Refuses (rather than silently skipping) an already-enabled
    proxy, an unexpected `linear_attn` class, or a distributed `sharding_group`."""
    if not layer.is_linear:
        return False
    attn = layer.linear_attn
    if isinstance(attn, GatedDeltaTrainingProxy):
        raise UnsupportedRecurrentError(
            "gated-delta training is already enabled on this model"
        )
    if not isinstance(attn, gated_delta_net_cls):
        raise UnsupportedRecurrentError(
            f"layer {i}'s linear_attn is not a GatedDeltaNet instance (got "
            f"{type(attn)!r})"
        )
    if attn.sharding_group is not None:
        raise UnsupportedRecurrentError(
            f"layer {i}'s GatedDeltaNet has a distributed sharding_group set; the "
            "training proxy does not wrap the distributed sum_gradients/all_sum calls "
            "the stock forward applies around a sharded layer"
        )
    return True


class GatedDeltaTrainingProxy(nn.Module):
    """Drop-in replacement for an mlx-lm `GatedDeltaNet` that will route the recurrence
    through this project's chunk-parallel op or the sequential oracle. Holds the
    original's submodules/array leaves under their original names and never retains the
    original module itself (see the module docstring for why). The forward pass lands in
    a later change; today `__call__` refuses a cache (a training-only path never serves
    one) and otherwise raises `NotImplementedError`."""

    def __init__(self, original: nn.Module, *, impl: str) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        # Submodules / array leaves, ORIGINAL names (LoRA discovery walks named_modules()
        # by path; nn.Module auto-registration keeps these in parameters()).
        self.in_proj_qkv = original.in_proj_qkv
        self.in_proj_z = original.in_proj_z
        self.in_proj_b = original.in_proj_b
        self.in_proj_a = original.in_proj_a
        self.conv1d = original.conv1d
        self.norm = original.norm
        self.out_proj = original.out_proj
        self.A_log = original.A_log
        self.dt_bias = original.dt_bias
        # Plain scalars the forward needs (nn.Module stores non-array/non-module values
        # as regular attributes, not dict children).
        self.num_v_heads = original.num_v_heads
        self.num_k_heads = original.num_k_heads
        self.head_k_dim = original.head_k_dim
        self.head_v_dim = original.head_v_dim
        self.key_dim = original.key_dim
        self.value_dim = original.value_dim
        self.conv_kernel_size = original.conv_kernel_size
        self.conv_dim = original.conv_dim
        self.layer_norm_epsilon = original.layer_norm_epsilon

        # Carry the freeze state over -- fail loud rather than silently losing it.
        missing = {k for k in original._no_grad if not hasattr(self, k)}
        if missing:
            raise UnsupportedRecurrentError(
                f"cannot carry frozen keys {sorted(missing)} onto the proxy -- "
                "the attribute copy must run first"
            )
        self._no_grad.update(original._no_grad)
        self._training = original._training

        if impl == "chunked":
            self._op: Callable[..., Any] = chunked_gated_delta
        elif impl == "sequential":
            self._op = sequential_gated_delta()
        else:
            raise RecurrentInputError(
                f"unknown impl {impl!r}; expected one of {_VALID_IMPLS}"
            )
        self._impl = impl

    def __call__(
        self,
        inputs: mx.array,  # noqa: ARG002 -- forward lands in a later change
        mask: Any = None,  # noqa: ARG002 -- forward lands in a later change
        cache: Any = None,
    ) -> mx.array:
        if cache is not None:
            raise RecurrentInputError(
                "GatedDeltaTrainingProxy is a training-only path; a cache is not "
                "supported"
            )
        raise NotImplementedError(
            "GatedDeltaTrainingProxy.__call__ forward is not implemented yet"
        )


def enable_gated_delta_training(model: Any, *, impl: _Impl = "chunked") -> Any:
    """Enable the chunked (or sequential-oracle) GatedDelta training path on a qwen3_5
    model IN PLACE. Replaces every structurally-linear decoder layer's `linear_attn` with
    a `GatedDeltaTrainingProxy` bound to `impl`; returns `model` for convenience.

    `impl`: `"chunked"` (default) routes through `recurrent.ops.chunked_gated_delta`;
    `"sequential"` routes through `recurrent.reference.sequential_gated_delta()` (the
    mlx-lm oracle, useful for parity tests). Anything else refuses with
    `RecurrentInputError`.

    Refuses (`UnsupportedRecurrentError`) rather than partially enabling: an unsupported
    model family (including `qwen3_next`, named explicitly, and any non-qwen3_5 module);
    a qwen3_5 MoE checkpoint (`num_experts > 0` -- the gated-delta training path was
    validated against dense text_config only, on a 32 GB bench box, and MoE's
    expert-parameter memory footprint is out of scope); a raw `text_config.layer_types`
    entry disagreeing with the structural `is_linear` split; a layer with a distributed
    `sharding_group` set; a model with zero linear-attention layers; and a model this
    function has already been called on. Every refusal is raised before any layer is
    mutated -- except double-enable, which legitimately leaves the earlier, successful
    enable call's proxies in place."""
    if impl not in _VALID_IMPLS:
        raise RecurrentInputError(
            f"unknown impl {impl!r}; expected one of {_VALID_IMPLS}"
        )

    module_name = type(model).__module__
    if module_name == _QWEN3_NEXT_MODULE:
        raise UnsupportedRecurrentError(
            "qwen3_next is a distinct model family from qwen3_5 (its own GatedDeltaNet/"
            "DecoderLayer shape) and is not supported by the gated-delta training path "
            "-- future work"
        )

    trunk = text_model(model)  # raises UnsupportedRecurrentError for any other family
    args = text_args(model)
    try:
        num_experts = args.num_experts
    except AttributeError as exc:
        raise UnsupportedRecurrentError(
            "qwen3_5 model's text args are missing num_experts"
        ) from exc
    if num_experts > 0:
        raise UnsupportedRecurrentError(
            "qwen3_5 MoE checkpoints (num_experts > 0) are unsupported: the "
            "gated-delta training path was validated against dense text_config only, "
            "on a 32 GB bench box -- MoE's expert-parameter memory footprint is out of "
            "scope"
        )

    layers = trunk.layers
    layer_types = _raw_layer_types(model, expected_len=len(layers))
    gated_delta_net_cls = _gated_delta_net_class()

    selected: list[int] = []
    for i, layer in enumerate(layers):
        _check_layer_types_agree(i, layer, layer_types)
        if _select_linear_layer(i, layer, gated_delta_net_cls):
            selected.append(i)

    if not selected:
        raise UnsupportedRecurrentError(
            "no linear-attention layers found to enable (full_attention_interval "
            "leaves every layer as full attention)"
        )

    for i in selected:
        layer = layers[i]
        layer.linear_attn = GatedDeltaTrainingProxy(layer.linear_attn, impl=impl)
    return model
