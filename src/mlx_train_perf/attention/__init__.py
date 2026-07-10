"""Pure-MLX attention oracles + the public flash_attention API -- re-exports.

See `mlx_train_perf.attention.reference` for the oracle implementation and the
layout/GQA conventions every later attention parity test (T4-T13) is checked against,
and `mlx_train_perf.attention.api` for the `flash_attention` custom_function boundary.
"""
from mlx_train_perf.attention.api import flash_attention, resolve_attention_impl
from mlx_train_perf.attention.reference import (
    flash_attention_reference,
    kv_head_for,
    math_attention,
)
from mlx_train_perf.attention.wrapper import (
    FlashAttentionWrapper,
    enable_flash_attention,
)

__all__ = [
    "FlashAttentionWrapper",
    "enable_flash_attention",
    "flash_attention",
    "flash_attention_reference",
    "kv_head_for",
    "math_attention",
    "resolve_attention_impl",
]
