"""Pure-MLX attention oracles -- public re-exports.

See `mlx_train_perf.attention.reference` for the implementation and the layout/GQA
conventions every later attention parity test (T4-T13) is checked against.
"""
from mlx_train_perf.attention.reference import (
    flash_attention_reference,
    kv_head_for,
    math_attention,
)

__all__ = ["flash_attention_reference", "kv_head_for", "math_attention"]
