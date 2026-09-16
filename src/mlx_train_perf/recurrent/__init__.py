"""Public recurrent (GatedDelta) training surface -- re-exports.

See `mlx_train_perf.recurrent.ops` for the chunk-parallel op and
`mlx_train_perf.recurrent.wrapper` for the qwen3_5 model-instance enable, mirroring
`mlx_train_perf.attention`'s subpackage-level re-export shape.
"""
from mlx_train_perf.recurrent.ops import chunked_gated_delta
from mlx_train_perf.recurrent.wrapper import enable_gated_delta_training

__all__ = ["chunked_gated_delta", "enable_gated_delta_training"]
