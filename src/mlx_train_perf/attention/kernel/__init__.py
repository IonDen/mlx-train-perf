"""Flash-attention FORWARD Metal kernel v0 (O + L), correctness-only (0.2.0 T5).

`source.build_fwd_source` is the sentinel-templated MSL builder; `launch.launch_flash_fwd`
is the query-range multi-dispatch driver (splits over disjoint query-row ranges, full
buffers + in-kernel offsets, calibrated launch-budget guard). See each module's docstring.
"""
from mlx_train_perf.attention.kernel.launch import (
    TileShape,
    calibrated_fwd_rate,
    check_fwd_budget,
    launch_flash_fwd,
)
from mlx_train_perf.attention.kernel.source import build_fwd_source

__all__ = [
    "TileShape",
    "build_fwd_source",
    "calibrated_fwd_rate",
    "check_fwd_budget",
    "launch_flash_fwd",
]
