"""Chained-launch driver for the dense MMA kernel. Full buffers + in-kernel offsets —
Python-side w[v0:v1] slices into chained launches cost 1.22 GB of retained copies
(measured; user-metal-kernels workflow-and-gotchas.md)."""
import functools
from collections.abc import Callable
from typing import cast

import mlx.core as mx

from mlx_train_perf.core.kernel.source import build_dense_source
from mlx_train_perf.errors import LaunchBudgetError

MAX_DISPATCH_SECONDS = 1.0
MAX_TOTAL_SECONDS = 60.0

# The installed mlx 0.31.2 stub types mx.fast.metal_kernel's return as `object` (nanobind
# gives it no more specific type); it is documented + actually a callable kernel invoker.
_MetalKernel = Callable[..., list[mx.array]]


@functools.cache
def _dense_kernel(row_tiles: int) -> _MetalKernel:
    kernel = mx.fast.metal_kernel(
        name=f"mtp_fused_ce_rt{row_tiles}",
        input_names=["hidden", "w", "targets", "offs", "lse_in", "tgt_in"],
        output_names=["lse_out", "tgt_out"],
        source=build_dense_source(row_tiles),
    )
    return cast(_MetalKernel, kernel)


def check_budget(*, n: int, d: int, v: int, tile: int, rate_macs_per_s: float) -> None:
    """Refuse-before-launch: the GPU watchdog SIGABRT is uncatchable."""
    per_dispatch = n * tile * d / rate_macs_per_s
    total = n * v * d / rate_macs_per_s
    if per_dispatch > MAX_DISPATCH_SECONDS or total > MAX_TOTAL_SECONDS:
        raise LaunchBudgetError(
            f"projected {per_dispatch:.2f} s/dispatch, {total:.0f} s total at "
            f"{rate_macs_per_s / 1e9:.0f} G MAC/s (budget {MAX_DISPATCH_SECONDS} s / "
            f"{MAX_TOTAL_SECONDS} s). Reduce tile/shape, or pass a measured rate."
        )


def forward(
    hidden: mx.array, w: mx.array, targets: mx.array, *,
    row_tiles: int, tile: int, rate_macs_per_s: float | None,
) -> tuple[mx.array, mx.array]:
    n, d = hidden.shape
    v = w.shape[0]
    if rate_macs_per_s is not None:
        check_budget(n=n, d=d, v=v, tile=tile, rate_macs_per_s=rate_macs_per_s)
    rows = 8 * row_tiles
    row_blocks = (n + rows - 1) // rows
    kernel = _dense_kernel(row_tiles)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    tgt = mx.zeros((n,), dtype=mx.float32)
    tg_y = min(8, row_blocks)
    for v0 in range(0, v, tile):
        v1 = min(v0 + tile, v)
        offs = mx.array([v0, v1], dtype=mx.uint32)
        lse, tgt = kernel(
            inputs=[hidden, w, targets, offs, lse, tgt],
            template=[("T", hidden.dtype)],
            grid=(32, row_blocks, 1),
            threadgroup=(32, tg_y, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.float32],
        )
    return lse, tgt
