"""Chained-launch driver for the dense MMA kernel. Full buffers + in-kernel offsets —
Python-side w[v0:v1] slices into chained launches cost 1.22 GB of retained copies
(measured; user-metal-kernels workflow-and-gotchas.md)."""
import functools
import time
from collections.abc import Callable
from typing import cast

import mlx.core as mx

from mlx_train_perf.core.kernel.source import build_dense_source
from mlx_train_perf.errors import LaunchBudgetError

MAX_DISPATCH_SECONDS = 1.0
MAX_TOTAL_SECONDS = 60.0

FLOOR_RATE = 10e9        # G MAC/s floor: v0-class, slowest rate ever measured for any variant
SAFETY_FACTOR = 0.5      # halve the measured rate: covers session drift (~12% measured) plus
                         # probe-timing noise with a deliberate 2x margin; costs nothing —
                         # production dispatches sit ~10x under budget even at half rate
_RATE_CACHE: dict[tuple[int, str, int], float] = {}


def _n_bucket(n: int) -> int:
    return 1 << max(9, (n - 1).bit_length())   # 512, 1024, 2048, ... occupancy regime key


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


def probe_tile_for(*, n: int, d: int, floor_rate: float = FLOOR_RATE,
                   budget_s: float = 0.5) -> int:
    """Largest power-of-two tile <= 8192 whose dispatch stays under budget_s at floor_rate."""
    tile = 8192
    while tile > 32 and n * tile * d / floor_rate > budget_s:
        tile //= 2
    return max(tile, 32)


def calibrated_rate(*, row_tiles: int, dtype: mx.Dtype, n: int, d: int, v: int) -> float:
    """Micro-probe the kernel at the caller's real n (small-n rates underestimate large-n
    rates due to occupancy), sized safe by probe_tile_for's floor-rate guard, then apply
    SAFETY_FACTOR. Cached per (row_tiles, dtype, n-bucket) so repeated calls at the same
    occupancy regime don't re-dispatch the probe."""
    key = (row_tiles, str(dtype), _n_bucket(n))
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    probe_tile = min(probe_tile_for(n=n, d=d), v)
    # Random probe data, NOT zeros: all-zero buffers are the best case for Apple's
    # lossless memory compression and could optimistically bias this safety-relevant
    # rate (T8 review finding). Seeded for determinism.
    mx.random.seed(0)
    hidden = mx.random.normal((n, d)).astype(dtype)
    w = (mx.random.normal((probe_tile, d)) * 0.05).astype(dtype)
    targets = mx.random.randint(0, probe_tile, (n,))
    # dispatch 1 pays the Metal JIT; time dispatch 2 only
    for _timed in (False, True):
        t0 = time.perf_counter()
        lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=probe_tile,
                           rate_macs_per_s=None)
        mx.eval(lse, tgt)
        elapsed = time.perf_counter() - t0
    rate = SAFETY_FACTOR * (n * probe_tile * d) / max(elapsed, 1e-9)
    _RATE_CACHE[key] = rate
    return rate
