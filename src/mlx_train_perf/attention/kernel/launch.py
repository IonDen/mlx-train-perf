"""Query-range multi-dispatch driver for the v0 flash-attention forward kernel.

Split from day one (review-mlx High): the GPU watchdog applies PER DISPATCH, and the
attention forward has the same O(N^2 . D . Hq) per-layer scaling as the backward -- at the
16-32k context ambition a single forward dispatch over all query blocks over-budgets at
any rate. Forward O/L rows are DISJOINT across query blocks, so the launcher loops
query-row-range dispatches, each writing its own tile-local (b, hq, rows, d) chunk (the CE
forward's disjoint-output pattern -- no accumulator chaining), and reassembles with
`mx.concatenate`. It SPLITS rather than refuses; `LaunchBudgetError` is raised only when
even one minimal range (or the whole forward's total time) over-budgets.

Full buffers + an in-kernel query-row offset (`qoffs`), never a Python-side `q[r0:r1]`
slice into a chained launch (the CE kernel's measured 1.22 GB retained-copy lesson --
user-metal-kernels workflow-and-gotchas.md).

Rate calibration (`calibrated_fwd_rate`) runs at first-launch time and is CACHED (mirrors
`core/kernel/launch._RATE_CACHE`), never inside a compiled region or a vjp (host-sync
timing is compile-hostile). It follows the CE calibration discipline -- warmup dispatch
unmeasured (pays the Metal JIT + cold clocks), median-of-3 timed, halved by a safety
factor -- but does NOT reuse the CE's vocab-tile ramp: that ramp exists because the MMA
kernel's rate is occupancy-dependent up to n~8192, whereas this v0 scalar kernel launches
b*hq*rows threads (thousands even at the small probe shape) and is latency-bound at
saturation, so a warm median at a fixed small probe shape is representative.
"""
import functools
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlx_train_perf.attention.kernel.source import build_fwd_source
from mlx_train_perf.errors import LaunchBudgetError

MAX_DISPATCH_SECONDS = 1.0
MAX_TOTAL_SECONDS = 60.0
SAFETY_FACTOR = 0.5      # halve the measured rate (session drift + probe noise, 2x margin)
_PROBE_N = 128           # fixed small probe shape; the scalar kernel saturates well below it

# key: (head_dim, dtype, causal, b, hq, n-bucket)
_FWD_RATE_CACHE: dict[tuple[int, str, bool, int, int, int], float] = {}

# The installed mlx stub types mx.fast.metal_kernel's return as `object`; it is a callable
# kernel invoker (same cast convention as core/kernel/launch._MetalKernel).
_MetalKernel = Callable[..., list[mx.array]]


@dataclass(frozen=True, slots=True, kw_only=True)
class TileShape:
    """v0 kernel tiling: `bq` = query rows per threadgroup (the x threadgroup dimension).
    v0 is one-thread-per-query-row, so `bq` sets only occupancy grouping, not the math;
    T6 introduces the KV-block tile and simdgroup_matrix dimensions."""

    bq: int = 32


def _n_bucket(n: int) -> int:
    return 1 << max(9, (n - 1).bit_length())   # 512, 1024, 2048, ... occupancy regime key


def _fwd_macs_per_row(*, n: int, d: int, b: int, hq: int) -> int:
    """Conservative per-query-row MAC upper bound: each of the n keys costs a D-wide QK dot
    plus a D-wide PV accumulate (2*D MACs), across every (batch, q-head). This over-counts
    causal (a row near the top attends to fewer keys), which is the safe direction for a
    launch-budget guard -- over-estimating cost splits MORE, never under-budgets."""
    return 2 * d * n * b * hq


def check_fwd_budget(*, n: int, d: int, b: int, hq: int, rows: int, rate: float) -> None:
    """Refuse-before-launch: the GPU watchdog SIGABRT is uncatchable. Projects both the
    per-dispatch time (this range of `rows` query rows) and the whole forward's total time
    from the conservative MAC model, and raises if either exceeds its budget."""
    per_row = _fwd_macs_per_row(n=n, d=d, b=b, hq=hq)
    per_dispatch = rows * per_row / rate
    total = n * per_row / rate
    if per_dispatch > MAX_DISPATCH_SECONDS or total > MAX_TOTAL_SECONDS:
        raise LaunchBudgetError(
            f"projected {per_dispatch:.2f} s/dispatch ({rows} rows), {total:.0f} s total "
            f"at {rate / 1e9:.1f} G MAC/s (budget {MAX_DISPATCH_SECONDS} s / "
            f"{MAX_TOTAL_SECONDS} s). Reduce shape/context, or pass a measured rate."
        )


@functools.cache
def _fwd_kernel(head_dim: int, causal: bool, flip_causal: bool) -> _MetalKernel:
    kernel = mx.fast.metal_kernel(
        name=f"mtp_flash_fwd_d{head_dim}_{'c' if causal else 'f'}{'x' if flip_causal else ''}",
        input_names=["q", "k", "v", "qoffs", "scale_in"],
        output_names=["o_out", "l_out"],
        source=build_fwd_source(head_dim, causal=causal, flip_causal=flip_causal),
    )
    return cast(_MetalKernel, kernel)


def _rows_per_dispatch(*, n: int, d: int, b: int, hq: int, rate: float) -> int:
    """Largest query-row range whose projected dispatch stays within the per-dispatch
    budget at `rate`. Floors at 1 row -- refusing an over-budget single row is
    `check_fwd_budget`'s job, not this sizing heuristic's."""
    per_row = _fwd_macs_per_row(n=n, d=d, b=b, hq=hq)
    rows = int(MAX_DISPATCH_SECONDS * rate / per_row)
    return max(1, min(n, rows))


def launch_flash_fwd(
    q: mx.array, k: mx.array, v: mx.array, *,
    scale: float, causal: bool, tile: TileShape,
    rate_macs_per_s: float | None = None,
    _flip_causal: bool = False,
) -> tuple[mx.array, mx.array]:
    """v0 flash-attention forward -> (O, L). O has q's shape/dtype; L is (B, Hq, N) fp32.

    `rate_macs_per_s`: when None, a single dispatch over all rows with no budget check
    (safe only at small N -- the direct/test path); when given, the launcher sizes the
    query-row split from it and refuses (`LaunchBudgetError`) if even one minimal range or
    the total over-budgets. The API path always passes a calibrated rate (see
    `calibrated_fwd_rate`), so a flagship call splits instead of tripping the watchdog.

    `_flip_causal` is TEST-ONLY (wrong-mask perturbation -- see source.py).
    """
    b, hq, n, d = q.shape
    if rate_macs_per_s is None:
        rows_per = n
    else:
        rows_per = _rows_per_dispatch(n=n, d=d, b=b, hq=hq, rate=rate_macs_per_s)
        check_fwd_budget(n=n, d=d, b=b, hq=hq, rows=rows_per, rate=rate_macs_per_s)

    kernel = _fwd_kernel(d, causal, _flip_causal)
    scale_in = mx.array([scale], dtype=mx.float32)
    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    for r0 in range(0, n, rows_per):
        r1 = min(r0 + rows_per, n)
        rows_this = r1 - r0
        qoffs = mx.array([r0, r1], dtype=mx.uint32)
        o_c, l_c = kernel(
            inputs=[q, k, v, qoffs, scale_in],
            template=[("T", q.dtype)],
            grid=(rows_this, b * hq, 1),
            threadgroup=(min(tile.bq, rows_this), 1, 1),
            output_shapes=[(b, hq, rows_this, d), (b, hq, rows_this)],
            output_dtypes=[q.dtype, mx.float32],
        )
        o_chunks.append(o_c)
        l_chunks.append(l_c)
    return mx.concatenate(o_chunks, axis=2), mx.concatenate(l_chunks, axis=2)


def calibrated_fwd_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool
) -> float:
    """Cached, safety-factored MAC/s throughput for the v0 forward kernel, used to size the
    query-row split. Probes at a fixed small shape (`_PROBE_N` rows and keys, the caller's
    real b/hq/hkv/d/causal): warmup dispatch unmeasured (Metal JIT + cold clocks), then the
    median of 3 timed dispatches, halved by `SAFETY_FACTOR`. Cached per
    (head_dim, dtype, causal, b, hq, n-bucket) so repeated calls in an occupancy regime
    don't re-probe. Must never be called inside a compiled region (host-sync timing)."""
    key = (head_dim, str(dtype), causal, b, hq, _n_bucket(n))
    if key in _FWD_RATE_CACHE:
        return _FWD_RATE_CACHE[key]

    np = min(n, _PROBE_N)
    mx.random.seed(0)
    q = mx.random.normal((b, hq, np, head_dim)).astype(dtype)
    kk = mx.random.normal((b, hkv, np, head_dim)).astype(dtype)
    vv = mx.random.normal((b, hkv, np, head_dim)).astype(dtype)
    mx.eval(q, kk, vv)
    scale = 1.0 / (head_dim ** 0.5)
    macs = _fwd_macs_per_row(n=np, d=head_dim, b=b, hq=hq) * np

    def once() -> float:
        t0 = time.perf_counter()
        o, lse = launch_flash_fwd(q, kk, vv, scale=scale, causal=causal, tile=TileShape())
        mx.eval(o, lse)
        return time.perf_counter() - t0

    once()  # warmup: pays the Metal JIT + cold GPU clocks; deliberately unmeasured
    median_s = statistics.median([once() for _ in range(3)])
    rate = SAFETY_FACTOR * macs / max(median_s, 1e-9)
    _FWD_RATE_CACHE[key] = rate
    return rate
