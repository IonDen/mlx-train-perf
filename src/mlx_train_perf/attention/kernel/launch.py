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
timing is compile-hostile). It RAMPS the probe key-count toward the caller's real n
(`_calibrate_fwd`, mirroring `core/kernel/launch.calibrate`'s ramp/sustain/median shape --
read that function's docstring for the DVFS rationale) rather than measuring at one fixed
small shape: v0 re-streams all N keys per query row with ZERO cross-row reuse (no
`simdgroup_matrix`, no threadgroup-resident K/V), so a small, cache-resident probe
(~128 KB/head) measures a fundamentally different memory regime than the flagship (8k+ N,
~8-16 MB/head) DRAM-bound working set -- a fixed micro-probe risks an optimistic rate the
same way the CE kernel's own micro-probe once read ~3.6x off at production shape (see
`core/kernel/launch`'s module docstring). This is a CACHE -> DRAM transition, not just an
occupancy effect, so ramping toward large n is load-bearing here, not cosmetic. The ramp's
own sizing (`_next_probe_n`) is QUADRATIC in probe size (a probe of size `np` sets BOTH the
query-row count and the key count, so its cost is O(np^2)), unlike the CE kernel's
tile-linear cost -- the ramp/sustain/median SHAPE is mirrored, the sizing arithmetic is
adapted, not reused. Each stage's own probe dispatch is sized so its projected cost stays
within budget at that stage's own SAFETY_FACTOR-halved measured rate, so calibration itself
never risks the watchdog. Probe QKV are drawn from a LOCAL `mx.random.key` (never
`mx.random.seed`), so calibration never mutates the caller's global RNG stream -- the first
kernel call can fire inside a user's grad/training run and desync downstream consumers.

T6's KV-block tiling changes the kernel's cache-residency and reuse pattern entirely
(threadgroup-resident K/V tiles instead of v0's zero-reuse re-stream), so this calibration
must be re-validated once T6 lands -- do not assume v0's rates or ramp behavior transfer.
"""
import functools
import math
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
_PROBE_N_FLOOR = 128     # ramp's minimum/starting probe key-count -- the original fixed
                         # probe shape, small enough to be safe at any plausible v0 rate
_PROBE_N_HARD_CAP = 8192 # never probe past this many keys/queries during calibration,
                         # regardless of the caller's real n (mirrors core/kernel/launch's
                         # 8192 tile cap)

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


def _start_probe_n(n: int) -> int:
    """Ramp's starting probe key-count: `_PROBE_N_FLOOR` (the original fixed probe size,
    small enough to be safe at any plausible v0 rate) capped at the caller's real `n` -- a
    caller whose real context is already smaller than the floor gets measured AT n
    directly rather than padded up to a probe size it will never actually run."""
    return min(_PROBE_N_FLOOR, n)


def _next_probe_n(
    *, rate_macs_per_s: float, n: int, d: int, b: int, hq: int,
    budget_s: float = MAX_DISPATCH_SECONDS,
) -> int:
    """Largest power-of-two probe key-count <= min(n, `_PROBE_N_HARD_CAP`) whose projected
    SELF-attention dispatch (`np` query rows over `np` keys -- QUADRATIC in `np`, unlike
    the CE kernel's tile-linear cost) stays within `budget_s` at `rate_macs_per_s`. Mirrors
    `core/kernel/launch.next_probe_tile`'s role in the ramp: callers pass the
    SAFETY_FACTOR-halved rate, and `_calibrate_fwd`'s ramp uses this to size the NEXT probe
    one stage ahead of what it has actually measured.

    When the caller's real `n` is already <= `_PROBE_N_FLOOR`, this returns `n` directly
    without a budget check -- that shape was already measured successfully as the ramp's
    OWN starting probe (`_start_probe_n` never exceeds the floor either), so re-deriving it
    here is redundant, not unsafe. Otherwise floors at `_PROBE_N_FLOOR` even if that probe
    still projects over budget -- refusing an over-budget PRODUCTION dispatch is
    `check_fwd_budget`'s job, not this sizing heuristic's."""
    cap = min(n, _PROBE_N_HARD_CAP)
    if cap <= _PROBE_N_FLOOR:
        return cap
    np_ = 1 << (cap.bit_length() - 1)   # largest power of two <= cap
    np_ = max(np_, _PROBE_N_FLOOR)
    while np_ > _PROBE_N_FLOOR and (
        _fwd_macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / rate_macs_per_s > budget_s
    ):
        np_ //= 2
    return np_


def _sustain_reps(*, per_dispatch_s: float, target_s: float = 0.75, cap: int = 8) -> int:
    """Extra back-to-back probe dispatches at the ramp's final size so cumulative
    sustained work reaches `target_s` before the timed median-of-3 -- Apple's GPU DVFS
    needs on the order of a second of continuous load to reach peak clocks from cold (same
    rationale as `core/kernel/launch.sustain_reps`, reimplemented here rather than imported
    so the two kernels' calibration paths stay independently reviewable -- see that
    function's docstring). Floors at 1 (always sustain past the rate-measuring dispatch
    itself); caps at 8 (bounds calibration wall time)."""
    if per_dispatch_s <= 0:
        return cap
    reps = math.ceil(target_s / per_dispatch_s)
    return min(max(reps, 1), cap)


def _calibrate_fwd(
    *, measure: Callable[[int], float], n: int, d: int, b: int, hq: int,
    start_n: int, max_stages: int = 3,
) -> float:
    """Ramp through probe key-counts under real (or, in unit tests, fake) dispatch
    timings, mirroring `core/kernel/launch.calibrate`'s ramp/sustain/median shape exactly
    (see that function's docstring) -- adapted to THIS kernel's quadratic-in-probe-size
    cost model via `_next_probe_n` in place of `next_probe_tile`. Each stage times one
    dispatch at the current probe size and projects a SAFETY_FACTOR-conservative next size
    from that stage's own rate, advancing only while the projection keeps growing; a
    caller whose real n is already small converges in one stage. The final size then gets
    extra sustained dispatches (`_sustain_reps`) to reach ramped GPU clocks, and the
    reported rate is the MEDIAN of 3 timed dispatches at that size -- a single sample is
    exactly the un-ramped, occupancy/cache-starved measurement this function exists to
    avoid. Returns the raw (un-halved) rate -- the caller applies SAFETY_FACTOR."""
    np_ = start_n
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(np_)
        raw_rate = _fwd_macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / max(per_dispatch_s, 1e-9)
        candidate = _next_probe_n(rate_macs_per_s=SAFETY_FACTOR * raw_rate, n=n, d=d, b=b, hq=hq)
        if candidate <= np_:
            break
        np_ = candidate
    for _rep in range(_sustain_reps(per_dispatch_s=per_dispatch_s)):
        measure(np_)
    timings = [measure(np_) for _ in range(3)]
    median_s = statistics.median(timings)
    return _fwd_macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / max(median_s, 1e-9)


def calibrated_fwd_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool
) -> float:
    """Cached, safety-factored, N-AWARE MAC/s throughput for the v0 forward kernel, used to
    size the query-row split. Ramps the probe key-count toward the caller's real `n` via
    `_calibrate_fwd` (see the module docstring for why a fixed small probe reads the wrong
    cache-resident regime at flagship N) rather than measuring at one fixed shape; probe
    QKV are drawn from a LOCAL `mx.random.key(0)` (split into per-tensor sub-keys), never
    `mx.random.seed`, so calibration never mutates the caller's global RNG stream. Cached
    per (head_dim, dtype, causal, b, hq, n-bucket) so repeated calls in an occupancy regime
    don't re-probe. Must never be called inside a compiled region (host-sync timing)."""
    key = (head_dim, str(dtype), causal, b, hq, _n_bucket(n))
    if key in _FWD_RATE_CACHE:
        return _FWD_RATE_CACHE[key]

    key_q, key_k, key_v = mx.random.split(mx.random.key(0), 3)
    scale = 1.0 / (head_dim ** 0.5)
    probes: dict[int, tuple[mx.array, mx.array, mx.array]] = {}

    def measure(np_: int) -> float:
        if np_ not in probes:
            q = mx.random.normal((b, hq, np_, head_dim), key=key_q).astype(dtype)
            kk = mx.random.normal((b, hkv, np_, head_dim), key=key_k).astype(dtype)
            vv = mx.random.normal((b, hkv, np_, head_dim), key=key_v).astype(dtype)
            mx.eval(q, kk, vv)
            probes[np_] = (q, kk, vv)
            # Metal JIT compiles on the first dispatch at a new probe shape, and GPU
            # clocks/caches are cold -- this dispatch is deliberately unmeasured (mirrors
            # core/kernel/launch.calibrated_rate's per-tile warmup).
            o, lse = launch_flash_fwd(q, kk, vv, scale=scale, causal=causal, tile=TileShape())
            mx.eval(o, lse)
        q, kk, vv = probes[np_]
        t0 = time.perf_counter()
        o, lse = launch_flash_fwd(q, kk, vv, scale=scale, causal=causal, tile=TileShape())
        mx.eval(o, lse)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n,
    )
    rate = SAFETY_FACTOR * raw_rate
    _FWD_RATE_CACHE[key] = rate
    return rate
