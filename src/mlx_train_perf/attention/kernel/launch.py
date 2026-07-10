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

T6's MMA body changes the kernel's cache-residency and reuse pattern entirely
(register/simdgroup-level K/V reuse within each 32-row query block -- NO threadgroup
staging; rung 2 removed all threadgroup memory -- versus v0's zero-reuse per-row
re-stream) -- do not assume v0's RATES transfer to the mma body (rung 3's dispatch table
treats every mma bucket off the one directly-measured saturation shape as PROVISIONAL for
exactly this reason, see `kernel/dispatch.py`). The cache key IS now variant/d_slab-aware
(rung 3: `calibrated_fwd_rate`'s `_FWD_RATE_CACHE` key includes `tile.variant`/
`tile.d_slab`, and `measure()` builds/dispatches the SAME kernel `tile` names -- probe
what you rate), so an mma call never reads a scalar-measured rate or vice versa. What
remains UNVALIDATED is the ramp/canary SHAPE itself: the DRAM-vs-cache-resident-probe
rationale above was derived from v0's zero-cross-row-reuse memory pattern, and the mma
body's block-level reuse has not been separately re-derived -- the ramp mechanics are
REUSED for mma calibration, not re-justified for it.
"""
import functools
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from mlx_train_perf.attention.kernel.source import (
    build_bwd_D_source,
    build_bwd_dkv_source,
    build_bwd_dq_mma_source,
    build_bwd_dq_source,
    build_fwd_mma_source,
    build_fwd_source,
)
from mlx_train_perf.errors import AttentionInputError, LaunchBudgetError

# 0.5, NOT 1.0 (T6 rung-0 kill evidence, macOS 26 / mlx 0.32.0): a projected-1.0s flagship
# dispatch sized from the SAFETY-halved calibrated rate was killed by the OS with
# kIOGPUCommandBufferCallbackErrorImpactingInteractivity -- a softer, EARLIER kill than the
# 5-10s GPU watchdog the 1.0s figure assumed. The CE kernel's shipped ~0.5s-real dispatches
# have never been killed (0.1.0 T13: zero watchdog events), so 0.5s projected (~0.25s real
# behind the 2x SAFETY margin) sits in the proven-safe class. Raising this needs new
# kill-threshold evidence.
MAX_DISPATCH_SECONDS = 0.5
_CANARY_BUDGET_S = 0.1   # projected cost of the calibration's final full-working-set probe
# 2.0, NOT 60 (T6 rung-0, second measurement): the OS kill is per COMMAND BUFFER /
# cumulative eval GPU-time, not per dispatch -- 35 honest ~0.25s-real range dispatches
# packed into one eval (~8.7s total) were killed even though every dispatch was inside its
# own budget; MLX packs consecutive custom-kernel dispatches, and Python cannot flush
# buffers from inside a compiled/traced region. The CE kernel's ~2.2s evals have never
# been killed, so ~2s is the proven-safe class for one eval's packed custom-kernel work.
# Consequence: a flagship v0-scalar forward REFUSES honestly (its 8.7s total cannot ship);
# MMA-class rates fit a full forward far inside this cap.
MAX_TOTAL_SECONDS = 2.0
SAFETY_FACTOR = 0.5      # halve the measured rate (session drift + probe noise, 2x margin)
_PROBE_N_FLOOR = 128     # ramp's minimum/starting probe key-count -- the original fixed
                         # probe shape, small enough to be safe at any plausible v0 rate
_PROBE_N_HARD_CAP = 8192 # never probe past this many keys/queries during calibration,
                         # regardless of the caller's real n (mirrors core/kernel/launch's
                         # 8192 tile cap)

# key: (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab) -- variant/d_slab-aware
# (T6 rung 3): probe what you rate, see calibrated_fwd_rate's own docstring.
_FWD_RATE_CACHE: dict[tuple[int, str, bool, int, int, int, str, int | None], float] = {}

# The installed mlx stub types mx.fast.metal_kernel's return as `object`; it is a callable
# kernel invoker (same cast convention as core/kernel/launch._MetalKernel).
_MetalKernel = Callable[..., list[mx.array]]


@dataclass(frozen=True, slots=True, kw_only=True)
class TileShape:
    """Forward-kernel tiling + variant selector.

    `variant` picks the kernel body: `"scalar"` (the v0 one-thread-per-query-row body,
    default -- unchanged behaviour) or `"mma"` (the rung-1 4x4 simdgroup-matrix body, one
    32-row query block per threadgroup). `bq` is the scalar body's query-rows-per-threadgroup
    occupancy grouping (no effect on the mma body, whose threadgroup is a fixed 32-lane
    simdgroup); the mma body's KV-block dimensions are baked into its source, not carried
    here.

    `d_slab` overrides the mma body's register-resident D-slab width (see `source.py`'s
    `build_fwd_mma_source` / `_FWD_MMA_D_SLAB`) -- `None` means "use the source builder's own
    default" (32, ignored by the scalar body entirely). `provisional` is selection metadata
    only: it marks a `TileShape` picked by `dispatch.select_fwd_tile` for a shape the T6
    ladder did not directly measure (same kernel body, unmeasured rate) -- it is never
    consumed by `_fwd_kernel`/`_dispatch_range`, only carried through for callers/logging. A
    `TileShape` built directly (every existing parity/determinism test does this) defaults
    `provisional=False`, since a caller who names a variant explicitly isn't deferring to the
    table's own confidence flag."""

    bq: int = 32
    variant: str = "scalar"
    d_slab: int | None = None
    provisional: bool = False


def _n_bucket(n: int) -> int:
    return 1 << max(9, (n - 1).bit_length())   # 512, 1024, 2048, ... occupancy regime key


def _fwd_macs_per_row(*, n: int, d: int, b: int, hq: int) -> int:
    """Conservative per-query-row MAC upper bound: each of the n keys costs a D-wide QK dot
    plus a D-wide PV accumulate (2*D MACs), across every (batch, q-head). This over-counts
    causal (a row near the top attends to fewer keys), which is the safe direction for a
    launch-budget guard -- over-estimating cost splits MORE, never under-budgets."""
    return 2 * d * n * b * hq


def _check_launch_budget(*, per_row: int, n: int, rows: int, rate: float) -> None:
    """Refuse-before-launch, shared by the forward and the dQ backward (query-range split
    kernels with the same watchdog exposure): the GPU watchdog SIGABRT is uncatchable.
    Projects both the per-dispatch time (this range of `rows` query rows at the given
    per-row MAC cost) and the whole pass's total time, and raises if either exceeds its
    budget. `per_row` is the caller's conservative per-query-row MAC upper bound (2*D*n*b*hq
    for the forward, 3*D*n*b*hq for dQ) -- the only kernel-specific input to this math."""
    per_dispatch = rows * per_row / rate
    total = n * per_row / rate
    if per_dispatch > MAX_DISPATCH_SECONDS or total > MAX_TOTAL_SECONDS:
        raise LaunchBudgetError(
            f"projected {per_dispatch:.2f} s/dispatch ({rows} rows), {total:.0f} s total "
            f"at {rate / 1e9:.1f} G MAC/s (budget {MAX_DISPATCH_SECONDS} s / "
            f"{MAX_TOTAL_SECONDS} s). Reduce shape/context, or pass a measured rate."
        )


def _rows_within_dispatch_budget(*, per_row: int, n: int, rate: float) -> int:
    """Largest query-row range whose projected dispatch stays within the per-dispatch budget
    at `rate` for the given per-row MAC cost -- shared by the forward and dQ splits. Floors at
    1 row (refusing an over-budget single row is `_check_launch_budget`'s job, not this sizing
    heuristic's); never exceeds `n`."""
    rows = int(MAX_DISPATCH_SECONDS * rate / per_row)
    return max(1, min(n, rows))


def check_fwd_budget(*, n: int, d: int, b: int, hq: int, rows: int, rate: float) -> None:
    """Refuse-before-launch for the forward (see `_check_launch_budget`)."""
    _check_launch_budget(
        per_row=_fwd_macs_per_row(n=n, d=d, b=b, hq=hq), n=n, rows=rows, rate=rate
    )


@functools.cache
def _fwd_kernel(
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None
) -> _MetalKernel:
    """Build (and cache) the forward kernel for a given (head_dim, causal, flip, variant,
    d_slab). `variant="scalar"` uses the v0 one-thread-per-row body (`d_slab` has no effect
    on its source, but stays part of the cache key regardless -- a harmless redundant cache
    entry, never a correctness issue, if a caller ever varies it for scalar); `"mma"` uses the
    rung-2 register-resident P@V MMA body, whose source genuinely changes with `d_slab` (see
    `source.build_fwd_mma_source`'s D_SLAB/D_SLAB_TILES templating) -- `d_slab=None` builds
    with the source builder's own default (`_FWD_MMA_D_SLAB`). Both variants share the same
    (q,k,v,qoffs,scale_in)->(o_out,l_out) contract, so `_dispatch_range` swaps only the
    grid/threadgroup shape between them."""
    if variant == "mma":
        source = build_fwd_mma_source(
            head_dim, causal=causal, flip_causal=flip_causal, d_slab=d_slab,
        )
    elif variant == "scalar":
        source = build_fwd_source(head_dim, causal=causal, flip_causal=flip_causal)
    else:
        raise ValueError(f"unknown forward kernel variant {variant!r}")
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtp_flash_fwd_{variant}_d{head_dim}_"
            f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
            + (f"_s{d_slab}" if variant == "mma" else "")
        ),
        input_names=["q", "k", "v", "qoffs", "scale_in"],
        output_names=["o_out", "l_out"],
        source=source,
    )
    return cast(_MetalKernel, kernel)


def _rows_per_dispatch(*, n: int, d: int, b: int, hq: int, rate: float) -> int:
    """Largest forward query-row range within the per-dispatch budget (see
    `_rows_within_dispatch_budget`)."""
    return _rows_within_dispatch_budget(
        per_row=_fwd_macs_per_row(n=n, d=d, b=b, hq=hq), n=n, rate=rate
    )


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

    kernel = _fwd_kernel(d, causal, _flip_causal, tile.variant, tile.d_slab)
    scale_in = mx.array([scale], dtype=mx.float32)
    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    for r0 in range(0, n, rows_per):
        r1 = min(r0 + rows_per, n)
        o_c, l_c = _dispatch_range(
            kernel, q, k, v, scale_in, r0=r0, r1=r1, tile=tile,
        )
        o_chunks.append(o_c)
        l_chunks.append(l_c)
    return mx.concatenate(o_chunks, axis=2), mx.concatenate(l_chunks, axis=2)


def _dispatch_range(
    kernel: Any, q: mx.array, k: mx.array, v: mx.array, scale_in: mx.array,
    *, r0: int, r1: int, tile: TileShape,
) -> tuple[mx.array, mx.array]:
    """One kernel dispatch covering query rows [r0, r1) of the full problem -- the loop
    body of `launch_flash_fwd`, extracted so the calibration canary can dispatch exactly
    one production-shaped range (the LAST rows: under causal masking only high row
    indices scan the full key working set).

    The two variants differ ONLY in the launch shape: `"scalar"` runs one thread per query
    row (grid.x == rows), while `"mma"` runs one 32-lane simdgroup per 32-row query block
    (grid.x == ceil(rows/32)*32, threadgroup.x == 32). Output shapes/dtypes and the full
    qoffs/buffer contract are identical, so the reassembly in `launch_flash_fwd` is
    variant-agnostic."""
    b, hq, _, d = q.shape
    rows_this = r1 - r0
    qoffs = mx.array([r0, r1], dtype=mx.uint32)
    if tile.variant == "mma":
        num_blocks = (rows_this + 31) // 32              # one 32-row query block per simdgroup
        grid = (num_blocks * 32, b * hq, 1)
        threadgroup = (32, 1, 1)
    else:
        grid = (rows_this, b * hq, 1)
        threadgroup = (min(tile.bq, rows_this), 1, 1)
    o_c, l_c = kernel(
        inputs=[q, k, v, qoffs, scale_in],
        template=[("T", q.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(b, hq, rows_this, d), (b, hq, rows_this)],
        output_dtypes=[q.dtype, mx.float32],
    )
    return o_c, l_c


def _start_probe_n(n: int) -> int:
    """Ramp's starting probe key-count: `_PROBE_N_FLOOR` (the original fixed probe size,
    small enough to be safe at any plausible v0 rate) capped at the caller's real `n` -- a
    caller whose real context is already smaller than the floor gets measured AT n
    directly rather than padded up to a probe size it will never actually run."""
    return min(_PROBE_N_FLOOR, n)


def _next_probe_n(
    *, rate_macs_per_s: float, n: int, d: int, b: int, hq: int,
    budget_s: float = MAX_DISPATCH_SECONDS,
    macs_per_row: Callable[..., int] = _fwd_macs_per_row,
) -> int:
    """Largest power-of-two probe key-count <= min(n, `_PROBE_N_HARD_CAP`) whose projected
    SELF-attention dispatch (`np` query rows over `np` keys -- QUADRATIC in `np`, unlike
    the CE kernel's tile-linear cost) stays within `budget_s` at `rate_macs_per_s`. Mirrors
    `core/kernel/launch.next_probe_tile`'s role in the ramp: callers pass the
    SAFETY_FACTOR-halved rate, and `_calibrate_fwd`'s ramp uses this to size the NEXT probe
    one stage ahead of what it has actually measured.

    `macs_per_row` is the per-query-row MAC cost model (default the forward's `2*D`; the
    backward rate ramp passes the dK/dV `4*D` cost via `calibrated_bwd_rate`) -- the only
    kernel-specific input, so the SAME ramp machinery sizes both the forward and the backward
    probe (design point 4). When the caller's real `n` is already <= `_PROBE_N_FLOOR`, this
    returns `n` directly without a budget check -- that shape was already measured successfully
    as the ramp's OWN starting probe (`_start_probe_n` never exceeds the floor either), so
    re-deriving it here is redundant, not unsafe. Otherwise floors at `_PROBE_N_FLOOR` even if
    that probe still projects over budget -- refusing an over-budget PRODUCTION dispatch is
    `check_fwd_budget`'s job, not this sizing heuristic's."""
    cap = min(n, _PROBE_N_HARD_CAP)
    if cap <= _PROBE_N_FLOOR:
        return cap
    np_ = 1 << (cap.bit_length() - 1)   # largest power of two <= cap
    np_ = max(np_, _PROBE_N_FLOOR)
    while np_ > _PROBE_N_FLOOR and (
        macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / rate_macs_per_s > budget_s
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


def _canary_rows(
    *, raw_ramp_rate: float, n: int, d: int, b: int, hq: int,
    macs_per_row: Callable[..., int] = _fwd_macs_per_row,
) -> int:
    """Row count for the calibration's final CANARY probe: the largest query-row range
    whose dispatch against the FULL n-key working set projects within `_CANARY_BUDGET_S`
    at the SAFETY-halved ramp rate. Floors at 1 (a 1-row full-n dispatch is the smallest
    measurable production-shaped unit); never exceeds n. `macs_per_row` (default the
    forward's `2*D`) selects the kernel's per-query-row cost model -- the backward rate ramp
    passes the dK/dV `4*D` cost."""
    per_row = macs_per_row(n=n, d=d, b=b, hq=hq)
    rows = int(_CANARY_BUDGET_S * SAFETY_FACTOR * raw_ramp_rate / per_row)
    return max(1, min(n, rows))


def _calibrate_fwd(
    *, measure: Callable[[int, int], float], n: int, d: int, b: int, hq: int,
    start_n: int, max_stages: int = 3,
    macs_per_row: Callable[..., int] = _fwd_macs_per_row,
) -> float:
    """Ramp through probe key-counts under real (or, in unit tests, fake) dispatch
    timings, mirroring `core/kernel/launch.calibrate`'s ramp/sustain/median shape exactly
    (see that function's docstring) -- adapted to THIS kernel's quadratic-in-probe-size
    cost model via `_next_probe_n` in place of `next_probe_tile`. `measure(rows, keys)`
    times one dispatch of `rows` query rows against `keys` keys; ramp stages are
    self-shaped (rows == keys == np_). The final ramp size gets sustained dispatches
    (`_sustain_reps`) to reach ramped GPU clocks and a median-of-3 measurement -- and then
    the rate is re-derived from a CANARY: a small-row-range dispatch against the FULL
    n-key working set. The ramp's self-shaped probes under-populate the production
    working set (few rows x ALL keys x every head is DRAM-bound where a self-probe still
    partly fits cache), and the measured consequence of trusting them was a
    macOS-interactivity-killed command buffer at flagship shape (T6 rung 0) -- so the
    returned rate comes from the canary, the only probe that sees production conditions.
    Skipped only when the ramp already measured the full n x n shape (harsher than any
    range dispatch). Returns the raw (un-halved) rate -- the caller applies SAFETY_FACTOR.

    `macs_per_row` (default the forward's `2*D`) is the per-query-row MAC cost model; the
    backward rate (`calibrated_bwd_rate`) reuses this exact ramp/canary machinery by passing
    the dK/dV `4*D` cost (design point 4) -- so the ONLY kernel-specific input is this one
    additive parameter. T6's KV-block tiling changes the cost model: re-validate both budgets
    then."""
    np_ = start_n
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(np_, np_)
        raw_rate = macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / max(per_dispatch_s, 1e-9)
        candidate = _next_probe_n(
            rate_macs_per_s=SAFETY_FACTOR * raw_rate, n=n, d=d, b=b, hq=hq,
            macs_per_row=macs_per_row,
        )
        if candidate <= np_:
            break
        np_ = candidate
    for _rep in range(_sustain_reps(per_dispatch_s=per_dispatch_s)):
        measure(np_, np_)
    timings = [measure(np_, np_) for _ in range(3)]
    median_s = statistics.median(timings)
    ramp_rate = macs_per_row(n=np_, d=d, b=b, hq=hq) * np_ / max(median_s, 1e-9)
    if np_ >= n:
        return ramp_rate
    rows = _canary_rows(
        raw_ramp_rate=ramp_rate, n=n, d=d, b=b, hq=hq, macs_per_row=macs_per_row,
    )
    canary_timings = [measure(rows, n) for _ in range(3)]
    canary_median = statistics.median(canary_timings)
    return macs_per_row(n=n, d=d, b=b, hq=hq) * rows / max(canary_median, 1e-9)


def calibrated_fwd_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool,
    tile: TileShape,
) -> float:
    """Cached, safety-factored, N-AWARE MAC/s throughput for the forward kernel `tile`
    actually names, used to size the query-row split. Ramps the probe key-count toward the
    caller's real `n` via `_calibrate_fwd` (see the module docstring for why a fixed small
    probe reads the wrong cache-resident regime at flagship N) rather than measuring at one
    fixed shape; probe QKV are drawn from a LOCAL `mx.random.key(0)` (split into per-tensor
    sub-keys), never `mx.random.seed`, so calibration never mutates the caller's global RNG
    stream.

    PROBE WHAT YOU RATE (T6 rung 3): `measure()` builds and dispatches the SAME
    (`tile.variant`, `tile.d_slab`) kernel the launcher will actually run -- rating one
    variant while dispatching another sizes the query-row split from the wrong rate. The
    cache is keyed on (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab), so an mma
    and a scalar call (or two mma calls with different `d_slab`) at the same shape are
    calibrated independently and never share a rate. `provisional` is deliberately NOT part
    of the key: it is a selection-confidence label on `tile`, not a distinct kernel/dispatch
    configuration -- the rate for a given (variant, d_slab) is the same physical number
    whichever confidence flag pointed at it. Must never be called inside a compiled region
    (host-sync timing)."""
    key = (head_dim, str(dtype), causal, b, hq, _n_bucket(n), tile.variant, tile.d_slab)
    if key in _FWD_RATE_CACHE:
        return _FWD_RATE_CACHE[key]

    key_q, key_k, key_v = mx.random.split(mx.random.key(0), 3)
    scale = 1.0 / (head_dim ** 0.5)
    probes: dict[tuple[int, int], tuple[mx.array, mx.array, mx.array]] = {}

    kernel = _fwd_kernel(head_dim, causal, False, tile.variant, tile.d_slab)
    scale_in = mx.array([scale], dtype=mx.float32)

    def measure(rows: int, keys: int) -> float:
        # Dispatches query rows [keys-rows, keys) against a full `keys`-key working set --
        # under causal masking only HIGH row indices scan every key, so the canary (rows <
        # keys) must be the LAST range, exactly the production tail dispatch. Self-shaped
        # ramp probes (rows == keys) reduce to the full [0, keys) dispatch.
        if (rows, keys) not in probes:
            q = mx.random.normal((b, hq, keys, head_dim), key=key_q).astype(dtype)
            kk = mx.random.normal((b, hkv, keys, head_dim), key=key_k).astype(dtype)
            vv = mx.random.normal((b, hkv, keys, head_dim), key=key_v).astype(dtype)
            mx.eval(q, kk, vv)
            probes[(rows, keys)] = (q, kk, vv)
            # Metal JIT compiles on the first dispatch at a new probe shape, and GPU
            # clocks/caches are cold -- this dispatch is deliberately unmeasured (mirrors
            # core/kernel/launch.calibrated_rate's per-tile warmup).
            o, lse = _dispatch_range(
                kernel, q, kk, vv, scale_in, r0=keys - rows, r1=keys, tile=tile,
            )
            mx.eval(o, lse)
        q, kk, vv = probes[(rows, keys)]
        t0 = time.perf_counter()
        o, lse = _dispatch_range(
            kernel, q, kk, vv, scale_in, r0=keys - rows, r1=keys, tile=tile,
        )
        mx.eval(o, lse)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n,
    )
    rate = SAFETY_FACTOR * raw_rate
    _FWD_RATE_CACHE[key] = rate
    return rate


# ---------------------------------------------------------------------------------------
# Backward: D-preprocess launcher -- T7. See source.py's block comment above
# `_BWD_D_TEMPLATE` for the full design (one simdgroup per (b, hq, row) triple, no
# splitting/chaining needed -- MEASURED 0.638 ms/dispatch at the flagship shape, ~780x
# under the 0.5 s per-dispatch budget (T7 review probe), so there is no LaunchBudgetError
# guard here, unlike the forward's query-range split or the CE kernel's chained vocab
# tiles).
# ---------------------------------------------------------------------------------------


def _validate_bwd_D_shapes(d_o: mx.array, o: mx.array) -> None:  # noqa: N802 -- D is the paper's name
    for name, arr in (("dO", d_o), ("O", o)):
        if arr.ndim != 4:
            raise AttentionInputError(
                f"{name} must be 4-D (B, Hq, N, D); got shape {arr.shape}"
            )
    if d_o.shape != o.shape:
        raise AttentionInputError(f"dO/O shape mismatch: dO={d_o.shape}, O={o.shape}")


@functools.cache
def _bwd_d_kernel(head_dim: int, drop_product: bool) -> _MetalKernel:
    """Build (and cache) the D-preprocess kernel for a given (head_dim, drop_product).
    `drop_product` is TEST-ONLY (see `build_bwd_D_source`'s docstring); it stays part of
    the cache key so a perturbed and a correct kernel at the same head_dim never collide."""
    kernel = mx.fast.metal_kernel(
        name=f"mtp_bwd_D_d{head_dim}" + ("_dropx" if drop_product else ""),
        input_names=["d_o", "o"],
        output_names=["d_out"],
        source=build_bwd_D_source(head_dim, drop_product=drop_product),
    )
    return cast(_MetalKernel, kernel)


def launch_bwd_D(  # noqa: N802 -- D is the paper's name
    d_o: mx.array, o: mx.array, *, _drop_product: bool = False,
) -> mx.array:
    """`D (B, Hq, N)` fp32 = rowsum(dO * O) -- the flash-attention paper's row-correction
    term for `dS`. `dO`/`O` are both `(B, Hq, N, D)`, bf16 or fp32 (matched by the `T`
    template, taken from `d_o.dtype`); `head_dim` (`d_o.shape[-1]`) must be one of
    `build_bwd_D_source`'s supported {64, 96, 128}. Raises `AttentionInputError` on a
    rank or shape mismatch between `dO` and `O`.

    `_drop_product` is TEST-ONLY (wrong-value perturbation -- see source.py's
    `build_bwd_D_source`). Never used by production code."""
    _validate_bwd_D_shapes(d_o, o)
    b, hq, n, d = d_o.shape
    kernel = _bwd_d_kernel(d, _drop_product)
    (d_out,) = kernel(
        inputs=[d_o, o],
        template=[("T", d_o.dtype)],
        grid=(32, n, b * hq),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, n)],
        output_dtypes=[mx.float32],
    )
    return d_out


# ---------------------------------------------------------------------------------------
# Backward: dQ launcher -- T8. One owner per query row (see source.py's block comment above
# `_BWD_DQ_TEMPLATE`). dQ rows are DISJOINT across query blocks -- no accumulator chaining --
# so this reuses the FORWARD's query-range multi-dispatch split machinery (`_check_launch_
# budget` / `_rows_within_dispatch_budget`, the same watchdog budgets), differing only in the
# per-row MAC cost (3*D vs the forward's 2*D: a QK dot + a dO.V dot + a dq accumulate per key)
# and the single dQ output. It SPLITS rather than refuses; `LaunchBudgetError` is raised only
# when even one minimal range (or the whole dQ pass's total) over-budgets.
#
# The BACKWARD-specific calibrated rate is NOT wired here: T8 accepts `rate_macs_per_s=None`
# (single dispatch, the tiny-shape/test path) or a caller-passed rate; the calibrated-rate
# wiring into api.py lands with T9. Calibration must never run inside the compiled backward
# (a host-synced probe there breaks compile / dumps ~1s of probe dispatches into every step).
# ---------------------------------------------------------------------------------------

_BWD_DQ_THREADGROUP = 32   # one thread per query row; 32 (SIMD width) groups them for occupancy


def _bwd_dq_macs_per_row(*, n: int, d: int, b: int, hq: int) -> int:
    """Conservative per-query-row MAC upper bound for dQ: each of the n keys costs a QK dot
    (D), a dO.V dot (D), and a dq accumulate (D) == 3*D MACs, across every (batch, q-head).
    Over-counts causal (a row near the top loops fewer keys), the safe direction for a
    launch-budget guard -- over-estimating cost splits MORE, never under-budgets."""
    return 3 * d * n * b * hq


def _validate_bwd_dq_shapes(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array,
) -> None:
    """Raise `AttentionInputError` on a rank/shape/dtype mismatch at the dQ boundary, before
    any Metal kernel is built. q/k/v/dO are 4-D (B,H,N,D); lse/D are 3-D (B,Hq,N); dO shares
    q's shape; k/v share Hkv with Hq a multiple of it; N/D match across q/k/v; and q/k/v/dO
    share one dtype (the kernel templates a single `T` and reads k/v/dO through it)."""
    for name, arr in (("q", q), ("k", k), ("v", v), ("dO", d_o)):
        if arr.ndim != 4:
            raise AttentionInputError(
                f"{name} must be 4-D (B, H, N, D); got shape {arr.shape}"
            )
    for name, arr in (("lse", lse), ("D", d_arr)):
        if arr.ndim != 3:
            raise AttentionInputError(
                f"{name} must be 3-D (B, Hq, N); got shape {arr.shape}"
            )
    b, hq, n, d = q.shape
    if not (k.shape[0] == v.shape[0] == d_o.shape[0] == b):
        raise AttentionInputError(
            f"batch mismatch: q={b}, k={k.shape[0]}, v={v.shape[0]}, dO={d_o.shape[0]}"
        )
    hkv = k.shape[1]
    if v.shape[1] != hkv:
        raise AttentionInputError(
            f"k/v head-count mismatch: k has {hkv} heads, v has {v.shape[1]}"
        )
    if hq % hkv != 0:
        raise AttentionInputError(f"Hq={hq} must be a multiple of Hkv={hkv} for GQA grouping")
    if d_o.shape != q.shape:
        raise AttentionInputError(f"dO/q shape mismatch: dO={d_o.shape}, q={q.shape}")
    if not (k.shape[2] == v.shape[2] == n):
        raise AttentionInputError(
            f"sequence length mismatch: q={n}, k={k.shape[2]}, v={v.shape[2]}"
        )
    if not (k.shape[3] == v.shape[3] == d):
        raise AttentionInputError(
            f"head_dim mismatch: q={d}, k={k.shape[3]}, v={v.shape[3]}"
        )
    if lse.shape != (b, hq, n) or d_arr.shape != (b, hq, n):
        raise AttentionInputError(
            f"lse/D must be (B={b}, Hq={hq}, N={n}); got lse={lse.shape}, D={d_arr.shape}"
        )
    _validate_bwd_residual_dtypes(lse, d_arr)
    if len({q.dtype, k.dtype, v.dtype, d_o.dtype}) != 1:
        raise AttentionInputError(
            f"q/k/v/dO must share a dtype; got q={q.dtype}, k={k.dtype}, "
            f"v={v.dtype}, dO={d_o.dtype}"
        )


def _validate_bwd_residual_dtypes(lse: mx.array, d_arr: mx.array) -> None:
    """L and D seed the backward and are read as FIXED fp32 device buffers the kernel never
    templates (the forward's fp32-L convention -- the residuals that reconstruct the backward
    stay fp32 always). A bf16 lse/D would be reinterpreted as raw fp32 bytes and silently
    corrupt every gradient, so both launchers reject a non-fp32 residual up front."""
    if lse.dtype != mx.float32 or d_arr.dtype != mx.float32:
        raise AttentionInputError(
            f"lse/D must be fp32 (the fixed backward residual dtype); got "
            f"lse={lse.dtype}, D={d_arr.dtype}"
        )


@functools.cache
def _bwd_dq_kernel(
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None
) -> _MetalKernel:
    """Build (and cache) the dQ kernel for a given (head_dim, causal, flip_causal, variant,
    d_slab). `variant="scalar"` uses the v1 one-thread-per-query-row body (`d_slab` has no
    effect on its source, but stays part of the cache key regardless -- a harmless redundant
    entry, never a correctness issue, if a caller ever varies it for scalar); `"mma"` uses the
    T9b rung-B1 register-resident D-slabbed body, whose source genuinely changes with `d_slab`
    (`d_slab=None` builds with the source builder's own default `_BWD_DQ_MMA_D_SLAB`). Both
    variants share the same (q,k,v,dO,lse,d_arr,qoffs,scale_in)->(dq_out) contract, so
    `_dispatch_bwd_dq_range` swaps only the grid/threadgroup shape between them. `flip_causal`
    is TEST-ONLY (see `build_bwd_dq_source` / `build_bwd_dq_mma_source`); it stays part of the
    cache key so a perturbed and a correct kernel at the same (head_dim, causal) never collide."""
    if variant == "mma":
        source = build_bwd_dq_mma_source(
            head_dim, causal=causal, flip_causal=flip_causal, d_slab=d_slab,
        )
    elif variant == "scalar":
        source = build_bwd_dq_source(head_dim, causal=causal, flip_causal=flip_causal)
    else:
        raise ValueError(f"unknown dQ kernel variant {variant!r}")
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtp_flash_bwd_dq_{variant}_d{head_dim}_"
            f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
            + (f"_s{d_slab}" if variant == "mma" else "")
        ),
        input_names=["q", "k", "v", "d_o", "lse", "d_arr", "qoffs", "scale_in"],
        output_names=["dq_out"],
        source=source,
    )
    return cast(_MetalKernel, kernel)


def _dispatch_bwd_dq_range(
    kernel: _MetalKernel, q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, scale_in: mx.array, *, r0: int, r1: int,
    variant: str = "scalar",
) -> mx.array:
    """One dQ dispatch covering query rows [r0, r1) of the full problem, writing this range's
    own tile-local (b, hq, rows, d) dQ chunk. Full q/k/v/dO/L/D buffers + an in-kernel `qoffs`
    row offset (never a Python-side slice).

    The two variants differ ONLY in the launch shape (same output shapes/dtypes + full qoffs
    contract, so the reassembly in `launch_bwd_dq` is variant-agnostic, mirroring the forward's
    `_dispatch_range`): `"scalar"` runs one thread per query row (grid.x == rows), while `"mma"`
    runs one 32-lane simdgroup per 32-row query block (grid.x == ceil(rows/32)*32,
    threadgroup.x == 32)."""
    b, hq, _, d = q.shape
    rows_this = r1 - r0
    qoffs = mx.array([r0, r1], dtype=mx.uint32)
    if variant == "mma":
        num_blocks = (rows_this + 31) // 32              # one 32-row query block per simdgroup
        grid = (num_blocks * 32, b * hq, 1)
        threadgroup = (32, 1, 1)
    else:
        grid = (rows_this, b * hq, 1)
        threadgroup = (min(_BWD_DQ_THREADGROUP, rows_this), 1, 1)
    (dq_c,) = kernel(
        inputs=[q, k, v, d_o, lse, d_arr, qoffs, scale_in],
        template=[("T", q.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(b, hq, rows_this, d)],
        output_dtypes=[q.dtype],
    )
    return dq_c


def launch_bwd_dq(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, *,
    scale: float, causal: bool, rate_macs_per_s: float | None = None,
    variant: str = "scalar", d_slab: int | None = None,
    _flip_causal: bool = False,
) -> mx.array:
    """dQ backward -> the query gradient, with q's shape/dtype. Consumes the forward's saved
    L (`lse`, fp32 (B, Hq, N)) and T7's D (`d_arr`, fp32 (B, Hq, N)); recomputes S/P from
    q/k and accumulates `dQ_i += scale*P*(dP - D)*k` in fp32 per causally-allowed key.

    `variant` picks the kernel body: `"scalar"` (default -- the v1 one-thread-per-query-row
    body, unchanged behaviour for every existing caller) or `"mma"` (the T9b rung-B1 4x4
    simdgroup-matrix body with a register-resident D-slabbed accumulator). `d_slab` overrides
    the mma body's slab width (`None` = the source builder's own default; ignored by the scalar
    body). The mma variant is a CORRECTNESS + small-shape rung: it is not wired into api.py or
    the calibrated-rate path (graduation + the saturation d_slab sweep are a later rung).

    `rate_macs_per_s`: when None, a single dispatch over all rows with no budget check (safe
    only at small N -- the tiny-shape/test path); when given, the launcher sizes the query-row
    split from it and refuses (`LaunchBudgetError`) if even one minimal range or the total
    over-budgets. dQ rows are disjoint across query blocks, so the reassembly (a plain
    `mx.concatenate` over the row axis) needs no accumulator chaining, and this holds for the
    mma variant too (each 32-row query block's dQ depends only on its own absolute rows).

    `_flip_causal` is TEST-ONLY (wrong-triangle causal-skip perturbation -- see source.py)."""
    _validate_bwd_dq_shapes(q, k, v, d_o, lse, d_arr)
    b, hq, n, d = q.shape
    if rate_macs_per_s is None:
        rows_per = n
    else:
        per_row = _bwd_dq_macs_per_row(n=n, d=d, b=b, hq=hq)
        rows_per = _rows_within_dispatch_budget(per_row=per_row, n=n, rate=rate_macs_per_s)
        _check_launch_budget(per_row=per_row, n=n, rows=rows_per, rate=rate_macs_per_s)

    kernel = _bwd_dq_kernel(d, causal, _flip_causal, variant, d_slab)
    scale_in = mx.array([scale], dtype=mx.float32)
    dq_chunks: list[mx.array] = []
    for r0 in range(0, n, rows_per):
        r1 = min(r0 + rows_per, n)
        dq_chunks.append(
            _dispatch_bwd_dq_range(
                kernel, q, k, v, d_o, lse, d_arr, scale_in, r0=r0, r1=r1, variant=variant,
            )
        )
    return mx.concatenate(dq_chunks, axis=2)


# ---------------------------------------------------------------------------------------
# Backward: dK/dV split-partials launcher -- T9. One owner per (batch, kv_head, key) (see
# source.py's block comment above `_BWD_DKV_TEMPLATE`). UNLIKE the forward/dQ splits (disjoint
# outputs, `mx.concatenate`), key j's dK/dV receive a contribution from EVERY query row i >= j
# across ALL query blocks, so the fp32 accumulators are CHAINED across query-range dispatches
# exactly like the CE forward chains lse/tgt: fresh `dk_out`/`dv_out` buffers per call, each
# seeded from the prior dispatch's output (`dk_in`/`dv_in`), full buffers + in-kernel offsets
# (never a Python-side slice -- the CE kernel's 1.22 GB retained-copy lesson). The fp32
# accumulator is cast down to k/v dtype exactly once, after the last dispatch. `plan_dkv_
# dispatches` sizes the ascending contiguous query-range split from the calibrated backward
# rate and refuses (`LaunchBudgetError`) if even one minimal range -- or the whole pass's
# total wall -- over-budgets (the shared `_check_launch_budget` math, at the dK/dV 4*D cost).
# ---------------------------------------------------------------------------------------

_BWD_DKV_THREADGROUP = 32   # one thread per key; 32 (SIMD width) groups them for occupancy

# key: (head_dim, dtype, causal, b, hq, n-bucket) -- the backward rate is a single MAC/s
# throughput number sizing BOTH the dQ and dK/dV query-range splits (each launcher applies its
# own per-row MAC cost), calibrated on the dK/dV kernel (the 4*D-per-pair heavier path).
_BWD_RATE_CACHE: dict[tuple[int, str, bool, int, int, int], float] = {}


def _bwd_dkv_macs_per_row(*, n: int, d: int, b: int, hq: int) -> int:
    """Conservative per-query-row MAC upper bound for dK/dV: each of the n keys costs an
    s = q.k dot (D), a dp = dO.v dot (D), a dV accumulate (D) and a dK accumulate (D) == 4*D
    MACs, across every (batch, q-head). Over-counts causal (a query near the top touches fewer
    keys), the safe direction for a launch-budget guard -- over-estimating cost splits MORE,
    never under-budgets. The 4*D (heavier) of the two backward kernels, so a rate calibrated on
    dK/dV also conservatively sizes the dQ split when the two share one throughput number."""
    return 4 * d * n * b * hq


def plan_dkv_dispatches(
    *, n: int, d: int, b: int, hq: int, rate: float,
) -> list[tuple[int, int]]:
    """Ascending contiguous query-range split `[(q_lo, q_hi), ...]` tiling `[0, n)` exactly for
    the chained dK/dV backward, sized from the calibrated backward `rate` via the shared budget
    helpers (`_rows_within_dispatch_budget` / `_check_launch_budget`) at the dK/dV `4*D` cost.
    Pure integer arithmetic -- no GPU, no allocation. Raises `LaunchBudgetError` (the 0.1.0
    refusal contract) when even one minimal range (`rows == 1`) over-budgets the per-dispatch
    bound OR the whole pass's total wall exceeds `MAX_TOTAL_SECONDS`, never returns an
    over-budget range for the uncatchable GPU watchdog to hit."""
    per_row = _bwd_dkv_macs_per_row(n=n, d=d, b=b, hq=hq)
    rows_per = _rows_within_dispatch_budget(per_row=per_row, n=n, rate=rate)
    _check_launch_budget(per_row=per_row, n=n, rows=rows_per, rate=rate)
    return [(q_lo, min(q_lo + rows_per, n)) for q_lo in range(0, n, rows_per)]


def _validate_bwd_dkv_shapes(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array,
) -> None:
    """Raise `AttentionInputError` on a rank/shape/dtype mismatch at the dK/dV boundary, before
    any Metal kernel is built. Identical contract to the dQ boundary (`_validate_bwd_dq_shapes`)
    -- q/k/v/dO 4-D (B,H,N,D) sharing one dtype, lse/D 3-D (B,Hq,N) fp32, GQA divisibility,
    matched N/D/batch -- so it delegates to the dQ validator verbatim rather than duplicating
    the checks."""
    _validate_bwd_dq_shapes(q, k, v, d_o, lse, d_arr)


@functools.cache
def _bwd_dkv_kernel(head_dim: int, causal: bool, flip_causal: bool) -> _MetalKernel:
    """Build (and cache) the dK/dV kernel for a given (head_dim, causal, flip_causal).
    `flip_causal` is TEST-ONLY (see `build_bwd_dkv_source`); it stays part of the cache key so
    a perturbed and a correct kernel at the same (head_dim, causal) never collide."""
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtp_flash_bwd_dkv_d{head_dim}_"
            f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
        ),
        input_names=[
            "q", "k", "v", "d_o", "lse", "d_arr", "dk_in", "dv_in", "qoffs", "scale_in",
        ],
        output_names=["dk_out", "dv_out"],
        source=build_bwd_dkv_source(head_dim, causal=causal, flip_causal=flip_causal),
    )
    return cast(_MetalKernel, kernel)


def _dispatch_bwd_dkv_range(
    kernel: _MetalKernel, q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, dk_in: mx.array, dv_in: mx.array, scale_in: mx.array,
    *, q_lo: int, q_hi: int,
) -> tuple[mx.array, mx.array]:
    """One dK/dV dispatch accumulating query rows [q_lo, q_hi) into the chained fp32 partials.
    One thread per key (grid.x == n, ALL keys are owners in every dispatch -- a key with no
    causally-allowed query in the range copies its `dk_in`->`dk_out` slot unchanged, carrying
    the accumulator forward). Full q/k/v/dO/L/D + `dk_in`/`dv_in` buffers + an in-kernel `qoffs`
    range offset (never a Python-side slice); returns the fresh fp32 `dk_out`/`dv_out`."""
    b, _hq, n, d = q.shape
    hkv = k.shape[1]
    qoffs = mx.array([q_lo, q_hi], dtype=mx.uint32)
    dk_out, dv_out = kernel(
        inputs=[q, k, v, d_o, lse, d_arr, dk_in, dv_in, qoffs, scale_in],
        template=[("T", q.dtype)],
        grid=(n, b * hkv, 1),
        threadgroup=(min(_BWD_DKV_THREADGROUP, n), 1, 1),
        output_shapes=[(b, hkv, n, d), (b, hkv, n, d)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return dk_out, dv_out


def launch_bwd_dkv(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, *,
    scale: float, causal: bool, rate_macs_per_s: float | None = None,
    _flip_causal: bool = False,
) -> tuple[mx.array, mx.array]:
    """dK/dV backward -> (dK, dV), with k/v's shape/dtype. Consumes the forward's saved L
    (`lse`, fp32 (B, Hq, N)) and T7's D (`d_arr`, fp32 (B, Hq, N)); recomputes S/P from q/k and
    accumulates `dV_j += P*dO`, `dK_j += scale*P*(dP - D)*q` in CHAINED fp32 partials over the
    causally-allowed queries, grouped over each kv head's contiguous q-head group.

    `rate_macs_per_s`: when None, a single dispatch over all query rows with no budget check
    (safe only at small N -- the tiny-shape/test path); when given, the launcher sizes the
    ascending query-range split via `plan_dkv_dispatches` and refuses (`LaunchBudgetError`) if
    even one minimal range or the total over-budgets. The fp32 accumulators are chained across
    dispatches (seeded from the prior's output), so a range split is bit-identical to a single
    dispatch, and cast to k/v dtype exactly once after the last dispatch.

    `_flip_causal` is TEST-ONLY (wrong-triangle causal-skip perturbation -- see source.py)."""
    _validate_bwd_dkv_shapes(q, k, v, d_o, lse, d_arr)
    b, hq, n, d = q.shape
    hkv = k.shape[1]
    if rate_macs_per_s is None:
        ranges = [(0, n)]
    else:
        ranges = plan_dkv_dispatches(n=n, d=d, b=b, hq=hq, rate=rate_macs_per_s)

    kernel = _bwd_dkv_kernel(d, causal, _flip_causal)
    scale_in = mx.array([scale], dtype=mx.float32)
    dk = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dv = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    for q_lo, q_hi in ranges:
        dk, dv = _dispatch_bwd_dkv_range(
            kernel, q, k, v, d_o, lse, d_arr, dk, dv, scale_in, q_lo=q_lo, q_hi=q_hi,
        )
    return dk.astype(k.dtype), dv.astype(v.dtype)


def calibrated_bwd_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool,
) -> float:
    """Cached, safety-factored, N-aware MAC/s throughput for the chained dK/dV backward kernel,
    used to size BOTH the dQ and dK/dV query-range splits at construction time (never inside the
    vjp -- host-sync timing is compile-hostile). A single backward throughput number: each
    launcher applies its own per-row MAC cost, and calibrating on the dK/dV kernel (the 4*D
    heavier of the two) keeps the shared rate conservative for the dQ split.

    Reuses the forward's ramp/canary machinery (`_calibrate_fwd`) via the dK/dV per-row cost
    model (`_bwd_dkv_macs_per_row`) -- the design-point-4 macs-per-row parameterization. Same
    discipline as `calibrated_fwd_rate`: probe QKV/dO are drawn from a LOCAL `mx.random.key(0)`
    (never `mx.random.seed`, so calibration never mutates the caller's global RNG stream), the
    rate is cached per occupancy regime, and it must never be called inside a compiled region."""
    key = (head_dim, str(dtype), causal, b, hq, _n_bucket(n))
    if key in _BWD_RATE_CACHE:
        return _BWD_RATE_CACHE[key]

    key_q, key_k, key_v, key_do = mx.random.split(mx.random.key(0), 4)
    scale = 1.0 / (head_dim ** 0.5)
    kernel = _bwd_dkv_kernel(head_dim, causal, False)
    scale_in = mx.array([scale], dtype=mx.float32)
    probes: dict[tuple[int, int], tuple[mx.array, ...]] = {}

    def measure(rows: int, keys: int) -> float:
        # Times one dK/dV dispatch of query rows [keys-rows, keys) against a full `keys`-key
        # working set -- the production tail range (high query indices scan the most keys under
        # causal), mirroring calibrated_fwd_rate.measure. lse/D are zeros (this is a TIMING
        # probe -- the kernel does identical FLOPs regardless of the residual values).
        if (rows, keys) not in probes:
            qp = mx.random.normal((b, hq, keys, head_dim), key=key_q).astype(dtype)
            kp = mx.random.normal((b, hkv, keys, head_dim), key=key_k).astype(dtype)
            vp = mx.random.normal((b, hkv, keys, head_dim), key=key_v).astype(dtype)
            dop = mx.random.normal((b, hq, keys, head_dim), key=key_do).astype(dtype)
            lsep = mx.zeros((b, hq, keys), dtype=mx.float32)
            dp = mx.zeros((b, hq, keys), dtype=mx.float32)
            dk0 = mx.zeros((b, hkv, keys, head_dim), dtype=mx.float32)
            dv0 = mx.zeros((b, hkv, keys, head_dim), dtype=mx.float32)
            mx.eval(qp, kp, vp, dop, lsep, dp, dk0, dv0)
            probes[(rows, keys)] = (qp, kp, vp, dop, lsep, dp, dk0, dv0)
            # Metal JIT + cold clocks/caches: this first dispatch is deliberately unmeasured
            # (mirrors calibrated_fwd_rate's per-shape warmup).
            wdk, wdv = _dispatch_bwd_dkv_range(
                kernel, qp, kp, vp, dop, lsep, dp, dk0, dv0, scale_in,
                q_lo=keys - rows, q_hi=keys,
            )
            mx.eval(wdk, wdv)
        qp, kp, vp, dop, lsep, dp, dk0, dv0 = probes[(rows, keys)]
        t0 = time.perf_counter()
        rdk, rdv = _dispatch_bwd_dkv_range(
            kernel, qp, kp, vp, dop, lsep, dp, dk0, dv0, scale_in,
            q_lo=keys - rows, q_hi=keys,
        )
        mx.eval(rdk, rdv)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n,
        macs_per_row=_bwd_dkv_macs_per_row,
    )
    rate = SAFETY_FACTOR * raw_rate
    _BWD_RATE_CACHE[key] = rate
    return rate
