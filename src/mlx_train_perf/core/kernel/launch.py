"""Chained-launch driver for the dense MMA kernel. Full buffers + in-kernel offsets —
Python-side w[v0:v1] slices into chained launches cost 1.22 GB of retained copies
(measured; user-metal-kernels workflow-and-gotchas.md).

Rate calibration ramps through a sequence of probe tiles rather than trusting a single
cold dispatch: Apple's GPU DVFS needs sustained work over roughly a second to reach peak
clocks, and a small probe tile starves occupancy, so a lone tile-128 micro-probe measured
the production-tile rate wrong by up to ~3.6x and once projected a safe dispatch as
over-budget. Each ramp stage times one dispatch, projects the next (larger) tile from
that stage's own rate, and stops once the projection stops growing; the final stage then
runs extra sustained dispatches to reach ramped clocks before taking a median-of-3 timed
measurement at the tile the production dispatch will actually use."""
import functools
import math
import statistics
import time
from collections.abc import Callable
from typing import cast

import mlx.core as mx

from mlx_train_perf.core.chunked import QuantSpec
from mlx_train_perf.core.kernel.source import (
    QUANT_HELPERS,
    build_backward_dhidden_source,
    build_backward_dw_source,
    build_dense_source,
    build_quant_source,
)
from mlx_train_perf.errors import LaunchBudgetError, UnsupportedHeadError

MAX_DISPATCH_SECONDS = 1.0
MAX_TOTAL_SECONDS = 60.0

FLOOR_RATE = 10e9        # G MAC/s floor: v0-class, slowest rate ever measured for any variant
SAFETY_FACTOR = 0.5      # halve the measured rate: covers session drift (~12% measured) plus
                         # probe-timing noise with a deliberate 2x margin; costs nothing —
                         # production dispatches sit ~10x under budget even at half rate
_RATE_CACHE: dict[tuple[str, int, str, int], float] = {}


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


@functools.cache
def _backward_dhidden_kernel(row_tiles: int) -> _MetalKernel:
    kernel = mx.fast.metal_kernel(
        name=f"mtp_bwd_dhidden_rt{row_tiles}",
        input_names=["hidden", "w", "targets", "offs", "lse", "cotangent", "d_hidden_in"],
        output_names=["d_hidden_out"],
        source=build_backward_dhidden_source(row_tiles),
    )
    return cast(_MetalKernel, kernel)


def backward_dhidden(
    hidden: mx.array,
    w: mx.array,
    targets: mx.array,
    lse: mx.array,
    tgt: mx.array,  # noqa: ARG001 — d_hidden's math needs only lse (see source.py's
    # derivation comment); accepted for interface parity with the (lse, tgt) residual
    # pair `forward` saves and the combined `launch.backward` call step 3+ introduces.
    cotangent: mx.array,
    *,
    row_tiles: int,
    tile: int,
    rate_macs_per_s: float | None,
) -> mx.array:
    """d_hidden-only backward (Task 16b step 2, v0-correct — frozen/QLoRA head path).

    d_hidden = cotangent * sum_j (P_ij - onehot(j == targets_i)) * w_j, with P_ij
    regenerated tile-wise from the SAVED lse residual (never recomputed) — see
    `core/kernel/source.py`'s derivation comment for the full math, cross-checked against
    `core.chunked.chunked_backward` (the proven oracle) before this kernel was written.

    Chained vocab-tile launches accumulate d_hidden across tiles exactly like `forward`
    chains lse/tgt: full buffers + in-kernel offsets, never a Python-side slice into a
    chained launch (measured 1.22 GB of retained copies otherwise — see
    user-metal-kernels workflow-and-gotchas.md). The fp32 accumulator is cast down to
    `hidden.dtype` exactly once, after the last tile launch.

    Backward recomputes logits tile-wise AND scatter-accumulates into d_hidden — roughly
    2x the forward's per-tile MAC count — so `check_budget` is called at half the supplied
    rate: this scales BOTH the per-dispatch and total-time projections by the same 2x
    factor `check_budget`'s own n*tile*d / n*v*d model uses for the forward.
    """
    n, d = hidden.shape
    v = w.shape[0]
    if rate_macs_per_s is not None:
        check_budget(n=n, d=d, v=v, tile=tile, rate_macs_per_s=rate_macs_per_s / 2.0)
    rows = 8 * row_tiles
    row_blocks = (n + rows - 1) // rows
    kernel = _backward_dhidden_kernel(row_tiles)
    d_hidden = mx.zeros((n, d), dtype=mx.float32)
    ct32 = cotangent.astype(mx.float32)
    tg_y = min(8, row_blocks)
    for v0 in range(0, v, tile):
        v1 = min(v0 + tile, v)
        offs = mx.array([v0, v1], dtype=mx.uint32)
        (d_hidden,) = kernel(
            inputs=[hidden, w, targets, offs, lse, ct32, d_hidden],
            template=[("T", hidden.dtype)],
            grid=(32, row_blocks, 1),
            threadgroup=(32, tg_y, 1),
            output_shapes=[(n, d)],
            output_dtypes=[mx.float32],
        )
    return d_hidden.astype(hidden.dtype)


@functools.cache
def _backward_dw_kernel(row_tiles: int) -> _MetalKernel:
    kernel = mx.fast.metal_kernel(
        name=f"mtp_bwd_dw_rt{row_tiles}",
        input_names=["hidden", "w", "targets", "offs", "lse", "cotangent"],
        output_names=["d_w_out"],
        source=build_backward_dw_source(row_tiles),
        atomic_outputs=True,
    )
    return cast(_MetalKernel, kernel)


def backward_dw(
    hidden: mx.array,
    w: mx.array,
    targets: mx.array,
    lse: mx.array,
    tgt: mx.array,  # noqa: ARG001 — d_w's math needs only lse (see source.py's derivation
    # comment); accepted for interface parity with the (lse, tgt) residual pair `forward`
    # saves and the combined `launch.backward` call step 3+ introduces — same convention
    # as `backward_dhidden`'s unused `tgt`.
    cotangent: mx.array,
    *,
    row_tiles: int,
    tile: int,
    rate_macs_per_s: float | None,
) -> mx.array:
    """d_w (trainable-head) backward (Task 16b step 3, v0-correct).

    d_w[j,:] = sum_i (P_ij - onehot(j == targets_i)) * cotangent_i * hidden_i,:, with
    P_ij regenerated tile-wise from the SAVED lse residual (never recomputed) — see
    `core/kernel/source.py`'s derivation comment for the full math, cross-checked against
    `core.chunked.chunked_backward` (the proven oracle) before this kernel was written.

    Cross-ROW-BLOCK accumulation (many context-row chunks contending on the same d_w
    column slice) is the ONE mechanism d_w needs that d_hidden doesn't — ground-truthed
    correct in Task 16b step 1 (scripts/ground_truth_atomic_outputs.py) via
    `atomic_outputs=True` + `atomic_fetch_add_explicit` on a native Metal
    `device atomic<float>*` output. This makes the result BIT-LEVEL NON-DETERMINISTIC
    run to run (atomics reorder float additions) — see
    tests/test_kernel_backward_parity.py's repeated-run tolerance test.

    Unlike `backward_dhidden`, tile launches need NO input-output accumulator chain: each
    vocab tile's d_w rows are structurally DISJOINT from every other tile's rows (a
    ground-truthed finding, not an assumption — `mx.fast.metal_kernel` allocates a
    genuinely fresh, independently-`init_value`'d output on every call, confirmed by
    calling the same atomic kernel repeatedly and observing zero bleed-through between
    calls), so each launch outputs only its own (tcols, d) fp32 slice and the launcher
    assembles the full (v, d) buffer with `mx.concatenate` — the same pattern
    `chunked_backward`'s own trainable-head branch already uses. Each tile's fp32
    accumulator is cast down to `w.dtype` BEFORE it is appended to the chunk list (not
    after the final concatenate): every chunk is already the COMPLETE final gradient for
    its disjoint rows (no further cross-chunk accumulation ever touches it), so per-chunk
    rounding commutes with concatenation — bit-identical to casting the concatenated
    whole — while halving the transient concat buffer's footprint (fp32 (v, d) -> bf16
    (v, d) at concat time, ~2.49 GB instead of ~4.98 GB at the production shape).

    Backward recomputes logits tile-wise AND scatter-accumulates into d_w — roughly 2x the
    forward's per-tile MAC count, same accounting `backward_dhidden` uses — so
    `check_budget` is called at half the supplied rate.
    """
    n, d = hidden.shape
    v = w.shape[0]
    if rate_macs_per_s is not None:
        check_budget(n=n, d=d, v=v, tile=tile, rate_macs_per_s=rate_macs_per_s / 2.0)
    rows = 8 * row_tiles
    row_blocks = (n + rows - 1) // rows
    kernel = _backward_dw_kernel(row_tiles)
    ct32 = cotangent.astype(mx.float32)
    tg_y = min(8, row_blocks)
    chunks: list[mx.array] = []
    for v0 in range(0, v, tile):
        v1 = min(v0 + tile, v)
        offs = mx.array([v0, v1], dtype=mx.uint32)
        (d_w_tile,) = kernel(
            inputs=[hidden, w, targets, offs, lse, ct32],
            template=[("T", hidden.dtype)],
            grid=(32, row_blocks, 1),
            threadgroup=(32, tg_y, 1),
            output_shapes=[(v1 - v0, d)],
            output_dtypes=[mx.float32],
            init_value=0.0,
        )
        chunks.append(d_w_tile.astype(w.dtype))
    return mx.concatenate(chunks, axis=0)


@functools.cache
def _quant_kernel(row_tiles: int) -> _MetalKernel:
    kernel = mx.fast.metal_kernel(
        name=f"mtp_fused_ce_quant_rt{row_tiles}",
        input_names=["hidden", "wq", "sc", "bi", "targets", "offs", "lse_in", "tgt_in"],
        output_names=["lse_out", "tgt_out"],
        source=build_quant_source(row_tiles),
        header=QUANT_HELPERS,
    )
    return cast(_MetalKernel, kernel)


def forward_quantized(
    hidden: mx.array, q: QuantSpec, targets: mx.array, *,
    row_tiles: int, tile: int, rate_macs_per_s: float | None,
) -> tuple[mx.array, mx.array]:
    """Dequant-in-kernel quantized forward. Same (lse, tgt) contract as `forward`.
    Constraint: 4-bit, group_size=64, d % 64 == 0 (no silent fallback — narrower coverage
    is post-0.1.0)."""
    if q.bits != 4 or q.group_size != 64:
        raise UnsupportedHeadError(
            f"forward_quantized only supports bits=4, group_size=64 (got bits={q.bits}, "
            f"group_size={q.group_size}); use impl='chunked' for other quant configs."
        )
    n, d = hidden.shape
    if d % 64 != 0:
        raise UnsupportedHeadError(
            f"forward_quantized requires d % 64 == 0 (got d={d}); use impl='chunked'."
        )
    v = q.w_q.shape[0]
    if rate_macs_per_s is not None:
        check_budget(n=n, d=d, v=v, tile=tile, rate_macs_per_s=rate_macs_per_s)
    rows = 8 * row_tiles
    row_blocks = (n + rows - 1) // rows
    kernel = _quant_kernel(row_tiles)
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    tgt = mx.zeros((n,), dtype=mx.float32)
    tg_y = min(8, row_blocks)
    for v0 in range(0, v, tile):
        v1 = min(v0 + tile, v)
        offs = mx.array([v0, v1], dtype=mx.uint32)
        lse, tgt = kernel(
            inputs=[hidden, q.w_q, q.scales, q.biases, targets, offs, lse, tgt],
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


def next_probe_tile(*, rate_macs_per_s: float, n: int, d: int, v: int,
                    budget_s: float = MAX_DISPATCH_SECONDS) -> int:
    """Largest power-of-two tile <= min(8192, v) whose projected dispatch time at
    rate_macs_per_s stays within budget_s. Callers pass the SAFETY_FACTOR-halved rate;
    `calibrate`'s ramp uses this the same way to size the NEXT probe one stage ahead of
    what it has actually measured. Floors at 32 even if that tile still projects over
    budget — refusing an over-budget dispatch is `check_budget`'s job, not this sizing
    heuristic's."""
    cap = min(8192, v)
    tile = (1 << (cap.bit_length() - 1)) if cap >= 1 else 32
    tile = max(tile, 32)
    while tile > 32 and n * tile * d / rate_macs_per_s > budget_s:
        tile //= 2
    return tile


def sustain_reps(*, per_dispatch_s: float, target_s: float = 0.75, cap: int = 8) -> int:
    """Extra back-to-back dispatches needed for cumulative sustained work to reach
    target_s. Apple's GPU DVFS needs on the order of a second of continuous load to ramp
    from cold to peak clocks; a single dispatch, however long, only ever runs at whatever
    clock state the GPU happened to already be in. Floors at 1 (always sustain past the
    rate-measuring dispatch itself); caps at 8 (bounds calibration wall time for a kernel
    fast enough that target_s would otherwise demand hundreds of reps)."""
    if per_dispatch_s <= 0:
        return cap
    reps = math.ceil(target_s / per_dispatch_s)
    return min(max(reps, 1), cap)


def calibrate(*, measure: Callable[[int], float], n: int, d: int, v: int,
             start_tile: int, max_stages: int = 3) -> float:
    """Ramp through probe tiles under real (or, in unit tests, fake) dispatch timings
    rather than trusting one cold, occupancy-starved measurement. Each stage times a
    single dispatch at the current tile and projects a (SAFETY_FACTOR-conservative) next
    tile from that stage's own rate, advancing only while the projection keeps growing —
    a kernel already at its safe production tile converges in one stage instead of
    burning the whole ramp. The final tile then gets extra sustained dispatches (see
    `sustain_reps`) to reach ramped clocks, and the reported rate is the MEDIAN of 3 timed
    dispatches at that tile: a single sample is exactly the un-ramped, occupancy-starved
    measurement this function exists to avoid. Returns the raw (un-halved) rate — the
    caller applies SAFETY_FACTOR."""
    tile = start_tile
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(tile)
        raw_rate = n * tile * d / max(per_dispatch_s, 1e-9)
        candidate = next_probe_tile(rate_macs_per_s=SAFETY_FACTOR * raw_rate, n=n, d=d, v=v)
        if candidate <= tile:
            break
        tile = candidate
    for _rep in range(sustain_reps(per_dispatch_s=per_dispatch_s)):
        measure(tile)
    timings = [measure(tile) for _ in range(3)]
    median_s = statistics.median(timings)
    return n * tile * d / max(median_s, 1e-9)


def _probe_hidden(*, dtype: mx.Dtype, n: int, d: int) -> mx.array:
    # Random probe data, NOT zeros: all-zero buffers are the best case for Apple's
    # lossless memory compression and could optimistically bias this safety-relevant
    # rate (T8 review finding). Seeded for determinism.
    mx.random.seed(0)
    hidden = mx.random.normal((n, d)).astype(dtype)
    mx.eval(hidden)
    return hidden


def calibrated_rate(*, row_tiles: int, dtype: mx.Dtype, n: int, d: int, v: int) -> float:
    """Ramped, sustained-load calibration for the dense kernel at the caller's real n
    (small-n rates underestimate large-n rates due to occupancy) — see the module
    docstring for why this ramps instead of micro-probing once. Cached per
    (row_tiles, dtype, n-bucket) so repeated calls at the same occupancy regime don't
    re-run the calibration."""
    key = ("dense", row_tiles, str(dtype), _n_bucket(n))
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    hidden = _probe_hidden(dtype=dtype, n=n, d=d)
    probes: dict[int, tuple[mx.array, mx.array]] = {}

    def measure(tile: int) -> float:
        if tile not in probes:
            w = (mx.random.normal((tile, d)) * 0.05).astype(dtype)
            targets = mx.random.randint(0, tile, (n,))
            mx.eval(w, targets)
            probes[tile] = (w, targets)
            # Metal JIT compiles on the first dispatch at a new tile, and GPU clocks/
            # caches are cold — this dispatch is deliberately unmeasured.
            lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                               rate_macs_per_s=None)
            mx.eval(lse, tgt)
        w, targets = probes[tile]
        t0 = time.perf_counter()
        lse, tgt = forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                           rate_macs_per_s=None)
        mx.eval(lse, tgt)
        return time.perf_counter() - t0

    start_tile = min(probe_tile_for(n=n, d=d), v)
    rate = SAFETY_FACTOR * calibrate(measure=measure, n=n, d=d, v=v, start_tile=start_tile)
    _RATE_CACHE[key] = rate
    return rate


def calibrated_rate_quantized(*, row_tiles: int, dtype: mx.Dtype, n: int, d: int, v: int) -> float:
    """Same ramped, sustained-load calibration as `calibrated_rate`, but the probe head is
    quantized (int4, group_size=64 — the only configuration `forward_quantized` supports)
    and measured through `forward_quantized`. This is the calibration path production
    code uses for the quantized kernel."""
    key = ("quantized", row_tiles, str(dtype), _n_bucket(n))
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    hidden = _probe_hidden(dtype=dtype, n=n, d=d)
    probes: dict[int, tuple[QuantSpec, mx.array]] = {}

    def measure(tile: int) -> float:
        if tile not in probes:
            w = (mx.random.normal((tile, d)) * 0.05).astype(dtype)
            targets = mx.random.randint(0, tile, (n,))
            w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
            mx.eval(w_q, scales, biases, targets)
            q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=64, bits=4)
            probes[tile] = (q, targets)
            # Metal JIT compiles on the first dispatch at a new tile, and GPU clocks/
            # caches are cold — this dispatch is deliberately unmeasured.
            lse, tgt = forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=tile,
                                         rate_macs_per_s=None)
            mx.eval(lse, tgt)
        q, targets = probes[tile]
        t0 = time.perf_counter()
        lse, tgt = forward_quantized(hidden, q, targets, row_tiles=row_tiles, tile=tile,
                                     rate_macs_per_s=None)
        mx.eval(lse, tgt)
        return time.perf_counter() - t0

    start_tile = min(probe_tile_for(n=n, d=d), v)
    rate = SAFETY_FACTOR * calibrate(measure=measure, n=n, d=d, v=v, start_tile=start_tile)
    _RATE_CACHE[key] = rate
    return rate
