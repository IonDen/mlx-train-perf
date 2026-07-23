"""Query-range multi-dispatch driver for the v0 flash-attention forward kernel.

Split from day one (review-mlx High): the GPU watchdog applies PER DISPATCH, and the
attention forward has the same O(N^2 . D . Hq) per-layer scaling as the backward -- at the
16-32k context ambition a single forward dispatch over all query blocks over-budgets at
any rate. Forward O/L rows are DISJOINT across query blocks, so the launcher loops
query-row-range dispatches, each writing its own tile-local (b, hq, rows, d) chunk (the CE
forward's disjoint-output pattern -- no accumulator chaining), and reassembles with
`mx.concatenate`. It SPLITS rather than refuses; `LaunchBudgetError` is raised only when
the shape cannot be planned within the per-command-buffer budget (the 0.3.0 buffer model,
backlog 0025 -- see `plan_budgeted_ranges` and the constants block below).

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
    build_bwd_dkv_mma_source,
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
# kill-threshold evidence. A 2026-07-10 ground-truth probe (13 conditions,
# _artifacts/attention_bwd_rungs/buffer_packing_probe/) found the kill SYSTEM-STATE-DEPENDENT:
# real 1.0s single dispatches SURVIVED, but only on a PERMISSIVE (idle-display) day -- the kill
# is display/UI-contention-driven, so this stays pinned to the WORST observed day and a
# permissive-day survival never licenses raising it.
MAX_DISPATCH_SECONDS = 0.5
_CANARY_BUDGET_S = 0.1   # projected cost of the calibration's final full-working-set probe
# 0.3.0 (backlog 0025, launch-budget evidence study): the OS kill unit is the individual
# COMMAND BUFFER -- there is no chain-total or per-eval kill mechanism (the retired
# MAX_TOTAL_SECONDS = 2.0 guarded a mis-attributed reading of the rung-0 kill: those "35
# packed dispatches" were really ~4-6-dispatch ~1.0-1.5 s packed BUFFERS dying on a
# kill-active day; external corroboration ml-explore/mlx#3267). mlx 0.32.0 commits a buffer
# when it holds >50 ops OR >50 M unique input+output ELEMENTS (device.cpp needs_commit,
# L512-515; array.data_size() counts elements; outputs count via set_output_array; both
# counters reset per commit; 50/50 is the LARGEST arch class ('s'/'d' -- max/ultra), so
# modeling with it never predicts a commit an arch with smaller limits would skip). The
# planner (`plan_budgeted_ranges`) models that composition and caps each MODELED BUFFER's
# summed projected time at MAX_DISPATCH_SECONDS -- which therefore stays the ONE pinned
# worst-day budget (projected ~0.5 s = ~0.25 s real behind the 2x SAFETY margin; never
# killed on any observed day, while ~0.5-1.0 s-real buffers died on the kill-active day).
# Full evidence: docs/superpowers/research/2026-07-14-mlx-train-perf-launch-budget-evidence.md
# (workspace root) + scripts/probe_command_buffer_packing.py.
_PACK_COMMIT_ELEMS = 51 << 20   # buffer_sizes_ >> 20 > 50  <=>  elements >= 51 * 2^20
_PACK_COMMIT_OPS = 51           # buffer_ops_ > 50          <=>  ops >= 51
SAFETY_FACTOR = 0.5      # halve the measured rate (session drift + probe noise, 2x margin)
_PROBE_N_FLOOR = 128     # ramp's minimum/starting probe key-count -- the original fixed
                         # probe shape, small enough to be safe at any plausible v0 rate
_PROBE_N_HARD_CAP = 8192 # never probe past this many keys/queries during calibration,
                         # regardless of the caller's real n (mirrors core/kernel/launch's
                         # 8192 tile cap)

# key: (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab, packed) -- variant/d_slab/
# packed-aware (T6 rung 3 + 0.4.0): probe what you rate, see calibrated_fwd_rate's own docstring.
_FWD_RATE_CACHE: dict[
    tuple[int, str, bool, int, int, int, str, int | None, bool], float
] = {}

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


def causal_pairs(r0: int, r1: int) -> int:
    """Exact (query, key) pair count for causal query rows [r0, r1): row i attends to the
    i+1 keys j <= i, so this is `sum_{i=r0}^{r1-1} (i+1) = (r1-r0)(r0+r1+1)/2`. The 0.3.0
    projection basis (backlog 0025): the old full-rectangle per-row cost over-charged causal
    work ~2x -- an upper bound the budget model no longer needs, since the exact count is
    itself an upper bound of nothing (it IS the work) and the 2x SAFETY_FACTOR on the rate
    carries the margin."""
    rows = r1 - r0
    return rows * (r0 + r1 + 1) // 2


def range_macs(*, r0: int, r1: int, n: int, pair_cost: int, causal: bool) -> int:
    """Exact MAC count for query rows [r0, r1) of an n-key pass. `pair_cost` is the MACs one
    (query, key) pair costs across every (batch, q-head): `2*D*b*hq` forward, `3*D*b*hq` dQ,
    `4*D*b*hq` dK/dV -- i.e. `macs_per_row(n=1, ...)` of the matching nominal cost model.
    Causal charges the exact triangle; non-causal charges the full rectangle (exact there)."""
    if causal:
        return causal_pairs(r0, r1) * pair_cost
    return (r1 - r0) * n * pair_cost


def _widest_within(
    *, r0: int, n: int, pair_cost: int, rate: float, causal: bool,
    budget_s: float, block_align: int,
) -> int:
    """Widest row count `w` (a `block_align` multiple, except a final tail reaching n) whose
    range [r0, r0+w) projects within `budget_s` at `rate`; 0 when even the minimal unit does
    not fit. Pure arithmetic; the closed-form causal solve is belt-verified against
    `range_macs` so float error can never return an over-budget width."""
    budget_macs = budget_s * rate
    if budget_macs <= 0:
        return 0
    if causal:
        # w rows starting at r0 cost w*(2*r0 + w + 1)/2 pairs: solve the quadratic bound.
        bp = budget_macs / pair_cost
        a = 2 * r0 + 1
        w = int((math.sqrt(a * a + 8.0 * bp) - a) / 2)
    else:
        w = int(budget_macs / (n * pair_cost))
    w = min(w, n - r0)
    if w < n - r0 and block_align > 1:
        w = (w // block_align) * block_align
    step = block_align if block_align > 1 else 1
    while w > 0 and (
        range_macs(r0=r0, r1=r0 + w, n=n, pair_cost=pair_cost, causal=causal) / rate
        > budget_s
    ):
        w -= step
    return max(0, w)


def plan_budgeted_ranges(
    *, n: int, pair_cost: int, rate: float, causal: bool,
    live_input_elems: int,
    output_elems_per_range: Callable[[int, int], int] | None = None,
    block_align: int = 1,
) -> list[tuple[int, int]]:
    """Ascending contiguous query ranges tiling [0, n) exactly, sized so that every MODELED
    COMMAND BUFFER's summed projected time stays within `MAX_DISPATCH_SECONDS` -- the 0.3.0
    launch guard (backlog 0025: the macOS interactivity kill applies to an individual command
    buffer, never a chain or eval total, so there is NO chain-total cap).

    Buffer composition follows mlx 0.32.0's verified commit rule (module constants above): a
    modeled buffer accumulates our dispatches until its GUARANTEED-counted unique elements
    reach `_PACK_COMMIT_ELEMS` or its ops reach `_PACK_COMMIT_OPS`, then resets.
    Guaranteed-counted = `live_input_elems` (the caller-held tensors every dispatch reads --
    stable buffers, counted once per command buffer by mlx) plus, when the caller's per-range
    outputs provably stay live across the whole chain (the forward's O/L chunks, dQ's chunks
    -- all held until the final concatenate), `output_elems_per_range` per dispatch. The
    chained dK/dV passes None: its intermediate accumulators are freed mid-chain and their
    buffers can be recycled by the allocator, so crediting them could model a commit reality
    skips -- the ONLY unsafe direction. Ops other frameworks interleave between our
    dispatches only ADD elements/ops (earlier real commits = shorter real buffers = safe).

    Consequences: at flagship shapes (unique inputs alone >= the threshold) every dispatch
    owns its buffer and a chain may project an UNBOUNDED total; at small-footprint shapes
    dispatches must be assumed to pack, so the whole chain is capped at one buffer budget --
    tighter than the retired 2.0 s chain cap, and honest about the mechanism.

    `block_align` rounds range widths DOWN to block multiples (the mma dK/dV variant passes
    32 so a split never bisects a query block -- the chained bit-identity contract), except a
    final tail that reaches n. Raises `LaunchBudgetError` (the 0.1.0 refusal contract) when
    even one minimal unit cannot fit its modeled buffer -- never returns an over-budget range
    for the uncatchable GPU watchdog to hit."""
    ranges: list[tuple[int, int]] = []
    r0 = 0
    buf_elems = 0
    buf_ops = 0
    buf_time = 0.0
    while r0 < n:
        if buf_ops == 0:
            buf_elems = live_input_elems
        w = _widest_within(
            r0=r0, n=n, pair_cost=pair_cost, rate=rate, causal=causal,
            budget_s=MAX_DISPATCH_SECONDS - buf_time, block_align=block_align,
        )
        if w == 0:
            unit = min(n - r0, max(1, block_align))
            t_unit = range_macs(
                r0=r0, r1=r0 + unit, n=n, pair_cost=pair_cost, causal=causal
            ) / rate
            if buf_ops == 0:
                raise LaunchBudgetError(
                    f"projected {t_unit:.2f} s for the minimal {unit}-row range at query "
                    f"row {r0} exceeds the {MAX_DISPATCH_SECONDS} s per-command-buffer "
                    f"budget at {rate / 1e9:.1f} G MAC/s. Reduce shape/context, or pass a "
                    "measured rate."
                )
            raise LaunchBudgetError(
                f"chained dispatches cannot be guaranteed their own command buffers "
                f"(unique input elements {live_input_elems} < the {_PACK_COMMIT_ELEMS} "
                f"commit threshold), and a packed buffer would exceed the "
                f"{MAX_DISPATCH_SECONDS} s budget ({buf_time:.2f} s already modeled across "
                f"{buf_ops} dispatches at {rate / 1e9:.1f} G MAC/s). Reduce shape/context, "
                "or pass a measured rate."
            )
        r1 = r0 + w
        buf_time += range_macs(r0=r0, r1=r1, n=n, pair_cost=pair_cost, causal=causal) / rate
        buf_ops += 1
        if output_elems_per_range is not None:
            buf_elems += output_elems_per_range(r0, r1)
        ranges.append((r0, r1))
        r0 = r1
        if buf_elems >= _PACK_COMMIT_ELEMS or buf_ops >= _PACK_COMMIT_OPS:
            buf_elems = 0
            buf_ops = 0
            buf_time = 0.0
    return ranges


def plan_fwd_dispatches(
    *, n: int, d: int, b: int, hq: int, hkv: int, rate: float, causal: bool,
) -> list[tuple[int, int]]:
    """The forward's query-range plan: 2*D per pair; live inputs q + k + v; O/L chunk
    outputs credited (all chunks stay live until the reassembling concatenate)."""
    return plan_budgeted_ranges(
        n=n, pair_cost=2 * d * b * hq, rate=rate, causal=causal,
        live_input_elems=b * hq * n * d + 2 * b * hkv * n * d,
        output_elems_per_range=lambda r0, r1: b * hq * (r1 - r0) * (d + 1),
    )


def check_fwd_budget(
    *, n: int, d: int, b: int, hq: int, hkv: int, rate: float, causal: bool,
) -> None:
    """Refuse-before-launch for the forward: raises `LaunchBudgetError` iff the shape cannot
    be planned within the per-command-buffer budget (see `plan_budgeted_ranges`)."""
    plan_fwd_dispatches(n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate, causal=causal)


@functools.cache
def _fwd_kernel(
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None,
    packed: bool = False,
) -> _MetalKernel:
    """Build (and cache) the forward kernel for a given (head_dim, causal, flip, variant,
    d_slab, packed). `variant="scalar"` uses the v0 one-thread-per-row body (`d_slab` has no
    effect on its source, but stays part of the cache key regardless -- a harmless redundant
    cache entry, never a correctness issue, if a caller ever varies it for scalar); `"mma"`
    uses the rung-2 register-resident P@V MMA body, whose source genuinely changes with
    `d_slab` (see `source.build_fwd_mma_source`'s D_SLAB/D_SLAB_TILES templating) --
    `d_slab=None` builds with the source builder's own default (`_FWD_MMA_D_SLAB`).

    `packed=True` (0.4.0) builds the block-diagonal-segment variant: the source's keep
    predicate gains a same-segment term and the launcher binds two extra int32 buffers
    (`seg_id`, `seg_start`), appended to `input_names` LAST so the non-packed buffer order
    (q,k,v,qoffs,scale_in -> o_out,l_out) is untouched (mlx binds `input_names[i]` to
    `inputs[i]` positionally). `packed` is the LAST cache-key component and defaults False, so
    every pre-0.4.0 caller (and the calibration path, which never packs) keeps its 5-argument
    call and its existing cache entry unchanged. Both variants share the same
    (...,qoffs,scale_in[,seg_id,seg_start])->(o_out,l_out) contract, so `_dispatch_range`
    swaps only the grid/threadgroup shape and (for packed) the two trailing inputs."""
    if variant == "mma":
        source = build_fwd_mma_source(
            head_dim, causal=causal, flip_causal=flip_causal, d_slab=d_slab, packed=packed,
        )
    elif variant == "scalar":
        source = build_fwd_source(
            head_dim, causal=causal, flip_causal=flip_causal, packed=packed,
        )
    else:
        raise ValueError(f"unknown forward kernel variant {variant!r}")
    input_names = ["q", "k", "v", "qoffs", "scale_in"]
    if packed:
        input_names += ["seg_id", "seg_start"]
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtp_flash_fwd_{variant}_d{head_dim}_"
            f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
            + (f"_s{d_slab}" if variant == "mma" else "")
            + ("_p" if packed else "")
        ),
        input_names=input_names,
        output_names=["o_out", "l_out"],
        source=source,
    )
    return cast(_MetalKernel, kernel)


def _validate_segments(
    seg_id: mx.array | None, seg_start: mx.array | None, *, b: int, n: int, causal: bool,
) -> bool:
    """Validate the optional packed-attention segment pair and return whether packing is on.

    Both-or-neither, both int32, both shape (B, N) -- the (B, N) row-contiguous int32 buffer
    contract the kernel indexes as `seg_id[b * n + pos]`. Packed attention is block-diagonal
    ON TOP of the causal triangle, so it requires `causal=True` (the reference oracle asserts
    the same). Raises `AttentionInputError` at the boundary before any kernel is built."""
    if (seg_id is None) != (seg_start is None):
        raise AttentionInputError(
            "seg_id and seg_start must be provided together (both or neither)"
        )
    if seg_id is None:
        return False
    if not causal:
        raise AttentionInputError("segments (packed attention) require causal=True")
    for name, arr in (("seg_id", seg_id), ("seg_start", cast(mx.array, seg_start))):
        if arr.dtype != mx.int32:
            raise AttentionInputError(f"{name} must be int32; got {arr.dtype}")
        if arr.shape != (b, n):
            raise AttentionInputError(
                f"{name} must be (B={b}, N={n}); got {arr.shape}"
            )
    return True


def launch_flash_fwd(
    q: mx.array, k: mx.array, v: mx.array, *,
    scale: float, causal: bool, tile: TileShape,
    rate_macs_per_s: float | None = None,
    seg_id: mx.array | None = None,
    seg_start: mx.array | None = None,
    _flip_causal: bool = False,
    _force_ranges: list[tuple[int, int]] | None = None,
) -> tuple[mx.array, mx.array]:
    """v0 flash-attention forward -> (O, L). O has q's shape/dtype; L is (B, Hq, N) fp32.

    `rate_macs_per_s`: when None, a single dispatch over all rows with no budget check
    (safe only at small N -- the direct/test path); when given, the launcher plans the
    query-row split via `plan_fwd_dispatches` (exact-causal costs, per-command-buffer
    budget) and refuses (`LaunchBudgetError`) when the shape cannot be planned. The API
    path always passes a calibrated rate (see `calibrated_fwd_rate`), so a flagship call
    splits instead of tripping the watchdog.

    `seg_id` / `seg_start` (0.4.0, both-or-neither, int32 (B, N), require `causal=True`)
    switch on PACKED block-diagonal-causal attention: key `kk` reaches query `row` only when
    both are in the same segment AND causal. The two buffers are bound LAST (after
    q/k/v/qoffs/scale_in) so the non-packed binding order is untouched, and each query-range
    dispatch reads the FULL (B, N) buffers with an in-kernel offset (never a Python-side
    slice). The split/reassembly is segment-agnostic -- a row's O/L depend only on its own
    absolute position, its segment, and the keys.

    `_flip_causal` is TEST-ONLY (wrong-mask perturbation -- see source.py). `_force_ranges`
    is TEST-ONLY too (the split-forcing seam: the production planner never splits a tiny
    packed-regime shape, but the split/reassembly contract still needs its own proof).
    """
    b, hq, n, d = q.shape
    hkv = k.shape[1]
    packed = _validate_segments(seg_id, seg_start, b=b, n=n, causal=causal)
    if _force_ranges is not None:
        ranges = _force_ranges
    elif rate_macs_per_s is None:
        ranges = [(0, n)]
    else:
        ranges = plan_fwd_dispatches(
            n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate_macs_per_s, causal=causal,
        )

    kernel = _fwd_kernel(d, causal, _flip_causal, tile.variant, tile.d_slab, packed)
    scale_in = mx.array([scale], dtype=mx.float32)
    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    for r0, r1 in ranges:
        o_c, l_c = _dispatch_range(
            kernel, q, k, v, scale_in, r0=r0, r1=r1, tile=tile,
            seg_id=seg_id, seg_start=seg_start,
        )
        o_chunks.append(o_c)
        l_chunks.append(l_c)
    if len(o_chunks) == 1:
        return o_chunks[0], l_chunks[0]
    return mx.concatenate(o_chunks, axis=2), mx.concatenate(l_chunks, axis=2)


def _dispatch_range(
    kernel: Any, q: mx.array, k: mx.array, v: mx.array, scale_in: mx.array,
    *, r0: int, r1: int, tile: TileShape,
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """One kernel dispatch covering query rows [r0, r1) of the full problem -- the loop
    body of `launch_flash_fwd`, extracted so the calibration canary can dispatch exactly
    one production-shaped range (the LAST rows: under causal masking only high row
    indices scan the full key working set).

    The two variants differ ONLY in the launch shape: `"scalar"` runs one thread per query
    row (grid.x == rows), while `"mma"` runs one 32-lane simdgroup per 32-row query block
    (grid.x == ceil(rows/32)*32, threadgroup.x == 32). Output shapes/dtypes and the full
    qoffs/buffer contract are identical, so the reassembly in `launch_flash_fwd` is
    variant-agnostic.

    `seg_id`/`seg_start` (both-or-neither) are the packed-attention buffers; when present they
    are appended to `inputs` LAST, in `input_names` order (the `_fwd_kernel` packed contract),
    and passed as the FULL (B, N) buffers -- the kernel offsets in with `seg_off = b * n`, so a
    query-range dispatch never slices them Python-side."""
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
    inputs = [q, k, v, qoffs, scale_in]
    if seg_id is not None:
        inputs += [seg_id, cast(mx.array, seg_start)]
    o_c, l_c = kernel(
        inputs=inputs,
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

    `macs_per_row` is the per-query-row MAC cost model (default the forward's `2*D`; the dQ rate
    ramp passes `3*D` via `calibrated_bwd_dq_rate`, the dK/dV rate ramp `4*D` via
    `calibrated_bwd_dkv_rate`) -- the only kernel-specific input, so the SAME ramp machinery sizes
    the forward and both backward probes (design point 4). When the caller's real `n` is already
    <= `_PROBE_N_FLOOR`, this
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
    causal: bool,
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

    `macs_per_row` (default the forward's `2*D`) is the per-query-row MAC cost model; the two
    per-kernel backward rates (`calibrated_bwd_dq_rate` at `3*D`, `calibrated_bwd_dkv_rate` at
    `4*D`) reuse this exact ramp/canary machinery by passing their own cost (design point 4) -- so
    the ONLY kernel-specific input is this one additive parameter. T6's KV-block tiling changes
    the cost model: re-validate both budgets then.

    0.3.0 (backlog 0025): rates are CREDITED with the probe's exact-causal work when
    `causal=True` (a self-shaped [0, np) probe does np(np+1)/2 pairs, not np^2; the tail
    canary [n-rows, n) does its own exact triangle slice), so the returned MAC/s means
    "causal-true MACs per second" -- consistent with `plan_budgeted_ranges`' projection
    accounting. Probe SIZING (`_next_probe_n`/`_canary_rows`) keeps the nominal full-
    rectangle cost: with causal-credited rates that over-estimates probe cost and sizes
    probes smaller, the safe direction."""
    pair_cost = macs_per_row(n=1, d=d, b=b, hq=hq)

    def _credit(rows: int, keys: int) -> int:
        return range_macs(r0=keys - rows, r1=keys, n=keys, pair_cost=pair_cost, causal=causal)

    np_ = start_n
    per_dispatch_s = 0.0
    for _stage in range(max_stages):
        per_dispatch_s = measure(np_, np_)
        raw_rate = _credit(np_, np_) / max(per_dispatch_s, 1e-9)
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
    ramp_rate = _credit(np_, np_) / max(median_s, 1e-9)
    if np_ >= n:
        return ramp_rate
    rows = _canary_rows(
        raw_ramp_rate=ramp_rate, n=n, d=d, b=b, hq=hq, macs_per_row=macs_per_row,
    )
    canary_timings = [measure(rows, n) for _ in range(3)]
    canary_median = statistics.median(canary_timings)
    return _credit(rows, n) / max(canary_median, 1e-9)


def _packed_probe_segs(
    b: int, packed: bool
) -> Callable[[int], tuple[mx.array | None, mx.array | None]]:
    """Segment buffers for a PACKED-rate probe: a synthetic SINGLE segment spanning the whole row
    (`seg_id`/`seg_start` all zeros, shape (b, keys)), cached per key-count so the timed dispatch
    reuses ready buffers. All-zeros = one segment covering every position, so the packed kernel
    walks the full causal triangle with the predicate always true -- the worst-case work a packed
    dispatch does (a real multi-segment layout does strictly less), which is exactly what the rate
    must size for. When `packed` is False the closure returns (None, None) so the causal probe
    binds no segment inputs and the non-packed rate path stays byte-identical."""
    cache: dict[int, tuple[mx.array, mx.array]] = {}

    def seg_for(keys: int) -> tuple[mx.array | None, mx.array | None]:
        if not packed:
            return None, None
        if keys not in cache:
            seg_id = mx.zeros((b, keys), dtype=mx.int32)
            seg_start = mx.zeros((b, keys), dtype=mx.int32)
            mx.eval(seg_id, seg_start)
            cache[keys] = (seg_id, seg_start)
        return cache[keys]

    return seg_for


def calibrated_fwd_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool,
    tile: TileShape, packed: bool = False,
) -> float:
    """Cached, safety-factored, N-AWARE MAC/s throughput for the forward kernel `tile`
    actually names, used to size the query-row split. Ramps the probe key-count toward the
    caller's real `n` via `_calibrate_fwd` (see the module docstring for why a fixed small
    probe reads the wrong cache-resident regime at flagship N) rather than measuring at one
    fixed shape; probe QKV are drawn from a LOCAL `mx.random.key(0)` (split into per-tensor
    sub-keys), never `mx.random.seed`, so calibration never mutates the caller's global RNG
    stream.

    PROBE WHAT YOU RATE (T6 rung 3): `measure()` builds and dispatches the SAME
    (`tile.variant`, `tile.d_slab`, `packed`) kernel the launcher will actually run -- rating one
    variant while dispatching another sizes the query-row split from the wrong rate. The
    cache is keyed on (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab, packed), so an
    mma and a scalar call (or two mma calls with different `d_slab`, or a packed vs a causal call)
    at the same shape are calibrated independently and never share a rate. `provisional` is
    deliberately NOT part of the key: it is a selection-confidence label on `tile`, not a distinct
    kernel/dispatch configuration -- the rate for a given (variant, d_slab) is the same physical
    number whichever confidence flag pointed at it. Must never be called inside a compiled region
    (host-sync timing).

    `packed=True` (0.4.0) rates the block-diagonal PACKED kernel via a synthetic SINGLE-segment
    layout (`seg_id`/`seg_start` all zeros): that walks the full causal triangle plus the segment
    predicate overhead -- the WORST-case work a packed dispatch does (a real multi-segment layout
    does strictly less), so the measured rate sizes the split conservatively. Same causal-true MAC
    accounting as the causal probe (single-segment = full triangle)."""
    key = (
        head_dim, str(dtype), causal, b, hq, _n_bucket(n), tile.variant, tile.d_slab, packed
    )
    if key in _FWD_RATE_CACHE:
        return _FWD_RATE_CACHE[key]

    key_q, key_k, key_v = mx.random.split(mx.random.key(0), 3)
    scale = 1.0 / (head_dim ** 0.5)
    probes: dict[tuple[int, int], tuple[mx.array, mx.array, mx.array]] = {}

    kernel = _fwd_kernel(head_dim, causal, False, tile.variant, tile.d_slab, packed)
    scale_in = mx.array([scale], dtype=mx.float32)
    seg_for = _packed_probe_segs(b, packed)

    def measure(rows: int, keys: int) -> float:
        # Dispatches query rows [keys-rows, keys) against a full `keys`-key working set --
        # under causal masking only HIGH row indices scan every key, so the canary (rows <
        # keys) must be the LAST range, exactly the production tail dispatch. Self-shaped
        # ramp probes (rows == keys) reduce to the full [0, keys) dispatch.
        seg_id, seg_start = seg_for(keys)
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
                seg_id=seg_id, seg_start=seg_start,
            )
            mx.eval(o, lse)
        q, kk, vv = probes[(rows, keys)]
        t0 = time.perf_counter()
        o, lse = _dispatch_range(
            kernel, q, kk, vv, scale_in, r0=keys - rows, r1=keys, tile=tile,
            seg_id=seg_id, seg_start=seg_start,
        )
        mx.eval(o, lse)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n, causal=causal,
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
# so this reuses the FORWARD's query-range planning machinery (`plan_budgeted_ranges`, the
# same per-command-buffer budget), differing only in the per-pair MAC cost (3*D vs the
# forward's 2*D: a QK dot + a dO.V dot + a dq accumulate per key) and the single dQ output.
# It SPLITS rather than refuses; `LaunchBudgetError` is raised only when the shape cannot be
# planned within the per-buffer budget (see `plan_dq_dispatches`).
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
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None,
    packed: bool = False,
) -> _MetalKernel:
    """Build (and cache) the dQ kernel for a given (head_dim, causal, flip_causal, variant,
    d_slab, packed). `variant="scalar"` uses the v1 one-thread-per-query-row body (`d_slab` has
    no effect on its source, but stays part of the cache key regardless -- a harmless redundant
    entry, never a correctness issue, if a caller ever varies it for scalar); `"mma"` uses the
    T9b rung-B1 register-resident D-slabbed body, whose source genuinely changes with `d_slab`
    (`d_slab=None` builds with the source builder's own default `_BWD_DQ_MMA_D_SLAB`). Both
    variants share the same (q,k,v,dO,lse,d_arr,qoffs,scale_in)->(dq_out) contract, so
    `_dispatch_bwd_dq_range` swaps only the grid/threadgroup shape between them. `flip_causal`
    is TEST-ONLY (see `build_bwd_dq_source` / `build_bwd_dq_mma_source`); it stays part of the
    cache key so a perturbed and a correct kernel at the same (head_dim, causal) never collide.

    `packed=True` (0.4.0) builds the block-diagonal-segment variant (see the source builders'
    packed docstrings): the keep predicate gains a same-segment term and the launcher binds two
    extra int32 buffers (`seg_id`, `seg_start`) appended to `input_names` LAST so the non-packed
    order (q,k,v,d_o,lse,d_arr,qoffs,scale_in -> dq_out) is untouched (mlx binds positionally).
    `packed` is the LAST cache-key component and defaults False, so every pre-0.4.0 caller keeps
    its existing 5-argument call and cache entry unchanged."""
    if variant == "mma":
        source = build_bwd_dq_mma_source(
            head_dim, causal=causal, flip_causal=flip_causal, d_slab=d_slab, packed=packed,
        )
    elif variant == "scalar":
        source = build_bwd_dq_source(
            head_dim, causal=causal, flip_causal=flip_causal, packed=packed,
        )
    else:
        raise ValueError(f"unknown dQ kernel variant {variant!r}")
    input_names = ["q", "k", "v", "d_o", "lse", "d_arr", "qoffs", "scale_in"]
    if packed:
        input_names += ["seg_id", "seg_start"]
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtp_flash_bwd_dq_{variant}_d{head_dim}_"
            f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
            + (f"_s{d_slab}" if variant == "mma" else "")
            + ("_p" if packed else "")
        ),
        input_names=input_names,
        output_names=["dq_out"],
        source=source,
    )
    return cast(_MetalKernel, kernel)


def _dispatch_bwd_dq_range(
    kernel: _MetalKernel, q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, scale_in: mx.array, *, r0: int, r1: int,
    variant: str = "scalar",
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
) -> mx.array:
    """One dQ dispatch covering query rows [r0, r1) of the full problem, writing this range's
    own tile-local (b, hq, rows, d) dQ chunk. Full q/k/v/dO/L/D buffers + an in-kernel `qoffs`
    row offset (never a Python-side slice).

    The two variants differ ONLY in the launch shape (same output shapes/dtypes + full qoffs
    contract, so the reassembly in `launch_bwd_dq` is variant-agnostic, mirroring the forward's
    `_dispatch_range`): `"scalar"` runs one thread per query row (grid.x == rows), while `"mma"`
    runs one 32-lane simdgroup per 32-row query block (grid.x == ceil(rows/32)*32,
    threadgroup.x == 32).

    `seg_id`/`seg_start` (both-or-neither) are the packed-attention buffers; when present they
    are appended to `inputs` LAST, in `input_names` order (the `_bwd_dq_kernel` packed contract),
    and passed as the FULL (B, N) buffers -- the kernel offsets in with `seg_off = b * n`, so a
    query-range dispatch never slices them Python-side."""
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
    inputs = [q, k, v, d_o, lse, d_arr, qoffs, scale_in]
    if seg_id is not None:
        inputs += [seg_id, cast(mx.array, seg_start)]
    (dq_c,) = kernel(
        inputs=inputs,
        template=[("T", q.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(b, hq, rows_this, d)],
        output_dtypes=[q.dtype],
    )
    return dq_c


def plan_dq_dispatches(
    *, n: int, d: int, b: int, hq: int, hkv: int, rate: float, causal: bool,
) -> list[tuple[int, int]]:
    """The dQ backward's query-range plan: 3*D per pair; live inputs q/k/v/dO/L/D; dQ chunk
    outputs credited (all chunks stay live until the reassembling concatenate)."""
    return plan_budgeted_ranges(
        n=n, pair_cost=3 * d * b * hq, rate=rate, causal=causal,
        live_input_elems=2 * b * hq * n * d + 2 * b * hkv * n * d + 2 * b * hq * n,
        output_elems_per_range=lambda r0, r1: b * hq * (r1 - r0) * d,
    )


def launch_bwd_dq(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, *,
    scale: float, causal: bool, rate_macs_per_s: float | None = None,
    variant: str = "scalar", d_slab: int | None = None,
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
    _flip_causal: bool = False,
    _force_ranges: list[tuple[int, int]] | None = None,
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
    only at small N -- the tiny-shape/test path); when given, the launcher plans the query-row
    split via `plan_dq_dispatches` (exact-causal costs, per-command-buffer budget) and refuses
    (`LaunchBudgetError`) when the shape cannot be planned. dQ rows are disjoint across query
    blocks, so the reassembly (a plain `mx.concatenate` over the row axis) needs no accumulator
    chaining, and this holds for the mma variant too (each 32-row query block's dQ depends only
    on its own absolute rows).

    `seg_id` / `seg_start` (0.4.0, both-or-neither, int32 (B, N), require `causal=True`) switch
    on PACKED block-diagonal-causal attention: key `kk` reaches query `row` only when both are
    in the same segment AND causal. The two buffers are bound LAST (after q/.../scale_in) so the
    non-packed binding order is untouched, and each query-range dispatch reads the FULL (B, N)
    buffers with an in-kernel offset (never a Python-side slice). The row-range split/reassembly
    is segment-agnostic -- a row's dQ depends only on its own absolute position, its segment, and
    the keys.

    `_flip_causal` is TEST-ONLY (wrong-triangle causal-skip perturbation -- see source.py);
    `_force_ranges` is TEST-ONLY too (the split-forcing seam -- the production planner never
    splits a tiny packed-regime shape)."""
    _validate_bwd_dq_shapes(q, k, v, d_o, lse, d_arr)
    b, hq, n, d = q.shape
    hkv = k.shape[1]
    packed = _validate_segments(seg_id, seg_start, b=b, n=n, causal=causal)
    if _force_ranges is not None:
        ranges = _force_ranges
    elif rate_macs_per_s is None:
        ranges = [(0, n)]
    else:
        ranges = plan_dq_dispatches(
            n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate_macs_per_s, causal=causal,
        )

    kernel = _bwd_dq_kernel(d, causal, _flip_causal, variant, d_slab, packed)
    scale_in = mx.array([scale], dtype=mx.float32)
    dq_chunks: list[mx.array] = []
    for r0, r1 in ranges:
        dq_chunks.append(
            _dispatch_bwd_dq_range(
                kernel, q, k, v, d_o, lse, d_arr, scale_in, r0=r0, r1=r1, variant=variant,
                seg_id=seg_id, seg_start=seg_start,
            )
        )
    if len(dq_chunks) == 1:
        return dq_chunks[0]
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
# dispatches` plans the ascending contiguous query-range split from the calibrated backward
# rate (exact-causal costs at the dK/dV 4*D pair cost, per-command-buffer budget) and refuses
# (`LaunchBudgetError`) when the chain cannot be planned -- see `plan_budgeted_ranges`.
# ---------------------------------------------------------------------------------------

_BWD_DKV_THREADGROUP = 32   # one thread per key; 32 (SIMD width) groups them for occupancy


def _bwd_dkv_macs_per_row(*, n: int, d: int, b: int, hq: int) -> int:
    """Conservative per-query-row MAC upper bound for dK/dV: each of the n keys costs an
    s = q.k dot (D), a dp = dO.v dot (D), a dV accumulate (D) and a dK accumulate (D) == 4*D
    MACs, across every (batch, q-head). Over-counts causal (a query near the top touches fewer
    keys), the safe direction for a launch-budget guard -- over-estimating cost splits MORE,
    never under-budgets. The 4*D (heavier) of the two backward kernels, so a rate calibrated on
    dK/dV also conservatively sizes the dQ split when the two share one throughput number."""
    return 4 * d * n * b * hq


def plan_dkv_dispatches(
    *, n: int, d: int, b: int, hq: int, hkv: int, rate: float, causal: bool = True,
    block_align: int = 1,
) -> list[tuple[int, int]]:
    """Ascending contiguous query-range split `[(q_lo, q_hi), ...]` tiling `[0, n)` exactly for
    the chained dK/dV backward, planned by `plan_budgeted_ranges` at the dK/dV `4*D` pair cost
    with exact-causal accounting. Pure integer arithmetic -- no GPU, no allocation. Raises
    `LaunchBudgetError` (the 0.1.0 refusal contract) when the chain cannot be planned within
    the per-command-buffer budget, never returns an over-budget range for the uncatchable GPU
    watchdog to hit.

    Live inputs (q/k/v/dO/L/D plus the chained dk_in/dv_in partials) are what mlx counts once
    per command buffer; the chain's INTERMEDIATE outputs are deliberately NOT credited (freed
    mid-chain, allocator-recyclable -- crediting them could model a commit reality skips). At
    flagship shapes the live inputs alone exceed the commit threshold, so every dispatch owns
    its buffer and the chain's total is unbounded; at small-footprint shapes the whole chain
    must fit one buffer budget (see `plan_budgeted_ranges`).

    `block_align` (default 1 == the scalar body's per-key owner) rounds range widths DOWN to
    a multiple of `block_align`: the mma body (the launcher passes 32) owns a 32-KEY block per
    simdgroup and loops the query range in 32-row query blocks, so a mid-block range split
    would merge different partial products inside one hardware MMA and break chained==single
    bit-identity -- 32-alignment restores the scalar accumulation-order argument at block
    granularity. The minimal unit becomes one query block; a rate where even one block cannot
    fit its modeled buffer still raises."""
    return plan_budgeted_ranges(
        n=n, pair_cost=4 * d * b * hq, rate=rate, causal=causal,
        live_input_elems=(
            2 * b * hq * n * d          # q, dO
            + 2 * b * hkv * n * d       # k, v
            + 2 * b * hq * n            # lse, D
            + 2 * b * hkv * n * d       # dk_in, dv_in (fp32 -- data_size counts ELEMENTS)
        ),
        output_elems_per_range=None,
        block_align=block_align,
    )


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


def _dkv_kernel_name(
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None,
    packed: bool, segment_bound: bool, break_early: bool,
) -> str:
    """Pure `mx.fast.metal_kernel(name=...)` string builder for the dK/dV kernel
    (checkpoint-A fix, 0.5.0): extracted out of `_bwd_dkv_kernel` so the name<->source
    correspondence is directly testable in the DEFAULT lane without dispatching a
    kernel. Byte-identical to the inline expression it replaces. mlx caches compiled
    kernels BY NAME -- a call with an unchanged name but different source silently
    returns the FIRST compiled binary -- so every flag that varies `build_bwd_dkv_mma_source`
    output (`segment_bound`, `break_early`, mma variant only) must also vary this name."""
    return (
        f"mtp_flash_bwd_dkv_{variant}_d{head_dim}_"
        f"{'c' if causal else 'f'}{'x' if flip_causal else ''}"
        + (f"_s{d_slab}" if variant == "mma" else "")
        + ("_p" if packed else "")
        + (("" if segment_bound else "_nb") if variant == "mma" else "")
        + (("_be" if break_early else "") if variant == "mma" else "")
    )


@functools.cache
def _bwd_dkv_kernel(
    head_dim: int, causal: bool, flip_causal: bool, variant: str, d_slab: int | None,
    packed: bool = False, segment_bound: bool = True, break_early: bool = False,
) -> _MetalKernel:
    """Build (and cache) the dK/dV kernel for a given (head_dim, causal, flip_causal, variant,
    d_slab, packed, segment_bound, break_early). `variant="scalar"` uses the v1 one-thread-per-key
    body (`d_slab` has no effect on its source, but stays part of the cache key regardless -- a
    harmless redundant entry, never a correctness issue, if a caller ever varies it for scalar);
    `"mma"` uses the T9b rung-B2 key-major register-resident D-slabbed body, whose source genuinely
    changes with `d_slab` (`d_slab=None` builds with the source builder's own default
    `_BWD_DKV_MMA_D_SLAB`). Both variants share the same
    (q,k,v,dO,lse,d_arr,dk_in,dv_in,qoffs,scale_in)->(dk_out,dv_out) chained contract, so
    `_dispatch_bwd_dkv_range` swaps only the grid/threadgroup shape between them. `flip_causal` is
    TEST-ONLY (see `build_bwd_dkv_source` / `build_bwd_dkv_mma_source`); it stays part of the
    cache key so a perturbed/correct kernel at the same (head_dim, causal, variant) never collide.

    `packed=True` (0.4.0) builds the block-diagonal-segment variant (see the source builders'
    packed docstrings): the keep predicate gains a same-segment term and the launcher binds two
    extra int32 buffers (`seg_id`, `seg_start`) appended to `input_names` LAST so the non-packed
    order is untouched (mlx binds positionally). `packed` is the LAST cache-key component and
    defaults False, so every pre-0.4.0 caller keeps its existing call and cache entry unchanged.

    `segment_bound`/`break_early` (0.5.0, spec D1/D5) add the packed MMA query-block segment-end
    bound; both are passed to `build_bwd_dkv_mma_source` in the mma arm ONLY -- the scalar builder
    does not accept them and stays the assumption-free oracle (D3). Both flags are threaded into
    BOTH the `functools.cache` key (this signature) AND the `mx.fast.metal_kernel` name below,
    never just one: mlx caches compiled kernels BY NAME, and a call with the same name but
    different source silently returns the FIRST compiled binary -- so a flag that changed the
    source but not the name (or vice versa) would risk a stale/wrong kernel on a later call with a
    different flag value. Defaults (`segment_bound=True, break_early=False`) preserve every
    pre-0.5.0 call and cache entry. Raises `ValueError` for `variant="scalar"` combined
    with `segment_bound=False` or `break_early=True` -- the scalar builder silently
    ignores both, so a caller that thinks it is toggling scalar behaviour with either
    flag is misusing the API rather than getting a no-op."""
    if variant == "scalar" and (segment_bound is False or break_early):
        raise ValueError("segment_bound/break_early apply to the mma variant only")
    if variant == "mma":
        source = build_bwd_dkv_mma_source(
            head_dim, causal=causal, flip_causal=flip_causal, d_slab=d_slab, packed=packed,
            segment_bound=segment_bound, break_early=break_early,
        )
    elif variant == "scalar":
        source = build_bwd_dkv_source(
            head_dim, causal=causal, flip_causal=flip_causal, packed=packed,
        )
    else:
        raise ValueError(f"unknown dK/dV kernel variant {variant!r}")
    input_names = [
        "q", "k", "v", "d_o", "lse", "d_arr", "dk_in", "dv_in", "qoffs", "scale_in",
    ]
    if packed:
        input_names += ["seg_id", "seg_start"]
    kernel = mx.fast.metal_kernel(
        name=_dkv_kernel_name(
            head_dim, causal, flip_causal, variant, d_slab, packed, segment_bound, break_early,
        ),
        input_names=input_names,
        output_names=["dk_out", "dv_out"],
        source=source,
    )
    return cast(_MetalKernel, kernel)


def _dispatch_bwd_dkv_range(
    kernel: _MetalKernel, q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, dk_in: mx.array, dv_in: mx.array, scale_in: mx.array,
    *, q_lo: int, q_hi: int, variant: str = "scalar",
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """One dK/dV dispatch accumulating query rows [q_lo, q_hi) into the chained fp32 partials.
    Full q/k/v/dO/L/D + `dk_in`/`dv_in` buffers + an in-kernel `qoffs` range offset (never a
    Python-side slice); returns the fresh fp32 `dk_out`/`dv_out`.

    The two variants differ ONLY in the launch shape (same output shapes/dtypes + full chained
    contract, so the accumulator threading in `launch_bwd_dkv` is variant-agnostic): `"scalar"`
    runs one thread per key (grid.x == n; ALL keys are owners, a key with no causally-allowed query
    in the range copies its `dk_in`->`dk_out` slot unchanged), while `"mma"` runs one 32-lane
    simdgroup per 32-KEY block (grid.x == ceil(n/32)*32, threadgroup.x == 32; each simdgroup seeds
    its key block from `dk_in`/`dv_in` and stores it, carrying the chained accumulator forward).

    `seg_id`/`seg_start` (both-or-neither) are the packed-attention buffers; when present they are
    appended to `inputs` LAST, in `input_names` order (the `_bwd_dkv_kernel` packed contract), and
    passed as the FULL (B, N) buffers -- the kernel offsets in with `seg_off = b * n`, and every
    chained dispatch receives the SAME full buffers (never sliced)."""
    b, _hq, n, d = q.shape
    hkv = k.shape[1]
    qoffs = mx.array([q_lo, q_hi], dtype=mx.uint32)
    if variant == "mma":
        num_key_blocks = (n + 31) // 32              # one 32-key block per simdgroup
        grid = (num_key_blocks * 32, b * hkv, 1)
        threadgroup = (32, 1, 1)
    else:
        grid = (n, b * hkv, 1)
        threadgroup = (min(_BWD_DKV_THREADGROUP, n), 1, 1)
    inputs = [q, k, v, d_o, lse, d_arr, dk_in, dv_in, qoffs, scale_in]
    if seg_id is not None:
        inputs += [seg_id, cast(mx.array, seg_start)]
    dk_out, dv_out = kernel(
        inputs=inputs,
        template=[("T", q.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(b, hkv, n, d), (b, hkv, n, d)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return dk_out, dv_out


def launch_bwd_dkv(
    q: mx.array, k: mx.array, v: mx.array, d_o: mx.array,
    lse: mx.array, d_arr: mx.array, *,
    scale: float, causal: bool, rate_macs_per_s: float | None = None,
    variant: str = "scalar", d_slab: int | None = None,
    seg_id: mx.array | None = None, seg_start: mx.array | None = None,
    segment_bound: bool = True, break_early: bool = False,
    _flip_causal: bool = False,
    _force_ranges: list[tuple[int, int]] | None = None,
) -> tuple[mx.array, mx.array]:
    """dK/dV backward -> (dK, dV), with k/v's shape/dtype. Consumes the forward's saved L
    (`lse`, fp32 (B, Hq, N)) and T7's D (`d_arr`, fp32 (B, Hq, N)); recomputes S/P from q/k and
    accumulates `dV_j += P*dO`, `dK_j += scale*P*(dP - D)*q` in CHAINED fp32 partials over the
    causally-allowed queries, grouped over each kv head's contiguous q-head group.

    `variant` picks the kernel body: `"scalar"` (default -- the v1 one-thread-per-key body,
    unchanged behaviour for every existing caller) or `"mma"` (the T9b rung-B2 key-major 4x4
    simdgroup-matrix body with register-resident D-slabbed accumulators). `d_slab` overrides the
    mma body's slab width (`None` = the source builder's own default; ignored by the scalar body).
    The mma variant is a CORRECTNESS + small-shape rung: it is not wired into api.py or the
    calibrated-rate path (graduation + the saturation d_slab sweep are a later rung).

    `rate_macs_per_s`: when None, a single dispatch over all query rows with no budget check
    (safe only at small N -- the tiny-shape/test path); when given, the launcher plans the
    ascending query-range split via `plan_dkv_dispatches` (exact-causal costs, per-command-
    buffer budget) and refuses (`LaunchBudgetError`) when the chain cannot be planned. The
    fp32 accumulators are chained across dispatches (seeded from the prior's output), so a
    range split is bit-identical to a single dispatch, and cast to k/v dtype exactly once
    after the last dispatch. The mma variant plans with a 32-row query-BLOCK alignment
    (`plan_dkv_dispatches(block_align=32)`) so a split never bisects a query block (a
    mid-block split would merge different partial products inside one hardware MMA and break
    the chained bit-identity).

    `seg_id` / `seg_start` (0.4.0, both-or-neither, int32 (B, N), require `causal=True`) switch
    on PACKED block-diagonal-causal attention: query `i` contributes to key `key`'s dK/dV only
    when both are in the same segment AND causal. The two buffers are bound LAST (after
    q/.../scale_in), and every chained dispatch receives the SAME full (B, N) buffers with an
    in-kernel offset (never a Python-side slice). The predicate zeros cross-segment contributions
    identically regardless of the range split, so a chained split stays bit-identical to a single
    dispatch under packing too.

    `segment_bound`/`break_early` (0.5.0, spec D1/D5, mma variant only): thread straight through
    to `_bwd_dkv_kernel`, which passes them to `build_bwd_dkv_mma_source` and folds both into the
    kernel's cache key AND its `mx.fast.metal_kernel` name -- mlx caches compiled kernels BY NAME
    (verified: same name + different source returns the FIRST compiled binary), so a
    source-varying flag that lived in only one of the two would risk a later call silently
    getting back a stale, wrong-flag kernel. Defaults (`segment_bound=True, break_early=False`)
    preserve every pre-0.5.0 call.

    `_flip_causal` is TEST-ONLY (wrong-triangle causal-skip perturbation -- see source.py);
    `_force_ranges` is TEST-ONLY too (the split-forcing seam -- the production planner never
    splits a tiny packed-regime shape; forced ranges for the mma variant must stay 32-aligned
    per the block contract above)."""
    _validate_bwd_dkv_shapes(q, k, v, d_o, lse, d_arr)
    b, hq, n, d = q.shape
    hkv = k.shape[1]
    packed = _validate_segments(seg_id, seg_start, b=b, n=n, causal=causal)
    if _force_ranges is not None:
        ranges = _force_ranges
    elif rate_macs_per_s is None:
        ranges = [(0, n)]
    else:
        block_align = 32 if variant == "mma" else 1
        ranges = plan_dkv_dispatches(
            n=n, d=d, b=b, hq=hq, hkv=hkv, rate=rate_macs_per_s, causal=causal,
            block_align=block_align,
        )

    kernel = _bwd_dkv_kernel(
        d, causal, _flip_causal, variant, d_slab, packed,
        segment_bound=segment_bound, break_early=break_early,
    )
    scale_in = mx.array([scale], dtype=mx.float32)
    dk = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    dv = mx.zeros((b, hkv, n, d), dtype=mx.float32)
    for q_lo, q_hi in ranges:
        dk, dv = _dispatch_bwd_dkv_range(
            kernel, q, k, v, d_o, lse, d_arr, dk, dv, scale_in, q_lo=q_lo, q_hi=q_hi,
            variant=variant, seg_id=seg_id, seg_start=seg_start,
        )
    return dk.astype(k.dtype), dv.astype(v.dtype)


# ---------------------------------------------------------------------------------------
# T9b Step 3 (graduation): PER-KERNEL calibrated backward rates. The Step-1 checkpoint
# measured the two backward kernels' scalar throughputs 2.35x apart (dQ streams k/v per row
# with no reuse; dK/dV reuses k/v registers across a GQA group's 4 q-heads), which measurably
# FAILS the shared-rate safety condition throughput_dq >= 0.5*throughput_dkv -- so the earlier
# single shared backward rate (which sized BOTH splits from the dK/dV kernel) is retired for
# TWO probe-what-you-rate rates: each split is sized by its OWN kernel's measured throughput,
# with no cross-kernel assumption at all. Each reuses the forward's ramp/canary machinery
# (`_calibrate_fwd`) via its own per-row MAC cost model, builds and probes the SAME
# (variant, d_slab) kernel the launcher will dispatch (the mma winner at the measured bucket),
# and caches on (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab) -- the same key
# shape as `_FWD_RATE_CACHE`. Same discipline as `calibrated_fwd_rate`: probe QKV/dO from a
# LOCAL `mx.random.key(0)` (never `mx.random.seed`), never call inside a compiled region.
# ---------------------------------------------------------------------------------------

# key: (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab, packed) -- variant/d_slab/
# packed-aware, one cache per backward kernel (dQ / dK/dV are separate measurements at separate
# rates); the trailing `packed` bool (0.4.0) keeps causal and packed rates from ever colliding.
_BWD_DQ_RATE_CACHE: dict[
    tuple[int, str, bool, int, int, int, str, int | None, bool], float
] = {}
_BWD_DKV_RATE_CACHE: dict[
    tuple[int, str, bool, int, int, int, str, int | None, bool], float
] = {}


def calibrated_bwd_dq_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool,
    tile: TileShape, packed: bool = False,
) -> float:
    """Cached, safety-factored, N-aware MAC/s throughput for the dQ backward kernel `tile`
    actually names, used to size the dQ query-row split. Ramps the probe key-count toward the
    caller's real `n` via `_calibrate_fwd` at the dQ `3*D` per-row cost (`_bwd_dq_macs_per_row`),
    and PROBES WHAT IT RATES: `measure()` builds and dispatches the SAME (`tile.variant`,
    `tile.d_slab`, `packed`) dQ kernel the launcher runs, so an mma call never reads a
    scalar-measured rate and a packed call never reads a causal-measured one. Cached per
    (head_dim, dtype, causal, b, hq, n-bucket, variant, d_slab, packed). Must never be called
    inside a compiled region (host-sync timing).

    `packed=True` (0.4.0) rates the PACKED dQ kernel via a synthetic single-segment layout
    (`seg_id`/`seg_start` all zeros) -- worst-case packed work (full causal triangle + predicate),
    sizing the split conservatively; see `calibrated_fwd_rate` for the full rationale."""
    key = (
        head_dim, str(dtype), causal, b, hq, _n_bucket(n), tile.variant, tile.d_slab, packed
    )
    if key in _BWD_DQ_RATE_CACHE:
        return _BWD_DQ_RATE_CACHE[key]

    key_q, key_k, key_v, key_do = mx.random.split(mx.random.key(0), 4)
    scale = 1.0 / (head_dim ** 0.5)
    kernel = _bwd_dq_kernel(head_dim, causal, False, tile.variant, tile.d_slab, packed)
    scale_in = mx.array([scale], dtype=mx.float32)
    probes: dict[tuple[int, int], tuple[mx.array, ...]] = {}
    seg_for = _packed_probe_segs(b, packed)

    def measure(rows: int, keys: int) -> float:
        # Times one dQ dispatch of query rows [keys-rows, keys) against a full `keys`-key working
        # set -- the production tail range (high query indices scan the most keys under causal),
        # mirroring calibrated_fwd_rate.measure. lse/D are zeros (this is a TIMING probe -- the
        # kernel does identical FLOPs regardless of the residual values).
        seg_id, seg_start = seg_for(keys)
        if (rows, keys) not in probes:
            qp = mx.random.normal((b, hq, keys, head_dim), key=key_q).astype(dtype)
            kp = mx.random.normal((b, hkv, keys, head_dim), key=key_k).astype(dtype)
            vp = mx.random.normal((b, hkv, keys, head_dim), key=key_v).astype(dtype)
            dop = mx.random.normal((b, hq, keys, head_dim), key=key_do).astype(dtype)
            lsep = mx.zeros((b, hq, keys), dtype=mx.float32)
            dp = mx.zeros((b, hq, keys), dtype=mx.float32)
            mx.eval(qp, kp, vp, dop, lsep, dp)
            probes[(rows, keys)] = (qp, kp, vp, dop, lsep, dp)
            # Metal JIT + cold clocks/caches: this first dispatch is deliberately unmeasured.
            wdq = _dispatch_bwd_dq_range(
                kernel, qp, kp, vp, dop, lsep, dp, scale_in,
                r0=keys - rows, r1=keys, variant=tile.variant,
                seg_id=seg_id, seg_start=seg_start,
            )
            mx.eval(wdq)
        qp, kp, vp, dop, lsep, dp = probes[(rows, keys)]
        t0 = time.perf_counter()
        rdq = _dispatch_bwd_dq_range(
            kernel, qp, kp, vp, dop, lsep, dp, scale_in,
            r0=keys - rows, r1=keys, variant=tile.variant,
            seg_id=seg_id, seg_start=seg_start,
        )
        mx.eval(rdq)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n,
        macs_per_row=_bwd_dq_macs_per_row, causal=causal,
    )
    rate = SAFETY_FACTOR * raw_rate
    _BWD_DQ_RATE_CACHE[key] = rate
    return rate


def calibrated_bwd_dkv_rate(
    *, head_dim: int, dtype: mx.Dtype, b: int, hq: int, hkv: int, n: int, causal: bool,
    tile: TileShape, packed: bool = False,
) -> float:
    """Cached, safety-factored, N-aware MAC/s throughput for the chained dK/dV backward kernel
    `tile` actually names, used to size the dK/dV query-range split. Ramps toward the caller's
    real `n` via `_calibrate_fwd` at the dK/dV `4*D` per-row cost (`_bwd_dkv_macs_per_row`), and
    PROBES WHAT IT RATES: `measure()` builds and dispatches the SAME (`tile.variant`,
    `tile.d_slab`, `packed`) dK/dV kernel the launcher runs. Cached per (head_dim, dtype, causal,
    b, hq, n-bucket, variant, d_slab, packed). Must never be called inside a compiled region
    (host-sync timing).

    `packed=True` (0.4.0) rates the PACKED dK/dV kernel via a synthetic single-segment layout
    (`seg_id`/`seg_start` all zeros) -- worst-case packed work (dK/dV scans the full causal query
    range regardless of packing, §6 YAGNI), sizing the split conservatively; see
    `calibrated_fwd_rate` for the full rationale."""
    key = (
        head_dim, str(dtype), causal, b, hq, _n_bucket(n), tile.variant, tile.d_slab, packed
    )
    if key in _BWD_DKV_RATE_CACHE:
        return _BWD_DKV_RATE_CACHE[key]

    key_q, key_k, key_v, key_do = mx.random.split(mx.random.key(0), 4)
    scale = 1.0 / (head_dim ** 0.5)
    kernel = _bwd_dkv_kernel(head_dim, causal, False, tile.variant, tile.d_slab, packed)
    scale_in = mx.array([scale], dtype=mx.float32)
    probes: dict[tuple[int, int], tuple[mx.array, ...]] = {}
    seg_for = _packed_probe_segs(b, packed)

    def measure(rows: int, keys: int) -> float:
        # Times one dK/dV dispatch of query rows [keys-rows, keys) against a full `keys`-key
        # working set -- the production tail range, mirroring calibrated_fwd_rate.measure. lse/D
        # are zeros (a TIMING probe: the kernel does identical FLOPs regardless of the residual
        # values); dk_in/dv_in seed the chained accumulator with zeros (single-dispatch probe).
        seg_id, seg_start = seg_for(keys)
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
            # Metal JIT + cold clocks/caches: this first dispatch is deliberately unmeasured.
            wdk, wdv = _dispatch_bwd_dkv_range(
                kernel, qp, kp, vp, dop, lsep, dp, dk0, dv0, scale_in,
                q_lo=keys - rows, q_hi=keys, variant=tile.variant,
                seg_id=seg_id, seg_start=seg_start,
            )
            mx.eval(wdk, wdv)
        qp, kp, vp, dop, lsep, dp, dk0, dv0 = probes[(rows, keys)]
        t0 = time.perf_counter()
        rdk, rdv = _dispatch_bwd_dkv_range(
            kernel, qp, kp, vp, dop, lsep, dp, dk0, dv0, scale_in,
            q_lo=keys - rows, q_hi=keys, variant=tile.variant,
            seg_id=seg_id, seg_start=seg_start,
        )
        mx.eval(rdk, rdv)
        return time.perf_counter() - t0

    start_n = _start_probe_n(n)
    raw_rate = _calibrate_fwd(
        measure=measure, n=n, d=head_dim, b=b, hq=hq, start_n=start_n,
        macs_per_row=_bwd_dkv_macs_per_row, causal=causal,
    )
    rate = SAFETY_FACTOR * raw_rate
    _BWD_DKV_RATE_CACHE[key] = rate
    return rate
