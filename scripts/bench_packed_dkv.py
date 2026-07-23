"""Op-level bench for the packed dK/dV segment-end bound (new in 0.5.0):
bounded (`segment_bound=True`) vs unbounded (`segment_bound=False`) at the SAME shape and
the SAME dispatch plan, isolating the block-skip's own wall-time effect from every other
variable.

ONE process per (layout, n) runs BOTH arms with their timed reps INTERLEAVED (a-b-a-b...)
so thermal/session drift (repo gotcha 7: cross-session absolute rates ~12% soft) cancels
out of the ratio rather than biasing whichever arm runs first/last -- op-level memory at
these shapes is modest (bf16, n<=8192, d=128), so subprocess-per-arm buys nothing here.

Both arms dispatch through IDENTICAL 32-aligned query ranges, forced via `launch_bwd_dkv`'s
`_force_ranges` seam (`forced_ranges`, below) -- FIXED, deterministic 1024-row chunks, never
a `plan_dkv_dispatches` call against a measured/calibrated rate: at this shape (hq=8, hkv=2,
d=128) the live dK/dV input-buffer element count sits under mlx 0.32.0's 51M-element command-
buffer commit threshold (`launch.py`'s `_PACK_COMMIT_ELEMS`), so `plan_dkv_dispatches` at ANY
plausible rate either returns a single [0, n) dispatch or refuses outright (verified: sweeping
150-928.97 G MAC/s, the conservative SAFETY_FACTOR-halved measured slab128 rate, at hq=8/hkv=2/
d=128 never produced more than one range at n=4096 or n=8192) -- there is no rate that yields a
genuinely multi-dispatch chain here. A single [0, n) causal dispatch is also exactly the shape
gotcha 1 warns could be killed by the macOS command-buffer-interactivity watchdog on a bad day,
so `forced_ranges` sidesteps the question entirely with FIXED 32-aligned chunks (never a rate
calibration probe -- no `calibrated_bwd_dkv_rate` call anywhere in this script), guaranteeing
BOTH arms run identically as a short chained sequence of small dispatches regardless of what
any rate model would have picked.

Correctness discipline mirrors the Checkpoint-A verified fact in
`tests/test_attention_kernel_bwd.py::test_bounded_dkv_bit_identical_to_unbounded`: on a cold
process, co-evaluating two structurally different freshly-JIT'd `mx.fast.metal_kernel`s in a
SINGLE `mx.eval` call corrupted both outputs (order-dependent). Each arm is therefore warmed
FULLY (its own `mx.eval`) before either arm's first TIMED rep, and every timed dispatch --
bounded or unbounded -- gets its own separate `mx.eval` call; only the *sequence* of reps is
interleaved, never a single eval spanning both arms.

Shapes: b=1, hq=8, hkv=2, head_dim=128, bf16 -- head_dim=128 is in `_KERNEL_HEAD_DIMS` and is
the ONLY head_dim either backward MMA ladder ever measured (`DKV_MEASURED[128][8192]`,
`attention/kernel/dispatch.py`), so `select_bwd_tiles` resolves a real (non-guessed)
`(variant="mma", d_slab=128)` tile for this bench rather than hardcoding one.

This script runs ONE (layout, n) per invocation and exits -- the orchestration loop (looping
layouts x n, pre-flight, ETA) is the caller's (main-session heavy-run) job, never this
script's. `--layout alpaca` reproduces the 0.4.0 packed-training bench's measured 4096-token
segment-length histogram (a decreasing-length tail typical of instruction-tuning data),
scaled in COUNT (not length) for n=8192, plus a trailing remainder segment so lengths always
sum to n exactly.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_train_perf.attention.kernel.dispatch import select_bwd_tiles
from mlx_train_perf.attention.kernel.launch import launch_bwd_dkv
from mlx_train_perf.bench.artifacts import run_identity
from mlx_train_perf.core.guards import install_guardrails

_SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUT = _SCRIPT_PATH.parent.parent / "_artifacts" / "packed_dkv"

_LAYOUTS = ("alpaca", "two_seg", "single")
_ALPACA_VALID_N = (4096, 8192)
# (segment_length, count) at n=4096 -- the 0.4.0 packed-training bench's measured
# histogram; `alpaca_lengths` scales COUNTS (not lengths) by n // 4096.
_ALPACA_BASE: tuple[tuple[int, int], ...] = ((256, 4), (128, 8), (64, 16), (32, 24))

# 32-aligned (the mma dK/dV block contract); both n=4096 and n=8192 are exact multiples,
# so `forced_ranges` never needs a shorter tail chunk for either supported --n.
_DISPATCH_CHUNK = 1024

B, HQ, HKV, HEAD_DIM = 1, 8, 2, 128
DTYPE = mx.bfloat16
SEED = 42


def alpaca_lengths(n: int) -> list[int]:
    """Segment-length histogram for `--layout alpaca`: the 0.4.0 packed-training bench's
    measured 4096-token histogram, with every count doubled at n=8192 (the same relative
    shape, twice the row budget), plus one trailing segment absorbing whatever length is
    left so the lengths sum to `n` EXACTLY. Defined only for n in `_ALPACA_VALID_N`."""
    if n not in _ALPACA_VALID_N:
        raise ValueError(f"alpaca layout is only defined for n in {_ALPACA_VALID_N}, got {n}")
    scale = n // 4096
    lens: list[int] = []
    for length, count in _ALPACA_BASE:
        lens.extend([length] * (count * scale))
    remainder = n - sum(lens)
    if remainder <= 0:
        raise ValueError(f"alpaca base histogram already reaches/exceeds n={n}")
    lens.append(remainder)
    return lens


def two_seg_lengths(n: int) -> list[int]:
    """`--layout two_seg`: one boundary at the midpoint."""
    return [n // 2, n - n // 2]


def single_lengths(n: int) -> list[int]:
    """`--layout single`: no packing at all (one segment covering every row) -- the
    no-win control (a bounded/unbounded ratio near 1.0x here is expected, not a bug)."""
    return [n]


def layout_lengths(name: str, n: int) -> list[int]:
    """Dispatch to the named layout's segment-length list. Raises `ValueError` on an
    unrecognized layout name (argparse's own `choices=` already rejects this at the CLI
    boundary; this function stays independently defensive for direct callers)."""
    if name == "alpaca":
        return alpaca_lengths(n)
    if name == "two_seg":
        return two_seg_lengths(n)
    if name == "single":
        return single_lengths(n)
    raise ValueError(f"unknown layout {name!r}")


def layout_to_segments(lens: list[int], *, b: int = B) -> tuple[mx.array, mx.array]:
    """(B, N) int32 `seg_id`/`seg_start` buffers for a fixed segment-length list, shared
    across every batch row -- same construction as
    `tests/test_attention_kernel_bwd.py::_packed_layout`."""
    seg_id_row: list[int] = []
    seg_start_row: list[int] = []
    start = 0
    for sid, ln in enumerate(lens):
        seg_id_row += [sid] * ln
        seg_start_row += [start] * ln
        start += ln
    seg_id = mx.array([seg_id_row] * b, dtype=mx.int32)
    seg_start = mx.array([seg_start_row] * b, dtype=mx.int32)
    return seg_id, seg_start


def forced_ranges(n: int, *, chunk: int = _DISPATCH_CHUNK) -> list[tuple[int, int]]:
    """Ascending, 32-aligned query ranges tiling `[0, n)` into fixed `chunk`-sized pieces
    (a shorter final tail only if `n` is not an exact multiple of `chunk`) -- pure,
    deterministic arithmetic, never a rate-calibration probe. Both arms dispatch through
    the SAME ranges via `launch_bwd_dkv`'s `_force_ranges` seam, so any wall-time
    difference between arms reflects only the segment-bound break, never a differing
    dispatch plan -- and splitting into multiple dispatches (rather than one [0, n) call)
    keeps a full n=8192 causal dispatch away from the single-command-buffer GPU-watchdog
    risk gotcha 1 documents."""
    if chunk % 32 != 0:
        raise ValueError(f"chunk must be 32-aligned, got {chunk}")
    ranges: list[tuple[int, int]] = []
    r0 = 0
    while r0 < n:
        r1 = min(r0 + chunk, n)
        ranges.append((r0, r1))
        r0 = r1
    return ranges


def artifact_path(out_dir: Path, *, n: int, layout: str) -> Path:
    """`dkv_n{n}_{layout}.json` under `out_dir` -- layout and n both ride the filename
    (repo gotcha 18: an attention-arm-shaped filename must never let two conditions
    collide in one `--out` directory)."""
    return out_dir / f"dkv_n{n}_{layout}.json"


def build_result(
    *, layout: str, n: int, ranges: list[tuple[int, int]],
    bounded_ms: list[float], unbounded_ms: list[float],
    peak_gb: float, code_sha: str,
) -> dict[str, object]:
    """Pure artifact-shape builder (no mx/GPU dependency): both arms' medians + raw reps,
    the ratio, the forced ranges, code_sha, and the run's peak memory since the pre-loop
    reset (the bench inputs stay resident through the window, so this is a total peak,
    not a marginal one)."""
    bounded_median = statistics.median(bounded_ms)
    unbounded_median = statistics.median(unbounded_ms)
    ratio = unbounded_median / bounded_median if bounded_median > 0 else None
    return {
        "layout": layout,
        "n": n,
        "arms": {
            "bounded": {
                "median_ms": round(bounded_median, 4),
                "reps_ms": [round(x, 4) for x in bounded_ms],
            },
            "unbounded": {
                "median_ms": round(unbounded_median, 4),
                "reps_ms": [round(x, 4) for x in unbounded_ms],
            },
        },
        # unbounded / bounded: >1.0 means the segment-end bound made dK/dV faster.
        "ratio": round(ratio, 4) if ratio is not None else None,
        "forced_ranges": [list(r) for r in ranges],
        "code_sha": code_sha,
        "peak_gb": round(peak_gb, 4),
    }


def write_artifact(path: Path, result: dict[str, object]) -> None:
    """Atomic write (tmp + rename) -- an interrupted run leaves either the PRIOR artifact
    or nothing at `path`, never a half-written JSON (same idiom as
    `bench.artifacts.write_result`, kept separate here since this schema is not the
    identity/status shape that helper writes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.rename(path)


def run_bench(*, layout: str, n: int, reps: int) -> dict[str, object]:
    """Build the shared q/k/v/dO/lse/D residuals and the forced dispatch plan once, then
    time the bounded and unbounded arms with reps INTERLEAVED (a-b-a-b...). `lse`/`D` are
    zeros: this is a pure TIMING probe, and the kernel's FLOPs are identical regardless of
    residual values (the same convention `calibrated_bwd_dkv_rate`'s own probe uses)."""
    lens = layout_lengths(layout, n)
    seg_id, seg_start = layout_to_segments(lens)
    ranges = forced_ranges(n)
    _, dkv_tile = select_bwd_tiles(n, HEAD_DIM)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    mx.random.seed(SEED)
    q = mx.random.normal((B, HQ, n, HEAD_DIM)).astype(DTYPE)
    k = mx.random.normal((B, HKV, n, HEAD_DIM)).astype(DTYPE)
    v = mx.random.normal((B, HKV, n, HEAD_DIM)).astype(DTYPE)
    d_o = mx.random.normal((B, HQ, n, HEAD_DIM)).astype(DTYPE)
    lse = mx.zeros((B, HQ, n), dtype=mx.float32)
    d_arr = mx.zeros((B, HQ, n), dtype=mx.float32)
    mx.eval(q, k, v, d_o, lse, d_arr, seg_id, seg_start)

    def dispatch(*, segment_bound: bool) -> tuple[mx.array, mx.array]:
        return launch_bwd_dkv(
            q, k, v, d_o, lse, d_arr, scale=scale, causal=True,
            variant=dkv_tile.variant, d_slab=dkv_tile.d_slab,
            seg_id=seg_id, seg_start=seg_start,
            segment_bound=segment_bound,
            _force_ranges=ranges,
        )

    # Warm each arm FULLY, arm-by-arm, before any timed rep (Checkpoint-A verified fact:
    # co-evaluating two freshly-JIT'd kernels in one `mx.eval` corrupted both outputs on a
    # cold process). Each warmup pays that arm's Metal JIT compile.
    mx.eval(*dispatch(segment_bound=True))
    mx.eval(*dispatch(segment_bound=False))

    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    bounded_ms: list[float] = []
    unbounded_ms: list[float] = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*dispatch(segment_bound=True))
        mx.synchronize()
        bounded_ms.append((time.perf_counter() - t0) * 1000.0)

        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*dispatch(segment_bound=False))
        mx.synchronize()
        unbounded_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_gb = mx.get_peak_memory() / 1024**3
    code_sha = str(run_identity()["code_sha"])
    return build_result(
        layout=layout, n=n, ranges=ranges, bounded_ms=bounded_ms,
        unbounded_ms=unbounded_ms, peak_gb=peak_gb, code_sha=code_sha,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True, choices=_LAYOUTS)
    ap.add_argument("--n", type=int, required=True, choices=_ALPACA_VALID_N)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_guardrails()
    result = run_bench(layout=args.layout, n=args.n, reps=args.reps)
    path = artifact_path(args.out, n=args.n, layout=args.layout)
    write_artifact(path, result)
    arms = result["arms"]
    assert isinstance(arms, dict)
    print(
        f"{args.layout} n={args.n}: "
        f"bounded={arms['bounded']['median_ms']}ms "
        f"unbounded={arms['unbounded']['median_ms']}ms "
        f"ratio={result['ratio']} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
