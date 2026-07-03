"""Tile-shape dispatch by n — encodes the persisted spike artifacts, not prose memory.
(An earlier draft claimed v2c-class wins at n~512; the artifacts refuted it: 210.8, last
in bucket. Caught by the 2026-07-03 design review.)"""
import math
from dataclasses import dataclass

MEASURED: dict[int, tuple[int, float]] = {   # bucket n -> (row_tiles, measured G MAC/s)
    512: (2, 310.7),      # results/bench_v2d_n512.json
    2048: (2, 879.2),     # results/bench_v2d_n2048.json
    8192: (4, 2423.7),    # results/bench_v2e_n8192.json
}


@dataclass(frozen=True, slots=True, kw_only=True)
class VariantChoice:
    row_tiles: int
    provisional: bool
    bucket: int


def select_variant(n: int) -> VariantChoice:
    buckets = sorted(MEASURED)
    best = min(buckets, key=lambda b: (abs(math.log2(n) - math.log2(b)), b))  # tie -> lower bucket
    provisional = not (best <= n < 2 * best)   # inside [bucket, 2*bucket) = same occupancy regime
    return VariantChoice(row_tiles=MEASURED[best][0], provisional=provisional, bucket=best)
