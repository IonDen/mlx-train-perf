"""PackedMask: the packed-sequence segment carrier (spec 2026-07-17, §3.2/§3.3)."""
from dataclasses import dataclass
from typing import cast

import mlx.core as mx


@dataclass(frozen=True)
class PackedMask:
    """Block-diagonal-causal segment description for one packed batch.

    seg_id: int32 (B, L) -- contiguous ascending segment ids per input position,
      starting at 0 each row; the pad tail is the row's last segment (gapless: every
      position in [0, L) belongs to exactly one segment).
    seg_start: int32 (B, L) -- row index where the position's segment begins
      (non-decreasing along the row).
    """
    seg_id: mx.array
    seg_start: mx.array


def segment_allowed(seg_id: mx.array) -> mx.array:
    """(B, 1, N, N) bool: key j visible to query i iff same segment AND j <= i."""
    # cast: mlx's `array.__eq__` stub returns `array | bool` (typeshed convention for
    # __eq__); two mx.array operands always produce an array here.
    same = cast(mx.array, seg_id[:, None, :, None] == seg_id[:, None, None, :])
    n = seg_id.shape[-1]
    causal = mx.tri(n, n, 0, dtype=mx.bool_)[None, None]
    return mx.logical_and(same, causal)
