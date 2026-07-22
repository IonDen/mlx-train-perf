"""PackedMask: the packed-sequence segment carrier (spec 2026-07-17, §3.2/§3.3)."""
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlx_train_perf.errors import PackingError


@dataclass(frozen=True)
class PackedMask:
    """Block-diagonal-causal segment description for one packed batch.

    seg_id: int32 (B, L) -- contiguous ascending segment ids per input position,
      starting at 0 each row; the pad tail is the row's last segment (gapless: every
      position in [0, L) belongs to exactly one segment).
    seg_start: int32 (B, L) -- row index where the position's segment begins
      (non-decreasing along the row).

    Both buffers MUST be non-decreasing along each row: the packed kernels' block-skip
    bounds -- the shipped 0.4.0 forward/dQ `kv_lo` skip AND the 0.5.0 dK/dV segment-end
    `break` -- assume contiguous ascending segments, with NO in-kernel guard (spec D2).
    `packed_iterate_batches` (`data/packing.py`) already produces this by construction
    and additionally asserts it at pack time (vectorized numpy, before the `mx.array`
    conversion), so a batch from the real packer is always safe. A HAND-BUILT
    `PackedMask` that skips the packer has no such guard -- call `validate()` on it
    explicitly before use.
    """
    seg_id: mx.array
    seg_start: mx.array

    def validate(self) -> None:
        """Host-side, EXPLICIT opt-in check that `seg_id` and `seg_start` are both
        non-decreasing along each row (spec D2). Never called on the packer/loss path
        (`packed_iterate_batches` already asserts this at pack time, and
        `make_packed_loss_fn`'s forward walk never calls `validate()`) -- this exists
        for a HAND-BUILT `PackedMask` that bypasses the packer, which otherwise has no
        guard at all: a non-monotone hand-built mask silently corrupts both the shipped
        0.4.0 forward/dQ `kv_lo` skip and the 0.5.0 dK/dV segment-end break. Evaluates
        its buffers host-side (a real sync) -- call it only outside a
        compiled/traced step, never on a per-training-step data path.
        """
        import numpy as np  # noqa: PLC0415
        for name in ("seg_id", "seg_start"):
            arr = np.asarray(getattr(self, name))
            if not bool(np.all(np.diff(arr.astype(np.int64), axis=-1) >= 0)):
                raise PackingError(
                    f"PackedMask.{name} must be non-decreasing along each row: the packed "
                    "kernels' block-skip bounds (forward/dQ kv_lo, dK/dV segment-end break) "
                    "assume contiguous ascending segments"
                )


def segment_allowed(seg_id: mx.array) -> mx.array:
    """(B, 1, N, N) bool: key j visible to query i iff same segment AND j <= i."""
    # cast: mlx's `array.__eq__` stub returns `array | bool` (typeshed convention for
    # __eq__); two mx.array operands always produce an array here.
    same = cast(mx.array, seg_id[:, None, :, None] == seg_id[:, None, None, :])
    n = seg_id.shape[-1]
    causal = mx.tri(n, n, 0, dtype=mx.bool_)[None, None]
    return mx.logical_and(same, causal)
