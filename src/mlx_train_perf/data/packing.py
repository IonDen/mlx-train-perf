"""Sequence packing: pure first-fit over a seeded per-epoch shuffle (spec §2.3, §4)."""
import random
from collections.abc import Sequence
from dataclasses import dataclass

from mlx_train_perf.errors import PackingError

_OPEN_POOL = 8  # open packs first-fit scans before closing the oldest


def pack_indices(
    lengths: Sequence[int], pack_len: int, *, seed: int, epoch: int
) -> list[list[int]]:
    """Group dataset indices into packs. A sequence costs `min(len, pack_len) + 1`
    slots (its trailing separator/pad slot, spec §4) against a capacity of
    `pack_len + 1`. Deterministic per (seed, epoch); the shuffle varies across epochs."""
    if pack_len < 1:
        raise PackingError(f"pack_len must be >= 1; got {pack_len}")
    if not lengths:
        raise PackingError("cannot pack an empty dataset")
    if any(n < 1 for n in lengths):
        raise PackingError("zero-length sequence in dataset")
    order = list(range(len(lengths)))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    capacity = pack_len + 1
    packs: list[list[int]] = []
    room: list[int] = []
    open_packs: list[int] = []
    for i in order:
        cost = min(lengths[i], pack_len) + 1
        for j in open_packs:
            if room[j] >= cost:
                packs[j].append(i)
                room[j] -= cost
                if room[j] <= 1:  # can't fit even a 1-token sequence
                    open_packs.remove(j)
                break
        else:
            packs.append([i])
            room.append(capacity - cost)
            open_packs.append(len(packs) - 1)
            if len(open_packs) > _OPEN_POOL:
                open_packs.pop(0)
    return packs


@dataclass(frozen=True, slots=True, kw_only=True)
class PackStats:
    """Utilization breakdown for a packed dataset, over a `pack_len`-token row budget
    (deliberately NOT `pack_len + 1` -- see `pack_stats`'s docstring for why the three
    fractions below sum to `1 + 1/pack_len`, not exactly 1)."""

    real_tokens: int
    capacity_tokens: int
    utilization: float
    separator_fraction: float
    tail_pad_fraction: float


def pack_stats(packs: list[list[int]], lengths: Sequence[int], pack_len: int) -> PackStats:
    """Compute utilization stats over `packs` (as produced by `pack_indices`).

    Accounting identity: each pack's full capacity (`pack_len + 1` slots) splits into
    real content tokens (`min(len, pack_len)` per placed sequence), one trailing
    separator slot per placed sequence, and unused tail padding. Summed over
    `len(packs)` packs: `real_tokens + separators + tail_pad ==
    len(packs) * (pack_len + 1) == capacity_tokens + len(packs)`, where
    `capacity_tokens = len(packs) * pack_len` intentionally omits the `+1` separator
    slot per pack (it reports the row's real-token budget, not raw capacity) -- so
    `utilization + separator_fraction + tail_pad_fraction == 1 + 1 / pack_len`, not 1.
    """
    if pack_len < 1:
        raise PackingError(f"pack_len must be >= 1; got {pack_len}")
    if not packs:
        raise PackingError("cannot compute stats over an empty pack list")
    capacity = pack_len + 1
    real_tokens = 0
    separators = 0
    tail_pad = 0
    for pack in packs:
        used = 0
        for i in pack:
            cost = min(lengths[i], pack_len) + 1
            real_tokens += cost - 1
            separators += 1
            used += cost
        tail_pad += capacity - used
    capacity_tokens = len(packs) * pack_len
    return PackStats(
        real_tokens=real_tokens,
        capacity_tokens=capacity_tokens,
        utilization=real_tokens / capacity_tokens,
        separator_fraction=separators / capacity_tokens,
        tail_pad_fraction=tail_pad / capacity_tokens,
    )
