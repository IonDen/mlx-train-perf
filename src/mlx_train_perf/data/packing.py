"""Sequence packing: pure first-fit over a seeded per-epoch shuffle (spec §2.3, §4)."""
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

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
                if room[j] <= 1:  # room <= 1 slot: 1-token seq costs 2 (token + separator)
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


def build_row(
    entries: list[tuple[list[int], int]], pack_len: int
) -> tuple[list[int], list[int], list[int], list[bool]]:
    """Lay out one packed row from `entries` (as drawn from one `pack_indices` pack):
    `(tokens, offset)` pairs in pack order (spec §4, the boundary algebra).

    Row axis (`pack_len + 1` wide, mirroring stock's `1 + ...` sizing -- the final
    column is target-only): each segment places its tokens, then a single `0` pad
    slot it owns (the slot stock's row sizing guarantees follows every sequence).

    Inputs axis (`pack_len` wide -- `seg_id`/`seg_start`/`loss_mask`): each segment
    spans its tokens plus its own pad slot (`seg_len + 1` positions); any remaining
    capacity forms one trailing tail segment, so every position in `[0, pack_len)`
    is covered by exactly one segment id -- gapless by construction, a NaN-safety
    invariant for the attention kernel's softmax (spec §4), not cosmetics.

    Supervised window per segment: `[t0 + max(offset, 1) - 1, t0 + seg_len - 1]`
    on the inputs axis (`t0` = the segment's `seg_start`) -- stock's window shifted
    by the segment's row offset.
    """
    row = [0] * (pack_len + 1)
    seg_id = [0] * pack_len
    seg_start = [0] * pack_len
    loss_mask = [False] * pack_len

    cursor = 0
    for seg, (tokens, offset) in enumerate(entries):
        t0 = cursor
        seg_len = len(tokens)
        pad_pos = t0 + seg_len
        row[t0:pad_pos] = tokens
        # row[pad_pos] stays 0: the segment's own pad slot.

        # The pad slot's inputs-axis position is normally `pad_pos`, but when the
        # pack fills the row's full `pack_len + 1` capacity exactly, that slot lands
        # in the final, target-only row column -- clip so this segment's own
        # inputs-axis span never runs past `pack_len - 1`.
        span_end = min(pad_pos, pack_len - 1)
        for pos in range(t0, span_end + 1):
            seg_id[pos] = seg
            seg_start[pos] = t0

        lo = t0 + max(offset, 1) - 1
        hi = t0 + seg_len - 1
        for pos in range(lo, hi + 1):
            loss_mask[pos] = True

        cursor = pad_pos + 1

    tail_seg = len(entries)
    for pos in range(cursor, pack_len):
        seg_id[pos] = tail_seg
        seg_start[pos] = cursor

    return row, seg_id, seg_start, loss_mask


def packed_iterate_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
    *,
    max_position_embeddings: int | None = None,
) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
    """Drop-in for `mlx_lm.tuner.trainer.iterate_batches` (host-side packed variant,
    spec §3.1). Parameter names/order mirror the installed stock signature exactly
    (`iterate_batches`, trainer.py:102-109) so `train(iterate_batches=partial(...))`
    and `evaluate(...)` need zero call-site changes.

    Each dataset item mirrors stock's tuple-or-bare contract (trainer.py:145-148):
    `dataset[j]` is a `(tokens, offset)` 2-tuple, or bare tokens (offset defaults to
    0). Overlong sequences are truncated to `max_seq_length`, printing stock's
    warning-text pattern once over the whole dataset (packing operates over the full
    dataset up front, unlike stock's per-batch truncation).

    Yields `(batch, seg_id, seg_start, loss_mask)`: int32 `(B, L+1)`, int32 `(B, L)`,
    int32 `(B, L)`, bool `(B, L)` `mx.array`s, `L = max_seq_length`, identical shapes
    every iteration -- no `S_max` axis exists, so a compiled `train()` step never
    retraces on packing composition. `loop=True` iterates epochs indefinitely (each
    epoch reshuffles via `pack_indices(seed=seed or 0, epoch=e)`); `loop=False` is
    one pass. Packs are grouped into `batch_size` rows in order; a trailing partial
    batch is dropped (matching stock's full-batch construction).

    Refuses (`PackingError`, no silent fallback): a `comm_group` with `size() > 1`
    (distributed packing is out of scope, spec §10 -- size 1 or `None` is accepted);
    `max_position_embeddings` given and `max_seq_length` exceeds it (positions run
    up to `pack_len` with no reset, spec §4 -- the real correctness fence).
    """
    if comm_group is not None and comm_group.size() > 1:
        raise PackingError(
            "packed_iterate_batches does not support distributed packing "
            f"(comm_group.size()={comm_group.size()}); pack on a single worker"
        )
    if max_position_embeddings is not None and max_seq_length > max_position_embeddings:
        raise PackingError(
            f"max_seq_length={max_seq_length} exceeds max_position_embeddings="
            f"{max_position_embeddings}"
        )

    items: list[tuple[list[int], int]] = []
    for j in range(len(dataset)):
        entry = dataset[j]
        if isinstance(entry, tuple) and len(entry) == 2:
            tokens, offset = entry
        else:
            tokens, offset = entry, 0
        items.append((list(tokens), int(offset)))

    lengths = [len(tokens) for tokens, _ in items]
    if lengths and max(lengths) > max_seq_length:
        print(
            f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
            f"The longest sentence {max(lengths)} will be truncated to "
            f"{max_seq_length}. Consider pre-splitting your data to save memory."
        )
    items = [(tokens[:max_seq_length], offset) for tokens, offset in items]
    lengths = [len(tokens) for tokens, _ in items]

    base_seed = seed or 0
    epoch = 0
    while True:
        packs = pack_indices(lengths, max_seq_length, seed=base_seed, epoch=epoch)
        num_batches = len(packs) // batch_size
        for b in range(num_batches):
            batch_packs = packs[b * batch_size : (b + 1) * batch_size]
            rows: list[list[int]] = []
            seg_ids: list[list[int]] = []
            seg_starts: list[list[int]] = []
            loss_masks: list[list[bool]] = []
            for pack in batch_packs:
                entries = [items[i] for i in pack]
                row, seg_id, seg_start, loss_mask = build_row(entries, max_seq_length)
                rows.append(row)
                seg_ids.append(seg_id)
                seg_starts.append(seg_start)
                loss_masks.append(loss_mask)
            yield (
                mx.array(np.array(rows, dtype=np.int32)),
                mx.array(np.array(seg_ids, dtype=np.int32)),
                mx.array(np.array(seg_starts, dtype=np.int32)),
                mx.array(np.array(loss_masks, dtype=bool)),
            )
        epoch += 1
        if not loop:
            break
