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


def _assert_pack_time_monotone(sid_arr: Any, sst_arr: Any) -> None:
    """Vectorized numpy check (spec D2): `seg_id` and `seg_start` must be
    non-decreasing along each row of a packed batch -- the packed kernels' block-skip
    bounds (forward/dQ `kv_lo`, the 0.5.0 dK/dV segment-end break) assume contiguous
    ascending segments, with no in-kernel guard. Called on the numpy host arrays
    BEFORE the `mx.array` conversion -- never an mx-side check (that would host-sync
    every batch on the training data path), never a Python per-element loop (real
    per-step host cost at L=4096). `build_row` already produces this by construction;
    this is the pack-time assert that guards it (`PackedMask.validate` is the
    equivalent opt-in check for a hand-built mask that bypasses the packer)."""
    for name, arr in (("seg_id", sid_arr), ("seg_start", sst_arr)):
        if not bool(np.all(np.diff(arr.astype(np.int64), axis=1) >= 0)):
            raise PackingError(
                f"packed batch {name} must be non-decreasing along each row: the "
                "packed kernels' block-skip bounds (forward/dQ kv_lo, dK/dV "
                "segment-end break) assume contiguous ascending segments"
            )


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
        if num_batches == 0:
            raise PackingError(
                f"packed dataset produced {len(packs)} pack(s) at pack_len={max_seq_length}, "
                f"fewer than batch_size={batch_size}; reduce batch_size or add more data"
            )
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
            # Pack-time invariant assert (spec D2). The numpy arrays are built ONCE
            # here and reused below in the `mx.array` conversion -- no double
            # conversion.
            sid_arr = np.array(seg_ids, dtype=np.int32)
            sst_arr = np.array(seg_starts, dtype=np.int32)
            _assert_pack_time_monotone(sid_arr, sst_arr)
            yield (
                mx.array(np.array(rows, dtype=np.int32)),
                mx.array(sid_arr),
                mx.array(sst_arr),
                mx.array(np.array(loss_masks, dtype=bool)),
            )
        epoch += 1
        if not loop:
            break


# ---------------------------------------------------------------------------------
# Dataset-level batching analytics (host-side, GPU-free) -- the numbers behind the
# packed-training bench (scripts/bench_packed_training.py). `stock_batching_stats`
# replays mlx-lm's own sort+batch+pad-to-max logic exactly (the padding-waste baseline);
# `packed_batching_stats` reports the per-step real-token/sample counts over the packs a
# whole number of batches consumes, plus the whole-dataset `pack_stats` fractions.
# ---------------------------------------------------------------------------------

_STOCK_PAD_TO = 32  # mlx_lm.tuner.trainer.iterate_batches pads to 1 + 32*ceil(max/32)


@dataclass(frozen=True, slots=True, kw_only=True)
class LengthHistogram:
    """Sequence-length summary for a tokenized dataset (all counts in tokens)."""

    count: int
    mean: float
    median: float
    p90: float
    minimum: int
    maximum: int
    total_tokens: int


def length_histogram(lengths: Sequence[int]) -> LengthHistogram:
    """Count / mean / median / p90 / min / max / total over a dataset's sequence lengths.
    p90 is numpy's default (linear-interpolation) percentile."""
    if not lengths:
        raise PackingError("cannot summarize an empty length list")
    arr: Any = np.asarray(lengths, dtype=np.int64)
    return LengthHistogram(
        count=len(lengths),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p90=float(np.percentile(arr, 90)),
        minimum=int(arr.min()),
        maximum=int(arr.max()),
        total_tokens=int(arr.sum()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class StockBatchingStats:
    """Padding-waste breakdown of stock (unpacked) batching at a fixed batch size."""

    num_batches: int
    real_tokens_total: int
    padded_tokens_total: int
    padding_waste_fraction: float
    mean_real_tokens_per_step: float
    mean_samples_per_step: float


def stock_batching_stats(
    lengths: Sequence[int], *, batch_size: int, max_seq_length: int
) -> StockBatchingStats:
    """Replay `mlx_lm.tuner.trainer.iterate_batches` (trainer.py:102-170) host-side to
    measure its padding waste: sort by length, group into `batch_size` batches (dropping
    the trailing partial), pad each batch to `min(1 + 32*ceil(max_in_batch/32),
    max_seq_length)`, and truncate content to `max_seq_length`. `padding_waste_fraction`
    is the share of the allocated `(num_batches * batch_size * width)` token budget that
    is padding rather than real content."""
    if batch_size < 1:
        raise PackingError(f"batch_size must be >= 1; got {batch_size}")
    if max_seq_length < 1:
        raise PackingError(f"max_seq_length must be >= 1; got {max_seq_length}")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    real_total = 0
    padded_total = 0
    num_batches = 0
    for start in range(0, len(order) - batch_size + 1, batch_size):
        batch_lengths = [lengths[i] for i in order[start : start + batch_size]]
        widest = max(batch_lengths)
        width = min(
            1 + _STOCK_PAD_TO * ((widest + _STOCK_PAD_TO - 1) // _STOCK_PAD_TO),
            max_seq_length,
        )
        real_total += sum(min(n, max_seq_length) for n in batch_lengths)
        padded_total += batch_size * width
        num_batches += 1
    waste = (padded_total - real_total) / padded_total if padded_total else 0.0
    mean_real = real_total / num_batches if num_batches else 0.0
    return StockBatchingStats(
        num_batches=num_batches,
        real_tokens_total=real_total,
        padded_tokens_total=padded_total,
        padding_waste_fraction=waste,
        mean_real_tokens_per_step=mean_real,
        mean_samples_per_step=float(batch_size),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedBatchingStats:
    """Per-step real-token / sample counts (over the CONSUMED packs) plus the
    whole-dataset utilization fractions (from `pack_stats`)."""

    num_batches: int
    real_tokens_total: int
    mean_real_tokens_per_step: float
    mean_samples_per_step: float
    utilization: float
    separator_fraction: float
    tail_pad_fraction: float


def packed_batching_stats(
    lengths: Sequence[int], pack_len: int, *, batch_size: int, seed: int, epoch: int = 0
) -> PackedBatchingStats:
    """Pack the dataset once (`pack_indices(seed, epoch)`), then measure the packs a
    whole number of `batch_size` batches consumes (the trailing partial batch is dropped,
    matching `packed_iterate_batches`). `real_tokens_total` / the per-step means are over
    those consumed packs; `utilization`/`separator_fraction`/`tail_pad_fraction` come from
    `pack_stats` over the full pack list."""
    if batch_size < 1:
        raise PackingError(f"batch_size must be >= 1; got {batch_size}")
    packs = pack_indices(lengths, pack_len, seed=seed, epoch=epoch)
    num_batches = len(packs) // batch_size
    consumed = packs[: num_batches * batch_size]
    real_consumed = sum(min(lengths[i], pack_len) for pack in consumed for i in pack)
    samples_consumed = sum(len(pack) for pack in consumed)
    stats = pack_stats(packs, lengths, pack_len)
    return PackedBatchingStats(
        num_batches=num_batches,
        real_tokens_total=real_consumed,
        mean_real_tokens_per_step=real_consumed / num_batches if num_batches else 0.0,
        mean_samples_per_step=samples_consumed / num_batches if num_batches else 0.0,
        utilization=stats.utilization,
        separator_fraction=stats.separator_fraction,
        tail_pad_fraction=stats.tail_pad_fraction,
    )
