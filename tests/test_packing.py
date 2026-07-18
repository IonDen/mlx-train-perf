import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlx_train_perf.data.packing import pack_indices, pack_stats
from mlx_train_perf.errors import PackingError


@given(st.lists(st.integers(1, 300), min_size=1, max_size=200),
       st.integers(0, 3), st.integers(0, 2))
def test_every_sequence_placed_exactly_once(lengths, seed, epoch):
    packs = pack_indices(lengths, 1024, seed=seed, epoch=epoch)
    placed = sorted(i for p in packs for i in p)
    assert placed == list(range(len(lengths)))


@given(st.lists(st.integers(1, 300), min_size=1, max_size=200))
def test_no_pack_exceeds_capacity(lengths):
    for p in pack_indices(lengths, 1024, seed=0, epoch=0):
        assert sum(min(lengths[i], 1024) + 1 for i in p) <= 1024 + 1


def test_deterministic_per_seed_epoch_and_varies_across_epochs():
    lengths = list(range(1, 120))
    a = pack_indices(lengths, 256, seed=7, epoch=0)
    assert a == pack_indices(lengths, 256, seed=7, epoch=0)
    assert a != pack_indices(lengths, 256, seed=7, epoch=1)


def test_refusals():
    with pytest.raises(PackingError):
        pack_indices([], 256, seed=0, epoch=0)
    with pytest.raises(PackingError):
        pack_indices([0, 5], 256, seed=0, epoch=0)
    with pytest.raises(PackingError):
        pack_indices([5], 0, seed=0, epoch=0)


def test_pack_stats_hand_computed_with_truncation() -> None:
    # pack_len=10 -> capacity 11/pack. seq0 len=15 truncates to 10 (cost 11, fills
    # its pack exactly); seq1 len=3 (cost 4). Two singleton packs, given explicitly
    # so this test exercises pack_stats alone, independent of the packer's shuffle.
    lengths = [15, 3]
    packs = [[0], [1]]
    stats = pack_stats(packs, lengths, 10)
    assert stats.real_tokens == 13  # min(15,10) + min(3,10)
    assert stats.capacity_tokens == 20  # len(packs) * pack_len
    assert stats.utilization == pytest.approx(13 / 20)
    assert stats.separator_fraction == pytest.approx(2 / 20)  # 1 separator/seq
    assert stats.tail_pad_fraction == pytest.approx(7 / 20)  # 22 full cap - 15 used


@given(st.lists(st.integers(1, 300), min_size=1, max_size=200),
       st.integers(0, 3), st.integers(0, 2), st.integers(1, 1024))
def test_pack_stats_accounting_identity(lengths, seed, epoch, pack_len):
    # real_tokens + separators + tail_pad == capacity_tokens + len(packs), because
    # each pack's full capacity (pack_len + 1) splits into: real content tokens
    # (min(len, pack_len) per placed sequence) + one separator slot per placed
    # sequence + whatever room is left unused (tail_pad). Summed over len(packs)
    # packs, capacity is len(packs) * (pack_len + 1) == capacity_tokens + len(packs).
    packs = pack_indices(lengths, pack_len, seed=seed, epoch=epoch)
    stats = pack_stats(packs, lengths, pack_len)
    separators = len(lengths)  # every sequence placed exactly once
    total_full_capacity = len(packs) * (pack_len + 1)
    tail_pad = total_full_capacity - stats.real_tokens - separators
    assert stats.tail_pad_fraction == pytest.approx(tail_pad / stats.capacity_tokens)
    lhs = (stats.real_tokens + separators + tail_pad)
    rhs = stats.capacity_tokens + len(packs)
    assert lhs == rhs
