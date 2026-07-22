import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlx_train_perf.data.packing import (
    _assert_pack_time_monotone,
    build_row,
    pack_indices,
    pack_stats,
    packed_iterate_batches,
)
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


def test_pack_stats_hand_computed_multi_sequence_pack() -> None:
    # pack_len=12 -> capacity 13/pack. Pack 0 holds TWO sequences (costs 6+5=11 <=
    # 13); pack 1 holds one (cost 10 <= 13). separators=3 but len(packs)=2, so this
    # fixture -- unlike the singleton-pack fixture above -- distinguishes a
    # per-sequence separator count from a (degenerately equal) per-pack count.
    lengths = [5, 4, 9]
    packs = [[0, 1], [2]]
    stats = pack_stats(packs, lengths, 12)
    assert stats.real_tokens == 18  # 5 + 4 + 9, none truncated
    assert stats.capacity_tokens == 24  # len(packs) * pack_len
    assert stats.utilization == pytest.approx(18 / 24)
    assert stats.separator_fraction == pytest.approx(3 / 24)  # 3 seqs, not 2 packs
    assert stats.tail_pad_fraction == pytest.approx(5 / 24)  # 2*13 - 18 - 3


def test_pack_stats_refusals() -> None:
    with pytest.raises(PackingError):
        pack_stats([], [], 5)
    with pytest.raises(PackingError):
        pack_stats([[0]], [5], 0)


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

    # Expected values computed INDEPENDENTLY from lengths + packs (every sequence is
    # placed exactly once) -- not solved backward from `stats.real_tokens`, which
    # would make the final identity assert a tautology for any real_tokens value.
    expected_real = sum(min(lengths[i], pack_len) for pack in packs for i in pack)
    expected_separators = sum(len(pack) for pack in packs)
    capacity_tokens = len(packs) * pack_len
    expected_tail = len(packs) * (pack_len + 1) - expected_real - expected_separators

    assert stats.real_tokens == expected_real
    assert stats.capacity_tokens == capacity_tokens
    assert stats.separator_fraction == pytest.approx(expected_separators / capacity_tokens)
    assert stats.tail_pad_fraction == pytest.approx(expected_tail / capacity_tokens)

    # The identity now falls out of the independently-pinned fields above.
    lhs = stats.real_tokens + expected_separators + expected_tail
    rhs = stats.capacity_tokens + len(packs)
    assert lhs == rhs


def test_build_row_worked_example():
    # seg A: 5 tokens [10..14], offset (prompt len) 2; seg B: 4 tokens [20..23], offset 1
    # pack_len 12 -> row capacity 13
    row, seg_id, seg_start, loss_mask = build_row(
        [([10, 11, 12, 13, 14], 2), ([20, 21, 22, 23], 1)], pack_len=12)
    #            A  A  A  A  A pad  B   B   B   B pad tail-pad     (13 wide)
    assert row == [10, 11, 12, 13, 14, 0, 20, 21, 22, 23, 0, 0, 0]
    # inputs axis (12 wide): A occupies [0,5], its pad slot idx 5; B [6,10]; tail [11]
    assert seg_id == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    assert seg_start == [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 11]
    # stock window per seg: [t0+max(off,1)-1, t0+Lseg-1]
    # A: t0=0, off=2, L=5 -> [1,4]; B: t0=6, off=1, L=4 -> [6,9]
    assert loss_mask == [False, True, True, True, True, False,
                          True, True, True, True, False, False]


def test_build_row_offset_zero_supervises_from_position_zero():
    _, _, _, m = build_row([([5, 6, 7], 0)], pack_len=8)
    assert m[:3] == [True, True, True]  # max(0,1)-1 == 0 .. Lseg-1 == 2


@given(
    entries=st.lists(
        st.tuples(st.lists(st.integers(1, 1000), min_size=1, max_size=6),
                  st.integers(0, 5)),
        min_size=0, max_size=5,
    ),
    slack=st.integers(0, 5),
)
def test_gapless_coverage_and_monotone(entries, slack):
    # `slack=0` puts the pack at EXACT capacity (the last segment's pad slot lands in
    # the row's final, target-only column) -- the boundary build_row must not overrun.
    required = sum(len(tokens) + 1 for tokens, _ in entries)
    pack_len = max(required - 1 + slack, 1)

    row, seg_id, seg_start, loss_mask = build_row(entries, pack_len)

    assert len(row) == pack_len + 1
    assert len(seg_id) == pack_len
    assert len(seg_start) == pack_len
    assert len(loss_mask) == pack_len

    # gapless + monotone: every inputs-axis position belongs to a segment, seg_id is
    # non-decreasing, and seg_start is self-consistent within each segment's span.
    assert seg_id == sorted(seg_id)
    for i in range(pack_len):
        start = seg_start[i]
        assert 0 <= start <= i
        assert seg_id[start] == seg_id[i]
        if i > 0:
            if seg_id[i] == seg_id[i - 1]:
                assert seg_start[i] == seg_start[i - 1]
            else:
                assert seg_start[i] == i  # a new segment begins exactly at itself


def test_iterator_accepts_stock_kwargs_and_shapes():
    ds = [([1, 2, 3, 4], 1)] * 8
    it = packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=16,
                                 loop=False, seed=3, comm_group=None)
    batch, sid, sst, lm = next(it)
    assert batch.shape == (2, 17)
    assert sid.shape == (2, 16)
    assert sst.shape == (2, 16)
    assert lm.shape == (2, 16)
    assert lm.dtype == mx.bool_
    assert batch.dtype == mx.int32
    assert sid.dtype == mx.int32
    assert sst.dtype == mx.int32


def test_distributed_refuses():
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    with pytest.raises(PackingError):
        next(packed_iterate_batches(dataset=[([1, 2], 0)] * 4, batch_size=2,
                                     max_seq_length=8, comm_group=FakeGroup()))


def test_comm_group_of_size_one_is_accepted():
    class FakeGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    ds = [([1, 2, 3], 0)] * 4
    it = packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=8,
                                 comm_group=FakeGroup())
    next(it)  # must not raise


def test_overlong_truncates_with_warning(capsys):
    long_tokens = list(range(20))
    ds = [(long_tokens, 0), ([1, 2, 3], 0)]
    it = packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=8,
                                 loop=False, seed=1, comm_group=None)
    batch, _, _, _ = next(it)

    captured = capsys.readouterr()
    assert "[WARNING]" in captured.out
    assert "longer than 8 tokens" in captured.out
    assert "truncated to 8" in captured.out
    # the truncated-away tokens (values 8..19) must never reach the packed row
    assert int(mx.max(batch).item()) < 8


def test_pack_len_exceeding_max_position_embeddings_refuses():
    ds = [([1, 2, 3], 0)] * 4
    with pytest.raises(PackingError):
        next(packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=100,
                                     max_position_embeddings=50, comm_group=None))


def test_iterator_refuses_when_packs_fall_below_batch_size() -> None:
    # One tiny sequence packs into a single pack; batch_size=2 -> num_batches==0. Stock's
    # iterate_batches fail-fasts when len(dataset) < batch_size (trainer.py:116-120); ours
    # must too, rather than silently yielding nothing (loop=False).
    with pytest.raises(PackingError, match="fewer than batch_size"):
        next(packed_iterate_batches(dataset=[([1, 2, 3], 0)], batch_size=2,
                                    max_seq_length=16, loop=False, seed=0))


def test_iterator_loop_true_refuses_instead_of_hanging() -> None:
    # loop=True (what mlx-lm's train() always passes) would spin forever re-packing with
    # zero progress when num_batches==0; the guard must raise on the first next() instead.
    with pytest.raises(PackingError, match="fewer than batch_size"):
        next(packed_iterate_batches(dataset=[([1, 2, 3], 0)], batch_size=2,
                                    max_seq_length=16, loop=True, seed=0))


def test_bare_token_dataset_items_default_offset_zero():
    # bare `list[int]` items (no offset) supervise from position 0, same as offset=0.
    ds = [[5, 6, 7, 8]] * 4
    it = packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=8, seed=2)
    _, _, _, loss_mask = next(it)
    assert bool(mx.all(loss_mask[:, 0]).item())  # every row supervises from position 0


def test_epochs_reshuffle_under_loop():
    ds = [([i, i + 1, i + 2, i + 3], 1) for i in range(40)]
    once = list(packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=16,
                                        loop=False, seed=5, comm_group=None))
    n_batches = len(once)
    assert n_batches > 1  # otherwise this test can't observe an epoch boundary

    looped = packed_iterate_batches(dataset=ds, batch_size=2, max_seq_length=16,
                                     loop=True, seed=5, comm_group=None)
    epoch0_first = next(looped)[0]
    for _ in range(n_batches - 1):
        next(looped)
    epoch1_first = next(looped)[0]

    assert epoch0_first.tolist() != epoch1_first.tolist()


# ---------------------------------------------------------------------------
# 0.5.0 T4 -- packer invariant: seg_id AND seg_start are non-decreasing along each
# row of every yielded batch (spec D2). Not a new guarantee -- `build_row` already
# lays segments contiguously left-to-right -- this pins it at the packer's public
# entry point, across seeds/datasets, independent of `test_build_row_worked_example`
# and the hypothesis `test_gapless_coverage_and_monotone` above (which exercise
# `build_row` directly, not the batched/int32-array output `packed_iterate_batches`
# actually yields).
# ---------------------------------------------------------------------------


def _monotone_dataset_a() -> list[tuple[list[int], int]]:
    # Short, fairly uniform sequence lengths -- many small packs per batch.
    return [([i % 250 + 1] * ((i % 4) + 1), i % 2) for i in range(30)]


def _monotone_dataset_b() -> list[tuple[list[int], int]]:
    # Wider length spread -- fewer, more varied sequences per pack.
    return [([i % 250 + 1] * ((i * 7) % 40 + 1), 0) for i in range(50)]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "make_dataset", [_monotone_dataset_a, _monotone_dataset_b],
    ids=["dataset_a", "dataset_b"],
)
def test_packed_batches_have_monotone_seg_id_and_seg_start(seed, make_dataset) -> None:
    dataset = make_dataset()
    it = packed_iterate_batches(
        dataset=dataset, batch_size=2, max_seq_length=64, loop=False, seed=seed,
    )
    saw_a_batch = False
    for _, seg_id, seg_start, _ in it:
        saw_a_batch = True
        sid = np.asarray(seg_id)
        sst = np.asarray(seg_start)
        assert np.all(np.diff(sid, axis=1) >= 0)
        assert np.all(np.diff(sst, axis=1) >= 0)
    assert saw_a_batch  # otherwise the loop above never ran and the test proves nothing


def test_assert_pack_time_monotone_raises_on_non_monotone_seg_id() -> None:
    """`_assert_pack_time_monotone`'s raise branch (spec D2) had zero direct coverage --
    every existing caller only ever fed it packer-produced, already-monotone buffers.
    Hand-build a seg_id row that decreases along its whole length while seg_start stays
    non-decreasing, isolating the seg_id check (`seg_id` is checked before `seg_start`,
    so this also proves the seg_id branch fires first when only seg_id is bad)."""
    sid_arr = np.array([[3, 2, 1, 0]], dtype=np.int32)      # decreasing throughout
    sst_arr = np.array([[0, 1, 2, 3]], dtype=np.int32)      # monotone (non-decreasing)
    with pytest.raises(PackingError, match="seg_id"):
        _assert_pack_time_monotone(sid_arr, sst_arr)


def test_assert_pack_time_monotone_raises_on_non_monotone_seg_start() -> None:
    """The seg_start counterpart of the test above: seg_id stays non-decreasing
    throughout while seg_start decreases along its whole length, isolating the
    seg_start check."""
    sid_arr = np.array([[0, 1, 2, 3]], dtype=np.int32)      # monotone (non-decreasing)
    sst_arr = np.array([[3, 2, 1, 0]], dtype=np.int32)      # decreasing throughout
    with pytest.raises(PackingError, match="seg_start"):
        _assert_pack_time_monotone(sid_arr, sst_arr)


@pytest.mark.metal
def test_validate_never_called_on_the_loss_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make_packed_loss_fn`'s forward walk builds a `PackedMask` (adapters/mlx_lm.py) but
    never calls `.validate()` -- the packer's own pack-time assert (this file, above) is
    the guard on that path, not a per-step host-side check (compile-hostile, spec D2).
    `PackedMask.validate` is monkeypatched to raise unconditionally; since `validate`
    itself does `np.asarray` (a real host sync), even a call reached only under tracing
    would still surface here -- a sound proxy for "never reached", not merely "never
    called eagerly". One tiny kernel-backed loss evaluation through the real, unpatched
    wrapper + adapter must not raise. Setup follows
    tests/test_adapter_packed.py::_parity_losses (tiny llama + enable_flash_attention
    with packed=True)."""
    pytest.importorskip("mlx_lm")
    from test_attention_wrapper import _tiny_llama_hd64  # noqa: PLC0415

    from mlx_train_perf.adapters.mlx_lm import make_packed_loss_fn  # noqa: PLC0415
    from mlx_train_perf.attention.segments import PackedMask  # noqa: PLC0415
    from mlx_train_perf.attention.wrapper import enable_flash_attention  # noqa: PLC0415

    def _raise(_self: "PackedMask") -> None:
        raise AssertionError("PackedMask.validate() must never be called on the loss path")

    monkeypatch.setattr(PackedMask, "validate", _raise)

    pack_len = 16
    mx.random.seed(0)
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=pack_len, batch_size=1, packed=True)
    mx.eval(model.parameters())

    tokens = mx.random.randint(1, 256, (pack_len,)).tolist()
    row, seg_id, seg_start, loss_mask = build_row([(tokens, 0)], pack_len)
    batch = mx.array([row], dtype=mx.int32)
    seg_id_arr = mx.array([seg_id], dtype=mx.int32)
    seg_start_arr = mx.array([seg_start], dtype=mx.int32)
    loss_mask_arr = mx.array([loss_mask], dtype=mx.bool_)

    loss_fn = make_packed_loss_fn(model, impl="kernel")  # type: ignore[arg-type]
    loss, ntoks = loss_fn(model, batch, seg_id_arr, seg_start_arr, loss_mask_arr)
    mx.eval(loss, ntoks)  # must not raise
