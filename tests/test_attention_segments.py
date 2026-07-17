"""0.4.0 T1 -- PackedMask carrier + block-diagonal segment support in the pure-MLX
attention oracles.

`PackedMask` describes a packed batch (multiple variable-length sequences concatenated
into one row) as a per-position segment id plus each position's segment-start offset.
`flash_attention_reference` / `math_attention` given `segments=` must behave exactly
like running the SAME oracle independently on each segment's slice -- block-diagonal
causal isolation, no cross-segment attention -- both in the forward output/logsumexp
and in the backward gradients (the composition tests below), and `segments=None` must
be byte-for-byte the existing pure-causal behavior (no regression for every caller that
predates packing).
"""
import mlx.core as mx
import pytest

from mlx_train_perf.attention import api
from mlx_train_perf.attention.reference import flash_attention_reference, math_attention
from mlx_train_perf.attention.segments import PackedMask, segment_allowed


def _two_segment_inputs(n1: int = 3, n2: int = 5, hq: int = 2, hkv: int = 1, d: int = 8):
    n = n1 + n2
    kq, kk, kv = (mx.random.key(s) for s in (0, 1, 2))
    q = mx.random.normal((1, hq, n, d), key=kq)
    k = mx.random.normal((1, hkv, n, d), key=kk)
    v = mx.random.normal((1, hkv, n, d), key=kv)
    seg_id = mx.array([[0] * n1 + [1] * n2], dtype=mx.int32)
    seg_start = mx.array([[0] * n1 + [n1] * n2], dtype=mx.int32)
    return q, k, v, PackedMask(seg_id=seg_id, seg_start=seg_start), n1


def test_packed_oracle_matches_per_segment_composition() -> None:
    q, k, v, pm, n1 = _two_segment_inputs()
    o_packed, lse_packed = flash_attention_reference(q, k, v, scale=0.125, segments=pm)
    o_a, lse_a = flash_attention_reference(
        q[:, :, :n1], k[:, :, :n1], v[:, :, :n1], scale=0.125)
    o_b, lse_b = flash_attention_reference(
        q[:, :, n1:], k[:, :, n1:], v[:, :, n1:], scale=0.125)
    assert mx.allclose(o_packed[:, :, :n1], o_a, atol=1e-6).item()
    assert mx.allclose(o_packed[:, :, n1:], o_b, atol=1e-6).item()
    assert mx.allclose(lse_packed[:, :, :n1], lse_a, atol=1e-6).item()
    assert mx.allclose(lse_packed[:, :, n1:], lse_b, atol=1e-6).item()


def test_packed_oracle_gradients_match_per_segment_composition() -> None:
    q, k, v, pm, n1 = _two_segment_inputs()
    def loss_packed(q_, k_, v_):
        return math_attention(q_, k_, v_, scale=0.125, segments=pm).sum()
    grads_p = mx.grad(loss_packed, argnums=(0, 1, 2))(q, k, v)
    def loss_solo(q_, k_, v_):
        return math_attention(q_, k_, v_, scale=0.125).sum()
    ga = mx.grad(loss_solo, argnums=(0, 1, 2))(q[:, :, :n1], k[:, :, :n1], v[:, :, :n1])
    gb = mx.grad(loss_solo, argnums=(0, 1, 2))(q[:, :, n1:], k[:, :, n1:], v[:, :, n1:])
    for gp, gs in zip(grads_p, ga, strict=True):
        assert mx.allclose(gp[:, :, :n1], gs, atol=1e-6).item()
    for gp, gs in zip(grads_p, gb, strict=True):  # BOTH segments: proves block-diagonal
        assert mx.allclose(gp[:, :, n1:], gs, atol=1e-6).item()  # isolation, not just A


def test_segments_none_is_pure_causal() -> None:
    q, k, v, _, _ = _two_segment_inputs()
    o1, l1 = flash_attention_reference(q, k, v, scale=0.125)
    o2, l2 = flash_attention_reference(q, k, v, scale=0.125, segments=None)
    assert mx.array_equal(o1, o2).item()
    assert mx.array_equal(l1, l2).item()


def test_segment_allowed_matches_expected_block_diagonal_causal_pattern() -> None:
    """Direct check of the (B, 1, N, N) mask against a hand-built expected pattern for
    a tiny two-segment example (n1=2, n2=2): a key is visible to a query iff same
    segment AND key index <= query index -- independent of `_two_segment_inputs` and
    of `flash_attention_reference`/`math_attention`, so a bug shared between the helper
    and its only caller would still show up here."""
    seg_id = mx.array([[0, 0, 1, 1]], dtype=mx.int32)
    allowed = segment_allowed(seg_id)
    expected = mx.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [False, False, True, True],
        ]
    )[None, None]  # (1, 1, 4, 4)
    mx.eval(allowed, expected)
    assert allowed.shape == (1, 1, 4, 4)
    assert allowed.dtype == mx.bool_
    assert mx.array_equal(allowed, expected).item()


def test_flash_attention_backward_matches_per_segment_composition() -> None:
    """`api._flash_attention_backward` (the reference vjp) must mask S identically to
    the forward oracle: dQ/dK/dV computed on the packed inputs with `segments=` must
    equal running the SAME backward on each segment's slice alone -- the backward
    analogue of `test_packed_oracle_gradients_match_per_segment_composition`, but
    exercising the hand-written vjp directly rather than via `mx.grad`."""
    q, k, v, pm, n1 = _two_segment_inputs()
    scale = 0.125
    o, lse = flash_attention_reference(q, k, v, scale=scale, segments=pm)
    kd = mx.random.key(7)
    d_o = mx.random.normal(o.shape, key=kd)
    mx.eval(o, lse, d_o)

    dq_p, dk_p, dv_p = api._flash_attention_backward(
        q, k, v, o, lse, d_o, scale=scale, causal=True, segments=pm)

    o_a, lse_a = flash_attention_reference(
        q[:, :, :n1], k[:, :, :n1], v[:, :, :n1], scale=scale)
    dq_a, dk_a, dv_a = api._flash_attention_backward(
        q[:, :, :n1], k[:, :, :n1], v[:, :, :n1], o_a, lse_a, d_o[:, :, :n1],
        scale=scale, causal=True)

    o_b, lse_b = flash_attention_reference(
        q[:, :, n1:], k[:, :, n1:], v[:, :, n1:], scale=scale)
    dq_b, dk_b, dv_b = api._flash_attention_backward(
        q[:, :, n1:], k[:, :, n1:], v[:, :, n1:], o_b, lse_b, d_o[:, :, n1:],
        scale=scale, causal=True)

    mx.eval(dq_p, dk_p, dv_p, dq_a, dk_a, dv_a, dq_b, dk_b, dv_b)

    # Measured (2026-07-17, mlx 0.32.0, M1 Max): all six dQ/dK/dV comparisons are
    # bit-identical (max abs diff 0.0e+00) -- the masked-out cross-segment/future
    # entries are exact fp32 zeros (exp(-inf) == 0.0), so they don't perturb the
    # accumulated sums. atol=1e-6 matches the sibling forward/gradient composition
    # tests above in this file rather than pinning to the measured 0.0 floor, which
    # would be fragile to a harmless future change in reduction order.
    assert mx.allclose(dq_p[:, :, :n1], dq_a, atol=1e-6).item()
    assert mx.allclose(dk_p[:, :, :n1], dk_a, atol=1e-6).item()
    assert mx.allclose(dv_p[:, :, :n1], dv_a, atol=1e-6).item()
    assert mx.allclose(dq_p[:, :, n1:], dq_b, atol=1e-6).item()
    assert mx.allclose(dk_p[:, :, n1:], dk_b, atol=1e-6).item()
    assert mx.allclose(dv_p[:, :, n1:], dv_b, atol=1e-6).item()


def test_flash_attention_backward_requires_causal_true() -> None:
    """`segments` is only meaningful under causal=True (block-diagonal-causal
    composition assumes the causal triangle); causal=False + segments must fail
    loudly rather than silently ignore the segment boundaries."""
    q, k, v, pm, _ = _two_segment_inputs()
    scale = 0.125
    o, lse = flash_attention_reference(q, k, v, scale=scale, segments=pm)
    d_o = mx.zeros_like(o)
    with pytest.raises(AssertionError):
        api._flash_attention_backward(
            q, k, v, o, lse, d_o, scale=scale, causal=False, segments=pm)
