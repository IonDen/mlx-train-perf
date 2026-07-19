"""0.2.0 T3 — pure-MLX attention oracles (spec Section 8 dual oracles, Section 9 P1).

`math_attention` and `flash_attention_reference` are the foundation every later parity
test (T4-T13) compares against, so this suite anchors two independent things:

1. Forward correctness against `mx.fast.scaled_dot_product_attention` (the trusted
   fused implementation) and against an independently-built logsumexp/causal-mask.
2. THE gradient anchor (review-tests High): every later backward-parity test
   differentiates `math_attention`'s graph. If a non-differentiable indexing op or a
   stray `stop_gradient` crept into `reference.py`, every later gradient test would be
   comparing a kernel against a silently-broken oracle and never notice. Finite
   differences, computed independently of MLX autodiff, are the only check that catches
   that class of bug.
"""
from collections.abc import Callable

import mlx.core as mx
import pytest

from mlx_train_perf.attention import flash_attention_reference, kv_head_for, math_attention
from mlx_train_perf.attention.segments import PackedMask


def _rand_qkv(
    *, b: int, hq: int, hkv: int, n: int, d: int, seed: int
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, n, d))
    k = mx.random.normal((b, hkv, n, d))
    v = mx.random.normal((b, hkv, n, d))
    mx.eval(q, k, v)
    return q, k, v


@pytest.mark.parametrize(
    ("hq", "hkv"), [(4, 4), (4, 2)], ids=["mha", "gqa"]
)
def test_math_attention_matches_fast_sdpa_forward(hq: int, hkv: int) -> None:
    """vs mx.fast.scaled_dot_product_attention(..., mask='causal') on fp32.

    Measured-first (mlx 0.32.0, fp32, n=64, d=32, seed=0): worst |diff| 6.78e-07 (mha),
    4.77e-07 (gqa) -> pin 3e-6 (~4.4x margin over the measured worst).
    """
    n, d = 64, 32
    scale = 1.0 / (d**0.5)
    q, k, v = _rand_qkv(b=1, hq=hq, hkv=hkv, n=n, d=d, seed=0)

    o_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    o_ours = math_attention(q, k, v, scale=scale, causal=True)
    mx.eval(o_ref, o_ours)

    diff = mx.abs(o_ours.astype(mx.float32) - o_ref.astype(mx.float32)).max()
    assert diff.item() < 3e-6


def test_flash_reference_L_is_row_logsumexp() -> None:  # noqa: N802 -- L is the spec's name
    """L must equal an INDEPENDENTLY built logsumexp of the masked, scaled scores --
    built here via mx.where (not reference.py's mx.triu-additive-mask path), so the
    check does not merely re-run the implementation against itself."""
    n, d = 16, 8
    scale = 1.0 / (d**0.5)
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=n, d=d, seed=1)

    _, l_ours = flash_attention_reference(q, k, v, scale=scale, causal=True)

    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    scores = (q32 @ k32.swapaxes(-1, -2)) * scale  # (B, H, N, N)
    i_idx = mx.arange(n)[:, None]
    j_idx = mx.arange(n)[None, :]
    causal = mx.where(j_idx <= i_idx, mx.array(0.0), mx.array(-mx.inf))
    scores = scores + causal
    l_ref = mx.logsumexp(scores, axis=-1)
    mx.eval(l_ours, l_ref)

    assert l_ours.dtype == mx.float32
    diff = mx.abs(l_ours - l_ref).max()
    # measured-first (mlx 0.32.0, fp32, n=16, d=8, seed=1): worst |diff| 0.0 exactly
    # (same fp32 arithmetic, independently constructed) -> pin 1e-6 exact-class tolerance.
    assert diff.item() < 1e-6


def test_flash_reference_O_equals_math_attention() -> None:  # noqa: N802 -- O is the spec's name
    """Bit-identical: both public entry points route O through the same code path."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=2, n=32, d=16, seed=2)
    scale = 1.0 / (16**0.5)

    o_math = math_attention(q, k, v, scale=scale, causal=True)
    o_flash, _ = flash_attention_reference(q, k, v, scale=scale, causal=True)
    mx.eval(o_math, o_flash)

    assert mx.array_equal(o_math, o_flash).item()


def test_causal_first_row_attends_only_itself() -> None:
    """Row 0 of O equals the mapped kv head's row 0 of V, per q head -- with causal
    masking, token 0 can only attend to itself."""
    hq, hkv, n, d = 4, 2, 8, 4
    group = hq // hkv
    q, k, v = _rand_qkv(b=1, hq=hq, hkv=hkv, n=n, d=d, seed=3)
    scale = 1.0 / (d**0.5)

    o = math_attention(q, k, v, scale=scale, causal=True)
    mx.eval(o)

    for h in range(hq):
        expected = v[0, kv_head_for(h, group), 0]
        assert mx.allclose(o[0, h, 0], expected, atol=1e-5).item()


@pytest.mark.parametrize(("hq", "hkv"), [(4, 2), (8, 2)])
def test_gqa_uses_pinned_convention(hq: int, hkv: int) -> None:
    """T2 pattern reapplied to math_attention: kv head j holds a constant (j+1), so
    softmax's convex combination reproduces that constant regardless of the attention
    weights -- discriminates the contiguous (h // group) mapping from the interleaved
    (h % Hkv) one."""
    n, d = 4, 8
    group = hq // hkv
    q = mx.random.normal((1, hq, n, d))
    k = mx.random.normal((1, hkv, n, d))
    v = mx.stack(
        [mx.full((n, d), float(j + 1)) for j in range(hkv)], axis=0
    )[None]  # (1, hkv, n, d)
    mx.eval(q, k, v)

    o = math_attention(q, k, v, scale=1.0, causal=True)
    mx.eval(o)

    contiguous = [h // group for h in range(hq)]
    interleaved = [h % hkv for h in range(hq)]
    assert contiguous != interleaved
    for h in range(hq):
        expected = mx.full((n, d), float(contiguous[h] + 1))
        assert mx.allclose(o[0, h], expected).item(), (
            f"q head {h}: math_attention did not use the contiguous // mapping"
        )


def _central_diff(
    fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    q: mx.array, k: mx.array, v: mx.array,
    *, which: int, idx: tuple[int, ...], eps: float,
) -> float:
    plus = [mx.array(q), mx.array(k), mx.array(v)]
    minus = [mx.array(q), mx.array(k), mx.array(v)]
    plus[which][idx] = plus[which][idx] + eps
    minus[which][idx] = minus[which][idx] - eps
    f_plus = fn(*plus).item()
    f_minus = fn(*minus).item()
    return (f_plus - f_minus) / (2.0 * eps)


@pytest.mark.parametrize(
    ("hq", "hkv", "n", "d"), [(2, 2, 8, 4), (4, 2, 8, 4)], ids=["mha", "gqa"]
)
def test_math_attention_grads_match_finite_differences(
    hq: int, hkv: int, n: int, d: int
) -> None:
    """THE independent gradient anchor. Central differences on a handful of q/k/v
    elements vs mx.grad of a scalar readout (a fixed random projection dotted with O,
    so the readout is sensitive to every output element -- a plain .sum() would let
    softmax's row-sums-to-1 cancellation hide indexing bugs).

    Measured-first (mlx 0.32.0, fp32, eps=1e-3, 9 sampled elements each): worst
    |fd - autodiff| 1.591e-04 (mha), 3.258e-04 (gqa) -> pin 1e-3 (~3.1x margin over the
    measured worst).
    """
    b = 1
    scale = 1.0 / (d**0.5)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, seed=4)
    mx.random.seed(5)
    proj = mx.random.normal((b, hq, n, d))
    mx.eval(proj)

    def readout(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        o = math_attention(q_, k_, v_, scale=scale, causal=True)
        return (o * proj).sum()

    g_q, g_k, g_v = mx.grad(readout, argnums=(0, 1, 2))(q, k, v)
    mx.eval(g_q, g_k, g_v)

    eps = 1e-3
    tol = 1e-3
    worst = 0.0
    cases: list[tuple[int, mx.array, tuple[int, ...]]] = [
        (0, g_q, (0, 0, 0, 0)),
        (0, g_q, (0, hq - 1, n - 1, d - 1)),
        (0, g_q, (0, 0, n // 2, 1)),
        (1, g_k, (0, 0, 0, 0)),
        (1, g_k, (0, hkv - 1, n - 1, d - 1)),
        (1, g_k, (0, 0, n // 2, 1)),
        (2, g_v, (0, 0, 0, 0)),
        (2, g_v, (0, hkv - 1, n - 1, d - 1)),
        (2, g_v, (0, 0, n // 2, 1)),
    ]
    for which, grad, idx in cases:
        fd = _central_diff(readout, q, k, v, which=which, idx=idx, eps=eps)
        auto = grad[idx].item()
        worst = max(worst, abs(fd - auto))
        assert abs(fd - auto) < tol, (
            f"which={which} idx={idx}: finite-diff {fd} vs autodiff {auto}"
        )


def test_math_attention_segments_requires_causal_true() -> None:
    """`segments` composes with the causal triangle (block-diagonal-CAUSAL isolation);
    causal=False + segments is an unsupported combination and must fail loudly rather
    than silently attend across the whole row."""
    n1, n2, d = 3, 5, 8
    n = n1 + n2
    q, k, v = _rand_qkv(b=1, hq=2, hkv=1, n=n, d=d, seed=6)
    seg_id = mx.array([[0] * n1 + [1] * n2], dtype=mx.int32)
    seg_start = mx.array([[0] * n1 + [n1] * n2], dtype=mx.int32)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)

    with pytest.raises(AssertionError):
        math_attention(q, k, v, scale=1.0 / (d**0.5), causal=False, segments=pm)
