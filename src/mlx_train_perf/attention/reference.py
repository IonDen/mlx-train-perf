"""Pure-MLX attention oracles: no Metal, no models.

Two oracles that every later Metal-kernel attention parity test (T4-T13) compares
against:

- `math_attention` -- the materialized reference: builds the full `(N, N)` score
  matrix, softmaxes it in fp32, and matmuls into V. MLX autodiff differentiates
  straight through it, so it doubles as the GRADIENT oracle every backward-parity
  test is checked against.
- `flash_attention_reference` -- returns `(O, L)`, the flash-attention forward
  contract: `O` is bit-identical to `math_attention` (same underlying code path), and
  `L` is the fp32 row logsumexp of the masked, scaled scores -- what a fused kernel
  saves to reconstruct the backward pass without re-materializing the full `(N, N)`
  matrix.

GQA convention (T2-pinned, `tests/test_attention_composition.py
::test_gqa_grouping_convention_matches_mlx`): query heads are grouped CONTIGUOUSLY, so
kv head index = `q_head // group_size` -- matching
`mx.fast.scaled_dot_product_attention`. K/V are never repeated or tiled; they are
gathered per q-head through this mapping.

Layouts (spec Section 4.1): q `(B, Hq, N, D)`, k/v `(B, Hkv, N, D)`, O `(B, Hq, N, D)`,
L `(B, Hq, N)` fp32. Softmax/logsumexp math runs in fp32 regardless of input dtype
(upcast internally); O is cast back to the input dtype at the end. L always stays fp32.

Packed sequences (0.4.0, spec 2026-07-17 §3.2/§3.3): `segments=None` on every public
function below is BYTE-IDENTICAL to pre-0.4.0 pure-causal behavior -- no regression for
callers written before packing existed. `segments=PackedMask(...)` replaces the causal
triangle with block-diagonal-causal isolation (`segment_allowed`): key j visible to
query i iff same segment AND j <= i. Requires `causal=True`.
"""
import mlx.core as mx

from mlx_train_perf.attention.segments import PackedMask, segment_allowed


def kv_head_for(q_head: int, group_size: int) -> int:
    """T2-pinned GQA convention: contiguous grouping, matching mx.fast.sdpa."""
    return q_head // group_size


def _masked_scaled_scores(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool,
    segments: PackedMask | None = None,
) -> tuple[mx.array, mx.array]:
    """fp32 upcast, gather K/V per q-head via `kv_head_for`, scale, optional causal
    mask. Returns `(scores, v_gathered)`, both fp32: scores `(B, Hq, N, N)`,
    v_gathered `(B, Hq, N, D)` -- the shared core both public entry points build on.

    `segments` (requires `causal=True`) swaps the causal triangle for block-diagonal-
    causal isolation via `segment_allowed`; the `segments=None` branch below is
    untouched from pre-0.4.0 -- byte-identical behavior for every existing caller.
    """
    hq = q.shape[1]
    hkv = k.shape[1]
    if hq % hkv != 0:
        raise ValueError(f"Hq={hq} must be a multiple of Hkv={hkv} for GQA grouping")
    group_size = hq // hkv
    head_idx = mx.array([kv_head_for(h, group_size) for h in range(hq)])

    q32 = q.astype(mx.float32)
    k_g = mx.take(k, head_idx, axis=1).astype(mx.float32)  # (B, Hq, N, D)
    v_g = mx.take(v, head_idx, axis=1).astype(mx.float32)  # (B, Hq, N, D)

    scores = (q32 @ k_g.swapaxes(-1, -2)) * scale  # (B, Hq, N, N)
    if segments is not None:
        assert causal, "segments requires causal=True"
        scores = mx.where(segment_allowed(segments.seg_id), scores, -mx.inf)
        return scores, v_g
    if causal:
        n = q.shape[2]
        neg_inf = mx.full((n, n), -mx.inf, dtype=mx.float32)
        scores = scores + mx.triu(neg_inf, k=1)  # additive -inf where j > i
    return scores, v_g


def _attention_core(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool,
    segments: PackedMask | None = None,
) -> tuple[mx.array, mx.array]:
    """Shared O, L computation. Both public functions call this exact function, so
    their O outputs are the same code path (bit-identical)."""
    scores, v_g = _masked_scaled_scores(
        q, k, v, scale=scale, causal=causal, segments=segments
    )
    lse = mx.logsumexp(scores, axis=-1)  # (B, Hq, N) fp32
    probs = mx.exp(scores - lse[..., None])
    o32 = probs @ v_g  # (B, Hq, N, D) fp32
    o = o32.astype(q.dtype)
    return o, lse


def math_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool = True,
    segments: PackedMask | None = None,
) -> mx.array:
    """`softmax(scale * Q K^T + causal_mask) @ V`, fp32 softmax internally, GQA via
    `kv_head_for`. Fully differentiable through MLX autodiff -- the gradient oracle.
    `segments` (requires `causal=True`) composes block-diagonal-causal isolation on
    top of the causal triangle -- see `mlx_train_perf.attention.segments.PackedMask`.
    """
    o, _ = _attention_core(q, k, v, scale=scale, causal=causal, segments=segments)
    return o


def flash_attention_reference(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool = True,
    segments: PackedMask | None = None,
) -> tuple[mx.array, mx.array]:
    """`(O, L)`: O identical to `math_attention` (same code path), L `(B, Hq, N)`
    fp32 row logsumexp of the masked scaled scores -- what a flash kernel saves for
    backward. `segments` (requires `causal=True`) composes block-diagonal-causal
    isolation on top of the causal triangle -- see
    `mlx_train_perf.attention.segments.PackedMask`.
    """
    return _attention_core(q, k, v, scale=scale, causal=causal, segments=segments)
