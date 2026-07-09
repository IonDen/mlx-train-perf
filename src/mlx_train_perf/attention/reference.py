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
"""
import mlx.core as mx


def kv_head_for(q_head: int, group_size: int) -> int:
    """T2-pinned GQA convention: contiguous grouping, matching mx.fast.sdpa."""
    return q_head // group_size


def _masked_scaled_scores(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool
) -> tuple[mx.array, mx.array]:
    """fp32 upcast, gather K/V per q-head via `kv_head_for`, scale, optional causal
    mask. Returns `(scores, v_gathered)`, both fp32: scores `(B, Hq, N, N)`,
    v_gathered `(B, Hq, N, D)` -- the shared core both public entry points build on.
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
    if causal:
        n = q.shape[2]
        neg_inf = mx.full((n, n), -mx.inf, dtype=mx.float32)
        scores = scores + mx.triu(neg_inf, k=1)  # additive -inf where j > i
    return scores, v_g


def _attention_core(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool
) -> tuple[mx.array, mx.array]:
    """Shared O, L computation. Both public functions call this exact function, so
    their O outputs are the same code path (bit-identical)."""
    scores, v_g = _masked_scaled_scores(q, k, v, scale=scale, causal=causal)
    lse = mx.logsumexp(scores, axis=-1)  # (B, Hq, N) fp32
    probs = mx.exp(scores - lse[..., None])
    o32 = probs @ v_g  # (B, Hq, N, D) fp32
    o = o32.astype(q.dtype)
    return o, lse


def math_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool = True
) -> mx.array:
    """`softmax(scale * Q K^T + causal_mask) @ V`, fp32 softmax internally, GQA via
    `kv_head_for`. Fully differentiable through MLX autodiff -- the gradient oracle."""
    o, _ = _attention_core(q, k, v, scale=scale, causal=causal)
    return o


def flash_attention_reference(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool = True
) -> tuple[mx.array, mx.array]:
    """`(O, L)`: O identical to `math_attention` (same code path), L `(B, Hq, N)`
    fp32 row logsumexp of the masked scaled scores -- what a flash kernel saves for
    backward."""
    return _attention_core(q, k, v, scale=scale, causal=causal)
