"""0.2.0 T4 -- `flash_attention` public API (spec Section 4.1, Section 9 P1).

The full custom_function boundary working BEFORE any Metal kernel exists: forward
routes through T3's `flash_attention_reference`, backward is a hand-written pure-MLX
tile-math vjp (the FlashAttention paper's backward equations, over full un-tiled
tensors -- an oracle, fenced to tiny N). This proves the vjp seam end-to-end so a
Metal kernel can swap in underneath in T5 without changing this surface.
"""
import math

import mlx.core as mx
import pytest

from mlx_train_perf.attention import api
from mlx_train_perf.attention.api import flash_attention, resolve_attention_impl
from mlx_train_perf.attention.reference import math_attention
from mlx_train_perf.errors import AttentionInputError, UnsupportedAttentionError


def _rand_qkv(
    *, b: int, hq: int, hkv: int, n: int, d: int, seed: int, dtype: mx.Dtype = mx.float32
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, n, d)).astype(dtype)
    k = mx.random.normal((b, hkv, n, d)).astype(dtype)
    v = mx.random.normal((b, hkv, n, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def test_flash_attention_value_matches_math_attention() -> None:
    """impl='reference' routes through the same T3 code path as math_attention, so O
    must be bit-identical (per T3's own contract, test_flash_reference_O_equals_..."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=2, n=32, d=16, seed=0)
    scale = 1.0 / (16**0.5)

    o_flash = flash_attention(q, k, v, scale=scale, causal=True, impl="reference")
    o_math = math_attention(q, k, v, scale=scale, causal=True)
    mx.eval(o_flash, o_math)

    assert mx.array_equal(o_flash, o_math).item()


@pytest.mark.parametrize("n", [96, 61], ids=["pow2-multiple", "n61-odd"])
def test_flash_attention_grads_match_autodiff_oracle(n: int) -> None:
    """mx.grad of sum(flash_attention(...)) vs mx.grad through math_attention (the T3
    autodiff gradient oracle) -- GQA (Hq=4, Hkv=2) at both a pow2-block-multiple N and
    an N that is NOT a multiple of any pow-2 block (61), which exercises the pure
    backward's edge math (no tiling assumption to break, but worth pinning explicitly).

    Measured-first (mlx 0.32.0, fp32, seed=10/11): worst |diff| 3.814697e-06 (n=96),
    1.668930e-06 (n=61) -> pin 2e-5 (~5.2x margin over the measured worst).
    """
    b, hq, hkv, d = 1, 4, 2, 32
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, seed=10 if n == 96 else 11)
    scale = 1.0 / math.sqrt(d)

    def flash_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="reference").sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True).sum()

    g_flash = mx.grad(flash_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_flash, *g_math)

    worst = max(
        float(mx.abs(gf - gm).max().item()) for gf, gm in zip(g_flash, g_math, strict=True)
    )
    assert worst < 2e-5, f"n={n}: worst |diff|={worst}"


def test_backward_D_identity() -> None:  # noqa: N802 -- D is the paper's name
    """`_bwd_D(dO, O) == rowsum(dO * O)` -- the flash-attention paper's row-correction
    term, checked in isolation from the rest of the backward."""
    mx.random.seed(12)
    d_o = mx.random.normal((2, 3, 5, 4))
    o = mx.random.normal((2, 3, 5, 4))
    mx.eval(d_o, o)

    d = api._bwd_D(d_o, o)
    expected = (d_o.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)
    mx.eval(d, expected)

    assert mx.array_equal(d, expected).item()


def test_impl_auto_refuses_fp16_hidden() -> None:
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=64, seed=13, dtype=mx.float16)
    with pytest.raises(UnsupportedAttentionError, match="fp32/bf16"):
        flash_attention(q, k, v, scale=1.0, causal=True, impl="auto")


def test_impl_auto_refuses_head_dim_80() -> None:
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=80, seed=14)
    with pytest.raises(UnsupportedAttentionError, match="head_dim"):
        flash_attention(q, k, v, scale=1.0, causal=True, impl="auto")


def test_impl_auto_refuses_non_causal() -> None:
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=64, seed=15)
    with pytest.raises(UnsupportedAttentionError, match="causal"):
        flash_attention(q, k, v, scale=1.0, causal=False, impl="auto")


@pytest.mark.metal
def test_impl_kernel_now_runs_the_metal_forward() -> None:
    """T5: a fully-supported config no longer refuses 'not built yet' -- impl='kernel'
    resolves and routes the FORWARD through the Metal kernel. O matches the math_attention
    oracle (the full parity grid lives in tests/test_attention_kernel_fwd.py). Metal-marked:
    resolution needs a real GPU and the wired kernel launch."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=64, seed=16)
    scale = 1.0 / math.sqrt(64)
    assert resolve_attention_impl(q, k, v, impl="kernel", causal=True) == "kernel"
    o_kernel = flash_attention(q, k, v, scale=scale, causal=True, impl="kernel")
    o_math = math_attention(q, k, v, scale=scale, causal=True)
    mx.eval(o_kernel, o_math)
    assert mx.abs(o_kernel - o_math).max().item() < 2e-6


def test_impl_reference_always_allowed() -> None:
    """'reference' is never refused by the kernel-support checks -- it's the oracle,
    not a production path (fenced to tiny N by every caller, never by this function)."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=80, seed=17, dtype=mx.float16)
    assert resolve_attention_impl(q, k, v, impl="reference") == "reference"


def test_validate_shapes_rejects_mismatched_kv() -> None:
    """Hq % Hkv != 0 -- GQA grouping is undefined."""
    q = mx.random.normal((1, 5, 16, 32))
    k = mx.random.normal((1, 2, 16, 32))
    v = mx.random.normal((1, 2, 16, 32))
    mx.eval(q, k, v)
    with pytest.raises(AttentionInputError, match="Hkv"):
        resolve_attention_impl(q, k, v, impl="reference")


def test_validate_shapes_rejects_non_4d() -> None:
    q = mx.random.normal((4, 16, 32))
    k = mx.random.normal((1, 4, 16, 32))
    v = mx.random.normal((1, 4, 16, 32))
    mx.eval(q, k, v)
    with pytest.raises(AttentionInputError, match="4-D"):
        resolve_attention_impl(q, k, v, impl="reference")


def test_custom_vjp_engaged_not_autodiff() -> None:
    """A dropped .vjp registration would silently autodiff through the reference
    forward with IDENTICAL gradient VALUES -- flash_attention_reference is built from
    ops MLX autodiff differentiates just fine (same code path as math_attention), so
    value-parity alone can't prove the hand vjp fired. This counter can (0.1.0
    chunked-vjp engagement precedent: tests/test_chunked.py
    ::test_custom_vjp_is_actually_engaged)."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=2, n=16, d=32, seed=18)
    scale = 1.0 / math.sqrt(32)
    api.VJP_CALLS.clear()

    def loss(q_: mx.array) -> mx.array:
        return flash_attention(q_, k, v, scale=scale, causal=True, impl="reference").sum()

    g = mx.grad(loss)(q)
    mx.eval(g)

    assert api.VJP_CALLS.get("flash_attention", 0) > 0
