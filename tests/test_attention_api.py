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
from mlx_train_perf.attention.segments import PackedMask
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


def _packed_layout(seg_lens: list[int], b: int) -> tuple[mx.array, mx.array]:
    """(B, N) int32 seg_id/seg_start for a fixed segment-length list, shared across batch rows
    (mirrors tests/test_attention_kernel_fwd.py::_packed_layout -- the PackedMask contract:
    seg_id contiguous ascending, seg_start each position's segment-start index)."""
    seg_id_row: list[int] = []
    seg_start_row: list[int] = []
    start = 0
    for sid, ln in enumerate(seg_lens):
        seg_id_row += [sid] * ln
        seg_start_row += [start] * ln
        start += ln
    seg_id = mx.array([seg_id_row] * b, dtype=mx.int32)
    seg_start = mx.array([seg_start_row] * b, dtype=mx.int32)
    return seg_id, seg_start


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


# =======================================================================================
# T5 (0.4.0): sequence packing threaded through the public `flash_attention` API.
# =======================================================================================


def test_int_primal_vjp_int_zero_cotangent_matches_autodiff() -> None:
    """SETTLED (mlx 0.32.0, plan-review empirical probe, verified eager + compiled): a
    custom_function vjp may return int-dtype zeros for an int primal's cotangent -- the grad
    wrt the FLOAT primal is unaffected and matches plain autodiff bit-for-bit. This is the
    contract `flash_attention`'s packed vjp relies on for `seg_id`/`seg_start` (the two int
    routing primals carry no gradient). Kept executable so the claim can't silently rot."""
    idx = mx.array([1, 2, 3], dtype=mx.int32)

    @mx.custom_function
    def scaled(x: mx.array, i: mx.array) -> mx.array:
        return x * i.astype(x.dtype)

    @scaled.vjp
    def scaled_vjp(
        primals: tuple[mx.array, mx.array], cotangents: mx.array, _out: mx.array
    ) -> tuple[mx.array, mx.array]:
        x, i = primals
        return cotangents * i.astype(x.dtype), mx.zeros(i.shape, dtype=i.dtype)

    def plain(x: mx.array, i: mx.array) -> mx.array:
        return x * i.astype(x.dtype)

    x = mx.array([0.5, -1.5, 2.0])
    g_custom = mx.grad(lambda x_: scaled(x_, idx).sum())(x)
    g_plain = mx.grad(lambda x_: plain(x_, idx).sum())(x)
    mx.eval(g_custom, g_plain)

    assert mx.array_equal(g_custom, g_plain).item()


def test_segments_with_non_causal_refuses() -> None:
    """Packed attention is block-diagonal ON TOP of the causal triangle, so `segments=` with
    `causal=False` is unsupported -- refused up front (never silently masked, never an assert
    deep in the oracle). Checked at both the resolver and the public entry point."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=2, n=8, d=32, seed=30)
    seg_id, seg_start = _packed_layout([4, 4], b=1)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    with pytest.raises(UnsupportedAttentionError, match="causal"):
        resolve_attention_impl(q, k, v, impl="reference", causal=False, segments=pm)
    with pytest.raises(UnsupportedAttentionError, match="causal"):
        flash_attention(q, k, v, scale=1.0, causal=False, impl="reference", segments=pm)


def test_reference_serves_segments_and_packed_vjp_engages() -> None:
    """The reference oracle serves `segments=` at tiny N: forward value AND grad match
    math_attention(segments=) autodiff (the block-diagonal-causal oracle), and the hand vjp
    fires (VJP_CALLS increments -- a dropped .vjp would autodiff to identical values, per the
    engagement-sentinel precedent). Fenced to tiny N: the reference backward materializes the
    (N, N) matrices by design."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 32
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, seed=31)
    seg_id, seg_start = _packed_layout([10, 14], b=b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    api.VJP_CALLS.clear()

    o_flash = flash_attention(q, k, v, scale=scale, causal=True, impl="reference", segments=pm)
    o_math = math_attention(q, k, v, scale=scale, causal=True, segments=pm)
    mx.eval(o_flash, o_math)
    assert mx.array_equal(o_flash, o_math).item()

    def flash_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(
            q_, k_, v_, scale=scale, causal=True, impl="reference", segments=pm
        ).sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True, segments=pm).sum()

    g_flash = mx.grad(flash_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_flash, *g_math)
    worst = max(
        float(mx.abs(gf - gm).max().item()) for gf, gm in zip(g_flash, g_math, strict=True)
    )
    assert worst < 2e-5, f"reference packed grad worst |diff|={worst}"
    assert api.VJP_CALLS.get("flash_attention", 0) > 0


# Measured-first through the PUBLIC kernel path vs the block-diagonal reference oracle (mlx
# 0.32.0, M1 Max, fp32, seed=32):
#   O:    n256 7.153e-7, n1024 8.345e-7 -> pin 2e-6 (~2.4x; same class as the packed O fp32 grid).
#   grad: n256 7.629e-6, n1024 1.431e-5 -> pin 3e-5 (~2.1x over the n1024 worst; the grad-of-sum
#         accumulates dQ+dK/dV over all rows, so it runs hotter than the n=24 causal grad pin).
_PACKED_API_O_TOL = 2e-6
_PACKED_API_GRAD_TOL = 3e-5


@pytest.mark.metal
@pytest.mark.parametrize("n", [256, 1024])
def test_packed_parity_through_public_api(n: int) -> None:
    """The whole packed path THROUGH `flash_attention(impl='kernel', segments=)`: forward value
    matches the block-diagonal oracle, grads wrt q/k/v match math_attention(segments=) autodiff,
    and the KERNEL-backward branch fires (`flash_attention_kernel_bwd` increments -- proves the
    vjp is kernel-backed, not the pure-MLX oracle; value parity alone can't tell them apart)."""
    b, hq, hkv, d = 1, 4, 2, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, seed=32)
    seg_lens = [100, 156] if n == 256 else [400, 300, 324]
    seg_id, seg_start = _packed_layout(seg_lens, b=b)
    pm = PackedMask(seg_id=seg_id, seg_start=seg_start)
    api.VJP_CALLS.clear()

    o_kernel = flash_attention(q, k, v, scale=scale, causal=True, impl="kernel", segments=pm)
    o_ref = math_attention(q, k, v, scale=scale, causal=True, segments=pm)
    mx.eval(o_kernel, o_ref)
    d_o = mx.abs(o_kernel - o_ref).max().item()

    def kernel_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(
            q_, k_, v_, scale=scale, causal=True, impl="kernel", segments=pm
        ).sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True, segments=pm).sum()

    g_kernel = mx.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_kernel, *g_math)
    worst = max(
        float(mx.abs(gk - gm).max().item()) for gk, gm in zip(g_kernel, g_math, strict=True)
    )
    print(f"[packed API n{n}] O={d_o:.3e} grad_worst={worst:.3e}")
    assert d_o < _PACKED_API_O_TOL, f"packed API O diff {d_o}"
    assert worst < _PACKED_API_GRAD_TOL, f"packed API grad worst {worst}"
    assert api.VJP_CALLS.get("flash_attention_kernel_bwd", 0) > 0


@pytest.mark.metal
def test_packed_fresh_segments_refeed_under_compile() -> None:
    """spec 8.3 / review F2: `seg_id`/`seg_start` are custom_function PRIMALS, never closure
    captures, so a COMPILED fn re-fed a DIFFERENT segment layout (same shapes -> one trace) must
    produce DIFFERENT masking. A closure capture would freeze layout A into the trace and layout
    B would return layout-A's (wrong) output. The oracle says the two layouts differ, and the
    kernel-backward branch must fire on the packed grad path."""
    b, hq, hkv, n, d = 1, 4, 2, 64, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, seed=33)
    seg_id_a, seg_start_a = _packed_layout([32, 32], b=b)
    seg_id_b, seg_start_b = _packed_layout([16, 48], b=b)

    def fn(
        q_: mx.array, k_: mx.array, v_: mx.array, sid: mx.array, sst: mx.array
    ) -> mx.array:
        return flash_attention(
            q_, k_, v_, scale=scale, causal=True, impl="kernel",
            segments=PackedMask(seg_id=sid, seg_start=sst),
        )

    compiled = mx.compile(fn)
    o_a = compiled(q, k, v, seg_id_a, seg_start_a)
    o_b = compiled(q, k, v, seg_id_b, seg_start_b)
    o_ref_a = math_attention(
        q, k, v, scale=scale, causal=True,
        segments=PackedMask(seg_id=seg_id_a, seg_start=seg_start_a),
    )
    o_ref_b = math_attention(
        q, k, v, scale=scale, causal=True,
        segments=PackedMask(seg_id=seg_id_b, seg_start=seg_start_b),
    )
    mx.eval(o_a, o_b, o_ref_a, o_ref_b)

    # The two layouts genuinely differ (oracle), so a threaded kernel must differ too.
    assert not mx.array_equal(o_ref_a, o_ref_b).item(), "layouts A/B do not differ -- weak test"
    assert mx.abs(o_a - o_ref_a).max().item() < 1e-2, "compiled A != oracle A"
    assert mx.abs(o_b - o_ref_b).max().item() < 1e-2, "compiled B != oracle B"
    assert not mx.array_equal(o_a, o_b).item(), (
        "re-fed layout B produced layout A's output -- seg buffers were captured, not threaded"
    )

    # The packed grad path also fires the kernel-backward branch under compile.
    api.VJP_CALLS.clear()

    def loss(q_: mx.array, sid: mx.array, sst: mx.array) -> mx.array:
        return flash_attention(
            q_, k, v, scale=scale, causal=True, impl="kernel",
            segments=PackedMask(seg_id=sid, seg_start=sst),
        ).sum()

    g = mx.compile(mx.grad(loss))(q, seg_id_b, seg_start_b)
    mx.eval(g)
    assert api.VJP_CALLS.get("flash_attention_kernel_bwd", 0) > 0
