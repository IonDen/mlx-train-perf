"""0.2.0 T5 -- flash-attention FORWARD Metal kernel v0 (O + L), correctness only.

Two lanes in one file (the `test_kernel_guard.py` mixed-file convention):

- Pure-arithmetic tests (source templating, range planning, budget math) stay in the
  DEFAULT lane -- no `@pytest.mark.metal`, no GPU.
- Every test that launches the kernel carries a PER-TEST `@pytest.mark.metal` -- never a
  module-level mark, so the arithmetic tests above run without `--run-metal`.

Parity is checked against BOTH oracles (T3 `math_attention` and `mx.fast.scaled_dot_
product_attention`) for O, and against the reference `L` (`flash_attention_reference`).
Tolerances are measured-first: the per-case worst is printed, and the committed bound is
the smallest honest value over the grid (see the pin comments).
"""
import math

import mlx.core as mx
import pytest

from mlx_train_perf.attention import api
from mlx_train_perf.attention.api import flash_attention, resolve_attention_impl
from mlx_train_perf.attention.kernel.launch import (
    TileShape,
    _fwd_macs_per_row,
    check_fwd_budget,
    launch_flash_fwd,
)
from mlx_train_perf.attention.kernel.source import build_fwd_source
from mlx_train_perf.attention.reference import (
    flash_attention_reference,
    kv_head_for,
    math_attention,
)
from mlx_train_perf.errors import LaunchBudgetError

GENEROUS_RATE = 1e13  # parity shapes are microscopic; the budget check must never trip


# ---------------------------------------------------------------------------------------
# Pure-arithmetic: source templating + budget math (DEFAULT lane, no GPU).
# ---------------------------------------------------------------------------------------


def test_build_fwd_source_substitutes_head_dim() -> None:
    for hd in (64, 96, 128):
        s = build_fwd_source(hd, causal=True)
        assert f"float qreg[{hd}];" in s
        assert f"float acc[{hd}];" in s
        assert f"dd < {hd}" in s
        assert "HEAD_DIM" not in s  # every sentinel substituted (lossless)


def test_build_fwd_source_causal_keep_comparison() -> None:
    s = build_fwd_source(64, causal=True)
    assert "kk <= row" in s
    assert "kk >= row" not in s


def test_build_fwd_source_noncausal_keeps_all_keys() -> None:
    s = build_fwd_source(64, causal=False)
    assert "kk <= row" not in s
    assert "kk >= row" not in s
    assert "bool keep = (true);" in s


def test_build_fwd_source_flip_causal_inverts_the_comparison() -> None:
    # The test-only wrong-mask arm: the causal comparison is flipped to the WRONG
    # triangle so a parity run FAILS -- proving the suite can fail.
    s = build_fwd_source(64, causal=True, flip_causal=True)
    assert "kk >= row" in s
    assert "kk <= row" not in s


def test_build_fwd_source_rejects_bad_head_dim() -> None:
    for hd in (0, 32, 80, 256):
        with pytest.raises(ValueError, match="head_dim"):
            build_fwd_source(hd, causal=True)


def test_build_fwd_source_rejects_flip_without_causal() -> None:
    with pytest.raises(ValueError, match="flip_causal"):
        build_fwd_source(64, causal=False, flip_causal=True)


def test_check_fwd_budget_passes_a_cheap_dispatch() -> None:
    # tiny per-row cost at a real rate: nowhere near the 1 s / 60 s bounds
    check_fwd_budget(n=64, d=64, b=1, hq=4, rows=64, rate=1e12)


def test_check_fwd_budget_refuses_when_one_row_over_budgets() -> None:
    # a single query row projected over 1 s at this (deliberately low) rate -> refuse
    per_row = _fwd_macs_per_row(n=16384, d=128, b=1, hq=32)
    slow_rate = per_row / 2.0  # 1 row projects to 2 s > the 1 s per-dispatch bound
    with pytest.raises(LaunchBudgetError):
        check_fwd_budget(n=16384, d=128, b=1, hq=32, rows=1, rate=slow_rate)


def test_check_fwd_budget_refuses_over_total_budget() -> None:
    # per-dispatch fine (1 row), but the whole forward over all rows blows the 60 s total
    per_row = _fwd_macs_per_row(n=16384, d=128, b=1, hq=32)
    rate = per_row * 100.0  # 1 row ~ 0.01 s (ok), all 16384 rows ~ 164 s (> 60 s)
    with pytest.raises(LaunchBudgetError):
        check_fwd_budget(n=16384, d=128, b=1, hq=32, rows=1, rate=rate)


# ---------------------------------------------------------------------------------------
# Metal parity (PER-TEST @pytest.mark.metal).
# ---------------------------------------------------------------------------------------

# head-config x N cases; the flagship (32/8) pattern only at N=64 to bound cost.
_HEAD_N_CASES = [
    (4, 4, 64), (4, 4, 61), (4, 4, 257),   # MHA
    (4, 2, 64), (4, 2, 61), (4, 2, 257),   # GQA
    (32, 8, 64),                            # flagship group_size-4 pattern
]

# Measured worsts over the whole grid (mlx 0.32.0, M1 Max, seed=7):
#   O vs math_attention / vs sdpa: fp32 9.537e-07, bf16 7.812e-03
#   L vs reference (always fp32):  fp32 9.537e-07, bf16 1.431e-06
# The kernel accumulates fp32 in-register for both input dtypes, so fp32 O/L diffs are pure
# reduction-order noise (~1e-6). bf16 O is written back in bf16, so its worst is one bf16
# ULP at an O value near 1-2 (2^-7 ~= 7.8e-3) -- the same single rounding the reference's
# o32.astype(bf16) does, differing by at most a ULP from the fp32 accumulation-order gap.
# Pins are the smallest honest bound over THIS grid (fp32 ~2.1x, bf16 O ~1.5x margin, same
# measure-first convention as tests/test_kernel_parity.py). A future case landing between a
# pin and 2 bf16 ULP is not a regression -- widen toward 2 ULP with a note.
_TOL_O = {mx.float32: 2e-6, mx.bfloat16: 1.2e-2}
_TOL_L = {mx.float32: 2e-6, mx.bfloat16: 5e-6}


def _rand_qkv(
    *, b: int, hq: int, hkv: int, n: int, d: int, dtype: mx.Dtype, seed: int = 7
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, n, d)).astype(dtype)
    k = mx.random.normal((b, hkv, n, d)).astype(dtype)
    v = mx.random.normal((b, hkv, n, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


@pytest.mark.metal
@pytest.mark.parametrize(("hq", "hkv", "n"), _HEAD_N_CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_fwd_parity_vs_both_oracles_and_reference_lse(
    hq: int, hkv: int, n: int, head_dim: int, batch: int, dtype: mx.Dtype
) -> None:
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(b=batch, hq=hq, hkv=hkv, n=n, d=head_dim, dtype=dtype)

    o_k, l_k = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=GENEROUS_RATE,
    )
    o_math = math_attention(q, k, v, scale=scale, causal=True)
    o_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    # L oracle: logsumexp of the masked scaled scores, fp32 (reference contract)
    _, l_ref = _reference_o_l(q, k, v, scale=scale)
    mx.eval(o_k, l_k, o_math, o_sdpa, l_ref)

    f = mx.float32
    d_math = mx.abs(o_k.astype(f) - o_math.astype(f)).max().item()
    d_sdpa = mx.abs(o_k.astype(f) - o_sdpa.astype(f)).max().item()
    d_l = mx.abs(l_k - l_ref).max().item()
    print(
        f"[{['fp32','bf16'][dtype==mx.bfloat16]} b{batch} {hq}/{hkv} n{n} d{head_dim}] "
        f"O-math={d_math:.3e} O-sdpa={d_sdpa:.3e} L={d_l:.3e}"
    )

    assert d_math < _TOL_O[dtype], f"O vs math {d_math}"
    assert d_sdpa < _TOL_O[dtype], f"O vs sdpa {d_sdpa}"
    assert d_l < _TOL_L[dtype], f"L vs reference {d_l}"


def _reference_o_l(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float
) -> tuple[mx.array, mx.array]:
    return flash_attention_reference(q, k, v, scale=scale, causal=True)


@pytest.mark.metal
def test_fwd_row0_attends_only_itself() -> None:
    """Causal row 0 attends only key 0: O[.,.,0]==V[.,kv,0], L[.,.,0]==scale*(q0.k0)."""
    b, hq, hkv, n, d = 2, 4, 2, 8, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=1)

    o_k, l_k = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=GENEROUS_RATE,
    )
    group = hq // hkv
    mx.eval(o_k, l_k)
    for bb in range(b):
        for h in range(hq):
            kvh = kv_head_for(h, group)
            o0 = o_k[bb, h, 0]
            expect_o = v[bb, kvh, 0]
            expect_l = (q[bb, h, 0].astype(mx.float32) * k[bb, kvh, 0].astype(mx.float32)
                        ).sum().item() * scale
            assert mx.abs(o0 - expect_o).max().item() < 1e-5
            assert abs(l_k[bb, h, 0].item() - expect_l) < 1e-4


@pytest.mark.metal
def test_fwd_bitwise_deterministic_across_runs() -> None:
    """No atomics by design -> bit-identical O and L across repeated runs. Lock it."""
    q, k, v = _rand_qkv(b=2, hq=4, hkv=2, n=129, d=64, dtype=mx.float32, seed=2)
    scale = 1.0 / math.sqrt(64)
    o0, l0 = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=GENEROUS_RATE,
    )
    mx.eval(o0, l0)
    for _ in range(4):
        o, lse = launch_flash_fwd(
            q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
            rate_macs_per_s=GENEROUS_RATE,
        )
        mx.eval(o, lse)
        assert mx.array_equal(o, o0).item()
        assert mx.array_equal(lse, l0).item()


@pytest.mark.metal
def test_fwd_split_matches_single_dispatch() -> None:
    """Query-range multi-dispatch writes DISJOINT O/L rows; the reassembled result must
    be bit-identical to a single dispatch. This is the outer-grid offset guard (a wrong
    r0 offset corrupts a chunk) -- run at batch>1 and an N that is not a block multiple."""
    b, hq, hkv, n, d = 2, 4, 2, 257, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=3)

    single_o, single_l = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=GENEROUS_RATE,   # one dispatch over all rows
    )
    # Force ~32 rows/dispatch (9 disjoint dispatches over n=257) via a low rate; the +0.5
    # keeps the projected per-dispatch strictly under the 1 s bound.
    per_row = _fwd_macs_per_row(n=n, d=d, b=b, hq=hq)
    split_rate = per_row * 32.5 / 1.0
    split_o, split_l = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=split_rate,
    )
    mx.eval(single_o, single_l, split_o, split_l)
    assert mx.array_equal(single_o, split_o).item()
    assert mx.array_equal(single_l, split_l).item()


@pytest.mark.metal
def test_fwd_wrong_mask_perturbation_fails_parity() -> None:
    """Deliberate wrong-mask: build the kernel with the causal comparison flipped to the
    WRONG triangle. Its O/L must DIVERGE from the causal reference -- if this ever matched,
    the parity tests above could not detect a real mask bug (the suite would be unfalsifiable)."""
    b, hq, hkv, n, d = 2, 4, 2, 16, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=4)

    o_wrong, l_wrong = launch_flash_fwd(
        q, k, v, scale=scale, causal=True, tile=TileShape(bq=32),
        rate_macs_per_s=GENEROUS_RATE, _flip_causal=True,
    )
    o_ref, l_ref = _reference_o_l(q, k, v, scale=scale)
    mx.eval(o_wrong, l_wrong, o_ref, l_ref)

    d_o = mx.abs(o_wrong.astype(mx.float32) - o_ref.astype(mx.float32)).max().item()
    d_l = mx.abs(l_wrong - l_ref).max().item()
    assert d_o > 1e-2 or d_l > 1e-2, (
        f"flipped-mask kernel matched the causal reference (O={d_o:.3e}, L={d_l:.3e}) -- "
        "the parity suite cannot detect a mask bug"
    )


# ---------------------------------------------------------------------------------------
# Step 5: impl="kernel" wired into the api -- Metal FORWARD, staged pure-MLX BACKWARD.
# ---------------------------------------------------------------------------------------


@pytest.mark.metal
def test_impl_kernel_resolves_now_that_forward_is_built() -> None:
    """A fully-supported config resolves to 'kernel' (T5), no longer refusing 'not built
    yet'. Metal-marked: resolution needs `mx.metal.is_available()` true."""
    q, k, v = _rand_qkv(b=1, hq=4, hkv=4, n=16, d=64, dtype=mx.float32, seed=6)
    assert resolve_attention_impl(q, k, v, impl="kernel", causal=True) == "kernel"
    assert resolve_attention_impl(q, k, v, impl="auto", causal=True) == "kernel"


@pytest.mark.metal
def test_kernel_forward_reference_backward_grads_match_oracle() -> None:
    """impl='kernel' routes the FORWARD through the Metal kernel; the BACKWARD is the
    staged pure-MLX vjp. Grads of sum(flash_attention(impl='kernel')) must match the
    `math_attention` autodiff oracle, and the hand vjp must actually fire (a dropped
    registration would autodiff through the kernel forward -- but the kernel forward has no
    autodiff, so it would error; the counter proves the vjp path, not just non-error)."""
    b, hq, hkv, n, d = 1, 4, 2, 24, 64
    scale = 1.0 / math.sqrt(d)
    q, k, v = _rand_qkv(b=b, hq=hq, hkv=hkv, n=n, d=d, dtype=mx.float32, seed=9)
    api.VJP_CALLS.clear()

    def kernel_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return flash_attention(q_, k_, v_, scale=scale, causal=True, impl="kernel").sum()

    def math_loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return math_attention(q_, k_, v_, scale=scale, causal=True).sum()

    g_kernel = mx.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v)
    g_math = mx.grad(math_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_kernel, *g_math)

    worst = max(
        float(mx.abs(gk - gm).max().item())
        for gk, gm in zip(g_kernel, g_math, strict=True)
    )
    # Kernel forward O/L feed the same pure-MLX backward the oracle uses; worst |grad diff|
    # is fp32 reduction-order noise (measured 3.815e-06, seed=9) -> pin 2e-5 (~5.2x, same
    # convention as test_attention_api.py::test_flash_attention_grads_match_autodiff_oracle).
    assert worst < 2e-5, f"worst |grad diff|={worst}"
    assert api.VJP_CALLS.get("flash_attention", 0) > 0
