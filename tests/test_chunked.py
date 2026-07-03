# Two comparison types below, with two different honest tolerance classes:
#   fp32-exact ref (naive on .astype(fp32) inputs): isolates chunked-vs-naive TRUE numeric
#   error -> e-6/e-7 class (fp32) but e-3 class for bf16, because the pure-MLX chunk matmul
#   rounds each logit to bf16 before upcasting (no fp32 accumulator), unlike a fused kernel.
#   same-dtype ref (naive on the SAME un-upcast bf16 inputs): both sides round logits to bf16
#   identically, so this isolates chunking LOGIC (boundaries/streaming lse/target gather) from
#   dtype rounding -> stays e-6/e-7 class even for bf16. This is the tight regression tripwire.
import mlx.core as mx
import pytest

from mlx_train_perf.core import chunked as ch
from mlx_train_perf.core.chunked import (
    QuantSpec,
    chunked_backward,
    make_chunked_dense,
    make_chunked_quantized,
    streamed_lse_and_target,
)
from mlx_train_perf.core.naive import naive_linear_ce

CASES = [  # (n, d, v, chunk) — boundary-planted: v not divisible, single-chunk, chunk>v
    (64, 32, 1000, 250), (64, 32, 1000, 333), (64, 32, 1000, 1000), (64, 32, 1000, 4096),
    (33, 17, 257, 100),  # nothing divides anything
]


def _data(n: int, d: int, v: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(7)
    hidden = mx.random.normal((n, d)).astype(dtype)
    w = (mx.random.normal((v, d)) * 0.05).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    # plant boundary targets: 0 (always a chunk start) and v-1 (lands on a chunk edge for
    # the non-dividing widths in CASES)
    targets[0] = 0
    targets[1] = v - 1
    return hidden, w, targets


@pytest.mark.parametrize(("n", "d", "v", "chunk"), CASES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_value_parity_vs_fp32_exact_naive(
    n: int, d: int, v: int, chunk: int, dtype: mx.Dtype,
) -> None:
    hidden, w, targets = _data(n, d, v, dtype)
    ref = naive_linear_ce(hidden.astype(mx.float32), w.astype(mx.float32), targets)
    ours = make_chunked_dense(chunk)(hidden, w, targets)
    # fp32-exact ref (see file header): fp32 measured worst 9.54e-7 (n=33,d=17,v=257,c=100)
    # -> pin 2e-6 (~2.1x margin). bf16 measured worst 1.5802e-3 (n=64,d=32,v=1000, all chunk
    # sizes) -> pin 4e-3 (~2.5x margin); see test_value_parity_same_dtype_control below for
    # the tight, dtype-rounding-independent regression check on the chunking logic itself.
    tol = 2e-6 if dtype == mx.float32 else 4e-3
    assert mx.abs(ours - ref).max().item() < tol


@pytest.mark.parametrize(("n", "d", "v", "chunk"), CASES)
def test_value_parity_same_dtype_control(n: int, d: int, v: int, chunk: int) -> None:
    """Chunked vs naive on IDENTICAL bf16 inputs: both round logits to bf16 the same way,
    so this isolates the chunking logic (boundaries, streaming lse, target gather) from
    dtype rounding. The fp32-exact tests above measure true error; THIS one is the tight
    regression tripwire for the algorithm."""
    hidden, w, targets = _data(n, d, v, mx.bfloat16)
    ours = make_chunked_dense(chunk)(hidden, w, targets)
    ref = naive_linear_ce(hidden, w, targets)  # same bf16 path, no pre-upcast
    # measured worst 4.7684e-7 (chunk 250/333/100; chunk 1000/4096 exact 0.0 — single chunk)
    # -> pin 2e-6 (~4.2x margin). fp32-level reassociation only (per-chunk logsumexp then
    # logaddexp vs naive's single logsumexp): confirms the chunking LOGIC is not the source
    # of the wider fp32-exact bf16 gates above — bf16 output-rounding is (see file header).
    assert mx.abs(ours - ref).max().item() < 2e-6


@pytest.mark.parametrize(("n", "d", "v", "chunk"), [(64, 32, 1000, 250), (33, 17, 257, 100)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gradient_parity_dense(n: int, d: int, v: int, chunk: int, dtype: mx.Dtype) -> None:
    hidden, w, targets = _data(n, d, v, dtype)
    ce = make_chunked_dense(chunk)

    def ours(h: mx.array, ww: mx.array) -> mx.array:
        return ce(h, ww, targets).mean()

    def ref(h: mx.array, ww: mx.array) -> mx.array:
        return naive_linear_ce(h, ww, targets).mean()

    (g_h, g_w) = mx.grad(ours, argnums=(0, 1))(hidden, w)
    (r_h, r_w) = mx.grad(ref, argnums=(0, 1))(hidden, w)
    # fp32: measured worst 7.45e-9 (n=33,d=17,v=257,c=100, d_w) -> pin 2e-8 (~2.7x margin).
    # bf16: measured worst 3.0518e-5 (n=33,d=17,v=257,c=100, d_hidden) -> pin 8e-5 (~2.6x margin,
    # same convention as the persisted-spike-derived gates below; the initial 2e-5 gate (based on
    # parity_dense.json's n=32,v=151936 shape, worst 7.63e-6) undershoots this test's smaller-V,
    # larger-weight-scale shapes).
    tol = 2e-8 if dtype == mx.float32 else 8e-5
    assert mx.abs(g_h.astype(mx.float32) - r_h.astype(mx.float32)).max().item() < tol
    assert mx.abs(g_w.astype(mx.float32) - r_w.astype(mx.float32)).max().item() < tol


def test_single_chunk_equals_naive_exactly() -> None:
    hidden, w, targets = _data(64, 32, 1000, mx.float32)
    ours = make_chunked_dense(4096)(hidden, w, targets)  # chunk > v -> one chunk
    ref = naive_linear_ce(hidden, w, targets)
    assert mx.abs(ours - ref).max().item() < 1e-6  # measured 0.0 exactly (single-chunk == naive)


def test_quantized_value_and_dhidden_parity() -> None:
    mx.random.seed(11)
    n, d, v = 64, 128, 1024  # d multiple of group 64
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    q = QuantSpec(w_q=w_q, scales=scales, biases=biases, group_size=64, bits=4)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)

    ce = make_chunked_quantized(256, q)
    ref_nll = naive_linear_ce(hidden.astype(mx.float32), w_dq.astype(mx.float32), targets)
    # fp32-exact ref (see file header): measured worst 3.3150e-3 -> pin 9e-3 (~2.7x margin).
    # ≈2x the dense bf16 value diff from the larger d (bigger logits → bigger bf16 output
    # rounding); 4-bit quant error is common-mode via w_dq.
    assert mx.abs(ce(hidden, targets) - ref_nll).max().item() < 9e-3

    def ours(h: mx.array) -> mx.array:
        return ce(h, targets).mean()

    def ref(h: mx.array) -> mx.array:
        return naive_linear_ce(h, w_dq, targets).mean()

    g = mx.grad(ours)(hidden)
    r = mx.grad(ref)(hidden)
    # measured worst 1.5259e-5 -> pin 4e-5 (~2.6x margin, same convention as parity_quantized.json's
    # worst 7.63e-6 gs64/b4 pinned gate).
    assert mx.abs(g.astype(mx.float32) - r.astype(mx.float32)).max().item() < 4e-5


def test_chunked_backward_engine_with_saved_lse_matches_autograd() -> None:
    """The Task-11 contract: given SAVED lse (no recompute), grads match the oracle."""
    hidden, w, targets = _data(64, 32, 1000, mx.float32)
    n = hidden.shape[0]
    v = w.shape[0]

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    lse, _ = streamed_lse_and_target(mm, targets, v=v, chunk_size=250, n=n)
    ct = mx.full((n,), 1.0 / n)  # cotangent of mean()
    d_h, d_w = chunked_backward(hidden=hidden, matmul_chunk=mm, w_chunk=lambda a, b: w[a:b],
                                targets=targets, lse=lse, cotangent=ct, v=v, chunk_size=250,
                                head_trainable=True)

    def ref(h: mx.array, ww: mx.array) -> mx.array:
        return naive_linear_ce(h, ww, targets).mean()

    r_h, r_w = mx.grad(ref, argnums=(0, 1))(hidden, w)
    assert d_w is not None
    # measured worst 1.75e-9 (d_h) / 3.73e-9 (d_w) -> pin 1e-8 (~2.7x margin).
    assert mx.abs(d_h - r_h).max().item() < 1e-8
    assert mx.abs(d_w - r_w).max().item() < 1e-8


def test_chunked_backward_engine_frozen_head_matches_autograd() -> None:
    """head_trainable=False: d_w is None AND d_hidden still matches the oracle by VALUE
    (a shape-only check would pass an all-zero d_hidden)."""
    hidden, w, targets = _data(64, 32, 1000, mx.float32)
    n = hidden.shape[0]
    v = w.shape[0]

    def mm(v0: int, v1: int) -> mx.array:
        return (hidden @ w[v0:v1].T).astype(mx.float32)

    lse, _ = streamed_lse_and_target(mm, targets, v=v, chunk_size=250, n=n)
    ct = mx.full((n,), 1.0 / n)
    d_h, d_w = chunked_backward(hidden=hidden, matmul_chunk=mm, w_chunk=lambda a, b: w[a:b],
                                targets=targets, lse=lse, cotangent=ct, v=v, chunk_size=250,
                                head_trainable=False)
    assert d_w is None
    r_h = mx.grad(lambda h: naive_linear_ce(h, w, targets).mean())(hidden)
    # measured worst 1.75e-9 -> pin 1e-8 (~5.7x margin; same d_hidden path as the trainable-head
    # test above, head_trainable only gates whether d_w is accumulated).
    assert mx.abs(d_h - r_h).max().item() < 1e-8


def test_custom_vjp_is_actually_engaged() -> None:
    """A dropped .vjp registration silently autodiffs through the pure-MLX forward with
    IDENTICAL gradients (verified 0.31.2) — parity can't catch it; this counter can."""
    hidden, w, targets = _data(64, 32, 1000, mx.float32)
    ce = make_chunked_dense(250)
    ch.VJP_CALLS.clear()
    g = mx.grad(lambda h: ce(h, w, targets).mean())(hidden)
    mx.eval(g)
    assert ch.VJP_CALLS.get("dense", 0) > 0
