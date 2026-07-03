"""Chunked linear-CE: pure-MLX streaming path. Never materializes full (N,V) logits.

Roles: (1) the explicit `impl="chunked"` fallback (its own custom_functions, recompute-lse
backward — spike-1's exact proven shape); (2) `chunked_backward` — the vjp engine the
kernel impl reuses with SAVED lse (no recompute).

Lazy discipline: no mx.eval inside chunk loops — under grad tracing it RETAINS chunks
(measured 6x worse peak at n=8192; user-metal-kernels workflow-and-gotchas.md).
"""
from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx


class QuantSpec(NamedTuple):
    """Frozen quantized head pieces (MLX affine layout; groups along D)."""
    w_q: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int


MatmulChunk = Callable[[int, int], mx.array]   # (v0, v1) -> (N, v1-v0) fp32 logits
WChunk = Callable[[int, int], mx.array]        # (v0, v1) -> (v1-v0, D) hidden-dtype weights

# Engagement counter: every registered vjp increments its key as its FIRST line
# ("dense" / "quantized"). A dropped .vjp registration silently autodiffs through the
# pure-MLX forward with identical gradients — only this counter (or a memory measurement)
# catches that regression. Tested by test_custom_vjp_is_actually_engaged.
VJP_CALLS: dict[str, int] = {}


def streamed_lse_and_target(
    matmul_chunk: MatmulChunk, targets: mx.array, *, v: int, chunk_size: int, n: int,
) -> tuple[mx.array, mx.array]:
    """Stream vocab chunks -> (lse (N,) fp32, target_logit (N,) fp32)."""
    lse = mx.full((n,), float("-inf"), dtype=mx.float32)
    tgt = mx.zeros((n,), dtype=mx.float32)
    for v0 in range(0, v, chunk_size):
        v1 = min(v0 + chunk_size, v)
        logits = matmul_chunk(v0, v1)
        lse = mx.logaddexp(lse, mx.logsumexp(logits, axis=-1))
        in_chunk = (targets >= v0) & (targets < v1)
        idx = mx.clip(targets - v0, 0, v1 - v0 - 1)
        got = mx.take_along_axis(logits, idx[:, None], axis=-1).squeeze(-1)
        tgt = mx.where(in_chunk, got, tgt)
    return lse, tgt


def chunked_backward(
    *,
    hidden: mx.array,
    matmul_chunk: MatmulChunk,
    w_chunk: WChunk,
    targets: mx.array,
    lse: mx.array,
    cotangent: mx.array,
    v: int,
    chunk_size: int,
    head_trainable: bool,
) -> tuple[mx.array, mx.array | None]:
    """Given SAVED lse from forward: d_hidden (hidden dtype) and d_w (or None if frozen).

    fp32 d_hidden accumulator: a low-precision running sum re-rounds after every chunk add
    (the classic mixed-precision accumulation bug) — cast down once at the end.
    """
    n, d = hidden.shape
    ct = cotangent.astype(mx.float32)
    d_hidden32 = mx.zeros((n, d), dtype=mx.float32)
    d_w_chunks: list[mx.array] = []
    for v0 in range(0, v, chunk_size):
        v1 = min(v0 + chunk_size, v)
        logits = matmul_chunk(v0, v1)
        p = mx.exp(logits - lse[:, None])
        # mx.equal (not `==`) keeps this typed as array; array.__eq__ is annotated
        # `-> array | bool` (Python object-equality convention), which mypy can't narrow.
        onehot = mx.equal(mx.arange(v0, v1)[None, :], targets[:, None])
        g32 = (p - onehot.astype(p.dtype)) * ct[:, None]
        wc = w_chunk(v0, v1)
        g = g32.astype(wc.dtype)
        d_hidden32 = d_hidden32 + (g @ wc).astype(mx.float32)
        if head_trainable:
            d_w_chunks.append(g.T @ hidden)
    if not head_trainable:
        return d_hidden32.astype(hidden.dtype), None
    # concatenate transiently holds 2x the (V,D) head-gradient footprint — the one known
    # chunked-backward double-up; counted in the peak accounting.
    return d_hidden32.astype(hidden.dtype), mx.concatenate(d_w_chunks, axis=0)


def make_chunked_dense(chunk_size: int) -> mx.custom_function:
    """Dense-head chunked CE. Head enters as an explicit primal (gradient contract)."""

    @mx.custom_function
    def dense_ce(hidden: mx.array, w: mx.array, targets: mx.array) -> mx.array:
        n = hidden.shape[0]
        v = w.shape[0]

        def mm(v0: int, v1: int) -> mx.array:
            return (hidden @ w[v0:v1].T).astype(mx.float32)

        lse, tgt = streamed_lse_and_target(mm, targets, v=v, chunk_size=chunk_size, n=n)
        return lse - tgt  # per-token NLL (N,)

    @dense_ce.vjp
    def dense_ce_vjp(
        primals: tuple[mx.array, mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,  # noqa: ARG001 — MLX-dictated signature
    ) -> tuple[mx.array, mx.array, mx.array]:
        VJP_CALLS["dense"] = VJP_CALLS.get("dense", 0) + 1
        hidden, w, targets = primals
        n = hidden.shape[0]
        v = w.shape[0]

        def mm(v0: int, v1: int) -> mx.array:
            return (hidden @ w[v0:v1].T).astype(mx.float32)

        # Recompute-in-backward: lse is the only residual we need. (This full extra vocab
        # pass is REMOVABLE overhead — the kernel path caches lse from the forward instead.)
        lse, _ = streamed_lse_and_target(mm, targets, v=v, chunk_size=chunk_size, n=n)
        d_hidden, d_w = chunked_backward(
            hidden=hidden, matmul_chunk=mm, w_chunk=lambda a, b: w[a:b],
            targets=targets, lse=lse, cotangent=cotangent, v=v, chunk_size=chunk_size,
            head_trainable=True,
        )
        assert d_w is not None  # head_trainable=True guarantees this; narrows for mypy
        return d_hidden, d_w, mx.zeros_like(targets)

    return dense_ce


def make_chunked_quantized(chunk_size: int, q: QuantSpec) -> mx.custom_function:
    """Quantized-head chunked CE: d_hidden only (head frozen — no d_head path exists)."""

    @mx.custom_function
    def quant_ce(hidden: mx.array, targets: mx.array) -> mx.array:
        n = hidden.shape[0]
        v = q.w_q.shape[0]

        def mm(v0: int, v1: int) -> mx.array:
            return mx.quantized_matmul(
                hidden, q.w_q[v0:v1], q.scales[v0:v1], q.biases[v0:v1],
                transpose=True, group_size=q.group_size, bits=q.bits,
            ).astype(mx.float32)

        lse, tgt = streamed_lse_and_target(mm, targets, v=v, chunk_size=chunk_size, n=n)
        return lse - tgt

    @quant_ce.vjp
    def quant_ce_vjp(
        primals: tuple[mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,  # noqa: ARG001 — MLX-dictated signature
    ) -> tuple[mx.array, mx.array]:
        VJP_CALLS["quantized"] = VJP_CALLS.get("quantized", 0) + 1
        hidden, targets = primals
        n = hidden.shape[0]
        v = q.w_q.shape[0]

        def mm(v0: int, v1: int) -> mx.array:
            return mx.quantized_matmul(
                hidden, q.w_q[v0:v1], q.scales[v0:v1], q.biases[v0:v1],
                transpose=True, group_size=q.group_size, bits=q.bits,
            ).astype(mx.float32)

        def w_chunk(v0: int, v1: int) -> mx.array:  # chunk-sized dequant only
            return mx.dequantize(
                q.w_q[v0:v1], q.scales[v0:v1], q.biases[v0:v1],
                group_size=q.group_size, bits=q.bits,
            )

        lse, _ = streamed_lse_and_target(mm, targets, v=v, chunk_size=chunk_size, n=n)
        d_hidden, _ = chunked_backward(
            hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk,
            targets=targets, lse=lse, cotangent=cotangent, v=v, chunk_size=chunk_size,
            head_trainable=False,
        )
        return d_hidden, mx.zeros_like(targets)

    return quant_ce
