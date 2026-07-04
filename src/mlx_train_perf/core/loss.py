"""Public loss API: HeadRef seam, impl resolution, custom_function wiring.

Residual contract: the inner custom_function returns (nll_rows, lse, tgt); the vjp reads
lse/tgt from `outputs` (the kernel already produces both — saving them is free); this
public wrapper consumes only nll_rows and applies `reduction` OUTSIDE the custom function.
Aux outputs never leak.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import mlx.core as mx

from mlx_train_perf._compat import check_mlx_verified
from mlx_train_perf.core import chunked as _chunked
from mlx_train_perf.core.kernel import dispatch as _dispatch
from mlx_train_perf.core.kernel import launch as _launch
from mlx_train_perf.core.naive import naive_linear_ce
from mlx_train_perf.errors import LossInputError, UnsupportedHeadError

# The kernel templates the hidden (and, for dense heads, the weight) pointer on one MSL
# type T — fp32 or bf16 only, verified by the parity suites in test_kernel_parity.py /
# test_kernel_quant_parity.py.
_KERNEL_DTYPES = (mx.float32, mx.bfloat16)

_KERNEL_TILE = 8192  # one vocab tile; fixed for the kernel impl (chunk_size is chunked-only)


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseHead:
    weight: mx.array          # (V, D) fp32/bf16; tied embeddings pass the same array
    trainable: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantizedHead:
    w_q: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int = 64
    bits: int = 4             # frozen by definition — no d_head path exists


HeadRef = DenseHead | QuantizedHead


def tied_head(embedding_weight: mx.array, *, trainable: bool = False) -> DenseHead:
    return DenseHead(weight=embedding_weight, trainable=trainable)


@dataclass(frozen=True, slots=True, kw_only=True)
class Resolution:
    impl: Literal["kernel", "chunked", "naive"]
    row_tiles: int | None
    provisional: bool
    reason: str


def _quantized_head_d(head: QuantizedHead) -> int:
    # MLX affine layout packs 32 // bits values per uint32 word (verified for bits=4 —
    # the only width the kernel and this formula need to support — by test_quant_layout.py).
    return head.w_q.shape[-1] * (32 // head.bits)


def _require_kernel_supported(*, head: HeadRef, dtype: mx.Dtype) -> None:
    """Structural gate for the kernel impl: raises UnsupportedHeadError (naming the
    'chunked' alternative) for any head/dtype combination the kernel cannot serve. Runs
    for BOTH `impl='auto'` and an explicit `impl='kernel'` request — the kernel genuinely
    cannot serve an unsupported config either way."""
    if dtype not in _KERNEL_DTYPES:
        raise UnsupportedHeadError(
            f"kernel impl only supports fp32/bf16 hidden dtype (got {dtype}); "
            "use impl='chunked' for other dtypes."
        )
    if isinstance(head, QuantizedHead):
        if head.bits != 4:
            raise UnsupportedHeadError(
                f"kernel impl only supports 4-bit quantized heads (got bits={head.bits}); "
                "use impl='chunked' for other bit widths."
            )
        if head.group_size != 64:
            raise UnsupportedHeadError(
                "kernel impl only supports group_size=64 quantized heads (got "
                f"group_size={head.group_size}); use impl='chunked' for other group sizes."
            )
        d = _quantized_head_d(head)
        if d % 64 != 0:
            raise UnsupportedHeadError(
                "kernel impl requires the quantized head's hidden dim to be a multiple of "
                f"64 (got d={d}); use impl='chunked'."
            )


def resolve_impl(*, head: HeadRef, dtype: mx.Dtype, n: int,
                 impl: str = "auto", allow_unverified_mlx: bool = False) -> Resolution:
    if n <= 0:
        # `select_variant` (below, on the kernel path) has no n<=0 floor of its own —
        # math.log2(0) is a domain error, not a typed one. Guard here so this function is
        # safe to call directly, independent of `linear_cross_entropy`'s own input checks.
        raise LossInputError(f"n must be positive (hidden's row count); got n={n}")
    if impl == "chunked":
        return Resolution(impl="chunked", row_tiles=None, provisional=False,
                          reason="explicit impl='chunked' opt-in")
    if impl == "naive":
        return Resolution(impl="naive", row_tiles=None, provisional=False,
                          reason="explicit impl='naive' opt-in")
    if impl not in ("auto", "kernel"):
        raise LossInputError(
            f"unknown impl {impl!r}; expected one of 'auto', 'kernel', 'chunked', 'naive'"
        )
    # "auto" and an explicit "kernel" request resolve identically: both attempt the
    # kernel and both raise the same typed errors on an unsupported/unverified config.
    # The only thing "auto" is not permitted to do is fall back to chunked SILENTLY on a
    # refusal — and per the checks below, it never falls back at all; the caller sees the
    # typed error either way and opts into impl="chunked" explicitly if that's what it wants.
    _require_kernel_supported(head=head, dtype=dtype)
    check_mlx_verified(allow_unverified=allow_unverified_mlx)
    variant = _dispatch.select_variant(n)
    reason = f"kernel row_tiles={variant.row_tiles} selected for n={n} (bucket {variant.bucket})"
    if variant.provisional:
        reason += " — nearest-measured bucket, provisional"
    return Resolution(impl="kernel", row_tiles=variant.row_tiles,
                      provisional=variant.provisional, reason=reason)


def _validate_inputs(*, hidden: mx.array, head: HeadRef, targets: mx.array) -> None:
    if hidden.ndim not in (2, 3):
        raise LossInputError(
            f"hidden must be 2D (N,D) or 3D (B,S,D), got shape {hidden.shape}"
        )
    leading = hidden.shape[:-1]
    n_total = 1
    for size in leading:
        n_total *= size
    if n_total <= 0:
        # `select_variant` (mlx_train_perf.core.kernel.dispatch) has no n<=0 floor of its
        # own — this must run before resolve_impl ever reaches it.
        raise LossInputError(f"hidden must have at least one row, got leading dims {leading}")
    if targets.shape != leading:
        raise LossInputError(
            f"targets shape {targets.shape} must match hidden's leading dims {leading}"
        )
    d = hidden.shape[-1]
    if isinstance(head, DenseHead):
        if head.weight.shape[-1] != d:
            raise LossInputError(
                f"head.weight's last dim ({head.weight.shape[-1]}) must match hidden's D ({d})"
            )
        if hidden.dtype != head.weight.dtype:
            # The kernel path templates both the hidden and weight pointers on one MSL
            # type T — a mismatch there is an opaque Metal JIT build error, not a clean
            # Python exception. Enforced uniformly across impls so switching impl never
            # changes which inputs are accepted.
            raise LossInputError(
                f"hidden dtype {hidden.dtype} must match dense head weight dtype "
                f"{head.weight.dtype}"
            )
        v = head.weight.shape[0]
    else:
        head_d = _quantized_head_d(head)
        if head_d != d:
            raise LossInputError(
                f"quantized head implies D={head_d} (from w_q shape and bits), which does "
                f"not match hidden's D={d}"
            )
        v = head.w_q.shape[0]
    # Deliberate PER-STEP host sync, not a one-time construction cost: an out-of-range
    # target on the kernel path silently produces a WRONG loss (the id never matches any
    # column, so `tgt` stays 0) instead of raising — no-silent-wrong-results outranks a
    # microsecond-scale sync next to a multi-second train step.
    tmin = int(mx.min(targets).item())
    tmax = int(mx.max(targets).item())
    if tmin < 0 or tmax >= v:
        raise LossInputError(f"targets must satisfy 0 <= t < V={v}; got range [{tmin}, {tmax}]")


def _flatten(hidden: mx.array, targets: mx.array) -> tuple[mx.array, mx.array, tuple[int, ...]]:
    leading = hidden.shape[:-1]
    d = hidden.shape[-1]
    n = 1
    for size in leading:
        n *= size
    return hidden.reshape(n, d), targets.reshape(n), leading


def _reduce(nll: mx.array, *, reduction: str, leading: tuple[int, ...]) -> mx.array:
    if reduction == "none":
        return nll.reshape(leading)
    if reduction == "mean":
        return nll.mean()
    return nll.sum()  # only remaining value once the entry-point has validated `reduction`


def _naive_nll(hidden2: mx.array, head: HeadRef, targets2: mx.array) -> mx.array:
    if isinstance(head, DenseHead):
        return naive_linear_ce(hidden2, head.weight, targets2)
    w_dq = mx.dequantize(head.w_q, head.scales, head.biases,
                         group_size=head.group_size, bits=head.bits)
    return naive_linear_ce(hidden2, w_dq, targets2)


def _chunked_nll(hidden2: mx.array, head: HeadRef, targets2: mx.array, *,
                 chunk_size: int) -> mx.array:
    if isinstance(head, DenseHead):
        dense_fn = cast(Callable[[mx.array, mx.array, mx.array], mx.array],
                       _chunked.make_chunked_dense(chunk_size))
        return dense_fn(hidden2, head.weight, targets2)
    q = _chunked.QuantSpec(w_q=head.w_q, scales=head.scales, biases=head.biases,
                          group_size=head.group_size, bits=head.bits)
    quant_fn = cast(Callable[[mx.array, mx.array], mx.array],
                   _chunked.make_chunked_quantized(chunk_size, q))
    return quant_fn(hidden2, targets2)


def _kernel_nll_dense(hidden2: mx.array, head: DenseHead, targets2: mx.array, *,
                      row_tiles: int) -> mx.array:
    rate = _launch.calibrated_rate(row_tiles=row_tiles, dtype=hidden2.dtype,
                                   n=hidden2.shape[0], d=hidden2.shape[1],
                                   v=head.weight.shape[0])
    tile = _KERNEL_TILE

    if head.trainable:
        @mx.custom_function
        def _ce(hidden: mx.array, w: mx.array,
                targets: mx.array) -> tuple[mx.array, mx.array, mx.array]:
            lse, tgt = _launch.forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                                       rate_macs_per_s=rate)
            return lse - tgt, lse, tgt

        @_ce.vjp
        def _ce_vjp(
            primals: tuple[mx.array, mx.array, mx.array],
            cotangent: tuple[mx.array, mx.array, mx.array],
            outputs: tuple[mx.array, mx.array, mx.array],
        ) -> tuple[mx.array, mx.array, mx.array]:
            hidden, w, targets = primals
            _, lse, _ = outputs
            ct, _, _ = cotangent
            v = w.shape[0]

            def mm(v0: int, v1: int) -> mx.array:
                return (hidden @ w[v0:v1].T).astype(mx.float32)

            d_hidden, d_w = _chunked.chunked_backward(
                hidden=hidden, matmul_chunk=mm, w_chunk=lambda a, b: w[a:b],
                targets=targets, lse=lse, cotangent=ct, v=v, chunk_size=tile,
                head_trainable=True,
            )
            assert d_w is not None  # head_trainable=True guarantees this; narrows for mypy
            return d_hidden, d_w, mx.zeros_like(targets)

        trainable_fn = cast(
            Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]], _ce
        )
        nll, _, _ = trainable_fn(hidden2, head.weight, targets2)
        return nll

    w = head.weight  # captured by closure — constant, no d_head path (verified contract)

    @mx.custom_function
    def _ce_frozen(hidden: mx.array,
                   targets: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        lse, tgt = _launch.forward(hidden, w, targets, row_tiles=row_tiles, tile=tile,
                                   rate_macs_per_s=rate)
        return lse - tgt, lse, tgt

    @_ce_frozen.vjp
    def _ce_frozen_vjp(
        primals: tuple[mx.array, mx.array],
        cotangent: tuple[mx.array, mx.array, mx.array],
        outputs: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        hidden, targets = primals
        _, lse, _ = outputs
        ct, _, _ = cotangent
        v = w.shape[0]

        def mm(v0: int, v1: int) -> mx.array:
            return (hidden @ w[v0:v1].T).astype(mx.float32)

        d_hidden, _ = _chunked.chunked_backward(
            hidden=hidden, matmul_chunk=mm, w_chunk=lambda a, b: w[a:b],
            targets=targets, lse=lse, cotangent=ct, v=v, chunk_size=tile,
            head_trainable=False,
        )
        return d_hidden, mx.zeros_like(targets)

    frozen_fn = cast(
        Callable[[mx.array, mx.array], tuple[mx.array, mx.array, mx.array]], _ce_frozen
    )
    nll, _, _ = frozen_fn(hidden2, targets2)
    return nll


def _kernel_nll_quantized(hidden2: mx.array, head: QuantizedHead, targets2: mx.array, *,
                          row_tiles: int) -> mx.array:
    q = _chunked.QuantSpec(w_q=head.w_q, scales=head.scales, biases=head.biases,
                          group_size=head.group_size, bits=head.bits)
    rate = _launch.calibrated_rate_quantized(row_tiles=row_tiles, dtype=hidden2.dtype,
                                             n=hidden2.shape[0], d=hidden2.shape[1],
                                             v=q.w_q.shape[0])
    tile = _KERNEL_TILE

    @mx.custom_function
    def _ce(hidden: mx.array, targets: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        lse, tgt = _launch.forward_quantized(hidden, q, targets, row_tiles=row_tiles,
                                             tile=tile, rate_macs_per_s=rate)
        return lse - tgt, lse, tgt

    @_ce.vjp
    def _ce_vjp(
        primals: tuple[mx.array, mx.array],
        cotangent: tuple[mx.array, mx.array, mx.array],
        outputs: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        hidden, targets = primals
        _, lse, _ = outputs
        ct, _, _ = cotangent
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

        d_hidden, _ = _chunked.chunked_backward(
            hidden=hidden, matmul_chunk=mm, w_chunk=w_chunk, targets=targets, lse=lse,
            cotangent=ct, v=v, chunk_size=tile, head_trainable=False,
        )
        return d_hidden, mx.zeros_like(targets)

    quant_fn = cast(
        Callable[[mx.array, mx.array], tuple[mx.array, mx.array, mx.array]], _ce
    )
    nll, _, _ = quant_fn(hidden2, targets2)
    return nll


def linear_cross_entropy(
    hidden: mx.array,
    head: HeadRef,
    targets: mx.array,
    *,
    impl: Literal["auto", "kernel", "chunked", "naive"] = "auto",
    chunk_size: int | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
    allow_unverified_mlx: bool = False,
) -> mx.array:
    if reduction not in ("none", "mean", "sum"):
        raise LossInputError(
            f"unknown reduction {reduction!r}; expected 'none', 'mean', or 'sum'"
        )
    _validate_inputs(hidden=hidden, head=head, targets=targets)
    hidden2, targets2, leading = _flatten(hidden, targets)
    n = hidden2.shape[0]
    res = resolve_impl(head=head, dtype=hidden2.dtype, n=n, impl=impl,
                       allow_unverified_mlx=allow_unverified_mlx)
    csize = _KERNEL_TILE if chunk_size is None else chunk_size

    if res.impl == "naive":
        nll = _naive_nll(hidden2, head, targets2)
    elif res.impl == "chunked":
        nll = _chunked_nll(hidden2, head, targets2, chunk_size=csize)
    else:
        assert res.row_tiles is not None  # guaranteed by resolve_impl's "kernel" branch
        if isinstance(head, DenseHead):
            nll = _kernel_nll_dense(hidden2, head, targets2, row_tiles=res.row_tiles)
        else:
            nll = _kernel_nll_quantized(hidden2, head, targets2, row_tiles=res.row_tiles)

    return _reduce(nll, reduction=reduction, leading=leading)
