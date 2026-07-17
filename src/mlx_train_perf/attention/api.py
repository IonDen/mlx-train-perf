"""`flash_attention` public API.

The full custom_function boundary. The reference forward routes through
`flash_attention_reference`; its backward is a hand-written pure-MLX vjp (the
FlashAttention paper's backward equations, over full un-tiled tensors). The Metal kernel
swaps in underneath this surface without changing it.

Residual contract: the inner `mx.custom_function`-decorated core returns
`(O, L)` -- `L` is a real tuple OUTPUT, never a closure-captured stash (closure arrays
are CONSTANTS under compile/checkpoint recompute). The public `flash_attention` does
`o, _ = core_fn(...)`, so L's cotangent is always zero and L never leaks to callers;
the vjp reads `O, L` from `outputs` and `q, k, v` from `primals`.

`impl='reference'` is an ORACLE, never a production path: its backward materializes
the full `(N, N)` score/probability matrices, reintroducing the exact O(N^2) peak this
feature exists to remove. Every caller (here and in tests) must fence it to
tiny N -- never run it at flagship context.
"""
from collections.abc import Callable
from typing import Literal, cast

import mlx.core as mx

from mlx_train_perf._compat import check_mlx_verified
from mlx_train_perf.attention.kernel.dispatch import select_bwd_tiles, select_fwd_tile
from mlx_train_perf.attention.kernel.launch import (
    TileShape,
    calibrated_bwd_dkv_rate,
    calibrated_bwd_dq_rate,
    calibrated_fwd_rate,
    launch_bwd_D,
    launch_bwd_dkv,
    launch_bwd_dq,
    launch_flash_fwd,
)
from mlx_train_perf.attention.reference import flash_attention_reference, kv_head_for
from mlx_train_perf.attention.segments import PackedMask, segment_allowed
from mlx_train_perf.errors import AttentionInputError, UnsupportedAttentionError

# The kernel (T5+) will template on one MSL type T -- fp32 or bf16 only, mirroring the
# loss-layer kernel's supported dtypes (core/loss.py::_KERNEL_DTYPES).
_KERNEL_DTYPES = (mx.float32, mx.bfloat16)
_KERNEL_HEAD_DIMS = (64, 96, 128)

# Engagement sentinel (0.1.0 chunked-vjp precedent, core/chunked.py::VJP_CALLS): proves
# the hand vjp actually fired, since flash_attention_reference is plain differentiable
# MLX ops -- a dropped .vjp registration would autodiff through it to IDENTICAL values.
VJP_CALLS: dict[str, int] = {}


def _validate_shapes(q: mx.array, k: mx.array, v: mx.array) -> None:
    for name, arr in (("q", q), ("k", k), ("v", v)):
        if arr.ndim != 4:
            raise AttentionInputError(
                f"{name} must be 4-D (B, H, N, D); got shape {arr.shape}"
            )
    b_q, hq, n_q, d_q = q.shape
    b_k, hkv, n_k, d_k = k.shape
    b_v, hv, n_v, d_v = v.shape
    if b_q < 1:
        raise AttentionInputError(f"batch must be >= 1; got {b_q}")
    if not (b_q == b_k == b_v):
        raise AttentionInputError(f"batch mismatch: q={b_q}, k={b_k}, v={b_v}")
    if hkv != hv:
        raise AttentionInputError(f"k/v head-count mismatch: k has {hkv} heads, v has {hv}")
    if hq % hkv != 0:
        raise AttentionInputError(f"Hq={hq} must be a multiple of Hkv={hkv} for GQA grouping")
    if not (n_q == n_k == n_v):
        raise AttentionInputError(f"sequence length mismatch: q={n_q}, k={n_k}, v={n_v}")
    if not (d_q == d_k == d_v):
        raise AttentionInputError(f"head_dim mismatch: q={d_q}, k={d_k}, v={d_v}")


def resolve_attention_impl(
    q: mx.array, k: mx.array, v: mx.array, *, impl: str, causal: bool = True
) -> str:
    """"reference" is always allowed (the oracle). "auto"/"kernel" run the full
    kernel-support gate (dtype, head_dim, causal, Metal availability, mlx-verified) and,
    when every gate passes, resolve to "kernel". An
    unsupported dtype/head_dim/causal/device raises `UnsupportedAttentionError` with a
    pointer to impl="reference"; there is no silent fallback."""
    _validate_shapes(q, k, v)
    if impl == "reference":
        return "reference"
    if impl not in ("auto", "kernel"):
        raise AttentionInputError(
            f"unknown impl {impl!r}; expected one of 'auto', 'kernel', 'reference'"
        )

    dtype = q.dtype
    if dtype not in _KERNEL_DTYPES:
        raise UnsupportedAttentionError(
            f"kernel impl only supports fp32/bf16 (got {dtype}); use impl='reference'."
        )
    head_dim = q.shape[-1]
    if head_dim not in _KERNEL_HEAD_DIMS:
        raise UnsupportedAttentionError(
            f"kernel impl only supports head_dim in {_KERNEL_HEAD_DIMS} (got {head_dim}); "
            "use impl='reference'."
        )
    if not causal:
        raise UnsupportedAttentionError(
            "kernel impl only supports causal=True (got causal=False); use impl='reference'."
        )
    if not mx.metal.is_available():
        raise UnsupportedAttentionError(
            "kernel impl requires Metal (no GPU device available); use impl='reference'."
        )
    check_mlx_verified(allow_unverified=False)

    return "kernel"


def _bwd_D(d_o: mx.array, o: mx.array) -> mx.array:  # noqa: N802 -- D is the paper's name
    """`D_i = sum_d dO_i,d * O_i,d`, fp32 -- the flash-attention paper's row-correction
    term for `dS`. Both inputs upcast to fp32 internally regardless of dtype."""
    return (d_o.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)


def _kv_gather(x: mx.array, *, hq: int, hkv: int) -> mx.array:
    """Gathers K or V per q-head via the pinned contiguous GQA convention
    (`kv_head_for`), matching flash_attention_reference's own forward gather."""
    group_size = hq // hkv
    head_idx = mx.array([kv_head_for(h, group_size) for h in range(hq)])
    return mx.take(x, head_idx, axis=1)  # (B, Hq, N, D)


def _flash_attention_backward(
    q: mx.array, k: mx.array, v: mx.array, o: mx.array, lse: mx.array, d_o: mx.array,
    *, scale: float, causal: bool, segments: PackedMask | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Hand-written flash-attention backward, pure MLX over FULL (un-tiled) tensors.
    This is the oracle path (impl='reference'): materializing S and P is BY DESIGN,
    fenced to tiny N by every caller. All arithmetic runs in fp32; results are cast to
    the input dtypes at the end.

    `D = (dO*O).sum(-1)`; `S = scale * q @ k^T` (causal-masked, matching the forward);
    `P = exp(S - L)`; `dV = P^T @ dO` (accumulated per kv head over its q-head group);
    `dP = dO @ v^T`; `dS = P * (dP - D)`; `dQ = scale * dS @ k`;
    `dK = scale * dS^T @ q` (grouped like dV).

    `segments` (requires `causal=True`) masks S with `segment_allowed`, the SAME helper
    the forward oracle uses, so forward and backward see identical masking -- see
    `mlx_train_perf.attention.segments.PackedMask`.
    """
    b = q.shape[0]
    hq, hkv = q.shape[1], k.shape[1]
    n = q.shape[2]
    d_dim = q.shape[-1]
    group_size = hq // hkv

    q32 = q.astype(mx.float32)
    k_g = _kv_gather(k, hq=hq, hkv=hkv).astype(mx.float32)
    v_g = _kv_gather(v, hq=hq, hkv=hkv).astype(mx.float32)
    o32 = o.astype(mx.float32)
    do32 = d_o.astype(mx.float32)
    lse32 = lse.astype(mx.float32)

    d = _bwd_D(do32, o32)  # (B, Hq, N)

    s = (q32 @ k_g.swapaxes(-1, -2)) * scale  # (B, Hq, N, N)
    if segments is not None:
        assert causal, "segments requires causal=True"
        s = mx.where(segment_allowed(segments.seg_id), s, -mx.inf)
    elif causal:
        neg_inf = mx.full((n, n), -mx.inf, dtype=mx.float32)
        s = s + mx.triu(neg_inf, k=1)  # additive -inf where j > i, matching the forward
    p = mx.exp(s - lse32[..., None])  # (B, Hq, N, N)

    d_p = do32 @ v_g.swapaxes(-1, -2)  # (B, Hq, N, N)
    d_s = p * (d_p - d[..., None])  # (B, Hq, N, N)

    d_q = scale * (d_s @ k_g)  # (B, Hq, N, D)
    d_v_per_head = p.swapaxes(-1, -2) @ do32  # (B, Hq, N, D)
    d_k_per_head = scale * (d_s.swapaxes(-1, -2) @ q32)  # (B, Hq, N, D)

    # accumulate per kv head over its query-head group (contiguous: head h belongs to
    # kv head h // group_size, so a (B, Hkv, group, N, D) reshape + sum(axis=2) is exact).
    d_v = d_v_per_head.reshape(b, hkv, group_size, n, d_dim).sum(axis=2)
    d_k = d_k_per_head.reshape(b, hkv, group_size, n, d_dim).sum(axis=2)

    return d_q.astype(q.dtype), d_k.astype(k.dtype), d_v.astype(v.dtype)


def flash_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool = True,
    impl: Literal["auto", "kernel", "reference"] = "auto",
) -> mx.array:
    """`softmax(scale * Q K^T + causal_mask) @ V`, GQA via the pinned contiguous
    convention. `impl='auto'`/`'kernel'` route the FORWARD through the Metal kernel
    (O + L, query-range split) AND the BACKWARD through the fully kernel-backed vjp
    (D + dQ + chained dK/dV) -- the (N, N) score matrix is never materialized on
    either pass; the two per-kernel backward split rates are calibrated at construction and
    closure-captured.
    `impl='reference'` is the oracle (never a production path): its backward materializes the
    full (N, N) score/probability matrices, so every caller must fence it to tiny N."""
    resolved = resolve_attention_impl(q, k, v, impl=impl, causal=causal)

    if resolved == "kernel":
        # Select the forward tile AND the two backward tiles (dQ, dK/dV) from the measured
        # dispatch tables (see attention/kernel/dispatch.py) OUTSIDE the custom_function -- a
        # fixed TileShape() (always scalar) is no longer used on the kernel path. Calibrate the
        # forward query-split rate and BOTH per-kernel backward split rates OUTSIDE the
        # custom_function (host-sync timing is compile-hostile) and close over them -- cached per
        # occupancy regime, so a compiled caller re-probes only on the first trace. The backward
        # rates are split PER KERNEL (T9b Step 3): the dQ and dK/dV throughputs differ measurably
        # (2.35x at scalar), so each split is sized by its own kernel's measured rate -- no
        # cross-kernel assumption. Rates are never measured inside the vjp.
        tile = select_fwd_tile(q.shape[2], q.shape[-1])
        rate = calibrated_fwd_rate(
            head_dim=q.shape[-1], dtype=q.dtype, b=q.shape[0], hq=q.shape[1],
            hkv=k.shape[1], n=q.shape[2], causal=causal, tile=tile,
        )
        dq_tile, dkv_tile = select_bwd_tiles(q.shape[2], q.shape[-1])
        dq_rate: float | None = calibrated_bwd_dq_rate(
            head_dim=q.shape[-1], dtype=q.dtype, b=q.shape[0], hq=q.shape[1],
            hkv=k.shape[1], n=q.shape[2], causal=causal, tile=dq_tile,
        )
        dkv_rate: float | None = calibrated_bwd_dkv_rate(
            head_dim=q.shape[-1], dtype=q.dtype, b=q.shape[0], hq=q.shape[1],
            hkv=k.shape[1], n=q.shape[2], causal=causal, tile=dkv_tile,
        )
    else:
        tile = TileShape()
        rate = None
        dq_tile = dkv_tile = TileShape()
        dq_rate = None
        dkv_rate = None

    @mx.custom_function
    def _core(q_: mx.array, k_: mx.array, v_: mx.array) -> tuple[mx.array, mx.array]:
        if resolved == "kernel":
            return launch_flash_fwd(
                q_, k_, v_, scale=scale, causal=causal, tile=tile,
                rate_macs_per_s=rate,
            )
        return flash_attention_reference(q_, k_, v_, scale=scale, causal=causal)

    @_core.vjp
    def _core_vjp(
        primals: tuple[mx.array, mx.array, mx.array],
        cotangents: tuple[mx.array, mx.array],
        outputs: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array, mx.array]:
        VJP_CALLS["flash_attention"] = VJP_CALLS.get("flash_attention", 0) + 1
        q_, k_, v_ = primals
        d_o, _d_l = cotangents  # L's cotangent is always zero -- see the module docstring
        o_, lse_ = outputs
        if resolved == "kernel":
            # Fully kernel-backed backward (T7 D + T8 dQ + T9 chained dK/dV) with the
            # construction-time `bwd_rate` closure-captured -- NO host-sync in the vjp path
            # (calibration ran once outside the custom_function). The counter proves this
            # kernel branch fired vs the pure-MLX oracle (value parity can't tell them apart).
            VJP_CALLS["flash_attention_kernel_bwd"] = (
                VJP_CALLS.get("flash_attention_kernel_bwd", 0) + 1
            )
            d_arr = launch_bwd_D(d_o, o_)
            d_q = launch_bwd_dq(
                q_, k_, v_, d_o, lse_, d_arr, scale=scale, causal=causal,
                rate_macs_per_s=dq_rate, variant=dq_tile.variant, d_slab=dq_tile.d_slab,
            )
            d_k, d_v = launch_bwd_dkv(
                q_, k_, v_, d_o, lse_, d_arr, scale=scale, causal=causal,
                rate_macs_per_s=dkv_rate, variant=dkv_tile.variant, d_slab=dkv_tile.d_slab,
            )
            return d_q, d_k, d_v
        return _flash_attention_backward(q_, k_, v_, o_, lse_, d_o, scale=scale, causal=causal)

    core_fn = cast(
        Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array]], _core
    )
    o, _ = core_fn(q, k, v)
    return o
