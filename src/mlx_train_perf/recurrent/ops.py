"""Recurrent / GatedDelta blocked solver core.

Blocked forward-substitution for (I - A)x = b where A is strictly-lower-triangular
(nilpotent). Used by the GatedDelta chunk kernel to solve recurrent states without
materializing high-rank intermediate products.

Computation follows the Qwen 3.5 GDN block structure (hidden x linear keys/values);
a later task chains this solver with the chunk kernel.

**Dtype protocol:** computations inherit the input dtype (float32 or bfloat16).
Accumulation error growth is bounded by sub-blocking (SUB_BLOCK=16) within each chunk.

**Closure invariant:** A is guaranteed strictly lower triangular (`A[i,j] == 0 for i <= j`).
The algorithm is nilpotent (successive doublings consume the strict-lower structure),
so (I - A)^{-1} = I + A + A^2 + ... is a finite Neumann series.

This implementation is adapted from mlx-lm PR #1389 (tsato081, head 6fc3a29, closed
unmerged 2026-08-21), released under MIT License.
"""

import mlx.core as mx

from mlx_train_perf.errors import RecurrentInputError

SUB_BLOCK = 16  # bounds fp32 error growth when keys repeat within a chunk


def _solve_strict_lower(A: mx.array, b: mx.array, sb: int = SUB_BLOCK) -> mx.array:  # noqa: N803
    """Solve (I - A) x = b for strictly lower-triangular (nilpotent) A.

    Args:
        A: Strictly lower-triangular matrix of shape `[..., C, C]`.
        b: Right-hand side of shape `[..., C, D]`.
        sb: Sub-block size for error control. Default SUB_BLOCK=16.

    Returns:
        Solution x of shape `[..., C, D]`.
    """
    C = A.shape[-1]  # noqa: N806

    def doubling(Aii: mx.array, rhs: mx.array, n: int) -> mx.array:  # noqa: N803
        x = rhs
        if n <= 1:
            return x
        P = Aii  # noqa: N806
        steps = (n - 1).bit_length()
        for s in range(steps):
            x = x + P @ x
            if s != steps - 1:
                P = P @ P  # noqa: N806
        return x

    if sb >= C:
        return doubling(A, b, C)
    nb = (C + sb - 1) // sb
    blocks: list[mx.array] = []
    for i in range(nb):
        lo, hi = i * sb, min((i + 1) * sb, C)
        rhs = b[..., lo:hi, :]
        if i > 0:
            prev = blocks[0] if i == 1 else mx.concatenate(blocks, axis=-2)
            rhs = rhs + A[..., lo:hi, :lo] @ prev
        blocks.append(doubling(A[..., lo:hi, lo:hi], rhs, hi - lo))
    return mx.concatenate(blocks, axis=-2)


CHUNK_SIZE = 64


def _gated_delta_chunk(
    state: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    repeat_factor: int = 1,
) -> tuple[mx.array, mx.array]:
    """One C-timestep chunk as a triangular solve (gated UT/WY transform).

    Exact reformulation of the sequential recurrence; internals run fp32.
    Chunk-local shapes: state [B,Hv,Dk,Dv], q/k [B,Hk,C,Dk], v [B,Hv,C,Dv],
    g/beta [B,Hv,C]. Under GQA the CxC Gram and qk^T products need only the
    Hk heads, so they are formed before the broadcast to Hv.

    INVARIANT: every differentiable array is a POSITIONAL argument — a
    closed-over array silently receives a ZERO gradient under mx.checkpoint
    (verified on mlx 0.32.0).
    """
    C, Hk, Hv = q.shape[2], q.shape[1], v.shape[1]  # noqa: N806
    orig_dtype = q.dtype
    q, k, v, g, beta, state = (t.astype(mx.float32)
                               for t in (q, k, v, g, beta, state))

    # KNOWN divergence (named test pins it): the clamp keeps -inf out of the
    # cumsum and zeroes dL/dg for g <= 1e-12.
    g_cumlog = mx.cumsum(mx.log(mx.maximum(g, 1e-12)), axis=-1)
    g_last = g_cumlog[..., -1:]

    # Zero the upper triangle BEFORE exp: it overflows, and inf * 0 = NaN.
    tril_ones = mx.tril(mx.ones((C, C), dtype=mx.float32), k=0)
    L_mask = mx.exp((g_cumlog[..., :, None] - g_cumlog[..., None, :]) * tril_ones) \
        * tril_ones  # noqa: N806

    kkT = k @ mx.swapaxes(k, -1, -2)  # noqa: N806
    qkT = q @ mx.swapaxes(k, -1, -2)  # noqa: N806
    if repeat_factor > 1:
        B = q.shape[0]  # noqa: N806

        def _to_hv(x: mx.array) -> mx.array:
            D = x.shape[-1]  # noqa: N806
            return mx.broadcast_to(
                x[:, :, None], (B, Hk, repeat_factor, C, D)).reshape(B, Hv, C, D)

        kkT, qkT, q, k = _to_hv(kkT), _to_hv(qkT), _to_hv(q), _to_hv(k)  # noqa: N806

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    strict_lower = mx.tril(mx.ones((C, C), dtype=mx.float32), k=-1)
    A = -(beta[..., :, None] * kkT) * L_mask * strict_lower  # noqa: N806

    decay_exp = mx.exp(g_cumlog)[..., None]
    sol = _solve_strict_lower(
        A, mx.concatenate([v_beta, k_beta * decay_exp], axis=-1))
    v_corrected, k_cumdecay = mx.split(sol, [v.shape[-1]], axis=-1)

    v_new = v_corrected - k_cumdecay @ state
    y = (q * decay_exp) @ state + (qkT * L_mask) @ v_new

    decay_to_end = mx.exp(g_last - g_cumlog)[..., None]
    new_state = state * mx.exp(g_last)[..., None] \
        + mx.swapaxes(k * decay_to_end, -1, -2) @ v_new
    # State stays fp32 across chunks; casting at boundaries drifts.
    return y.astype(orig_dtype), new_state


_chunk_ckpt = mx.checkpoint(_gated_delta_chunk)


def chunked_gated_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    chunk_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Chunk-parallel GatedDelta forward — the mlx#4020 swap point.

    Argument-compatible with mlx-lm's ``gated_delta_ops``: ``q, k`` are
    ``[B, T, Hk, Dk]``, ``v`` is ``[B, T, Hv, Dv]``, ``g, beta`` are
    ``[B, T, Hv]``, ``state`` is ``[B, Hv, Dv, Dk] | None``, ``mask`` is
    ``[B, T] bool | None``. Returns ``y: [B, T, Hv, Dv]``, ``state:
    [B, Hv, Dv, Dk]``.

    Argument-compatible means shapes and dtypes only. At masked positions
    (``mask`` False), ``y`` is unspecified and differs from
    ``gated_delta_ops``'s own output there by design -- ``state`` and every
    valid (unmasked) position's ``y`` match. Callers must mask their own loss
    rather than rely on a masked position's output.
    """
    if g.ndim != 3:
        raise RecurrentInputError(
            f"scalar gating only (g.ndim == 3); got {g.ndim}. Vectorized "
            "gating stays on the sequential path."
        )
    B, T, Hk, Dk = q.shape  # noqa: N806
    Hv, Dv = v.shape[-2:]  # noqa: N806
    if Hv % Hk != 0:
        raise RecurrentInputError(
            f"v's head count Hv={Hv} must be a multiple of q/k's head count Hk={Hk} "
            "(grouped-query broadcast requires an integer repeat factor); Hv < Hk would "
            "otherwise floor to a zero repeat factor and skip the broadcast entirely, "
            "failing later on an unrelated shape mismatch"
        )
    C = chunk_size or CHUNK_SIZE  # noqa: N806
    repeat_factor = Hv // Hk
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # Masked steps become IDENTITY steps: g <- 1, beta <- 0. (Shape-only
    # branches; never a value-dependent Python branch.)
    if mask is not None:
        m = mask[..., None]
        g = mx.where(m, g, mx.ones_like(g))
        beta = beta * m

    # Pad T to a multiple of C with identity steps: q/k/v <- ZEROS,
    # g <- ONES, beta <- ZEROS.
    pad_len = (-T) % C
    if pad_len > 0:
        pad4 = [(0, 0), (0, pad_len), (0, 0), (0, 0)]
        q, k, v = (mx.pad(t, pad4) for t in (q, k, v))
        g = mx.concatenate([g, mx.ones((B, pad_len, Hv), dtype=g.dtype)], axis=1)
        beta = mx.pad(beta, [(0, 0), (0, pad_len), (0, 0)])

    nc = (T + pad_len) // C
    q = mx.swapaxes(q, 1, 2).reshape(B, Hk, nc, C, Dk)
    k = mx.swapaxes(k, 1, 2).reshape(B, Hk, nc, C, Dk)
    v = mx.swapaxes(v, 1, 2).reshape(B, Hv, nc, C, Dv)
    g = mx.swapaxes(g, 1, 2).reshape(B, Hv, nc, C)
    beta = mx.swapaxes(beta, 1, 2).reshape(B, Hv, nc, C)

    st = mx.swapaxes(state.astype(mx.float32), -1, -2)  # [B,Hv,Dv,Dk] -> [B,Hv,Dk,Dv]
    ys = []
    for ci in range(nc):  # no mx.eval in this loop (gotcha 2)
        y_c, st = _chunk_ckpt(st, q[:, :, ci], k[:, :, ci], v[:, :, ci],
                              g[:, :, ci], beta[:, :, ci], repeat_factor)
        ys.append(y_c)
    y = mx.concatenate(ys, axis=2)
    if pad_len > 0:
        y = y[:, :, :T, :]
    return mx.swapaxes(y, 1, 2), mx.swapaxes(st, -1, -2)
