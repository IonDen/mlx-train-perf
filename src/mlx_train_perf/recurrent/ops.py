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

import mlx.core as mx  # noqa: I001


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
