"""Full-materialization reference loss — the memory spike the library exists to remove.

Test oracle only. The fp32-exact reference (what parity gates are pinned against) is
this function called on pre-upcast fp32 inputs.
"""
import mlx.core as mx


def naive_linear_ce(hidden: mx.array, w: mx.array, targets: mx.array) -> mx.array:
    """(N,D) x (V,D) -> per-token NLL (N,), float32. Materializes full (N,V) logits."""
    logits = (hidden @ w.T).astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
    return lse - tgt
