"""Direct input-validation tests for `chunked_gated_delta` -- distinct from the parity
suites, which only ever call it with valid shapes.
"""

import mlx.core as mx
import pytest

from mlx_train_perf.errors import RecurrentInputError
from mlx_train_perf.recurrent.ops import chunked_gated_delta


def test_hv_not_a_multiple_of_hk_refuses():
    # Catches: Hv < Hk (or any Hv not a multiple of Hk) silently reaching
    # `repeat_factor = Hv // Hk == 0`, which skips the GQA broadcast entirely and fails
    # later on an obscure shape mismatch deep inside the chunk body instead of naming
    # the real problem at the call boundary. Upstream's own `gated_delta_ops` validates
    # this at construction; a direct-API caller of this op has no such guard.
    b, t, hk, hv, dk, dv = 1, 8, 4, 2, 8, 8
    mx.random.seed(0)
    q = mx.random.normal(shape=(b, t, hk, dk))
    k = mx.random.normal(shape=(b, t, hk, dk))
    v = mx.random.normal(shape=(b, t, hv, dv))
    g = mx.sigmoid(mx.random.normal(shape=(b, t, hv)))
    beta = mx.sigmoid(mx.random.normal(shape=(b, t, hv)))
    mx.eval(q, k, v, g, beta)

    with pytest.raises(RecurrentInputError, match="Hv"):
        chunked_gated_delta(q, k, v, g, beta, chunk_size=4)
