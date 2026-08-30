import mlx.core as mx
import numpy as np

from mlx_train_perf.recurrent.ops import _solve_strict_lower


def _dense_solve(A: mx.array, b: mx.array) -> mx.array:  # noqa: N803
    ident = np.eye(A.shape[-1], dtype=np.float32)
    return mx.array(np.linalg.solve(ident - np.asarray(A), np.asarray(b)))


def test_solve_matches_dense_inverse():
    # Catches: wrong triangular orientation or an off-by-one in the block walk.
    mx.random.seed(0)
    A = mx.tril(mx.random.normal((48, 48)) * 0.3, k=-1)  # noqa: N806
    b = mx.random.normal((48, 8))
    x = _solve_strict_lower(A, b)
    mx.eval(x)
    # measured 4.05e-6 (fp32 accumulation floor ~1e-6); pin 1e-4.
    assert float(mx.abs(x - _dense_solve(A, b)).max()) < 1e-4


def test_solve_beats_global_doubling_on_the_collinear_gram():
    # Catches: dropping the sub-blocking (a global Neumann/doubling expansion).
    # This IS the matrix the chunk builds at repeat_keys + beta->1 + g->1.
    mx.random.seed(0)
    A = -0.999 * mx.tril(mx.ones((64, 64), dtype=mx.float32), k=-1)  # noqa: N806
    b = mx.random.normal((64, 8))
    ref = _dense_solve(A, b)
    x = _solve_strict_lower(A, b)               # sub-blocked
    x_global = _solve_strict_lower(A, b, sb=64)  # the bug this guards against
    mx.eval(x, x_global)
    assert float(mx.abs(x - ref).max()) < 1e-2        # measured 3.9e-3
    assert float(mx.abs(x_global - ref).max()) > 1e3  # measured 1.4e+11
