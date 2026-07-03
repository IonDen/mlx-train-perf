import math

import mlx.core as mx
from mlx import nn

from mlx_train_perf.core.naive import naive_linear_ce


def test_hand_computed_two_rows() -> None:
    # D=2, V=3. logits row0 = [1, 0, -1] (w = eye-ish), target 0 -> nll = lse - 1
    hidden = mx.array([[1.0, 0.0], [0.0, 1.0]])
    w = mx.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
    targets = mx.array([0, 1])
    nll = naive_linear_ce(hidden, w, targets)
    lse0 = math.log(math.exp(1.0) + math.exp(0.0) + math.exp(-1.0))
    lse1 = math.log(3.0)
    assert abs(nll[0].item() - (lse0 - 1.0)) < 1e-6
    assert abs(nll[1].item() - (lse1 - 0.0)) < 1e-6


def test_matches_mlx_builtin_cross_entropy() -> None:
    mx.random.seed(0)
    hidden = mx.random.normal((32, 16))
    w = mx.random.normal((100, 16))
    targets = mx.random.randint(0, 100, (32,))
    ours = naive_linear_ce(hidden, w, targets)
    theirs = nn.losses.cross_entropy(hidden @ w.T, targets, reduction="none")
    assert mx.abs(ours - theirs).max().item() < 1e-5


def test_output_is_fp32_even_for_bf16_inputs() -> None:
    hidden = mx.random.normal((4, 8)).astype(mx.bfloat16)
    w = mx.random.normal((10, 8)).astype(mx.bfloat16)
    nll = naive_linear_ce(hidden, w, mx.array([0, 1, 2, 3]))
    assert nll.dtype == mx.float32
