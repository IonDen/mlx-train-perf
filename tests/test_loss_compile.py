"""`linear_cross_entropy` must run inside `mx.compile` for the training path.

`mlx_lm.tuner.trainer.train()` wraps its per-step function in `mx.compile`, and MLX
forbids evaluating a traced array during a compile trace. The default `validate_targets`
path does exactly that (`mx.min/max(targets).item()`, the out-of-range safety check), so
it cannot run under compile. `validate_targets=False` (the training/trusted path the
mlx-lm adapter uses) skips ONLY that host sync — the loss math is unchanged — so it
traces cleanly. These tests lock both properties: compile-compatibility of the skipped
path, and that the default path still catches out-of-range targets.
"""
import mlx.core as mx
import pytest

from mlx_train_perf import DenseHead, linear_cross_entropy
from mlx_train_perf.errors import LossInputError


def _fixture(n: int = 64, d: int = 64, v: int = 1024):
    mx.random.seed(0)
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    return hidden, DenseHead(weight=w, trainable=False), targets


@pytest.mark.parametrize("impl", ["chunked", "naive"])
def test_loss_runs_under_compile_with_validation_skipped(impl: str) -> None:
    hidden, head, targets = _fixture()

    @mx.compile
    def step(h: mx.array, t: mx.array) -> mx.array:
        return linear_cross_entropy(h, head, t, impl=impl, reduction="mean",
                                    validate_targets=False)

    compiled = step(hidden, targets)
    uncompiled = linear_cross_entropy(hidden, head, targets, impl=impl, reduction="mean",
                                      validate_targets=False)
    assert mx.abs(compiled - uncompiled).item() < 1e-6


@pytest.mark.metal
def test_kernel_loss_runs_under_compile_with_validation_skipped() -> None:
    hidden, head, targets = _fixture()

    @mx.compile
    def step(h: mx.array, t: mx.array) -> mx.array:
        return linear_cross_entropy(h, head, t, impl="kernel", reduction="mean",
                                    validate_targets=False)

    compiled = step(hidden, targets)
    uncompiled = linear_cross_entropy(hidden, head, targets, impl="kernel",
                                      reduction="mean", validate_targets=False)
    assert mx.abs(compiled - uncompiled).item() < 1e-6


@pytest.mark.parametrize("impl", ["chunked", "naive"])
def test_default_validation_path_breaks_under_compile(impl: str) -> None:
    """The DEFAULT (validate_targets=True) path host-syncs and cannot be compiled — this
    is the exact incompatibility the training path routes around; pin it so a future
    change that silently removed the host sync (or the flag) is noticed."""
    hidden, head, targets = _fixture()

    @mx.compile
    def step(h: mx.array, t: mx.array) -> mx.array:
        return linear_cross_entropy(h, head, t, impl=impl, reduction="mean")

    with pytest.raises(ValueError, match="eval an array during"):
        step(hidden, targets)


def test_validate_targets_true_still_catches_out_of_range() -> None:
    hidden, head, targets = _fixture()
    targets[0] = 9999  # >= V=1024
    with pytest.raises(LossInputError, match="0 <= t < V"):
        linear_cross_entropy(hidden, head, targets, impl="naive", reduction="mean")


def test_validate_targets_false_skips_the_range_check() -> None:
    """Trusted-mode contract: the range check is skipped, so an out-of-range target does
    NOT raise (the caller — the mlx-lm adapter — guarantees in-range tokenizer ids)."""
    hidden, head, targets = _fixture()
    targets[0] = 9999
    # No LossInputError about the range; must not raise from validation.
    out = linear_cross_entropy(hidden, head, targets, impl="naive", reduction="mean",
                               validate_targets=False)
    assert out.shape == ()


def test_validate_targets_is_numerically_inert_on_valid_input() -> None:
    hidden, head, targets = _fixture()
    checked = linear_cross_entropy(hidden, head, targets, impl="naive", reduction="mean",
                                   validate_targets=True)
    skipped = linear_cross_entropy(hidden, head, targets, impl="naive", reduction="mean",
                                   validate_targets=False)
    assert mx.abs(checked - skipped).item() == 0.0


def test_non_syncing_validation_still_runs_when_targets_skipped() -> None:
    """validate_targets=False skips ONLY the range host-sync — the cheap structural checks
    (dtype, shape) still fire."""
    hidden, head, _ = _fixture()
    float_targets = mx.zeros((64,), dtype=mx.float32)
    with pytest.raises(LossInputError, match="integer dtype"):
        linear_cross_entropy(hidden, head, float_targets, impl="naive",
                             validate_targets=False)
