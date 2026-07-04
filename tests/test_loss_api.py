import mlx.core as mx
import pytest

from mlx_train_perf import (
    DenseHead,
    QuantizedHead,
    linear_cross_entropy,
    resolve_impl,
    tied_head,
)
from mlx_train_perf.core.naive import naive_linear_ce
from mlx_train_perf.errors import LossInputError, UnsupportedHeadError, UnverifiedMlxError


def _dense(n: int = 32, d: int = 16, v: int = 100) -> tuple[mx.array, DenseHead, mx.array]:
    mx.random.seed(21)
    hidden = mx.random.normal((n, d))
    head = DenseHead(weight=mx.random.normal((v, d)) * 0.05)
    targets = mx.random.randint(0, v, (n,))
    return hidden, head, targets


def test_auto_resolution_is_inspectable_and_kernel_on_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    _, head, _ = _dense()
    r = resolve_impl(head=head, dtype=mx.float32, n=512)
    assert r.impl == "kernel" and r.row_tiles == 2 and "512" in r.reason  # noqa: PT018


def test_auto_refuses_unverified_mlx_no_silent_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.99.0")
    _, head, _ = _dense()
    with pytest.raises(UnverifiedMlxError):
        resolve_impl(head=head, dtype=mx.float32, n=512)
    r = resolve_impl(head=head, dtype=mx.float32, n=512, impl="chunked")  # explicit opt-in OK
    assert r.impl == "chunked"


def test_auto_refuses_unsupported_quant_config() -> None:
    q = QuantizedHead(w_q=mx.zeros((8, 8), dtype=mx.uint32), scales=mx.ones((8, 1)),
                      biases=mx.zeros((8, 1)), group_size=32, bits=4)  # gs32 unsupported
    with pytest.raises(UnsupportedHeadError) as ei:
        resolve_impl(head=q, dtype=mx.bfloat16, n=512)
    assert "chunked" in str(ei.value)      # error names the explicit alternative


def test_kernel_rejects_unsupported_hidden_dtype() -> None:
    _, head, _ = _dense()
    with pytest.raises(UnsupportedHeadError) as ei:
        resolve_impl(head=head, dtype=mx.float16, n=512)
    assert "chunked" in str(ei.value)


def test_kernel_rejects_quantized_bits_other_than_4() -> None:
    q = QuantizedHead(w_q=mx.zeros((8, 8), dtype=mx.uint32), scales=mx.ones((8, 1)),
                      biases=mx.zeros((8, 1)), group_size=64, bits=8)  # bits=8 unsupported
    with pytest.raises(UnsupportedHeadError) as ei:
        resolve_impl(head=q, dtype=mx.bfloat16, n=512)
    assert "chunked" in str(ei.value)


def test_kernel_rejects_quantized_d_not_multiple_of_64() -> None:
    # w_q shape (8, 12) -> d = 12 * (32 // 4) = 96, not a multiple of 64
    q = QuantizedHead(w_q=mx.zeros((8, 12), dtype=mx.uint32), scales=mx.ones((8, 2)),
                      biases=mx.zeros((8, 2)), group_size=64, bits=4)
    with pytest.raises(UnsupportedHeadError) as ei:
        resolve_impl(head=q, dtype=mx.bfloat16, n=512)
    assert "chunked" in str(ei.value)


def test_chunked_impl_value_and_reductions_match_naive() -> None:
    hidden, head, targets = _dense()
    ref = naive_linear_ce(hidden, head.weight, targets)
    for red, expect in (("none", ref), ("mean", ref.mean()), ("sum", ref.sum())):
        got = linear_cross_entropy(hidden, head, targets, impl="chunked", chunk_size=30,
                                   reduction=red)
        assert mx.abs(got - expect).max().item() < 1e-5


def test_public_api_returns_single_array_no_aux_leak() -> None:
    hidden, head, targets = _dense()
    out = linear_cross_entropy(hidden, head, targets, impl="chunked", reduction="none")
    assert isinstance(out, mx.array) and out.shape == (32,)  # noqa: PT018


def test_bsd_input_flattens_consistently() -> None:
    hidden, head, targets = _dense()
    h3 = hidden.reshape(2, 16, 16)
    t2 = targets.reshape(2, 16)
    flat = linear_cross_entropy(hidden, head, targets, impl="chunked", reduction="none")
    shaped = linear_cross_entropy(h3, head, t2, impl="chunked", reduction="none")
    assert shaped.shape == (2, 16)
    assert mx.abs(shaped.reshape(32) - flat).max().item() < 1e-6


def test_target_out_of_range_is_typed_error() -> None:
    hidden, head, _ = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, mx.full((32,), 100, dtype=mx.int32), impl="naive")


def test_hidden_wrong_ndim_is_typed_error() -> None:
    hidden, head, targets = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden[None, None], head, targets, impl="naive")  # 4D


def test_targets_shape_mismatch_is_typed_error() -> None:
    hidden, head, _ = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, mx.zeros((16,), dtype=mx.int32), impl="naive")


def test_dense_head_d_mismatch_is_typed_error() -> None:
    hidden, head, targets = _dense()
    mismatched = DenseHead(weight=mx.random.normal((head.weight.shape[0], 8)))
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, mismatched, targets, impl="naive")


def test_quantized_head_d_mismatch_is_typed_error() -> None:
    hidden, _, targets = _dense()  # d=16
    mx.random.seed(3)
    w = mx.random.normal((100, 128)).astype(mx.float32) * 0.05  # d=128 != hidden's d=16
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    head = QuantizedHead(w_q=w_q, scales=scales, biases=biases)
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, targets, impl="naive")


def test_dense_head_dtype_mismatch_is_typed_error() -> None:
    # a mismatch on the kernel path is an opaque Metal JIT build error — must be caught
    # up front, uniformly across impls
    hidden, head, targets = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, DenseHead(weight=head.weight.astype(mx.bfloat16)),
                             targets, impl="chunked")


def test_frozen_dense_head_d_hidden_still_correct() -> None:
    hidden, head, targets = _dense()
    frozen = DenseHead(weight=head.weight, trainable=False)

    def loss(h: mx.array) -> mx.array:
        return linear_cross_entropy(h, frozen, targets, impl="chunked").mean()

    g = mx.grad(loss)(hidden)  # w not a diffable arg anywhere -> must simply work
    # VALUE check, not just shape — an all-zero d_hidden from the frozen branch must fail
    r = mx.grad(lambda h: naive_linear_ce(h, head.weight, targets).mean())(hidden)
    assert mx.abs(g - r).max().item() < 1e-5


def test_zero_rows_input_is_typed_error_before_dispatch() -> None:
    """`select_variant` has no n<=0 floor of its own (math.log2(0) is a domain error) —
    input validation must reject an empty batch before resolve_impl ever calls it."""
    head = DenseHead(weight=mx.random.normal((100, 16)))
    hidden = mx.zeros((0, 16))
    targets = mx.zeros((0,), dtype=mx.int32)
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, targets, impl="auto")
    with pytest.raises(LossInputError):
        resolve_impl(head=head, dtype=mx.float32, n=0)  # would otherwise crash select_variant


def test_unknown_reduction_is_typed_error() -> None:
    hidden, head, targets = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, targets, impl="naive", reduction="bogus")


def test_unknown_impl_is_typed_error() -> None:
    hidden, head, targets = _dense()
    with pytest.raises(LossInputError):
        linear_cross_entropy(hidden, head, targets, impl="bogus")


def test_tied_head_shares_weight_and_defaults_frozen() -> None:
    hidden, head, targets = _dense()
    tied = tied_head(head.weight)
    assert tied.weight is head.weight
    assert tied.trainable is False
    got = linear_cross_entropy(hidden, tied, targets, impl="chunked")
    expect = naive_linear_ce(hidden, head.weight, targets).mean()
    assert abs(got.item() - expect.item()) < 1e-5


def test_naive_impl_matches_reference_dense() -> None:
    hidden, head, targets = _dense()
    got = linear_cross_entropy(hidden, head, targets, impl="naive", reduction="none")
    ref = naive_linear_ce(hidden, head.weight, targets)
    assert mx.abs(got - ref).max().item() < 1e-6


def test_naive_impl_dequantizes_quantized_head() -> None:
    mx.random.seed(11)
    n, d, v = 32, 128, 256
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    head = QuantizedHead(w_q=w_q, scales=scales, biases=biases)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
    got = linear_cross_entropy(hidden, head, targets, impl="naive", reduction="none")
    ref = naive_linear_ce(hidden, w_dq, targets)
    assert mx.abs(got - ref).max().item() < 1e-6


def test_chunked_impl_quantized_head_matches_naive() -> None:
    mx.random.seed(11)
    n, d, v = 64, 128, 1024  # d multiple of group 64 — same shape/seed as
    # test_chunked.py::test_quantized_value_and_dhidden_parity, so the same measured
    # tolerance (worst 3.3150e-3 -> pin 9e-3) applies to this wrapper unchanged.
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    head = QuantizedHead(w_q=w_q, scales=scales, biases=biases)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
    got = linear_cross_entropy(hidden, head, targets, impl="chunked", chunk_size=256,
                               reduction="none")
    ref = naive_linear_ce(hidden.astype(mx.float32), w_dq.astype(mx.float32), targets)
    assert mx.abs(got - ref).max().item() < 9e-3


@pytest.mark.metal
def test_kernel_impl_gradient_parity_all_reductions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    hidden, head, targets = _dense(64, 32, 1000)
    for red in ("mean", "sum"):
        # `ours`/`ref` are built AND called within this same loop iteration (never stored
        # for later), so the late-binding closure hazard B023 warns about doesn't apply.
        def ours(h: mx.array, w: mx.array) -> mx.array:
            return linear_cross_entropy(h, DenseHead(weight=w), targets, reduction=red)  # noqa: B023

        def ref(h: mx.array, w: mx.array) -> mx.array:
            nll = naive_linear_ce(h, w, targets)
            return nll.mean() if red == "mean" else nll.sum()  # noqa: B023

        g = mx.grad(ours, argnums=(0, 1))(hidden, head.weight)
        r = mx.grad(ref, argnums=(0, 1))(hidden, head.weight)
        assert mx.abs(g[0] - r[0]).max().item() < 1e-5
        assert mx.abs(g[1] - r[1]).max().item() < 1e-5


@pytest.mark.metal
def test_kernel_impl_forward_value_parity_and_no_aux_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vjp never reads the main output, so a sign/scale bug in nll leaves every
    gradient test green while training ascends — the VALUE must be checked through the
    public API, and the aux (lse, tgt) outputs must not leak from it."""
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    hidden, head, targets = _dense(64, 32, 1000)
    ref = naive_linear_ce(hidden, head.weight, targets)
    out = linear_cross_entropy(hidden, head, targets, impl="kernel", reduction="none")
    assert isinstance(out, mx.array) and out.shape == (64,)  # noqa: PT018 — single array, no aux tuple
    assert mx.abs(out - ref).max().item() < 1e-5
    for red, expect in (("mean", ref.mean()), ("sum", ref.sum())):
        got = linear_cross_entropy(hidden, head, targets, impl="kernel", reduction=red)
        assert got.shape == () and abs(got.item() - expect.item()) < 1e-4  # noqa: PT018


@pytest.mark.metal
def test_kernel_impl_nonuniform_cotangent_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """reduction='none' + weighted sum is EXACTLY the adapter's (nll*mask)/ntoks shape;
    mean/sum feed the vjp a uniform cotangent, which hides per-row broadcast bugs."""
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    hidden, head, targets = _dense(64, 32, 1000)
    mx.random.seed(29)
    weights = mx.random.uniform(shape=(64,))

    def ours(h: mx.array, w: mx.array) -> mx.array:
        nll = linear_cross_entropy(h, DenseHead(weight=w), targets, reduction="none")
        return (nll * weights).sum()

    def ref(h: mx.array, w: mx.array) -> mx.array:
        return (naive_linear_ce(h, w, targets) * weights).sum()

    g = mx.grad(ours, argnums=(0, 1))(hidden, head.weight)
    r = mx.grad(ref, argnums=(0, 1))(hidden, head.weight)
    assert mx.abs(g[0] - r[0]).max().item() < 1e-5
    assert mx.abs(g[1] - r[1]).max().item() < 1e-5


@pytest.mark.metal
def test_kernel_impl_quantized_head_d_hidden_parity() -> None:
    mx.random.seed(23)
    n, d, v = 64, 128, 1024
    hidden = mx.random.normal((n, d)).astype(mx.bfloat16)
    w = (mx.random.normal((v, d)) * 0.05).astype(mx.bfloat16)
    targets = mx.random.randint(0, v, (n,))
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    qhead = QuantizedHead(w_q=w_q, scales=scales, biases=biases)
    w_dq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)

    def ours(h: mx.array) -> mx.array:
        return linear_cross_entropy(h, qhead, targets).mean()

    def ref(h: mx.array) -> mx.array:
        return naive_linear_ce(h, w_dq, targets).mean()

    g = mx.grad(ours)(hidden)
    r = mx.grad(ref)(hidden)
    # measured worst 1.5259e-5 -> pin 4e-5 (~2.6x margin, same convention as
    # test_chunked.py's quantized d_hidden gate); the 2e-2 placeholder was never this loose
    # in practice.
    assert mx.abs(g.astype(mx.float32) - r.astype(mx.float32)).max().item() < 4e-5


@pytest.mark.metal
def test_kernel_impl_frozen_dense_head_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the KERNEL + frozen-dense-head branch (w captured by closure, no d_head
    path) — distinct code from the trainable-kernel branch above and from the
    frozen-CHUNKED branch in test_frozen_dense_head_d_hidden_still_correct."""
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    hidden, head, targets = _dense(64, 32, 1000)
    frozen = DenseHead(weight=head.weight, trainable=False)
    ref = naive_linear_ce(hidden, head.weight, targets)

    out = linear_cross_entropy(hidden, frozen, targets, impl="kernel", reduction="none")
    assert mx.abs(out - ref).max().item() < 1e-5

    def loss(h: mx.array) -> mx.array:
        return linear_cross_entropy(h, frozen, targets, impl="kernel").mean()

    def ref_loss(h: mx.array) -> mx.array:
        return naive_linear_ce(h, head.weight, targets).mean()

    g = mx.grad(loss)(hidden)
    r = mx.grad(ref_loss)(hidden)
    assert mx.abs(g - r).max().item() < 1e-5
