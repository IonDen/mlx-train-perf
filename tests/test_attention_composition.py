"""0.2.0 T2 — the composition GO/NO-GO gate (spec §9 P0b-d).

Proves, on tiny shapes, the make-or-break unknowns BEFORE any attention kernel exists:
(1) a hand-written `mx.custom_function` vjp fires under `mx.compile`;
(2) it fires under `mx.checkpoint` (the double-forward-recompute path), and checkpointing
    genuinely trades stored intermediates for recompute (peak-memory delta);
(3) it survives mlx_lm's CLASS-LEVEL `grad_checkpoint` patch inside the real compiled
    training step — exercised in a SUBPROCESS (`tests/_composition_gc_child.py`), because
    `tests/test_worker_train_step.py` already patches `mlx_lm.models.llama` in-process and
    the patch never reverts (repo gotcha 13; `tests/test_attention_wrapper.py` will hold
    the third gc=True site, also subprocess-isolated);
(4) the GQA head-grouping convention of `mx.fast.scaled_dot_product_attention` is the
    contiguous `q_head // group_size` mapping, not the interleaved `q_head % Hkv` one
    (pinned here; `attention/reference.py::kv_head_for` implements what this proves);
(5) the memory class of the installed mlx's SDPA training backward: bounded as O(N^2)
    on the mlx versions where that was measured (0.31.2, 0.32.0), recorded elsewhere.

Sentinel discipline: the toy vjp multiplies the true gradient by _SENTINEL, so a sentinel-
valued gradient PROVES our vjp produced it (autodiff would give the unscaled value). All
assertions are on gradient VALUES or measured peaks — never on Python trace-time side
effects, which are unobservable under mx.compile.
"""

import math
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import pytest
from conftest import needs_comparative_peak_room

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.reference import math_attention
from mlx_train_perf.plan.calibration import STOCK_ATTENTION_CALIBRATED_MLX_VERSIONS

pytest.importorskip("mlx_lm")

_SENTINEL = 3.0
_CHILD = Path(__file__).parent / "_composition_gc_child.py"


@mx.custom_function
def toy_mul(x: mx.array, w: mx.array) -> mx.array:
    """y = x * w, with a hand vjp (registered below) that scales grads by _SENTINEL."""
    return x * w


@toy_mul.vjp
def _toy_mul_vjp(
    primals: tuple[mx.array, mx.array],
    cotangent: mx.array,
    _outputs: mx.array,
) -> tuple[mx.array, mx.array]:
    x, w = primals
    return cotangent * w * _SENTINEL, cotangent * x * _SENTINEL


def test_custom_vjp_fires_under_compile() -> None:
    w = mx.array([2.0, 4.0])
    x = mx.array([1.0, 3.0])

    def loss(x_: mx.array) -> mx.array:
        return toy_mul(x_, w).sum()

    g = mx.compile(mx.grad(loss))(x)
    mx.eval(g)
    # autodiff gives d/dx = w; ONLY our vjp gives w * _SENTINEL
    assert mx.allclose(g, w * _SENTINEL).item()


def test_custom_vjp_fires_inside_checkpoint() -> None:
    w = mx.array([2.0, 4.0])
    x = mx.array([1.0, 3.0])

    def inner(x_: mx.array) -> mx.array:
        return toy_mul(x_, w).sum()

    g = mx.grad(mx.checkpoint(inner))(x)
    mx.eval(g)
    assert mx.allclose(g, w * _SENTINEL).item()


def test_custom_vjp_fires_inside_checkpoint_under_compile() -> None:
    """The full nesting the training path uses: compile(grad(checkpoint(fn)))."""
    w = mx.array([2.0, 4.0])
    x = mx.array([1.0, 3.0])

    def inner(x_: mx.array) -> mx.array:
        return toy_mul(x_, w).sum()

    g = mx.compile(mx.grad(mx.checkpoint(inner)))(x)
    mx.eval(g)
    assert mx.allclose(g, w * _SENTINEL).item()


@needs_comparative_peak_room
def test_checkpoint_recomputes_instead_of_storing_under_compile() -> None:
    """Memory proof of recompute, measured in the regime the trainer actually uses.
    VERIFIED FACT (mlx 0.32.0, this gate, 2026-07-09): mx.checkpoint's peak reduction
    materializes ONLY under mx.compile -- uncompiled, the lazy scheduler does not
    interleave recompute/free and checkpointing measured slightly WORSE (824 vs 736 MB on
    this exact chain); compiled, it measured 448 vs 632 MB. mlx_lm's trainer compiles the
    step, so compiled is the only regime that matters -- and any future memory assertion
    about checkpointing must be made under compile. Segments expand 8x internally so
    stored-intermediates dominate the boundaries (a flat elementwise chain shows NO delta
    -- the scheduler already streams it)."""
    n = 2_000_000
    ones8 = mx.ones((1, 8))

    def segment(h: mx.array) -> mx.array:
        t = mx.tanh(h[:, None] * ones8)  # (n, 8) internal -- 8x the (n,) boundary
        t = mx.tanh(t * 1.01)
        return t.sum(axis=1) / 8.0

    def plain(x_: mx.array) -> mx.array:
        return segment(segment(segment(x_))).sum()

    ck = mx.checkpoint(segment)

    def checkpointed(x_: mx.array) -> mx.array:
        return ck(ck(ck(x_))).sum()

    x = mx.random.normal((n,))
    mx.eval(x)

    def compiled_grad_peak(fn) -> int:  # type: ignore[no-untyped-def]
        # Gotcha-15 measurement discipline (backlog 0023 -- this comparison flaked
        # INVERTED once on a 7 GB CI runner): the warmup output stays ALIVE through the
        # measured window (a deferred release mid-window perturbs the peak
        # nondeterministically), and every snapshot boundary gets mx.synchronize() +
        # mx.clear_cache() so both windows start from the same allocator state.
        g_fn = mx.compile(mx.grad(fn))
        warm = g_fn(x)
        mx.eval(warm)  # warmup/trace OUTSIDE the measurement window
        mx.synchronize()  # no in-flight frees racing the reset
        mx.clear_cache()
        mx.reset_peak_memory()
        out = g_fn(x)
        mx.eval(out)
        mx.synchronize()
        peak = int(mx.get_peak_memory())
        del warm, out  # released only after the snapshot
        return peak

    plain_peak = compiled_grad_peak(plain)
    ckpt_peak = compiled_grad_peak(checkpointed)
    assert ckpt_peak < plain_peak, (
        f"compiled checkpointed grad peak {ckpt_peak} not below plain {plain_peak} -- "
        "recompute did not reduce stored intermediates in the compiled regime"
    )


def test_custom_vjp_survives_mlx_lm_class_patch() -> None:
    """Subprocess-isolated (gotcha 13): the child applies mlx_lm's class-level
    grad_checkpoint patch to a tiny llama whose every attention output routes through a
    sentinel custom_function, then runs 2 real compiled train() steps. Its stdout carries
    the verdict."""
    proc = subprocess.run(
        [sys.executable, str(_CHILD)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "COMPOSITION_OK" in proc.stdout, proc.stdout


@pytest.mark.parametrize(("hq", "hkv"), [(4, 2), (8, 2)])
def test_gqa_grouping_convention_matches_mlx(hq: int, hkv: int) -> None:
    """Head-identifiable V contents: kv head j is constant (j+1), so each query head's
    output IS the value of the kv head it attended (softmax weights cancel on a constant).
    The contiguous mapping (h // group_size) and the interleaved one (h % Hkv) predict
    different outputs for at least one head at both ratios -- the test genuinely
    discriminates (8q/2kv is the flagship's group_size-4 ratio class)."""
    n, d = 4, 8
    group = hq // hkv
    q = mx.ones((1, hq, n, d))
    k = mx.ones((1, hkv, n, d))
    v = mx.stack([mx.full((n, d), float(j + 1)) for j in range(hkv)], axis=0)[
        None
    ]  # (1, hkv, n, d)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)
    mx.eval(o)

    contiguous = [h // group for h in range(hq)]
    interleaved = [h % hkv for h in range(hq)]
    assert contiguous != interleaved  # the two candidates genuinely differ at this shape
    for h in range(hq):
        expected = mx.full((n, d), float(contiguous[h] + 1))
        assert mx.allclose(o[0, h], expected).item(), (
            f"q head {h}: mlx did not use the contiguous // mapping"
        )


def _judge_sdpa_peak_ratio(mlx_version: str, peak_small: float, peak_large: float) -> float:
    """Return peak_large / peak_small. Fails on a broken reading on any version, and on a
    ratio at or below 3.0 on a version where the O(N^2) backward was observed.

    The observed versions are the planner's STOCK_ATTENTION_CALIBRATED_MLX_VERSIONS (stock
    SDPA's fwd+bwd peak grew ~4x per N doubling there), imported so this canary and the
    planner's quadratic term cannot drift apart."""
    assert peak_small > 0, f"broken measurement: N=1024 peak {peak_small} is not positive"
    assert peak_large > 0, f"broken measurement: N=2048 peak {peak_large} is not positive"
    ratio = peak_large / peak_small
    assert math.isfinite(ratio), f"broken measurement: peak ratio {ratio} is not finite"
    if mlx_version in STOCK_ATTENTION_CALIBRATED_MLX_VERSIONS:
        assert ratio > 3.0, (
            f"SDPA fwd+bwd peak ratio {ratio:.2f} for N 1024->2048 on mlx {mlx_version} is "
            "not O(N^2)-class, where an O(N^2) backward was measured before"
        )
    return ratio


@pytest.mark.parametrize("mlx_version", ["0.31.2", "0.32.0"])
@pytest.mark.parametrize(
    ("peak_large", "should_fail"),
    [(299, True), (300, True), (301, False)],
)
def test_sdpa_peak_ratio_bound_applies_on_observed_versions(
    mlx_version: str, peak_large: int, should_fail: bool
) -> None:
    """Bug caught: the 3.0 floor drifts (`>` to `>=`, or a shifted constant), or one of
    the two observed versions drops out of the bounded set, so a broken O(N^2) reading
    on 0.31.2/0.32.0 would pass. Peaks 100 -> 299/300/301 sit just below, on and just
    above the floor."""
    if should_fail:
        with pytest.raises(AssertionError, match="not O\\(N\\^2\\)-class"):
            _judge_sdpa_peak_ratio(mlx_version, 100, peak_large)
    else:
        assert _judge_sdpa_peak_ratio(mlx_version, 100, peak_large) == pytest.approx(3.01)


@pytest.mark.parametrize("mlx_version", ["0.32.4", "0.33.0"])
def test_sdpa_peak_ratio_is_recorded_unbounded_on_other_versions(mlx_version: str) -> None:
    """Bug caught: the floor is applied to every version again (or a number is pinned
    for an unobserved one), so a low reading outside 0.31.2/0.32.0 turns the lane red
    instead of being recorded."""
    assert _judge_sdpa_peak_ratio(mlx_version, 100, 119) == pytest.approx(1.19)


@pytest.mark.parametrize("mlx_version", ["0.32.0", "0.32.4"])
@pytest.mark.parametrize(
    ("peak_small", "peak_large"),
    [(0, 100), (100, 0), (-5, 100), (100, -5), (100, math.inf)],
)
def test_sdpa_peak_ratio_rejects_broken_measurements(
    mlx_version: str, peak_small: float, peak_large: float
) -> None:
    """Bug caught: either positivity check or the finiteness check is dropped (or `>`
    becomes `>=`), so a zero, negative or infinite peak passes as a recorded ratio on
    any version."""
    with pytest.raises(AssertionError, match="broken measurement"):
        _judge_sdpa_peak_ratio(mlx_version, peak_small, peak_large)


def test_sdpa_backward_memory_class_on_installed_mlx(
    record_property: Callable[[str, object], None],
) -> None:
    """Stock attention's fwd+bwd peak at N=1024 vs N=2048, on the installed mlx.

    On mlx 0.31.2 and 0.32.0 the peak grew ~4x per N doubling (an O(N^2) backward),
    and on those versions the ratio must stay above 3.0. mlx 0.32.4 changed this
    behaviour. On any other version the ratio is recorded (test property, warning,
    printed line) for a re-measure and the test passes; a low reading is not a clean
    O(N) result either, because `reset_peak_memory` counts the live inputs and earlier
    process state, and this is one N pair, fp32, d=64, with grad on q only. A broken
    reading (a zero or negative peak, a non-finite ratio) fails on every version.
    """
    heads, d = 8, 64

    def peak_at(n: int) -> int:
        q = mx.random.normal((1, heads, n, d))
        k = mx.random.normal((1, heads, n, d))
        v = mx.random.normal((1, heads, n, d))
        mx.eval(q, k, v)

        def f(q_: mx.array) -> mx.array:
            return mx.fast.scaled_dot_product_attention(q_, k, v, scale=1.0, mask="causal").sum()

        mx.reset_peak_memory()
        g = mx.grad(f)(q)
        mx.eval(g)
        return int(mx.get_peak_memory())

    mlx_version = _installed_mlx_version()
    p1, p2 = peak_at(1024), peak_at(2048)
    ratio = _judge_sdpa_peak_ratio(mlx_version, p1, p2)
    record_property("mlx_version", mlx_version)
    record_property("sdpa_peak_ratio_n1024_n2048", ratio)
    line = f"mlx {mlx_version}: SDPA fwd+bwd peak ratio N 1024->2048 = {ratio:.4f}"
    print(line)
    if mlx_version not in STOCK_ATTENTION_CALIBRATED_MLX_VERSIONS:
        warnings.warn(f"{line} (recorded, not bounded on this version)", stacklevel=1)


def test_flash_attention_under_compile_and_checkpoint() -> None:
    """T4's real `flash_attention` (not a toy) under the T2-proven nesting: gradient
    parity against the `math_attention` autodiff oracle, both INSIDE
    `mx.compile(mx.grad(...))` and under `mx.grad(mx.checkpoint(...))`. Tiny shapes,
    `impl='reference'` (the oracle -- never a production path)."""
    b, hq, hkv, n, d = 1, 2, 1, 8, 8
    mx.random.seed(20)
    q = mx.random.normal((b, hq, n, d))
    k = mx.random.normal((b, hkv, n, d))
    v = mx.random.normal((b, hkv, n, d))
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(d)

    def flash_loss(q_: mx.array) -> mx.array:
        return flash_attention(q_, k, v, scale=scale, causal=True, impl="reference").sum()

    def math_loss(q_: mx.array) -> mx.array:
        return math_attention(q_, k, v, scale=scale, causal=True).sum()

    g_oracle = mx.grad(math_loss)(q)
    g_compiled = mx.compile(mx.grad(flash_loss))(q)
    g_checkpoint = mx.grad(mx.checkpoint(flash_loss))(q)
    mx.eval(g_oracle, g_compiled, g_checkpoint)

    # measured-first (mlx 0.32.0, fp32, seed=20, tiny shape 1x2x8x8): worst |diff|
    # 2.980232e-07 (compiled), 2.384186e-07 (checkpoint) -- compile's op fusion
    # reorders fp32 rounding vs the uncompiled oracle -> pin 1e-6 (~3.4x margin).
    diff_compiled = mx.abs(g_compiled - g_oracle).max().item()
    diff_checkpoint = mx.abs(g_checkpoint - g_oracle).max().item()
    assert diff_compiled < 1e-6, diff_compiled
    assert diff_checkpoint < 1e-6, diff_checkpoint
