import pytest
from test_memfit_imports import run_with_blocked_imports

from mlx_train_perf.memfit import FitSample, MemfitInputError, fit_linear


def _samples(rows: list[tuple[float, float]], phase: str | None = None) -> list[FitSample]:
    # truth: bytes = 500 + 3 * tokens
    return [FitSample(features={"tokens": t}, measured_bytes=500 + 3 * t + noise,
                      phase=phase, label=f"t{int(t)}") for t, noise in rows]


def test_recovers_hand_set_coefficients_exactly_without_noise() -> None:
    fit = fit_linear(_samples([(100, 0), (200, 0), (400, 0)]), features=["tokens"])
    assert fit.coefficients["intercept"] == pytest.approx(500, rel=1e-9)
    assert fit.coefficients["tokens"] == pytest.approx(3, rel=1e-9)
    assert fit.max_abs_rel_error == pytest.approx(0, abs=1e-12)
    assert set(fit.residuals) == {"t100", "t200", "t400"}


def test_reports_the_worst_relative_residual() -> None:
    # Measured 800, 1400, 1130 at t = 100, 300, 200. Hand least squares: slope 3,
    # intercept 510, fitted 810 / 1410 / 1110, so residuals -10 / -10 / +20 and the worst
    # relative one is 20 / 1130 (relative to the measured value).
    fit = fit_linear(_samples([(100, 0), (300, 0), (200, 30)]), features=["tokens"])
    assert fit.max_abs_rel_error == pytest.approx(20 / 1130)
    assert fit.residuals == {"t100": pytest.approx(-10), "t300": pytest.approx(-10),
                             "t200": pytest.approx(20)}


def test_phase_filters_the_samples() -> None:
    samples = [
        *_samples([(100, 0), (200, 0)], phase="denoise"),
        FitSample(features={"tokens": 100.0}, measured_bytes=10**9, phase="vae", label="v"),
    ]
    fit = fit_linear(samples, features=["tokens"], phase="denoise")
    assert fit.coefficients["tokens"] == pytest.approx(3, rel=1e-9)


def test_a_rank_deficient_design_is_refused() -> None:
    """Two features that are exact multiples cannot be separated."""
    samples = [FitSample(features={"a": t, "b": 2 * t}, measured_bytes=7 * t, label=str(t))
               for t in (1.0, 2.0, 3.0)]
    with pytest.raises(MemfitInputError, match="rank"):
        fit_linear(samples, features=["a", "b"], intercept=False)


def test_too_few_samples_are_refused() -> None:
    with pytest.raises(MemfitInputError, match="2 unknowns"):
        fit_linear(_samples([(100, 0)]), features=["tokens"])


def test_a_negative_coefficient_is_refused_unless_allowed() -> None:
    # truth: bytes = 1000 - 2 * tokens
    samples = [FitSample(features={"tokens": t}, measured_bytes=1000 - 2 * t, label=str(t))
               for t in (10.0, 20.0, 30.0)]
    with pytest.raises(MemfitInputError, match="tokens"):
        fit_linear(samples, features=["tokens"])
    fit = fit_linear(samples, features=["tokens"], allow_negative=frozenset({"tokens"}))
    assert fit.coefficients["tokens"] == pytest.approx(-2, rel=1e-9)


def test_a_sample_missing_a_feature_is_named() -> None:
    samples = [
        *_samples([(100, 0), (200, 0)]),
        FitSample(features={"pixels": 1.0}, measured_bytes=1.0, label="odd-one"),
    ]
    with pytest.raises(MemfitInputError, match="odd-one"):
        fit_linear(samples, features=["tokens"])


def test_duplicate_labels_are_refused() -> None:
    samples = [FitSample(features={"tokens": t}, measured_bytes=500 + 3 * t, label="same")
               for t in (1.0, 2.0, 3.0)]
    with pytest.raises(MemfitInputError, match="same"):
        fit_linear(samples, features=["tokens"])


@pytest.mark.parametrize("measured", [0.0, -5.0])
def test_non_positive_measurements_are_refused(measured: float) -> None:
    samples = [
        *_samples([(100, 0), (200, 0)]),
        FitSample(features={"tokens": 300.0}, measured_bytes=measured, label="bad"),
    ]
    with pytest.raises(MemfitInputError, match="bad"):
        fit_linear(samples, features=["tokens"])


def test_missing_numpy_raises_the_typed_error() -> None:
    r = run_with_blocked_imports("""
from mlx_train_perf.memfit import FitSample, MemfitDependencyError, fit_linear
try:
    fit_linear([FitSample(features={"x": 1.0}, measured_bytes=1.0)], features=["x"])
except MemfitDependencyError as exc:
    print("typed:", "mlx-train-perf[fit]" in str(exc))
""")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "typed: True"


@pytest.mark.parametrize("measured", [float("inf")])
def test_non_finite_measured_bytes_are_refused(measured: float) -> None:
    samples = [
        *_samples([(100, 0), (200, 0)]),
        FitSample(features={"tokens": 300.0}, measured_bytes=measured, label="bad"),
    ]
    with pytest.raises(MemfitInputError, match="bad"):
        fit_linear(samples, features=["tokens"])


@pytest.mark.parametrize("feature_value", [float("nan"), float("inf")])
def test_non_finite_feature_values_are_refused(feature_value: float) -> None:
    samples = [
        *_samples([(100, 0), (200, 0)]),
        FitSample(features={"tokens": feature_value}, measured_bytes=700.0, label="bad"),
    ]
    with pytest.raises(MemfitInputError, match="bad"):
        fit_linear(samples, features=["tokens"])


def test_intercept_is_a_reserved_feature_name() -> None:
    samples = _samples([(100, 0), (200, 0)])
    with pytest.raises(MemfitInputError, match="reserved"):
        fit_linear(samples, features=["intercept"])


def test_intercept_false_recovers_coefficients_without_constant() -> None:
    # truth: bytes = 3 * tokens (no intercept)
    samples = [FitSample(features={"tokens": t}, measured_bytes=3 * t, label=str(t))
               for t in (100.0, 200.0, 400.0)]
    fit = fit_linear(samples, features=["tokens"], intercept=False)
    assert "intercept" not in fit.coefficients
    assert fit.coefficients["tokens"] == pytest.approx(3, rel=1e-9)
