"""Ordinary least squares over named features, for fitting memory coefficients.

Needs numpy (`pip install "mlx-train-perf[fit]"`); imported only when `fit_linear` runs.
A rank-deficient design, a missing feature, a non-positive measurement, a duplicate
label, or a negative coefficient on a feature not listed in `allow_negative` is refused,
never patched over: each of those means the functional form or the data is wrong.
"""
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlx_train_perf.memfit.errors import MemfitDependencyError, MemfitInputError

__all__ = ["FitSample", "LinearFit", "fit_linear"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FitSample:
    features: Mapping[str, float]
    measured_bytes: float
    phase: str | None = None
    label: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearFit:
    coefficients: dict[str, float]
    residuals: dict[str, float]
    max_abs_rel_error: float


def _validate_sample(
    sample: FitSample, label: str, features: Sequence[str],
) -> list[float]:
    """Validate a sample and return its feature values.
    Raises MemfitInputError on validation failure."""
    missing = [f for f in features if f not in sample.features]
    if missing:
        raise MemfitInputError(f"sample {label} lacks features {missing}")
    if not math.isfinite(float(sample.measured_bytes)):
        raise MemfitInputError(
            f"sample {label} has a non-finite measured_bytes={sample.measured_bytes!r}")
    if not sample.measured_bytes > 0:
        raise MemfitInputError(
            f"sample {label} has measured_bytes={sample.measured_bytes}; must be > 0")
    feature_values = []
    for f in features:
        v = float(sample.features[f])
        if not math.isfinite(v):
            raise MemfitInputError(f"sample {label} has a non-finite {f}={v!r}")
        feature_values.append(v)
    return feature_values


def fit_linear(
    samples: Sequence[FitSample], *, features: Sequence[str], intercept: bool = True,
    phase: str | None = None, allow_negative: frozenset[str] = frozenset(),
) -> LinearFit:
    """Fit `measured_bytes ~ intercept + sum(coef[f] * features[f])` over the samples
    (those whose `phase` matches, when `phase` is given). Coefficient names are the
    feature names plus `"intercept"`; residuals are keyed by sample label (`#<index>`
    for an unlabelled sample). The rank check refuses exactly collinear features and is
    scale-independent; it is not a conditioning check, so validate a fit on hold-out points.

    `LinearFit.max_abs_rel_error` is relative to the measured value; the hold-out band
    (`memfit.holdout`) is relative to the prediction."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise MemfitDependencyError(
            'fit_linear needs numpy: pip install "mlx-train-perf[fit]"') from exc
    if not features or len(set(features)) != len(features):
        raise MemfitInputError(f"features must be non-empty and unique (got {list(features)})")
    if intercept and "intercept" in features:
        raise MemfitInputError("'intercept' is reserved when intercept=True")
    chosen = [s for s in samples if phase is None or s.phase == phase]
    labels = [s.label or f"#{i}" for i, s in enumerate(chosen)]
    dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
    if dupes:
        raise MemfitInputError(f"duplicate sample labels: {dupes}")
    names = (["intercept"] if intercept else []) + list(features)
    rows: list[list[float]] = []
    for sample, label in zip(chosen, labels, strict=True):
        feature_values = _validate_sample(sample, label, features)
        rows.append(([1.0] if intercept else []) + feature_values)
    if len(rows) < len(names):
        raise MemfitInputError(
            f"{len(rows)} samples cannot determine {len(names)} unknowns ({names})")
    design = np.array(rows, dtype=float)
    measured = np.array([float(s.measured_bytes) for s in chosen], dtype=float)
    scale = np.abs(design).max(axis=0)
    scale[scale == 0] = 1.0
    if int(np.linalg.matrix_rank(design / scale)) < len(names):
        raise MemfitInputError(
            f"rank-deficient design: the samples cannot separate {names}; vary the "
            "features independently or drop one")
    solution = np.linalg.lstsq(design, measured, rcond=None)[0]
    coefficients = {n: float(c) for n, c in zip(names, solution, strict=True)}
    negative = sorted(n for n, c in coefficients.items() if c < 0 and n not in allow_negative)
    if negative:
        raise MemfitInputError(
            f"negative fitted coefficients {negative}: a negative bytes-per-unit means "
            "the functional form is wrong (list a feature in allow_negative to probe it)")
    predicted = design @ solution
    residuals = {lab: float(m - p) for lab, m, p in zip(labels, measured, predicted, strict=True)}
    worst = max(abs(r) / float(m) for r, m in zip(residuals.values(), measured, strict=True))
    return LinearFit(coefficients=coefficients, residuals=residuals, max_abs_rel_error=worst)
