"""memfit's own error hierarchy. It does not derive from mlx_train_perf's errors so the
package can move into its own distribution without changing what callers catch;
`mlx_train_perf.errors.PlanInputError` / `DoesNotFitError` subclass these instead."""

__all__ = ["CalibrationMismatchError", "DoesNotFitError", "MemfitDependencyError",
           "MemfitError", "MemfitInputError"]


class MemfitError(Exception):
    """Root of every memfit error."""


class MemfitInputError(MemfitError):
    """Invalid model, parameters, samples or file contents (no silent fallback)."""


class DoesNotFitError(MemfitError):
    """Nothing in the searched range fits the budget."""


class CalibrationMismatchError(MemfitInputError):
    """A calibration or hold-out file measured a different quantity than the caller expects."""


class MemfitDependencyError(MemfitError):
    """An optional dependency (numpy, for fitting) is not installed."""
