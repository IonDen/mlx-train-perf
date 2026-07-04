class MlxTrainPerfError(Exception):
    """Root of all mlx-train-perf errors."""


class UnsupportedHeadError(MlxTrainPerfError):
    """Head kind/dtype the requested impl cannot serve (no silent fallback)."""


class UnverifiedMlxError(MlxTrainPerfError):
    """Installed mlx version is not on the kernel's verified list."""


class LaunchBudgetError(MlxTrainPerfError):
    """A dispatch would exceed the GPU-watchdog budget; refused before launch."""


class LossInputError(MlxTrainPerfError):
    """Shape/dtype/target-range validation failure at the loss boundary."""


class PlanInputError(MlxTrainPerfError):
    """Shape/config validation failure in the fit planner (e.g. an unrecognized dtype)."""


class DoesNotFitError(MlxTrainPerfError):
    """Planner verdict: predicted peak exceeds the memory budget."""


class AdapterError(MlxTrainPerfError):
    """mlx-lm model can't be split (unsupported architecture, missing modules)."""


class MissingDependencyError(MlxTrainPerfError):
    """A lazily-imported optional dependency is absent."""


class BenchInputError(MlxTrainPerfError):
    """Bench condition input validation failure (e.g. a reserved identity param key)."""
