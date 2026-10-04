"""Memory-fit core: phase-structured peak estimates, the largest setting that fits a
budget, calibration files, least-squares fits and hold-out scoring. Pure Python; numpy is
needed only for `fit_linear` (`pip install "mlx-train-perf[fit]"`)."""
from mlx_train_perf.memfit.calibration import (
    CALIBRATION_FORMAT,
    MAX_FOOTPRINT_MLX_ACTIVE_CACHE,
    MLX_ACTIVE_MARGINAL,
    MLX_ACTIVE_TOTAL,
    CalibrationFile,
    dump_calibration_file,
    load_calibration_file,
)
from mlx_train_perf.memfit.errors import (
    CalibrationMismatchError,
    DoesNotFitError,
    MemfitDependencyError,
    MemfitError,
    MemfitInputError,
)
from mlx_train_perf.memfit.fit import (
    FitSample,
    LinearFit,
    fit_linear,
)
from mlx_train_perf.memfit.holdout import (
    HOLDOUT_FORMAT,
    HoldoutBand,
    HoldoutPoint,
    HoldoutReport,
    score_holdout,
    write_holdout_predictions,
)
from mlx_train_perf.memfit.model import (
    MemoryModel,
    PeakEstimate,
    Phase,
    estimate,
    fits,
    max_int_within_budget,
)

__all__ = [
    "CALIBRATION_FORMAT",
    "HOLDOUT_FORMAT",
    "MAX_FOOTPRINT_MLX_ACTIVE_CACHE",
    "MLX_ACTIVE_MARGINAL",
    "MLX_ACTIVE_TOTAL",
    "CalibrationFile",
    "CalibrationMismatchError",
    "DoesNotFitError",
    "FitSample",
    "HoldoutBand",
    "HoldoutPoint",
    "HoldoutReport",
    "LinearFit",
    "MemfitDependencyError",
    "MemfitError",
    "MemfitInputError",
    "MemoryModel",
    "PeakEstimate",
    "Phase",
    "dump_calibration_file",
    "estimate",
    "fit_linear",
    "fits",
    "load_calibration_file",
    "max_int_within_budget",
    "score_holdout",
    "write_holdout_predictions",
]
