from mlx_train_perf import memfit


def test_memfit_public_surface_is_exactly_the_documented_names() -> None:
    """The stability promise from 0.9.0 covers these names; mlx-dfloat imports them."""
    assert sorted(memfit.__all__) == sorted([
        "CALIBRATION_FORMAT", "CalibrationFile", "CalibrationMismatchError",
        "DoesNotFitError", "FitSample", "HOLDOUT_FORMAT", "HoldoutBand", "HoldoutPoint",
        "HoldoutReport", "LinearFit", "MAX_FOOTPRINT_MLX_ACTIVE_CACHE", "MLX_ACTIVE_MARGINAL",
        "MLX_ACTIVE_TOTAL", "MemfitDependencyError", "MemfitError", "MemfitInputError",
        "MemoryModel", "PeakEstimate", "Phase", "dump_calibration_file", "estimate",
        "fit_linear", "fits", "load_calibration_file", "max_int_within_budget",
        "score_holdout", "write_holdout_predictions",
    ])
    for name in memfit.__all__:
        assert hasattr(memfit, name), name
