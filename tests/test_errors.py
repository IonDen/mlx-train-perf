from mlx_train_perf.errors import (
    MlxTrainPerfError,
    RecurrentInputError,
    UnsupportedRecurrentError,
)


def test_recurrent_errors_are_package_rooted():
    """Catches: a refusal raising bare Exception/AttributeError that callers can't filter."""
    assert issubclass(UnsupportedRecurrentError, MlxTrainPerfError)
    assert issubclass(RecurrentInputError, MlxTrainPerfError)
