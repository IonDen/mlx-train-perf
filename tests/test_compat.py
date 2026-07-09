import pytest

from mlx_train_perf._compat import (
    VERIFIED_MLX_VERSIONS,
    _installed_mlx_version,
    check_mlx_verified,
)
from mlx_train_perf.errors import MlxTrainPerfError, UnverifiedMlxError


def test_verified_list_contains_the_spike_version() -> None:
    assert "0.31.2" in VERIFIED_MLX_VERSIONS


def test_verified_list_contains_the_0_2_0_bump_version() -> None:
    """0.2.0 T1: mlx 0.32.0 joins the allowlist ONLY after the full 0.1.0 kernel-contract
    re-verification (metal lane + parity pins + regpressure ceilings) passes on it."""
    assert "0.32.0" in VERIFIED_MLX_VERSIONS


def test_installed_mlx_is_verified() -> None:
    """The dev environment must always run a kernel-verified mlx — a bump without the
    re-verification ritual (and allowlist entry) fails here before anything else does."""
    assert _installed_mlx_version() in VERIFIED_MLX_VERSIONS


def test_current_mlx_passes_when_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.31.2")
    check_mlx_verified(allow_unverified=False)  # must not raise


def test_unlisted_mlx_raises_with_list_and_override_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.99.0")
    with pytest.raises(UnverifiedMlxError) as ei:
        check_mlx_verified(allow_unverified=False)
    msg = str(ei.value)
    assert "0.99.0" in msg and "0.31.2" in msg and "allow_unverified_mlx" in msg  # noqa: PT018


def test_override_permits_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.99.0")
    check_mlx_verified(allow_unverified=True)  # must not raise


def test_errors_are_package_rooted() -> None:
    assert issubclass(UnverifiedMlxError, MlxTrainPerfError)
