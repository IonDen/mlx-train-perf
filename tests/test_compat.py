import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

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


def test_mlx_0_32_4_is_refused_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """mlx 0.32.4 changed the recurrent path's numerics against an fp64 reference, so it is
    not kernel-verified: the kernel impl needs the `allow_unverified_mlx` opt-in there.
    Goes red if 0.32.4 is added to the allowlist without the re-verification."""
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.32.4")
    with pytest.raises(UnverifiedMlxError) as ei:
        check_mlx_verified(allow_unverified=False)
    assert "0.32.4" in str(ei.value)


def test_unlisted_mlx_raises_with_list_and_override_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.99.0")
    with pytest.raises(UnverifiedMlxError) as ei:
        check_mlx_verified(allow_unverified=False)
    msg = str(ei.value)
    assert "0.99.0" in msg and "0.31.2" in msg and "allow_unverified_mlx" in msg  # noqa: PT018


def test_override_permits_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlx_train_perf._compat._installed_mlx_version", lambda: "0.99.0")
    check_mlx_verified(allow_unverified=True)  # must not raise


def test_guard_extra_and_dev_group_pin_the_same_exact_mlx_guard() -> None:
    """mlx-guard is pre-1.0 and its reports are readable only by the version that wrote
    them, so the `guard` extra and the dev group (the `--run-guard` lane) carry one exact
    pin. Goes red if either drifts, or is loosened to a range. The dev entry is found by
    its requirement name, so a neighbour such as `mlx-guardian` is not taken for it."""
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    guard_extra = pyproject["project"]["optional-dependencies"]["guard"]
    dev_guard = [
        r for r in pyproject["dependency-groups"]["dev"] if Requirement(r).name == "mlx-guard"
    ]
    assert guard_extra == ["mlx-guard==0.3.0"]
    assert dev_guard == ["mlx-guard==0.3.0"]


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )


def _specifier(requirements: list[str], name: str) -> SpecifierSet:
    matches = [Requirement(r) for r in requirements if Requirement(r).name == name]
    assert len(matches) == 1, f"expected exactly one {name} requirement in {requirements}"
    return matches[0].specifier


def test_core_mlx_requirement_is_exactly_0_32_0() -> None:
    """Fresh installs must resolve the mlx the kernel contract is verified on. Goes red if
    the core requirement is loosened back to a range (a fresh install would then pick up
    0.32.4 and be refused by the gate) or moved to another version."""
    spec = _specifier(_pyproject()["project"]["dependencies"], "mlx")
    assert spec == SpecifierSet("==0.32.0")


def test_pinned_core_mlx_is_kernel_verified() -> None:
    """The version a fresh install resolves must pass the gate without the override. Goes
    red if the pin is bumped without the matching allowlist entry, or the pinned version
    is dropped from the allowlist."""
    spec = _specifier(_pyproject()["project"]["dependencies"], "mlx")
    pinned = [s.version for s in spec if s.operator == "=="]
    assert len(pinned) == 1, f"core mlx requirement is not one exact pin: {spec}"
    assert pinned[0] in VERIFIED_MLX_VERSIONS


def _mlx_lm_pin_sources() -> dict[str, list[str]]:
    pyproject = _pyproject()
    return {
        "mlx-lm extra": pyproject["project"]["optional-dependencies"]["mlx-lm"],
        "dev group": pyproject["dependency-groups"]["dev"],
    }


@pytest.mark.parametrize("source", ["mlx-lm extra", "dev group"])
@pytest.mark.parametrize(
    ("version", "admitted"),
    [("0.31.2", False), ("0.31.3", True), ("0.32.0", False), ("0.32.2", False)],
)
def test_mlx_lm_pin_admits_0_31_x_from_0_31_3_and_stops_below_0_32(
    source: str, version: str, admitted: bool
) -> None:
    """mlx-lm 0.32.x changed the qwen3_5 GatedDelta sources the recurrent path mirrors
    (gated_delta_update, GatedDeltaNet and RMSNormGated among them), so both pins stay below
    0.32 until the mirror is redone; the 0.31.3 floor stays. Goes red if the cap is widened
    to admit 0.32.x, dropped, or the floor moves."""
    spec = _specifier(_mlx_lm_pin_sources()[source], "mlx-lm")
    assert spec.contains(version) is admitted


@pytest.mark.parametrize("source", ["mlx-lm extra", "dev group"])
@pytest.mark.parametrize(
    ("version", "admitted"),
    [("4.57.0", False), ("5.0.0", True), ("5.12.0", True), ("5.13.0", False)],
)
def test_transformers_pin_stops_below_5_13(source: str, version: str, admitted: bool) -> None:
    """transformers 5.13 breaks `import mlx_lm` (AutoTokenizer.register with a string class
    name). Goes red if the <5.13 cap is widened or dropped in either place."""
    spec = _specifier(_mlx_lm_pin_sources()[source], "transformers")
    assert spec.contains(version) is admitted


def test_mlx_lm_extra_and_dev_group_carry_the_same_pins() -> None:
    """The dev environment must test what the extra promises. Goes red if the mlx-lm or
    transformers specifier in one place drifts from the other."""
    sources = _mlx_lm_pin_sources()
    for name in ("mlx-lm", "transformers"):
        assert _specifier(sources["mlx-lm extra"], name) == _specifier(sources["dev group"], name)


def test_errors_are_package_rooted() -> None:
    assert issubclass(UnverifiedMlxError, MlxTrainPerfError)
