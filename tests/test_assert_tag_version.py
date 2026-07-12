"""Tests for `scripts/assert_tag_version.py` -- the publish-workflow gate that refuses
to ship a dist whose built version does not exactly match the pushed `vX.Y.Z` tag (RC
review finding 4: a mistyped or dev-versioned tag must fail BEFORE the PyPI upload, which
is immutable).

`scripts/` has no `__init__.py` (matches `bench_attention_op.py`'s existing convention),
so the module is loaded by path rather than via a package import.
"""
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from assert_tag_version import (  # noqa: E402 -- import must follow the sys.path insert
    check_dist_dir,
    expected_dist_names,
    main,
)

# ---------------------------------------------------------------------------
# expected_dist_names: pure tag -> exact dist-filename mapping
# ---------------------------------------------------------------------------


def test_expected_dist_names_for_a_release_tag() -> None:
    assert expected_dist_names("v0.2.0") == (
        "mlx_train_perf-0.2.0.tar.gz",
        "mlx_train_perf-0.2.0-py3-none-any.whl",
    )


@pytest.mark.parametrize(
    "tag",
    [
        "0.2.0",        # missing the v prefix
        "v0.2",         # not X.Y.Z
        "v0.2.0rc1",    # pre-release suffix
        "v0.2.0.dev1",  # dev suffix
        "v0.2.0-fix",   # arbitrary suffix
        "release",      # not a version at all
    ],
)
def test_expected_dist_names_rejects_non_release_tags(tag: str) -> None:
    with pytest.raises(ValueError, match=r"not an exact vX\.Y\.Z release tag"):
        expected_dist_names(tag)


# ---------------------------------------------------------------------------
# check_dist_dir: the gate decision over a real directory
# ---------------------------------------------------------------------------


def _make_dist(tmp_path: Path, *names: str) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in names:
        (dist / name).write_bytes(b"")
    return dist


def test_check_dist_dir_passes_on_an_exact_match(tmp_path: Path) -> None:
    dist = _make_dist(
        tmp_path,
        "mlx_train_perf-0.2.0.tar.gz",
        "mlx_train_perf-0.2.0-py3-none-any.whl",
    )
    assert check_dist_dir("v0.2.0", dist) == []


def test_check_dist_dir_reports_a_missing_artifact(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path, "mlx_train_perf-0.2.0.tar.gz")
    problems = check_dist_dir("v0.2.0", dist)
    assert len(problems) == 1
    assert "missing" in problems[0]
    assert "mlx_train_perf-0.2.0-py3-none-any.whl" in problems[0]


def test_check_dist_dir_reports_a_dev_versioned_stowaway(tmp_path: Path) -> None:
    # hatch-vcs builds a dev version when the checkout is not exactly at the tag --
    # that build must never reach PyPI.
    dist = _make_dist(
        tmp_path,
        "mlx_train_perf-0.2.0.tar.gz",
        "mlx_train_perf-0.2.0-py3-none-any.whl",
        "mlx_train_perf-0.1.1.dev34+g343d1ce4b.d20260712-py3-none-any.whl",
    )
    problems = check_dist_dir("v0.2.0", dist)
    assert len(problems) == 1
    assert "unexpected" in problems[0]
    assert "dev34" in problems[0]


def test_check_dist_dir_reports_a_bad_tag_as_the_single_problem(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path, "mlx_train_perf-0.2.0.tar.gz")
    problems = check_dist_dir("v0.2.0rc1", dist)
    assert len(problems) == 1
    assert "not an exact vX.Y.Z release tag" in problems[0]


# ---------------------------------------------------------------------------
# main: thin CLI shell over check_dist_dir
# ---------------------------------------------------------------------------


def test_main_returns_0_on_a_clean_gate(tmp_path: Path) -> None:
    dist = _make_dist(
        tmp_path,
        "mlx_train_perf-0.2.0.tar.gz",
        "mlx_train_perf-0.2.0-py3-none-any.whl",
    )
    assert main(["v0.2.0", str(dist)]) == 0


def test_main_returns_1_on_a_gate_failure(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path, "mlx_train_perf-0.2.0.tar.gz")
    assert main(["v0.2.0", str(dist)]) == 1


def test_main_returns_2_on_usage_error() -> None:
    assert main(["v0.2.0"]) == 2
