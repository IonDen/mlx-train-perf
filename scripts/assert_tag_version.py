"""Publish-workflow gate: the pushed tag must be an exact `vX.Y.Z` release tag and the
built dist must contain EXACTLY that version's sdist + wheel, nothing else.

hatch-vcs derives the package version from the git tag, so a mistyped tag (or a build
from a checkout that is not exactly at the tag) produces a dev-suffixed version -- and a
PyPI upload is immutable. This gate fails the workflow before the upload step instead.

Deliberately stdlib-only: `publish.yml`'s Ubuntu build job runs it with the runner's
`python3`, no project environment (mlx does not need to be importable there).

Usage: python3 scripts/assert_tag_version.py <tag> <dist-dir>
Exit codes: 0 gate passed, 1 gate failed, 2 usage error.
"""
import re
import sys
from pathlib import Path

_DIST_NAME = "mlx_train_perf"
_RELEASE_TAG_RE = re.compile(r"v(\d+\.\d+\.\d+)")


def expected_dist_names(tag: str) -> tuple[str, str]:
    """`vX.Y.Z` -> the exact (sdist, wheel) filenames `uv build` produces for it (the
    package is pure Python, so the wheel is always `py3-none-any`). Raises `ValueError`
    for anything that is not an exact three-component release tag -- pre-release, dev,
    or arbitrary suffixes included."""
    m = _RELEASE_TAG_RE.fullmatch(tag)
    if m is None:
        raise ValueError(f"tag {tag!r} is not an exact vX.Y.Z release tag")
    version = m.group(1)
    return (
        f"{_DIST_NAME}-{version}.tar.gz",
        f"{_DIST_NAME}-{version}-py3-none-any.whl",
    )


def check_dist_dir(tag: str, dist_dir: Path) -> list[str]:
    """The gate decision: an empty list means pass; otherwise each entry is one
    human-readable problem (bad tag, missing artifact, or a stowaway file such as a
    dev-versioned build from a checkout not exactly at the tag)."""
    try:
        expected = set(expected_dist_names(tag))
    except ValueError as exc:
        return [str(exc)]
    actual = {p.name for p in dist_dir.iterdir() if p.is_file()}
    problems = [f"missing from {dist_dir}: {name}" for name in sorted(expected - actual)]
    problems += [
        f"unexpected in {dist_dir} (a stale or dev-versioned build?): {name}"
        for name in sorted(actual - expected)
    ]
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: assert_tag_version.py <tag> <dist-dir>", file=sys.stderr)
        return 2
    tag, dist_dir = argv[0], Path(argv[1])
    problems = check_dist_dir(tag, dist_dir)
    for problem in problems:
        print(f"TAG/VERSION GATE: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"TAG/VERSION GATE: ok -- {tag} matches {dist_dir} exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
