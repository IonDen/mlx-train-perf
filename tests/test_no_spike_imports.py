"""Mechanical guard: nothing shipped ever references the reference-only 0036 spike tree.

`mlx-train-perf-0036-spike/` is a throwaway feasibility-probe directory outside this
repo's `src`/`tests`/`scripts` -- it must never be imported, copied from, or even
mentioned in a path a committed file carries, or a reader (or a fresh checkout without
that sibling directory) hits a dead reference.
"""

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def test_no_reference_to_the_0036_spike_tree():
    # Catches: a stray "see mlx-train-perf-0036-spike/..." comment, docstring, or import
    # left behind by a prototype-to-production port -- committed code must stand on its
    # own, not point at a directory that is never checked in.
    #
    # THIS file is excluded from its own scan: the banned marker has to appear once,
    # literally, right here, to define what is banned -- without the exclusion the guard
    # would flag itself and could never pass.
    root = _THIS_FILE.parents[1]
    offenders = [
        str(p.relative_to(root))
        for sub in ("src", "tests", "scripts")
        for p in (root / sub).rglob("*.py")
        if p != _THIS_FILE and "0036-spike" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"reference-only spike tree referenced by: {offenders}"
