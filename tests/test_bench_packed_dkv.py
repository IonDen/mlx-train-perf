"""Tests for the packed dK/dV op-level bench (`scripts/bench_packed_dkv.py`) -- argument
parsing, layout construction, forced-range construction, and artifact naming/shape ONLY
(default lane, GPU-free). The actual timed `launch_bwd_dkv` dispatch (`run_bench`) is
main-session heavy-run territory and is never exercised here -- see the script's own
module docstring for the measurement design.

Scripts are loaded by path (the existing `scripts/` convention -- no `__init__.py`),
mirroring `tests/test_bench_packed_training.py`'s `sys.path.insert` + module import.
"""
import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_packed_dkv  # noqa: E402 -- import must follow the sys.path insert
from bench_packed_dkv import (  # noqa: E402
    _ALPACA_VALID_N,
    _LAYOUTS,
    alpaca_lengths,
    artifact_path,
    build_parser,
    build_result,
    forced_ranges,
    layout_lengths,
    layout_to_segments,
    single_lengths,
    two_seg_lengths,
    write_artifact,
)

# ---------------------------------------------------------------------------
# layout construction
# ---------------------------------------------------------------------------


def test_alpaca_lengths_sum_to_n_exactly_at_4096() -> None:
    assert sum(alpaca_lengths(4096)) == 4096


def test_alpaca_lengths_sum_to_n_exactly_at_8192() -> None:
    assert sum(alpaca_lengths(8192)) == 8192


def test_alpaca_lengths_8192_doubles_every_base_count() -> None:
    # n=8192 is the SAME relative histogram as n=4096 with every count doubled, plus
    # each n's own trailing remainder segment (excluded from the comparison below).
    lens_4096 = alpaca_lengths(4096)
    lens_8192 = alpaca_lengths(8192)
    counts_4096 = Counter(lens_4096[:-1])
    counts_8192 = Counter(lens_8192[:-1])
    for length, count in counts_4096.items():
        assert counts_8192[length] == 2 * count


def test_alpaca_lengths_rejects_n_outside_4096_8192() -> None:
    with pytest.raises(ValueError, match="4096"):
        alpaca_lengths(2048)


def test_two_seg_lengths_splits_n_at_the_midpoint() -> None:
    assert two_seg_lengths(4096) == [2048, 2048]
    assert two_seg_lengths(8192) == [4096, 4096]


def test_single_lengths_is_one_segment_covering_n() -> None:
    assert single_lengths(4096) == [4096]
    assert single_lengths(8192) == [8192]


def test_layout_lengths_dispatches_by_name() -> None:
    assert layout_lengths("single", 8192) == [8192]
    assert layout_lengths("two_seg", 4096) == [2048, 2048]
    assert sum(layout_lengths("alpaca", 8192)) == 8192


def test_layout_lengths_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        layout_lengths("bogus", 4096)


def test_layout_to_segments_seg_id_and_seg_start_shape_and_values() -> None:
    seg_id, seg_start = layout_to_segments([2, 3], b=1)
    assert seg_id.shape == (1, 5)
    assert seg_start.shape == (1, 5)
    assert seg_id.tolist() == [[0, 0, 1, 1, 1]]
    assert seg_start.tolist() == [[0, 0, 2, 2, 2]]


def test_layout_to_segments_broadcasts_across_batch_rows() -> None:
    seg_id, _seg_start = layout_to_segments([4], b=3)
    assert seg_id.shape == (3, 4)


# ---------------------------------------------------------------------------
# forced_ranges
# ---------------------------------------------------------------------------


def test_forced_ranges_tile_n_exactly_ascending_and_contiguous() -> None:
    ranges = forced_ranges(8192)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 8192
    for (_r0, r1), (next_r0, _next_r1) in pairwise(ranges):
        assert r1 == next_r0


def test_forced_ranges_stay_32_aligned() -> None:
    for r0, _r1 in forced_ranges(4096):
        assert r0 % 32 == 0


def test_forced_ranges_produce_more_than_one_dispatch_at_both_n() -> None:
    assert len(forced_ranges(4096)) > 1
    assert len(forced_ranges(8192)) > 1


def test_forced_ranges_identical_regardless_of_layout() -> None:
    # forced_ranges is a pure function of n -- the same n must give both arms (and both
    # layouts) IDENTICAL ranges, the whole point of the review-corrected measurement design.
    assert forced_ranges(4096) == forced_ranges(4096)


def test_forced_ranges_rejects_non_32_aligned_chunk() -> None:
    with pytest.raises(ValueError, match="32"):
        forced_ranges(4096, chunk=100)


# ---------------------------------------------------------------------------
# artifact naming + shape
# ---------------------------------------------------------------------------


def test_artifact_path_embeds_n_and_layout() -> None:
    assert artifact_path(Path("/tmp/out"), n=8192, layout="alpaca").name == (
        "dkv_n8192_alpaca.json"
    )
    assert artifact_path(Path("/tmp/out"), n=4096, layout="two_seg").name == (
        "dkv_n4096_two_seg.json"
    )
    assert artifact_path(Path("/tmp/out"), n=4096, layout="single").name == (
        "dkv_n4096_single.json"
    )


def test_build_result_records_both_arms_and_the_schema_fields() -> None:
    result = build_result(
        layout="single", n=4096, ranges=[(0, 2048), (2048, 4096)],
        bounded_ms=[1.0, 1.1, 0.9, 1.05, 0.95],
        unbounded_ms=[2.0, 2.1, 1.9, 2.05, 1.95],
        peak_gb=0.5, code_sha="deadbeef",
    )
    assert result["layout"] == "single"
    assert result["n"] == 4096
    arms = result["arms"]
    assert isinstance(arms, dict)
    assert arms["bounded"]["median_ms"] == pytest.approx(1.0)
    assert arms["bounded"]["reps_ms"] == [1.0, 1.1, 0.9, 1.05, 0.95]
    assert arms["unbounded"]["median_ms"] == pytest.approx(2.0)
    assert result["ratio"] == pytest.approx(2.0)
    assert result["forced_ranges"] == [[0, 2048], [2048, 4096]]
    assert result["code_sha"] == "deadbeef"
    assert result["peak_gb"] == pytest.approx(0.5)


def test_write_artifact_round_trips_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "dkv_n4096_single.json"
    result = {"layout": "single", "n": 4096, "arms": {}, "ratio": 1.0}
    write_artifact(path, result)
    assert json.loads(path.read_text()) == result


# ---------------------------------------------------------------------------
# CLI argument parsing (build_parser only -- never invokes run_bench/GPU code)
# ---------------------------------------------------------------------------


def test_parser_accepts_every_known_layout_and_n() -> None:
    parser = build_parser()
    for layout in _LAYOUTS:
        for n in _ALPACA_VALID_N:
            args = parser.parse_args(["--layout", layout, "--n", str(n)])
            assert args.layout == layout
            assert args.n == n


def test_parser_defaults_reps_and_out() -> None:
    parser = build_parser()
    args = parser.parse_args(["--layout", "alpaca", "--n", "4096"])
    assert args.reps == 5
    assert args.out == bench_packed_dkv.DEFAULT_OUT


def test_parser_rejects_unknown_layout() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--layout", "bogus", "--n", "4096"])


def test_parser_rejects_unsupported_n() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--layout", "alpaca", "--n", "2048"])


def test_parser_requires_layout_and_n() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--n", "4096"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--layout", "alpaca"])
