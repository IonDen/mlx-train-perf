"""GPU-free tests for `scripts/bench_attention_marginal.py`.

`scripts/` has no `__init__.py`, so the module is loaded by path (the
`test_bench_attention_op.py` convention). Nothing here dispatches a Metal kernel: the
mask builder runs on the CPU stream, and the orchestrator runs with a fake spawner that
writes the record a worker would have written.

Each test names the one-line bug it exists to catch.
"""
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

import mlx.core as mx
import pytest

from mlx_train_perf import _compat
from mlx_train_perf.attention import api
from mlx_train_perf.attention.kernel import launch
from mlx_train_perf.errors import UnverifiedMlxError

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_attention_marginal as bam  # noqa: E402 -- import must follow the sys.path insert

SHA = "0123456789abcdef"
VER = "0.32.4"
GIB = 1024**3
# The current run's identity as `unit_is_done` and `run_units` see it.
RUN: dict[str, Any] = {"source": None, "bypassed": False, "reps": 3}


def _rep(pre: int, peak: int, wall: float) -> dict[str, Any]:
    return {"pre_bytes": pre, "peak_bytes": peak, "marginal_bytes": peak - pre,
            "wall_s": wall}


def _ok_record(*, mode: str = "causal", impl: str = "flash", n: int = 1024,
               marginal_median: int = 100, script_sha: str = SHA,
               mlx_version: str = VER, source: str | None = None, bypassed: bool = False,
               reps_requested: int = 3) -> dict[str, Any]:
    reps = [_rep(1000, 1000 + marginal_median, 0.02)] * 3
    return {
        "schema": bam.SCHEMA, "unit": bam.unit_name(mode, impl, n), "mode": mode,
        "impl": impl, "n": n, "status": "ok", "mlx_version": mlx_version,
        "script_sha": script_sha, "reps": reps, "grads_finite": True,
        "reps_requested": reps_requested, "verification_gate_bypassed": bypassed,
        "mlx_train_perf_source": source,
        "marginal_median_bytes": marginal_median, "wall_median_s": 0.02,
    }


# ------------------------------------------------------------------- marginal + median


def test_marginal_is_peak_minus_pre() -> None:
    # Bug caught: `pre - peak` (sign flip) or returning `peak` (total, not marginal).
    assert bam.marginal_bytes(pre=1_000, peak=1_750) == 750


def test_rep_summary_takes_the_median_not_the_mean_or_the_first() -> None:
    # Bug caught: `mean(...)` (would give 400) or `reps[0]` (would give 1000).
    reps = [_rep(0, 1000, 0.5), _rep(0, 100, 0.1), _rep(0, 100, 0.3)]
    s = bam.summarize_reps(reps)
    assert s["marginal_median_bytes"] == 100
    assert s["wall_median_s"] == pytest.approx(0.3)
    assert s["n_reps"] == 3


def test_rep_summary_median_of_an_even_count_averages_the_middle_pair() -> None:
    # Bug caught: `sorted(x)[len(x) // 2]` (upper middle, 300) instead of the median 200.
    reps = [_rep(0, 100, 0.1), _rep(0, 300, 0.1), _rep(0, 100, 0.1), _rep(0, 300, 0.1)]
    assert bam.summarize_reps(reps)["marginal_median_bytes"] == 200


def test_rep_summary_refuses_an_empty_rep_list() -> None:
    # Bug caught: a crashed measure loop silently producing a record with no reps.
    with pytest.raises(ValueError, match="no reps"):
        bam.summarize_reps([])


# ------------------------------------------------------------------- stock/flash ratio


def test_ratio_is_stock_over_flash_per_mode_and_n() -> None:
    # Bug caught: flash/stock (inverted: 0.874) or mixing a causal and a packed row.
    recs = [
        _ok_record(impl="flash", n=8192, marginal_median=596),
        _ok_record(impl="stock", n=8192, marginal_median=682),
        _ok_record(mode="packed", impl="flash", n=8192, marginal_median=600),
        _ok_record(mode="packed", impl="stock", n=8192, marginal_median=6000),
    ]
    ratios = bam.stock_over_flash(recs)
    assert ratios["causal-n8192"] == pytest.approx(1.144)
    assert ratios["packed-n8192"] == pytest.approx(10.0)


def test_ratio_is_none_unless_both_arms_are_ok() -> None:
    # Bug caught: dividing by an aborted record's leftover number, or a KeyError on a
    # missing arm instead of an honest null.
    aborted = {**_ok_record(impl="stock", n=4096), "status": "aborted_watchdog"}
    recs = [_ok_record(impl="flash", n=4096), aborted, _ok_record(impl="flash", n=2048)]
    ratios = bam.stock_over_flash(recs)
    assert ratios["causal-n4096"] is None
    assert ratios["causal-n2048"] is None


def test_ratio_is_none_when_flash_marginal_is_zero() -> None:
    # Bug caught: ZeroDivisionError aborting the summary of an otherwise finished run.
    recs = [_ok_record(impl="flash", n=1024, marginal_median=0),
            _ok_record(impl="stock", n=1024, marginal_median=10)]
    assert bam.stock_over_flash(recs)["causal-n1024"] is None


# ------------------------------------------------------------------- schema validation


def test_a_complete_ok_record_validates() -> None:
    # Bug caught: a validator so strict it rejects the records the worker writes.
    assert bam.validate_unit_record(_ok_record()) == []


def test_validation_flags_fewer_than_three_reps() -> None:
    # Bug caught: dropping the >=3 measured reps rule the public number depends on.
    rec = _ok_record()
    rec["reps"] = rec["reps"][:2]
    assert any("reps" in p for p in bam.validate_unit_record(rec))


def test_validation_flags_a_median_that_does_not_match_the_reps() -> None:
    # Bug caught: a hand-edited or stale median that no longer follows from the raw reps.
    rec = _ok_record(marginal_median=100)
    rec["marginal_median_bytes"] = 999
    assert any("median" in p for p in bam.validate_unit_record(rec))


def test_validation_flags_a_rep_whose_marginal_is_not_peak_minus_pre() -> None:
    # Bug caught: a rep recording `peak` as its marginal (pre included).
    rec = _ok_record()
    rec["reps"] = [*rec["reps"][:2], {**rec["reps"][2], "marginal_bytes": 5}]
    assert any("peak - pre" in p for p in bam.validate_unit_record(rec))


def test_validation_flags_a_missing_required_field() -> None:
    # Bug caught: a record without the mlx version being accepted into the table.
    rec = _ok_record()
    del rec["mlx_version"]
    assert any("mlx_version" in p for p in bam.validate_unit_record(rec))


def test_an_aborted_record_validates_without_reps() -> None:
    # Bug caught: requiring reps on an abort record, so an honest abort is reported as
    # schema-invalid instead of as an abort.
    rec = {"schema": bam.SCHEMA, "unit": "causal-stock-n8192", "mode": "causal",
           "impl": "stock", "n": 8192, "status": "aborted_watchdog", "mlx_version": VER,
           "script_sha": SHA}
    assert bam.validate_unit_record(rec) == []


def test_validation_flags_an_ok_record_whose_gradients_are_not_finite() -> None:
    # Bug caught: a NaN/inf backward accepted as an `ok` measurement (the marginal of a
    # broken kernel served as a result), including an ok record with no finiteness field.
    nonfinite = {**_ok_record(), "grads_finite": False}
    assert any("finite" in p for p in bam.validate_unit_record(nonfinite))
    unchecked = _ok_record()
    del unchecked["grads_finite"]
    assert any("finite" in p for p in bam.validate_unit_record(unchecked))


def test_non_finite_gradients_give_a_status_outside_the_done_set() -> None:
    # Bug caught: the worker writing status `ok` whatever `grads_finite` says, so a resume
    # skips the unit and the exit code reports success.
    assert bam.measured_status(grads_finite=True) == "ok"
    assert bam.measured_status(grads_finite=False) == "nonfinite"
    assert "nonfinite" not in bam.DONE_STATUSES


# ------------------------------------------------------------------- resume skip


def test_unit_is_done_for_ok_and_aborts_from_this_script_and_mlx() -> None:
    # Bug caught: re-running finished units on resume (or never re-running anything).
    assert bam.unit_is_done(_ok_record(), script_sha=SHA, mlx_version=VER, **RUN)
    aborted = {**_ok_record(), "status": "aborted_watchdog"}
    assert bam.unit_is_done(aborted, script_sha=SHA, mlx_version=VER, **RUN)


def test_unit_is_not_done_for_an_error_a_missing_file_or_another_script_or_mlx() -> None:
    # Bug caught: a crashed unit's error record, or a record from an edited script or a
    # different mlx, being served as a result.
    assert not bam.unit_is_done(None, script_sha=SHA, mlx_version=VER, **RUN)
    assert not bam.unit_is_done({**_ok_record(), "status": "error"}, script_sha=SHA,
                                mlx_version=VER, **RUN)
    assert not bam.unit_is_done(_ok_record(script_sha="ffffffffffffffff"), script_sha=SHA,
                                mlx_version=VER, **RUN)
    assert not bam.unit_is_done(_ok_record(mlx_version="0.32.0"), script_sha=SHA,
                                mlx_version=VER, **RUN)


def test_unit_is_not_done_for_a_record_from_another_source_tree() -> None:
    # Bug caught: a re-run with `--src` on edited code skipping every unit the installed
    # package (or an older tree) measured, so the new code is never measured.
    installed = _ok_record(source=None)
    assert not bam.unit_is_done(installed, script_sha=SHA, mlx_version=VER,
                                source="src@aaaaaaaaaaaaaaaa", bypassed=False, reps=3)
    older = _ok_record(source="src@aaaaaaaaaaaaaaaa")
    assert not bam.unit_is_done(older, script_sha=SHA, mlx_version=VER,
                                source="src@bbbbbbbbbbbbbbbb", bypassed=False, reps=3)
    assert bam.unit_is_done(older, script_sha=SHA, mlx_version=VER,
                            source="src@aaaaaaaaaaaaaaaa", bypassed=False, reps=3)


def test_unit_is_not_done_when_the_bypass_flag_differs() -> None:
    # Bug caught: a bypassed (measurement-only) record served as the result of a normal
    # run, or the reverse.
    bypassed = _ok_record(bypassed=True)
    assert not bam.unit_is_done(bypassed, script_sha=SHA, mlx_version=VER, **RUN)
    assert bam.unit_is_done(bypassed, script_sha=SHA, mlx_version=VER, source=None,
                            bypassed=True, reps=3)
    unflagged = _ok_record()
    del unflagged["verification_gate_bypassed"]
    assert not bam.unit_is_done(unflagged, script_sha=SHA, mlx_version=VER, **RUN)


def test_unit_is_not_done_when_the_requested_reps_differ() -> None:
    # Bug caught: `--reps 7` silently keeping the 3-rep records of an earlier run.
    assert not bam.unit_is_done(_ok_record(reps_requested=3), script_sha=SHA,
                                mlx_version=VER, source=None, bypassed=False, reps=7)
    assert bam.unit_is_done(_ok_record(reps_requested=7), script_sha=SHA,
                            mlx_version=VER, source=None, bypassed=False, reps=7)


def test_an_ok_record_with_non_finite_gradients_is_not_done() -> None:
    # Bug caught: resume skipping a unit whose stored gradients were NaN/inf.
    assert not bam.unit_is_done({**_ok_record(), "grads_finite": False}, script_sha=SHA,
                                mlx_version=VER, **RUN)


class _FakeSpawner:
    """Stands in for the worker subprocess: writes the record a worker would write, or
    nothing, and reports the outcome the parent would see."""

    def __init__(self, *, write: bool = True, returncode: int = 0,
                 timed_out: bool = False) -> None:
        self.write, self.returncode, self.timed_out = write, returncode, timed_out
        self.spawned: list[str] = []

    def __call__(self, unit: bam.Unit, out_path: Path, _log_path: Path) -> bam.WorkerOutcome:
        self.spawned.append(unit.name)
        if self.write:
            out_path.write_text(json.dumps(
                _ok_record(mode=unit.mode, impl=unit.impl, n=unit.n)))
        return bam.WorkerOutcome(returncode=self.returncode, timed_out=self.timed_out,
                                 duration_s=1.0, stderr_tail="boom")


def _units() -> list[bam.Unit]:
    return bam.build_units(modes=["causal"], impls=["flash", "stock"], seq_lens=[1024])


def test_run_units_skips_a_done_unit_and_spawns_the_rest(tmp_path: Path) -> None:
    # Bug caught: the resume check ignored, so a finished unit is measured again.
    (tmp_path / "causal-flash-n1024.json").write_text(json.dumps(_ok_record()))
    spawn = _FakeSpawner()
    bam.run_units(_units(), out_dir=tmp_path, spawn=spawn, script_sha=SHA, mlx_version=VER,
                  **RUN)
    assert spawn.spawned == ["causal-stock-n1024"]


def test_run_units_reruns_a_unit_recorded_from_another_source(tmp_path: Path) -> None:
    # Bug caught: `run_units` checking resume against script and mlx only, so a `--src`
    # re-run on new code keeps the old code's numbers.
    (tmp_path / "causal-flash-n1024.json").write_text(json.dumps(_ok_record(source=None)))
    spawn = _FakeSpawner()
    bam.run_units(_units(), out_dir=tmp_path, spawn=spawn, script_sha=SHA, mlx_version=VER,
                  source="src@aaaaaaaaaaaaaaaa", bypassed=False, reps=3)
    assert spawn.spawned == ["causal-flash-n1024", "causal-stock-n1024"]


def test_run_units_moves_a_stale_record_aside_instead_of_deleting_it(tmp_path: Path) -> None:
    # Bug caught: a stale record unlinked (lost) or left in place (then read as this
    # attempt's handoff even if the worker wrote nothing).
    stale = {**_ok_record(), "status": "error"}
    (tmp_path / "causal-flash-n1024.json").write_text(json.dumps(stale))
    bam.run_units(_units(), out_dir=tmp_path, spawn=_FakeSpawner(write=False, returncode=1),
                  script_sha=SHA, mlx_version=VER, **RUN)
    kept = list((tmp_path / "attempts").iterdir())
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["status"] == "error"


def test_run_units_records_an_error_when_the_worker_leaves_no_record(tmp_path: Path) -> None:
    # Bug caught: a crashed worker leaving no trace, so the summary silently has a hole.
    paths = bam.run_units(_units(), out_dir=tmp_path,
                          spawn=_FakeSpawner(write=False, returncode=1),
                          script_sha=SHA, mlx_version=VER, **RUN)
    rec = json.loads(paths[0].read_text())
    assert rec["status"] == "error"
    assert rec["unit_exit"] == 1
    assert "boom" in rec["stderr_tail"]


def test_run_units_records_a_wall_abort_when_the_parent_stopped_the_worker(
    tmp_path: Path,
) -> None:
    # Bug caught: a wall-limit stop recorded as a generic error (retried forever) instead
    # of the honest abort it is.
    paths = bam.run_units(_units(), out_dir=tmp_path,
                          spawn=_FakeSpawner(write=False, returncode=-15, timed_out=True),
                          script_sha=SHA, mlx_version=VER, **RUN)
    assert json.loads(paths[0].read_text())["status"] == "aborted_wall"


def test_a_parent_wall_abort_record_is_done_on_the_next_run(tmp_path: Path) -> None:
    # Bug caught: the parent's own abort record lacking the run identity (source, bypass,
    # reps), so it never counts as done and the unit that hit the wall runs again forever.
    run = {"source": "src@aaaaaaaaaaaaaaaa", "bypassed": True, "reps": 5}
    paths = bam.run_units(_units(), out_dir=tmp_path,
                          spawn=_FakeSpawner(write=False, returncode=-15, timed_out=True),
                          script_sha=SHA, mlx_version=VER, **run)
    rec = json.loads(paths[0].read_text())
    assert bam.unit_is_done(rec, script_sha=SHA, mlx_version=VER, **run)


def test_build_units_orders_n_ascending_flash_before_stock() -> None:
    # Bug caught: stock at the largest N running first, before the cheap units have a
    # record (the heaviest unit belongs last).
    units = bam.build_units(modes=["causal"], impls=["stock", "flash"], seq_lens=[2048, 1024])
    assert [u.name for u in units] == [
        "causal-flash-n1024", "causal-stock-n1024", "causal-flash-n2048", "causal-stock-n2048",
    ]


# ------------------------------------------------------------------- packed layout


@pytest.mark.parametrize("n", [1024, 4096, 8192])
def test_segment_lengths_fill_n_exactly_within_the_range(n: int) -> None:
    # Bug caught: a final segment that overshoots N, or a tail shorter than 256 tokens.
    lens = bam.segment_lengths(n, seed=0)
    assert sum(lens) == n
    assert all(256 <= ln <= 2048 for ln in lens)


def test_segment_lengths_are_mixed_at_the_bench_sizes() -> None:
    # Bug caught: a degenerate generator emitting one segment, or one repeated length.
    lens = bam.segment_lengths(8192, seed=0)
    assert len(lens) >= 4
    assert len(set(lens)) > 1


def test_segment_lengths_are_deterministic_for_a_seed() -> None:
    # Bug caught: drawing from the global `random` state (or the clock) instead of a
    # seeded generator, so the pinned layout drifts between runs.
    assert bam.segment_lengths(8192, seed=0) == bam.segment_lengths(8192, seed=0)
    assert bam.segment_lengths(8192, seed=0) != bam.segment_lengths(8192, seed=1)


def test_segment_lengths_refuse_an_n_below_the_shortest_segment() -> None:
    # Bug caught: returning a 100-token "segment" below the pinned 256 minimum.
    with pytest.raises(ValueError, match="256"):
        bam.segment_lengths(100, seed=0)


def test_segment_ids_and_starts_follow_the_lengths() -> None:
    # Bug caught: seg_start holding the segment index, or ids not restarting at 0.
    seg_id, seg_start = bam.segment_ids([2, 3, 1])
    assert seg_id == [0, 0, 1, 1, 1, 2]
    assert seg_start == [0, 0, 2, 2, 2, 5]


def test_packed_mask_is_block_diagonal_and_causal_within_segments() -> None:
    # Bug caught: a mask that is causal but not block-diagonal (cross-segment leak), or
    # block-diagonal but not causal (future-token leak), or transposed (keys as rows).
    with mx.stream(mx.cpu):
        mask = bam.packed_mask([2, 3, 1])
        assert mask.shape == (1, 1, 6, 6)
        assert mask.dtype == mx.bool_
        rows = mask[0, 0].tolist()
    t, f = True, False
    assert rows == [
        [t, f, f, f, f, f],
        [t, t, f, f, f, f],
        [f, f, t, f, f, f],
        [f, f, t, t, f, f],
        [f, f, t, t, t, f],
        [f, f, f, f, f, t],
    ]


# ------------------------------------------------------------------- misc pure helpers


def test_wired_limit_stays_strictly_below_the_recommendation() -> None:
    # Bug caught: wiring the full recommended working set (the limit must sit below it).
    gib = 1024**3
    assert bam.wired_limit_for(32 * gib) == 20 * gib
    assert bam.wired_limit_for(16 * gib) == 14 * gib
    assert bam.wired_limit_for(3 * gib) < 3 * gib


def test_watchdog_ceiling_is_the_requested_limit_on_a_large_machine() -> None:
    # Bug caught: the ceiling always set to memory - 4 GiB, ignoring --watchdog-gib.
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=64 * GIB) == 22 * GIB


def test_watchdog_ceiling_is_memory_minus_4_gib_on_a_small_machine() -> None:
    # Bug caught: the fixed 22 GiB ceiling on a 16 or 24 GB Mac, which pages long before
    # the watchdog trips.
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=16 * GIB) == 12 * GIB
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=24 * GIB) == 20 * GIB


def test_watchdog_ceiling_boundary_between_the_two_terms() -> None:
    # Bug caught: `max` for `min`, or a 4 GiB margin off by a byte (`>=` vs `>`).
    mem = 26 * GIB  # memory - 4 GiB == the request
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=mem) == 22 * GIB
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=mem + 1) == 22 * GIB
    assert bam.watchdog_ceiling_bytes(requested_gib=22.0, memory_size=mem - 1) == 22 * GIB - 1


def test_the_watchdog_record_keeps_the_requested_and_the_effective_ceiling() -> None:
    # Bug caught: only one of the two ceilings recorded, so a record from a small machine
    # cannot show that its limit was lowered below --watchdog-gib.
    fields = bam.watchdog_fields(requested_gib=22.0, memory_size=16 * GIB, wall_budget_s=270.0)
    assert fields["requested_limit_bytes"] == 22 * GIB
    assert fields["limit_bytes"] == 12 * GIB
    assert fields["device_margin_bytes"] == 4 * GIB
    assert fields["wall_budget_s"] == 270.0


def test_summary_table_and_ratios_come_from_valid_records() -> None:
    # Bug caught: results.json built from schema-invalid records, or the invalid ones
    # dropped without a trace.
    bad = _ok_record(impl="stock", n=8192)
    bad["reps"] = bad["reps"][:1]
    recs = [_ok_record(impl="flash", n=8192, marginal_median=596), bad]
    summary = bam.summarize(recs, source=None, bypassed=False)
    assert summary["ratio_stock_over_flash"]["causal-n8192"] is None
    assert summary["invalid_units"] == {"causal-stock-n8192": bam.validate_unit_record(bad)}
    assert [r["unit"] for r in summary["units"]] == ["causal-flash-n8192"]


def test_summary_refuses_records_from_another_source_or_bypass_setting() -> None:
    # Bug caught: results.json mixing records measured on different code (or past the
    # gate and not), with the machine context taken from whichever record sorts first.
    current = _ok_record(impl="flash", n=4096, source="src@aaaaaaaaaaaaaaaa")
    other_src = _ok_record(impl="stock", n=4096, source="src@bbbbbbbbbbbbbbbb")
    bypassed = _ok_record(impl="flash", n=8192, source="src@aaaaaaaaaaaaaaaa", bypassed=True)
    summary = bam.summarize([other_src, current, bypassed], source="src@aaaaaaaaaaaaaaaa",
                            bypassed=False)
    assert [r["unit"] for r in summary["units"]] == ["causal-flash-n4096"]
    assert set(summary["invalid_units"]) == {"causal-stock-n4096", "causal-flash-n8192"}
    assert any("mlx_train_perf_source" in p
               for p in summary["invalid_units"]["causal-stock-n4096"])
    assert any("verification_gate_bypassed" in p
               for p in summary["invalid_units"]["causal-flash-n8192"])
    assert summary["context"]["mlx_train_perf_source"] == "src@aaaaaaaaaaaaaaaa"
    assert summary["ratio_stock_over_flash"]["causal-n4096"] is None


def test_mlx_train_perf_version_is_null_under_src() -> None:
    # Bug caught: the installed distribution's version recorded while `--src` measured a
    # source tree, so the record names code it did not run.
    assert bam.package_versions(src_given=True)["mlx_train_perf_version"] is None
    assert bam.package_versions(src_given=False)["mlx_train_perf_version"] == (
        importlib.metadata.version("mlx-train-perf"))


# ------------------------------------------------------------------- flash dispatch plan

# The cache key the flash path writes for this bench's shape (head_dim 128, bf16, causal,
# b=1, hq=32, n-bucket, tile variant, d_slab, packed).
_KEY_8192 = (128, "mlx.core.bfloat16", True, 1, 32, 8192, "mma", 128, False)


def test_the_dispatch_plan_reads_the_cached_rates_tile_and_dkv_range_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: the dq rate (or the forward rate) read as the dK/dV rate, the dK/dV
    # plan made without the mma body's 32-row block alignment, or the packed rates read
    # for a causal unit.
    # A dK/dV rate where the 32-row alignment changes the range count (unaligned: fewer).
    fwd, dq, dkv = 3.0e11, 4.0e11, 3.0e10
    monkeypatch.setattr(launch, "_FWD_RATE_CACHE", {_KEY_8192: fwd})
    monkeypatch.setattr(launch, "_BWD_DQ_RATE_CACHE", {_KEY_8192: dq})
    packed_key = (*_KEY_8192[:-1], True)
    monkeypatch.setattr(launch, "_BWD_DKV_RATE_CACHE", {_KEY_8192: dkv, packed_key: 9.9e12})
    plan = bam.read_dispatch_plan(n=8192, packed=False)
    assert plan["rates_macs_per_s"] == {"fwd": fwd, "dq": dq, "dkv": dkv}
    assert plan["dkv_tile"] == {"variant": "mma", "d_slab": 128, "provisional": False}
    expected = len(launch.plan_dkv_dispatches(n=8192, d=128, b=1, hq=32, hkv=8, rate=dkv,
                                              causal=True, block_align=32))
    assert expected > 1
    assert expected != len(launch.plan_dkv_dispatches(n=8192, d=128, b=1, hq=32, hkv=8,
                                                      rate=dkv, causal=True, block_align=1))
    assert plan["dkv_block_align"] == 32
    assert plan["dkv_ranges"] == expected
    assert plan["dkv_ranges"] != len(launch.plan_dkv_dispatches(
        n=8192, d=128, b=1, hq=32, hkv=8, rate=dq, causal=True, block_align=32))


def test_the_dispatch_plan_records_null_for_a_rate_the_process_did_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: a missing cache entry raising (losing the measured record) or being
    # filled in by a fresh calibration outside the warm-up.
    monkeypatch.setattr(launch, "_FWD_RATE_CACHE", {})
    monkeypatch.setattr(launch, "_BWD_DQ_RATE_CACHE", {})
    monkeypatch.setattr(launch, "_BWD_DKV_RATE_CACHE", {})
    plan = bam.read_dispatch_plan(n=4096, packed=True)
    assert plan["rates_macs_per_s"] == {"fwd": None, "dq": None, "dkv": None}
    assert plan["dkv_ranges"] is None
    assert launch._BWD_DKV_RATE_CACHE == {}


# ------------------------------------------------------------------- recorded paths

_FAKE_HOME = Path("/home/someone")
_FAKE_REPO = _FAKE_HOME / "Work" / "mlx-train-perf"


def test_a_record_built_with_repo_internal_paths_stores_them_relative() -> None:
    # Bug caught: `worker_command` written from `sys.orig_argv` as is, so the interpreter,
    # the script, the --out record path and the nested --command-line carry the absolute
    # checkout path (home directory and user name included).
    parent_cmd = [f"{_FAKE_REPO}/.venv/bin/python", "-I",
                  f"{_FAKE_REPO}/scripts/bench_attention_marginal.py", "--mode", "causal"]
    worker_argv = [
        f"{_FAKE_REPO}/.venv/bin/python", "-I", "-B",
        f"{_FAKE_REPO}/scripts/bench_attention_marginal.py", "--single-unit",
        "--out", f"{_FAKE_REPO}/_artifacts/bench_attention_marginal/mlx-0.32.4/x.json",
        "--command-line", json.dumps(parent_cmd),
    ]
    rec = bam.provenance(orig_argv=worker_argv, executable=f"{_FAKE_REPO}/.venv/bin/python",
                         command_line=bam.portable_argv(parent_cmd, repo_root=_FAKE_REPO,
                                                        home=_FAKE_HOME),
                         repo_root=_FAKE_REPO, home=_FAKE_HOME)
    expected_parent = [".venv/bin/python", "-I", "scripts/bench_attention_marginal.py",
                       "--mode", "causal"]
    assert rec["interpreter"] == ".venv/bin/python"
    assert rec["command_line"] == expected_parent
    assert rec["worker_command"] == [
        ".venv/bin/python", "-I", "-B", "scripts/bench_attention_marginal.py",
        "--single-unit", "--out", "_artifacts/bench_attention_marginal/mlx-0.32.4/x.json",
        "--command-line", json.dumps(expected_parent),
    ]
    assert isinstance(rec["python_version"], str)
    assert rec["python_version"].count(".") == 2


def test_an_outside_path_under_home_becomes_tilde() -> None:
    # Bug caught: a path outside the checkout (an --out-dir elsewhere, a uv cache
    # interpreter's site-packages in a traceback) written with the home directory spelled
    # out, or a sibling directory `/home/someone2` mangled into `~2`.
    assert bam.portable_path(f"{_FAKE_HOME}/.cache/uv/archive-v0/abc/bin/python",
                             repo_root=_FAKE_REPO, home=_FAKE_HOME) == (
        "~/.cache/uv/archive-v0/abc/bin/python")
    assert bam.portable_path("/home/someone2/x", repo_root=_FAKE_REPO,
                             home=_FAKE_HOME) == "<path>"


def test_an_absolute_path_outside_repo_and_home_becomes_a_placeholder() -> None:
    # Bug caught: only the repo and home prefixes rewritten, so a `/private/tmp` out-dir,
    # an external volume, a Homebrew traceback frame or a symlinked script path reaches
    # the committed record verbatim.
    def p(text: str) -> str:
        return bam.scrub_text(text, repo_root=_FAKE_REPO, home=_FAKE_HOME)

    assert p("/private/tmp/run1/causal-flash-n1024.json") == "<path>"
    assert p("/Volumes/Data/bench") == "<path>"
    assert p('  File "/opt/homebrew/lib/python3.12/a.py", line 3') == (
        '  File "<path>", line 3')
    assert p("--out=/private/tmp/r.json --x") == "--out=<path> --x"
    assert p(json.dumps(["/tmp/link-to-repo/scripts/b.py", "-I"])) == (
        json.dumps(["<path>", "-I"]))
    assert p(f"{_FAKE_REPO}/scripts/b.py and {_FAKE_HOME}/x") == "scripts/b.py and ~/x"


def test_the_scrub_leaves_relative_paths_ratios_and_urls_alone() -> None:
    # Bug caught: a path pattern so loose it rewrites units (`G MAC/s`), ratios
    # (`stock/flash`), relative or `~` paths, or a URL.
    text = ("rate 695 G MAC/s; stock/flash 1.14; 1/2; scripts/x.py; ./a/b; ~/c/d; "
            "https://ineshin.space/papers/x/")
    assert bam.scrub_text(text, repo_root=_FAKE_REPO, home=_FAKE_HOME) == text


def test_an_interpreter_outside_the_repo_is_recorded_as_python() -> None:
    # Bug caught: the isolated leg's uv cache interpreter (or a Homebrew framework path)
    # recorded in full, so the record names the machine's cache layout and user.
    for exe in (f"{_FAKE_HOME}/.cache/uv/builds-v0/.tmpX/bin/python",
                "/opt/homebrew/Cellar/python@3.12/3.12.12/bin/python3.12"):
        assert bam.portable_interpreter(exe, repo_root=_FAKE_REPO, home=_FAKE_HOME) == "python"


def test_a_written_record_holds_no_home_directory(tmp_path: Path) -> None:
    # Bug caught: a record field added later with a raw local path (here a parent error
    # record whose stderr tail is a traceback) reaching the committed JSON, because only
    # the argv fields were made portable and the writer serializes whatever it gets.
    home = str(Path.home())
    repo = str(bam.REPO_ROOT)
    tail = (f'Traceback (most recent call last):\n  File "{repo}/scripts/x.py", line 1\n'
            f'  File "{home}/.cache/uv/archive-v0/q/lib/python3.12/site-packages/mlx/a.py"\n')
    record = {
        **bam._parent_record(bam.Unit(mode="causal", impl="flash", n=1024), status="error",
                             script_sha=SHA, mlx_version=VER,
                             outcome=bam.WorkerOutcome(returncode=1, timed_out=False,
                                                       duration_s=1.0, stderr_tail=tail),
                             **RUN),
        "worker_command": [sys.executable, f"{repo}/scripts/x.py", "--out",
                           f"{repo}/_artifacts/y.json", "--out-dir", f"{home}/elsewhere"],
        "nested": {"paths": [f"{home}/a", repo]},
    }
    out = tmp_path / "r.json"
    bam._atomic_write(out, record)
    text = out.read_text()
    assert home not in text
    assert f"/{Path.home().name}/" not in text
    assert "~/elsewhere" in text
    assert '"scripts/x.py"' in text


def test_a_worker_log_is_rewritten_without_the_home_directory(tmp_path: Path) -> None:
    # Bug caught: the per-unit log (tracked next to the records) keeping a worker
    # traceback's absolute paths, and the stderr tail taken from the unscrubbed text.
    home = str(Path.home())
    log = tmp_path / "u.log"
    log.write_text(f'  File "{bam.REPO_ROOT}/scripts/x.py"\n  File "{home}/lib/y.py"\n')
    tail = bam._scrub_log(log)
    assert home not in log.read_text()
    assert tail == log.read_text()
    assert tail == '  File "scripts/x.py"\n  File "~/lib/y.py"\n'


def test_a_log_moved_to_attempts_is_scrubbed_first(tmp_path: Path) -> None:
    # Bug caught: a log left by an interrupted parent kept in attempts/ with the home
    # directory in it.
    home = str(Path.home())
    log = tmp_path / "logs" / "u.log"
    log.parent.mkdir()
    log.write_text(f'  File "{home}/lib/y.py"\n')
    bam._move_aside(log, tmp_path / "attempts")
    (kept,) = (tmp_path / "attempts").iterdir()
    assert kept.read_text() == '  File "~/lib/y.py"\n'


class _InterruptedPopen:
    """A worker process that writes a traceback line into its log, then the parent is
    interrupted while waiting for it (what `_on_signal` does on SIGTERM/SIGINT)."""

    def __init__(self, _cmd: list[str], *, stdout: IO[bytes], **_kw: Any) -> None:
        stdout.write(f'  File "{Path.home()}/lib/y.py"\n'.encode())
        stdout.flush()
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002 -- Popen's signature
        raise SystemExit(143)

    def poll(self) -> int:
        return self.returncode


def test_an_interrupted_unit_still_leaves_a_scrubbed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: the log scrub running only after a normal wait, so a SIGTERM or Ctrl-C
    # during a unit leaves the raw log (home directory included) for attempts/.
    monkeypatch.setattr(bam.subprocess, "Popen", _InterruptedPopen)
    spawn = bam.make_spawner(reps=3, watchdog_gib=22.0, unit_wall_s=60.0, mlx_version=VER,
                             command_line=[])
    log = tmp_path / "u.log"
    with pytest.raises(SystemExit):
        spawn(bam.Unit(mode="causal", impl="flash", n=1024), tmp_path / "u.json", log)
    assert log.read_text() == '  File "~/lib/y.py"\n'


def test_attempts_dirs_under_the_bench_artifacts_are_gitignored() -> None:
    # Bug caught: moved-aside records and logs (unscrubbed history) showing up as
    # untracked files under the tracked artifact directory, one `git add` from a commit.
    if shutil.which("git") is None or not (_REPO / ".git").exists():
        pytest.skip("needs a git checkout")
    probe = "_artifacts/bench_attention_marginal/mlx-0.32.4/attempts/causal-flash-n1024.log"
    kept = "_artifacts/bench_attention_marginal/mlx-0.32.4/causal-flash-n1024.json"
    ignored = subprocess.run(["git", "-C", str(_REPO), "check-ignore", "-q", "--no-index",
                              probe], check=False)
    tracked = subprocess.run(["git", "-C", str(_REPO), "check-ignore", "-q", "--no-index",
                              kept], check=False)
    assert ignored.returncode == 0
    assert tracked.returncode == 1


def test_the_ci_local_path_check_covers_the_bench_artifacts() -> None:
    # Bug caught: the CI grep for local paths scanning only the older rung directories, so
    # a record from this bench with a home path passes CI.
    ci = (_REPO / ".github" / "workflows" / "ci.yml").read_text()
    start = ci.index("name: Tracked artifacts carry no local paths")
    step = ci[start:ci.index("\n      - ", start)]
    covered = [w.rstrip("/;") for w in step.split()]
    assert "_artifacts/bench_attention_marginal" in covered or "_artifacts" in covered


# ------------------------------------------------------------------- verification-gate bypass

_VERIFIED = frozenset({"0.31.2", "0.32.0"})


def test_the_gate_bypass_is_off_by_default() -> None:
    # Bug caught: `default=True` (or a store_false), so every run measures past the gate.
    assert bam._build_parser().parse_args([]).bypass_verification_gate is False


def test_the_gate_bypass_is_refused_on_a_verified_mlx() -> None:
    # Bug caught: the bypass accepted where the gate already passes, so a record claims a
    # bypass that changed nothing (or the check inverted, refusing the unverified case).
    refusal = bam.bypass_refusal(bypass=True, installed="0.32.0", verified=_VERIFIED)
    assert refusal is not None
    assert "0.32.0" in refusal
    assert bam.bypass_refusal(bypass=True, installed="0.32.4", verified=_VERIFIED) is None
    assert bam.bypass_refusal(bypass=False, installed="0.32.0", verified=_VERIFIED) is None


def test_main_refuses_the_bypass_before_any_unit_runs_on_a_verified_mlx(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Bug caught: the refusal checked only inside the workers (or not at all), so a
    # pointless bypass still launches a sweep and writes records.
    installed = bam._dist("mlx")
    if installed not in _compat.VERIFIED_MLX_VERSIONS:
        pytest.skip(f"installed mlx {installed} is not verified here")
    with pytest.raises(SystemExit) as exc:
        bam.main(["--bypass-verification-gate", "--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "verified" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


class _PastTheGateError(Exception):
    """Raised by the first step after the verification gate in `flash_attention`."""


def _stop_past_the_gate(*_a: Any, **_kw: Any) -> Any:
    raise _PastTheGateError


def test_the_bypass_reaches_the_gate_at_the_flash_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: the bypass rebinding a copy (or a name the gate does not read), so the
    # real `api.flash_attention` call the worker makes still raises on the unverified
    # version. Everything below the gate is stopped, so no kernel is built or dispatched.
    monkeypatch.setattr(_compat, "VERIFIED_MLX_VERSIONS", _compat.VERIFIED_MLX_VERSIONS)
    monkeypatch.setattr(_compat, "_installed_mlx_version", lambda: "9.9.9")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(api, "select_fwd_tile", _stop_past_the_gate)
    q = mx.zeros((1, 4, 8, 64), dtype=mx.float32)
    k = v = mx.zeros((1, 2, 8, 64), dtype=mx.float32)
    with pytest.raises(UnverifiedMlxError):
        api.flash_attention(q, k, v, scale=0.125, causal=True, impl="kernel")
    bam.apply_verification_bypass("9.9.9")
    with pytest.raises(_PastTheGateError):
        api.flash_attention(q, k, v, scale=0.125, causal=True, impl="kernel")


def test_a_bypassed_run_is_marked_in_the_record_results_and_table_header() -> None:
    # Bug caught: the bypass dropped from the unit record, from results.json, or from the
    # table header, so a number measured past the gate reads like a verified one.
    flags = bam.run_flags(bypassed=True, src_identity=None)
    assert flags["verification_gate_bypassed"] is True
    rec = {**_ok_record(), **flags}
    summary = bam.summarize([rec], source=None, bypassed=True)
    assert summary["verification_gate_bypassed"] is True
    assert "verification gate bypassed" in summary["table"].splitlines()[0]


def test_an_unbypassed_run_says_so_and_leaves_the_header_plain() -> None:
    # Bug caught: `any(...)` over a missing field defaulting to True, marking a normal run.
    rec = {**_ok_record(), **bam.run_flags(bypassed=False, src_identity=None)}
    summary = bam.summarize([rec, _ok_record(impl="stock")], source=None, bypassed=False)
    assert summary["verification_gate_bypassed"] is False
    assert "bypass" not in summary["table"]


# ------------------------------------------------------------------- --src identity


def _fake_src(root: Path) -> Path:
    pkg = root / "src" / "mlx_train_perf"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n")
    (pkg / "api.py").write_text("Y = 2\n")
    return root / "src"


def test_the_src_identity_is_repo_relative_with_a_content_sha(tmp_path: Path) -> None:
    # Bug caught: the absolute source path recorded, or a sha that does not follow the
    # source contents (an edited tree reported as the same code).
    src = _fake_src(tmp_path)
    ident = bam.src_identity(src, repo_root=tmp_path, home=_FAKE_HOME)
    name, sha = ident.split("@")
    assert name == "src"
    assert len(sha) == 16
    assert int(sha, 16) >= 0
    (src / "mlx_train_perf" / "api.py").write_text("Y = 3\n")
    assert bam.src_identity(src, repo_root=tmp_path, home=_FAKE_HOME) != ident
    assert bam.run_flags(bypassed=False, src_identity=ident)["mlx_train_perf_source"] == ident


def test_prepare_src_refuses_a_path_without_the_package(tmp_path: Path) -> None:
    # Bug caught: a typo'd --src silently ignored, so the installed package is measured
    # while the record claims the source tree.
    with pytest.raises(ValueError, match="mlx_train_perf"):
        bam.prepare_src(tmp_path)


def test_prepare_src_puts_the_tree_first_on_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bug caught: the path appended (an installed mlx_train_perf would shadow the tree).
    monkeypatch.setattr(sys, "path", ["/somewhere/site-packages"])
    src = _fake_src(tmp_path)
    assert bam.prepare_src(src) == src.resolve()
    assert sys.path[0] == str(src.resolve())


# Runs the script's `--help` in a fresh isolated interpreter where `mlx` and
# `mlx_train_perf` cannot be imported at all.
_HELP_WITH_IMPORTS_BLOCKED = """\
import runpy
import sys


class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("mlx", "mlx_train_perf"):
            raise ImportError(f"blocked for this test: {name}")
        return None


sys.meta_path.insert(0, _Block())
script = sys.argv[1]
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
"""


def test_help_runs_without_importing_mlx_or_mlx_train_perf() -> None:
    # Bug caught: a new top-level `mlx` or `mlx_train_perf` import in the script, which
    # breaks `--help` and the parent process on an environment without them (or with an
    # older mlx-train-perf) -- invisible in-process, where both are already imported.
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _HELP_WITH_IMPORTS_BLOCKED,
         str(_SCRIPTS_DIR / "bench_attention_marginal.py")],
        capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "--mode" in proc.stdout
