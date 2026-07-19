"""Bench harness: identity-keyed artifacts, resume integrity, same-session ratios.

All staleness/identity logic is pure and default-lane (no model loads); the
`run_conditions`/`worker` integration tests below spawn a real subprocess but only ever
at a tiny synthetic shape with `impl="naive"`/`"chunked"` -- fast, no Metal JIT, no
`--run-metal`/`--run-smoke` gate needed.
"""
import json
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from mlx_train_perf.bench import artifacts, runner, worker
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    make_watchdog_on_breach,
    new_session_id,
    result_is_fresh,
    run_identity,
    write_result,
)
from mlx_train_perf.bench.runner import Condition, report, run_conditions
from mlx_train_perf.core.guards import (
    DEFAULT_WALL_BUDGET_S,
    EffectiveCeiling,
    effective_memory_ceiling,
)
from mlx_train_perf.errors import (
    BenchInputError,
    LaunchBudgetError,
    MemoryBudgetError,
    MlxTrainPerfError,
)


@pytest.fixture(autouse=True)
def _plentiful_memory_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small CI runners (7 GB, ~3.5 GB free) trip guards' safe-start floor inside
    `worker.main` -- that is the runner's environment, not the behavior under test. The
    in-process tests in this module run as if the machine had honest room; the floor's
    own decision logic is covered with injected readers in the guards tests, and the
    real-subprocess tests below skip instead (their children compute the real ceiling,
    out of any monkeypatch's reach)."""
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )


def _machine_refuses_worker_start() -> bool:
    """True when THIS machine's real availability trips guards' safe-start floor. Tests
    marked with `_needs_room_for_real_worker` spawn real worker subprocesses whose
    children compute the real ceiling, so on such a machine (e.g. a 7 GB CI runner) the
    honest outcome is a skip, not a failure that says nothing about the code."""
    try:
        effective_memory_ceiling()
    except MemoryBudgetError:
        return True
    return False


_needs_room_for_real_worker = pytest.mark.skipif(
    _machine_refuses_worker_start(),
    reason="machine trips guards' safe-start floor; a real worker subprocess would refuse",
)

_GIB = 1024**3

# --- Task 14 brief's mandated Step 1 tests (verbatim) -----------------------------


def test_fresh_roundtrip(tmp_path: Path) -> None:
    ident = run_identity(model="m", impl="kernel", session_id="s1")
    p = tmp_path / "r.json"
    write_result(p, ident, "ok", g_mac_per_s=100.0)
    assert result_is_fresh(p, ident)


def test_stale_on_any_identity_change(tmp_path: Path) -> None:
    ident = run_identity(model="m", impl="kernel", session_id="s1")
    p = tmp_path / "r.json"
    write_result(p, ident, "ok", g_mac_per_s=100.0)
    changed = run_identity(model="m", impl="chunked", session_id="s1")
    assert not result_is_fresh(p, changed)


def test_not_fresh_on_error_status_or_corrupt(tmp_path: Path) -> None:
    ident = run_identity(model="m", session_id="s1")
    p = tmp_path / "r.json"
    write_result(p, ident, "error", error_type="OOM")
    assert not result_is_fresh(p, ident)
    p.write_text("{not json")
    assert not result_is_fresh(p, ident)


def test_report_refuses_cross_session_ratios(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_result(a, run_identity(model="m", impl="kernel", session_id="s1"), "ok", wall_s=1.0)
    write_result(b, run_identity(model="m", impl="naive", session_id="s2"), "ok", wall_s=2.0)
    rep = report([a, b])
    assert rep["ratios"] == {}
    assert rep["cross_session_excluded"]


def test_session_ids_unique() -> None:
    assert new_session_id() != new_session_id()


# --- Additional artifacts.py coverage (identity contents, atomicity) ---------------


def test_run_identity_carries_environment_facts_and_caller_kwargs() -> None:
    ident = run_identity(model="m", session_id="s1")
    assert ident["schema_version"] == 1
    assert ident["mlx_version"]
    assert ident["machine"]
    assert ident["macos"] or ident["macos"] == ""  # mac_ver() can be '' off-macOS
    assert ident["code_sha"]
    assert ident["package_version"]
    assert ident["model"] == "m"
    assert ident["session_id"] == "s1"


def test_code_sha_changes_when_a_dep_file_changes_and_flips_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`result_is_fresh` must NOT serve a stale artifact after a measured-path file
    changes -- an installed `importlib.metadata.version` is hatch-vcs tag-derived and
    only refreshes on an explicit reinstall (NOT on every `uv sync`), so it alone cannot
    catch this. `code_sha` is computed over a DECLARED dep list (monkeypatched here to an
    isolated tmp file so the test never touches the real source tree)."""
    dep = tmp_path / "fake_dep.py"
    dep.write_text("VALUE = 1\n")
    monkeypatch.setattr(artifacts, "CODE_SHA_DEPS", (dep,))

    ident_before = run_identity(model="m", session_id="s1")
    p = tmp_path / "r.json"
    write_result(p, ident_before, "ok", g_mac_per_s=1.0)
    assert result_is_fresh(p, ident_before)

    dep.write_text("VALUE = 2\n")  # byte-change the "measured path" dependency
    ident_after = run_identity(model="m", session_id="s1")
    assert ident_after["code_sha"] != ident_before["code_sha"]
    assert not result_is_fresh(p, ident_after)


_ATTENTION_CODE_SHA_DEPS: tuple[str, ...] = (
    "attention/reference.py",
    "attention/api.py",
    "attention/kernel/source.py",
    "attention/kernel/launch.py",
    "attention/kernel/dispatch.py",
    "attention/wrapper.py",
)


@pytest.mark.parametrize("rel_path", _ATTENTION_CODE_SHA_DEPS, ids=_ATTENTION_CODE_SHA_DEPS)
def test_editing_attention_source_changes_code_sha(
    rel_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each attention-path file a measured attention condition depends on must be
    declared in `CODE_SHA_DEPS` (T10, spec amendment §8/§10.13 -- MUST land before T11's
    attention artifact exists). Proven two ways: (1) the real, on-disk file is actually a
    member of the production `CODE_SHA_DEPS` tuple -- this is what fails BEFORE the dep
    list is extended, with the readable reason "must be declared in CODE_SHA_DEPS"; (2)
    editing its bytes flips `code_sha`, through the SAME mechanism
    `test_code_sha_changes_when_a_dep_file_changes_and_flips_freshness` proves generically
    -- isolated to a tmp copy so this test never mutates the real source tree."""
    real_path = artifacts._PACKAGE_ROOT / rel_path
    assert real_path in artifacts.CODE_SHA_DEPS, f"{rel_path} must be declared in CODE_SHA_DEPS"

    dep = tmp_path / real_path.name
    dep.write_bytes(real_path.read_bytes())
    monkeypatch.setattr(artifacts, "CODE_SHA_DEPS", (dep,))

    ident_before = run_identity(model="m", session_id="s1")
    dep.write_bytes(dep.read_bytes() + b"\n# perturb\n")
    ident_after = run_identity(model="m", session_id="s1")
    assert ident_after["code_sha"] != ident_before["code_sha"]


_PACKED_CODE_SHA_DEPS: tuple[str, ...] = (
    "data/packing.py",
    "attention/segments.py",
)


@pytest.mark.parametrize("rel_path", _PACKED_CODE_SHA_DEPS, ids=_PACKED_CODE_SHA_DEPS)
def test_editing_packed_path_source_changes_code_sha(
    rel_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each file a measured `packed_train` condition depends on must be declared in
    `CODE_SHA_DEPS` (Task 12, 0.4.0 packing cycle). `data/packing.py` supplies the
    packer/iterator/stats `bench/worker.py` calls directly for the packed arm;
    `attention/segments.py` supplies `PackedMask`/`segment_allowed`, imported by both
    `attention/api.py` and `attention/wrapper.py` (already-listed deps) -- a byte-change
    to either must invalidate a prior `packed_train` artifact. Proven the same two ways as
    `test_editing_attention_source_changes_code_sha`: (1) the real, on-disk file is
    actually a member of the production `CODE_SHA_DEPS` tuple; (2) editing its bytes flips
    `code_sha`, isolated to a tmp copy so this test never mutates the real source tree."""
    real_path = artifacts._PACKAGE_ROOT / rel_path
    assert real_path in artifacts.CODE_SHA_DEPS, f"{rel_path} must be declared in CODE_SHA_DEPS"

    dep = tmp_path / real_path.name
    dep.write_bytes(real_path.read_bytes())
    monkeypatch.setattr(artifacts, "CODE_SHA_DEPS", (dep,))

    ident_before = run_identity(model="m", session_id="s1")
    dep.write_bytes(dep.read_bytes() + b"\n# perturb\n")
    ident_after = run_identity(model="m", session_id="s1")
    assert ident_after["code_sha"] != ident_before["code_sha"]


def test_condition_identity_rejects_reserved_param_key_kind() -> None:
    with pytest.raises(BenchInputError, match="kind"):
        condition_identity(kind="loss_layer", session_id="s1", params={"kind": "oops"})


def test_condition_identity_rejects_reserved_param_key_session_id() -> None:
    with pytest.raises(BenchInputError, match="session_id"):
        condition_identity(kind="loss_layer", session_id="s1", params={"session_id": "oops"})


def test_condition_identity_differs_by_attention_impl() -> None:
    """`attention_impl` is a dedicated identity input (T10, spec §8 amendment) -- two
    conditions differing only in it get different identities, and a condition that never
    supplies it (every loss_layer/train_step condition today) gets an identity dict with
    the SAME keys as before this change -- omitted, not defaulted to `None` -- so
    SCHEMA_VERSION does not need to bump for existing condition kinds."""
    ident_flash = condition_identity(
        kind="attention_op", session_id="s1", params={}, attention_impl="flash",
    )
    ident_stock = condition_identity(
        kind="attention_op", session_id="s1", params={}, attention_impl="stock",
    )
    assert ident_flash["attention_impl"] == "flash"
    assert ident_flash != ident_stock

    ident_unset = condition_identity(kind="loss_layer", session_id="s1", params={"impl": "kernel"})
    assert "attention_impl" not in ident_unset


def test_condition_identity_differs_by_dkv_split_policy() -> None:
    ident_none = condition_identity(
        kind="attention_op", session_id="s1", params={}, dkv_split_policy="none",
    )
    ident_chunked = condition_identity(
        kind="attention_op", session_id="s1", params={}, dkv_split_policy="chunked",
    )
    assert ident_none["dkv_split_policy"] == "none"
    assert ident_none != ident_chunked

    ident_unset = condition_identity(kind="loss_layer", session_id="s1", params={"impl": "kernel"})
    assert "dkv_split_policy" not in ident_unset


def test_condition_identity_differs_by_attention_variant() -> None:
    ident_a = condition_identity(
        kind="attention_op", session_id="s1", params={}, attention_variant="mma_slab128",
    )
    ident_b = condition_identity(
        kind="attention_op", session_id="s1", params={}, attention_variant="mma_slab256",
    )
    assert ident_a["attention_variant"] == "mma_slab128"
    assert ident_a != ident_b

    ident_unset = condition_identity(kind="loss_layer", session_id="s1", params={"impl": "kernel"})
    assert "attention_variant" not in ident_unset


def test_condition_identity_rejects_reserved_param_key_attention_impl() -> None:
    with pytest.raises(BenchInputError, match="attention_impl"):
        condition_identity(kind="attention_op", session_id="s1", params={"attention_impl": "oops"})


def test_condition_identity_rejects_reserved_param_key_dkv_split_policy() -> None:
    with pytest.raises(BenchInputError, match="dkv_split_policy"):
        condition_identity(
            kind="attention_op", session_id="s1", params={"dkv_split_policy": "oops"},
        )


def test_condition_identity_rejects_reserved_param_key_attention_variant() -> None:
    with pytest.raises(BenchInputError, match="attention_variant"):
        condition_identity(
            kind="attention_op", session_id="s1", params={"attention_variant": "oops"},
        )


def test_run_identity_rejects_param_colliding_with_internal_field() -> None:
    """`_RESERVED_PARAM_KEYS` only covers `kind`/`session_id` -- this is the deeper,
    self-maintaining guard: ANY caller kwarg reusing one of `run_identity`'s OWN internal
    field names (e.g. `code_sha`) must be rejected, not silently hijack that field."""
    with pytest.raises(BenchInputError, match="code_sha"):
        run_identity(model="m", code_sha="HIJACKED")


def test_installed_mlx_lm_version_is_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_name: str) -> str:
        raise PackageNotFoundError("mlx-lm")

    monkeypatch.setattr(artifacts, "version", _raise)
    assert artifacts._installed_mlx_lm_version() is None


def test_write_result_is_atomic_no_tmp_file_left_behind(tmp_path: Path) -> None:
    ident = run_identity(model="m", session_id="s1")
    p = tmp_path / "r.json"
    write_result(p, ident, "ok", wall_s=1.0)
    assert p.exists()
    assert not p.with_suffix(".tmp").exists()


# --- report(): same-session ratio computed when identities otherwise match --------


def test_ratio_label_and_value_none_on_non_numeric_or_missing_wall_time() -> None:
    assert runner._ratio_label_and_value("kernel", "naive", None, 1.0) is None
    assert runner._ratio_label_and_value("kernel", "naive", "n/a", 1.0) is None


def test_ratio_label_and_value_none_on_zero_wall_time() -> None:
    assert runner._ratio_label_and_value("kernel", "naive", 0.0, 1.0) is None


def test_report_computes_ratio_within_one_session(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_result(a, run_identity(model="m", impl="kernel", session_id="s1"), "ok", wall_s=1.0)
    write_result(b, run_identity(model="m", impl="naive", session_id="s1"), "ok", wall_s=4.0)
    rep = report([a, b])
    assert rep["cross_session_excluded"] == []
    # keyed by MEASURED speed (naive is 4x slower), not alphabetical order
    assert rep["ratios"] == {"naive/kernel": pytest.approx(4.0)}


def test_report_ratio_direction_is_by_measured_speed_not_alphabetical(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    # "chunked" sorts BEFORE "kernel" alphabetically, but chunked is the SLOWER impl here
    # -- an alphabetical convention would silently mislabel this comparison's direction.
    write_result(a, run_identity(model="m", impl="chunked", session_id="s1"), "ok", wall_s=4.0)
    write_result(b, run_identity(model="m", impl="kernel", session_id="s1"), "ok", wall_s=1.0)
    rep = report([a, b])
    assert rep["ratios"] == {"chunked/kernel": pytest.approx(4.0)}


def test_report_skips_pairs_with_the_same_impl(tmp_path: Path) -> None:
    """Two artifacts that happen to share an identity (e.g. a re-run of the identical
    condition) are not a `kernel` vs `naive`-style comparison -- no ratio, no exclusion."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ident = run_identity(model="m", impl="kernel", session_id="s1")
    write_result(a, ident, "ok", wall_s=1.0)
    write_result(b, ident, "ok", wall_s=1.1)
    rep = report([a, b])
    assert rep["ratios"] == {}
    assert rep["cross_session_excluded"] == []


def test_report_ignores_non_ok_and_corrupt_entries(tmp_path: Path) -> None:
    ok_path = tmp_path / "ok.json"
    err_path = tmp_path / "err.json"
    corrupt_path = tmp_path / "corrupt.json"
    write_result(ok_path, run_identity(model="m", impl="kernel", session_id="s1"), "ok",
                 wall_s=1.0)
    write_result(err_path, run_identity(model="m", impl="naive", session_id="s1"), "error",
                 error_type="OOM")
    corrupt_path.write_text("{not json")
    rep = report([ok_path, err_path, corrupt_path])
    assert rep["ratios"] == {}
    assert rep["cross_session_excluded"] == []


# --- runner.run_conditions(): real subprocess integration, tiny shapes -------------


def _tiny_loss_layer(name: str, impl: str = "naive") -> Condition:
    # n*d*v large enough that `g_mac_per_s` (rounded to 3 decimals in `run_loss_layer`)
    # doesn't round away to 0.000 under subprocess-dispatch overhead noise -- still a
    # trivially fast shape (no Metal JIT, sub-millisecond compute either way).
    return Condition(
        name=name, kind="loss_layer",
        params={"n": 64, "d": 64, "v": 128, "dtype": "float32", "impl": impl, "reps": 1},
    )


@_needs_room_for_real_worker
def test_run_conditions_spawns_worker_and_writes_ok_artifact(tmp_path: Path) -> None:
    session_id = new_session_id()
    paths = run_conditions([_tiny_loss_layer("naive_tiny")], tmp_path, session_id=session_id)
    assert len(paths) == 1
    data = json.loads(paths[0].read_text())
    assert data["status"] == "ok"
    assert data["identity"]["session_id"] == session_id
    assert data["wall_s"] > 0
    assert data["g_mac_per_s"] > 0


def test_run_conditions_skips_fresh_without_spawning_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = new_session_id()
    condition = _tiny_loss_layer("naive_tiny")
    out_path = tmp_path / f"{condition.name}.json"
    ident = condition_identity(kind=condition.kind, session_id=session_id, params=condition.params)
    write_result(out_path, ident, "ok", wall_s=0.01, g_mac_per_s=1.0)

    def _must_not_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called for a fresh artifact")

    monkeypatch.setattr(subprocess, "run", _must_not_spawn)
    paths = run_conditions([condition], tmp_path, session_id=session_id)
    assert paths == [out_path]
    assert json.loads(out_path.read_text())["wall_s"] == 0.01  # untouched


@_needs_room_for_real_worker
def test_run_conditions_records_error_envelope_on_worker_crash(tmp_path: Path) -> None:
    # Spawns a REAL worker (unsupported kind -> crash -> WorkerCrashed envelope). On a
    # machine that trips guards' safe-start floor (small CI runner) the worker refuses at
    # `effective_memory_ceiling()` with `refused_environment` BEFORE it reaches the kind
    # crash, so the "error" assertion only holds where the worker has room to run -- hence
    # the skipif its sibling real-worker tests carry.
    session_id = new_session_id()
    bad = Condition(name="bad_kind", kind="not_a_real_kind", params={"n": 8, "d": 4, "v": 16})
    paths = run_conditions([bad], tmp_path, session_id=session_id)
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"]
    assert data["error_msg"]


def test_run_conditions_reserved_param_key_is_a_typed_error(tmp_path: Path) -> None:
    bad = Condition(name="bad", kind="loss_layer", params={"kind": "oops"})
    with pytest.raises(BenchInputError):
        run_conditions([bad], tmp_path, session_id="s1")


def test_run_conditions_records_error_when_worker_exits_zero_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stub worker standing in for the real one: exits cleanly but writes nothing --
    e.g. a worker that silently swallowed its own crash. Must be recorded as an error,
    not treated as an implicit success just because the process exited 0."""
    stub = tmp_path / "silent_stub_worker.py"
    stub.write_text("import sys\nsys.exit(0)\n")

    def _spawn_stub(config_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(stub), "--config", str(config_path)],
            capture_output=True, text=True, check=False,
        )

    monkeypatch.setattr(runner, "_spawn_worker", _spawn_stub)
    condition = _tiny_loss_layer("silent_stub")
    paths = run_conditions([condition], tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerExitedWithoutArtifact"


def test_run_conditions_stale_artifact_not_served_as_ok_when_worker_fails_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A STALE artifact (old `session_id`, so identity mismatches -- exactly why the
    worker gets spawned at all) must not be left in place and reported as this run's
    truth if the worker that was supposed to replace it fails silently (exits 0, writes
    nothing). Reproduces the exact scenario: stale artifact with `wall_s=999.0` under an
    old session, a silently-failing stub worker -- the recorded result must be the
    error envelope, never the stale `"ok"`."""
    condition = _tiny_loss_layer("stale_then_silent")
    out_path = tmp_path / f"{condition.name}.json"
    stale_ident = condition_identity(
        kind=condition.kind, session_id="OLD_SESSION", params=condition.params,
    )
    write_result(out_path, stale_ident, "ok", wall_s=999.0)

    stub = tmp_path / "silent_stub_worker2.py"
    stub.write_text("import sys\nsys.exit(0)\n")

    def _spawn_stub(config_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(stub), "--config", str(config_path)],
            capture_output=True, text=True, check=False,
        )

    monkeypatch.setattr(runner, "_spawn_worker", _spawn_stub)
    paths = run_conditions([condition], tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerExitedWithoutArtifact"
    assert data.get("wall_s") != 999.0


# --- worker.py: pure(ish) measurement + CLI shell, tiny shapes, no Metal marker ----


def test_run_loss_layer_dense_naive_reports_expected_fields() -> None:
    # shape large enough that `g_mac_per_s` (rounded to 3 decimals) doesn't round away
    # to 0.000 under call-overhead noise at a truly tiny n/d/v -- see `_tiny_loss_layer`.
    fields = worker.run_loss_layer({"n": 64, "d": 64, "v": 128, "dtype": "float32",
                                    "impl": "naive", "reps": 1})
    assert fields["wall_s"] > 0
    assert len(fields["wall_s_all"]) == 1
    assert fields["g_mac_per_s"] > 0
    assert fields["active_before_gb"] >= 0
    assert fields["marginal_peak_gb"] >= 0


def test_run_loss_layer_quantized_chunked_runs() -> None:
    fields = worker.run_loss_layer({"n": 8, "d": 64, "v": 128, "dtype": "bfloat16",
                                    "impl": "chunked", "quantized": True, "reps": 1})
    assert fields["wall_s"] > 0


def test_run_loss_layer_unknown_dtype_is_typed_error() -> None:
    with pytest.raises(MlxTrainPerfError):
        worker.run_loss_layer({"n": 8, "d": 4, "v": 16, "dtype": "bogus", "impl": "naive"})


def test_worker_main_reserved_param_key_is_a_typed_error(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer", "params": {"session_id": "oops"}, "session_id": "s1",
        "out": str(tmp_path / "r.json"),
    }))
    with pytest.raises(BenchInputError):
        worker.main(["--config", str(cfg)])


def test_worker_main_writes_ok_artifact(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1",
        "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["identity"]["kind"] == "loss_layer"


def test_worker_main_records_refusal_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer", "params": {"n": 8, "d": 4, "v": 16}, "session_id": "s1",
        "out": str(out),
    }))

    def _refuse(_params: dict[str, object]) -> dict[str, object]:
        raise LaunchBudgetError("projected dispatch exceeds the watchdog budget")

    monkeypatch.setattr(worker, "run_loss_layer", _refuse)
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0                    # a refusal IS a result, not a crash
    data = json.loads(out.read_text())
    assert data["status"] == "refused"
    assert "watchdog" in data["error"]


def test_worker_main_records_environment_refusal_distinctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0022d: a too-crowded-at-start refusal (guards' `MemoryBudgetError` out of
    `effective_memory_ceiling`) is ENVIRONMENT-transient -- a distinct
    `refused_environment` status, never a `WorkerCrashed` envelope and never the
    condition-intrinsic `refused`. Only `"ok"` is fresh on resume, so a later, quieter
    invocation re-runs it automatically."""
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer", "params": {"n": 8, "d": 4, "v": 16}, "session_id": "s1",
        "out": str(out),
    }))

    def _too_crowded() -> EffectiveCeiling:
        raise MemoryBudgetError("machine too crowded to start safely")

    monkeypatch.setattr(worker, "effective_memory_ceiling", _too_crowded)
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0                    # transient environment refusal IS a result
    data = json.loads(out.read_text())
    assert data["status"] == "refused_environment"
    assert "crowded" in data["error"]


def test_worker_main_crashes_on_unsupported_kind(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "not_a_real_kind", "params": {}, "session_id": "s1", "out": str(out),
    }))
    with pytest.raises(MlxTrainPerfError):
        worker.main(["--config", str(cfg)])
    assert not out.exists()           # worker itself writes nothing on an uncaught crash


def test_worker_main_installs_guardrails_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1",
        "out": str(out),
    }))
    calls: list[str] = []
    monkeypatch.setattr(worker, "install_guardrails", lambda: calls.append("guardrails"))
    monkeypatch.setattr(worker, "run_loss_layer", lambda _params: calls.append("run") or {})  # type: ignore[func-returns-value]
    worker.main(["--config", str(cfg)])
    assert calls == ["guardrails", "run"]


@_needs_room_for_real_worker
def test_bench_worker_module_runnable_as_main(tmp_path: Path) -> None:
    """`python -m mlx_train_perf.bench.worker --config ...` — the exact subprocess
    invocation `run_conditions` uses."""
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1",
        "out": str(out),
    }))
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_train_perf.bench.worker", "--config", str(cfg)],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["status"] == "ok"


# --- attention_impl is a FIRST-CLASS identity field through the runner seam ---------
# Regression for the T13 step-1 crash: an attention-bearing condition must reach
# `condition_identity` via `run_conditions`' DEDICATED slot, NOT through `params` (where
# the reserved-key guard raises `BenchInputError`). The worker is stubbed at the
# subprocess boundary, so the error-envelope path writes the SAME identity the runner
# computed -- reading it back verifies the runner's own identity call, the exact line
# that crashed. These tests drive the REAL `run_conditions`, above the identity
# construction (unlike the 012fadb tests, which stubbed below it and so never built one).


def _crash_worker(_config_path: Path) -> subprocess.CompletedProcess[str]:
    """A `_spawn_worker` stand-in that "crashes" (nonzero exit, wrote nothing) -- so
    `run_conditions` takes its error-envelope branch and writes the identity it built,
    without ever loading a real model."""
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="stub crash")


_TRAIN_STEP_PARAMS: dict[str, object] = {
    "model": "m/a", "revision": None, "seq_len": 16, "batch": 1, "steps": 2,
}


def test_run_conditions_carries_attention_impl_into_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_spawn_worker", _crash_worker)
    cond = Condition(
        name="flash_c", kind="train_step", params=dict(_TRAIN_STEP_PARAMS),
        attention_impl="flash",
    )
    paths = run_conditions([cond], tmp_path, session_id=new_session_id())
    identity = json.loads(paths[0].read_text())["identity"]
    assert identity["attention_impl"] == "flash"


def test_run_conditions_stock_and_flash_conditions_get_different_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_spawn_worker", _crash_worker)
    session_id = new_session_id()
    stock = Condition(name="stock_c", kind="train_step", params=dict(_TRAIN_STEP_PARAMS),
                      attention_impl="stock")
    flash = Condition(name="flash_c", kind="train_step", params=dict(_TRAIN_STEP_PARAMS),
                      attention_impl="flash")
    paths = run_conditions([stock, flash], tmp_path, session_id=session_id)
    id_stock = json.loads(paths[0].read_text())["identity"]
    id_flash = json.loads(paths[1].read_text())["identity"]
    assert id_stock["attention_impl"] == "stock"
    assert id_flash["attention_impl"] == "flash"
    assert id_stock != id_flash


def test_run_conditions_without_attention_impl_omits_it_from_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility: a `Condition` that never sets `attention_impl` (every
    0.1.0-era loss_layer/train_step condition) produces an identity with the SAME keys as
    before -- the field is omitted, not defaulted to `None`, so no prior artifact's shape
    changes."""
    monkeypatch.setattr(runner, "_spawn_worker", _crash_worker)
    cond = _tiny_loss_layer("no_attn")
    paths = run_conditions([cond], tmp_path, session_id=new_session_id())
    identity = json.loads(paths[0].read_text())["identity"]
    assert "attention_impl" not in identity


# --- active-memory watchdog: honest-abort artifact + runner-respect + config seam -----


def test_make_watchdog_on_breach_writes_aborted_memory_artifact_then_exits(
    tmp_path: Path,
) -> None:
    """A memory breach writes THIS condition's artifact with an honest
    `aborted_memory_ceiling` status + the observed numbers, via the SAME `write_result`
    identity path the worker's ok/refused write uses, then hard-exits (70). `exit_fn` is
    injected so the test asserts the write + code WITHOUT terminating the runner."""
    out = tmp_path / "r.json"
    ident = run_identity(model="m", session_id="s1")
    exits: list[int] = []
    on_breach = make_watchdog_on_breach(
        out, ident, 28 * _GIB, exit_fn=exits.append,
    )
    on_breach(
        "memory_ceiling",
        {"active_bytes": 32 * _GIB, "elapsed_s": 12.5, "wall_budget_s": 3600.0},
    )
    data = json.loads(out.read_text())
    assert data["status"] == "aborted_memory_ceiling"
    assert data["identity"] == ident
    assert data["observed_active_gb"] == 32.0
    assert data["ceiling_gb"] == 28.0
    assert data["elapsed_s"] == 12.5
    assert exits == [70]


def test_make_watchdog_on_breach_maps_wall_reason_to_aborted_wall_budget_status(
    tmp_path: Path,
) -> None:
    out = tmp_path / "r.json"
    ident = run_identity(model="m", session_id="s1")
    exits: list[int] = []
    on_breach = make_watchdog_on_breach(out, ident, 28 * _GIB, exit_fn=exits.append)
    on_breach(
        "wall_budget",
        {"active_bytes": 1 << 20, "elapsed_s": 4000.0, "wall_budget_s": 3600.0},
    )
    data = json.loads(out.read_text())
    assert data["status"] == "aborted_wall_budget"
    assert exits == [70]


def test_make_watchdog_on_breach_still_exits_when_write_result_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FINDING C1 (safety review): `write_result` runs BEFORE `exit_fn` inside
    `on_breach`, and the watchdog thread (`core.guards._watchdog_step`) swallows any
    exception `on_breach` raises so a failing callback can't disarm the guard thread. If
    `write_result` itself raises (MemoryError/OSError -- plausible in the exact paging
    storm this watchdog exists to catch) with no `finally`, the swallowed exception means
    `exit_fn(70)` never runs and the watchdog silently disarms while the storm continues.
    `exit_fn` must ALWAYS fire, artifact write or not."""
    out = tmp_path / "r.json"
    ident = run_identity(model="m", session_id="s1")
    exits: list[int] = []

    def _raising_write_result(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full mid-storm")

    monkeypatch.setattr(artifacts, "write_result", _raising_write_result)
    on_breach = make_watchdog_on_breach(out, ident, 28 * _GIB, exit_fn=exits.append)
    # In production `exit_fn` is `os._exit`, which never returns -- the write's OSError
    # never gets a chance to propagate past it. The injected `exit_fn` test double DOES
    # return, so the `finally` block's re-raise of the original OSError is still visible
    # here; what matters for C1 is that `exit_fn` was already called before that happens.
    with pytest.raises(OSError, match="disk full mid-storm"):
        on_breach(
            "memory_ceiling",
            {"active_bytes": 32 * _GIB, "elapsed_s": 12.5, "wall_budget_s": 3600.0},
        )
    assert exits == [70]


def _spawn_breach_worker(config_path: Path) -> subprocess.CompletedProcess[str]:
    """A `_spawn_worker` stand-in that mimics the watchdog's `os._exit(70)` breach path:
    it writes an honest `aborted_memory_ceiling` artifact to the config's `out`, then
    reports a NONZERO exit -- exactly the "worker wrote its own record then hard-exited"
    shape the runner must respect rather than clobber with a generic crash envelope."""
    cfg = json.loads(config_path.read_text())
    out = Path(cfg["out"])
    ident = condition_identity(
        kind=cfg["kind"], session_id=cfg["session_id"], params=cfg["params"],
        attention_impl=cfg.get("attention_impl"),
    )
    write_result(out, ident, "aborted_memory_ceiling", observed_active_gb=32.4, ceiling_gb=28.0)
    return subprocess.CompletedProcess(args=[], returncode=70, stdout="", stderr="paging storm")


def test_run_conditions_respects_a_breach_artifact_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner-respect contract: a worker that wrote its OWN honest artifact before
    exiting nonzero (the watchdog `os._exit(70)` path) must have that record surfaced,
    NOT overwritten by the generic `WorkerCrashed` envelope. Because the runner unlinks a
    stale artifact BEFORE spawning, an artifact present after a nonzero exit is
    unambiguously this worker's own write."""
    monkeypatch.setattr(runner, "_spawn_worker", _spawn_breach_worker)
    cond = _tiny_loss_layer("breached")
    paths = run_conditions([cond], tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "aborted_memory_ceiling"
    assert data["observed_active_gb"] == 32.4
    assert data.get("error_type") != "WorkerCrashed"


def test_run_conditions_still_records_crash_envelope_when_worker_wrote_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the respect rule: a nonzero exit that left NO artifact (an
    ordinary crash before any write) still gets the generic error envelope."""
    monkeypatch.setattr(runner, "_spawn_worker", _crash_worker)
    cond = _tiny_loss_layer("plain_crash")
    paths = run_conditions([cond], tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerCrashed"


def _capture_config_spawn(
    captured: dict[str, object],
) -> Callable[[Path], subprocess.CompletedProcess[str]]:
    def _spawn(config_path: Path) -> subprocess.CompletedProcess[str]:
        cfg = json.loads(config_path.read_text())
        captured.update(cfg)
        out = Path(cfg["out"])
        ident = condition_identity(
            kind=cfg["kind"], session_id=cfg["session_id"], params=cfg["params"],
            attention_impl=cfg.get("attention_impl"),
        )
        write_result(out, ident, "ok", wall_s=0.01, g_mac_per_s=1.0)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    return _spawn


def test_run_conditions_threads_wall_budget_s_into_the_worker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wall_budget_s` rides the SAME top-level config seam `attention_impl` does (commit
    7e611f0), NOT inside `params` -- so the worker reads one authoritative value. Unlike
    `attention_impl`, it is deliberately NOT an identity field: a safety limit is not a
    measurement dimension, so two runs differing only in their budget must share an
    identity."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "_spawn_worker", _capture_config_spawn(captured))
    cond = Condition(
        name="c", kind="loss_layer", params={"n": 8, "d": 4, "v": 16}, wall_budget_s=123.0,
    )
    paths = run_conditions([cond], tmp_path, session_id="s1")
    assert captured["wall_budget_s"] == 123.0
    # NOT an identity field -- must not leak into the artifact identity.
    assert "wall_budget_s" not in json.loads(paths[0].read_text())["identity"]


def test_run_conditions_wall_budget_s_is_none_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror of the `attention_impl` seam: the config always carries the key, `None`
    when the `Condition` did not set it (worker maps `None` -> the module default)."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "_spawn_worker", _capture_config_spawn(captured))
    cond = _tiny_loss_layer("no_budget")
    run_conditions([cond], tmp_path, session_id="s1")
    assert "wall_budget_s" in captured
    assert captured["wall_budget_s"] is None


class _FakeWatchdogHandle:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def is_alive(self) -> bool:  # pragma: no cover -- interface parity, unused here
        return not self.stopped


def test_worker_main_installs_watchdog_and_threads_resolved_wall_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`worker.main` installs the watchdog (right after `install_guardrails`) with the
    EFFECTIVE ceiling `effective_memory_ceiling` computes (static + dynamic-availability
    min), threads the config's explicit `wall_budget_s` through, and STOPS it on the
    normal completion path."""
    captured: dict[str, object] = {}
    handle = _FakeWatchdogHandle()

    def _fake_install(
        *, ceiling_bytes: int, wall_budget_s: float | None, on_breach: object,  # noqa: ARG001
    ) -> _FakeWatchdogHandle:
        captured["ceiling_bytes"] = ceiling_bytes
        captured["wall_budget_s"] = wall_budget_s
        return handle

    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    monkeypatch.setattr(worker, "install_memory_watchdog", _fake_install)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1", "wall_budget_s": 222.0, "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    assert captured["wall_budget_s"] == 222.0
    assert captured["ceiling_bytes"] == 987654321  # the effective ceiling flowed through


def test_worker_main_watchdog_defaults_wall_budget_when_config_omits_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_install(
        *, ceiling_bytes: int, wall_budget_s: float | None, on_breach: object,  # noqa: ARG001
    ) -> _FakeWatchdogHandle:
        captured["wall_budget_s"] = wall_budget_s
        return _FakeWatchdogHandle()

    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    monkeypatch.setattr(worker, "install_memory_watchdog", _fake_install)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1", "out": str(out),
    }))
    worker.main(["--config", str(cfg)])
    assert captured["wall_budget_s"] == DEFAULT_WALL_BUDGET_S


def test_worker_main_stops_watchdog_even_when_the_condition_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watchdog must be stopped on EVERY exit path -- including an uncaught crash
    (unsupported kind) -- so an in-process caller never leaks a sampling thread."""
    handle = _FakeWatchdogHandle()
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    monkeypatch.setattr(
        worker, "install_memory_watchdog",
        lambda *, ceiling_bytes, wall_budget_s, on_breach: handle,  # noqa: ARG005
    )
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "not_a_real_kind", "params": {}, "session_id": "s1", "out": str(out),
    }))
    with pytest.raises(MlxTrainPerfError):
        worker.main(["--config", str(cfg)])
    assert handle.stopped is True


def test_worker_main_records_memory_warning_in_the_artifact_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded-start warning from `effective_memory_ceiling` (crowded machine /
    unmeasurable availability) is carried into the worker's `ok` artifact as
    `memory_warning`, so a campaign record shows the degraded-start state."""
    fake_handle = _FakeWatchdogHandle()
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning="only 8 GiB free"),
    )
    monkeypatch.setattr(
        worker, "install_memory_watchdog",
        lambda *, ceiling_bytes, wall_budget_s, on_breach: fake_handle,  # noqa: ARG005
    )
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1", "out": str(out),
    }))
    assert worker.main(["--config", str(cfg)]) == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["memory_warning"] == "only 8 GiB free"


def test_worker_main_omits_memory_warning_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No warning (nominal start) -> the artifact omits the key entirely (omit-when-None,
    matching the identity conventions), never a null field."""
    fake_handle = _FakeWatchdogHandle()
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    monkeypatch.setattr(
        worker, "install_memory_watchdog",
        lambda *, ceiling_bytes, wall_budget_s, on_breach: fake_handle,  # noqa: ARG005
    )
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {"n": 8, "d": 4, "v": 16, "dtype": "float32", "impl": "naive", "reps": 1},
        "session_id": "s1", "out": str(out),
    }))
    assert worker.main(["--config", str(cfg)]) == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert "memory_warning" not in data
