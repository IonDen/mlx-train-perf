"""Bench harness: identity-keyed artifacts, resume integrity, same-session ratios.

All staleness/identity logic is pure and default-lane (no model loads); the
`run_conditions`/`worker` integration tests below spawn a real subprocess but only ever
at a tiny synthetic shape with `impl="naive"`/`"chunked"` -- fast, no Metal JIT, no
`--run-metal`/`--run-smoke` gate needed.
"""
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from mlx_train_perf.bench import artifacts, runner, worker
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    new_session_id,
    result_is_fresh,
    run_identity,
    write_result,
)
from mlx_train_perf.bench.runner import Condition, report, run_conditions
from mlx_train_perf.errors import BenchInputError, LaunchBudgetError, MlxTrainPerfError

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


def test_condition_identity_rejects_reserved_param_key_kind() -> None:
    with pytest.raises(BenchInputError, match="kind"):
        condition_identity(kind="loss_layer", session_id="s1", params={"kind": "oops"})


def test_condition_identity_rejects_reserved_param_key_session_id() -> None:
    with pytest.raises(BenchInputError, match="session_id"):
        condition_identity(kind="loss_layer", session_id="s1", params={"session_id": "oops"})


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


def test_run_conditions_records_error_envelope_on_worker_crash(tmp_path: Path) -> None:
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


def test_worker_main_crashes_on_unsupported_kind(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "train_step", "params": {}, "session_id": "s1", "out": str(out),
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
