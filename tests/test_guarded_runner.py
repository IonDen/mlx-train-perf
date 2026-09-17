from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_train_perf.bench import runner
from mlx_train_perf.bench.runner import ExternalGuardConfig
from mlx_train_perf.errors import BenchInputError


class _DiscoveryError(RuntimeError):
    pass


class _ReportError(RuntimeError):
    pass


class _FakeRunConfig:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class _ConfigError(ValueError):
    pass


def _fake_module(
    run: Any,
    *,
    start_error: BaseException | None = None,
    run_config: Any = _FakeRunConfig,
) -> object:
    """A stand-in for the `mlx_guard` package (a true external boundary: a native binary).
    `run` plays `GuardProcess.wait()`: it executes only AFTER `start()` returned, i.e. only
    once a supervisor -- and so possibly a worker -- exists."""

    def start(config: object, *, capture_output: bool) -> object:
        if start_error is not None:
            raise start_error
        return SimpleNamespace(wait=lambda: run(config, capture_output=capture_output))

    return SimpleNamespace(
        RunConfig=run_config,
        start=start,
        SupervisorDiscoveryError=_DiscoveryError,
    )


def _result(
    *, returncode: int = 0, kind: str = "child_exited",
    checkpoint: dict[str, object] | None = None, child_status: object = None,
) -> object:
    outcome: dict[str, object] = {"kind": kind}
    if child_status is not None:
        outcome["child_status"] = child_status
    payload: dict[str, object] = {"outcome": outcome}
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    return SimpleNamespace(
        returncode=returncode, stdout=b"", stderr=b"",
        report=SimpleNamespace(
            outcome=SimpleNamespace(kind=SimpleNamespace(value=kind)), payload=payload,
        ),
    )


def test_guard_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_footprint_bytes"):
        ExternalGuardConfig(max_footprint_bytes=1)
    with pytest.raises(ValueError, match="sample_interval_ms"):
        ExternalGuardConfig(max_footprint_bytes=1024, sample_interval_ms=9)
    with pytest.raises(ValueError, match="wall_time_ms"):
        ExternalGuardConfig(max_footprint_bytes=1024, wall_time_ms=0)


@pytest.mark.parametrize("bad", [9, 60_001, 1.5, True])
def test_guard_config_rejects_checkpoint_timeout_outside_10ms_to_60s(bad: Any) -> None:
    # Catches: forwarding an out-of-range timeout, which the supervisor would refuse only
    # AFTER the runner had already cleared the stale condition artifact.
    with pytest.raises(ValueError, match="checkpoint_timeout_ms"):
        ExternalGuardConfig(max_footprint_bytes=1 << 30, checkpoint_timeout_ms=bad)


def test_guard_config_rejects_unknown_on_parent_exit() -> None:
    with pytest.raises(ValueError, match="on_parent_exit"):
        ExternalGuardConfig(max_footprint_bytes=1 << 30, on_parent_exit="orphan")


def test_checkpoint_timeout_and_parent_exit_reach_run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the knob is accepted and silently dropped, so a multi-second training step
    # always loses its checkpoint to the supervisor's 1 s default.
    captured: dict[str, Any] = {}

    def fake_run(config: object, *, capture_output: bool) -> object:  # noqa: ARG001
        captured["config"] = config
        return _result()

    monkeypatch.setattr(
        runner.importlib, "import_module", lambda _name: _fake_module(fake_run),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")
    policy = ExternalGuardConfig(
        max_footprint_bytes=1 << 30, checkpoint_timeout_ms=5_000, on_parent_exit="detach",
    )

    runner._spawn_guarded_worker(config_path, tmp_path / "guard.json", policy)

    assert captured["config"].checkpoint_timeout_ms == 5_000
    assert captured["config"].on_parent_exit == "detach"


def test_guarded_spawn_passes_literal_worker_command_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(config: object, *, capture_output: bool) -> object:
        captured["config"] = config
        captured["capture_output"] = capture_output
        return _result(returncode=75, kind="policy_intervention")

    monkeypatch.setattr(
        runner.importlib, "import_module", lambda _name: _fake_module(fake_run),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")
    report_path = tmp_path / "guard.json"
    policy = ExternalGuardConfig(
        max_footprint_bytes=512 << 20,
        sample_interval_ms=10,
        wall_time_ms=500,
    )

    result = runner._spawn_guarded_worker(config_path, report_path, policy)

    config = captured["config"]
    assert config.command == (
        runner.sys.executable,
        "-m",
        "mlx_train_perf.bench.worker",
        "--config",
        str(config_path),
    )
    assert config.report == report_path
    assert config.max_footprint_bytes == 512 << 20
    assert config.sample_interval_ms == 10
    assert config.wall_time_ms == 500
    assert captured["capture_output"] is True
    assert result.process.returncode == 75
    assert result.fallback_reason is None
    assert result.client_error is None


def test_missing_package_falls_back_before_any_supervisor_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_calls: list[Path] = []
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("mlx_guard")),
    )
    monkeypatch.setattr(
        runner,
        "_spawn_worker",
        lambda path: direct_calls.append(path)
        or subprocess.CompletedProcess([], 0, "direct", ""),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path,
        tmp_path / "guard.json",
        ExternalGuardConfig(max_footprint_bytes=512 << 20),
    )

    assert direct_calls == [config_path]
    assert result.fallback_reason == "mlx-guard package is unavailable"
    assert result.client_error is None


def _direct_launch_spy(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr(
        runner,
        "_spawn_worker",
        lambda path: calls.append(path) or subprocess.CompletedProcess([], 0, "direct", ""),
    )
    return calls


def _raising(error: BaseException) -> Any:
    def wait(_config: object, *, capture_output: bool) -> object:  # noqa: ARG001
        raise error

    return wait


def test_broken_install_metadata_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: `import mlx_guard` reads its own distribution metadata at import; on a broken
    # install that raises PackageNotFoundError, a ModuleNotFoundError whose `.name` is the
    # DISTRIBUTION name "mlx-guard" -- re-raising it would abort the whole sweep for a case
    # that must fall back.
    direct_calls = _direct_launch_spy(monkeypatch)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("mlx-guard")),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "guard.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert direct_calls == [config_path]
    assert result.fallback_reason == "mlx-guard package is unavailable"


def test_discovery_failure_before_the_supervisor_starts_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_calls = _direct_launch_spy(monkeypatch)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(
            _raising(AssertionError("wait() must not run when start() failed")),
            start_error=_DiscoveryError("version mismatch"),
        ),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "a.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert direct_calls == [config_path]
    assert result.fallback_reason == "mlx-guard supervisor discovery failed"
    assert result.client_error is None


def test_discovery_failure_after_the_worker_ran_never_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: deciding fallback by exception TYPE. The guard re-verifies its binary while
    # loading the final report, so the very same SupervisorDiscoveryError can surface AFTER
    # the supervised worker already ran (a slow `--version` probe on a stressed machine).
    # Relaunching there runs the condition twice and clobbers the first artifact.
    direct_calls = _direct_launch_spy(monkeypatch)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(_raising(_DiscoveryError("version probe timed out"))),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "a.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert direct_calls == []
    assert result.fallback_reason is None
    assert result.client_error == "guard client failed after launch: _DiscoveryError"
    assert result.process is None


def test_report_failure_never_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_calls = _direct_launch_spy(monkeypatch)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(_raising(_ReportError("report write failed"))),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "b.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert direct_calls == []
    assert result.client_error == "guard client failed after launch: _ReportError"


def test_rejected_configuration_is_recorded_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: RunConfig validation living outside the try, where one refused policy
    # aborts every remaining condition of the sweep.
    def refusing_config(**_fields: object) -> object:
        raise _ConfigError("cwd must be absolute")

    direct_calls = _direct_launch_spy(monkeypatch)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(
            _raising(AssertionError("unreachable")), run_config=refusing_config,
        ),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "c.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert direct_calls == []
    assert result.client_error == "guard client could not start the supervisor: _ConfigError"


def test_an_interrupted_wait_cancels_the_supervisor_before_propagating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: Ctrl-C during a supervised condition leaving the native supervisor and its
    # worker running. The unsupervised path gets this for free (`subprocess.run` kills its
    # child on any exception); the supervised one has to ask. `cancel()` is the external
    # boundary's only stop signal, so the call itself is the observable effect.
    cancelled: list[bool] = []

    def start(_config: object, *, capture_output: bool) -> object:  # noqa: ARG001
        def wait() -> object:
            raise KeyboardInterrupt

        return SimpleNamespace(wait=wait, cancel=lambda: cancelled.append(True))

    module = SimpleNamespace(
        RunConfig=_FakeRunConfig, start=start, SupervisorDiscoveryError=_DiscoveryError,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: module)
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    with pytest.raises(KeyboardInterrupt):
        runner._spawn_guarded_worker(
            config_path, tmp_path / "a.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
        )
    assert cancelled == [True]


def test_an_unreadable_report_shape_fails_one_condition_not_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: outcome extraction outside any `try`. The worker has already run; a report
    # shape this version does not expect must cost one record, not the rest of the sweep.
    broken = SimpleNamespace(returncode=0, stdout=b"", stderr=b"", report=None)
    monkeypatch.setattr(
        runner.importlib, "import_module",
        lambda _name: _fake_module(lambda _config, *, capture_output: broken),  # noqa: ARG005
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")

    result = runner._spawn_guarded_worker(
        config_path, tmp_path / "a.json", ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    assert result.outcome is None
    assert result.client_error == "guard report could not be summarized: AttributeError"
    assert result.process is not None
    assert result.process.returncode == 0


def test_guard_outcome_keeps_how_the_command_ended_and_what_was_saved() -> None:
    result = _result(
        returncode=75, kind="policy_intervention", child_status={"signal": 15},
        checkpoint={
            "status": "acknowledged_unverified_durability", "request_id": 7,
            "reason": "wall_time", "artifact": {"kind": "file", "size_bytes": 641},
        },
    )
    assert runner._guard_outcome(result) == {
        "kind": "policy_intervention",
        "returncode": 75,
        "child_status": {"signal": 15},
        "checkpoint_status": "acknowledged_unverified_durability",
        "checkpoint_request_id": 7,
        "checkpoint_reason": "wall_time",
        "checkpoint_artifact": {"kind": "file", "size_bytes": 641},
    }


def test_guard_outcome_tolerates_a_report_without_a_checkpoint_section() -> None:
    assert runner._guard_outcome(_result())["checkpoint_status"] is None


def _guarded_condition() -> runner.Condition:
    return runner.Condition(
        name="guarded", kind="loss_layer", params={"n": 8, "d": 4, "v": 16, "impl": "naive"},
    )


def test_each_attempt_gets_its_own_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: reusing `<condition>.json`. The guard retains `.<name>.journal` beside every
    # report and refuses a path whose journal exists, so the resume retry of a condition
    # would never run under supervision again.
    reports: list[Path] = []

    def fake_spawn(
        _config_path: Path, report_path: Path, _policy: ExternalGuardConfig,
    ) -> runner.GuardedWorkerResult:
        reports.append(report_path)
        return runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], 75, "", ""),
            fallback_reason=None, client_error=None,
            outcome={"kind": "policy_intervention", "returncode": 75},
        )

    monkeypatch.setattr(runner, "_spawn_guarded_worker", fake_spawn)
    policy = ExternalGuardConfig(max_footprint_bytes=1 << 30)
    runner.run_conditions([_guarded_condition()], tmp_path, session_id="s1", guard=policy)
    runner.run_conditions([_guarded_condition()], tmp_path, session_id="s1", guard=policy)

    assert len(reports) == 2
    assert reports[0] != reports[1]
    assert {r.parent for r in reports} == {tmp_path / "_mlx_guard"}
    assert all(r.name.startswith("guarded.") and r.suffix == ".json" for r in reports)


def test_intervention_without_an_artifact_is_an_external_abort_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: recording a deliberate policy intervention as WorkerCrashed -- the sweep
    # would report a bug where the guard did its job.
    outcome: dict[str, object] = {"kind": "policy_intervention", "returncode": 75}
    monkeypatch.setattr(
        runner,
        "_spawn_guarded_worker",
        lambda _c, _r, _p: runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], 75, "", ""),
            fallback_reason=None, client_error=None, outcome=outcome,
        ),
    )
    paths = runner.run_conditions(
        [_guarded_condition()], tmp_path, session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    data = json.loads(paths[0].read_text())
    assert data["status"] == "aborted_external_guard"
    assert data["guard_outcome"] == outcome
    record = json.loads((tmp_path / "_mlx_guard" / "guarded.supervision.json").read_text())
    assert record["status"] == "guard_supervised"
    assert record["outcome"] == outcome
    assert record["report"].startswith("guarded.")


@pytest.mark.parametrize("kind", ["supervisor_failure", "launch_not_found"])
def test_supervisor_side_failure_is_not_blamed_on_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    # Catches: labelling a supervisor-side failure `WorkerCrashed` -- whoever debugs the
    # sweep would go looking for a bug in the condition instead of in the launch.
    returncode = {"supervisor_failure": 70, "launch_not_found": 127}[kind]
    outcome: dict[str, object] = {"kind": kind, "returncode": returncode}
    monkeypatch.setattr(
        runner,
        "_spawn_guarded_worker",
        lambda _c, _r, _p: runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], returncode, "", ""),
            fallback_reason=None, client_error=None, outcome=outcome,
        ),
    )
    paths = runner.run_conditions(
        [_guarded_condition()], tmp_path, session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "SupervisorReportedFailure"
    assert data["guard_outcome"] == outcome


def test_a_supervised_worker_crash_is_still_a_worker_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome: dict[str, object] = {"kind": "child_exited", "returncode": 1}
    monkeypatch.setattr(
        runner,
        "_spawn_guarded_worker",
        lambda _c, _r, _p: runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], 1, "", "Traceback: boom"),
            fallback_reason=None, client_error=None, outcome=outcome,
        ),
    )
    paths = runner.run_conditions(
        [_guarded_condition()], tmp_path, session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    data = json.loads(paths[0].read_text())
    assert data["error_type"] == "WorkerCrashed"
    assert data["error_msg"] == "Traceback: boom"


def test_a_fallback_is_recorded_and_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Catches: a silent fallback. The condition's own artifact looks identical whether or not
    # a supervisor watched it, so this record and this stderr line are the only trace that a
    # sweep which asked for supervision ran without it.
    def fake_spawn(
        config_path: Path, _report: Path, _policy: ExternalGuardConfig,
    ) -> runner.GuardedWorkerResult:
        config = json.loads(config_path.read_text())
        identity = runner.condition_identity(
            kind=config["kind"], session_id=config["session_id"], params=config["params"],
            attention_impl=config["attention_impl"],
        )
        runner.write_result(Path(config["out"]), identity, "ok", wall_s=1.0)
        return runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], 0, "", ""),
            fallback_reason="mlx-guard supervisor discovery failed", client_error=None,
        )

    monkeypatch.setattr(runner, "_spawn_guarded_worker", fake_spawn)
    guard_dir = tmp_path / "_mlx_guard"
    guard_dir.mkdir(mode=0o700)
    (guard_dir / "guarded.client.json").write_text("{}")  # a previous attempt's record

    runner.run_conditions(
        [_guarded_condition()], tmp_path, session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    record = json.loads((guard_dir / "guarded.launch.json").read_text())
    assert record["status"] == "guard_fallback"
    assert record["reason"] == "mlx-guard supervisor discovery failed"
    assert not (guard_dir / "guarded.client.json").exists()
    err = capsys.readouterr().err
    assert "guarded" in err
    assert "UNSUPERVISED" in err


def test_a_guard_directory_that_is_a_symlink_is_refused_before_any_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: following a pre-planted `_mlx_guard` symlink in a shared output directory and
    # writing the runner's records wherever it points. The native supervisor refuses such a
    # directory for its own report; the runner's records deserve the same care.
    launches: list[Path] = []
    monkeypatch.setattr(
        runner, "_spawn_guarded_worker",
        lambda config_path, _r, _p: launches.append(config_path),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "_mlx_guard").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(BenchInputError, match="_mlx_guard"):
        runner.run_conditions(
            [_guarded_condition()], out_dir, session_id="s1",
            guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
        )
    assert launches == []
    assert list(elsewhere.iterdir()) == []


def test_client_failure_without_an_artifact_is_named_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: fabricating exit code 70 for a client-side failure -- the same number the
    # in-process watchdog and the supervisor's own failure outcome already use.
    monkeypatch.setattr(
        runner,
        "_spawn_guarded_worker",
        lambda _c, _r, _p: runner.GuardedWorkerResult(
            process=None, fallback_reason=None,
            client_error="guard client failed after launch: _ReportError",
        ),
    )
    paths = runner.run_conditions(
        [_guarded_condition()], tmp_path, session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=1 << 30),
    )

    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "GuardClientError"
    assert "returncode" not in data


def test_run_conditions_records_guard_metadata_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_spawn(
        config_path: Path, _report_path: Path, _policy: ExternalGuardConfig,
    ) -> runner.GuardedWorkerResult:
        config = json.loads(config_path.read_text())
        identity = runner.condition_identity(
            kind=config["kind"],
            session_id=config["session_id"],
            params=config["params"],
            attention_impl=config["attention_impl"],
        )
        runner.write_result(Path(config["out"]), identity, "checkpointed_partial")
        return runner.GuardedWorkerResult(
            process=subprocess.CompletedProcess([], 75, "", ""),
            fallback_reason=None,
            client_error="guard client failed after launch: _ReportError",
        )

    monkeypatch.setattr(runner, "_spawn_guarded_worker", fake_spawn)
    condition = runner.Condition(
        name="guarded",
        kind="loss_layer",
        params={"n": 8, "d": 4, "v": 16, "impl": "naive"},
    )
    paths = runner.run_conditions(
        [condition],
        tmp_path,
        session_id="s1",
        guard=ExternalGuardConfig(max_footprint_bytes=512 << 20),
    )

    assert json.loads(paths[0].read_text())["status"] == "checkpointed_partial"
    guard_dir = tmp_path / "_mlx_guard"
    assert guard_dir.stat().st_mode & 0o777 == 0o700
    sidecar = json.loads((guard_dir / "guarded.client.json").read_text())
    assert sidecar["status"] == "guard_client_error"
    assert sidecar["error"] == "guard client failed after launch: _ReportError"
