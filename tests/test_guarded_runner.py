from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_train_perf.bench import runner
from mlx_train_perf.bench.runner import ExternalGuardConfig


class _DiscoveryError(RuntimeError):
    pass


class _ReportError(RuntimeError):
    pass


class _FakeRunConfig:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def _fake_module(run: Any) -> object:
    return SimpleNamespace(
        RunConfig=_FakeRunConfig,
        run=run,
        SupervisorDiscoveryError=_DiscoveryError,
    )


def test_guard_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_footprint_bytes"):
        ExternalGuardConfig(max_footprint_bytes=1)
    with pytest.raises(ValueError, match="sample_interval_ms"):
        ExternalGuardConfig(max_footprint_bytes=1024, sample_interval_ms=9)
    with pytest.raises(ValueError, match="wall_time_ms"):
        ExternalGuardConfig(max_footprint_bytes=1024, wall_time_ms=0)


def test_guarded_spawn_passes_literal_worker_command_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(config: object, *, capture_output: bool) -> object:
        captured["config"] = config
        captured["capture_output"] = capture_output
        return SimpleNamespace(returncode=75, stdout=b"worker out", stderr=b"worker err")

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


def test_discovery_failure_falls_back_but_report_failure_never_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_calls: list[Path] = []
    monkeypatch.setattr(
        runner,
        "_spawn_worker",
        lambda path: direct_calls.append(path)
        or subprocess.CompletedProcess([], 0, "direct", ""),
    )
    config_path = tmp_path / "condition.json"
    config_path.write_text("{}")
    policy = ExternalGuardConfig(max_footprint_bytes=512 << 20)

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(
            lambda _config, *, capture_output: (_ for _ in ()).throw(  # noqa: ARG005
                _DiscoveryError("version mismatch")
            )
        ),
    )
    discovery = runner._spawn_guarded_worker(config_path, tmp_path / "a.json", policy)
    assert discovery.fallback_reason == "mlx-guard supervisor discovery failed"
    assert direct_calls == [config_path]

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: _fake_module(
            lambda _config, *, capture_output: (_ for _ in ()).throw(  # noqa: ARG005
                _ReportError("report write failed")
            )
        ),
    )
    report_failure = runner._spawn_guarded_worker(config_path, tmp_path / "b.json", policy)
    assert direct_calls == [config_path]
    assert report_failure.fallback_reason is None
    assert report_failure.client_error == "guard client failed after launch: _ReportError"
    assert report_failure.process.returncode != 0


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
