from __future__ import annotations

import inspect
import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_train_perf.bench import artifacts, checkpoint, worker
from mlx_train_perf.core.guards import EffectiveCeiling


class _FakeWorker:
    def __init__(self) -> None:
        self.poll_calls = 0
        self.closed = False

    def poll(self) -> object | None:
        self.poll_calls += 1
        return None

    def close(self) -> None:
        self.closed = True


class _FakeCheckpointWorker:
    callback: Any = None
    worker = _FakeWorker()

    @classmethod
    def connect(cls, callback: Any) -> _FakeWorker:
        cls.callback = callback
        return cls.worker


class _FakeArtifactKind:
    FILE = "file"


class _FakeArtifact:
    def __init__(self, *, kind: object, size_bytes: int) -> None:
        self.kind = kind
        self.size_bytes = size_bytes


class _FakeResponse:
    @classmethod
    def completed(cls, artifact: object) -> object:
        return SimpleNamespace(status="completed", artifact=artifact)

    @classmethod
    def cancelled(cls) -> object:
        return SimpleNamespace(status="cancelled", artifact=None)


def _fake_mlx_guard() -> object:
    _FakeCheckpointWorker.callback = None
    _FakeCheckpointWorker.worker = _FakeWorker()
    return SimpleNamespace(
        CheckpointWorker=_FakeCheckpointWorker,
        CheckpointArtifactKind=_FakeArtifactKind,
        CheckpointArtifact=_FakeArtifact,
        CheckpointResponse=_FakeResponse,
    )


def test_connect_outside_supervisor_does_not_import_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_GUARD_CHECKPOINT_FD", raising=False)

    def unexpected_import(_name: str) -> object:
        raise AssertionError("optional dependency must stay lazy")

    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {"session_id": "s1"}, import_module=unexpected_import,
    )
    session.mark(stage="loss_layer", completed=1, total=3)
    assert session.poll() is None
    session.close()


def test_missing_optional_dependency_leaves_supervisor_to_apply_its_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")

    def missing(_name: str) -> object:
        raise ModuleNotFoundError("mlx_guard")

    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {"session_id": "s1"}, import_module=missing,
    )
    assert session.poll() is None
    session.close()


class _FakeCheckpointError(RuntimeError):
    pass


def test_failed_handshake_degrades_to_an_unsupervised_checkpoint_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: letting a checkpoint-channel hiccup (a bad descriptor, a hello that misses
    # the channel's 1 s timeout) kill the whole condition. Without a negotiated channel the
    # supervisor still enforces its limits; only the cooperative checkpoint is lost, and
    # its report says so (`not_negotiated`).
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")

    class _RefusingWorker:
        @classmethod
        def connect(cls, _callback: Any) -> object:
            raise _FakeCheckpointError("checkpoint descriptor is invalid")

    module = SimpleNamespace(CheckpointWorker=_RefusingWorker, CheckpointError=_FakeCheckpointError)
    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {"session_id": "s1"}, import_module=lambda _name: module,
    )

    session.mark(stage="loss_layer", completed=1, total=3)
    assert session.poll() is None
    assert not (tmp_path / "result.json").exists()


def test_a_failed_checkpoint_does_not_crash_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Catches: letting the helper's post-failure exception kill the worker. Measured against
    # the real supervisor (mlx-guard 0.2.0): a worker that exits right after a failed
    # acknowledgement is gone before the supervisor's TERM arrives, and the run is recorded
    # as `supervisor_failure` instead of the policy intervention it was. The failed
    # acknowledgement has already been sent; the worker's job is to stay alive for the TERM.
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")

    class _FailedCallbackWorker:
        def poll(self) -> object:
            raise _FakeCheckpointError("checkpoint callback failed")

        def close(self) -> None:
            return None

    class _Connecting:
        @classmethod
        def connect(cls, _callback: Any) -> object:
            return _FailedCallbackWorker()

    module = SimpleNamespace(CheckpointWorker=_Connecting, CheckpointError=_FakeCheckpointError)
    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {"session_id": "s1"}, import_module=lambda _name: module,
    )

    assert session.poll() is None
    assert session.poll() is None  # and it stops asking the dead channel
    assert "checkpoint" in capsys.readouterr().err


def test_broken_install_metadata_inside_the_supervisor_degrades_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: comparing only against the import name. A broken install raises
    # PackageNotFoundError, whose `.name` is the distribution name "mlx-guard".
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")

    def broken_import(_name: str) -> object:
        raise PackageNotFoundError("mlx-guard")

    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {"session_id": "s1"}, import_module=broken_import,
    )
    assert session.poll() is None


def test_checkpoint_writes_and_syncs_partial_artifact_before_acknowledging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")
    syncs: list[int] = []
    monkeypatch.setattr(checkpoint.os, "fsync", syncs.append)
    out = tmp_path / "result.json"
    identity = {"session_id": "s1", "kind": "loss_layer"}
    module = _fake_mlx_guard()

    session = checkpoint.connect_external_checkpoint(
        out, identity, import_module=lambda _name: module,
    )
    session.mark(stage="loss_layer", completed=2, total=5)
    request = SimpleNamespace(request_id=17, supervisor_deadline_ns=123_456)
    response = _FakeCheckpointWorker.callback(request)

    data = json.loads(out.read_text())
    assert data["identity"] == identity
    assert data["status"] == "checkpointed_partial"
    assert data["checkpoint_request_id"] == 17
    assert data["supervisor_deadline_ns"] == 123_456
    assert data["progress"] == {"stage": "loss_layer", "completed": 2, "total": 5}
    assert len(syncs) == 2
    assert response.artifact.kind == _FakeArtifactKind.FILE
    assert response.artifact.size_bytes == out.stat().st_size
    assert not hasattr(response.artifact, "path")


def test_checkpoint_stands_down_when_a_breach_is_already_being_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the checkpoint replacing the watchdog's breach record. Both layers react to
    # the same memory event, so the callback can run while `aborted_memory_ceiling` is
    # being written; it must write nothing and tell the supervisor so.
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")
    out = tmp_path / "result.json"
    session = checkpoint.connect_external_checkpoint(
        out, {"session_id": "s1"}, import_module=lambda _name: _fake_mlx_guard(),
        result_writer=lambda *_args, **_fields: False,  # what the breach-aware writer returns
    )

    response = _FakeCheckpointWorker.callback(
        SimpleNamespace(request_id=7, supervisor_deadline_ns=1),
    )

    assert response.status == "cancelled"
    assert not out.exists()
    session.close()


def test_the_default_checkpoint_writer_is_the_breach_aware_one() -> None:
    default = inspect.signature(checkpoint.connect_external_checkpoint).parameters[
        "result_writer"
    ].default
    assert default is checkpoint.write_result_unless_breached


def test_checkpoint_session_delegates_poll_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")
    module = _fake_mlx_guard()
    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json", {}, import_module=lambda _name: module,
    )

    assert session.poll() is None
    session.close()
    assert _FakeCheckpointWorker.worker.poll_calls == 1
    assert _FakeCheckpointWorker.worker.closed is True


def test_checkpoint_write_failure_propagates_before_completed_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLX_GUARD_CHECKPOINT_FD", "9")
    module = _fake_mlx_guard()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    session = checkpoint.connect_external_checkpoint(
        tmp_path / "result.json",
        {"session_id": "s1"},
        import_module=lambda _name: module,
        result_writer=fail_write,
    )
    session.mark(stage="loss_layer", completed=1, total=3)

    with pytest.raises(OSError, match="disk full"):
        _FakeCheckpointWorker.callback(SimpleNamespace(request_id=1, supervisor_deadline_ns=2))


class _RecordingSession:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.closed = False

    def mark(self, *, stage: str, completed: int, total: int) -> None:
        self.events.append(("mark", stage, completed, total))

    def poll(self) -> None:
        self.events.append("poll")

    def close(self) -> None:
        self.closed = True


def test_training_callback_polls_after_each_completed_step() -> None:
    session = _RecordingSession()
    callback = worker._RecordingCallback(
        checkpoint=session, stage="train_step", total=2,
    )

    callback.on_train_loss_report({"iteration": 1})
    callback.on_train_loss_report({"iteration": 2})

    assert callback.train_info == [{"iteration": 1}, {"iteration": 2}]
    assert session.events == [
        ("mark", "train_step", 1, 2),
        "poll",
        ("mark", "train_step", 2, 2),
        "poll",
    ]


def test_loss_layer_polls_after_each_completed_repetition() -> None:
    session = _RecordingSession()
    fields = worker.run_loss_layer(
        {
            "n": 8,
            "d": 4,
            "v": 16,
            "dtype": "float32",
            "impl": "naive",
            "reps": 3,
        },
        checkpoint=session,
    )

    assert len(fields["wall_s_all"]) == 3
    assert session.events == [
        ("mark", "loss_layer", 1, 3),
        "poll",
        ("mark", "loss_layer", 2, 3),
        "poll",
        ("mark", "loss_layer", 3, 3),
        "poll",
    ]


class _FakeWatchdog:
    def stop(self) -> None:
        return None


def test_worker_main_owns_and_closes_external_checkpoint_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _RecordingSession()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        worker,
        "connect_external_checkpoint",
        lambda out, identity: captured.update(out=out, identity=identity) or session,
    )
    monkeypatch.setattr(
        worker,
        "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )
    monkeypatch.setattr(
        worker,
        "install_memory_watchdog",
        lambda **_kwargs: _FakeWatchdog(),
    )

    def fake_run(
        _params: dict[str, object], *, checkpoint: checkpoint.CheckpointSession,
    ) -> dict[str, object]:
        captured["session"] = checkpoint
        return {"wall_s": 0.1}

    monkeypatch.setattr(worker, "run_loss_layer", fake_run)
    out = tmp_path / "result.json"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "kind": "loss_layer",
        "params": {},
        "session_id": "s1",
        "out": str(out),
    }))

    assert worker.main(["--config", str(config)]) == 0
    assert captured["out"] == out
    assert captured["session"] is session
    assert session.closed is True


def test_a_result_computed_during_a_breach_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the worker's final `ok` landing after the watchdog recorded a breach but before
    # its hard exit. That `ok` would count as fresh forever: a measurement taken past the
    # memory ceiling, served as a clean result, with the breach record gone.
    monkeypatch.setattr(
        worker, "connect_external_checkpoint", lambda _out, _identity: _RecordingSession(),
    )
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )
    monkeypatch.setattr(worker, "install_memory_watchdog", lambda **_kwargs: _FakeWatchdog())
    out = tmp_path / "result.json"

    def run_while_the_watchdog_fires(
        _params: dict[str, object], *, checkpoint: checkpoint.CheckpointSession,  # noqa: ARG001
    ) -> dict[str, object]:
        on_breach = artifacts.make_watchdog_on_breach(
            out, {"session_id": "s1"}, 1 << 30, exit_fn=lambda _code: None,
        )
        on_breach("memory_ceiling", {"active_bytes": 2 << 30, "elapsed_s": 1.0})
        return {"wall_s": 0.1}

    monkeypatch.setattr(worker, "run_loss_layer", run_while_the_watchdog_fires)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "kind": "loss_layer", "params": {}, "session_id": "s1", "out": str(out),
    }))

    worker.main(["--config", str(config)])

    assert json.loads(out.read_text())["status"] == "aborted_memory_ceiling"
