"""Optional mlx-guard checkpoint integration for benchmark workers."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from mlx_train_perf.bench.artifacts import write_result

_CHECKPOINT_FD_ENV = "MLX_GUARD_CHECKPOINT_FD"


class ResultWriter(Protocol):
    def __call__(
        self,
        path: Path,
        identity: dict[str, object],
        status: str,
        **fields: object,
    ) -> None: ...


class CheckpointSession(Protocol):
    """Small worker-facing seam shared by guarded and direct runs."""

    def mark(self, *, stage: str, completed: int, total: int) -> None: ...

    def poll(self) -> object | None: ...

    def close(self) -> None: ...


class _NullCheckpointSession:
    def mark(self, *, stage: str, completed: int, total: int) -> None:  # noqa: ARG002
        return None

    def poll(self) -> None:
        return None

    def close(self) -> None:
        return None


class _ExternalCheckpointSession:
    def __init__(
        self,
        out: Path,
        identity: dict[str, object],
        module: Any,
        result_writer: ResultWriter,
    ) -> None:
        self._out = out
        self._identity = identity
        self._module = module
        self._result_writer = result_writer
        self._progress: dict[str, object] = {
            "stage": "starting",
            "completed": 0,
            "total": 0,
        }
        self._worker: Any = None

    def connect(self) -> bool:
        self._worker = self._module.CheckpointWorker.connect(self._checkpoint)
        return self._worker is not None

    def mark(self, *, stage: str, completed: int, total: int) -> None:
        if completed < 0 or total < 0 or completed > total:
            raise ValueError("checkpoint progress must satisfy 0 <= completed <= total")
        self._progress = {"stage": stage, "completed": completed, "total": total}

    def poll(self) -> object | None:
        if self._worker is None:
            return None
        return cast("object | None", self._worker.poll())

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _checkpoint(self, request: Any) -> object:
        self._result_writer(
            self._out,
            self._identity,
            "checkpointed_partial",
            checkpoint_request_id=request.request_id,
            supervisor_deadline_ns=request.supervisor_deadline_ns,
            progress=dict(self._progress),
        )
        _sync_file_and_parent(self._out)
        artifact = self._module.CheckpointArtifact(
            kind=self._module.CheckpointArtifactKind.FILE,
            size_bytes=self._out.stat().st_size,
        )
        return self._module.CheckpointResponse.completed(artifact)


def connect_external_checkpoint(
    out: Path,
    identity: dict[str, object],
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    result_writer: ResultWriter = write_result,
) -> CheckpointSession:
    """Connect only inside mlx-guard, with no required dependency for direct runs."""
    if _CHECKPOINT_FD_ENV not in os.environ:
        return _NullCheckpointSession()
    try:
        module = import_module("mlx_guard")
    except ModuleNotFoundError as error:
        if error.name not in (None, "mlx_guard"):
            raise
        return _NullCheckpointSession()
    session = _ExternalCheckpointSession(out, identity, module, result_writer)
    if not session.connect():
        return _NullCheckpointSession()
    return session


def _sync_file_and_parent(path: Path) -> None:
    with path.open("rb") as artifact:
        os.fsync(artifact.fileno())
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_fd = os.open(path.parent, flags)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
