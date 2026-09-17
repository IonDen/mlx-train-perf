"""The real `mlx-guard` supervisor through the runner's launch boundary (`--run-guard`).

Every other guard test fakes the package; these are the ones that notice when the fake and
the real 0.2.0 surface drift apart. Synthetic `loss_layer` conditions only -- no model.
"""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mlx_train_perf.bench.runner import Condition, ExternalGuardConfig, run_conditions
from mlx_train_perf.core.guards import MemoryBudgetError, effective_memory_ceiling

_GIB = 1024**3
# Far more repetitions than any limit below lets finish (sub-millisecond each).
_ENDLESS_REPS = 2_000_000


def _machine_refuses_worker_start() -> bool:
    try:
        effective_memory_ceiling()
    except MemoryBudgetError:
        return True
    return False


pytestmark = [
    pytest.mark.guard,
    pytest.mark.skipif(
        _machine_refuses_worker_start(),
        reason="machine trips guards' safe-start floor; a real worker would refuse",
    ),
]


@pytest.fixture
def mlx_guard() -> Any:
    # Imported per test, never at module scope: collecting this file in the default lane
    # must not import the optional package.
    return pytest.importorskip("mlx_guard")


def _condition(name: str, *, reps: int, wall_budget_s: float | None = None) -> Condition:
    return Condition(
        name=name, kind="loss_layer", wall_budget_s=wall_budget_s,
        params={"n": 64, "d": 64, "v": 128, "dtype": "float32", "impl": "naive", "reps": reps},
    )


def _wall_limited(limit_ms: int) -> ExternalGuardConfig:
    # The acknowledgement budget is fixed when the request is created, and the worker can
    # only answer at its next poll. 10 s covers a request that lands during interpreter
    # start-up or the warm-up pass, so the outcome does not depend on where the limit falls.
    return ExternalGuardConfig(
        max_footprint_bytes=8 * _GIB, wall_time_ms=limit_ms, sample_interval_ms=10,
        checkpoint_timeout_ms=10_000,
    )


def _supervision(out_dir: Path, name: str) -> dict[str, Any]:
    record = json.loads((out_dir / "_mlx_guard" / f"{name}.supervision.json").read_text())
    assert record["status"] == "guard_supervised"
    return dict(record["outcome"])


def test_supervised_condition_completes_ok(tmp_path: Path, mlx_guard: Any) -> None:  # noqa: ARG001
    # Catches: a RunConfig/argv mismatch with the real package (every fake-module test
    # stays green while the real supervisor refuses the launch).
    paths = run_conditions(
        [_condition("tiny", reps=3)], tmp_path, session_id="s",
        guard=ExternalGuardConfig(max_footprint_bytes=8 * _GIB),
    )

    assert json.loads(paths[0].read_text())["status"] == "ok"
    outcome = _supervision(tmp_path, "tiny")
    assert outcome["kind"] == "child_exited"
    assert outcome["returncode"] == 0
    assert not (tmp_path / "_mlx_guard" / "tiny.launch.json").exists()


def test_wall_limit_checkpoints_the_worker_then_stops_it(
    tmp_path: Path, mlx_guard: Any,  # noqa: ARG001
) -> None:
    # Catches: a worker that never polls (or polls off the main thread) -- the request
    # would time out and no partial artifact would exist.
    paths = run_conditions(
        [_condition("endless", reps=_ENDLESS_REPS)], tmp_path, session_id="s",
        guard=_wall_limited(1_500),
    )

    data = json.loads(paths[0].read_text())
    outcome = _supervision(tmp_path, "endless")
    assert outcome["kind"] == "policy_intervention"
    assert outcome["checkpoint_reason"] == "wall_time"
    assert outcome["checkpoint_status"] == "acknowledged_unverified_durability"
    assert outcome["checkpoint_artifact"]["kind"] == "file"
    assert data["status"] == "checkpointed_partial"
    assert data["checkpoint_request_id"] == outcome["checkpoint_request_id"]
    assert data["progress"]["stage"] == "loss_layer"
    assert 0 < data["progress"]["completed"] < _ENDLESS_REPS


def test_retry_after_an_intervention_launches_again(tmp_path: Path, mlx_guard: Any) -> None:  # noqa: ARG001
    # Catches: a fixed report path. The supervisor keeps `.<report>.journal` and refuses the
    # path next time, so the resume retry would die as a client error before any launch.
    condition = _condition("endless", reps=_ENDLESS_REPS)
    policy = _wall_limited(1_500)
    run_conditions([condition], tmp_path, session_id="s", guard=policy)
    paths = run_conditions([condition], tmp_path, session_id="s", guard=policy)

    guard_dir = tmp_path / "_mlx_guard"
    assert not (guard_dir / "endless.client.json").exists()
    assert _supervision(tmp_path, "endless")["kind"] == "policy_intervention"
    assert json.loads(paths[0].read_text())["status"] == "checkpointed_partial"
    reports = [p for p in guard_dir.glob("endless.*.json") if p.name.count(".") == 2
               and p.name.split(".")[1] not in ("launch", "client", "supervision")]
    assert len(reports) == 2


def test_the_in_process_wall_backstop_still_fires_under_supervision(
    tmp_path: Path, mlx_guard: Any,  # noqa: ARG001
) -> None:
    # Catches: supervision quietly replacing the worker's own protections. No supervisor
    # limit can trip here, so the abort can only come from the in-process watchdog -- and
    # its exit code 70 must read as the WORKER's status, not as a supervisor failure.
    paths = run_conditions(
        [_condition("backstop", reps=_ENDLESS_REPS, wall_budget_s=2.0)], tmp_path,
        session_id="s", guard=ExternalGuardConfig(max_footprint_bytes=8 * _GIB),
    )

    assert json.loads(paths[0].read_text())["status"] == "aborted_wall_budget"
    outcome = _supervision(tmp_path, "backstop")
    assert outcome["kind"] == "child_exited"
    assert outcome["returncode"] == 70


def test_a_failed_checkpoint_write_is_never_acknowledged_as_completed(
    tmp_path: Path, mlx_guard: Any,
) -> None:
    # Catches: acknowledging `completed` before (or regardless of) the partial artifact
    # reaching the disk. The artifact directory is read-only, so the callback's write
    # fails; the supervisor must not record a completed acknowledgement or an artifact.
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    out = sealed / "endless.json"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "kind": "loss_layer", "session_id": "s", "attention_impl": None,
        "wall_budget_s": None, "out": str(out),
        "params": {"n": 64, "d": 64, "v": 128, "dtype": "float32", "impl": "naive",
                   "reps": _ENDLESS_REPS},
    }))
    reports = tmp_path / "reports"
    reports.mkdir(mode=0o700)
    sealed.chmod(0o500)
    try:
        result = mlx_guard.run(
            mlx_guard.RunConfig(
                command=(sys.executable, "-m", "mlx_train_perf.bench.worker",
                         "--config", str(config)),
                report=reports / "endless.json", max_footprint_bytes=8 * _GIB,
                wall_time_ms=1_500, sample_interval_ms=10, checkpoint_timeout_ms=10_000,
            ),
            capture_output=True,
        )
    finally:
        sealed.chmod(0o700)

    checkpoint = result.report.payload["checkpoint"]
    assert checkpoint["status"] == "requested_unverified"  # asked, never completed
    assert "artifact" not in checkpoint
    assert not out.exists()
    # And the worker outlived its failed checkpoint long enough to be stopped cleanly. A
    # slow-exiting worker that dies first can be gone before the TERM lands, and the
    # supervisor then records `supervisor_failure` (exit 70) instead -- observed with this
    # worker on mlx-guard 0.2.0; it is a race, so only the safe outcome is pinned here.
    assert result.report.outcome.kind.value == "policy_intervention"
    assert dict(result.report.payload["outcome"]["child_status"]) == {
        "status": "signaled", "signal": 15,
    }
