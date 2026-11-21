"""The self-writing bench scripts' final writes: a breach record is never replaced, and a
non-finite recorded loss is never written as `ok`.

`scripts/` is not a package, so each script is loaded by path (the convention of
`tests/test_bench_attention_op.py`). Every process-global call a script's per-condition
child makes before measuring (the wired cap, the cache limit, the memory ceiling, the
watchdog thread) is stubbed, and the measurement itself is replaced by a function that
returns literal fields: nothing here allocates on the GPU or loads a model.

A breach is simulated as in the worker's
`test_a_result_computed_during_a_breach_never_replaces_the_breach_record`: the stubbed
measurement builds a `make_watchdog_on_breach` callback for the same artifact with an
injected `exit_fn` and fires it before the script's own final write.
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_attention_op  # noqa: E402 -- import must follow the sys.path insert
import bench_qwen35_convergence  # noqa: E402
import bench_qwen35_training  # noqa: E402
import bench_recurrent_layer  # noqa: E402
from bench_attention_op import AttnCondition  # noqa: E402
from bench_qwen35_convergence import ConvergenceCondition  # noqa: E402
from bench_qwen35_training import TrainCondition  # noqa: E402
from bench_recurrent_layer import LayerCondition  # noqa: E402

from mlx_train_perf.bench import artifacts  # noqa: E402
from mlx_train_perf.core.guards import EffectiveCeiling  # noqa: E402
from mlx_train_perf.errors import LaunchBudgetError  # noqa: E402


class _FakeWatchdog:
    def stop(self) -> None:
        return None


def _fire_breach(out_path: Path) -> None:
    """Record a memory-ceiling breach for `out_path` the way the installed watchdog
    would, minus the hard exit."""
    on_breach = artifacts.make_watchdog_on_breach(
        out_path, {"session_id": "s1"}, 1 << 30, exit_fn=lambda _code: None,
    )
    on_breach("memory_ceiling", {"active_bytes": 2 << 30, "elapsed_s": 1.0})


def _stub_process_globals(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setattr(
        module, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )
    monkeypatch.setattr(module, "install_memory_watchdog", lambda **_kwargs: _FakeWatchdog())
    if hasattr(module, "install_guardrails"):
        monkeypatch.setattr(module, "install_guardrails", lambda: 0)
    monkeypatch.setattr(mx, "set_cache_limit", lambda _limit: 0)


def _status(path: Path) -> str:
    return str(json.loads(path.read_text())["status"])


# --- bench_attention_op.py ---------------------------------------------------------------


def _attn_condition() -> AttnCondition:
    return AttnCondition(name="flash_n256", impl="flash", n=256, head_dim=64, heads=4, kv_heads=2)


def test_attention_op_ok_computed_during_a_breach_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: `_run_single_condition`'s final `ok` written with plain `write_result`, landing
    # after the watchdog recorded the breach: a measurement past the ceiling served as fresh.
    _stub_process_globals(monkeypatch, bench_attention_op)
    out_path = tmp_path / "flash_n256.json"

    def measure_while_the_watchdog_fires(_c: AttnCondition) -> dict[str, object]:
        _fire_breach(out_path)
        return {"impl": "flash", "n": 256, "wall_s": 0.001}

    monkeypatch.setattr(bench_attention_op, "measure_condition", measure_while_the_watchdog_fires)

    bench_attention_op._run_single_condition(_attn_condition(), out_dir=tmp_path, session_id="s1")

    assert _status(out_path) == "aborted_memory_ceiling"


def test_attention_op_refusal_computed_during_a_breach_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the `refused` write left as plain `write_result` next to the watchdog.
    _stub_process_globals(monkeypatch, bench_attention_op)
    out_path = tmp_path / "flash_n256.json"

    def refuse_while_the_watchdog_fires(_c: AttnCondition) -> dict[str, object]:
        _fire_breach(out_path)
        raise LaunchBudgetError("shape exceeds the launch budget")

    monkeypatch.setattr(bench_attention_op, "measure_condition", refuse_while_the_watchdog_fires)

    bench_attention_op._run_single_condition(_attn_condition(), out_dir=tmp_path, session_id="s1")

    assert _status(out_path) == "aborted_memory_ceiling"


# --- bench_qwen35_training.py ------------------------------------------------------------


def _train_condition() -> TrainCondition:
    return TrainCondition(
        name="stock_s512", model="m", revision=None, arm="stock", seq_len=512, batch=1,
        steps=3, lora_rank=8, lora_layers=4, learning_rate=1e-5, seed=0,
        grad_checkpoint=False,
    )


def _run_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loss_all: list[float], *,
    breach: bool = False,
) -> dict[str, object]:
    _stub_process_globals(monkeypatch, bench_qwen35_training)
    out_path = tmp_path / "stock_s512.json"

    def measure(_c: TrainCondition) -> dict[str, object]:
        if breach:
            _fire_breach(out_path)
        return {"arm": "stock", "seq_len": 512, "loss_all": loss_all, "marginal_peak_gb": 1.5}

    monkeypatch.setattr(bench_qwen35_training, "measure_condition", measure)
    written = bench_qwen35_training._run_single_condition(
        _train_condition(), out_dir=tmp_path, session_id="s1",
    )
    assert written == out_path
    return dict(json.loads(out_path.read_text()))


def test_training_ok_computed_during_a_breach_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the final `ok` written with plain `write_result` after the breach record.
    record = _run_training(tmp_path, monkeypatch, [2.5, 2.25, 2.0], breach=True)
    assert record["status"] == "aborted_memory_ceiling"


def test_training_inf_loss_is_recorded_as_nonfinite_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: no finiteness check before the final write (a diverged run recorded `ok`).
    record = _run_training(tmp_path, monkeypatch, [2.5, math.inf, 2.0])
    assert record["status"] == "nonfinite_loss"
    assert record["error"] == "non-finite loss at loss_all[1]"
    assert record["loss_all"] == [2.5, math.inf, 2.0]
    assert record["marginal_peak_gb"] == 1.5


def test_training_nan_loss_is_recorded_as_nonfinite_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: a check that only looks for infinities (`math.isinf`) and misses NaN.
    record = _run_training(tmp_path, monkeypatch, [math.nan, 2.25, 2.0])
    assert record["status"] == "nonfinite_loss"
    assert record["error"] == "non-finite loss at loss_all[0]"


def test_training_finite_loss_still_writes_ok_with_its_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: a check that fires on every run, or one that drops or renames fields.
    record = _run_training(tmp_path, monkeypatch, [2.5, 2.25, 2.0])
    assert record["status"] == "ok"
    assert "error" not in record
    assert record["loss_all"] == [2.5, 2.25, 2.0]
    assert record["arm"] == "stock"
    assert record["marginal_peak_gb"] == 1.5


def test_training_nonfinite_write_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the `nonfinite_loss` write made with plain `write_result`.
    record = _run_training(tmp_path, monkeypatch, [2.5, math.nan, 2.0], breach=True)
    assert record["status"] == "aborted_memory_ceiling"


# --- bench_qwen35_convergence.py ---------------------------------------------------------


def _convergence_condition() -> ConvergenceCondition:
    return ConvergenceCondition(
        name="stock", model="m", revision=None, data="d.jsonl", dataset_sha="0" * 64,
        arm="stock", max_seq_length=512, batch_size=1, steps=4, val_every=2,
        val_examples=2, lora_rank=8, lora_layers=4, learning_rate=1e-5, seed=0,
        grad_checkpoint=False,
    )


def _run_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    train_losses: list[float], val_losses: list[float], breach: bool = False,
) -> dict[str, object]:
    _stub_process_globals(monkeypatch, bench_qwen35_convergence)
    out_path = tmp_path / "stock.json"

    def run(_c: ConvergenceCondition, *, out_path: Path, identity: object) -> dict[str, object]:  # noqa: ARG001
        if breach:
            _fire_breach(out_path)
        # The shape `mlx_lm.tuner.trainer.train` reports through the callback: one dict per
        # report, the loss a Python float under `train_loss` / `val_loss`.
        return {
            "train_info": [
                {"iteration": i + 1, "train_loss": loss} for i, loss in enumerate(train_losses)
            ],
            "val_info": [
                {"iteration": i * 2, "val_loss": loss} for i, loss in enumerate(val_losses)
            ],
            "marginal_peak_gb": 1.5,
        }

    monkeypatch.setattr(bench_qwen35_convergence, "run_convergence", run)
    returned = bench_qwen35_convergence._run_single_condition(
        _convergence_condition(), out_path=out_path, identity={"session_id": "s1"},
    )
    on_disk = dict(json.loads(out_path.read_text()))
    assert returned["status"] == on_disk["status"]
    return on_disk


def test_convergence_ok_computed_during_a_breach_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the final `ok` written with plain `write_result` after the breach record.
    record = _run_convergence(
        tmp_path, monkeypatch, train_losses=[2.5, 2.0], val_losses=[2.6, 2.1], breach=True,
    )
    assert record["status"] == "aborted_memory_ceiling"


def test_convergence_inf_train_loss_is_recorded_as_nonfinite_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: no check, or a check handed the record's own `train_info` (a list of dicts,
    # which the shared helper skips) instead of the projected loss values.
    record = _run_convergence(
        tmp_path, monkeypatch, train_losses=[2.5, math.inf], val_losses=[2.6, 2.1],
    )
    assert record["status"] == "nonfinite_loss"
    assert record["error"] == "non-finite loss at train_info[1]"
    assert record["train_info"] == [
        {"iteration": 1, "train_loss": 2.5}, {"iteration": 2, "train_loss": math.inf},
    ]
    assert record["marginal_peak_gb"] == 1.5


def test_convergence_nan_validation_loss_is_recorded_as_nonfinite_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: a check that projects the training history but not the validation one.
    record = _run_convergence(
        tmp_path, monkeypatch, train_losses=[2.5, 2.0], val_losses=[math.nan, 2.1],
    )
    assert record["status"] == "nonfinite_loss"
    assert record["error"] == "non-finite loss at val_info[0]"


def test_convergence_finite_history_still_writes_ok_with_its_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: a check that fires on every run, or one that rewrites the recorded history.
    record = _run_convergence(
        tmp_path, monkeypatch, train_losses=[2.5, 2.0], val_losses=[2.6, 2.1],
    )
    assert record["status"] == "ok"
    assert "error" not in record
    assert record["train_info"] == [
        {"iteration": 1, "train_loss": 2.5}, {"iteration": 2, "train_loss": 2.0},
    ]
    assert record["val_info"] == [
        {"iteration": 0, "val_loss": 2.6}, {"iteration": 2, "val_loss": 2.1},
    ]


def test_convergence_nonfinite_write_never_replaces_the_breach_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the `nonfinite_loss` write made with plain `write_result`.
    record = _run_convergence(
        tmp_path, monkeypatch, train_losses=[math.nan, 2.0], val_losses=[2.6, 2.1],
        breach=True,
    )
    assert record["status"] == "aborted_memory_ceiling"


def test_convergence_progress_flush_after_a_breach_never_replaces_the_breach_record(
    tmp_path: Path,
) -> None:
    # Catches: `_FlushingCallback._flush` writing `in_progress` with plain `write_result`, so a
    # loss report between the breach record and the hard exit erases the abort reason.
    out_path = tmp_path / "stock.json"
    callback = bench_qwen35_convergence._FlushingCallback(out_path, {"session_id": "s1"})
    callback.on_train_loss_report({"iteration": 1, "train_loss": 2.5})
    assert _status(out_path) == "in_progress"

    _fire_breach(out_path)
    callback.on_train_loss_report({"iteration": 2, "train_loss": 2.25})
    callback.on_val_loss_report({"iteration": 2, "val_loss": 2.3})

    assert _status(out_path) == "aborted_memory_ceiling"


# --- bench_recurrent_layer.py --------------------------------------------------------------
# Its breach callback writes no record and exits 2, so the orchestrator is what keeps a
# breached measurement out of the results: an `ok` left by a worker that exited nonzero is
# replaced by the crash record.


def _layer_condition() -> LayerCondition:
    return LayerCondition(
        name="kernel_t512", impl="kernel", seq_len=512, batch=1, dtype="bfloat16",
        chunk_size=64, reps=3,
    )


_UNREADABLE = "<unreadable>"


def _run_grid(
    module: Any, condition: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    writes: str | None, returncode: int, stderr: str = "",
) -> dict[str, object] | None:
    """`module.run_grid` over one condition with a fake worker that writes `writes` (a
    status, `_UNREADABLE` for a truncated file, or nothing) for the artifact it was given,
    then exits `returncode`. Its stderr defaults to empty: a watchdog breach hard-exits
    without printing anything."""

    def fake_spawn(
        condition: object, *, out_dir: Path, session_id: str,
    ) -> subprocess.CompletedProcess[str]:
        out_path = module._out_path(out_dir, condition)
        if writes == _UNREADABLE:
            out_path.write_text('{"status": "o')
        elif writes is not None:
            artifacts.write_result(
                out_path, module._identity_for(condition, session_id=session_id),
                writes, fwd=0.01, fwd_backward=0.03,
            )
        return subprocess.CompletedProcess(
            args=["worker"], returncode=returncode, stdout="", stderr=stderr,
        )

    monkeypatch.setattr(module, "_spawn_condition", fake_spawn)
    [path] = module.run_grid([condition], out_dir=tmp_path, session_id="s1")
    return dict(json.loads(path.read_text())) if path.exists() else None


def _run_recurrent_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, writes: str | None, returncode: int,
) -> dict[str, object] | None:
    return _run_grid(
        bench_recurrent_layer, _layer_condition(), tmp_path, monkeypatch,
        writes=writes, returncode=returncode,
    )


def test_recurrent_grid_never_keeps_an_ok_from_a_worker_that_exited_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: `run_grid` recording a crash only when no artifact exists, so an `ok` that
    # landed just before the watchdog's `os._exit(2)` is kept as fresh forever.
    record = _run_recurrent_grid(tmp_path, monkeypatch, writes="ok", returncode=2)
    assert record is not None
    assert record["status"] == "error"
    assert record["error_type"] == "WorkerCrashed"
    assert record["returncode"] == 2


def test_recurrent_grid_keeps_an_ok_from_a_clean_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: replacing every `ok` (the rc test inverted or dropped).
    record = _run_recurrent_grid(tmp_path, monkeypatch, writes="ok", returncode=0)
    assert record is not None
    assert record["status"] == "ok"
    assert record["fwd_backward"] == 0.03


def test_recurrent_grid_records_a_crash_when_the_worker_wrote_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: the new `ok` check replacing, instead of extending, the missing-artifact case.
    record = _run_recurrent_grid(tmp_path, monkeypatch, writes=None, returncode=2)
    assert record is not None
    assert record["error_type"] == "WorkerCrashed"
    assert record["returncode"] == 2


def test_recurrent_grid_keeps_a_non_ok_record_from_a_worker_that_exited_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches: replacing whatever a failed worker wrote, which would erase its own non-`ok`
    # reason (only an `ok` is unsafe to keep).
    record = _run_recurrent_grid(tmp_path, monkeypatch, writes="refused", returncode=1)
    assert record is not None
    assert record["status"] == "refused"
    assert "error_type" not in record


# --- every grid: a nonzero exit never keeps an `ok` or an unreadable artifact ------------
# The attention and training children write their breach record over an `ok`, but that
# write can fail (or lose the lock) after the `ok` landed, and the watchdog still exits 70.
# The orchestrator is the backstop, the same rule as the recurrent grid's above.

_GRID_SITES = [
    pytest.param(bench_attention_op, _attn_condition, id="attention"),
    pytest.param(bench_qwen35_training, _train_condition, id="training"),
    pytest.param(bench_recurrent_layer, _layer_condition, id="recurrent"),
]


@pytest.mark.parametrize(("module", "make_condition"), _GRID_SITES)
def test_grid_never_keeps_an_ok_from_a_worker_that_exited_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, make_condition: Any,
) -> None:
    # Catches: `run_grid` recording a crash only when no artifact exists, so an `ok` left by
    # a worker whose breach write failed before its exit 70 is served as fresh on resume.
    record = _run_grid(
        module, make_condition(), tmp_path, monkeypatch, writes="ok", returncode=70,
    )
    assert record is not None
    assert record["status"] == "error"
    assert record["error_type"] == "WorkerCrashed"
    assert record["returncode"] == 70


@pytest.mark.parametrize(("module", "make_condition"), _GRID_SITES)
def test_grid_records_a_crash_over_an_unreadable_artifact_after_a_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, make_condition: Any,
) -> None:
    # Catches: keeping a truncated artifact from a worker that died mid-write, which loses
    # the exit code and the traceback that say why.
    traceback = "Traceback (most recent call last):\nOSError: [Errno 28] No space left on device\n"
    record = _run_grid(
        module, make_condition(), tmp_path, monkeypatch,
        writes=_UNREADABLE, returncode=1, stderr=traceback,
    )
    assert record is not None
    assert record["error_type"] == "WorkerCrashed"
    assert record["returncode"] == 1
    assert record["error_msg"] == traceback


@pytest.mark.parametrize(("module", "make_condition"), _GRID_SITES)
def test_grid_keeps_an_ok_from_a_clean_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, make_condition: Any,
) -> None:
    # Catches: replacing every `ok` (the exit-code test inverted or dropped).
    record = _run_grid(module, make_condition(), tmp_path, monkeypatch, writes="ok", returncode=0)
    assert record is not None
    assert record["status"] == "ok"
    assert record["fwd_backward"] == 0.03


@pytest.mark.parametrize(("module", "make_condition"), _GRID_SITES)
def test_grid_keeps_a_breach_record_from_a_worker_that_exited_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, make_condition: Any,
) -> None:
    # Catches: replacing whatever a failed worker wrote, which would erase the breach
    # record the watchdog wrote before its exit 70.
    record = _run_grid(
        module, make_condition(), tmp_path, monkeypatch,
        writes="aborted_memory_ceiling", returncode=70,
    )
    assert record is not None
    assert record["status"] == "aborted_memory_ceiling"
    assert "error_type" not in record


# --- exit codes for a diverged run ---------------------------------------------------------


@pytest.mark.parametrize(("status", "rc"), [("nonfinite_loss", 0), ("error", 1), ("ok", 0)])
def test_training_single_condition_exit_code_per_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, rc: int,
) -> None:
    # Catches: a change to the rc mapping that makes a diverged training child exit nonzero,
    # which a wrapping run would then read as a crash (only `error` exits 1).
    def run_single(condition: TrainCondition, *, out_dir: Path, session_id: str) -> Path:
        out_path = bench_qwen35_training._out_path(out_dir, condition)
        artifacts.write_result(
            out_path, bench_qwen35_training._identity_for(condition, session_id=session_id),
            status, loss_all=[2.5, math.nan],
        )
        return out_path

    monkeypatch.setattr(bench_qwen35_training, "_run_single_condition", run_single)
    argv = [
        "--model", "m", "--arm", "stock", "--seq-lens", "512", "--single-condition",
        "--session-id", "s1", "--out-dir", str(tmp_path),
    ]
    assert bench_qwen35_training.main(argv) == rc


@pytest.mark.parametrize(("status", "rc"), [("nonfinite_loss", 1), ("ok", 0)])
def test_convergence_exit_code_per_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, rc: int,
) -> None:
    # Catches: a change to the rc mapping that lets a diverged convergence run exit 0, so a
    # wrapping run counts it as a clean measurement.
    data = tmp_path / "d.jsonl"
    data.write_text('{"text": "x"}\n')

    def run_single(
        _c: ConvergenceCondition, *, out_path: Path, identity: dict[str, object],
    ) -> dict[str, object]:
        artifacts.write_result(out_path, identity, status, train_info=[])
        return dict(json.loads(out_path.read_text()))

    monkeypatch.setattr(bench_qwen35_convergence, "_run_single_condition", run_single)
    argv = [
        "--model", "m", "--data", str(data), "--arm", "stock", "--out", str(tmp_path / "out"),
    ]
    assert bench_qwen35_convergence.main(argv) == rc
