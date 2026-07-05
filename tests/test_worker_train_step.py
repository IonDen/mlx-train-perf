"""Tests for `bench/worker.py`'s `train_step` condition kind.

Requires the optional `mlx-lm` extra; the whole module is skipped (not failed) when it
is absent, matching `tests/test_adapter.py`'s own convention.

The wired-limit re-assert mechanism this module cares about is exercised at TWO levels:
  - `wired_cap_holds` (pure, in `core/guards.py`) already has its own direct tests in
    `tests/test_guards.py` -- not repeated here.
  - `run_train_step`'s "artifact records the observed limit" contract IS repeated here,
    end to end, against a real (but tiny and freshly-constructed, never downloaded)
    `llama.Model` -- `_load_model` is the ONLY call site that would touch the network
    or a real checkpoint, and every test below monkeypatches it. No test in this file
    loads a real pretrained model.
"""
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm")

from mlx_lm.models import llama

from mlx_train_perf.bench import worker
from mlx_train_perf.core.guards import clamped_caps, install_guardrails
from mlx_train_perf.errors import MissingDependencyError, WiredCapRegressionError


def _tiny_llama() -> llama.Model:
    args = llama.ModelArgs(
        model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    return llama.Model(args)


def _stub_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_load(_model_id: str, _revision: str | None) -> tuple[Any, Any]:
        return _tiny_llama(), object()
    monkeypatch.setattr(worker, "_load_model", _fake_load)


_BASE_PARAMS: dict[str, object] = {
    "model": "tiny-llama", "seq_len": 16, "batch": 1, "steps": 2,
    "impl": "chunked", "lora_rank": 4, "lora_layers": 1, "seed": 0,
}


# ---------------------------------------------------------------------------
# run_train_step: ok-path field contract (ours and stock)
# ---------------------------------------------------------------------------


def test_run_train_step_ours_reports_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    fields = worker.run_train_step(dict(_BASE_PARAMS))

    assert len(fields["tokens_per_sec_all"]) == 2  # type: ignore[arg-type]
    assert len(fields["loss_all"]) == 2  # type: ignore[arg-type]
    tps_all = fields["tokens_per_sec_all"]
    assert isinstance(tps_all, list)
    # `tokens_per_sec_median` is computed from the RAW (unrounded) per-step values,
    # then rounded once (the same house convention `run_loss_layer`/
    # `bench_backward_ladder.py` already use) -- recomputing it from the ALREADY-
    # rounded `tokens_per_sec_all` list can differ at the 3rd decimal by a rounding-
    # order artifact, so this only checks it lands near the right value, not bit-exact.
    assert fields["tokens_per_sec_median"] == pytest.approx(statistics.median(tps_all), abs=0.01)
    assert all(v > 0 for v in tps_all)
    assert fields["active_before_gb"] >= 0  # type: ignore[operator]
    assert fields["marginal_peak_gb"] >= 0  # type: ignore[operator]


def test_run_train_step_stock_reports_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    params = {**_BASE_PARAMS, "stock": True}
    fields = worker.run_train_step(params)

    assert len(fields["loss_all"]) == 2  # type: ignore[arg-type]
    assert len(fields["tokens_per_sec_all"]) == 2  # type: ignore[arg-type]


def test_run_train_step_ours_with_grad_checkpoint_still_reports_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`grad_checkpoint=True` on the manual-loop (`ours`) path applies mlx-lm's own
    `grad_checkpoint` helper to `model.layers[0]` (matching what `train()` itself does
    internally for `stock`) -- exercised here as a real, not mocked, call."""
    _stub_load(monkeypatch)
    install_guardrails()
    params = {**_BASE_PARAMS, "grad_checkpoint": True}
    fields = worker.run_train_step(params)
    assert len(fields["loss_all"]) == 2  # type: ignore[arg-type]


def test_require_mlx_lm_raises_missing_dependency_error_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same CPython mechanism `tests/test_adapter.py::test_missing_mlx_lm_dependency_
    is_typed_error` already established: an `import mlx_lm` inside a `try` block sees
    an `ImportError` when the module is present in `sys.modules` as `None`."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(MissingDependencyError, match="mlx-lm"):
        worker._require_mlx_lm()


# ---------------------------------------------------------------------------
# THE load-bearing contract: the artifact records the OBSERVED wired limit, and it
# must equal the house cap -- a mismatch is a FAILED condition, not a silent pass.
# ---------------------------------------------------------------------------


def test_run_train_step_records_observed_wired_limit_at_the_house_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    fields = worker.run_train_step(dict(_BASE_PARAMS))

    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _soft = clamped_caps(dev_max)
    expected_gb = round(wired / 1024**3, 4)

    assert fields["observed_wired_limit_gb"] == expected_gb
    assert fields["house_wired_limit_gb"] == expected_gb
    # The in-loop one-shot reassert actually fired and observed the hazard: mlx_lm's
    # train() raised the limit to the device max at its own entry before we lowered it.
    assert fields["wired_limit_before_reassert_gb"] is not None
    assert fields["wired_limit_before_reassert_gb"] >= expected_gb  # type: ignore[operator]


def test_run_train_step_raises_when_observed_limit_does_not_match_the_house_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the exact hazard this condition kind exists to catch: the wired
    limit read back after training does NOT match the house cap (e.g. because the
    in-loop reassert never fired, or something re-raised it afterward). Directly
    forces the failing branch by making the SECOND `install_guardrails` call (the
    post-train observation -- the FIRST call is the in-loop one-shot reassert, which
    must still behave normally so the rest of the pipeline runs) return a bogus value,
    rather than depending on mlx-lm's own internals to reproduce it."""
    _stub_load(monkeypatch)
    install_guardrails()
    real_install = worker.install_guardrails
    calls = {"n": 0}

    def _flaky_install(**kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_install(**kwargs)  # type: ignore[arg-type]
        return 999_999_999_999  # bogus: simulates a regressed/never-reasserted cap

    monkeypatch.setattr(worker, "install_guardrails", _flaky_install)

    with pytest.raises(WiredCapRegressionError, match="wired limit"):
        worker.run_train_step(dict(_BASE_PARAMS))


# ---------------------------------------------------------------------------
# worker.main: kind="train_step" end-to-end dispatch
# ---------------------------------------------------------------------------


def test_worker_main_train_step_writes_ok_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_load(monkeypatch)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "train_step", "params": dict(_BASE_PARAMS), "session_id": "s1",
        "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["identity"]["kind"] == "train_step"
    assert len(data["loss_all"]) == 2


def test_worker_main_train_step_wired_cap_regression_is_not_caught_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `WiredCapRegressionError` is NOT a `LaunchBudgetError` -- `worker.main` must
    not swallow it as a "refused" result. It propagates uncaught (same as an
    unsupported kind), so `runner.run_conditions` records the sweep-level crash
    envelope on the CALLER's side, never a silent "ok"."""
    _stub_load(monkeypatch)

    def _raise(_params: dict[str, object]) -> dict[str, object]:
        raise WiredCapRegressionError("wired limit regressed")

    monkeypatch.setattr(worker, "run_train_step", _raise)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "train_step", "params": dict(_BASE_PARAMS), "session_id": "s1",
        "out": str(out),
    }))
    with pytest.raises(WiredCapRegressionError):
        worker.main(["--config", str(cfg)])
    assert not out.exists()
