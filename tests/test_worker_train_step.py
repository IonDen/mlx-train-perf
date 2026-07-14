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

import mlx_lm.tuner.utils as lora_utils
from mlx_lm.models import llama

from mlx_train_perf.bench import worker
from mlx_train_perf.core.guards import EffectiveCeiling, clamped_caps, install_guardrails
from mlx_train_perf.errors import (
    MissingDependencyError,
    MlxTrainPerfError,
    WiredCapRegressionError,
)


@pytest.fixture(autouse=True)
def _plentiful_memory_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small CI runners (7 GB, ~3.5 GB free) trip guards' safe-start floor inside
    `worker.main` -- that is the runner's environment, not the behavior under test.
    Every in-process test in this module runs as if the machine had honest room; the
    floor's own decision logic is covered with injected readers in the guards tests."""
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )


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


def test_run_train_step_casts_model_to_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`compute_dtype` casts the loaded model's floating params right after load, before
    freeze/LoRA injection, so the kernel loss path (which accepts only fp32/bf16 hidden)
    can run on a model that otherwise computes in fp16 -- the real reason this exists:
    4-bit MLX checkpoints store fp16 scales and compute in fp16 regardless of their
    config `torch_dtype`. The final RMSNorm weight is the stable observable: a trunk
    param `linear_to_lora_layers` never wraps and `model.freeze()` keeps frozen, so it
    holds the cast dtype unchanged through training."""
    model = _tiny_llama()
    assert model.model.norm.weight.dtype == mx.float32  # constructed fp32
    monkeypatch.setattr(worker, "_load_model", lambda _m, _r: (model, object()))
    install_guardrails()

    worker.run_train_step({**_BASE_PARAMS, "compute_dtype": "bfloat16"})

    assert model.model.norm.weight.dtype == mx.bfloat16


def test_run_train_step_evaluates_setup_before_the_memory_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """review-mlx finding: `set_dtype` and `linear_to_lora_layers` build LAZY graphs
    (`Module.apply` astype; LoRA's random-init), and `mlx_lm.load(lazy=False)` only
    materializes the ORIGINAL weights -- so without a forced eval, the one-time cast +
    adapter init execute inside the `reset_peak_memory()`->`get_peak_memory()` window
    and land in `marginal_peak_gb` (and in step 1's wall time). The contract: at least
    one `mx.eval` runs BEFORE the first `reset_peak_memory` call. The spies call
    through, so the step still trains for real."""
    model = _tiny_llama()
    monkeypatch.setattr(worker, "_load_model", lambda _m, _r: (model, object()))
    install_guardrails()

    calls: list[str] = []
    real_eval = worker.mx.eval
    real_reset = worker.mx.reset_peak_memory

    def spy_eval(*args: Any) -> None:
        calls.append("eval")
        real_eval(*args)

    def spy_reset() -> None:
        calls.append("reset")
        real_reset()

    monkeypatch.setattr(worker.mx, "eval", spy_eval)
    monkeypatch.setattr(worker.mx, "reset_peak_memory", spy_reset)

    worker.run_train_step({**_BASE_PARAMS, "compute_dtype": "bfloat16"})

    assert "reset" in calls
    assert "eval" in calls[: calls.index("reset")]


def test_run_train_step_without_compute_dtype_leaves_model_dtype_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (no `compute_dtype`): the model runs at its loaded dtype, uncast -- the
    smoke/chunked path relies on this (fp16 models stay fp16 for chunked/naive, which
    accept them; only the kernel arm needs the bf16 cast)."""
    model = _tiny_llama()
    monkeypatch.setattr(worker, "_load_model", lambda _m, _r: (model, object()))
    install_guardrails()

    worker.run_train_step(dict(_BASE_PARAMS))  # no compute_dtype key

    assert model.model.norm.weight.dtype == mx.float32


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


# ---------------------------------------------------------------------------
# attention_impl threading (T13 step 1): "stock" (default) leaves the model's
# self_attn untouched; "flash" must call `enable_flash_attention` with THIS run's
# training shape (seq_len/batch), AFTER the compute-dtype cast is materialized and
# BEFORE freeze/LoRA injection -- LoRA target discovery walks `named_modules()` by
# path, so the wrapper must already be in the tree when LoRA lands inside it
# (`FlashAttentionWrapper`'s own docstring, reason 1). `enable_flash_attention` is
# stubbed in every test below: the tiny llama fixture's head_dim (16) is outside the
# kernel's supported set ({64, 96, 128}), and exercising the REAL wrapper/kernel here
# would add a GPU dispatch the default pytest lane does not otherwise pay for --
# `attention/wrapper.py` and `attention/api.py` have their own dedicated test
# coverage for the real wrapping/kernel behavior.
# ---------------------------------------------------------------------------


def test_run_train_step_default_attention_impl_is_stock_and_skips_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    called = {"n": 0}

    def spy_enable(*_a: Any, **_kw: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr(worker, "enable_flash_attention", spy_enable)

    fields = worker.run_train_step(dict(_BASE_PARAMS))  # no attention_impl key

    assert called["n"] == 0
    assert len(fields["loss_all"]) == 2  # type: ignore[arg-type]


def test_run_train_step_explicit_stock_attention_impl_skips_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    called = {"n": 0}

    def spy_enable(*_a: Any, **_kw: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr(worker, "enable_flash_attention", spy_enable)

    worker.run_train_step(dict(_BASE_PARAMS), attention_impl="stock")

    assert called["n"] == 0


def test_run_train_step_flash_attention_impl_enables_before_lora_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing ordering contract: `enable_flash_attention` fires with THIS
    run's seq_len/batch hints, before `model.freeze()`/`linear_to_lora_layers`."""
    model = _tiny_llama()
    monkeypatch.setattr(worker, "_load_model", lambda _m, _r: (model, object()))
    install_guardrails()

    calls: list[str] = []
    enable_args: dict[str, object] = {}

    def spy_enable(m: Any, **kw: Any) -> None:
        calls.append("enable_flash_attention")
        enable_args["model"] = m
        enable_args["seq_len"] = kw.get("seq_len")
        enable_args["batch_size"] = kw.get("batch_size")

    monkeypatch.setattr(worker, "enable_flash_attention", spy_enable)

    real_freeze = model.freeze

    def spy_freeze(*a: Any, **kw: Any) -> Any:
        calls.append("freeze")
        return real_freeze(*a, **kw)

    monkeypatch.setattr(model, "freeze", spy_freeze)

    real_lora = lora_utils.linear_to_lora_layers

    def spy_lora(*a: Any, **kw: Any) -> Any:
        calls.append("linear_to_lora_layers")
        return real_lora(*a, **kw)

    monkeypatch.setattr(lora_utils, "linear_to_lora_layers", spy_lora)

    fields = worker.run_train_step(dict(_BASE_PARAMS), attention_impl="flash")

    assert calls == ["enable_flash_attention", "freeze", "linear_to_lora_layers"]
    assert enable_args["model"] is model
    assert enable_args["seq_len"] == _BASE_PARAMS["seq_len"]
    assert enable_args["batch_size"] == _BASE_PARAMS["batch"]
    assert len(fields["loss_all"]) == 2  # type: ignore[arg-type]


def test_run_train_step_evaluates_parameters_before_enabling_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gotcha 14: `enable_flash_attention`'s prewarm runs a REAL calibration forward
    with its own `mx.eval` -- the compute_dtype cast above must already be
    materialized before that happens, not implicitly forced as a side effect of the
    calibration's own eval (which would land the one-time cast cost inside whatever
    the calibration itself measures/attributes)."""
    model = _tiny_llama()
    monkeypatch.setattr(worker, "_load_model", lambda _m, _r: (model, object()))
    install_guardrails()

    calls: list[str] = []
    real_eval = worker.mx.eval

    def spy_eval(*args: Any) -> None:
        calls.append("eval")
        real_eval(*args)

    def spy_enable(*_a: Any, **_kw: Any) -> None:
        calls.append("enable_flash_attention")

    monkeypatch.setattr(worker.mx, "eval", spy_eval)
    monkeypatch.setattr(worker, "enable_flash_attention", spy_enable)

    worker.run_train_step(
        {**_BASE_PARAMS, "compute_dtype": "bfloat16"}, attention_impl="flash",
    )

    assert "eval" in calls
    assert "enable_flash_attention" in calls
    assert calls.index("eval") < calls.index("enable_flash_attention")


def test_run_train_step_rejects_unknown_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_load(monkeypatch)
    install_guardrails()
    with pytest.raises(MlxTrainPerfError, match="attention_impl"):
        worker.run_train_step(dict(_BASE_PARAMS), attention_impl="bogus")


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


def test_worker_main_threads_config_attention_impl_into_identity_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`worker.main` must read the TOP-LEVEL `attention_impl` config field (the dedicated
    slot `run_conditions` now writes it into, out of `params`) and thread it BOTH into its
    own `condition_identity` call (so the artifact's identity carries it) AND into
    `run_train_step`. Regression for the T13 step-1 seam: before the fix, `worker.main`
    ignored the config field entirely, so the identity never carried `attention_impl` and
    the run always defaulted to stock. `run_train_step` is stubbed here to capture the
    keyword and keep this off the real training path."""
    _stub_load(monkeypatch)
    captured: dict[str, object] = {}

    def _fake_run(
        _params: dict[str, object], *, attention_impl: str | None = None,
    ) -> dict[str, object]:
        captured["attention_impl"] = attention_impl
        return {"loss_all": [1.0, 1.0]}

    monkeypatch.setattr(worker, "run_train_step", _fake_run)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "train_step", "params": dict(_BASE_PARAMS), "session_id": "s1",
        "attention_impl": "flash", "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["identity"]["attention_impl"] == "flash"
    assert captured["attention_impl"] == "flash"


def test_worker_main_train_step_without_attention_impl_omits_it_from_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility: a config with NO top-level `attention_impl` (every
    0.1.0-era train_step config) leaves the identity's keys unchanged -- the field is
    omitted -- and the run defaults to stock (`run_train_step` gets `None`)."""
    _stub_load(monkeypatch)
    captured: dict[str, object] = {}

    def _fake_run(
        _params: dict[str, object], *, attention_impl: str | None = None,
    ) -> dict[str, object]:
        captured["attention_impl"] = attention_impl
        return {"loss_all": [1.0, 1.0]}

    monkeypatch.setattr(worker, "run_train_step", _fake_run)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "train_step", "params": dict(_BASE_PARAMS), "session_id": "s1",
        "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "attention_impl" not in data["identity"]
    assert captured["attention_impl"] is None


def test_worker_main_train_step_wired_cap_regression_is_not_caught_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `WiredCapRegressionError` is NOT a `LaunchBudgetError` -- `worker.main` must
    not swallow it as a "refused" result. It propagates uncaught (same as an
    unsupported kind), so `runner.run_conditions` records the sweep-level crash
    envelope on the CALLER's side, never a silent "ok"."""
    _stub_load(monkeypatch)

    def _raise(
        _params: dict[str, object], *, attention_impl: str | None = None,  # noqa: ARG001
    ) -> dict[str, object]:
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
