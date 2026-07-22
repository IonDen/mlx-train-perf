"""Tests for `scripts/fit_calibration.py`. `scripts/` has no `__init__.py` (matches
the existing convention), so the module is loaded by path.

Every test here is fully synthetic: fabricated `config.json`/`run_train_step`-shaped
artifact files written to `tmp_path`, and a `--calibration-data` pointed at a temp
copy -- this test module NEVER touches the real, committed
`src/mlx_train_perf/plan/calibration_data.json` (which now carries measured constants;
the controller ran this script for real, on the production artifacts). The underlying
fitting MATH (`fit_memory_coeffs`) has its own
dedicated, RED-first-TDD'd tests in `tests/test_plan.py`; this file covers only the
I/O/manifest-reading glue around it.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_train_perf.plan.calibration import Calibration, load_calibration
from mlx_train_perf.plan.estimate import (
    ModelShape,
    TrainConfig,
    _flash_saved_state_bytes,
    _lora_bytes,
    _loss_bytes,
    _optimizer_bytes,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import fit_calibration  # noqa: E402 -- import must follow the sys.path insert
from fit_calibration import (  # noqa: E402
    build_updated_calibration_data,
    load_fit_points,
)

_SCRIPT_PATH = _SCRIPTS_DIR / "fit_calibration.py"

_CONFIG = {
    "vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
    "tie_word_embeddings": False,
}

_EXISTING_CALIBRATION = {
    "base_transient_bytes": 1.0,
    "act_bytes_per_token_hidden_layer_ckpt": 1.0,
    "act_bytes_per_token_hidden_layer_full": 50.0,
    "attn_bytes_per_head_token2": 1.0,
    "attn_bytes_per_head_token_flash": 1.0,
    "optimizer_bytes_per_param": 8.0,
    "overhead_frac": 0.10,
    "naive_loss_bytes_per_nv": 12.0,
    "provenance": {
        "machine": "arm64-placeholder", "macos": "0.0.0", "mlx_version": "0.0.0",
        "measured_date": "placeholder, pending measured calibration",
    },
}


def _write_artifact(
    path: Path, *, status: str = "ok", marginal_peak_gb: float = 1.0, impl: str = "kernel",
    attention_impl: str = "stock",
) -> None:
    path.write_text(json.dumps({
        "status": status, "marginal_peak_gb": marginal_peak_gb,
        "identity": {"impl": impl, "attention_impl": attention_impl},
    }))


def _write_manifest(
    path: Path, entries: list[dict[str, object]],
) -> None:
    path.write_text(json.dumps(entries))


def _write_gc_true_manifest(tmp_path: Path, config_path: Path) -> Path:
    """3 grad_checkpoint=True kernel points spanning 3 distinct seq_len (512/1024/2048) --
    the minimum full-rank design `fit_memory_coeffs` accepts in the batch-fixed regime
    (>= 3 distinct seq_len to separate base/linear/quadratic). Arbitrary distinct
    marginals; the fit just needs a non-singular design."""
    manifest: list[dict[str, object]] = []
    for name, seq, marg in [("a", 512, 2.0), ("b", 1024, 3.5), ("c", 2048, 8.0)]:
        artifact = tmp_path / f"{name}.json"
        _write_artifact(artifact, marginal_peak_gb=marg)
        manifest.append({
            "config": str(config_path), "artifact": str(artifact), "batch": 1,
            "seq_len": seq, "lora_rank": 8, "lora_layers": 2, "grad_checkpoint": True,
        })
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path


# ---------------------------------------------------------------------------
# load_fit_points: manifest + config + artifact -> FitPoint list
# ---------------------------------------------------------------------------


def test_load_fit_points_builds_one_fitpoint_per_manifest_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    artifact_a = tmp_path / "a.json"
    artifact_b = tmp_path / "b.json"
    _write_artifact(artifact_a, marginal_peak_gb=2.0)
    _write_artifact(artifact_b, marginal_peak_gb=4.0)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        {"config": str(config_path), "artifact": str(artifact_a), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},
        {"config": str(config_path), "artifact": str(artifact_b), "batch": 2,
         "seq_len": 512, "lora_rank": 16, "lora_layers": 2},
    ])
    points = load_fit_points(manifest_path)
    assert len(points) == 2
    assert points[0].marginal_peak_bytes == 2.0 * 1024**3
    assert points[1].cfg.batch == 2
    assert points[1].cfg.lora_rank == 16
    assert points[0].shape.vocab == 1000
    assert points[0].cfg.impl == "kernel"


def test_load_fit_points_defaults_omitted_grad_checkpoint_to_true(tmp_path: Path) -> None:
    """review item: a manifest entry that OMITS `grad_checkpoint` defaults to True (the
    realistic calibration regime) -- a deliberate flip from the pre-rework False default.
    Untested, a revert would silently misroute future calibration points into the wrong
    ckpt/full fit bucket."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    artifact_path = tmp_path / "a.json"
    _write_artifact(artifact_path, marginal_peak_gb=2.0)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        {"config": str(config_path), "artifact": str(artifact_path), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},  # no grad_checkpoint key
    ])
    points = load_fit_points(manifest_path)
    assert points[0].cfg.grad_checkpoint is True


def test_load_fit_points_reads_impl_from_the_artifact_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    artifact_path = tmp_path / "a.json"
    _write_artifact(artifact_path, impl="chunked")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        {"config": str(config_path), "artifact": str(artifact_path), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},
    ])
    points = load_fit_points(manifest_path)
    assert points[0].cfg.impl == "chunked"


def test_load_fit_points_reads_attention_impl_from_the_artifact_identity(
    tmp_path: Path,
) -> None:
    """The train-step artifact identity carries `attention_impl` ("stock"/"flash"); it
    threads into the FitPoint's `cfg.attention` so `fit_memory_coeffs` routes the point
    into the correct branch. Absent (old artifacts) it defaults to "stock"."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    flash_art = tmp_path / "flash.json"
    stock_art = tmp_path / "stock.json"
    _write_artifact(flash_art, attention_impl="flash")
    _write_artifact(stock_art, attention_impl="stock")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        {"config": str(config_path), "artifact": str(flash_art), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},
        {"config": str(config_path), "artifact": str(stock_art), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},
    ])
    points = load_fit_points(manifest_path)
    assert points[0].cfg.attention == "flash"
    assert points[1].cfg.attention == "stock"


def test_load_fit_points_rejects_a_non_ok_artifact(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    artifact_path = tmp_path / "a.json"
    _write_artifact(artifact_path, status="refused")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [
        {"config": str(config_path), "artifact": str(artifact_path), "batch": 1,
         "seq_len": 512, "lora_rank": 8, "lora_layers": 2},
    ])
    with pytest.raises(ValueError, match="refused"):
        load_fit_points(manifest_path)


# ---------------------------------------------------------------------------
# build_updated_calibration_data: preserves overhead_frac/naive_loss_bytes_per_nv,
# replaces act/optimizer + provenance
# ---------------------------------------------------------------------------


_COEFFS = {
    "base_transient_bytes": 5.0, "act_bytes_per_token_hidden_layer_ckpt": 2.0,
    "act_bytes_per_token_hidden_layer_full": 60.0, "attn_bytes_per_head_token2": 3.0,
    "attn_bytes_per_head_token_flash": 7.0,
}


def test_build_updated_calibration_data_preserves_untouched_constants(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]")
    updated = build_updated_calibration_data(
        existing=_EXISTING_CALIBRATION, coeffs=_COEFFS,
        optimizer_bytes_per_param=8.0, manifest_path=manifest_path, num_points=3,
    )
    assert updated["base_transient_bytes"] == 5.0
    assert updated["act_bytes_per_token_hidden_layer_ckpt"] == 2.0
    assert updated["act_bytes_per_token_hidden_layer_full"] == 60.0
    assert updated["attn_bytes_per_head_token2"] == 3.0
    assert updated["attn_bytes_per_head_token_flash"] == 7.0
    assert updated["optimizer_bytes_per_param"] == 8.0
    # untouched by this fit:
    assert updated["overhead_frac"] == _EXISTING_CALIBRATION["overhead_frac"]
    assert updated["naive_loss_bytes_per_nv"] == _EXISTING_CALIBRATION["naive_loss_bytes_per_nv"]


def test_build_updated_calibration_data_provenance_has_the_required_keys(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]")
    updated = build_updated_calibration_data(
        existing=_EXISTING_CALIBRATION, coeffs=_COEFFS,
        optimizer_bytes_per_param=8.0, manifest_path=manifest_path, num_points=2,
    )
    provenance = updated["provenance"]
    for key in ("machine", "macos", "mlx_version", "measured_date"):
        assert provenance[key]
    assert "manifest.json" in provenance["fit_source"]
    assert "2" in provenance["fit_source"]


# ---------------------------------------------------------------------------
# CLI shell, end to end against SYNTHETIC files only -- never the real
# calibration_data.json
# ---------------------------------------------------------------------------


def test_main_dry_run_does_not_write_the_calibration_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    calibration_path = tmp_path / "calibration_data.json"
    calibration_path.write_text(json.dumps(_EXISTING_CALIBRATION))
    manifest_path = _write_gc_true_manifest(tmp_path, config_path)
    original_text = calibration_path.read_text()

    rc = fit_calibration.main([
        "--manifest", str(manifest_path), "--calibration-data", str(calibration_path),
        "--dry-run",
    ])
    assert rc == 0
    assert calibration_path.read_text() == original_text   # untouched


def test_main_writes_the_updated_calibration_file_without_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    calibration_path = tmp_path / "calibration_data.json"
    calibration_path.write_text(json.dumps(_EXISTING_CALIBRATION))
    manifest_path = _write_gc_true_manifest(tmp_path, config_path)

    rc = fit_calibration.main([
        "--manifest", str(manifest_path), "--calibration-data", str(calibration_path),
    ])
    assert rc == 0
    updated = json.loads(calibration_path.read_text())
    # the fit replaced the memory coefficients (base moved off its placeholder 1.0):
    assert updated["base_transient_bytes"] != _EXISTING_CALIBRATION["base_transient_bytes"]
    assert "attn_bytes_per_head_token2" in updated
    assert "attn_bytes_per_head_token_flash" in updated
    assert updated["overhead_frac"] == _EXISTING_CALIBRATION["overhead_frac"]
    # review item: main() wires optimizer_bytes_per_param from the EXISTING file (it is
    # analytic, not fitted -- a behavior change in the rework, previously fit-returned):
    assert (updated["optimizer_bytes_per_param"]
            == _EXISTING_CALIBRATION["optimizer_bytes_per_param"])
    assert updated["provenance"]["measured_date"]


# ---------------------------------------------------------------------------
# flash_fit selection (Task 8, 0.5.0): main() fits with "ols", checks one-sidedness
# (predicted cushioned TOTAL >= measured at every flash anchor) via the public
# estimate_peak, and refits with "envelope" on a violation.
# ---------------------------------------------------------------------------


def _stand_in_calibration() -> Calibration:
    """The SAME stand-in `Calibration` `main()` builds from `_EXISTING_CALIBRATION`
    when `--calibration-data` is a non-default (temp) path -- used here to synthesize
    flash points whose true coefficients `main()` will fit against exactly this calib."""
    return Calibration(
        base_transient_bytes=float(_EXISTING_CALIBRATION["base_transient_bytes"]),
        act_bytes_per_token_hidden_layer_ckpt=float(
            _EXISTING_CALIBRATION["act_bytes_per_token_hidden_layer_ckpt"]),
        act_bytes_per_token_hidden_layer_full=float(
            _EXISTING_CALIBRATION["act_bytes_per_token_hidden_layer_full"]),
        attn_bytes_per_head_token2=float(_EXISTING_CALIBRATION["attn_bytes_per_head_token2"]),
        attn_bytes_per_head_token_flash=float(
            _EXISTING_CALIBRATION["attn_bytes_per_head_token_flash"]),
        optimizer_bytes_per_param=float(_EXISTING_CALIBRATION["optimizer_bytes_per_param"]),
        overhead_frac=float(_EXISTING_CALIBRATION["overhead_frac"]),
        naive_loss_bytes_per_nv=float(_EXISTING_CALIBRATION["naive_loss_bytes_per_nv"]),
        provenance=dict(_EXISTING_CALIBRATION["provenance"]),
    )


def _write_flash_point(
    tmp_path: Path, name: str, *, config_path: Path, calib: Calibration, shape: ModelShape,
    seq_len: int, a_flash: float,
) -> dict[str, object]:
    """Synthesizes ONE flash manifest entry (config + artifact) whose measured
    `marginal_peak_gb` is generated FORWARD from a known `a_flash`, mirroring
    `test_plan.py::_synthesize_flash_fit_point`'s forward-construction pattern but
    writing a real artifact JSON file instead of a `FitPoint` object -- this is what
    lets `main()`'s check-then-select logic be exercised end to end through the real
    manifest-ingestion path (`load_fit_points`)."""
    cfg = TrainConfig(batch=1, seq_len=seq_len, dtype="bfloat16", lora_rank=8,
                      lora_layers=2, grad_checkpoint=True, impl="kernel", attention="flash")
    analytic = (_lora_bytes(cfg, shape) + _optimizer_bytes(cfg, shape, calib)
                + _loss_bytes(cfg, shape, calib))
    a_lin = calib.act_bytes_per_token_hidden_layer_ckpt
    x_lin = cfg.batch * cfg.seq_len * shape.hidden * shape.layers
    x_flash = cfg.batch * shape.heads * cfg.seq_len
    o_l = _flash_saved_state_bytes(cfg, shape)
    marginal_bytes = calib.base_transient_bytes + a_lin * x_lin + a_flash * x_flash + analytic + o_l
    artifact_path = tmp_path / f"{name}.json"
    _write_artifact(artifact_path, marginal_peak_gb=marginal_bytes / 1024**3,
                    attention_impl="flash")
    return {"config": str(config_path), "artifact": str(artifact_path), "batch": 1,
            "seq_len": seq_len, "lora_rank": 8, "lora_layers": 2, "grad_checkpoint": True}


def test_main_selects_envelope_flash_fit_when_ols_under_predicts_an_anchor(
    tmp_path: Path,
) -> None:
    """Two flash-only points (a flash-only manifest needs no stock points --
    `fit_memory_coeffs` keeps the stock coefficients from `existing`) with WIDELY
    different true `a_flash` -- a small-seq_len anchor with a HUGE true coefficient
    and a large-seq_len anchor with a TINY one. OLS's weighted average is dominated by
    the large-seq_len point's own huge `x_flash^2` weight, so it lands far below the
    small anchor's true coefficient: at that anchor, the OLS candidate's predicted
    cushioned TOTAL (`estimate_peak`) under-shoots the measured total by orders of
    magnitude, well past the planner's own 10% `overhead_frac` cushion -- main() must
    detect the violation and refit with `flash_fit="envelope"`."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    calib = _stand_in_calibration()
    shape = ModelShape.from_config(_CONFIG)
    small_seq_huge_a_flash = _write_flash_point(
        tmp_path, "small_seq_huge_a_flash", config_path=config_path, calib=calib,
        shape=shape, seq_len=512, a_flash=1_000_000.0,
    )
    big_seq_tiny_a_flash = _write_flash_point(
        tmp_path, "big_seq_tiny_a_flash", config_path=config_path, calib=calib,
        shape=shape, seq_len=8192, a_flash=1.0,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [small_seq_huge_a_flash, big_seq_tiny_a_flash])
    calibration_path = tmp_path / "calibration_data.json"
    calibration_path.write_text(json.dumps(_EXISTING_CALIBRATION))

    rc = fit_calibration.main([
        "--manifest", str(manifest_path), "--calibration-data", str(calibration_path),
    ])
    assert rc == 0
    updated = json.loads(calibration_path.read_text())
    assert updated["provenance"]["flash_fit"] == "envelope"
    # the envelope recovers the under-predicted anchor's own true coefficient, not the
    # OLS weighted average (which lands far below it -- see the docstring above).
    assert updated["attn_bytes_per_head_token_flash"] == pytest.approx(1_000_000.0, rel=1e-6)


def test_main_selects_ols_flash_fit_for_benign_points(tmp_path: Path) -> None:
    """Counterpart: flash points that all imply the SAME true coefficient never trip
    the one-sidedness guard (OLS recovers that coefficient exactly, and the 10%
    `overhead_frac` cushion comfortably covers any float rounding) -- `flash_fit` stays
    `"ols"`, the shipped default, not the envelope fallback."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    calib = _stand_in_calibration()
    shape = ModelShape.from_config(_CONFIG)
    a = _write_flash_point(tmp_path, "a", config_path=config_path, calib=calib, shape=shape,
                           seq_len=512, a_flash=500.0)
    b = _write_flash_point(tmp_path, "b", config_path=config_path, calib=calib, shape=shape,
                           seq_len=1024, a_flash=500.0)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [a, b])
    calibration_path = tmp_path / "calibration_data.json"
    calibration_path.write_text(json.dumps(_EXISTING_CALIBRATION))

    rc = fit_calibration.main([
        "--manifest", str(manifest_path), "--calibration-data", str(calibration_path),
    ])
    assert rc == 0
    updated = json.loads(calibration_path.read_text())
    assert updated["provenance"]["flash_fit"] == "ols"
    assert updated["attn_bytes_per_head_token_flash"] == pytest.approx(500.0, rel=1e-6)


def test_help_runs_without_touching_a_model() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--manifest" in proc.stdout
    assert "--dry-run" in proc.stdout


def test_real_calibration_data_json_is_never_touched_by_this_test_module() -> None:
    """Sanity guard: the REAL, committed calibration_data.json must still carry its
    MEASURED provenance after this whole test module runs -- every test above points
    `--calibration-data` at a temp copy, so a stray write to the real default would be
    caught here (a synthetic overwrite's provenance would not mention the real Qwen3-8B
    calibration)."""
    calib = load_calibration()
    assert "Qwen3-8B" in calib.provenance["fit_source"]
