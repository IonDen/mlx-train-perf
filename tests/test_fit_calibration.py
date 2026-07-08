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

from mlx_train_perf.plan.calibration import load_calibration

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
) -> None:
    path.write_text(json.dumps({
        "status": status, "marginal_peak_gb": marginal_peak_gb,
        "identity": {"impl": impl},
    }))


def _write_manifest(
    path: Path, entries: list[dict[str, object]],
) -> None:
    path.write_text(json.dumps(entries))


def _write_gc_true_manifest(tmp_path: Path, config_path: Path) -> Path:
    """3 grad_checkpoint=True kernel points spanning 3 distinct seq_len (512/1024/2048) --
    the minimum `fit_memory_coeffs` needs (>= 3 gc=True points, >= 2 distinct seq_len, to
    separate base/linear/quadratic). Arbitrary distinct marginals; the fit just needs a
    non-singular design."""
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
    assert updated["overhead_frac"] == _EXISTING_CALIBRATION["overhead_frac"]
    assert updated["provenance"]["measured_date"]


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
