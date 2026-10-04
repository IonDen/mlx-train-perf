import json
from importlib import resources
from pathlib import Path

import pytest

from mlx_train_perf.errors import PlanInputError
from mlx_train_perf.memfit import MLX_ACTIVE_MARGINAL, CalibrationFile
from mlx_train_perf.plan.calibration import calibration_from_file, load_calibration


def test_packaged_calibration_is_in_the_memfit_format() -> None:
    raw = json.loads(
        resources.files("mlx_train_perf.plan").joinpath("calibration_data.json").read_text()
    )
    assert raw["format"] == "mlx-train-perf.calibration"
    assert raw["measured_quantity"] == "mlx_active_marginal"


def test_lora_calibration_numbers_are_the_shipped_0_8_0_values() -> None:
    """Literal values from the 0.8.0 calibration_data.json; the format move must not
    change a single number."""
    c = load_calibration()
    assert c.base_transient_bytes == 1600848316.0
    assert c.act_bytes_per_token_hidden_layer_ckpt == 1.867
    assert c.act_bytes_per_token_hidden_layer_full == 85.12
    assert c.attn_bytes_per_head_token2 == 8.716
    assert c.attn_bytes_per_head_token_flash_kernel == 21145.736137516276
    assert c.attn_bytes_per_head_token_flash_stock == 34125.414004182945
    assert c.optimizer_bytes_per_param == 8.0
    assert c.overhead_frac == 0.1
    assert c.naive_loss_bytes_per_nv == 12.0
    assert c.provenance["flash_fit"] == "envelope"


def test_a_missing_coefficient_is_refused_by_name() -> None:
    cal = CalibrationFile(
        measured_quantity=MLX_ACTIVE_MARGINAL,
        coefficients={"base_transient_bytes": 1.0},
        provenance={},
    )
    with pytest.raises(PlanInputError, match="act_bytes_per_token_hidden_layer_ckpt"):
        calibration_from_file(cal)


def test_refit_drops_the_old_holdout_record(tmp_path: Path) -> None:
    """Bug caught: `scripts/fit_calibration.py` copied the old file's hold-out score next
    to freshly fitted coefficients, so the file claimed a validation it never had."""
    # test_fit_calibration puts scripts/ on sys.path as it imports, so it comes first.
    from test_fit_calibration import (  # noqa: PLC0415
        _CONFIG,
        _write_existing_calibration,
        _write_gc_true_manifest,
        fit_calibration,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_CONFIG))
    calibration_path = tmp_path / "calibration_data.json"
    _write_existing_calibration(calibration_path)
    doc = json.loads(calibration_path.read_text())
    doc["holdout"] = {"predictions_sha256": "abc", "max_under_rel": 0.01, "scored": 3}
    calibration_path.write_text(json.dumps(doc))
    manifest_path = _write_gc_true_manifest(tmp_path, config_path)

    rc = fit_calibration.main([
        "--manifest", str(manifest_path), "--calibration-data", str(calibration_path),
    ])

    assert rc == 0
    assert json.loads(calibration_path.read_text())["holdout"] is None
