import json
from pathlib import Path

import pytest

from mlx_train_perf import cli
from mlx_train_perf.memfit.errors import CalibrationMismatchError


def test_plan_command_maps_a_memfit_error_to_the_tool_error_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug caught: the CLI only catching MlxTrainPerfError, so a pure memfit error (here a
    calibration-quantity mismatch) escapes as a traceback (exit 1) instead of exit 2."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "tie_word_embeddings": False,
    }))

    def _raise(*_a: object, **_kw: object) -> object:
        raise CalibrationMismatchError("file holds 'x', caller expects 'y'")

    monkeypatch.setattr(cli, "plan_fit", _raise)
    rc = cli.main(["plan", "--config", str(config), "--batch", "1", "--seq-len", "512",
                   "--lora-rank", "8", "--budget-gb", "8", "--json"])
    assert rc == 2
    assert "caller expects 'y'" in capsys.readouterr().err
