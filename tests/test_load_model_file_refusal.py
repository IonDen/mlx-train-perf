"""Loaders refuse a model whose config names a Python file to execute.

mlx-lm 0.31.x `load` imports and runs the file a repo's `config.json` names under
`"model_file"`, with no opt-in. The package's loaders read the config first and refuse
such a model with a typed error before `mlx_lm.load` runs. These tests use fake local
model directories; nothing is downloaded and no model is loaded.
"""
import ast
import json
from pathlib import Path
from typing import Any

import mlx_lm
import mlx_lm.utils
import pytest

from mlx_train_perf.adapters.model_files import load_mlx_lm_model
from mlx_train_perf.bench import worker
from mlx_train_perf.errors import MlxTrainPerfError, ModelCodeRefusedError

_LLAMA = {"model_type": "llama", "vocab_size": 1000, "hidden_size": 64,
          "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4,
          "num_key_value_heads": 2, "rms_norm_eps": 1e-5}


def _model_dir(root: Path, config: dict[str, Any]) -> Path:
    """A model directory with `config` and a `custom_model.py` that, if ever executed,
    leaves a marker file behind."""
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    marker = root / "executed"
    (model / "custom_model.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    return model


@pytest.fixture
def load_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("mlx_lm.load was reached")
    monkeypatch.setattr(mlx_lm, "load", _fail)


@pytest.mark.usefixtures("load_must_not_run")
def test_worker_refuses_a_config_with_model_file_before_any_load(tmp_path: Path) -> None:
    """Bug caught: `_load_model` calling `mlx_lm.load` without reading the config first,
    so the repo's `custom_model.py` runs inside the bench worker."""
    model = _model_dir(tmp_path, {**_LLAMA, "model_file": "custom_model.py"})
    with pytest.raises(ModelCodeRefusedError) as info:
        worker._load_model(str(model), None)
    assert "model_file" in str(info.value)
    assert not (tmp_path / "executed").exists()


@pytest.mark.usefixtures("load_must_not_run")
def test_loader_refuses_a_config_with_model_file(tmp_path: Path) -> None:
    """Bug caught: the shared loader (used by the example and the smoke loaders) checking
    a different key, or checking after the load."""
    model = _model_dir(tmp_path, {**_LLAMA, "model_file": "custom_model.py"})
    with pytest.raises(ModelCodeRefusedError) as info:
        load_mlx_lm_model(str(model))
    message = str(info.value)
    assert "model_file" in message
    assert "custom_model.py" in message
    assert not (tmp_path / "executed").exists()


def test_refusal_is_a_package_error() -> None:
    """Bug caught: the refusal raised as a bare exception the CLI's typed-error exit path
    (`except MlxTrainPerfError`) does not catch, so it escapes as a traceback."""
    assert issubclass(ModelCodeRefusedError, MlxTrainPerfError)


@pytest.mark.usefixtures("load_must_not_run")
def test_hub_id_is_resolved_and_checked_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug caught: the check reading `config.json` only for local paths, so a Hub repo id
    (the usual case) goes straight to `mlx_lm.load`."""
    model = _model_dir(tmp_path, {**_LLAMA, "model_file": "custom_model.py"})
    resolved: list[tuple[str, str | None]] = []

    def _fake_download(repo: str, revision: str | None = None, **_kw: object) -> Path:
        resolved.append((repo, revision))
        return model  # stands in for the Hub snapshot; no network

    monkeypatch.setattr(mlx_lm.utils, "_download", _fake_download)
    with pytest.raises(ModelCodeRefusedError):
        worker._load_model("someone/some-model", "abc123")
    assert resolved == [("someone/some-model", "abc123")]
    assert not (tmp_path / "executed").exists()


def test_a_normal_config_loads_from_the_checked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug caught: the check refusing every model (e.g. testing the key's absence the
    wrong way round), or `mlx_lm.load` getting the original id and resolving it a second
    time, so the files it loads are not the ones that were checked."""
    model = _model_dir(tmp_path, _LLAMA)
    loaded_from: list[str] = []

    def _fake_download(_repo: str, **_kw: object) -> Path:
        return model  # stands in for the Hub snapshot; no network

    def _fake_load(path: str, **_kw: object) -> tuple[str, str]:
        loaded_from.append(path)
        return "the model", "the tokenizer"

    monkeypatch.setattr(mlx_lm.utils, "_download", _fake_download)
    monkeypatch.setattr(mlx_lm, "load", _fake_load)
    assert worker._load_model("someone/some-model", None) == ("the model", "the tokenizer")
    assert [Path(p).resolve() for p in loaded_from] == [model.resolve()]


# --- the example and the smoke loaders go through the checked loader ------------------

_ROOT = Path(__file__).resolve().parents[1]
_CHECKED_LOADER = _ROOT / "src" / "mlx_train_perf" / "adapters" / "model_files.py"
_ROUTED_SCRIPTS = ("examples/finetune_long_context.py", "tests/test_packed_smoke.py",
                   "tests/_qwen35_smoke_child.py")


def _direct_mlx_lm_loads(source: str) -> list[int]:
    """Lines that call `mlx_lm.load` directly or import `load` from `mlx_lm`."""
    def _is_direct_load(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return (node.attr == "load" and isinstance(node.value, ast.Name)
                    and node.value.id == "mlx_lm")
        if isinstance(node, ast.ImportFrom):
            return node.module == "mlx_lm" and any(a.name == "load" for a in node.names)
        return False

    return sorted(getattr(node, "lineno", 0) for node in ast.walk(ast.parse(source))
                  if _is_direct_load(node))


def test_direct_load_scan_finds_both_spellings() -> None:
    """Bug caught: the scan below matching neither spelling, so it passes on anything."""
    assert _direct_mlx_lm_loads("import mlx_lm\nm = mlx_lm.load('x')\n") == [2]
    assert _direct_mlx_lm_loads("from mlx_lm import load\n") == [1]
    assert _direct_mlx_lm_loads("from mlx_train_perf.adapters.model_files import "
                                "load_mlx_lm_model\n") == []


@pytest.mark.parametrize("relpath", _ROUTED_SCRIPTS)
def test_script_loads_through_the_checked_loader(relpath: str) -> None:
    """Bug caught: the example or a smoke loader calling `mlx_lm.load` directly, which
    runs a repo's `model_file` with no check (these paths load real models, so only
    their source can be checked in the default lane)."""
    source = (_ROOT / relpath).read_text(encoding="utf-8")
    assert _direct_mlx_lm_loads(source) == []
    assert "load_mlx_lm_model" in source


def test_package_calls_mlx_lm_load_only_in_the_checked_loader() -> None:
    """Bug caught: a new package call site of `mlx_lm.load` that skips the check."""
    offenders = [
        f"{path.relative_to(_ROOT).as_posix()}:{line}"
        for path in sorted((_ROOT / "src").rglob("*.py"))
        if path != _CHECKED_LOADER
        for line in _direct_mlx_lm_loads(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
