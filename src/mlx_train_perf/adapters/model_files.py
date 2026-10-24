"""Loading an mlx-lm model without running code the model repo ships.

mlx-lm 0.31.x `load` imports and executes the Python file a model's `config.json` names
under `"model_file"`, and has no opt-in that turns this off. `load_mlx_lm_model` resolves
the model directory first (the same local-path-or-Hub-snapshot resolution `mlx_lm.load`
does), reads its config, refuses a config that sets `model_file`, and then loads from
that same directory, so the files loaded are the files that were checked.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from mlx_train_perf.errors import ModelCodeRefusedError

_MODEL_FILE_KEY = "model_file"


def refuse_model_file(config: Mapping[str, object], *, model: str) -> None:
    """Raises `ModelCodeRefusedError` when `config` (a model's parsed `config.json`)
    sets `model_file`, whatever its value."""
    if _MODEL_FILE_KEY in config:
        raise ModelCodeRefusedError(
            f"{model}: config.json sets {_MODEL_FILE_KEY} "
            f"({config[_MODEL_FILE_KEY]!r}); mlx-lm 0.31.x would execute that file at "
            "load and has no trust_remote_code opt-in, so mlx-train-perf refuses this model"
        )


def load_mlx_lm_model(model_id: str, *, revision: str | None = None) -> tuple[Any, Any]:
    """`mlx_lm.load(model_id, revision=revision)`, refusing a model whose config sets
    `model_file` before anything is loaded. Returns mlx-lm's `(model, tokenizer)`."""
    import mlx_lm  # noqa: PLC0415 -- optional dependency, imported on use
    import mlx_lm.utils  # noqa: PLC0415

    # `_download` is mlx-lm 0.31.x's own path resolution for `load` (a local directory
    # as is, otherwise a Hub snapshot); the package pins mlx-lm below 0.32.
    model_path = Path(mlx_lm.utils._download(model_id, revision=revision))
    refuse_model_file(mlx_lm.utils.load_config(model_path), model=model_id)
    return cast("tuple[Any, Any]", mlx_lm.load(str(model_path)))
