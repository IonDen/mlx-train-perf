from pathlib import Path

import pytest

import mlx_train_perf.recurrent.ops as ops_mod
import mlx_train_perf.recurrent.reference as reference_mod
import mlx_train_perf.recurrent.wrapper as wrapper_mod


def _assert_no_module_level_mlx_lm_import(module: object) -> None:
    src = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    for line in src.splitlines():
        s = line.strip()
        assert not s.startswith(("import mlx_lm", "from mlx_lm")), line


def test_reference_module_has_no_module_level_mlx_lm_import():
    # Catches: a top-level `import mlx_lm` in reference.py, which breaks
    # collection for users who installed without the mlx-lm extra.
    _assert_no_module_level_mlx_lm_import(reference_mod)


@pytest.mark.parametrize("module", [ops_mod, wrapper_mod])
def test_module_has_no_module_level_mlx_lm_import(module):
    # Catches: a top-level `import mlx_lm` in ops.py or wrapper.py, which would break
    # collection for users who installed without the mlx-lm extra -- load-bearing since
    # `recurrent/__init__.py` now re-exports `enable_gated_delta_training`, so a plain
    # `import mlx_train_perf.recurrent` pulls wrapper.py in at import time.
    _assert_no_module_level_mlx_lm_import(module)
