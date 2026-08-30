from pathlib import Path

import mlx_train_perf.recurrent.reference as mod


def test_reference_module_has_no_module_level_mlx_lm_import():
    # Catches: a top-level `import mlx_lm` in reference.py, which breaks
    # collection for users who installed without the mlx-lm extra.
    src = Path(mod.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        assert not s.startswith(("import mlx_lm", "from mlx_lm")), line
