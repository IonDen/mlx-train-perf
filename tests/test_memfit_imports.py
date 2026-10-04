# tests/test_memfit_imports.py
"""Import-boundary contracts for `mlx_train_perf.memfit` and the lazy package root."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

import mlx_train_perf
from mlx_train_perf import machine
from mlx_train_perf.core import loss
from mlx_train_perf.errors import DoesNotFitError, PlanInputError
from mlx_train_perf.memfit import errors as mf

_BLOCKER = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in ("mlx", "numpy"):
            raise ImportError(f"blocked for this test: {name}")
        return None
sys.meta_path.insert(0, _Block())
"""


def run_with_blocked_imports(code: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-c", _BLOCKER + code],
                          capture_output=True, text=True, timeout=120, check=False)


def test_memfit_imports_without_mlx_or_numpy() -> None:
    """Bug caught: the package root (or memfit itself) importing mlx at import time."""
    r = run_with_blocked_imports("import mlx_train_perf.memfit\nprint('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_machine_imports_without_mlx_or_numpy() -> None:
    """Bug caught: machine.py importing mlx (or a module that does) at import time."""
    r = run_with_blocked_imports("import mlx_train_perf.machine\nprint('ok')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_memfit_sources_import_nothing_outside_memfit_but_stdlib_and_numpy() -> None:
    """Bug caught: a memfit module importing mlx_train_perf.errors / plan / core, which
    would break the later move into its own distribution."""
    root = Path(__file__).resolve().parents[1] / "src" / "mlx_train_perf" / "memfit"
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level >= 2:
                offenders.append(f"{path.name}: relative import reaching above memfit "
                                 f"(level {node.level})")
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top == "mlx_train_perf" and not name.startswith("mlx_train_perf.memfit"):
                    offenders.append(f"{path.name}: {name}")
                if top == "mlx":
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


def test_root_still_exports_the_loss_api() -> None:
    assert mlx_train_perf.linear_cross_entropy is loss.linear_cross_entropy
    assert sorted(mlx_train_perf.__all__) == [
        "DenseHead", "HeadRef", "QuantizedHead", "Resolution", "linear_cross_entropy",
        "resolve_impl", "tied_head",
    ]


def test_root_unknown_attribute_raises_attribute_error() -> None:
    """Bug caught: a __getattr__ that raises KeyError breaks `from mlx_train_perf import
    cli`, which relies on AttributeError to fall through to the submodule import."""
    with pytest.raises(AttributeError, match="no_such_name"):
        mlx_train_perf.no_such_name  # noqa: B018


def test_submodule_from_import_still_works() -> None:
    assert machine.__name__ == "mlx_train_perf.machine"


def test_planner_input_errors_are_memfit_input_errors() -> None:
    with pytest.raises(mf.MemfitInputError):
        raise PlanInputError("x")
    with pytest.raises(mf.DoesNotFitError):
        raise DoesNotFitError("x")


def test_public_error_hierarchy() -> None:
    """Bug caught: callers catching MemfitInputError / MemfitError stop seeing the
    planner's or calibration's refusals if a base class changes."""
    assert issubclass(mf.CalibrationMismatchError, mf.MemfitInputError)
    assert issubclass(mf.MemfitInputError, mf.MemfitError)
    assert issubclass(mf.DoesNotFitError, mf.MemfitError)
    assert issubclass(mf.MemfitDependencyError, mf.MemfitError)
    assert issubclass(PlanInputError, mf.MemfitInputError)
    assert issubclass(DoesNotFitError, mf.DoesNotFitError)
    assert not issubclass(mf.DoesNotFitError, mf.MemfitInputError)
