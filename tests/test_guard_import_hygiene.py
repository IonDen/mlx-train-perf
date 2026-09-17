"""`mlx-guard` is an optional extra: nothing may import it unless supervision is in use."""
import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mlx_train_perf"


def _module_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_no_module_level_mlx_guard_import() -> None:
    # Catches: an eager import, which turns the optional extra into a hard dependency of
    # every bench run.
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if any(
            name == "mlx_guard" or name.startswith("mlx_guard.")
            for name in _module_level_imports(ast.parse(path.read_text()))
        )
    ]
    assert offenders == []


def test_importing_the_runner_and_worker_does_not_load_mlx_guard() -> None:
    # Catches what the AST scan cannot: an import reached transitively at import time
    # (a helper module, a conditional at module scope).
    code = (
        "import sys\n"
        "import mlx_train_perf.bench.runner\n"
        "import mlx_train_perf.bench.worker\n"
        "sys.exit(1 if 'mlx_guard' in sys.modules else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0
