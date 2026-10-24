"""Gate: no code here passes a remote-code opt-in or calls mlx-lm's sharded loaders.

What the gate checks: no file under src/, scripts/, examples/ or tests/ passes mlx-lm
0.32's `trust_remote_code` opt-in (as a keyword, or as the string key of a dict or
subscript), and none names `sharded_load` or `pipeline_load`. Both loaders turn remote
code on for the tokenizer by default, and `pipeline_load` calls `sharded_load`. Every
model id here comes from params or the command line, so any of these would run code from
whatever repo a user names. The scan is an AST walk, so a comment or docstring that
mentions the names does not trip it.

What it does not check: the pinned mlx-lm 0.31.3 executes a repo's `model_file` from
plain `load` with no opt-in at all, and an absent flag leaves nothing in the source to
scan. That case is covered by the package's loaders refusing such configs, with its own
tests; a green run here says nothing about it.
"""

import ast
from pathlib import Path

import pytest

_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parents[1]
SCANNED_DIRS = ("src", "scripts", "examples", "tests")


_FLAG = "trust_remote_code"
_LOADERS = ("sharded_load", "pipeline_load")


def _remote_code_sites(source: str) -> list[str]:
    """Every place in `source` that opts in to remote code, as `line N: what`.

    Flagged: a `trust_remote_code=` keyword on any call; the exact string
    "trust_remote_code" (a `tokenizer_config` dict key, a kwargs subscript); any name,
    attribute or import of `sharded_load` or `pipeline_load`, and the exact string of
    either (getattr). A comment is not in the AST, and a docstring is a longer string, so
    neither matches.
    """
    sites: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == _FLAG:
            sites.append((node.lineno, f"keyword {_FLAG}"))
        elif isinstance(node, ast.Constant) and node.value in (_FLAG, *_LOADERS):
            sites.append((node.lineno, f"string {node.value!r}"))
        elif isinstance(node, ast.Name) and node.id in _LOADERS:
            sites.append((node.lineno, f"name {node.id}"))
        elif isinstance(node, ast.Attribute) and node.attr in _LOADERS:
            sites.append((node.lineno, f"attribute {node.attr}"))
        elif isinstance(node, ast.alias) and (name := node.name.rsplit(".", 1)[-1]) in _LOADERS:
            sites.append((node.lineno, f"import {name}"))
    return [f"line {line}: {what}" for line, what in sorted(sites)]


def _offenders(root: Path, skip: Path | None = None) -> list[str]:
    """`path: line N: what` for every remote-code opt-in under the scanned directories.

    `skip` is one file, compared by resolved path, so a same-named file elsewhere is
    still scanned.
    """
    return sorted(
        f"{path.relative_to(root).as_posix()}: {site}"
        for sub in SCANNED_DIRS
        for path in (root / sub).rglob("*.py")
        if skip is None or path.resolve() != skip
        for site in _remote_code_sites(path.read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("mlx_lm.load(model_id, trust_remote_code=True)\n", id="load-kwarg"),
        pytest.param("load_model(path, trust_remote_code=False)\n", id="kwarg-any-value"),
        pytest.param(
            "mlx_lm.load(model_id, tokenizer_config={'trust_remote_code': True})\n",
            id="tokenizer-config-dict-key",
        ),
        pytest.param("kwargs['trust_remote_code'] = True\n", id="subscript-key"),
        pytest.param("cfg = dict(trust_remote_code=True)\n", id="dict-call-kwarg"),
        pytest.param("mlx_lm.utils.sharded_load(repo)\n", id="sharded-load-attribute"),
        pytest.param("from mlx_lm.utils import sharded_load\n", id="sharded-load-import"),
        pytest.param(
            "from mlx_lm.utils import sharded_load as sl\nsl(repo)\n",
            id="sharded-load-aliased-import",
        ),
        pytest.param("fn = sharded_load\n", id="sharded-load-name"),
        pytest.param("fn = getattr(mlx_lm.utils, 'sharded_load')\n", id="sharded-load-getattr"),
        pytest.param("mlx_lm.utils.pipeline_load(repo)\n", id="pipeline-load-attribute"),
        pytest.param("from mlx_lm.utils import pipeline_load\n", id="pipeline-load-import"),
        pytest.param(
            "from mlx_lm.utils import pipeline_load as pl\npl(repo)\n",
            id="pipeline-load-aliased-import",
        ),
        pytest.param("fn = pipeline_load\n", id="pipeline-load-name"),
        pytest.param(
            "fn = getattr(mlx_lm.utils, 'pipeline_load')\n", id="pipeline-load-getattr"
        ),
    ],
)
def test_scanner_flags_each_remote_code_opt_in(source: str) -> None:
    # Catches: a scanner blind to one spelling of the opt-in (e.g. it checks call
    # keywords but not the `tokenizer_config={"trust_remote_code": True}` dict key that
    # plain `load` forwards to the tokenizer, or it bans `sharded_load` but not
    # `pipeline_load`, which calls it), so the repo gate below passes vacuously.
    assert len(_remote_code_sites(source)) == 1, source


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("mlx_lm.load(model_id, revision=revision)\n", id="plain-load"),
        pytest.param("# never pass trust_remote_code or call sharded_load\n", id="comment"),
        pytest.param('"""We do not use trust_remote_code or sharded_load."""\n', id="docstring"),
        pytest.param("trust_remote_code_note = 1\nload(x, sharded=True)\n", id="near-names"),
        pytest.param("pipeline_loader = 1\nx.pipeline_load_count\n", id="near-pipeline-names"),
    ],
)
def test_scanner_ignores_mentions_that_do_not_opt_in(source: str) -> None:
    # Catches: a scanner that degrades to a substring grep, which would fail the gate on
    # this module's own explanatory comments and on unrelated identifiers.
    assert _remote_code_sites(source) == []


def test_scanner_reports_the_offending_line() -> None:
    # Catches: a finding that drops the line number, leaving the reader to hunt for it.
    source = "import mlx_lm\n\nmodel = mlx_lm.load(repo, trust_remote_code=True)\n"
    assert _remote_code_sites(source) == ["line 3: keyword trust_remote_code"]


_BAD = "mlx_lm.load(m, trust_remote_code=True)\n"


def _write_tree(root: Path, rels: tuple[str, ...]) -> None:
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_BAD, encoding="utf-8")


def test_offenders_walks_each_scanned_dir(tmp_path: Path) -> None:
    # Catches: a directory dropped from the scanned set (or a non-recursive glob), which
    # lets an opt-in in a nested script, example or smoke test go unseen.
    _write_tree(tmp_path, ("src/pkg/deep/a.py", "scripts/b.py", "examples/c.py", "tests/d.py"))
    assert _offenders(tmp_path) == [
        "examples/c.py: line 1: keyword trust_remote_code",
        "scripts/b.py: line 1: keyword trust_remote_code",
        "src/pkg/deep/a.py: line 1: keyword trust_remote_code",
        "tests/d.py: line 1: keyword trust_remote_code",
    ]


def test_offenders_flags_a_pipeline_load_call_in_a_scanned_file(tmp_path: Path) -> None:
    # Catches: `pipeline_load` banned in the per-source scanner but lost on the way to the
    # repo gate (e.g. a separate name list for the tree walk), so a script that calls it
    # passes. pipeline_load calls sharded_load, which trusts the tokenizer's remote code.
    script = tmp_path / "scripts" / "distributed_bench.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from mlx_lm.utils import pipeline_load\n\nmodel, tok = pipeline_load(repo)\n",
        encoding="utf-8",
    )
    assert _offenders(tmp_path) == [
        "scripts/distributed_bench.py: line 1: import pipeline_load",
        "scripts/distributed_bench.py: line 3: name pipeline_load",
    ]


def test_skip_excludes_only_the_one_resolved_path(tmp_path: Path) -> None:
    # Catches: an exclusion by file name, glob or directory (e.g. `p.name == ...` or
    # skipping all of tests/) that would hide another test file's opt-in. The nested file
    # shares this module's name and must still be flagged; only the exact path is skipped.
    _write_tree(
        tmp_path,
        (
            "tests/test_no_remote_code.py",
            "tests/sub/test_no_remote_code.py",
            "tests/test_packed_smoke.py",
        ),
    )
    skipped = (tmp_path / "tests" / "sub" / ".." / "test_no_remote_code.py").resolve()
    assert _offenders(tmp_path, skip=skipped) == [
        "tests/sub/test_no_remote_code.py: line 1: keyword trust_remote_code",
        "tests/test_packed_smoke.py: line 1: keyword trust_remote_code",
    ]


def test_the_known_load_sites_are_inside_the_scanned_set() -> None:
    # Catches: SCANNED_DIRS or ROOT pointing at the wrong place, so the gate below scans
    # nothing and passes. worker.py holds the package's one `mlx_lm.load` call; the two
    # test files hold real-model loads run by the --run-smoke lane on this machine.
    scanned = {p for d in SCANNED_DIRS for p in (ROOT / d).rglob("*.py")}
    for rel in (
        "src/mlx_train_perf/bench/worker.py",
        "tests/test_packed_smoke.py",
        "tests/_qwen35_smoke_child.py",
    ):
        assert ROOT / rel in scanned, rel


def test_no_code_opts_in_to_remote_code() -> None:
    # Catches: `trust_remote_code=True` (or a `sharded_load`/`pipeline_load` call) added
    # to any load site in src/, scripts/, examples/ or tests/, which would execute Python
    # shipped by the model repo. This module is skipped: it spells the banned names out.
    assert _offenders(ROOT, skip=_THIS_FILE) == []
