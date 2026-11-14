"""The July 2026 run records under `evidence/2026-07-run-records/`, and the run records
already tracked under `_artifacts/` that the project README and the papers cite, bound to
their script and library revisions.

`MANIFEST.json` holds two lists: `records`, the files in the evidence folder, and
`tracked_elsewhere`, the cited run records tracked under `_artifacts/`. Each row gives a
record's sha256 and, for each of the record's own fingerprints (`identity.script_sha` over
the producing script, `identity.code_sha` over the library files), a commit on main that
reproduces the fingerprint; the manifest generator chose the earliest. These tests read
the records and the git history only: no mlx import, no Metal.

They need a full git checkout. A missing evidence folder, a missing `git`, a shallow
clone or an absent commit FAILS here, it never skips; that is why the sdist leaves this
file out. A missing folder fails this file only: collection of the rest of the suite
goes on.
"""

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PREFIX = "evidence/2026-07-run-records/"
# Every record is read through this folder (rebinding it points the tests elsewhere);
# git always runs against REPO_ROOT.
EVIDENCE_DIR = REPO_ROOT / EVIDENCE_PREFIX
# The only files in the folder that are not records.
NOT_RECORDS = frozenset({"MANIFEST.json", "README.md"})
# The manifest's two lists: the records in the evidence folder, and the cited run records
# tracked elsewhere in the repository (read from REPO_ROOT, whatever EVIDENCE_DIR is).
RECORDS_KEY = "records"
TRACKED_KEY = "tracked_elsewhere"
TRACKED_PREFIX = "_artifacts/"

ROW_FIELDS = (
    "path",
    "sha256",
    "session_id",
    "mlx_version",
    "script",
    "script_sha",
    "commit",
    "fingerprint",
    "code_sha",
    "code_commit",
    "code_deps",
)
COMMIT_RE = r"[0-9a-f]{40}"
SCRIPT_RE = r"scripts/[A-Za-z0-9_]+\.py"
CODE_DEP_RE = r"src/mlx_train_perf/[A-Za-z0-9_/]+\.py"

NOT_CHECKABLE = "not checkable: the record carries no script fingerprint"
PROBE_PATH = EVIDENCE_PREFIX + "launch_budget_evidence/probe_results.json"
SWEEP_23040_PATH = (
    EVIDENCE_PREFIX
    + "northstar_context_sweep/northstar_mlx-community__Qwen3-8B-4bit_seq23040_ours.json"
)
UNREADABLE_ID = "manifest-unreadable"

# The sets chosen for publication: directory under the evidence folder -> file count.
PUBLISHED_SET_SIZES = {
    "northstar_context_sweep": 20,
    "bench_train_step_stockattn": 4,
    "bench_train_step": 4,
    "bench_train_step_gradckpt": 4,
    "pre-rc-refresh-backup-2026-07-12/northstar_context_sweep": 24,
    "launch_budget_evidence": 1,
}


class RunSet(NamedTuple):
    sessions: dict[str, int]  # session_id prefix (8 hex) -> number of records
    mlx_version: str
    script_sha: str


# What each set's records carry in their own `identity` (the probe set has none). A set
# may hold several runs, each with its own mlx version and script revision.
RUN_SETS: dict[str, tuple[RunSet, ...]] = {
    "northstar_context_sweep": (RunSet({"6cdf97ec": 20}, "0.32.0", "304787652e37ab2e"),),
    "bench_train_step_stockattn": (
        RunSet({"97d6b7fb": 3, "2f9f3185": 1}, "0.32.0", "191fc55b97263907"),
    ),
    "bench_train_step": (RunSet({"77982f98": 4}, "0.32.0", "191fc55b97263907"),),
    "bench_train_step_gradckpt": (RunSet({"00b615e3": 4}, "0.31.2", "afb0ac0e9a72c80e"),),
    "pre-rc-refresh-backup-2026-07-12/northstar_context_sweep": (
        RunSet({"435c2eff": 4}, "0.31.2", "2521d5810a1ceb7b"),
        # Read from the 20 records (frozen set); git 01b2f7b reproduces the script_sha.
        RunSet({"68b48e5f": 20}, "0.32.0", "cdb2a14be0c07fca"),
    ),
}

# The folder above holds two sessions; the files of each, by name.
PRE_RC_SWEEP_DIR = "pre-rc-refresh-backup-2026-07-12/northstar_context_sweep"


def _sweep_file(seq_len: int, arm: str) -> str:
    return f"northstar_mlx-community__Qwen3-8B-4bit_seq{seq_len}_{arm}.json"


PRE_RC_SESSION_FILES = {
    # The four rows the fc45488 MANIFEST held for this folder.
    "435c2eff": frozenset(
        {
            _sweep_file(8448, "ours"),
            _sweep_file(8704, "ours"),
            _sweep_file(8704, "stock"),
            _sweep_file(8960, "stock"),
        }
    ),
    # The 2026-07-11 sweep's source folder listing: ten lengths, both arms.
    "68b48e5f": frozenset(
        _sweep_file(n, arm)
        for n in (1024, 2048, 4096, 8192, 9216, 9728, 9984, 10240, 12288, 16384)
        for arm in ("ours", "stock")
    ),
}

# The 2026-07-11 sweep's outcome per (length, arm), as the evidence README's 2026-07-11
# section states it: flash (ours) completed up to 9,984 and was refused by the launch
# budget above; stock completed up to 9,728 and hit the 28.0 GiB ceiling from 9,984.
SWEEP_0711_OUTCOMES = {
    **{(n, "ours"): "ok" for n in (1024, 2048, 4096, 8192, 9216, 9728, 9984)},
    **{(n, "ours"): "refused" for n in (10240, 12288, 16384)},
    **{(n, "stock"): "ok" for n in (1024, 2048, 4096, 8192, 9216, 9728)},
    **{(n, "stock"): "aborted_memory_ceiling" for n in (9984, 10240, 12288, 16384)},
}
SWEEP_0711_CEILING_GB = 28.0  # the fixed ceiling the same README section names

# The cited sets already tracked under `_artifacts/`: directory -> run-record count.
TRACKED_SET_SIZES = {
    "_artifacts/bench_train_step_flash": 4,
    "_artifacts/bench_train_step_flash_llama3b": 4,
    "_artifacts/calib_050/flash_n10240": 2,
    "_artifacts/calib_050/flash_n12288": 2,
    "_artifacts/calib_050/flash_n16384": 2,
}
# What each tracked set's run records carry in their own `identity`.
TRACKED_RUN_SETS = {
    "_artifacts/bench_train_step_flash": RunSet({"51580640": 4}, "0.32.0", "191fc55b97263907"),
    "_artifacts/bench_train_step_flash_llama3b": RunSet(
        {"b76a41d9": 4}, "0.32.0", "191fc55b97263907"
    ),
    "_artifacts/calib_050": RunSet(
        {"a293d664": 2, "142d93a9": 2, "9f3357a9": 2}, "0.32.0", "89ee79937c01655a"
    ),
}
# Tracked files in those sets that are not run records (a model config and the refit's
# own manifest): no identity, no fingerprints, so no manifest row.
TRACKED_NOT_RECORDS = frozenset(
    {"_artifacts/calib_050/qwen3-8b-4bit.config.json", "_artifacts/calib_050/refit_manifest.json"}
)


def _read_json(path: Path) -> tuple[Any, list[str]]:
    """Parse a JSON file. Also return every non-finite constant the parser met
    (`Infinity`, `-Infinity`, `NaN`: Python's json writer emits these bare words)."""
    seen: list[str] = []

    def keep(name: str) -> float:
        seen.append(name)
        return float(name)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=keep), seen


def _load_json(path: Path) -> Any:
    return _read_json(path)[0]


class _ManifestError(Exception):
    pass


def _evidence_dir() -> Path:
    if not EVIDENCE_DIR.is_dir():
        pytest.fail(_missing_folder_message(EVIDENCE_DIR))
    return EVIDENCE_DIR


def _missing_folder_message(folder: Path) -> str:
    return (
        f"the evidence folder {folder} is missing: these checks need a full git "
        "checkout of the repository (an sdist does not carry the records or the history)"
    )


@cache
def _read_manifest(folder: Path) -> dict[str, list[dict[str, Any]]]:
    if not folder.is_dir():
        raise _ManifestError(_missing_folder_message(folder))
    path = folder / "MANIFEST.json"
    try:
        manifest = _load_json(path)
    except (OSError, ValueError) as exc:
        raise _ManifestError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {RECORDS_KEY, TRACKED_KEY}:
        raise _ManifestError(
            f"{path} must be a JSON object holding exactly the lists "
            f"{RECORDS_KEY!r} and {TRACKED_KEY!r}"
        )
    for key, rows in manifest.items():
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise _ManifestError(f"{path}: {key!r} must be a JSON array of objects")
    return manifest


def _section(key: str) -> list[dict[str, Any]]:
    """One manifest list; fails the calling test with a clear message when the folder
    or MANIFEST.json is missing or unreadable."""
    try:
        return _read_manifest(EVIDENCE_DIR)[key]
    except _ManifestError as exc:
        pytest.fail(str(exc))


def _manifest_rows() -> list[dict[str, Any]]:
    """The rows for the files in the evidence folder."""
    return _section(RECORDS_KEY)


def _tracked_rows() -> list[dict[str, Any]]:
    """The rows for the cited run records tracked under `_artifacts/`."""
    return _section(TRACKED_KEY)


def _all_rows() -> list[dict[str, Any]]:
    return _manifest_rows() + _tracked_rows()


def _row_id(path: str) -> str:
    return path.removeprefix(EVIDENCE_PREFIX)


def _params(keep: Callable[[dict[str, Any]], bool] = lambda _row: True) -> list[Any]:
    """Collection-time parameters, one per manifest row of either list. When the
    manifest cannot be read, a single placeholder whose test then fails, so collection
    never stops."""
    try:
        manifest = _read_manifest(EVIDENCE_DIR)
    except _ManifestError:
        return [pytest.param(None, id=UNREADABLE_ID)]
    rows = manifest[RECORDS_KEY] + manifest[TRACKED_KEY]
    paths = [str(r.get("path")) for r in rows if keep(r)]
    return [pytest.param(p, id=_row_id(p)) for p in paths]


def _row(path: str | None) -> dict[str, Any]:
    rows = _all_rows()
    matching = [r for r in rows if r.get("path") == path]
    if len(matching) != 1:
        pytest.fail(f"MANIFEST.json has {len(matching)} rows for {path!r}")
    return matching[0]


def _record_path(path: str) -> Path:
    if path.startswith(EVIDENCE_PREFIX):
        return _evidence_dir() / path.removeprefix(EVIDENCE_PREFIX)
    if path.startswith(TRACKED_PREFIX):
        return REPO_ROOT / path
    pytest.fail(f"{path!r} is neither in {EVIDENCE_PREFIX} nor in {TRACKED_PREFIX}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> bytes:
    """Run read-only git against this repo; any failure fails the test with git's message."""
    git = shutil.which("git")
    if git is None:
        pytest.fail("git is not installed: the record fingerprints cannot be checked")
    try:
        result = subprocess.run([git, "-C", str(REPO_ROOT), *args], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"`git {' '.join(args)}` failed ({exc.returncode}): "
            f"{exc.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _require_full_history() -> None:
    """A shallow clone lacks the July revisions; say so instead of skipping."""
    shallow = _git("rev-parse", "--is-shallow-repository").decode().strip()
    if shallow != "false":
        pytest.fail(
            "this checkout is a shallow clone (git rev-parse --is-shallow-repository "
            f"printed {shallow!r}); fetch the full history (fetch-depth: 0) to check "
            "the record fingerprints"
        )


@cache
def _require_commit_in_history(commit: str) -> None:
    _require_full_history()
    if _git("cat-file", "-t", commit).decode().strip() != "commit":
        pytest.fail(f"{commit} is not a commit in this repository")
    # The commit must be in this branch's history, which contains main.
    _git("merge-base", "--is-ancestor", commit, "HEAD")


@cache
def _blob_at(commit: str, path: str) -> bytes:
    _require_commit_in_history(commit)
    return _git("show", f"{commit}:{path}")


def test_collection_survives_a_missing_evidence_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug caught: the manifest read at collection raising when the folder is missing
    (an sdist), which stops collection of every test file, not just this one."""
    monkeypatch.setattr(sys.modules[__name__], "EVIDENCE_DIR", tmp_path / "absent")
    assert [p.id for p in _params()] == [UNREADABLE_ID]
    assert [p.values for p in _params()] == [(None,)]


def test_a_missing_evidence_folder_fails_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bug caught: a missing evidence folder turned into a skip or an empty pass instead
    of a failure that says what is missing."""
    monkeypatch.setattr(sys.modules[__name__], "EVIDENCE_DIR", tmp_path / "absent")
    # BaseException: a pytest.skip() raised here must turn this test red, not skip it.
    with pytest.raises(BaseException, match=r"evidence folder .*absent is missing") as outcome:
        _manifest_rows()
    assert outcome.type is pytest.fail.Exception


def test_folder_files_equal_manifest_paths() -> None:
    """Bug caught: a file in the evidence folder with no manifest row (whatever its name
    or extension), or a row whose file was never copied."""
    rows = _manifest_rows()
    folder = _evidence_dir()
    on_disk = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        # Finder writes .DS_Store, .gitignore excludes it, and no record can have that name.
        if p.is_file() and p.name != ".DS_Store"
    } - NOT_RECORDS
    listed = [str(r["path"]) for r in rows]
    assert [p for p in listed if not p.startswith(EVIDENCE_PREFIX)] == []
    ids = [_row_id(p) for p in listed]
    assert len(ids) == len(set(ids)), "MANIFEST.json lists a path twice"
    assert on_disk == set(ids), (
        f"not in manifest: {sorted(on_disk - set(ids))}; "
        f"missing on disk: {sorted(set(ids) - on_disk)}"
    )


def test_manifest_rows_carry_exactly_the_published_fields() -> None:
    """Bug caught: a row carrying a path into the maintainer's local `_artifacts/` folder
    (the dropped `source` field), or a row of either list missing a field the checks
    below read."""
    assert {tuple(r) for r in _all_rows()} == {ROW_FIELDS}


def _git_files(*pathspecs: str) -> set[str]:
    return {p for p in _git("ls-files", "-z", "--", *pathspecs).decode().split("\0") if p}


def test_tracked_rows_are_exactly_the_tracked_run_records_of_their_sets() -> None:
    """Bug caught: a `tracked_elsewhere` row naming a file git does not track (a local-only
    record under the ignored `_artifacts/`), a run record of a cited set left without a
    row, or a non-record file (the model config, the refit manifest) given one."""
    listed = [str(r["path"]) for r in _tracked_rows()]
    assert len(listed) == len(set(listed)), "MANIFEST.json lists a tracked path twice"
    set_dirs = sorted({f"{d}/" for d in TRACKED_RUN_SETS})
    in_git = _git_files(*set_dirs)
    assert set(listed) <= in_git, f"not in git ls-files: {sorted(set(listed) - in_git)}"
    assert set(listed) == in_git - TRACKED_NOT_RECORDS


def test_each_tracked_set_has_its_size() -> None:
    """Bug caught: a run record removed from git together with its `tracked_elsewhere`
    row (the git-ls-files check cannot see that), or the whole list dropped."""
    counts: dict[str, int] = {}
    for row in _tracked_rows():
        parent = PurePosixPath(str(row["path"])).parent.as_posix()
        counts[parent] = counts.get(parent, 0) + 1
    assert counts == TRACKED_SET_SIZES


def test_each_published_set_has_its_size() -> None:
    """Bug caught: a record dropped from a published set together with its manifest row
    (the folder-vs-manifest check cannot see that). These are the sizes of the sets
    chosen for publication, not a claim that every run of July 2026 is here."""
    counts: dict[str, int] = {}
    for row in _manifest_rows():
        parent = PurePosixPath(_row_id(str(row["path"]))).parent.as_posix()
        counts[parent] = counts.get(parent, 0) + 1
    assert counts == PUBLISHED_SET_SIZES


def _expected_runs(runs: tuple[RunSet, ...]) -> Counter[tuple[str, str, str]]:
    """(session prefix, mlx version, script fingerprint) -> number of records."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for run in runs:
        for session, n in run.sessions.items():
            counts[(session, run.mlx_version, run.script_sha)] += n
    return counts


def _runs_in(identities: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (str(i.get("session_id"))[:8], str(i.get("mlx_version")), str(i.get("script_sha")))
        for i in identities
    )


@pytest.mark.parametrize("set_dir", sorted(RUN_SETS))
def test_each_run_set_holds_only_records_of_its_own_run(set_dir: str) -> None:
    """Bug caught: a record from another run moved into a set (another session, mlx
    version or script revision), even with its manifest row rebuilt to match, or a record
    of one run carrying another run's mlx version or script revision. Checked against each
    record's own `identity`, not the manifest."""
    files = sorted((_evidence_dir() / set_dir).glob("*.json"))
    identities = [_load_json(p).get("identity", {}) for p in files]
    assert _runs_in(identities) == _expected_runs(RUN_SETS[set_dir])


def test_each_session_in_the_two_session_folder_holds_its_own_files() -> None:
    """Bug caught: two records of the two sessions in the pre-rc folder trading their
    identities (session, mlx version, script revision) with both manifest rows rebuilt;
    the per-run counts stay balanced, so only the file names tell the swap."""
    by_session: dict[str, set[str]] = {}
    for path in sorted((_evidence_dir() / PRE_RC_SWEEP_DIR).glob("*.json")):
        session = str(_load_json(path).get("identity", {}).get("session_id"))[:8]
        by_session.setdefault(session, set()).add(path.name)
    assert by_session == PRE_RC_SESSION_FILES


def test_the_0711_sweep_keeps_its_outcomes() -> None:
    """Bug caught: a point of the 2026-07-11 sweep whose status changed after copying (a
    refused flash point at 10,240 reading `ok`, a stock abort turned into a completion),
    an abort whose ceiling is not 28.0 GiB, or two files that traded their bytes (the
    9,984 and 10,240 flash records, say), each even with its manifest hash rebuilt. Each
    file is read by its name, `seq{n}_{arm}`, and must hold that length and arm."""
    assert {_sweep_file(n, arm) for n, arm in SWEEP_0711_OUTCOMES} == PRE_RC_SESSION_FILES[
        "68b48e5f"
    ]
    named: dict[tuple[int, str], tuple[Any, Any]] = {}
    outcomes: dict[tuple[int, str], str] = {}
    ceilings: set[Any] = set()
    for n, arm in SWEEP_0711_OUTCOMES:
        record = _load_json(_evidence_dir() / PRE_RC_SWEEP_DIR / _sweep_file(n, arm))
        identity = record["identity"]
        named[(n, arm)] = (identity["seq_len"], identity["stock"])
        outcomes[(n, arm)] = record["status"]
        if record["status"] == "aborted_memory_ceiling":
            ceilings.add(record["ceiling_gb"])
    assert named == {(n, arm): (n, arm == "stock") for n, arm in SWEEP_0711_OUTCOMES}
    assert outcomes == SWEEP_0711_OUTCOMES
    assert ceilings == {SWEEP_0711_CEILING_GB}


@pytest.mark.parametrize("set_dir", sorted(TRACKED_RUN_SETS))
def test_each_tracked_set_holds_only_records_of_its_own_run(set_dir: str) -> None:
    """Bug caught: a run record from another run (another session, mlx version or script
    revision) committed into a cited set, even with its manifest row rebuilt to match.
    Read from what git tracks in the set, checked against each record's own `identity`."""
    expected = TRACKED_RUN_SETS[set_dir]
    files = sorted(_git_files(f"{set_dir}/") - TRACKED_NOT_RECORDS)
    identities = [_load_json(REPO_ROOT / p).get("identity", {}) for p in files]
    assert dict(Counter(str(i.get("session_id"))[:8] for i in identities)) == expected.sessions
    assert {i.get("mlx_version") for i in identities} == {expected.mlx_version}
    assert {i.get("script_sha") for i in identities} == {expected.script_sha}


@pytest.mark.parametrize("path", _params())
def test_record_bytes_match_manifest_hash(path: str | None) -> None:
    """Bug caught: a record of either list changed after it was committed (re-saved,
    reformatted or edited) without its manifest row. This proves the bytes are unchanged
    since this commit; that the evidence-folder copies equal the July originals was shown
    once, when they were copied."""
    row = _row(path)
    assert _sha256(_record_path(str(row["path"])).read_bytes()) == row["sha256"]


@pytest.mark.parametrize("path", _params())
def test_record_identity_matches_manifest_row(path: str | None) -> None:
    """Bug caught: a manifest row that names another record's script or library
    fingerprint, session or mlx version than the record itself carries."""
    row = _row(path)
    identity = _load_json(_record_path(str(row["path"]))).get("identity", {})
    assert identity.get("script_sha") == row["script_sha"]
    assert identity.get("code_sha") == row["code_sha"]
    assert identity.get("session_id") == row["session_id"]
    assert identity.get("mlx_version") == row["mlx_version"]


def _non_finite_floats(obj: Any) -> list[float]:
    """Every float in a parsed JSON value that is inf or NaN, however it was spelled
    (an overflowing literal such as `1e400` parses to inf without the parser's hook)."""
    if isinstance(obj, float):
        return [] if math.isfinite(obj) else [obj]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _non_finite_floats(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _non_finite_floats(v)]
    return []


def test_only_the_23040_record_holds_a_non_finite_value() -> None:
    """Bug caught: an Infinity, NaN or overflowing literal such as `1e400` in any other
    record (another diverged run, or a value corrupted on copying), even with its
    manifest hash updated to match."""
    rows = _manifest_rows()
    assert len(rows) == 57
    non_finite: set[str] = set()
    for row in rows:
        value, constants = _read_json(_record_path(str(row["path"])))
        if constants or _non_finite_floats(value):
            non_finite.add(str(row["path"]))
    assert non_finite == {SWEEP_23040_PATH}


@pytest.mark.parametrize("path", _params(lambda row: row.get("fingerprint") == "checked"))
def test_script_commit_reproduces_record_fingerprint(path: str | None) -> None:
    """Bug caught: a manifest row pointing at a commit whose script bytes did not
    produce the record (another revision of the same script, or a wrong script path)."""
    row = _row(path)
    commit, script = str(row["commit"]), str(row["script"])
    assert re.fullmatch(COMMIT_RE, commit), f"commit must be 40 lowercase hex, got {commit!r}"
    assert re.fullmatch(SCRIPT_RE, script), f"script must be scripts/<name>.py, got {script!r}"
    assert _sha256(_blob_at(commit, script))[:16] == row["script_sha"]


@pytest.mark.parametrize("path", _params(lambda row: row.get("code_commit") is not None))
def test_library_commit_reproduces_record_code_fingerprint(path: str | None) -> None:
    """Bug caught: a manifest row naming a library revision, or a dependency list, whose
    files did not produce the record's own `identity.code_sha`. The recipe is the bench's
    `_code_sha`: sha256 over each dependency's file name and bytes, in order, first 16 hex."""
    row = _row(path)
    commit, deps = str(row["code_commit"]), row["code_deps"]
    assert re.fullmatch(COMMIT_RE, commit), f"code_commit must be 40 lowercase hex, got {commit!r}"
    assert isinstance(deps, list), "code_deps must be a list"
    assert deps, "code_deps must not be empty"
    assert [d for d in deps if not re.fullmatch(CODE_DEP_RE, str(d))] == []
    digest = hashlib.sha256()
    for dep in deps:
        digest.update(PurePosixPath(dep).name.encode())
        digest.update(_blob_at(commit, dep))
    record = _load_json(_record_path(str(row["path"])))
    assert digest.hexdigest()[:16] == record["identity"]["code_sha"]


def test_every_row_is_checked_except_the_probe() -> None:
    """Bug caught: a fingerprinted row of either list marked not checkable, or given no
    library commit, which silently drops it from the commit checks above. Every tracked
    record's library fingerprint is reproduced by a commit on main, so no tracked row is
    exempt."""
    rows = _all_rows()
    assert [r["path"] for r in rows if r["fingerprint"] != "checked"] == [PROBE_PATH]
    assert [r["path"] for r in rows if r["code_commit"] is None] == [PROBE_PATH]


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _keys(v)}
    return set()


def test_probe_record_is_marked_not_checkable_and_its_script_exists() -> None:
    """Bug caught: the probe row given a made-up script or library fingerprint or commit,
    or naming a script that never existed in the history."""
    (row,) = [r for r in _manifest_rows() if r["path"] == PROBE_PATH]
    record_keys = _keys(_load_json(_record_path(PROBE_PATH)))
    assert "script_sha" not in record_keys
    assert "code_sha" not in record_keys
    assert row["fingerprint"] == NOT_CHECKABLE
    assert row["script_sha"] is None
    assert row["commit"] is None
    assert row["code_sha"] is None
    assert row["code_commit"] is None
    assert row["code_deps"] is None
    assert row["script"] == "scripts/probe_command_buffer_packing.py"
    _require_full_history()
    added = _git("log", "--format=%H", "--diff-filter=A", "HEAD", "--", row["script"])
    assert added.split(), f"{row['script']} was never added in this branch's history"


def test_sweep_23040_record_keeps_its_non_finite_loss() -> None:
    """Bug caught: the 23,040-token sweep record "fixed" after copying (its first loss
    rewritten, its second loss made finite or NaN, a loss added or dropped, its status
    changed). The record reports `ok` with the losses [12.809766, Infinity]."""
    record = _load_json(_record_path(SWEEP_23040_PATH))
    assert record["loss_all"] == [12.809766, math.inf]
    assert record["status"] == "ok"
