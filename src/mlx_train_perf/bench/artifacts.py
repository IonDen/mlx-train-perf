"""Bench artifact conventions: identity-keyed JSON results with resume integrity.

Same shape as the committed `scripts/bench_quant_thresholds.py` house pattern (itself a
port of `mlx-train-perf-spike/common.py:47-62`): a run's identity captures everything a
stale-vs-fresh decision depends on, results are written atomically (`.tmp` + rename), and
freshness requires an EXACT identity match plus `status == "ok"`.

One refinement over the ad hoc bench script: identity keys off this package's own
(hatch-vcs tag-derived) installed version rather than a source-file hash. A source hash
made sense for a standalone script whose own `.py` files ARE the versioned artifact; this
harness ships inside the installed package, so `importlib.metadata.version` already
captures "which code produced this result" without re-deriving it from the filesystem.
"""
import json
import platform
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mlx_train_perf._compat import _installed_mlx_version

SCHEMA_VERSION = 1


def _installed_mlx_lm_version() -> str | None:
    """`mlx-lm` is an optional extra -- absent installs must not fail identity
    construction, they just record `None` (still a stable, comparable identity value)."""
    try:
        return version("mlx-lm")
    except PackageNotFoundError:
        return None


def new_session_id() -> str:
    """A fresh identity token for one bench invocation -- distinguishes artifacts from
    different runs so `report`'s ratio logic can refuse to compare across them."""
    return uuid.uuid4().hex


def run_identity(**kw: object) -> dict[str, object]:
    """Everything a result's freshness depends on: mlx/mlx-lm versions, machine, macos,
    this package's own installed version, plus whatever identity-relevant kwargs the
    caller supplies (condition kind, grid point, dtype, impl, tile/chunk, session_id,
    ...). Two calls returning equal dicts describe the SAME run in every way that
    matters for reuse; any difference is what `result_is_fresh` treats as staleness."""
    return {
        "schema_version": SCHEMA_VERSION,
        "mlx_version": _installed_mlx_version(),
        "mlx_lm_version": _installed_mlx_lm_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "package_version": version("mlx-train-perf"),
        **kw,
    }


def write_result(path: Path, identity: dict[str, object], status: str, **fields: object) -> None:
    """Atomic write (tmp + rename) -- an interrupted worker leaves either the PRIOR
    artifact or nothing at `path`, never a half-written JSON `result_is_fresh` could
    misparse as fresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"identity": identity, "status": status, **fields}, indent=2))
    tmp.rename(path)


def result_is_fresh(path: Path, identity: dict[str, object]) -> bool:
    """Fresh = parses, `status == "ok"`, identity matches EXACTLY. A missing file, a
    corrupt one, a recorded error/refusal, or ANY identity field drift (including one the
    caller didn't think to check) all trigger a recompute rather than a stale reuse."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("status") == "ok" and data.get("identity") == identity)
