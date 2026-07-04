"""Bench artifact conventions: identity-keyed JSON results with resume integrity.

Same shape as the committed `scripts/bench_quant_thresholds.py` house pattern (itself a
port of `mlx-train-perf-spike/common.py:47-62`): a run's identity captures everything a
stale-vs-fresh decision depends on, results are written atomically (`.tmp` + rename), and
freshness requires an EXACT identity match plus `status == "ok"`.

Identity carries a `code_sha` -- the same SHA-256-over-name+bytes pattern
`scripts/bench_quant_thresholds.py._code_sha` uses -- over a declared list of files the
measured path actually depends on (the bench worker plus the loss/kernel/guard modules
it calls into). This is the field `result_is_fresh` actually needs: an installed
package's `importlib.metadata.version` is hatch-vcs tag-derived and only refreshes on an
explicit reinstall, NOT on every `uv sync` -- a plain checkout can therefore have edited
measured-path files while `version("mlx-train-perf")` still reports last release's
string, which would let a stale artifact be served as fresh (the exact failure this
harness exists to prevent). `package_version` is kept in the identity too, but purely as
informational provenance, not as the staleness signal.
"""
import hashlib
import json
import platform
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.errors import BenchInputError

SCHEMA_VERSION = 1

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../src/mlx_train_perf

# Every file whose bytes change what a `loss_layer` (or later `train_step`) condition
# actually measures -- the bench worker itself, plus everything on the loss-computation
# path it calls into. Deliberately explicit rather than "every .py in the repo": a
# docs-only or CLI-only edit must NOT invalidate every bench artifact.
CODE_SHA_DEPS: tuple[Path, ...] = tuple(
    _PACKAGE_ROOT / rel for rel in (
        "bench/worker.py",
        "core/loss.py",
        "core/chunked.py",
        "core/kernel/launch.py",
        "core/kernel/dispatch.py",
        "core/kernel/source.py",
        "core/guards.py",
    )
)

_RESERVED_PARAM_KEYS = ("kind", "session_id")


def _code_sha(deps: tuple[Path, ...]) -> str:
    """SHA-256 over name+bytes of each dep file, in the given order -- any edit to a
    measured-path file changes this, which is what `result_is_fresh` uses to invalidate
    a prior artifact (same convention as `scripts/bench_quant_thresholds.py._code_sha`).
    Recomputed fresh on every call (no caching): identity is meant to reflect the
    ON-DISK state of these files at the moment it's built, not a snapshot from import
    time."""
    h = hashlib.sha256()
    for p in deps:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


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
    a `code_sha` over the measured-path dependency files (see `CODE_SHA_DEPS` -- the
    staleness signal `result_is_fresh` actually relies on), this package's own installed
    version (informational only), plus whatever identity-relevant kwargs the caller
    supplies (condition kind, grid point, dtype, impl, tile/chunk, session_id, ...). Two
    calls returning equal dicts describe the SAME run in every way that matters for
    reuse; any difference is what `result_is_fresh` treats as staleness."""
    return {
        "schema_version": SCHEMA_VERSION,
        "mlx_version": _installed_mlx_version(),
        "mlx_lm_version": _installed_mlx_lm_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "code_sha": _code_sha(CODE_SHA_DEPS),
        "package_version": version("mlx-train-perf"),
        **kw,
    }


def condition_identity(
    *, kind: str, session_id: str, params: dict[str, object],
) -> dict[str, object]:
    """The single call site both `runner.run_conditions` and `worker.main` use to build
    one condition's identity. `kind`/`session_id` are supplied separately by the
    caller -- a `params` dict that happens to reuse either name would otherwise reach
    `run_identity(kind=kind, session_id=session_id, **params)` and fail with a raw
    `TypeError: got multiple values for keyword argument`; this raises a clean, named
    error instead."""
    for key in _RESERVED_PARAM_KEYS:
        if key in params:
            raise BenchInputError(
                f"condition params must not use the reserved key {key!r} -- it is "
                "supplied separately by the bench runner/worker"
            )
    return run_identity(kind=kind, session_id=session_id, **params)


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
