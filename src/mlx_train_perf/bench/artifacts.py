"""Bench artifact conventions: identity-keyed JSON results with resume integrity.

Same shape as the committed `scripts/bench_quant_thresholds.py` house pattern: a run's
identity captures everything a stale-vs-fresh decision depends on, results are written
atomically (`.tmp` + rename), and
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
import os
import platform
import sys
import uuid
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.errors import BenchInputError

# T10 (0.2.0, spec §8 amendment): the attention-path CODE_SHA_DEPS entries and the
# attention_impl/dkv_split_policy/attention_variant identity fields below are additive.
# Existing condition kinds (loss_layer, train_step) never pass the new
# condition_identity() kwargs, so their identity dict keeps EXACTLY the same keys as
# before -- the new fields are omitted (not defaulted to `None`) when unset. No existing
# artifact's shape changes, so SCHEMA_VERSION stays 1. T11's attention artifact is a NEW
# `kind`, not a reshape of an existing one, which is the other reason a bump isn't due.
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
        "core/naive.py",
        "core/kernel/launch.py",
        "core/kernel/dispatch.py",
        "core/kernel/source.py",
        "core/guards.py",
        "adapters/mlx_lm.py",
        # T10/T12 (0.2.0): the attention-path files T11/T13's measured conditions depend on.
        # `attention/wrapper.py` (T12's `enable_flash_attention` integration surface) joins
        # here now that it exists -- a train_step condition that enables flash attention runs
        # through it, so a byte-change to it must invalidate a prior artifact.
        "attention/reference.py",
        "attention/api.py",
        "attention/kernel/source.py",
        "attention/kernel/launch.py",
        "attention/kernel/dispatch.py",
        "attention/wrapper.py",
    )
)

_RESERVED_PARAM_KEYS = (
    "kind", "session_id",
    # T10 (0.2.0): reserved because they are supplied to `condition_identity` as
    # dedicated keyword args (see below), not as free-form `params` entries.
    "attention_impl", "dkv_split_policy", "attention_variant",
)


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
    reuse; any difference is what `result_is_fresh` treats as staleness.

    A caller kwarg that happens to reuse one of THIS function's own internal field names
    (e.g. a condition param literally named `code_sha`) would otherwise silently hijack
    that field via `{**internal, **kw}` -- caught here by checking for overlap against
    `internal`'s own keys directly, so the guard can never drift out of sync with the
    field list (unlike a separately maintained reserved-name constant)."""
    internal: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mlx_version": _installed_mlx_version(),
        "mlx_lm_version": _installed_mlx_lm_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "code_sha": _code_sha(CODE_SHA_DEPS),
        "package_version": version("mlx-train-perf"),
    }
    collision = set(kw) & internal.keys()
    if collision:
        raise BenchInputError(
            f"condition params must not use reserved identity key(s) {sorted(collision)} "
            "-- computed internally by run_identity"
        )
    return {**internal, **kw}


def condition_identity(
    *, kind: str, session_id: str, params: dict[str, object],
    attention_impl: str | None = None,
    dkv_split_policy: str | None = None,
    attention_variant: str | None = None,
) -> dict[str, object]:
    """The single call site both `runner.run_conditions` and `worker.main` use to build
    one condition's identity. `kind`/`session_id` are supplied separately by the
    caller -- a `params` dict that happens to reuse either name would otherwise reach
    `run_identity(kind=kind, session_id=session_id, **params)` and fail with a raw
    `TypeError: got multiple values for keyword argument`; this raises a clean, named
    error instead.

    `attention_impl`/`dkv_split_policy`/`attention_variant` (T10, 0.2.0, spec §8
    amendment) are the same kind of dedicated, reserved identity input as `kind`/
    `session_id` -- an attention-measuring condition (T11/T13) passes them explicitly so
    two conditions differing only in one of them get different identities; `params` may
    not also set them (rejected below, same collision-avoidance reasoning as `kind`/
    `session_id`). Each is OMITTED from the returned identity (not defaulted to `None`)
    when the caller leaves it unset, so a non-attention condition kind's identity keeps
    the exact same keys it had before this parameter existed."""
    for key in _RESERVED_PARAM_KEYS:
        if key in params:
            raise BenchInputError(
                f"condition params must not use the reserved key {key!r} -- it is "
                "supplied separately by the bench runner/worker"
            )
    attention_fields: dict[str, object] = {
        field_name: value for field_name, value in (
            ("attention_impl", attention_impl),
            ("dkv_split_policy", dkv_split_policy),
            ("attention_variant", attention_variant),
        )
        if value is not None
    }
    return run_identity(kind=kind, session_id=session_id, **attention_fields, **params)


def write_result(path: Path, identity: dict[str, object], status: str, **fields: object) -> None:
    """Atomic write (tmp + rename) -- an interrupted worker leaves either the PRIOR
    artifact or nothing at `path`, never a half-written JSON `result_is_fresh` could
    misparse as fresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"identity": identity, "status": status, **fields}, indent=2))
    tmp.rename(path)


def make_watchdog_on_breach(
    out: Path,
    identity: dict[str, object],
    ceiling_bytes: int,
    *,
    exit_fn: Callable[[int], object] = os._exit,
    exit_code: int = 70,
) -> Callable[[str, dict[str, object]], None]:
    """Build the `on_breach` callback `core.guards.install_memory_watchdog` fires when a
    bench condition breaches the active-memory ceiling or the wall budget. It records the
    breach AS A RESULT -- writing THIS condition's artifact with an honest
    `aborted_<reason>` status (`aborted_memory_ceiling` / `aborted_wall_budget`) plus the
    observed numbers, via the SAME `write_result` identity path the worker's own
    ok/refused write uses -- flushes stdio, then HARD-exits the process (`os._exit`, no
    cleanup: a GPU paging storm cannot be cleanly unwound, and a normal `raise`/`sys.exit`
    from a daemon thread would not stop the main thread that is mid-allocation).

    The written artifact is the durable record of the breach; because its status is not
    `"ok"`, `result_is_fresh` treats it as stale, so a later resume run retries the
    condition (e.g. under a tighter budget). `exit_fn`/`exit_code` are injectable so a
    test can assert the write and the exit code without terminating the test runner."""

    def on_breach(reason: str, details: dict[str, object]) -> None:
        active_bytes = int(cast(int, details.get("active_bytes", 0)))
        elapsed_s = float(cast(float, details.get("elapsed_s", 0.0)))
        write_result(
            out, identity, f"aborted_{reason}",
            observed_active_gb=round(active_bytes / 1024**3, 4),
            ceiling_gb=round(ceiling_bytes / 1024**3, 4),
            elapsed_s=round(elapsed_s, 3),
            wall_budget_s=details.get("wall_budget_s"),
        )
        sys.stdout.flush()
        sys.stderr.flush()
        exit_fn(exit_code)

    return on_breach


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
