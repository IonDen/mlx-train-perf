"""Subprocess-per-condition bench runner: resume-safe, same-session ratio reporting.

Each `Condition` gets its OWN Python process (`python -m mlx_train_perf.bench.worker`) --
the spike-proven isolation pattern (MLX's lazy allocator otherwise holds buffers across
runs within one process) -- and its own artifact, written the instant it finishes. A
condition whose artifact is already fresh is skipped entirely, never spawned; a worker
that exits nonzero (crashes) gets its failure recorded as a `status="error"` result here,
on the CALLER's side, so one bad condition never aborts the rest of the sweep.
"""
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mlx_train_perf.bench.artifacts import result_is_fresh, run_identity, write_result

_STDERR_TAIL_CHARS = 4000  # enough to see the failing assertion/traceback, not a full dump


@dataclass(frozen=True, slots=True, kw_only=True)
class Condition:
    name: str
    kind: str
    params: dict[str, object]


def _spawn_worker(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mlx_train_perf.bench.worker", "--config", str(config_path)],
        capture_output=True, text=True, check=False,
    )


def run_conditions(
    conditions: list[Condition], out_dir: Path, *, session_id: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for condition in conditions:
        out_path = out_dir / f"{condition.name}.json"
        ident = run_identity(kind=condition.kind, session_id=session_id, **condition.params)
        paths.append(out_path)
        if result_is_fresh(out_path, ident):
            continue

        config = {
            "kind": condition.kind, "params": condition.params, "session_id": session_id,
            "out": str(out_path),
        }
        # The config lives in the SYSTEM temp dir, deliberately never inside `out_dir` --
        # an interrupted run must not leave a stray `.json` there for a later glob over
        # the artifact directory to misread as a result.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)
        try:
            proc = _spawn_worker(config_path)
            if proc.returncode != 0:
                # The worker itself wrote nothing (it crashed before/without reaching its
                # own `write_result`) -- this IS the sweep-level failure envelope, keyed
                # by the SAME identity the worker would have used, so a later resume run
                # (after the underlying bug is fixed) still sees it as stale and retries.
                stderr_tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
                write_result(
                    out_path, ident, "error", error_type="WorkerCrashed",
                    error_msg=stderr_tail, returncode=proc.returncode,
                )
        finally:
            config_path.unlink(missing_ok=True)
    return paths


def _identity_of(entry: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], entry["identity"])


def _group_key(identity: dict[str, object]) -> tuple[tuple[str, object], ...]:
    """The "same experimental grid point" key: every identity field EXCEPT `impl` (the
    dimension being compared) and `session_id` (checked separately, as a gate on whether
    a ratio may be emitted at all -- see `report`)."""
    return tuple(sorted((k, v) for k, v in identity.items() if k not in ("impl", "session_id")))


def report(paths: list[Path]) -> dict[str, object]:
    """Aggregate a set of artifacts into pairwise `impl` ratios (by `wall_s`) at each
    shared grid point. A pair that shares a grid point but NOT a `session_id` is a
    cross-machine/cross-run comparison -- refused, and named in `cross_session_excluded`,
    rather than silently blended into `ratios`."""
    entries: list[dict[str, object]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") != "ok":
            continue
        entries.append(data)

    groups: dict[tuple[tuple[str, object], ...], list[dict[str, object]]] = defaultdict(list)
    for data in entries:
        groups[_group_key(_identity_of(data))].append(data)

    ratios: dict[str, float] = {}
    cross_session_excluded: list[dict[str, object]] = []
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                ident_a, ident_b = _identity_of(a), _identity_of(b)
                impl_a, impl_b = ident_a.get("impl"), ident_b.get("impl")
                if impl_a == impl_b:
                    continue  # a genuine re-run of the same impl, not a comparison
                lo, hi = sorted((str(impl_a), str(impl_b)))
                label = f"{lo}/{hi}"
                if ident_a.get("session_id") != ident_b.get("session_id"):
                    cross_session_excluded.append({
                        "impl_a": impl_a, "session_a": ident_a.get("session_id"),
                        "impl_b": impl_b, "session_b": ident_b.get("session_id"),
                    })
                    continue
                wall_lo = a.get("wall_s") if str(impl_a) == lo else b.get("wall_s")
                wall_hi = b.get("wall_s") if str(impl_b) == hi else a.get("wall_s")
                if (
                    isinstance(wall_lo, int | float)
                    and isinstance(wall_hi, int | float)
                    and wall_lo
                ):
                    ratios[label] = wall_hi / wall_lo
    return {"ratios": ratios, "cross_session_excluded": cross_session_excluded}
