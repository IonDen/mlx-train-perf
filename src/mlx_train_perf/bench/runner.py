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

from mlx_train_perf.bench.artifacts import condition_identity, result_is_fresh, write_result

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
        ident = condition_identity(
            kind=condition.kind, session_id=session_id, params=condition.params,
        )
        paths.append(out_path)
        if result_is_fresh(out_path, ident):
            continue
        # By definition stale (missing, corrupt, an old error/refusal, or an identity
        # mismatch) -- remove it BEFORE spawning so `out_path.exists()` after the worker
        # returns means exactly "THIS worker wrote it". Without this, a worker that fails
        # silently (exits 0, writes nothing) would leave the stale artifact in place and
        # the sweep would report someone else's old "ok" result as this run's truth.
        out_path.unlink(missing_ok=True)

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
            elif not out_path.exists():
                # A clean exit that wrote nothing is still a sweep-level failure (e.g. a
                # worker that swallowed its own crash) -- recorded the same way, so a
                # later resume still sees this condition as stale and retries it.
                write_result(
                    out_path, ident, "error", error_type="WorkerExitedWithoutArtifact",
                    error_msg="worker exited 0 without writing an artifact", returncode=0,
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


def _ratio_label_and_value(
    impl_a: object, impl_b: object, wall_a: object, wall_b: object,
) -> tuple[str, float] | None:
    """Direction is by MEASURED speed (`f"{slower_impl}/{faster_impl}"`, value = slower
    `wall_s` / faster `wall_s`) -- never by the impls' alphabetical name order, which is
    an accident of spelling and not a proxy for which one actually ran slower. `None`
    when either wall time is missing or non-numeric (can't rank), or one is zero (can't
    divide by it)."""
    if not (isinstance(wall_a, int | float) and isinstance(wall_b, int | float)):
        return None
    if not wall_a or not wall_b:
        return None
    if wall_a >= wall_b:
        slower_impl, slower_wall, faster_impl, faster_wall = impl_a, wall_a, impl_b, wall_b
    else:
        slower_impl, slower_wall, faster_impl, faster_wall = impl_b, wall_b, impl_a, wall_a
    return f"{slower_impl}/{faster_impl}", slower_wall / faster_wall


def report(paths: list[Path]) -> dict[str, object]:
    """Aggregate a set of artifacts into pairwise `impl` ratios at each shared grid
    point (direction/magnitude: see `_ratio_label_and_value`). A pair that shares a grid
    point but NOT a `session_id` is a cross-machine/cross-run comparison -- refused, and
    named in `cross_session_excluded`, rather than silently blended into `ratios`."""
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
                if ident_a.get("session_id") != ident_b.get("session_id"):
                    cross_session_excluded.append({
                        "impl_a": impl_a, "session_a": ident_a.get("session_id"),
                        "impl_b": impl_b, "session_b": ident_b.get("session_id"),
                    })
                    continue
                result = _ratio_label_and_value(impl_a, impl_b, a.get("wall_s"), b.get("wall_s"))
                if result is not None:
                    label, value = result
                    ratios[label] = value
    return {"ratios": ratios, "cross_session_excluded": cross_session_excluded}
