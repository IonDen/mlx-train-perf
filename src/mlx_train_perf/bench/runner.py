"""Subprocess-per-condition bench runner: resume-safe, same-session ratio reporting.

Each `Condition` gets its OWN Python process (`python -m mlx_train_perf.bench.worker`) --
the spike-proven isolation pattern (MLX's lazy allocator otherwise holds buffers across
runs within one process) -- and its own artifact, written the instant it finishes. A
condition whose artifact is already fresh is skipped entirely, never spawned; a worker
that exits nonzero (crashes) gets its failure recorded as a `status="error"` result here,
on the CALLER's side, so one bad condition never aborts the rest of the sweep.
"""
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mlx_train_perf.bench.artifacts import condition_identity, result_is_fresh, write_result
from mlx_train_perf.errors import BenchInputError

_STDERR_TAIL_CHARS = 4000  # enough to see the failing assertion/traceback, not a full dump
# Supervisor outcome kinds that describe how the WORKER ended (every other kind describes
# the supervisor, the launch, or a policy decision).
_CHILD_OUTCOMES = frozenset({"child_exited", "child_signaled"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Condition:
    name: str
    kind: str
    params: dict[str, object]
    # A dedicated, reserved identity input carried OUT of `params` -- the same kind of
    # first-class field as `kind`, threaded to `condition_identity(attention_impl=...)`
    # here and forwarded to the worker via its config so `worker.main` rebuilds the SAME
    # identity and runs the matching attention path. `params` may NOT also set it
    # (`condition_identity` rejects the reserved key). Left `None` for every condition
    # that does not select an attention implementation (all loss_layer conditions, and any
    # 0.1.0-era train_step config) -- `condition_identity` then OMITS it, keeping those
    # identities byte-identical to before this field existed.
    attention_impl: str | None = None
    # Wall-clock budget (seconds) the worker's memory/wall watchdog fails this condition
    # against. Rides the SAME top-level config seam as `attention_impl` -- carried out of
    # `params`, threaded into the worker config below -- but, unlike `attention_impl`, it
    # is deliberately NOT an identity input: a safety limit is not a measurement
    # dimension, so it is never passed to `condition_identity` (two runs differing only in
    # their budget must share an identity). `None` -> the worker applies its module
    # default (`core.guards.DEFAULT_WALL_BUDGET_S`).
    wall_budget_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalGuardConfig:
    """Opt-in external supervision limits for each condition worker."""

    max_footprint_bytes: int
    sample_interval_ms: int = 50
    wall_time_ms: int | None = None
    # How long the supervisor waits for the worker's checkpoint acknowledgement before it
    # escalates (`None` -> the supervisor's own 1 s default). The worker can only answer
    # at its next poll, i.e. after the repetition or training step in flight, so size this
    # to the step time: a multi-second step under the default always loses its checkpoint.
    checkpoint_timeout_ms: int | None = None
    # `None` defers to the supervisor default ("terminate": a runner that dies takes its
    # worker down). "detach" lets a condition outlive the runner.
    on_parent_exit: str | None = None

    def __post_init__(self) -> None:
        limit = self.max_footprint_bytes
        if type(limit) is not int or not 2 <= limit <= (1 << 64) - 1:
            raise ValueError("max_footprint_bytes must be within 2..=u64::MAX")
        if limit + max(limit // 10, 1) > (1 << 64) - 1:
            raise ValueError("max_footprint_bytes cannot represent the emergency band")
        if (
            type(self.sample_interval_ms) is not int
            or not 10 <= self.sample_interval_ms <= 10_000
        ):
            raise ValueError("sample_interval_ms must be within 10..=10000")
        wall_time = self.wall_time_ms
        if wall_time is not None and (
            type(wall_time) is not int or not 1 <= wall_time <= 30 * 24 * 60 * 60 * 1000
        ):
            raise ValueError("wall_time_ms must be within 1ms..=30d")
        timeout = self.checkpoint_timeout_ms
        if timeout is not None and (
            type(timeout) is not int or not 10 <= timeout <= 60_000
        ):
            raise ValueError("checkpoint_timeout_ms must be within 10ms..=60s")
        if self.on_parent_exit not in (None, "terminate", "detach"):
            raise ValueError("on_parent_exit must be 'terminate', 'detach', or None")


@dataclass(frozen=True, slots=True)
class GuardedWorkerResult:
    """Worker process result plus launch-boundary supervision metadata. `process` is
    `None` when the guard client failed before it could report how the worker ended --
    nothing is fabricated in its place."""

    process: subprocess.CompletedProcess[str] | None
    fallback_reason: str | None
    client_error: str | None
    outcome: dict[str, object] | None = None


def _spawn_worker(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mlx_train_perf.bench.worker", "--config", str(config_path)],
        capture_output=True, text=True, check=False,
    )


def _new_attempt_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def _guard_outcome(result: Any) -> dict[str, object]:
    """What the supervisor's report says happened, in the runner's own record. Read from
    the outcome KIND, never from exit-code numerics: 70 is shared by the in-process
    watchdog's breach exit and the supervisor's own failure outcome, and the supervisor
    ranks its codes (failure over intervention over the command's status)."""
    payload = result.report.payload
    checkpoint = payload.get("checkpoint") or {}
    artifact = checkpoint.get("artifact")
    child_status = (payload.get("outcome") or {}).get("child_status")
    return {
        "kind": str(result.report.outcome.kind.value),
        "returncode": int(result.returncode),
        "child_status": dict(child_status) if isinstance(child_status, Mapping) else child_status,
        "checkpoint_status": checkpoint.get("status"),
        "checkpoint_request_id": checkpoint.get("request_id"),
        "checkpoint_reason": checkpoint.get("reason"),
        "checkpoint_artifact": dict(artifact) if isinstance(artifact, Mapping) else None,
    }


def _spawn_guarded_worker(
    config_path: Path,
    report_path: Path,
    policy: ExternalGuardConfig,
) -> GuardedWorkerResult:
    """Supervise one worker. Direct launch is a fallback ONLY while no supervisor exists:
    the boundary is the call site, not the exception type. `start()` returns once the
    native supervisor is up, so anything it raises about discovery happened before a
    worker could exist. `wait()` re-verifies the binary while loading the final report, so
    the same discovery error there means the worker already ran -- never relaunch."""
    try:
        mlx_guard = importlib.import_module("mlx_guard")
    except ModuleNotFoundError as error:
        # "mlx-guard" (the distribution name) is what a broken install raises: the package
        # reads its own metadata at import, and PackageNotFoundError is a
        # ModuleNotFoundError. Any other missing module is a real bug -- let it surface.
        if error.name not in (None, "mlx_guard", "mlx-guard"):
            raise
        return GuardedWorkerResult(
            process=_spawn_worker(config_path),
            fallback_reason="mlx-guard package is unavailable",
            client_error=None,
        )

    command = (
        sys.executable,
        "-m",
        "mlx_train_perf.bench.worker",
        "--config",
        str(config_path),
    )
    try:
        config = mlx_guard.RunConfig(
            command=command,
            report=report_path,
            max_footprint_bytes=policy.max_footprint_bytes,
            sample_interval_ms=policy.sample_interval_ms,
            wall_time_ms=policy.wall_time_ms,
            checkpoint_timeout_ms=policy.checkpoint_timeout_ms,
            on_parent_exit=policy.on_parent_exit,
        )
        supervised = mlx_guard.start(config, capture_output=True)
    except mlx_guard.SupervisorDiscoveryError:
        return GuardedWorkerResult(
            process=_spawn_worker(config_path),
            fallback_reason="mlx-guard supervisor discovery failed",
            client_error=None,
        )
    except Exception as error:
        # A refused configuration is strictly pre-launch, but it means the caller's policy
        # is wrong -- running unsupervised instead would hide that. A start error is NOT
        # provably pre-launch (the readiness handshake can fail after the worker exists).
        return GuardedWorkerResult(
            process=None,
            fallback_reason=None,
            client_error=(
                f"guard client could not start the supervisor: {type(error).__name__}"
            ),
        )
    try:
        result = supervised.wait()
    except Exception as error:
        return GuardedWorkerResult(
            process=None,
            fallback_reason=None,
            client_error=f"guard client failed after launch: {type(error).__name__}",
        )
    except BaseException:
        # Ctrl-C or an exit request while a condition runs. `subprocess.run` kills its
        # child on any exception; a supervised launch has to ask the supervisor, which
        # then stops the worker's whole process group.
        supervised.cancel()
        raise
    process = subprocess.CompletedProcess(
        command, result.returncode, _text_output(result.stdout), _text_output(result.stderr),
    )
    try:
        outcome = _guard_outcome(result)
    except Exception as error:
        # The worker has already run and its artifact stands. A report this version
        # cannot summarize costs one record, never the rest of the sweep.
        return GuardedWorkerResult(
            process=process,
            fallback_reason=None,
            client_error=f"guard report could not be summarized: {type(error).__name__}",
        )
    return GuardedWorkerResult(
        process=process, fallback_reason=None, client_error=None, outcome=outcome,
    )


def _text_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _prepare_guard_dir(guard_dir: Path) -> None:
    """A private directory this user owns, or a refusal before anything launches. The native
    supervisor demands exactly that (owner-only, no symlink) for its own report; the
    runner's records sit beside it and get the same care, so a pre-planted `_mlx_guard`
    in a shared output directory cannot redirect them."""
    guard_dir.mkdir(parents=True, exist_ok=True)
    info = guard_dir.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BenchInputError(f"{guard_dir} (_mlx_guard) must be a real directory, not a link")
    if info.st_uid != os.geteuid():
        raise BenchInputError(f"{guard_dir} (_mlx_guard) is owned by another user")
    guard_dir.chmod(0o700)


def _launch_under_guard(
    config_path: Path,
    out_path: Path,
    ident: dict[str, object],
    name: str,
    guard_dir: Path,
    guard: ExternalGuardConfig,
) -> subprocess.CompletedProcess[str] | None:
    """One supervised launch plus its records under `_mlx_guard/`. Returns the worker's
    process result, or `None` when the guard client failed (already recorded here)."""
    # A fresh report path per launch attempt: the supervisor retains `.<report>.journal`
    # beside every report and refuses a path whose journal exists, so a fixed name would
    # block the resume retry of any condition. Reports and journals are evidence -- never
    # deleted here.
    guard_report = guard_dir / f"{name}.{_new_attempt_id()}.json"
    records = {
        kind: guard_dir / f"{name}.{kind}.json" for kind in ("launch", "client", "supervision")
    }
    for record in records.values():
        record.unlink(missing_ok=True)
    guarded = _spawn_guarded_worker(config_path, guard_report, guard)
    if guarded.fallback_reason is not None:
        write_result(
            records["launch"], ident, "guard_fallback", reason=guarded.fallback_reason,
        )
        # The condition's own artifact looks the same either way, so say it out loud too.
        print(
            f"mlx-train-perf: {name}: ran UNSUPERVISED ({guarded.fallback_reason})",
            file=sys.stderr,
        )
    if guarded.client_error is not None:
        write_result(
            records["client"], ident, "guard_client_error",
            error=guarded.client_error, report=guard_report.name,
        )
    if guarded.outcome is not None:
        write_result(
            records["supervision"], ident, "guard_supervised",
            outcome=guarded.outcome, report=guard_report.name,
        )
    if not out_path.exists():
        # No worker artifact to respect, and the supervision layer knows why.
        if guarded.client_error is not None and guarded.process is None:
            write_result(
                out_path, ident, "error", error_type="GuardClientError",
                error_msg=guarded.client_error,
            )
        elif guarded.outcome is not None and guarded.outcome["kind"] == "policy_intervention":
            # The guard did its job before the worker reached a poll: an honest abort, the
            # external twin of `aborted_memory_ceiling`.
            write_result(
                out_path, ident, "aborted_external_guard",
                guard_outcome=guarded.outcome, report=guard_report.name,
            )
        elif guarded.outcome is not None and guarded.outcome["kind"] not in _CHILD_OUTCOMES:
            # The supervisor or the launch failed, not the condition. A `child_*` outcome
            # falls through to the caller's ordinary worker-crash envelope instead.
            write_result(
                out_path, ident, "error", error_type="SupervisorReportedFailure",
                guard_outcome=guarded.outcome, report=guard_report.name,
            )
    return guarded.process


def run_conditions(
    conditions: list[Condition],
    out_dir: Path,
    *,
    session_id: str,
    guard: ExternalGuardConfig | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    guard_dir = out_dir / "_mlx_guard"
    if guard is not None:
        _prepare_guard_dir(guard_dir)
    paths: list[Path] = []
    for condition in conditions:
        out_path = out_dir / f"{condition.name}.json"
        ident = condition_identity(
            kind=condition.kind, session_id=session_id, params=condition.params,
            attention_impl=condition.attention_impl,
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
            "attention_impl": condition.attention_impl,
            # Top-level (never inside `params`), always present (`None` when unset) -- the
            # worker reads it with `.get`. NOT part of `ident` above: a wall budget is a
            # safety limit, not a measurement dimension.
            "wall_budget_s": condition.wall_budget_s,
            "out": str(out_path),
        }
        # The config lives in the SYSTEM temp dir, deliberately never inside `out_dir` --
        # an interrupted run must not leave a stray `.json` there for a later glob over
        # the artifact directory to misread as a result.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)
        try:
            proc: subprocess.CompletedProcess[str] | None
            if guard is None:
                proc = _spawn_worker(config_path)
            else:
                proc = _launch_under_guard(
                    config_path, out_path, ident, condition.name, guard_dir, guard,
                )
            if proc is None:
                pass  # recorded above; a client failure never fabricates a worker status
            elif proc.returncode != 0 and not out_path.exists():
                # A nonzero exit that left NO artifact: the worker crashed before/without
                # reaching any `write_result`. THIS is the sweep-level failure envelope,
                # keyed by the SAME identity the worker would have used, so a later resume
                # run (after the underlying bug is fixed) still sees it as stale and
                # retries. If the artifact DOES exist, the worker wrote its own honest
                # record before hard-exiting -- the memory/wall watchdog's `os._exit(70)`
                # breach path (status `aborted_*`) -- and the pre-spawn `unlink` above
                # guarantees it is THIS worker's write, so it is respected, not clobbered.
                stderr_tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
                write_result(
                    out_path, ident, "error", error_type="WorkerCrashed",
                    error_msg=stderr_tail, returncode=proc.returncode,
                )
            elif proc.returncode == 0 and not out_path.exists():
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
