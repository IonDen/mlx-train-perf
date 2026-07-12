"""Community hardware-measurement contribution kit.

One command (`mlx-train-perf contribute`) that any Apple-Silicon user can run to measure
this project's committed benchmarks on THEIR machine and submit the numbers back. It
detects the machine, picks RAM-proportional shapes, prints an honest up-front time
estimate, runs a pre-flight memory/power check, drives the committed benches
(subprocess-per-condition, resumable per-condition artifacts), and writes ONE
provenance-complete `community-benchmarks/<chip>-<ram>gb-<date>.json` plus a pre-filled
PR title/body -- no hand-editing of numbers.

Functional core, imperative shell: the machine parse, the RAM->shape table, the ETA sum,
the pre-flight decision, and the artifact assembly are all pure functions (unit-tested
GPU-free with fakes at the boundary). The measurement itself is a single injectable seam
(`_measure_bench`) so tests never spawn a real bench, load a model, or touch the GPU.

Safety on UNKNOWN hardware is the design center. The kit does NOT install its own
watchdog: every bench it runs (via `bench.runner.run_conditions` for loss_layer/
train_step, and the committed `bench_attention_op`/`northstar_context_sweep` scripts for
the attention and context probes) already installs the device-clamped wired cap plus the
two-term dynamic memory watchdog. The kit's job is provenance -- it
propagates any `memory_warning` those artifacts recorded into the community artifact, and
surfaces the divergence warning prominently in the pre-flight so someone on a crowded
machine sees "expected ~58 GB free, measured 20 GB" before burning an hour.

Claims policy: community numbers are "measured on contributor hardware", never
extrapolated -- the aggregation table (`scripts/aggregate_community.py`) states each row's
own measured numbers with a PR reference, and the maintainer campaign (not the
contributor) owns the stock-attention baseline comparisons.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import cast

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.bench.artifacts import run_identity, write_result
from mlx_train_perf.bench.runner import Condition, run_conditions
from mlx_train_perf.core.guards import EffectiveCeiling, effective_memory_ceiling
from mlx_train_perf.errors import (
    BenchInputError,
    MachineDetectionError,
    MemoryBudgetError,
    MissingDependencyError,
)

COMMUNITY_SCHEMA_VERSION = 1

# The kit runs the FLASH arm (the library's own path -- that's what community numbers are
# for). Both attention arms run in the model-free single-op bench (the O(N) vs O(N^2)
# proof); the end-to-end train-step/context probes run ours-only (stock comparisons are
# the maintainer campaign's job).
_ATTENTION_IMPLS = ("flash", "stock")
_LOSS_IMPLS = ("kernel", "chunked", "naive")


# --- machine detection ----------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MachineInfo:
    chip: str
    ram_gib: int
    ram_bytes: int
    macos: str
    mlx_version: str
    package_version: str


def parse_chip(brand_string: str) -> str:
    """Normalize the `sysctl machdep.cpu.brand_string` output: strip and collapse any
    internal whitespace run to a single space (`"Apple  M2   Ultra"` -> `"Apple M2
    Ultra"`)."""
    return " ".join(brand_string.split())


def ram_gib_from_bytes(ram_bytes: int) -> int:
    """Physical RAM in GiB, rounded to the nearest whole GiB -- `mx.device_info()`'s
    `memory_size` is exact powers of two on Apple Silicon (32 GiB -> exactly 32)."""
    return round(ram_bytes / 1024**3)


def machine_slug(*, chip: str, ram_gib: int) -> str:
    """Filesystem-safe machine identifier carrying the RAM class, e.g.
    `apple-m1-max-32gb` -- the stem of the submitted artifact filename."""
    return f"{chip.lower().replace(' ', '-')}-{ram_gib}gb"


def artifact_filename(*, chip: str, ram_gib: int, date: str) -> str:
    """`<chip>-<ram>gb-<yyyy-mm-dd>.json`."""
    return f"{machine_slug(chip=chip, ram_gib=ram_gib)}-{date}.json"


def _read_chip() -> str:
    """The `sysctl` brand-string reader -- unlike `_read_memory_pressure`/
    `_read_on_ac_power` (which degrade gracefully on failure; a stale-but-safe default is
    fine there), a chip read failure has no honest default, so `check=True` raises. A
    subprocess/OS failure (missing binary, nonzero exit, timeout) is mapped to the typed
    `MachineDetectionError` here, not left to escape as a raw traceback -- `main` only
    catches `MlxTrainPerfError`, so an unmapped `CalledProcessError` would exit 1 (an
    uncaught crash) instead of this package's tool-error exit 2."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise MachineDetectionError(
            f"failed to read the CPU brand string via `sysctl`: {exc}"
        ) from exc
    return parse_chip(out)


def _read_ram_bytes() -> int:  # pragma: no cover -- Metal device query boundary
    import mlx.core as mx  # noqa: PLC0415

    return int(mx.device_info()["memory_size"])


def _read_macos() -> str:
    return platform.mac_ver()[0]


def _read_package_version() -> str:
    return version("mlx-train-perf")


def detect_machine(
    *,
    chip_reader: Callable[[], str] = _read_chip,
    ram_bytes_reader: Callable[[], int] = _read_ram_bytes,
    macos_reader: Callable[[], str] = _read_macos,
    mlx_version_reader: Callable[[], str] = _installed_mlx_version,
    package_version_reader: Callable[[], str] = _read_package_version,
) -> MachineInfo:
    """Assemble a `MachineInfo` from injectable readers -- the real ones read `sysctl`,
    `mx.device_info()`, `platform.mac_ver()`, and installed package versions; tests inject
    fakes so this is fully GPU-free and subprocess-free under test."""
    ram_bytes = ram_bytes_reader()
    return MachineInfo(
        chip=parse_chip(chip_reader()),
        ram_gib=ram_gib_from_bytes(ram_bytes),
        ram_bytes=ram_bytes,
        macos=macos_reader(),
        mlx_version=mlx_version_reader(),
        package_version=package_version_reader(),
    )


# --- RAM -> shape scaling table (pure) ------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeGrid:
    ram_class_gib: int
    loss_n: tuple[int, ...]
    loss_d: int
    loss_v: int
    attn_seq: tuple[int, ...]
    attn_head_dim: int
    attn_heads: int
    attn_kv_heads: int
    train_model: str
    train_seq: tuple[int, ...]
    context_start: int
    context_granularity: int


# Machine classes aligned with the 0021 device-proportional wired-cap ladder. 32 GiB is
# the calibration anchor -- this project's own measurement campaign (loss-layer at the
# flagship shape; single-op attention at 2048/4096/8192; train-step at 2048+8192), so its
# grid is byte-for-byte the flagship reference. Smaller machines cap the attention seq
# grid below the O(N^2) stock-attention wall; bigger machines scale the seq grid up
# conservatively (the stock arm honestly aborts past its ceiling -- the O(N) flash arm
# carries through, which IS the demonstration). Head geometry (32q/8kv, d128), the
# loss-layer shape, and the single flagship model are held constant across classes so the
# proofs stay comparable across hardware; only the seq grids and the context-probe start
# scale with RAM.
_FLAGSHIP_MODEL = "mlx-community/Qwen3-8B-4bit"
_LOSS_N = (512, 2048, 8192)
_LOSS_D = 4096
_LOSS_V = 151936
_HEAD_DIM = 128
_HEADS = 32
_KV_HEADS = 8
_GRANULARITY = 256

_CLASSES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)

# Per-class (attn_seq, train_seq, context_start). Everything else is constant.
_CLASS_TABLE: dict[int, tuple[tuple[int, ...], tuple[int, ...], int]] = {
    16:   ((2048, 4096),                                (2048,),                       1024),
    32:   ((2048, 4096, 8192),                          (2048, 8192),                  1024),
    64:   ((2048, 4096, 8192, 16384),                   (2048, 8192),                  2048),
    128:  ((2048, 4096, 8192, 16384, 32768),            (2048, 8192, 16384),           4096),
    256:  ((2048, 4096, 8192, 16384, 32768),            (2048, 8192, 16384),           4096),
    512:  ((2048, 4096, 8192, 16384, 32768, 65536),     (2048, 8192, 16384, 32768),    8192),
    1024: ((2048, 4096, 8192, 16384, 32768, 65536),     (2048, 8192, 16384, 32768),    8192),
}


def ram_class_for(ram_gib: int) -> int:
    """Snap a measured RAM size DOWN to the nearest defined machine class (floor 16 GiB,
    cap 1024 GiB) -- a 48 GiB machine runs the 32 GiB grid, a 2 TB machine the 1024 grid."""
    eligible = [c for c in _CLASSES if c <= ram_gib]
    return eligible[-1] if eligible else _CLASSES[0]


def shapes_for_ram(ram_gib: int) -> ShapeGrid:
    ram_class = ram_class_for(ram_gib)
    attn_seq, train_seq, context_start = _CLASS_TABLE[ram_class]
    return ShapeGrid(
        ram_class_gib=ram_class,
        loss_n=_LOSS_N, loss_d=_LOSS_D, loss_v=_LOSS_V,
        attn_seq=attn_seq, attn_head_dim=_HEAD_DIM, attn_heads=_HEADS, attn_kv_heads=_KV_HEADS,
        train_model=_FLAGSHIP_MODEL, train_seq=train_seq,
        context_start=context_start, context_granularity=_GRANULARITY,
    )


# --- tiers + ETA (pure) ---------------------------------------------------------------


TIERS: dict[str, tuple[str, ...]] = {
    "quick": ("loss_layer", "attention_op"),
    "full": ("loss_layer", "attention_op", "train_step", "context_probe"),
}

# Honest per-bench wall estimate ranges (minutes). Summed per tier: quick = (10, 15) min
# (the brief's ~10-15 min), full = (70, 135) min (~1-2 h). These are deliberately wide,
# machine-independent ranges -- a coarse expectation, not a per-shape prediction.
_BENCH_ETA_MIN: dict[str, tuple[float, float]] = {
    "loss_layer": (5.0, 8.0),
    "attention_op": (5.0, 7.0),
    "train_step": (20.0, 40.0),
    "context_probe": (40.0, 80.0),
}


def bench_eta_label(bench: str) -> str:
    """The per-bench ETA range as a short `"~N-M min"` label (or `"~N min"` when the
    range collapses to a point) -- the kit-level progress line's "before" text quotes
    this so a contributor sees WHY a bench is expected to take a while, not just its
    name."""
    low, high = _BENCH_ETA_MIN[bench]
    if low == high:
        return f"~{low:.0f} min"
    return f"~{low:.0f}-{high:.0f} min"


def benches_for_tier(tier: str) -> tuple[str, ...]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {tuple(TIERS)}")
    return TIERS[tier]


def eta_minutes_for_tier(tier: str) -> tuple[float, float]:
    """Sum the per-bench ETA ranges for the tier's benches (`benches_for_tier` validates
    the tier)."""
    low = sum(_BENCH_ETA_MIN[b][0] for b in benches_for_tier(tier))
    high = sum(_BENCH_ETA_MIN[b][1] for b in benches_for_tier(tier))
    return (low, high)


def format_eta(tier: str) -> str:
    low, high = eta_minutes_for_tier(tier)
    if high >= 90.0:
        return (
            f"{tier} tier: ~{low:.0f}-{high:.0f} min "
            f"(~{low / 60:.1f}-{high / 60:.1f} h)"
        )
    return f"{tier} tier: ~{low:.0f}-{high:.0f} min"


# --- pre-flight (pure decision) -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Preflight:
    ok: bool
    refusal: str | None
    warnings: tuple[str, ...]


_RED_FREE_PCT = 10.0
_WARN_FREE_PCT = 25.0


def classify_memory_pressure(text: str) -> str:
    """Classify `memory_pressure`'s output as `"normal"`/`"warn"`/`"red"` off its
    `System-wide memory free percentage: N%` line. A missing/unparseable line degrades to
    `"normal"` -- the real panic guard is the effective-ceiling refusal (the two-term
    watchdog each bench installs), not this coarse gate, so a parse hiccup must never
    falsely block a healthy machine."""
    import re  # noqa: PLC0415

    match = re.search(r"free percentage:\s*([\d.]+)\s*%", text)
    if match is None:
        return "normal"
    free_pct = float(match.group(1))
    if free_pct < _RED_FREE_PCT:
        return "red"
    if free_pct < _WARN_FREE_PCT:
        return "warn"
    return "normal"


def evaluate_preflight(
    *, memory_pressure_state: str, on_ac_power: bool, ceiling: EffectiveCeiling,
) -> Preflight:
    """Pure pre-flight decision. REFUSES only on a red memory-pressure state (a
    genuinely-crowded machine also refuses upstream via `effective_memory_ceiling` raising
    `MemoryBudgetError`, handled in `run_contribution`). Everything else proceeds with
    WARNINGS: running on battery (measurements drift under power throttling), an elevated
    memory-pressure state, and -- surfaced prominently, this is the kit's audience -- the
    divergence warning (`expected ~N GB free, measured M GB`) the effective ceiling
    recorded."""
    warnings: list[str] = []
    refusal: str | None = None
    if memory_pressure_state == "red":
        refusal = (
            "system memory pressure is critical (red); refusing to start a heavy GPU run "
            "-- close other applications and retry"
        )
    if not on_ac_power:
        warnings.append(
            "running on battery power -- plug in AC power for stable measurements "
            "(power throttling on battery distorts wall-clock timing)"
        )
    if ceiling.warning is not None:
        warnings.append(ceiling.warning)
    if memory_pressure_state == "warn":
        warnings.append(
            "system memory pressure is elevated -- other processes are using memory; "
            "measurements may be affected"
        )
    return Preflight(ok=refusal is None, refusal=refusal, warnings=tuple(warnings))


# --- community artifact assembly (pure) -----------------------------------------------


def summarize_artifact_file(path: Path) -> dict[str, object]:
    """One bench condition's community summary: `{name, status, identity, result}`. The
    `identity` block is embedded VERBATIM (provenance the maintainer trusts); `result` is
    every other top-level artifact field (the measured numbers), with `identity`/`status`
    stripped so they are not duplicated. A missing/corrupt artifact reads as an `error`
    status with an empty identity/result -- the same conservative direction the bench
    runner takes for a crashed condition."""
    name = path.stem
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"name": name, "status": "error", "identity": {}, "result": {}}
    identity = data.get("identity", {})
    result = {k: v for k, v in data.items() if k not in ("identity", "status")}
    return {
        "name": name,
        "status": data.get("status", "error"),
        "identity": identity,
        "result": result,
    }


def summarize_bench(bench: str, paths: list[Path]) -> dict[str, object]:
    return {"bench": bench, "conditions": [summarize_artifact_file(p) for p in paths]}


def _bench_status_label(summary: dict[str, object]) -> str:
    """Short completion status for the kit-level progress line's "after" text -- `"N/M
    ok"` over the conditions this bench produced. An
    empty-condition bench reads as `"0/0 ok"`, not an error -- some benches legitimately
    produce zero conditions on a degenerate grid."""
    conditions = cast(list[dict[str, object]], summary["conditions"])
    total = len(conditions)
    ok = sum(1 for c in conditions if c.get("status") == "ok")
    return f"{ok}/{total} ok"


def collect_memory_warnings(bench_summaries: list[dict[str, object]]) -> list[str]:
    """Every distinct `memory_warning` any underlying bench artifact recorded (divergence
    / degraded-vm_stat start), in first-seen order -- the provenance the maintainer needs
    to weight a crowded-machine submission."""
    seen: list[str] = []
    for bench in bench_summaries:
        for condition in cast(list[dict[str, object]], bench["conditions"]):
            result = cast(dict[str, object], condition.get("result", {}))
            warning = result.get("memory_warning")
            if isinstance(warning, str) and warning not in seen:
                seen.append(warning)
    return seen


def build_community_artifact(
    *,
    machine: MachineInfo,
    tier: str,
    grid: ShapeGrid,
    bench_summaries: list[dict[str, object]],
    generated_date: str,
) -> dict[str, object]:
    """One provenance-complete community artifact: the machine block, the tier + shape
    grid it ran, every bench's per-condition summaries (identity blocks embedded
    verbatim), and every propagated `memory_warning`."""
    return {
        "schema_version": COMMUNITY_SCHEMA_VERSION,
        "generated_date": generated_date,
        "tier": tier,
        "machine": asdict(machine),
        "shapes": asdict(grid),
        "benches": bench_summaries,
        "memory_warnings": collect_memory_warnings(bench_summaries),
    }


# --- PR text --------------------------------------------------------------------------


def pr_title(machine: MachineInfo) -> str:
    return f"Community benchmark: {machine.chip} {machine.ram_gib} GB"


def pr_body(machine: MachineInfo, *, artifact_filename: str, tier: str) -> str:
    """A pre-filled PR body the contributor pastes -- no number editing. Names the
    submitted artifact and states the honesty convention (community-measured on the
    contributor's own hardware, not extrapolated)."""
    return (
        f"Community-measured benchmark on {machine.chip} ({machine.ram_gib} GB, "
        f"macOS {machine.macos}, mlx {machine.mlx_version}), {tier} tier.\n\n"
        f"This PR adds one provenance-complete artifact, "
        f"`community-benchmarks/{artifact_filename}`, produced by "
        f"`mlx-train-perf contribute`. The numbers are measured on my own hardware and "
        f"are not hand-edited or extrapolated.\n\n"
        f"Steps taken:\n"
        f"1. Forked the repo and installed the package.\n"
        f"2. Ran `mlx-train-perf contribute --tier {tier}`.\n"
        f"3. Committed the generated artifact and opened this PR.\n"
    )


# --- measurement seam (imperative shell; stubbed in unit tests) -----------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ContributionResult:
    refused: bool
    refusal: str | None
    artifact_path: Path | None
    warnings: tuple[str, ...]
    pr_title: str | None
    pr_body: str | None


def _bench_scripts_dir() -> Path:
    """Resolve the directory holding the committed bench scripts the kit shells to
    (`bench_attention_op.py`, `northstar_context_sweep.py`). A fresh `pip install` ships
    them alongside the package (`_bench_scripts/`, via the wheel force-include); a dev
    checkout finds them at the repo-root `scripts/`. `MLX_TRAIN_PERF_SCRIPTS_DIR`
    overrides both. No silent fallback: an unresolvable location is a clear typed error."""
    override = os.environ.get("MLX_TRAIN_PERF_SCRIPTS_DIR")
    if override:
        return Path(override)
    packaged = Path(__file__).resolve().parent / "_bench_scripts"
    if (packaged / "bench_attention_op.py").exists():
        return packaged
    repo = Path(__file__).resolve().parents[2] / "scripts"
    if (repo / "bench_attention_op.py").exists():
        return repo
    raise MissingDependencyError(
        "cannot locate the committed bench scripts (bench_attention_op.py / "
        "northstar_context_sweep.py); run from a source checkout or reinstall the package. "
        "Set MLX_TRAIN_PERF_SCRIPTS_DIR to override."
    )


def _loss_conditions(grid: ShapeGrid) -> list[Condition]:
    """The `bench_loss_layer` grid, in-package: kernel/chunked/naive at each `n`."""
    conditions: list[Condition] = []
    for n in grid.loss_n:
        for impl in _LOSS_IMPLS:
            conditions.append(Condition(
                name=f"loss_layer_n{n}_{impl}", kind="loss_layer",
                params={"n": n, "d": grid.loss_d, "v": grid.loss_v, "dtype": "bfloat16",
                        "impl": impl, "reps": 3, "seed": 0},
            ))
    return conditions


def _train_conditions(grid: ShapeGrid) -> list[Condition]:
    """The `bench_train_step` OURS arm (the library's own path) at each scaled seq, flash
    attention on -- the stock-attention baseline is the maintainer campaign's job."""
    return [
        Condition(
            name=f"train_step_seq{seq}_ours", kind="train_step",
            params={"model": grid.train_model, "revision": None, "seq_len": seq,
                    "batch": 1, "steps": 20, "lora_rank": 8, "lora_layers": -1,
                    "impl": "auto", "stock": False, "learning_rate": 1e-5, "seed": 0,
                    "compute_dtype": "bfloat16", "grad_checkpoint": True},
            attention_impl="flash",
        )
        for seq in grid.train_seq
    ]


_SPAWN_STDERR_TAIL_CHARS = 4000  # same convention as bench.runner._STDERR_TAIL_CHARS


def _spawn_crash_artifact(
    out_dir: Path, *, argv: list[str], proc: subprocess.CompletedProcess[str],
) -> Path:
    """Mirror `bench.runner.run_conditions`'s `WorkerCrashed` envelope for a
    spawned bench SCRIPT (`bench_attention_op.py` / `northstar_context_sweep.py`), not
    just the in-package `Condition` workers. A script that crashes (nonzero exit) before
    writing ANY artifact of its own must not glob-read as a clean, empty bench -- that
    silently records a total measurement loss as a successful zero-condition result.
    Written with the SAME envelope shape (`status="error"`, `error_type`, `error_msg` =
    stderr tail, `returncode`) via `bench.artifacts.write_result` so it renders through
    `summarize_bench` / the community artifact exactly like any other crashed
    condition."""
    identity = run_identity(argv=argv)
    stderr_tail = (proc.stderr or proc.stdout or "")[-_SPAWN_STDERR_TAIL_CHARS:]
    path = out_dir / "_spawn_crashed.json"
    write_result(
        path, identity, "error", error_type="WorkerCrashed",
        error_msg=stderr_tail, returncode=proc.returncode,
    )
    return path


def _spawn_script(argv: list[str], *, out_dir: Path) -> list[Path]:
    """Run a committed bench script as a subprocess and return the artifacts it wrote.
    The script owns its own subprocess-per-condition + resume + watchdog machinery; the
    kit just points it at `out_dir` and collects the resulting per-condition artifacts.
    Output is captured (not streamed) -- the same convention `bench.runner._spawn_worker`
    already uses for in-package workers -- so a crash's stderr is available for
    `_spawn_crash_artifact`. A nonzero exit that left an artifact behind (the
    script wrote its own honest partial record, e.g. a condition-level watchdog breach,
    before dying) is respected as-is, never overwritten by a synthetic crash record --
    same reasoning as `run_conditions`'s own crash-envelope path.

    `_spawn_crash_artifact` writes a FIXED filename
    (`_spawn_crashed.json`), and this kit's session id is now deterministic,
    so the SAME `out_dir` is reused across invocations for the same recipe. A stale
    marker from an earlier crash must not glob-read as part of THIS call's result
    forever -- unlink it before respawning, mirroring `bench.runner.run_conditions`'s
    remove-stale-artifact-before-respawn reasoning (runner.py:69-74). This is safe for
    the honest-crash case too: if THIS call also crashes with no artifact, the unlink
    above already ran before the subprocess started, so `_spawn_crash_artifact` below
    writes a fresh marker that is never touched by it again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_spawn_crashed.json").unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, check=False,
    )
    paths = sorted(out_dir.glob("*.json"))
    if proc.returncode != 0 and not paths:
        return [_spawn_crash_artifact(out_dir, argv=argv, proc=proc)]
    return paths


def _spawn_attention(grid: ShapeGrid, *, out_dir: Path, session_id: str) -> list[Path]:
    script = _bench_scripts_dir() / "bench_attention_op.py"
    argv = [
        str(script), "--impl", *_ATTENTION_IMPLS,
        "--seq-lens", *[str(n) for n in grid.attn_seq],
        "--head-dim", str(grid.attn_head_dim), "--heads", str(grid.attn_heads),
        "--kv-heads", str(grid.attn_kv_heads),
        "--out-dir", str(out_dir), "--session-id", session_id,
    ]
    return _spawn_script(argv, out_dir=out_dir)


def _spawn_context(grid: ShapeGrid, *, out_dir: Path) -> list[Path]:
    script = _bench_scripts_dir() / "northstar_context_sweep.py"
    argv = [
        str(script), "--arm", "ours", "--model", grid.train_model,
        "--start-seq-len", str(grid.context_start),
        "--granularity", str(grid.context_granularity), "--out", str(out_dir),
    ]
    return _spawn_script(argv, out_dir=out_dir)


def _measure_bench(
    bench: str, *, grid: ShapeGrid, out_dir: Path, session_id: str, machine: MachineInfo,
) -> list[Path]:
    """The single measurement seam (stubbed wholesale in unit tests). `loss_layer` and
    `train_step` are in-package worker condition kinds driven through `run_conditions`
    (subprocess-per-condition, resumable). `attention_op` and `context_probe` have their
    own script-level orchestration and run as subprocesses of the committed scripts.
    `machine` is unused here (the caller records it in the artifact) but kept in the seam
    signature so a future bench could scale by it."""
    _ = machine
    out_dir.mkdir(parents=True, exist_ok=True)
    if bench == "loss_layer":
        return run_conditions(_loss_conditions(grid), out_dir, session_id=session_id)
    if bench == "train_step":
        return run_conditions(_train_conditions(grid), out_dir, session_id=session_id)
    if bench == "attention_op":
        return _spawn_attention(grid, out_dir=out_dir, session_id=session_id)
    if bench == "context_probe":
        return _spawn_context(grid, out_dir=out_dir)
    raise BenchInputError(f"unknown bench {bench!r}")


def _read_memory_pressure() -> str:
    try:
        return subprocess.run(
            ["memory_pressure"], capture_output=True, text=True, check=False, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_on_ac_power() -> bool:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True  # can't tell -> don't nag; the memory watchdog is the real guard
    return "AC Power" in out


def _today() -> str:
    return date.today().isoformat()


def run_preflight(
    *,
    ceiling_reader: Callable[[], EffectiveCeiling] = effective_memory_ceiling,
    memory_pressure_reader: Callable[[], str] = _read_memory_pressure,
    ac_power_reader: Callable[[], bool] = _read_on_ac_power,
) -> Preflight:
    """Run the pre-flight checks (the memory ceiling/divergence, `memory_pressure`,
    AC power) and return the `Preflight` decision, WITHOUT taking any confirmation input
    -- this is the standalone seam the CLI calls BEFORE prompting for confirmation.
    The kit's README promises the pre-flight warning names the
    expected-vs-measured memory gap "so you can close other apps and retry"; that promise
    only holds if the warning is on screen before any heavy work starts, which requires it
    to run (and be printed) ahead of the confirmation prompt, not folded into the same
    call that also drives the measurement.

    `effective_memory_ceiling` may itself REFUSE (typed `MemoryBudgetError`) on a
    genuinely-crowded machine; that becomes a clean `Preflight(ok=False, ...)` here rather
    than a raised exception, so every caller gets the same clean-refusal shape without its
    own try/except."""
    try:
        ceiling = ceiling_reader()
    except MemoryBudgetError as exc:
        return Preflight(ok=False, refusal=str(exc), warnings=())
    mp_state = classify_memory_pressure(memory_pressure_reader())
    return evaluate_preflight(
        memory_pressure_state=mp_state, on_ac_power=ac_power_reader(), ceiling=ceiling,
    )


_MODULE_PATH = Path(__file__).resolve()


def _module_sha() -> str:
    """Fingerprint of THIS module's own bytes -- same reasoning as
    `scripts/northstar_context_sweep.py::script_sha()`: an edit to the condition-building
    logic in this file must invalidate any session id derived below, so a stale,
    incompatible session is never silently resumed after a code change."""
    return hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()[:16]


def _contribute_session_id(*, machine: MachineInfo, tier: str, grid: ShapeGrid) -> str:
    """Deterministic (NOT random) session id for one `mlx-train-perf contribute` run --
    mirrors `scripts/northstar_context_sweep.py::_recipe_session_id`'s reasoning (finding
    D). The `loss_layer`/`attention_op`/`train_step` conditions this session id feeds are
    all dispatched through machinery that resume-skips a condition whose identity --
    `session_id` included -- already has a FRESH artifact on disk (`bench.runner.
    run_conditions`, and `bench_attention_op.py`'s own `--session-id`-driven grid). A
    fresh `uuid4` session id on every invocation (the prior behavior) defeats that resume
    property ACROSS separate `contribute` runs: an interrupted quick/full-tier kit run,
    re-invoked, would re-measure every condition from scratch even though most of them
    already finished.

    Folding in the machine identity, the tier, the FULL shape grid (not just the RAM
    class int), and this module's own `_module_sha()` keeps the id sensitive to anything
    that changes what actually gets measured: a different machine, a wider seq grid, or
    an edit to the condition-building logic in this file all hash to a DIFFERENT id, so a
    stale/incompatible session is never silently resumed. `context_probe` is excluded on
    purpose -- `northstar_context_sweep.py` already derives its own stable id via
    `_recipe_session_id` and does not accept one from the caller."""
    h = hashlib.sha256()
    for part in (
        machine.chip, str(machine.ram_gib), tier,
        json.dumps(asdict(grid), sort_keys=True), _module_sha(),
    ):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


def run_contribution(
    *,
    tier: str,
    out_dir: Path,
    confirm: bool,
    machine: MachineInfo | None = None,
    preflight: Preflight | None = None,
    ceiling_reader: Callable[[], EffectiveCeiling] = effective_memory_ceiling,
    memory_pressure_reader: Callable[[], str] = _read_memory_pressure,
    ac_power_reader: Callable[[], bool] = _read_on_ac_power,
    measure: Callable[..., list[Path]] = _measure_bench,
    today: Callable[[], str] = _today,
) -> ContributionResult:
    """Run the contribution kit end to end and return a `ContributionResult`. Order of
    operations is safety-first: validate the tier, resolve the shape grid, require explicit
    confirmation, run the pre-flight (a red memory state or a too-crowded
    `MemoryBudgetError` REFUSES before any measurement starts), then drive each bench
    through the injectable `measure` seam, and finally write ONE provenance-complete
    community artifact plus the pre-filled PR text.

    `preflight`, when given, is used AS-IS instead of being recomputed: the
    CLI runs `run_preflight` itself BEFORE the confirmation prompt (so its warnings print
    ahead of it, and a refusal is shown before ever asking to proceed) and passes the
    result through here rather than paying for the readers a second time. A direct
    caller that omits it (every test in this suite, and any other direct caller) gets the
    SAME decision computed fresh via the injected readers -- this function's own
    stand-alone refusal semantics are unchanged either way.

    All hardware/subprocess touchpoints are injectable (`ceiling_reader`,
    `memory_pressure_reader`, `ac_power_reader`, `measure`, `today`) so the whole flow is
    unit-tested GPU-free. The ETA print + interactive confirmation live in the CLI (before
    this call); `confirm=False` here is a belt-and-suspenders refusal."""
    benches = benches_for_tier(tier)  # validates the tier up front
    if machine is None:
        machine = detect_machine()
    grid = shapes_for_ram(machine.ram_gib)

    if not confirm:
        return ContributionResult(
            refused=True,
            refusal="not confirmed -- pass --yes or confirm at the prompt to start",
            artifact_path=None, warnings=(), pr_title=None, pr_body=None,
        )

    if preflight is None:
        preflight = run_preflight(
            ceiling_reader=ceiling_reader, memory_pressure_reader=memory_pressure_reader,
            ac_power_reader=ac_power_reader,
        )
    if not preflight.ok:
        return ContributionResult(
            refused=True, refusal=preflight.refusal, artifact_path=None,
            warnings=preflight.warnings, pr_title=None, pr_body=None,
        )

    # Measure. Per-bench artifacts live under out_dir/_work/<bench> (resumable across kit
    # runs, intermediate); the single community artifact is written at out_dir/<name>.json.
    # `session_id` is DETERMINISTIC (finding D, `_contribute_session_id`), not a fresh
    # uuid4 -- a `contribute` run interrupted and re-invoked with the SAME (machine, tier,
    # grid) resumes by skipping already-fresh conditions instead of re-measuring
    # everything.
    session_id = _contribute_session_id(machine=machine, tier=tier, grid=grid)
    work_root = out_dir / "_work"
    summaries: list[dict[str, object]] = []
    total_benches = len(benches)
    # Review round 2, finding 2: between the confirmation prompt and the final "wrote
    # ..." line, a full-tier run (~1-2 h) previously printed NOTHING -- subprocess output
    # is captured, not streamed (the honest-crash mechanism at `_spawn_script` needs the
    # capture), so silence read as a hang. These kit-level lines are the liveness signal:
    # printed by THIS function around each `measure` call, independent of whatever the
    # (captured) subprocess does or doesn't print itself. Flushed explicitly since stdout
    # may be line-buffered-off when not attached to a TTY (e.g. piped to a log file).
    for i, bench in enumerate(benches, start=1):
        bench_dir = work_root / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{total_benches}] {bench} -- running ({bench_eta_label(bench)})...")
        sys.stdout.flush()
        started = time.monotonic()
        paths = measure(bench, grid=grid, out_dir=bench_dir, session_id=session_id,
                        machine=machine)
        elapsed_s = time.monotonic() - started
        summary = summarize_bench(bench, paths)
        summaries.append(summary)
        print(
            f"[{i}/{total_benches}] {bench} -- done in {elapsed_s:.0f}s "
            f"({_bench_status_label(summary)})"
        )
        sys.stdout.flush()

    generated_date = today()
    artifact = build_community_artifact(
        machine=machine, tier=tier, grid=grid, bench_summaries=summaries,
        generated_date=generated_date,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = artifact_filename(chip=machine.chip, ram_gib=machine.ram_gib, date=generated_date)
    artifact_path = out_dir / fname
    artifact_path.write_text(json.dumps(artifact, indent=2))
    return ContributionResult(
        refused=False, refusal=None, artifact_path=artifact_path,
        warnings=preflight.warnings,
        pr_title=pr_title(machine),
        pr_body=pr_body(machine, artifact_filename=fname, tier=tier),
    )
