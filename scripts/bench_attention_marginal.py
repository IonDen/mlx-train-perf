"""Attention marginal-peak bench: stock `mx.fast.scaled_dot_product_attention` against
`mlx_train_perf.attention.flash_attention`, forward plus backward, one op in isolation.

Model-free: synthetic bf16 tensors at a pinned shape, B=1, Hq=32, Hkv=8 (GQA), d=128,
seed 0. Loss is `out.sum()`; gradients are taken with respect to q, k and v
(`mx.grad(loss, argnums=(0, 1, 2))`) and all three are evaluated.

Modes (`--mode`, one or both):

  causal  -- causal attention. stock: SDPA with `mask="causal"`; flash:
             `flash_attention(causal=True, impl="kernel")`. Default N: 1024 2048 4096 8192.
  packed  -- block-diagonal causal attention over a pinned, seeded segment layout (mixed
             segment lengths from 256 to 2048 tokens that fill N exactly; the layout is in
             every record). stock: SDPA with a materialized boolean (1, 1, N, N) mask
             (True = attend), the only block-diagonal form SDPA takes; flash:
             `flash_attention(segments=PackedMask(...))`, which reads O(N) per-token segment
             ids instead. Default N: 4096 8192.

Metric, per measured call: `mx.synchronize()`, `mx.clear_cache()`, `pre = active memory`,
`mx.reset_peak_memory()`, one forward+backward with `mx.eval`, `mx.synchronize()`, then
marginal = `mx.get_peak_memory() - pre`. One warm-up call (Metal JIT, and flash's
construction-time rate calibration) precedes at least three measured calls; every call's
gradients stay alive until the unit ends, so a deferred release can never land inside a
later call's `pre` snapshot. The record keeps every call's raw `pre`/`peak`/marginal and
wall time; the reported value is the median. The packed stock mask and the packed flash
segment buffers are inputs allocated before the window: they sit in `pre`, outside the
marginal, and their own bytes are recorded separately. Wall time is recorded, not
headlined. Every gradient of every call is checked for finiteness after the last window; a
non-finite gradient makes the unit `nonfinite`, never `ok`. A flash record also carries
`dispatch_plan`: the three rates the flash path calibrated and cached in this process
(forward, dQ, dK/dV), the dK/dV tile, and how many dK/dV dispatch ranges the planner makes
for the unit's shape at that rate, read from the kernel module's caches after the
measured calls (`null` where the process cached no rate).

Isolation and bounds: one fresh worker process per (mode, impl, N), in the parent's
process group so an outer supervisor (e.g. `mlx-guard run`) accounts for it. The worker
pins the wired limit strictly below the device's recommended working set, sets a 1 GiB
cache limit, and runs a watchdog sampling active + cache memory every 50 ms against
`min(--watchdog-gib, device memory - 4 GiB)` (`--watchdog-gib` default 22; both the
requested and the effective ceiling are recorded) plus a wall budget; a breach writes an
honest `aborted_watchdog` / `aborted_wall` record and exits 3. The parent stops a worker
that outlives `--unit-wall-s` (default 300 s) and records `aborted_wall` itself. Aborts
are results; a worker that dies without a record gets an `error` record.

Output: `<out-dir>/mlx-<mlx version>/<mode>-<impl>-n<N>.json` per unit, plus
`results.json` (the table, the stock/flash marginal ratios per (mode, N), and machine
context) rebuilt from the unit records in that directory on each run; a record from
another source tree or bypass setting than this run's is listed under `invalid_units`,
never mixed into the table. Re-running skips every unit whose record is ok or an abort
written by this exact script version under the same mlx, source tree
(`mlx_train_perf_source`), bypass setting and `--reps`; any other record (`error`,
`nonfinite`, another script version or run setting) is moved to `attempts/` (never
deleted) and the unit runs again. Default `<out-dir>` is
`_artifacts/bench_attention_marginal` in the repository.

Records and logs hold no machine-local path. In every string written (records,
results.json, and each worker log, which is rewritten when the worker ends, however it
ends, and before it is moved to `attempts/`): a path inside the repository is written
relative to it (the root itself as `.`); a path under the home directory as `~/...`; and
any other absolute path as `<path>`: a `/` not preceded by a letter, digit, `_`, `.`,
`~`, `:`, `/` or `-` (so relative paths, `~/...`, URLs and ratios such as `tok/s` stay),
up to the next whitespace, quote, backtick, bracket, `,`, `;`, `:`, `=` or `|`. A
symlinked path counts by its spelling, not its target. The recorded interpreter is
repository-relative when inside the repository, else `python`.

Runs on the repository environment, or on any environment with `mlx` and
`mlx-train-perf>=0.9.0` (stdlib, `mlx` and `mlx_train_perf` imports only), e.g.

    python scripts/bench_attention_marginal.py --mode causal
    python scripts/bench_attention_marginal.py --mode packed
    uv run --isolated --no-project --python 3.12 --with "mlx-train-perf==0.9.0" \\
        --with "mlx==0.32.0" python -I scripts/bench_attention_marginal.py --mode causal

`--src <path>` imports `mlx_train_perf` from a source tree instead (it goes first on
sys.path in the parent and in every worker, which `python -I` would otherwise not allow),
so an environment holding only `mlx` can measure the repository code; the records carry
`mlx_train_perf_source` = `<repo-relative path>@<sha of the tree's .py files>`, and
`mlx_train_perf_version` is null (no installed distribution was measured).

`--bypass-verification-gate` is a measurement aid, not a user option. It lets the flash arm
run on an mlx that `mlx_train_perf._compat.VERIFIED_MLX_VERSIONS` does not list (each worker
adds the installed version to that set), so a not-yet-verified mlx can be measured before
the verification ritual. It is off by default, refused when the installed mlx is already
verified, recorded as `verification_gate_bypassed` in every unit record and in
results.json, and named in the table header. It says nothing about kernel correctness on
that mlx.

Exit: 0 = every unit has an ok or abort record; 1 = at least one unit ended otherwise
(`error`, `nonfinite`). Worker exits: 0 ok, 3 watchdog/wall abort (record written), 2 wrong
mlx (no record), 1 non-finite gradients (record written) or any other failure (the parent
records it).
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, cast

SCHEMA = "bench_attention_marginal/1"
GIB = 1024**3
B, HQ, HKV, HEAD_DIM = 1, 32, 8, 128
SEED = 0
LAYOUT_SEED = 0
SEG_MIN, SEG_MAX = 256, 2048
MODES = ("causal", "packed")
IMPLS = ("flash", "stock")
DEFAULT_SEQ_LENS: dict[str, tuple[int, ...]] = {
    "causal": (1024, 2048, 4096, 8192),
    "packed": (4096, 8192),
}
MIN_REPS = 3
DONE_STATUSES = ("ok", "aborted_watchdog", "aborted_wall")
NONFINITE = "nonfinite"
WIRED_CAP_BYTES = 20 * GIB
WIRED_MARGIN_BYTES = 2 * GIB
CACHE_LIMIT_BYTES = 1 * GIB
WATCHDOG_INTERVAL_S = 0.05
DEFAULT_WATCHDOG_GIB = 22.0
WATCHDOG_DEVICE_MARGIN_BYTES = 4 * GIB  # the ceiling never exceeds memory_size - this
DEFAULT_UNIT_WALL_S = 300.0
WORKER_WALL_MARGIN_S = 30.0
STOP_GRACE_S = 15.0
EXIT_ABORT = 3
EXIT_WRONG_MLX = 2
STDERR_TAIL_CHARS = 4000
SDPA_VJP_FALLBACK_ENV = "MLX_SDPA_VJP_FALLBACK"

_SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = _SCRIPT_PATH.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "_artifacts" / "bench_attention_marginal"
_REQUIRED_FIELDS = ("schema", "unit", "mode", "impl", "n", "status", "mlx_version",
                    "script_sha")
_REP_FIELDS = ("pre_bytes", "peak_bytes", "marginal_bytes", "wall_s")


# ----------------------------------------------------------------- pure core


@dataclass(frozen=True, slots=True, kw_only=True)
class Unit:
    mode: str
    impl: str
    n: int

    @property
    def name(self) -> str:
        return unit_name(self.mode, self.impl, self.n)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerOutcome:
    returncode: int
    timed_out: bool
    duration_s: float
    stderr_tail: str


Spawner = Callable[[Unit, Path, Path], WorkerOutcome]


def unit_name(mode: str, impl: str, n: int) -> str:
    return f"{mode}-{impl}-n{n}"


def build_units(*, modes: Sequence[str], impls: Sequence[str],
                seq_lens: Sequence[int]) -> list[Unit]:
    """Mode by mode, N ascending, flash before stock: the heaviest unit runs last."""
    ordered_impls = [i for i in IMPLS if i in impls]
    return [Unit(mode=m, impl=i, n=n)
            for m in modes for n in sorted(set(seq_lens)) for i in ordered_impls]


def marginal_bytes(*, pre: int, peak: int) -> int:
    return peak - pre


def summarize_reps(reps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reps:
        raise ValueError("no reps to summarize")
    marginals = [r["marginal_bytes"] for r in reps]
    return {
        "n_reps": len(reps),
        "marginal_median_bytes": statistics.median(marginals),
        "marginal_min_bytes": min(marginals),
        "marginal_max_bytes": max(marginals),
        "wall_median_s": statistics.median(r["wall_s"] for r in reps),
    }


def wired_limit_for(max_recommended: int) -> int:
    """Strictly below the device recommendation: 20 GiB, or 2 GiB under the
    recommendation on a smaller machine, or half of a very small one."""
    if max_recommended > 2 * WIRED_MARGIN_BYTES:
        return min(WIRED_CAP_BYTES, max_recommended - WIRED_MARGIN_BYTES)
    return max_recommended // 2


def watchdog_ceiling_bytes(*, requested_gib: float, memory_size: int) -> int:
    """The worker's active + cache ceiling: `--watchdog-gib`, lowered to 4 GiB under the
    device's memory on a machine where the request would let the unit page first."""
    return min(int(requested_gib * GIB), memory_size - WATCHDOG_DEVICE_MARGIN_BYTES)


def watchdog_fields(*, requested_gib: float, memory_size: int,
                    wall_budget_s: float) -> dict[str, Any]:
    """The `watchdog` block of a worker record: the requested and the effective ceiling."""
    return {
        "requested_limit_bytes": int(requested_gib * GIB),
        "limit_bytes": watchdog_ceiling_bytes(requested_gib=requested_gib,
                                              memory_size=memory_size),
        "device_margin_bytes": WATCHDOG_DEVICE_MARGIN_BYTES,
        "interval_s": WATCHDOG_INTERVAL_S, "samples": "active + cache",
        "wall_budget_s": wall_budget_s,
    }


def measured_status(*, grads_finite: bool) -> str:
    """A measured unit is `ok` only when every gradient of every call was finite."""
    return "ok" if grads_finite else NONFINITE


def segment_lengths(n: int, *, seed: int, lo: int = SEG_MIN, hi: int = SEG_MAX) -> list[int]:
    """Seeded segment lengths in [lo, hi] that sum to exactly n. A draw never leaves a
    remainder shorter than `lo`; a remainder that fits in one segment and cannot be split
    into two becomes the last segment."""
    if hi < 2 * lo - 1:
        raise ValueError(f"need hi >= 2 * lo - 1 to fill any n >= lo; got lo={lo} hi={hi}")
    if n < lo:
        raise ValueError(f"n={n} is below the shortest segment length {lo}")
    rng = random.Random(seed)
    lengths: list[int] = []
    remaining = n
    while remaining > 0:
        if remaining <= hi and remaining < 2 * lo:
            lengths.append(remaining)
            break
        length = rng.randint(lo, min(hi, remaining - lo))
        lengths.append(length)
        remaining -= length
    return lengths


def segment_ids(lengths: Sequence[int]) -> tuple[list[int], list[int]]:
    """Per-token segment id (0, 1, ...) and the index where each token's segment starts."""
    seg_id: list[int] = []
    seg_start: list[int] = []
    start = 0
    for sid, length in enumerate(lengths):
        seg_id += [sid] * length
        seg_start += [start] * length
        start += length
    return seg_id, seg_start


def packed_mask(lengths: Sequence[int]) -> Any:
    """Boolean (1, 1, N, N) mask, True where key j is visible to query i: same segment
    and j <= i. Built with mlx ops on the current default stream."""
    import mlx.core as mx  # noqa: PLC0415 -- the parent and --help never load mlx

    seg, _ = segment_ids(lengths)
    n = len(seg)
    s = mx.array(seg, dtype=mx.int32)
    same = cast(mx.array, s[:, None] == s[None, :])
    causal = mx.tri(n, n, 0, dtype=mx.bool_)
    return mx.logical_and(same, causal).reshape(1, 1, n, n)


def _root_pattern(root: Path) -> re.Pattern[str]:
    """`root` as a whole path prefix: followed by `/`, or by nothing that could continue
    a path component (so `/home/a` never matches inside `/home/ab`)."""
    return re.compile(re.escape(str(root)) + r"(/|(?![\w.\-]))")


# An absolute path: a `/` that does not continue a word, a relative path, `~`, `.` or a
# URL scheme, running to the next whitespace, quote, bracket or separator.
_ABSOLUTE_PATH = re.compile(r"(?<![\w.~:/\-])/[^\s\"'`()\[\]{}<>,;:=|]+")
PATH_PLACEHOLDER = "<path>"


def scrub_text(text: str, *, repo_root: Path = REPO_ROOT, home: Path | None = None) -> str:
    """Every repository path in `text` made repository-relative (the root itself becomes
    `.`), then every remaining home-directory path written with `~`, then every remaining
    absolute path replaced by `<path>`. Paths match by spelling: a symlink into the
    repository or home is not resolved, so it becomes `<path>`."""
    home = home if home is not None else Path.home()
    text = _root_pattern(repo_root).sub(lambda m: "" if m.group(1) == "/" else ".", text)
    text = _root_pattern(home).sub(lambda m: "~" + m.group(1), text)
    return _ABSOLUTE_PATH.sub(PATH_PLACEHOLDER, text)


def portable_path(path: str, *, repo_root: Path = REPO_ROOT, home: Path | None = None) -> str:
    """A path as recorded: relative to the repository when inside it, `~/...` when under
    the home directory, `<path>` for any other absolute path, a relative path unchanged."""
    return scrub_text(path, repo_root=repo_root, home=home)


def portable_interpreter(path: str, *, repo_root: Path = REPO_ROOT,
                         home: Path | None = None) -> str:
    """The interpreter as recorded: repository-relative (`.venv/bin/python`) when inside
    the repository, `python` for any absolute path outside it (a uv cache or system
    interpreter), a relative invocation unchanged."""
    if not Path(path).is_absolute():
        return path
    if Path(path).is_relative_to(repo_root):
        return portable_path(path, repo_root=repo_root, home=home)
    return "python"


def portable_argv(argv: Sequence[str], *, repo_root: Path = REPO_ROOT,
                  home: Path | None = None) -> list[str]:
    """argv as recorded: the interpreter by `portable_interpreter`, every other argument
    (paths, `--out=...` forms, the nested JSON command line) by `scrub_text`."""
    if not argv:
        return []
    return [portable_interpreter(argv[0], repo_root=repo_root, home=home),
            *(scrub_text(a, repo_root=repo_root, home=home) for a in argv[1:])]


def scrub(obj: Any, *, repo_root: Path = REPO_ROOT, home: Path | None = None) -> Any:
    """`scrub_text` applied to every string in a JSON-shaped object."""
    if isinstance(obj, str):
        return scrub_text(obj, repo_root=repo_root, home=home)
    if isinstance(obj, dict):
        return {k: scrub(v, repo_root=repo_root, home=home) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [scrub(v, repo_root=repo_root, home=home) for v in obj]
    return obj


def provenance(*, orig_argv: Sequence[str], executable: str, command_line: Sequence[str],
               repo_root: Path = REPO_ROOT, home: Path | None = None) -> dict[str, Any]:
    """The invocation fields of a worker record, with no machine-local path in them."""
    return {
        "interpreter": portable_interpreter(executable, repo_root=repo_root, home=home),
        "python_version": platform.python_version(),
        "command_line": portable_argv(command_line, repo_root=repo_root, home=home),
        "worker_command": portable_argv(orig_argv, repo_root=repo_root, home=home),
    }


def bypass_refusal(*, bypass: bool, installed: str, verified: Iterable[str]) -> str | None:
    """Why `--bypass-verification-gate` is refused, or None: it is a measurement aid for an
    mlx the kernel gate does not list, and pointless (so refused) where the gate passes."""
    verified = frozenset(verified)
    if bypass and installed in verified:
        return (f"--bypass-verification-gate is refused: mlx {installed} is already "
                f"verified ({', '.join(sorted(verified))})")
    return None


def apply_verification_bypass(installed: str) -> None:
    """Add `installed` to `mlx_train_perf._compat.VERIFIED_MLX_VERSIONS`, the module
    global `check_mlx_verified` reads on every call, so every flash call site in this
    process passes the gate."""
    from mlx_train_perf import _compat  # noqa: PLC0415

    _compat.VERIFIED_MLX_VERSIONS = _compat.VERIFIED_MLX_VERSIONS | {installed}


def prepare_src(src: Path) -> Path:
    """Resolve a `--src` tree, check it holds `mlx_train_perf`, and put it first on
    sys.path (ahead of any installed copy)."""
    root = src.resolve()
    if not (root / "mlx_train_perf" / "__init__.py").is_file():
        raise ValueError(f"--src {src} has no mlx_train_perf/__init__.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def src_identity(src: Path, *, repo_root: Path = REPO_ROOT, home: Path | None = None) -> str:
    """`<path>@<sha>` for a `--src` tree: the path as `portable_path` writes it, and the
    first 16 hex of a sha256 over every `*.py` file's relative path and bytes."""
    root = src.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return f"{portable_path(str(root), repo_root=repo_root, home=home)}@" \
           f"{digest.hexdigest()[:16]}"


def run_flags(*, bypassed: bool, src_identity: str | None) -> dict[str, Any]:
    """The measurement-mode fields every unit record carries."""
    return {"verification_gate_bypassed": bypassed, "mlx_train_perf_source": src_identity}


def validate_unit_record(record: dict[str, Any]) -> list[str]:
    """Problems with one unit record (empty when it is usable). An ok record must carry
    `grads_finite: true`, at least MIN_REPS raw reps, each with marginal == peak - pre, and
    medians that follow from them; any other record needs only its identity fields."""
    problems = [f"missing field {f}" for f in _REQUIRED_FIELDS if f not in record]
    if problems:
        return problems
    if record["schema"] != SCHEMA:
        problems.append(f"schema {record['schema']!r} is not {SCHEMA!r}")
    if record["status"] != "ok":
        return problems
    if record.get("grads_finite") is not True:
        problems.append("ok record without grads_finite: true (non-finite or unchecked "
                        "gradients)")
    reps = record.get("reps")
    if not isinstance(reps, list) or len(reps) < MIN_REPS:
        count = len(reps) if isinstance(reps, list) else 0
        return [*problems, f"ok record needs >= {MIN_REPS} reps, has {count}"]
    for i, rep in enumerate(reps):
        missing = [f for f in _REP_FIELDS if not isinstance(rep.get(f), int | float)]
        if missing:
            problems.append(f"rep {i} lacks numeric {', '.join(missing)}")
        elif rep["marginal_bytes"] != rep["peak_bytes"] - rep["pre_bytes"]:
            problems.append(f"rep {i} marginal_bytes is not peak - pre")
    if problems:
        return problems
    expected = summarize_reps(reps)
    if record.get("marginal_median_bytes") != expected["marginal_median_bytes"]:
        problems.append("marginal_median_bytes does not match the median of the reps")
    if record.get("wall_median_s") != expected["wall_median_s"]:
        problems.append("wall_median_s does not match the median of the reps")
    return problems


def run_mismatches(record: dict[str, Any], *, source: str | None,
                   bypassed: bool) -> list[str]:
    """How a record's source tree and bypass setting differ from this run's (empty when
    they match; a missing field never matches)."""
    problems: list[str] = []
    if "mlx_train_perf_source" not in record or record["mlx_train_perf_source"] != source:
        problems.append(f"mlx_train_perf_source {record.get('mlx_train_perf_source')!r} is "
                        f"not this run's {source!r}")
    if record.get("verification_gate_bypassed") is not bypassed:
        problems.append(f"verification_gate_bypassed "
                        f"{record.get('verification_gate_bypassed')!r} is not this run's "
                        f"{bypassed!r}")
    return problems


def unit_is_done(record: dict[str, Any] | None, *, script_sha: str, mlx_version: str,
                 source: str | None, bypassed: bool, reps: int) -> bool:
    """A unit is done when its record is an ok or abort result from this script version
    under this mlx, source tree, bypass setting and requested rep count, and is
    schema-valid."""
    if record is None or record.get("status") not in DONE_STATUSES:
        return False
    if record.get("script_sha") != script_sha or record.get("mlx_version") != mlx_version:
        return False
    if run_mismatches(record, source=source, bypassed=bypassed):
        return False
    if record.get("reps_requested") != reps:
        return False
    return validate_unit_record(record) == []


def stock_over_flash(records: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """Stock marginal median over flash marginal median, keyed `<mode>-n<N>`, for every
    (mode, N) with a record; None unless both arms are ok and flash is nonzero."""
    by_key = {(r.get("mode"), r.get("impl"), r.get("n")): r for r in records}
    keys = sorted({(m, n) for (m, _i, n) in by_key}, key=lambda k: (str(k[0]), int(k[1] or 0)))
    ratios: dict[str, float | None] = {}
    for mode, n in keys:
        flash, stock = by_key.get((mode, "flash", n)), by_key.get((mode, "stock", n))
        ratio: float | None = None
        if (flash is not None and stock is not None
                and flash.get("status") == "ok" and stock.get("status") == "ok"
                and flash.get("marginal_median_bytes")):
            ratio = round(stock["marginal_median_bytes"] / flash["marginal_median_bytes"], 3)
        ratios[f"{mode}-n{n}"] = ratio
    return ratios


def _table(records: Sequence[dict[str, Any]], *, bypassed: bool = False) -> str:
    title = "marginal peak, GiB, median of the measured calls (forward + backward)"
    if bypassed:
        title += "; mlx verification gate bypassed (measurement only)"
    lines = [title, "",
             "| mode | impl | N | status | marginal GiB | min..max GiB | median call s |",
             "|---|---|---|---|---|---|---|"]
    for r in records:
        if r["status"] == "ok":
            s = summarize_reps(r["reps"])
            lines.append(
                f"| {r['mode']} | {r['impl']} | {r['n']} | ok "
                f"| {s['marginal_median_bytes'] / GIB:.3f} "
                f"| {s['marginal_min_bytes'] / GIB:.3f}..{s['marginal_max_bytes'] / GIB:.3f} "
                f"| {s['wall_median_s']:.4f} |")
        else:
            lines.append(f"| {r['mode']} | {r['impl']} | {r['n']} | {r['status']} | - | - | - |")
    return "\n".join(lines)


def summarize(records: Iterable[dict[str, Any]], *, source: str | None,
              bypassed: bool) -> dict[str, Any]:
    """results.json content: the schema-valid unit records from this run's source tree and
    bypass setting (sorted), the per-(mode, N) stock/flash ratios over them, a markdown
    table, machine context from the first of them, and every other record's problems under
    `invalid_units` (a set mixing sources or bypass settings is never summarized as one)."""
    valid: list[dict[str, Any]] = []
    invalid: dict[str, list[str]] = {}
    for r in records:
        problems = validate_unit_record(r) or run_mismatches(r, source=source,
                                                             bypassed=bypassed)
        if problems:
            invalid[str(r.get("unit", "?"))] = problems
        else:
            valid.append(r)
    valid.sort(key=lambda r: (MODES.index(r["mode"]) if r["mode"] in MODES else 99, r["n"],
                              IMPLS.index(r["impl"]) if r["impl"] in IMPLS else 99))
    context_keys = ("mlx_version", "mlx_metal_version", "mlx_train_perf_version",
                    "mlx_train_perf_source", "python", "macos", "device")
    context = {k: valid[0].get(k) for k in context_keys} if valid else {}
    return {
        "schema": SCHEMA,
        "context": context,
        "verification_gate_bypassed": bypassed,
        "ratio_stock_over_flash": stock_over_flash(valid),
        "table": _table(valid, bypassed=bypassed),
        "units": valid,
        "invalid_units": invalid,
    }


# ----------------------------------------------------------------- parent (no mlx import)


def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def script_sha() -> str:
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def _dist(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions(*, src_given: bool) -> dict[str, str | None]:
    """The installed distribution versions a worker record names; the mlx-train-perf one
    is null under `--src`, where the measured code is the source tree, not the install."""
    return {"mlx_metal_version": _dist("mlx-metal"),
            "mlx_train_perf_version": None if src_given else _dist("mlx-train-perf")}


def _atomic_write(path: Path, obj: dict[str, Any]) -> None:
    """Every record and results.json goes through here, scrubbed of local paths."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w") as fh:
        json.dump(scrub(obj), fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _move_aside(path: Path, attempts: Path) -> None:
    """Move a record or log to `attempts/` (never delete it); a log is scrubbed first."""
    if path.exists():
        if path.suffix == ".log":
            _scrub_log(path)
        attempts.mkdir(parents=True, exist_ok=True)
        path.replace(attempts / f"{path.stem}.{_stamp()}.{os.getpid()}{path.suffix}")


def _parent_record(unit: Unit, *, status: str, script_sha: str, mlx_version: str,
                   outcome: WorkerOutcome, source: str | None, bypassed: bool,
                   reps: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "unit": unit.name, "mode": unit.mode, "impl": unit.impl,
        "n": unit.n, "status": status, "recorded_by": "parent",
        "mlx_version": mlx_version, "script_sha": script_sha, "reps_requested": reps,
        **run_flags(bypassed=bypassed, src_identity=source),
        "unit_exit": outcome.returncode, "unit_duration_s": round(outcome.duration_s, 3),
        "stderr_tail": outcome.stderr_tail, "finished_at": _utc(),
    }


def run_units(units: Sequence[Unit], *, out_dir: Path, spawn: Spawner, script_sha: str,
              mlx_version: str, source: str | None, bypassed: bool,
              reps: int) -> list[Path]:
    """Run every unit not already done (see `unit_is_done`), one worker each. The handoff
    is the record file: an earlier record is moved to `attempts/` before the worker starts,
    so a record at the path afterwards is this worker's. No record -> `aborted_wall` when
    the parent stopped the worker at the wall limit, else `error` with the log tail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts, logs = out_dir / "attempts", out_dir / "logs"
    logs.mkdir(exist_ok=True)
    def parent_record(unit: Unit, status: str, outcome: WorkerOutcome) -> dict[str, Any]:
        return _parent_record(unit, status=status, script_sha=script_sha,
                              mlx_version=mlx_version, outcome=outcome, source=source,
                              bypassed=bypassed, reps=reps)

    paths: list[Path] = []
    for unit in units:
        path = out_dir / f"{unit.name}.json"
        paths.append(path)
        if unit_is_done(_read_json(path), script_sha=script_sha, mlx_version=mlx_version,
                        source=source, bypassed=bypassed, reps=reps):
            print(f"[skip] {unit.name}: done", flush=True)
            continue
        log = logs / f"{unit.name}.log"
        _move_aside(path, attempts)
        _move_aside(log, attempts)
        print(f"[run] {unit.name}", flush=True)
        outcome = spawn(unit, path, log)
        record = _read_json(path)
        if record is not None:
            record.update(unit_exit=outcome.returncode,
                          unit_duration_s=round(outcome.duration_s, 3))
        elif outcome.timed_out:
            record = parent_record(unit, "aborted_wall", outcome)
        else:
            _move_aside(path, attempts)  # an unreadable partial file, if any
            record = parent_record(unit, "error", outcome)
        _atomic_write(path, record)
        marginal = record.get("marginal_median_bytes")
        shown = f"{marginal / GIB:.4f} GiB" if isinstance(marginal, int | float) else "-"
        print(f"[end] {unit.name}: {record['status']} marginal {shown} "
              f"(exit {outcome.returncode}, {outcome.duration_s:.1f} s)", flush=True)
    return paths


def _scrub_log(log_path: Path) -> str:
    """Rewrite a finished worker's log without local paths; return the scrubbed text."""
    text = log_path.read_text(errors="replace")
    clean = scrub_text(text)
    if clean != text:
        log_path.write_text(clean)
    return clean


_CURRENT: list[subprocess.Popen[bytes]] = []


def _stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _on_signal(signum: int, _frame: FrameType | None) -> None:
    for proc in list(_CURRENT):
        if proc.poll() is None:
            _stop(proc)
    raise SystemExit(128 + signum)


def make_spawner(*, reps: int, watchdog_gib: float, unit_wall_s: float, mlx_version: str,
                 command_line: list[str], bypass: bool = False,
                 src: Path | None = None) -> Spawner:
    """The real spawner: `sys.executable -I -B <this script> --single-unit ...`, same
    process group as the parent, stdout+stderr to the unit log, stopped (TERM, then KILL
    after a grace period) once it outlives `unit_wall_s`."""
    worker_wall_s = max(unit_wall_s - WORKER_WALL_MARGIN_S, WORKER_WALL_MARGIN_S)

    def spawn(unit: Unit, out_path: Path, log_path: Path) -> WorkerOutcome:
        cmd = [sys.executable, "-I", "-B", str(_SCRIPT_PATH), "--single-unit",
               "--mode", unit.mode, "--impl", unit.impl, "--seq-lens", str(unit.n),
               "--reps", str(reps), "--watchdog-gib", str(watchdog_gib),
               "--worker-wall-s", str(worker_wall_s), "--expect-mlx", mlx_version,
               "--out", str(out_path), "--command-line", json.dumps(command_line)]
        if src is not None:
            cmd += ["--src", str(src)]
        if bypass:
            cmd.append("--bypass-verification-gate")
        start = time.monotonic()
        timed_out = False
        try:
            with log_path.open("wb") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)
                _CURRENT.append(proc)
                try:
                    proc.wait(timeout=unit_wall_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    print(f"[stop] {unit.name}: wall limit {unit_wall_s:.0f} s reached",
                          flush=True)
                    _stop(proc)
                finally:
                    _CURRENT.remove(proc)
        finally:  # an interrupted unit (SIGTERM, Ctrl-C) still leaves a scrubbed log
            tail = _scrub_log(log_path)[-STDERR_TAIL_CHARS:]
        return WorkerOutcome(returncode=proc.returncode, timed_out=timed_out,
                             duration_s=time.monotonic() - start, stderr_tail=tail)

    return spawn


def write_results(version_dir: Path, *, source: str | None, bypassed: bool) -> dict[str, Any]:
    records = [r for p in sorted(version_dir.glob("*-n*.json"))
               if (r := _read_json(p)) is not None]
    summary = {**summarize(records, source=source, bypassed=bypassed), "written_at": _utc()}
    _atomic_write(version_dir / "results.json", summary)
    return summary


# ----------------------------------------------------------------- worker (loads mlx)


class _OnceWriter:
    """One record per unit: the first writer (the measurement or the watchdog) wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._closed = False

    def write_once(self, record: dict[str, Any]) -> bool:
        if not self._lock.acquire(timeout=2.0):
            return False
        try:
            if self._closed:
                return False
            _atomic_write(self.path, record)
            self._closed = True
            return True
        finally:
            self._lock.release()


AttnFn = Callable[[Any, Any, Any], Any]


def _build_arm(unit: Unit, *, scale: float) -> tuple[AttnFn, list[Any], dict[str, Any]]:
    """The attention call for one unit, the extra inputs it closes over (mask or segment
    buffers, to be evaluated before the window), and what the record says about them."""
    import mlx.core as mx  # noqa: PLC0415

    from mlx_train_perf.attention import flash_attention  # noqa: PLC0415
    from mlx_train_perf.attention.segments import PackedMask  # noqa: PLC0415

    if unit.mode == "causal" and unit.impl == "stock":
        def stock_causal(q: Any, k: Any, v: Any) -> Any:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        return stock_causal, [], {"op": "mx.fast.scaled_dot_product_attention(mask='causal')"}
    if unit.mode == "causal":
        def flash_causal(q: Any, k: Any, v: Any) -> Any:
            return flash_attention(q, k, v, scale=scale, causal=True, impl="kernel")
        return flash_causal, [], {"op": "flash_attention(causal=True, impl='kernel')"}

    lengths = segment_lengths(unit.n, seed=LAYOUT_SEED)
    extra: dict[str, Any] = {"layout": {"lengths": lengths, "n_segments": len(lengths),
                                        "seed": LAYOUT_SEED, "min_len": SEG_MIN,
                                        "max_len": SEG_MAX}}
    if unit.impl == "stock":
        mask = packed_mask(lengths)
        extra["op"] = "mx.fast.scaled_dot_product_attention(mask=<bool (1, 1, N, N)>)"
        extra["mask"] = {
            "form": "boolean (True = attend)", "shape": list(mask.shape),
            "dtype": str(mask.dtype), "bytes": mask.nbytes,
            "accounting": "allocated before the window: inside pre_bytes, not inside the "
                          "marginal",
        }

        def stock_packed(q: Any, k: Any, v: Any) -> Any:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return stock_packed, [mask], extra

    seg, start = segment_ids(lengths)
    segments = PackedMask(seg_id=mx.array([seg], dtype=mx.int32),
                          seg_start=mx.array([start], dtype=mx.int32))
    extra["op"] = "flash_attention(causal=True, impl='kernel', segments=PackedMask)"
    extra["segment_buffers"] = {
        "shape": [B, unit.n], "dtype": "int32",
        "bytes": segments.seg_id.nbytes + segments.seg_start.nbytes,
        "accounting": "allocated before the window: inside pre_bytes",
    }

    def flash_packed(q: Any, k: Any, v: Any) -> Any:
        return flash_attention(q, k, v, scale=scale, causal=True, impl="kernel",
                               segments=segments)
    return flash_packed, [segments.seg_id, segments.seg_start], extra


def _measure(grad: Callable[..., Any], qkv: tuple[Any, Any, Any], *, reps: int,
             state: dict[str, Any]) -> dict[str, Any]:
    """One warm-up call, then `reps` measured calls, each in its own
    synchronize / clear_cache / pre / reset_peak window. Every call's gradients are kept
    alive until the end, then checked for finiteness outside every window."""
    import mlx.core as mx  # noqa: PLC0415

    state["phase"] = "warmup"
    t0 = time.perf_counter()
    warm = grad(*qkv)
    mx.eval(*warm)
    mx.synchronize()
    warmup_wall = time.perf_counter() - t0
    kept = [warm]

    rep_records: list[dict[str, Any]] = []
    for r in range(reps):
        state["phase"] = f"rep{r}"
        mx.synchronize()
        mx.clear_cache()
        pre = mx.get_active_memory()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        grads = grad(*qkv)
        mx.eval(*grads)
        mx.synchronize()
        wall = time.perf_counter() - t0
        peak = mx.get_peak_memory()
        kept.append(grads)
        rep_records.append({"rep": r, "pre_bytes": pre, "peak_bytes": peak,
                            "marginal_bytes": marginal_bytes(pre=pre, peak=peak),
                            "wall_s": round(wall, 6)})

    state["phase"] = "finite-check"
    finite = all(bool(mx.all(mx.isfinite(g)).item()) for gs in kept for g in gs)
    return {"reps": rep_records, **summarize_reps(rep_records),
            "warmup_wall_s": round(warmup_wall, 6), "grads_finite": finite}


def read_dispatch_plan(*, n: int, packed: bool) -> dict[str, Any]:
    """What the flash path dispatched for this bench's shape, read after it ran: the three
    rates it calibrated and cached in this process (forward, dQ, dK/dV; null when not
    cached), the dK/dV tile, and the number of dK/dV dispatch ranges the planner makes at
    the cached dK/dV rate (the launcher's own call, so the same ranges). Reads the caches
    and pure planners only: no calibration, no allocation, no kernel."""
    import mlx.core as mx  # noqa: PLC0415

    from mlx_train_perf.attention.kernel import launch  # noqa: PLC0415
    from mlx_train_perf.attention.kernel.dispatch import (  # noqa: PLC0415
        select_bwd_tiles,
        select_fwd_tile,
    )

    fwd_tile = select_fwd_tile(n, HEAD_DIM)
    dq_tile, dkv_tile = select_bwd_tiles(n, HEAD_DIM)

    def cached(cache: Mapping[Any, float], tile: Any) -> float | None:
        key = (HEAD_DIM, str(mx.bfloat16), True, B, HQ, launch._n_bucket(n),
               tile.variant, tile.d_slab, packed)
        return cache.get(key)

    rates = {"fwd": cached(launch._FWD_RATE_CACHE, fwd_tile),
             "dq": cached(launch._BWD_DQ_RATE_CACHE, dq_tile),
             "dkv": cached(launch._BWD_DKV_RATE_CACHE, dkv_tile)}
    block_align = 32 if dkv_tile.variant == "mma" else 1
    ranges: int | None = None
    if rates["dkv"] is not None:
        ranges = len(launch.plan_dkv_dispatches(n=n, d=HEAD_DIM, b=B, hq=HQ, hkv=HKV,
                                                rate=rates["dkv"], causal=True,
                                                block_align=block_align))
    return {
        "rates_macs_per_s": rates,
        "dkv_tile": {"variant": dkv_tile.variant, "d_slab": dkv_tile.d_slab,
                     "provisional": dkv_tile.provisional},
        "dkv_block_align": block_align,
        "dkv_ranges": ranges,
        "read": "process rate caches and the dK/dV planner, after the measured calls",
    }


def _worker_preflight(*, ver: str, expect_mlx: str, bypass: bool, src: Path | None) -> int | None:
    """The worker's exit code when it must not measure (no record written), else None.
    Applies the verification-gate bypass when asked for and allowed."""
    import mlx_train_perf  # noqa: PLC0415
    from mlx_train_perf import _compat  # noqa: PLC0415

    if ver != expect_mlx:
        print(f"[worker] mlx is {ver}, expected {expect_mlx}: no record written",
              file=sys.stderr, flush=True)
        return EXIT_WRONG_MLX
    if src is not None and not Path(str(mlx_train_perf.__file__)).resolve().is_relative_to(src):
        print("[worker] mlx_train_perf was not imported from --src", file=sys.stderr,
              flush=True)
        return 1
    if bypass:
        refusal = bypass_refusal(bypass=True, installed=ver,
                                 verified=_compat.VERIFIED_MLX_VERSIONS)
        if refusal is not None:
            print(f"[worker] {refusal}", file=sys.stderr, flush=True)
            return 1
        apply_verification_bypass(ver)
    return None


def _pin_device_limits(*, watchdog_gib: float,
                       worker_wall_s: float) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Set the wired and cache limits for this device; return the device info, the wired
    limit, and the watchdog block (its ceiling relative to the device's memory)."""
    import mlx.core as mx  # noqa: PLC0415

    dev = dict(mx.device_info())
    wired = wired_limit_for(int(dev["max_recommended_working_set_size"]))
    mx.set_wired_limit(wired)
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    watchdog = watchdog_fields(requested_gib=watchdog_gib, memory_size=int(dev["memory_size"]),
                               wall_budget_s=worker_wall_s)
    return dev, wired, watchdog


def run_worker(*, unit: Unit, reps: int, expect_mlx: str, out: Path, watchdog_gib: float,
               worker_wall_s: float, command_line: list[str], bypass: bool = False,
               src: Path | None = None) -> int:
    import mlx.core as mx  # noqa: PLC0415

    from mlx_train_perf.core.guards import install_memory_watchdog  # noqa: PLC0415

    ver = str(mx.__version__)  # type: ignore[attr-defined]
    refused = _worker_preflight(ver=ver, expect_mlx=expect_mlx, bypass=bypass, src=src)
    if refused is not None:
        return refused

    dev, wired, watchdog = _pin_device_limits(watchdog_gib=watchdog_gib,
                                              worker_wall_s=worker_wall_s)
    limit = int(watchdog["limit_bytes"])
    scale = HEAD_DIM ** -0.5
    fallback = os.environ.get(SDPA_VJP_FALLBACK_ENV)

    base: dict[str, Any] = {
        "schema": SCHEMA, "unit": unit.name, "mode": unit.mode, "impl": unit.impl,
        "n": unit.n, "b": B, "hq": HQ, "hkv": HKV, "head_dim": HEAD_DIM,
        "dtype": "bfloat16", "scale": scale, "seed": SEED, "reps_requested": reps,
        "loss": "out.sum()", "grads": "mx.grad(loss, argnums=(0, 1, 2)), all evaluated",
        "measured_call": ("forward+backward after one warm-up call; every call's gradients "
                          "stay alive until the unit ends"),
        "mlx_version": ver, **package_versions(src_given=src is not None),
        "python": platform.python_version(), "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "device": {k: dev.get(k) for k in
                   ("device_name", "architecture", "memory_size",
                    "max_recommended_working_set_size")},
        "wired_limit_bytes": wired, "cache_limit_bytes": CACHE_LIMIT_BYTES,
        "watchdog": watchdog,
        "env": {SDPA_VJP_FALLBACK_ENV: {"set": fallback is not None, "value": fallback}},
        **provenance(orig_argv=sys.orig_argv, executable=sys.executable,
                     command_line=command_line),
        **run_flags(bypassed=bypass,
                    src_identity=src_identity(src) if src is not None else None),
        "script_sha": script_sha(), "started_at": _utc(),
    }

    writer = _OnceWriter(out)
    state: dict[str, Any] = {"phase": "setup"}
    last = {"active": 0, "cache": 0}
    started = time.monotonic()

    def sampler() -> int:
        active, cache = mx.get_active_memory(), mx.get_cache_memory()
        last["active"], last["cache"] = active, cache
        return active + cache

    def on_breach(reason: str, _details: dict[str, object]) -> None:
        record = {
            **base, "status": "aborted_watchdog" if reason == "memory_ceiling"
            else "aborted_wall",
            "abort_reason": reason, "phase": state["phase"],
            "watchdog_active_bytes": last["active"], "watchdog_cache_bytes": last["cache"],
            "elapsed_s": round(time.monotonic() - started, 3), "finished_at": _utc(),
        }
        try:
            won = writer.write_once(record)
        except Exception:  # a failed write must not keep the process alive
            won = True
        if won:
            print(f"[watchdog] {reason} in phase {state['phase']}: exiting {EXIT_ABORT}",
                  file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(EXIT_ABORT)

    dog = install_memory_watchdog(ceiling_bytes=limit, wall_budget_s=worker_wall_s,
                                  on_breach=on_breach, sampler=sampler,
                                  interval_s=WATCHDOG_INTERVAL_S)
    try:
        state["phase"] = "inputs"
        mx.random.seed(SEED)
        q = mx.random.normal((B, HQ, unit.n, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((B, HKV, unit.n, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((B, HKV, unit.n, HEAD_DIM)).astype(mx.bfloat16)
        attn, extra_inputs, extra = _build_arm(unit, scale=scale)
        mx.eval(q, k, v, *extra_inputs)

        def loss(q_: Any, k_: Any, v_: Any) -> Any:
            return attn(q_, k_, v_).sum()

        measured = _measure(mx.grad(loss, argnums=(0, 1, 2)), (q, k, v), reps=reps,
                            state=state)

        if unit.impl == "flash":
            state["phase"] = "dispatch-plan"
            extra["dispatch_plan"] = read_dispatch_plan(n=unit.n,
                                                        packed=unit.mode == "packed")
        state["phase"] = "write"
        status = measured_status(grads_finite=measured["grads_finite"])
        record = {
            **base, **extra, "status": status, **measured,
            "active_after_bytes": mx.get_active_memory(),
            "cache_after_bytes": mx.get_cache_memory(), "finished_at": _utc(),
        }
        if not writer.write_once(record):
            return EXIT_ABORT  # the watchdog's record stands
        print(json.dumps({"unit": unit.name, "status": status,
                          "marginal_median_gib": round(record["marginal_median_bytes"] / GIB,
                                                       4),
                          "grads_finite": record["grads_finite"]}), flush=True)
        return 0 if status == "ok" else 1
    finally:
        dog.stop()


# ----------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Marginal peak memory of stock SDPA vs flash_attention, forward + "
                    "backward, one fresh process per (mode, impl, N).")
    ap.add_argument("--mode", nargs="+", choices=MODES, default=["causal"],
                    help="causal (default N 1024..8192) and/or packed (default N 4096 8192)")
    ap.add_argument("--impl", nargs="+", choices=IMPLS, default=list(IMPLS))
    ap.add_argument("--seq-lens", nargs="+", type=int, default=None,
                    help="override the per-mode default N list (applies to every mode)")
    ap.add_argument("--reps", type=int, default=MIN_REPS,
                    help=f"measured calls per unit after one warm-up (>= {MIN_REPS})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="results go to <out-dir>/mlx-<mlx version>/")
    ap.add_argument("--watchdog-gib", type=float, default=DEFAULT_WATCHDOG_GIB,
                    help="worker active+cache ceiling in GiB")
    ap.add_argument("--unit-wall-s", type=float, default=DEFAULT_UNIT_WALL_S,
                    help="the parent stops a worker that runs longer than this")
    ap.add_argument("--src", type=Path, default=None,
                    help="import mlx_train_perf from this source tree (parent and workers)")
    ap.add_argument("--bypass-verification-gate", action="store_true",
                    help="measurement only: let flash run on an mlx the kernel gate does "
                         "not list; refused on a verified mlx; marked in every record")
    # Internal worker surface (the parent re-invokes this script once per unit).
    ap.add_argument("--single-unit", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-mlx", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--worker-wall-s", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--command-line", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    if args.reps < MIN_REPS:
        ap.error(f"--reps must be >= {MIN_REPS}")
    src: Path | None = None
    if args.src is not None:
        try:
            src = prepare_src(args.src)
        except ValueError as exc:
            ap.error(str(exc))

    if args.single_unit:
        if (len(args.mode) != 1 or len(args.impl) != 1 or not args.seq_lens
                or len(args.seq_lens) != 1 or None in (args.expect_mlx, args.out,
                                                       args.worker_wall_s)):
            ap.error("--single-unit needs one mode, impl and N, --expect-mlx, --out and "
                     "--worker-wall-s")
        unit = Unit(mode=args.mode[0], impl=args.impl[0], n=args.seq_lens[0])
        return run_worker(unit=unit, reps=args.reps, expect_mlx=args.expect_mlx, out=args.out,
                          watchdog_gib=args.watchdog_gib, worker_wall_s=args.worker_wall_s,
                          command_line=json.loads(args.command_line or "[]"),
                          bypass=args.bypass_verification_gate, src=src)

    units: list[Unit] = []
    for mode in args.mode:
        seq_lens = args.seq_lens or DEFAULT_SEQ_LENS[mode]
        if mode == "packed" and min(seq_lens) < SEG_MIN:
            ap.error(f"packed mode needs N >= {SEG_MIN}")
        units += build_units(modes=[mode], impls=args.impl, seq_lens=seq_lens)

    mlx_version = _dist("mlx")
    if mlx_version is None:
        print("[stop] mlx is not installed in this environment", file=sys.stderr)
        return 1
    if args.bypass_verification_gate:
        from mlx_train_perf import _compat  # noqa: PLC0415 -- after --src is on sys.path

        refusal = bypass_refusal(bypass=True, installed=mlx_version,
                                 verified=_compat.VERIFIED_MLX_VERSIONS)
        if refusal is not None:
            ap.error(refusal)
    version_dir = args.out_dir / f"mlx-{mlx_version}"
    source = src_identity(src) if src is not None else None
    bypassed = bool(args.bypass_verification_gate)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)
    command_line = portable_argv(sys.orig_argv)
    print(f"[bench] mlx {mlx_version}; {len(units)} units; out {version_dir}", flush=True)
    spawn = make_spawner(reps=args.reps, watchdog_gib=args.watchdog_gib,
                         unit_wall_s=args.unit_wall_s, mlx_version=mlx_version,
                         command_line=command_line, bypass=bypassed, src=src)
    paths = run_units(units, out_dir=version_dir, spawn=spawn, script_sha=script_sha(),
                      mlx_version=mlx_version, source=source, bypassed=bypassed,
                      reps=args.reps)
    summary = write_results(version_dir, source=source, bypassed=bypassed)
    print(summary["table"], flush=True)
    for key, ratio in summary["ratio_stock_over_flash"].items():
        print(f"stock/flash marginal, {key}: {ratio if ratio is not None else 'n/a'}")
    statuses = [(_read_json(p) or {}).get("status") for p in paths]
    return 0 if all(s in DONE_STATUSES for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
