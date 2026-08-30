"""Layer-level GatedDelta bench: real Qwen3.5-0.8B `GatedDeltaNet` geometry, sequential
(mlx-lm's own python-loop oracle) vs chunked (ours) arms.

Unlike the op-level parity cases in `tests/recurrent_parity_cases.py` (synthetic random
q/k/v/g/beta tensors at the same Hk/Hv/Dk/Dv shapes), this bench instantiates the REAL
`mlx_lm.models.qwen3_5.GatedDeltaNet` module from EXPLICIT real-0.8B `TextModelArgs`
(`QWEN35_08B_TEXT_CONFIG` below) -- never `TextModelArgs` defaults, which describe a
~10x larger, `repeat_factor=4` block the 0.8B model never takes (see
`mlx_train_perf.recurrent.reference`'s own module docstring). Every projection weight
(`in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`/`out_proj`), the depthwise `conv1d`,
and the gated `RMSNorm` are real (randomly initialized, seed-deterministic) and run
IDENTICALLY in both arms -- only the layer's own core recurrent state update is swapped.

`GatedDeltaNet.__call__` normally routes that update through `gated_delta_update`
(imported into `mlx_lm.models.qwen3_5`'s own module namespace), which itself picks
between mlx-lm's python-loop `gated_delta_ops` (training mode) or mlx-lm's OWN fused
Metal kernel (eval mode) -- neither of which is this project's chunked op. This script
monkeypatches that one module-global name (restored via a context manager, scoped to a
single measured call) to route instead to `sequential_gated_delta()` (mlx-lm's own
`gated_delta_ops` -- the "sequential" arm) or `chunked_gated_delta` (ours -- the
"chunked" arm), computing `beta`/`g` the exact same way mlx-lm's own `gated_delta_update`
does (`mx.sigmoid(b)` / `compute_g(A_log, a, dt_bias)`) so a swap changes nothing else
about what the layer computes.

Each condition (one `(impl, seq_len)` pair) measures forward-only and forward+backward
(`nn.value_and_grad` over the layer's own trainable parameters, loss = squared output
sum) with REP-ISOLATED memory/wall boundaries: `mx.synchronize()` + `mx.clear_cache()` +
`mx.reset_peak_memory()` before EACH of 5 reps (not one shared reset across all reps, the
convention this project's other bench scripts use) -- min/median/max are reported over 5
INDEPENDENT peak readings. The warmup call (outside any reset window, kept alive) pays
Metal JIT + the layer's own one-time lazy-graph materialization before any timed rep.

Memory discipline: `install_guardrails()` (wired+soft caps) FIRST, then
`mx.set_cache_limit(...)` to bound the retained pool, then a daemon
`install_memory_watchdog` sampling `active + cache` (NOT active alone -- dropped buffers
move to MLX's retained cache pool, which an active-only sampler misses) every 50ms
against a CAMPAIGN-PINNED ceiling (not the dynamic `effective_memory_ceiling()` this
project's other bench scripts use -- this bench's shapes are small and well-characterized:
the sequential oracle's own backward tops out ~8.2 GiB at T=2048, B=1, bf16, so a fixed,
generous ceiling well under the house wired cap is deliberately simpler here). The breach
callback is intentionally NOT `bench.artifacts.make_watchdog_on_breach` (which writes an
artifact on breach): it is non-blocking (no `mx.synchronize()`, no subprocess spawn, no
disk I/O) and fails CLOSED -- record counters in the caller's own dict, then
UNCONDITIONALLY `os._exit(2)` from a `finally`, so a paging storm is stopped as fast as
possible rather than risk a blocking write while memory is already critical.

subprocess-per-condition (workspace convention -- MLX's lazy allocator otherwise holds
buffers across runs within one process), same shape as `scripts/bench_attention_op.py`:
the top-level invocation builds the full `(impl, seq_len)` grid and self-invokes THIS
script once per stale condition via `subprocess.run([sys.executable, __file__, ...])`
with the internal `--single-condition` marker pinned to exactly one pair -- resume-by-skip
identical to `bench.runner.run_conditions`'s own convention.

`--dry-run` builds the condition list and resolves every artifact path + identity WITHOUT
touching MLX or `mlx_lm` at all -- no model load, no GPU allocation, nothing written. Every
run THIS task performs is `--dry-run` only (workspace rule: a subagent implementing a bench
script never executes a real condition); a full measurement campaign owns the real run
(main session, ETA-stated, AC power, serialized against other heavy runs).
"""

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx import nn

from mlx_train_perf.bench.artifacts import (
    condition_identity,
    new_session_id,
    result_is_fresh,
    write_result,
)
from mlx_train_perf.core.guards import (
    DEFAULT_WALL_BUDGET_S,
    install_guardrails,
    install_memory_watchdog,
)
from mlx_train_perf.errors import MissingDependencyError
from mlx_train_perf.recurrent.ops import CHUNK_SIZE

_IMPLS: tuple[str, ...] = ("sequential", "chunked")
DEFAULT_SEQ_LENS: tuple[int, ...] = (128, 256, 512, 1024, 2048)
DEFAULT_BATCH = 1
DEFAULT_REPS = 5
_DTYPES: dict[str, mx.Dtype] = {
    "float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16,
}
DEFAULT_DTYPE = "bfloat16"

# The real mlx-community/Qwen3.5-0.8B-4bit text_config (verified against the cached
# snapshot) -- explicit, NOT `TextModelArgs` defaults (a ~10x larger, repeat_factor=4
# block the 0.8B never takes). Same values as this project's `QWEN35_08B_GEOM` op-level
# view (`tests/recurrent_parity_cases.py`: Hk=16, Hv=16, Dk=128, Dv=128), derived here:
# linear_num_key_heads == linear_num_value_heads == 16 (repeat_factor 1), Dk ==
# linear_key_head_dim == 128, Dv == linear_value_head_dim == 128.
QWEN35_08B_TEXT_CONFIG: dict[str, object] = {
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
}
GEOM_SOURCE = (
    "mlx-community/Qwen3.5-0.8B-4bit text_config, verified against the cached snapshot"
)

# Active+cache ceiling for this bench's daemon watchdog -- campaign-pinned (not the
# dynamic `effective_memory_ceiling()` this project's other bench scripts use): the
# sequential oracle's own backward tops out ~8.2 GiB at T=2048/B=1/bf16 (measured,
# op-level), leaving generous headroom for the real layer's own conv1d/projection
# activations while staying under the 20 GiB house wired cap `install_guardrails`
# installs.
CAMPAIGN_CEILING_BYTES = 18 * 1024**3
CAMPAIGN_CACHE_LIMIT_BYTES = 2 * 1024**3
CAMPAIGN_WALL_BUDGET_S = DEFAULT_WALL_BUDGET_S

_STDERR_TAIL_CHARS = 4000  # enough to see the failing assertion/traceback, not a full dump

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUT_DIR = _SCRIPTS_DIR.parent / "_artifacts" / "bench_recurrent_layer"


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- `bench.artifacts.CODE_SHA_DEPS`
    deliberately excludes ad hoc bench scripts under `scripts/`, so without this, an edit
    to THIS script's own measurement logic would not invalidate a previously-written
    artifact."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


def _require_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise MissingDependencyError(
            "the layer-level recurrent bench requires the 'mlx-lm' extra: "
            "pip install 'mlx-train-perf[mlx-lm]'"
        ) from exc


def _resolve_dtype(name: str) -> mx.Dtype:
    return _DTYPES[name]


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerCondition:
    name: str
    impl: str
    seq_len: int
    batch: int
    dtype: str
    chunk_size: int
    reps: int


def build_conditions(
    *, impls: Sequence[str], seq_lens: Sequence[int], batch: int, dtype: str,
    chunk_size: int, reps: int,
) -> list[LayerCondition]:
    """The full `(impl, seq_len)` grid, pure -- `seq_len` outer (matches
    `scripts/bench_attention_op.py`'s own grid-construction ordering). Every arm
    dimension that could otherwise let two conditions collide on one artifact path
    (impl/seq_len/batch/dtype) is part of the condition NAME, not just the identity."""
    return [
        LayerCondition(
            name=f"{impl}_T{seq_len}_B{batch}_{dtype}", impl=impl, seq_len=seq_len,
            batch=batch, dtype=dtype, chunk_size=chunk_size, reps=reps,
        )
        for seq_len in seq_lens
        for impl in impls
    ]


def _params_for(condition: LayerCondition) -> dict[str, object]:
    return {
        "impl": condition.impl, "seq_len": condition.seq_len, "batch": condition.batch,
        "dtype": condition.dtype, "chunk_size": condition.chunk_size,
        "reps": condition.reps, "text_config": QWEN35_08B_TEXT_CONFIG,
        "geometry_source": GEOM_SOURCE, "script_sha": script_sha(),
    }


def _identity_for(condition: LayerCondition, *, session_id: str) -> dict[str, object]:
    return condition_identity(
        kind="recurrent_layer", session_id=session_id, params=_params_for(condition),
    )


def _out_path(out_dir: Path, condition: LayerCondition) -> Path:
    return out_dir / f"{condition.name}.json"


def _build_layer(*, dtype: mx.Dtype, seed: int) -> Any:
    """Build the real `GatedDeltaNet` from the explicit real-0.8B config -- NOT
    `TextModelArgs` defaults (see the module docstring). Casting to `dtype` and forcing
    the lazy init graph (`mx.eval`) BEFORE the measured window mirrors
    `bench/worker.py`'s own gotcha-14 discipline (a lazy cast/init cost must not leak
    into the first rep's wall/peak)."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs  # noqa: PLC0415

    mx.random.seed(seed)
    config = TextModelArgs(
        model_type="qwen3_5", **cast("dict[str, Any]", QWEN35_08B_TEXT_CONFIG)
    )
    layer: Any = GatedDeltaNet(config)
    layer.set_dtype(dtype)
    mx.eval(layer.parameters())
    return layer


@contextlib.contextmanager
def _patched_gated_delta_update(impl: str, chunk_size: int) -> Iterator[None]:
    """Swap ONLY the layer's core recurrent update, scoped to this context manager and
    restored on exit -- see the module docstring for why a plain `layer(...)` call
    cannot exercise the `chunked` arm at all (`GatedDeltaNet.__call__`'s own
    `use_kernel=not self.training` routes to mlx-lm's OWN sequential ops or its OWN
    fused Metal kernel, never to this project's op)."""
    import mlx_lm.models.qwen3_5 as qwen3_5_mod  # noqa: PLC0415
    from mlx_lm.models.gated_delta import compute_g  # noqa: PLC0415

    from mlx_train_perf.recurrent.ops import chunked_gated_delta  # noqa: PLC0415
    from mlx_train_perf.recurrent.reference import sequential_gated_delta  # noqa: PLC0415

    def update(
        q: mx.array, k: mx.array, v: mx.array, a: mx.array, b: mx.array,
        A_log: mx.array, dt_bias: mx.array,  # noqa: N803
        state: mx.array | None = None, mask: mx.array | None = None,
        use_kernel: bool = True,  # noqa: ARG001 -- this bench's --impl choice overrides it
    ) -> tuple[mx.array, mx.array]:
        beta = mx.sigmoid(b)
        g = compute_g(A_log, a, dt_bias)
        if impl == "sequential":
            return cast(
                "tuple[mx.array, mx.array]",
                sequential_gated_delta()(q, k, v, g, beta, state, mask),
            )
        return chunked_gated_delta(q, k, v, g, beta, state, mask, chunk_size=chunk_size)

    # `getattr`/`setattr` (not attribute syntax) deliberately: `gated_delta_update` is
    # imported into `qwen3_5`'s namespace without being re-exported (no `__all__`), which
    # mypy's strict `no_implicit_reexport` would otherwise reject on a plain
    # `qwen3_5_mod.gated_delta_update` access -- this monkeypatch is intentionally
    # reaching past that boundary, scoped and restored by this context manager.
    original = getattr(qwen3_5_mod, "gated_delta_update")  # noqa: B009
    setattr(qwen3_5_mod, "gated_delta_update", update)  # noqa: B010
    try:
        yield
    finally:
        setattr(qwen3_5_mod, "gated_delta_update", original)  # noqa: B010


def _measure_arm(
    call: Callable[[], object], *, eval_fn: Callable[[object], None], reps: int,
) -> dict[str, object]:
    """Rep-isolated measurement: EACH rep gets its own `mx.synchronize()` +
    `mx.clear_cache()` + `mx.reset_peak_memory()` boundary (not one shared reset across
    all reps, the convention this project's other bench scripts use) --
    min/median/max are reported over `reps` INDEPENDENT peak readings. The warmup call
    is measured separately (outside any reset window) and kept ALIVE (bound to `warm`
    for this function's whole lifetime) through the loop below -- gotcha 15: freeing it
    early would race the allocator's deferred release against a rep's own
    `active_before` snapshot and can read a stale-high baseline."""
    warm = call()
    eval_fn(warm)
    walls: list[float] = []
    peaks_gb: list[float] = []
    for _ in range(reps):
        mx.synchronize()
        mx.clear_cache()
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = call()
        eval_fn(out)
        walls.append(time.perf_counter() - t0)
        peaks_gb.append((mx.get_peak_memory() - active_before) / 1024**3)
    return {
        "wall_s_all": [round(w, 6) for w in walls],
        "wall_s_min": round(min(walls), 6),
        "wall_s_median": round(statistics.median(walls), 6),
        "wall_s_max": round(max(walls), 6),
        "peak_gb_all": [round(p, 4) for p in peaks_gb],
        "peak_gb_min": round(min(peaks_gb), 4),
        "peak_gb_median": round(statistics.median(peaks_gb), 4),
        "peak_gb_max": round(max(peaks_gb), 4),
    }


def measure_condition(condition: LayerCondition) -> dict[str, object]:
    """The worker body: build the real layer, one synthetic `hidden_states` input, then
    measure forward-only and forward+backward (`nn.value_and_grad` over the layer's own
    trainable parameters, loss = squared output sum) under the swapped-in `--impl` arm.
    `install_guardrails`/`mx.set_cache_limit`/the watchdog are installed by the CALLER
    (`_run_single_condition`) before this runs, matching `bench/worker.py`'s own
    ordering (guardrails first, before any allocation this condition makes)."""
    _require_mlx_lm()
    hidden_size = int(cast(int, QWEN35_08B_TEXT_CONFIG["hidden_size"]))
    dtype = _resolve_dtype(condition.dtype)

    layer = _build_layer(dtype=dtype, seed=0)
    mx.random.seed(1)
    hidden = mx.random.normal((condition.batch, condition.seq_len, hidden_size)).astype(dtype)
    mx.eval(hidden)

    def loss_fn(model: Any, h: mx.array) -> mx.array:
        return cast(mx.array, (model(h, mask=None, cache=None).astype(mx.float32) ** 2).sum())

    value_and_grad = nn.value_and_grad(layer, loss_fn)

    def forward() -> mx.array:
        return cast(mx.array, layer(hidden, mask=None, cache=None))

    def forward_backward() -> tuple[mx.array, object]:
        return cast("tuple[mx.array, object]", value_and_grad(hidden))

    with _patched_gated_delta_update(condition.impl, condition.chunk_size):
        fwd = _measure_arm(forward, eval_fn=mx.eval, reps=condition.reps)
        fwd_backward = _measure_arm(
            forward_backward,
            eval_fn=lambda out: mx.eval(cast("tuple[Any, Any]", out)[0],
                                        cast("tuple[Any, Any]", out)[1]),
            reps=condition.reps,
        )

    return {
        "impl": condition.impl, "seq_len": condition.seq_len, "batch": condition.batch,
        "dtype": condition.dtype, "chunk_size": condition.chunk_size,
        "fwd": fwd, "fwd_backward": fwd_backward,
    }


def _make_on_breach(counters: dict[str, object]) -> Callable[[str, dict[str, object]], None]:
    """Breach callback: NON-BLOCKING and fail-CLOSED. `core.guards._watchdog_step` calls
    `on_breach` under `contextlib.suppress(Exception)` -- an exception here would
    silently disarm the guard while a paging storm continues, so the process-exit call
    sits in a `finally` and nothing before it can plausibly raise (a dict write only, no
    I/O, no `mx.synchronize()`, no subprocess spawn). Deliberately NOT
    `bench.artifacts.make_watchdog_on_breach` (which writes an `aborted_*` artifact):
    a breach here should exit as fast as possible rather than risk a blocking write
    while memory is already critical -- `counters` is the caller's own dict, mutated in
    place so a test can assert what was recorded without terminating the test runner."""

    def on_breach(reason: str, details: dict[str, object]) -> None:
        try:
            counters["breach_reason"] = reason
            counters["breach_details"] = details
        finally:
            os._exit(2)

    return on_breach


def _run_single_condition(condition: LayerCondition, *, out_dir: Path, session_id: str) -> Path:
    """Measures exactly ONE condition and writes its artifact unconditionally (no
    freshness check here -- the orchestrator, `run_grid`, already unlinks a stale
    artifact before spawning, matching `bench.runner.run_conditions`'s convention)."""
    out_path = _out_path(out_dir, condition)
    ident = _identity_for(condition, session_id=session_id)
    install_guardrails()  # FIRST -- before any allocation this condition makes
    mx.set_cache_limit(CAMPAIGN_CACHE_LIMIT_BYTES)
    counters: dict[str, object] = {}
    watchdog = install_memory_watchdog(
        ceiling_bytes=CAMPAIGN_CEILING_BYTES,
        sampler=lambda: mx.get_active_memory() + mx.get_cache_memory(),
        interval_s=0.05,
        wall_budget_s=CAMPAIGN_WALL_BUDGET_S,
        on_breach=_make_on_breach(counters),
    )
    try:
        fields = measure_condition(condition)
        write_result(out_path, ident, "ok", **fields)
        return out_path
    finally:
        # A breach never reaches here -- `on_breach` already hard-exited the process.
        watchdog.stop()


def _spawn_condition(
    condition: LayerCondition, *, out_dir: Path, session_id: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable, str(_SCRIPT_PATH),
        "--impl", condition.impl,
        "--seq-lens", str(condition.seq_len),
        "--batch", str(condition.batch),
        "--dtype", condition.dtype,
        "--chunk-size", str(condition.chunk_size),
        "--reps", str(condition.reps),
        "--out-dir", str(out_dir),
        "--session-id", session_id,
        "--single-condition",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_grid(
    conditions: list[LayerCondition], *, out_dir: Path, session_id: str,
) -> list[Path]:
    """Subprocess-per-condition orchestration -- same shape as
    `scripts/bench_attention_op.py`'s own `run_grid`: a fresh artifact is skipped
    without spawning; a stale one is unlinked BEFORE spawning; a nonzero exit or a
    clean exit that wrote nothing is recorded as an `"error"` result on this side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for condition in conditions:
        out_path = _out_path(out_dir, condition)
        ident = _identity_for(condition, session_id=session_id)
        paths.append(out_path)
        if result_is_fresh(out_path, ident):
            continue
        out_path.unlink(missing_ok=True)
        proc = _spawn_condition(condition, out_dir=out_dir, session_id=session_id)
        if proc.returncode != 0 and not out_path.exists():
            stderr_tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
            write_result(
                out_path, ident, "error", error_type="WorkerCrashed",
                error_msg=stderr_tail, returncode=proc.returncode,
            )
        elif proc.returncode == 0 and not out_path.exists():
            write_result(
                out_path, ident, "error", error_type="WorkerExitedWithoutArtifact",
                error_msg="subprocess exited 0 without writing an artifact", returncode=0,
            )
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="+", choices=_IMPLS, default=list(_IMPLS))
    ap.add_argument("--seq-lens", nargs="+", type=int, default=list(DEFAULT_SEQ_LENS))
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default=DEFAULT_DTYPE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the condition list and resolve every artifact path + "
                        "identity WITHOUT touching MLX or mlx_lm -- no model load, no "
                        "GPU allocation, nothing written")
    # Internal self-reinvocation surface (subprocess-per-condition -- see the module
    # docstring): not part of the documented top-level CLI, so both are suppressed from
    # --help.
    ap.add_argument("--session-id", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single-condition", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    session_id = args.session_id or new_session_id()

    conditions = build_conditions(
        impls=args.impl, seq_lens=args.seq_lens, batch=args.batch, dtype=args.dtype,
        chunk_size=args.chunk_size, reps=args.reps,
    )

    if args.dry_run:
        # Builds the condition list and resolves every artifact path + identity --
        # `condition_identity` only reads this project's own source bytes (`code_sha`)
        # and installed-package metadata, never touches MLX or mlx_lm. Writes nothing.
        report = [
            {
                "name": c.name, "out_path": str(_out_path(out_dir, c)),
                "identity": _identity_for(c, session_id=session_id),
            }
            for c in conditions
        ]
        print(json.dumps(report, indent=2))
        return 0

    if args.single_condition:
        if len(args.impl) != 1 or len(args.seq_lens) != 1:
            raise SystemExit(
                "--single-condition requires exactly one --impl and one --seq-lens value"
            )
        condition = conditions[0]
        out_path = _run_single_condition(condition, out_dir=out_dir, session_id=session_id)
        data = json.loads(out_path.read_text())
        print(json.dumps(data, indent=2))
        return 0 if data.get("status") != "error" else 1

    paths = run_grid(conditions, out_dir=out_dir, session_id=session_id)
    print(json.dumps({"paths": [str(p) for p in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
