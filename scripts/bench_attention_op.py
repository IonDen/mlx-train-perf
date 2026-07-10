"""Single-op attention memory-scaling bench (spec §8/§10.6) -- the O(N) proof.

Model-free: synthetic `(B=1, Hq, Hkv, N, D)` tensors, no `mlx_lm`/model load. Two `impl`
arms, both measured at the SAME shape so the comparison is apples to apples:

  flash -- `attention.api.flash_attention(..., impl="kernel")`, the production
           Metal-backed path (T9b: forward AND backward fully kernel-backed, calibrated
           split rates). Expected O(N)-class memory growth: the `(N, N)` score/probability
           matrix is never materialized on either pass.
  stock -- `attention.reference.math_attention` under plain MLX autodiff -- the pure-MLX
           O(N^2) oracle this release replaces (same measurement SHAPE as T2's moat check,
           `tests/test_attention_composition.py::
           test_sdpa_backward_is_still_quadratic_on_installed_mlx`, through OUR math
           reference rather than `mx.fast.scaled_dot_product_attention` since that upstream
           op is the one this project's flash arm stands in for).

Each condition (one `(impl, n)` pair) measures TWO peaks with independent
reset-peak/eval boundaries (worker discipline from `bench/worker.py`: warmup pays Metal
JIT -- and, for `flash`, the construction-time rate/table calibration `attention/api.py`'s
own module docstring requires -- OUTSIDE any timed/peak window):

  fwd_peak_gb    -- one forward-only call.
  fwdbwd_peak_gb -- `mx.value_and_grad` over `attn(...).sum()` w.r.t. (q, k, v); `wall_s`
                    is this call's wall-clock.

The O(N) PROOF is `fwdbwd_peak_gb`'s doubling ratio across `--seq-lens`: flash should show
~2x per doubling (O(N)-class), stock ~3.8x (the measured O(N^2) baseline, spec §8). That
assertion is never made here against a real GPU number -- it lives in
`compute_doubling_ratios`'s own unit test against SYNTHETIC artifacts
(`tests/test_bench_attention_op.py`); the real numbers are T13's measurement campaign.

subprocess-per-condition (workspace convention -- MLX's lazy allocator otherwise holds
buffers across runs within one process): the top-level invocation

    python scripts/bench_attention_op.py --impl flash stock --seq-lens 2048 4096 8192 \\
        --head-dim 128 --heads 32 --kv-heads 8 [--out-dir ...]

builds the full `(impl, n)` grid and self-invokes THIS script once per stale condition via
`subprocess.run([sys.executable, __file__, ...])` with the internal `--single-condition`
marker pinned to exactly that one `(impl, n)` pair -- resume-by-skip identical to
`bench.runner.run_conditions`'s own convention (a condition whose artifact identity is
already fresh is never spawned; a crashed/silently-exited subprocess gets its failure
recorded as an `"error"` result on the ORCHESTRATOR's side, so one bad condition never
aborts the rest of the grid). `--impl` defaults to both arms; each artifact carries the
T10-extended identity (`attention_impl` -- see `bench.artifacts.condition_identity`), plus
`impl`/`n`/`fwd_peak_gb`/`fwdbwd_peak_gb`/`wall_s` as top-level result fields.

Every RUN this script performs in THIS task is a tiny synthetic shape (the `--run-benchmark`
gated smoke, N=256) -- never a flagship dispatch. T13's campaign owns the real measurement
run (main session, ETA-stated, AC power, serialized against other heavy runs).
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_train_perf.attention.api import flash_attention
from mlx_train_perf.attention.reference import math_attention
from mlx_train_perf.bench.artifacts import (
    condition_identity,
    new_session_id,
    result_is_fresh,
    write_result,
)
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.errors import LaunchBudgetError

DEFAULT_SEQ_LENS: tuple[int, ...] = (2048, 4096, 8192)
DEFAULT_HEAD_DIM = 128
DEFAULT_HEADS = 32
DEFAULT_KV_HEADS = 8
DTYPE = mx.bfloat16
SEED = 0

_STDERR_TAIL_CHARS = 4000  # enough to see the failing assertion/traceback, not a full dump

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUT_DIR = _SCRIPTS_DIR.parent / "_artifacts" / "bench_attention_op"


def script_sha() -> str:
    """Fingerprint of THIS script's own bytes -- `bench.artifacts.CODE_SHA_DEPS`
    deliberately excludes ad hoc bench scripts under `scripts/` (same convention
    `bench_backward_ladder.py`/`ground_truth_atomic_outputs.py` already established), so
    without this, an edit to THIS script's own measurement logic would not invalidate a
    previously-written artifact."""
    return hashlib.sha256(_SCRIPT_PATH.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True, slots=True, kw_only=True)
class AttnCondition:
    name: str
    impl: str
    n: int
    head_dim: int
    heads: int
    kv_heads: int


def build_conditions(
    *, impls: Sequence[str], seq_lens: Sequence[int],
    head_dim: int, heads: int, kv_heads: int,
) -> list[AttnCondition]:
    """The full `(impl, n)` grid, pure -- one condition per pair, `n` outer (matches
    `scripts/bench_loss_layer.py`'s own grid-construction ordering)."""
    return [
        AttnCondition(name=f"{impl}_n{n}", impl=impl, n=n, head_dim=head_dim, heads=heads,
                     kv_heads=kv_heads)
        for n in seq_lens
        for impl in impls
    ]


def compute_doubling_ratios(
    entries: Sequence[dict[str, object]],
) -> dict[str, dict[str, float]]:
    """Per-impl `{"n->2n": ratio}` for every seq-len present whose exact double is ALSO
    present, computed off `fwdbwd_peak_gb` -- the O(N) proof this bench exists to produce
    (see the module docstring). Pure: plain dicts in, plain dict out, never touches MLX or
    the filesystem. Entries missing `impl`/`n`/`fwdbwd_peak_gb`, or with the wrong types,
    are silently skipped (defensive against a non-"ok" or hand-built entry) -- callers that
    care about status filter to `status == "ok"` before calling this."""
    by_impl: dict[str, dict[int, float]] = defaultdict(dict)
    for e in entries:
        impl, n, peak = e.get("impl"), e.get("n"), e.get("fwdbwd_peak_gb")
        if not isinstance(impl, str):
            continue
        if not (isinstance(n, int) and not isinstance(n, bool)):
            continue
        if not (isinstance(peak, int | float) and not isinstance(peak, bool)):
            continue
        by_impl[impl][n] = float(peak)

    ratios: dict[str, dict[str, float]] = {}
    for impl, peaks in by_impl.items():
        impl_ratios = {
            f"{n}->{2 * n}": peaks[2 * n] / peaks[n]
            for n in peaks
            if 2 * n in peaks and peaks[n] > 0
        }
        if impl_ratios:
            ratios[impl] = impl_ratios
    return ratios


def _params_for(condition: AttnCondition) -> dict[str, object]:
    return {
        "n": condition.n, "head_dim": condition.head_dim, "heads": condition.heads,
        "kv_heads": condition.kv_heads, "dtype": str(DTYPE), "causal": True,
        "seed": SEED, "script_sha": script_sha(),
    }


def _identity_for(condition: AttnCondition, *, session_id: str) -> dict[str, object]:
    return condition_identity(
        kind="attention_op", session_id=session_id, params=_params_for(condition),
        attention_impl=condition.impl,
    )


def _out_path(out_dir: Path, condition: AttnCondition) -> Path:
    return out_dir / f"{condition.name}.json"


def _build_qkv(condition: AttnCondition, *, dtype: mx.Dtype, seed: int) -> tuple[
    mx.array, mx.array, mx.array,
]:
    mx.random.seed(seed)
    q = mx.random.normal((1, condition.heads, condition.n, condition.head_dim)).astype(dtype)
    k = mx.random.normal((1, condition.kv_heads, condition.n, condition.head_dim)).astype(dtype)
    v = mx.random.normal((1, condition.kv_heads, condition.n, condition.head_dim)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _attn_call(
    condition: AttnCondition, q: mx.array, k: mx.array, v: mx.array, *, scale: float,
) -> mx.array:
    if condition.impl == "flash":
        return flash_attention(q, k, v, scale=scale, causal=True, impl="kernel")
    return math_attention(q, k, v, scale=scale, causal=True)


def measure_condition(
    condition: AttnCondition, *, dtype: mx.Dtype = DTYPE, seed: int = SEED,
) -> dict[str, object]:
    """The worker body: install the house wired caps, build synthetic q/k/v, warm the
    call OUTSIDE any timed/peak window, then measure fwd-only peak and fwd+bwd peak+wall
    with INDEPENDENT `reset_peak_memory`/`mx.eval` boundaries (worker discipline from
    `bench/worker.py`). May raise `LaunchBudgetError` -- the flash arm's kernel launch
    guard is a real refusal, not caught here; `_run_single_condition` records it."""
    install_guardrails()
    scale = 1.0 / math.sqrt(condition.head_dim)
    q, k, v = _build_qkv(condition, dtype=dtype, seed=seed)

    warm = _attn_call(condition, q, k, v, scale=scale)
    mx.eval(warm)
    mx.clear_cache()

    active_before_fwd = mx.get_active_memory()
    mx.reset_peak_memory()
    o = _attn_call(condition, q, k, v, scale=scale)
    mx.eval(o)
    fwd_peak_gb = (mx.get_peak_memory() - active_before_fwd) / 1024**3

    def loss(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
        return _attn_call(condition, q_, k_, v_, scale=scale).sum()

    vag = mx.value_and_grad(loss, argnums=(0, 1, 2))

    mx.clear_cache()
    active_before_bwd = mx.get_active_memory()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    val, grads = vag(q, k, v)
    mx.eval(val, *grads)
    wall_s = time.perf_counter() - t0
    fwdbwd_peak_gb = (mx.get_peak_memory() - active_before_bwd) / 1024**3

    return {
        "impl": condition.impl,
        "n": condition.n,
        "fwd_peak_gb": round(fwd_peak_gb, 4),
        "fwdbwd_peak_gb": round(fwdbwd_peak_gb, 4),
        "wall_s": round(wall_s, 6),
    }


def _run_single_condition(condition: AttnCondition, *, out_dir: Path, session_id: str) -> Path:
    """Measures exactly ONE condition and writes its artifact unconditionally (no
    freshness check here -- the orchestrator, `run_grid`, already unlinks a stale
    artifact before spawning, matching `bench.runner.run_conditions`'s convention). A
    `LaunchBudgetError` refusal IS a recorded result, not a crash."""
    out_path = _out_path(out_dir, condition)
    ident = _identity_for(condition, session_id=session_id)
    try:
        fields = measure_condition(condition)
    except LaunchBudgetError as exc:
        write_result(out_path, ident, "refused", error=str(exc))
        return out_path
    write_result(out_path, ident, "ok", **fields)
    return out_path


def _spawn_condition(
    condition: AttnCondition, *, out_dir: Path, session_id: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable, str(_SCRIPT_PATH),
        "--impl", condition.impl,
        "--seq-lens", str(condition.n),
        "--head-dim", str(condition.head_dim),
        "--heads", str(condition.heads),
        "--kv-heads", str(condition.kv_heads),
        "--out-dir", str(out_dir),
        "--session-id", session_id,
        "--single-condition",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_grid(
    conditions: list[AttnCondition], *, out_dir: Path, session_id: str,
) -> list[Path]:
    """Subprocess-per-condition orchestration -- same shape as `bench.runner.
    run_conditions`: a fresh artifact is skipped without spawning; a stale one is
    unlinked BEFORE spawning (so `out_path.exists()` after the subprocess returns means
    exactly "this subprocess wrote it"); a nonzero exit or a clean exit that wrote
    nothing is recorded as an `"error"` result on this side, keyed by the identity the
    subprocess would have used, so a later resume run still sees it as stale."""
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
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
            write_result(
                out_path, ident, "error", error_type="WorkerCrashed",
                error_msg=stderr_tail, returncode=proc.returncode,
            )
        elif not out_path.exists():
            write_result(
                out_path, ident, "error", error_type="WorkerExitedWithoutArtifact",
                error_msg="subprocess exited 0 without writing an artifact", returncode=0,
            )
    return paths


def _read_ok_entries(paths: list[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") == "ok":
            entries.append(data)
    return entries


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="+", choices=("flash", "stock"),
                    default=("flash", "stock"))
    ap.add_argument("--seq-lens", nargs="+", type=int, default=list(DEFAULT_SEQ_LENS))
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    ap.add_argument("--out-dir", type=Path, default=None)
    # Internal self-reinvocation surface (subprocess-per-condition -- see the module
    # docstring): not part of the documented top-level CLI, so both are suppressed from
    # --help. `--session-id` lets every condition spawned by ONE top-level invocation
    # share an identity session; `--single-condition` marks "run exactly the one
    # (impl, n) pair these args describe, in-process, no further spawn."
    ap.add_argument("--session-id", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single-condition", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    session_id = args.session_id or new_session_id()

    if args.single_condition:
        if len(args.impl) != 1 or len(args.seq_lens) != 1:
            raise SystemExit(
                "--single-condition requires exactly one --impl and one --seq-lens value"
            )
        condition = build_conditions(
            impls=args.impl, seq_lens=args.seq_lens, head_dim=args.head_dim,
            heads=args.heads, kv_heads=args.kv_heads,
        )[0]
        out_path = _run_single_condition(condition, out_dir=out_dir, session_id=session_id)
        data = json.loads(out_path.read_text())
        print(json.dumps(data, indent=2))
        return 0 if data.get("status") != "error" else 1

    conditions = build_conditions(
        impls=args.impl, seq_lens=args.seq_lens, head_dim=args.head_dim,
        heads=args.heads, kv_heads=args.kv_heads,
    )
    paths = run_grid(conditions, out_dir=out_dir, session_id=session_id)
    ratios = compute_doubling_ratios(_read_ok_entries(paths))
    print(json.dumps({"ratios": ratios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
