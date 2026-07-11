"""Command-line interface: `plan` (RAM-fit check) and `bench` (loss-layer harness).

Two subcommands only. `plan` renders a `FitReport` for a given model config + training
shape (exit 0 fits, 1 does not fit, 2 tool error). `bench` drives the subprocess-per-
condition bench runner over the `loss_layer` suite (exit 0 all conditions ok, 1 any
condition not ok, 2 tool error). Argument parsing and rendering are kept in small pure
functions; `main` and the `_cmd_*` handlers are thin glue around them.
"""
import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import Condition, report, run_conditions
from mlx_train_perf.errors import BenchInputError, MlxTrainPerfError, PlanInputError
from mlx_train_perf.plan.estimate import FitReport, ModelShape, TrainConfig, plan_fit

Command = Callable[[argparse.Namespace], int]

_BENCH_SUITES = ("loss-layer",)
_PLAN_IMPLS = ("kernel", "chunked", "naive")
_BENCH_IMPLS = ("auto", "kernel", "chunked", "naive")
_DTYPES = ("float32", "bfloat16", "float16")
_ATTENTIONS = ("stock", "flash")

# Fixed pairing this project uses everywhere a quantized shape is priced (worker.py's
# own quantized-condition default, and the planner's own quantized test fixture): a
# `--quant-bits` override with no group size of its own gets this default group.
_DEFAULT_QUANT_GROUP = 64


# --- plan: pure helpers --------------------------------------------------------------


def _load_model_shape(config_path: str) -> ModelShape:
    """Reads a HF `config.json` and builds a `ModelShape` from it. Both a missing file
    and invalid JSON are reported as a `PlanInputError` -- a tool-level input problem,
    not a program crash."""
    path = Path(config_path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise PlanInputError(f"cannot read config file {config_path!r}: {exc}") from exc
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanInputError(f"config file {config_path!r} is not valid JSON: {exc}") from exc
    return ModelShape.from_config(config)


def _apply_quant_override(shape: ModelShape, quant_bits: int | None) -> ModelShape:
    """`None` is a passthrough (the shape's own config-derived quantization metadata, if
    any, is left alone). A given `quant_bits` overrides the bit width; the group size is
    preserved from the config when the shape already carries one (a config.json with its
    own `quantization.group_size` is ground truth and must not be silently overwritten --
    e.g. a real gs=32 checkpoint flipped to the fixed gs=64 default would understate the
    quantized weights bytes and produce an over-optimistic fit verdict). Only a shape
    with NO quantization metadata at all (`quant_group is None`) falls back to the
    project's fixed default group size, since pricing at the quantized rate needs both
    `quant_bits` and `quant_group` set (see `estimate._weights_bytes`)."""
    if quant_bits is None:
        return shape
    quant_group = shape.quant_group if shape.quant_group is not None else _DEFAULT_QUANT_GROUP
    return replace(shape, quant_bits=quant_bits, quant_group=quant_group)


def _train_config_from_args(
    *, batch: int, seq_len: int, lora_rank: int, impl: str, shape_layers: int,
    attention: str = "stock",
) -> TrainConfig:
    """`lora_rank == 0` means full fine-tuning in this planner's terms (see
    `estimate._head_trainable`) -- no LoRA layers apply, so `lora_layers` is 0. Otherwise
    every model layer gets a LoRA adapter, the conventional default this planner already
    assumes for its `q_proj`/`v_proj` target set. `attention` selects the stock O(N^2) or
    flash O(N) attention-backward memory model (default "stock")."""
    lora_layers = shape_layers if lora_rank > 0 else 0
    return TrainConfig(
        batch=batch, seq_len=seq_len, dtype="bfloat16", lora_rank=lora_rank,
        lora_layers=lora_layers, grad_checkpoint=True, impl=impl, attention=attention,
    )


def _render_plan_json(fit_report: FitReport) -> str:
    return json.dumps(asdict(fit_report), indent=2)


def _render_plan_text(fit_report: FitReport) -> str:
    gib = 1024**3
    lines = [
        f"fits: {'yes' if fit_report.fits else 'no'}",
        f"predicted peak: {fit_report.predicted_peak_bytes / gib:.3f} GiB",
        f"budget: {fit_report.budget_bytes / gib:.3f} GiB",
        f"headroom: {fit_report.headroom_bytes / gib:+.3f} GiB",
        "components:",
    ]
    lines += [f"  {name}: {nbytes / gib:.3f} GiB" for name, nbytes in fit_report.components.items()]
    lines.append(f"is_estimate: {fit_report.is_estimate}")
    lines.append("provenance:")
    lines += [f"  {key}: {value}" for key, value in fit_report.provenance.items()]
    if fit_report.suggestion is not None:
        s = fit_report.suggestion
        lines.append(
            f"suggestion: batch={s.batch} seq_len={s.seq_len} lora_rank={s.lora_rank} "
            f"lora_layers={s.lora_layers} impl={s.impl}"
        )
    return "\n".join(lines)


def _plan_exit_code(fit_report: FitReport) -> int:
    return 0 if fit_report.fits else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    shape = _load_model_shape(args.config)
    shape = _apply_quant_override(shape, args.quant_bits)
    cfg = _train_config_from_args(
        batch=args.batch, seq_len=args.seq_len, lora_rank=args.lora_rank, impl=args.impl,
        shape_layers=shape.layers, attention=args.attention,
    )
    budget_bytes = int(args.budget_gb * 1024**3) if args.budget_gb is not None else None
    fit_report = plan_fit(shape, cfg, budget_bytes=budget_bytes)
    rendered = _render_plan_json(fit_report) if args.json else _render_plan_text(fit_report)
    print(rendered)
    return _plan_exit_code(fit_report)


# --- bench: pure helpers --------------------------------------------------------------


def _conditions_for_suite(
    suite: str, *, n_values: list[int], d: int, v: int, dtype: str, impl: str,
    quantized: bool, group_size: int, bits: int, chunk_size: int | None, reps: int,
    seed: int,
) -> list[Condition]:
    """Builds one `loss_layer` condition per `n`. 0.1.0 supports exactly one suite; an
    unsupported name is a typed error rather than a silently empty sweep."""
    if suite not in _BENCH_SUITES:
        raise BenchInputError(
            f"unsupported bench suite {suite!r}; only {_BENCH_SUITES!r} implemented in 0.1.0"
        )
    conditions = []
    for n in n_values:
        params: dict[str, object] = {
            "n": n, "d": d, "v": v, "dtype": dtype, "impl": impl, "quantized": quantized,
            "group_size": group_size, "bits": bits, "chunk_size": chunk_size, "reps": reps,
            "seed": seed,
        }
        conditions.append(Condition(name=f"loss_layer_n{n}", kind="loss_layer", params=params))
    return conditions


def _read_status(path: Path) -> str:
    """A missing or corrupt artifact reads as `"error"` -- the same conservative
    direction `bench/artifacts.result_is_fresh` already takes for staleness."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "error"
    return str(data.get("status", "error"))


def _bench_exit_code(statuses: list[str]) -> int:
    return 0 if all(status == "ok" for status in statuses) else 1


def _render_bench_summary(paths: list[Path], statuses: list[str]) -> str:
    summary: dict[str, object] = {
        "conditions": [
            {"name": path.stem, "status": status, "path": str(path)}
            for path, status in zip(paths, statuses, strict=True)
        ],
        **report(paths),
    }
    return json.dumps(summary, indent=2)


def _cmd_bench(args: argparse.Namespace) -> int:
    conditions = _conditions_for_suite(
        args.suite, n_values=args.n, d=args.d, v=args.v, dtype=args.dtype, impl=args.impl,
        quantized=args.quantized, group_size=args.group_size, bits=args.bits,
        chunk_size=args.chunk_size, reps=args.reps, seed=args.seed,
    )
    paths = run_conditions(conditions, Path(args.out), session_id=new_session_id())
    statuses = [_read_status(path) for path in paths]
    print(_render_bench_summary(paths, statuses))
    return _bench_exit_code(statuses)


# --- argument parser -------------------------------------------------------------------


def _add_plan_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    plan = subparsers.add_parser("plan", help="check whether a training config fits the RAM budget")
    plan.add_argument("--config", required=True, help="path to a HF config.json")
    plan.add_argument("--batch", type=int, required=True, help="per-step batch size")
    plan.add_argument("--seq-len", type=int, required=True, help="sequence length")
    plan.add_argument("--lora-rank", type=int, required=True,
                       help="LoRA rank (0 means full fine-tuning)")
    plan.add_argument("--quant-bits", type=int, default=None,
                       help="assume this per-parameter bit width (overrides the config's "
                            "own quantization metadata, if any)")
    plan.add_argument("--impl", choices=_PLAN_IMPLS, default="kernel",
                       help="loss-layer implementation to plan for (default: kernel)")
    plan.add_argument("--attention", choices=_ATTENTIONS, default="stock",
                       help="attention-backward memory model: stock O(N^2) (default) or "
                            "the 0.2.0 flash O(N) path")
    plan.add_argument("--budget-gb", type=float, default=None,
                       help="memory budget in GiB (default: this project's own device-"
                            "clamped wired cap)")
    plan.add_argument("--json", action="store_true", help="render the FitReport as JSON")
    plan.set_defaults(func=_cmd_plan)


def _add_bench_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    bench = subparsers.add_parser(
        "bench", help="run the benchmark harness",
        description="Run the benchmark harness. Exit 1 covers both an 'error' and a "
                     "'refused' condition status -- neither is a clean 'ok' result.",
    )
    bench.add_argument("--suite", required=True, choices=_BENCH_SUITES, help="bench suite")
    bench.add_argument("--out", required=True, help="output directory for result artifacts")
    bench.add_argument("--n", type=int, nargs="+", default=[512, 2048, 8192],
                        help="row counts (batch*seq_len) to sweep")
    bench.add_argument("--d", type=int, default=4096, help="hidden dim")
    bench.add_argument("--v", type=int, default=151936, help="vocab size")
    bench.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    bench.add_argument("--impl", choices=_BENCH_IMPLS, default="auto")
    bench.add_argument("--quantized", action="store_true", help="build a quantized head")
    bench.add_argument("--group-size", type=int, default=64, help="quantization group size")
    bench.add_argument("--bits", type=int, default=4, help="quantization bit width")
    bench.add_argument("--chunk-size", type=int, default=None,
                        help="chunked-impl vocab tile size (default: chunked's own default)")
    bench.add_argument("--reps", type=int, default=3, help="timed repetitions per condition")
    bench.add_argument("--seed", type=int, default=0)
    bench.set_defaults(func=_cmd_bench)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-train-perf",
        description="RAM-fit planner and benchmark harness for MLX fine-tuning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_plan_parser(subparsers)
    _add_bench_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the command-line interface and returns a process exit code (never raises
    `SystemExit` itself, so it composes as a plain function in tests and other callers)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    func = cast(Command, args.func)
    try:
        return func(args)
    except MlxTrainPerfError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
