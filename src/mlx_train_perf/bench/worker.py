"""Bench worker: the subprocess entry point for exactly one `Condition`, invoked by
`runner.run_conditions` as `python -m mlx_train_perf.bench.worker --config <path>`.

Guardrail-reassert note for the FUTURE `train_step` condition kind (Task 17 -- recorded
here now so that task doesn't have to re-derive it): `mlx_lm.tuner.trainer.train()` raises
the wired limit to the device max AT ENTRY
(site-packages/mlx_lm/tuner/trainer.py:229-230) and then blocks until the training loop
finishes. A worker that calls `install_guardrails()` once before `train()` and again after
it returns protects nothing in between -- the stricter cap is silently overridden for the
entire run. The re-assert has to live INSIDE the loop: the loss callable handed to
`train(...)` calls `install_guardrails()` on its own first invocation (a one-shot flag),
and the worker records the OBSERVED wired limit mid-run in its artifact so a cap
regression is visible rather than assumed away (drive the step loop manually instead of
calling `train()` if that callback point proves awkward). 0.1.0 only implements the
`loss_layer` kind below, which never calls `train()`, so this hazard does not apply yet.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from mlx_train_perf.bench.artifacts import run_identity, write_result
from mlx_train_perf.core.guards import install_guardrails
from mlx_train_perf.core.loss import DenseHead, HeadRef, QuantizedHead, linear_cross_entropy
from mlx_train_perf.errors import LaunchBudgetError, MlxTrainPerfError

_DTYPES: dict[str, mx.Dtype] = {
    "float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16,
}


def _resolve_dtype(name: str) -> mx.Dtype:
    if name not in _DTYPES:
        raise MlxTrainPerfError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def _build_head(*, v: int, d: int, dtype: mx.Dtype, quantized: bool, group_size: int,
                bits: int, seed: int) -> HeadRef:
    mx.random.seed(seed)
    w = (mx.random.normal((v, d)) * 0.02).astype(dtype)
    mx.eval(w)
    if not quantized:
        return DenseHead(weight=w)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(w_q, scales, biases)
    return QuantizedHead(w_q=w_q, scales=scales, biases=biases, group_size=group_size, bits=bits)


def run_loss_layer(params: dict[str, object]) -> dict[str, object]:
    """Times `linear_cross_entropy` at one synthetic grid point. Reset-peak semantics
    (warmup pays Metal JIT OUTSIDE the measured window; `active_before` is snapshotted
    right before the reset so `marginal_peak_gb` is the incremental cost of the forward
    passes themselves) -- the same convention `scripts/bench_quant_thresholds.py` uses."""
    n = int(cast(int, params["n"]))
    d = int(cast(int, params["d"]))
    v = int(cast(int, params["v"]))
    dtype = _resolve_dtype(str(params.get("dtype", "bfloat16")))
    impl = cast(Literal["auto", "kernel", "chunked", "naive"], params.get("impl", "auto"))
    quantized = bool(params.get("quantized", False))
    group_size = int(cast(int, params.get("group_size", 64)))
    bits = int(cast(int, params.get("bits", 4)))
    chunk_size = cast(int | None, params.get("chunk_size"))
    reps = int(cast(int, params.get("reps", 3)))
    seed = int(cast(int, params.get("seed", 0)))

    mx.random.seed(seed)
    hidden = mx.random.normal((n, d)).astype(dtype)
    targets = mx.random.randint(0, v, (n,))
    mx.eval(hidden, targets)
    head = _build_head(v=v, d=d, dtype=dtype, quantized=quantized, group_size=group_size,
                       bits=bits, seed=seed + 1)

    def run_once() -> mx.array:
        return linear_cross_entropy(hidden, head, targets, impl=impl, chunk_size=chunk_size,
                                    reduction="mean")

    loss = run_once()
    mx.eval(loss)
    mx.clear_cache()
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    walls: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        loss = run_once()
        mx.eval(loss)
        walls.append(time.perf_counter() - t0)
    marginal_peak_gb = (mx.get_peak_memory() - active_before) / 1024**3
    med = statistics.median(walls)
    g_mac_per_s = (n * v * d) / med / 1e9
    return {
        "wall_s": round(med, 6),
        "wall_s_all": [round(x, 6) for x in walls],
        "g_mac_per_s": round(g_mac_per_s, 3),
        "active_before_gb": round(active_before / 1024**3, 4),
        "marginal_peak_gb": round(marginal_peak_gb, 4),
        "total_peak_gb": round(active_before / 1024**3 + marginal_peak_gb, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mlx_train_perf.bench.worker")
    ap.add_argument("--config", required=True, help="path to a JSON condition config")
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    kind = str(config["kind"])
    params = cast(dict[str, object], config["params"])
    session_id = str(config["session_id"])
    out = Path(cast(str, config["out"]))

    install_guardrails()  # FIRST -- before any allocation this condition makes

    ident = run_identity(kind=kind, session_id=session_id, **params)
    if kind != "loss_layer":
        # Deliberately uncaught: an unsupported kind is a program error (a bad Condition
        # was constructed), not a recorded run outcome -- it crashes this worker process
        # with a nonzero exit, and `runner.run_conditions` records the failure envelope
        # on the CALLER's side instead of this worker writing anything. Referencing the
        # bare `run_loss_layer` name below (not a dict bound at import time) also keeps
        # this dispatch monkeypatch-friendly for tests.
        raise MlxTrainPerfError(
            f"unsupported bench condition kind {kind!r}; only 'loss_layer' is implemented "
            "in 0.1.0 ('train_step' lands in a later task)"
        )
    try:
        fields = run_loss_layer(params)
    except LaunchBudgetError as exc:
        # A guard refusal IS a result: the calibrated rate cannot serve this shape within
        # the watchdog budget -- record it, don't crash the sweep.
        write_result(out, ident, "refused", error=str(exc))
        return 0
    write_result(out, ident, "ok", **fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
