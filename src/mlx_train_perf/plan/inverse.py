"""Inverse planner queries -- given a `TrainConfig` with every other field held fixed,
find the largest `seq_len` or `batch` whose predicted peak (`estimate_peak`) still fits a
memory budget.

Every component `estimate_peak` sums is non-decreasing in `n = batch * seq_len` (weights,
lora, and optimizer are batch/seq_len-independent; activations, attention, and loss all
grow with `batch` and/or `seq_len`, never shrink), so for either search variable held
fixed at any value of the other, the predicted peak is monotonically non-decreasing.
Monotonic bisection over the search variable is therefore well-founded: once a value no
longer fits, no larger value does either.
"""
from dataclasses import replace

import mlx.core as mx

from mlx_train_perf.core.guards import clamped_caps
from mlx_train_perf.errors import DoesNotFitError
from mlx_train_perf.plan.calibration import Calibration, load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak


def _peak(shape: ModelShape, cfg: TrainConfig, calib: Calibration) -> int:
    """`estimate_peak(...)[0]` alone -- the bisections below only need the scalar
    predicted peak, never the component breakdown."""
    peak, _ = estimate_peak(shape, cfg, calib)
    return peak


def _resolve_budget_bytes(budget_bytes: int | None) -> int:
    """Mirrors `plan_fit`'s own budget resolution exactly: when the caller passes none,
    default to THIS project's own guarded wired cap (`core.guards.clamped_caps`,
    device-clamped) -- the conservative budget our own benches run under. This is
    deliberately NOT what stock `mlx_lm.tuner.trainer.train()` enforces: it calls
    `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` at entry
    (verified against the installed mlx-lm==0.31.3 source, `tuner/trainer.py:229-230`),
    overriding any stricter cap to the raw device max. A caller planning specifically for
    the stock trainer's own path should pass
    `budget_bytes=int(mx.device_info()["max_recommended_working_set_size"])` explicitly
    for an honest stock-trainer budget rather than relying on this stricter default."""
    if budget_bytes is not None:
        return budget_bytes
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _ = clamped_caps(dev_max)
    return wired


def max_seq_len_for_budget(
    shape: ModelShape, cfg: TrainConfig, *, budget_bytes: int | None = None,
    seq_ceiling: int = 65536,
) -> int:
    """Largest `seq_len` (every other `cfg` field, including `batch`, held fixed) whose
    predicted peak fits `budget_bytes`, found by monotonic bisection over
    `[1, seq_ceiling]`. See the module docstring for why bisection is well-founded here.

    `budget_bytes` resolution mirrors `plan_fit` exactly -- see `_resolve_budget_bytes`.

    Raises `DoesNotFitError` if the config does not fit even at the floor
    (`seq_len=1`) -- never silently returns 0. If the config still fits at
    `seq_ceiling`, returns `seq_ceiling` (documented saturation, not a search failure).
    """
    calib = load_calibration()
    budget = _resolve_budget_bytes(budget_bytes)
    floor_peak = _peak(shape, replace(cfg, seq_len=1), calib)
    if floor_peak > budget:
        raise DoesNotFitError(
            f"no seq_len >= 1 fits budget_bytes={budget} (predicted peak at seq_len=1 "
            f"is {floor_peak} bytes)"
        )
    lo, hi = 1, seq_ceiling
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _peak(shape, replace(cfg, seq_len=mid), calib) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def max_batch_for_budget(
    shape: ModelShape, cfg: TrainConfig, *, budget_bytes: int | None = None,
    batch_ceiling: int = 4096,
) -> int:
    """Largest `batch` (every other `cfg` field, including `seq_len`, held fixed) whose
    predicted peak fits `budget_bytes`, found by monotonic bisection over
    `[1, batch_ceiling]`. See the module docstring for why bisection is well-founded here.

    `budget_bytes` resolution mirrors `plan_fit` exactly -- see `_resolve_budget_bytes`.

    Raises `DoesNotFitError` if the config does not fit even at the floor (`batch=1`) --
    never silently returns 0. If the config still fits at `batch_ceiling`, returns
    `batch_ceiling` (documented saturation, not a search failure).
    """
    calib = load_calibration()
    budget = _resolve_budget_bytes(budget_bytes)
    floor_peak = _peak(shape, replace(cfg, batch=1), calib)
    if floor_peak > budget:
        raise DoesNotFitError(
            f"no batch >= 1 fits budget_bytes={budget} (predicted peak at batch=1 is "
            f"{floor_peak} bytes)"
        )
    lo, hi = 1, batch_ceiling
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _peak(shape, replace(cfg, batch=mid), calib) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo
