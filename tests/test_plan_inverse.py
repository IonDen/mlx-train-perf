from dataclasses import replace

import mlx.core as mx
import pytest

from mlx_train_perf.core.guards import clamped_caps
from mlx_train_perf.errors import DoesNotFitError
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak
from mlx_train_perf.plan.inverse import max_batch_for_budget, max_seq_len_for_budget

# Same micro "llama" shape used throughout test_plan.py: small enough that every
# component of estimate_peak is cheap to recompute inside a bisection loop.
_SHAPE = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                    kv_heads=2, tied=False, quant_bits=None, quant_group=None)

_ATTENTION_GC_GRID = [
    (attention, grad_checkpoint)
    for attention in ("stock", "flash")
    for grad_checkpoint in (True, False)
]

# (label, budget_bytes, ceiling): "interior" budgets sit well below what either
# attention arm costs at the full-size default ceiling (stock's O(N^2) term and
# flash's O(N) term both blow past 5 GiB by seq_len=65536 / batch=4096 at this shape),
# so bisection must land strictly inside the range. "saturated" pairs a tiny ceiling
# with a huge budget so even the ceiling config is cheap enough to fit -- the
# saturation return path (no crossing point within range).
_SEQ_BUDGETS = [
    ("interior", 5 * 1024**3, 65536),
    ("saturated", 200 * 1024**3, 16),
]
_BATCH_BUDGETS = [
    ("interior", 5 * 1024**3, 4096),
    ("saturated", 200 * 1024**3, 8),
]

# Absurdly small: below even the fixed ~1.49 GiB base_transient_bytes, so the config
# cannot fit at the floor (seq_len=1 / batch=1) regardless of any other field.
_DEGENERATE_BUDGET = 1000


def _cfg(
    *, batch: int = 1, seq_len: int = 512, grad_checkpoint: bool, attention: str
) -> TrainConfig:
    return TrainConfig(batch=batch, seq_len=seq_len, dtype="bfloat16", lora_rank=8,
                       lora_layers=2, grad_checkpoint=grad_checkpoint, impl="kernel",
                       attention=attention)


def _peak_bytes(cfg: TrainConfig) -> int:
    calib = load_calibration()
    peak, _ = estimate_peak(_SHAPE, cfg, calib)
    return peak


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
@pytest.mark.parametrize(("label", "budget_bytes", "seq_ceiling"), _SEQ_BUDGETS)
def test_max_seq_len_for_budget_consistency(
    label: str, budget_bytes: int, seq_ceiling: int, attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(batch=1, grad_checkpoint=grad_checkpoint, attention=attention)
    v = max_seq_len_for_budget(_SHAPE, cfg, budget_bytes=budget_bytes, seq_ceiling=seq_ceiling)
    assert 1 <= v <= seq_ceiling
    assert _peak_bytes(replace(cfg, seq_len=v)) <= budget_bytes
    if v < seq_ceiling:
        assert _peak_bytes(replace(cfg, seq_len=v + 1)) > budget_bytes
    else:
        assert label == "saturated"


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
@pytest.mark.parametrize(("label", "budget_bytes", "batch_ceiling"), _BATCH_BUDGETS)
def test_max_batch_for_budget_consistency(
    label: str, budget_bytes: int, batch_ceiling: int, attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(seq_len=512, grad_checkpoint=grad_checkpoint, attention=attention)
    v = max_batch_for_budget(_SHAPE, cfg, budget_bytes=budget_bytes, batch_ceiling=batch_ceiling)
    assert 1 <= v <= batch_ceiling
    assert _peak_bytes(replace(cfg, batch=v)) <= budget_bytes
    if v < batch_ceiling:
        assert _peak_bytes(replace(cfg, batch=v + 1)) > budget_bytes
    else:
        assert label == "saturated"


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
def test_max_seq_len_for_budget_refuses_at_degenerate_budget(
    attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(batch=1, grad_checkpoint=grad_checkpoint, attention=attention)
    with pytest.raises(DoesNotFitError):
        max_seq_len_for_budget(_SHAPE, cfg, budget_bytes=_DEGENERATE_BUDGET)


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
def test_max_batch_for_budget_refuses_at_degenerate_budget(
    attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(seq_len=512, grad_checkpoint=grad_checkpoint, attention=attention)
    with pytest.raises(DoesNotFitError):
        max_batch_for_budget(_SHAPE, cfg, budget_bytes=_DEGENERATE_BUDGET)


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
def test_max_seq_len_for_budget_monotonic_in_budget(
    attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(batch=1, grad_checkpoint=grad_checkpoint, attention=attention)
    small = max_seq_len_for_budget(_SHAPE, cfg, budget_bytes=3 * 1024**3)
    large = max_seq_len_for_budget(_SHAPE, cfg, budget_bytes=50 * 1024**3)
    assert large >= small


@pytest.mark.parametrize(("attention", "grad_checkpoint"), _ATTENTION_GC_GRID)
def test_max_batch_for_budget_monotonic_in_budget(
    attention: str, grad_checkpoint: bool
) -> None:
    cfg = _cfg(seq_len=512, grad_checkpoint=grad_checkpoint, attention=attention)
    small = max_batch_for_budget(_SHAPE, cfg, budget_bytes=3 * 1024**3)
    large = max_batch_for_budget(_SHAPE, cfg, budget_bytes=50 * 1024**3)
    assert large >= small


def test_max_seq_len_for_budget_uses_default_clamped_budget_like_plan_fit() -> None:
    """No `budget_bytes` -- resolves via the same `core.guards.clamped_caps` default
    `plan_fit` uses, so a huge (unreachable) budget and an omitted budget behave the
    same at this tiny micro shape (both saturate at the ceiling)."""
    cfg = _cfg(batch=1, grad_checkpoint=True, attention="stock")
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _ = clamped_caps(dev_max)
    default_result = max_seq_len_for_budget(_SHAPE, cfg, seq_ceiling=32)
    explicit_result = max_seq_len_for_budget(_SHAPE, cfg, budget_bytes=wired, seq_ceiling=32)
    assert default_result == explicit_result
