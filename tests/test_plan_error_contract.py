"""The planner's refusal contract: every bad input surfaces as `PlanInputError`
(a `MlxTrainPerfError` and a `MemfitInputError`), never a bare memfit error or a number."""
from dataclasses import replace

import pytest

from mlx_train_perf.errors import MlxTrainPerfError, PlanInputError
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak
from mlx_train_perf.plan.inverse import max_batch_for_budget, max_seq_len_for_budget

_SHAPE = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                    kv_heads=2, tied=False, quant_bits=None, quant_group=None)
_CFG = TrainConfig(batch=1, seq_len=512, dtype="bfloat16", lora_rank=8, lora_layers=2,
                   grad_checkpoint=True, impl="kernel", attention="stock")
_BUDGET = 200 * 1024**3


@pytest.mark.parametrize(("field", "value"), [
    ("batch", 0), ("batch", -1), ("seq_len", 0), ("seq_len", -5), ("lora_rank", -1),
])
def test_out_of_range_config_is_a_plan_input_error(field: str, value: int) -> None:
    """Bug caught: a negative value reached a memfit term and escaped as a bare
    MemfitInputError, and batch=0 returned a number."""
    with pytest.raises(PlanInputError, match=field) as info:
        estimate_peak(_SHAPE, replace(_CFG, **{field: value}), load_calibration())
    assert isinstance(info.value, MlxTrainPerfError)


def test_lora_rank_zero_is_still_accepted() -> None:
    peak, _ = estimate_peak(_SHAPE, replace(_CFG, lora_rank=0), load_calibration())
    assert peak > 0


def test_dtype_error_wins_over_attention_error() -> None:
    """Bug caught: wrapping term errors (or validating attention up front) would change
    which of two bad fields the caller hears about first."""
    cfg = replace(_CFG, dtype="float7", attention="sideways")
    with pytest.raises(PlanInputError, match="float7"):
        estimate_peak(_SHAPE, cfg, load_calibration())


def test_seq_ceiling_below_one_is_a_plan_input_error() -> None:
    with pytest.raises(PlanInputError, match="seq_ceiling"):
        max_seq_len_for_budget(_SHAPE, _CFG, budget_bytes=_BUDGET, seq_ceiling=0)


def test_batch_ceiling_below_one_is_a_plan_input_error() -> None:
    with pytest.raises(PlanInputError, match="batch_ceiling"):
        max_batch_for_budget(_SHAPE, _CFG, budget_bytes=_BUDGET, batch_ceiling=-3)
