from test_memfit_imports import run_with_blocked_imports


def test_planner_estimates_without_mlx() -> None:
    """Bug caught: a module-level `import mlx.core` (or `core.guards`, which imports it)
    in plan/estimate.py or plan/inverse.py."""
    code = """
from mlx_train_perf.plan.calibration import load_calibration
from mlx_train_perf.plan.estimate import ModelShape, TrainConfig, estimate_peak
from mlx_train_perf.plan.inverse import max_seq_len_for_budget
shape = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4, kv_heads=2,
                   tied=False, quant_bits=None, quant_group=None)
cfg = TrainConfig(batch=1, seq_len=256, dtype="bfloat16", lora_rank=8, lora_layers=2,
                  grad_checkpoint=True, impl="kernel")
peak, _ = estimate_peak(shape, cfg, load_calibration())
print(peak > 0, max_seq_len_for_budget(shape, cfg, budget_bytes=peak * 4) > 256)
"""
    r = run_with_blocked_imports(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True True"
