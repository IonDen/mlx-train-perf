"""Calibration constants for the fit planner.

Loaded from `calibration_data.json`, a versioned data file that carries its own
measurement provenance (machine, macOS, mlx version, measured date) so a `FitReport`
can always show where its non-analytic constants came from. Most of 0.1.0's constants
are HONEST placeholders -- the file's own `provenance.measured_date` says so explicitly
-- but `naive_loss_bytes_per_nv` is a real measured value: see its field docstring below
for the derivation against a persisted benchmark artifact.
"""
import json
from dataclasses import dataclass
from importlib import resources

_PACKAGE = "mlx_train_perf.plan"
_DATA_FILE = "calibration_data.json"


@dataclass(frozen=True, slots=True, kw_only=True)
class Calibration:
    act_bytes_per_token_layer: float
    optimizer_bytes_per_param: float
    overhead_frac: float
    # Bytes per (row, vocab) element for the naive loss impl's base term (before the
    # separate d_w term). Measured (task-13 review item 3) against
    # mlx-train-perf-spike/results/gate_naive_n8192.json (reference-only artifact, never
    # executed by this project): that gate ran naive_linear_ce under
    # mx.value_and_grad(argnums=(0, 1)) (trainable head) at n=8192, V=151936, D=4096,
    # bf16 hidden, measuring marginal_peak_gb=18.547 (GiB). Converting to bytes
    # (18.547 * 1024**3 ~= 19,914,689,610) and holding the d_w term (V*D*4*2 =
    # 4,978,638,848, unchanged by this item) fixed, the remaining ~14,936,050,762 bytes
    # divided by n*V = 1,244,659,712 gives ~12.0 bytes per (n, V) pair -- 3 fp32
    # (N,V)-shaped buffers under MLX's naive autodiff (logits, softmax probabilities,
    # d_logits), not the 2 the brief's literal `n*V*4*2` formula assumed (~1.9x too low
    # at production shape).
    naive_loss_bytes_per_nv: float
    provenance: dict[str, str]


def load_calibration() -> Calibration:
    raw = json.loads(resources.files(_PACKAGE).joinpath(_DATA_FILE).read_text())
    return Calibration(
        act_bytes_per_token_layer=float(raw["act_bytes_per_token_layer"]),
        optimizer_bytes_per_param=float(raw["optimizer_bytes_per_param"]),
        overhead_frac=float(raw["overhead_frac"]),
        naive_loss_bytes_per_nv=float(raw["naive_loss_bytes_per_nv"]),
        provenance=dict(raw["provenance"]),
    )
