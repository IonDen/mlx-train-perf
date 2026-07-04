"""Calibration constants for the fit planner.

Loaded from `calibration_data.json`, a versioned data file that carries its own
measurement provenance (machine, macOS, mlx version, measured date) so a `FitReport`
can always show where its non-analytic constants came from. 0.1.0 ships HONEST
placeholder values -- the file's own `provenance.measured_date` says so explicitly;
a later task replaces them with numbers measured on real training runs.
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
    provenance: dict[str, str]


def load_calibration() -> Calibration:
    raw = json.loads(resources.files(_PACKAGE).joinpath(_DATA_FILE).read_text())
    return Calibration(
        act_bytes_per_token_layer=float(raw["act_bytes_per_token_layer"]),
        optimizer_bytes_per_param=float(raw["optimizer_bytes_per_param"]),
        overhead_frac=float(raw["overhead_frac"]),
        provenance=dict(raw["provenance"]),
    )
