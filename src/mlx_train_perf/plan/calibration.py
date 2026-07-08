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
    # Fixed per-run framework/base transient (MLX runtime + model buffers) that does not
    # scale with seq -- a real ~1.5 GB floor, measured, not the tiny analytic lora/opt terms.
    base_transient_bytes: float
    # Linear activation memory, bytes per (token * hidden * layer). gc-aware: `_ckpt` is the
    # grad_checkpoint=True case (only layer-boundary residuals stored); `_full` is
    # grad_checkpoint=False (every layer's forward activations stored, ~45x larger).
    act_bytes_per_token_hidden_layer_ckpt: float
    act_bytes_per_token_hidden_layer_full: float
    # Quadratic attention-backward memory, bytes per (head * seq^2). ONE layer for BOTH gc
    # settings (mlx's O(N^2) SDPA backward materializes one (N,N) at a time as the backward
    # walks the stack). bf16-calibrated (dtype folded into the coefficient).
    attn_bytes_per_head_token2: float
    # AdamW analytic: two fp32 moments per trainable param == 8 bytes/param (not fitted).
    optimizer_bytes_per_param: float
    overhead_frac: float
    # Bytes per (row, vocab) element for the naive loss impl's base term (before the
    # separate d_w term). This is an EMPIRICAL FIT to a single measured production-shape
    # anchor: a measured production-shape gate at n=8192, V=151936, D=4096, bf16 hidden,
    # trainable head, marginal_peak_gb=18.547 (GiB). Converting to bytes
    # (~19,914,689,610) and holding the d_w term (V*D*4*2 = 4,978,638,848) fixed, the
    # remainder divided by n*V = 1,244,659,712 gives ~12.0 bytes per (n, V) pair.
    #
    # This is NOT a validated buffer-by-buffer decomposition. At this exact anchor shape
    # 2*D == n, so V*D*4*2 == n*V*4 exactly -- the split between "the d_w term" and "the
    # n*V coefficient" is numerically unidentifiable from this one point alone. The fit
    # also does not extrapolate linearly to other n: the sibling n=2048 measurement (same
    # code path) is marginal_peak_gb=4.057,
    # while this coefficient plus the fixed d_w term predicts ~8.11 GiB there -- about
    # 2x too high. At n=8192 (this project's flagship shape) the estimate is accurate;
    # at smaller n it over-predicts the naive path's cost, which is the conservative
    # (safe) direction for a planner steering callers away from naive.
    naive_loss_bytes_per_nv: float
    provenance: dict[str, str]


def load_calibration() -> Calibration:
    raw = json.loads(resources.files(_PACKAGE).joinpath(_DATA_FILE).read_text())
    return Calibration(
        base_transient_bytes=float(raw["base_transient_bytes"]),
        act_bytes_per_token_hidden_layer_ckpt=float(raw["act_bytes_per_token_hidden_layer_ckpt"]),
        act_bytes_per_token_hidden_layer_full=float(raw["act_bytes_per_token_hidden_layer_full"]),
        attn_bytes_per_head_token2=float(raw["attn_bytes_per_head_token2"]),
        optimizer_bytes_per_param=float(raw["optimizer_bytes_per_param"]),
        overhead_frac=float(raw["overhead_frac"]),
        naive_loss_bytes_per_nv=float(raw["naive_loss_bytes_per_nv"]),
        provenance=dict(raw["provenance"]),
    )
