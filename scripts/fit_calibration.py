"""Controller-run mechanism: fits the memory-model coefficients (base transient +
gc-aware linear activation + O(N^2) attention backward) from real `train_step` bench
artifacts (`bench/worker.py` `run_train_step`'s own JSON output) + their models'
`config.json` files, and updates `calibration_data.json` with provenance -- the src-side
math this wraps is `mlx_train_perf.plan.estimate.fit_memory_coeffs` (pure, unit-tested
against synthetic FitPoints; see `tests/test_plan.py`). `optimizer_bytes_per_param` is
analytic (AdamW, 8 B/param), preserved unchanged rather than fitted.

Reads a MANIFEST: a JSON list of `{"config": "<path to HF config.json>", "artifact":
"<path to a run_train_step artifact, impl='kernel', status='ok'>", "batch": int,
"seq_len": int, "lora_rank": int, "lora_layers": int, "grad_checkpoint": bool}` objects.
`batch`/`seq_len`/`lora_rank`/`lora_layers`/`grad_checkpoint` are given explicitly in the
manifest rather than reverse-engineered from the artifact's own (opaque, nested)
`identity` dict -- makes this script's inputs auditable at a glance. `grad_checkpoint`
defaults True; the fit needs >= 3 grad_checkpoint=True points spanning >= 2 seq_len values
(to separate base/linear/quadratic) plus optional grad_checkpoint=False points (for the
`_full` linear coefficient).

The flash coefficient (`attn_bytes_per_head_token_flash`) is fit twice if needed: first
by `fit_memory_coeffs(..., flash_fit="ols")` (the least-squares default), then checked
for one-sidedness (`_flash_fit_is_one_sided` -- the candidate's predicted cushioned
TOTAL peak, via the public `estimate_peak`, must be >= every flash anchor's own
measured total). A violation triggers a refit with `flash_fit="envelope"` (the largest
per-point residual/x_flash ratio, strictly conservative). Whichever one ran is recorded
as `provenance["flash_fit"]`.

This is BUILD-verified only: `tests/test_fit_calibration.py` exercises this script
end-to-end against SYNTHETIC (fabricated) manifest/config/artifact files written to a
temp directory -- never against a real `run_train_step` artifact, since none exist yet
(the production benches have not been run). `--dry-run` prints the fitted constants
without writing anything -- the safe default for a first look at real numbers. The
controller runs this for real, against real artifacts, after the production runs. The
committed `src/mlx_train_perf/plan/calibration_data.json` now carries measured
coefficients: the stock terms are the original 0.31.2-era campaign carried forward, and
the flash coefficient was fit on 0.32.0.
"""
import argparse
import json
import platform
from dataclasses import replace
from datetime import date
from pathlib import Path

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.plan.calibration import Calibration, load_calibration
from mlx_train_perf.plan.estimate import (
    FitPoint,
    ModelShape,
    TrainConfig,
    estimate_peak,
    fit_memory_coeffs,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DATA = (
    _SCRIPTS_DIR.parent / "src" / "mlx_train_perf" / "plan" / "calibration_data.json"
)


def load_fit_points(manifest_path: Path) -> list[FitPoint]:
    """Builds one `FitPoint` per manifest entry. Raises `ValueError` (naming the
    offending artifact) if an artifact's own recorded `status` is not `"ok"` -- a
    refused or crashed condition carries no usable `marginal_peak_gb`, and fitting
    from it would silently corrupt the calibration constants."""
    manifest = json.loads(manifest_path.read_text())
    points: list[FitPoint] = []
    for entry in manifest:
        config = json.loads(Path(entry["config"]).read_text())
        shape = ModelShape.from_config(config)
        artifact_path = entry["artifact"]
        artifact = json.loads(Path(artifact_path).read_text())
        if artifact.get("status") != "ok":
            raise ValueError(
                f"artifact {artifact_path!r} has status={artifact.get('status')!r}, "
                "not 'ok' -- cannot fit from a refused/crashed condition"
            )
        identity = artifact.get("identity", {})
        impl = str(identity.get("impl", "kernel"))
        # `attention_impl` ("stock"/"flash") threads from the artifact identity into the
        # FitPoint's cfg so `fit_memory_coeffs` routes the point into the right branch.
        # Absent (pre-0.2.0 artifacts) it defaults to "stock".
        attention = str(identity.get("attention_impl", "stock"))
        cfg = TrainConfig(
            batch=int(entry["batch"]), seq_len=int(entry["seq_len"]), dtype="bfloat16",
            lora_rank=int(entry["lora_rank"]), lora_layers=int(entry["lora_layers"]),
            grad_checkpoint=bool(entry.get("grad_checkpoint", True)), impl=impl,
            attention=attention,
        )
        marginal_peak_bytes = float(artifact["marginal_peak_gb"]) * 1024**3
        points.append(FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=marginal_peak_bytes))
    return points


def _candidate_calibration(*, calib: Calibration, coeffs: dict[str, float]) -> Calibration:
    """Builds the FULL candidate `Calibration` `main()` is about to ship -- all five
    post-fit memory coefficients (base + gc-aware linear + O(N^2) stock attention +
    O(N) flash attention), not just `calib` with `attn_bytes_per_head_token_flash`
    swapped in. A mixed stock+flash manifest refits base/a_lin/a_quad from the stock
    points too, and those fitted values can differ sharply from `calib`'s own
    (pre-refit) values -- validating only `replace(calib, attn_bytes_per_head_token_flash=...)`
    left the actually-shipped combination unchecked (reviewer reproduced a
    ~49.9 MB under-prediction that passed the old, stale-base/a_lin check)."""
    return replace(
        calib,
        base_transient_bytes=coeffs["base_transient_bytes"],
        act_bytes_per_token_hidden_layer_ckpt=coeffs["act_bytes_per_token_hidden_layer_ckpt"],
        act_bytes_per_token_hidden_layer_full=coeffs["act_bytes_per_token_hidden_layer_full"],
        attn_bytes_per_head_token2=coeffs["attn_bytes_per_head_token2"],
        attn_bytes_per_head_token_flash=coeffs["attn_bytes_per_head_token_flash"],
    )


def _flash_fit_is_one_sided(points: list[FitPoint], *, candidate: Calibration) -> bool:
    """True iff, for EVERY flash `FitPoint`, the CANDIDATE calibration's predicted
    cushioned TOTAL peak (the public `estimate_peak` call a real planner caller makes)
    is >= the point's own measured total. `candidate` must be the FULL post-fit
    `Calibration` `main()` is about to write (see `_candidate_calibration`) -- checking
    a stale `calib` with only `attn_bytes_per_head_token_flash` swapped in validates a
    combination that is never actually shipped whenever a mixed stock+flash manifest
    also refits base/a_lin/a_quad. The measured total is reconstructed from
    `estimate_peak`'s own `weights` component (shape/dtype-only, independent of
    `calib`) plus the point's measured MARGINAL (`marginal_peak_bytes`) --
    `estimate_peak`'s remaining components (base + activations + attention + lora +
    optimizer + loss) are exactly what the marginal measures, by construction (see
    `fit_memory_coeffs`'s docstring). A violation here is what triggers the
    `flash_fit="envelope"` fallback in `main()`: the planner's own never-under-predict
    invariant, not a numeric-accuracy nicety."""
    for p in points:
        if p.cfg.attention != "flash":
            continue
        predicted_total, components = estimate_peak(p.shape, p.cfg, candidate)
        measured_total = components["weights"] + p.marginal_peak_bytes
        if predicted_total < measured_total:
            return False
    return True


def build_updated_calibration_data(
    *, existing: dict[str, object], coeffs: dict[str, float],
    optimizer_bytes_per_param: float, manifest_path: Path, num_points: int,
    flash_fit: str = "ols",
) -> dict[str, object]:
    """Preserves `overhead_frac`/`naive_loss_bytes_per_nv` from `existing` UNCHANGED
    (this fit never touches them) and replaces the five fitted memory coefficients (base +
    gc-aware linear + O(N^2) stock attention + O(N) flash attention) +
    `optimizer_bytes_per_param` (analytic) + `provenance`.

    `provenance` keeps the four keys `load_calibration`'s own tests require truthy
    (`machine`, `macos`, `mlx_version`, `measured_date`), plus `flash_fit`
    (`"ols"|"envelope"`, Task 8 0.5.0) naming which flash-coefficient fit `main()`
    selected -- `"ols"` unless the one-sidedness check (`_flash_fit_is_one_sided`)
    found a violation and refit with the envelope fallback. Its `fit_source` PRESERVES
    the existing note (so a flash-only refit -- which keeps the stock coefficients
    unchanged -- doesn't erase the stock fit's provenance) and appends this run's
    clause naming the manifest + point count for auditability."""
    prior_provenance = existing.get("provenance", {})
    prior_source = ""
    if isinstance(prior_provenance, dict):
        prior_source = str(prior_provenance.get("fit_source", "")).strip()
    this_clause = (
        f"refit via fit_memory_coeffs from {num_points} train_step (impl='kernel') marginal "
        f"peaks (stock points fit base/gc-aware-linear/O(N^2)-attention by full-rank OLS; "
        f"flash points fit attn_bytes_per_head_token_flash by 1-var residual OLS holding "
        f"base/a_lin fixed; a branch with no points keeps its prior value); "
        f"optimizer_bytes_per_param analytic (AdamW, 8 B/param); manifest={manifest_path.name}"
    )
    fit_source = f"{prior_source} || {this_clause}" if prior_source else this_clause
    return {
        "base_transient_bytes": coeffs["base_transient_bytes"],
        "act_bytes_per_token_hidden_layer_ckpt": coeffs["act_bytes_per_token_hidden_layer_ckpt"],
        "act_bytes_per_token_hidden_layer_full": coeffs["act_bytes_per_token_hidden_layer_full"],
        "attn_bytes_per_head_token2": coeffs["attn_bytes_per_head_token2"],
        "attn_bytes_per_head_token_flash": coeffs["attn_bytes_per_head_token_flash"],
        "optimizer_bytes_per_param": optimizer_bytes_per_param,
        "overhead_frac": existing["overhead_frac"],
        "naive_loss_bytes_per_nv": existing["naive_loss_bytes_per_nv"],
        "provenance": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "mlx_version": _installed_mlx_version(),
            "measured_date": date.today().isoformat(),
            "fit_source": fit_source,
            "flash_fit": flash_fit,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="path to a JSON manifest (see this script's module docstring)")
    ap.add_argument("--calibration-data", default=str(DEFAULT_CALIBRATION_DATA),
                    help="path to calibration_data.json to update")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the fitted constants without writing anything")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    calibration_path = Path(args.calibration_data)
    existing = json.loads(calibration_path.read_text())
    calib = load_calibration() if calibration_path == DEFAULT_CALIBRATION_DATA else None
    if calib is None:
        # A non-default --calibration-data path (e.g. a test's temp file) can't go
        # through load_calibration (which always reads the INSTALLED package data file) --
        # build a stand-in Calibration from `existing` instead. `fit_memory_coeffs` reads
        # calib only for the analytic small terms it subtracts (optimizer + naive loss,
        # the latter only for impl="naive" points, which this script never builds).
        calib = Calibration(
            base_transient_bytes=float(existing["base_transient_bytes"]),
            act_bytes_per_token_hidden_layer_ckpt=float(
                existing["act_bytes_per_token_hidden_layer_ckpt"]),
            act_bytes_per_token_hidden_layer_full=float(
                existing["act_bytes_per_token_hidden_layer_full"]),
            attn_bytes_per_head_token2=float(existing["attn_bytes_per_head_token2"]),
            attn_bytes_per_head_token_flash=float(existing["attn_bytes_per_head_token_flash"]),
            optimizer_bytes_per_param=float(existing["optimizer_bytes_per_param"]),
            overhead_frac=float(existing["overhead_frac"]),
            naive_loss_bytes_per_nv=float(existing["naive_loss_bytes_per_nv"]),
            provenance=dict(existing["provenance"]),
        )

    points = load_fit_points(manifest_path)
    coeffs = fit_memory_coeffs(points, calib=calib, flash_fit="ols")
    flash_fit = "ols"
    candidate = _candidate_calibration(calib=calib, coeffs=coeffs)
    if not _flash_fit_is_one_sided(points, candidate=candidate):
        # OLS's least-squares average under-predicted at least one flash anchor's own
        # cushioned TOTAL, checked against the FULL candidate (not a stale calib with
        # only a_flash swapped) -- refit with the conservative (over-predict-safe)
        # envelope.
        coeffs = fit_memory_coeffs(points, calib=calib, flash_fit="envelope")
        flash_fit = "envelope"
        candidate = _candidate_calibration(calib=calib, coeffs=coeffs)
    updated = build_updated_calibration_data(
        existing=existing, coeffs=coeffs,
        optimizer_bytes_per_param=float(existing["optimizer_bytes_per_param"]),
        manifest_path=manifest_path, num_points=len(points), flash_fit=flash_fit,
    )
    print(json.dumps(updated, indent=2))
    if args.dry_run:
        print(f"--dry-run: {calibration_path} NOT written")
        return 0
    calibration_path.write_text(json.dumps(updated, indent=2) + "\n")
    print(f"wrote {calibration_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
