"""Controller-run mechanism: fits `act_bytes_per_token_layer` and
`optimizer_bytes_per_param` from real `train_step` bench artifacts (`bench/worker.py`
`run_train_step`'s own JSON output) + their models' `config.json` files, and updates
`calibration_data.json` with provenance -- the src-side math this wraps is
`mlx_train_perf.plan.estimate.fit_activation_and_optimizer_bytes` (pure, unit-tested
against synthetic FitPoints; see `tests/test_plan.py`).

Reads a MANIFEST: a JSON list of `{"config": "<path to HF config.json>", "artifact":
"<path to a run_train_step artifact, impl='kernel', status='ok'>", "batch": int,
"seq_len": int, "lora_rank": int, "lora_layers": int}` objects. `batch`/`seq_len`/
`lora_rank`/`lora_layers` are given explicitly in the manifest rather than
reverse-engineered from the artifact's own (opaque, nested) `identity` dict -- makes
this script's inputs auditable at a glance.

This is BUILD-verified only: `tests/test_fit_calibration.py` exercises this script
end-to-end against SYNTHETIC (fabricated) manifest/config/artifact files written to a
temp directory -- never against a real `run_train_step` artifact, since none exist yet
(the production benches have not been run). `--dry-run` prints the fitted constants
without writing anything -- the safe default for a first look at real numbers. The
controller runs this for real, against real artifacts, after the production runs;
`src/mlx_train_perf/plan/calibration_data.json`'s own committed constants are left
untouched by this task.
"""
import argparse
import json
import platform
from datetime import date
from pathlib import Path

from mlx_train_perf._compat import _installed_mlx_version
from mlx_train_perf.plan.calibration import Calibration, load_calibration
from mlx_train_perf.plan.estimate import (
    FitPoint,
    ModelShape,
    TrainConfig,
    fit_activation_and_optimizer_bytes,
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
        cfg = TrainConfig(
            batch=int(entry["batch"]), seq_len=int(entry["seq_len"]), dtype="bfloat16",
            lora_rank=int(entry["lora_rank"]), lora_layers=int(entry["lora_layers"]),
            grad_checkpoint=bool(entry.get("grad_checkpoint", False)), impl=impl,
        )
        marginal_peak_bytes = float(artifact["marginal_peak_gb"]) * 1024**3
        points.append(FitPoint(shape=shape, cfg=cfg, marginal_peak_bytes=marginal_peak_bytes))
    return points


def build_updated_calibration_data(
    *, existing: dict[str, object], act_bytes_per_token_layer: float,
    optimizer_bytes_per_param: float, manifest_path: Path, num_points: int,
) -> dict[str, object]:
    """Preserves `overhead_frac`/`naive_loss_bytes_per_nv` from `existing` UNCHANGED
    (this fit never touches them) and replaces `act_bytes_per_token_layer`/
    `optimizer_bytes_per_param` + `provenance` -- `provenance` keeps the four keys
    `load_calibration`'s own tests require truthy (`machine`, `macos`, `mlx_version`,
    `measured_date`) and adds a `fit_source` note naming the manifest + point count
    for auditability."""
    return {
        "act_bytes_per_token_layer": act_bytes_per_token_layer,
        "optimizer_bytes_per_param": optimizer_bytes_per_param,
        "overhead_frac": existing["overhead_frac"],
        "naive_loss_bytes_per_nv": existing["naive_loss_bytes_per_nv"],
        "provenance": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "mlx_version": _installed_mlx_version(),
            "measured_date": date.today().isoformat(),
            "fit_source": (
                f"act_bytes_per_token_layer and optimizer_bytes_per_param fitted "
                f"from {num_points} train_step (impl='kernel') artifacts via "
                f"fit_activation_and_optimizer_bytes; manifest={manifest_path.name}"
            ),
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
        # through load_calibration (which always reads the INSTALLED package data
        # file) -- read `overhead_frac`/`naive_loss_bytes_per_nv` from `existing`
        # directly instead; fit_activation_and_optimizer_bytes only reads
        # naive_loss_bytes_per_nv when a FitPoint's impl=="naive", which this script
        # never builds (see load_fit_points), so an approximate stand-in Calibration
        # here is inert either way.
        calib = Calibration(
            act_bytes_per_token_layer=float(existing["act_bytes_per_token_layer"]),
            optimizer_bytes_per_param=float(existing["optimizer_bytes_per_param"]),
            overhead_frac=float(existing["overhead_frac"]),
            naive_loss_bytes_per_nv=float(existing["naive_loss_bytes_per_nv"]),
            provenance=dict(existing["provenance"]),
        )

    points = load_fit_points(manifest_path)
    act_bytes_per_token_layer, optimizer_bytes_per_param = fit_activation_and_optimizer_bytes(
        points, calib=calib,
    )
    updated = build_updated_calibration_data(
        existing=existing, act_bytes_per_token_layer=act_bytes_per_token_layer,
        optimizer_bytes_per_param=optimizer_bytes_per_param, manifest_path=manifest_path,
        num_points=len(points),
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
