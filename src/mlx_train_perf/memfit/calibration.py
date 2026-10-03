"""Calibration files: fitted coefficients plus where they came from and what they measure.

Format (JSON, schema 1): `format`, `schema_version`, `measured_quantity`, `coefficients`
(name -> number), optional `terms` (name -> functional-form note), `provenance`
(name -> string; recommended keys: machine, chip, ram_gib, macos, mlx_version,
package_version, git_sha, date), optional `holdout` (a scored hold-out record).
Loading requires the caller to state the measured quantity it expects; a file that
measured something else is refused.
"""
import json
import math
import os
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from pathlib import Path

from mlx_train_perf.memfit.errors import CalibrationMismatchError, MemfitInputError

__all__ = ["CALIBRATION_FORMAT", "MAX_FOOTPRINT_MLX_ACTIVE_CACHE", "MLX_ACTIVE_MARGINAL",
           "MLX_ACTIVE_TOTAL", "CalibrationFile", "dump_calibration_file",
           "load_calibration_file"]

CALIBRATION_FORMAT = "mlx-train-perf.calibration"
_SCHEMA_VERSION = 1
# Measured-quantity names. A calibration file records which one it measured, and loading
# refuses a file that measured another.
# MLX_ACTIVE_MARGINAL: the peak of `mx.get_active_memory()` during the measured window
#   minus its value just before the window (MLX's cache pool excluded).
MLX_ACTIVE_MARGINAL = "mlx_active_marginal"
# MLX_ACTIVE_TOTAL: the absolute peak of MLX active memory (cache pool excluded).
MLX_ACTIVE_TOTAL = "mlx_active_total"
# MAX_FOOTPRINT_MLX_ACTIVE_CACHE: the larger of the OS process footprint (phys_footprint)
#   and MLX active + cache memory.
MAX_FOOTPRINT_MLX_ACTIVE_CACHE = "max_footprint_mlx_active_cache"


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationFile:
    measured_quantity: str
    coefficients: dict[str, float]
    provenance: dict[str, str]
    terms: dict[str, str] = field(default_factory=dict)
    holdout: dict[str, object] | None = None
    schema_version: int = _SCHEMA_VERSION


def _str_map(raw: object, where: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise MemfitInputError(f"{where} must be an object")
    for key, value in raw.items():
        if not isinstance(value, str):
            raise MemfitInputError(f"{where}[{key!r}] must be a string (got {value!r})")
    return dict(raw)


def _number_map(raw: object, where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise MemfitInputError(f"{where} must be an object")
    out: dict[str, float] = {}
    for key, value in raw.items():
        is_bool = isinstance(value, bool)
        is_number = isinstance(value, (int, float))
        if is_bool or not is_number:
            raise MemfitInputError(f"{where}[{key!r}] must be a finite number (got {value!r})")
        try:
            number = float(value)
        except OverflowError:
            raise MemfitInputError(
                f"{where}[{key!r}] must be a finite number "
                "(got an integer too large for a float)") from None
        if not math.isfinite(number):
            raise MemfitInputError(f"{where}[{key!r}] must be a finite number (got {value!r})")
        out[key] = number
    return out


def load_calibration_file(
    source: str | os.PathLike[str] | Traversable, *, expect_quantity: str,
) -> CalibrationFile:
    """Read and validate a calibration file. Raises `MemfitInputError` naming the bad
    field, and `CalibrationMismatchError` when `measured_quantity != expect_quantity`."""
    try:
        text = (Path(source).read_text(encoding="utf-8")
                if isinstance(source, (str, os.PathLike)) else source.read_text(encoding="utf-8"))
        raw = json.loads(text)
    except (ValueError, RecursionError) as exc:
        # ValueError covers bad JSON and non-UTF-8 bytes (UnicodeDecodeError);
        # RecursionError is what json raises on absurdly deep nesting.
        raise MemfitInputError(f"{source}: not valid JSON ({type(exc).__name__})") from exc
    if not isinstance(raw, dict) or raw.get("format") != CALIBRATION_FORMAT:
        raise MemfitInputError(f"{source}: format must be {CALIBRATION_FORMAT!r}")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise MemfitInputError(
            f"{source}: unsupported schema_version {raw.get('schema_version')!r} "
            f"(this version reads {_SCHEMA_VERSION})")
    quantity = raw.get("measured_quantity")
    if not isinstance(quantity, str):
        raise MemfitInputError(f"{source}: measured_quantity must be a string")
    if quantity != expect_quantity:
        raise CalibrationMismatchError(
            f"{source} measured {quantity!r}, but the caller expects {expect_quantity!r}")
    holdout = raw.get("holdout")
    if holdout is not None and not isinstance(holdout, dict):
        raise MemfitInputError(f"{source}: holdout must be an object or null")
    return CalibrationFile(
        measured_quantity=quantity,
        coefficients=_number_map(raw.get("coefficients"), "coefficients"),
        provenance=_str_map(raw.get("provenance", {}), "provenance"),
        terms=_str_map(raw.get("terms", {}), "terms"),
        holdout=holdout,
        schema_version=_SCHEMA_VERSION,
    )


def dump_calibration_file(cal: CalibrationFile, path: str | os.PathLike[str]) -> None:
    """Write `cal` in the schema-1 format, keys in a stable order, newline-terminated.
    Raises `MemfitInputError` for a schema version this module does not write or for a
    value JSON cannot hold (NaN, infinity, a non-serializable object); no file is created."""
    if cal.schema_version != _SCHEMA_VERSION:
        raise MemfitInputError(
            f"cannot write schema_version {cal.schema_version!r} (this version writes "
            f"{_SCHEMA_VERSION})")
    doc = {
        "format": CALIBRATION_FORMAT, "schema_version": cal.schema_version,
        "measured_quantity": cal.measured_quantity, "coefficients": cal.coefficients,
        "terms": cal.terms, "provenance": cal.provenance, "holdout": cal.holdout,
    }
    try:
        serialized = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError) as exc:
        raise MemfitInputError(f"calibration is not JSON-serializable: {exc}") from exc
    Path(path).write_text(serialized)
