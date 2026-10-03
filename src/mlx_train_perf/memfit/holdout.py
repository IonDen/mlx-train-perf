"""Hold-out validation: predictions are written (and committed) before the measuring
runs, then the measurements are scored against them.

A point is in band when `predicted * (1 - over_rel) <= measured <=
predicted * (1 + under_rel)`. The band is asymmetric because the two misses differ:
measuring above the prediction (under-prediction) risks running out of memory, measuring
below it only wastes headroom. A predictions file is never overwritten.

To budget with the band: peak + band + reserve <= B becomes
`budget_bytes = floor((B - reserve) / (1 + under_rel))` for `max_int_within_budget`.
"""
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlx_train_perf.memfit.errors import CalibrationMismatchError, MemfitInputError

__all__ = ["HOLDOUT_FORMAT", "HoldoutBand", "HoldoutPoint", "HoldoutReport", "score_holdout",
           "write_holdout_predictions"]

HOLDOUT_FORMAT = "mlx-train-perf.holdout-predictions"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldoutBand:
    under_rel: float
    over_rel: float

    def __post_init__(self) -> None:
        for name in ("under_rel", "over_rel"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise MemfitInputError(f"{name} must be finite and >= 0 (got {value!r})")


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldoutPoint:
    label: str
    predicted: int
    measured: int
    rel_error: float
    in_band: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldoutReport:
    points: tuple[HoldoutPoint, ...]
    out_of_band: tuple[str, ...]
    max_under_rel: float
    unscored: tuple[str, ...]
    unpredicted: tuple[str, ...]
    band: HoldoutBand
    measured_quantity: str
    predictions_sha256: str

    def to_record(self) -> dict[str, object]:
        """The scored result, for a calibration file's `holdout` field."""
        return {
            "predictions_sha256": self.predictions_sha256,
            "measured_quantity": self.measured_quantity,
            "band": {"under_rel": self.band.under_rel, "over_rel": self.band.over_rel},
            "max_under_rel": self.max_under_rel,
            "out_of_band": list(self.out_of_band),
            "scored": len(self.points),
            "unscored": list(self.unscored),
            "unpredicted": list(self.unpredicted),
        }


# Beyond 2**53 an int no longer converts to a float exactly, and a 10**400 byte count makes
# `predicted * (1 + under_rel)` raise OverflowError; no real byte count is near it.
_MAX_BYTES = 2**53


def _positive_ints(values: Mapping[str, int], what: str) -> None:
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _MAX_BYTES:
            # An out-of-range int is not echoed: repr of a 5,000-digit int raises.
            shown = ("an out-of-range int" if isinstance(value, int) and abs(value) > 10**15
                     else repr(value))
            raise MemfitInputError(
                f"{what} {label!r} must be a positive int of at most 2**53 (got {shown})")


def write_holdout_predictions(
    path: str | os.PathLike[str], predictions: Mapping[str, int], *, band: HoldoutBand,
    measured_quantity: str, provenance: Mapping[str, str],
) -> None:
    """Write the predictions file once. Refuses an existing path: re-predicting after
    seeing results is what the protocol forbids, so moving the old file is a deliberate
    act by the caller."""
    if not predictions:
        raise MemfitInputError("no predictions to write")
    _positive_ints(predictions, "prediction")
    doc = {
        "format": HOLDOUT_FORMAT, "schema_version": _SCHEMA_VERSION,
        "measured_quantity": measured_quantity,
        "band": {"under_rel": band.under_rel, "over_rel": band.over_rel},
        "provenance": dict(provenance), "predictions": dict(predictions),
    }
    # Serialize to string FIRST (before opening file), so a serialization error
    # doesn't leave an empty file
    try:
        serialized = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError) as exc:
        raise MemfitInputError(f"predictions file is not JSON-serializable: {exc}") from exc
    try:
        with Path(path).open("x") as fh:
            fh.write(serialized)
    except FileExistsError as exc:
        raise MemfitInputError(
            f"{path} exists; hold-out predictions are written once, before the measuring "
            "runs") from exc


def score_holdout(
    path: str | os.PathLike[str], measured: Mapping[str, int], *, expect_quantity: str,
) -> HoldoutReport:
    """Score `measured` (label -> bytes) against the predictions file at `path`."""
    path_obj = Path(path)
    raw_bytes = path_obj.read_bytes()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise MemfitInputError(f"{path_obj} is not valid JSON") from exc

    # Validate doc is a dict
    if not isinstance(doc, dict):
        raise MemfitInputError(f"{path_obj} top-level must be a dict, not {type(doc).__name__}")

    if doc.get("format") != HOLDOUT_FORMAT or doc.get("schema_version") != _SCHEMA_VERSION:
        raise MemfitInputError(f"{path_obj} is not a schema-1 {HOLDOUT_FORMAT} file")

    try:
        measured_quantity = doc["measured_quantity"]
        band_dict = doc["band"]
        predictions = doc["predictions"]
    except KeyError as exc:
        raise MemfitInputError(f"{path_obj} is missing required key {exc}") from exc

    # Validate predictions is a dict with positive int values
    if not isinstance(predictions, dict):
        raise MemfitInputError(
            f"{path_obj} predictions must be a dict, not {type(predictions).__name__}"
        )
    _positive_ints(predictions, "prediction")

    # Validate band is a dict with exactly the right keys and correct types
    if not isinstance(band_dict, dict):
        raise MemfitInputError(f"{path_obj} band must be a dict, not {type(band_dict).__name__}")

    required_band_keys = {"under_rel", "over_rel"}
    if set(band_dict.keys()) != required_band_keys:
        raise MemfitInputError(
            f"{path_obj} band must have exactly keys {required_band_keys}, "
            f"got {set(band_dict.keys())}"
        )

    for key in required_band_keys:
        value = band_dict[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MemfitInputError(
                f"{path_obj} band[{key!r}] must be a real number, not {type(value).__name__}"
            )

    if measured_quantity != expect_quantity:
        raise CalibrationMismatchError(
            f"{path_obj} predicted {measured_quantity!r}, but the caller expects "
            f"{expect_quantity!r}")
    _positive_ints(measured, "measurement")
    band = HoldoutBand(**band_dict)
    points: list[HoldoutPoint] = []
    for label, predicted in predictions.items():
        if label not in measured:
            continue
        m = measured[label]
        points.append(HoldoutPoint(
            label=label, predicted=predicted, measured=m,
            rel_error=(m - predicted) / predicted,
            in_band=predicted * (1 - band.over_rel) <= m <= predicted * (1 + band.under_rel),
        ))
    return HoldoutReport(
        points=tuple(points),
        out_of_band=tuple(p.label for p in points if not p.in_band),
        max_under_rel=max((max(p.rel_error, 0.0) for p in points), default=0.0),
        unscored=tuple(lab for lab in predictions if lab not in measured),
        unpredicted=tuple(lab for lab in measured if lab not in predictions),
        band=band, measured_quantity=measured_quantity,
        predictions_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
