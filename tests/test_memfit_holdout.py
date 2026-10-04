import hashlib
import json
from pathlib import Path

import pytest

from mlx_train_perf.memfit import (
    CalibrationMismatchError,
    HoldoutBand,
    MemfitInputError,
    score_holdout,
    write_holdout_predictions,
)

_BAND = HoldoutBand(under_rel=0.25, over_rel=0.5)   # 1000 predicted -> [500, 1250] in band


def _predict(tmp_path: Path, preds: dict[str, int]) -> Path:
    path = tmp_path / "pred.json"
    write_holdout_predictions(path, preds, band=_BAND, measured_quantity="mlx_active_marginal",
                              provenance={"git_sha": "abc123"})
    return path


@pytest.mark.parametrize(("measured", "in_band"),
                         [(1249, True), (1250, True), (1251, False),
                          (501, True), (500, True), (499, False)])
def test_band_edges_are_inclusive_and_relative_to_the_prediction(
    tmp_path: Path, measured: int, in_band: bool,
) -> None:
    report = score_holdout(_predict(tmp_path, {"p": 1000}), {"p": measured},
                           expect_quantity="mlx_active_marginal")
    assert report.points[0].in_band is in_band
    assert report.out_of_band == (() if in_band else ("p",))


def test_rel_error_is_measured_minus_predicted_over_predicted(tmp_path: Path) -> None:
    report = score_holdout(_predict(tmp_path, {"a": 1000, "b": 2000}), {"a": 1100, "b": 1900},
                           expect_quantity="mlx_active_marginal")
    assert [p.rel_error for p in report.points] == [pytest.approx(0.1), pytest.approx(-0.05)]
    assert report.max_under_rel == pytest.approx(0.1)


def test_unscored_and_unpredicted_labels_are_reported(tmp_path: Path) -> None:
    report = score_holdout(_predict(tmp_path, {"a": 1000, "b": 1000}), {"a": 1000, "c": 5},
                           expect_quantity="mlx_active_marginal")
    assert report.unscored == ("b",)
    assert report.unpredicted == ("c",)


def test_predictions_are_never_overwritten(tmp_path: Path) -> None:
    path = _predict(tmp_path, {"a": 1000})
    with pytest.raises(MemfitInputError, match="exists"):
        write_holdout_predictions(path, {"a": 999}, band=_BAND,
                                  measured_quantity="mlx_active_marginal", provenance={})
    assert json.loads(path.read_text())["predictions"] == {"a": 1000}


def test_the_predictions_file_is_versioned(tmp_path: Path) -> None:
    doc = json.loads(_predict(tmp_path, {"a": 1000}).read_text())
    assert doc["format"] == "mlx-train-perf.holdout-predictions"
    assert doc["schema_version"] == 1
    assert doc["provenance"] == {"git_sha": "abc123"}


def test_scoring_refuses_a_different_quantity(tmp_path: Path) -> None:
    with pytest.raises(CalibrationMismatchError):
        score_holdout(_predict(tmp_path, {"a": 1000}), {"a": 1000},
                      expect_quantity="max_footprint_mlx_active_cache")


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_measurements_are_refused(tmp_path: Path, bad: int) -> None:
    with pytest.raises(MemfitInputError, match="measurement 'a'"):
        score_holdout(_predict(tmp_path, {"a": 1000}), {"a": bad},
                      expect_quantity="mlx_active_marginal")


@pytest.mark.parametrize("bad", [{}, {"a": 0}])
def test_empty_or_non_positive_predictions_are_refused(tmp_path: Path, bad: dict[str, int]) -> None:
    with pytest.raises(MemfitInputError):
        write_holdout_predictions(tmp_path / "x.json", bad, band=_BAND,
                                  measured_quantity="q", provenance={})


def test_record_carries_the_predictions_file_hash(tmp_path: Path) -> None:
    path = _predict(tmp_path, {"a": 1000})
    record = score_holdout(path, {"a": 1300}, expect_quantity="mlx_active_marginal").to_record()
    assert record["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["band"] == {"under_rel": 0.25, "over_rel": 0.5}
    assert record["out_of_band"] == ["a"]


def test_invalid_json_raises_memfit_input_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(MemfitInputError, match="not valid JSON"):
        score_holdout(path, {"a": 1000}, expect_quantity="mlx_active_marginal")


def test_missing_predictions_key_raises_memfit_input_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    doc = {
        "format": "mlx-train-perf.holdout-predictions",
        "schema_version": 1,
        "measured_quantity": "mlx_active_marginal",
        "band": {"under_rel": 0.25, "over_rel": 0.5},
        "provenance": {},
    }
    path.write_text(json.dumps(doc))
    with pytest.raises(MemfitInputError, match="predictions"):
        score_holdout(path, {"a": 1000}, expect_quantity="mlx_active_marginal")


@pytest.mark.parametrize(("case", "mutation"), [
    ("top_level_list", lambda doc: [doc]),
    ("predictions_list", lambda doc: {**doc, "predictions": [1000]}),
    ("prediction_zero", lambda doc: {**doc, "predictions": {"a": 0}}),
    ("prediction_string", lambda doc: {**doc, "predictions": {"a": "1000"}}),
    ("prediction_negative", lambda doc: {**doc, "predictions": {"a": -5}}),
    ("band_list", lambda doc: {**doc, "band": [0.25, 0.5]}),
    ("band_missing_over_rel", lambda doc: {**doc, "band": {"under_rel": 0.25}}),
    ("band_value_string", lambda doc: {**doc, "band": {"under_rel": 0.25, "over_rel": "x"}}),
])
def test_invalid_prediction_file_structures_raise_memfit_input_error(
    tmp_path: Path, case: str, mutation,
) -> None:
    # Create a valid predictions file, then hand-edit it for this case
    base_path = tmp_path / "valid.json"
    write_holdout_predictions(base_path, {"a": 1000}, band=_BAND,
                              measured_quantity="mlx_active_marginal", provenance={})
    doc = json.loads(base_path.read_text())

    # Mutate the valid doc
    mutated = mutation(doc)

    # Write the mutated doc
    case_path = tmp_path / f"{case}.json"
    case_path.write_text(json.dumps(mutated))

    # Should raise MemfitInputError
    with pytest.raises(MemfitInputError):
        score_holdout(case_path, {"a": 1000}, expect_quantity="mlx_active_marginal")


def test_write_holdout_predictions_with_non_serializable_provenance_fails_safely(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nonser.json"
    # Try to write with a non-serializable provenance value
    with pytest.raises(MemfitInputError):
        write_holdout_predictions(path, {"a": 1000}, band=_BAND,
                                  measured_quantity="mlx_active_marginal",
                                  provenance={"obj": object()})
    # File should not exist (serialization happened before file write)
    assert not path.exists()


def test_deeply_nested_json_is_a_memfit_input_error(tmp_path: Path) -> None:
    """Bug caught: json.loads raises RecursionError on 100k nested arrays, which escaped
    score_holdout as a raw stdlib exception."""
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000)
    with pytest.raises(MemfitInputError):
        score_holdout(path, {"a": 1000}, expect_quantity="mlx_active_marginal")


def test_huge_prediction_in_a_hand_edited_file_is_refused(tmp_path: Path) -> None:
    """Bug caught: 10**400 passes a positive-int check, then `predicted * (1 + under_rel)`
    raises OverflowError."""
    path = _predict(tmp_path, {"a": 1000})
    doc = json.loads(path.read_text())
    doc["predictions"]["a"] = 10**400
    path.write_text(json.dumps(doc))
    with pytest.raises(MemfitInputError, match="prediction"):
        score_holdout(path, {"a": 1000}, expect_quantity="mlx_active_marginal")


def test_huge_measurement_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MemfitInputError, match="measurement"):
        score_holdout(_predict(tmp_path, {"a": 1000}), {"a": 10**400},
                      expect_quantity="mlx_active_marginal")


def test_writing_a_huge_prediction_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    with pytest.raises(MemfitInputError, match="prediction"):
        write_holdout_predictions(path, {"a": 10**400}, band=_BAND,
                                  measured_quantity="mlx_active_marginal", provenance={})
    assert not path.exists()


def test_max_under_rel_is_zero_when_every_point_is_over_predicted(tmp_path: Path) -> None:
    """Bug caught: taking max(abs(rel_error)) instead of the positive part would report
    0.2 here, treating over-prediction as under-prediction."""
    report = score_holdout(_predict(tmp_path, {"a": 1000, "b": 1000}),
                           {"a": 800, "b": 800}, expect_quantity="mlx_active_marginal")
    assert report.max_under_rel == 0.0


@pytest.mark.parametrize("field", ["under_rel", "over_rel"])
@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_holdout_band_refuses_negative_and_non_finite(field: str, bad: float) -> None:
    kwargs = {"under_rel": 0.1, "over_rel": 0.1, field: bad}
    with pytest.raises(MemfitInputError, match=field):
        HoldoutBand(**kwargs)
