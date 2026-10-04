import json
from pathlib import Path

import pytest

from mlx_train_perf.memfit import (
    MLX_ACTIVE_MARGINAL,
    CalibrationFile,
    CalibrationMismatchError,
    MemfitInputError,
    dump_calibration_file,
    load_calibration_file,
)


def _doc(**over: object) -> dict[str, object]:
    doc: dict[str, object] = {
        "format": "mlx-train-perf.calibration", "schema_version": 1,
        "measured_quantity": "mlx_active_marginal",
        "coefficients": {"base": 1600848316.0, "per_token": 1.867},
        "terms": {"per_token": "a * tokens"},
        "provenance": {"machine": "arm64", "measured_date": "2026-10-03"},
        "holdout": None,
    }
    doc.update(over)
    return doc


def _write(tmp_path: Path, doc: dict[str, object]) -> Path:
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(doc))
    return p


def test_load_reads_every_field(tmp_path: Path) -> None:
    cal = load_calibration_file(_write(tmp_path, _doc()), expect_quantity=MLX_ACTIVE_MARGINAL)
    assert cal == CalibrationFile(
        measured_quantity="mlx_active_marginal",
        coefficients={"base": 1600848316.0, "per_token": 1.867},
        terms={"per_token": "a * tokens"},
        provenance={"machine": "arm64", "measured_date": "2026-10-03"},
        holdout=None, schema_version=1,
    )


def test_a_different_measured_quantity_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, _doc(measured_quantity="max_footprint_mlx_active_cache"))
    with pytest.raises(CalibrationMismatchError, match="max_footprint_mlx_active_cache"):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)


def test_terms_and_holdout_are_optional(tmp_path: Path) -> None:
    doc = _doc()
    del doc["terms"]
    del doc["holdout"]
    cal = load_calibration_file(_write(tmp_path, doc), expect_quantity=MLX_ACTIVE_MARGINAL)
    assert cal.terms == {}
    assert cal.holdout is None


def test_provenance_has_no_required_keys(tmp_path: Path) -> None:
    cal = load_calibration_file(_write(tmp_path, _doc(provenance={})),
                                expect_quantity=MLX_ACTIVE_MARGINAL)
    assert cal.provenance == {}


@pytest.mark.parametrize(("field", "value", "needle"), [
    ("format", "something-else", "format"),
    ("schema_version", 2, "schema_version"),
    ("coefficients", {"base": True}, "base"),          # JSON bool is not a number here
    ("coefficients", {"base": "1.0"}, "base"),
    ("provenance", {"machine": 64}, "machine"),
    ("measured_quantity", 3, "measured_quantity"),
])
def test_malformed_files_are_refused_naming_the_field(
    tmp_path: Path, field: str, value: object, needle: str,
) -> None:
    with pytest.raises(MemfitInputError, match=needle):
        load_calibration_file(_write(tmp_path, _doc(**{field: value})),
                              expect_quantity=MLX_ACTIVE_MARGINAL)


def test_dump_then_load_round_trips(tmp_path: Path) -> None:
    cal = CalibrationFile(measured_quantity="mlx_active_marginal",
                          coefficients={"a": 2.5}, provenance={"git_sha": "abc"},
                          holdout={"max_under_rel": 0.02, "out_of_band": []})
    path = tmp_path / "out.json"
    dump_calibration_file(cal, path)
    assert load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL) == cal
    assert json.loads(path.read_text())["format"] == "mlx-train-perf.calibration"


@pytest.mark.parametrize("content", ["not json", ""])
def test_non_json_file_is_refused(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(MemfitInputError, match="not valid JSON"):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)


def test_huge_integer_coefficient_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    doc = {
        "format": "mlx-train-perf.calibration", "schema_version": 1,
        "measured_quantity": "mlx_active_marginal",
        "coefficients": {"base": 10**400},
        "provenance": {},
    }
    path.write_text(json.dumps(doc))
    with pytest.raises(MemfitInputError, match="base"):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)


def test_nan_coefficient_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    # Manually write NaN token since json.dumps(float('nan')) produces NaN
    doc_text = ('{"format": "mlx-train-perf.calibration", "schema_version": 1, '
                 '"measured_quantity": "mlx_active_marginal", '
                 '"coefficients": {"base": NaN}, "provenance": {}}')
    path.write_text(doc_text)
    with pytest.raises(MemfitInputError, match="base"):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)


def test_dump_with_nan_coefficient_raises(tmp_path: Path) -> None:
    cal = CalibrationFile(measured_quantity="mlx_active_marginal",
                          coefficients={"base": float("nan")}, provenance={})
    path = tmp_path / "nan_out.json"
    with pytest.raises(MemfitInputError, match="JSON"):
        dump_calibration_file(cal, path)
    assert not path.exists()


def test_dump_refuses_an_unsupported_schema_version(tmp_path: Path) -> None:
    """Bug caught: a CalibrationFile(schema_version=2) written out as a file this
    version's loader then refuses."""
    cal = CalibrationFile(measured_quantity="mlx_active_marginal",
                          coefficients={"base": 1.0}, provenance={}, schema_version=2)
    path = tmp_path / "v2.json"
    with pytest.raises(MemfitInputError, match="schema_version"):
        dump_calibration_file(cal, path)
    assert not path.exists()


def test_deeply_nested_json_is_a_memfit_input_error(tmp_path: Path) -> None:
    """Bug caught: json.loads raises RecursionError on 100k nested arrays, which escaped
    the loader as a raw stdlib exception."""
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000)
    with pytest.raises(MemfitInputError):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)


def test_non_utf8_file_is_a_memfit_input_error(tmp_path: Path) -> None:
    """Bug caught: the utf-8 read sat outside the try, so a binary file raised a raw
    UnicodeDecodeError."""
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(MemfitInputError):
        load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)
