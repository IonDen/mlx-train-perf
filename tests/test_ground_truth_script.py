"""Pure-logic tests for `scripts/ground_truth_atomic_outputs.py` (mlx-train-perf-0008,
Task 16b Step 1). The GPU correctness assertions live in the script itself (it's an
experiment, not product code, per the workspace TDD exception for spike-style scripts) --
these tests cover the shape-defaulting and exact-reference-computation helpers, which are
pure and always run in the default lane.

`scripts/` has no `__init__.py` (matches `bench_quant_thresholds.py`'s existing
convention), so the module is loaded by path rather than via a package import.
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from ground_truth_atomic_outputs import (  # noqa: E402 -- import must follow the sys.path insert
    FLOAT32_EXACT_INT_CEILING,
    expected_totals,
    fits_float32_exact,
    max_expected_value,
    resolve_shape,
    script_sha,
)

# ---------------------------------------------------------------------------
# expected_totals / max_expected_value: the exact closed-form reference
# ---------------------------------------------------------------------------


def test_expected_totals_is_weight_times_triangular_number() -> None:
    # row_blocks=3 -> triangular = 1+2+3 = 6; weight(elem) = elem+1 -> [6, 12, 18, 24]
    assert expected_totals(n_elem=4, row_blocks=3) == [6.0, 12.0, 18.0, 24.0]


def test_expected_totals_zero_row_blocks_is_all_zero() -> None:
    assert expected_totals(n_elem=3, row_blocks=0) == [0.0, 0.0, 0.0]


def test_expected_totals_single_row_block_is_bare_weight() -> None:
    assert expected_totals(n_elem=3, row_blocks=1) == [1.0, 2.0, 3.0]


def test_max_expected_value_is_n_elem_times_triangular() -> None:
    assert max_expected_value(n_elem=4, row_blocks=3) == 24  # 4 * 6


# ---------------------------------------------------------------------------
# fits_float32_exact: the exactness guard (any dropped atomic add must be
# detectable exactly, never masked by float rounding)
# ---------------------------------------------------------------------------


def test_float32_exact_int_ceiling_is_2_pow_24() -> None:
    assert FLOAT32_EXACT_INT_CEILING == 1 << 24


def test_fits_float32_exact_true_well_below_ceiling() -> None:
    assert fits_float32_exact(n_elem=16, row_blocks=200) is True


def test_fits_float32_exact_false_at_the_ceiling() -> None:
    # n_elem * triangular(row_blocks) == exactly the ceiling -> NOT strictly below it
    assert max_expected_value(n_elem=1, row_blocks=1) == 1
    huge = FLOAT32_EXACT_INT_CEILING
    assert fits_float32_exact(n_elem=1, row_blocks=0) is True  # 0 is trivially fine
    assert fits_float32_exact(n_elem=huge, row_blocks=1) is False  # == ceiling, not below


def test_fits_float32_exact_false_above_ceiling() -> None:
    assert fits_float32_exact(n_elem=1000, row_blocks=10_000) is False


# ---------------------------------------------------------------------------
# resolve_shape: pure CLI-defaulting logic
# ---------------------------------------------------------------------------


def test_resolve_shape_correctness_defaults() -> None:
    assert resolve_shape(mode="correctness", tile=None, d=None, splits=None) == (16, 16, 200)


def test_resolve_shape_cost_defaults() -> None:
    assert resolve_shape(mode="cost", tile=None, d=None, splits=None) == (2048, 4096, 16)


def test_resolve_shape_overrides_take_precedence_over_defaults() -> None:
    assert resolve_shape(mode="correctness", tile=8, d=None, splits=50) == (8, 16, 50)
    assert resolve_shape(mode="cost", tile=None, d=1024, splits=None) == (2048, 1024, 16)


# ---------------------------------------------------------------------------
# script_sha: identity-provenance helper
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)
