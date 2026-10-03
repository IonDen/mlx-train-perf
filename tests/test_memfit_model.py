import math
from dataclasses import dataclass

import numpy as np
import pytest

from mlx_train_perf import memfit
from mlx_train_perf.memfit import (
    DoesNotFitError,
    MemfitInputError,
    MemoryModel,
    Phase,
    estimate,
    fits,
    max_int_within_budget,
)


@dataclass(frozen=True)
class _P:
    n: int


def _two_phase(overhead: float = 0.0) -> MemoryModel[_P]:
    # load: 100 + 10n ; run: 50 + 30n.  n=2: load 120, run 110 -> load binds.
    #                                   n=10: load 200, run 350 -> run binds.
    return MemoryModel(phases=(
        Phase(name="load", terms={"weights": lambda p: 100, "buffers": lambda p: 10 * p.n}),  # noqa: ARG005
        Phase(name="run", terms={"weights": lambda p: 50, "acts": lambda p: 30 * p.n}),  # noqa: ARG005
    ), overhead_frac=overhead)


def test_peak_is_the_largest_phase_not_the_sum() -> None:
    est = estimate(_two_phase(), _P(n=10))
    assert est.peak_bytes == 350
    assert est.peak_phase == "run"
    assert est.phase_totals == {"load": 200, "run": 350}
    assert est.components == {"load": {"weights": 100, "buffers": 100},
                              "run": {"weights": 50, "acts": 300}}


def test_the_binding_phase_moves_with_the_parameters() -> None:
    est = estimate(_two_phase(), _P(n=2))
    assert (est.peak_phase, est.peak_bytes) == ("load", 120)


def test_overhead_is_applied_to_every_phase_total() -> None:
    est = estimate(_two_phase(overhead=0.5), _P(n=10))
    assert est.phase_totals == {"load": 300, "run": 525}
    assert est.peak_bytes == 525


def test_a_tie_goes_to_the_first_phase() -> None:
    model = MemoryModel(phases=(Phase(name="a", terms={"x": lambda p: 7}),  # noqa: ARG005
                                Phase(name="b", terms={"x": lambda p: 7})))  # noqa: ARG005
    assert estimate(model, _P(n=0)).peak_phase == "a"


def test_terms_are_truncated_to_whole_bytes() -> None:
    model = MemoryModel(phases=(Phase(name="a", terms={"x": lambda p: 2.9, "y": lambda p: 1}),))  # noqa: ARG005
    est = estimate(model, _P(n=0))
    assert est.components == {"a": {"x": 2, "y": 1}}
    assert est.peak_bytes == 3


def test_numpy_scalars_are_accepted_as_terms() -> None:
    model = MemoryModel(phases=(Phase(name="a", terms={
        "i": lambda p: np.int64(5), "f": lambda p: np.float64(2.5)}),))  # noqa: ARG005
    assert estimate(model, _P(n=0)).peak_bytes == 7


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf])
def test_a_negative_or_non_finite_term_is_refused_with_its_name(bad: float) -> None:
    model = MemoryModel(phases=(Phase(name="run", terms={"acts": lambda p: bad}),))  # noqa: ARG005
    with pytest.raises(MemfitInputError, match="run/acts"):
        estimate(model, _P(n=0))


def test_an_exception_inside_a_term_propagates_unwrapped() -> None:
    class _OwnError(Exception):
        pass

    def _boom(_p: _P) -> int:
        raise _OwnError("caller's own error")

    model = MemoryModel(phases=(Phase(name="a", terms={"x": _boom}),))
    with pytest.raises(_OwnError):
        estimate(model, _P(n=0))


def test_a_model_needs_a_phase() -> None:
    with pytest.raises(MemfitInputError):
        MemoryModel(phases=())


def test_duplicate_phase_names_are_refused() -> None:
    with pytest.raises(MemfitInputError, match="dup"):
        MemoryModel(phases=(Phase(name="dup", terms={}), Phase(name="dup", terms={})))


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_overhead_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(MemfitInputError, match="overhead_frac"):
        MemoryModel(phases=(Phase(name="a", terms={}),), overhead_frac=bad)


def test_fits_compares_the_peak_with_the_budget_inclusively() -> None:
    est = estimate(_two_phase(), _P(n=10))      # peak 350
    assert fits(est, budget_bytes=350) is True
    assert fits(est, budget_bytes=349) is False


def test_an_estimate_is_labelled_as_a_prediction() -> None:
    assert estimate(_two_phase(), _P(n=1)).is_estimate is True


def test_memfit_public_names() -> None:
    expected = {"Phase", "MemoryModel", "PeakEstimate", "estimate", "fits", "max_int_within_budget"}
    assert expected <= set(memfit.__all__)


def _linear() -> MemoryModel[_P]:
    # peak = 1000 + 100 n
    return MemoryModel(phases=(Phase(name="a", terms={
        "base": lambda p: 1000, "per_n": lambda p: 100 * p.n}),))  # noqa: ARG005


def test_inverse_returns_the_exact_boundary() -> None:
    # budget 1750 -> n=7 gives 1700 (fits), n=8 gives 1800 (does not)
    got = max_int_within_budget(_linear(), lambda n: _P(n=n), lo=0, hi=1000, budget_bytes=1750)
    assert got == 7


def test_inverse_includes_a_value_landing_exactly_on_the_budget() -> None:
    got = max_int_within_budget(_linear(), lambda n: _P(n=n), lo=0, hi=1000, budget_bytes=1800)
    assert got == 8


def test_inverse_saturates_at_hi() -> None:
    got = max_int_within_budget(_linear(), lambda n: _P(n=n), lo=0, hi=5, budget_bytes=10**9)
    assert got == 5


def test_inverse_refuses_when_lo_does_not_fit() -> None:
    with pytest.raises(DoesNotFitError, match="999"):
        max_int_within_budget(_linear(), lambda n: _P(n=n), lo=0, hi=10, budget_bytes=999)


def test_inverse_refuses_non_monotone_endpoints() -> None:
    falling = MemoryModel(phases=(Phase(name="a", terms={"x": lambda p: 1000 - p.n}),))
    with pytest.raises(MemfitInputError, match="non-monotone"):
        max_int_within_budget(falling, lambda n: _P(n=n), lo=0, hi=10, budget_bytes=995)


@pytest.mark.parametrize(("lo", "hi"), [(-1, 5), (6, 5)])
def test_inverse_refuses_a_bad_range(lo: int, hi: int) -> None:
    with pytest.raises(MemfitInputError):
        max_int_within_budget(_linear(), lambda n: _P(n=n), lo=lo, hi=hi, budget_bytes=10**9)


def test_inverse_lets_the_callers_own_error_through() -> None:
    class _DomainError(Exception):
        pass

    def _vary(n: int) -> _P:
        if n == 500:
            raise _DomainError("no such resolution")
        return _P(n=n)

    with pytest.raises(_DomainError):
        max_int_within_budget(_linear(), _vary, lo=0, hi=1000, budget_bytes=1750)


def test_inverse_maps_steps_inside_vary() -> None:
    # sides in multiples of 64: peak = 1000 + 100 * side; budget 7500 -> side <= 65 -> k=1
    got = max_int_within_budget(_linear(), lambda k: _P(n=64 * k), lo=0, hi=50, budget_bytes=7500)
    assert got == 1
