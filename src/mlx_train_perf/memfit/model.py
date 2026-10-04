"""Phase-structured peak-memory estimates.

A `MemoryModel` is a tuple of `Phase`s. Each phase is an ordered mapping of named terms;
each term is a function from the caller's own parameter object to bytes. A phase's total
is the sum of its terms times `(1 + overhead_frac)`, and the predicted peak is the
largest phase total. Every result is a prediction (`PeakEstimate.is_estimate`).
"""
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from mlx_train_perf.memfit.errors import DoesNotFitError, MemfitInputError

__all__ = ["MemoryModel", "PeakEstimate", "Phase", "estimate", "fits", "max_int_within_budget"]

P = TypeVar("P")


@dataclass(frozen=True, slots=True, kw_only=True)
class Phase(Generic[P]):
    """One phase of a run. `terms` maps a term name to a function of the parameters that
    returns bytes; insertion order is the evaluation and report order."""

    name: str
    terms: Mapping[str, Callable[[P], int | float]]


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryModel(Generic[P]):
    """At least one phase, unique names; `overhead_frac` (finite, >= 0) scales every
    phase total."""

    phases: tuple[Phase[P], ...]
    overhead_frac: float = 0.0

    def __post_init__(self) -> None:
        if not self.phases:
            raise MemfitInputError("a MemoryModel needs at least one phase")
        names = [phase.name for phase in self.phases]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise MemfitInputError(f"duplicate phase names: {dupes}")
        if not math.isfinite(self.overhead_frac) or self.overhead_frac < 0:
            raise MemfitInputError(
                f"overhead_frac must be finite and >= 0 (got {self.overhead_frac!r})"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PeakEstimate:
    """A predicted peak: the largest phase total, which phase it is (the first one on a
    tie), every phase total, and every term's whole-byte value."""

    peak_bytes: int
    peak_phase: str
    phase_totals: dict[str, int]
    components: dict[str, dict[str, int]]
    is_estimate: bool = True


def _term_bytes(phase: str, term: str, value: object) -> int:
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0):
        raise MemfitInputError(
            f"term {phase}/{term} returned {value!r}; bytes must be a finite number >= 0"
        )
    return int(float(value))


def estimate(model: MemoryModel[P], params: P) -> PeakEstimate:
    """Evaluate every term once, in order. A term's own exception propagates unchanged;
    a negative or non-finite value raises `MemfitInputError` naming `phase/term`."""
    components: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for phase in model.phases:
        values = {name: _term_bytes(phase.name, name, term(params))
                  for name, term in phase.terms.items()}
        components[phase.name] = values
        totals[phase.name] = int(sum(values.values()) * (1 + model.overhead_frac))
    peak_phase = max(totals, key=totals.__getitem__)
    return PeakEstimate(peak_bytes=totals[peak_phase], peak_phase=peak_phase,
                        phase_totals=totals, components=components)


def fits(est: PeakEstimate, *, budget_bytes: int) -> bool:
    """True when the predicted peak is at or under `budget_bytes`."""
    return est.peak_bytes <= budget_bytes


def max_int_within_budget(
    model: MemoryModel[P], vary: Callable[[int], P], *, lo: int, hi: int, budget_bytes: int,
) -> int:
    """Largest `v` in `[lo, hi]` whose predicted peak `estimate(model, vary(v))` is at
    or under `budget_bytes`, by bisection.

    The caller owns monotonicity: the peak must not decrease as `v` grows. Only the two
    endpoints are checked (a falling pair raises `MemfitInputError`). Raises
    `DoesNotFitError` when `lo` itself does not fit, and returns `hi` when `hi` fits.
    Map steps inside `vary` (e.g. `vary=lambda k: params_at(side=64 * k)`). Exceptions
    raised by `vary` or a term propagate unchanged. The `hi` endpoint is evaluated and
    its monotonicity checked before the `lo` fit verdict, so a non-monotone pair raises
    `MemfitInputError` even when `lo` does not fit, and any exception from `vary(hi)`
    surfaces before `DoesNotFitError`."""
    if lo < 0 or lo > hi:
        raise MemfitInputError(f"need 0 <= lo <= hi (got lo={lo}, hi={hi})")

    def peak(v: int) -> int:
        return estimate(model, vary(v)).peak_bytes

    lo_peak = peak(lo)
    hi_peak = peak(hi)
    if hi_peak < lo_peak:
        raise MemfitInputError(
            f"non-monotone model: predicted peak falls from {lo_peak} at {lo} to "
            f"{hi_peak} at {hi}; bisection needs a peak that never decreases"
        )
    if lo_peak > budget_bytes:
        raise DoesNotFitError(
            f"nothing in [{lo}, {hi}] fits budget_bytes={budget_bytes} "
            f"(predicted peak at {lo} is {lo_peak} bytes)"
        )
    if hi_peak <= budget_bytes:
        return hi
    fit_v, miss_v = lo, hi              # invariant: peak(fit_v) fits, peak(miss_v) does not
    while miss_v - fit_v > 1:
        mid = (fit_v + miss_v) // 2
        if peak(mid) <= budget_bytes:
            fit_v = mid
        else:
            miss_v = mid
    return fit_v
