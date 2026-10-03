# Planning memory for your own pipeline

`mlx_train_perf.memfit` is a small toolkit for answering one question before you launch a heavy run: will it fit? You describe your pipeline as a few phases, each phase as a handful of named byte counts, and the library tells you the predicted peak, which phase sets it, and the largest setting (an image side, a batch size, a context length) that stays under a budget. It also reads and writes calibration files, fits coefficients from your own measurements, and scores predictions against held-out runs.

It is plain Python. Importing it does not load MLX, and it needs numpy only for `fit_linear`. The names are not re-exported from the package root, so import them from the area:

```python
from mlx_train_perf.memfit import MemoryModel, Phase, estimate, fits, max_int_within_budget
```

`memfit` and `mlx_train_perf.machine` follow semantic versioning from 0.9.0. While the project is below 1.0, a breaking change to either one bumps the minor version and is listed under "Changed" in the changelog.

## One phase

A model is a tuple of phases. A phase maps term names to functions that take your own parameter object and return bytes. Here is a single-phase model of a transformer forward pass.

```python
from dataclasses import dataclass

from mlx_train_perf.memfit import MemoryModel, Phase, estimate, fits


@dataclass(frozen=True)
class Run:
    batch: int
    seq: int


GIB = 1024**3

model = MemoryModel(
    phases=(
        Phase(
            name="forward",
            terms={
                "weights": lambda r: 8 * GIB,
                "activations": lambda r: r.batch * r.seq * 4096 * 32 * 2,
            },
        ),
    ),
    overhead_frac=0.10,
)

est = estimate(model, Run(batch=1, seq=8192))
print(est.peak_phase, est.peak_bytes / GIB)
print(fits(est, budget_bytes=24 * GIB))
```

`overhead_frac` scales every phase total, so 0.10 means ten percent headroom on top of the sum of the terms. `estimate` evaluates each term once and returns a `PeakEstimate`. Its `components` field holds every term's whole-byte value, which is what you want when you are hunting for the term that dominates. A term that returns a negative or non-finite number raises an error that names the phase and the term.

Everything `memfit` returns is a prediction. `PeakEstimate.is_estimate` is always `True`.

## Several phases

Most pipelines run their stages one after another and free memory in between, so the peak is the largest stage, not the sum. A `MemoryModel` takes that shape directly: the predicted peak is the largest phase total. This example is a generic image pipeline with three stages. The numbers are made up.

```python
from dataclasses import dataclass

from mlx_train_perf.memfit import MemoryModel, Phase, estimate

GIB = 1024**3


@dataclass(frozen=True)
class Job:
    side: int      # image side in pixels
    batch: int = 1


def tokens(j: Job) -> int:
    return (j.side // 16) ** 2


pipeline = MemoryModel(
    phases=(
        Phase(name="encode", terms={
            "text_weights": lambda j: 5 * GIB,
            "activations": lambda j: 600 * 1024**2,
        }),
        Phase(name="denoise", terms={
            "weights": lambda j: 9 * GIB,
            "latents": lambda j: j.batch * tokens(j) * 3072 * 2,
            "attention": lambda j: j.batch * 24 * tokens(j) ** 2 * 2,
        }),
        Phase(name="decode", terms={
            "decoder_weights": lambda j: GIB // 3,
            "activations": lambda j: j.batch * j.side**2 * 3000,
        }),
    ),
    overhead_frac=0.10,
)

for side in (512, 1024, 2048):
    est = estimate(pipeline, Job(side=side))
    print(side, est.peak_phase, round(est.peak_bytes / GIB, 2),
          {k: round(v / GIB, 2) for k, v in est.phase_totals.items()})
```

Denoise sets the peak at every size here. At 512 and 1024 it is mostly weights. At 2048 the attention term, which grows with the fourth power of the side, takes it to 23 GiB while decode stays near 13. `est.components` shows which term to attack first.

## The largest setting that fits

`max_int_within_budget` bisects over an integer you choose and returns the largest value whose predicted peak is at or under the budget. You give it a function, `vary`, that turns the integer into your parameter object.

```python
from dataclasses import dataclass

from mlx_train_perf.memfit import MemoryModel, Phase, max_int_within_budget

GIB = 1024**3


@dataclass(frozen=True)
class Job:
    side: int


model = MemoryModel(phases=(
    Phase(name="decode", terms={
        "weights": lambda j: 6 * GIB,
        "activations": lambda j: j.side**2 * 3000,
    }),
))

# Sides come in multiples of 64, so search over the multiplier k.
k = max_int_within_budget(
    model, lambda k: Job(side=64 * k), lo=1, hi=64, budget_bytes=12 * GIB,
)
print(k, 64 * k)
```

Do the step mapping inside `vary`, as above, so the search never proposes a size your pipeline cannot run. How the search behaves:

- The peak must not decrease as the integer grows. You own that. The function checks only the two endpoints, and a falling pair raises `MemfitInputError`.
- If `lo` does not fit, you get `DoesNotFitError`. If `hi` fits, you get `hi` back.
- The `hi` endpoint is evaluated before the verdict on `lo`. A non-monotone pair, or an exception raised by `vary(hi)`, therefore shows up before a `DoesNotFitError` does.

## Calibration files

A calibration file stores fitted coefficients together with what they measure and where they came from. The format is JSON. Loading requires you to say what quantity you expect, and a file that measured something else is refused, so a number fit against one kind of memory reading cannot quietly price another.

Three quantity names are provided:

- `MLX_ACTIVE_MARGINAL`: the peak of `mx.get_active_memory()` during the measured window, minus its value just before the window. MLX's cache pool is not counted.
- `MLX_ACTIVE_TOTAL`: the absolute peak of MLX active memory. The cache pool is not counted here either.
- `MAX_FOOTPRINT_MLX_ACTIVE_CACHE`: the larger of the operating system's footprint for the process (`phys_footprint`) and MLX active plus cache memory.

```python
import tempfile
from pathlib import Path

from mlx_train_perf.memfit import (
    MLX_ACTIVE_TOTAL,
    MLX_ACTIVE_MARGINAL,
    CalibrationFile,
    CalibrationMismatchError,
    dump_calibration_file,
    load_calibration_file,
)

cal = CalibrationFile(
    measured_quantity=MLX_ACTIVE_TOTAL,
    coefficients={"bytes_per_pixel": 3000.0},
    terms={"bytes_per_pixel": "decode activations = bytes_per_pixel * side**2"},
    provenance={"chip": "Apple M1 Max", "ram_gib": "32", "date": "2026-10-03"},
)

path = Path(tempfile.mkdtemp()) / "decode.json"
dump_calibration_file(cal, path)

loaded = load_calibration_file(path, expect_quantity=MLX_ACTIVE_TOTAL)
print(loaded.coefficients)

try:
    load_calibration_file(path, expect_quantity=MLX_ACTIVE_MARGINAL)
except CalibrationMismatchError as exc:
    print("refused:", type(exc).__name__)
```

Provenance values are strings. The suggested keys are `machine`, `chip`, `ram_gib`, `macos`, `mlx_version`, `package_version`, `git_sha` and `date`. A file can also carry a scored hold-out record in its `holdout` field, described below.

## Fitting coefficients from measurements

`fit_linear` runs ordinary least squares: measured bytes against named features, with an intercept unless you turn it off. It needs numpy, which the `fit` extra installs:

```bash
pip install "mlx-train-perf[fit]"
```

```python
from mlx_train_perf.memfit import FitSample, fit_linear

GIB = 1024**3

samples = [
    FitSample(label=f"side{s}", features={"pixels": s * s},
              measured_bytes=6 * GIB + 3000 * s * s)
    for s in (512, 768, 1024, 1536)
]

fit = fit_linear(samples, features=["pixels"])
print({k: round(v, 1) for k, v in fit.coefficients.items()})
print(fit.max_abs_rel_error)
```

`max_abs_rel_error` is relative to the measured value. The hold-out band below is relative to the prediction.

The fit refuses to guess. It raises `MemfitInputError` for a rank-deficient design (two features that move together, or too few samples), for a missing, non-finite or duplicate-labelled sample, for a measurement that is not positive, and for a negative coefficient. A negative bytes-per-unit means the shape of the model is wrong, not that you found a memory refund. If you want to probe one anyway, list that feature in `allow_negative`. Pass `phase=` to fit only the samples tagged with a phase.

The rank check catches exactly collinear columns. It does not measure how badly conditioned the fit is, so a good fit on its own training points proves little. Check it on runs it has not seen.

## Hold-out runs

The protocol is: predict first, commit the predictions, measure, then score. `write_holdout_predictions` creates the predictions file once and refuses to overwrite it, because re-predicting after you have seen the results is exactly what the check exists to prevent.

```python
import tempfile
from pathlib import Path

from mlx_train_perf.memfit import (
    MLX_ACTIVE_TOTAL,
    HoldoutBand,
    score_holdout,
    write_holdout_predictions,
)

GIB = 1024**3
band = HoldoutBand(under_rel=0.05, over_rel=0.30)

path = Path(tempfile.mkdtemp()) / "holdout-predictions.json"
write_holdout_predictions(
    path,
    {"side1280": 14 * GIB, "side1792": 21 * GIB},
    band=band,
    measured_quantity=MLX_ACTIVE_TOTAL,
    provenance={"date": "2026-10-03"},
)

# ... commit the file, run the two jobs, record the peaks ...
measured = {"side1280": int(14.4 * GIB), "side1792": int(20.2 * GIB)}

report = score_holdout(path, measured, expect_quantity=MLX_ACTIVE_TOTAL)
print(report.out_of_band, round(report.max_under_rel, 4))
print(report.to_record()["band"])
```

A point is in band when `predicted * (1 - over_rel) <= measured <= predicted * (1 + under_rel)`. The band is lopsided on purpose. Measuring more than predicted can run you out of memory, while measuring less only wastes headroom. `report.unscored` lists predictions you never measured, `report.unpredicted` lists measurements nobody predicted, and `report.to_record()` produces the dictionary to store in a calibration file's `holdout` field.

## From a band to a budget

Once the band is known, give the search a smaller budget so that the predicted peak, plus the worst under-prediction you validated, plus a reserve for everything else on the machine, still stays under the memory you have. With `B` the total you can use and `reserve` the bytes you hold back:

```python
GIB = 1024**3
B = 26 * GIB
reserve = 2 * GIB
under_rel = 0.05

budget_bytes = int((B - reserve) // (1 + under_rel))
print(round(budget_bytes / GIB, 2))
```

That is the `budget_bytes` to pass to `max_int_within_budget`. A 5 percent band on a 26 GiB total with a 2 GiB reserve leaves about 22.86 GiB.

`mlx_train_perf.machine.detect_machine()` reports the working set Metal recommends (`recommended_working_set_bytes`). On a 32 GB M1 Max it reads 26,800,603,136 bytes, about 78 percent of RAM. Treat it as an upper bound, not a budget to plan against: subtract a reserve and your validated hold-out band first. An active-memory estimate leaves out MLX's retained buffer cache, so either bound that cache with `mx.set_cache_limit(...)` or make the reserve large enough to cover it. Never pass the working set to `mx.set_wired_limit` as it is; a wired cap has to sit strictly below it.
