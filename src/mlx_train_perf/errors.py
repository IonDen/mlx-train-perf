class MlxTrainPerfError(Exception):
    """Root of all mlx-train-perf errors."""


class UnsupportedHeadError(MlxTrainPerfError):
    """Head kind/dtype the requested impl cannot serve (no silent fallback)."""


class UnverifiedMlxError(MlxTrainPerfError):
    """Installed mlx version is not on the kernel's verified list."""


class LaunchBudgetError(MlxTrainPerfError):
    """A dispatch would exceed the GPU-watchdog budget; refused before launch."""


class LossInputError(MlxTrainPerfError):
    """Shape/dtype/target-range validation failure at the loss boundary."""


class PlanInputError(MlxTrainPerfError):
    """Shape/config validation failure in the fit planner (e.g. an unrecognized dtype)."""


class DoesNotFitError(MlxTrainPerfError):
    """Planner verdict: predicted peak exceeds the memory budget."""


class AdapterError(MlxTrainPerfError):
    """mlx-lm model can't be split (unsupported architecture, missing modules)."""


class MissingDependencyError(MlxTrainPerfError):
    """A lazily-imported optional dependency is absent."""


class BenchInputError(MlxTrainPerfError):
    """Bench condition input validation failure (e.g. a reserved identity param key)."""


class RegisterProbeError(MlxTrainPerfError):
    """Standalone MSL recompile for the register-pressure probe failed: a bad capture,
    a Metal compiler error, or an unexpected number of compiled kernel functions."""


class WiredCapRegressionError(MlxTrainPerfError):
    """A `train_step` condition's wired limit was not at this project's house cap
    after training completed -- the in-loop re-assert did not hold; the condition is a
    failed result, not a silent pass."""


class MemoryBudgetError(MlxTrainPerfError):
    """Measured available memory at watchdog-install time leaves an effective ceiling
    below the safe-start floor (memory_size // 4): the machine is too crowded to start
    this run safely. Raised per-rank at install time -- no silent degradation."""


class UnsupportedAttentionError(MlxTrainPerfError):
    """Attention impl/config combination the requested impl cannot serve (no silent
    fallback) -- e.g. an unsupported dtype, head_dim, or causal=False for impl='kernel'."""


class AttentionInputError(MlxTrainPerfError):
    """Shape validation failure at the flash_attention boundary (non-4D tensors,
    Hq not a multiple of Hkv, mismatched N/D/batch across q/k/v)."""


class MachineDetectionError(MlxTrainPerfError):
    """A machine-detection reader (e.g. `sysctl` for the CPU brand string) failed --
    subprocess/OS errors are mapped here so they hit the package's typed tool-error exit
    policy (2) instead of escaping as an uncaught traceback (exit 1)."""


class PackingError(MlxTrainPerfError):
    """Sequence-packing input validation failure (empty dataset, zero-length sequence,
    non-positive pack_len) -- no silent fallback."""
