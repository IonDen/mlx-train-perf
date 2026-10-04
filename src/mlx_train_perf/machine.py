"""Machine detection and the pre-flight decision, as a public API.

`detect_machine()` reports what a benchmark or a memory budget needs to know about this
Mac: the chip, physical RAM, the GPU working set Metal recommends, the GPU architecture,
and the macOS / mlx / package versions. It records no hostname, user name or path.

The recommended working set is `mx.device_info()["max_recommended_working_set_size"]`,
the GPU working set Metal recommends (advisory, not enforced). It is not a fixed share of
RAM: an M1 Max with 32 GB reports 26,800,603,136 bytes (78 %), and smaller machines report
a lower share.

`evaluate_preflight()` is the pure go / no-go decision before a heavy run: it refuses on
critical memory pressure and warns on battery power, elevated pressure, or a
caller-supplied memory warning.

Both are stable public API from 0.9.0. This module imports mlx only inside the device
reader, never at import time.
"""
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version

from mlx_train_perf.errors import MachineDetectionError

__all__ = [
    "MachineInfo",
    "Preflight",
    "classify_memory_pressure",
    "detect_machine",
    "evaluate_preflight",
    "machine_slug",
    "parse_chip",
    "ram_gib_from_bytes",
]


# --- machine detection ----------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MachineInfo:
    """One machine's identity for benchmark provenance and memory budgets.

    `ram_bytes` is physical RAM (`memory_size`); `recommended_working_set_bytes` is the
    GPU budget Metal recommends (`max_recommended_working_set_size`); `gpu_architecture`
    is the device-info `architecture` string (e.g. `"applegpu_g13s"`), `None` when the
    installed mlx does not report one."""

    chip: str
    ram_gib: int
    ram_bytes: int
    recommended_working_set_bytes: int
    gpu_architecture: str | None
    macos: str
    mlx_version: str
    package_version: str


def parse_chip(brand_string: str) -> str:
    """Normalize the `sysctl machdep.cpu.brand_string` output: strip and collapse any
    internal whitespace run to a single space (`"Apple  M2   Ultra"` -> `"Apple M2
    Ultra"`)."""
    return " ".join(brand_string.split())


def ram_gib_from_bytes(ram_bytes: int) -> int:
    """Physical RAM in GiB, rounded to the nearest whole GiB -- `mx.device_info()`'s
    `memory_size` is exact powers of two on Apple Silicon (32 GiB -> exactly 32)."""
    return round(ram_bytes / 1024**3)


def machine_slug(*, chip: str, ram_gib: int) -> str:
    """Filesystem-safe machine identifier carrying the RAM class, e.g.
    `apple-m1-max-32gb`."""
    return f"{chip.lower().replace(' ', '-')}-{ram_gib}gb"


def _read_chip() -> str:
    """The `sysctl` brand-string reader. A chip read failure has no honest default, so a
    subprocess/OS failure (missing binary, nonzero exit, timeout) is mapped to the typed
    `MachineDetectionError`, not left to escape as a raw traceback -- the CLI only
    catches `MlxTrainPerfError`, so an unmapped `CalledProcessError` would exit 1 (an
    uncaught crash) instead of this package's tool-error exit 2."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise MachineDetectionError(
            f"failed to read the CPU brand string via `sysctl`: {exc}"
        ) from exc
    return parse_chip(out)


def _read_device_info() -> Mapping[str, object]:  # pragma: no cover -- Metal device query
    import mlx.core as mx  # noqa: PLC0415

    return dict(mx.device_info())


def _read_macos() -> str:
    return platform.mac_ver()[0]


def _read_package_version() -> str:
    return version("mlx-train-perf")


def _read_mlx_version() -> str:
    """The installed mlx distribution's version, read from package metadata so that
    importing mlx (and initialising Metal) is not needed just to learn a version string."""
    return version("mlx")


def _required_int(device_info: Mapping[str, object], key: str) -> int:
    value = device_info.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MachineDetectionError(
            f"mx.device_info() has no positive integer {key!r} (got {value!r})"
        )
    return value


def detect_machine(
    *,
    chip_reader: Callable[[], str] = _read_chip,
    device_info_reader: Callable[[], Mapping[str, object]] = _read_device_info,
    macos_reader: Callable[[], str] = _read_macos,
    mlx_version_reader: Callable[[], str] = _read_mlx_version,
    package_version_reader: Callable[[], str] = _read_package_version,
) -> MachineInfo:
    """Assemble a `MachineInfo` from injectable readers -- the real ones read `sysctl`,
    `mx.device_info()`, `platform.mac_ver()`, and installed package versions; tests inject
    fakes so this is GPU-free and subprocess-free under test.

    Raises `MachineDetectionError` when the device info lacks a positive integer `memory_size`
    or `max_recommended_working_set_size`: neither has an honest default."""
    device_info = device_info_reader()
    ram_bytes = _required_int(device_info, "memory_size")
    working_set = _required_int(device_info, "max_recommended_working_set_size")
    arch = device_info.get("architecture")
    return MachineInfo(
        chip=parse_chip(chip_reader()),
        ram_gib=ram_gib_from_bytes(ram_bytes),
        ram_bytes=ram_bytes,
        recommended_working_set_bytes=working_set,
        gpu_architecture=arch if isinstance(arch, str) and arch else None,
        macos=macos_reader(),
        mlx_version=mlx_version_reader(),
        package_version=package_version_reader(),
    )


# --- pre-flight (pure decision) -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Preflight:
    """The pre-flight verdict: `ok` is False exactly when `refusal` is set; `warnings`
    are reasons to distrust the measurement that do not block the run."""

    ok: bool
    refusal: str | None
    warnings: tuple[str, ...]


_RED_FREE_PCT = 10.0
_WARN_FREE_PCT = 25.0


def classify_memory_pressure(text: str) -> str:
    """Classify `memory_pressure`'s output as `"normal"`/`"warn"`/`"red"` off its
    `System-wide memory free percentage: N%` line (red below 10 %, warn below 25 %). A
    missing/unparseable line degrades to `"normal"` -- the real panic guard is the
    run's own memory watchdog, not this coarse gate, so a parse hiccup must never
    falsely block a healthy machine."""
    match = re.search(r"free percentage:\s*([\d.]+)\s*%", text)
    if match is None:
        return "normal"
    free_pct = float(match.group(1))
    if free_pct < _RED_FREE_PCT:
        return "red"
    if free_pct < _WARN_FREE_PCT:
        return "warn"
    return "normal"


def evaluate_preflight(
    *, memory_pressure_state: str, on_ac_power: bool, ceiling_warning: str | None,
) -> Preflight:
    """Pure pre-flight decision. REFUSES only on a red memory-pressure state. Everything
    else proceeds with WARNINGS: running on battery (measurements drift under power
    throttling), the caller's `ceiling_warning` (e.g. this package's
    `EffectiveCeiling.warning`: "expected ~N GB free, measured M GB"), and an elevated
    memory-pressure state, in that order."""
    warnings: list[str] = []
    refusal: str | None = None
    if memory_pressure_state == "red":
        refusal = (
            "system memory pressure is critical (red); refusing to start a heavy GPU run "
            "-- close other applications and retry"
        )
    if not on_ac_power:
        warnings.append(
            "running on battery power -- plug in AC power for stable measurements "
            "(power throttling on battery distorts wall-clock timing)"
        )
    if ceiling_warning is not None:
        warnings.append(ceiling_warning)
    if memory_pressure_state == "warn":
        warnings.append(
            "system memory pressure is elevated -- other processes are using memory; "
            "measurements may be affected"
        )
    return Preflight(ok=refusal is None, refusal=refusal, warnings=tuple(warnings))
