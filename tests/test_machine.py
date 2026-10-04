"""Public machine-detection and preflight surface (`mlx_train_perf.machine`).

GPU-free: `detect_machine` takes injectable readers, so the device-info dict here is a
literal shaped like mlx 0.32.0's `mx.device_info()` on an M1 Max 32 GB (values copied
from a real read on 2026-10-03), and the preflight decision is pure.
"""
import subprocess
from importlib.metadata import version

import pytest
from test_memfit_imports import run_with_blocked_imports

from mlx_train_perf import machine
from mlx_train_perf.contribute import (
    build_community_artifact,
    shapes_for_ram,
)
from mlx_train_perf.contribute import (
    parse_chip as contribute_parse_chip,
)
from mlx_train_perf.contribute import (
    ram_gib_from_bytes as contribute_ram_gib_from_bytes,
)
from mlx_train_perf.errors import MachineDetectionError
from mlx_train_perf.machine import (
    MachineInfo,
    classify_memory_pressure,
    detect_machine,
    evaluate_preflight,
    machine_slug,
    parse_chip,
    ram_gib_from_bytes,
)

_M1_MAX_DEVICE_INFO: dict[str, object] = {
    "device_name": "Apple M1 Max",
    "max_recommended_working_set_size": 26800603136,
    "memory_size": 34359738368,
    "architecture": "applegpu_g13s",
    "max_buffer_length": 20100448256,
    "resource_limit": 499000,
}


def _detect(device_info: dict[str, object]) -> MachineInfo:
    return detect_machine(
        chip_reader=lambda: "Apple M1 Max\n",
        device_info_reader=lambda: device_info,
        macos_reader=lambda: "26.5.2",
        mlx_version_reader=lambda: "0.32.0",
        package_version_reader=lambda: "0.9.0",
    )


def test_public_surface_is_exactly_the_documented_names() -> None:
    """The stability promise covers these names; a rename or a dropped export breaks
    mlx-dfloat, which imports them."""
    assert sorted(machine.__all__) == [
        "MachineInfo", "Preflight", "classify_memory_pressure", "detect_machine",
        "evaluate_preflight", "machine_slug", "parse_chip", "ram_gib_from_bytes",
    ]


def test_detect_machine_records_ram_and_the_recommended_working_set_separately() -> None:
    info = _detect(_M1_MAX_DEVICE_INFO)
    assert info == MachineInfo(
        chip="Apple M1 Max", ram_gib=32, ram_bytes=34359738368,
        recommended_working_set_bytes=26800603136, gpu_architecture="applegpu_g13s",
        macos="26.5.2", mlx_version="0.32.0", package_version="0.9.0",
    )


def test_detect_machine_leaves_the_architecture_empty_when_device_info_lacks_it() -> None:
    info = _detect({k: v for k, v in _M1_MAX_DEVICE_INFO.items() if k != "architecture"})
    assert info.gpu_architecture is None
    assert info.recommended_working_set_bytes == 26800603136


@pytest.mark.parametrize("missing", ["memory_size", "max_recommended_working_set_size"])
def test_detect_machine_refuses_when_a_memory_field_is_missing(missing: str) -> None:
    """No honest default exists for either number: a 0 would fill a budget table with a
    machine that has no GPU budget."""
    device_info = {k: v for k, v in _M1_MAX_DEVICE_INFO.items() if k != missing}
    with pytest.raises(MachineDetectionError, match=missing):
        _detect(device_info)


def test_detect_machine_refuses_a_non_integer_memory_field() -> None:
    with pytest.raises(MachineDetectionError, match="max_recommended_working_set_size"):
        _detect({**_M1_MAX_DEVICE_INFO, "max_recommended_working_set_size": "26800603136"})


@pytest.mark.parametrize("bad", [True, False, 0, -1])
def test_detect_machine_refuses_bool_and_non_positive_memory_fields(bad: object) -> None:
    """Bug caught: `isinstance(True, int)` is true, so memory_size=True read as 1 byte, and
    a 0 working set would price every plan as not fitting."""
    with pytest.raises(MachineDetectionError, match="memory_size"):
        _detect({**_M1_MAX_DEVICE_INFO, "memory_size": bad})
    with pytest.raises(MachineDetectionError, match="max_recommended_working_set_size"):
        _detect({**_M1_MAX_DEVICE_INFO, "max_recommended_working_set_size": bad})


def test_default_mlx_version_reader_reads_metadata_without_loading_mlx() -> None:
    """Bug caught: reading `mx.__version__` initialises Metal just to get a version
    string. With mlx imports blocked the default reader must still answer."""
    r = run_with_blocked_imports(
        "from mlx_train_perf.machine import _read_mlx_version\n"
        "print(_read_mlx_version())"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == version("mlx")


def test_contribute_still_exports_the_parsers_it_used_to_define() -> None:
    assert contribute_parse_chip is parse_chip
    assert contribute_ram_gib_from_bytes is ram_gib_from_bytes


def test_evaluate_preflight_takes_a_plain_warning_string() -> None:
    """A consumer with its own watchdog has a warning string, not this package's
    `EffectiveCeiling`; the decision must accept that directly."""
    pf = evaluate_preflight(
        memory_pressure_state="normal", on_ac_power=True,
        ceiling_warning="measured available 20 GB is far below 58 GB",
    )
    assert pf.ok is True
    assert pf.warnings == ("measured available 20 GB is far below 58 GB",)


def test_community_artifact_is_schema_2_and_carries_the_working_set() -> None:
    art = build_community_artifact(
        machine=_detect(_M1_MAX_DEVICE_INFO), tier="quick", grid=shapes_for_ram(32),
        bench_summaries=[], generated_date="2026-10-03",
    )
    assert art["schema_version"] == 2
    block = art["machine"]
    assert isinstance(block, dict)
    assert block["recommended_working_set_bytes"] == 26800603136
    assert block["gpu_architecture"] == "applegpu_g13s"
    assert block["ram_bytes"] == 34359738368


def test_community_artifact_machine_block_holds_no_host_identity() -> None:
    """Privacy rule: chip, RAM, OS and versions only -- no hostname, user or path."""
    art = build_community_artifact(
        machine=_detect(_M1_MAX_DEVICE_INFO), tier="quick", grid=shapes_for_ram(32),
        bench_summaries=[], generated_date="2026-10-03",
    )
    block = art["machine"]
    assert isinstance(block, dict)
    assert sorted(block) == [
        "chip", "gpu_architecture", "macos", "mlx_version", "package_version", "ram_bytes",
        "ram_gib", "recommended_working_set_bytes",
    ]


# --- pure parsing ------------------------------------------------------------------------


def test_parse_chip_strips_the_sysctl_brand_string() -> None:
    assert parse_chip("Apple M1 Max\n") == "Apple M1 Max"


def test_parse_chip_collapses_internal_whitespace() -> None:
    assert parse_chip("  Apple  M2   Ultra  ") == "Apple M2 Ultra"


def test_ram_gib_from_bytes_rounds_to_the_nearest_gib() -> None:
    assert ram_gib_from_bytes(34359738368) == 32     # exactly 32 GiB
    assert ram_gib_from_bytes(68719476736) == 64
    assert ram_gib_from_bytes(17179869184) == 16


def test_machine_slug_is_filesystem_safe_and_carries_ram() -> None:
    assert machine_slug(chip="Apple M1 Max", ram_gib=32) == "apple-m1-max-32gb"


# --- pre-flight decision (pure) -------------------------------------------------------


def test_classify_memory_pressure_reads_the_free_percentage_line() -> None:
    assert classify_memory_pressure("System-wide memory free percentage: 91%") == "normal"
    assert classify_memory_pressure("System-wide memory free percentage: 20%") == "warn"
    assert classify_memory_pressure("System-wide memory free percentage: 4%") == "red"


def test_classify_memory_pressure_degrades_to_normal_when_unparseable() -> None:
    """A missing free-percentage line must NOT read as red (the real panic guard is the
    effective-ceiling refusal, not this coarse gate) -- it degrades to normal."""
    assert classify_memory_pressure("garbage with no percentage line") == "normal"


def test_evaluate_preflight_refuses_on_red_memory() -> None:
    pf = evaluate_preflight(
        memory_pressure_state="red", on_ac_power=True,
        ceiling_warning=None,
    )
    assert pf.ok is False
    assert pf.refusal is not None
    assert "memory" in pf.refusal.lower()


def test_evaluate_preflight_warns_on_battery_but_proceeds() -> None:
    pf = evaluate_preflight(
        memory_pressure_state="normal", on_ac_power=False,
        ceiling_warning=None,
    )
    assert pf.ok is True
    assert pf.refusal is None
    assert any("AC" in w or "battery" in w.lower() for w in pf.warnings)


def test_evaluate_preflight_surfaces_the_divergence_warning_prominently() -> None:
    """The 0021 memory-divergence warning ('expected ~58 GB free, measured 20 GB') is
    exactly the kit's audience -- someone on a crowded machine must see it up front."""
    pf = evaluate_preflight(
        memory_pressure_state="normal", on_ac_power=True,
        ceiling_warning="measured available 20 GB is far below 58 GB",
    )
    assert pf.ok is True
    assert any("20 GB" in w for w in pf.warnings)


def test_real_non_metal_machine_readers_return_plausible_values() -> None:
    """The subprocess/platform readers run on any macOS without a Metal device --
    exercised for real, unlike the Metal `mx.device_info()` reader (pragma'd)."""
    assert machine._read_chip()                       # non-empty sysctl brand string
    assert machine._read_package_version()
    assert isinstance(machine._read_macos(), str)


# --- _read_chip: subprocess failures map to the typed tool-error path (finding E) -----


def test_read_chip_wraps_a_called_process_error_in_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing `sysctl` (nonzero exit under `check=True`) must not escape as a raw
    `CalledProcessError` -- that traceback would bypass `main`'s `MlxTrainPerfError`
    catch and exit 1 (an uncaught crash) instead of the package's tool-error exit 2."""
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["sysctl", "-n", "machdep.cpu.brand_string"])

    monkeypatch.setattr(machine.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError, match="sysctl"):
        machine._read_chip()


def test_read_chip_wraps_a_timeout_in_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["sysctl"], timeout=10)

    monkeypatch.setattr(machine.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError):
        machine._read_chip()


def test_read_chip_wraps_a_missing_binary_in_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(machine.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError):
        machine._read_chip()
