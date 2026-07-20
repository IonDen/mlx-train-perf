import warnings

import mlx.core as mx
import pytest

from mlx_train_perf.core.guards import install_guardrails

_GATED = ("metal", "smoke", "benchmark", "network")


def _markers_to_skip(run_flags: dict[str, bool]) -> set[str]:
    """Pure gating decision: a gated marker is skipped unless its --run-<name> flag is set."""
    return {m for m in _GATED if not run_flags.get(m, False)}


_COMPARATIVE_PEAK_FLOOR_BYTES = 16 * 1024**3


def _machine_supports_comparative_peaks(memory_size_bytes: int) -> bool:
    """Pure decision: tests that compare two real `mx.get_peak_memory()` readings are
    only meaningful on machines with >= 16 GiB physical memory. On the 7 GB GitHub
    macos-15 runners the packed-walk arm's peak swings run to run (measured ratios
    0.7446/0.7595/0.9039 against a locally-deterministic 0.988), landing on both sides
    of the broken-regime signatures -- the comparison discriminates nothing there. The
    fix is this honest skip, NOT a wider assertion band: the tight bands are what let
    the comparisons catch a real checkpointing regression on developer machines."""
    return memory_size_bytes >= _COMPARATIVE_PEAK_FLOOR_BYTES


def _comparative_peaks_unsupported_here() -> bool:
    try:
        memory_size = int(mx.device_info()["memory_size"])
    except Exception:  # no usable Metal device -- peak comparisons are moot anyway
        return True
    return not _machine_supports_comparative_peaks(memory_size)


needs_comparative_peak_room = pytest.mark.skipif(
    _comparative_peaks_unsupported_here(),
    reason="machine below the 16 GiB floor: comparative peak-memory readings are "
    "nondeterministic on small shared runners",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    for m in _GATED:
        parser.addoption(f"--run-{m}", action="store_true", default=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    flags = {m: config.getoption(f"--run-{m}") for m in _GATED}
    to_skip = _markers_to_skip(flags)
    for item in items:
        for m in to_skip:
            if m in item.keywords:
                item.add_marker(pytest.mark.skip(reason=f"needs --run-{m}"))


@pytest.fixture(scope="session", autouse=True)
def _memory_guard() -> None:
    """House wired-limit caps, device-clamped (panic guard on any machine). Degrades
    gracefully where no usable Metal device exists (a CI runner without GPU must still
    run the pure default lane, not hard-fail at session setup)."""
    try:
        install_guardrails()
    except Exception as exc:  # broad by design — device query only; tests decide their own lane
        warnings.warn(f"memory guard not installed (no Metal device?): {exc}", stacklevel=1)
