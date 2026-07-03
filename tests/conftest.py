import warnings

import mlx.core as mx
import pytest

_GATED = ("metal", "smoke", "benchmark", "network")


def _markers_to_skip(run_flags: dict[str, bool]) -> set[str]:
    """Pure gating decision: a gated marker is skipped unless its --run-<name> flag is set."""
    return {m for m in _GATED if not run_flags.get(m, False)}


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
        dev_max = int(mx.device_info()["max_recommended_working_set_size"])
        mx.set_wired_limit(min(20 * 1024**3, int(0.85 * dev_max)))
        mx.set_memory_limit(min(22 * 1024**3, int(0.92 * dev_max)))
    except Exception as exc:  # broad by design — device query only; tests decide their own lane
        warnings.warn(f"memory guard not installed (no Metal device?): {exc}", stacklevel=1)
