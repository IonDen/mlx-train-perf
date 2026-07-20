from conftest import _machine_supports_comparative_peaks, _markers_to_skip


def test_no_flags_skips_all_gated() -> None:
    assert _markers_to_skip({}) == {"metal", "smoke", "benchmark", "network"}


def test_run_metal_unlocks_metal_only() -> None:
    assert _markers_to_skip({"metal": True}) == {"smoke", "benchmark", "network"}


def test_all_flags_skip_nothing() -> None:
    flags = {"metal": True, "smoke": True, "benchmark": True, "network": True}
    assert _markers_to_skip(flags) == set()


def test_comparative_peaks_unsupported_on_ci_runner_class() -> None:
    # The 7 GB GitHub macos-15 runner class, where the measurements are nondeterministic.
    assert _machine_supports_comparative_peaks(7 * 1024**3) is False


def test_comparative_peaks_supported_at_the_16gib_floor_and_above() -> None:
    assert _machine_supports_comparative_peaks(16 * 1024**3) is True
    assert _machine_supports_comparative_peaks(32 * 1024**3) is True


def test_comparative_peaks_unsupported_just_below_the_floor() -> None:
    assert _machine_supports_comparative_peaks(16 * 1024**3 - 1) is False
