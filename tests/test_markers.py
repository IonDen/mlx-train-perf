from conftest import _markers_to_skip


def test_no_flags_skips_all_gated() -> None:
    assert _markers_to_skip({}) == {"metal", "smoke", "benchmark", "network"}


def test_run_metal_unlocks_metal_only() -> None:
    assert _markers_to_skip({"metal": True}) == {"smoke", "benchmark", "network"}


def test_all_flags_skip_nothing() -> None:
    flags = {"metal": True, "smoke": True, "benchmark": True, "network": True}
    assert _markers_to_skip(flags) == set()
