import mlx.core as mx

from mlx_train_perf.core.guards import clamped_caps, install_guardrails, wired_cap_holds


def test_caps_below_requested_on_big_machine() -> None:
    dev_max = 25 * 1024**3  # ~M1 Max 32 GB
    wired, soft = clamped_caps(dev_max)
    assert wired == 20 * 1024**3 and soft == 22 * 1024**3  # noqa: PT018


def test_caps_clamp_on_small_machine() -> None:
    dev_max = 11 * 1024**3  # 16 GB Mac class — hardcoded 20 GB would DISABLE the guard
    wired, soft = clamped_caps(dev_max)
    assert wired == int(0.85 * dev_max) and soft == int(0.92 * dev_max)  # noqa: PT018
    assert wired < dev_max


def test_caps_unchanged_on_the_m1max_baseline() -> None:
    """0019 preservation pin: the REAL M1 Max 32 GB `max_recommended_working_set_size`
    (26,800,603,136 B, just under the 25 GiB proportional reference) must produce the
    EXACT 0.1.0 caps — every shipped calibration constant and bench gate was measured
    under them, so any drift here silently re-bases the project's memory evidence."""
    wired, soft = clamped_caps(26_800_603_136)
    assert wired == 20 * 1024**3
    assert soft == 22 * 1024**3


def test_caps_scale_proportionally_above_the_reference() -> None:
    """0019: the fixed ~20 GiB ceiling was 32 GB-tuned — on big machines it would forbid
    measuring big shapes (the community kit's whole purpose). Above the 25 GiB reference
    the wired cap grows with the device while always keeping headroom below dev_max."""
    for dev_gib in (57.6, 115.2, 460.8):  # 64 / 128 / 512 GB unified-memory classes
        dev_max = int(dev_gib * 1024**3)
        wired, soft = clamped_caps(dev_max)
        assert wired > 20 * 1024**3, f"{dev_gib}: cap stuck at the 32 GB-tuned ceiling"
        assert wired < dev_max  # panic-proof property
        assert wired < soft < dev_max


def test_caps_are_monotone_in_device_size() -> None:
    sizes = [int(g * 1024**3) for g in (8, 11, 24.96, 25, 40, 57.6, 115.2, 230, 460.8)]
    wireds = [clamped_caps(s)[0] for s in sizes]
    assert wireds == sorted(wireds)


def test_install_runs_on_this_machine() -> None:
    install_guardrails()  # must not raise; conftest already installed session caps
    assert mx.device_info()["max_recommended_working_set_size"] > 0


def test_install_guardrails_returns_the_previously_active_wired_limit() -> None:
    """`install_guardrails` now returns whatever wired limit was ACTIVE immediately
    before it re-asserted the house cap (`mx.set_wired_limit`'s own contract — MLX
    exposes no separate getter). `bench/worker.py`'s `train_step` condition reads this
    return value to OBSERVE whether the cap held through an mlx-lm training loop that
    raises it to the device max at its own entry."""
    install_guardrails()  # our own cap is active now
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, _soft = clamped_caps(dev_max)
    previous = install_guardrails()  # re-asserting a cap that's already active is a no-op
    assert previous == wired


def test_wired_cap_holds_true_when_observed_matches_expected() -> None:
    assert wired_cap_holds(observed_bytes=100, expected_bytes=100) is True


def test_wired_cap_holds_false_when_observed_differs_from_expected() -> None:
    assert wired_cap_holds(observed_bytes=100, expected_bytes=200) is False
