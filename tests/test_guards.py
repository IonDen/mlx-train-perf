import mlx.core as mx

from mlx_train_perf.core.guards import clamped_caps, install_guardrails


def test_caps_below_requested_on_big_machine() -> None:
    dev_max = 25 * 1024**3  # ~M1 Max 32 GB
    wired, soft = clamped_caps(dev_max)
    assert wired == 20 * 1024**3 and soft == 22 * 1024**3  # noqa: PT018


def test_caps_clamp_on_small_machine() -> None:
    dev_max = 11 * 1024**3  # 16 GB Mac class — hardcoded 20 GB would DISABLE the guard
    wired, soft = clamped_caps(dev_max)
    assert wired == int(0.85 * dev_max) and soft == int(0.92 * dev_max)  # noqa: PT018
    assert wired < dev_max


def test_install_runs_on_this_machine() -> None:
    install_guardrails()  # must not raise; conftest already installed session caps
    assert mx.device_info()["max_recommended_working_set_size"] > 0
