"""Wired-limit guardrails: a hard wired cap is what turns an over-budget run into a
clean MLX OOM instead of a kernel panic (unified memory; wired pages can't swap)."""
import mlx.core as mx


def clamped_caps(dev_max: int, *, wired_gb: int = 20, soft_gb: int = 22) -> tuple[int, int]:
    wired = min(wired_gb * 1024**3, int(0.85 * dev_max))
    soft = min(soft_gb * 1024**3, int(0.92 * dev_max))
    if wired >= dev_max:  # pragma: no cover — arithmetic guarantee, kept as a tripwire
        raise AssertionError(f"wired cap {wired} >= device max {dev_max}")
    return wired, soft


def install_guardrails(*, wired_gb: int = 20, soft_gb: int = 22) -> None:
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, soft = clamped_caps(dev_max, wired_gb=wired_gb, soft_gb=soft_gb)
    mx.set_wired_limit(wired)
    mx.set_memory_limit(soft)
