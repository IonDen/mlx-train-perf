"""Wired-limit guardrails: a hard wired cap is what turns an over-budget run into a
clean MLX OOM instead of a kernel panic (unified memory; wired pages can't swap)."""
import mlx.core as mx


def clamped_caps(dev_max: int, *, wired_gb: int = 20, soft_gb: int = 22) -> tuple[int, int]:
    wired = min(wired_gb * 1024**3, int(0.85 * dev_max))
    soft = min(soft_gb * 1024**3, int(0.92 * dev_max))
    if wired >= dev_max:  # pragma: no cover — arithmetic guarantee, kept as a tripwire
        raise AssertionError(f"wired cap {wired} >= device max {dev_max}")
    return wired, soft


def install_guardrails(*, wired_gb: int = 20, soft_gb: int = 22) -> int:
    """Re-asserts this project's device-clamped wired/soft caps. Returns the wired
    limit that was ACTIVE immediately before this call (`mx.set_wired_limit`'s own
    contract: it returns the previous limit, and MLX exposes no separate getter) --
    every caller before this return value existed simply discarded it (a bare
    `install_guardrails()` statement), so adding it is backward compatible. A caller
    that needs to OBSERVE whether a cap held across intervening code that might have
    overridden it (e.g. `bench/worker.py`'s `train_step` condition, re-asserting after
    `mlx_lm.tuner.trainer.train()`'s own entry-time override) reads this return value --
    see `wired_cap_holds` for the decision that reading feeds."""
    dev_max = int(mx.device_info()["max_recommended_working_set_size"])
    wired, soft = clamped_caps(dev_max, wired_gb=wired_gb, soft_gb=soft_gb)
    previous = mx.set_wired_limit(wired)
    mx.set_memory_limit(soft)
    return previous


def wired_cap_holds(*, observed_bytes: int, expected_bytes: int) -> bool:
    """Pure decision: does an OBSERVED wired limit (an `install_guardrails` return
    value, read again after code that might have overridden the cap) match the
    EXPECTED house cap? Isolated as its own function so the decision is testable
    without any MLX device -- the hazard it exists to catch: `mlx_lm.tuner.trainer.
    train()` raises the wired limit to the device max at its own entry (site-packages/
    mlx_lm/tuner/trainer.py:229), silently overriding a cap installed before it runs."""
    return observed_bytes == expected_bytes
