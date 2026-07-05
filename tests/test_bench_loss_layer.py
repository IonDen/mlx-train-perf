"""Pure-logic tests for `scripts/bench_loss_layer.py`. `scripts/` has no `__init__.py`
(matches `bench_quant_thresholds.py`'s/`bench_backward_ladder.py`'s existing
convention), so the module is loaded by path rather than via a package import.

Only the GPU-free helpers are covered here: `script_sha`, `build_conditions` (pure
condition-list construction), and `check_acceptance` (reads plain JSON artifacts from
disk, no MLX). The real `loss_layer` condition measurement itself is exercised
end-to-end via `--tiny` on the main session, not unit-tested here (same convention
`test_bench_backward_ladder.py` uses for its own script's Metal-backed conditions).
"""
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_loss_layer import (  # noqa: E402 -- import must follow the sys.path insert
    ACCEPTANCE_N,
    ACCEPTANCE_RATIO,
    IMPLS,
    N_VALUES,
    TINY_D,
    TINY_N_VALUES,
    TINY_V,
    D,
    V,
    build_conditions,
    check_acceptance,
    script_sha,
)


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_build_conditions_covers_the_full_production_grid() -> None:
    conditions = build_conditions(tiny=False)
    assert len(conditions) == len(N_VALUES) * len(IMPLS)
    names = {c.name for c in conditions}
    for n in N_VALUES:
        for impl in IMPLS:
            assert f"loss_layer_n{n}_{impl}" in names


def test_build_conditions_production_params_match_the_declared_grid() -> None:
    conditions = build_conditions(tiny=False)
    by_name = {c.name: c for c in conditions}
    cond = by_name[f"loss_layer_n{ACCEPTANCE_N}_kernel"]
    assert cond.kind == "loss_layer"
    assert cond.params["n"] == ACCEPTANCE_N
    assert cond.params["d"] == D
    assert cond.params["v"] == V
    assert cond.params["impl"] == "kernel"
    assert cond.params["dtype"] == "bfloat16"
    assert cond.params["script_sha"] == script_sha()


def test_build_conditions_tiny_uses_the_cheap_synthetic_shape() -> None:
    conditions = build_conditions(tiny=True)
    assert len(conditions) == len(TINY_N_VALUES) * len(IMPLS)
    for cond in conditions:
        assert cond.params["d"] == TINY_D
        assert cond.params["v"] == TINY_V
        assert cond.params["n"] in TINY_N_VALUES


def test_build_conditions_tiny_and_production_names_never_collide() -> None:
    """`--tiny` writes to a SEPARATE directory (`RESULTS_TINY`), but the condition
    NAMES should still differ too whenever the grids overlap in `n` -- here they
    don't (TINY_N_VALUES vs N_VALUES are disjoint), which this test pins so a future
    edit that made them overlap would be caught."""
    assert not (set(TINY_N_VALUES) & set(N_VALUES))


# ---------------------------------------------------------------------------
# check_acceptance: reads plain JSON artifacts directly, no MLX
# ---------------------------------------------------------------------------


def _write(path: Path, *, status: str, session_id: str = "s1", **fields: object) -> None:
    path.write_text(json.dumps({"status": status, "identity": {"session_id": session_id},
                                **fields}))


def test_check_acceptance_passes_when_kernel_at_or_under_the_ratio(tmp_path: Path) -> None:
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    _write(kernel, status="ok", wall_s=1.63)
    _write(naive, status="ok", wall_s=1.0)
    passed, reason = check_acceptance([kernel, naive])
    assert passed is True
    assert "1.630" in reason


def test_check_acceptance_fails_when_kernel_wall_exceeds_the_ratio(tmp_path: Path) -> None:
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    _write(kernel, status="ok", wall_s=2.5)
    _write(naive, status="ok", wall_s=1.0)
    passed, reason = check_acceptance([kernel, naive])
    assert passed is False
    assert "2.500" in reason


def test_check_acceptance_boundary_exactly_at_the_ratio_passes(tmp_path: Path) -> None:
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    _write(kernel, status="ok", wall_s=ACCEPTANCE_RATIO)
    _write(naive, status="ok", wall_s=1.0)
    passed, _reason = check_acceptance([kernel, naive])
    assert passed is True


def test_check_acceptance_skips_when_conditions_are_missing(tmp_path: Path) -> None:
    passed, reason = check_acceptance([tmp_path / "not_the_right_name.json"])
    assert passed is None
    assert "no n=" in reason


def test_check_acceptance_skips_when_a_condition_is_not_ok(tmp_path: Path) -> None:
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    _write(kernel, status="refused", error="watchdog budget")
    _write(naive, status="ok", wall_s=1.0)
    passed, reason = check_acceptance([kernel, naive])
    assert passed is None
    assert "refused" in reason


def test_check_acceptance_skips_when_artifact_is_corrupt(tmp_path: Path) -> None:
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    kernel.write_text("{not json")
    _write(naive, status="ok", wall_s=1.0)
    passed, reason = check_acceptance([kernel, naive])
    assert passed is None
    assert "error" in reason


def test_check_acceptance_refuses_a_cross_session_comparison(tmp_path: Path) -> None:
    """The exact hazard `bench.runner.report`'s own `cross_session_excluded` guards
    against: a resumed run where one condition's artifact survives from an EARLIER
    session (stale-but-fresh, so `run_conditions` skipped re-measuring it) while the
    other was measured fresh just now under a NEW session -- comparing their `wall_s`
    values would be an unsafe cross-session comparison, not a same-session ratio."""
    kernel = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_kernel.json"
    naive = tmp_path / f"loss_layer_n{ACCEPTANCE_N}_naive.json"
    _write(kernel, status="ok", wall_s=1.5, session_id="old_session")
    _write(naive, status="ok", wall_s=1.0, session_id="new_session")
    passed, reason = check_acceptance([kernel, naive])
    assert passed is None
    assert "different sessions" in reason
