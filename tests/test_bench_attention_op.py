"""Pure-logic + gated-smoke tests for `scripts/bench_attention_op.py` (Task 11, spec
§8/§10.6 -- the single-op O(N) memory-scaling proof).

`scripts/` has no `__init__.py` (matches `bench_backward_ladder.py`'s existing
convention), so the module is loaded by path rather than via a package import.

Everything below the smoke test is GPU-free: `build_conditions` (pure grid math),
`compute_doubling_ratios` (pure ratio math over SYNTHETIC artifact dicts -- the flash
~2x / stock ~3.8x assertions belong HERE, never against a real GPU measurement in a
committed test; the real numbers are T13's campaign), and `run_grid`'s resume-by-skip
orchestration (subprocess spawning stubbed out, same pattern
`test_bench_resume.py::test_run_conditions_skips_fresh_without_spawning_a_subprocess`
uses for `bench.runner.run_conditions`). Only the final smoke test actually drives the
Metal kernel + `math_attention` autodiff at a tiny shape, gated behind
`--run-benchmark`.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_attention_op  # noqa: E402 -- import must follow the sys.path insert
from bench_attention_op import (  # noqa: E402
    AttnCondition,
    build_conditions,
    compute_doubling_ratios,
    decompose_walls,
    main,
    run_grid,
    script_sha,
)

from mlx_train_perf.bench.artifacts import (  # noqa: E402
    condition_identity,
    new_session_id,
    write_result,
)
from mlx_train_perf.core.guards import (  # noqa: E402
    DEFAULT_WALL_BUDGET_S,
    EffectiveCeiling,
)

# ---------------------------------------------------------------------------
# build_conditions: pure grid (impl x seq_lens) construction
# ---------------------------------------------------------------------------


def test_build_conditions_grid_times_impl() -> None:
    conditions = build_conditions(
        impls=("flash", "stock"), seq_lens=(2048, 4096),
        head_dim=128, heads=32, kv_heads=8,
    )
    assert len(conditions) == 4
    pairs = {(c.impl, c.n) for c in conditions}
    assert pairs == {("flash", 2048), ("flash", 4096), ("stock", 2048), ("stock", 4096)}


def test_build_conditions_carries_shape_fields() -> None:
    conditions = build_conditions(
        impls=("flash",), seq_lens=(2048,), head_dim=64, heads=4, kv_heads=2,
    )
    assert conditions == [
        AttnCondition(name="flash_n2048", impl="flash", n=2048, head_dim=64, heads=4, kv_heads=2)
    ]


def test_build_conditions_names_are_unique() -> None:
    conditions = build_conditions(
        impls=("flash", "stock"), seq_lens=(2048, 4096, 8192),
        head_dim=128, heads=32, kv_heads=8,
    )
    names = [c.name for c in conditions]
    assert len(names) == len(set(names))


def test_build_conditions_single_impl_single_seqlen_is_one_condition() -> None:
    conditions = build_conditions(
        impls=("flash",), seq_lens=(256,), head_dim=128, heads=32, kv_heads=8,
    )
    assert len(conditions) == 1
    assert conditions[0].name == "flash_n256"


# ---------------------------------------------------------------------------
# compute_doubling_ratios: pure ratio math over SYNTHETIC artifact dicts -- the
# flash ~2x (O(N)-class) / stock ~3.8x (spec §8's measured O(N^2) baseline) assertions
# live HERE, not against any real GPU measurement.
# ---------------------------------------------------------------------------


def _entry(*, impl: str, n: int, fwdbwd_peak_gb: float) -> dict[str, object]:
    return {
        "impl": impl, "n": n, "fwd_peak_gb": fwdbwd_peak_gb / 2,
        "fwdbwd_peak_gb": fwdbwd_peak_gb,
    }


def test_compute_doubling_ratios_flash_is_approximately_2x() -> None:
    entries = [
        _entry(impl="flash", n=2048, fwdbwd_peak_gb=1.0),
        _entry(impl="flash", n=4096, fwdbwd_peak_gb=2.0),
        _entry(impl="flash", n=8192, fwdbwd_peak_gb=4.0),
    ]
    ratios = compute_doubling_ratios(entries)
    assert ratios["flash"]["2048->4096"] == pytest.approx(2.0)
    assert ratios["flash"]["4096->8192"] == pytest.approx(2.0)


def test_compute_doubling_ratios_stock_is_approximately_3_8x() -> None:
    entries = [
        _entry(impl="stock", n=2048, fwdbwd_peak_gb=1.0),
        _entry(impl="stock", n=4096, fwdbwd_peak_gb=3.8),
        _entry(impl="stock", n=8192, fwdbwd_peak_gb=14.44),
    ]
    ratios = compute_doubling_ratios(entries)
    assert ratios["stock"]["2048->4096"] == pytest.approx(3.8)
    assert ratios["stock"]["4096->8192"] == pytest.approx(3.8, rel=1e-6)


def test_compute_doubling_ratios_only_pairs_exact_doublings() -> None:
    # 4096 -> 16384 is NOT an exact doubling (missing the 8192 midpoint) -- must not
    # be reported as a ratio pair.
    entries = [
        _entry(impl="flash", n=2048, fwdbwd_peak_gb=1.0),
        _entry(impl="flash", n=4096, fwdbwd_peak_gb=2.0),
        _entry(impl="flash", n=16384, fwdbwd_peak_gb=8.0),
    ]
    ratios = compute_doubling_ratios(entries)
    assert ratios["flash"] == {"2048->4096": pytest.approx(2.0)}


def test_compute_doubling_ratios_ignores_non_ok_shaped_entries() -> None:
    entries = [
        {"impl": "flash", "n": 2048},  # missing fwdbwd_peak_gb
        {"n": 4096, "fwdbwd_peak_gb": 2.0},  # missing impl
        {"impl": "flash", "n": "bogus", "fwdbwd_peak_gb": 2.0},  # n not an int
    ]
    assert compute_doubling_ratios(entries) == {}


def test_compute_doubling_ratios_keeps_impls_independent() -> None:
    entries = [
        _entry(impl="flash", n=2048, fwdbwd_peak_gb=1.0),
        _entry(impl="flash", n=4096, fwdbwd_peak_gb=2.0),
        _entry(impl="stock", n=2048, fwdbwd_peak_gb=1.0),
        _entry(impl="stock", n=4096, fwdbwd_peak_gb=3.8),
    ]
    ratios = compute_doubling_ratios(entries)
    assert set(ratios) == {"flash", "stock"}
    assert ratios["flash"]["2048->4096"] != ratios["stock"]["2048->4096"]


# ---------------------------------------------------------------------------
# decompose_walls: pure fwd/bwd wall decomposition over the two per-rep wall lists --
# the recorded quantities behind the README's forward/backward split claim (RC review
# finding 2: the split must come from THIS committed script's artifact, not from
# gitignored development runs).
# ---------------------------------------------------------------------------


def test_decompose_walls_medians_and_backward_remainder() -> None:
    split = decompose_walls([0.1, 0.2, 0.3], [0.5, 0.7, 0.6])
    assert split["fwd_wall_s"] == pytest.approx(0.2)
    assert split["bwd_wall_s"] == pytest.approx(0.4)
    assert split["bwd_over_fwd"] == pytest.approx(2.0)


def test_decompose_walls_uses_medians_not_means() -> None:
    # One wild outlier per list must not move the decomposition.
    split = decompose_walls([0.1, 0.1, 10.0], [0.3, 0.3, 30.0])
    assert split["fwd_wall_s"] == pytest.approx(0.1)
    assert split["bwd_wall_s"] == pytest.approx(0.2)


def test_decompose_walls_omits_ratio_for_a_nonpositive_forward() -> None:
    # Degenerate (never expected from perf_counter deltas, but the helper is pure):
    # no division blow-up, the ratio key is simply absent.
    split = decompose_walls([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    assert split["fwd_wall_s"] == 0.0
    assert split["bwd_wall_s"] == pytest.approx(0.5)
    assert "bwd_over_fwd" not in split


# ---------------------------------------------------------------------------
# script_sha: identity-provenance helper (same convention as
# bench_backward_ladder.py's own script_sha -- CODE_SHA_DEPS excludes ad hoc bench
# scripts, so this is what invalidates a stale artifact when THIS script's own
# measurement logic changes).
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


# ---------------------------------------------------------------------------
# run_grid: resume-by-skip orchestration (subprocess spawning stubbed, matching
# test_bench_resume.py's `runner.run_conditions` coverage pattern).
# ---------------------------------------------------------------------------


def _condition() -> AttnCondition:
    return AttnCondition(name="flash_n256", impl="flash", n=256, head_dim=64, heads=4, kv_heads=2)


def test_run_grid_skips_fresh_without_spawning_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = new_session_id()
    condition = _condition()
    out_path = tmp_path / f"{condition.name}.json"
    ident = condition_identity(
        kind="attention_op", session_id=session_id,
        params=bench_attention_op._params_for(condition),
        attention_impl=condition.impl,
    )
    write_result(out_path, ident, "ok", impl="flash", n=256, fwd_peak_gb=0.01,
                fwdbwd_peak_gb=0.02, wall_s=0.001)

    def _must_not_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called for a fresh artifact")

    monkeypatch.setattr(bench_attention_op, "_spawn_condition", _must_not_spawn)
    paths = run_grid([condition], out_dir=tmp_path, session_id=session_id)
    assert paths == [out_path]
    assert json.loads(out_path.read_text())["wall_s"] == 0.001  # untouched


def test_run_grid_spawns_for_a_stale_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = new_session_id()
    condition = _condition()
    calls: list[AttnCondition] = []

    def _fake_spawn(
        c: AttnCondition, *, out_dir: Path, session_id: str,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(c)
        ident = condition_identity(
            kind="attention_op", session_id=session_id,
            params=bench_attention_op._params_for(c), attention_impl=c.impl,
        )
        write_result(out_dir / f"{c.name}.json", ident, "ok", impl=c.impl, n=c.n,
                    fwd_peak_gb=0.01, fwdbwd_peak_gb=0.02, wall_s=0.001)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench_attention_op, "_spawn_condition", _fake_spawn)
    paths = run_grid([condition], out_dir=tmp_path, session_id=session_id)
    assert calls == [condition]
    assert json.loads(paths[0].read_text())["status"] == "ok"


def test_run_grid_records_error_envelope_on_subprocess_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _condition()

    def _fake_crash(
        _c: AttnCondition, *, out_dir: Path, session_id: str,  # noqa: ARG001
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(bench_attention_op, "_spawn_condition", _fake_crash)
    paths = run_grid([condition], out_dir=tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerCrashed"
    assert "boom" in data["error_msg"]


def test_run_grid_records_error_when_subprocess_exits_zero_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _condition()

    def _fake_silent(
        _c: AttnCondition, *, out_dir: Path, session_id: str,  # noqa: ARG001
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench_attention_op, "_spawn_condition", _fake_silent)
    paths = run_grid([condition], out_dir=tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerExitedWithoutArtifact"


def test_run_grid_respects_a_breach_artifact_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FINDING H1 (safety review): a subprocess that wrote its OWN honest `aborted_*`
    artifact before hard-exiting (the watchdog `os._exit(70)` breach path) must have that
    record PRESERVED, not clobbered by the generic `WorkerCrashed` envelope -- mirrors
    `bench.runner.run_conditions`'s respect contract (`out_path.exists()` guard) exactly.
    Because the pre-spawn `unlink` already guarantees an existing artifact is this
    subprocess's own write, a nonzero exit + an existing artifact means the subprocess
    recorded its own honest breach and must not be overwritten."""
    condition = _condition()

    def _fake_breach(
        c: AttnCondition, *, out_dir: Path, session_id: str,
    ) -> subprocess.CompletedProcess[str]:
        ident = condition_identity(
            kind="attention_op", session_id=session_id,
            params=bench_attention_op._params_for(c), attention_impl=c.impl,
        )
        write_result(
            out_dir / f"{c.name}.json", ident, "aborted_memory_ceiling",
            observed_active_gb=32.4, ceiling_gb=28.0,
        )
        return subprocess.CompletedProcess(args=[], returncode=70, stdout="", stderr="paging storm")

    monkeypatch.setattr(bench_attention_op, "_spawn_condition", _fake_breach)
    paths = run_grid([condition], out_dir=tmp_path, session_id=new_session_id())
    data = json.loads(paths[0].read_text())
    assert data["status"] == "aborted_memory_ceiling"
    assert data["observed_active_gb"] == 32.4
    assert data.get("error_type") != "WorkerCrashed"


# ---------------------------------------------------------------------------
# --run-benchmark-gated tiny smoke: real subprocesses, real Metal kernel + autodiff,
# N=256 -- never a flagship dispatch (binding constraint: T13 owns real runs).
# ---------------------------------------------------------------------------


class _FakeWatchdogHandle:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_run_single_condition_installs_and_stops_the_memory_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attention bench's per-condition child installs the SAME active-memory
    watchdog the loss/train worker does (this script measured the 32.4 GB paging
    condition), with the EFFECTIVE ceiling `effective_memory_ceiling` computes, and stops
    it on completion. `measure_condition` is stubbed so no real GPU allocation happens."""
    condition = _condition()
    captured: dict[str, object] = {}
    handle = _FakeWatchdogHandle()

    def _fake_install(
        *, ceiling_bytes: int, wall_budget_s: float | None, on_breach: object,  # noqa: ARG001
    ) -> _FakeWatchdogHandle:
        captured["ceiling_bytes"] = ceiling_bytes
        captured["wall_budget_s"] = wall_budget_s
        return handle

    def _fake_measure(c: AttnCondition) -> dict[str, object]:
        return {
            "impl": c.impl, "n": c.n, "fwd_peak_gb": 0.0, "fwdbwd_peak_gb": 0.0,
            "wall_s": 0.001, "walls_s": [0.001],
        }

    monkeypatch.setattr(
        bench_attention_op, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    monkeypatch.setattr(bench_attention_op, "install_memory_watchdog", _fake_install)
    monkeypatch.setattr(bench_attention_op, "measure_condition", _fake_measure)

    out_path = bench_attention_op._run_single_condition(
        condition, out_dir=tmp_path, session_id="s1",
    )
    assert captured["ceiling_bytes"] == 987654321  # the effective ceiling flowed through
    assert captured["wall_budget_s"] == DEFAULT_WALL_BUDGET_S
    assert handle.stopped is True
    assert json.loads(out_path.read_text())["status"] == "ok"


def test_run_single_condition_records_memory_warning_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded-start warning is carried into the attention child's `ok` artifact as
    `memory_warning`; a nominal start (None) omits the key entirely."""
    def _fake_measure(c: AttnCondition) -> dict[str, object]:
        return {
            "impl": c.impl, "n": c.n, "fwd_peak_gb": 0.0, "fwdbwd_peak_gb": 0.0,
            "wall_s": 0.001, "walls_s": [0.001],
        }

    fake_handle = _FakeWatchdogHandle()
    monkeypatch.setattr(bench_attention_op, "measure_condition", _fake_measure)
    monkeypatch.setattr(
        bench_attention_op, "install_memory_watchdog",
        lambda *, ceiling_bytes, wall_budget_s, on_breach: fake_handle,  # noqa: ARG005
    )

    monkeypatch.setattr(
        bench_attention_op, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning="only 8 GiB free"),
    )
    warned = bench_attention_op._run_single_condition(
        _condition(), out_dir=tmp_path, session_id="s1",
    )
    assert json.loads(warned.read_text())["memory_warning"] == "only 8 GiB free"

    monkeypatch.setattr(
        bench_attention_op, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=987654321, warning=None),
    )
    clean = bench_attention_op._run_single_condition(
        _condition(), out_dir=tmp_path, session_id="s2",
    )
    assert "memory_warning" not in json.loads(clean.read_text())


@pytest.mark.benchmark
def test_bench_attention_op_tiny_smoke_writes_ok_artifacts_for_both_impls(
    tmp_path: Path,
) -> None:
    rc = main(["--seq-lens", "256", "--out-dir", str(tmp_path)])
    assert rc == 0

    flash = json.loads((tmp_path / "flash_n256.json").read_text())
    stock = json.loads((tmp_path / "stock_n256.json").read_text())
    for data in (flash, stock):
        assert data["status"] == "ok"
        assert data["n"] == 256
        assert data["fwd_peak_gb"] >= 0
        assert data["fwdbwd_peak_gb"] >= 0
        assert data["wall_s"] > 0
        # wall_s is the median of WALL_REPS per-rep walls (T11 review fix: backward
        # JIT warmed outside the window, wall no longer single-shot)
        assert len(data["walls_s"]) == bench_attention_op.WALL_REPS
        assert data["wall_s"] == statistics.median(data["walls_s"])
        # The forward window is rep-timed the same way, and the artifact records the
        # fwd/bwd decomposition itself (RC review finding 2: the split claim must be
        # reproducible from this one committed script's output).
        assert data["fwd_wall_s"] > 0
        assert len(data["fwd_walls_s"]) == bench_attention_op.WALL_REPS
        assert data["fwd_wall_s"] == statistics.median(data["fwd_walls_s"])
        # Identity up to the per-field round(..., 6) applied when the artifact is built.
        assert data["bwd_wall_s"] == pytest.approx(
            data["wall_s"] - data["fwd_wall_s"], abs=2e-6,
        )
        assert data["bwd_over_fwd"] == pytest.approx(
            data["bwd_wall_s"] / data["fwd_wall_s"], rel=1e-2,
        )
    assert flash["impl"] == "flash"
    assert stock["impl"] == "stock"
