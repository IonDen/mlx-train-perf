"""Tests for `scripts/bench_train_step.py`. `scripts/` has no `__init__.py` (matches
the existing convention), so the module is loaded by path rather than a package import.

Only the GPU-free, model-free helpers are covered directly here: `script_sha`,
`model_slug`, `condition_name`, `build_conditions` (pure condition-list construction),
and `compare_ours_vs_stock`/`build_report` (read plain JSON artifacts from disk, no
MLX, no mlx-lm). The real `train_step` condition measurement -- which always loads a
model, even the tiny `--smoke` default -- is exercised only through the gated
`@pytest.mark.smoke` test at the bottom, collected and confirmed skipped by default,
never executed here (same convention `tests/test_adapter.py`'s own smoke test uses).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_train_perf.bench import runner as bench_runner
from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import run_conditions

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_train_step  # noqa: E402 -- import must follow the sys.path insert
from bench_train_step import (  # noqa: E402
    RESULTS,
    RESULTS_SMOKE,
    SMOKE_MODEL,
    SMOKE_SEQ_LEN,
    SMOKE_STEPS,
    build_conditions,
    build_report,
    compare_ours_vs_stock,
    condition_name,
    model_slug,
    script_sha,
)

_SCRIPT_PATH = _SCRIPTS_DIR / "bench_train_step.py"


# ---------------------------------------------------------------------------
# script_sha / model_slug / condition_name: pure string/hash helpers
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_model_slug_is_filesystem_safe() -> None:
    slug = model_slug("mlx-community/Qwen3-8B-4bit")
    assert "/" not in slug
    assert slug == "mlx-community__Qwen3-8B-4bit"


def test_condition_name_embeds_slug_seq_len_attention_and_arm() -> None:
    name = condition_name(
        model="mlx-community/Qwen3-8B-4bit", seq_len=2048, arm="ours",
        attention_impl="stock",
    )
    assert name == "train_step_mlx-community__Qwen3-8B-4bit_seq2048_attn-stock_ours"


def test_condition_names_differ_across_attention_arms() -> None:
    """Gotcha 18 / 0022c: two invocations differing ONLY by --attention against the same
    --out dir must never share artifact FILENAMES (the identity already differed; the
    filename did not, and the T13 flash run silently overwrote the stock artifacts)."""
    kwargs: dict = {
        "models": ["m/a"], "seq_lens": [1024], "batch": 1, "steps": 20, "lora_rank": 8,
        "lora_layers": -1, "learning_rate": 1e-5, "seed": 0, "revision": None,
    }
    stock_names = {c.name for c in build_conditions(attention_impl="stock", **kwargs)}
    flash_names = {c.name for c in build_conditions(attention_impl="flash", **kwargs)}
    assert stock_names
    assert flash_names
    assert stock_names.isdisjoint(flash_names)


# ---------------------------------------------------------------------------
# build_conditions: pure condition-list construction over the (model, seq_len) x
# {ours, stock} cross product
# ---------------------------------------------------------------------------


def test_build_conditions_covers_the_full_model_seq_len_arm_cross_product() -> None:
    conditions = build_conditions(
        models=["m/a", "m/b"], seq_lens=[1024, 2048], batch=1, steps=20, lora_rank=8,
        lora_layers=-1, learning_rate=1e-5, seed=0, revision=None,
    )
    assert len(conditions) == 2 * 2 * 2   # 2 models x 2 seq_lens x {ours, stock}
    names = {c.name for c in conditions}
    for model in ("m/a", "m/b"):
        for seq_len in (1024, 2048):
            for arm in ("ours", "stock"):
                assert condition_name(
                    model=model, seq_len=seq_len, arm=arm, attention_impl="stock",
                ) in names


def test_build_conditions_ours_and_stock_share_every_param_except_stock_and_impl() -> None:
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=2, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=7, revision="main",
    )
    by_name = {c.name: c for c in conditions}
    ours = by_name[condition_name(model="m/a", seq_len=1024, arm="ours", attention_impl="stock")]
    stock = by_name[condition_name(model="m/a", seq_len=1024, arm="stock", attention_impl="stock")]
    assert ours.kind == "train_step"
    assert stock.kind == "train_step"
    assert ours.params["stock"] is False
    assert stock.params["stock"] is True
    for key in ("model", "revision", "seq_len", "batch", "steps", "lora_rank",
                "lora_layers", "learning_rate", "seed", "script_sha"):
        assert ours.params[key] == stock.params[key]
    # attention is held constant across both arms -- now a dedicated `Condition` field
    # (out of `params`), not a params entry.
    assert ours.attention_impl == stock.attention_impl


def test_build_conditions_threads_compute_dtype_into_both_arms() -> None:
    """`--compute-dtype` reaches EVERY condition's params uniformly (both `ours` and
    `stock`), so the kernel arm gets a bf16-cast model AND the comparison holds the
    trunk dtype constant across the two arms -- isolating the loss layer."""
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None, compute_dtype="bfloat16",
    )
    assert conditions
    assert all(c.params["compute_dtype"] == "bfloat16" for c in conditions)


def test_build_conditions_compute_dtype_defaults_to_none() -> None:
    """Absent `--compute-dtype`, the key is present and None -- no cast (the
    smoke/chunked path, and any explicit-chunked/naive run, keeps the loaded dtype)."""
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None,
    )
    assert all(c.params["compute_dtype"] is None for c in conditions)


def test_build_conditions_threads_grad_checkpoint_into_both_arms() -> None:
    """`--grad-checkpoint` reaches every condition's params uniformly -- the realistic
    long-context QLoRA setup (activations recomputed), where ours' flat loss-layer
    memory becomes visible against a small trunk footprint."""
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None, grad_checkpoint=True,
    )
    assert conditions
    assert all(c.params["grad_checkpoint"] is True for c in conditions)


def test_build_conditions_grad_checkpoint_defaults_to_false() -> None:
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None,
    )
    assert all(c.params["grad_checkpoint"] is False for c in conditions)


def test_build_conditions_threads_attention_impl_into_both_arms() -> None:
    """`--attention` reaches EVERY condition's dedicated `attention_impl` field uniformly
    (both `ours` and `stock`) -- this bench's `ours`/`stock` arms compare the loss layer at
    a held-constant attention implementation, unlike the North-Star sweep, where attention
    IS the per-arm dimension (bench/worker.py's `attention_impl` knob)."""
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None, attention_impl="flash",
    )
    assert conditions
    assert all(c.attention_impl == "flash" for c in conditions)


def test_build_conditions_attention_impl_defaults_to_stock() -> None:
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None,
    )
    assert all(c.attention_impl == "stock" for c in conditions)


def _crash_worker(_config_path: Path) -> subprocess.CompletedProcess[str]:
    """A `_spawn_worker` stand-in that "crashes" -- so `run_conditions` takes its
    error-envelope branch and writes the identity it built, never loading a real model."""
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="stub crash")


def test_build_conditions_flow_reaches_run_conditions_without_reserved_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (T13 step 1): the exact production path that crashed. `build_conditions`'
    `attention_impl` must travel to `condition_identity` through `run_conditions`' DEDICATED
    identity slot, NOT through `params` (where the reserved-key guard raises
    `BenchInputError` before any worker spawns). Worker stubbed at the subprocess boundary;
    the error-envelope path writes the identity `run_conditions` computed, so reading it
    back proves the field reached the identity."""
    monkeypatch.setattr(bench_runner, "_spawn_worker", _crash_worker)
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=2, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None, attention_impl="flash",
    )
    paths = run_conditions(conditions, tmp_path, session_id=new_session_id())
    assert paths
    for p in paths:
        assert json.loads(p.read_text())["identity"]["attention_impl"] == "flash"


def test_build_conditions_stock_and_flash_invocations_get_different_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two invocations of the bench differing ONLY in `--attention` (stock vs flash) must
    produce different artifact identities -- otherwise a flash run would resume-skip a
    stock artifact (same script_sha, same params) and silently report the wrong numbers."""
    monkeypatch.setattr(bench_runner, "_spawn_worker", _crash_worker)
    session_id = new_session_id()
    stock_paths = run_conditions(
        build_conditions(
            models=["m/a"], seq_lens=[1024], batch=1, steps=2, lora_rank=8, lora_layers=-1,
            learning_rate=1e-5, seed=0, revision=None, attention_impl="stock",
        ),
        tmp_path / "stock", session_id=session_id,
    )
    flash_paths = run_conditions(
        build_conditions(
            models=["m/a"], seq_lens=[1024], batch=1, steps=2, lora_rank=8, lora_layers=-1,
            learning_rate=1e-5, seed=0, revision=None, attention_impl="flash",
        ),
        tmp_path / "flash", session_id=session_id,
    )
    id_stock = json.loads(stock_paths[0].read_text())["identity"]
    id_flash = json.loads(flash_paths[0].read_text())["identity"]
    assert id_stock["attention_impl"] == "stock"
    assert id_flash["attention_impl"] == "flash"
    assert id_stock != id_flash


def test_build_conditions_all_share_the_same_script_sha() -> None:
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=1, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=0, revision=None,
    )
    assert all(c.params["script_sha"] == script_sha() for c in conditions)


# ---------------------------------------------------------------------------
# compare_ours_vs_stock / build_report: read plain JSON artifacts, no MLX
# ---------------------------------------------------------------------------


def _write_result(path: Path, *, status: str, session_id: str = "s1", **fields: object) -> None:
    path.write_text(json.dumps({
        "status": status, "identity": {"session_id": session_id}, **fields,
    }))


def _pair_paths(tmp_path: Path, *, seq_len: int = 1024) -> tuple[Path, Path]:
    ours = condition_name(
        model="m/a", seq_len=seq_len, arm="ours", attention_impl="stock",
    )
    stock = condition_name(
        model="m/a", seq_len=seq_len, arm="stock", attention_impl="stock",
    )
    return tmp_path / f"{ours}.json", tmp_path / f"{stock}.json"


def test_compare_ok_pair_reports_tps_ratio_and_loss_curve_diff(tmp_path: Path) -> None:
    ours_path, stock_path = _pair_paths(tmp_path)
    _write_result(ours_path, status="ok", tokens_per_sec_median=200.0,
                  loss_all=[3.0, 2.9, 2.8])
    _write_result(stock_path, status="ok", tokens_per_sec_median=100.0,
                  loss_all=[3.0, 2.91, 2.79])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024, attention_impl="stock",
    )
    assert entry["status"] == "ok"
    assert entry["ours_tps_over_stock_tps"] == 2.0   # ours is 2x faster
    assert entry["loss_curve_worst_diff"] == pytest.approx(0.01, abs=1e-9)


def test_compare_reports_missing_when_a_condition_is_absent() -> None:
    entry = compare_ours_vs_stock({}, model="m/a", seq_len=1024, attention_impl="stock")
    assert entry["status"] == "missing"


def test_compare_reports_corrupt_artifact(tmp_path: Path) -> None:
    ours_path, stock_path = _pair_paths(tmp_path)
    ours_path.write_text("{not json")
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024, attention_impl="stock",
    )
    assert entry["status"] == "corrupt"


def test_compare_reports_incomplete_when_stock_oomed(tmp_path: Path) -> None:
    """The exact flagship scenario: stock legitimately crashes (recorded as the
    standard `runner.run_conditions` crash envelope) while ours succeeds -- this is
    the EXPECTED, demonstrating result, not a bug in this reporting function."""
    ours_path, stock_path = _pair_paths(tmp_path, seq_len=8192)
    _write_result(ours_path, status="ok", tokens_per_sec_median=50.0, loss_all=[1.0])
    _write_result(stock_path, status="error", error_type="WorkerCrashed",
                  error_msg="RuntimeError: [malloc] Attempting to allocate ...")
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=8192, attention_impl="stock",
    )
    assert entry["status"] == "incomplete"
    assert entry["ours_status"] == "ok"
    assert entry["stock_status"] == "error"


def test_compare_refuses_a_cross_session_comparison(tmp_path: Path) -> None:
    ours_path, stock_path = _pair_paths(tmp_path)
    _write_result(ours_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0],
                  session_id="old")
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0],
                  session_id="new")
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024, attention_impl="stock",
    )
    assert entry["status"] == "cross_session"


def test_compare_skips_loss_curve_diff_when_lengths_differ(tmp_path: Path) -> None:
    ours_path, stock_path = _pair_paths(tmp_path)
    _write_result(ours_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0, 2.0])
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024, attention_impl="stock",
    )
    assert entry["status"] == "ok"
    assert "loss_curve_worst_diff" not in entry


def test_build_report_covers_every_model_seq_len_pair(tmp_path: Path) -> None:
    for model in ("m/a", "m/b"):
        for arm in ("ours", "stock"):
            p = tmp_path / (
                f"{condition_name(model=model, seq_len=1024, arm=arm, attention_impl='stock')}"
                ".json"
            )
            _write_result(p, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    paths = list(tmp_path.glob("*.json"))
    rows = build_report(paths, models=["m/a", "m/b"], seq_lens=[1024], attention_impl="stock")
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"m/a", "m/b"}
    assert all(row["status"] == "ok" for row in rows)


# ---------------------------------------------------------------------------
# CLI shell: --help (no model, no MLX)
# ---------------------------------------------------------------------------


def test_help_runs_without_touching_a_model() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--smoke" in proc.stdout
    assert "--model" in proc.stdout
    assert "--attention" in proc.stdout


def test_main_requires_model_and_seq_len_unless_smoke() -> None:
    with pytest.raises(SystemExit):
        bench_train_step.main([])


# ---------------------------------------------------------------------------
# Gated smoke test: a real (small, pre-downloaded) model, end to end via the CLI.
# Collected and confirmed SKIPPED by default -- never executed with --run-smoke here.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bench_train_step_smoke_end_to_end(tmp_path: Path) -> None:
    """`--run-smoke`: invokes `bench_train_step.py --smoke` as a real subprocess
    against the pre-downloaded `mlx-community/Llama-3.2-1B-Instruct-4bit` (2 steps,
    ours vs stock) and asserts both arms report `status == "ok"` with a finite
    tokens/sec median."""
    out_dir = tmp_path / "bench_train_step_smoke"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--smoke", "--out", str(out_dir)],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert len(rows) == 1
    assert rows[0]["model"] == SMOKE_MODEL
    assert rows[0]["seq_len"] == SMOKE_SEQ_LEN
    assert rows[0]["status"] == "ok"
    assert rows[0]["ours_tokens_per_sec_median"] > 0
    assert rows[0]["stock_tokens_per_sec_median"] > 0
    ours_name = condition_name(model=SMOKE_MODEL, seq_len=SMOKE_SEQ_LEN, arm="ours")
    ours_artifact = json.loads((out_dir / f"{ours_name}.json").read_text())
    assert len(ours_artifact["loss_all"]) == SMOKE_STEPS


def test_default_results_directories_are_distinct() -> None:
    assert RESULTS != RESULTS_SMOKE
    assert RESULTS_SMOKE.is_relative_to(RESULTS)
