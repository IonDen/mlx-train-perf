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


def test_condition_name_embeds_slug_seq_len_and_arm() -> None:
    name = condition_name(model="mlx-community/Qwen3-8B-4bit", seq_len=2048, arm="ours")
    assert name == "train_step_mlx-community__Qwen3-8B-4bit_seq2048_ours"


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
                assert condition_name(model=model, seq_len=seq_len, arm=arm) in names


def test_build_conditions_ours_and_stock_share_every_param_except_stock_and_impl() -> None:
    conditions = build_conditions(
        models=["m/a"], seq_lens=[1024], batch=2, steps=20, lora_rank=8, lora_layers=-1,
        learning_rate=1e-5, seed=7, revision="main",
    )
    by_name = {c.name: c for c in conditions}
    ours = by_name[condition_name(model="m/a", seq_len=1024, arm="ours")]
    stock = by_name[condition_name(model="m/a", seq_len=1024, arm="stock")]
    assert ours.kind == "train_step"
    assert stock.kind == "train_step"
    assert ours.params["stock"] is False
    assert stock.params["stock"] is True
    for key in ("model", "revision", "seq_len", "batch", "steps", "lora_rank",
                "lora_layers", "learning_rate", "seed", "script_sha"):
        assert ours.params[key] == stock.params[key]


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


def test_compare_ok_pair_reports_tps_ratio_and_loss_curve_diff(tmp_path: Path) -> None:
    ours_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='ours')}.json"
    stock_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='stock')}.json"
    _write_result(ours_path, status="ok", tokens_per_sec_median=200.0,
                  loss_all=[3.0, 2.9, 2.8])
    _write_result(stock_path, status="ok", tokens_per_sec_median=100.0,
                  loss_all=[3.0, 2.91, 2.79])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024,
    )
    assert entry["status"] == "ok"
    assert entry["ours_tps_over_stock_tps"] == 2.0   # ours is 2x faster
    assert entry["loss_curve_worst_diff"] == pytest.approx(0.01, abs=1e-9)


def test_compare_reports_missing_when_a_condition_is_absent() -> None:
    entry = compare_ours_vs_stock({}, model="m/a", seq_len=1024)
    assert entry["status"] == "missing"


def test_compare_reports_corrupt_artifact(tmp_path: Path) -> None:
    ours_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='ours')}.json"
    stock_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='stock')}.json"
    ours_path.write_text("{not json")
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024,
    )
    assert entry["status"] == "corrupt"


def test_compare_reports_incomplete_when_stock_oomed(tmp_path: Path) -> None:
    """The exact flagship scenario: stock legitimately crashes (recorded as the
    standard `runner.run_conditions` crash envelope) while ours succeeds -- this is
    the EXPECTED, demonstrating result, not a bug in this reporting function."""
    ours_path = tmp_path / f"{condition_name(model='m/a', seq_len=8192, arm='ours')}.json"
    stock_path = tmp_path / f"{condition_name(model='m/a', seq_len=8192, arm='stock')}.json"
    _write_result(ours_path, status="ok", tokens_per_sec_median=50.0, loss_all=[1.0])
    _write_result(stock_path, status="error", error_type="WorkerCrashed",
                  error_msg="RuntimeError: [malloc] Attempting to allocate ...")
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=8192,
    )
    assert entry["status"] == "incomplete"
    assert entry["ours_status"] == "ok"
    assert entry["stock_status"] == "error"


def test_compare_refuses_a_cross_session_comparison(tmp_path: Path) -> None:
    ours_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='ours')}.json"
    stock_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='stock')}.json"
    _write_result(ours_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0],
                  session_id="old")
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0],
                  session_id="new")
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024,
    )
    assert entry["status"] == "cross_session"


def test_compare_skips_loss_curve_diff_when_lengths_differ(tmp_path: Path) -> None:
    ours_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='ours')}.json"
    stock_path = tmp_path / f"{condition_name(model='m/a', seq_len=1024, arm='stock')}.json"
    _write_result(ours_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0, 2.0])
    _write_result(stock_path, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    entry = compare_ours_vs_stock(
        {ours_path.stem: ours_path, stock_path.stem: stock_path},
        model="m/a", seq_len=1024,
    )
    assert entry["status"] == "ok"
    assert "loss_curve_worst_diff" not in entry


def test_build_report_covers_every_model_seq_len_pair(tmp_path: Path) -> None:
    for model in ("m/a", "m/b"):
        for arm in ("ours", "stock"):
            p = tmp_path / f"{condition_name(model=model, seq_len=1024, arm=arm)}.json"
            _write_result(p, status="ok", tokens_per_sec_median=1.0, loss_all=[1.0])
    paths = list(tmp_path.glob("*.json"))
    rows = build_report(paths, models=["m/a", "m/b"], seq_lens=[1024])
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
    against the pre-downloaded `mlx-community/Qwen2.5-0.5B-Instruct-bf16` (2 steps,
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
