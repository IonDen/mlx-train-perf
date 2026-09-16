"""Tests for `scripts/bench_qwen35_convergence.py`. `scripts/` has no `__init__.py`
(matches the existing convention), so the module is loaded by path rather than a
package import.

Only the GPU-free, model-free surface is covered: the batch-order seeding guard
(`_seed_batch_order`), `condition_name`/`build_condition`, `split_held_out`,
`build_comparison` (reads plain JSON artifacts from disk, no MLX, no mlx-lm), and the
CLI shell (`--help`, required-argument validation). The real `run_convergence`
measurement -- which always loads a model and drives real `mlx_lm.tuner.trainer.train()`
iterations -- is out of scope for this test file, same convention
`tests/test_bench_train_step.py` uses for its own heavy measurement function.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_qwen35_convergence  # noqa: E402 -- import must follow the sys.path insert
from bench_qwen35_convergence import (  # noqa: E402
    build_comparison,
    build_condition,
    condition_name,
    dataset_sha,
    model_slug,
    script_sha,
    split_held_out,
)

_SCRIPT_PATH = _SCRIPTS_DIR / "bench_qwen35_convergence.py"


# ---------------------------------------------------------------------------
# _seed_batch_order: the batch-order determinism regression this file exists to pin.
# ---------------------------------------------------------------------------


def test_seed_batch_order_overrides_ambient_numpy_state_at_seed_zero() -> None:
    # Catches: relying on mlx-lm's own `if seed: np.random.seed(seed)` gate inside
    # `iterate_batches` instead of seeding numpy explicitly. That gate is a NO-OP when
    # seed == 0 (this script's own --seed default) -- without `_seed_batch_order`'s own
    # explicit `np.random.seed` call, two runs both passed `seed=0` but started from
    # DIFFERENT ambient numpy global-RNG state (the normal case across two independent
    # subprocess launches) would silently draw different batch-order permutations,
    # breaking the "both arms see the same example order" premise the whole convergence
    # comparison rests on.
    np.random.seed(111)  # simulate one process's ambient global-RNG state
    bench_qwen35_convergence._seed_batch_order(0)
    first = np.random.permutation(10).tolist()

    np.random.seed(222222)  # simulate a DIFFERENT process's ambient state
    bench_qwen35_convergence._seed_batch_order(0)
    second = np.random.permutation(10).tolist()

    assert first == second


def test_seed_batch_order_is_deterministic_for_a_nonzero_seed_too() -> None:
    np.random.seed(1)
    bench_qwen35_convergence._seed_batch_order(7)
    first = np.random.permutation(10).tolist()

    np.random.seed(999999)
    bench_qwen35_convergence._seed_batch_order(7)
    second = np.random.permutation(10).tolist()

    assert first == second


def test_seed_batch_order_different_seeds_diverge() -> None:
    # A sanity control for the two tests above: _seed_batch_order must actually change
    # the draw, not just make every seed collapse to the same fixed permutation.
    bench_qwen35_convergence._seed_batch_order(0)
    zero_draw = np.random.permutation(10).tolist()
    bench_qwen35_convergence._seed_batch_order(1)
    one_draw = np.random.permutation(10).tolist()
    assert zero_draw != one_draw


# ---------------------------------------------------------------------------
# script_sha / model_slug / dataset_sha / condition_name: pure string/hash helpers
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    assert a == script_sha()
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_model_slug_is_filesystem_safe() -> None:
    assert model_slug("mlx-community/Qwen3.5-0.8B-4bit") == "mlx-community__Qwen3.5-0.8B-4bit"
    assert "/" not in model_slug("a/b:c d")


def test_dataset_sha_is_a_stable_content_digest(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('{"tokens":[1,2,3],"offset":1}\n')
    a = dataset_sha(p)
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)
    assert dataset_sha(p) == a
    p.write_text('{"tokens":[1,2,4],"offset":1}\n')
    assert dataset_sha(p) != a


def test_condition_name_embeds_slug_arm_seq_and_batch() -> None:
    name = condition_name(model="mlx-community/Qwen3.5-0.8B-4bit", arm="enabled",
                          max_seq_length=512, batch_size=1)
    assert name == "qwen35_convergence_mlx-community__Qwen3.5-0.8B-4bit_enabled_seq512_B1"


def test_condition_names_differ_across_arms() -> None:
    """Two arms against the same --out dir must never collide on FILENAME -- the arm is
    part of the name, not only the artifact identity."""
    stock = condition_name(model="m/a", arm="stock", max_seq_length=512, batch_size=1)
    enabled = condition_name(model="m/a", arm="enabled", max_seq_length=512, batch_size=1)
    assert stock != enabled
    assert "_stock_" in stock
    assert "_enabled_" in enabled


# ---------------------------------------------------------------------------
# build_condition: pure Condition construction
# ---------------------------------------------------------------------------


def _build_one(**overrides: object) -> bench_qwen35_convergence.ConvergenceCondition:
    kwargs: dict[str, object] = {
        "model": "m/a", "revision": None, "data": "/prepped.jsonl",
        "dataset_sha": "cafef00d", "arm": "enabled", "max_seq_length": 512,
        "batch_size": 1, "steps": 200, "val_every": 20, "val_examples": 16,
        "lora_rank": 8, "lora_layers": -1, "learning_rate": 1e-5, "seed": 0,
        "grad_checkpoint": False,
    }
    kwargs.update(overrides)
    return build_condition(**kwargs)  # type: ignore[arg-type]


def test_build_condition_name_carries_the_arm() -> None:
    assert "_stock_" in _build_one(arm="stock").name
    assert "_enabled_" in _build_one(arm="enabled").name


def test_build_condition_defaults_seed_to_zero() -> None:
    assert _build_one().seed == 0


# ---------------------------------------------------------------------------
# split_held_out: deterministic prefix split
# ---------------------------------------------------------------------------


def test_split_held_out_takes_a_deterministic_prefix() -> None:
    pairs = [([i], 0) for i in range(10)]
    train_pairs, val_pairs = split_held_out(pairs, val_examples=3)
    assert val_pairs == pairs[:3]
    assert train_pairs == pairs[3:]
    assert len(train_pairs) + len(val_pairs) == len(pairs)


def test_split_held_out_is_the_same_for_repeated_calls() -> None:
    # "held out" only means the same physical examples across both arms if the split
    # itself carries no hidden randomness.
    pairs = [([i], 0) for i in range(20)]
    a = split_held_out(pairs, val_examples=5)
    b = split_held_out(pairs, val_examples=5)
    assert a == b


# ---------------------------------------------------------------------------
# build_comparison: reads plain JSON artifacts, no MLX
# ---------------------------------------------------------------------------


def _write(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields))


def test_build_comparison_missing_files_is_corrupt(tmp_path: Path) -> None:
    result = build_comparison(tmp_path / "a.json", tmp_path / "b.json")
    assert result["status"] == "corrupt"
    assert "note" not in result  # the field is named validation_implementation_note
    assert "validation_implementation_note" in result


def test_build_comparison_incomplete_when_one_arm_did_not_finish(tmp_path: Path) -> None:
    stock = tmp_path / "stock.json"
    enabled = tmp_path / "enabled.json"
    _write(stock, status="ok", identity={"arm": "stock", "seed": 0})
    _write(enabled, status="refused_environment", identity={"arm": "enabled", "seed": 0})
    result = build_comparison(stock, enabled)
    assert result["status"] == "incomplete"
    assert result["stock_status"] == "ok"
    assert result["enabled_status"] == "refused_environment"


def test_build_comparison_refuses_on_identity_mismatch(tmp_path: Path) -> None:
    # The exact regression class this file's seeding tests exist to prevent from
    # slipping past unnoticed: two runs whose identity disagrees on `seed` (or any
    # other measurement dimension) must never be folded into one comparison.
    stock = tmp_path / "stock.json"
    enabled = tmp_path / "enabled.json"
    _write(stock, status="ok", identity={"arm": "stock", "session_id": "s1", "seed": 0},
          train_info=[], val_info=[])
    _write(enabled, status="ok",
          identity={"arm": "enabled", "session_id": "s2", "seed": 1},
          train_info=[], val_info=[])
    result = build_comparison(stock, enabled)
    assert result["status"] == "identity_mismatch"
    assert result["differing_identity_fields"] == ["seed"]


def test_build_comparison_ok_pair_reports_train_and_val_series_plus_disclosure(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock.json"
    enabled = tmp_path / "enabled.json"
    ident = {"arm": None, "session_id": None, "seed": 0, "model": "m"}
    _write(
        stock, status="ok", identity={**ident, "arm": "stock", "session_id": "s1"},
        train_info=[{"iteration": 1, "train_loss": 2.0}],
        val_info=[{"iteration": 0, "val_loss": 2.5}],
    )
    _write(
        enabled, status="ok", identity={**ident, "arm": "enabled", "session_id": "s2"},
        train_info=[{"iteration": 1, "train_loss": 1.9}],
        val_info=[{"iteration": 0, "val_loss": 2.4}],
    )
    result = build_comparison(stock, enabled)
    assert result["status"] == "ok"
    assert result["train_loss_by_iteration"] == {
        "stock": {1: 2.0}, "enabled": {1: 1.9},
    }
    assert result["val_loss_by_iteration"] == {
        "stock": {0: 2.5}, "enabled": {0: 2.4},
    }
    note = result["validation_implementation_note"]
    assert isinstance(note, str)
    assert "chunked" in note
    assert "inference kernel" in note


# ---------------------------------------------------------------------------
# CLI shell: --help, required-argument validation (no model, no MLX)
# ---------------------------------------------------------------------------


def test_help_runs_without_touching_a_model() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--arm" in proc.stdout
    assert "--data" in proc.stdout
    assert "--compare" in proc.stdout
    # The fix under test must be visible in --help, not just in a code comment.
    assert "_seed_batch_order" in proc.stdout


def test_main_requires_model_data_and_arm_unless_compare() -> None:
    with pytest.raises(SystemExit):
        bench_qwen35_convergence.main([])


def test_main_compare_requires_both_artifact_paths() -> None:
    with pytest.raises(SystemExit):
        bench_qwen35_convergence.main(["--compare"])


def test_default_results_directory_is_under_artifacts() -> None:
    assert bench_qwen35_convergence.RESULTS.name == "bench_qwen35_convergence"
    assert "_artifacts" in bench_qwen35_convergence.RESULTS.parts
