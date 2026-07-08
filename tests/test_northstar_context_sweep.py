"""Tests for `scripts/northstar_context_sweep.py`. `scripts/` has no `__init__.py`
(matches the existing convention), so the module is loaded by path.

This script is BUILD-verified only -- per this project's heavy-run discipline, the
real sweep (a 1-2 hour binary search against the real flagship 8B model) is run only
by the controller, with an explicit go-ahead, never from this test suite or an agent
session. Every test here is pure: `find_max_context` is driven entirely by a FAKE
oracle function (no MLX, no subprocess, no model), and `_recipe_session_id`/
`build_probe`/`probe_condition_name`/`script_sha`/`model_slug` are plain string/hash
logic. `sweep_arm`/`main` (which call `run_conditions` and would spawn a real
subprocess trying to load the flagship model) are exercised only via `--help`.
"""
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from northstar_context_sweep import (  # noqa: E402 -- follows the sys.path insert
    DEFAULT_GRANULARITY,
    DEFAULT_START_SEQ_LEN,
    FLAGSHIP_MODEL,
    MAX_CONTEXT_CEILING,
    _recipe_session_id,
    build_parser,
    build_probe,
    find_max_context,
    model_slug,
    probe_condition_name,
    script_sha,
)

_SCRIPT_PATH = _SCRIPTS_DIR / "northstar_context_sweep.py"

_RECIPE = {
    "model": "mlx-community/Qwen3-8B-4bit", "revision": None, "batch": 1,
    "lora_rank": 8, "lora_layers": -1, "seed": 0,
}


# ---------------------------------------------------------------------------
# find_max_context: pure binary/doubling search over a fake "does it fit" oracle
# ---------------------------------------------------------------------------


def _fits_up_to(limit: int) -> Callable[[int], bool]:
    def oracle(seq_len: int) -> bool:
        return seq_len <= limit
    return oracle


def test_find_max_context_converges_within_granularity() -> None:
    probe = _fits_up_to(5000)
    max_fitting, min_failing = find_max_context(probe, start=1024, granularity=256)
    assert max_fitting <= 5000 < (min_failing or 0)
    assert min_failing is not None
    assert min_failing - max_fitting <= 256


def test_find_max_context_doubles_before_bisecting() -> None:
    calls: list[int] = []
    probe = _fits_up_to(5000)

    def recording_probe(seq_len: int) -> bool:
        calls.append(seq_len)
        return probe(seq_len)

    find_max_context(recording_probe, start=1024, granularity=256)
    # Doubling phase: 1024 (fits) -> 2048 (fits) -> 4096 (fits) -> 8192 (fails) --
    # these four calls MUST appear, in this order, before any bisection call.
    assert calls[:4] == [1024, 2048, 4096, 8192]


def test_find_max_context_returns_zero_and_start_when_start_itself_does_not_fit() -> None:
    probe = _fits_up_to(512)
    max_fitting, min_failing = find_max_context(probe, start=1024, granularity=256)
    assert max_fitting == 0
    assert min_failing == 1024


def test_find_max_context_never_probes_below_start() -> None:
    calls: list[int] = []

    def probe(seq_len: int) -> bool:
        calls.append(seq_len)
        return seq_len <= 100_000   # fits forever within any reasonable ceiling

    find_max_context(probe, start=1024, granularity=256, ceiling=8192)
    assert min(calls) == 1024


def test_find_max_context_reports_no_min_failing_when_ceiling_reached_without_a_failure() -> None:
    """Every doubling step fits, all the way to the ceiling -- the search stops
    (bounded work), reporting `min_failing_seq_len=None` so the caller knows this
    converged on an artificial stop, not the model's real limit."""
    probe = _fits_up_to(10**9)   # never fails within any ceiling this test uses
    max_fitting, min_failing = find_max_context(probe, start=1024, granularity=256,
                                                ceiling=8192)
    assert min_failing is None
    assert max_fitting == 8192


def test_find_max_context_boundary_probes_are_granularity_aligned() -> None:
    calls: list[int] = []
    probe = _fits_up_to(5000)

    def recording_probe(seq_len: int) -> bool:
        calls.append(seq_len)
        return probe(seq_len)

    find_max_context(recording_probe, start=1024, granularity=256)
    for seq_len in calls:
        assert seq_len % 256 == 0 or seq_len == 1024   # start itself is exempt


def test_find_max_context_exact_boundary_at_start_value() -> None:
    """`start` itself is the max fitting value (the very next doubling step, 2048,
    fails) -- the search must still converge (bisecting between 1024 and 2048 down to
    granularity, not simply reporting the first failing doubling step), not loop
    forever, and never probe below start."""
    probe = _fits_up_to(1024)
    max_fitting, min_failing = find_max_context(probe, start=1024, granularity=256)
    assert max_fitting == 1024
    assert min_failing is not None
    assert min_failing - max_fitting <= 256
    assert min_failing > 1024


# ---------------------------------------------------------------------------
# _recipe_session_id: deterministic (resume-safe), sensitive to the recipe
# ---------------------------------------------------------------------------


def test_recipe_session_id_is_deterministic_for_the_same_recipe() -> None:
    a = _recipe_session_id(**_RECIPE)
    b = _recipe_session_id(**_RECIPE)
    assert a == b


def test_recipe_session_id_differs_for_a_different_model() -> None:
    other = {**_RECIPE, "model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}
    assert _recipe_session_id(**_RECIPE) != _recipe_session_id(**other)


def test_recipe_session_id_differs_for_a_different_lora_rank() -> None:
    other = {**_RECIPE, "lora_rank": 16}
    assert _recipe_session_id(**_RECIPE) != _recipe_session_id(**other)


def test_recipe_session_id_differs_for_a_different_seed() -> None:
    other = {**_RECIPE, "seed": 1}
    assert _recipe_session_id(**_RECIPE) != _recipe_session_id(**other)


# ---------------------------------------------------------------------------
# build_probe / probe_condition_name: pure Condition construction
# ---------------------------------------------------------------------------


def test_build_probe_ours_and_stock_differ_only_in_stock_and_name() -> None:
    ours = build_probe(
        model=_RECIPE["model"], revision=None, batch=1, lora_rank=8, lora_layers=-1,
        seed=0, arm="ours", seq_len=2048,
    )
    stock = build_probe(
        model=_RECIPE["model"], revision=None, batch=1, lora_rank=8, lora_layers=-1,
        seed=0, arm="stock", seq_len=2048,
    )
    assert ours.kind == stock.kind == "train_step"
    assert ours.params["stock"] is False
    assert stock.params["stock"] is True
    for key in ("model", "revision", "seq_len", "batch", "steps", "lora_rank",
                "lora_layers", "seed", "script_sha", "dataset_recipe"):
        assert ours.params[key] == stock.params[key]
    assert ours.name != stock.name


def test_build_probe_pins_the_dataset_recipe_and_script_sha() -> None:
    cond = build_probe(
        model=_RECIPE["model"], revision="main", batch=1, lora_rank=8, lora_layers=-1,
        seed=0, arm="ours", seq_len=2048,
    )
    assert cond.params["dataset_recipe"]
    assert cond.params["script_sha"] == script_sha()
    assert cond.params["revision"] == "main"


def test_build_probe_threads_compute_dtype_into_both_arms() -> None:
    """The `ours` probe runs the kernel, which needs bf16-compute hidden -- the 4-bit
    flagship loads fp16, so every probe casts the model to `compute_dtype` (else `auto`
    refuses the kernel on fp16, the worker crashes, and the probe reads as "does not
    fit" -> ours' max context would collapse to 0). Applied to BOTH arms, like the
    train-step bench, so the max-context comparison holds the trunk dtype constant.
    Defaults to bfloat16 -- the only sensible value for the 4-bit flagship sweep."""
    for arm in ("ours", "stock"):
        cond = build_probe(model=_RECIPE["model"], revision=None, batch=1, lora_rank=8,
                           lora_layers=-1, seed=0, arm=arm, seq_len=2048)
        assert cond.params["compute_dtype"] == "bfloat16"


def test_build_probe_honors_an_explicit_compute_dtype() -> None:
    cond = build_probe(model=_RECIPE["model"], revision=None, batch=1, lora_rank=8,
                       lora_layers=-1, seed=0, arm="ours", seq_len=2048,
                       compute_dtype="float32")
    assert cond.params["compute_dtype"] == "float32"


def test_recipe_session_id_differs_for_a_different_compute_dtype() -> None:
    """A different compute_dtype is a different recipe -- it must not silently resume a
    sweep measured at another dtype."""
    a = _recipe_session_id(**_RECIPE)  # default bfloat16
    b = _recipe_session_id(**_RECIPE, compute_dtype="float16")
    assert a != b


def test_default_compute_dtype_is_bfloat16() -> None:
    assert build_parser().parse_args([]).compute_dtype == "bfloat16"


def test_build_probe_threads_grad_checkpoint_defaulting_true() -> None:
    """North-Star measures the realistic long-context QLoRA setup, where activations are
    recomputed (grad_checkpoint=True) -- the regime where ours' flat loss-layer memory
    is the binding constraint and the max-context advantage over stock appears. Default
    True; applied to both arms."""
    for arm in ("ours", "stock"):
        cond = build_probe(model=_RECIPE["model"], revision=None, batch=1, lora_rank=8,
                           lora_layers=-1, seed=0, arm=arm, seq_len=2048)
        assert cond.params["grad_checkpoint"] is True


def test_build_probe_honors_explicit_grad_checkpoint_false() -> None:
    cond = build_probe(model=_RECIPE["model"], revision=None, batch=1, lora_rank=8,
                       lora_layers=-1, seed=0, arm="ours", seq_len=2048,
                       grad_checkpoint=False)
    assert cond.params["grad_checkpoint"] is False


def test_recipe_session_id_differs_for_a_different_grad_checkpoint() -> None:
    a = _recipe_session_id(**_RECIPE)  # default grad_checkpoint True
    b = _recipe_session_id(**_RECIPE, grad_checkpoint=False)
    assert a != b


def test_default_grad_checkpoint_is_true() -> None:
    assert build_parser().parse_args([]).grad_checkpoint is True


def test_build_probe_rejects_an_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_probe(model=_RECIPE["model"], revision=None, batch=1, lora_rank=8,
                    lora_layers=-1, seed=0, arm="bogus", seq_len=2048)


def test_probe_condition_name_is_filesystem_safe_and_seq_len_specific() -> None:
    name_a = probe_condition_name(model=_RECIPE["model"], seq_len=2048, arm="ours")
    name_b = probe_condition_name(model=_RECIPE["model"], seq_len=4096, arm="ours")
    assert "/" not in name_a
    assert name_a != name_b


# ---------------------------------------------------------------------------
# script_sha / model_slug: pure hash/string helpers
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    b = script_sha()
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_model_slug_is_filesystem_safe() -> None:
    assert "/" not in model_slug(FLAGSHIP_MODEL)


def test_defaults_are_sane() -> None:
    assert DEFAULT_START_SEQ_LEN > 0
    assert DEFAULT_GRANULARITY > 0
    assert MAX_CONTEXT_CEILING > DEFAULT_START_SEQ_LEN


# ---------------------------------------------------------------------------
# CLI shell: --help only (no model, no subprocess against a real model)
# ---------------------------------------------------------------------------


def test_help_runs_without_touching_a_model() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--start-seq-len" in proc.stdout


def test_default_model_is_the_flagship() -> None:
    """Checked via `build_parser()` directly (not by scraping `--help` text, which
    argparse line-wraps in a way that can split the repo id across lines)."""
    args = build_parser().parse_args([])
    assert args.model == FLAGSHIP_MODEL
    assert args.arm == "both"
