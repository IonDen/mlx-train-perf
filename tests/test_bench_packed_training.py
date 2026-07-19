"""Tests for the packed-training benchmark (`scripts/bench_packed_training.py`), its
Alpaca prep step (`scripts/prep_alpaca.py`), and the GPU-free helpers they lean on
(`data.packing`'s batching-stats family + `bench.worker`'s packed throughput helpers).

Everything here is GPU-free and model-free (the brief's hard constraint): the pure
padding-waste / real-token / histogram / throughput math is unit-tested against
hand-computed mini datasets; the worker's `packed_train` dispatch is exercised with a
STUBBED `run_packed_train` (never a real model); the scripts are loaded by path (the
existing `scripts/` convention — no `__init__.py`) and their model-loading condition is
only reachable through the runner, stubbed at the subprocess boundary. The one network
touch (the real Alpaca download) is `@pytest.mark.network`, collected and skipped by
default, and additionally `importorskip`s the parquet reader so it skips cleanly even
under `--run-network` on a machine without it.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_train_perf.bench import runner as bench_runner
from mlx_train_perf.bench import worker
from mlx_train_perf.bench.artifacts import new_session_id
from mlx_train_perf.bench.runner import run_conditions
from mlx_train_perf.core.guards import EffectiveCeiling
from mlx_train_perf.data.packing import (
    length_histogram,
    pack_indices,
    packed_batching_stats,
    stock_batching_stats,
)
from mlx_train_perf.errors import LaunchBudgetError, MlxTrainPerfError

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_packed_training  # noqa: E402 -- import must follow the sys.path insert
import prep_alpaca  # noqa: E402
from bench_packed_training import (  # noqa: E402
    RESULTS,
    build_condition,
    condition_name,
    dataset_sha,
    model_slug,
    script_sha,
)

_BENCH_SCRIPT = _SCRIPTS_DIR / "bench_packed_training.py"
_PREP_SCRIPT = _SCRIPTS_DIR / "prep_alpaca.py"


# ---------------------------------------------------------------------------
# data.packing: length_histogram (pure, GPU-free)
# ---------------------------------------------------------------------------


def test_length_histogram_reports_count_mean_median_p90_max() -> None:
    hist = length_histogram([10, 20, 30, 40])
    assert hist.count == 4
    assert hist.mean == pytest.approx(25.0)
    assert hist.median == pytest.approx(25.0)
    assert hist.minimum == 10
    assert hist.maximum == 40
    assert hist.total_tokens == 100
    # numpy's linear-interpolation 90th percentile of [10,20,30,40] is 37.0.
    assert hist.p90 == pytest.approx(37.0)


def test_length_histogram_rejects_empty() -> None:
    with pytest.raises(MlxTrainPerfError, match="empty"):
        length_histogram([])


# ---------------------------------------------------------------------------
# data.packing: stock_batching_stats -- the exact stock sort+batch+pad replay
# (mlx_lm.tuner.trainer.iterate_batches, trainer.py:102-170). Padding-waste is the
# public claim, so it is pinned against a hand-computed mini dataset.
# ---------------------------------------------------------------------------


def test_stock_batching_stats_padding_waste_on_hand_computed_dataset() -> None:
    # lengths [10,20,30,40], batch_size 2, max_seq_length 256. Stock sorts ascending,
    # then batches in pairs (dropping any trailing partial). pad_to=32, width =
    # min(1 + 32*ceil(max_in_batch/32), msl):
    #   batch [10,20]: width 33, real 30, padded 2*33 = 66
    #   batch [30,40]: width 65, real 70, padded 2*65 = 130
    stats = stock_batching_stats([10, 20, 30, 40], batch_size=2, max_seq_length=256)
    assert stats.num_batches == 2
    assert stats.real_tokens_total == 100
    assert stats.padded_tokens_total == 196
    assert stats.padding_waste_fraction == pytest.approx(96 / 196)
    assert stats.mean_real_tokens_per_step == pytest.approx(50.0)
    assert stats.mean_samples_per_step == pytest.approx(2.0)


def test_stock_batching_stats_drops_the_trailing_partial_batch() -> None:
    # 5 sequences, batch_size 2 -> exactly 2 full batches (4 consumed), 1 dropped --
    # mirroring stock's `range(0, len(idx) - batch_size + 1, batch_size)`.
    stats = stock_batching_stats([1, 2, 3, 4, 5], batch_size=2, max_seq_length=256)
    assert stats.num_batches == 2


def test_stock_batching_stats_truncates_to_max_seq_length() -> None:
    # A single over-long sequence: real content counts only min(len, msl); the padded
    # width is capped at msl too (never msl+1) -- stock's `min(..., max_seq_length)`.
    stats = stock_batching_stats([100], batch_size=1, max_seq_length=64)
    assert stats.real_tokens_total == 64
    assert stats.padded_tokens_total == 64
    assert stats.padding_waste_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# data.packing: packed_batching_stats -- per-step real tokens over the CONSUMED packs,
# plus the whole-dataset pack_stats utilization/separator/tail fractions.
# ---------------------------------------------------------------------------


def test_packed_batching_stats_on_hand_computed_dataset() -> None:
    # 6 sequences of length 4, pack_len 10 (capacity 11), cost = min(4,10)+1 = 5 each.
    # First-fit closes each pack after 2 sequences (room 11 - 5 - 5 = 1 <= 1) ->
    # 3 packs of 2. batch_size 1 -> 3 steps, all consumed.
    stats = packed_batching_stats([4] * 6, 10, batch_size=1, seed=0)
    assert stats.num_batches == 3
    assert stats.real_tokens_total == 24
    assert stats.mean_real_tokens_per_step == pytest.approx(8.0)
    assert stats.mean_samples_per_step == pytest.approx(2.0)
    # capacity_tokens = 3 packs * 10 = 30. real 24 -> 0.8; 6 separators -> 0.2;
    # 3 tail slots (one per pack) -> 0.1.
    assert stats.utilization == pytest.approx(0.8)
    assert stats.separator_fraction == pytest.approx(0.2)
    assert stats.tail_pad_fraction == pytest.approx(0.1)


def test_packed_batching_stats_real_tokens_match_the_consumed_packs() -> None:
    # Cross-check against the SAME pack layout: real_tokens_total must equal the sum of
    # min(len, pack_len) over exactly the packs a whole number of batches consumes.
    lengths = [7, 3, 11, 5, 9, 2, 8, 4]
    pack_len, batch_size = 16, 2
    packs = pack_indices(lengths, pack_len, seed=1, epoch=0)
    num_batches = len(packs) // batch_size
    consumed = packs[: num_batches * batch_size]
    expected_real = sum(min(lengths[i], pack_len) for pack in consumed for i in pack)
    stats = packed_batching_stats(lengths, pack_len, batch_size=batch_size, seed=1)
    assert stats.num_batches == num_batches
    assert stats.real_tokens_total == expected_real


# ---------------------------------------------------------------------------
# bench.worker: median_post_warmup + packed_throughput_fields (pure, GPU-free)
# ---------------------------------------------------------------------------


def test_median_post_warmup_drops_the_warmup_prefix() -> None:
    assert worker.median_post_warmup([10.0, 10.0, 1.0, 2.0, 3.0], 2) == pytest.approx(2.0)


def test_median_post_warmup_falls_back_to_all_when_warmup_exceeds_samples() -> None:
    assert worker.median_post_warmup([5.0], 3) == pytest.approx(5.0)


def test_median_post_warmup_rejects_empty() -> None:
    with pytest.raises(MlxTrainPerfError, match="wall"):
        worker.median_post_warmup([], 2)


def test_packed_throughput_fields_composes_real_tps_and_samples_hour() -> None:
    fields = worker.packed_throughput_fields(
        mean_real_tokens_per_step=100.0, mean_samples_per_step=2.0, median_step_wall_s=0.5,
    )
    assert fields["real_tokens_per_second"] == pytest.approx(200.0)
    assert fields["samples_per_hour"] == pytest.approx(14400.0)
    assert fields["median_step_wall_s"] == pytest.approx(0.5)


def test_packed_throughput_fields_zero_wall_is_zero_not_a_crash() -> None:
    fields = worker.packed_throughput_fields(
        mean_real_tokens_per_step=100.0, mean_samples_per_step=2.0, median_step_wall_s=0.0,
    )
    assert fields["real_tokens_per_second"] == pytest.approx(0.0)
    assert fields["samples_per_hour"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bench.worker.main: kind="packed_train" dispatch (run_packed_train STUBBED -- no model)
# ---------------------------------------------------------------------------


@pytest.fixture
def _plentiful_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small CI runners trip guards' safe-start floor inside `worker.main`; that is the
    runner environment, not the behavior under test (mirrors test_worker_train_step)."""
    monkeypatch.setattr(
        worker, "effective_memory_ceiling",
        lambda: EffectiveCeiling(ceiling_bytes=64 << 30, warning=None),
    )


_PACKED_PARAMS: dict[str, object] = {
    "model": "m/x", "data": "/x.jsonl", "pack_len": 4096, "batch": 1, "steps": 2,
    "arm": "packed", "seed": 0, "dataset_sha": "deadbeef",
}


@pytest.mark.usefixtures("_plentiful_ceiling")
def test_worker_main_packed_train_dispatches_and_writes_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(params: dict[str, object]) -> dict[str, object]:
        captured["params"] = params
        return {"arm": params["arm"], "real_tokens_per_second": 1.0, "loss_all": [1.0]}

    monkeypatch.setattr(worker, "run_packed_train", _fake_run)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "packed_train", "params": dict(_PACKED_PARAMS), "session_id": "s1",
        "attention_impl": "flash", "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["identity"]["kind"] == "packed_train"
    assert data["identity"]["attention_impl"] == "flash"
    assert data["identity"]["arm"] == "packed"
    assert data["identity"]["pack_len"] == 4096
    assert data["identity"]["dataset_sha"] == "deadbeef"
    assert data["arm"] == "packed"


@pytest.mark.usefixtures("_plentiful_ceiling")
def test_worker_main_packed_train_launch_budget_refusal_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `LaunchBudgetError` out of the flash path is a RESULT (`status="refused"`,
    rc 0), the same envelope `run_train_step`'s refusals use -- the script turns it into
    a nonzero process exit (the repo bench exit policy), tested separately below."""
    def _raise(_params: dict[str, object]) -> dict[str, object]:
        raise LaunchBudgetError("no launch budget at this shape")

    monkeypatch.setattr(worker, "run_packed_train", _raise)
    out = tmp_path / "r.json"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "kind": "packed_train", "params": dict(_PACKED_PARAMS), "session_id": "s1",
        "attention_impl": "flash", "out": str(out),
    }))
    rc = worker.main(["--config", str(cfg)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["status"] == "refused"


# ---------------------------------------------------------------------------
# bench_packed_training.py: script_sha / model_slug / dataset_sha / condition_name
# ---------------------------------------------------------------------------


def test_script_sha_is_a_stable_short_hex_digest() -> None:
    a = script_sha()
    assert a == script_sha()
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_model_slug_is_filesystem_safe() -> None:
    assert model_slug("mlx-community/Qwen3-8B-4bit") == "mlx-community__Qwen3-8B-4bit"
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


def test_condition_name_embeds_slug_pack_len_and_arm() -> None:
    name = condition_name(model="mlx-community/Qwen3-8B-4bit", pack_len=4096, arm="packed")
    assert name == "packed_train_mlx-community__Qwen3-8B-4bit_pack4096_packed"


def test_condition_names_differ_across_arms() -> None:
    """Gotcha 18: two arms against the same --out dir must never collide on FILENAME --
    the arm is part of the name, not only the artifact identity."""
    stock = condition_name(model="m/a", pack_len=4096, arm="stock")
    packed = condition_name(model="m/a", pack_len=4096, arm="packed")
    assert stock != packed
    assert stock.endswith("_stock")
    assert packed.endswith("_packed")


# ---------------------------------------------------------------------------
# bench_packed_training.py: build_condition (pure Condition construction)
# ---------------------------------------------------------------------------


def _build_one(**overrides: object) -> bench_runner.Condition:
    kwargs: dict[str, object] = {
        "model": "m/a", "data": "/prepped.jsonl", "pack_len": 4096, "batch_size": 1,
        "arm": "packed", "steps": 30, "warmup": 5, "lora_rank": 8, "lora_layers": -1,
        "impl": "auto", "learning_rate": 1e-5, "seed": 0, "compute_dtype": None,
        "grad_checkpoint": False, "revision": None, "dataset_sha": "cafef00d",
    }
    kwargs.update(overrides)
    return build_condition(**kwargs)  # type: ignore[arg-type]


def test_build_condition_is_a_packed_train_flash_condition() -> None:
    cond = _build_one()
    assert cond.kind == "packed_train"
    # Both arms use flash + fused CE -- the ONLY variable is the batching strategy, so
    # the attention arm rides the dedicated identity field, fixed to flash.
    assert cond.attention_impl == "flash"


def test_build_condition_params_carry_arm_pack_len_dataset_sha() -> None:
    cond = _build_one(arm="stock", pack_len=2048, dataset_sha="abc123")
    assert cond.params["arm"] == "stock"
    assert cond.params["pack_len"] == 2048
    assert cond.params["dataset_sha"] == "abc123"
    assert cond.params["data"] == "/prepped.jsonl"
    assert cond.params["batch"] == 1
    assert cond.params["script_sha"] == script_sha()


def test_build_condition_name_carries_the_arm(tmp_path: Path) -> None:  # noqa: ARG001
    assert _build_one(arm="stock").name.endswith("_stock")
    assert _build_one(arm="packed").name.endswith("_packed")


def _crash_worker(_config_path: Path) -> subprocess.CompletedProcess[str]:
    """A `_spawn_worker` stand-in that "crashes" so `run_conditions` writes the identity
    it built (never loading a real model) -- the same stub `test_bench_train_step` uses."""
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="stub crash")


def test_condition_identity_carries_arm_pack_len_dataset_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full production path through `run_conditions`: `arm`/`pack_len`/`dataset_sha`
    reach `condition_identity` through `params` (they are NOT reserved keys), so the
    written artifact's identity carries them -- and `attention_impl` rides its dedicated
    slot. Worker stubbed at the subprocess boundary; the crash-envelope branch writes
    the computed identity, so reading it back proves the fields landed."""
    monkeypatch.setattr(bench_runner, "_spawn_worker", _crash_worker)
    cond = _build_one(arm="packed", pack_len=4096, dataset_sha="deadbeef")
    paths = run_conditions([cond], tmp_path, session_id=new_session_id())
    ident = json.loads(paths[0].read_text())["identity"]
    assert ident["arm"] == "packed"
    assert ident["pack_len"] == 4096
    assert ident["dataset_sha"] == "deadbeef"
    assert ident["attention_impl"] == "flash"
    assert ident["kind"] == "packed_train"


def test_stock_and_packed_invocations_get_different_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench_runner, "_spawn_worker", _crash_worker)
    session_id = new_session_id()
    stock = run_conditions([_build_one(arm="stock")], tmp_path / "stock", session_id=session_id)
    packed = run_conditions(
        [_build_one(arm="packed")], tmp_path / "packed", session_id=session_id,
    )
    id_stock = json.loads(stock[0].read_text())["identity"]
    id_packed = json.loads(packed[0].read_text())["identity"]
    assert id_stock["arm"] == "stock"
    assert id_packed["arm"] == "packed"
    assert id_stock != id_packed


# ---------------------------------------------------------------------------
# bench_packed_training.py main: exit policy (refused -> nonzero) + CLI shell
# ---------------------------------------------------------------------------


def _stub_run_conditions_writing(status: str, **fields: object):
    def fake(conditions, out_dir, *, session_id):  # type: ignore[no-untyped-def]  # noqa: ARG001
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{conditions[0].name}.json"
        path.write_text(json.dumps({"status": status, "identity": {}, **fields}))
        return [path]
    return fake


def _run_main_with_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, status: str, **fields: object,
) -> int:
    data = tmp_path / "d.jsonl"
    data.write_text('{"tokens":[1,2,3,4],"offset":1}\n')
    monkeypatch.setattr(
        bench_packed_training, "run_conditions",
        _stub_run_conditions_writing(status, **fields),
    )
    return bench_packed_training.main([
        "--model", "m/x", "--data", str(data), "--arm", "packed",
        "--out", str(tmp_path / "out"), "--steps", "2",
    ])


def test_main_exits_zero_on_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rc = _run_main_with_artifact(
        tmp_path, monkeypatch, status="ok", real_tokens_per_second=100.0,
    )
    assert rc == 0


def test_main_exits_one_on_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rc = _run_main_with_artifact(tmp_path, monkeypatch, status="refused", error="no budget")
    assert rc == 1


def test_main_exits_one_on_crash_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = _run_main_with_artifact(
        tmp_path, monkeypatch, status="error", error_type="WorkerCrashed",
    )
    assert rc == 1


def test_main_requires_model_data_and_arm() -> None:
    with pytest.raises(SystemExit):
        bench_packed_training.main([])


def test_bench_help_runs_without_touching_a_model() -> None:
    proc = subprocess.run(
        [sys.executable, str(_BENCH_SCRIPT), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--arm" in proc.stdout
    assert "--pack-len" in proc.stdout
    assert "--data" in proc.stdout


def test_default_results_directory_is_under_artifacts() -> None:
    assert RESULTS.name == "packed_bench"
    assert "_artifacts" in RESULTS.parts


# ---------------------------------------------------------------------------
# prep_alpaca.py: chat-template + tokenize-to-(tokens, offset) (pure, fake tokenizer)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """A minimal `apply_chat_template` stand-in: one token per character of each
    message's content, a `500` end-of-turn marker after every message, and a `1000`
    generation-prompt marker when `add_generation_prompt` is set. Deterministic, and it
    makes the prompt tokenization a strict prefix of the full one -- the ideal (tokens,
    offset) case the real chat template targets."""

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *,
        add_generation_prompt: bool = False, **_kw: object,
    ) -> list[int]:
        ids: list[int] = []
        for message in conversation:
            ids.extend(ord(c) for c in message["content"])
            ids.append(500)
        if add_generation_prompt:
            ids.append(1000)
        return ids


def test_user_content_joins_instruction_and_input() -> None:
    assert prep_alpaca.user_content(
        {"instruction": "ab", "input": "X", "output": "y"}
    ) == "ab\n\nX"


def test_user_content_omits_empty_input() -> None:
    assert prep_alpaca.user_content(
        {"instruction": "ab", "input": "", "output": "y"}
    ) == "ab"


def test_tokenize_record_returns_full_tokens_and_prompt_offset() -> None:
    tokens, offset = prep_alpaca.tokenize_record(
        _FakeTokenizer(), {"instruction": "abc", "input": "", "output": "de"},
    )
    # prompt = a,b,c,<eot>,<gen> -> len 5; full = a,b,c,<eot>,d,e,<eot> -> len 7.
    assert tokens == [97, 98, 99, 500, 100, 101, 500]
    assert offset == 5
    assert offset < len(tokens)  # the completion is supervised (non-empty region)


def test_records_to_pairs_respects_max_samples() -> None:
    records = [
        {"instruction": f"i{i}", "input": "", "output": f"o{i}"} for i in range(5)
    ]
    pairs = prep_alpaca.records_to_pairs(_FakeTokenizer(), records, max_samples=3)
    assert len(pairs) == 3
    assert all(isinstance(toks, list) and isinstance(off, int) for toks, off in pairs)


def test_dataset_stats_block_has_histogram_stock_waste_and_packed_utilization() -> None:
    pairs = [([0] * n, 0) for n in (10, 20, 30, 40)]
    stats = prep_alpaca.dataset_stats(pairs, batch_size=2, pack_len=256, seed=0)
    assert stats["count"] == 4
    assert stats["max"] == 40
    stock = stats["stock_batching"]
    assert isinstance(stock, dict)
    assert stock["batch_size"] == 2
    assert stock["padding_waste_fraction"] == pytest.approx(96 / 196)
    packed = stats["packed"]
    assert isinstance(packed, dict)
    assert packed["pack_len"] == 256
    assert 0.0 < packed["utilization"] <= 1.0


def test_prep_help_runs_without_touching_the_network() -> None:
    proc = subprocess.run(
        [sys.executable, str(_PREP_SCRIPT), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--model" in proc.stdout
    assert "--revision" in proc.stdout


def test_pinned_alpaca_revision_is_a_full_commit_sha() -> None:
    # A branch name (e.g. "main") would make the prep non-deterministic; the pin must be
    # a resolved 40-hex commit sha.
    rev = prep_alpaca.ALPACA_REVISION
    assert len(rev) == 40
    assert all(c in "0123456789abcdef" for c in rev)


@pytest.mark.network
def test_download_alpaca_returns_instruction_records() -> None:
    """The only network touch: skipped by default (collection-gated), and `importorskip`s
    the parquet reader so it also skips cleanly under --run-network without it."""
    pytest.importorskip("pyarrow")
    records = prep_alpaca.download_alpaca(prep_alpaca.ALPACA_REVISION)
    assert len(records) > 1000
    assert {"instruction", "input", "output"} <= set(records[0])
