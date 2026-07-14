"""CLI surface: `plan` and `bench` subcommands, exit-code policy.

Pure helpers (arg parsing -> dataclass, rendering -> string) are exercised directly;
`main(argv)` end-to-end coverage stays at tiny/stub scale (no production-shape runs, no
Metal kernel dispatch -- `impl="naive"`/`"chunked"` only) so the suite runs fast in the
default (non-metal) lane.
"""
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from mlx_train_perf import cli
from mlx_train_perf.bench.artifacts import run_identity, write_result
from mlx_train_perf.cli import (
    _apply_quant_override,
    _bench_exit_code,
    _conditions_for_suite,
    _contribution_confirmed,
    _load_model_shape,
    _plan_exit_code,
    _read_status,
    _render_bench_summary,
    _render_plan_json,
    _render_plan_text,
    _train_config_from_args,
    main,
)
from mlx_train_perf.contribute import ContributionResult, MachineInfo, Preflight
from mlx_train_perf.core.guards import effective_memory_ceiling
from mlx_train_perf.errors import (
    BenchInputError,
    MachineDetectionError,
    MemoryBudgetError,
    PlanInputError,
)
from mlx_train_perf.plan.estimate import FitReport, ModelShape, TrainConfig


def _machine_refuses_worker_start() -> bool:
    """True when THIS machine's real availability trips guards' safe-start floor. The
    bench tests marked with `_needs_room_for_real_worker` spawn real worker subprocesses
    (subprocess-per-condition), and the child computes the real ceiling, out of any
    monkeypatch's reach -- on such a machine (e.g. a 7 GB CI runner) the honest outcome
    is a skip."""
    try:
        effective_memory_ceiling()
    except MemoryBudgetError:
        return True
    return False


_needs_room_for_real_worker = pytest.mark.skipif(
    _machine_refuses_worker_start(),
    reason="machine trips guards' safe-start floor; a real worker subprocess would refuse",
)

# --- brief's mandated Step 1 tests (verbatim) --------------------------------------


def _config(tmp_path: Path) -> Path:
    cfg = {"vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 2,
           "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
           "tie_word_embeddings": False, "model_type": "llama"}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_plan_fits_exit_zero_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plan", "--config", str(_config(tmp_path)), "--batch", "1",
               "--seq-len", "512", "--lora-rank", "8", "--budget-gb", "8", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["fits"] is True
    assert out["is_estimate"] is True


def test_plan_refuses_exit_one_with_suggestion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["plan", "--config", str(_config(tmp_path)), "--batch", "4096",
               "--seq-len", "8192", "--lora-rank", "8", "--budget-gb", "3", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["fits"] is False
    assert out["suggestion"]


def test_unknown_subcommand_exit_two() -> None:
    assert main(["frobnicate"]) == 2


# --- pure helper: arg-derived shape (I/O boundary) ---------------------------------


def test_load_model_shape_matches_from_config(tmp_path: Path) -> None:
    shape = _load_model_shape(str(_config(tmp_path)))
    assert shape == ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                                kv_heads=2, tied=False, quant_bits=None, quant_group=None)


def test_load_model_shape_missing_file_is_plan_input_error(tmp_path: Path) -> None:
    with pytest.raises(PlanInputError):
        _load_model_shape(str(tmp_path / "does-not-exist.json"))


def test_load_model_shape_invalid_json_is_plan_input_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PlanInputError):
        _load_model_shape(str(bad))


# --- pure helper: --quant-bits override --------------------------------------------


def test_apply_quant_override_none_is_passthrough() -> None:
    shape = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                        kv_heads=2, tied=False, quant_bits=None, quant_group=None)
    assert _apply_quant_override(shape, None) == shape


def test_apply_quant_override_sets_bits_and_default_group() -> None:
    shape = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                        kv_heads=2, tied=False, quant_bits=None, quant_group=None)
    overridden = _apply_quant_override(shape, 4)
    assert overridden.quant_bits == 4
    assert overridden.quant_group == 64


def test_apply_quant_override_preserves_config_derived_group() -> None:
    """A config.json that already carries non-default quantization metadata
    (group_size=32) must keep that group size when --quant-bits overrides the bit
    width -- silently flipping it to the fixed 64 default would understate the
    quantized weights bytes (rate 4/8 + 4/32 vs 4/8 + 4/64), producing an
    over-optimistic fit verdict."""
    shape = ModelShape(vocab=1000, hidden=64, layers=2, intermediate=128, heads=4,
                        kv_heads=2, tied=False, quant_bits=8, quant_group=32)
    overridden = _apply_quant_override(shape, 4)
    assert overridden.quant_bits == 4
    assert overridden.quant_group == 32


def test_plan_quant_bits_flows_through_to_weights_component(tmp_path: Path) -> None:
    base = main(["plan", "--config", str(_config(tmp_path)), "--batch", "1",
                 "--seq-len", "512", "--lora-rank", "8", "--json"])
    assert base == 0

    def _weights(extra_args: list[str]) -> int:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["plan", "--config", str(_config(tmp_path)), "--batch", "1",
                  "--seq-len", "512", "--lora-rank", "8", "--json", *extra_args])
        return int(json.loads(buf.getvalue())["components"]["weights"])

    dense_weights = _weights([])
    quantized_weights = _weights(["--quant-bits", "4"])
    assert quantized_weights < dense_weights


# --- pure helper: arg -> TrainConfig ------------------------------------------------


def test_train_config_from_args_lora_layers_follows_lora_rank() -> None:
    lora_cfg = _train_config_from_args(batch=1, seq_len=512, lora_rank=8, impl="kernel",
                                        shape_layers=12)
    assert lora_cfg.lora_layers == 12
    assert lora_cfg.lora_rank == 8

    full_ft_cfg = _train_config_from_args(batch=1, seq_len=512, lora_rank=0, impl="kernel",
                                           shape_layers=12)
    assert full_ft_cfg.lora_layers == 0


def test_train_config_from_args_threads_attention() -> None:
    """`--attention` threads into the TrainConfig; it defaults to "stock" (the 0.1.0
    behavior) when omitted."""
    default_cfg = _train_config_from_args(batch=1, seq_len=512, lora_rank=8, impl="kernel",
                                          shape_layers=12)
    assert default_cfg.attention == "stock"
    flash_cfg = _train_config_from_args(batch=1, seq_len=512, lora_rank=8, impl="kernel",
                                        shape_layers=12, attention="flash")
    assert flash_cfg.attention == "flash"


def test_plan_attention_flash_flag_shrinks_attention_component(tmp_path: Path) -> None:
    """`--attention flash` routes the planner through the O(N) flash term; at the flagship
    long context (seq=8192) its attention component is far below the stock O(N^2) term."""
    def _attention_component(extra_args: list[str]) -> int:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["plan", "--config", str(_config(tmp_path)), "--batch", "1",
                  "--seq-len", "8192", "--lora-rank", "8", "--json", *extra_args])
        return int(json.loads(buf.getvalue())["components"]["attention"])

    stock = _attention_component([])
    flash = _attention_component(["--attention", "flash"])
    assert flash < stock


# --- pure helper: FitReport rendering ------------------------------------------------


def _report(*, fits: bool, suggestion: TrainConfig | None) -> FitReport:
    return FitReport(
        fits=fits, predicted_peak_bytes=1024, budget_bytes=2048, headroom_bytes=1024,
        components={"weights": 512, "loss": 512}, suggestion=suggestion, is_estimate=True,
        provenance={"machine": "test", "macos": "0", "mlx_version": "0", "measured_date": "x"},
    )


def test_render_plan_json_round_trips_fields() -> None:
    report = _report(fits=True, suggestion=None)
    parsed = json.loads(_render_plan_json(report))
    assert parsed["fits"] is True
    assert parsed["suggestion"] is None
    assert parsed["components"] == {"weights": 512, "loss": 512}


def test_render_plan_json_serializes_suggestion() -> None:
    suggestion = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8,
                              lora_layers=4, grad_checkpoint=True, impl="kernel")
    report = _report(fits=False, suggestion=suggestion)
    parsed = json.loads(_render_plan_json(report))
    assert parsed["suggestion"]["batch"] == 1
    assert parsed["suggestion"]["seq_len"] == 1024


def test_render_plan_text_mentions_fit_verdict_and_components() -> None:
    text = _render_plan_text(_report(fits=True, suggestion=None))
    assert "fits: yes" in text
    assert "weights" in text


def test_render_plan_text_includes_suggestion_when_present() -> None:
    suggestion = TrainConfig(batch=1, seq_len=1024, dtype="bfloat16", lora_rank=8,
                              lora_layers=4, grad_checkpoint=True, impl="kernel")
    text = _render_plan_text(_report(fits=False, suggestion=suggestion))
    assert "fits: no" in text
    assert "suggestion" in text


def test_plan_exit_code() -> None:
    assert _plan_exit_code(_report(fits=True, suggestion=None)) == 0
    assert _plan_exit_code(_report(fits=False, suggestion=None)) == 1


def test_plan_text_output_when_not_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["plan", "--config", str(_config(tmp_path)), "--batch", "1",
               "--seq-len", "512", "--lora-rank", "8", "--budget-gb", "8"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fits: yes" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_plan_config_tool_error_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plan", "--config", "/no/such/config.json", "--batch", "1",
               "--seq-len", "512", "--lora-rank", "8"])
    assert rc == 2
    assert "Error" in capsys.readouterr().err


# --- pure helper: bench suite -> Condition list --------------------------------------


def test_conditions_for_suite_builds_one_condition_per_n() -> None:
    conditions = _conditions_for_suite(
        "loss-layer", n_values=[64, 128], d=8, v=16, dtype="float32", impl="naive",
        quantized=False, group_size=64, bits=4, chunk_size=None, reps=1, seed=0,
    )
    assert [c.name for c in conditions] == ["loss_layer_n64", "loss_layer_n128"]
    assert all(c.kind == "loss_layer" for c in conditions)
    assert conditions[0].params["n"] == 64
    assert conditions[1].params["n"] == 128


def test_conditions_for_suite_rejects_unsupported_suite() -> None:
    with pytest.raises(BenchInputError):
        _conditions_for_suite("bogus-suite", n_values=[64], d=8, v=16, dtype="float32",
                               impl="naive", quantized=False, group_size=64, bits=4,
                               chunk_size=None, reps=1, seed=0)


# --- pure helper: bench exit-code / status reading ----------------------------------


def test_bench_exit_code_zero_when_all_ok() -> None:
    assert _bench_exit_code(["ok", "ok"]) == 0


def test_bench_exit_code_one_on_any_error_or_refusal() -> None:
    assert _bench_exit_code(["ok", "error"]) == 1
    assert _bench_exit_code(["ok", "refused"]) == 1


def test_bench_exit_code_zero_on_empty() -> None:
    assert _bench_exit_code([]) == 0


def test_read_status_parses_written_result(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    write_result(path, run_identity(model="m", session_id="s1"), "ok", wall_s=1.0)
    assert _read_status(path) == "ok"


def test_read_status_corrupt_file_is_error(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text("{not json")
    assert _read_status(path) == "error"


def test_render_bench_summary_lists_conditions(tmp_path: Path) -> None:
    path = tmp_path / "loss_layer_n64.json"
    write_result(path, run_identity(model="m", impl="naive", session_id="s1"), "ok",
                 wall_s=1.0)
    summary = json.loads(_render_bench_summary([path], ["ok"]))
    assert summary["conditions"] == [{"name": "loss_layer_n64", "status": "ok",
                                       "path": str(path)}]
    assert "ratios" in summary
    assert "cross_session_excluded" in summary


@_needs_room_for_real_worker
def test_bench_subcommand_in_process_tiny_scale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Drives `_cmd_bench` in-process (not via subprocess) at tiny/stub scale --
    `impl="naive"` needs no Metal JIT, so this stays in the default (non-metal) lane.
    Still needs machine room: `run_conditions` spawns one real worker child per
    condition, and that child computes the real memory ceiling."""
    out_dir = tmp_path / "results"
    rc = main(["bench", "--suite", "loss-layer", "--out", str(out_dir), "--n", "64",
               "--d", "8", "--v", "16", "--dtype", "float32", "--impl", "naive",
               "--reps", "1"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["conditions"] == [
        {"name": "loss_layer_n64", "status": "ok",
         "path": str(out_dir / "loss_layer_n64.json")}
    ]


# --- end-to-end subprocess invocations, tiny/stub scale (one per subcommand) --------


def test_plan_end_to_end_subprocess(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_train_perf.cli", "plan", "--config",
         str(_config(tmp_path)), "--batch", "1", "--seq-len", "512", "--lora-rank", "8",
         "--budget-gb", "8", "--json"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["fits"] is True


@_needs_room_for_real_worker
def test_bench_end_to_end_subprocess(tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_train_perf.cli", "bench", "--suite", "loss-layer",
         "--out", str(out_dir), "--n", "64", "--d", "8", "--v", "16", "--dtype",
         "float32", "--impl", "naive", "--reps", "1"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["conditions"] == [
        {"name": "loss_layer_n64", "status": "ok", "path": str(out_dir / "loss_layer_n64.json")}
    ]


def test_bench_unsupported_suite_tool_error_exit_two(tmp_path: Path) -> None:
    assert main(["bench", "--suite", "bogus-suite", "--out", str(tmp_path)]) == 2


def test_bench_help_documents_exit_code_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """The exit-1 policy (both 'error' and 'refused' count as non-ok) must be visible in
    `bench -h`, not only in the module docstring. `main` catches argparse's own
    `SystemExit(0)` for `-h`, so this asserts on the return code, not a raised exception."""
    rc = main(["bench", "-h"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "error" in out
    assert "refused" in out


# --- contribute subcommand ----------------------------------------------------------


def test_contribution_confirmed_yes_flag_always_proceeds() -> None:
    assert _contribution_confirmed(yes=True, isatty=False, prompt=lambda _p: "n") is True


def test_contribution_confirmed_non_tty_without_yes_refuses() -> None:
    """A non-interactive run without --yes must NOT start a heavy GPU job unattended."""
    assert _contribution_confirmed(yes=False, isatty=False, prompt=lambda _p: "y") is False


def test_contribution_confirmed_tty_prompt_gates_on_yes_response() -> None:
    assert _contribution_confirmed(yes=False, isatty=True, prompt=lambda _p: "y") is True
    assert _contribution_confirmed(yes=False, isatty=True, prompt=lambda _p: "") is False


def test_contribute_help_documents_the_tiers(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["contribute", "-h"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "quick" in out
    assert "full" in out


def _machine_info() -> MachineInfo:
    return MachineInfo(chip="Apple M1 Max", ram_gib=32, ram_bytes=34359738368,
                       macos="15.5", mlx_version="0.32.0", package_version="0.2.0")


def _ok_preflight(warnings: tuple[str, ...] = ()) -> Preflight:
    return Preflight(ok=True, refusal=None, warnings=warnings)


def test_cmd_contribute_prints_eta_and_pr_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """`contribute --yes` on a stubbed measurement path: the ETA prints up front, the
    pre-flight warning prints, and the artifact path + suggested PR text print on
    success, exit 0."""
    art_path = tmp_path / "apple-m1-max-32gb-2026-07-12.json"
    art_path.write_text("{}")
    result = ContributionResult(
        refused=False, refusal=None, artifact_path=art_path,
        warnings=("running on battery power",),
        pr_title="Community benchmark: Apple M1 Max 32 GB", pr_body="body",
    )
    monkeypatch.setattr(cli, "detect_machine", _machine_info)
    monkeypatch.setattr(cli, "run_preflight",
                         lambda **_kw: _ok_preflight(("running on battery power",)))
    monkeypatch.setattr(cli, "run_contribution", lambda **_kw: result)

    rc = main(["contribute", "--yes", "--tier", "quick", "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "estimated time" in captured.out
    assert "10" in captured.out          # the quick ETA range low bound
    assert "15" in captured.out          # ... and high bound
    assert "Community benchmark: Apple M1 Max 32 GB" in captured.out
    assert "battery" in captured.err                              # warning to stderr


def test_cmd_contribute_returns_one_on_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    result = ContributionResult(
        refused=True, refusal="not confirmed -- pass --yes or confirm at the prompt to start",
        artifact_path=None, warnings=(), pr_title=None, pr_body=None,
    )
    monkeypatch.setattr(cli, "detect_machine", _machine_info)
    monkeypatch.setattr(cli, "run_preflight", lambda **_kw: _ok_preflight())
    monkeypatch.setattr(cli, "run_contribution", lambda **_kw: result)

    rc = main(["contribute", "--yes"])
    assert rc == 1
    assert "refused" in capsys.readouterr().err


# --- finding A: pre-flight runs BEFORE the confirmation prompt / any measurement -----


def test_cmd_contribute_preflight_warning_prints_before_the_confirmation_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """The kit's README promises the pre-flight warning names the expected-vs-measured
    memory gap "so you can close other apps and retry" -- that promise only holds if the
    warning is on screen BEFORE the confirmation prompt (and well before any heavy
    measurement), not after. Order is asserted via the fake seams: the preflight, the
    confirmation prompt, and the measurement call each record themselves into `order`,
    and the fake prompt snapshots stdout the instant it is invoked."""
    order: list[str] = []
    snapshot_at_confirm: list[str] = []

    def _fake_run_preflight(**_kw: object) -> Preflight:
        order.append("preflight")
        return _ok_preflight(("expected ~58 GB free, measured 20 GB",))

    def _fake_input(_msg: str) -> str:
        order.append("confirm")
        snapshot_at_confirm.append(capsys.readouterr().err)
        return "y"

    def _fake_run_contribution(**_kw: object) -> ContributionResult:
        order.append("measure")
        return ContributionResult(
            refused=False, refusal=None, artifact_path=tmp_path / "a.json",
            warnings=(), pr_title="t", pr_body="b",
        )

    monkeypatch.setattr(cli, "detect_machine", _machine_info)
    monkeypatch.setattr(cli, "run_preflight", _fake_run_preflight)
    monkeypatch.setattr(cli, "run_contribution", _fake_run_contribution)
    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    rc = main(["contribute", "--tier", "quick", "--out", str(tmp_path)])

    assert rc == 0
    assert order == ["preflight", "confirm", "measure"]
    assert "58 GB" in snapshot_at_confirm[0]     # already printed by the time confirm ran


def test_cmd_contribute_red_preflight_refuses_before_any_confirmation_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """A red/too-crowded pre-flight refuses outright and must never reach the
    confirmation prompt or the measurement seam."""
    prompted: list[str] = []

    def _fake_run_preflight(**_kw: object) -> Preflight:
        return Preflight(
            ok=False,
            refusal="system memory pressure is critical (red); refusing to start",
            warnings=(),
        )

    def _fake_input(msg: str) -> str:
        prompted.append(msg)
        return "y"

    def _fake_run_contribution(**_kw: object) -> ContributionResult:
        raise AssertionError("run_contribution must not run when the pre-flight refuses")

    monkeypatch.setattr(cli, "detect_machine", _machine_info)
    monkeypatch.setattr(cli, "run_preflight", _fake_run_preflight)
    monkeypatch.setattr(cli, "run_contribution", _fake_run_contribution)
    monkeypatch.setattr("builtins.input", _fake_input)

    rc = main(["contribute", "--tier", "quick", "--out", str(tmp_path)])

    assert rc == 1
    assert prompted == []                                    # never reached the prompt
    assert "red" in capsys.readouterr().err


# --- finding E: a machine-detection failure maps to the tool-error exit (2) ----------


def test_cmd_contribute_machine_detection_failure_is_a_tool_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def _raise() -> MachineInfo:
        raise MachineDetectionError(
            "failed to read the CPU brand string via `sysctl`: exit status 1"
        )

    monkeypatch.setattr(cli, "detect_machine", _raise)

    rc = main(["contribute", "--yes"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "sysctl" in captured.err
