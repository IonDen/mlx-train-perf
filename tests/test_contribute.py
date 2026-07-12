"""Fakes-at-the-boundary unit tests for the community contribution kit
(`src/mlx_train_perf/contribute.py`, backlog 0015, spec §7).

Everything here is GPU-free and model-free: machine detection reads fake sysctl/
device-info/version strings, the RAM->shape scaling table is a pure function tested at
exact values, the ETA is summed from a pure tier table, the pre-flight decision is pure,
the community artifact is assembled from synthetic per-bench summaries, and the
end-to-end `run_contribution` orchestration stubs its measurement seam (`_measure_bench`)
so no real bench subprocess, GPU dispatch, or model load happens -- the same
stub-the-spawn-seam discipline `tests/test_bench_attention_op.py` uses for `run_grid`.
The real quick-tier run (a heavy GPU job) is the controller's `--run-smoke` step, never
this suite.
"""
import json
import subprocess
from pathlib import Path

import pytest

from mlx_train_perf import contribute
from mlx_train_perf.contribute import (
    COMMUNITY_SCHEMA_VERSION,
    ContributionResult,
    MachineInfo,
    Preflight,
    artifact_filename,
    benches_for_tier,
    build_community_artifact,
    classify_memory_pressure,
    collect_memory_warnings,
    detect_machine,
    eta_minutes_for_tier,
    evaluate_preflight,
    format_eta,
    machine_slug,
    parse_chip,
    pr_body,
    pr_title,
    ram_class_for,
    ram_gib_from_bytes,
    run_contribution,
    run_preflight,
    shapes_for_ram,
    summarize_artifact_file,
)
from mlx_train_perf.core.guards import EffectiveCeiling
from mlx_train_perf.errors import BenchInputError, MachineDetectionError, MemoryBudgetError

# --- machine detection: pure parsing --------------------------------------------------


def test_parse_chip_strips_the_sysctl_brand_string() -> None:
    assert parse_chip("Apple M1 Max\n") == "Apple M1 Max"


def test_parse_chip_collapses_internal_whitespace() -> None:
    assert parse_chip("  Apple  M2   Ultra  ") == "Apple M2 Ultra"


def test_ram_gib_from_bytes_rounds_to_the_nearest_gib() -> None:
    assert ram_gib_from_bytes(34359738368) == 32     # exactly 32 GiB
    assert ram_gib_from_bytes(68719476736) == 64
    assert ram_gib_from_bytes(17179869184) == 16


def test_machine_slug_is_filesystem_safe_and_carries_ram() -> None:
    assert machine_slug(chip="Apple M1 Max", ram_gib=32) == "apple-m1-max-32gb"


def test_artifact_filename_matches_the_spec_convention() -> None:
    assert artifact_filename(chip="Apple M1 Max", ram_gib=32, date="2026-07-12") == (
        "apple-m1-max-32gb-2026-07-12.json"
    )


def test_detect_machine_assembles_from_injected_readers() -> None:
    info = detect_machine(
        chip_reader=lambda: "Apple M2 Ultra\n",
        ram_bytes_reader=lambda: 68719476736,
        macos_reader=lambda: "15.5",
        mlx_version_reader=lambda: "0.32.0",
        package_version_reader=lambda: "0.2.0",
    )
    assert info == MachineInfo(
        chip="Apple M2 Ultra", ram_gib=64, ram_bytes=68719476736, macos="15.5",
        mlx_version="0.32.0", package_version="0.2.0",
    )


# --- RAM -> shape scaling table (pure, exact-value tested) -----------------------------


def test_ram_class_for_snaps_down_to_the_nearest_class() -> None:
    assert ram_class_for(16) == 16
    assert ram_class_for(24) == 16     # between classes -> the lower one
    assert ram_class_for(32) == 32
    assert ram_class_for(48) == 32
    assert ram_class_for(192) == 128
    assert ram_class_for(2048) == 1024  # above the top class -> the top class


def test_ram_class_for_floors_tiny_machines_at_16() -> None:
    assert ram_class_for(8) == 16


def test_shapes_for_ram_32gib_is_the_flagship_measured_reference() -> None:
    """32 GiB = this project's own measurement campaign: loss-layer at the flagship shape,
    single-op attention at 2048/4096/8192, train-step at 2048+8192 (spec §7 note)."""
    grid = shapes_for_ram(32)
    assert grid.ram_class_gib == 32
    assert grid.loss_n == (512, 2048, 8192)
    assert grid.attn_seq == (2048, 4096, 8192)
    assert grid.train_seq == (2048, 8192)
    assert grid.train_model == "mlx-community/Qwen3-8B-4bit"


def test_shapes_for_ram_16gib_shrinks_the_seq_grid() -> None:
    grid = shapes_for_ram(16)
    assert grid.ram_class_gib == 16
    assert grid.attn_seq == (2048, 4096)     # capped below the O(N^2) stock wall
    assert grid.train_seq == (2048,)


def test_shapes_for_ram_512gib_scales_the_grid_up() -> None:
    grid = shapes_for_ram(512)
    assert grid.ram_class_gib == 512
    assert grid.attn_seq == (2048, 4096, 8192, 16384, 32768, 65536)
    assert grid.train_seq == (2048, 8192, 16384, 32768)
    assert grid.context_start == 8192


def test_shapes_for_ram_is_monotone_nondecreasing_in_grid_size() -> None:
    """A bigger machine never runs a SMALLER attention seq grid than a smaller one."""
    classes = (16, 32, 64, 128, 256, 512, 1024)
    lengths = [len(shapes_for_ram(c).attn_seq) for c in classes]
    assert lengths == sorted(lengths)


def test_shapes_for_ram_holds_attention_head_config_constant() -> None:
    """Head geometry is the flagship's (32q/8kv, d128) at every class -- only the seq grid
    scales, so the single-op O(N) proof is comparable across hardware."""
    for c in (16, 32, 64, 128, 256, 512, 1024):
        grid = shapes_for_ram(c)
        assert (grid.attn_head_dim, grid.attn_heads, grid.attn_kv_heads) == (128, 32, 8)


# --- ETA computation from the tier table ----------------------------------------------


def test_benches_for_tier_quick_is_loss_and_attention_only() -> None:
    assert benches_for_tier("quick") == ("loss_layer", "attention_op")


def test_benches_for_tier_full_adds_train_step_and_context_probe() -> None:
    assert benches_for_tier("full") == (
        "loss_layer", "attention_op", "train_step", "context_probe",
    )


def test_benches_for_tier_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        benches_for_tier("bogus")


def test_eta_minutes_for_tier_quick_is_the_briefs_10_to_15_min() -> None:
    assert eta_minutes_for_tier("quick") == (10.0, 15.0)


def test_eta_minutes_for_tier_full_is_roughly_one_to_two_hours() -> None:
    low, high = eta_minutes_for_tier("full")
    assert low >= 60.0            # at least ~1 h
    assert high <= 150.0          # no more than ~2.5 h
    assert (low, high) == (70.0, 135.0)


def test_format_eta_mentions_the_tier_and_a_range() -> None:
    text = format_eta("quick")
    assert "quick" in text
    assert "10" in text
    assert "15" in text


# --- pre-flight decision (pure) -------------------------------------------------------


def test_classify_memory_pressure_reads_the_free_percentage_line() -> None:
    assert classify_memory_pressure("System-wide memory free percentage: 91%") == "normal"
    assert classify_memory_pressure("System-wide memory free percentage: 20%") == "warn"
    assert classify_memory_pressure("System-wide memory free percentage: 4%") == "red"


def test_classify_memory_pressure_degrades_to_normal_when_unparseable() -> None:
    """A missing free-percentage line must NOT read as red (the real panic guard is the
    effective-ceiling refusal, not this coarse gate) -- it degrades to normal."""
    assert classify_memory_pressure("garbage with no percentage line") == "normal"


def test_evaluate_preflight_refuses_on_red_memory() -> None:
    pf = evaluate_preflight(
        memory_pressure_state="red", on_ac_power=True,
        ceiling=EffectiveCeiling(ceiling_bytes=1, warning=None),
    )
    assert pf.ok is False
    assert pf.refusal is not None
    assert "memory" in pf.refusal.lower()


def test_evaluate_preflight_warns_on_battery_but_proceeds() -> None:
    pf = evaluate_preflight(
        memory_pressure_state="normal", on_ac_power=False,
        ceiling=EffectiveCeiling(ceiling_bytes=1, warning=None),
    )
    assert pf.ok is True
    assert pf.refusal is None
    assert any("AC" in w or "battery" in w.lower() for w in pf.warnings)


def test_evaluate_preflight_surfaces_the_divergence_warning_prominently() -> None:
    """The 0021 memory-divergence warning ('expected ~58 GB free, measured 20 GB') is
    exactly the kit's audience -- someone on a crowded machine must see it up front."""
    ceiling = EffectiveCeiling(
        ceiling_bytes=1, warning="measured available 20 GB is far below 58 GB",
    )
    pf = evaluate_preflight(memory_pressure_state="normal", on_ac_power=True, ceiling=ceiling)
    assert pf.ok is True
    assert any("20 GB" in w for w in pf.warnings)


# --- run_preflight: the standalone, pre-confirmation seam (finding A) -----------------


def test_run_preflight_runs_before_any_confirmation_is_required() -> None:
    """`run_preflight` is a self-contained seam the CLI calls BEFORE prompting for
    confirmation (finding A) -- it takes no `confirm` input at all, so there is no way
    to gate it behind a confirmation step."""
    pf = run_preflight(
        ceiling_reader=lambda: EffectiveCeiling(ceiling_bytes=1, warning=None),
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
    )
    assert pf == Preflight(ok=True, refusal=None, warnings=())


def test_run_preflight_surfaces_the_divergence_warning() -> None:
    pf = run_preflight(
        ceiling_reader=lambda: EffectiveCeiling(
            ceiling_bytes=1, warning="measured available 20 GB is far below 58 GB",
        ),
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
    )
    assert pf.ok is True
    assert any("20 GB" in w for w in pf.warnings)


def test_run_preflight_refuses_on_red_memory_without_raising() -> None:
    pf = run_preflight(
        ceiling_reader=lambda: EffectiveCeiling(ceiling_bytes=1, warning=None),
        memory_pressure_reader=lambda: "System-wide memory free percentage: 3%",
        ac_power_reader=lambda: True,
    )
    assert pf.ok is False
    assert pf.refusal is not None


def test_run_preflight_turns_a_too_crowded_memory_budget_error_into_a_clean_refusal() -> None:
    def _raises() -> EffectiveCeiling:
        raise MemoryBudgetError("machine too crowded to start safely")

    pf = run_preflight(
        ceiling_reader=_raises,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
    )
    assert pf.ok is False
    assert "crowded" in (pf.refusal or "")


def test_run_contribution_accepts_a_precomputed_preflight_and_skips_its_own_readers(
    tmp_path: Path,
) -> None:
    """When the caller (the CLI, per finding A) already ran `run_preflight` before the
    confirmation prompt, `run_contribution` must use that decision AS-IS rather than
    re-invoking the readers a second time."""
    reader_calls: list[str] = []

    def _spy_ceiling() -> EffectiveCeiling:
        reader_calls.append("ceiling")
        return EffectiveCeiling(ceiling_bytes=20 * 1024**3, warning=None)

    def _spy_mp() -> str:
        reader_calls.append("memory_pressure")
        return "System-wide memory free percentage: 91%"

    def _spy_ac() -> bool:
        reader_calls.append("ac_power")
        return True

    precomputed = Preflight(ok=True, refusal=None, warnings=("precomputed warning",))
    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        preflight=precomputed,
        ceiling_reader=_spy_ceiling, memory_pressure_reader=_spy_mp, ac_power_reader=_spy_ac,
        measure=lambda *a, **k: [_fake_artifact(tmp_path, "loss_layer_c0", "ok")],  # noqa: ARG005
        today=lambda: "2026-07-12",
    )
    assert reader_calls == []                          # the precomputed decision was reused
    assert result.warnings == ("precomputed warning",)


def test_run_contribution_refuses_using_a_precomputed_red_preflight_before_measuring(
    tmp_path: Path,
) -> None:
    called: list[str] = []

    def _spy_measure(bench: str, **_kw: object) -> list[Path]:
        called.append(bench)
        return []

    precomputed = Preflight(ok=False, refusal="system memory pressure is critical (red)",
                             warnings=())
    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        preflight=precomputed,
        measure=_spy_measure, today=lambda: "2026-07-12",
    )
    assert result.refused is True
    assert result.refusal == "system memory pressure is critical (red)"
    assert called == []


# --- community artifact schema --------------------------------------------------------


def _fake_artifact(tmp_path: Path, name: str, status: str, **fields: object) -> Path:
    ident = {
        "schema_version": 1, "mlx_version": "0.32.0", "machine": "arm64",
        "code_sha": "deadbeef", "session_id": "s1", "impl": "kernel", "n": 8192,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"identity": ident, "status": status, **fields}))
    return path


def test_summarize_artifact_file_embeds_identity_verbatim(tmp_path: Path) -> None:
    path = _fake_artifact(tmp_path, "loss_layer_n8192_kernel", "ok", marginal_peak_gb=0.0006,
                          wall_s=1.2)
    summary = summarize_artifact_file(path)
    assert summary["name"] == "loss_layer_n8192_kernel"
    assert summary["status"] == "ok"
    assert summary["identity"]["session_id"] == "s1"      # verbatim provenance block
    assert summary["result"]["marginal_peak_gb"] == 0.0006
    assert "identity" not in summary["result"]            # identity is not duplicated into result
    assert "status" not in summary["result"]


def test_summarize_artifact_file_missing_reads_as_error(tmp_path: Path) -> None:
    summary = summarize_artifact_file(tmp_path / "nope.json")
    assert summary["status"] == "error"


def _machine() -> MachineInfo:
    return MachineInfo(chip="Apple M1 Max", ram_gib=32, ram_bytes=34359738368,
                       macos="15.5", mlx_version="0.32.0", package_version="0.2.0")


def test_build_community_artifact_has_all_required_keys(tmp_path: Path) -> None:
    path = _fake_artifact(tmp_path, "loss_layer_n8192_kernel", "ok", marginal_peak_gb=0.0006)
    summaries = [{"bench": "loss_layer", "conditions": [summarize_artifact_file(path)]}]
    art = build_community_artifact(
        machine=_machine(), tier="quick", grid=shapes_for_ram(32),
        bench_summaries=summaries, generated_date="2026-07-12",
    )
    for key in ("schema_version", "generated_date", "tier", "machine", "shapes",
                "benches", "memory_warnings"):
        assert key in art
    assert art["schema_version"] == COMMUNITY_SCHEMA_VERSION
    assert art["machine"]["chip"] == "Apple M1 Max"
    assert art["machine"]["ram_gib"] == 32
    assert art["tier"] == "quick"
    assert art["benches"][0]["conditions"][0]["identity"]["session_id"] == "s1"


def test_build_community_artifact_propagates_memory_warnings(tmp_path: Path) -> None:
    """A `memory_warning` recorded by an underlying bench artifact (0021 divergence, or a
    degraded vm_stat start) must ride into the community artifact -- provenance the
    maintainer needs to weight a crowded-machine submission."""
    path = _fake_artifact(tmp_path, "attn_flash_n8192", "ok", fwdbwd_peak_gb=4.0,
                          memory_warning="measured available 20 GB is far below 58 GB")
    summaries = [{"bench": "attention_op", "conditions": [summarize_artifact_file(path)]}]
    art = build_community_artifact(
        machine=_machine(), tier="quick", grid=shapes_for_ram(32),
        bench_summaries=summaries, generated_date="2026-07-12",
    )
    assert art["memory_warnings"] == ["measured available 20 GB is far below 58 GB"]


def test_collect_memory_warnings_dedupes_and_skips_clean(tmp_path: Path) -> None:
    clean = summarize_artifact_file(_fake_artifact(tmp_path, "a", "ok", wall_s=1.0))
    warned = summarize_artifact_file(
        _fake_artifact(tmp_path, "b", "ok", memory_warning="crowded")
    )
    warned2 = summarize_artifact_file(
        _fake_artifact(tmp_path, "c", "ok", memory_warning="crowded")
    )
    summaries = [{"bench": "x", "conditions": [clean, warned, warned2]}]
    assert collect_memory_warnings(summaries) == ["crowded"]


# --- PR text --------------------------------------------------------------------------


def test_pr_title_names_the_machine() -> None:
    title = pr_title(_machine())
    assert "Apple M1 Max" in title
    assert "32" in title


def test_pr_body_references_the_artifact_and_says_no_number_editing() -> None:
    body = pr_body(_machine(), artifact_filename="apple-m1-max-32gb-2026-07-12.json",
                   tier="quick")
    assert "apple-m1-max-32gb-2026-07-12.json" in body
    assert "community-measured" in body.lower() or "measured on" in body.lower()


# --- run_contribution orchestration (measurement seam stubbed) ------------------------


def _healthy_ceiling() -> EffectiveCeiling:
    return EffectiveCeiling(ceiling_bytes=20 * 1024**3, warning=None)


def test_run_contribution_refuses_when_memory_is_red(tmp_path: Path) -> None:
    """A red pre-flight refuses BEFORE any measurement -- the measurement seam is never
    called and no community artifact is written."""
    called: list[str] = []

    def _spy_measure(bench: str, **_kw: object) -> list[Path]:
        called.append(bench)
        return []

    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 3%",
        ac_power_reader=lambda: True,
        measure=_spy_measure,
        today=lambda: "2026-07-12",
    )
    assert result.refused is True
    assert result.refusal is not None
    assert called == []                              # heavy work never started
    assert result.artifact_path is None
    assert list(tmp_path.glob("*.json")) == []


def test_run_contribution_refuses_when_machine_is_too_crowded(tmp_path: Path) -> None:
    """`effective_memory_ceiling` raising `MemoryBudgetError` (the 0021 too-crowded
    refusal) becomes a clean kit refusal, not a crash."""
    def _raises() -> EffectiveCeiling:
        raise MemoryBudgetError("machine too crowded to start safely")

    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_raises,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
        measure=lambda *a, **k: [],  # noqa: ARG005
        today=lambda: "2026-07-12",
    )
    assert result.refused is True
    assert "crowded" in (result.refusal or "")


def test_run_contribution_happy_path_writes_a_provenance_complete_artifact(
    tmp_path: Path,
) -> None:
    def _fake_measure(bench: str, *, out_dir: Path, **_kw: object) -> list[Path]:
        p = _fake_artifact(out_dir, f"{bench}_c0", "ok", marginal_peak_gb=0.0006, wall_s=1.2)
        return [p]

    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
        measure=_fake_measure,
        today=lambda: "2026-07-12",
    )
    assert result.refused is False
    assert result.artifact_path is not None
    assert result.artifact_path.name == "apple-m1-max-32gb-2026-07-12.json"
    art = json.loads(result.artifact_path.read_text())
    assert art["schema_version"] == COMMUNITY_SCHEMA_VERSION
    assert art["tier"] == "quick"
    assert {b["bench"] for b in art["benches"]} == {"loss_layer", "attention_op"}
    assert result.pr_title is not None
    assert "Apple M1 Max" in result.pr_title
    assert result.pr_body is not None


def test_run_contribution_carries_a_battery_warning_without_refusing(tmp_path: Path) -> None:
    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: False,           # on battery
        measure=lambda *a, **k: [_fake_artifact(tmp_path, "loss_layer_c0", "ok")],  # noqa: ARG005
        today=lambda: "2026-07-12",
    )
    assert result.refused is False
    assert any("AC" in w or "battery" in w.lower() for w in result.warnings)


def test_run_contribution_refuses_when_not_confirmed(tmp_path: Path) -> None:
    """`confirm=False` (no --yes, no TTY confirmation) refuses before any measurement."""
    called: list[str] = []
    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=False, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
        measure=lambda bench, **_kw: called.append(bench) or [],  # type: ignore[func-returns-value]
        today=lambda: "2026-07-12",
    )
    assert result.refused is True
    assert called == []
    assert result.artifact_path is None


def test_run_contribution_prints_kit_level_progress_per_bench_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Review round 2, finding 2: a full-tier run sat in total silence between the
    confirmation prompt and the final 'wrote ...' line -- the attention_op/context_probe
    arm regressed from live inherited subprocess output to fully-captured-and-discarded.
    `run_contribution` must print a kit-level progress line naming each bench and its
    position BEFORE calling the measurement seam, and a completion line (status +
    elapsed) AFTER -- this is the kit's own liveness signal, independent of whatever the
    (here faked) measurement seam does or doesn't print itself."""
    def _fake_measure(bench: str, *, out_dir: Path, **_kw: object) -> list[Path]:
        return [_fake_artifact(out_dir, f"{bench}_c0", "ok")]

    run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
        measure=_fake_measure,
        today=lambda: "2026-07-12",
    )
    out = capsys.readouterr().out
    loss_running = out.index("[1/2] loss_layer")
    loss_done = out.index("done", loss_running)
    attn_running = out.index("[2/2] attention_op", loss_done)
    attn_done = out.index("done", attn_running)
    assert loss_running < loss_done < attn_running < attn_done


def test_run_contribution_returns_a_result_dataclass(tmp_path: Path) -> None:
    result = run_contribution(
        tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
        ceiling_reader=_healthy_ceiling,
        memory_pressure_reader=lambda: "System-wide memory free percentage: 91%",
        ac_power_reader=lambda: True,
        measure=lambda *a, **k: [_fake_artifact(tmp_path, "loss_layer_c0", "ok")],  # noqa: ARG005
        today=lambda: "2026-07-12",
    )
    assert isinstance(result, ContributionResult)


# --- measurement seam internals (still GPU-free / model-free) --------------------------


def test_loss_conditions_reproduce_the_flagship_grid() -> None:
    conditions = contribute._loss_conditions(shapes_for_ram(32))
    assert [c.name for c in conditions] == [
        "loss_layer_n512_kernel", "loss_layer_n512_chunked", "loss_layer_n512_naive",
        "loss_layer_n2048_kernel", "loss_layer_n2048_chunked", "loss_layer_n2048_naive",
        "loss_layer_n8192_kernel", "loss_layer_n8192_chunked", "loss_layer_n8192_naive",
    ]
    assert all(c.kind == "loss_layer" for c in conditions)
    assert conditions[0].params["v"] == 151936


def test_train_conditions_are_ours_arm_flash_only() -> None:
    conditions = contribute._train_conditions(shapes_for_ram(32))
    assert [c.name for c in conditions] == [
        "train_step_seq2048_ours", "train_step_seq8192_ours",
    ]
    for c in conditions:
        assert c.attention_impl == "flash"        # the library's own path
        assert c.params["stock"] is False         # ours arm only
        assert c.params["grad_checkpoint"] is True
        assert c.params["model"] == "mlx-community/Qwen3-8B-4bit"


def test_measure_bench_dispatches_each_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        contribute, "run_conditions",
        lambda conds, out, *, session_id: calls.append("run_conditions") or [],  # noqa: ARG005
    )
    monkeypatch.setattr(
        contribute, "_spawn_attention",
        lambda grid, *, out_dir, session_id: calls.append("attention") or [],  # noqa: ARG005
    )
    monkeypatch.setattr(
        contribute, "_spawn_context",
        lambda grid, *, out_dir: calls.append("context") or [],  # noqa: ARG005
    )
    grid = shapes_for_ram(32)
    for bench in ("loss_layer", "train_step", "attention_op", "context_probe"):
        contribute._measure_bench(bench, grid=grid, out_dir=tmp_path, session_id="s1",
                                  machine=_machine())
    assert calls == ["run_conditions", "run_conditions", "attention", "context"]


def test_measure_bench_rejects_unknown_bench(tmp_path: Path) -> None:
    with pytest.raises(BenchInputError):
        contribute._measure_bench("bogus", grid=shapes_for_ram(32), out_dir=tmp_path,
                                  session_id="s1", machine=_machine())


def test_bench_scripts_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLX_TRAIN_PERF_SCRIPTS_DIR", str(tmp_path))
    assert contribute._bench_scripts_dir() == tmp_path


def test_bench_scripts_dir_finds_the_repo_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLX_TRAIN_PERF_SCRIPTS_DIR", raising=False)
    scripts = contribute._bench_scripts_dir()
    assert (scripts / "bench_attention_op.py").exists()


def test_spawn_script_runs_a_subprocess_and_globs_the_output(tmp_path: Path) -> None:
    """`_spawn_script` shells out and returns the artifacts written to `out_dir`. Driven
    here with a trivial inline python (no GPU, no model) that writes one JSON file."""
    code = f"import pathlib, json; (pathlib.Path({str(tmp_path)!r})/'c0.json')" \
           f".write_text(json.dumps({{'status': 'ok'}}))"
    paths = contribute._spawn_script(["-c", code], out_dir=tmp_path)
    assert [p.name for p in paths] == ["c0.json"]


def test_spawn_script_records_a_crash_envelope_when_rc_nonzero_and_nothing_was_written(
    tmp_path: Path,
) -> None:
    """Finding B: a script that crashes before writing ANY artifact must not glob-read
    as a clean, empty bench -- `_spawn_script` mirrors `bench.runner.run_conditions`'s
    `WorkerCrashed` envelope (status "error", error_type, a stderr tail) instead."""
    code = "import sys; print('boom explanation', file=sys.stderr); sys.exit(3)"
    paths = contribute._spawn_script(["-c", code], out_dir=tmp_path)
    assert len(paths) == 1
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerCrashed"
    assert "boom explanation" in data["error_msg"]
    assert data["returncode"] == 3


def test_spawn_script_respects_a_partial_artifact_the_crashed_script_already_wrote(
    tmp_path: Path,
) -> None:
    """A script that wrote its OWN honest partial record before crashing (e.g. a
    condition-level watchdog breach) must be respected, not clobbered by a synthetic
    crash envelope -- same reasoning as `bench.runner.run_conditions`'s own crash path."""
    code = (
        f"import pathlib, json, sys; "
        f"(pathlib.Path({str(tmp_path)!r})/'c0.json').write_text(json.dumps("
        f"{{'status': 'aborted_memory_ceiling'}})); sys.exit(70)"
    )
    paths = contribute._spawn_script(["-c", code], out_dir=tmp_path)
    assert [p.name for p in paths] == ["c0.json"]
    data = json.loads(paths[0].read_text())
    assert data["status"] == "aborted_memory_ceiling"    # untouched -- not overwritten


def test_spawn_script_clean_exit_with_no_output_is_not_a_crash(tmp_path: Path) -> None:
    """A script that exits 0 but legitimately writes nothing (an empty grid) is not the
    crash case this finding targets -- no synthetic error artifact is invented."""
    paths = contribute._spawn_script(["-c", "pass"], out_dir=tmp_path)
    assert paths == []


def test_spawn_script_supersedes_a_stale_crash_marker_on_a_successful_retry(
    tmp_path: Path,
) -> None:
    """Review round 2, finding 1: `_spawn_crash_artifact` writes `out_dir/
    _spawn_crashed.json` with a FIXED name. Because the kit's session id is now
    deterministic (finding D), the SAME out_dir is reused across invocations for the
    same recipe -- a transient crash leaves that fixed-name marker on disk forever, and
    `_spawn_script`'s glob picks it up on every later call even after a clean retry.
    Reproduced live: crash -> `['_spawn_crashed.json']`; then a successful retry against
    the same dir -> `['_spawn_crashed.json', 'attn_seq2048_flash.json']` (the phantom
    error rides along with the real result). `_spawn_script` must unlink any
    pre-existing crash marker BEFORE respawning (mirrors bench/runner.py:69-74's
    remove-stale-artifact-before-respawn reasoning)."""
    crash_code = "import sys; print('boom explanation', file=sys.stderr); sys.exit(3)"
    contribute._spawn_script(["-c", crash_code], out_dir=tmp_path)
    assert (tmp_path / "_spawn_crashed.json").exists()  # sanity: the crash was recorded

    success_code = (
        f"import pathlib, json; (pathlib.Path({str(tmp_path)!r})/"
        f"'attn_seq2048_flash.json').write_text(json.dumps({{'status': 'ok'}}))"
    )
    paths = contribute._spawn_script(["-c", success_code], out_dir=tmp_path)

    assert [p.name for p in paths] == ["attn_seq2048_flash.json"]  # no phantom entry
    assert not (tmp_path / "_spawn_crashed.json").exists()  # superseded, not lingering

    summary = contribute.summarize_bench("attention_op", paths)
    assert [c["name"] for c in summary["conditions"]] == ["attn_seq2048_flash"]
    assert all(c["status"] != "error" for c in summary["conditions"])


def test_spawn_script_crash_marker_still_present_with_no_retry(tmp_path: Path) -> None:
    """The honest-crash behavior (finding B, round 1) must survive the round-2 fix: a
    single crash with no follow-up call still leaves the crash envelope on disk and in
    the returned paths -- unlinking a stale marker before respawning must not also erase
    a marker THIS call itself just wrote."""
    crash_code = "import sys; print('boom explanation', file=sys.stderr); sys.exit(3)"
    paths = contribute._spawn_script(["-c", crash_code], out_dir=tmp_path)
    assert [p.name for p in paths] == ["_spawn_crashed.json"]
    data = json.loads(paths[0].read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "WorkerCrashed"


# --- _contribute_session_id: deterministic, resumable across invocations (finding D) ---


def test_contribute_session_id_is_stable_for_the_same_recipe() -> None:
    grid = shapes_for_ram(32)
    id_a = contribute._contribute_session_id(machine=_machine(), tier="quick", grid=grid)
    id_b = contribute._contribute_session_id(machine=_machine(), tier="quick", grid=grid)
    assert id_a == id_b


def test_contribute_session_id_changes_with_tier() -> None:
    grid = shapes_for_ram(32)
    id_quick = contribute._contribute_session_id(machine=_machine(), tier="quick", grid=grid)
    id_full = contribute._contribute_session_id(machine=_machine(), tier="full", grid=grid)
    assert id_quick != id_full


def test_contribute_session_id_changes_with_the_shape_grid() -> None:
    id_32 = contribute._contribute_session_id(
        machine=_machine(), tier="quick", grid=shapes_for_ram(32),
    )
    id_16 = contribute._contribute_session_id(
        machine=_machine(), tier="quick", grid=shapes_for_ram(16),
    )
    assert id_32 != id_16


def test_contribute_session_id_changes_with_the_machine() -> None:
    other_machine = MachineInfo(chip="Apple M2 Ultra", ram_gib=32, ram_bytes=34359738368,
                                macos="15.5", mlx_version="0.32.0", package_version="0.2.0")
    grid = shapes_for_ram(32)
    id_a = contribute._contribute_session_id(machine=_machine(), tier="quick", grid=grid)
    id_b = contribute._contribute_session_id(machine=other_machine, tier="quick", grid=grid)
    assert id_a != id_b


def _fake_artifact_with_session(
    out_dir: Path, name: str, session_id: str, status: str, **fields: object,
) -> Path:
    ident = {
        "schema_version": 1, "mlx_version": "0.32.0", "machine": "arm64",
        "code_sha": "deadbeef", "session_id": session_id, "impl": "kernel", "n": 8192,
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"identity": ident, "status": status, **fields}))
    return path


def test_run_contribution_reuses_the_session_id_and_resumes_across_invocations(
    tmp_path: Path,
) -> None:
    """Finding D: two `run_contribution` calls with the SAME (machine, tier, grid) must
    get the SAME session id, and a `measure` seam that resume-skips fresh artifacts
    (exactly what `bench.runner.run_conditions` does, gated on `identity.session_id`)
    must spawn nothing the second time."""
    session_ids: list[str] = []
    spawned: list[str] = []

    def _resuming_measure(
        bench: str, *, grid: object, out_dir: Path, session_id: str, machine: object,  # noqa: ARG001
    ) -> list[Path]:
        session_ids.append(session_id)
        out_path = out_dir / f"{bench}_c0.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing["identity"]["session_id"] == session_id:
                return [out_path]              # fresh -- resume-skip, no "spawn"
        spawned.append(bench)
        return [_fake_artifact_with_session(out_dir, f"{bench}_c0", session_id, "ok",
                                            marginal_peak_gb=0.0006, wall_s=1.2)]

    def _run() -> ContributionResult:
        return run_contribution(
            tier="quick", out_dir=tmp_path, confirm=True, machine=_machine(),
            preflight=Preflight(ok=True, refusal=None, warnings=()),
            measure=_resuming_measure, today=lambda: "2026-07-12",
        )

    _run()
    first_ids, first_spawned = list(session_ids), list(spawned)
    session_ids.clear()
    spawned.clear()

    _run()                                                 # SAME recipe, second invocation
    assert session_ids == first_ids                        # stable session id
    assert first_spawned != []                              # first run actually measured
    assert spawned == []                                    # second run resumed, spawned nothing


def test_real_non_metal_readers_return_plausible_values() -> None:
    """The subprocess/platform readers (chip, macOS, memory_pressure, AC power, date,
    package version) run on any macOS without a Metal device -- exercised for real, unlike
    the Metal `mx.device_info()` RAM reader (pragma'd)."""
    assert contribute._read_chip()                       # non-empty sysctl brand string
    assert isinstance(contribute._read_on_ac_power(), bool)
    assert "percentage" in contribute._read_memory_pressure().lower()
    assert contribute._read_package_version()
    assert len(contribute._today()) == 10                # YYYY-MM-DD
    assert isinstance(contribute._read_macos(), str)


# --- _read_chip: subprocess failures map to the typed tool-error path (finding E) -----


def test_read_chip_wraps_a_called_process_error_in_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing `sysctl` (nonzero exit under `check=True`) must not escape as a raw
    `CalledProcessError` -- that traceback would bypass `main`'s `MlxTrainPerfError`
    catch and exit 1 (an uncaught crash) instead of the package's tool-error exit 2."""
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["sysctl", "-n", "machdep.cpu.brand_string"])

    monkeypatch.setattr(contribute.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError, match="sysctl"):
        contribute._read_chip()


def test_read_chip_wraps_a_timeout_in_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["sysctl"], timeout=10)

    monkeypatch.setattr(contribute.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError):
        contribute._read_chip()


def test_read_chip_wraps_a_missing_binary_in_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(contribute.subprocess, "run", _raise)
    with pytest.raises(MachineDetectionError):
        contribute._read_chip()


# --- controller's Step 2: the real quick-tier run (gated, NOT run in this suite) -------


@pytest.mark.smoke
def test_contribute_quick_tier_end_to_end(tmp_path: Path) -> None:
    """Step 2 (controller only): the real quick-tier run on THIS machine -- a HEAVY GPU
    job (~10-15 min: loss-layer + single-op attention). Collected but skipped by default;
    the controller runs it with `--run-smoke` after this task's review, writing the first
    community-benchmark row. NEVER executed from an agent session."""
    result = run_contribution(tier="quick", out_dir=tmp_path, confirm=True)
    assert result.refused is False
    assert result.artifact_path is not None
    art = json.loads(result.artifact_path.read_text())
    assert art["schema_version"] == COMMUNITY_SCHEMA_VERSION
    assert {b["bench"] for b in art["benches"]} == {"loss_layer", "attention_op"}
