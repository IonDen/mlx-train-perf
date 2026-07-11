"""Pure-logic tests for `scripts/aggregate_community.py` -- folds submitted
`community-benchmarks/*.json` artifacts into the README's community table. Every test is
GPU-free and reads only synthetic artifact dicts; the exact-markdown assertions live
HERE, never against a real measurement.

`scripts/` has no `__init__.py` (matches `bench_attention_op.py`'s convention), so the
module is loaded by path.
"""
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from aggregate_community import (  # noqa: E402
    CommunityRow,
    load_community_artifacts,
    main,
    render_markdown_table,
    summarize_row,
)


def _artifact(
    *, chip: str = "Apple M1 Max", ram_gib: int = 32, mlx: str = "0.32.0",
    tier: str = "quick", loss_peak: float | None = 0.0006,
    flash_peaks: dict[int, float] | None = None,
    train_tps: float | None = None, pr: str | None = None,
) -> dict[str, object]:
    benches: list[dict[str, object]] = []
    if loss_peak is not None:
        benches.append({"bench": "loss_layer", "conditions": [
            {"name": "loss_layer_n8192_kernel", "status": "ok",
             "identity": {"impl": "kernel", "n": 8192},
             "result": {"marginal_peak_gb": loss_peak}},
            {"name": "loss_layer_n8192_naive", "status": "ok",
             "identity": {"impl": "naive", "n": 8192},
             "result": {"marginal_peak_gb": 2.318}},
        ]})
    if flash_peaks is not None:
        conditions = [
            {"name": f"flash_n{n}", "status": "ok",
             "identity": {"attention_impl": "flash", "n": n},
             "result": {"impl": "flash", "n": n, "fwdbwd_peak_gb": peak}}
            for n, peak in sorted(flash_peaks.items())
        ]
        benches.append({"bench": "attention_op", "conditions": conditions})
    if train_tps is not None:
        benches.append({"bench": "train_step", "conditions": [
            {"name": "train_step_seq2048_ours", "status": "ok",
             "identity": {"stock": False, "seq_len": 2048, "attention_impl": "flash"},
             "result": {"tokens_per_sec_median": train_tps}},
        ]})
    art: dict[str, object] = {
        "schema_version": 1, "tier": tier,
        "machine": {"chip": chip, "ram_gib": ram_gib, "mlx_version": mlx},
        "benches": benches,
    }
    if pr is not None:
        art["pr"] = pr
    return art


# --- summarize_row: pure extraction ---------------------------------------------------


def test_summarize_row_pulls_machine_and_tier() -> None:
    row = summarize_row(_artifact(chip="Apple M2 Ultra", ram_gib=192, mlx="0.32.0"))
    assert row.chip == "Apple M2 Ultra"
    assert row.ram_gib == 192
    assert row.mlx == "0.32.0"
    assert row.tier == "quick"


def test_summarize_row_reads_loss_kernel_peak_at_max_n() -> None:
    row = summarize_row(_artifact(loss_peak=0.0006))
    assert row.loss_kernel_peak_gb == 0.0006


def test_summarize_row_computes_flash_doubling_at_the_top_pair() -> None:
    """The O(N) proof: flash fwd+bwd peak roughly doubles per seq doubling. The row
    reports the ratio at the largest consecutive pair present."""
    row = summarize_row(_artifact(flash_peaks={2048: 1.0, 4096: 2.0, 8192: 4.04}))
    assert row.attn_flash_doubling == 2.02          # 4.04 / 2.0, the top (4096->8192) pair


def test_summarize_row_flash_doubling_is_none_with_one_point() -> None:
    row = summarize_row(_artifact(flash_peaks={2048: 1.0}))
    assert row.attn_flash_doubling is None


def test_summarize_row_reads_train_tps_when_present() -> None:
    row = summarize_row(_artifact(tier="full", train_tps=91.4))
    assert row.train_tps_flash == 91.4


def test_summarize_row_train_tps_is_none_for_quick_tier() -> None:
    row = summarize_row(_artifact(tier="quick"))
    assert row.train_tps_flash is None


def test_summarize_row_reads_optional_pr_field() -> None:
    assert summarize_row(_artifact(pr="#42")).pr == "#42"
    assert summarize_row(_artifact()).pr is None


# --- render_markdown_table: exact output ----------------------------------------------


def test_render_markdown_table_exact_for_one_full_row() -> None:
    row = CommunityRow(
        chip="Apple M1 Max", ram_gib=32, mlx="0.32.0", tier="full",
        loss_kernel_peak_gb=0.0006, attn_flash_doubling=2.02, train_tps_flash=91.4, pr="#42",
    )
    expected = (
        "| Chip | RAM (GB) | mlx | Tier | Loss kernel peak (GB) | Attn flash 2x ratio | "
        "Train tok/s (flash) | PR |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| Apple M1 Max | 32 | 0.32.0 | full | 0.0006 | 2.02 | 91.4 | #42 |"
    )
    assert render_markdown_table([row]) == expected


def test_render_markdown_table_renders_missing_values_as_em_dash() -> None:
    row = CommunityRow(
        chip="Apple M4", ram_gib=16, mlx="0.32.0", tier="quick",
        loss_kernel_peak_gb=0.0006, attn_flash_doubling=None, train_tps_flash=None, pr=None,
    )
    table = render_markdown_table([row])
    last_line = table.splitlines()[-1]
    assert last_line == "| Apple M4 | 16 | 0.32.0 | quick | 0.0006 | — | — | — |"


def test_render_markdown_table_sorts_rows_by_ram_then_chip() -> None:
    rows = [
        CommunityRow(chip="B", ram_gib=64, mlx="0.32.0", tier="quick",
                     loss_kernel_peak_gb=0.0006, attn_flash_doubling=None,
                     train_tps_flash=None, pr=None),
        CommunityRow(chip="A", ram_gib=32, mlx="0.32.0", tier="quick",
                     loss_kernel_peak_gb=0.0006, attn_flash_doubling=None,
                     train_tps_flash=None, pr=None),
    ]
    body = render_markdown_table(rows).splitlines()[2:]
    assert body[0].startswith("| A | 32")
    assert body[1].startswith("| B | 64")


# --- load + main ----------------------------------------------------------------------


def test_load_community_artifacts_skips_the_readme_and_bad_json(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(_artifact()))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "README.md").write_text("# submission how-to")
    loaded = load_community_artifacts(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["machine"]["chip"] == "Apple M1 Max"


def test_main_prints_the_table(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.json").write_text(json.dumps(_artifact(chip="Apple M1 Max", ram_gib=32)))
    rc = main(["--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| Chip | RAM (GB) | mlx |" in out
    assert "Apple M1 Max" in out


def test_main_empty_dir_exits_zero_with_a_note(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["--dir", str(tmp_path)])
    assert rc == 0
    assert "no community" in capsys.readouterr().out.lower()
