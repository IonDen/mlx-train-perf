"""Fold submitted `community-benchmarks/*.json` artifacts into the README's community
table (backlog 0015, spec §7). Pure extraction + exact-markdown rendering -- the
maintainer runs this after merging a contributor's artifact PR and pastes the table into
the README's community section.

Honesty convention preserved: every row is one contributor's own measured numbers with an
optional PR reference; nothing is extrapolated or hand-edited. The kit
(`mlx_train_perf.contribute`) writes the artifacts; this script only reads them.

`scripts/` has no `__init__.py` (matches `bench_attention_op.py`'s convention), so this is
run by path: `python scripts/aggregate_community.py --dir community-benchmarks`.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_EM_DASH = "—"
_COLUMNS = (
    "Chip", "RAM (GB)", "mlx", "Tier", "Loss kernel peak (GB)", "Attn flash 2x ratio",
    "Train tok/s (flash)", "PR",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CommunityRow:
    chip: str
    ram_gib: int
    mlx: str
    tier: str
    loss_kernel_peak_gb: float | None
    attn_flash_doubling: float | None
    train_tps_flash: float | None
    pr: str | None


def _bench(artifact: dict[str, object], name: str) -> dict[str, object] | None:
    for bench in cast(list[dict[str, object]], artifact.get("benches", [])):
        if bench.get("bench") == name:
            return bench
    return None


def _conditions(bench: dict[str, object] | None) -> list[dict[str, object]]:
    if bench is None:
        return []
    return cast(list[dict[str, object]], bench.get("conditions", []))


def _field(condition: dict[str, object], key: str) -> object:
    """A condition's value for `key`, checked in `result` (measured numbers) first, then
    `identity` (grid-point provenance) -- so `impl`/`n` are found whether the underlying
    bench recorded them as top-level result fields or only in the identity block."""
    result = cast(dict[str, object], condition.get("result", {}))
    if key in result:
        return result[key]
    identity = cast(dict[str, object], condition.get("identity", {}))
    return identity.get(key)


def _loss_kernel_peak(artifact: dict[str, object]) -> float | None:
    """Kernel loss-layer marginal peak at the largest `n` -- the near-zero memory number."""
    best_n = -1
    best_peak: float | None = None
    for condition in _conditions(_bench(artifact, "loss_layer")):
        if condition.get("status") != "ok" or _field(condition, "impl") != "kernel":
            continue
        n = _field(condition, "n")
        peak = _field(condition, "marginal_peak_gb")
        if isinstance(n, int) and isinstance(peak, int | float) and n > best_n:
            best_n, best_peak = n, float(peak)
    return best_peak


def _flash_doubling(artifact: dict[str, object]) -> float | None:
    """Flash fwd+bwd peak doubling ratio at the largest consecutive (n, 2n) pair present --
    the O(N) proof (~2.0 confirms linear-in-N saved state)."""
    peaks: dict[int, float] = {}
    for condition in _conditions(_bench(artifact, "attention_op")):
        if condition.get("status") != "ok" or _field(condition, "impl") != "flash":
            continue
        n = _field(condition, "n")
        peak = _field(condition, "fwdbwd_peak_gb")
        if isinstance(n, int) and isinstance(peak, int | float) and peak > 0:
            peaks[n] = float(peak)
    pairs = [(n, peaks[n] / peaks[n // 2]) for n in peaks if n // 2 in peaks]
    if not pairs:
        return None
    _, ratio = max(pairs)  # the largest-n pair
    return round(ratio, 2)


def _train_tps_flash(artifact: dict[str, object]) -> float | None:
    """Ours-arm (flash) tokens/sec at the smallest measured train-step seq."""
    best_seq = None
    best_tps: float | None = None
    for condition in _conditions(_bench(artifact, "train_step")):
        if condition.get("status") != "ok" or _field(condition, "stock") is True:
            continue
        seq = _field(condition, "seq_len")
        tps = _field(condition, "tokens_per_sec_median")
        if isinstance(seq, int) and isinstance(tps, int | float) and (
            best_seq is None or seq < best_seq
        ):
            best_seq, best_tps = seq, float(tps)
    return best_tps


def summarize_row(artifact: dict[str, object]) -> CommunityRow:
    machine = cast(dict[str, object], artifact.get("machine", {}))
    pr = artifact.get("pr")
    return CommunityRow(
        chip=str(machine.get("chip", "?")),
        ram_gib=int(cast(int, machine.get("ram_gib", 0))),
        mlx=str(machine.get("mlx_version", "?")),
        tier=str(artifact.get("tier", "?")),
        loss_kernel_peak_gb=_loss_kernel_peak(artifact),
        attn_flash_doubling=_flash_doubling(artifact),
        train_tps_flash=_train_tps_flash(artifact),
        pr=str(pr) if isinstance(pr, str) else None,
    )


def _cell_float(value: float | None, fmt: str) -> str:
    return _EM_DASH if value is None else format(value, fmt)


def _row_cells(row: CommunityRow) -> tuple[str, ...]:
    return (
        row.chip,
        str(row.ram_gib),
        row.mlx,
        row.tier,
        _cell_float(row.loss_kernel_peak_gb, ".4f"),
        _cell_float(row.attn_flash_doubling, ".2f"),
        _cell_float(row.train_tps_flash, ".1f"),
        row.pr if row.pr is not None else _EM_DASH,
    )


def render_markdown_table(rows: list[CommunityRow]) -> str:
    """A GitHub-flavored markdown table, rows sorted by RAM then chip -- deterministic so
    the README diff is stable across re-runs."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    ordered = sorted(rows, key=lambda r: (r.ram_gib, r.chip))
    body = ["| " + " | ".join(_row_cells(row)) + " |" for row in ordered]
    return "\n".join([header, separator, *body])


def _artifact_is_complete(data: dict[str, object]) -> bool:
    """A well-formed community artifact carries a `machine` block with a real `chip` and
    an int `ram_gib` (what `summarize_row` needs to avoid the "?" chip / "0" RAM
    degenerate row) and a `benches` list. Either missing/malformed is a partial artifact
    -- caught here so `load_community_artifacts` can skip it with a named warning
    instead of silently rendering the degenerate row."""
    machine = data.get("machine")
    if not isinstance(machine, dict):
        return False
    if not machine.get("chip") or not isinstance(machine.get("ram_gib"), int):
        return False
    return isinstance(data.get("benches"), list)


def load_community_artifacts(directory: Path) -> list[dict[str, object]]:
    """Every well-formed `*.json` artifact in `directory`. A corrupt/unparseable file, or
    a valid-but-partial one (missing/malformed `machine` or `benches`), is skipped with a
    warning NAMING the file printed to stderr -- never silently, and never folded into a
    degenerate `?` chip / `0` RAM table row. The submission `README.md` is not `*.json`,
    so it is naturally ignored."""
    artifacts: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: skipping unparseable community artifact {path}: {exc}",
                  file=sys.stderr)
            continue
        if not isinstance(data, dict) or not _artifact_is_complete(data):
            print(
                f"warning: skipping partial/malformed community artifact {path} "
                "(missing or incomplete 'machine'/'benches')",
                file=sys.stderr,
            )
            continue
        artifacts.append(cast(dict[str, object], data))
    return artifacts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="community-benchmarks",
                    help="directory of submitted community artifacts")
    args = ap.parse_args(argv)

    artifacts = load_community_artifacts(Path(args.dir))
    if not artifacts:
        print(f"no community benchmark artifacts found in {args.dir!r}")
        return 0
    rows = [summarize_row(a) for a in artifacts]
    print(render_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
