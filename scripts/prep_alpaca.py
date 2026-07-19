"""Prepare the Alpaca dataset for the packed-training bench (scripts/bench_packed_training.py).

Downloads `tatsu-lab/alpaca` at a PINNED commit via the `hf` CLI, applies the target
model's chat template, tokenizes each example into an `(tokens, offset)` pair -- `offset`
is the prompt length in tokens, so the trainer supervises only the assistant completion
(the mlx-lm SFT convention) -- and writes a jsonl (one `{"tokens": [...], "offset": N}`
object per line) plus a sibling `<out>.stats.json` describing the length distribution and
the EXACT padding-waste fraction of stock batching at the bench batch size.

Deterministic given the pinned revision + the model tokenizer: the same inputs always
produce the same jsonl. Heavy dependencies (`huggingface_hub`/`hf`, `pyarrow`,
`transformers`) are imported lazily inside the functions that need them, so `--help` and
the pure parsing/stats helpers stay importable (and unit-testable) without them.

Run on the main session (T14), not an agent: the download touches the network and the
tokenize pass loads a real tokenizer.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlx_train_perf.data.packing import (
    length_histogram,
    packed_batching_stats,
    stock_batching_stats,
)
from mlx_train_perf.errors import MissingDependencyError

# The Alpaca dataset repo + a RESOLVED commit sha (not a branch): a branch name would
# make the prep non-deterministic. Resolved once via the Hub API (2026-07-19).
ALPACA_REPO = "tatsu-lab/alpaca"
ALPACA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"


# ---------------------------------------------------------------------------
# Pure: chat-template + tokenize (no network, no heavy deps -- a tokenizer object
# exposing `apply_chat_template` is the only input)
# ---------------------------------------------------------------------------


def user_content(record: dict[str, str]) -> str:
    """The Alpaca user turn: the instruction, plus the optional `input` context joined
    with a blank line when present (the standard Alpaca prompt composition)."""
    instruction = record["instruction"]
    extra = record.get("input", "")
    return f"{instruction}\n\n{extra}" if extra else instruction


def tokenize_record(tokenizer: Any, record: dict[str, str]) -> tuple[list[int], int]:
    """Tokenize one Alpaca record into `(tokens, offset)`: `tokens` is the full
    user+assistant conversation under the model chat template; `offset` is the length of
    the prompt-only tokenization (user turn + generation prompt), i.e. the number of
    leading tokens the trainer's mask leaves unsupervised."""
    content = user_content(record)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": record["output"]},
        ],
        add_generation_prompt=False, tokenize=True, return_dict=False,
    )
    return [int(t) for t in full_ids], len(prompt_ids)


def records_to_pairs(
    tokenizer: Any, records: list[dict[str, str]], *, max_samples: int | None = None
) -> list[tuple[list[int], int]]:
    """Tokenize every record (optionally the first `max_samples`) into `(tokens, offset)`
    pairs. Drops any pair whose completion is empty (offset >= len(tokens)) -- an
    unsupervised example the trainer's `ntoks` denominator could not divide by."""
    chosen = records[:max_samples] if max_samples is not None else records
    pairs: list[tuple[list[int], int]] = []
    for record in chosen:
        tokens, offset = tokenize_record(tokenizer, record)
        if offset < len(tokens):  # keep only examples with >= 1 supervised token
            pairs.append((tokens, offset))
    return pairs


def dataset_stats(
    pairs: list[tuple[list[int], int]], *, batch_size: int, pack_len: int, seed: int
) -> dict[str, object]:
    """The stats block: sequence-length histogram, the EXACT stock padding-waste fraction
    at `batch_size`, and the packed-utilization preview at `pack_len` -- all deterministic
    from the tokenized lengths (the padding-waste fraction is part of the public claim)."""
    lengths = [len(tokens) for tokens, _ in pairs]
    hist = length_histogram(lengths)
    stock = stock_batching_stats(lengths, batch_size=batch_size, max_seq_length=pack_len)
    packed = packed_batching_stats(lengths, pack_len, batch_size=batch_size, seed=seed)
    return {
        "count": hist.count,
        "mean": round(hist.mean, 3),
        "median": hist.median,
        "p90": round(hist.p90, 3),
        "min": hist.minimum,
        "max": hist.maximum,
        "total_tokens": hist.total_tokens,
        "stock_batching": {
            "batch_size": batch_size,
            "max_seq_length": pack_len,
            "num_batches": stock.num_batches,
            "real_tokens_total": stock.real_tokens_total,
            "padded_tokens_total": stock.padded_tokens_total,
            "padding_waste_fraction": round(stock.padding_waste_fraction, 6),
        },
        "packed": {
            "pack_len": pack_len,
            "batch_size": batch_size,
            "num_batches": packed.num_batches,
            "real_tokens_total": packed.real_tokens_total,
            "utilization": round(packed.utilization, 6),
            "separator_fraction": round(packed.separator_fraction, 6),
            "tail_pad_fraction": round(packed.tail_pad_fraction, 6),
        },
    }


# ---------------------------------------------------------------------------
# I/O shell: hf-CLI download + parquet read (lazy, network), tokenizer load, write
# ---------------------------------------------------------------------------


def download_alpaca(revision: str) -> list[dict[str, str]]:
    """Download `tatsu-lab/alpaca` at `revision` via the `hf` CLI and read its single
    parquet into `{"instruction", "input", "output"}` records. `pyarrow` is required for
    the parquet read (a `MissingDependencyError` names it if absent)."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 -- lazy: only the network path needs it
    except ImportError as exc:
        raise MissingDependencyError(
            "reading the Alpaca parquet requires pyarrow (pip install pyarrow)"
        ) from exc
    result = subprocess.run(
        ["hf", "download", ALPACA_REPO, "--repo-type", "dataset", "--revision", revision],
        check=True, capture_output=True, text=True,
    )
    local = Path(result.stdout.strip().splitlines()[-1])
    records: list[dict[str, str]] = []
    for parquet_path in sorted(local.glob("data/*.parquet")):
        cols = pq.read_table(parquet_path).to_pydict()
        count = len(cols["instruction"])
        inputs = cols.get("input", [""] * count)
        for i in range(count):
            records.append({
                "instruction": str(cols["instruction"][i]),
                "input": str(inputs[i] or ""),
                "output": str(cols["output"][i]),
            })
    return records


def load_tokenizer(model_id: str, revision: str | None = None) -> Any:
    """The model's tokenizer (via `transformers.AutoTokenizer` -- lighter than a full
    `mlx_lm.load`, and the chat template ships with the tokenizer, not the weights)."""
    from transformers import AutoTokenizer  # noqa: PLC0415 -- lazy heavy dep

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def write_jsonl(path: Path, pairs: list[tuple[list[int], int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for tokens, offset in pairs:
            handle.write(json.dumps({"tokens": tokens, "offset": offset}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id whose chat template + "
                    "tokenizer are applied (e.g. mlx-community/Qwen3-8B-4bit)")
    ap.add_argument("--out", required=True, help="output jsonl path; the stats block is "
                    "written to a sibling <out>.stats.json")
    ap.add_argument("--revision", default=ALPACA_REVISION,
                    help=f"Alpaca dataset commit (default: the pinned {ALPACA_REVISION})")
    ap.add_argument("--model-revision", default=None, help="revision for the tokenizer repo")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch size the stock padding-waste fraction is computed at")
    ap.add_argument("--pack-len", type=int, default=4096,
                    help="pack length / max sequence length for the stats block")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap the number of examples (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="packing seed for the stats block")
    args = ap.parse_args(argv)

    records = download_alpaca(args.revision)
    tokenizer = load_tokenizer(args.model, args.model_revision)
    pairs = records_to_pairs(tokenizer, records, max_samples=args.max_samples)

    out = Path(args.out)
    write_jsonl(out, pairs)
    stats = {
        "repo": ALPACA_REPO,
        "revision": args.revision,
        "model": args.model,
        "num_examples": len(pairs),
        **dataset_stats(pairs, batch_size=args.batch_size, pack_len=args.pack_len,
                        seed=args.seed),
    }
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
