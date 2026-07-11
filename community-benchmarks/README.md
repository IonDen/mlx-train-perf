# Community benchmarks

This folder collects benchmark numbers measured on hardware the maintainer does not own.
The project is developed on an M1 Max (32 GB); the fused loss and the flash-attention path
should scale to 64/128/256/512 GB machines, but "should scale" is not data. Each file here
is one contributor's own measured run, submitted as-is.

## Submit yours in three steps

1. **Fork this repo and install the package** into an environment on your Apple-Silicon
   Mac:

   ```bash
   pip install "mlx-train-perf[mlx-lm]"
   ```

2. **Run the kit.** It detects your machine, picks shapes for your RAM, prints an honest
   time estimate, and (after you confirm) runs the committed benchmarks:

   ```bash
   mlx-train-perf contribute --tier quick        # ~10-15 min: loss-layer + single-op attention
   # or, for the full picture (loads a model, ~1-2 h):
   mlx-train-perf contribute --tier full
   ```

   It writes one file, `community-benchmarks/<chip>-<ram>gb-<date>.json`, and prints a
   ready-to-paste PR title and body.

3. **Commit that one file and open a PR**, pasting the title and body the tool printed.
   That is all — do not edit any numbers by hand.

The kit runs on unknown hardware safely: it caps GPU-wired memory below your device ceiling
and installs a memory watchdog on every bench, so an over-large shape fails cleanly instead
of destabilizing the machine. If your machine is unusually busy at start, the pre-flight
prints a warning naming how much memory it expected free versus how much it measured — you
can close other apps and retry.

## What gets measured, and what the numbers mean

- **Loss-layer** — the fused kernel's memory and forward cost versus the materialized-logits
  baseline, at the flagship shape.
- **Single-op attention** — the flash forward+backward peak memory as sequence length
  doubles, next to the stock O(N²) path. A flash ratio near 2× per doubling is the O(N)
  behavior this release adds.
- **Train-step** (full tier) — end-to-end tokens/sec for a real LoRA fine-tune step on the
  library's own path (fused loss + flash attention). This is the number your machine
  actually delivers; it is not compared against a stock-attention baseline here — that
  comparison is the maintainer's campaign, run on reference hardware.
- **Context probe** (full tier) — a binary search for the longest sequence that still fits,
  on the library's own path.

Every number is measured on the contributor's hardware and reported as-is. Nothing here is
extrapolated to machines no one has run, and no submitted number implies a context ceiling
beyond what that machine measured.

## Turning submissions into the README table

The maintainer folds merged submissions into the README's community table:

```bash
python scripts/aggregate_community.py --dir community-benchmarks
```
