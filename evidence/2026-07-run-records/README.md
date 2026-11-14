# July 2026 run records

This folder holds six sets of result files from July 2026 training runs, copied unchanged. We chose these sets because they hold the runs behind figures the project README carried until 2026-11-13, now marked withdrawn in the CHANGELOG and ROADMAP. A session is one launch of a benchmark script, which writes one file per point. Each set is whole within one session, except the stock-attention set, which holds two. The files were kept outside the repository until 2026-11-14, months after the sessions. Each file is one point (one sequence length, one arm), written by the benchmark scripts in `scripts/`. This note says what each file states by itself and what we added afterwards.

## What the runs compare

Every training point here fine-tunes Qwen3-8B-4bit with LoRA rank 8, gradient checkpointing and batch size 1, at one fixed sequence length. Only the two context-sweep folders name the data: synthetic fixed-length random tokens (`dataset_recipe`). The other records name none.

Each set compares the fused loss with the stock loss, with attention fixed within the set. The context sweep changes both together: flash attention with the fused loss against stock attention with the stock loss. Comparisons across sets mix sessions. An arm is one side of a comparison. In file names, `ours` is the fused-loss arm and `stock` is the stock-loss arm.

Memory figures use three words. The peak is the most memory MLX had in use during the point. The marginal peak is what the point added on top of what was already held before it started. The ceiling is the level at which the benchmark's watchdog stops a point. A point stopped this way is recorded as `aborted_memory_ceiling`. The ceiling follows the free memory at launch, so it differs from point to point.

Every `*_gb` field in these files is GiB (bytes divided by 1024³).

## Two memory figures, and how to read them

- 20.0 GiB (`house_wired_limit_gb`) is the wired cap the benchmark held. Peak active memory above it is not by itself a failure, and the records do not measure wired memory.
- 24.96 GiB is the wired limit the record shows before the benchmark put its own 20 GiB cap back (`wired_limit_before_reassert_gb`). A peak above it went past that limit, so it is not evidence of a safe working size.

Three completed points are above 24.96 GiB: the 8,192-token stock-loss step in the stock-attention set (25.6808 GiB), and two points on mlx 0.31.2, at 8,448 tokens (26.2299 GiB) and 8,704 tokens (26.473 GiB). Two aborted points also observed more than 24.96 GiB before they were stopped: 25.4168 GiB in the stock-attention set and 25.6379 GiB at 24,576 tokens in the sweep.

## Three things that apply to every set

- The files do not name the chip. We add "M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record". That is our statement, not the file's.
- The files carry no run date, except the probe in the last set. We add a last-written window: the times the files were last written, taken from the original local files. Git does not keep those times. All times are CEST. The build string helps a little: the part after `.d` (for example `.d20260714`) is the date of a build that had uncommitted changes, so the session happened on or after that day.
- Each training record carries a script fingerprint: the first 16 hex characters of the SHA-256 of the benchmark script that wrote it.

Figures from these files carry one label: recorded (records published 2026-11-14; chip and run date annotated). It means the file says it, and we added the chip and the date. It does not mean anyone ran the measurement again.

Throughput is the median tokens per second. In the 20-step sets that median includes the first step. In the sweep each point was set to two training steps, so its median covers two values and cannot be compared with the 20-step sets.

## Context sweep (mlx 0.32.0)

Folder `northstar_context_sweep/`, 20 files. Flash attention with the fused loss (`ours`) and stock attention with the stock loss (`stock`), at rising sequence lengths. Refusals and aborts are included. Every point was set to two training steps.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.2, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.2.1.dev0+g37660d762.d20260714` | Last-written window: 2026-07-14, 19:59 to 22:10 |
| Script fingerprint `304787652e37ab2e` | Built from base revision `37660d762` plus uncommitted changes. The measured-path library files match revision `3fea237`; the rest of the source tree was not checked |

Read these before using any number from this set.

- Completed points record `total_peak_gb`. Aborted points record `observed_active_gb` against `ceiling_gb`.
- The 23,040-token point says `status: ok`, but its losses are `[12.809766, Infinity]`. The second step produced a non-finite loss, so that point does not show that training at 23,040 tokens works. Nothing in the file marks it as failed. The file holds a bare `Infinity`, which is not strict JSON (Python's `json` module reads it).
- In this sweep the last completed stock point is 7,936 tokens (24.5339 GiB total peak), and the last flash point with finite losses is 22,528 tokens (24.0189 GiB total peak). These are where this sweep's points stop. They are not limits of either path, for four reasons:
  - every point was set to two training steps;
  - the sweep's stock arm ran slowly (at 2,048 tokens it reads 27.219 tokens per second, against 97.794 for the same condition in the stock-attention set, with the same 8.041 GiB peak);
  - the abort ceilings varied by point, from 23.1137 to 24.6956 GiB, with free memory at launch (23.5141 for stock at 8,192 tokens, 24.6956 at 23,552, 23.1137 at 24,576);
  - stock attention completed 8,192 tokens in another session, the stock-attention set below.
- Stock attention at 8,192 tokens aborted in this sweep: 24.0674 GiB observed against a 23.5141 GiB ceiling. The flash points at 23,552 and 24,576 tokens aborted the same way (24.7523 against 24.6956, and 25.6379 against 23.1137).
- The 23,296 and 32,768 points were refused before they ran. In both, the launch budget projected more than 1.0 s per dispatch. They are launch-budget refusals, not memory failures.

## Stock-attention train steps (mlx 0.32.0)

Folder `bench_train_step_stockattn/`, 4 files: the 2,048 and 8,192 token pairs with stock attention, one arm with the stock loss and one with the fused loss. All are 20-step points. The folder mixes two sessions. The fused-loss 8,192-token abort is its own session. The other three files are one session.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.2, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.1.dev31+g1717b077f.d20260711` | Last-written window: 2026-07-13, 16:29 to 23:55 |
| Script fingerprint `191fc55b97263907` | Built from base revision `1717b077f` plus uncommitted changes. The measured-path library files match revision `0192588b` (same files through `ae450246`); the rest of the source tree was not checked |

The stock-loss run at 8,192 tokens finished 20 steps with finite losses and a total peak of 25.6808 GiB (marginal peak 21.3082 GiB, 78.14 tokens per second). Its peak active memory was above both memory figures listed above, and the record does not measure wired memory. Because the peak is above 24.96 GiB, the record is not evidence that 25.68 GiB is a safe working size on this machine.

The fused-loss run at 8,192 tokens in the same folder aborted with `aborted_memory_ceiling`: 25.4168 GiB observed against a ceiling of 24.5423 GiB.

At 2,048 tokens the fused-loss arm reads 91.466 tokens per second and the stock-loss arm 97.794, with total peaks of 7.3212 and 8.041 GiB.

Stock attention at 8,192 tokens also appears in the context sweep, where it aborted. It completed here and aborted there.

## Flash-attention train steps (mlx 0.32.0)

Folder `bench_train_step/`, 4 files: the 2,048 and 8,192 token pairs with flash attention, fused loss against stock loss. All four are 20-step points from one session.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.2, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.1.dev31+g1717b077f.d20260711` | Last-written window: 2026-07-13, 10:00 to 11:22 |
| Script fingerprint `191fc55b97263907` | Built from base revision `1717b077f` plus uncommitted changes. The measured-path library files match revision `0192588b` (same files through `ae450246`); the rest of the source tree was not checked |

Three of the four files carry a `memory_warning`: the 2,048-token fused-loss file, the 2,048-token stock-loss file and the 8,192-token fused-loss file. In each, the available memory at launch was far below the 30,064,771,072 B expected for this class of machine, and the run went ahead under a reduced ceiling. For the 8,192-token fused-loss file the available memory was 21,644,869,632 B and the reduced ceiling 19,497,385,984 B (about 18.2 GiB). The warning ends "(other heavy processes resident?)". That is a question, and the record does not answer it.

Throughput, fused loss against stock loss: 86.398 against 92.072 tokens per second at 2,048 tokens, and 72.903 against 74.009 at 8,192 tokens. Total peaks are 7.0831 and 7.803 GiB at 2,048 tokens, and 12.7462 and 15.9153 GiB at 8,192 tokens.

At 8,192 tokens, two sessions give two results. The records already in the repository under `_artifacts/bench_train_step_flash/` give 71.86 tokens per second for the fused loss and 68.473 for the stock loss (1.05×), and the project README uses that pair. The pair in this folder gives 72.903 and 74.009 (0.985×), with a memory warning on the fused arm.

## Gradient-checkpointing train steps (mlx 0.31.2)

Folder `bench_train_step_gradckpt/`, 4 files: the 1,024 and 2,048 token pairs, fused loss against stock loss. All are 20-step runs. The records name no attention implementation, and the script at that revision has no flash path, so the two arms differ in the loss only.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.1, mlx 0.31.2 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.dev18+gf5c5aa587.d20260704` | Last-written window: 2026-07-08, 08:16 to 08:35 |
| Script fingerprint `afb0ac0e9a72c80e` | Built from base revision `f5c5aa587` plus uncommitted changes. The measured-path library files match revision `5626be24` (same files through `e32527b8`); the rest of the source tree was not checked |

The memory held before training starts (`active_before_gb`) is 4.2913 GiB in all four files. The marginal peaks, fused loss against stock loss, are 2.1533 and 2.1171 GiB at 1,024 tokens, and 3.0299 and 3.7497 GiB at 2,048 tokens.

Throughput, fused loss against stock loss: 84.972 against 92.013 tokens per second at 1,024 tokens, and 84.791 against 96.699 at 2,048 tokens. The worst per-step loss difference between the two arms over the 20 steps is 2.42e-3 at 1,024 tokens and 1.46e-3 at 2,048 tokens.

## mlx 0.31.2 context points

Folder `pre-rc-refresh-backup-2026-07-12/northstar_context_sweep/`, 4 files, at 8,448, 8,704 and 8,960 tokens. The folder name comes from a copy kept when newer runs replaced the older ones. These are the only points from that sweep that survive. The script has no flash path, so the arms are called fused-loss and stock-loss, and the records name no attention implementation.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.1, mlx 0.31.2 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.dev18+gf5c5aa587.d20260704` | Last-written window: 2026-07-08, 08:54 to 09:14 |
| Script fingerprint `2521d5810a1ceb7b` | Built from base revision `f5c5aa587` plus uncommitted changes. The measured-path library files match revision `5626be24` (same files through `e32527b8`); the rest of the source tree was not checked |

Two points are marked `ok` and both peaks are above 24.96 GiB. The fused-loss point at 8,448 tokens has a total peak of 26.2299 GiB, and the stock-loss point at 8,704 tokens has 26.473 GiB. Each has two loss values. Neither is evidence of a safe working size.

The other two points are `error`: the fused-loss point at 8,704 tokens and the stock-loss point at 8,960 tokens. In both, the worker process crashed (return code −6), and the Metal message in each is "Insufficient Memory".

## Command-buffer probe

Folder `launch_budget_evidence/`, 1 file: `probe_results.json`, a timing probe of how the GPU command buffers are packed, not a training run.

| In the record | Added afterwards |
|---|---|
| `machine` macOS-26.5.2-arm64; `date` 2026-07-14 09:17:13 | M1 Max 32 GB: the maintainer's only Mac when this run was made; not stated in the record |
| No mlx runtime version (it cites mlx v0.32.0 source only) and no script fingerprint | Nothing can be matched to a script revision, so this file cannot be checked against one |

The figure here comes from the third of its three probes, the dK/dV chain at 12,288 tokens. It reports a peak of 1,563,426,940 B. That is 1.56 GB, which is 1.46 GiB. Read it as 1.46 GiB.

## Records already in the repository

Three more sets were committed to the repository earlier, under `_artifacts/`. The project README and the papers cite them, so they are listed here with the same two columns as the sets above. Their files are unchanged. All three use mlx 0.32.0, and in each set the two arms came from the same session.

### Flash-attention train steps, Qwen3-8B (`_artifacts/bench_train_step_flash/`)

4 files: the 2,048 and 8,192 token pairs, fused loss against stock loss, from one session.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.1, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.1.dev1+g30f05857c.d20260709` | Run date: no earlier than 2026-07-09 (the build date), no later than 2026-07-23 (the day the files were first committed) |
| Script fingerprint `191fc55b97263907` | Built from base revision `30f05857c` plus uncommitted changes |
| Library fingerprint `14cdad93fdab8137` | The measured-path library files match revision `facc5b8` (the earliest revision with those files, and the only one on main). The run used a checkout with changes committed later, so the matching revision can come after the run. The rest of the source tree was not checked |

### Flash-attention train steps, Llama-3.2-3B (`_artifacts/bench_train_step_flash_llama3b/`)

4 files: the 2,048 and 8,192 token pairs on Llama-3.2-3B-Instruct-4bit, fused loss against stock loss, from one session.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.2, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.1.1.dev31+g1717b077f.d20260711` | Run date: no earlier than 2026-07-11 (the build date), no later than 2026-08-27 (the day the files were first committed) |
| Script fingerprint `191fc55b97263907` | Built from base revision `1717b077f` plus uncommitted changes |
| Library fingerprint `f1765993faef3a4e` | The measured-path library files match revision `0192588b` (the earliest revision with those files; they stay the same through `ae450246`). The July stock-attention and flash sets above share this revision. The run used a checkout with changes committed later, so the matching revision can come after the run. The rest of the source tree was not checked |

### Calibration runs (`_artifacts/calib_050/`)

6 run records at 10,240, 12,288 and 16,384 tokens (flash attention, fused loss against stock loss, 20 steps each), one folder per length. The set spans three sessions, one per length, so the lengths were not measured together. The folder also holds a model config and a refit manifest. They are not run records and are not listed here.

| In the record | Added afterwards |
|---|---|
| `arm64`, macOS 26.5.2, mlx 0.32.0 | M1 Max 32 GB: the maintainer's only Mac when these runs were made; not stated in the record |
| Build `0.3.2.dev1+g5c64ffceb` | Run date: no later than 2026-07-23 (the day the files were first committed). The build string has no date, so there is no earliest day |
| Script fingerprint `89ee79937c01655a` | The build string names revision `5c64ffceb`, but it comes from the package's install metadata, which can lag behind the code. The library fingerprint is the reliable link |
| Library fingerprint `fbfc6294e39dc547` | The measured-path library files match revision `e46d83ee` (the earliest revision with those files; twelve revisions carry the same files). The run used a checkout with changes committed later, so the matching revision can come after the run. The rest of the source tree was not checked |

The 10,240-token fused-loss file carries a `memory_warning`. The memory available at launch was 19,222,315,008 B, far below the 30,064,771,072 B expected for this class of machine, and the run went ahead under a reduced ceiling of 17,074,831,360 B. The warning ends "(other heavy processes resident?)". That is a question, and the record does not answer it.

## What is not here

These sets were chosen for the figures they back. Other records exist and are held back: an earlier mlx 0.31.2 pair at 1,024 tokens from 2026-07-07, whose records show no gradient checkpointing, an earlier context sweep from 2026-07-11, other July runs, and the run summary logs. No raw record survives for the mlx 0.31.2 points at 4,096 and 8,192 tokens, or for the mlx 0.31.2 attention probe, so figures that came from those runs have no file in this folder.

Runs of related shapes are already in the repository: `_artifacts/bench_train_step_flash/` and `_artifacts/calib_050/` hold flash-attention step records, and the project README uses them.

## How to check

`MANIFEST.json` has two lists: `records`, for the files in this folder, and `tracked_elsewhere`, for the records already in the repository under `_artifacts/`, which the test checks the same way. The first list gives every file here with its SHA-256 hash, the script fingerprint inside it, and the script revision whose bytes match that fingerprint (or "not checkable" for the probe). It also lists the library fingerprint (`code_sha`), the earliest commit whose files reproduce it (`code_commit`) and the files it covers (`code_deps`). `session_id` is the benchmark's run id, written in each record's identity block. To run the check:

```
uv run pytest tests/test_evidence_records.py
```

The test hashes each file and compares it with the manifest, ties each record's script fingerprint to the committed script, and recomputes the library fingerprint from the `code_deps` files at `code_commit`. It needs the full git history, and it fails rather than skips when the history is missing. It does not check the chip, the dates or the library code outside `code_deps`. Those are the parts we added or did not check.

To recompute a script fingerprint by hand, run the command below and keep the first 16 hex characters. Use the revision and script path the manifest lists for that file.

```
git show <commit>:<script> | shasum -a 256
```

To recompute a library fingerprint by hand, take one SHA-256 over each file in `code_deps`, in the listed order: first the file's base name (without its directory), then its bytes at `code_commit`. Keep the first 16 hex characters.
