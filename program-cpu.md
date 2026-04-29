# autoresearch CPU

This is the CPU-only experiment protocol for iterating on the local smoke-test trainer. It mirrors the spirit of `program.md`, but it does **not** modify or replace the original CUDA benchmark.

## Setup

To set up a new CPU experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, for example `cpu-apr29`. The branch `autoresearch-cpu/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-cpu/<tag>` from the current branch or from `master`, as agreed with the user.
3. **Read the in-scope files**:
   - `README.md` — project context.
   - `prepare.py` — data preparation and tokenizer. Do not modify.
   - `cpu/README.md` — CPU trainer usage and limits.
   - `cpu/train_cpu.py` — the only Python file edited during CPU experiments.
   - `program-cpu.md` — this protocol.
4. **Verify data exists**: check that `~/.cache/autoresearch/data` contains at least one training shard and the pinned validation shard, and that `~/.cache/autoresearch/tokenizer` contains `tokenizer.pkl`. If missing, run `uv run prepare.py --num-shards 1`.
5. **Initialize `results_cpu.tsv`** with only the header row. This file stays untracked.
6. **Run the baseline** before making any training-code changes.

## Experimentation

Each experiment runs on CPU using the independent smoke-test trainer:

```bash
uv run cpu/train_cpu.py --seconds 300 > run_cpu.log 2>&1
```

**What you CAN do:**
- Modify `cpu/train_cpu.py` only. You may change the CPU model architecture, optimizer, learning rate, batch size, sequence length defaults, data packing, evaluation batches, schedules, and training loop.

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, or `program.md`.
- Modify the downloaded data or tokenizer to improve metrics.
- Install packages or add dependencies.
- Compare CPU `val_loss` against CUDA `val_bpb`; they are different metrics and different trainers.

The goal is to reduce CPU `val_loss` under the same command and local data setup. Lower is better. Treat runtime as a constraint: the default comparison run is 300 seconds unless the user explicitly asks for shorter iteration.

Prefer simple improvements. A tiny improvement with large complexity is usually not worth keeping. A simplification with equal or better `val_loss` is worth keeping.

## Output Format

The CPU trainer prints a summary like:

```text
---
val_loss:        6.363547
last_train_loss: 5.889253
train_seconds:   300.0
total_seconds:   300.1
num_steps:       12442
num_params_M:    2.510
device:          cpu
```

Extract key results with:

```bash
grep "^val_loss:\|^num_steps:\|^num_params_M:" run_cpu.log
```

## Logging Results

Record every experiment in `results_cpu.tsv` as tab-separated values:

```text
commit	val_loss	num_steps	num_params_M	status	description
```

Use these statuses:

- `keep`: `val_loss` improved or code became meaningfully simpler with no regression.
- `discard`: run completed but did not improve enough to keep.
- `crash`: run failed or timed out.

Example:

```text
commit	val_loss	num_steps	num_params_M	status	description
a1b2c3d	6.363547	12442	2.510	keep	baseline CPU trainer
b2c3d4e	6.221000	11890	2.640	keep	increase hidden size to 144
c3d4e5f	6.500000	9000	3.100	discard	deeper model too slow
d4e5f6g	0.000000	0	0.000	crash	invalid attention shape
```

## Experiment Loop

LOOP until the user stops you:

1. Check current git state and note the starting commit.
2. Choose one experiment idea scoped to `cpu/train_cpu.py`.
3. Edit `cpu/train_cpu.py`.
4. Commit the change.
5. Run `uv run cpu/train_cpu.py --seconds 300 > run_cpu.log 2>&1`.
6. Read `val_loss`, `num_steps`, and `num_params_M` from `run_cpu.log`.
7. If the run crashed, inspect the last 50 log lines and either fix an obvious bug or record `crash`.
8. Append the result to `results_cpu.tsv` and leave that file untracked.
9. If the result should be kept, continue from the new commit.
10. If the result should be discarded, reset back to the starting commit for that experiment.

Timeout: if a CPU experiment exceeds 10 minutes, stop it and record `crash` unless the user requested a longer run.

This CPU loop is for local iteration and functional research only. It is not a replacement for the original CUDA benchmark loop.
