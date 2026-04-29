# CPU Smoke-Test Trainer

This directory contains a CPU-only training path for validating the autoresearch data and model loop without CUDA. It is not benchmark-equivalent to the main `train.py` path and its `val_loss` should not be compared with the CUDA `val_bpb` metric.

The CPU script is intentionally separate from the original project files. It does not modify `prepare.py`, `train.py`, or `program.md`, and it avoids CUDA-only components such as Flash Attention and the original GPU dataloader.

## Usage

Prepare one real training shard plus the pinned validation shard and tokenizer:

```bash
uv run prepare.py --num-shards 1
```

Run the default short CPU smoke test:

```bash
uv run cpu/train_cpu.py
```

Attempt a 5-minute CPU run:

```bash
uv run cpu/train_cpu.py --seconds 300
```

For a faster path check during development:

```bash
uv run cpu/train_cpu.py --seconds 10
```

## Notes

- Defaults are deliberately small: sequence length 128, batch size 2, 2 layers, 128 hidden size, and 4 attention heads.
- The script reads real parquet shards from `~/.cache/autoresearch/data` and the tokenizer from `~/.cache/autoresearch/tokenizer`.
- Output includes `val_loss`, `train_seconds`, `total_seconds`, `num_steps`, `num_params_M`, and `device=cpu`.
