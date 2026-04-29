"""
CPU smoke-test trainer for autoresearch.

This is intentionally separate from train.py. It validates the data,
tokenizer, dataloader, and next-token training path without requiring CUDA.
Usage: uv run cpu/train_cpu.py --seconds 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare import DATA_DIR, TOKENIZER_DIR, Tokenizer, VAL_FILENAME  # noqa: E402


def parquet_paths(split: str) -> list[str]:
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}. Run `uv run prepare.py --num-shards 1` first.")

    paths = sorted(
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.endswith(".parquet") and not name.endswith(".tmp")
    )
    if not paths:
        raise FileNotFoundError(f"No parquet shards found in {DATA_DIR}. Run `uv run prepare.py --num-shards 1` first.")

    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        train_paths = [path for path in paths if path != val_path]
        if not train_paths:
            raise FileNotFoundError("No training shards found. Run `uv run prepare.py --num-shards 1` first.")
        return train_paths
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation shard missing: {val_path}")
    return [val_path]


def text_batches(split: str, tokenizer_batch_size: int):
    paths = parquet_paths(split)
    while True:
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group, columns=["text"])
                texts = table.column("text").to_pylist()
                for i in range(0, len(texts), tokenizer_batch_size):
                    yield texts[i : i + tokenizer_batch_size]


def make_cpu_dataloader(tokenizer: Tokenizer, batch_size: int, seq_len: int, split: str, tokenizer_batch_size: int):
    bos = tokenizer.get_bos_token_id()
    batches = text_batches(split, tokenizer_batch_size)
    token_buffer: list[int] = []
    needed = batch_size * (seq_len + 1)

    while True:
        while len(token_buffer) < needed:
            docs = next(batches)
            encoded = tokenizer.encode(docs, prepend=bos, num_threads=1)
            for row in encoded:
                token_buffer.extend(row)

        chunk = token_buffer[:needed]
        del token_buffer[:needed]
        rows = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
        yield rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


@dataclass
class CPUConfig:
    vocab_size: int
    seq_len: int
    n_layer: int
    n_embd: int
    n_head: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CPUConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: CPUConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: CPUConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.seq_len = config.seq_len

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model limit {self.seq_len}")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = next(loader)
        losses.append(float(model(x, y).item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU smoke-test trainer for autoresearch")
    parser.add_argument("--seconds", type=float, default=60.0, help="Training wall-clock budget")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="CPU batch size")
    parser.add_argument("--n-layer", type=int, default=2, help="Number of Transformer blocks")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--eval-batches", type=int, default=4, help="Validation batches for final loss")
    parser.add_argument("--tokenizer-batch-size", type=int, default=32, help="Documents encoded per tokenizer batch")
    parser.add_argument("--log-interval", type=int, default=10, help="Training log interval in steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(TOKENIZER_DIR):
        raise FileNotFoundError(f"Tokenizer directory not found: {TOKENIZER_DIR}. Run `uv run prepare.py --num-shards 1` first.")

    torch.manual_seed(42)
    device = torch.device("cpu")
    tokenizer = Tokenizer.from_directory()
    config = CPUConfig(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = make_cpu_dataloader(tokenizer, args.batch_size, args.seq_len, "train", args.tokenizer_batch_size)
    val_loader = make_cpu_dataloader(tokenizer, args.batch_size, args.seq_len, "val", args.tokenizer_batch_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Vocab size: {config.vocab_size:,}")
    print(f"Model params: {num_params / 1e6:.3f}M")
    print(f"Training seconds: {args.seconds:.1f}")

    t_start = time.time()
    t_train_start = time.time()
    step = 0
    last_loss = math.nan
    while time.time() - t_train_start < args.seconds:
        x, y = next(train_loader)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        step += 1
        if args.log_interval > 0 and step % args.log_interval == 0:
            elapsed = time.time() - t_train_start
            print(f"step {step:05d} | loss: {last_loss:.6f} | elapsed: {elapsed:.1f}s", flush=True)

    train_seconds = time.time() - t_train_start
    val_loss = evaluate_loss(model, val_loader, args.eval_batches)
    total_seconds = time.time() - t_start

    print("---")
    print(f"val_loss:        {val_loss:.6f}")
    print(f"last_train_loss: {last_loss:.6f}")
    print(f"train_seconds:   {train_seconds:.1f}")
    print(f"total_seconds:   {total_seconds:.1f}")
    print(f"num_steps:       {step}")
    print(f"num_params_M:    {num_params / 1e6:.3f}")
    print("device:          cpu")


if __name__ == "__main__":
    main()
