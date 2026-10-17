"""Standalone Keras 3 transformer CPU benchmark (no existing checkpoints modified).

Install: pip install keras torch numpy
Run: python experiments/keras_benchmark.py --steps 100 --batch-size 4 --threads 2
Keras uses its PyTorch backend here; results compare API overhead, not TF vs Torch.
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import keras
from keras import layers, ops


class TransformerBlock(layers.Layer):
    def __init__(self, dim=128, heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=heads, key_dim=dim // heads, dropout=dropout
        )
        self.attn_drop = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff1 = layers.Dense(dim * 4, activation=keras.activations.gelu)
        self.ff2 = layers.Dense(dim)
        self.ff_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        a = self.attn(self.ln1(x), self.ln1(x),
                      use_causal_mask=True, training=training)
        x = x + self.attn_drop(a, training=training)
        y = self.ff2(self.ff1(self.ln2(x)))
        return x + self.ff_drop(y, training=training)


class TinyKerasGPT(keras.Model):
    def __init__(self, vocab, block=128, dim=128):
        super().__init__()
        self.tokens = layers.Embedding(vocab, dim)
        self.positions = layers.Embedding(block, dim)
        self.blocks = [TransformerBlock() for _ in range(4)]
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(vocab)
        self.block = block

    def call(self, tokens, training=False):
        positions = ops.arange(ops.shape(tokens)[1])
        x = self.tokens(tokens) + self.positions(positions)
        for block in self.blocks:
            x = block(x, training=training)
        return self.head(self.norm(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.threads) < 1:
        p.error("steps, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    np.random.seed(1337)
    with (args.data_dir / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("tokenizer") != "character":
        raise ValueError("Character-level dataset required")
    dtype = np.dtype(meta["dtype"])
    train = np.memmap(args.data_dir / "train.bin", dtype=dtype, mode="r")
    val = np.memmap(args.data_dir / "val.bin", dtype=dtype, mode="r")
    block = 128
    if min(len(train), len(val)) <= block:
        raise ValueError("Dataset too short for 128-token context")
    rng = np.random.default_rng(1337)

    def batch(data):
        starts = rng.integers(0, len(data) - block, size=args.batch_size)
        x = np.stack([data[i:i + block] for i in starts]).astype("int32")
        y = np.stack([data[i + 1:i + block + 1] for i in starts]).astype("int32")
        return x, y

    model = TinyKerasGPT(int(meta["vocab_size"]))
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=False,
    )
    x, y = batch(train)
    model(x)  # Build outside timing
    parameters = model.count_params()
    print(f"Backend: {keras.backend.backend()}; parameters: {parameters:,}")
    print(f"Dataset: {meta.get('dataset_name', meta.get('source_file', 'Unknown'))}")
    print(f"Steps: {args.steps}; batch: {args.batch_size}; threads: {args.threads}")
    started = time.perf_counter()
    final_loss = None
    for step in range(args.steps):
        x, y = batch(train)
        final_loss = float(model.train_on_batch(x, y))
        if (step + 1) % 25 == 0:
            print(f"step {step + 1}: loss {final_loss:.4f}")
    seconds = time.perf_counter() - started
    x, y = batch(val)
    val_loss = float(model.test_on_batch(x, y))
    print(f"Training time: {seconds:.3f}s; steps/s: {args.steps / seconds:.3f}")
    print(f"Train loss: {final_loss:.4f}; validation batch loss: {val_loss:.4f}")
    print("Note: Keras uses untied output embeddings; parameter count differs from PyTorch.")
    print("This is an exploratory benchmark, not an apples-to-apples speed comparison.")


if __name__ == "__main__":
    main()
