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
import socket
import platform
import uuid
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
    p.add_argument("--dataset-name", default=None, help="Override dataset display name for upload")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--upload", action="store_true", help="Upload results to MariaDB")
    p.add_argument("--db-host", default=None, help="MariaDB host override")
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
    dataset_name = args.dataset_name or meta.get("dataset_name")
    if args.upload and not dataset_name:
        p.error("Dataset name missing from meta.json; supply --dataset-name before uploading")
    print(f"Dataset: {dataset_name or meta.get('source_file', 'Unknown')}")
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

    if args.upload:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmark_upload import upload_benchmark
        from hardware_info import detect_hardware

        hardware = detect_hardware()
        ram_bytes = hardware.get("ram_bytes")
        backend = keras.backend.backend()
        record = {
            "upload_id": str(uuid.uuid4()),
            "hostname": socket.gethostname().split(".")[0].lower(),
            "cpu_model": hardware.get("cpu_model") or "Unknown",
            "cpu_cores": hardware.get("cpu_cores"),
            "cpu_threads_available": os.cpu_count(),
            "ram_mb": round(ram_bytes / (1024 * 1024)) if ram_bytes is not None else None,
            "operating_system": platform.platform(),
            "dataset_name": dataset_name,
            "vocabulary_size": int(meta["vocab_size"]),
            "dataset_notes": f"source={meta.get('source', '')}; source_mode={meta.get('source_mode', '')}",
            "model_name": f"TinyKerasGPT-{parameters // 1000}K-keras3-{backend}",
            "parameter_count": parameters,
            "n_embd": 128, "n_head": 4, "n_layer": 4,
            "block_size": block, "dropout": 0.1, "tokenizer": "character",
            "start_step": 0, "training_steps": args.steps, "steps_this_run": args.steps,
            "batch_size": args.batch_size, "learning_rate": 3e-4,
            "pytorch_version": torch.__version__,
            "pytorch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "train_loss": final_loss, "validation_loss": val_loss,
            "real_seconds": seconds,
            "notes": (
                f"Keras {keras.__version__}; backend={backend}; "
                "single validation batch; untied embeddings; exploratory comparison"
            ),
        }
        try:
            status, run_id = upload_benchmark(record, db_host_override=args.db_host)
            print(f"MariaDB benchmark {status}: run #{run_id}")
        except Exception as exc:
            print(f"WARNING: benchmark upload failed: {exc}")
            print("Training succeeded; database upload did not complete.")


if __name__ == "__main__":
    main()
