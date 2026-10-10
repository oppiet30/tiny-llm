import argparse
import json
import os
import platform
import shutil
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from model import GPTConfig, TinyGPT


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train the TinyGPT character-level language model."
)

parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data"),
    help="Prepared dataset directory. Default: data"
)

parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=Path("."),
    help="Checkpoint directory. Default: current directory"
)

parser.add_argument(
    "--steps",
    type=int,
    default=10000,
    help="Target training step. Default: 10000"
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Checkpoint file to resume from."
)

parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=250,
    help="Save a checkpoint every N steps. Default: 250"
)

parser.add_argument(
    "--upload",
    action="store_true",
    help="Upload the completed benchmark to MariaDB after saving its SQL file.",
)

parser.add_argument(
    "--db-host",
    default=None,
    help="MariaDB server IP/hostname for --upload (overrides benchmark-upload.local.json).",
)

parser.add_argument(
    "--threads",
    type=int,
    default=None,
    help="Number of PyTorch CPU threads to use."
)

args = parser.parse_args()

checkpoint_directory = args.checkpoint_dir

checkpoint_directory.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------

if args.steps < 1:
    parser.error("--steps must be at least 1")

if args.checkpoint_interval < 1:
    parser.error("--checkpoint-interval must be at least 1")

if args.threads is not None and args.threads < 1:
    parser.error("--threads must be at least 1")


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

batch_size = 16
block_size = 128

max_steps = args.steps
checkpoint_interval = args.checkpoint_interval

eval_interval = 250
learning_rate = 3e-4
eval_iters = 10

device = "cpu"

torch.manual_seed(1337)


# ---------------------------------------------------------
# CPU thread configuration
# ---------------------------------------------------------

cpu_threads_available = os.cpu_count()

if args.threads is not None:
    torch.set_num_threads(args.threads)

pytorch_threads = torch.get_num_threads()
interop_threads = torch.get_num_interop_threads()

print("CPU configuration:")
print(f"  Logical CPUs available: {cpu_threads_available}")
print(f"  PyTorch threads:        {pytorch_threads}")
print(f"  PyTorch interop:        {interop_threads}")
print()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_cpu_model():
    try:
        with open(
            "/proc/cpuinfo",
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                if line.startswith("model name"):
                    return line.split(
                        ":",
                        1
                    )[1].strip()

    except OSError:
        pass

    return platform.processor() or "Unknown"


def sql_string(value):
    if value is None:
        return "NULL"

    value = str(value)

    value = value.replace(
        "\\",
        "\\\\"
    )

    value = value.replace(
        "'",
        "''"
    )

    return "'" + value + "'"


def write_checkpoint_info(
    filename,
    step
):
    with open(
        checkpoint_directory / "checkpoint.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"checkpoint_file={filename}\n"
        )

        f.write(
            f"step={step}\n"
        )

        f.write(
            f"target_steps={max_steps}\n"
        )


def save_checkpoint(step):

    checkpoint = {
        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        # Save RNG state so a resumed run follows exactly
        # the same random training sequence.
        "rng_state":
            torch.get_rng_state(),

        "config": {
            "vocab_size":
                vocab_size,

            "block_size":
                block_size,

            "n_embd":
                config.n_embd,

            "n_head":
                config.n_head,

            "n_layer":
                config.n_layer,

            "dropout":
                config.dropout,
        },

        "stoi": stoi,
        "itos": itos,
    }

    checkpoint_filename = (
        f"checkpoint-{step:08d}.pt"
    )

    checkpoint_path = (
        checkpoint_directory
        / checkpoint_filename
    )

    torch.save(
        checkpoint,
        checkpoint_path
    )

    shutil.copyfile(
        checkpoint_path,
        checkpoint_directory / "model.pt"
    )

    write_checkpoint_info(
        checkpoint_filename,
        step
    )

    return checkpoint_path


# ---------------------------------------------------------
# Dataset files
# ---------------------------------------------------------

data_directory = args.data_dir

meta_file = (
    data_directory / "meta.json"
)

train_file = (
    data_directory / "train.bin"
)

val_file = (
    data_directory / "val.bin"
)


# ---------------------------------------------------------
# Load dataset metadata
# ---------------------------------------------------------

if not meta_file.exists():
    raise FileNotFoundError(
        f"{meta_file} does not exist. "
        "Run prepare_data.py first."
    )

if not train_file.exists():
    raise FileNotFoundError(
        f"{train_file} does not exist. "
        "Run prepare_data.py first."
    )

if not val_file.exists():
    raise FileNotFoundError(
        f"{val_file} does not exist. "
        "Run prepare_data.py first."
    )

with open(
    meta_file,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# ---------------------------------------------------------
# Validate metadata
# ---------------------------------------------------------

if metadata.get(
    "format_version"
) not in (1, 2):

    raise RuntimeError(
        "Unsupported dataset format version: "
        f"{metadata.get('format_version')}"
    )


if metadata.get(
    "tokenizer"
) != "character":

    raise RuntimeError(
        "This trainer currently requires "
        "a character-level dataset."
    )


vocab_size = int(
    metadata["vocab_size"]
)

dataset_name = metadata.get(
    "dataset_name",
    "Unknown Dataset"
)

dtype_name = metadata[
    "dtype"
]

try:
    token_dtype = np.dtype(
        dtype_name
    )

except TypeError as exc:
    raise RuntimeError(
        f"Unsupported token dtype: "
        f"{dtype_name}"
    ) from exc


if token_dtype.kind != "u":
    raise RuntimeError(
        "Token data type must be an "
        "unsigned integer type."
    )


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

stoi = metadata[
    "stoi"
]

itos_list = metadata[
    "itos"
]

itos = {
    i: ch
    for i, ch in enumerate(
        itos_list
    )
}


if len(stoi) != vocab_size:
    raise RuntimeError(
        "stoi size does not match "
        "vocab_size."
    )

if len(itos) != vocab_size:
    raise RuntimeError(
        "itos size does not match "
        "vocab_size."
    )


def encode(s):
    return [
        stoi[c]
        for c in s
    ]


def decode(tokens):
    return "".join(
        itos[int(i)]
        for i in tokens
    )


# ---------------------------------------------------------
# Memory-map binary datasets
# ---------------------------------------------------------

train_data = np.memmap(
    train_file,
    dtype=token_dtype,
    mode="r"
)

val_data = np.memmap(
    val_file,
    dtype=token_dtype,
    mode="r"
)


# ---------------------------------------------------------
# Validate binary datasets
# ---------------------------------------------------------

expected_train_tokens = int(
    metadata["train_tokens"]
)

expected_val_tokens = int(
    metadata["val_tokens"]
)


if len(train_data) != expected_train_tokens:
    raise RuntimeError(
        "train.bin token count does not "
        "match meta.json."
    )


if len(val_data) != expected_val_tokens:
    raise RuntimeError(
        "val.bin token count does not "
        "match meta.json."
    )


if len(train_data) <= block_size:
    raise RuntimeError(
        "Training dataset is too small "
        "for the configured block size."
    )


if len(val_data) <= block_size:
    raise RuntimeError(
        "Validation dataset is too small "
        "for the configured block size."
    )


print("Dataset configuration:")

print(
    f"  Dataset:               "
    f"{dataset_name}"
)

print(
    f"  Source:                "
    f"{metadata.get('source_file', 'Unknown')}"
)

print(
    f"  Tokenizer:             "
    f"{metadata['tokenizer']}"
)

print(
    f"  Vocabulary size:       "
    f"{vocab_size}"
)

print(
    f"  Token data type:       "
    f"{token_dtype.name}"
)

print(
    f"  Training tokens:       "
    f"{len(train_data):,}"
)

print(
    f"  Validation tokens:     "
    f"{len(val_data):,}"
)

print(
    f"  Training data bytes:   "
    f"{train_data.nbytes:,}"
)

print(
    f"  Validation data bytes: "
    f"{val_data.nbytes:,}"
)

print(
    "  Loading method:        "
    "NumPy memory map"
)

print()


# ---------------------------------------------------------
# Create batches
#
# Only the selected batch is converted from the memory-mapped
# token file into PyTorch int64 tensors.
# ---------------------------------------------------------

def get_batch(split):

    source = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        torch.from_numpy(
            np.asarray(
                source[
                    int(i):
                    int(i) + block_size
                ]
            ).copy()
        )
        for i in ix
    ]).long()

    y = torch.stack([
        torch.from_numpy(
            np.asarray(
                source[
                    int(i) + 1:
                    int(i) + block_size + 1
                ]
            ).copy()
        )
        for i in ix
    ]).long()

    return (
        x.to(device),
        y.to(device)
    )


# ---------------------------------------------------------
# Configure model
# ---------------------------------------------------------

config = GPTConfig()

config.vocab_size = vocab_size
config.block_size = block_size
config.n_embd = 128
config.n_head = 4
config.n_layer = 4
config.dropout = 0.1

model = TinyGPT(
    config
).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

model_name = (
    f"TinyGPT-{parameter_count // 1000}K"
)

print(
    f"Model parameters: "
    f"{parameter_count:,}"
)


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

start_step = 0

if args.resume is not None:

    print()

    print(
        f"Loading checkpoint: "
        f"{args.resume}"
    )

    checkpoint = torch.load(
        args.resume,
        map_location=device,
        weights_only=False
    )

    if "step" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain a "
            "step number."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if "optimizer_state_dict" in checkpoint:

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        print(
            "Optimizer state restored."
        )

    else:

        print(
            "Checkpoint has no "
            "optimizer state."
        )

        print(
            "AdamW optimizer will "
            "start fresh."
        )

    start_step = int(
        checkpoint["step"]
    )

    print(
        f"Checkpoint step: "
        f"{start_step:,}"
    )

    if "rng_state" in checkpoint:

        torch.set_rng_state(
            checkpoint["rng_state"]
        )

        print(
            "PyTorch RNG state restored."
        )

    else:

        print(
            "Checkpoint has no RNG state."
        )

        print(
            "Resume will work, but exact "
            "random-sequence reproducibility "
            "is not guaranteed."
        )

    if start_step > max_steps:

        raise RuntimeError(
            f"Checkpoint is at step "
            f"{start_step:,}, but target "
            f"is only {max_steps:,}."
        )

    if start_step == max_steps:

        print(
            "Checkpoint is already at "
            "the requested target step."
        )

else:

    print()
    print("Starting new model.")


# ---------------------------------------------------------
# Estimate training and validation loss
#
# Evaluation uses random batches. Preserve the RNG state so
# evaluation does not alter the training random sequence.
# ---------------------------------------------------------

@torch.no_grad()
def estimate_loss():

    rng_state = torch.get_rng_state()

    try:

        results = {}

        model.eval()

        for split in [
            "train",
            "val"
        ]:

            losses = torch.zeros(
                eval_iters
            )

            for k in range(
                eval_iters
            ):

                xb, yb = get_batch(
                    split
                )

                logits, loss = model(
                    xb,
                    yb
                )

                losses[k] = (
                    loss.item()
                )

            results[split] = (
                losses.mean().item()
            )

        return results

    finally:

        model.train()

        torch.set_rng_state(
            rng_state
        )


# ---------------------------------------------------------
# Training summary
# ---------------------------------------------------------

print()
print("Training configuration:")

print(
    f"  Start step:          "
    f"{start_step:,}"
)

print(
    f"  Target step:         "
    f"{max_steps:,}"
)

print(
    f"  Steps this run:      "
    f"{max_steps - start_step:,}"
)

print(
    f"  Checkpoint interval: "
    f"{checkpoint_interval:,}"
)

print(
    f"  Checkpoint directory: "
    f"{checkpoint_directory}"
)

print(
    f"  Evaluation interval: "
    f"{eval_interval:,}"
)

print(
    f"  Batch size:          "
    f"{batch_size}"
)

print(
    f"  Block size:          "
    f"{block_size}"
)

print(
    f"  Learning rate:       "
    f"{learning_rate}"
)

print()


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

model.train()

wall_start = (
    time.perf_counter()
)

interval_start = wall_start
interval_step = start_step

for step in range(
    start_step,
    max_steps
):

    if step % eval_interval == 0:

        losses = estimate_loss()

        if step == start_step:

            print(
                f"step {step:5d} | "
                f"train "
                f"{losses['train']:.4f} | "
                f"val "
                f"{losses['val']:.4f}"
            )

        else:

            interval_seconds = (
                time.perf_counter()
                - interval_start
            )

            steps_in_interval = (
                step
                - interval_step
            )

            print(
                f"step {step:5d} | "
                f"train "
                f"{losses['train']:.4f} | "
                f"val "
                f"{losses['val']:.4f} | "
                f"last "
                f"{steps_in_interval}: "
                f"{interval_seconds:.2f}s"
            )

            interval_start = (
                time.perf_counter()
            )

            interval_step = step

    xb, yb = get_batch(
        "train"
    )

    logits, loss = model(
        xb,
        yb
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

    completed_step = (
        step + 1
    )

    if (
        completed_step
        % checkpoint_interval
        == 0
        and completed_step
        < max_steps
    ):

        filename = save_checkpoint(
            completed_step
        )

        print(
            f"Checkpoint saved: "
            f"{filename}"
        )


# ---------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------

losses = estimate_loss()

wall_seconds = (
    time.perf_counter()
    - wall_start
)

final_train_loss = (
    losses["train"]
)

final_val_loss = (
    losses["val"]
)

print()
print("Training complete.")

print(
    f"Final train loss: "
    f"{final_train_loss:.4f}"
)

print(
    f"Final validation loss: "
    f"{final_val_loss:.4f}"
)

print(
    f"Training wall time: "
    f"{wall_seconds:.3f} seconds"
)


# ---------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------

final_checkpoint = (
    save_checkpoint(
        max_steps
    )
)

print(
    f"Saved final checkpoint "
    f"at step {max_steps:,}"
)

print(
    f"Checkpoint file: "
    f"{final_checkpoint}"
)

print(
    f"Latest checkpoint: "
    f"{checkpoint_directory / 'model.pt'}"
)

print(
    f"Checkpoint status: "
    f"{checkpoint_directory / 'checkpoint.txt'}"
)


# ---------------------------------------------------------
# Collect benchmark information
# ---------------------------------------------------------

hostname = (
    socket.gethostname()
    .split(".")[0]
)

cpu_model = (
    get_cpu_model()
)

pytorch_version = (
    torch.__version__
)

pytorch_threads = (
    torch.get_num_threads()
)

interop_threads = (
    torch.get_num_interop_threads()
)

operating_system = (
    platform.platform()
)

steps_this_run = (
    max_steps
    - start_step
)


# ---------------------------------------------------------
# Generate benchmark SQL
# ---------------------------------------------------------

os.makedirs(
    "benchmarks",
    exist_ok=True
)

timestamp = (
    datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
)

upload_id = str(uuid.uuid4())
benchmark_notes = (
    "Automatically generated by train.py v2.1 - memmap dataset; "
    f"upload_id={upload_id}"
)

sql_filename = (
    f"benchmarks/"
    f"{hostname}-"
    f"{start_step}-to-{max_steps}-"
    f"{timestamp}.sql"
)

sql = f"""USE tiny_llm_benchmarks;

INSERT INTO machines (
    hostname,
    cpu_model,
    cpu_threads,
    operating_system
)
VALUES (
    {sql_string(hostname)},
    {sql_string(cpu_model)},
    {cpu_threads_available},
    {sql_string(operating_system)}
)
ON DUPLICATE KEY UPDATE
    cpu_model = VALUES(cpu_model),
    cpu_threads = VALUES(cpu_threads),
    operating_system = VALUES(operating_system);

INSERT INTO benchmark_runs (
    machine_id,
    dataset_id,
    model_id,
    start_step,
    training_steps,
    steps_this_run,
    batch_size,
    learning_rate,
    pytorch_version,
    pytorch_threads,
    interop_threads,
    train_loss,
    validation_loss,
    real_seconds,
    notes
)
VALUES (
    (
        SELECT machine_id
        FROM machines
        WHERE hostname =
        {sql_string(hostname)}
    ),
    (
        SELECT dataset_id
        FROM datasets
        WHERE name =
        {sql_string(dataset_name)}
    ),
    (
        SELECT model_id
        FROM models
        WHERE name =
        {sql_string(model_name)}
    ),
    {start_step},
    {max_steps},
    {steps_this_run},
    {batch_size},
    {learning_rate},
    {sql_string(pytorch_version)},
    {pytorch_threads},
    {interop_threads},
    {final_train_loss:.8f},
    {final_val_loss:.8f},
    {wall_seconds:.3f},
    {sql_string(benchmark_notes)}
);
"""

with open(
    sql_filename,
    "w",
    encoding="utf-8"
) as f:

    f.write(sql)

print(
    f"Benchmark SQL saved to "
    f"{sql_filename}"
)


# ---------------------------------------------------------
# Optional direct database upload
# ---------------------------------------------------------

if args.upload:
    from benchmark_upload import upload_benchmark

    benchmark_record = {
        "upload_id": upload_id,
        "notes": benchmark_notes,
        "hostname": hostname.lower(),
        "cpu_model": cpu_model,
        "cpu_threads_available": cpu_threads_available,
        "operating_system": operating_system,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "start_step": start_step,
        "training_steps": max_steps,
        "steps_this_run": steps_this_run,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pytorch_version": pytorch_version,
        "pytorch_threads": pytorch_threads,
        "interop_threads": interop_threads,
        "train_loss": final_train_loss,
        "validation_loss": final_val_loss,
        "real_seconds": wall_seconds,
    }

    try:
        status, database_run_id = upload_benchmark(
            benchmark_record,
            db_host_override=args.db_host,
        )
        if status == "inserted":
            print(f"Benchmark uploaded to MariaDB as run #{database_run_id}")
        else:
            print(f"Benchmark was already present in MariaDB as run #{database_run_id}")
    except Exception as exc:
        print(f"WARNING: database upload failed: {exc}")
        print(f"The SQL fallback is still available at {sql_filename}")
