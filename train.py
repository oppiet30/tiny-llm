import argparse
import os
import platform
import shutil
import socket
import time
from datetime import datetime

import torch
from model import GPTConfig, TinyGPT


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train the TinyGPT character-level language model."
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
    "--threads",
    type=int,
    default=None,
    help="Number of PyTorch CPU threads to use."
)

args = parser.parse_args()


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
        "checkpoint.txt",
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

        # Save the PyTorch RNG state so that an interrupted
        # training run can continue the same random sequence.
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

    torch.save(
        checkpoint,
        checkpoint_filename
    )

    # model.pt always contains the latest checkpoint.
    shutil.copyfile(
        checkpoint_filename,
        "model.pt"
    )

    write_checkpoint_info(
        checkpoint_filename,
        step
    )

    return checkpoint_filename


# ---------------------------------------------------------
# Load training text
# ---------------------------------------------------------

with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print(
    f"Characters in dataset: "
    f"{len(text):,}"
)


# ---------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------

chars = sorted(
    list(set(text))
)

vocab_size = len(chars)

print(
    f"Vocabulary size: "
    f"{vocab_size}"
)

print(
    f"Characters: "
    f"{repr(''.join(chars))}"
)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [
        stoi[c]
        for c in s
    ]


def decode(tokens):
    return "".join(
        itos[i]
        for i in tokens
    )


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ---------------------------------------------------------
# Training / validation split
# ---------------------------------------------------------

n = int(
    0.9 * len(data)
)

train_data = data[:n]
val_data = data[n:]

print(
    f"Training characters:   "
    f"{len(train_data):,}"
)

print(
    f"Validation characters: "
    f"{len(val_data):,}"
)


# ---------------------------------------------------------
# Create batches
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
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[
            i + 1:
            i + block_size + 1
        ]
        for i in ix
    ])

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

    # Restore RNG after model and optimizer creation.
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
# IMPORTANT:
# Evaluation uses random batches. Saving/restoring the RNG
# state prevents evaluation from changing the random sequence
# used later by training.
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
    "Latest checkpoint: model.pt"
)

print(
    "Checkpoint status: checkpoint.txt"
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
        'Adventures of Huckleberry Finn'
    ),
    (
        SELECT model_id
        FROM models
        WHERE name =
        'TinyGPT-821K'
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
    'Automatically generated by train.py v2'
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

