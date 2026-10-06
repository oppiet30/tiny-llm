import os
import platform
import socket
import time
from datetime import datetime

import torch
from model import GPTConfig, TinyGPT


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor() or "Unknown"


def sql_string(value):
    if value is None:
        return "NULL"

    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "''")

    return "'" + value + "'"


def save_checkpoint(filename, step):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
        },
        "stoi": stoi,
        "itos": itos,
    }

    torch.save(checkpoint, filename)


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

batch_size = 16
block_size = 128

max_steps = 10000

eval_interval = 250
learning_rate = 3e-4
eval_iters = 10

device = "cpu"

torch.manual_seed(1337)


# ---------------------------------------------------------
# Resume configuration
# ---------------------------------------------------------

# None means start a brand-new model.
#
# To continue our original 3000-step model:
resume_checkpoint = None

# The old checkpoint does not contain its step number,
# so we tell train.py where it came from.
old_checkpoint_step = 3000


# ---------------------------------------------------------
# Load training text
# ---------------------------------------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Characters in dataset: {len(text):,}")


# ---------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {repr(''.join(chars))}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(tokens):
    return "".join(itos[i] for i in tokens)


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ---------------------------------------------------------
# Training / validation split
# ---------------------------------------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print(f"Training characters:   {len(train_data):,}")
print(f"Validation characters: {len(val_data):,}")


# ---------------------------------------------------------
# Create batches
# ---------------------------------------------------------

def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


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

model = TinyGPT(config).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Model parameters: {parameter_count:,}")


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

if resume_checkpoint is not None:

    print()
    print(f"Loading checkpoint: {resume_checkpoint}")

    checkpoint = torch.load(
        resume_checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if "optimizer_state_dict" in checkpoint:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print("Optimizer state restored.")

    else:

        print(
            "Checkpoint has no optimizer state."
        )

        print(
            "AdamW optimizer will start fresh."
        )

    if "step" in checkpoint:
        start_step = checkpoint["step"]
    else:
        start_step = old_checkpoint_step

    print(f"Resuming from step {start_step:,}")

else:

    print()
    print("Starting new model.")


# ---------------------------------------------------------
# Estimate training and validation loss
# ---------------------------------------------------------

@torch.no_grad()
def estimate_loss():

    results = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            xb, yb = get_batch(split)

            _, loss = model(xb, yb)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

model.train()

wall_start = time.perf_counter()
interval_start = wall_start

for step in range(start_step, max_steps):

    if step % eval_interval == 0:

        losses = estimate_loss()

        if step == start_step:

            print(
                f"step {step:5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

        else:

            interval_seconds = (
                time.perf_counter() - interval_start
            )

            print(
                f"step {step:5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"last {eval_interval}: "
                f"{interval_seconds:.2f}s"
            )

            interval_start = time.perf_counter()

            save_checkpoint(
                "model.pt",
                step
            )

            print(
                f"Checkpoint saved at step {step:,}"
            )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


# ---------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------

losses = estimate_loss()

wall_seconds = (
    time.perf_counter() - wall_start
)

final_train_loss = losses["train"]
final_val_loss = losses["val"]

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

save_checkpoint(
    "model.pt",
    max_steps
)

print(
    f"Saved final checkpoint "
    f"at step {max_steps:,} to model.pt"
)


# ---------------------------------------------------------
# Collect benchmark information
# ---------------------------------------------------------

hostname = socket.gethostname().split(".")[0]

cpu_model = get_cpu_model()
cpu_threads = os.cpu_count()

pytorch_version = torch.__version__

pytorch_threads = (
    torch.get_num_threads()
)

interop_threads = (
    torch.get_num_interop_threads()
)

operating_system = platform.platform()

steps_this_run = max_steps - start_step


# ---------------------------------------------------------
# Generate benchmark SQL
# ---------------------------------------------------------

os.makedirs(
    "benchmarks",
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d-%H%M%S"
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
    {cpu_threads},
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
        WHERE hostname = {sql_string(hostname)}
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
        WHERE name = 'TinyGPT-821K'
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
    'Automatically generated by train.py'
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

