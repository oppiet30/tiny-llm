import torch
from model import GPTConfig, TinyGPT

# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

batch_size = 16
block_size = 128
max_steps = 3000
eval_interval = 250
learning_rate = 3e-4
eval_iters = 10

# Use CPU for now.
device = "cpu"

torch.manual_seed(1337)

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

data = torch.tensor(encode(text), dtype=torch.long)

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

parameter_count = sum(p.numel() for p in model.parameters())

print(f"Model parameters: {parameter_count:,}")

# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

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

for step in range(max_steps):

    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {step:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

# ---------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------

losses = estimate_loss()

print()
print("Training complete.")
print(
    f"Final train loss: {losses['train']:.4f}"
)
print(
    f"Final validation loss: {losses['val']:.4f}"
)

# ---------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
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

torch.save(checkpoint, "model.pt")

print("Saved model to model.pt")
