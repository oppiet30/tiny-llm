import torch

from model import GPTConfig, TinyGPT

device = "cpu"

checkpoint = torch.load(
    "model.pt",
    map_location=device,
    weights_only=False
)

cfg = checkpoint["config"]

config = GPTConfig()
config.vocab_size = cfg["vocab_size"]
config.block_size = cfg["block_size"]
config.n_embd = cfg["n_embd"]
config.n_head = cfg["n_head"]
config.n_layer = cfg["n_layer"]
config.dropout = cfg["dropout"]

model = TinyGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

def encode(text):
    return [stoi[c] for c in text]

def decode(tokens):
    return "".join(itos[i] for i in tokens)

prompt = "Huck"

tokens = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

output = model.generate(
    tokens,
    max_new_tokens=1000,
    temperature=0.8
)

print(decode(output[0].tolist()))

