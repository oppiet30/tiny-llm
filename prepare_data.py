import json
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

input_file = Path("data/input.txt")
train_file = Path("data/train.bin")
val_file = Path("data/val.bin")
meta_file = Path("data/meta.json")

train_fraction = 0.90


# ---------------------------------------------------------
# Load source text
# ---------------------------------------------------------

print(f"Reading: {input_file}")

with open(
    input_file,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

print(
    f"Characters in dataset: "
    f"{len(text):,}"
)


# ---------------------------------------------------------
# Build character-level vocabulary
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


# ---------------------------------------------------------
# Select binary token type
# ---------------------------------------------------------

# 88 characters fit easily in uint8.
#
# Keep this automatic so the script can support larger
# character vocabularies later.

if vocab_size <= 256:
    dtype = np.uint8

elif vocab_size <= 65536:
    dtype = np.uint16

else:
    dtype = np.uint32

print(
    f"Token data type: "
    f"{np.dtype(dtype).name}"
)


# ---------------------------------------------------------
# Split source text
#
# This intentionally matches the existing train.py:
#
#     n = int(0.9 * len(data))
# ---------------------------------------------------------

split_index = int(
    train_fraction * len(text)
)

train_text = text[:split_index]
val_text = text[split_index:]

print(
    f"Training characters:   "
    f"{len(train_text):,}"
)

print(
    f"Validation characters: "
    f"{len(val_text):,}"
)


# ---------------------------------------------------------
# Encode and write binary token files
# ---------------------------------------------------------

def write_tokens(
    source_text,
    filename
):

    tokens = np.fromiter(
        (
            stoi[ch]
            for ch in source_text
        ),
        dtype=dtype,
        count=len(source_text)
    )

    tokens.tofile(
        filename
    )

    return len(tokens)


train_tokens = write_tokens(
    train_text,
    train_file
)

val_tokens = write_tokens(
    val_text,
    val_file
)


# ---------------------------------------------------------
# Save tokenizer metadata
# ---------------------------------------------------------

metadata = {
    "format_version": 1,
    "tokenizer": "character",
    "source_file": str(input_file),
    "characters": len(text),
    "train_tokens": train_tokens,
    "val_tokens": val_tokens,
    "vocab_size": vocab_size,
    "dtype": np.dtype(dtype).name,
    "train_fraction": train_fraction,

    # JSON requires string keys, so stoi is straightforward.
    "stoi": stoi,

    # Save itos as a list. The token ID is the list index.
    "itos": chars,
}

with open(
    meta_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )

    f.write("\n")


# ---------------------------------------------------------
# Verify generated files
# ---------------------------------------------------------

train_size = os.path.getsize(
    train_file
)

val_size = os.path.getsize(
    val_file
)

bytes_per_token = np.dtype(
    dtype
).itemsize

expected_train_size = (
    train_tokens
    * bytes_per_token
)

expected_val_size = (
    val_tokens
    * bytes_per_token
)

if train_size != expected_train_size:
    raise RuntimeError(
        "train.bin size does not match "
        "the expected token count."
    )

if val_size != expected_val_size:
    raise RuntimeError(
        "val.bin size does not match "
        "the expected token count."
    )


# ---------------------------------------------------------
# Verify decoding
# ---------------------------------------------------------

train_check = np.memmap(
    train_file,
    dtype=dtype,
    mode="r"
)

val_check = np.memmap(
    val_file,
    dtype=dtype,
    mode="r"
)

decoded_train = "".join(
    chars[int(token)]
    for token in train_check
)

decoded_val = "".join(
    chars[int(token)]
    for token in val_check
)

if decoded_train != train_text:
    raise RuntimeError(
        "train.bin failed decode verification."
    )

if decoded_val != val_text:
    raise RuntimeError(
        "val.bin failed decode verification."
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("Dataset preparation complete.")

print(
    f"  Train tokens: "
    f"{train_tokens:,}"
)

print(
    f"  Validation tokens: "
    f"{val_tokens:,}"
)

print(
    f"  Bytes per token: "
    f"{bytes_per_token}"
)

print(
    f"  train.bin: "
    f"{train_size:,} bytes"
)

print(
    f"  val.bin:   "
    f"{val_size:,} bytes"
)

print(
    f"  Metadata:  "
    f"{meta_file}"
)

print()
print(
    "Binary files successfully decoded "
    "back to the original text."
)

