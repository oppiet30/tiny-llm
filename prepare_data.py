import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description=(
        "Prepare character-level binary datasets "
        "for TinyGPT."
    )
)

source_group = parser.add_mutually_exclusive_group()

source_group.add_argument(
    "--input",
    type=Path,
    default=None,
    help="Single UTF-8 text file."
)

source_group.add_argument(
    "--input-dir",
    type=Path,
    default=None,
    help="Directory containing UTF-8 .txt files."
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("data"),
    help="Output directory. Default: data"
)

parser.add_argument(
    "--train-fraction",
    type=float,
    default=0.90,
    help="Training fraction. Default: 0.90"
)

parser.add_argument(
    "--chunk-size",
    type=int,
    default=1024 * 1024,
    help="Characters read per chunk. Default: 1048576"
)

args = parser.parse_args()


# ---------------------------------------------------------
# Defaults / validation
# ---------------------------------------------------------

# Preserve existing behavior when no source is specified.
if args.input is None and args.input_dir is None:
    args.input = Path("data/input.txt")

if not 0.0 < args.train_fraction < 1.0:
    parser.error(
        "--train-fraction must be greater than 0 "
        "and less than 1"
    )

if args.chunk_size < 1:
    parser.error(
        "--chunk-size must be at least 1"
    )

output_dir = args.output_dir
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

train_file = output_dir / "train.bin"
val_file = output_dir / "val.bin"
meta_file = output_dir / "meta.json"


# ---------------------------------------------------------
# Find source files
# ---------------------------------------------------------

if args.input is not None:

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{args.input}"
        )

    source_mode = "single-file"

    source_files = [
        args.input
    ]

else:

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{args.input_dir}"
        )

    source_mode = "directory"

    source_files = sorted(
        path
        for path in args.input_dir.rglob("*.txt")
        if path.is_file()
    )

    if not source_files:
        raise RuntimeError(
            f"No .txt files found under "
            f"{args.input_dir}"
        )


print(
    f"Source mode: "
    f"{source_mode}"
)

print(
    f"Source files: "
    f"{len(source_files):,}"
)

print()


# ---------------------------------------------------------
# Read text in chunks
# ---------------------------------------------------------

def read_chunks(
    filename,
    chunk_size
):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            yield chunk


# ---------------------------------------------------------
# Pass 1
#
# Count characters and build the character vocabulary.
#
# Memory usage depends on the vocabulary and file list,
# not the total corpus size.
# ---------------------------------------------------------

print(
    "Pass 1: scanning corpus..."
)

characters = set()
file_character_counts = {}
total_characters = 0

for file_number, filename in enumerate(
    source_files,
    start=1
):

    file_characters = 0

    for chunk in read_chunks(
        filename,
        args.chunk_size
    ):

        chunk_length = len(
            chunk
        )

        file_characters += (
            chunk_length
        )

        total_characters += (
            chunk_length
        )

        characters.update(
            chunk
        )

    file_character_counts[
        filename
    ] = file_characters

    if (
        file_number % 100 == 0
        or file_number == len(source_files)
    ):

        print(
            f"  Scanned "
            f"{file_number:,}/"
            f"{len(source_files):,} files"
        )


chars = sorted(
    characters
)

vocab_size = len(
    chars
)

stoi = {
    ch: i
    for i, ch in enumerate(
        chars
    )
}


print()

print(
    f"Characters in corpus: "
    f"{total_characters:,}"
)

print(
    f"Vocabulary size: "
    f"{vocab_size:,}"
)

print(
    f"Characters: "
    f"{repr(''.join(chars))}"
)


# ---------------------------------------------------------
# Select token dtype
# ---------------------------------------------------------

if vocab_size <= 256:

    token_dtype = np.dtype(
        np.uint8
    )

elif vocab_size <= 65536:

    token_dtype = np.dtype(
        np.uint16
    )

else:

    token_dtype = np.dtype(
        np.uint32
    )


print(
    f"Token data type: "
    f"{token_dtype.name}"
)


# ---------------------------------------------------------
# Choose train / validation files
# ---------------------------------------------------------

if source_mode == "single-file":

    # Preserve the original exact 90/10 character split
    # for the Huck Finn benchmark.

    split_character = int(
        args.train_fraction
        * total_characters
    )

    train_source_files = [
        source_files[0]
    ]

    val_source_files = [
        source_files[0]
    ]

else:

    # Directory mode keeps complete files together.
    #
    # Files are sorted deterministically. Add complete
    # files to training until adding another file would
    # move us farther from the requested train fraction.

    target_train_characters = int(
        args.train_fraction
        * total_characters
    )

    train_source_files = []
    val_source_files = []

    running_train_characters = 0

    for filename in source_files:

        file_size = (
            file_character_counts[
                filename
            ]
        )

        current_difference = abs(
            target_train_characters
            - running_train_characters
        )

        new_difference = abs(
            target_train_characters
            - (
                running_train_characters
                + file_size
            )
        )

        if (
            new_difference
            <= current_difference
        ):

            train_source_files.append(
                filename
            )

            running_train_characters += (
                file_size
            )

        else:

            val_source_files.append(
                filename
            )

    # Ensure both datasets contain at least one file.
    if not train_source_files:
        raise RuntimeError(
            "No files were assigned to "
            "the training dataset."
        )

    if not val_source_files:

        if len(train_source_files) < 2:
            raise RuntimeError(
                "Directory mode requires at least "
                "two source files to create separate "
                "training and validation datasets."
            )

        moved_file = (
            train_source_files.pop()
        )

        val_source_files.append(
            moved_file
        )


# ---------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------

def encode_chunk(
    chunk
):

    return np.fromiter(
        (
            stoi[ch]
            for ch in chunk
        ),
        dtype=token_dtype,
        count=len(chunk)
    )


# ---------------------------------------------------------
# Write single-file dataset
#
# This preserves the original character-position split.
# ---------------------------------------------------------

def write_single_file_dataset(
    filename,
    split_position
):

    train_tokens = 0
    val_tokens = 0
    position = 0

    with open(
        train_file,
        "wb"
    ) as train_output, open(
        val_file,
        "wb"
    ) as val_output:

        for chunk in read_chunks(
            filename,
            args.chunk_size
        ):

            chunk_start = (
                position
            )

            chunk_end = (
                position
                + len(chunk)
            )

            # Entire chunk belongs to training.
            if chunk_end <= split_position:

                encoded = encode_chunk(
                    chunk
                )

                encoded.tofile(
                    train_output
                )

                train_tokens += len(
                    encoded
                )

            # Entire chunk belongs to validation.
            elif chunk_start >= split_position:

                encoded = encode_chunk(
                    chunk
                )

                encoded.tofile(
                    val_output
                )

                val_tokens += len(
                    encoded
                )

            # Split falls inside this chunk.
            else:

                local_split = (
                    split_position
                    - chunk_start
                )

                train_chunk = (
                    chunk[:local_split]
                )

                val_chunk = (
                    chunk[local_split:]
                )

                if train_chunk:

                    encoded = encode_chunk(
                        train_chunk
                    )

                    encoded.tofile(
                        train_output
                    )

                    train_tokens += len(
                        encoded
                    )

                if val_chunk:

                    encoded = encode_chunk(
                        val_chunk
                    )

                    encoded.tofile(
                        val_output
                    )

                    val_tokens += len(
                        encoded
                    )

            position = (
                chunk_end
            )

    return (
        train_tokens,
        val_tokens
    )


# ---------------------------------------------------------
# Write directory dataset
#
# Complete files are kept in either training or validation.
# ---------------------------------------------------------

def write_file_group(
    files,
    output_filename,
    label
):

    token_count = 0

    with open(
        output_filename,
        "wb"
    ) as output:

        for file_number, filename in enumerate(
            files,
            start=1
        ):

            for chunk in read_chunks(
                filename,
                args.chunk_size
            ):

                encoded = encode_chunk(
                    chunk
                )

                encoded.tofile(
                    output
                )

                token_count += len(
                    encoded
                )

            if (
                file_number % 100 == 0
                or file_number == len(files)
            ):

                print(
                    f"  {label}: "
                    f"{file_number:,}/"
                    f"{len(files):,} files"
                )

    return token_count


# ---------------------------------------------------------
# Pass 2
# ---------------------------------------------------------

print()
print(
    "Pass 2: encoding corpus..."
)


if source_mode == "single-file":

    train_tokens, val_tokens = (
        write_single_file_dataset(
            source_files[0],
            split_character
        )
    )

else:

    train_tokens = write_file_group(
        train_source_files,
        train_file,
        "train"
    )

    val_tokens = write_file_group(
        val_source_files,
        val_file,
        "validation"
    )


# ---------------------------------------------------------
# Verify output sizes
# ---------------------------------------------------------

bytes_per_token = (
    token_dtype.itemsize
)

train_bytes = (
    train_file.stat().st_size
)

val_bytes = (
    val_file.stat().st_size
)

expected_train_bytes = (
    train_tokens
    * bytes_per_token
)

expected_val_bytes = (
    val_tokens
    * bytes_per_token
)


if train_bytes != expected_train_bytes:

    raise RuntimeError(
        "train.bin size does not match "
        "the encoded token count."
    )


if val_bytes != expected_val_bytes:

    raise RuntimeError(
        "val.bin size does not match "
        "the encoded token count."
    )


if (
    train_tokens
    + val_tokens
    != total_characters
):

    raise RuntimeError(
        "Encoded token count does not "
        "match source character count."
    )


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

if source_mode == "single-file":

    source_description = str(
        source_files[0]
    )

    train_file_names = [
        str(source_files[0])
    ]

    val_file_names = [
        str(source_files[0])
    ]

else:

    source_description = str(
        args.input_dir
    )

    train_file_names = [
        str(path)
        for path in train_source_files
    ]

    val_file_names = [
        str(path)
        for path in val_source_files
    ]


metadata = {
    "format_version": 2,

    "tokenizer":
        "character",

    "source_mode":
        source_mode,

    "source":
        source_description,

    # Retain this for compatibility with the current
    # train.py display.
    "source_file":
        source_description,

    "source_files":
        len(source_files),

    "characters":
        total_characters,

    "train_tokens":
        train_tokens,

    "val_tokens":
        val_tokens,

    "vocab_size":
        vocab_size,

    "dtype":
        token_dtype.name,

    "train_fraction":
        args.train_fraction,

    "bytes_per_token":
        bytes_per_token,

    "train_files":
        train_file_names,

    "val_files":
        val_file_names,

    "stoi":
        stoi,

    "itos":
        chars,
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

    f.write(
        "\n"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

actual_train_fraction = (
    train_tokens
    / total_characters
)

actual_val_fraction = (
    val_tokens
    / total_characters
)


print()
print(
    "Dataset preparation complete."
)

print()

print(
    f"  Source files:       "
    f"{len(source_files):,}"
)

if source_mode == "directory":

    print(
        f"  Training files:     "
        f"{len(train_source_files):,}"
    )

    print(
        f"  Validation files:   "
        f"{len(val_source_files):,}"
    )

print(
    f"  Total tokens:       "
    f"{total_characters:,}"
)

print(
    f"  Training tokens:    "
    f"{train_tokens:,}"
)

print(
    f"  Validation tokens:  "
    f"{val_tokens:,}"
)

print(
    f"  Training fraction:  "
    f"{actual_train_fraction:.4%}"
)

print(
    f"  Validation fraction:"
    f"  {actual_val_fraction:.4%}"
)

print(
    f"  Vocabulary size:    "
    f"{vocab_size:,}"
)

print(
    f"  Token dtype:        "
    f"{token_dtype.name}"
)

print(
    f"  Bytes per token:    "
    f"{bytes_per_token}"
)

print(
    f"  train.bin:          "
    f"{train_bytes:,} bytes"
)

print(
    f"  val.bin:            "
    f"{val_bytes:,} bytes"
)

print(
    f"  Metadata:           "
    f"{meta_file}"
)

