#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path


DEFAULT_ARCHIVE = Path.home() / "gutenberg-temp" / "txt-files.tar"
DEFAULT_OUTPUT = Path("data/gutenberg-corpus")

MEMBER_RE = re.compile(
    r"^cache/epub/([0-9]+)/pg\1\.txt$"
)

START_RE = re.compile(
    r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE | re.MULTILINE,
)

END_RE = re.compile(
    r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE | re.MULTILINE,
)

LANGUAGE_RE = re.compile(
    r"^Language:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_size(value):
    """
    Parse sizes such as:
        50000000
        50M
        50MB
        50MiB
        1G
    """

    value = value.strip().upper()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*"
        r"(B|K|KB|KIB|M|MB|MIB|G|GB|GIB)?",
        value,
    )

    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid size: {value}"
        )

    number = float(match.group(1))
    unit = match.group(2) or "B"

    multipliers = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024 ** 2,
        "MB": 1024 ** 2,
        "MIB": 1024 ** 2,
        "G": 1024 ** 3,
        "GB": 1024 ** 3,
        "GIB": 1024 ** 3,
    }

    return int(
        number * multipliers[unit]
    )


def deterministic_key(ebook_id, seed):
    """
    Produce a stable pseudo-random ordering key.

    Python's built-in hash() is deliberately randomized
    between processes, so SHA-256 is used instead.
    """

    value = (
        f"{seed}:{ebook_id}"
        .encode("ascii")
    )

    return hashlib.sha256(
        value
    ).digest()


def decode_book(raw):
    """
    Gutenberg's current txt-files archive is expected to
    contain UTF-8 text. A UTF-8 BOM is accepted.
    """

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def get_language(text):
    """
    Gutenberg metadata is near the beginning of each file.
    Only search the first 32 KiB of text.
    """

    header = text[:32768]

    match = LANGUAGE_RE.search(
        header
    )

    if not match:
        return None

    return match.group(1).strip()


def is_english(language):
    if language is None:
        return False

    normalized = language.casefold()

    # Accept:
    #   English
    #   English, French
    #   English and French
    #
    # Reject language names that merely contain the
    # substring "english" as part of another word.
    return bool(
        re.search(
            r"\benglish\b",
            normalized,
        )
    )


def strip_gutenberg_wrapper(text):
    """
    Remove the Gutenberg header and footer using the
    standard START/END markers.

    Return None if either marker cannot be found.
    """

    start_match = START_RE.search(
        text
    )

    if not start_match:
        return None

    body_start = text.find(
        "\n",
        start_match.end(),
    )

    if body_start == -1:
        return None

    body_start += 1

    end_match = END_RE.search(
        text,
        body_start,
    )

    if not end_match:
        return None

    body = text[
        body_start:end_match.start()
    ]

    # Normalize line endings while preserving the actual
    # text otherwise.
    body = body.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    body = body.strip()

    if not body:
        return None

    return body + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic cleaned English "
            "Project Gutenberg corpus."
        )
    )

    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=(
            "Path to txt-files.tar. "
            f"Default: {DEFAULT_ARCHIVE}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Directory for cleaned books. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )

    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=parse_size("50MiB"),
        help=(
            "Target cleaned corpus size. "
            "Default: 50MiB"
        ),
    )

    parser.add_argument(
        "--min-book-size",
        type=parse_size,
        default=parse_size("10KiB"),
        help=(
            "Minimum cleaned UTF-8 book size. "
            "Default: 10KiB"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help=(
            "Seed used for deterministic book selection. "
            "Default: 1337"
        ),
    )

    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(
            f"Archive does not exist: "
            f"{args.archive}"
        )

    if args.max_size <= 0:
        raise ValueError(
            "--max-size must be greater than zero"
        )

    if args.min_book_size <= 0:
        raise ValueError(
            "--min-book-size must be greater than zero"
        )

    # -----------------------------------------------------
    # Start with a clean output directory.
    # -----------------------------------------------------

    if args.output_dir.exists():
        shutil.rmtree(
            args.output_dir
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Pass 1:
    #
    # Read TAR metadata only and collect the Gutenberg IDs.
    # This does not extract book contents.
    # -----------------------------------------------------

    print(
        f"Archive: {args.archive}"
    )

    print(
        f"Output:  {args.output_dir}"
    )

    print(
        f"Target:  "
        f"{args.max_size:,} bytes"
    )

    print(
        f"Minimum book size: "
        f"{args.min_book_size:,} bytes"
    )

    print(
        f"Selection seed: "
        f"{args.seed}"
    )

    print()
    print(
        "Pass 1: indexing archive..."
    )

    candidates = []

    with tarfile.open(
        args.archive,
        mode="r:",
    ) as archive:

        for member in archive:

            if not member.isfile():
                continue

            match = MEMBER_RE.fullmatch(
                member.name
            )

            if not match:
                continue

            ebook_id = int(
                match.group(1)
            )

            candidates.append(
                (
                    ebook_id,
                    member.name,
                    member.size,
                )
            )

    print(
        f"  Candidate books: "
        f"{len(candidates):,}"
    )

    # Deterministic pseudo-random order.
    candidates.sort(
        key=lambda item:
            deterministic_key(
                item[0],
                args.seed,
            )
    )

    # -----------------------------------------------------
    # Pass 2:
    #
    # Reopen the TAR and inspect books in deterministic
    # order. tarfile can seek directly to members in an
    # uncompressed TAR.
    # -----------------------------------------------------

    print()
    print(
        "Pass 2: selecting and cleaning books..."
    )

    selected = []
    selected_bytes = 0

    rejected_language = 0
    rejected_decode = 0
    rejected_markers = 0
    rejected_small = 0

    with tarfile.open(
        args.archive,
        mode="r:",
    ) as archive:

        for (
            ebook_id,
            member_name,
            source_size,
        ) in candidates:

            try:
                member = archive.getmember(
                    member_name
                )
            except KeyError:
                continue

            source = archive.extractfile(
                member
            )

            if source is None:
                continue

            raw = source.read()

            text = decode_book(
                raw
            )

            if text is None:
                rejected_decode += 1
                continue

            language = get_language(
                text
            )

            if not is_english(
                language
            ):
                rejected_language += 1
                continue

            cleaned = strip_gutenberg_wrapper(
                text
            )

            if cleaned is None:
                rejected_markers += 1
                continue

            cleaned_bytes = cleaned.encode(
                "utf-8"
            )

            if (
                len(cleaned_bytes)
                < args.min_book_size
            ):
                rejected_small += 1
                continue

            output_name = (
                f"pg{ebook_id}.txt"
            )

            output_path = (
                args.output_dir
                / output_name
            )

            output_path.write_bytes(
                cleaned_bytes
            )

            sha256 = hashlib.sha256(
                cleaned_bytes
            ).hexdigest()

            selected.append(
                {
                    "ebook_id":
                        ebook_id,

                    "source_path":
                        member_name,

                    "output_file":
                        output_name,

                    "language":
                        language,

                    "source_bytes":
                        source_size,

                    "cleaned_bytes":
                        len(cleaned_bytes),

                    "characters":
                        len(cleaned),

                    "sha256":
                        sha256,
                }
            )

            selected_bytes += len(
                cleaned_bytes
            )

            print(
                f"  {len(selected):4d} books | "
                f"{selected_bytes / (1024 ** 2):8.2f} MiB | "
                f"ebook {ebook_id}"
            )

            if (
                selected_bytes
                >= args.max_size
            ):
                break

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not selected:
        raise RuntimeError(
            "No books were selected."
        )

    if selected_bytes < args.max_size:
        print()
        print(
            "WARNING: archive was exhausted before "
            "the requested corpus size was reached."
        )

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    manifest = {
        "format_version": 1,
        "source_archive":
            str(args.archive),
        "selection_method":
            "sha256-seeded-gutenberg-id",
        "seed":
            args.seed,
        "target_bytes":
            args.max_size,
        "minimum_book_bytes":
            args.min_book_size,
        "books":
            len(selected),
        "cleaned_bytes":
            selected_bytes,
        "rejected": {
            "decode":
                rejected_decode,
            "language":
                rejected_language,
            "markers":
                rejected_markers,
            "too_small":
                rejected_small,
        },
        "files":
            selected,
    }

    manifest_path = (
        args.output_dir
        / "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write(
            "\n"
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print(
        "Corpus build complete."
    )

    print()

    print(
        f"  Books selected:       "
        f"{len(selected):,}"
    )

    print(
        f"  Cleaned bytes:        "
        f"{selected_bytes:,}"
    )

    print(
        f"  Cleaned size:         "
        f"{selected_bytes / (1024 ** 2):.2f} MiB"
    )

    print(
        f"  Rejected decode:      "
        f"{rejected_decode:,}"
    )

    print(
        f"  Rejected language:    "
        f"{rejected_language:,}"
    )

    print(
        f"  Rejected markers:     "
        f"{rejected_markers:,}"
    )

    print(
        f"  Rejected too small:   "
        f"{rejected_small:,}"
    )

    print(
        f"  Manifest:             "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()

