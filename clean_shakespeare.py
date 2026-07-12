from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADER_MARKERS = (
    "Characters in the Play",
    "Dramatis Personae",
    "ACT 1",
    "ACT I",
)

DOCUMENT_SEPARATOR = "\n\n\n"


def read_text(path: Path) -> str:
    raw = path.read_bytes()

    # UTF-8 with optional BOM first; fall back to Windows-1252.
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not decode {path}")


def remove_folger_header(text: str) -> str:
    """
    Remove front-matter before the first recognized Shakespeare-content marker.
    Preserve character lists, acts, scenes, and stage directions.
    """
    positions: list[int] = []

    for marker in HEADER_MARKERS:
        match = re.search(
            rf"(?im)^[ \t]*{re.escape(marker)}[ \t]*$",
            text,
        )
        if match:
            positions.append(match.start())

    if positions:
        return text[min(positions):]

    # Fallback: locate the first act heading.
    act_match = re.search(
        r"(?im)^[ \t]*ACT[ \t]+(?:[IVXLCDM]+|\d+)[ \t]*$",
        text,
    )
    if act_match:
        return text[act_match.start():]

    # If no known marker exists, keep the document rather than deleting content.
    return text


def normalize_text(
    text: str,
    normalize_unicode: bool = False,
) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")

    if normalize_unicode:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "--",
            "\u2026": "...",
            "\u00a0": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

    # Remove trailing spaces while preserving indentation.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Reduce very large blank regions without flattening dramatic layout.
    text = re.sub(r"\n{5,}", "\n\n\n\n", text)

    return text.strip() + "\n"


def clean_file(
    source: Path,
    destination: Path,
    normalize_unicode: bool,
) -> tuple[int, int]:
    original = read_text(source)

    # Normalize line endings before applying line-based regexes.
    normalized = original.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.lstrip("\ufeff")

    cleaned = remove_folger_header(normalized)
    cleaned = normalize_text(cleaned, normalize_unicode)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        cleaned,
        encoding="utf-8",
        newline="\n",
    )

    return len(original), len(cleaned)


def clean_corpus(
    input_dir: Path,
    output_dir: Path,
    combined_path: Path,
    normalize_unicode: bool,
) -> None:
    files = sorted(
        path for path in input_dir.rglob("*.txt")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt files found beneath {input_dir}"
        )

    combined_documents: list[str] = []
    manifest_lines = [
        "filename\toriginal_chars\tcleaned_chars\tremoved_chars"
    ]

    total_original = 0
    total_cleaned = 0

    for source in files:
        relative = source.relative_to(input_dir)
        destination = output_dir / relative

        original_count, cleaned_count = clean_file(
            source,
            destination,
            normalize_unicode,
        )

        cleaned_text = destination.read_text(encoding="utf-8")
        combined_documents.append(cleaned_text.rstrip())

        removed = original_count - cleaned_count

        manifest_lines.append(
            f"{relative.as_posix()}\t"
            f"{original_count}\t"
            f"{cleaned_count}\t"
            f"{removed}"
        )

        total_original += original_count
        total_cleaned += cleaned_count

        print(
            f"[CLEANED] {relative} "
            f"{original_count:,} -> {cleaned_count:,} chars"
        )

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(
        DOCUMENT_SEPARATOR.join(combined_documents) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = output_dir / "cleaning_manifest.tsv"
    manifest_lines.append(
        f"TOTAL\t{total_original}\t{total_cleaned}\t"
        f"{total_original - total_cleaned}"
    )
    manifest_path.write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"Files processed: {len(files)}")
    print(f"Original characters: {total_original:,}")
    print(f"Cleaned characters:  {total_cleaned:,}")
    print(
        f"Removed characters:  "
        f"{total_original - total_cleaned:,}"
    )
    print(f"Combined corpus: {combined_path}")
    print(f"Manifest:        {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Folger Shakespeare text files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shakespeare_complete"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shakespeare_cleaned"),
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path("data/shakespeare_complete_cleaned.txt"),
    )
    parser.add_argument(
        "--normalize-unicode",
        action="store_true",
        help=(
            "Convert curly quotes, em dashes, and similar punctuation "
            "to ASCII equivalents."
        ),
    )

    args = parser.parse_args()

    clean_corpus(
        input_dir=args.input,
        output_dir=args.output,
        combined_path=args.combined,
        normalize_unicode=args.normalize_unicode,
    )


if __name__ == "__main__":
    main()
