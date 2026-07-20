from pathlib import Path
import csv
import re


INPUT_CSV = Path("preprocessed_data.csv")
OUTPUT_TEXT = Path("edgar_allan_poe_corpus.txt")


def format_title(title: str) -> str:
    return title.strip().title()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)

    # This dataset appears to represent paragraph breaks as
    # runs of four or more spaces.
    text = re.sub(r" {4,}", "\n\n", text)

    # Normalize remaining accidental whitespace.
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_corpus(input_csv: Path, output_text: Path) -> None:
    works: list[str] = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)

        required_columns = {"title", "text"}
        missing = required_columns.difference(reader.fieldnames or [])

        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"CSV is missing required column(s): {missing_list}"
            )

        for row_number, row in enumerate(reader, start=2):
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()

            if not title or not text:
                print(
                    f"Skipping row {row_number}: "
                    "missing title or text."
                )
                continue

            work = f"{format_title(title)} -\n\n{clean_text(text)}"
            works.append(work)

    if not works:
        raise ValueError("No usable works were found in the CSV.")

    corpus = "\n\n\n\n".join(works) + "\n"
    output_text.write_text(corpus, encoding="utf-8")

    print(f"Wrote {len(works)} works to: {output_text.resolve()}")


if __name__ == "__main__":
    build_corpus(INPUT_CSV, OUTPUT_TEXT)