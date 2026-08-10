"""
training/dataset_loader.py

Handles loading a category's dataset and normalizing it into a simple,
consistent shape that Milestone 4 (model training) can use directly:

    texts:  list[str]   - raw text samples
    labels: list[int]   - 1 = flagged (spam/phishing/etc.), 0 = not flagged

WHY NORMALIZE?
Public datasets all use different column names and label formats
(e.g. "ham"/"spam", "0"/"1", "real"/"deceptive", multiple toxicity
columns, etc). Rather than writing a different one-off script for each
raw format, we standardize: whoever downloads a dataset converts it to
one simple CSV with exactly two columns, "text" and "label" (already
0/1), and drops it into training/datasets/<category>/data.csv.

That conversion step (raw Kaggle/UCI file -> our standard CSV) is a
few lines of pandas per dataset - see training/datasets/DATASETS.md
for exact column-mapping instructions for each of our 5 categories.

This file only needs to do ONE job after that: read the standardized
CSV and hand back clean Python lists.
"""

import os
import csv


# Folder where each category's data.csv should live.
DATASETS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

VALID_CATEGORIES = ["spam", "phishing", "clickbait", "toxic", "fake_review"]


class DatasetLoadError(Exception):
    """Raised when a category's dataset file is missing or malformed."""
    pass


def load_dataset(category: str) -> tuple[list[str], list[int]]:
    """
    Loads training/datasets/<category>/data.csv and returns
    (texts, labels) as two plain lists, ready for train/test splitting
    and TF-IDF vectorization in Milestone 4.

    Expected CSV format (exactly two columns, with a header row):
        text,label
        "click here to win a free prize",1
        "let's meet for lunch tomorrow",0

    Raises DatasetLoadError with a clear message if the file is
    missing or the category name isn't recognized.
    """
    if category not in VALID_CATEGORIES:
        raise DatasetLoadError(
            f"Unknown category '{category}'. Must be one of: {VALID_CATEGORIES}"
        )

    csv_path = os.path.join(DATASETS_ROOT, category, "data.csv")

    if not os.path.isfile(csv_path):
        raise DatasetLoadError(
            f"No dataset found for '{category}'. Expected a file at:\n"
            f"  {csv_path}\n"
            f"See training/datasets/DATASETS.md for how to download and "
            f"format this category's dataset."
        )

    texts = []
    labels = []

    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != ["text", "label"]:
            raise DatasetLoadError(
                f"'{csv_path}' must have exactly two columns named "
                f"'text' and 'label' (found: {reader.fieldnames})."
            )

        for row_number, row in enumerate(reader, start=2):  # row 1 is the header
            text_value = row["text"].strip()
            label_value = row["label"].strip()

            if not text_value:
                continue  # skip blank rows rather than failing the whole load

            if label_value not in ("0", "1"):
                raise DatasetLoadError(
                    f"Invalid label '{label_value}' on line {row_number} of "
                    f"'{csv_path}'. Labels must be 0 or 1 - see DATASETS.md "
                    f"for how to normalize each dataset's original labels."
                )

            texts.append(text_value)
            labels.append(int(label_value))

    if not texts:
        raise DatasetLoadError(f"'{csv_path}' contained no usable rows.")

    return texts, labels


# Lets a teammate quickly test this file directly, e.g.:
#   python training/dataset_loader.py spam
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python training/dataset_loader.py <category>")
        print(f"Categories: {VALID_CATEGORIES}")
    else:
        try:
            loaded_texts, loaded_labels = load_dataset(sys.argv[1])
            print(f"Loaded {len(loaded_texts)} rows for '{sys.argv[1]}'")
            print(f"Flagged (1): {sum(loaded_labels)}  Not flagged (0): {len(loaded_labels) - sum(loaded_labels)}")
            print("First 3 rows:")
            for i in range(min(3, len(loaded_texts))):
                print(f"  [{loaded_labels[i]}] {loaded_texts[i]}")
        except DatasetLoadError as e:
            print(f"[dataset_loader] Error: {e}")
