"""
tests/test_dataset_loader.py

Basic tests for training/dataset_loader.py (Milestone 3).

Run with:
    pytest tests/test_dataset_loader.py -v

These tests write small temporary CSV files rather than relying on the
real downloaded datasets (which won't exist on every machine/CI run).
"""

import os
import pytest

from training.dataset_loader import load_dataset, DatasetLoadError


def _write_csv(folder, rows):
    """Helper: writes a data.csv file with the given rows (list of strings)."""
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, "data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("text,label\n")
        for row in rows:
            f.write(row + "\n")
    return csv_path


def test_valid_dataset_loads_correctly(tmp_path, monkeypatch):
    """A correctly formatted CSV should load into matching texts/labels lists."""
    import training.dataset_loader as loader_module

    # Point the loader at a temporary folder instead of the real one.
    monkeypatch.setattr(loader_module, "DATASETS_ROOT", str(tmp_path))
    _write_csv(
        os.path.join(tmp_path, "spam"),
        ['"free prize click now",1', '"lunch tomorrow?",0'],
    )

    texts, labels = loader_module.load_dataset("spam")
    assert texts == ["free prize click now", "lunch tomorrow?"]
    assert labels == [1, 0]


def test_missing_file_raises_clear_error(tmp_path, monkeypatch):
    """If data.csv doesn't exist yet, we should get a helpful error, not a crash."""
    import training.dataset_loader as loader_module

    monkeypatch.setattr(loader_module, "DATASETS_ROOT", str(tmp_path))

    with pytest.raises(DatasetLoadError):
        loader_module.load_dataset("clickbait")


def test_invalid_category_raises_error():
    """An unrecognized category name should be rejected immediately."""
    with pytest.raises(DatasetLoadError):
        load_dataset("not_a_real_category")


def test_invalid_label_value_raises_error(tmp_path, monkeypatch):
    """Labels must be 0 or 1 - anything else should raise a clear error."""
    import training.dataset_loader as loader_module

    monkeypatch.setattr(loader_module, "DATASETS_ROOT", str(tmp_path))
    _write_csv(os.path.join(tmp_path, "toxic"), ['"some comment",maybe'])

    with pytest.raises(DatasetLoadError):
        loader_module.load_dataset("toxic")
