"""
tests/test_report_generator.py

Tests for core/report_generator.py (Milestone 6).

Run with:
    pytest tests/test_report_generator.py -v
"""

import os
import csv

from core.report_generator import export_csv, export_pdf


SAMPLE_RECORDS = [
    {
        "analysis_date": "2026-08-08 10:00:00",
        "input_type": "text",
        "prediction": "Spam, Clickbait",
        "confidence": 91.2,
        "credibility_score": 24.5,
        "risk_level": "High",
    },
    {
        "analysis_date": "2026-08-08 10:05:00",
        "input_type": "pdf",
        "prediction": "Safe",
        "confidence": 0.0,
        "credibility_score": 100.0,
        "risk_level": "Low",
    },
]


def test_export_csv_creates_correct_rows(tmp_path):
    """The exported CSV should have a header row plus one row per record."""
    csv_path = os.path.join(tmp_path, "report.csv")
    export_csv(SAMPLE_RECORDS, csv_path)

    assert os.path.isfile(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["Date", "Input Type", "Prediction", "Confidence (%)", "Credibility Score", "Risk Level"]
    assert len(rows) == 3  # header + 2 records
    assert rows[1][2] == "Spam, Clickbait"
    assert rows[2][2] == "Safe"


def test_export_csv_with_no_records(tmp_path):
    """Exporting an empty history shouldn't crash - just a header row."""
    csv_path = os.path.join(tmp_path, "empty.csv")
    export_csv([], csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 1  # header only


def test_export_pdf_creates_a_valid_pdf_file(tmp_path):
    """The exported PDF should exist and start with the standard PDF file signature."""
    pdf_path = os.path.join(tmp_path, "report.pdf")
    export_pdf(SAMPLE_RECORDS, pdf_path)

    assert os.path.isfile(pdf_path)
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_export_pdf_with_no_records_does_not_crash(tmp_path):
    """Exporting an empty history to PDF should still produce a valid (mostly empty) file."""
    pdf_path = os.path.join(tmp_path, "empty.pdf")
    export_pdf([], pdf_path)

    assert os.path.isfile(pdf_path)
