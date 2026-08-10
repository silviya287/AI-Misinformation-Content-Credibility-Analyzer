"""
core/report_generator.py

Module C - Reports (Milestone 6).

Takes a list of analysis history records (the same dicts that
core.database.get_history() returns) and writes them out as a PDF or
CSV file the user can save/share.
"""

import csv

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# The columns we export, in order, and their friendly header labels.
REPORT_COLUMNS = [
    ("analysis_date", "Date"),
    ("input_type", "Input Type"),
    ("prediction", "Prediction"),
    ("confidence", "Confidence (%)"),
    ("credibility_score", "Credibility Score"),
    ("risk_level", "Risk Level"),
]


def export_csv(records: list[dict], file_path: str) -> None:
    """
    Writes the given analysis records to a CSV file at file_path.
    Uses the standard library's csv module - no extra dependency needed.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        field_keys = [key for key, _label in REPORT_COLUMNS]
        header_labels = [label for _key, label in REPORT_COLUMNS]

        writer = csv.writer(csv_file)
        writer.writerow(header_labels)

        for record in records:
            writer.writerow([record.get(key, "") for key in field_keys])


def export_pdf(records: list[dict], file_path: str) -> None:
    """
    Writes the given analysis records to a simple PDF report at
    file_path, using reportlab. The PDF has a title and a table of all
    the records.
    """
    document = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("AI Content Trust & Spam Detection - Analysis Report", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 0.25 * inch))

    if not records:
        elements.append(Paragraph("No analysis history to report.", styles["Normal"]))
    else:
        header_labels = [label for _key, label in REPORT_COLUMNS]
        table_data = [header_labels]

        for record in records:
            row = [str(record.get(key, "")) for key, _label in REPORT_COLUMNS]
            table_data.append(row)

        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        elements.append(table)

    document.build(elements)


# Lets a teammate quickly test this file directly, e.g.:
#   python core/report_generator.py
if __name__ == "__main__":
    sample_records = [
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
    export_csv(sample_records, "sample_report.csv")
    export_pdf(sample_records, "sample_report.pdf")
    print("Wrote sample_report.csv and sample_report.pdf")
