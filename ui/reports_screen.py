"""
ui/reports_screen.py

Module D - User Interface (Milestone 6).

Lets the user export their full analysis history as a PDF or CSV file.
Talks directly to core/database.py (to get the records) and
core/report_generator.py (to write the file).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)

from core.database import get_history
from core.report_generator import export_csv, export_pdf


class ReportsScreen(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Reports")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            "Export your full analysis history as a PDF report or a CSV "
            "spreadsheet."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        button_row = QHBoxLayout()
        export_pdf_button = QPushButton("Export as PDF")
        export_pdf_button.clicked.connect(self.on_export_pdf_clicked)
        export_csv_button = QPushButton("Export as CSV")
        export_csv_button.clicked.connect(self.on_export_csv_clicked)
        button_row.addWidget(export_pdf_button)
        button_row.addWidget(export_csv_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def on_export_pdf_clicked(self):
        self._export(export_pdf, "PDF Files (*.pdf)", ".pdf")

    def on_export_csv_clicked(self):
        self._export(export_csv, "CSV Files (*.csv)", ".csv")

    def _export(self, export_function, file_filter: str, default_extension: str):
        """
        Shared logic for both export buttons: fetch history, ask the
        user where to save, call the right export_* function, and show
        a success/error message.
        """
        records = get_history()
        if not records:
            QMessageBox.information(self, "No Data", "There is no analysis history to export yet.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", f"analysis_report{default_extension}", file_filter
        )
        if not file_path:
            return  # user cancelled the dialog

        try:
            export_function(records, file_path)
            self.status_label.setText(f"Report saved to: {file_path}")
            self.status_label.setStyleSheet("color: #2e7d32;")
        except Exception as error:
            self.status_label.setText(f"Failed to export report: {error}")
            self.status_label.setStyleSheet("color: #c62828;")
