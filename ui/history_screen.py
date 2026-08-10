"""
ui/history_screen.py

Module D - User Interface (Milestone 6).

Shows every past analysis in a table, newest first, with a button to
delete a selected row. Talks directly to core/database.py.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView, QMessageBox
)

from core.database import get_history, delete_analysis


COLUMN_HEADERS = ["ID", "Date", "Input Type", "Prediction", "Confidence (%)", "Credibility", "Risk Level"]


class HistoryScreen(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("History")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMN_HEADERS))
        self.table.setHorizontalHeaderLabels(COLUMN_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Hide the raw ID column visually - we still need the id value
        # internally to support deleting the right row.
        layout.addWidget(self.table)

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.on_delete_clicked)
        button_row.addWidget(refresh_button)
        button_row.addWidget(delete_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.setLayout(layout)

    def refresh(self):
        """Reloads the table from the database. Call this any time the
        underlying data might have changed (e.g. after a new analysis,
        or when this tab is opened)."""
        records = get_history()
        self.table.setRowCount(len(records))

        for row_index, record in enumerate(records):
            values = [
                record["id"],
                record["analysis_date"],
                record["input_type"],
                record["prediction"],
                record["confidence"],
                record["credibility_score"],
                record["risk_level"],
            ]
            for col_index, value in enumerate(values):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))

        self.table.setColumnHidden(0, True)  # keep id in the table for lookups, just don't show it

    def on_delete_clicked(self):
        """Deletes whichever row is currently selected, after confirming."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a row to delete first.")
            return

        row_index = selected_rows[0].row()
        analysis_id = int(self.table.item(row_index, 0).text())

        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to delete this analysis record?",
        )
        if confirm == QMessageBox.Yes:
            delete_analysis(analysis_id)
            self.refresh()
