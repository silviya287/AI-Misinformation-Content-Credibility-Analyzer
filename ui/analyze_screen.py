"""
ui/analyze_screen.py

Module D - User Interface (Milestone 6).

The main working screen of the app. Lets the user either paste text or
upload a file, click Analyze, and see full results: per-category
flags, confidence, risk level, credibility score, highlighted
influencing words, and a plain-English explanation. Every analysis is
also saved to history automatically.

This screen talks DIRECTLY to the core/ modules - no controller layer,
per our finalized simplified architecture.
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QFileDialog, QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt

from core.input_handler import extract_text, InputExtractionError
from core.ml_engine import MLEngine, CATEGORY_DISPLAY_NAMES
from core.database import save_analysis


# Colors used for risk badges - kept simple and readable.
RISK_COLORS = {
    "Low": "#2e7d32",     # green
    "Medium": "#f9a825",  # amber
    "High": "#c62828",    # red
}


class AnalyzeScreen(QWidget):
    def __init__(self):
        super().__init__()

        # The MLEngine loads all 5 models from disk - we only want to
        # do that ONCE, when this screen is created, not on every click.
        self.ml_engine = MLEngine()

        # Tracks a file the user has uploaded (if any). None means the
        # user is analyzing whatever's typed in the text box instead.
        self.selected_file_path = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Analyze Content")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)

        # --- Text input area ---
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Paste text here, or upload a PDF / image / video file below..."
        )
        self.text_input.setFixedHeight(120)
        main_layout.addWidget(self.text_input)

        # --- File upload row ---
        file_row = QHBoxLayout()
        upload_button = QPushButton("Upload File (PDF / Image / Video)")
        upload_button.clicked.connect(self.on_upload_clicked)
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        file_row.addWidget(upload_button)
        file_row.addWidget(self.file_label)
        file_row.addStretch()
        main_layout.addLayout(file_row)

        # --- Analyze button ---
        analyze_button = QPushButton("Analyze")
        analyze_button.setStyleSheet(
            "font-weight: bold; padding: 8px; background-color: #1565c0; color: white;"
        )
        analyze_button.clicked.connect(self.on_analyze_clicked)
        main_layout.addWidget(analyze_button)

        # --- Status / error label ---
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #c62828;")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        # --- Scrollable results area ---
        self.results_container = QVBoxLayout()
        results_widget = QWidget()
        results_widget.setLayout(self.results_container)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(results_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

    def on_upload_clicked(self):
        """Opens a file picker restricted to our supported file types."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file to analyze",
            "",
            "Supported Files (*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.mp4 *.mov *.avi *.mkv)",
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.file_label.setStyleSheet("color: black;")

    def on_analyze_clicked(self):
        """
        Main action: figures out the input source, extracts text,
        runs the ML engine, displays results, and saves to history.
        """
        self.status_label.setText("")
        self._clear_results()

        # Decide the input source: an uploaded file takes priority
        # over typed text if both are present.
        if self.selected_file_path:
            source = self.selected_file_path
            input_type = self._detect_input_type(self.selected_file_path)
        else:
            source = self.text_input.toPlainText().strip()
            input_type = "text"

        if not source:
            self.status_label.setText("Please type some text or upload a file first.")
            return

        # Step 1: extract plain text (Module A)
        try:
            extracted_text = extract_text(source)
        except InputExtractionError as error:
            self.status_label.setText(f"Could not analyze this input: {error}")
            return

        if not extracted_text.strip():
            self.status_label.setText("No text could be found in this input.")
            return

        # Step 2 + 3: run the ML engine (cleans text internally, then predicts)
        result = self.ml_engine.predict(extracted_text)

        # Step 4: save to history (Module C)
        save_analysis(input_type, result)

        # Step 5: display results
        self._render_results(result)

    def _detect_input_type(self, file_path: str) -> str:
        """Maps a file extension to the input_type string we store in the database."""
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return "pdf"
        if extension in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            return "image"
        if extension in (".mp4", ".mov", ".avi", ".mkv"):
            return "video"
        return "text"

    def _clear_results(self):
        """Removes all widgets from the results area before showing a new result."""
        while self.results_container.count():
            item = self.results_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_results(self, result: dict):
        """Builds and displays the results panel for one MLEngine.predict() result."""
        overview = QLabel(
            f"<b>Credibility Score:</b> {result['credibility_score']}/100 &nbsp;&nbsp; "
            f"<b>Overall Risk Level:</b> "
            f"<span style='color:{RISK_COLORS.get(result['overall_risk_level'], 'black')}'>"
            f"{result['overall_risk_level']}</span>"
        )
        overview.setStyleSheet("font-size: 15px; padding: 6px;")
        self.results_container.addWidget(overview)

        if not result["flagged_categories"]:
            safe_label = QLabel("No spam, phishing, clickbait, toxic, or fake-review patterns detected.")
            safe_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 6px;")
            self.results_container.addWidget(safe_label)
            return

        for category in result["flagged_categories"]:
            category_result = result["categories"][category]
            self.results_container.addWidget(self._build_category_card(category, category_result))

    def _build_category_card(self, category: str, category_result: dict) -> QFrame:
        """Builds a small bordered panel showing one flagged category's details."""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 6px; padding: 8px; }")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        layout = QVBoxLayout()

        risk_color = RISK_COLORS.get(category_result["risk_level"], "black")
        header = QLabel(
            f"<b>{CATEGORY_DISPLAY_NAMES.get(category, category.title())}</b> — "
            f"{category_result['confidence']}% confidence — "
            f"<span style='color:{risk_color}'>{category_result['risk_level']} risk</span>"
        )
        layout.addWidget(header)

        if category_result["top_words"]:
            highlighted = ", ".join(
                f"<span style='background-color:#ffe082;'>{word}</span>"
                for word in category_result["top_words"]
            )
            words_label = QLabel(f"Influencing words: {highlighted}")
            words_label.setTextFormat(Qt.RichText)
            layout.addWidget(words_label)

        explanation_label = QLabel(category_result["explanation"])
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("color: #444;")
        layout.addWidget(explanation_label)

        card.setLayout(layout)
        return card
