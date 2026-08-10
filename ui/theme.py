"""
ui/theme.py

Module D - User Interface (Milestone 7).

Two simple Qt stylesheets (QSS - Qt's version of CSS) for a light and
dark theme, plus one helper to apply either of them to the whole app.
Kept as plain strings in one small file rather than a styles/ folder
with separate .qss files - simple enough not to need its own folder.
"""

from PySide6.QtWidgets import QApplication

LIGHT_QSS = ""  # Qt's default look is already a clean light theme - nothing to override.

DARK_QSS = """
    QWidget {
        background-color: #2b2b2b;
        color: #e0e0e0;
    }
    QTextEdit, QTableWidget, QLineEdit {
        background-color: #3c3c3c;
        color: #e0e0e0;
        border: 1px solid #555;
    }
    QPushButton {
        background-color: #4a4a4a;
        color: #e0e0e0;
        border: 1px solid #666;
        padding: 6px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #5a5a5a;
    }
    QTabWidget::pane {
        border: 1px solid #555;
    }
    QTabBar::tab {
        background: #3c3c3c;
        color: #e0e0e0;
        padding: 8px;
    }
    QTabBar::tab:selected {
        background: #555;
    }
    QHeaderView::section {
        background-color: #3c3c3c;
        color: #e0e0e0;
        padding: 4px;
    }
    QFrame {
        border-color: #555;
    }
"""

THEMES = {
    "light": LIGHT_QSS,
    "dark": DARK_QSS,
}


def apply_theme(theme_name: str) -> None:
    """
    Applies the given theme ("light" or "dark") to the whole running
    application. Falls back to light if an unrecognized name is given.
    """
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(THEMES.get(theme_name, LIGHT_QSS))
