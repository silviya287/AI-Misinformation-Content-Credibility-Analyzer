"""
ui/main_window.py

This is the navigation shell of the app (Module D - User Interface).
It creates the main window and puts all 5 screens into tabs:
    Home | Analyze | History | Reports | Settings

The Home tab is simple enough that we build it directly here instead
of creating a separate home_screen.py file - this keeps the file count
low, per the simplified architecture.

The other 4 tabs are separate screen classes (in their own files)
because they will each grow more complex in later milestones.
"""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
)
from PySide6.QtCore import Qt

from ui.analyze_screen import AnalyzeScreen
from ui.history_screen import HistoryScreen
from ui.reports_screen import ReportsScreen
from ui.settings_screen import SettingsScreen


def build_home_tab() -> QWidget:
    """
    Builds the simple Home/welcome tab.
    Kept as a small function (not its own file) since it's just a
    static welcome message for now.
    """
    home_widget = QWidget()
    layout = QVBoxLayout()

    title = QLabel("AI Content Trust & Spam Detection System")
    title.setStyleSheet("font-size: 22px; font-weight: bold;")
    title.setAlignment(Qt.AlignCenter)

    subtitle = QLabel(
        "Analyze text, PDFs, images, and videos for spam, phishing, "
        "clickbait, toxic content, and fake reviews.\n\n"
        "Use the tabs above to get started."
    )
    subtitle.setAlignment(Qt.AlignCenter)
    subtitle.setWordWrap(True)

    layout.addStretch()
    layout.addWidget(title)
    layout.addWidget(subtitle)
    layout.addStretch()

    home_widget.setLayout(layout)
    return home_widget


class MainWindow(QMainWindow):
    """
    The main application window.
    Holds a QTabWidget with all 5 screens as tabs.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Content Trust & Spam Detection System")
        self.resize(900, 600)

        # QTabWidget is the simplest way in PySide6 to give us
        # navigation between screens without building a custom sidebar.
        tabs = QTabWidget()

        tabs.addTab(build_home_tab(), "Home")
        tabs.addTab(AnalyzeScreen(), "Analyze")

        self.history_screen = HistoryScreen()
        tabs.addTab(self.history_screen, "History")

        tabs.addTab(ReportsScreen(), "Reports")
        tabs.addTab(SettingsScreen(), "Settings")

        # Refresh the History table every time the user switches to it,
        # so a new analysis from the Analyze tab shows up immediately
        # without needing a manual "Refresh" click.
        tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs = tabs

        # QMainWindow needs a "central widget" - that's the tabs widget.
        self.setCentralWidget(tabs)

    def _on_tab_changed(self, index: int):
        """If the History tab was just selected, refresh its table."""
        if self.tabs.widget(index) is self.history_screen:
            self.history_screen.refresh()
