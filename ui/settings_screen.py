"""
ui/settings_screen.py

Module D - User Interface (Milestone 7).

Lets the user switch between Light and Dark theme. The choice is saved
to the database (via core.database.get_setting/save_setting) so it's
remembered the next time the app is opened.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Signal

from core.database import get_setting, save_setting
from ui.theme import apply_theme


class SettingsScreen(QWidget):
    # Emitted whenever the user picks a new theme, so MainWindow (or
    # anything else interested) can react if it needs to. Right now we
    # apply the theme directly here too, but the signal keeps this
    # screen decoupled from anyone else who might care later.
    theme_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        theme_label = QLabel("Theme:")
        layout.addWidget(theme_label)

        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark"])

        # Load the previously saved theme (defaulting to "light" if
        # this is the first time the app has ever been run).
        saved_theme = get_setting("theme", default="light")
        self.theme_selector.setCurrentText(saved_theme.capitalize())

        self.theme_selector.currentTextChanged.connect(self.on_theme_changed)
        layout.addWidget(self.theme_selector)

        layout.addStretch()
        self.setLayout(layout)

    def on_theme_changed(self, selected_text: str):
        """Saves the new theme choice and applies it immediately."""
        theme_name = selected_text.lower()  # "Light" -> "light", "Dark" -> "dark"
        save_setting("theme", theme_name)
        apply_theme(theme_name)
        self.theme_changed.emit(theme_name)
