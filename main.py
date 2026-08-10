"""
main.py

This is the entry point of the whole application.
Running "python main.py" is how anyone (teammates, graders) starts the app.

What it does, in order:
    1. Initializes the SQLite database (creates the file + tables if
       they don't exist yet).
    2. Creates the PySide6 QApplication (required by every PySide6 app).
    3. Creates and shows the MainWindow (the 5-tab navigation shell).
    4. Starts the Qt event loop, which keeps the app running and
       responsive until the user closes the window.
"""

import sys
from PySide6.QtWidgets import QApplication

from core.database import init_db, get_setting
from ui.main_window import MainWindow
from ui.theme import apply_theme


def main():
    # Step 1: make sure the database and its tables exist before the
    # UI opens. Every later milestone that reads/writes the DB depends
    # on this having already run.
    init_db()

    # Step 2: every PySide6 app needs exactly one QApplication instance.
    app = QApplication(sys.argv)

    # Step 2b: apply whichever theme the user picked last time (defaults
    # to "light" the very first time the app is ever run).
    saved_theme = get_setting("theme", default="light")
    apply_theme(saved_theme)

    # Step 3: build and display the main window.
    window = MainWindow()
    window.show()

    # Step 4: start the event loop. sys.exit() ensures the app closes
    # with the correct exit code when the window is closed.
    sys.exit(app.exec())


# This check means "only run main() if this file is being run directly
# (python main.py), not if it's imported by another file."
if __name__ == "__main__":
    main()
