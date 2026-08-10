"""
tests/test_theme.py

Tests for ui/theme.py (Milestone 7).

Run with:
    pytest tests/test_theme.py -v
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from ui.theme import apply_theme, LIGHT_QSS, DARK_QSS


def _get_app():
    """PySide6 needs exactly one QApplication - reuse it if a test already made one."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_apply_light_theme_clears_stylesheet():
    app = _get_app()
    apply_theme("dark")  # start from dark so we can prove light actually changes it
    apply_theme("light")
    assert app.styleSheet() == LIGHT_QSS


def test_apply_dark_theme_sets_dark_stylesheet():
    app = _get_app()
    apply_theme("dark")
    assert app.styleSheet() == DARK_QSS


def test_unknown_theme_name_falls_back_to_light():
    app = _get_app()
    apply_theme("dark")
    apply_theme("some_theme_that_does_not_exist")
    assert app.styleSheet() == LIGHT_QSS
