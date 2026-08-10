"""
tests/test_database.py

Tests for the Milestone 6 additions to core/database.py:
save_analysis(), get_history(), delete_analysis().

Run with:
    pytest tests/test_database.py -v

These tests point the database module at a temporary file instead of
the real app_data.db, so running tests never touches your real history.
"""

import os
import pytest

import core.database as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirects DB_PATH to a temporary file for the duration of one test."""
    temp_db_path = os.path.join(tmp_path, "test_app_data.db")
    monkeypatch.setattr(db, "DB_PATH", temp_db_path)
    db.init_db()
    return temp_db_path


def _fake_ml_result(flagged_categories, category_confidences, credibility_score, risk_level):
    """Builds a minimal fake MLEngine.predict()-shaped dict for testing save_analysis()."""
    categories = {
        category: {"confidence": category_confidences[category]}
        for category in flagged_categories
    }
    return {
        "flagged_categories": flagged_categories,
        "categories": categories,
        "credibility_score": credibility_score,
        "overall_risk_level": risk_level,
    }


def test_save_and_get_flagged_analysis(temp_db):
    """A flagged result should be saved with the right prediction/confidence."""
    fake_result = _fake_ml_result(
        flagged_categories=["spam", "clickbait"],
        category_confidences={"spam": 91.0, "clickbait": 70.0},
        credibility_score=19.5,
        risk_level="High",
    )
    new_id = db.save_analysis("text", fake_result)
    assert new_id is not None

    history = db.get_history()
    assert len(history) == 1
    row = history[0]
    assert row["input_type"] == "text"
    assert row["prediction"] == "Spam, Clickbait"
    assert row["confidence"] == 91.0  # the higher of the two flagged confidences
    assert row["credibility_score"] == 19.5
    assert row["risk_level"] == "High"


def test_save_safe_analysis(temp_db):
    """A result with nothing flagged should save as 'Safe' with 0 confidence."""
    fake_result = _fake_ml_result(
        flagged_categories=[],
        category_confidences={},
        credibility_score=100.0,
        risk_level="Low",
    )
    db.save_analysis("text", fake_result)

    history = db.get_history()
    assert history[0]["prediction"] == "Safe"
    assert history[0]["confidence"] == 0.0


def test_get_history_most_recent_first(temp_db):
    """get_history() should return newest rows first."""
    fake_result = _fake_ml_result([], {}, 100.0, "Low")
    first_id = db.save_analysis("text", fake_result)
    second_id = db.save_analysis("pdf", fake_result)

    history = db.get_history()
    assert history[0]["id"] == second_id
    assert history[1]["id"] == first_id


def test_delete_analysis(temp_db):
    """delete_analysis() should remove the row and return True; repeat calls return False."""
    fake_result = _fake_ml_result([], {}, 100.0, "Low")
    new_id = db.save_analysis("text", fake_result)

    assert db.delete_analysis(new_id) is True
    assert db.get_history() == []
    assert db.delete_analysis(new_id) is False  # already gone


def test_get_setting_returns_default_when_not_saved(temp_db):
    """An unset setting should return the given default, not crash."""
    assert db.get_setting("theme", default="light") == "light"
    assert db.get_setting("nonexistent_key") is None


def test_save_and_get_setting(temp_db):
    """A saved setting should be retrievable afterward."""
    db.save_setting("theme", "dark")
    assert db.get_setting("theme") == "dark"


def test_save_setting_overwrites_existing_value(temp_db):
    """Saving the same key twice should update it, not create a duplicate row."""
    db.save_setting("theme", "light")
    db.save_setting("theme", "dark")
    assert db.get_setting("theme") == "dark"

    connection = db._get_connection()
    count = connection.execute("SELECT COUNT(*) FROM settings WHERE key = 'theme'").fetchone()[0]
    connection.close()
    assert count == 1
