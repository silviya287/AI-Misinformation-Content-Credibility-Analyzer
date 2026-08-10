"""
core/database.py

This file owns ALL SQLite database code for the app (per the finalized
architecture, Module C - Database & Reports).

In Milestone 1, we only need one function: init_db().
It creates the database file (if it doesn't already exist) and creates
the two tables the whole team agreed on:
    - analysis_history
    - settings

Later milestones (Module C) will add functions like save_analysis(),
get_history(), delete_analysis(), get_setting(), save_setting() to this
same file. We are NOT building those yet - Milestone 1 is just the
skeleton so the rest of the team has a working database to build against.
"""

import sqlite3
import os
import sys

# The database file needs to live in a PERSISTENT location - not
# wherever this script happens to be running from. This matters a lot
# once the app is packaged into a standalone executable: PyInstaller
# (in "onefile" mode) extracts everything into a temporary folder that
# gets deleted when the app closes. If we stored the database there,
# every restart would silently wipe the user's entire history.
#
# So: when running as a packaged executable, we store the database
# next to the .exe itself (a folder that persists between runs).
# When running normally as a Python script (during development), we
# use the project root, same as before.
if getattr(sys, "frozen", False):
    _base_dir = os.path.dirname(sys.executable)
else:
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(_base_dir, "app_data.db")


def init_db():
    """
    Creates the SQLite database file and the two required tables,
    if they don't already exist.

    This is safe to call every time the app starts - "CREATE TABLE IF
    NOT EXISTS" means it will never wipe out existing data.
    """

    # sqlite3.connect() will create the .db file automatically if it
    # doesn't exist yet.
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # Table 1: stores every analysis a user runs (text/pdf/image/video).
    # This matches the finalized database design from the architecture doc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            credibility_score REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)

    # Table 2: simple key-value store for app settings (theme, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Save changes to disk and close the connection cleanly.
    connection.commit()
    connection.close()

    print(f"[database] Database ready at: {DB_PATH}")


def _get_connection():
    """
    Small internal helper so every function below opens the database
    the same way (and always via DB_PATH), instead of repeating this
    line in every function.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row  # lets us access columns by name
    return connection


def save_analysis(input_type: str, ml_result: dict) -> int:
    """
    Saves one analysis result to the analysis_history table.

    `ml_result` is the dictionary returned by core.ml_engine.MLEngine.predict().
    Since the app can flag MULTIPLE categories at once but our table has
    a single `prediction` column, we summarize:
        - prediction: comma-separated flagged category names, or "Safe"
          if nothing was flagged
        - confidence: the highest confidence among flagged categories
          (0 if nothing was flagged)
        - credibility_score / risk_level: taken directly from the
          engine's overall scores

    Returns the new row's id (useful if the UI wants to jump straight
    to viewing this result).
    """
    flagged_categories = ml_result["flagged_categories"]

    if flagged_categories:
        prediction = ", ".join(
            category.replace("_", " ").title() for category in flagged_categories
        )
        confidence = max(
            ml_result["categories"][category]["confidence"] for category in flagged_categories
        )
    else:
        prediction = "Safe"
        confidence = 0.0

    connection = _get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO analysis_history
            (input_type, prediction, confidence, credibility_score, risk_level)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            input_type,
            prediction,
            confidence,
            ml_result["credibility_score"],
            ml_result["overall_risk_level"],
        ),
    )
    connection.commit()
    new_id = cursor.lastrowid
    connection.close()
    return new_id


def get_history(limit: int = None) -> list[dict]:
    """
    Returns all saved analyses, most recent first, as a list of plain
    dictionaries (easy for the UI's table widget to loop over).

    `limit`: optionally cap how many rows come back (e.g. for a
    "recent activity" preview). None means "return everything".
    """
    connection = _get_connection()
    cursor = connection.cursor()

    query = "SELECT * FROM analysis_history ORDER BY analysis_date DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        cursor.execute(query, (limit,))
    else:
        cursor.execute(query)

    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()
    return rows


def delete_analysis(analysis_id: int) -> bool:
    """
    Deletes one analysis record by id.
    Returns True if a row was actually deleted, False if no row with
    that id existed (so the UI can show "already deleted" if needed).
    """
    connection = _get_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
    connection.commit()
    deleted = cursor.rowcount > 0
    connection.close()
    return deleted


def get_setting(key: str, default: str = None) -> str:
    """
    Returns the saved value for `key`, or `default` if that setting
    has never been saved. Values are always stored/returned as strings
    (e.g. "light"/"dark") - convert on the calling side if you need a
    different type.
    """
    connection = _get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    connection.close()

    return row["value"] if row is not None else default


def save_setting(key: str, value: str) -> None:
    """
    Saves (or updates) a single setting. "INSERT OR REPLACE" means this
    works whether the key already exists or not - no need to check
    first.
    """
    connection = _get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value),
    )
    connection.commit()
    connection.close()


# This lets a teammate test this file directly by running:
#   python core/database.py
# without needing to launch the whole app.
if __name__ == "__main__":
    init_db()
