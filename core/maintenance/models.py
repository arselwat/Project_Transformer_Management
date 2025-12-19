import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inventory.db"

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # pratique: dict(row)
    conn.execute("PRAGMA foreign_keys = ON;")

    with conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_task (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_code TEXT NOT NULL,
                title TEXT NOT NULL,
                periodicity_days INTEGER DEFAULT 0,
                next_due_date TEXT,
                last_done_date TEXT,
                status TEXT DEFAULT 'ACTIVE'
            )"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_template (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                group_name TEXT DEFAULT 'Préventif',
                periodicity_days INTEGER DEFAULT 0,
                default_enabled INTEGER DEFAULT 1
            )"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_threshold (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_key TEXT UNIQUE NOT NULL,
                unit TEXT,
                rule TEXT,
                min_val REAL,
                max_val REAL,
                ref_text TEXT
            )"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                equipment_code TEXT NOT NULL,
                test_key TEXT NOT NULL,
                measured REAL,
                unit TEXT,
                status TEXT,
                comment TEXT,
                operator TEXT
            )"""
        )

    return conn
