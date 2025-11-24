import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inventory.db"

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Tâches planifiées (préventif)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pm_task (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_code TEXT,
            title TEXT,
            periodicity_days INTEGER,   -- 0 si ponctuel
            next_due_date TEXT,         -- ISO YYYY-MM-DD
            last_done_date TEXT,        -- ISO YYYY-MM-DD
            status TEXT DEFAULT 'ACTIVE' -- ACTIVE | SUSPENDED
        )"""
    )
    # Modèles (génèrent des pm_task)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pm_template (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,                 -- ex: 'Inspection visuelle'
            group_name TEXT,            -- ex: 'Préventif'
            periodicity_days INTEGER,   -- 30, 180, 365, 0...
            default_enabled INTEGER DEFAULT 1
        )"""
    )
    # Seuils de référence (tests)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pm_threshold (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_key TEXT UNIQUE,       -- ex: 'resistance_isolement'
            unit TEXT,                  -- ex: 'MΩ'
            rule TEXT,                  -- ex: '>=', '<=', 'range'
            min_val REAL,
            max_val REAL,
            ref_text TEXT               -- texte d'aide
        )"""
    )
    # Résultats de tests
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pm_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,                  -- ISO date
            equipment_code TEXT,
            test_key TEXT,              -- référence à pm_threshold.test_key
            measured REAL,
            unit TEXT,
            status TEXT,                -- 'OK' / 'NOK'
            comment TEXT,
            operator TEXT
        )"""
    )
    return conn
