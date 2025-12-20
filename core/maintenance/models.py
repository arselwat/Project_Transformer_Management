# core/maintenance/models.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# ---------------------------------------------------------
# Streamlit Cloud persistant:
# - si tu as activé le "Persistent storage", le chemin est /mount/data
# - sinon fallback sur ./data (volatile)
# ---------------------------------------------------------
PERSIST_DIR = os.getenv("STREAMLIT_PERSIST_DIR", "/mount/data")
DEFAULT_DIR = Path(PERSIST_DIR) if Path(PERSIST_DIR).exists() else (Path(__file__).resolve().parent.parent.parent / "data")
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("MAINTENANCE_DB_PATH", str(DEFAULT_DIR / "inventory.db")))

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    with conn:
        # --- pm_task (tâches planifiées) ---
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_task (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_code TEXT NOT NULL,
                title TEXT NOT NULL,
                periodicity_days INTEGER DEFAULT 0,
                next_due_date TEXT,
                last_done_date TEXT,
                status TEXT DEFAULT 'ACTIVE',

                -- champs "bridge optimisation"
                maintenance_type TEXT,
                interval_h REAL,
                source_hash TEXT,
                updated_at TEXT
            )"""
        )

        # clé fonctionnelle : éviter doublons
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS ux_pm_task_equipment_title
               ON pm_task(equipment_code, title)"""
        )

        # --- templates ---
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pm_template (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                group_name TEXT DEFAULT 'Préventif',
                periodicity_days INTEGER DEFAULT 0,
                default_enabled INTEGER DEFAULT 1
            )"""
        )

        # --- seuils ---
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

        # --- résultats ---
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

        # ---------------------------------------------------------
        # Migration "safe" (si table existe déjà mais sans colonnes)
        # ---------------------------------------------------------
        def _has_col(table: str, col: str) -> bool:
            cur = conn.execute(f"PRAGMA table_info({table})")
            cols = [r["name"] for r in cur.fetchall()]
            return col in cols

        # Ajouter colonnes si absentes (anciens DB)
        for col, ddl in [
            ("maintenance_type", "ALTER TABLE pm_task ADD COLUMN maintenance_type TEXT"),
            ("interval_h", "ALTER TABLE pm_task ADD COLUMN interval_h REAL"),
            ("source_hash", "ALTER TABLE pm_task ADD COLUMN source_hash TEXT"),
            ("updated_at", "ALTER TABLE pm_task ADD COLUMN updated_at TEXT"),
        ]:
            if not _has_col("pm_task", col):
                conn.execute(ddl)

    return conn
