import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inventory.db"

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS part (
            code TEXT PRIMARY KEY,
            nom TEXT,
            quantite INTEGER,
            localisation TEXT,
            seuil_min INTEGER,
            fournisseur TEXT,
            prix_unitaire REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS movement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_part TEXT,
            type TEXT,
            quantite INTEGER,
            date TEXT,
            ref_operation TEXT,
            FOREIGN KEY(code_part) REFERENCES part(code)
        )"""
    )
    return conn
