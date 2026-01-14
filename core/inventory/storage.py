# core/inventory/storage.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import csv, time
import os

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("FS_DATA_DIR", BASE_DIR / "data")).resolve()
DATA_DIR.mkdir(exist_ok=True, parents=True)

PARTS_CSV = (DATA_DIR / "inventory_parts.csv").resolve()
MOVES_CSV = (DATA_DIR / "stock_movements.csv").resolve()  # ✅ aligné avec services.py

PARTS_COLUMNS = [
    "code","nom","famille","quantite_dispo","seuil_min",
    "localisation","prix_unitaire","fournisseur"
]

# ✅ aligné (task_id)
MOVES_COLUMNS = ["ts","type","code","qty","reason","task_id","ref","user"]

def _ensure_files() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    if not PARTS_CSV.exists() or PARTS_CSV.stat().st_size == 0:
        with open(PARTS_CSV, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=PARTS_COLUMNS).writeheader()

    if not MOVES_CSV.exists() or MOVES_CSV.stat().st_size == 0:
        with open(MOVES_CSV, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=MOVES_COLUMNS).writeheader()

def load_parts() -> List[Dict[str, Any]]:
    _ensure_files()
    out: List[Dict[str, Any]] = []
    with open(PARTS_CSV, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = {k: row.get(k) for k in PARTS_COLUMNS}
            d["quantite_dispo"] = float(d.get("quantite_dispo") or 0)
            d["seuil_min"] = float(d.get("seuil_min") or 0)
            d["prix_unitaire"] = float(d.get("prix_unitaire") or 0)
            out.append(d)
    return out

def save_parts(rows: List[Dict[str, Any]]) -> None:
    _ensure_files()
    with open(PARTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARTS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "code": str(r.get("code","")).strip(),
                "nom": r.get("nom",""),
                "famille": r.get("famille",""),
                "quantite_dispo": float(r.get("quantite_dispo") or 0),
                "seuil_min": float(r.get("seuil_min") or 0),
                "localisation": r.get("localisation",""),
                "prix_unitaire": float(r.get("prix_unitaire") or 0),
                "fournisseur": r.get("fournisseur",""),
            })

def append_movement(m: Dict[str, Any]) -> None:
    _ensure_files()
    row = {
        "ts": float(m.get("ts") or time.time()),
        "type": str(m.get("type","")).upper(),
        "code": str(m.get("code","")).strip(),
        "qty": float(m.get("qty") or 0),
        "reason": m.get("reason",""),
        "task_id": str(m.get("task_id","") or ""),
        "ref": m.get("ref",""),
        "user": m.get("user",""),
    }
    with open(MOVES_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MOVES_COLUMNS)
        w.writerow(row)

def load_movements(limit: int | None = None) -> List[Dict[str, Any]]:
    _ensure_files()
    rows: List[Dict[str, Any]] = []
    with open(MOVES_CSV, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["ts"] = float(row.get("ts") or 0)
            row["qty"] = float(row.get("qty") or 0)
            rows.append(row)
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return rows
