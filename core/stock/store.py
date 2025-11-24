from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os, csv

DATA_DIR = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SPARES_CSV = DATA_DIR / "spares.csv"

def _ensure_header():
    if not SPARES_CSV.exists() or SPARES_CSV.stat().st_size == 0:
        with open(SPARES_CSV, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=["code","name","qty","unit","min_qty"]).writeheader()

def list_spares() -> List[dict]:
    if not SPARES_CSV.exists():
        _ensure_header()
        return []
    rows = []
    with open(SPARES_CSV, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def upsert_spare(code: str, name: str, qty: float, unit: str = "pcs", min_qty: float = 0.0) -> Tuple[bool,str]:
    _ensure_header()
    rows = list_spares()
    found = False
    for r in rows:
        if r["code"] == code:
            r["name"] = name; r["qty"] = str(qty); r["unit"] = unit; r["min_qty"] = str(min_qty)
            found = True; break
    if not found:
        rows.append({"code":code, "name":name, "qty":str(qty), "unit":unit, "min_qty":str(min_qty)})
    with open(SPARES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code","name","qty","unit","min_qty"])
        w.writeheader(); w.writerows(rows)
    return True, "OK"

def change_stock(code: str, delta: float) -> Tuple[bool,str]:
    rows = list_spares()
    found = False
    for r in rows:
        if r["code"] == code:
            new_qty = float(r.get("qty","0") or 0) + float(delta)
            r["qty"] = f"{new_qty:.3f}"
            found = True
            break
    if not found:
        return False, "Article introuvable"
    with open(SPARES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code","name","qty","unit","min_qty"])
        w.writeheader(); w.writerows(rows)
    return True, "OK"
