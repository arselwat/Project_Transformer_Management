# core/transformer/store.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os, csv
import pandas as pd

BASE = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
BASE.mkdir(parents=True, exist_ok=True)
CSV_PATH = (BASE / "transformers.csv").resolve()

COLS = [
    "equipment_code","name","site",
    "rated_mva","V1n_kV","V2n_kV","f_nominal","vector_group",
    "status","commissioned_on","notes"
]

def _ensure_header():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=COLS).writeheader()

def _read_df() -> pd.DataFrame:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(CSV_PATH, dtype=str)
    except Exception:
        df = pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns: df[c] = ""
    def _num(x):
        try: return float(str(x).replace(",", "."))
        except Exception: return float("nan")
    for c in ("rated_mva","V1n_kV","V2n_kV","f_nominal"):
        df[c] = df[c].apply(_num)
    df["equipment_code"] = df["equipment_code"].astype(str).str.strip()
    def _stat(s):
        s = str(s).strip().lower()
        return "retired" if s in ("retired","inactive","1","true","oui") else "active"
    df["status"] = df["status"].apply(_stat)
    df["name"]  = df["name"].fillna("").astype(str)
    df["site"]  = df["site"].fillna("").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    df = df[COLS].drop_duplicates(subset=["equipment_code"]).reset_index(drop=True)
    return df

def _write_df(df: pd.DataFrame):
    _ensure_header()
    df = df[COLS].copy()
    df.to_csv(CSV_PATH, index=False)

def list_transformers(include_retired: bool = True) -> List[Dict]:
    d = _read_df()
    if not include_retired:
        d = d[d["status"] == "active"]
    return d.to_dict("records")

def get_transformer(code: str) -> Optional[Dict]:
    if not code: return None
    d = _read_df()
    m = d[d["equipment_code"].astype(str) == str(code)]
    return (m.iloc[0].to_dict() if not m.empty else None)

def upsert_transformer(rec: Dict) -> Tuple[bool, str]:
    d = _read_df()
    r = {k: rec.get(k, "") for k in COLS}
    r["equipment_code"] = str(r["equipment_code"]).strip()
    if not r["equipment_code"]:
        return False, "equipment_code est obligatoire"
    for c in ("rated_mva","V1n_kV","V2n_kV","f_nominal"):
        try: r[c] = float(str(r[c]).replace(",", ".")) if str(r[c]).strip() != "" else ""
        except Exception: r[c] = ""
    r["status"] = "retired" if str(r["status"]).strip().lower() in ("retired","inactive","1","true","oui") else "active"
    idx = d.index[d["equipment_code"] == r["equipment_code"]]
    if len(idx) > 0:
        for k, v in r.items(): d.at[idx[0], k] = v
        msg = f"updated: {r['equipment_code']}"
    else:
        d = pd.concat([d, pd.DataFrame([r])], ignore_index=True)
        msg = f"inserted: {r['equipment_code']}"
    _write_df(d)
    return True, msg

def delete_transformer(code: str) -> Tuple[bool, str]:
    d = _read_df()
    before = len(d)
    d = d[d["equipment_code"].astype(str) != str(code)]
    if len(d) == before:
        return False, f"introuvable: {code}"
    _write_df(d)
    return True, f"deleted: {code}"

def set_status(code: str, active: bool) -> Tuple[bool, str]:
    d = _read_df()
    m = d["equipment_code"].astype(str) == str(code)
    if not m.any():
        return False, f"introuvable: {code}"
    d.loc[m, "status"] = "active" if active else "retired"
    _write_df(d)
    return True, f"status set to {'active' if active else 'retired'} for {code}"
