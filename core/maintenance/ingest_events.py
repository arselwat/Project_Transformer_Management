# core/maintenance/ingest_events.py
from __future__ import annotations
import os, datetime as dt
from pathlib import Path
from typing import Dict, List
import pandas as pd

try:
    from core.inventory import services as inv
except Exception:
    inv = None

try:
    from core.inventory.recommendations import build_pm_kit_for_equipment
except Exception:
    def build_pm_kit_for_equipment(eq, beta, parts): return []

def _data_dir() -> Path:
    # même logique que pages/4_... (FS_DATA_DIR prioritaire)
    root = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root

EVT_CSV = (_data_dir() / "realtime_events.csv").resolve()
FAIL_CSV = (_data_dir() / "failures_saved.csv").resolve()

def _read_csv_flex(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # 1) essai classique
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        pass
    # 2) essai python engine + skip bad lines
    try:
        return pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def _beta_by_equipment() -> Dict[str, float]:
    from core.reliability.weibull import fit_weibull
    if not FAIL_CSV.exists():
        return {}
    try:
        df = pd.read_csv(FAIL_CSV)
    except Exception:
        return {}
    if "equipment_code" not in df.columns or "ttf_h" not in df.columns:
        return {}
    betas={}
    for eq, g in df.groupby("equipment_code"):
        x = pd.to_numeric(g["ttf_h"], errors="coerce").dropna().values
        if len(x) >= 3:
            try:
                ft = fit_weibull(x)
                betas[str(eq)] = float(ft.beta)
            except Exception:
                pass
    return betas

def _priority(level: str) -> str:
    level = (level or "").upper()
    if level == "ALARM": return "High"
    if level == "WARN":  return "Medium"
    return "Low"

def ingest_events_to_tasks(events_csv: str | None = None) -> Dict:
    p = Path(events_csv).resolve() if events_csv else EVT_CSV
    df = _read_csv_flex(p)
    if df.empty:
        return {"tasks": [], "kits_by_eq": {}, "events_loaded": 0, "marked_processed": 0}

    # Colonnes attendues — on comble ce qui manque
    for col in ["ts","site","equipment","level","code","msg","value","threshold","processed"]:
        if col not in df.columns:
            df[col] = "" if col != "processed" else 0

    # On ne traite que les nouvelles (processed != 1) — tolérance si processed est string
    df["processed"] = pd.to_numeric(df["processed"], errors="coerce").fillna(0).astype(int)
    df_new = df[df["processed"] != 1].copy()
    if df_new.empty:
        return {"tasks": [], "kits_by_eq": {}, "events_loaded": len(df), "marked_processed": 0}

    # Pièces et β
    parts = []
    if inv is not None:
        try:
            if hasattr(inv, "list_parts_as_dicts"):
                parts = inv.list_parts_as_dicts() or []
            elif hasattr(inv, "list_parts"):
                parts = list(inv.list_parts() or [])
        except Exception:
            parts = []
    betas = _beta_by_equipment()

    # Tâches + kits
    tasks: List[Dict] = []
    kits_by_eq: Dict[str, List[Dict]] = {}
    today = dt.date.today().isoformat()

    for _, r in df_new.iterrows():
        eq  = str(r.get("equipment", "")) or "UNKNOWN"
        lvl = str(r.get("level", "INFO"))
        code = str(r.get("code", "EVENT"))
        msg  = str(r.get("msg", ""))

        tasks.append({
            "equipment_code": eq,
            "title": f"[{lvl}] {code}",
            "priority": _priority(lvl),
            "due_date": today,
            "type": "corrective",
            "source": "realtime_event",
            "description": msg or f"Auto-généré depuis événement {code}",
        })

        if eq not in kits_by_eq:
            b = betas.get(eq, 1.3)
            try:
                kits_by_eq[eq] = build_pm_kit_for_equipment(eq, b, parts) or []
            except Exception:
                kits_by_eq[eq] = []

    # Marquage processed=1 et sauvegarde tolérante (conserve colonnes en place)
    df.loc[df_new.index, "processed"] = 1
    df.to_csv(p, index=False)

    return {
        "tasks": tasks,
        "kits_by_eq": kits_by_eq,
        "events_loaded": int(len(df)),
        "marked_processed": int(len(df_new)),
    }
