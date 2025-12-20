# core/maintenance/bridge.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

from core.maintenance import services as pm
from core.datahub import get_failures_meta  # pour dataset_hash

@dataclass
class BridgeParams:
    min_days: int = 7
    only_if_preventive: bool = True

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v): 
            return default
        return v
    except Exception:
        return default

def _is_preventive(maintenance_type: str | None) -> bool:
    if not maintenance_type:
        return False
    return "préventive" in str(maintenance_type).lower()

def upsert_tasks_from_optimization(
    opt_df: pd.DataFrame,
    start_date: Optional[str] = None,
    params: BridgeParams = BridgeParams(),
) -> Dict[str, Any]:

    if opt_df is None or opt_df.empty:
        return {"ok": False, "created": 0, "updated": 0, "skipped": 0, "errors": ["opt_df vide"]}

    df = opt_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "equipment_code" not in df.columns:
        return {"ok": False, "created": 0, "updated": 0, "skipped": 0, "errors": ["equipment_code manquant"]}

    # intervalle choisi
    interval_col = None
    for c in ["T_recommended_h", "T_R_h", "T_cost_h"]:
        if c in df.columns:
            interval_col = c
            break
    if interval_col is None:
        return {"ok": False, "created": 0, "updated": 0, "skipped": 0, "errors": ["Aucune colonne d’intervalle (h)"]}

    # dataset hash (traçabilité)
    meta = get_failures_meta()
    ds_hash = meta.get("hash") if meta.get("ok") else None

    base = date.fromisoformat(start_date) if start_date else date.today()

    created = updated = skipped = 0
    errors: List[str] = []

    existing = pm.list_tasks() or []
    # clé : (equipment_code, title, source)
    key_map = {(str(t.get("equipment_code")), str(t.get("title")), str(t.get("source","MANUAL"))): t for t in existing}

    for _, r in df.iterrows():
        eq = str(r.get("equipment_code") or "").strip()
        if not eq:
            skipped += 1
            continue

        mtype = str(r.get("maintenance_type") or "").strip() or None

        # si option “only preventive”
        if params.only_if_preventive and not _is_preventive(mtype):
            skipped += 1
            continue

        interval_h = _safe_float(r.get(interval_col), 0.0)
        if interval_h <= 0:
            skipped += 1
            continue

        per_days = max(int(params.min_days), int(round(interval_h / 24.0)))
        next_due = (base + timedelta(days=per_days)).isoformat()

        title = "Maintenance (issue de l’optimisation)"
        src = "OPTIMISATION"

        old = key_map.get((eq, title, src))
        payload = {
            "equipment_code": eq,
            "title": title,
            "periodicity_days": per_days,
            "next_due_date": next_due,
            "last_done_date": old.get("last_done_date") if old else None,
            "status": old.get("status","ACTIVE") if old else "ACTIVE",
            "source": src,
            "maintenance_type": mtype,
            "opt_interval_h": interval_h,
            "dataset_hash": ds_hash,
        }

        try:
            if old and old.get("id"):
                payload["id"] = int(old["id"])
                pm.upsert_task(payload)
                updated += 1
            else:
                pm.upsert_task(payload)
                created += 1
        except Exception as e:
            errors.append(f"{eq}: {e}")

    ok = len(errors) == 0
    return {"ok": ok, "created": created, "updated": updated, "skipped": skipped, "errors": errors, "interval_col": interval_col}
