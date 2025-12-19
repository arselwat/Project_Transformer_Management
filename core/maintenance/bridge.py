# core/maintenance/bridge.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import math

from core.maintenance import services as pm


@dataclass
class BridgeParams:
    # seuil minimum en jours pour éviter des périodicités absurdes
    min_days: int = 7
    # plafond pour éviter des périodicités trop longues (tu peux changer)
    max_days: int = 365 * 5
    # si True, on ne crée des tâches que si maintenance_type est "préventive"
    only_if_preventive: bool = True


def _to_days(hours: Optional[float], *, min_days: int, max_days: int) -> Optional[int]:
    if hours is None:
        return None
    try:
        h = float(hours)
        if not math.isfinite(h) or h <= 0:
            return None
        d = int(round(h / 24.0))
        d = max(min_days, d)
        d = min(max_days, d)
        return d
    except Exception:
        return None


def _build_title(maintenance_type: str) -> str:
    """
    Titre stable (sert de "clé" fonctionnelle).
    On garde simple: un seul type de tâche par équipement provenant de l'optimisation.
    """
    mt = (maintenance_type or "").lower()
    if "condition" in mt or "inspection" in mt:
        return "Maintenance conditionnelle / inspection (optimisation)"
    if "corrective" in mt:
        return "Maintenance corrective (optimisation)"
    return "Maintenance préventive planifiée (optimisation)"


def _is_preventive(maintenance_type: str) -> bool:
    mt = (maintenance_type or "").lower()
    return ("préventive" in mt) or ("prevent" in mt)


def upsert_tasks_from_optimization(
    *,
    opt_df,  # DataFrame ou liste dict
    start_date: Optional[str] = None,
    params: Optional[BridgeParams] = None,
) -> Dict[str, Any]:
    """
    Entrée attendue (par équipement) :
      - equipment_code
      - T_recommended_h (priorité)
      - sinon T_R_h
      - sinon T_cost_h
      - maintenance_type

    Sortie : dict (résumé)
    """
    cfg = params or BridgeParams()

    # normaliser en liste de dict
    rows: List[Dict[str, Any]] = []
    if opt_df is None:
        return {"ok": False, "error": "opt_df vide", "created": 0, "updated": 0, "skipped": 0}
    if hasattr(opt_df, "to_dict"):
        rows = opt_df.to_dict("records")
    elif isinstance(opt_df, list):
        rows = opt_df
    else:
        return {"ok": False, "error": "opt_df format inconnu", "created": 0, "updated": 0, "skipped": 0}

    # date de base
    base = date.today()
    if start_date:
        try:
            base = date.fromisoformat(str(start_date))
        except Exception:
            pass

    created = 0
    updated = 0
    skipped = 0
    errors: List[str] = []

    # pour savoir si une tâche existe déjà : (equipment_code, title)
    existing = pm.list_tasks() or []
    key_to_id: Dict[Tuple[str, str], int] = {}
    for t in existing:
        try:
            key_to_id[(str(t.get("equipment_code")), str(t.get("title")))] = int(t.get("id"))
        except Exception:
            continue

    for r in rows:
        eq = str(r.get("equipment_code") or "").strip()
        if not eq:
            skipped += 1
            continue

        maintenance_type = str(r.get("maintenance_type") or "")
        title = _build_title(maintenance_type)

        # intervalle en h -> jours (priorité recommended, sinon R, sinon cost)
        h = r.get("T_recommended_h")
        if h is None:
            h = r.get("T_R_h")
        if h is None:
            h = r.get("T_cost_h")

        # si pas d’intervalle => on skip (inspection/correctif non périodiques)
        days = _to_days(h, min_days=cfg.min_days, max_days=cfg.max_days)

        if cfg.only_if_preventive and (not _is_preventive(maintenance_type)):
            # pas de planification calendaire stricte
            skipped += 1
            continue

        if days is None:
            skipped += 1
            continue

        next_due = (base + timedelta(days=int(days))).isoformat()

        payload = {
            "equipment_code": eq,
            "title": title,
            "periodicity_days": int(days),
            "next_due_date": next_due,
            "last_done_date": None,
            "status": "ACTIVE",
        }

        try:
            k = (eq, title)
            if k in key_to_id:
                payload["id"] = key_to_id[k]
                pm.upsert_task(payload)
                updated += 1
            else:
                pm.upsert_task(payload)
                created += 1
        except Exception as e:
            errors.append(f"{eq}: {e}")

    return {
        "ok": len(errors) == 0,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:20],
    }
