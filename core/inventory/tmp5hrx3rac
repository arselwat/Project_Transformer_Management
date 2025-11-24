# core/inventory/safety.py
from __future__ import annotations
from typing import List, Dict
import math

def safety_stock_stat(
    history_daily_demand: List[float],
    lead_time_days: float,
    service_level_z: float = 1.65,  # ~95%
) -> int:
    """SS = z * sigma_demand * sqrt(lead_time)."""
    if not history_daily_demand:
        return 0
    n = len(history_daily_demand)
    mean = sum(history_daily_demand) / n
    var = sum((x - mean)**2 for x in history_daily_demand) / max(1, (n-1))
    sigma = math.sqrt(max(0.0, var))
    ss = service_level_z * sigma * math.sqrt(max(0.0, lead_time_days))
    return int(math.ceil(ss))

def safety_stock_risk(
    failure_prob_30d: float,
    mean_usage_per_event: float,
    criticality: float = 1.0,  # 1 (faible) à 3 (élevée)
    buffer: float = 0.5,
) -> int:
    """SS ≈ (proba défaut 30j) × (usage moyen) × criticité + buffer."""
    val = failure_prob_30d * mean_usage_per_event * max(1.0, criticality) + buffer
    return int(math.ceil(val))

def propose_safety_stock_for_parts(
    parts: List[Dict],
    risks_by_eq: Dict[str, float] | None = None,
    default_lead_time_days: float = 14.0,
) -> List[Dict]:
    """
    Retourne une liste avec proposition de SS par pièce (colonne 'ss_propose').
    Si aucun historique → fallback risk-based avec failure_prob_30d globale (moyenne).
    """
    out = []
    global_risk = 0.15
    if risks_by_eq:
        vals = [max(0.0, min(1.0, v)) for v in risks_by_eq.values()]
        if vals:
            global_risk = sum(vals)/len(vals)

    for p in parts:
        hist = p.get("daily_history", [])  # optionnel
        if hist:
            ss = safety_stock_stat(hist, default_lead_time_days, 1.65)
        else:
            # suppose 1 unité par event en moyenne
            ss = safety_stock_risk(global_risk, mean_usage_per_event=1.0, criticality=float(p.get("criticite",1.0)))
        row = dict(p)
        row["ss_propose"] = ss
        out.append(row)
    return out
