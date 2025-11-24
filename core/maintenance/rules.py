from __future__ import annotations
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from core.maintenance.store import new_task
# Règles simples paramétrables
RULES = {
    "hotspot_alarm_s_per_24h": 60.0,     # si θhs > TEMP_ALARM cumulé > 60 s → tâche
    "pf_low_percent": 20.0,              # si >20% du temps PF < seuil → tâche
    "overload_events_per_24h": 5,        # si >5 events OVERLOAD en 24h → tâche
}

def evaluate_and_create_tasks(transformer_code: str,
                              kpis: Dict[str, float],
                              events_24h: List[dict],
                              temp_alarm_seconds_24h: float,
                              pf_low_percent_24h: float) -> List[str]:
    """
    Retourne la liste des IDs de tâches créées.
    kpis: dict venant de l'onglet KPI (énergie, durées, etc.)
    events_24h: liste d'événements sur 24h glissantes
    """
    created = []

    # 1) Hot-spot > ALARM cumulé en 24h
    if temp_alarm_seconds_24h >= RULES["hotspot_alarm_s_per_24h"]:
        ok, tid = new_task(
            transformer_code, "HIGH",
            "Surchauffe hot-spot (24h)",
            f"θhs > ALARM cumulé {temp_alarm_seconds_24h:.0f} s / 24h. Vérifier refroidissement (ventilos, circulation huile), encrassement radiateurs.",
            "R_HOTSPOT_24H",
            spare_suggestion="Ventilateurs, huile diélectrique, sondes T°"
        )
        if ok: created.append(tid)

    # 2) PF bas trop fréquent
    if pf_low_percent_24h >= RULES["pf_low_percent"]:
        ok, tid = new_task(
            transformer_code, "MEDIUM",
            "PF faible fréquent",
            f"{pf_low_percent_24h:.1f}% du temps avec PF < seuil. Vérifier compensation (bancs de condos), déséquilibres charges.",
            "R_PF_LOW",
            spare_suggestion="Banc de condensateurs"
        )
        if ok: created.append(tid)

    # 3) OVERLOAD events
    overloads = sum(1 for e in events_24h if str(e.get("code")).upper() == "OVERLOAD")
    if overloads >= RULES["overload_events_per_24h"]:
        ok, tid = new_task(
            transformer_code, "MEDIUM",
            "Surcharges répétées",
            f"{overloads} événements OVERLOAD en 24h. Vérifier dispatching, délestage, profiles de charge.",
            "R_OVERLOAD_24H"
        )
        if ok: created.append(tid)

    return created
