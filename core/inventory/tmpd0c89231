from __future__ import annotations
from typing import List, Dict, Any

# Kit simple basé sur beta + familles
def build_pm_kit_for_equipment(equipment_code: str, beta: float | None, parts: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if not isinstance(parts, list) or not parts:
        return []

    fam_map = {
        "refroidissement": ["huile","filtre","ventilateur","radiateur"],
        "connexion": ["borne","cosse","connecteur","serrage"],
        "mesure": ["capteur","thermique","sonde","capteur temp"],
        "joint": ["joint","garniture"],
    }

    def match_any(p: Dict[str, Any], words: List[str]) -> bool:
        s = (p.get("nom","") + " " + p.get("famille","")).lower()
        return any(w in s for w in words)

    kit: List[Dict[str, Any]] = []
    if beta is None:
        # profil inconnu → sélectionner des essentiels
        for p in parts:
            if match_any(p, fam_map["joint"]) or match_any(p, fam_map["refroidissement"]):
                p = p.copy(); p["quantite_recommandee"] = max(1, int(p.get("seuil_min", 0) or 1))
                kit.append(p)
    elif beta > 1.0:
        # usure → refroidissement + joints
        for p in parts:
            if match_any(p, fam_map["refroidissement"]) or match_any(p, fam_map["joint"]):
                p = p.copy(); p["quantite_recommandee"] = max(1, int(p.get("seuil_min", 0) or 1))
                kit.append(p)
    elif beta < 1.0:
        # défauts précoces → connexions/mesure
        for p in parts:
            if match_any(p, fam_map["connexion"]) or match_any(p, fam_map["mesure"]):
                p = p.copy(); p["quantite_recommandee"] = max(1, int(p.get("seuil_min", 0) or 1))
                kit.append(p)
    else:
        # beta==1 → un peu de tout minimal
        for p in parts:
            if match_any(p, fam_map["joint"]) and match_any(p, fam_map["connexion"]):
                qrec = max(1, int(p.get("seuil_min", 0) or 1))
                p = p.copy(); p["quantite_recommandee"] = qrec
                kit.append(p)

    # dédoublonnage par code
    seen = set(); out = []
    for p in kit:
        c = (p.get("code") or "").strip()
        if not c or c in seen: 
            continue
        seen.add(c); out.append(p)
    return out
