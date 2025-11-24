# core/reliability/policy.py
from __future__ import annotations
from typing import List

def suggested_actions(beta: float) -> List[str]:
    """
    Recommandations qualitatives selon la forme de Weibull :
      β<1   : mortalité infantile (défauts précoces) → focus qualité, rodage, contrôles d'installation
      β≈1   : hasard pur (taux ~ constant) → surveillance régulière, checklists standard
      β>1   : usure/aging → préventif conditionnel, renouvellement de composants, inspection ciblée
    """
    b = float(beta)
    if b < 0.8:
        return [
            "Renforcer les contrôles à la mise en service (rodage, serrage, connexions).",
            "Revoir la qualité fournisseurs / procédures d’installation.",
            "Mettre en place un suivi rapproché post-intervention."
        ]
    if b < 1.2:
        return [
            "Appliquer une maintenance régulière basée temps (checks planifiés).",
            "Surveiller dérives de température et facteur de puissance (PF).",
            "Maintenir un stock minimal pour pannes aléatoires."
        ]
    # β>1
    return [
        "Passer à de la maintenance conditionnelle (CBM) sur T°, I, PF.",
        "Planifier le remplacement préventif des composants critiques.",
        "Ajuster l’intervalle de PM via la cible de fiabilité (R_target)."
    ]
