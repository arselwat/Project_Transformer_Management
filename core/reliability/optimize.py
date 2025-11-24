# core/reliability/optimize.py
from __future__ import annotations
from typing import Dict, Any
import math

"""
Principe:
- On cible une fiabilité R_target (ex: 0.8) entre deux maintenances.
- Pour un fit Weibull (β, η, γ≈0), R(t) = exp(-((t-γ)/η)^β) → t* = γ + η * (-ln R_target)^(1/β)
- On retourne un dictionnaire {equipment_code: interval_opt_h}
"""

def propose_intervals(fits: Dict[str, Any], R_target: float = 0.80) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not fits:
        return out

    if R_target <= 0 or R_target >= 1:
        R_target = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta  = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0))
            if beta > 0 and eta > 0:
                t = gamma + eta * ( -math.log(R_target) )**(1.0 / beta)
                out[str(eq)] = float(t)
        except Exception:
            pass
    return out
