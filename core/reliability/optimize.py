# core/reliability/optimize.py
from __future__ import annotations
from typing import Dict, Any
import math

def _interval_weibull(beta: float, eta: float, gamma: float, R_target: float) -> float | None:
    if beta <= 0 or eta <= 0:
        return None
    if not (0.0 < R_target < 1.0):
        R_target = 0.8
    return gamma + eta * (-math.log(R_target)) ** (1.0 / beta)

def _interval_exponential(lmbda: float, R_target: float) -> float | None:
    # R(t) = exp(-λ t) → t* = -ln(R_target)/λ
    if lmbda <= 0:
        return None
    if not (0.0 < R_target < 1.0):
        R_target = 0.8
    return -math.log(R_target) / lmbda

def propose_intervals_from_models(
    fits: Dict[str, Dict[str, Any]],
    R_target: float = 0.80,
) -> Dict[str, Dict[str, float | str]]:
    """
    fits[code] = sortie de select_best_model() pour un équipement.
    Retourne pour chaque code:
      - interval_h : intervalle suggéré en heures (si dispo)
      - policy_hint : texte court sur la politique
      - model_name  : loi retenue
    """
    out: Dict[str, Dict[str, float | str]] = {}
    if not fits:
        return out

    if not (0.0 < R_target < 1.0):
        R_target = 0.8

    for eq_code, res in fits.items():
        if not res.get("ok"):
            continue
        name = res.get("best_name")
        params = res.get("params", {}) or {}
        interval = None
        policy = "undefined"

        if name == "weibull":
            beta = float(params.get("beta", 0.0))
            eta  = float(params.get("eta", 0.0))
            gamma = float(params.get("gamma", 0.0)) if "gamma" in params else 0.0
            interval = _interval_weibull(beta, eta, gamma, R_target)
            # Ajuster le hint selon beta
            if beta < 1.0:
                policy = "surveillance/predictive (β<1, peu de sens de remplacer par âge)"
            elif 0.95 <= beta <= 1.05:
                policy = "corrective + inspections périodiques (β≈1)"
            else:
                policy = "préventive calée sur l'âge (β>1, usure)"
        elif name == "exponential":
            lmbda = float(params.get("lambda", 0.0))
            interval = _interval_exponential(lmbda, R_target)
            policy = "corrective + opportuniste (taux constant)"
        elif name == "lognormal":
            # pour la lognormale ou gamma, tu peux continuer à utiliser
            # la formule Weibull en t'appuyant sur un équivalent ou rester sur un hint textuel
            policy = "préventive conditionnelle (dégradation multiplicative)"
        elif name == "gamma":
            policy = "préventive liée à la charge/usage (fatigue cumulée)"

        if interval is not None and interval > 0:
            out[str(eq_code)] = {
                "interval_h": float(interval),
                "policy_hint": policy,
                "model_name": str(name),
            }

    return out
