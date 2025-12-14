from __future__ import annotations
from typing import Dict, Any
import math
import numpy as np


# =========================
# 1. Intervalle par fiabilité cible (Weibull 3p)
# =========================

def interval_weibull_target(beta: float, eta: float, gamma: float, R_target: float) -> float | None:
    """
    Intervalle T_R tel que R(T_R) = R_target pour une Weibull (β, η, γ).
    R(t) = exp(-((t-γ)/η)^β) pour t >= γ.
    """
    if beta <= 0 or eta <= 0:
        return None
    if not (0.0 < R_target < 1.0):
        R_target = 0.8
    return float(gamma + eta * (-math.log(R_target)) ** (1.0 / beta))


# =========================
# 2. Intervalle coût minimal (Weibull)
# =========================

def optimize_interval_cost_weibull(
    beta: float,
    eta: float,
    gamma: float,
    C_prev: float,
    C_corr: float,
    R_min: float = 0.0,
    t_max_mult: float = 3.0,
    steps: int = 200,
) -> dict[str, float | None]:
    """
    Cherche T qui minimise le coût moyen par heure pour une politique d'âge:
        C(T) ≈ (C_prev + C_corr * (1 - R(T))) / T
    avec R(t) Weibull (β, η, γ).

    Si R_min > 0, on ne garde que les T tels que R(T) >= R_min.

    Retourne:
      {
        "T_cost": T* (h),
        "C_min": C(T*) (coût moyen / h),
        "R_at_T": R(T*),
      }
    """
    if beta <= 0 or eta <= 0 or C_prev <= 0 or C_corr <= 0:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    def R(t: float) -> float:
        if t <= gamma:
            return 1.0
        return math.exp(-(((t - gamma) / eta) ** beta))

    t_grid = np.linspace(eta * 0.1, eta * t_max_mult, steps)
    best_T, best_C = None, float("inf")
    best_R = None

    for T in t_grid:
        R_T = R(T)
        if R_min > 0.0 and R_T < R_min:
            # ne respecte pas la fiabilité minimale
            continue
        p_fail = 1.0 - R_T
        C_T = (C_prev + C_corr * p_fail) / T
        if C_T < best_C:
            best_C = C_T
            best_T = T
            best_R = R_T

    if best_T is None:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    return {
        "T_cost": float(best_T),
        "C_min": float(best_C),
        "R_at_T": float(best_R) if best_R is not None else None,
    }


# =========================
# 3. Intégration: coût + fiabilité
# =========================

def propose_intervals_cost_and_reliability(
    fits: Dict[str, Any],
    C_prev: float,
    C_corr: float,
    R_target: float = 0.80,
    R_min_cost: float = 0.0,
) -> Dict[str, dict]:
    """
    fits[code] = objet avec .beta, .eta, éventuellement .gamma.

    Retourne, pour chaque équipement, un dict:
      {
        "T_R":      intervalle par fiabilité cible (h),
        "T_cost":   intervalle coût minimal (h),
        "R_at_T":   fiabilité à T_cost,
        "C_min":    coût moyen minimal (/h),
      }
    """
    out: Dict[str, dict] = {}
    if not fits or C_prev <= 0 or C_corr <= 0:
        return out

    if not (0.0 < R_target < 1.0):
        R_target = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0))

            T_R = interval_weibull_target(beta, eta, gamma, R_target)
            cost_res = optimize_interval_cost_weibull(
                beta=beta,
                eta=eta,
                gamma=gamma,
                C_prev=C_prev,
                C_corr=C_corr,
                R_min=R_min_cost,
            )

            out[str(eq)] = {
                "T_R": T_R,
                "T_cost": cost_res.get("T_cost"),
                "R_at_T": cost_res.get("R_at_T"),
                "C_min": cost_res.get("C_min"),
            }
        except Exception:
            continue

    return out
