# core/reliability/optimize.py
from __future__ import annotations

from typing import Dict, Any, Optional
import math
import numpy as np


# =========================
# 1. Intervalle par fiabilité cible (Weibull 3p)
# =========================

def interval_weibull_target(
    beta: float,
    eta: float,
    gamma: float,
    R_target: float,
) -> Optional[float]:
    """
    Intervalle T_R tel que R(T_R) = R_target pour une Weibull 3p (β, η, γ).
    R(t) = exp(-((t-γ)/η)^β) pour t >= γ.
    """
    if beta <= 0 or eta <= 0:
        return None

    if not (0.0 < float(R_target) < 1.0):
        R_target = 0.8

    # Inversion de R(t)
    return float(gamma + eta * (-math.log(R_target)) ** (1.0 / beta))


# =========================
# 2. Intervalle coût minimal (Weibull) - Politique type âge
# =========================

def optimize_interval_cost_weibull(
    beta: float,
    eta: float,
    gamma: float,
    C_prev: float,       # coût préventif Cp
    C_corr: float,       # coût correctif Cf
    R_min: float = 0.0,  # contrainte R(T) >= R_min
    t_max_mult: float = 3.0,
    steps: int = 200,
    integ_steps: int = 400,
) -> Dict[str, Optional[float]]:
    """
    Cherche T qui minimise le coût moyen par heure (politique type âge / renewal).

    Formule:
        C(T) = (Cp*R(T) + Cf*(1 - R(T))) / ∫_0^T R(t) dt
    """
    if beta <= 0 or eta <= 0 or C_prev <= 0 or C_corr <= 0:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    def Rf(t: float) -> float:
        if t <= gamma:
            return 1.0
        return math.exp(-(((t - gamma) / eta) ** beta))

    def integral_R_0_T(T: float) -> float:
        if T <= 0:
            return 0.0
        xs = np.linspace(0.0, T, max(10, int(integ_steps)))
        ys = np.array([Rf(float(x)) for x in xs], dtype=float)
        return float(np.trapz(ys, xs))

    eps = max(1e-6, 1e-4 * eta)
    t_min = max(eps, gamma + eps)
    t_max = gamma + eta * float(t_max_mult)
    if t_max <= t_min:
        t_max = t_min + eta

    t_grid = np.linspace(t_min, t_max, max(20, int(steps)))

    best_T: Optional[float] = None
    best_C: float = float("inf")
    best_R: Optional[float] = None

    for T in t_grid:
        T = float(T)
        R_T = float(Rf(T))

        if R_min > 0.0 and R_T < R_min:
            continue

        denom = integral_R_0_T(T)
        if denom <= 1e-12:
            continue

        num = (C_prev * R_T) + (C_corr * (1.0 - R_T))
        C_T = float(num / denom)

        if C_T < best_C:
            best_C = C_T
            best_T = T
            best_R = R_T

    if best_T is None:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    return {"T_cost": float(best_T), "C_min": float(best_C), "R_at_T": float(best_R)}


# =========================
# 3. Cost + fiabilité (version “riche”)
# =========================

def propose_intervals_cost_and_reliability(
    fits: Dict[str, Any],
    C_prev: float,
    C_corr: float,
    R_target: float = 0.80,
    R_min_cost: float = 0.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    fits[code] = objet avec .beta, .eta, éventuellement .gamma.

    Retourne, pour chaque équipement:
      {
        "T_R":      intervalle basé fiabilité cible (h),
        "T_cost":   intervalle coût minimal (h),
        "R_at_T":   fiabilité à T_cost,
        "C_min":    coût moyen minimal (/h),
      }
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    if not fits or C_prev <= 0 or C_corr <= 0:
        return out

    if not (0.0 < float(R_target) < 1.0):
        R_target = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

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


# =========================
# 4. COMPAT: propose_intervals attendu par unify/compute_bundle
# =========================

def propose_intervals(
    fits: Dict[str, Any],
    R_target: float = 0.80,
    t_min: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compatibilité: attendu par plusieurs modules:
        from core.reliability.optimize import propose_intervals

    Retour attendu:
        { "TR-01": {"interval_opt_h": 123.4}, ... }

    Ici: intervalle basé sur fiabilité cible Weibull (T_R).
    """
    out: Dict[str, Dict[str, float]] = {}

    if not fits:
        return out

    if not (0.0 < float(R_target) < 1.0):
        R_target = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

            T_R = interval_weibull_target(beta, eta, gamma, R_target)
            if T_R is None:
                continue

            T_R = max(float(T_R), float(t_min))
            out[str(eq)] = {"interval_opt_h": float(T_R)}
        except Exception:
            continue

    return out
