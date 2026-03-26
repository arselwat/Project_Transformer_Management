from __future__ import annotations

from typing import Dict, Any, Optional
import math
import numpy as np


# ============================================================
# Helpers
# ============================================================

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _weibull_survival_3p(t: float, beta: float, eta: float, gamma: float = 0.0) -> float:
    """
    Fonction de survie Weibull 3p :
        R(t) = 1                              si t <= gamma
             = exp(-((t-gamma)/eta)^beta)     sinon
    """
    if beta <= 0 or eta <= 0:
        return 0.0
    if t <= gamma:
        return 1.0
    return math.exp(-(((t - gamma) / eta) ** beta))


def _integral_survival_trapz(
    beta: float,
    eta: float,
    gamma: float,
    T: float,
    integ_steps: int = 400,
) -> float:
    if T <= 0:
        return 0.0
    xs = np.linspace(0.0, float(T), max(10, int(integ_steps)))
    ys = np.array([_weibull_survival_3p(float(x), beta, eta, gamma) for x in xs], dtype=float)
    return float(np.trapz(ys, xs))


# ============================================================
# 1) Intervalle par fiabilité cible (Weibull 3p)
# ============================================================

def interval_weibull_target(
    beta: float,
    eta: float,
    gamma: float,
    R_target: float,
) -> Optional[float]:
    """
    Calcule l’intervalle T_R tel que R(T_R)=R_target.

    Pour une Weibull 3p :
        R(t)=exp(-((t-gamma)/eta)^beta) pour t>=gamma
    d’où :
        T_R = gamma + eta * (-ln(R_target))^(1/beta)
    """
    beta = _safe_float(beta, None)
    eta = _safe_float(eta, None)
    gamma = _safe_float(gamma, 0.0) or 0.0
    R_target = _safe_float(R_target, 0.8) or 0.8

    if beta is None or eta is None or beta <= 0 or eta <= 0:
        return None

    if not (0.0 < R_target < 1.0):
        R_target = 0.8

    try:
        return float(gamma + eta * ((-math.log(R_target)) ** (1.0 / beta)))
    except Exception:
        return None


# ============================================================
# 2) Intervalle coût minimal (politique âge sur Weibull)
# ============================================================

def optimize_interval_cost_weibull(
    beta: float,
    eta: float,
    gamma: float,
    C_prev: float,
    C_corr: float,
    R_min: float = 0.0,
    t_max_mult: float = 3.0,
    steps: int = 200,
    integ_steps: int = 400,
) -> Dict[str, Optional[float]]:
    """
    Cherche T qui minimise le coût moyen par heure sous politique type âge.

    Coût moyen :
        C(T) = (Cp*R(T) + Cf*(1-R(T))) / ∫_0^T R(t)dt

    avec :
        - Cp = coût maintenance préventive
        - Cf = coût corrective / panne
        - R(T) = fiabilité à T

    Une contrainte peut être imposée :
        R(T) >= R_min
    """
    beta = _safe_float(beta, None)
    eta = _safe_float(eta, None)
    gamma = _safe_float(gamma, 0.0) or 0.0
    C_prev = _safe_float(C_prev, None)
    C_corr = _safe_float(C_corr, None)
    R_min = _safe_float(R_min, 0.0) or 0.0

    if beta is None or eta is None or C_prev is None or C_corr is None:
        return {"T_cost": None, "C_min": None, "R_at_T": None}
    if beta <= 0 or eta <= 0 or C_prev <= 0 or C_corr <= 0:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    eps = max(1e-6, 1e-4 * eta)
    t_min = max(eps, gamma + eps)
    t_max = gamma + eta * float(max(t_max_mult, 1.2))
    if t_max <= t_min:
        t_max = t_min + eta

    t_grid = np.linspace(t_min, t_max, max(20, int(steps)))

    best_T: Optional[float] = None
    best_C: float = float("inf")
    best_R: Optional[float] = None

    for T in t_grid:
        T = float(T)
        R_T = float(_weibull_survival_3p(T, beta, eta, gamma))

        if R_min > 0.0 and R_T < R_min:
            continue

        denom = _integral_survival_trapz(beta, eta, gamma, T, integ_steps=integ_steps)
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

    return {
        "T_cost": float(best_T),
        "C_min": float(best_C),
        "R_at_T": float(best_R) if best_R is not None else None,
    }


# ============================================================
# 3) Cost + fiabilité (interface principale)
# ============================================================

def propose_intervals_cost_and_reliability(
    fits: Dict[str, Any],
    C_prev: float,
    C_corr: float,
    R_target: float = 0.80,
    R_min_cost: float = 0.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    fits[code] = objet possédant .beta, .eta, éventuellement .gamma.

    Retourne pour chaque équipement :
      {
        "T_R":    intervalle basé fiabilité cible,
        "T_cost": intervalle coût minimal,
        "R_at_T": fiabilité à T_cost,
        "C_min":  coût moyen minimal,
      }
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    if not fits:
        return out

    C_prev_v = _safe_float(C_prev, None)
    C_corr_v = _safe_float(C_corr, None)
    R_target_v = _safe_float(R_target, 0.8) or 0.8
    R_min_cost_v = _safe_float(R_min_cost, 0.0) or 0.0

    if C_prev_v is None or C_corr_v is None or C_prev_v <= 0 or C_corr_v <= 0:
        return out

    if not (0.0 < R_target_v < 1.0):
        R_target_v = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

            T_R = interval_weibull_target(beta, eta, gamma, R_target_v)

            cost_res = optimize_interval_cost_weibull(
                beta=beta,
                eta=eta,
                gamma=gamma,
                C_prev=C_prev_v,
                C_corr=C_corr_v,
                R_min=R_min_cost_v,
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


# ============================================================
# 4) COMPAT : propose_intervals
# ============================================================

def propose_intervals(
    fits: Dict[str, Any],
    R_target: float = 0.80,
    t_min: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Interface de compatibilité avec les anciens modules.

    Retour :
        { "EQ-01": {"interval_opt_h": 123.4}, ... }

    Ici l’intervalle retourné est T_R (basé fiabilité cible).
    """
    out: Dict[str, Dict[str, float]] = {}

    if not fits:
        return out

    R_target_v = _safe_float(R_target, 0.8) or 0.8
    t_min_v = _safe_float(t_min, 0.0) or 0.0

    if not (0.0 < R_target_v < 1.0):
        R_target_v = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

            T_R = interval_weibull_target(beta, eta, gamma, R_target_v)
            if T_R is None:
                continue

            out[str(eq)] = {"interval_opt_h": float(max(T_R, t_min_v))}
        except Exception:
            continue

    return out
