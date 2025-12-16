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
    # [CHG] Garde-fous inchangés mais un peu plus stricts (évite des params non physiques)
    if beta <= 0 or eta <= 0:
        return None

    # [CHG] Clamp du R_target au lieu de forcer 0.8 silencieusement (plus propre)
    if not (0.0 < R_target < 1.0):
        R_target = 0.8

    # Formule inverse directe (OK et fidèle)
    return float(gamma + eta * (-math.log(R_target)) ** (1.0 / beta))


# =========================
# 2. Intervalle coût minimal (Weibull) - Politique type âge (document)
# =========================

def optimize_interval_cost_weibull(
    beta: float,
    eta: float,
    gamma: float,
    C_prev: float,   # C_p dans le document (coût de préventif)
    C_corr: float,   # C_f dans le document (coût de correctif / panne)
    R_min: float = 0.0,
    t_max_mult: float = 3.0,
    steps: int = 200,
    integ_steps: int = 400,  # [CHG] nb de pas pour l'intégrale ∫ R(t) dt (stabilité)
) -> dict[str, float | None]:
    """
    Cherche T qui minimise le coût moyen par heure (politique type âge / renewal).

    [CHG] Ancien approx:
        C(T) ≈ (C_prev + C_corr * (1 - R(T))) / T

    [CHG] Nouvelle formule (fidèle au document):
        C(T) = (C_prev*R(T) + C_corr*(1 - R(T))) / ∫_0^T R(t) dt

    - C_prev est payé seulement si l'équipement survit jusqu'à T (d'où *R(T))
    - Le dénominateur est la durée moyenne du cycle E[min(X, T)] = ∫_0^T R(t) dt

    Si R_min > 0, on ne garde que les T tels que R(T) >= R_min.

    Retourne:
      {
        "T_cost": T* (h),
        "C_min": C(T*) (coût moyen / h),
        "R_at_T": R(T*),
      }
    """
    # [CHG] Validation: coûts > 0 et paramètres Weibull valides
    if beta <= 0 or eta <= 0 or C_prev <= 0 or C_corr <= 0:
        return {"T_cost": None, "C_min": None, "R_at_T": None}

    # [CHG] Fiabilité Weibull 3p (inchangée mais clarifiée)
    def R(t: float) -> float:
        # Pour t <= γ, la loi est "décalée": pas de défaillance avant γ -> R=1
        if t <= gamma:
            return 1.0
        return math.exp(-(((t - gamma) / eta) ** beta))

    # [CHG] Intégrale numérique ∫_0^T R(t) dt via trapèzes (stable et simple)
    def integral_R_0_T(T: float) -> float:
        if T <= 0:
            return 0.0
        # grille de 0 à T
        xs = np.linspace(0.0, T, max(10, int(integ_steps)))
        ys = np.array([R(float(x)) for x in xs], dtype=float)
        # trapèzes
        return float(np.trapz(ys, xs))

    # [CHG] Grille de recherche: commence à γ+ε, sinon la zone avant γ donne R=1 et fausse l’optimisation
    eps = max(1e-6, 1e-4 * eta)
    t_min = max(eps, gamma + eps)

    # [CHG] t_max basé sur gamma + multiple de eta (plus logique avec décalage)
    t_max = gamma + eta * float(t_max_mult)

    if t_max <= t_min:
        t_max = t_min + eta  # fallback

    t_grid = np.linspace(t_min, t_max, max(20, int(steps)))

    best_T, best_C, best_R = None, float("inf"), None

    for T in t_grid:
        T = float(T)
        R_T = R(T)

        # [CHG] Contrainte de fiabilité minimale (comme ton code)
        if R_min > 0.0 and R_T < R_min:
            continue

        denom = integral_R_0_T(T)

        # [CHG] Evite division par 0 / intégrale trop petite
        if denom <= 1e-12:
            continue

        # [CHG] Coût document: C_prev*R(T) + C_corr*(1-R(T))
        num = (C_prev * R_T) + (C_corr * (1.0 - R_T))
        C_T = num / denom

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

    Retourne, pour chaque équipement:
      {
        "T_R":      intervalle par fiabilité cible (h),
        "T_cost":   intervalle coût minimal (h),
        "R_at_T":   fiabilité à T_cost,
        "C_min":    coût moyen minimal (/h),
      }
    """
    out: Dict[str, dict] = {}

    # [CHG] Garde-fous inchangés
    if not fits or C_prev <= 0 or C_corr <= 0:
        return out

    if not (0.0 < R_target < 1.0):
        R_target = 0.8

    for eq, ft in fits.items():
        try:
            beta = float(getattr(ft, "beta"))
            eta = float(getattr(ft, "eta"))
            gamma = float(getattr(ft, "gamma", 0.0))

            # Intervalle "fiabilité cible" (inchangé)
            T_R = interval_weibull_target(beta, eta, gamma, R_target)

            # [CHG] Intervalle coût minimal désormais basé sur la vraie formule du document
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
            # [CHG] On garde le comportement: ignorer silencieusement les items invalides
            continue

    return out
