# core/reliability/weibull.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Iterable
import numpy as np
import math

# ============================================================
# Modèle paramétrique minimal
# ============================================================
@dataclass
class WeibullFit:
    beta: float
    eta: float
    gamma: float = 0.0  # on force 2 paramètres dans notre MLE -> gamma=0

# ============================================================
# Helpers robustes d’extraction des paramètres
# ============================================================
def _params_from_fit(fit: Any) -> Tuple[float, float, float]:
    """
    Extrait (beta, eta, gamma) depuis:
      - instance avec attributs .beta/.eta/(.gamma)
      - dict {"beta":..,"eta":..,"gamma":..}
    Lève ValueError si β ou η invalides.
    """
    beta = None
    eta  = None
    gamma = 0.0

    if fit is None:
        raise ValueError("fit is None")

    # objet avec attributs
    if hasattr(fit, "beta") and hasattr(fit, "eta"):
        beta = getattr(fit, "beta", None)
        eta  = getattr(fit, "eta", None)
        gamma = getattr(fit, "gamma", 0.0)

    # dict
    if beta is None and isinstance(fit, dict):
        beta = fit.get("beta", None)
        eta  = fit.get("eta",  None)
        gamma = fit.get("gamma", 0.0)

    # conversion + garde-fous
    try:
        beta = float(beta)
        eta  = float(eta)
        gamma = float(gamma if gamma is not None else 0.0)
    except Exception as e:
        raise ValueError(f"Invalid β/η values on fit: {e}")

    if not (beta > 0.0 and eta > 0.0):
        raise ValueError("β and η must be > 0")

    return beta, eta, gamma

def _x_from_t(t, gamma):
    """Décale le temps (γ), borne à ≥0."""
    x = np.asarray(t, dtype=float)
    return np.maximum(x, gamma) - gamma

# ============================================================
# Fonctions universelles (pas d’appel à fit.R())
# ============================================================
def R(t, fit: Any):
    beta, eta, gamma = _params_from_fit(fit)
    x = _x_from_t(t, gamma)
    eta = max(eta, 1e-12)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = (x / eta) ** beta
        out = np.exp(-z)
    return out

def F(t, fit: Any):
    return 1.0 - R(t, fit)

def pdf(t, fit: Any):
    beta, eta, gamma = _params_from_fit(fit)
    x = _x_from_t(t, gamma)
    eta = max(eta, 1e-12)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        base = (x / eta)
        dens = (beta / eta) * np.power(base, np.maximum(beta - 1.0, 0.0)) * np.exp(-np.power(base, beta))
    dens = np.where((np.asarray(t, dtype=float) - gamma) >= 0.0, dens, 0.0)
    return dens

def hazard(t, fit: Any):
    f = pdf(t, fit)
    r = R(t, fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(r > 0.0, f / np.clip(r, 1e-300, None), np.nan)
    return h

def as_weibull_fit(fit: Any) -> WeibullFit:
    """Transforme objet/dict en WeibullFit."""
    b, e, g = _params_from_fit(fit)
    return WeibullFit(beta=b, eta=e, gamma=g)

# ============================================================
# Estimation MLE 2 paramètres (γ=0) — sans SciPy
# ============================================================
def fit_weibull(x: Iterable[float],
                max_iter: int = 100,
                tol: float = 1e-7) -> WeibullFit:
    """
    MLE Weibull 2 paramètres (β, η), γ=0.
    - Nettoie x<=0
    - Newton-Raphson sur l’équation:
        f(k) = n/k + sum(ln x) - n*(sum x^k ln x)/(sum x^k) = 0
      f'(k) = -n/k^2 - n*( (sum x^k (ln x)^2)/(sum x^k) - ( (sum x^k ln x)/(sum x^k) )^2 )
    - η = ( (1/n) * sum x^k )^(1/k)
    Retourne WeibullFit(beta, eta, gamma=0.0)
    """
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    n = x.size
    if n < 2:
        # fallback très robuste
        if n == 1:
            return WeibullFit(beta=1.0, eta=float(x[0]), gamma=0.0)
        raise ValueError("Not enough data to fit Weibull.")

    lx = np.log(x)
    s = float(np.std(lx, ddof=1)) if n > 1 else 1.0
    # guess initial standard (borne pour éviter l’extrême)
    if s < 1e-8:
        k = 3.0
    else:
        k = max(0.2, min(10.0, 1.2 / s))

    S_lx = float(np.sum(lx))

    for _ in range(max_iter):
        xk   = np.power(x, k)
        A    = float(np.sum(xk))
        B    = float(np.sum(xk * lx))
        C    = float(np.sum(xk * lx * lx))

        if A <= 0.0:
            break

        f   = n / k + S_lx - n * (B / A)
        var = (C / A) - (B / A) ** 2
        fp  = -n / (k * k) - n * var

        if not np.isfinite(f) or not np.isfinite(fp) or fp == 0.0:
            break

        k_new = k - f / fp
        # bornes
        k_new = float(np.clip(k_new, 0.1, 50.0))

        if abs(k_new - k) <= tol * (1.0 + k):
            k = k_new
            break
        k = k_new

    # eta via MLE fermé
    xk = np.power(x, k)
    eta = float((np.sum(xk) / n) ** (1.0 / k))
    # sauvegardes sécu
    if not (np.isfinite(k) and k > 0.0):
        k = 1.0
    if not (np.isfinite(eta) and eta > 0.0):
        eta = float(np.mean(x))

    return WeibullFit(beta=float(k), eta=float(eta), gamma=0.0)
# ============================================================
# Backward-compat shims (pour anciens modules)
# ============================================================

def fit_weibull_mle(x, *args, **kwargs):
    """Alias rétro-compatible vers fit_weibull (MLE 2P, gamma=0)."""
    return fit_weibull(x, *args, **kwargs)

def reliability_weibull(t, fit):
    """Alias rétro-compatible : même chose que R(t, fit)."""
    return R(t, fit)

# Autres alias éventuels utilisés ailleurs
weibull_R = R
weibull_F = F
weibull_pdf = pdf
weibull_hazard = hazard
as_fit = as_weibull_fit
