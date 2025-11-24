# core/reliability/diag_tests.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# ---------- Utilitaires ----------
def _as_clean_array(ttf) -> np.ndarray:
    x = np.asarray(ttf, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x

@dataclass
class TrendResult:
    method: str
    statistic: float
    pvalue: float
    conclusion: str  # "trend_up" / "trend_down" / "no_trend" / "inconclusive"

@dataclass
class IndependenceResult:
    method: str
    statistic: float
    pvalue: float
    conclusion: str  # "independent" / "dependent" / "inconclusive"

# ---------- Test de tendance : Laplace (MIL-STD-781) ----------
def laplace_trend_test(ttf: List[float]) -> TrendResult:
    """
    Test de Laplace sur les temps inter-pannes (TTF).
    H0 : taux constant (pas de tendance).
    Stat L = (sum(t_i) - n*mean)/ (mean*sqrt(n/12)) ~ N(0,1) approx.
    Ici, on utilise la version standardisée classique.
    """
    x = _as_clean_array(ttf)
    n = len(x)
    if n < 6:
        return TrendResult("laplace", float("nan"), float("nan"), "inconclusive")

    mean = x.mean()
    # Temps cumulatifs centrés
    t = np.cumsum(x)
    L = (t.sum() - n * (t[-1] / 2)) / ( (t[-1]/np.sqrt(12)) )
    # sous H0, approx ~ N(0,1). Bilatéral.
    z = abs(L)
    # p-value ~ 2*(1-Phi(|z|))
    # approximation rapide sans SciPy:
    def phi_tail(zval):
        return 0.5*math.erfc(zval/np.sqrt(2))
    p = 2*phi_tail(z)
    if z > 1.96:
        # signe de L donne la tendance
        conc = "trend_up" if L > 0 else "trend_down"
    else:
        conc = "no_trend"
    return TrendResult("laplace", float(L), float(p), conc)

# ---------- Test de tendance : Cox–Stuart (version simple) ----------
def cox_stuart_trend_test(ttf: List[float]) -> TrendResult:
    x = _as_clean_array(ttf)
    n = len(x)
    if n < 10:
        return TrendResult("cox_stuart", float("nan"), float("nan"), "inconclusive")
    half = n // 2
    a, b = x[:half], x[-half:]
    diffs = b - a
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    # sous H0, pos ~ Binom(half, 0.5)
    # Use normal approx
    mu = half * 0.5
    sigma = math.sqrt(half * 0.5 * 0.5)
    z = (pos - mu) / (sigma if sigma > 0 else 1e-9)
    # bilatéral
    def phi_tail(zval): return 0.5*math.erfc(abs(zval)/np.sqrt(2))
    p = 2*phi_tail(z)
    if abs(z) > 1.96:
        conc = "trend_up" if z > 0 else "trend_down"
    else:
        conc = "no_trend"
    return TrendResult("cox_stuart", float(z), float(p), conc)

# ---------- Test d’indépendance : runs test sur la médiane ----------
def runs_test_median(ttf: List[float]) -> IndependenceResult:
    x = _as_clean_array(ttf)
    n = len(x)
    if n < 8:
        return IndependenceResult("runs_median", float("nan"), float("nan"), "inconclusive")
    med = np.median(x)
    s = np.where(x >= med, 1, 0)
    # compte de runs
    runs = 1 + np.sum(s[1:] != s[:-1])
    n1 = np.sum(s == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return IndependenceResult("runs_median", float("nan"), float("nan"), "inconclusive")
    mu = 1 + (2*n0*n1)/n
    var = (2*n0*n1*(2*n0*n1 - n)) / (n**2 * (n-1))
    z = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
    def phi_tail(zval): return 0.5*math.erfc(abs(zval)/np.sqrt(2))
    p = 2*phi_tail(z)
    conc = "independent" if abs(z) <= 1.96 else "dependent"
    return IndependenceResult("runs_median", float(z), float(p), conc)

# ---------- Ajustements (paramètres) sans SciPy ----------
def fit_exponential(ttf: List[float]) -> Dict:
    x = _as_clean_array(ttf)
    if len(x) == 0:
        return {"name": "exponential", "ok": False}
    lam = 1.0 / x.mean()
    ll = np.sum(np.log(lam) - lam*x)
    k = 1  # nb params
    aic = -2*ll + 2*k
    return {"name": "exponential", "ok": True, "lambda": float(lam), "loglik": float(ll), "aic": float(aic)}

def fit_lognormal(ttf: List[float]) -> Dict:
    x = _as_clean_array(ttf)
    if len(x) == 0:
        return {"name": "lognormal", "ok": False}
    y = np.log(x)
    mu = float(np.mean(y))
    sigma = float(np.std(y, ddof=1) if len(y) > 1 else 1e-9)
    # log-vraisemblance
    ll = np.sum(-np.log(x*sigma*np.sqrt(2*np.pi)) - (np.log(x)-mu)**2/(2*sigma**2))
    k = 2
    aic = -2*ll + 2*k
    return {"name":"lognormal","ok":True,"mu":mu,"sigma":sigma,"loglik":float(ll),"aic":float(aic)}

def fit_gamma_mm(ttf: List[float]) -> Dict:
    x = _as_clean_array(ttf)
    if len(x) == 0:
        return {"name": "gamma", "ok": False}
    m = x.mean()
    v = x.var(ddof=1) if len(x) > 1 else (m**2)
    if v <= 0: v = 1e-9
    k_shape = m**2 / v
    theta = v / m
    # loglik (shape k, scale θ)
    ll = np.sum((k_shape-1)*np.log(x) - x/theta - k_shape*np.log(theta) - math.lgamma(k_shape))
    k = 2
    aic = -2*ll + 2*k
    return {"name":"gamma","ok":True,"k":float(k_shape),"theta":float(theta),"loglik":float(ll),"aic":float(aic)}

def ks_distance(sample: np.ndarray, cdf_fn) -> float:
    s = np.sort(sample)
    n = len(s)
    if n == 0: return float("inf")
    u = np.arange(1, n+1)/n
    F = np.array([cdf_fn(val) for val in s])
    return float(np.max(np.abs(F - u)))

def cdf_exponential(x, lam): return 1 - np.exp(-lam*max(x, 0.0))
def cdf_lognormal(x, mu, sigma):
    if x <= 0: return 0.0
    z = (math.log(x)-mu)/(sigma+1e-12)
    # approx Phi(z)
    return 0.5*(1+math.erf(z/math.sqrt(2)))
def cdf_gamma_mm(x, k, theta):
    # CDF gamma(k, theta) via série incomplète (approx) — simple & robuste
    if x <= 0: return 0.0
    # Reg: P(k, x/theta) ≈ igam(k, x/θ) / Γ(k), ici on fait une approximation par somme tronquée
    t = x/theta
    # série de Poisson gamma (k entier ~ approchée si non entier)
    # on tronque à 50 termes
    s = 0.0
    term = 1.0
    for i in range(50):
        if i > 0:
            term *= t / i
        s += term
    return 1 - math.exp(-t) * s

# Weibull → on utilise ton fit existant (beta, eta) pour la comparaison KS (CDF)
def cdf_weibull(x, beta, eta):
    if x <= 0: return 0.0
    return 1 - math.exp(- (x/eta)**beta )
