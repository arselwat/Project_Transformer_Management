# core/reliability/organigram.py
from __future__ import annotations

"""
Organigramme d'analyse des TTF (Time To Failure)

But :
- Déterminer le type de processus (RP / NHPP / BPP) avant optimisation
- Ajuster des lois candidates (MLE)
- Choisir la meilleure loi (AIC) et exposer KS/Chi2 + paramètres

Sortie UNIFIÉE (schéma stable) :
{
  "cleaned_n": int,
  "model": "RP"|"NHPP"|"BPP",
  "distribution": "expon"|"norm"|"lognorm"|"weibull_2p"|"weibull_3p",
  "goodness": {"aic": float, "ks_p": float, "chi2_p": float},
  "params": {"raw": tuple, "beta": float|None, "eta": float|None, "gamma": float|None},
  "tests": {
      "trend_mk": {"z": float, "p": float, "has_trend": bool, "direction": "up"|"down"|"none"},
      "dependence": {"r": float, "p": float, "has_dep": bool, "method": "spearman"|"pearson"}
  },
  "candidates": {name: {"aic":..., "ks_p":..., "chi2_p":..., "params":..., "loglik":...}, ...}
}
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np
from scipy import stats as sst


# ----------------------------- Utils -----------------------------
def _clean_positive(series: List[float]) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


# ------------------ 1) Mann–Kendall (correction ties) ------------------
def mann_kendall_test(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    x = _clean_positive(series)
    n = len(x)
    if n < 3:
        return {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"}

    diffs = x[np.newaxis, :] - x[:, np.newaxis]
    S = float(np.sign(np.triu(diffs, 1)).sum())

    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    varS = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if varS <= 0:
        return {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"}

    if S > 0:
        z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        z = (S + 1) / math.sqrt(varS)
    else:
        z = 0.0

    p = 2 * (1 - sst.norm.cdf(abs(z)))
    has = bool(p < alpha)
    direction = "up" if (has and z > 0) else "down" if (has and z < 0) else "none"
    return {"z": float(z), "p": float(p), "has_trend": has, "direction": direction}


# --------------- 2) Dépendance robuste (lag-1) ---------------
def test_dependence(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    x = _clean_positive(series)
    if len(x) < 3:
        return {"r": 0.0, "p": 1.0, "has_dep": False, "method": "spearman"}

    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"r": 0.0, "p": 1.0, "has_dep": False, "method": "spearman"}

    # Spearman par défaut (plus robuste aux distributions asymétriques)
    method = "spearman"
    try:
        r, p = sst.spearmanr(a, b)
    except Exception:
        method = "pearson"
        r, p = sst.pearsonr(a, b)

    return {"r": float(r), "p": float(p), "has_dep": bool(p < alpha), "method": method}


# ----------------------- 3) Ajustement des lois -----------------------
def _aic_from_loglik(loglik: float, k: int) -> float:
    return float(2 * k - 2 * loglik)

def _safe_logpdf_sum(dist, data: np.ndarray, params: Tuple[float, ...]) -> float:
    try:
        lp = dist.logpdf(data, *params).sum()
        return float(lp) if np.isfinite(lp) else -np.inf
    except Exception:
        return -np.inf

def _merge_small_bins(observed: np.ndarray, expected: np.ndarray, min_exp: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    obs, exp = observed.astype(float).tolist(), expected.astype(float).tolist()
    i = 0
    while i < len(exp):
        if exp[i] < min_exp:
            if i + 1 < len(exp):
                exp[i] += exp[i + 1]; obs[i] += obs[i + 1]
                del exp[i + 1]; del obs[i + 1]
                continue
            elif i - 1 >= 0:
                exp[i - 1] += exp[i]; obs[i - 1] += obs[i]
                del exp[i]; del obs[i]
                i = max(i - 1, 0)
                continue
        i += 1
    return np.asarray(obs), np.asarray(exp)

def _chi2_gof_prob_bins(data: np.ndarray, cdf_func, params: Tuple[float, ...], nbins: int = 8) -> float:
    n = data.size
    if n < 10:
        return 1.0
    nbins = max(4, min(nbins, int(np.sqrt(n)) + 2))
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(data, qs)
    edges = np.unique(edges)
    if edges.size < 4:
        return 1.0

    observed, _ = np.histogram(data, bins=edges)
    expected = []
    for i in range(len(edges) - 1):
        p = cdf_func(edges[i + 1], *params) - cdf_func(edges[i], *params)
        expected.append(float(p) * n)
    expected = np.asarray(expected)

    obs, exp = _merge_small_bins(observed, expected, min_exp=5.0)
    if exp.size < 2 or (exp <= 0).any() or not np.all(np.isfinite(exp)):
        return 1.0
    try:
        _, p = sst.chisquare(f_obs=obs, f_exp=exp)
        return float(p)
    except Exception:
        return 1.0

def _fit_distribution(name: str, data: np.ndarray) -> Dict[str, Any]:
    res = {"name": name, "params": None, "loglik": None, "aic": np.inf, "ks_p": None, "chi2_p": None}

    try:
        # ⚠️ choix pragmatiques :
        # - lois "durée de vie" souvent positives → on contraint floc=0 quand c'est pertinent
        if name == "expon":
            dist = sst.expon
            params = dist.fit(data, floc=0.0)
            k = 1 + 1  # scale + (loc fixé) -> approx k=1, mais on garde 2 (robuste)
            k = 2
        elif name == "norm":
            dist = sst.norm
            params = dist.fit(data)
            k = 2
        elif name == "lognorm":
            dist = sst.lognorm
            params = dist.fit(data, floc=0.0)
            k = 2  # shape + scale (loc fixé)
            k = 3
        elif name == "weibull_2p":
            dist = sst.weibull_min
            params = dist.fit(data, floc=0.0)
            k = 2  # shape + scale
        elif name == "weibull_3p":
            dist = sst.weibull_min
            params = dist.fit(data)
            k = 3  # shape + loc + scale
        else:
            return res

        loglik = _safe_logpdf_sum(dist, data, params)
        if not np.isfinite(loglik):
            return res

        res["params"] = tuple(float(v) for v in params)
        res["loglik"] = float(loglik)
        res["aic"] = _aic_from_loglik(loglik, k=k)

        _, ks_p = sst.kstest(data, dist.cdf, args=params)
        res["ks_p"] = float(ks_p)

        res["chi2_p"] = _chi2_gof_prob_bins(data, dist.cdf, params)
        return res
    except Exception:
        return res

def _normalize_weibull_params(name: str, params: Tuple[float, ...]) -> Tuple[float, float, float]:
    # scipy weibull_min.fit -> (c, loc, scale)
    if name == "weibull_2p":
        c, loc, scale = params
        return float(c), 0.0, float(scale)
    c, loc, scale = params
    return float(c), float(loc), float(scale)

def fit_and_compare_distributions(data: np.ndarray) -> Dict[str, Any]:
    candidates = ["expon", "norm", "lognorm", "weibull_2p", "weibull_3p"]
    all_fits = {name: _fit_distribution(name, data) for name in candidates}
    best_name = min(all_fits.keys(), key=lambda n: all_fits[n]["aic"])
    best = all_fits[best_name]

    params = best.get("params")
    beta = eta = gamma = None
    if best_name.startswith("weibull") and params:
        b, g, e = _normalize_weibull_params(best_name, params)
        beta, eta, gamma = b, e, g

    out = {
        "best_name": best_name,
        "best": best,
        "all": all_fits,
        "weibull": {"beta": beta, "eta": eta, "gamma": gamma} if beta is not None else None,
    }
    return out


# -------------------------- 4) Pipeline principal --------------------------
def analyze_ttf_pipeline(ttf_series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    data = _clean_positive(ttf_series)
    n = int(data.size)
    if n < 3:
        return {
            "error": "TTF insuffisants (<3 après nettoyage).",
            "cleaned_n": n,
            "model": "RP",
            "distribution": "weibull_2p",
            "goodness": {"aic": None, "ks_p": None, "chi2_p": None},
            "params": {"raw": None, "beta": None, "eta": None, "gamma": None},
            "tests": {
                "trend_mk": {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
                "dependence": {"r": 0.0, "p": 1.0, "has_dep": False, "method": "spearman"},
            },
            "candidates": {},
        }

    mk = mann_kendall_test(data.tolist(), alpha=alpha)
    dep = test_dependence(data.tolist(), alpha=alpha)

    # Décision modèle (avant optimisation)
    if mk["has_trend"]:
        model = "NHPP"
    elif dep["has_dep"]:
        model = "BPP"
    else:
        model = "RP"

    fits = fit_and_compare_distributions(data)
    best = fits["best"]
    best_name = fits["best_name"]

    beta = eta = gamma = None
    if fits["weibull"] is not None:
        beta = fits["weibull"]["beta"]
        eta = fits["weibull"]["eta"]
        gamma = fits["weibull"]["gamma"]

    return {
        "cleaned_n": n,
        "model": model,
        "distribution": best_name,
        "goodness": {
            "aic": best.get("aic"),
            "ks_p": best.get("ks_p"),
            "chi2_p": best.get("chi2_p"),
        },
        "params": {
            "raw": best.get("params"),
            "beta": beta,
            "eta": eta,
            "gamma": gamma,
        },
        "tests": {
            "trend_mk": mk,
            "dependence": dep,
        },
        "candidates": fits["all"],
    }
