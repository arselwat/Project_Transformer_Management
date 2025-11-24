from __future__ import annotations
"""
Organigramme d'analyse des TTF (Times To Failure)

Pipeline :
1) Test tendance (Mann–Kendall) → tendance ? NHPP
2) Test dépendance (corrélation TTF[t] vs TTF[t+1]) → dépendance ? BPP
3) Sinon → RP
4) Ajustement lois (expon, norm, lognorm, weibull 2p, 3p) via MLE
5) Sélection par AIC (expose aussi KS/Chi2)
6) Si Weibull retenue : β (shape), η (scale), γ (loc)

Sortie compacte et tolérante :
- 'model' = 'NHPP'|'BPP'|'RP'
- 'distribution' (str) + 'fit' = {params, ks_p, chi2_p}
- 'beta','eta','gamma' si Weibull
- 'details' = {'mann_kendall': bool, 'correlation': bool}
- 'distribution_full' (détails AIC/KS/Chi2/params)
"""
from typing import List, Dict, Any, Tuple
import math
import numpy as np
from scipy import stats as sst


# ----------------------------- Utils -----------------------------
def _clean_positive(series: List[float]) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    return x[x > 0.0]


# ------------------ 1) Mann–Kendall avec correction ties ------------------
def mann_kendall_test(series: List[float]) -> Dict[str, float | bool]:
    x = _clean_positive(series)
    n = len(x)
    if n < 3:
        return {"z": 0.0, "p": 1.0, "has_trend": False}

    # S = sum_{i<j} sign(xj - xi)
    # vectorisé
    diffs = x[np.newaxis, :] - x[:, np.newaxis]
    S = float(np.sign(np.triu(diffs, 1)).sum())

    # variance avec correction des ex-aequo
    # Var(S) = [n(n-1)(2n+5) - sum(ti(ti-1)(2ti+5))]/18
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    varS = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if varS <= 0:
        return {"z": 0.0, "p": 1.0, "has_trend": False}

    if S > 0:
        z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        z = (S + 1) / math.sqrt(varS)
    else:
        z = 0.0

    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return {"z": float(z), "p": float(p), "has_trend": bool(p < 0.05)}


# --------------- 2) Corrélation robuste (Pearson/Spearman) ---------------
def test_correlation(series: List[float]) -> Dict[str, float | bool]:
    x = _clean_positive(series)
    if len(x) < 3:
        return {"r": 0.0, "p": 1.0, "has_dep": False}

    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"r": 0.0, "p": 1.0, "has_dep": False}

    # si distribution log-skewed → Spearman plus robuste
    try:
        r, p = sst.spearmanr(a, b)
    except Exception:
        r, p = sst.pearsonr(a, b)

    return {"r": float(r), "p": float(p), "has_dep": bool(p < 0.05)}


# ----------------------- 3) Ajustement de lois -----------------------
def _aic_from_loglik(loglik: float, k: int) -> float:
    return float(2 * k - 2 * loglik)

def _safe_logpdf_sum(dist, data: np.ndarray, params: Tuple[float, float, float]) -> float:
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
                del exp[i + 1]; del obs[i + 1]; continue
            elif i - 1 >= 0:
                exp[i - 1] += exp[i]; obs[i - 1] += obs[i]
                del exp[i]; del obs[i]; i = max(i - 1, 0); continue
        i += 1
    return np.asarray(obs), np.asarray(exp)

def _chi2_gof_prob_bins(data: np.ndarray, cdf_func, params: Tuple[float, float, float], nbins: int = 8) -> float:
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
    if exp.size < 2 or not np.all(np.isfinite(exp)) or (exp <= 0).any():
        return 1.0
    try:
        _, p = sst.chisquare(f_obs=obs, f_exp=exp)
        return float(p)
    except Exception:
        return 1.0

def _fit_distribution(name: str, data: np.ndarray) -> Dict[str, Any]:
    res = {"name": name, "params": None, "loglik": None, "aic": np.inf, "ks_p": None, "chi2_p": None}
    try:
        if name == "expon":
            dist = sst.expon; params = dist.fit(data)
            k = 2
        elif name == "norm":
            dist = sst.norm; params = dist.fit(data)
            k = 2
        elif name == "lognorm":
            dist = sst.lognorm; params = dist.fit(data)
            k = 3
        elif name == "weibull_2p":
            dist = sst.weibull_min; params = dist.fit(data, floc=0.0)
            k = 2
        elif name == "weibull_3p":
            dist = sst.weibull_min; params = dist.fit(data)
            k = 3
        else:
            return res

        loglik = _safe_logpdf_sum(dist, data, params)
        if not np.isfinite(loglik):  # invalide → on abandonne cette loi
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

def _normalize_weibull_params(name: str, params: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if name == "weibull_2p":
        c, _, scale = params
        return float(c), 0.0, float(scale)
    c, loc, scale = params
    return float(c), float(loc), float(scale)

def fit_and_compare_distributions(data: np.ndarray) -> Dict[str, Any]:
    candidates = ["expon", "norm", "lognorm", "weibull_2p", "weibull_3p"]
    all_fits = {name: _fit_distribution(name, data) for name in candidates}
    best_name = min(all_fits.keys(), key=lambda n: all_fits[n]["aic"])
    best = all_fits[best_name]
    wb = None
    if best_name.startswith("weibull") and best.get("params"):
        b, g, e = _normalize_weibull_params(best_name, best["params"])
        wb = {"beta": b, "eta": e, "gamma": g}
    return {"best_name": best_name, "best": best, "all": all_fits, "weibull_params": wb}


# -------------------------- 4) Pipeline principal --------------------------
def analyze_ttf_pipeline(ttf_series: List[float]) -> Dict[str, Any]:
    data = _clean_positive(ttf_series)
    n = data.size
    if n < 3:
        return {"error": "TTF insuffisants (<3 après nettoyage).", "cleaned_n": int(n)}

    mk = mann_kendall_test(data.tolist())
    dep = test_correlation(data.tolist())

    if mk["has_trend"]:
        model = "NHPP"
    elif dep["has_dep"]:
        model = "BPP"
    else:
        model = "RP"

    fits = fit_and_compare_distributions(data)
    best, best_name = fits["best"], fits["best_name"]

    beta = eta = gamma = None
    if fits["weibull_params"] is not None:
        beta = fits["weibull_params"]["beta"]
        eta  = fits["weibull_params"]["eta"]
        gamma = fits["weibull_params"]["gamma"]

    return {
        "cleaned_n": int(n),
        "model": model,
        "distribution": best_name,
        "fit": {"params": best.get("params"), "ks_p": best.get("ks_p"), "chi2_p": best.get("chi2_p")},
        "beta": beta, "eta": eta, "gamma": gamma,
        "details": {"mann_kendall": bool(mk["has_trend"]), "correlation": bool(dep["has_dep"])},
        "distribution_full": {"name": best_name, "aic": best.get("aic"), "ks_p": best.get("ks_p"),
                              "chi2_p": best.get("chi2_p"), "params": best.get("params")},
        "trend_mk": mk, "dependence": dep, "all_fits": fits["all"],
    }
