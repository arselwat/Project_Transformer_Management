from __future__ import annotations

"""
core/reliability/organigram.py

Pipeline fiabiliste pur (sans thermique), aligné sur l'organigramme métier.

Logique suivie :
1) Test de tendance
   - Méthode graphique (Crow-AMSAA / MIL-HDBK-189 sur log N(t) vs log t)
   - Test de Mann-Kendall
   - Test de Laplace
   => si tendance : NHPP (Power Law Process / Crow-AMSAA)

2) Si pas de tendance : test de dépendance
   - Méthode graphique (lag plot TTF_i vs TTF_{i+1})
   - Corrélation de Pearson
   - Corrélation de Spearman
   => si dépendance : BPP (approximation Hawkes exponentiel)

3) Si pas de dépendance : RP
   - Données considérées iid
   - Lois candidates : exponentielle, normale, lognormale, Weibull 2p, Weibull 3p
   - Estimation par maximum de vraisemblance
   - Estimation graphique / moindres carrés pour Weibull en complément
   - Tests d'ajustement : KS, Chi2, Cramér-von Mises
   - Si exponentielle retenue : signalement du cas particulier HPP

Sorties :
- reliability : résultats complets
- tables      : tableaux prêts pour Streamlit / PDF / Excel
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _clean_positive(series: List[float] | np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.asarray([], dtype=float)
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


def _to_event_times(ttf_series: List[float] | np.ndarray) -> np.ndarray:
    return np.cumsum(_clean_positive(ttf_series))


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else default
    except Exception:
        return default


def _aic_from_loglik(loglik: float, k: int) -> float:
    return float(2 * k - 2 * loglik)


def _round_df(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def _dominant_direction(values: List[str]) -> str:
    clean = [str(v) for v in values if str(v) in {"up", "down"}]
    if not clean:
        return "none"
    counts = Counter(clean)
    if counts["up"] > counts["down"]:
        return "up"
    if counts["down"] > counts["up"]:
        return "down"
    return clean[0]


def _strength_label(x: Optional[float]) -> str:
    v = abs(_safe_float(x, 0.0) or 0.0)
    if v < 0.20:
        return "very_low"
    if v < 0.40:
        return "low"
    if v < 0.60:
        return "medium"
    if v < 0.80:
        return "high"
    return "very_high"


def _merge_small_bins(
    observed: np.ndarray,
    expected: np.ndarray,
    min_exp: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = observed.astype(float).tolist()
    exp = expected.astype(float).tolist()
    i = 0
    while i < len(exp):
        if exp[i] < min_exp:
            if i + 1 < len(exp):
                exp[i] += exp[i + 1]
                obs[i] += obs[i + 1]
                del exp[i + 1]
                del obs[i + 1]
                continue
            if i - 1 >= 0:
                exp[i - 1] += exp[i]
                obs[i - 1] += obs[i]
                del exp[i]
                del obs[i]
                i = max(i - 1, 0)
                continue
        i += 1
    return np.asarray(obs), np.asarray(exp)


def _chi2_gof_prob_bins(
    data: np.ndarray,
    cdf_func,
    params: Tuple[float, ...],
    nbins: int = 8,
) -> float:
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

    obs, exp = _merge_small_bins(observed, np.asarray(expected), min_exp=5.0)
    if exp.size < 2 or (exp <= 0).any() or not np.all(np.isfinite(exp)):
        return 1.0

    try:
        _, p = sst.chisquare(f_obs=obs, f_exp=exp)
        return float(p)
    except Exception:
        return 1.0


def _cvm_gof(data: np.ndarray, cdf_func, params: Tuple[float, ...]) -> float:
    if data.size < 3:
        return 1.0
    try:
        res = sst.cramervonmises(data, lambda z: cdf_func(z, *params))
        return float(res.pvalue)
    except Exception:
        return 1.0


def _safe_logpdf_sum(dist, data: np.ndarray, params: Tuple[float, ...]) -> float:
    try:
        val = float(dist.logpdf(data, *params).sum())
        return val if np.isfinite(val) else -np.inf
    except Exception:
        return -np.inf


def _safe_hazard(dist, t: float, params: Tuple[float, ...]) -> Optional[float]:
    try:
        sf = float(dist.sf(t, *params))
        if sf <= 0:
            return None
        hz = float(dist.pdf(t, *params) / sf)
        return hz if np.isfinite(hz) else None
    except Exception:
        return None


def _gof_acceptance_for_rp(ks_p: Any, chi2_p: Any, alpha: float) -> Optional[bool]:
    checks = []
    ks_v = _safe_float(ks_p)
    chi_v = _safe_float(chi2_p)
    if ks_v is not None:
        checks.append(ks_v >= alpha)
    if chi_v is not None:
        checks.append(chi_v >= alpha)
    if not checks:
        return None
    return bool(all(checks))


def _gof_acceptance_generic(ks_p: Any, cvm_p: Any, alpha: float) -> Optional[bool]:
    checks = []
    ks_v = _safe_float(ks_p)
    cvm_v = _safe_float(cvm_p)
    if ks_v is not None:
        checks.append(ks_v >= alpha)
    if cvm_v is not None:
        checks.append(cvm_v >= alpha)
    if not checks:
        return None
    return bool(all(checks))


# ---------------------------------------------------------------------
# 1) Tendance
# ---------------------------------------------------------------------
def graphical_trend_test(ttf_series: List[float]) -> Dict[str, Any]:
    """
    Méthode graphique inspirée de Crow-AMSAA / MIL-HDBK-189.
    On ajuste log(N(t)) ~ a + b log(t).
    - b > 1  : intensité croissante -> tendance up
    - b < 1  : intensité décroissante -> tendance down
    - b ~= 1 : pas de tendance claire
    """
    t = _to_event_times(ttf_series)
    n = len(t)
    if n < 3:
        return {
            "beta_graph": 1.0,
            "slope_loglog": 1.0,
            "intercept_loglog": 0.0,
            "r2": None,
            "has_trend": False,
            "direction": "none",
            "graphical_signal": "none",
            "method": "crow_amsaa_loglog",
        }

    idx = np.arange(1, n + 1, dtype=float)
    x = np.log(t)
    y = np.log(idx)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    if slope > 1.05:
        direction = "up"
    elif slope < 0.95:
        direction = "down"
    else:
        direction = "none"

    return {
        "beta_graph": float(slope),
        "slope_loglog": float(slope),
        "intercept_loglog": float(intercept),
        "r2": r2,
        "has_trend": direction != "none",
        "direction": direction,
        "graphical_signal": direction,
        "method": "crow_amsaa_loglog",
    }


def mann_kendall_test(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    x = _clean_positive(series)
    n = len(x)
    if n < 3:
        return {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"}

    diffs = x[np.newaxis, :] - x[:, np.newaxis]
    s_val = float(np.sign(np.triu(diffs, 1)).sum())

    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"}

    if s_val > 0:
        z = (s_val - 1) / math.sqrt(var_s)
    elif s_val < 0:
        z = (s_val + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - sst.norm.cdf(abs(z)))
    has = bool(p < alpha)
    direction = "up" if (has and z > 0) else "down" if (has and z < 0) else "none"
    return {"z": float(z), "p": float(p), "has_trend": has, "direction": direction}


def laplace_trend_test(ttf_series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    t = _to_event_times(ttf_series)
    n = len(t)
    if n < 3:
        return {"u": 0.0, "p": 1.0, "has_trend": False, "direction": "none"}

    T = float(t[-1])
    mean_t = float(np.mean(t))
    u = math.sqrt(12.0 * n) / T * (mean_t - T / 2.0)
    p = 2.0 * (1.0 - sst.norm.cdf(abs(u)))
    has = bool(p < alpha)
    direction = "up" if (has and u > 0) else "down" if (has and u < 0) else "none"
    return {"u": float(u), "p": float(p), "has_trend": has, "direction": direction}


def combine_trend_evidence(
    graphical: Dict[str, Any],
    mk: Dict[str, Any],
    lap: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    sig_votes: List[str] = []
    if mk.get("has_trend"):
        sig_votes.append(str(mk.get("direction", "none")))
    if lap.get("has_trend"):
        sig_votes.append(str(lap.get("direction", "none")))

    support_votes = sig_votes.copy()
    g_dir = str(graphical.get("direction", "none"))
    if g_dir in {"up", "down"}:
        support_votes.append(g_dir)

    has_trend = bool(mk.get("has_trend") or lap.get("has_trend"))
    direction = _dominant_direction(support_votes)

    if mk.get("has_trend") and lap.get("has_trend") and str(mk.get("direction")) == str(lap.get("direction")):
        confidence = "high"
    elif has_trend and g_dir != "none" and direction == g_dir:
        confidence = "medium"
    elif has_trend:
        confidence = "medium"
    elif g_dir != "none":
        confidence = "weak"
    else:
        confidence = "none"

    return {
        "has_trend": has_trend,
        "direction": direction,
        "confidence": confidence,
        "graphical_direction": g_dir,
        "mk_sig": bool(mk.get("has_trend")),
        "lap_sig": bool(lap.get("has_trend")),
        "reason": (
            "Tendance confirmée par les tests de tendance; branche NHPP retenue."
            if has_trend
            else "Pas de tendance statistiquement significative; passage au test de dépendance."
        ),
    }


# ---------------------------------------------------------------------
# 2) Dépendance
# ---------------------------------------------------------------------
def graphical_dependence_test(series: List[float]) -> Dict[str, Any]:
    """
    Méthode graphique sur lag plot (TTF_i, TTF_{i+1}).
    Le signal graphique est qualitatif; la décision reste confirmée
    par les tests de corrélation Pearson / Spearman.
    """
    x = _clean_positive(series)
    if len(x) < 3:
        return {
            "lag1_r": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": None,
            "direction": "none",
            "has_dependence": False,
            "strength": "very_low",
            "method": "lag_plot",
        }

    a = x[:-1]
    b = x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return {
            "lag1_r": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": None,
            "direction": "none",
            "has_dependence": False,
            "strength": "very_low",
            "method": "lag_plot",
        }

    r = float(np.corrcoef(a, b)[0, 1])
    slope, intercept = np.polyfit(a, b, 1)
    y_hat = intercept + slope * a
    ss_res = float(np.sum((b - y_hat) ** 2))
    ss_tot = float(np.sum((b - np.mean(b)) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    if r > 0.20:
        direction = "positive"
    elif r < -0.20:
        direction = "negative"
    else:
        direction = "none"

    return {
        "lag1_r": r,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "direction": direction,
        "has_dependence": abs(r) >= 0.20,
        "strength": _strength_label(r),
        "method": "lag_plot",
    }


def dependence_correlation_test(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    x = _clean_positive(series)
    if len(x) < 3:
        return {
            "r": 0.0,
            "p": 1.0,
            "has_dep": False,
            "method": "spearman",
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "strength": "very_low",
        }

    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return {
            "r": 0.0,
            "p": 1.0,
            "has_dep": False,
            "method": "spearman",
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "strength": "very_low",
        }

    pr, pp = sst.pearsonr(a, b)
    sr, sp = sst.spearmanr(a, b)

    method = "spearman" if abs(sr) >= abs(pr) else "pearson"
    r = float(sr if method == "spearman" else pr)
    p = float(sp if method == "spearman" else pp)
    has_dep = bool((sp < alpha) or (pp < alpha))

    return {
        "r": r,
        "p": p,
        "has_dep": has_dep,
        "method": method,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "strength": _strength_label(r),
    }


def combine_dependence_evidence(
    graphical: Dict[str, Any],
    corr: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    has_dep = bool(corr.get("has_dep"))
    return {
        "has_dep": has_dep,
        "method": corr.get("method"),
        "r": corr.get("r"),
        "p": corr.get("p"),
        "pearson_r": corr.get("pearson_r"),
        "pearson_p": corr.get("pearson_p"),
        "spearman_r": corr.get("spearman_r"),
        "spearman_p": corr.get("spearman_p"),
        "graphical_direction": graphical.get("direction"),
        "graphical_r": graphical.get("lag1_r"),
        "strength": corr.get("strength"),
        "reason": (
            "Dépendance détectée; branche BPP retenue."
            if has_dep
            else "Pas de dépendance significative; passage à la branche RP."
        ),
    }


# ---------------------------------------------------------------------
# 3) Branche RP : lois iid
# ---------------------------------------------------------------------
def weibull_probability_plot_ls(data: np.ndarray) -> Dict[str, Any]:
    """
    Estimation graphique / moindres carrés pour Weibull 2 paramètres.
    Utilisée comme estimation complémentaire, non comme règle principale de sélection.
    """
    x = np.sort(_clean_positive(data))
    n = len(x)
    if n < 3:
        return {
            "beta_ls": None,
            "eta_ls": None,
            "intercept": None,
            "r2": None,
            "method": "weibull_probability_plot_ls",
        }

    ranks = np.arange(1, n + 1, dtype=float)
    F = (ranks - 0.3) / (n + 0.4)
    F = np.clip(F, 1e-8, 1.0 - 1e-8)

    X = np.log(x)
    Y = np.log(-np.log(1.0 - F))
    slope, intercept = np.polyfit(X, Y, 1)
    if slope <= 0:
        return {
            "beta_ls": None,
            "eta_ls": None,
            "intercept": float(intercept),
            "r2": None,
            "method": "weibull_probability_plot_ls",
        }

    Y_hat = intercept + slope * X
    ss_res = float(np.sum((Y - Y_hat) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    beta_ls = float(slope)
    eta_ls = float(np.exp(-intercept / slope))
    return {
        "beta_ls": beta_ls,
        "eta_ls": eta_ls,
        "intercept": float(intercept),
        "r2": r2,
        "method": "weibull_probability_plot_ls",
    }


def _fit_distribution(name: str, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    res = {
        "name": name,
        "params": None,
        "loglik": None,
        "aic": np.inf,
        "ks_p": None,
        "chi2_p": None,
        "cvm_p": None,
        "accepted": None,
        "estimation_method": "MLE",
        "ls_params": None,
    }

    try:
        if name == "expon":
            dist = sst.expon
            params = dist.fit(data, floc=0.0)
            k = 2
        elif name == "norm":
            dist = sst.norm
            params = dist.fit(data)
            k = 2
        elif name == "lognorm":
            dist = sst.lognorm
            params = dist.fit(data, floc=0.0)
            k = 3
        elif name == "weibull_2p":
            dist = sst.weibull_min
            params = dist.fit(data, floc=0.0)
            k = 2
        elif name == "weibull_3p":
            dist = sst.weibull_min
            params = dist.fit(data)
            k = 3
        else:
            return res

        loglik = _safe_logpdf_sum(dist, data, params)
        if not np.isfinite(loglik):
            return res

        _, ks_p = sst.kstest(data, dist.cdf, args=params)
        chi2_p = _chi2_gof_prob_bins(data, dist.cdf, params)
        cvm_p = _cvm_gof(data, dist.cdf, params)

        res["params"] = tuple(float(v) for v in params)
        res["loglik"] = float(loglik)
        res["aic"] = _aic_from_loglik(loglik, k)
        res["ks_p"] = float(ks_p)
        res["chi2_p"] = float(chi2_p)
        res["cvm_p"] = float(cvm_p)
        res["accepted"] = _gof_acceptance_for_rp(ks_p, chi2_p, alpha)

        if name == "weibull_2p":
            ls = weibull_probability_plot_ls(data)
            if ls.get("beta_ls") is not None:
                res["ls_params"] = {
                    "beta_ls": ls.get("beta_ls"),
                    "eta_ls": ls.get("eta_ls"),
                    "r2": ls.get("r2"),
                }

        return res
    except Exception:
        return res


def _normalize_weibull_params(name: str, params: Tuple[float, ...]) -> Tuple[float, float, float]:
    c, loc, scale = params
    if name == "weibull_2p":
        return float(c), float(scale), 0.0
    return float(c), float(scale), float(loc)


def fit_and_compare_distributions(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    candidates = ["expon", "norm", "lognorm", "weibull_2p", "weibull_3p"]
    all_fits = {name: _fit_distribution(name, data, alpha=alpha) for name in candidates}

    accepted_names = [
        name for name, fit in all_fits.items()
        if fit.get("accepted") is True and np.isfinite(_safe_float(fit.get("aic"), np.inf))
    ]

    if accepted_names:
        best_name = min(accepted_names, key=lambda n: all_fits[n]["aic"])
        selected_by = "accepted_min_aic"
    else:
        best_name = min(all_fits.keys(), key=lambda n: all_fits[n]["aic"])
        selected_by = "min_aic"

    best = all_fits[best_name]

    beta = eta = gamma = None
    lambda_h = None
    weibull_ls = None

    if best_name.startswith("weibull") and best.get("params"):
        beta, eta, gamma = _normalize_weibull_params(best_name, best["params"])
        weibull_ls = best.get("ls_params")
    elif best_name == "expon" and best.get("params"):
        _loc, scale = best["params"]
        if scale and scale > 0:
            lambda_h = float(1.0 / scale)

    return {
        "best_name": best_name,
        "best": best,
        "all": all_fits,
        "selected_by": selected_by,
        "weibull": {"beta": beta, "eta": eta, "gamma": gamma} if beta is not None else None,
        "weibull_ls": weibull_ls,
        "hpp": {"lambda_h": lambda_h} if lambda_h is not None else None,
    }


# ---------------------------------------------------------------------
# 4) Branche NHPP : Crow-AMSAA / Power Law Process
# ---------------------------------------------------------------------
def fit_power_law_nhpp(ttf_series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    t = _to_event_times(ttf_series)
    n = len(t)
    res = {
        "name": "power_law_nhpp",
        "params": None,
        "loglik": None,
        "aic": np.inf,
        "ks_p": None,
        "chi2_p": None,
        "cvm_p": None,
        "accepted": None,
        "beta": None,
        "eta": None,
        "T_end": None,
        "estimation_method": "MLE_closed_form",
    }
    if n < 2:
        return res

    T = float(t[-1])
    denom = float(np.sum(np.log(T / t)))
    if denom <= 0:
        return res

    beta = float(n / denom)
    eta = float(T / (n ** (1.0 / beta)))

    ll = (
        n * math.log(beta)
        - n * beta * math.log(eta)
        + (beta - 1.0) * float(np.sum(np.log(t)))
        - (T / eta) ** beta
    )

    m_prev = 0.0
    z = []
    for ti in t:
        m_i = (ti / eta) ** beta
        z.append(m_i - m_prev)
        m_prev = m_i
    z = np.asarray(z, dtype=float)

    try:
        _, ks_p = sst.kstest(z, sst.expon.cdf, args=(0.0, 1.0))
    except Exception:
        ks_p = 1.0

    chi2_p = _chi2_gof_prob_bins(z, sst.expon.cdf, (0.0, 1.0))
    cvm_p = _cvm_gof(z, sst.expon.cdf, (0.0, 1.0))

    res.update(
        {
            "params": (beta, eta),
            "loglik": float(ll),
            "aic": _aic_from_loglik(ll, 2),
            "ks_p": float(ks_p),
            "chi2_p": float(chi2_p),
            "cvm_p": float(cvm_p),
            "accepted": _gof_acceptance_generic(ks_p, cvm_p, alpha),
            "beta": beta,
            "eta": eta,
            "T_end": float(T),
        }
    )
    return res


# ---------------------------------------------------------------------
# 5) Branche BPP : approximation Hawkes exponentiel
# ---------------------------------------------------------------------
def _hawkes_neg_loglik(theta: np.ndarray, t: np.ndarray) -> float:
    log_mu, log_beta, q = theta
    mu = math.exp(log_mu)
    beta = math.exp(log_beta)
    branch_ratio = 0.98 / (1.0 + math.exp(-q))
    alpha = branch_ratio * beta

    T = float(t[-1])
    if mu <= 0 or beta <= 0 or alpha < 0:
        return 1e100

    r_prev = 0.0
    log_terms = []
    t_prev = 0.0
    for i, ti in enumerate(t):
        if i == 0:
            r_i = 0.0
        else:
            dt = ti - t_prev
            r_i = math.exp(-beta * dt) * (1.0 + r_prev)
        lam = mu + alpha * r_i
        if lam <= 0 or not np.isfinite(lam):
            return 1e100
        log_terms.append(math.log(lam))
        r_prev = r_i
        t_prev = ti

    integral = mu * T + (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (T - t))))
    nll = -(float(np.sum(log_terms)) - integral)
    return nll if np.isfinite(nll) else 1e100


def fit_hawkes_bpp(ttf_series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    t = _to_event_times(ttf_series)
    n = len(t)
    res = {
        "name": "hawkes_exp_bpp",
        "params": None,
        "loglik": None,
        "aic": np.inf,
        "ks_p": None,
        "chi2_p": None,
        "cvm_p": None,
        "accepted": None,
        "mu": None,
        "alpha": None,
        "beta_kernel": None,
        "branch_ratio": None,
        "T_end": None,
        "estimation_method": "MLE_numerical",
    }
    if n < 4:
        return res

    mean_gap = float(np.mean(np.diff(np.insert(t, 0, 0.0))))
    x0 = np.array([
        math.log(max(1e-6, 1.0 / max(mean_gap, 1e-6))),
        math.log(1.0),
        0.0,
    ])

    opt = minimize(_hawkes_neg_loglik, x0=x0, args=(t,), method="L-BFGS-B")
    if not opt.success:
        return res

    log_mu, log_beta, q = opt.x
    mu = math.exp(log_mu)
    beta_k = math.exp(log_beta)
    branch_ratio = 0.98 / (1.0 + math.exp(-q))
    alpha_h = branch_ratio * beta_k
    ll = -float(opt.fun)
    T = float(t[-1])

    w = []
    A_prev = 0.0
    t_prev = 0.0
    for ti in t:
        dt = ti - t_prev
        compensator = mu * dt + (alpha_h / beta_k) * A_prev * (1.0 - math.exp(-beta_k * dt))
        w.append(compensator)
        A_before = A_prev * math.exp(-beta_k * dt)
        A_prev = A_before + 1.0
        t_prev = ti
    w = np.asarray(w, dtype=float)

    try:
        _, ks_p = sst.kstest(w, sst.expon.cdf, args=(0.0, 1.0))
    except Exception:
        ks_p = 1.0

    chi2_p = _chi2_gof_prob_bins(w, sst.expon.cdf, (0.0, 1.0))
    cvm_p = _cvm_gof(w, sst.expon.cdf, (0.0, 1.0))

    res.update(
        {
            "params": (float(mu), float(alpha_h), float(beta_k), float(branch_ratio)),
            "loglik": ll,
            "aic": _aic_from_loglik(ll, 3),
            "ks_p": float(ks_p),
            "chi2_p": float(chi2_p),
            "cvm_p": float(cvm_p),
            "accepted": _gof_acceptance_generic(ks_p, cvm_p, alpha),
            "mu": float(mu),
            "alpha": float(alpha_h),
            "beta_kernel": float(beta_k),
            "branch_ratio": float(branch_ratio),
            "T_end": float(T),
        }
    )
    return res


# ---------------------------------------------------------------------
# 6) Courbes fiabilistes
# ---------------------------------------------------------------------
def build_reliability_curves(
    ttf_series: List[float] | np.ndarray,
    model: str,
    distribution: str,
    params: Dict[str, Any],
    points: int = 200,
) -> pd.DataFrame:
    ttf = _clean_positive(ttf_series)
    if ttf.size == 0:
        return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])

    t_max = max(float(np.max(ttf)) * 1.5, float(np.mean(ttf)) * 2.0)
    grid = np.linspace(1e-8, t_max, points)

    if model == "RP":
        raw = params.get("raw")
        if not raw:
            return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])

        if distribution == "expon":
            dist = sst.expon
        elif distribution == "norm":
            dist = sst.norm
        elif distribution == "lognorm":
            dist = sst.lognorm
        elif distribution.startswith("weibull"):
            dist = sst.weibull_min
        else:
            return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])

        f_t = dist.pdf(grid, *raw)
        F_t = dist.cdf(grid, *raw)
        R_t = dist.sf(grid, *raw)
        h_t = np.divide(f_t, R_t, out=np.full_like(f_t, np.nan), where=R_t > 1e-12)

    elif model == "NHPP":
        beta = _safe_float(params.get("beta"))
        eta = _safe_float(params.get("eta"))
        if beta is None or eta is None or beta <= 0 or eta <= 0:
            return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])
        F_t = 1.0 - np.exp(-((grid / eta) ** beta))
        R_t = 1.0 - F_t
        f_t = (beta / eta) * ((grid / eta) ** (beta - 1.0)) * np.exp(-((grid / eta) ** beta))
        h_t = (beta / eta) * ((grid / eta) ** (beta - 1.0))

    elif model == "BPP":
        mu = _safe_float(params.get("mu"))
        if mu is None or mu <= 0:
            return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])
        F_t = 1.0 - np.exp(-mu * grid)
        R_t = np.exp(-mu * grid)
        f_t = mu * np.exp(-mu * grid)
        h_t = np.full_like(grid, mu)

    else:
        return pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])

    return _round_df(pd.DataFrame({
        "t": grid,
        "R_t": R_t,
        "F_t": F_t,
        "f_t": f_t,
        "h_t": h_t,
    }))


# ---------------------------------------------------------------------
# 7) Indicateurs fiabilistes
# ---------------------------------------------------------------------
def _distribution_mean(name: str, params: Optional[Tuple[float, ...]]) -> Optional[float]:
    if not params:
        return None
    try:
        if name == "expon":
            return float(sst.expon.mean(*params))
        if name == "norm":
            return float(sst.norm.mean(*params))
        if name == "lognorm":
            return float(sst.lognorm.mean(*params))
        if name.startswith("weibull"):
            return float(sst.weibull_min.mean(*params))
        return None
    except Exception:
        return None


def _distribution_hazard_at_mean(name: str, params: Optional[Tuple[float, ...]]) -> Optional[float]:
    if not params:
        return None
    mean_ = _distribution_mean(name, params)
    if mean_ is None or mean_ <= 0:
        return None
    try:
        if name == "expon":
            return _safe_hazard(sst.expon, mean_, params)
        if name == "norm":
            return _safe_hazard(sst.norm, mean_, params)
        if name == "lognorm":
            return _safe_hazard(sst.lognorm, mean_, params)
        if name.startswith("weibull"):
            return _safe_hazard(sst.weibull_min, mean_, params)
        return None
    except Exception:
        return None


def _maintenance_recommendation(
    model: str,
    process_variant: str,
    distribution: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    beta = _safe_float(params.get("beta"))

    if model == "NHPP":
        return {
            "maintenance_type": "Préventif renforcé",
            "priority": "Élevée",
            "reason": "Tendance détectée sur le processus de défaillance; une action proactive est recommandée.",
        }

    if model == "BPP":
        return {
            "maintenance_type": "Surveillance active",
            "priority": "Élevée",
            "reason": "Dépendance entre événements détectée; le processus n'est pas iid.",
        }

    if process_variant == "HPP":
        return {
            "maintenance_type": "Maintenance systématique simple",
            "priority": "Normale",
            "reason": "La loi exponentielle suggère un taux de défaillance à peu près constant.",
        }

    if distribution.startswith("weibull") and beta is not None:
        if beta > 1.2:
            return {
                "maintenance_type": "Préventif planifié",
                "priority": "Élevée",
                "reason": "Beta > 1 : phase d'usure probable, cohérente avec le vieillissement d'équipements de puissance.",
            }
        if beta < 0.8:
            return {
                "maintenance_type": "Correctif surveillé",
                "priority": "Modérée",
                "reason": "Beta < 1 : défaillances précoces ou aléatoires dominantes.",
            }

    return {
        "maintenance_type": "Surveillance nominale",
        "priority": "Normale",
        "reason": "Aucun signal critique dominant n'est observé dans la branche retenue.",
    }


def compute_reliability_indicators(
    ttf_series: List[float] | np.ndarray,
    repair_series: Optional[List[float] | np.ndarray] = None,
    *,
    model: str,
    distribution: str,
    process_variant: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    ttf = _clean_positive(ttf_series)
    rep = _clean_positive(repair_series) if repair_series is not None else np.asarray([], dtype=float)

    empirical_mttf = float(np.mean(ttf)) if ttf.size else None
    theoretical_mttf = None
    mean_hazard = None

    if model == "RP":
        theoretical_mttf = _distribution_mean(distribution, params.get("raw"))
        mean_hazard = _distribution_hazard_at_mean(distribution, params.get("raw"))
    elif model == "NHPP":
        theoretical_mttf = empirical_mttf
        beta = _safe_float(params.get("beta"))
        eta = _safe_float(params.get("eta"))
        if beta is not None and eta is not None:
            try:
                t_ref = float(np.mean(_to_event_times(ttf.tolist())))
                mean_hazard = float((beta / eta) * ((t_ref / eta) ** (beta - 1.0)))
            except Exception:
                mean_hazard = None
    elif model == "BPP":
        theoretical_mttf = empirical_mttf
        mu = _safe_float(params.get("mu"))
        if mu is not None and mu > 0:
            mean_hazard = float(mu)

    mttr = float(np.mean(rep)) if rep.size else None
    mtbf = theoretical_mttf if theoretical_mttf is not None else empirical_mttf
    availability = None
    if mtbf is not None and mttr is not None and (mtbf + mttr) > 0:
        availability = float(mtbf / (mtbf + mttr))

    recommendation = _maintenance_recommendation(
        model=model,
        process_variant=process_variant,
        distribution=distribution,
        params=params,
    )

    return {
        "cleaned_failures_n": int(ttf.size),
        "cleaned_repairs_n": int(rep.size),
        "empirical_mttf_h": None if empirical_mttf is None else float(empirical_mttf),
        "theoretical_mttf_h": None if theoretical_mttf is None else float(theoretical_mttf),
        "mtbf_h": None if mtbf is None else float(mtbf),
        "mttr_h": None if mttr is None else float(mttr),
        "availability_intrinsic": availability,
        "sample_std_ttf_h": float(np.std(ttf, ddof=1)) if ttf.size >= 2 else None,
        "mean_failure_rate_h": mean_hazard,
        "empirical_failure_rate_h": None if empirical_mttf in (None, 0) else float(1.0 / empirical_mttf),
        **recommendation,
    }


# ---------------------------------------------------------------------
# 8) Moteur fiabiliste principal
# ---------------------------------------------------------------------
def analyze_reliability_only(
    ttf_series: List[float],
    repair_series: Optional[List[float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    data = _clean_positive(ttf_series)
    n = int(data.size)

    if n < 3:
        default_params = {
            "raw": None,
            "beta": None,
            "eta": None,
            "gamma": None,
            "mu": None,
            "alpha": None,
            "beta_kernel": None,
            "branch_ratio": None,
            "lambda_hpp_h": None,
            "beta_ls": None,
            "eta_ls": None,
        }
        indicators = compute_reliability_indicators(
            ttf_series=data,
            repair_series=repair_series,
            model="RP",
            distribution="weibull_2p",
            process_variant="RP",
            params=default_params,
        )
        return {
            "error": "TTF insuffisants (<3 après nettoyage).",
            "cleaned_n": n,
            "model": "RP",
            "process_variant": "RP",
            "distribution": "weibull_2p",
            "decision": {
                "has_trend": False,
                "trend_direction": "none",
                "trend_confidence": "none",
                "has_dependence": False,
                "selected_process": "RP",
                "selected_variant": "RP",
                "entity_assumption": "Données insuffisantes pour conclure",
                "law_selected": "weibull_2p",
                "law_accepted": None,
                "selection_rule": None,
                "reason": "Données insuffisantes pour statuer, hypothèse RP par défaut.",
            },
            "goodness": {"aic": None, "ks_p": None, "chi2_p": None, "cvm_p": None, "accepted": None},
            "params": default_params,
            "tests": {
                "trend_graphical": {"direction": "none", "has_trend": False},
                "trend_mk": {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
                "trend_laplace": {"u": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
                "trend_combined": {"has_trend": False, "direction": "none", "confidence": "none"},
                "dependence_graphical": {"direction": "none", "has_dependence": False},
                "dependence_correlation": {"has_dep": False, "r": 0.0, "p": 1.0},
                "dependence": {"has_dep": False, "r": 0.0, "p": 1.0},
            },
            "candidates": {},
            "curves": build_reliability_curves(data, "RP", "weibull_2p", default_params),
            "indicators": indicators,
        }

    trend_graphical = graphical_trend_test(data.tolist())
    mk = mann_kendall_test(data.tolist(), alpha=alpha)
    lap = laplace_trend_test(data.tolist(), alpha=alpha)
    trend = combine_trend_evidence(trend_graphical, mk, lap, alpha=alpha)

    dep_graphical = graphical_dependence_test(data.tolist())
    dep_corr = dependence_correlation_test(data.tolist(), alpha=alpha)
    dep = combine_dependence_evidence(dep_graphical, dep_corr, alpha=alpha)

    beta = eta = gamma = None
    mu = alpha_bpp = beta_kernel = branch_ratio = None
    lambda_hpp_h = None
    beta_ls = eta_ls = None
    candidates: Dict[str, Any] = {}
    selected_by = None

    if trend["has_trend"]:
        model = "NHPP"
        process_variant = "NHPP"
        fit_res = fit_power_law_nhpp(data.tolist(), alpha=alpha)
        distribution = "power_law_nhpp"
        candidates = {"power_law_nhpp": fit_res}
        beta = fit_res.get("beta")
        eta = fit_res.get("eta")
        best = fit_res
        reason = "Tendance détectée : la logique de l'organigramme retient directement NHPP."
        entity_assumption = "Entité réparable / minimal repair"

    elif dep["has_dep"]:
        model = "BPP"
        process_variant = "BPP"
        fit_res = fit_hawkes_bpp(data.tolist(), alpha=alpha)
        distribution = "hawkes_exp_bpp"
        candidates = {"hawkes_exp_bpp": fit_res}
        mu = fit_res.get("mu")
        alpha_bpp = fit_res.get("alpha")
        beta_kernel = fit_res.get("beta_kernel")
        branch_ratio = fit_res.get("branch_ratio")
        best = fit_res
        reason = "Pas de tendance significative, mais dépendance détectée : la logique de l'organigramme retient BPP."
        entity_assumption = "Entité réparable avec dépendance entre événements"

    else:
        model = "RP"
        fit_bundle = fit_and_compare_distributions(data, alpha=alpha)
        best = fit_bundle["best"]
        distribution = fit_bundle["best_name"]
        candidates = fit_bundle["all"]
        selected_by = fit_bundle.get("selected_by")
        process_variant = "HPP" if distribution == "expon" else "RP"

        if fit_bundle["weibull"] is not None:
            beta = fit_bundle["weibull"]["beta"]
            eta = fit_bundle["weibull"]["eta"]
            gamma = fit_bundle["weibull"]["gamma"]

        weibull_ls = fit_bundle.get("weibull_ls") or {}
        beta_ls = weibull_ls.get("beta_ls")
        eta_ls = weibull_ls.get("eta_ls")

        if fit_bundle["hpp"] is not None:
            lambda_hpp_h = fit_bundle["hpp"].get("lambda_h")

        if process_variant == "HPP":
            reason = (
                "Ni tendance ni dépendance : branche RP retenue. La loi exponentielle étant sélectionnée, "
                "le cas particulier HPP est signalé."
            )
        else:
            reason = "Ni tendance ni dépendance : branche RP retenue avec choix de loi probabiliste."

        entity_assumption = "Entité non réparable ou réparable comme neuve"

    params = {
        "raw": best.get("params"),
        "beta": beta,
        "eta": eta,
        "gamma": gamma,
        "mu": mu,
        "alpha": alpha_bpp,
        "beta_kernel": beta_kernel,
        "branch_ratio": branch_ratio,
        "lambda_hpp_h": lambda_hpp_h,
        "beta_ls": beta_ls,
        "eta_ls": eta_ls,
    }

    indicators = compute_reliability_indicators(
        ttf_series=data,
        repair_series=repair_series,
        model=model,
        distribution=distribution,
        process_variant=process_variant,
        params=params,
    )

    curves = build_reliability_curves(
        ttf_series=data,
        model=model,
        distribution=distribution,
        params=params,
    )

    return {
        "cleaned_n": n,
        "model": model,
        "process_variant": process_variant,
        "distribution": distribution,
        "decision": {
            "has_trend": bool(trend["has_trend"]),
            "trend_direction": trend["direction"],
            "trend_confidence": trend["confidence"],
            "has_dependence": bool(dep["has_dep"]),
            "dependence_strength": dep.get("strength"),
            "selected_process": model,
            "selected_variant": process_variant,
            "entity_assumption": entity_assumption,
            "law_selected": distribution,
            "law_accepted": best.get("accepted"),
            "selection_rule": selected_by,
            "reason": reason,
        },
        "goodness": {
            "aic": best.get("aic"),
            "ks_p": best.get("ks_p"),
            "chi2_p": best.get("chi2_p"),
            "cvm_p": best.get("cvm_p"),
            "accepted": best.get("accepted"),
        },
        "params": params,
        "tests": {
            "trend_graphical": trend_graphical,
            "trend_mil_hdbk_189": trend_graphical,
            "trend_mk": mk,
            "trend_laplace": lap,
            "trend_combined": trend,
            "dependence_graphical": dep_graphical,
            "dependence_correlation": dep_corr,
            "dependence": dep,
        },
        "candidates": candidates,
        "curves": curves,
        "indicators": indicators,
    }


# ---------------------------------------------------------------------
# 9) Tables globales
# ---------------------------------------------------------------------
def build_reliability_tables(reliability_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tests = reliability_result.get("tests", {})
    decision = reliability_result.get("decision", {})
    params = reliability_result.get("params", {})
    goodness = reliability_result.get("goodness", {})
    indicators = reliability_result.get("indicators", {})

    trend_df = pd.DataFrame(
        [
            {
                "Test": "Méthode graphique",
                "Statistique": tests.get("trend_graphical", {}).get("beta_graph"),
                "p_value": None,
                "Décision": "Oui" if tests.get("trend_graphical", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_graphical", {}).get("direction"),
                "R2": tests.get("trend_graphical", {}).get("r2"),
            },
            {
                "Test": "Mann-Kendall",
                "Statistique": tests.get("trend_mk", {}).get("z"),
                "p_value": tests.get("trend_mk", {}).get("p"),
                "Décision": "Oui" if tests.get("trend_mk", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_mk", {}).get("direction"),
                "R2": None,
            },
            {
                "Test": "Laplace",
                "Statistique": tests.get("trend_laplace", {}).get("u"),
                "p_value": tests.get("trend_laplace", {}).get("p"),
                "Décision": "Oui" if tests.get("trend_laplace", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_laplace", {}).get("direction"),
                "R2": None,
            },
            {
                "Test": "Décision finale tendance",
                "Statistique": None,
                "p_value": None,
                "Décision": "Oui" if tests.get("trend_combined", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_combined", {}).get("direction"),
                "R2": None,
            },
        ]
    )

    dependence_df = pd.DataFrame(
        [
            {
                "Méthode": "Graphique (lag plot)",
                "r": tests.get("dependence_graphical", {}).get("lag1_r"),
                "p_value": None,
                "Dépendance": "Oui" if tests.get("dependence_graphical", {}).get("has_dependence") else "Non",
                "Direction": tests.get("dependence_graphical", {}).get("direction"),
            },
            {
                "Méthode": "Pearson",
                "r": tests.get("dependence_correlation", {}).get("pearson_r"),
                "p_value": tests.get("dependence_correlation", {}).get("pearson_p"),
                "Dépendance": "Oui" if (tests.get("dependence_correlation", {}).get("pearson_p") is not None and tests.get("dependence_correlation", {}).get("pearson_p") < 0.05) else "Non",
                "Direction": None,
            },
            {
                "Méthode": "Spearman",
                "r": tests.get("dependence_correlation", {}).get("spearman_r"),
                "p_value": tests.get("dependence_correlation", {}).get("spearman_p"),
                "Dépendance": "Oui" if (tests.get("dependence_correlation", {}).get("spearman_p") is not None and tests.get("dependence_correlation", {}).get("spearman_p") < 0.05) else "Non",
                "Direction": None,
            },
            {
                "Méthode": "Décision finale dépendance",
                "r": tests.get("dependence", {}).get("r"),
                "p_value": tests.get("dependence", {}).get("p"),
                "Dépendance": "Oui" if tests.get("dependence", {}).get("has_dep") else "Non",
                "Direction": tests.get("dependence", {}).get("graphical_direction"),
            },
        ]
    )

    process_df = pd.DataFrame(
        [
            {
                "Tendance": "Oui" if decision.get("has_trend") else "Non",
                "Direction tendance": decision.get("trend_direction"),
                "Dépendance": "Oui" if decision.get("has_dependence") else "Non",
                "Processus retenu": decision.get("selected_process"),
                "Variant": decision.get("selected_variant"),
                "Hypothèse entité": decision.get("entity_assumption"),
                "Justification": decision.get("reason"),
            }
        ]
    )

    rows = []
    for name, fit in reliability_result.get("candidates", {}).items():
        rows.append(
            {
                "Modèle": name,
                "Paramètres": str(fit.get("params")),
                "Méthode estimation": fit.get("estimation_method"),
                "KS p": fit.get("ks_p"),
                "Chi2 p": fit.get("chi2_p"),
                "CvM p": fit.get("cvm_p"),
                "Acceptée": fit.get("accepted"),
                "Retenue": "Oui" if name == reliability_result.get("distribution") else "Non",
            }
        )
    fits_df = pd.DataFrame(rows)

    reliability_summary_df = pd.DataFrame(
        [
            {
                "Processus": reliability_result.get("model"),
                "Variant": reliability_result.get("process_variant"),
                "Distribution": reliability_result.get("distribution"),
                "Beta": params.get("beta"),
                "Eta": params.get("eta"),
                "Gamma": params.get("gamma"),
                "Lambda_HPP (1/h)": params.get("lambda_hpp_h"),
                "Mu": params.get("mu"),
                "Alpha": params.get("alpha"),
                "KS p": goodness.get("ks_p"),
                "Chi2 p": goodness.get("chi2_p"),
                "CvM p": goodness.get("cvm_p"),
                "Ajustement accepté": goodness.get("accepted"),
                "MTTF (h)": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
                "MTBF (h)": indicators.get("mtbf_h"),
                "MTTR (h)": indicators.get("mttr_h"),
                "Disponibilité": indicators.get("availability_intrinsic"),
                "Taux de défaillance moyen (1/h)": indicators.get("mean_failure_rate_h"),
                "Type maintenance": indicators.get("maintenance_type"),
                "Priorité": indicators.get("priority"),
            }
        ]
    )

    recommendation_df = pd.DataFrame(
        [
            {
                "Type de maintenance": indicators.get("maintenance_type"),
                "Priorité": indicators.get("priority"),
                "Raison": indicators.get("reason"),
            }
        ]
    )

    curves_df = reliability_result.get("curves")
    if not isinstance(curves_df, pd.DataFrame):
        curves_df = pd.DataFrame(columns=["t", "R_t", "F_t", "f_t", "h_t"])

    return {
        "trend_results": _round_df(trend_df),
        "dependence_results": _round_df(dependence_df),
        "process_choice": _round_df(process_df),
        "fit_candidates": _round_df(fits_df),
        "reliability_summary": _round_df(reliability_summary_df),
        "maintenance_recommendation": _round_df(recommendation_df),
        "reliability_curves": _round_df(curves_df),
    }


def build_global_result_tables(
    reliability_result: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    tables.update(build_reliability_tables(reliability_result))
    return tables


# ---------------------------------------------------------------------
# 10) Fonction principale intégrée pour le logiciel
# ---------------------------------------------------------------------
def analyze_ttf_pipeline(
    ttf_series: List[float],
    alpha: float = 0.05,
    repair_series: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Entrées :
    - ttf_series    : série des temps entre défaillances (heures)
    - alpha         : seuil de significativité
    - repair_series : série des temps de réparation (heures), optionnelle

    Sorties :
    - reliability : résultats fiabilistes complets
    - tables      : tableaux prêts pour Streamlit / PDF
    """
    reliability = analyze_reliability_only(
        ttf_series=ttf_series,
        repair_series=repair_series,
        alpha=alpha,
    )

    tables = build_global_result_tables(
        reliability_result=reliability,
    )

    return {
        "reliability": reliability,
        "tables": tables,
    }


analyze_integrated_pipeline = analyze_ttf_pipeline
