from __future__ import annotations

"""
core/reliability/organigram.py

Pipeline fiabiliste + thermo-fiabiliste aligné sur l'organigramme métier.

Logique respectée :
1) Déterminer s'il existe une tendance
   - Méthode graphique (proxy MIL-HDBK-189)
   - Test de Mann-Kendall
   - Test de Laplace
   => si tendance : NHPP

2) Sinon, tester la dépendance
   - Corrélation Pearson + Spearman
   => si dépendance : BPP

3) Sinon, retenir un processus de renouvellement (RP)
   - Données indépendantes et identiquement distribuées
   - Choix d'une loi : expon, norm, lognorm, weibull_2p, weibull_3p
   - Estimation des paramètres
   - Tests d'ajustement : KS, Chi2, Cramér-von Mises
   - Cas particulier : si la loi retenue est exponentielle, on signale aussi le variant HPP

4) En parallèle si fourni :
   - bloc thermique dynamique
   - top-oil, point chaud, FAA, perte de vie

Sorties conservées compatibles avec les pages Streamlit existantes :
- reliability
- thermal
- tables
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
    x = _clean_positive(ttf_series)
    return np.cumsum(x)


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
        lp = dist.logpdf(data, *params).sum()
        return float(lp) if np.isfinite(lp) else -np.inf
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

    p = 2 * (1 - sst.norm.cdf(abs(z)))
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
    u = math.sqrt(12 * n) / T * (mean_t - T / 2.0)
    p = 2 * (1 - sst.norm.cdf(abs(u)))
    has = bool(p < alpha)
    direction = "up" if (has and u > 0) else "down" if (has and u < 0) else "none"
    return {"u": float(u), "p": float(p), "has_trend": has, "direction": direction}


def mil_hdbk_189_indicator(ttf_series: List[float]) -> Dict[str, Any]:
    t = _to_event_times(ttf_series)
    n = len(t)
    if n < 3:
        return {
            "beta_graph": 1.0,
            "slope_loglog": 1.0,
            "intercept_loglog": 0.0,
            "interpreted_trend": "none",
        }

    idx = np.arange(1, n + 1, dtype=float)
    x = np.log(t)
    y = np.log(idx)
    slope, intercept = np.polyfit(x, y, 1)
    beta = float(slope)

    if beta > 1.05:
        tr = "up"
    elif beta < 0.95:
        tr = "down"
    else:
        tr = "none"

    return {
        "beta_graph": beta,
        "slope_loglog": beta,
        "intercept_loglog": float(intercept),
        "interpreted_trend": tr,
    }


def combine_trend_evidence(
    mk: Dict[str, Any],
    lap: Dict[str, Any],
    mil: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    sig_votes = []
    if mk.get("has_trend"):
        sig_votes.append(str(mk.get("direction", "none")))
    if lap.get("has_trend"):
        sig_votes.append(str(lap.get("direction", "none")))

    support_votes = sig_votes.copy()
    mil_dir = str(mil.get("interpreted_trend", "none"))
    if mil_dir in {"up", "down"}:
        support_votes.append(mil_dir)

    has_trend = bool(mk.get("has_trend") or lap.get("has_trend"))
    direction = _dominant_direction(support_votes)

    if mk.get("has_trend") and lap.get("has_trend") and str(mk.get("direction")) == str(lap.get("direction")):
        confidence = "high"
    elif has_trend:
        confidence = "medium"
    elif mil_dir != "none":
        confidence = "weak"
    else:
        confidence = "none"

    return {
        "has_trend": has_trend,
        "direction": direction,
        "confidence": confidence,
        "mk_sig": bool(mk.get("has_trend")),
        "lap_sig": bool(lap.get("has_trend")),
        "mil_direction": mil_dir,
        "reason": (
            "Tendance confirmée par tests statistiques."
            if has_trend
            else "Pas de tendance statistiquement significative."
        ),
    }


# ---------------------------------------------------------------------
# 2) Dépendance
# ---------------------------------------------------------------------
def test_dependence(series: List[float], alpha: float = 0.05) -> Dict[str, Any]:
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

    method = "spearman"
    r = float(sr)
    p = float(sp)
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


# ---------------------------------------------------------------------
# 3) Branche RP : lois iid
# ---------------------------------------------------------------------
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
        return res
    except Exception:
        return res


def _normalize_weibull_params(name: str, params: Tuple[float, ...]) -> Tuple[float, float, float]:
    c, loc, scale = params
    if name == "weibull_2p":
        return float(c), 0.0, float(scale)
    return float(c), float(loc), float(scale)


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

    if best_name.startswith("weibull") and best.get("params"):
        beta, gamma, eta = _normalize_weibull_params(best_name, best["params"])
    elif best_name == "expon" and best.get("params"):
        loc, scale = best["params"]
        if scale and scale > 0:
            lambda_h = float(1.0 / scale)

    return {
        "best_name": best_name,
        "best": best,
        "all": all_fits,
        "selected_by": selected_by,
        "weibull": {"beta": beta, "eta": eta, "gamma": gamma} if beta is not None else None,
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

    beta = n / denom
    eta = T / (n ** (1.0 / beta))

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
            "params": (float(beta), float(eta)),
            "loglik": float(ll),
            "aic": _aic_from_loglik(ll, 2),
            "ks_p": float(ks_p),
            "chi2_p": float(chi2_p),
            "cvm_p": float(cvm_p),
            "accepted": _gof_acceptance_generic(ks_p, cvm_p, alpha),
            "beta": float(beta),
            "eta": float(eta),
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
# 6) Indicateurs fiabilistes
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


def compute_reliability_indicators(
    ttf_series: List[float] | np.ndarray,
    repair_series: Optional[List[float] | np.ndarray] = None,
    *,
    model: str,
    distribution: str,
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
        beta = params.get("beta")
        eta = params.get("eta")
        if beta is not None and eta is not None:
            try:
                t_ref = float(np.mean(_to_event_times(ttf.tolist())))
                mean_hazard = float((beta / eta) * ((t_ref / eta) ** (beta - 1.0)))
            except Exception:
                mean_hazard = None
    elif model == "BPP":
        theoretical_mttf = empirical_mttf
        mu = params.get("mu")
        if mu is not None and mu > 0:
            mean_hazard = float(mu)

    mttr = float(np.mean(rep)) if rep.size else None
    mtbf = theoretical_mttf if theoretical_mttf is not None else empirical_mttf
    availability = None
    if mtbf is not None and mttr is not None and (mtbf + mttr) > 0:
        availability = float(mtbf / (mtbf + mttr))

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
    }


# ---------------------------------------------------------------------
# 7) Moteur fiabiliste principal
# ---------------------------------------------------------------------
def analyze_reliability_only(
    ttf_series: List[float],
    repair_series: Optional[List[float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    data = _clean_positive(ttf_series)
    n = int(data.size)

    default_tests = {
        "trend_mk": {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
        "trend_laplace": {"u": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
        "trend_mil_hdbk_189": {"beta_graph": 1.0, "slope_loglog": 1.0, "intercept_loglog": 0.0, "interpreted_trend": "none"},
        "trend_combined": {"has_trend": False, "direction": "none", "confidence": "none", "reason": "TTF insuffisants."},
        "dependence": {
            "r": 0.0,
            "p": 1.0,
            "has_dep": False,
            "method": "spearman",
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "strength": "very_low",
        },
    }

    if n < 3:
        result = {
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
                "reason": "Données insuffisantes pour statuer, hypothèse RP par défaut.",
            },
            "goodness": {"aic": None, "ks_p": None, "chi2_p": None, "cvm_p": None, "accepted": None},
            "params": {
                "raw": None,
                "beta": None,
                "eta": None,
                "gamma": None,
                "mu": None,
                "alpha": None,
                "beta_kernel": None,
                "branch_ratio": None,
                "lambda_hpp_h": None,
            },
            "tests": default_tests,
            "candidates": {},
        }
        result["indicators"] = compute_reliability_indicators(
            ttf_series=data,
            repair_series=repair_series,
            model=result["model"],
            distribution=result["distribution"],
            params=result["params"],
        )
        return result

    mk = mann_kendall_test(data.tolist(), alpha=alpha)
    lap = laplace_trend_test(data.tolist(), alpha=alpha)
    mil = mil_hdbk_189_indicator(data.tolist())
    trend = combine_trend_evidence(mk, lap, mil, alpha=alpha)
    dep = test_dependence(data.tolist(), alpha=alpha)

    beta = eta = gamma = None
    mu = alpha_bpp = beta_kernel = branch_ratio = None
    lambda_hpp_h = None
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
        reason = (
            "Tendance détectée : l'organigramme oriente vers un processus de Poisson non homogène (NHPP)."
        )
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
        reason = (
            "Absence de tendance significative mais présence de dépendance : "
            "l'organigramme oriente vers un BPP."
        )
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

        if fit_bundle["hpp"] is not None:
            lambda_hpp_h = fit_bundle["hpp"].get("lambda_h")

        if process_variant == "HPP":
            reason = (
                "Ni tendance ni dépendance : hypothèse RP retenue. "
                "La loi exponentielle étant retenue, le cas particulier HPP est signalé."
            )
        else:
            reason = (
                "Ni tendance significative ni dépendance : "
                "l'organigramme oriente vers un processus de renouvellement (RP) avec choix de loi."
            )

        entity_assumption = "Entité non réparable ou réparable comme neuve / hybride"

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
    }

    out = {
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
            "trend_mk": mk,
            "trend_laplace": lap,
            "trend_mil_hdbk_189": mil,
            "trend_combined": trend,
            "dependence": dep,
        },
        "candidates": candidates,
    }

    out["indicators"] = compute_reliability_indicators(
        ttf_series=data,
        repair_series=repair_series,
        model=model,
        distribution=distribution,
        params=params,
    )
    return out


# ---------------------------------------------------------------------
# 8) Modélisation thermique dynamique
# ---------------------------------------------------------------------
def simulate_thermal_dynamic(
    df: pd.DataFrame,
    *,
    sn_mva: float = 100.0,
    R: float = 5.0,
    delta_to_r: float = 55.0,
    delta_h_r: float = 30.0,
    tau_to_min: float = 180.0,
    tau_w_min: float = 10.0,
    n_exp: float = 0.8,
    m_exp: float = 0.8,
    forced_tau_to_factor: float = 0.75,
    forced_delta_to_factor: float = 0.92,
    forced_delta_h_factor: float = 0.92,
    normal_insulation_life_h: float = 180000.0,
) -> pd.DataFrame:
    req = {"timestamp", "temp_amb_C"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la thermique: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "K" not in out.columns:
        if "charge_pct" in out.columns:
            out["K"] = pd.to_numeric(out["charge_pct"], errors="coerce") / 100.0
        elif "load_factor" in out.columns:
            out["K"] = pd.to_numeric(out["load_factor"], errors="coerce")
        elif "load_mva" in out.columns:
            out["K"] = pd.to_numeric(out["load_mva"], errors="coerce") / float(sn_mva)
        else:
            raise ValueError("Il faut fournir K, charge_pct, load_factor ou load_mva.")

    if "etat_ventilateurs" not in out.columns:
        out["etat_ventilateurs"] = 0

    out["temp_amb_C"] = pd.to_numeric(out["temp_amb_C"], errors="coerce")
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    out["etat_ventilateurs"] = (
        pd.to_numeric(out["etat_ventilateurs"], errors="coerce").fillna(0).astype(int)
    )

    out = out.dropna(subset=["temp_amb_C", "K"]).reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("La série thermique doit contenir au moins 2 points.")

    dt_h_series = out["timestamp"].diff().dt.total_seconds().div(3600.0)
    dt_h_default = float(dt_h_series.iloc[1:].median())
    if not np.isfinite(dt_h_default) or dt_h_default <= 0:
        dt_h_default = 1.0

    n = len(out)
    dTO = np.zeros(n, dtype=float)
    dH = np.zeros(n, dtype=float)

    tau_to_onan_h = tau_to_min / 60.0
    tau_w_h = tau_w_min / 60.0

    K0 = float(out.loc[0, "K"])
    fans0 = int(out.loc[0, "etat_ventilateurs"])

    dTO_ult0 = delta_to_r * ((((K0 ** 2) * R) + 1.0) / (R + 1.0)) ** n_exp
    dH_ult0 = delta_h_r * (K0 ** (2.0 * m_exp))

    if fans0 == 1:
        dTO_ult0 *= forced_delta_to_factor
        dH_ult0 *= forced_delta_h_factor

    dTO[0] = 0.6 * dTO_ult0
    dH[0] = 0.6 * dH_ult0

    for i in range(1, n):
        K = float(out.loc[i, "K"])
        fans = int(out.loc[i, "etat_ventilateurs"])

        dt_h_i = _safe_float(dt_h_series.iloc[i], dt_h_default)
        if dt_h_i is None or dt_h_i <= 0:
            dt_h_i = dt_h_default

        tau_to_h = tau_to_onan_h * (forced_tau_to_factor if fans == 1 else 1.0)
        tau_to_h = max(tau_to_h, 1e-6)
        tau_w_h_eff = max(tau_w_h, 1e-6)

        dTO_ult = delta_to_r * ((((K ** 2) * R) + 1.0) / (R + 1.0)) ** n_exp
        dH_ult = delta_h_r * (K ** (2.0 * m_exp))

        if fans == 1:
            dTO_ult *= forced_delta_to_factor
            dH_ult *= forced_delta_h_factor

        a_to = math.exp(-dt_h_i / tau_to_h)
        a_w = math.exp(-dt_h_i / tau_w_h_eff)

        dTO[i] = dTO_ult + (dTO[i - 1] - dTO_ult) * a_to
        dH[i] = dH_ult + (dH[i - 1] - dH_ult) * a_w

        dTO[i] = float(np.clip(dTO[i], 0.0, 120.0))
        dH[i] = float(np.clip(dH[i], 0.0, 80.0))

    out["Delta_theta_TO"] = dTO
    out["Delta_theta_H"] = dH
    out["theta_TO_est_C"] = out["temp_amb_C"] + out["Delta_theta_TO"]
    out["theta_HS_est_C"] = out["temp_amb_C"] + out["Delta_theta_TO"] + out["Delta_theta_H"]

    theta_hs = out["theta_HS_est_C"].to_numpy()
    out["FAA"] = np.exp((15000.0 / 383.0) - (15000.0 / (theta_hs + 273.0)))

    step_h = dt_h_series.fillna(dt_h_default).clip(lower=dt_h_default).to_numpy(dtype=float)
    out["dt_h_step"] = step_h
    out["aging_hours_step"] = out["FAA"] * out["dt_h_step"]
    out["aging_hours_cum"] = out["aging_hours_step"].cumsum()
    out["life_consumed_pct_cum"] = 100.0 * out["aging_hours_cum"] / float(normal_insulation_life_h)
    out["remaining_life_pct"] = 100.0 - out["life_consumed_pct_cum"]

    out.attrs["dt_h_default"] = float(dt_h_default)
    out.attrs["normal_insulation_life_h"] = float(normal_insulation_life_h)
    out.attrs["thermal_params"] = {
        "sn_mva": float(sn_mva),
        "R": float(R),
        "delta_to_r": float(delta_to_r),
        "delta_h_r": float(delta_h_r),
        "tau_to_min": float(tau_to_min),
        "tau_w_min": float(tau_w_min),
        "n_exp": float(n_exp),
        "m_exp": float(m_exp),
        "forced_tau_to_factor": float(forced_tau_to_factor),
        "forced_delta_to_factor": float(forced_delta_to_factor),
        "forced_delta_h_factor": float(forced_delta_h_factor),
        "normal_insulation_life_h": float(normal_insulation_life_h),
    }
    return out


def summarize_thermal_results(df_sim: pd.DataFrame) -> Dict[str, Any]:
    dt_h = float(df_sim.attrs.get("dt_h_default", 1.0))
    normal_life = float(df_sim.attrs.get("normal_insulation_life_h", 180000.0))
    thermal_params = dict(df_sim.attrs.get("thermal_params", {}))

    total_aging_hours = float(df_sim["aging_hours_step"].sum())
    life_consumed_pct = 100.0 * total_aging_hours / normal_life

    daily = df_sim.copy()
    daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
    daily = (
        daily.groupby("date")
        .agg(
            charge_mean_pct=("K", lambda s: float(np.mean(s) * 100.0)),
            charge_max_pct=("K", lambda s: float(np.max(s) * 100.0)),
            amb_mean=("temp_amb_C", "mean"),
            theta_TO_mean=("theta_TO_est_C", "mean"),
            theta_HS_max=("theta_HS_est_C", "max"),
            theta_HS_p95=("theta_HS_est_C", lambda s: float(pd.Series(s).quantile(0.95))),
            FAA_max=("FAA", "max"),
            FAA_mean=("FAA", "mean"),
            aging_hours=("aging_hours_step", "sum"),
            fans_share=("etat_ventilateurs", "mean"),
        )
        .reset_index()
    )

    daily["fans_share_pct"] = 100.0 * daily["fans_share"]
    daily["life_consumed_pct"] = 100.0 * daily["aging_hours"] / normal_life
    daily["aging_hours_cum"] = daily["aging_hours"].cumsum()
    daily["life_consumed_pct_cum"] = 100.0 * daily["aging_hours_cum"] / normal_life
    daily["remaining_life_pct"] = 100.0 - daily["life_consumed_pct_cum"]

    top5 = daily.sort_values("aging_hours", ascending=False).head(5).copy()

    summary = {
        "theta_hs_max": float(df_sim["theta_HS_est_C"].max()),
        "theta_hs_p95": float(df_sim["theta_HS_est_C"].quantile(0.95)),
        "theta_hs_mean": float(df_sim["theta_HS_est_C"].mean()),
        "faa_max": float(df_sim["FAA"].max()),
        "faa_mean": float(df_sim["FAA"].mean()),
        "loss_of_life_hours": total_aging_hours,
        "loss_of_life_pct": float(life_consumed_pct),
        "dt_h_default": dt_h,
    }

    return {
        "summary": summary,
        "timeseries": df_sim,
        "daily": daily,
        "top5_critical_days": top5,
        "params": thermal_params,
    }


# ---------------------------------------------------------------------
# 9) Tables fiabilité / thermique / globales
# ---------------------------------------------------------------------
def build_reliability_tables(reliability_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tests = reliability_result.get("tests", {})
    dep = tests.get("dependence", {})
    trend_combined = tests.get("trend_combined", {})
    decision = reliability_result.get("decision", {})
    params = reliability_result.get("params", {})
    goodness = reliability_result.get("goodness", {})
    indicators = reliability_result.get("indicators", {})

    trend_df = pd.DataFrame(
        [
            {
                "Test": "Mann-Kendall",
                "Statistique": tests.get("trend_mk", {}).get("z"),
                "p_value": tests.get("trend_mk", {}).get("p"),
                "Décision": "Oui" if tests.get("trend_mk", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_mk", {}).get("direction"),
            },
            {
                "Test": "Laplace",
                "Statistique": tests.get("trend_laplace", {}).get("u"),
                "p_value": tests.get("trend_laplace", {}).get("p"),
                "Décision": "Oui" if tests.get("trend_laplace", {}).get("has_trend") else "Non",
                "Direction": tests.get("trend_laplace", {}).get("direction"),
            },
            {
                "Test": "MIL-HDBK-189",
                "Statistique": tests.get("trend_mil_hdbk_189", {}).get("beta_graph"),
                "p_value": None,
                "Décision": trend_combined.get("mil_direction"),
                "Direction": tests.get("trend_mil_hdbk_189", {}).get("interpreted_trend"),
            },
            {
                "Test": "Décision finale tendance",
                "Statistique": None,
                "p_value": None,
                "Décision": "Oui" if trend_combined.get("has_trend") else "Non",
                "Direction": trend_combined.get("direction"),
            },
        ]
    )

    dependence_df = pd.DataFrame(
        [
            {
                "Méthode": "Pearson",
                "r": dep.get("pearson_r"),
                "p_value": dep.get("pearson_p"),
                "Dépendance": "Oui" if (dep.get("pearson_p") is not None and dep.get("pearson_p") < 0.05) else "Non",
            },
            {
                "Méthode": "Spearman",
                "r": dep.get("spearman_r"),
                "p_value": dep.get("spearman_p"),
                "Dépendance": "Oui" if (dep.get("spearman_p") is not None and dep.get("spearman_p") < 0.05) else "Non",
            },
            {
                "Méthode": "Décision finale",
                "r": dep.get("r"),
                "p_value": dep.get("p"),
                "Dépendance": "Oui" if dep.get("has_dep") else "Non",
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
                "LogLik": fit.get("loglik"),
                "AIC": fit.get("aic"),
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
                "Beta_kernel": params.get("beta_kernel"),
                "Branch_ratio": params.get("branch_ratio"),
                "AIC": goodness.get("aic"),
                "KS p": goodness.get("ks_p"),
                "Chi2 p": goodness.get("chi2_p"),
                "CvM p": goodness.get("cvm_p"),
                "Ajustement accepté": goodness.get("accepted"),
                "MTTF (h)": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
                "MTBF (h)": indicators.get("mtbf_h"),
                "MTTR (h)": indicators.get("mttr_h"),
                "Disponibilité": indicators.get("availability_intrinsic"),
                "Taux de défaillance moyen (1/h)": indicators.get("mean_failure_rate_h"),
            }
        ]
    )

    return {
        "trend_results": _round_df(trend_df),
        "dependence_results": _round_df(dependence_df),
        "process_choice": _round_df(process_df),
        "fit_candidates": _round_df(fits_df),
        "reliability_summary": _round_df(reliability_summary_df),
    }


def build_thermal_tables(thermal_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    ts = thermal_result["timeseries"].copy()
    daily = thermal_result["daily"].copy()
    top5 = thermal_result["top5_critical_days"].copy()
    summary = thermal_result["summary"]
    params = thermal_result.get("params", {})

    table_dataset = pd.DataFrame(
        [
            {
                "Période début": str(ts["timestamp"].min()),
                "Période fin": str(ts["timestamp"].max()),
                "Nombre de points": int(len(ts)),
                "Pas de temps par défaut (h)": summary.get("dt_h_default"),
                "Charge min (%)": float(ts["K"].min() * 100.0),
                "Charge max (%)": float(ts["K"].max() * 100.0),
                "Temp ambiante min (°C)": float(ts["temp_amb_C"].min()),
                "Temp ambiante max (°C)": float(ts["temp_amb_C"].max()),
                "Ventilation forcée (% du temps)": float((ts["etat_ventilateurs"] == 1).mean() * 100.0),
            }
        ]
    )

    table_params = pd.DataFrame([{"Paramètre": k, "Valeur": v} for k, v in params.items()])

    table_indicators = pd.DataFrame(
        [
            {
                "θHS max (°C)": summary.get("theta_hs_max"),
                "θHS P95 (°C)": summary.get("theta_hs_p95"),
                "θHS mean (°C)": summary.get("theta_hs_mean"),
                "FAA max": summary.get("faa_max"),
                "FAA mean": summary.get("faa_mean"),
                "Perte de vie (h)": summary.get("loss_of_life_hours"),
                "Perte de vie (%)": summary.get("loss_of_life_pct"),
            }
        ]
    )

    thermal_summary = pd.DataFrame(
        [
            {
                "θHS max": summary.get("theta_hs_max"),
                "θHS P95": summary.get("theta_hs_p95"),
                "θHS mean": summary.get("theta_hs_mean"),
                "FAA max": summary.get("faa_max"),
                "FAA mean": summary.get("faa_mean"),
                "Loss of life (h)": summary.get("loss_of_life_hours"),
                "Loss of life (%)": summary.get("loss_of_life_pct"),
            }
        ]
    )

    return {
        "thermal_table_dataset": _round_df(table_dataset),
        "thermal_table_params": _round_df(table_params),
        "thermal_table_indicators": _round_df(table_indicators),
        "thermal_summary": _round_df(thermal_summary),
        "thermal_daily": _round_df(daily),
        "thermal_top5_days": _round_df(top5),
    }


def build_global_result_tables(
    reliability_result: Dict[str, Any],
    thermal_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    tables = {}
    tables.update(build_reliability_tables(reliability_result))
    if thermal_result is not None:
        tables.update(build_thermal_tables(thermal_result))
    return tables


# ---------------------------------------------------------------------
# 10) Fonction principale intégrée pour le logiciel
# ---------------------------------------------------------------------
def analyze_ttf_pipeline(
    ttf_series: List[float],
    alpha: float = 0.05,
    repair_series: Optional[List[float]] = None,
    thermal_df: Optional[pd.DataFrame] = None,
    thermal_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entrées :
    - ttf_series     : série des temps entre défaillances (heures)
    - alpha          : seuil de significativité
    - repair_series  : série des temps de réparation (heures), optionnelle
    - thermal_df     : série temporelle thermique, optionnelle
    - thermal_config : paramètres du modèle thermique dynamique

    Sorties :
    - reliability : résultats fiabilistes complets
    - thermal     : résultats thermiques complets si fournis
    - tables      : tableaux prêts pour Streamlit / PDF
    """
    reliability = analyze_reliability_only(
        ttf_series=ttf_series,
        repair_series=repair_series,
        alpha=alpha,
    )

    thermal = None
    if thermal_df is not None and not thermal_df.empty:
        cfg = dict(thermal_config or {})
        df_sim = simulate_thermal_dynamic(thermal_df, **cfg)
        thermal = summarize_thermal_results(df_sim)

    tables = build_global_result_tables(
        reliability_result=reliability,
        thermal_result=thermal,
    )

    return {
        "reliability": reliability,
        "thermal": thermal,
        "tables": tables,
    }


analyze_integrated_pipeline = analyze_ttf_pipeline