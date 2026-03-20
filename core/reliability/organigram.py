from __future__ import annotations

"""
core/reliability/organigram.py

Pipeline fiabiliste + thermo-fiabiliste inspiré de l'organigramme :
- Test de tendance (graphique/diagnostic + Mann-Kendall + Laplace + indicateur MIL-HDBK-189)
- Test de dépendance lag-1
- Choix du processus : RP / NHPP / BPP
- Ajustement selon la branche :
    * RP   -> lois iid candidates (expon, norm, lognorm, weibull_2p, weibull_3p)
    * NHPP -> Power Law Process / Crow-AMSAA
    * BPP  -> approximation branchement de Poisson par Hawkes exponentiel
- Validation statistique : AIC + KS + Chi² + Cramér-von Mises
- Extension thermique : point chaud, FAA, perte de vie cumulée

NOTE:
- La branche NHPP est implémentée via Crow-AMSAA (power law NHPP), cohérente
  avec la logique "loi de puissance" de l'organigramme.
- La branche BPP est approchée par un processus de type Hawkes exponentiel
  (auto-excitant), qui est un proxy pratique d'un processus de branchement.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
from scipy import stats as sst
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _clean_positive(series: List[float]) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


def _to_event_times(ttf_series: List[float]) -> np.ndarray:
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


def _merge_small_bins(
    observed: np.ndarray,
    expected: np.ndarray,
    min_exp: float = 5.0
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
    nbins: int = 8
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
        return {"beta_graph": 1.0, "slope_loglog": 1.0, "interpreted_trend": "none"}

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
        }

    pr, pp = sst.pearsonr(a, b)
    sr, sp = sst.spearmanr(a, b)

    method = "spearman"
    r = float(sr)
    p = float(sp)
    has_dep = bool(sp < alpha or pp < alpha)

    return {
        "r": r,
        "p": p,
        "has_dep": has_dep,
        "method": method,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# ---------------------------------------------------------------------
# 3) Branche RP : lois iid
# ---------------------------------------------------------------------
def _fit_distribution(name: str, data: np.ndarray) -> Dict[str, Any]:
    res = {
        "name": name,
        "params": None,
        "loglik": None,
        "aic": np.inf,
        "ks_p": None,
        "chi2_p": None,
        "cvm_p": None,
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
        return res
    except Exception:
        return res


def _normalize_weibull_params(name: str, params: Tuple[float, ...]) -> Tuple[float, float, float]:
    c, loc, scale = params
    if name == "weibull_2p":
        return float(c), 0.0, float(scale)
    return float(c), float(loc), float(scale)


def fit_and_compare_distributions(data: np.ndarray) -> Dict[str, Any]:
    candidates = ["expon", "norm", "lognorm", "weibull_2p", "weibull_3p"]
    all_fits = {name: _fit_distribution(name, data) for name in candidates}
    best_name = min(all_fits.keys(), key=lambda n: all_fits[n]["aic"])
    best = all_fits[best_name]

    beta = eta = gamma = None
    if best_name.startswith("weibull") and best.get("params"):
        beta, gamma, eta = _normalize_weibull_params(best_name, best["params"])

    return {
        "best_name": best_name,
        "best": best,
        "all": all_fits,
        "weibull": {"beta": beta, "eta": eta, "gamma": gamma} if beta is not None else None,
    }


# ---------------------------------------------------------------------
# 4) Branche NHPP : Crow-AMSAA / Power Law Process
# ---------------------------------------------------------------------
def fit_power_law_nhpp(ttf_series: List[float]) -> Dict[str, Any]:
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
        "beta": None,
        "eta": None,
        "T_end": None,
    }
    if n < 2:
        return res

    T = float(t[-1])
    denom = float(np.sum(np.log(T / t)))
    if denom <= 0:
        return res

    beta = n / denom
    eta = T / (n ** (1.0 / beta))

    # log-likelihood of PLP on [0,T]
    ll = (
        n * math.log(beta)
        - n * beta * math.log(eta)
        + (beta - 1.0) * float(np.sum(np.log(t)))
        - (T / eta) ** beta
    )

    # Time-rescaling theorem: transformed inter-event compensators ~ Exp(1)
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
    branch_ratio = 0.98 / (1.0 + math.exp(-q))  # in (0, 0.98)
    alpha = branch_ratio * beta

    T = float(t[-1])
    if mu <= 0 or beta <= 0 or alpha < 0:
        return 1e100

    # Recursive evaluation of intensities
    r_prev = 0.0  # sum exp(-beta*(t_i-t_j)) for j < i
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


def fit_hawkes_bpp(ttf_series: List[float]) -> Dict[str, Any]:
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
        "mu": None,
        "alpha": None,
        "beta_kernel": None,
        "branch_ratio": None,
        "T_end": None,
    }
    if n < 4:
        return res

    mean_gap = float(np.mean(np.diff(np.insert(t, 0, 0.0))))
    x0 = np.array([math.log(max(1e-6, 1.0 / max(mean_gap, 1e-6))), math.log(1.0), 0.0])

    opt = minimize(
        _hawkes_neg_loglik,
        x0=x0,
        args=(t,),
        method="L-BFGS-B",
    )

    if not opt.success:
        return res

    log_mu, log_beta, q = opt.x
    mu = math.exp(log_mu)
    beta_k = math.exp(log_beta)
    branch_ratio = 0.98 / (1.0 + math.exp(-q))
    alpha = branch_ratio * beta_k
    ll = -float(opt.fun)
    T = float(t[-1])

    # Time-rescaling theorem: integrated intensity increments ~ Exp(1)
    w = []
    A_prev = 0.0  # state immediately after previous event
    t_prev = 0.0
    for ti in t:
        dt = ti - t_prev
        compensator = mu * dt + (alpha / beta_k) * A_prev * (1.0 - math.exp(-beta_k * dt))
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
            "params": (float(mu), float(alpha), float(beta_k), float(branch_ratio)),
            "loglik": ll,
            "aic": _aic_from_loglik(ll, 3),
            "ks_p": float(ks_p),
            "chi2_p": float(chi2_p),
            "cvm_p": float(cvm_p),
            "mu": float(mu),
            "alpha": float(alpha),
            "beta_kernel": float(beta_k),
            "branch_ratio": float(branch_ratio),
            "T_end": float(T),
        }
    )
    return res


# ---------------------------------------------------------------------
# 6) Bloc thermique : point chaud, FAA, perte de vie
# ---------------------------------------------------------------------
def estimate_hotspot_profile(
    theta_amb: List[float] | np.ndarray,
    load_factor: Optional[List[float] | np.ndarray] = None,
    *,
    dt_hours: float = 1.0,
    delta_theta_to_r: float = 55.0,
    delta_theta_h_r: float = 30.0,
    R: float = 5.0,
    n_exp: float = 0.8,
    m_exp: float = 1.6,
    tau_to_hours: float = 4.0,
    tau_h_hours: float = 1.0,
    theta_to_init: Optional[float] = None,
    theta_h_init: Optional[float] = None,
    direct_top_oil_rise: Optional[List[float] | np.ndarray] = None,
    direct_hotspot_gradient: Optional[List[float] | np.ndarray] = None,
    normal_life_hours: float = 180000.0,
) -> Dict[str, Any]:
    """
    Estimation thermique cohérente avec la structure du mémoire :
        θHS = θamb + ΔθTO + ΔθH
        FAA = exp(15000/383 - 15000/(θHS + 273))
        L    = Σ FAA_i * Δt_i

    Deux modes:
    1) Direct:
       - direct_top_oil_rise et direct_hotspot_gradient fournis
    2) Estimé à partir du facteur de charge K:
       - load_factor fourni + paramètres thermiques usuels
    """
    amb = np.asarray(theta_amb, dtype=float)
    if amb.ndim != 1 or amb.size == 0:
        raise ValueError("theta_amb doit être une série 1D non vide.")

    N = amb.size

    if direct_top_oil_rise is not None and direct_hotspot_gradient is not None:
        d_to = np.asarray(direct_top_oil_rise, dtype=float)
        d_h = np.asarray(direct_hotspot_gradient, dtype=float)
        if d_to.size != N or d_h.size != N:
            raise ValueError("Les séries directes doivent avoir la même taille que theta_amb.")
    else:
        if load_factor is None:
            raise ValueError(
                "Fournir soit load_factor, soit direct_top_oil_rise + direct_hotspot_gradient."
            )
        K = np.asarray(load_factor, dtype=float)
        if K.size != N:
            raise ValueError("load_factor doit avoir la même taille que theta_amb.")

        d_to = np.zeros(N, dtype=float)
        d_h = np.zeros(N, dtype=float)

        d_to_ult = delta_theta_to_r * (((K ** 2) * R + 1.0) / (R + 1.0)) ** n_exp
        d_h_ult = delta_theta_h_r * (K ** (2.0 * m_exp))

        d_to[0] = d_to_ult[0] if theta_to_init is None else float(theta_to_init)
        d_h[0] = d_h_ult[0] if theta_h_init is None else float(theta_h_init)

        a_to = 1.0 - math.exp(-dt_hours / max(tau_to_hours, 1e-9))
        a_h = 1.0 - math.exp(-dt_hours / max(tau_h_hours, 1e-9))

        for i in range(1, N):
            d_to[i] = d_to[i - 1] + a_to * (d_to_ult[i] - d_to[i - 1])
            d_h[i] = d_h[i - 1] + a_h * (d_h_ult[i] - d_h[i - 1])

    theta_hs = amb + d_to + d_h
    faa = np.exp((15000.0 / 383.0) - (15000.0 / (theta_hs + 273.0)))
    loss_hours = float(np.sum(faa * dt_hours))
    loss_pu = loss_hours / float(normal_life_hours) if normal_life_hours > 0 else None
    loss_pct = 100.0 * loss_pu if loss_pu is not None else None

    return {
        "theta_hs": theta_hs.tolist(),
        "delta_theta_to": d_to.tolist(),
        "delta_theta_h": d_h.tolist(),
        "faa": faa.tolist(),
        "faa_mean": float(np.mean(faa)),
        "faa_max": float(np.max(faa)),
        "theta_hs_mean": float(np.mean(theta_hs)),
        "theta_hs_max": float(np.max(theta_hs)),
        "theta_hs_min": float(np.min(theta_hs)),
        "loss_of_life_hours": loss_hours,
        "loss_of_life_pu": None if loss_pu is None else float(loss_pu),
        "loss_of_life_pct": None if loss_pct is None else float(loss_pct),
        "inputs": {
            "dt_hours": float(dt_hours),
            "delta_theta_to_r": float(delta_theta_to_r),
            "delta_theta_h_r": float(delta_theta_h_r),
            "R": float(R),
            "n_exp": float(n_exp),
            "m_exp": float(m_exp),
            "tau_to_hours": float(tau_to_hours),
            "tau_h_hours": float(tau_h_hours),
            "normal_life_hours": float(normal_life_hours),
        },
    }


# ---------------------------------------------------------------------
# 7) Fonction principale
# ---------------------------------------------------------------------
def analyze_ttf_pipeline(
    ttf_series: List[float],
    alpha: float = 0.05,
    thermal_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = _clean_positive(ttf_series)
    n = int(data.size)

    if n < 3:
        out = {
            "error": "TTF insuffisants (<3 après nettoyage).",
            "cleaned_n": n,
            "model": "RP",
            "distribution": "weibull_2p",
            "goodness": {"aic": None, "ks_p": None, "chi2_p": None, "cvm_p": None},
            "params": {
                "raw": None,
                "beta": None,
                "eta": None,
                "gamma": None,
                "mu": None,
                "alpha": None,
                "beta_kernel": None,
                "branch_ratio": None,
            },
            "tests": {
                "trend_mk": {"z": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
                "trend_laplace": {"u": 0.0, "p": 1.0, "has_trend": False, "direction": "none"},
                "trend_mil_hdbk_189": {"beta_graph": 1.0, "slope_loglog": 1.0, "interpreted_trend": "none"},
                "dependence": {
                    "r": 0.0,
                    "p": 1.0,
                    "has_dep": False,
                    "method": "spearman",
                    "pearson_r": 0.0,
                    "pearson_p": 1.0,
                    "spearman_r": 0.0,
                    "spearman_p": 1.0,
                },
            },
            "candidates": {},
            "thermal": None,
        }
        if thermal_config:
            out["thermal"] = estimate_hotspot_profile(**thermal_config)
        return out

    mk = mann_kendall_test(data.tolist(), alpha=alpha)
    lap = laplace_trend_test(data.tolist(), alpha=alpha)
    mil = mil_hdbk_189_indicator(data.tolist())
    dep = test_dependence(data.tolist(), alpha=alpha)

    has_trend = bool(mk["has_trend"] or lap["has_trend"])
    if has_trend:
        model = "NHPP"
    elif dep["has_dep"]:
        model = "BPP"
    else:
        model = "RP"

    beta = eta = gamma = None
    mu = alpha_bpp = beta_kernel = branch_ratio = None

    if model == "NHPP":
        best = fit_power_law_nhpp(data.tolist())
        distribution = "power_law_nhpp"
        candidates = {"power_law_nhpp": best}
        beta = best.get("beta")
        eta = best.get("eta")
    elif model == "BPP":
        best = fit_hawkes_bpp(data.tolist())
        distribution = "hawkes_exp_bpp"
        candidates = {"hawkes_exp_bpp": best}
        mu = best.get("mu")
        alpha_bpp = best.get("alpha")
        beta_kernel = best.get("beta_kernel")
        branch_ratio = best.get("branch_ratio")
    else:
        fits = fit_and_compare_distributions(data)
        best = fits["best"]
        distribution = fits["best_name"]
        candidates = fits["all"]
        if fits["weibull"] is not None:
            beta = fits["weibull"]["beta"]
            eta = fits["weibull"]["eta"]
            gamma = fits["weibull"]["gamma"]

    out = {
        "cleaned_n": n,
        "model": model,
        "distribution": distribution,
        "goodness": {
            "aic": best.get("aic"),
            "ks_p": best.get("ks_p"),
            "chi2_p": best.get("chi2_p"),
            "cvm_p": best.get("cvm_p"),
        },
        "params": {
            "raw": best.get("params"),
            "beta": beta,
            "eta": eta,
            "gamma": gamma,
            "mu": mu,
            "alpha": alpha_bpp,
            "beta_kernel": beta_kernel,
            "branch_ratio": branch_ratio,
        },
        "tests": {
            "trend_mk": mk,
            "trend_laplace": lap,
            "trend_mil_hdbk_189": mil,
            "dependence": dep,
        },
        "candidates": candidates,
        "thermal": None,
    }

    if thermal_config:
        out["thermal"] = estimate_hotspot_profile(**thermal_config)

    return out