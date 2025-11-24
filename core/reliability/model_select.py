# core/reliability/model_select.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

from core.reliability.diag_tests import (
    _as_clean_array, laplace_trend_test, cox_stuart_trend_test, runs_test_median,
    fit_exponential, fit_lognormal, fit_gamma_mm, ks_distance,
    cdf_exponential, cdf_lognormal, cdf_gamma_mm, cdf_weibull
)
from core.reliability.weibull import fit_weibull

def select_best_model(ttf: List[float]) -> Dict:
    """
    Compare Weibull / Exponentielle / Lognormale / Gamma.
    Critères: AIC (prioritaire), KS (secondaire).
    Retourne un dict avec:
      - best_name, params
      - aic_rank, ks_rank
      - tests (trend / independence)
      - all_models (détails)
      - maintenance_hint (policy)
    """
    x = _as_clean_array(ttf)
    out = {
        "ok": False, "reason": "",
        "best_name": None, "params": {},
        "aic_rank": [], "ks_rank": [],
        "tests": {},
        "all_models": {}
    }
    if len(x) < 3:
        out["reason"] = "Not enough TTF (need ≥ 3)."
        return out

    # 1) Tests de tendance & indépendance
    lap = laplace_trend_test(x)
    cox = cox_stuart_trend_test(x)
    run = runs_test_median(x)
    out["tests"] = {
        "laplace": lap.__dict__,
        "cox_stuart": cox.__dict__,
        "runs": run.__dict__
    }

    # 2) Ajustements
    models = []

    # Weibull
    try:
        wb = fit_weibull(x)
        aic_w = 2*2 - 2*float(wb.loglik)  # k=2, -2*LL + 2k <=> 2k - 2LL
        ks_w  = ks_distance(x, lambda v: cdf_weibull(v, float(wb.beta), float(wb.eta)))
        models.append(("weibull", aic_w, ks_w, {"beta":float(wb.beta), "eta":float(wb.eta)}))
    except Exception:
        pass

    # Expo
    ex = fit_exponential(x)
    if ex.get("ok"):
        ks_e = ks_distance(x, lambda v: cdf_exponential(v, ex["lambda"]))
        models.append(("exponential", ex["aic"], ks_e, {"lambda":ex["lambda"]}))

    # Lognormal
    ln = fit_lognormal(x)
    if ln.get("ok"):
        ks_l = ks_distance(x, lambda v: cdf_lognormal(v, ln["mu"], ln["sigma"]))
        models.append(("lognormal", ln["aic"], ks_l, {"mu":ln["mu"], "sigma":ln["sigma"]}))

    # Gamma (méthode des moments)
    ga = fit_gamma_mm(x)
    if ga.get("ok"):
        ks_g = ks_distance(x, lambda v: cdf_gamma_mm(v, ga["k"], ga["theta"]))
        models.append(("gamma", ga["aic"], ks_g, {"k":ga["k"], "theta":ga["theta"]}))

    if not models:
        out["reason"] = "No model fitted."
        return out

    # 3) Classements
    aic_sorted = sorted(models, key=lambda m: m[1])
    ks_sorted  = sorted(models, key=lambda m: m[2])

    out["aic_rank"] = [{"name":n, "aic":float(a), "params":p} for n,a,_,p in aic_sorted]
    out["ks_rank"]  = [{"name":n, "ks":float(k), "params":p} for n,_,k,p in ks_sorted]
    out["all_models"] = {
        n: {"aic": float(a), "ks": float(k), "params": p} for n,a,k,p in models
    }

    # 4) Choix final (AIC d’abord, KS en tie-break)
    best = aic_sorted[0]
    # tie-break si AIC proches (< 2)
    close = [m for m in aic_sorted if abs(m[1] - best[1]) < 2.0]
    if len(close) > 1:
        # départage au KS
        close = sorted(close, key=lambda m: m[2])
        best = close[0]

    out["best_name"] = best[0]
    out["params"] = best[3]
    out["ok"] = True

    # 5) Suggestion de politique maintenance (hint)
    hint = "preventive"
    if out["best_name"] == "weibull":
        b = out["params"].get("beta", 1.0)
        if b < 1.0:
            hint = "predictive (surveillance/condition-based)"
        elif 0.95 <= b <= 1.05:
            hint = "corrective + périodique légère"
        else:
            hint = "préventive basée âge + inspections"
    elif out["best_name"] == "exponential":
        hint = "corrective (taux constant) + opportuniste"
    elif out["best_name"] == "lognormal":
        hint = "préventive conditionnelle (dégradation multiplicative)"
    elif out["best_name"] == "gamma":
        hint = "préventive sur charge/usage (fatigue cumulée)"
    out["maintenance_hint"] = hint
    return out
