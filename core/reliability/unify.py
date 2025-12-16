# core/reliability/unify.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.reliability.weibull import fit_weibull
from core.reliability.metrics_plus import (
    compute_mtbf_mttr_by_eq,
    compute_weibull_params_by_eq,
    merge_metrics_and_optim,
)
from core.reliability.optimize import propose_intervals

try:
    from core.reliability.organigram import analyze_ttf_pipeline
except Exception:
    analyze_ttf_pipeline = None

DATA_CSV = Path("data/failures_saved.csv")


@dataclass
class UnifyOptions:
    force_weibull_2p: bool = True   # γ=0 partout pour cohérence
    R_target: float = 0.80
    min_points: int = 3


@dataclass
class UnifyBundle:
    ttf: pd.DataFrame
    fits_df: pd.DataFrame
    optim: Dict[str, Dict[str, Any]]
    metrics_df: pd.DataFrame
    pipeline_by_eq: Dict[str, Dict[str, Any]]  # pipeline COMPLET normalisé


# ------------------ utilitaires sûrs ------------------

def _ensure_df(obj: Any, cols: List[str]) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])
    else:
        df = pd.DataFrame()

    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="float64") if c in (
                "beta", "eta", "gamma", "MTBF", "MTTR", "MTBF_opt", "MTTR_opt", "interval_opt_h"
            ) else ""
    return df


def _load_ttf_df(session_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(session_df, pd.DataFrame):
        df = session_df.copy()
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        return pd.DataFrame(columns=["equipment_code", "ttf_h"])

    # equipment_code
    if "equipment_code" not in df.columns:
        for c in ("equipment", "equip", "eq", "EQ", "Equipment"):
            if c in df.columns:
                df["equipment_code"] = df[c].astype(str)
                break
    if "equipment_code" not in df.columns:
        return pd.DataFrame(columns=["equipment_code", "ttf_h"])

    # ttf_h
    if "ttf_h" in df.columns:
        df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
    else:
        for c in ("hours", "time_to_fail_h", "ttf"):
            if c in df.columns:
                df["ttf_h"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "ttf_h" not in df.columns:
        df["ttf_h"] = np.nan

    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0]
    df["equipment_code"] = df["equipment_code"].astype(str)
    return df


def _fit_weibull_2p(x: np.ndarray) -> Dict[str, float]:
    """
    On n’a PAS besoin de modifier weibull.py :
    - ta MLE fit_weibull() renvoie (beta, eta, gamma=0)
    - on force gamma=0 ici pour cohérence bundle
    """
    ft = fit_weibull(x)
    beta = float(getattr(ft, "beta"))
    eta = float(getattr(ft, "eta"))
    return {"beta": beta, "eta": eta, "gamma": 0.0}


def _compute_fits(df_ttf: pd.DataFrame, opt: UnifyOptions) -> pd.DataFrame:
    if df_ttf.empty:
        return pd.DataFrame(columns=["equipment_code", "beta", "eta", "gamma"])

    recs: List[Dict[str, Any]] = []
    for eq, grp in df_ttf.groupby("equipment_code"):
        x = grp["ttf_h"].dropna().values
        if len(x) < opt.min_points:
            continue
        try:
            if opt.force_weibull_2p:
                rec = _fit_weibull_2p(x)
            else:
                ft = fit_weibull(x)
                rec = {
                    "beta": float(getattr(ft, "beta")),
                    "eta": float(getattr(ft, "eta")),
                    "gamma": float(getattr(ft, "gamma", 0.0)),
                }
            rec["equipment_code"] = str(eq)
            recs.append(rec)
        except Exception:
            continue

    return pd.DataFrame(recs, columns=["equipment_code", "beta", "eta", "gamma"])


# -------- Normalisation pipeline (évite sections mortes) --------
def _normalize_pipeline(res: Any) -> Dict[str, Any]:
    """
    On accepte:
    - Nouveau schéma (déjà riche) -> on le conserve
    - Ancien schéma (distribution_full, trend_mk, fit...) -> on convertit
    - Schéma minimal précédent (distribution/name, goodness/ks_p, trend/p_value) -> on convertit
    """
    if not isinstance(res, dict):
        return {}

    # Déjà au schéma "riche" (recommandé)
    if "tests" in res and "goodness" in res and "params" in res:
        # garantir distribution string
        dist = res.get("distribution")
        if isinstance(dist, dict):
            res["distribution"] = dist.get("name", "weibull_2p")
        elif dist is None:
            res["distribution"] = "weibull_2p"
        return res

    out: Dict[str, Any] = {}

    # model
    out["model"] = res.get("model", "RP")

    # distribution
    d = res.get("distribution")
    if isinstance(d, dict):
        dist_name = d.get("name", "weibull_2p")
    elif isinstance(d, str):
        dist_name = d
    else:
        dist_name = res.get("distribution_full", {}).get("name", "weibull_2p")
    out["distribution"] = dist_name

    # goodness
    good = res.get("goodness") or res.get("gof") or {}
    if not isinstance(good, dict):
        good = {}
    # anciens formats: distribution_full.ks_p / fit.ks_p
    dfull = res.get("distribution_full", {}) if isinstance(res.get("distribution_full"), dict) else {}
    fit = res.get("fit", {}) if isinstance(res.get("fit"), dict) else {}
    out["goodness"] = {
        "aic": dfull.get("aic") or good.get("aic"),
        "ks_p": dfull.get("ks_p") or fit.get("ks_p") or good.get("ks_p"),
        "chi2_p": dfull.get("chi2_p") or fit.get("chi2_p") or good.get("chi2_p"),
    }

    # tests
    mk = res.get("trend_mk") or res.get("trend") or {}
    if not isinstance(mk, dict):
        mk = {}
    dep = res.get("dependence") or res.get("correlation") or {}
    if not isinstance(dep, dict):
        dep = {}

    mk_p = mk.get("p", mk.get("p_value"))
    mk_z = mk.get("z", 0.0)
    mk_has = mk.get("has_trend", mk.get("hasTrend", False))
    direction = "up" if (mk_has and mk_z and mk_z > 0) else "down" if (mk_has and mk_z and mk_z < 0) else "none"

    out["tests"] = {
        "trend_mk": {
            "z": float(mk_z) if mk_z is not None else 0.0,
            "p": float(mk_p) if mk_p is not None else 1.0,
            "has_trend": bool(mk_has),
            "direction": direction,
        },
        "dependence": {
            "r": float(dep.get("r", 0.0) or 0.0),
            "p": float(dep.get("p", 1.0) or 1.0),
            "has_dep": bool(dep.get("has_dep", dep.get("has_dep", False))),
            "method": dep.get("method", "spearman"),
        },
    }

    # params
    beta = res.get("beta")
    eta = res.get("eta")
    gamma = res.get("gamma", 0.0)
    out["params"] = {"raw": fit.get("params"), "beta": beta, "eta": eta, "gamma": gamma}

    # candidates si dispo
    out["candidates"] = res.get("all_fits") or res.get("candidates") or {}
    return out


def _compute_pipeline_per_eq(df_ttf: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if df_ttf.empty or analyze_ttf_pipeline is None:
        return out

    for eq, grp in df_ttf.groupby("equipment_code"):
        x = grp["ttf_h"].dropna().astype(float).values
        if len(x) < 3:
            continue
        try:
            # signature stable : list[float]
            res = analyze_ttf_pipeline(x.tolist())
            out[str(eq)] = _normalize_pipeline(res)
        except Exception:
            out[str(eq)] = {}
    return out


# ------------------ API principale ------------------

def compute_bundle(session_df: Optional[pd.DataFrame] = None, options: Optional[UnifyOptions] = None) -> UnifyBundle:
    opt = options or UnifyOptions()
    df_ttf = _load_ttf_df(session_df)

    # 1) Fits Weibull cohérents (source unique)
    fits_df = _compute_fits(df_ttf, opt)

    # 2) Métriques mesurées + paramètres Weibull "metrics_plus"
    m_meas = compute_mtbf_mttr_by_eq(df_ttf) if not df_ttf.empty else []
    w_params = compute_weibull_params_by_eq(df_ttf) if not df_ttf.empty else []
    w_params = [r for r in (w_params or []) if isinstance(r, dict)]

    if opt.force_weibull_2p:
        for r in w_params:
            r["gamma"] = 0.0

    # 3) Intervalles d’optimisation (cohérents)
    fits_dict = {
        row["equipment_code"]: type("F", (), {
            "beta": float(row["beta"]),
            "eta": float(row["eta"]),
            "gamma": float(row.get("gamma", 0.0) or 0.0),
        })
        for _, row in fits_df.iterrows()
    }
    try:
        props = propose_intervals(fits_dict, R_target=opt.R_target) or {}
    except Exception:
        props = {}

    # 4) Fusion métriques + optimisation
    metrics = merge_metrics_and_optim(m_meas, w_params, optim_intervals=props, optim_mttr=None) or []
    metrics_df = _ensure_df(metrics, [
        "equipment_code", "MTBF", "MTTR", "MTBF_opt", "MTTR_opt",
        "beta", "eta", "gamma", "interval_opt_h"
    ])

    # 5) Forcer la cohérence finale β/η/γ : on impose fits_df (vérité unique)
    if not metrics_df.empty and not fits_df.empty:
        metrics_df = (
            metrics_df
            .drop(columns=[c for c in ("beta", "eta", "gamma") if c in metrics_df.columns])
            .merge(fits_df, on="equipment_code", how="left")
        )

    # 6) Pipeline avant optimisation (organigramme complet)
    pipeline_by_eq = _compute_pipeline_per_eq(df_ttf)

    # 7) Cast numeric
    for col in ("beta", "eta", "gamma", "MTBF", "MTTR", "MTBF_opt", "MTTR_opt", "interval_opt_h"):
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    return UnifyBundle(
        ttf=df_ttf,
        fits_df=fits_df,
        optim=props,
        metrics_df=metrics_df,
        pipeline_by_eq=pipeline_by_eq,
    )
