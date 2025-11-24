# core/reliability/unify.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Modules existants
from core.reliability.weibull import fit_weibull
from core.reliability.metrics_plus import (
    compute_mtbf_mttr_by_eq, compute_weibull_params_by_eq, merge_metrics_and_optim
)
from core.reliability.optimize import propose_intervals

# (Optionnel) Organigramme complet, si dispo
try:
    from core.reliability.organigram import analyze_ttf_pipeline  # pipeline “Day 2”
except Exception:
    analyze_ttf_pipeline = None  # fallback si non dispo

DATA_CSV = Path("data/failures_saved.csv")

@dataclass
class UnifyOptions:
    force_weibull_2p: bool = True       # γ=0 partout pour cohérence industrielle
    R_target: float = 0.80              # fiabilité cible pour propose_intervals
    min_points: int = 3                 # nb mini d’observations par équipement

@dataclass
class UnifyBundle:
    ttf: pd.DataFrame                   # données TTF normalisées
    fits_df: pd.DataFrame               # β/η/γ par équipement (γ=0 si force_weibull_2p)
    optim: Dict[str, Dict[str, Any]]    # propose_intervals par équipement (toujours {"interval_opt_h": float, ...})
    metrics_df: pd.DataFrame            # tableau final fusionné et “comblé”
    pipeline_by_eq: Dict[str, Dict]     # résumé organigramme par équipement (si dispo, structure normalisée)

# ---------- utilitaires internes ----------
def _load_ttf_df(session_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(session_df, pd.DataFrame):
        df = session_df.copy()
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        return pd.DataFrame()

    if "equipment_code" not in df.columns:
        for c in ("equipment", "equip", "eq", "EQ", "Equipment"):
            if c in df.columns:
                df["equipment_code"] = df[c].astype(str)
                break

    if "equipment_code" not in df.columns:
        return pd.DataFrame()

    if "ttf_h" in df.columns:
        df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
        df = df.dropna(subset=["ttf_h"])
        df = df[df["ttf_h"] > 0]
    else:
        for c in ("hours", "time_to_fail_h", "ttf"):
            if c in df.columns:
                df["ttf_h"] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["ttf_h"])
                df = df[df["ttf_h"] > 0]
                break

    return df

def _fit_weibull_2p(x: np.ndarray) -> Dict[str, float]:
    """Fit Weibull en forçant γ=0 (2 paramètres) pour cohérence globale."""
    ft = fit_weibull(x)  # suppose MLE déterministe
    beta = float(getattr(ft, "beta"))
    eta  = float(getattr(ft, "eta"))
    return {"beta": beta, "eta": eta, "gamma": 0.0}

def _compute_fits(df_ttf: pd.DataFrame, opt: UnifyOptions) -> pd.DataFrame:
    if df_ttf.empty:
        return pd.DataFrame(columns=["equipment_code", "beta", "eta", "gamma"])
    recs: List[Dict[str, Any]] = []
    for eq, grp in df_ttf.groupby("equipment_code"):
        x = grp["ttf_h"].dropna().values
        if len(x) >= opt.min_points:
            try:
                if opt.force_weibull_2p:
                    rec = _fit_weibull_2p(x)
                else:
                    ft = fit_weibull(x)  # 3P si ta fonction sort γ
                    rec = {
                        "beta": float(getattr(ft, "beta")),
                        "eta":  float(getattr(ft, "eta")),
                        "gamma": float(getattr(ft, "gamma", 0.0)),
                    }
                rec["equipment_code"] = str(eq)
                recs.append(rec)
            except Exception:
                pass
    return pd.DataFrame(recs, columns=["equipment_code", "beta", "eta", "gamma"])

def _safe_get(d: Any, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur

def _compute_pipeline_per_eq(df_ttf: pd.DataFrame) -> Dict[str, Dict]:
    """
    Exécute l’organigramme complet par équipement si disponible.
    Sortie NORMALISÉE:
    {
      "EQ": {
        "distribution": {"name": "Weibull2P"},
        "goodness": {"ks_p": float|None, "chi2_p": float|None},
        "trend": {"name": str|None, "p_value": float|None},
        "decision": <any> (optionnel)
      }, ...
    }
    """
    out: Dict[str, Dict] = {}
    if df_ttf.empty or analyze_ttf_pipeline is None:
        return out

    for eq, grp in df_ttf.groupby("equipment_code"):
        x = grp["ttf_h"].dropna().values
        if len(x) < 3:
            continue
        try:
            res = analyze_ttf_pipeline(pd.DataFrame({"ttf_h": x, "equipment_code": [eq]*len(x)}))
            dist_name = _safe_get(res, ["distribution", "name"], "Weibull2P")
            ks_p      = _safe_get(res, ["goodness", "ks_p"])
            chi2_p    = _safe_get(res, ["goodness", "chi2_p"])
            trend_nm  = _safe_get(res, ["trend", "name"])
            trend_p   = _safe_get(res, ["trend", "p_value"])
            out[str(eq)] = {
                "distribution": {"name": dist_name},
                "goodness": {"ks_p": ks_p, "chi2_p": chi2_p},
                "trend": {"name": trend_nm, "p_value": trend_p},
                "decision": res.get("decision"),
            }
        except Exception:
            out[str(eq)] = {}
    return out

def compute_bundle(session_df: Optional[pd.DataFrame] = None,
                   options: Optional[UnifyOptions] = None) -> UnifyBundle:
    """
    === PORTE D’ENTRÉE UNIQUE ===
    Charge les TTF, calcule β/η/γ, propose les intervalles, fusionne MTBF/MTTR,
    et récupère le résumé pipeline (organigramme) si dispo.
    """
    opt = options or UnifyOptions()
    df_ttf = _load_ttf_df(session_df)

    fits_df = _compute_fits(df_ttf, opt)

    # Mesurés
    m_meas = compute_mtbf_mttr_by_eq(df_ttf) if not df_ttf.empty else []
    # Paramètres dérivés
    w_params = compute_weibull_params_by_eq(df_ttf) if not df_ttf.empty else []
    if opt.force_weibull_2p:
        for r in w_params:
            r["gamma"] = 0.0

    # Optimisation — mapping d’objets minimal pour rester compatible
    fits_dict = {
        r["equipment_code"]: type("F", (), r)  # attrs .beta, .eta, .gamma
        for r in fits_df.to_dict(orient="records")
    }
    try:
        raw_props = propose_intervals(fits_dict, R_target=opt.R_target)  # dict eq -> float ou dict
    except Exception:
        raw_props = {}

    # 🔧 NORMALISATION: toujours dict eq -> {"interval_opt_h": float, ...}
    props: Dict[str, Dict[str, Any]] = {}
    for k, v in (raw_props or {}).items():
        kk = str(k)
        if isinstance(v, dict):
            vv = dict(v)
            if "interval_opt_h" not in vv:
                if "interval" in vv:
                    vv["interval_opt_h"] = vv["interval"]
                elif "t_opt_h" in vv:
                    vv["interval_opt_h"] = vv["t_opt_h"]
            props[kk] = vv
        else:
            try:
                props[kk] = {"interval_opt_h": float(v)}
            except Exception:
                props[kk] = {}

    # Fusion propre
    metrics = merge_metrics_and_optim(m_meas, w_params, optim_intervals=props, optim_mttr=None) or []
    # Normalisations anti-trous
    for r in metrics:
        r["gamma"] = 0.0 if (opt.force_weibull_2p or r.get("gamma") is None) else r["gamma"]
        r["MTBF_opt"] = r.get("MTBF_opt")
        r["interval_opt_h"] = r.get("interval_opt_h")

    metrics_df = pd.DataFrame(metrics)
    # Join explicite avec fits_df pour garantir cohérence β/η/γ
    if not metrics_df.empty and not fits_df.empty:
        metrics_df = (
            metrics_df.drop(columns=[c for c in ("beta","eta","gamma") if c in metrics_df.columns])
            .merge(fits_df, on("equipment_code"), how="left")
        )

    # Pipeline/organigramme par équipement (structure normalisée)
    pipeline_by_eq = _compute_pipeline_per_eq(df_ttf)

    return UnifyBundle(
        ttf=df_ttf,
        fits_df=fits_df,
        optim=props,
        metrics_df=metrics_df,
        pipeline_by_eq=pipeline_by_eq,
    )
