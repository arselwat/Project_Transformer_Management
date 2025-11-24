# core/reliability/unify.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Tes modules existants
from core.reliability.weibull import fit_weibull
from core.reliability.metrics_plus import (
    compute_mtbf_mttr_by_eq,          # -> List[Dict]
    compute_weibull_params_by_eq,     # -> List[Dict]
    merge_metrics_and_optim           # -> List[Dict]
)
from core.reliability.optimize import propose_intervals

# (Optionnel) Organigramme complet
try:
    from core.reliability.organigram import analyze_ttf_pipeline  # Day 2
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
    ttf: pd.DataFrame                 # colonnes au moins: equipment_code, ttf_h
    fits_df: pd.DataFrame             # colonnes: equipment_code, beta, eta, gamma
    optim: Dict[str, Dict[str, Any]]  # propose_intervals
    metrics_df: pd.DataFrame          # tableau final fusionné
    pipeline_by_eq: Dict[str, Dict]   # résumé organigramme

# ------------------ utilitaires sûrs ------------------

def _ensure_df(obj: Any, cols: List[str]) -> pd.DataFrame:
    """
    Convertit 'obj' en DataFrame avec au moins les colonnes 'cols'.
    - Si obj == list[dict] -> DataFrame
    - Si obj == dict -> DataFrame([obj])
    - Si obj == DataFrame -> retourne
    - Sinon -> DataFrame vide
    """
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
            df[c] = pd.Series(dtype="float64") if c in ("beta","eta","gamma","MTBF","MTTR") else ""
    return df

def _load_ttf_df(session_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(session_df, pd.DataFrame):
        df = session_df.copy()
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        return pd.DataFrame(columns=["equipment_code","ttf_h"])

    # normalise equipment_code
    if "equipment_code" not in df.columns:
        for c in ("equipment", "equip", "eq", "EQ", "Equipment"):
            if c in df.columns:
                df["equipment_code"] = df[c].astype(str)
                break
    if "equipment_code" not in df.columns:
        return pd.DataFrame(columns=["equipment_code","ttf_h"])

    # normalise ttf_h
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
    ft = fit_weibull(x)  # suppose MLE déterministe
    beta = float(getattr(ft, "beta"))
    eta  = float(getattr(ft, "eta"))
    return {"beta": beta, "eta": eta, "gamma": 0.0}

def _compute_fits(df_ttf: pd.DataFrame, opt: UnifyOptions) -> pd.DataFrame:
    if df_ttf.empty:
        return pd.DataFrame(columns=["equipment_code","beta","eta","gamma"])
    recs: List[Dict[str, Any]] = []
    for eq, grp in df_ttf.groupby("equipment_code"):
        x = grp["ttf_h"].dropna().values
        if len(x) >= opt.min_points:
            try:
                if opt.force_weibull_2p:
                    rec = _fit_weibull_2p(x)
                else:
                    ft = fit_weibull(x)
                    rec = {
                        "beta": float(getattr(ft, "beta")),
                        "eta":  float(getattr(ft, "eta")),
                        "gamma": float(getattr(ft, "gamma", 0.0)),
                    }
                rec["equipment_code"] = str(eq)
                recs.append(rec)
            except Exception:
                pass
    df = pd.DataFrame(recs, columns=["equipment_code","beta","eta","gamma"])
    return df

def _normalize_pipeline_output(res: dict) -> dict:
    """Uniformise la sortie de analyze_ttf_pipeline vers un schéma minimal."""
    out = {}
    if not isinstance(res, dict):
        return out

    # Distribution
    d = res.get("distribution")
    if isinstance(d, dict):
        out["distribution"] = {"name": d.get("name", "Weibull2P")}
    elif isinstance(d, str):
        out["distribution"] = {"name": d}
    else:
        out["distribution"] = {"name": "Weibull2P"}

    # Goodness-of-fit
    g = res.get("goodness") or res.get("gof") or {}
    if isinstance(g, dict):
        out["goodness"] = {
            "ks_p":  g.get("ks_p")  or g.get("ks_pvalue")  or g.get("ks_pv"),
            "chi2_p": g.get("chi2_p") or g.get("chi2_pvalue") or g.get("chi2_pv"),
        }

    # Trend
    tr = res.get("trend") or res.get("trend_mk") or {}
    if isinstance(tr, dict):
        out["trend"] = {"name": tr.get("name", "MK"),
                        "p_value": tr.get("p_value") or tr.get("p")}

    # Model (si présent)
    if "model" in res:
        out["model"] = res.get("model")

    return out

def _compute_pipeline_per_eq(df_ttf: pd.DataFrame) -> Dict[str, Dict]:
    """
    Exécute ton organigramme complet par équipement si disponible.
    Tolérant aux signatures (liste/ndarray/DataFrame) et aux anciens formats de retour.
    """
    out: Dict[str, Dict] = {}
    if df_ttf.empty or analyze_ttf_pipeline is None:
        return out

    for eq, grp in df_ttf.groupby("equipment_code"):
        try:
            x = grp["ttf_h"].dropna().astype(float).values
        except Exception:
            continue
        if len(x) < 3:
            continue

        res = None
        # 1) Tentative : vecteur numpy
        try:
            res = analyze_ttf_pipeline(x)
        except TypeError:
            res = None
        except Exception:
            res = None

        # 2) Tentative fallback : DataFrame
        if not isinstance(res, dict):
            try:
                res = analyze_ttf_pipeline(pd.DataFrame({"ttf_h": x, "equipment_code": [eq]*len(x)}))
            except Exception:
                res = None

        if isinstance(res, dict):
            try:
                out[str(eq)] = _normalize_pipeline_output(res)
            except Exception:
                out[str(eq)] = {}
        else:
            out[str(eq)] = {}

    return out


# ------------------ API principale ------------------

def compute_bundle(session_df: Optional[pd.DataFrame] = None,
                   options: Optional[UnifyOptions] = None) -> UnifyBundle:
    """
    === PORTE D’ENTRÉE UNIQUE ===
    - Charge les TTF
    - Calcule β/η/γ cohérents (γ=0 si force_weibull_2p=True)
    - Propose les intervalles d’entretien
    - Fusionne MTBF/MTTR + β/η/γ + interval_opt_h
    - Récupère un résumé pipeline (si dispo)
    Retourne toujours des DataFrame (pas des dict bruts).
    """
    opt = options or UnifyOptions()
    df_ttf = _load_ttf_df(session_df)

    # 1) Fits Weibull (cohérents)
    fits_df = _compute_fits(df_ttf, opt)  # DF: equipment_code, beta, eta, gamma

    # 2) Mesurés & paramètres (robustes)
    m_meas = compute_mtbf_mttr_by_eq(df_ttf) if not df_ttf.empty else []
    w_params = compute_weibull_params_by_eq(df_ttf) if not df_ttf.empty else []

    # tolérance : si des éléments ne sont pas des dict, on filtre
    w_params = [r for r in (w_params or []) if isinstance(r, dict)]
    if opt.force_weibull_2p:
        for r in w_params:
            # certains impl peuvent renvoyer gamma None
            r["gamma"] = 0.0
    w_params_df = _ensure_df(w_params, ["equipment_code","beta","eta","gamma"])

    # 3) Optimisation (propose_intervals attend un mapping eq -> objet .beta/.eta/.gamma)
    fits_dict = {
        row["equipment_code"]: type("F", (), {
            "beta": float(row["beta"]),
            "eta":  float(row["eta"]),
            "gamma": float(row.get("gamma", 0.0) or 0.0),
        })
        for _, row in fits_df.iterrows()
    }
    try:
        props = propose_intervals(fits_dict, R_target=opt.R_target) or {}
    except Exception:
        props = {}

    # 4) Fusion métriques
    metrics = merge_metrics_and_optim(m_meas, w_params, optim_intervals=props, optim_mttr=None) or []
    metrics_df = _ensure_df(metrics, [
        "equipment_code","MTBF","MTTR","MTBF_opt","MTTR_opt","beta","eta","gamma","interval_opt_h"
    ])

    # 5) Forcer la cohérence finale des β/η/γ (on prend ceux de fits_df)
    if not metrics_df.empty and not fits_df.empty:
        metrics_df = (
            metrics_df
            .drop(columns=[c for c in ("beta","eta","gamma") if c in metrics_df.columns])
            .merge(fits_df, on="equipment_code", how="left")
        )

    # 6) Pipeline (organigramme) résumé
    pipeline_by_eq = _compute_pipeline_per_eq(df_ttf)

    # 7) Types numériques propres (évite 'object')
    for col in ("beta","eta","gamma","MTBF","MTTR","MTBF_opt","MTTR_opt","interval_opt_h"):
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    return UnifyBundle(
        ttf=df_ttf,
        fits_df=fits_df,
        optim=props,
        metrics_df=metrics_df,
        pipeline_by_eq=pipeline_by_eq,
    )
