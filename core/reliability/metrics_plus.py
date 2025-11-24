# core/reliability/metrics_plus.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Union

import pandas as pd 
import numpy as np
import pandas as pd

from core.reliability.weibull import fit_weibull


def compute_mtbf_mttr_by_eq(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    df: attend au moins 'equipment_code'
        - si 'ttf_h' existe => MTBF = moyenne(t tf_h)
        - si 'duree_rep_h' existe => MTTR = moyenne(duree_rep_h)
    Retour: {eq: {"MTBF": val|None, "MTTR": val|None}}
    """
    out: Dict[str, Dict[str, float]] = {}
    if "equipment_code" not in df.columns:
        return out

    g = df.groupby("equipment_code", dropna=True)
    for eq, d in g:
        mtbf = None
        mttr = None
        if "ttf_h" in d.columns:
            x = pd.to_numeric(d["ttf_h"], errors="coerce").dropna()
            x = x[x > 0]
            if len(x) > 0:
                mtbf = float(np.mean(x))
        if "duree_rep_h" in d.columns:
            r = pd.to_numeric(d["duree_rep_h"], errors="coerce").dropna()
            r = r[r >= 0]
            if len(r) > 0:
                mttr = float(np.mean(r))
        out[str(eq)] = {"MTBF": mtbf, "MTTR": mttr}
    return out


def compute_weibull_params_by_eq(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit Weibull par équipement si >=3 TTF.
    Retour: {eq: {"beta":..., "eta":..., "gamma":...}}
    - gamma n'existe pas forcément dans ton fit; on met 0.0 si absent.
    """
    out: Dict[str, Dict[str, float]] = {}
    if "equipment_code" not in df.columns or "ttf_h" not in df.columns:
        return out

    g = df.groupby("equipment_code", dropna=True)
    for eq, d in g:
        x = pd.to_numeric(d["ttf_h"], errors="coerce").dropna()
        x = x[x > 0]
        if len(x) >= 3:
            try:
                ft = fit_weibull(x.values)
                beta = float(getattr(ft, "beta"))
                eta  = float(getattr(ft, "eta"))
                gamma = float(getattr(ft, "gamma", 0.0))  # si non dispo → 0.0
                out[str(eq)] = {"beta": beta, "eta": eta, "gamma": gamma}
            except Exception:
                pass
    return out


def _to_map(obj: Any, key: str = "equipment_code") -> Dict[str, Dict[str, Any]]:
    """
    Normalise en mapping {equipment_code: {…}}.
    - dict déjà au bon format: renvoie tel quel (si valeurs sont des dicts)
    - list[dict]: indexe par item[key]
    - DataFrame: records -> indexés par key
    - Autre: {}
    """
    if isinstance(obj, dict):
        # Si c'est déjà {eq: {..}} on garde
        if all(isinstance(v, dict) for v in obj.values()):
            return {str(k): dict(v) for k, v in obj.items()}
        # Sinon, peut-être un flat dict => on l'emballe si key présent
        if key in obj:
            return {str(obj[key]): dict(obj)}
        return {}

    if isinstance(obj, list):
        out: Dict[str, Dict[str, Any]] = {}
        for it in obj:
            if isinstance(it, dict) and key in it:
                out[str(it[key])] = dict(it)
        return out

    if isinstance(obj, pd.DataFrame):
        if key in obj.columns:
            return {str(r[key]): dict(r) for r in obj.to_dict(orient="records")}
        return {}

    return {}

def _get_interval_opt(optim_intervals: Any, eq: str):
    """
    Récupère un intervalle optimisé depuis un mapping hétérogène.
    - float direct
    - dict avec 'interval_opt_h'
    - sinon None
    """
    if not optim_intervals:
        return None
    v = None
    if isinstance(optim_intervals, dict):
        v = optim_intervals.get(eq)
    # liste de paires éventuelle ?
    if v is None and isinstance(optim_intervals, list):
        for it in optim_intervals:
            if isinstance(it, dict) and str(it.get("equipment_code")) == str(eq):
                v = it
                break
    if isinstance(v, dict):
        return v.get("interval_opt_h")
    try:
        return float(v)
    except Exception:
        return None

def merge_metrics_and_optim(
    mtbf_mttr: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame],
    weibull_params: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame],
    optim_intervals: Any = None,
    optim_mttr: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, None] = None,
) -> List[Dict[str, Any]]:
    """
    Fusionne de manière robuste:
      - mtbf_mttr: peut être dict | list[dict] | DataFrame
      - weibull_params: idem
      - optim_intervals: dict {eq: float|{interval_opt_h:..}} ou list d'objets
      - optim_mttr: optionnel (même logique que mtbf_mttr)

    Sortie: list[dict] avec au minimum:
      equipment_code, MTBF, MTTR, MTBF_opt, MTTR_opt, beta, eta, gamma, interval_opt_h
    """
    mm = _to_map(mtbf_mttr, key="equipment_code")
    ww = _to_map(weibull_params, key="equipment_code")
    om = _to_map(optim_mttr, key="equipment_code") if optim_mttr is not None else {}

    eqs = set(mm.keys()) | set(ww.keys()) | set(om.keys())
    out: List[Dict[str, Any]] = []

    for eq in sorted(eqs):
        m = mm.get(eq, {})
        w = ww.get(eq, {})
        o = om.get(eq, {})
        row: Dict[str, Any] = {"equipment_code": eq}

        # MTBF/MTTR mesurés
        row["MTBF"] = _safe_float(m.get("MTBF"))
        row["MTTR"] = _safe_float(m.get("MTTR"))

        # Params Weibull
        row["beta"]  = _safe_float(w.get("beta"))
        row["eta"]   = _safe_float(w.get("eta"))
        # gamma peut manquer ou être None -> garder None (ou 0 si ta politique le force ailleurs)
        row["gamma"] = _safe_float(w.get("gamma"))

        # Optimisation
        row["MTBF_opt"]     = _safe_float(o.get("MTBF_opt"))
        row["MTTR_opt"]     = _safe_float(o.get("MTTR_opt"))
        row["interval_opt_h"] = _safe_float(_get_interval_opt(optim_intervals, eq))

        out.append(row)

    return out

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and (pd.isna(x))):
            return None
        return float(x)
    except Exception:
        return None


def merge_metrics_and_optim(
    mtbf_mttr: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame],
    weibull_params: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame],
    optim_intervals: Any = None,
    optim_mttr: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, None] = None,
) -> List[Dict[str, Any]]:
    mm = _to_map(mtbf_mttr)
    ww = _to_map(weibull_params)
    om = _to_map(optim_mttr) if optim_mttr is not None else {}

    eqs = set(mm.keys()) | set(ww.keys()) | set(om.keys())
    out: List[Dict[str, Any]] = []

    for eq in sorted(eqs):
        m = mm.get(eq, {})
        w = ww.get(eq, {})
        o = om.get(eq, {})
        row: Dict[str, Any] = {"equipment_code": eq}
        row["MTBF"] = _safe_float(m.get("MTBF"))
        row["MTTR"] = _safe_float(m.get("MTTR"))
        row["beta"]  = _safe_float(w.get("beta"))
        row["eta"]   = _safe_float(w.get("eta"))
        row["gamma"] = _safe_float(w.get("gamma"))
        row["MTBF_opt"]     = _safe_float(o.get("MTBF_opt"))
        row["MTTR_opt"]     = _safe_float(o.get("MTTR_opt"))
        row["interval_opt_h"] = _safe_float(_get_interval_opt(optim_intervals, eq))
        out.append(row)

    return out