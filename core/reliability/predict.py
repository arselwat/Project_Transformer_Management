import pandas as pd
import numpy as np
from typing import Dict, List
from .weibull import fit_weibull_mle, reliability_weibull

def load_failures_csv_for_scheduler(path: str = "data/failures_saved.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df["heures_fonct"] = pd.to_numeric(df["heures_fonct"], errors="coerce")
        df = df.dropna(subset=["equipment_code","heures_fonct"])
        return df
    except Exception:
        return pd.DataFrame()

def failure_risk_by_equipment(df: pd.DataFrame, horizon_h: float = 720.0, min_points: int = 3) -> List[Dict]:
    """
    Retourne une liste de dicts: {equipment_code, beta, eta, risk}
    risk = P(failure within horizon) = 1 - R(horizon).
    """
    out = []
    if df is None or df.empty: return out
    for eq, g in df.groupby("equipment_code"):
        times = g["heures_fonct"].dropna().astype(float).values
        if len(times) < min_points:
            continue
        fit = fit_weibull_mle(times)
        if not fit:
            continue
        beta, eta = fit
        R = reliability_weibull(beta, eta, horizon_h)
        if R is None: 
            continue
        risk = 1.0 - float(R)
        out.append({"equipment_code": eq, "beta": beta, "eta": eta, "risk": risk})
    # tri décroissant risque
    out.sort(key=lambda x: x["risk"], reverse=True)
    return out
