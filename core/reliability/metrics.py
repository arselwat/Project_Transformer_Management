# core/reliability/metrics.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

def compute_mtbf_mttr(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrée : df avec colonnes
      - 'ttf_h' (float) : temps entre pannes (heures)
      - 'duree_rep_h' (float, optionnel) : durée de réparation
    Retour :
      {'global': {'MTBF': float|None, 'MTTR': float|None}}
    """
    out = {"global": {"MTBF": None, "MTTR": None}}
    if df is None or df.empty:
        return out

    if "ttf_h" in df.columns:
        ttfs = pd.to_numeric(df["ttf_h"], errors="coerce").dropna()
        ttfs = ttfs[ttfs > 0]
        if not ttfs.empty:
            out["global"]["MTBF"] = float(ttfs.mean())

    if "duree_rep_h" in df.columns:
        reps = pd.to_numeric(df["duree_rep_h"], errors="coerce").dropna()
        reps = reps[reps >= 0]
        if not reps.empty:
            out["global"]["MTTR"] = float(reps.mean())

    return out
