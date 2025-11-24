from __future__ import annotations
from typing import Literal, Dict, Tuple
import pandas as pd

UnitMode = Literal["SI", "HT"]

# Facteurs SI -> HT (affichage)
_FACTORS = {
    "V": 1e-3,    # V -> kV
    "A": 1e-3,    # A -> kA
    "W": 1e-6,    # W -> MW
    "VA": 1e-6,   # VA -> MVA
}

_LABELS = {
    ("SI","V"): "V", ("HT","V"): "kV",
    ("SI","A"): "A", ("HT","A"): "kA",
    ("SI","W"): "W", ("HT","W"): "MW",
    ("SI","VA"): "VA", ("HT","VA"): "MVA",
    ("SI","Hz"): "Hz", ("HT","Hz"): "Hz",
    ("SI","PF"): "",  ("HT","PF"): "",
    ("SI","C"): "°C", ("HT","C"): "°C",
    ("SI","MU"): "Pa·s", ("HT","MU"): "Pa·s",
}

def scale_value(x, base: str, mode: UnitMode) -> float:
    if x is None: return None
    if mode == "SI": return float(x)
    f = _FACTORS.get(base, 1.0)
    try: return float(x) * f
    except Exception: return float("nan")

def unit_label(base: str, mode: UnitMode) -> str:
    return _LABELS.get((mode, base), base)

def convert_df_for_display(df: pd.DataFrame, mode: UnitMode) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Retourne un DataFrame converti + labels par colonne."""
    g = df.copy()
    labels: Dict[str,str] = {}
    # colonnes usuelles de ta page 4
    col_defs = {
        "v_sec": ("V","V"), "v_prim_rms": ("V","V"),
        "i_sec": ("A","A"), "i_prim_rms": ("A","A"),
        "p_sec": ("W","W"), # puissance active affichée
        "S_sec": ("VA","VA"), # si un jour tu ajoutes S
        "pf_prim": ("PF","PF"), "freq": ("Hz","Hz"),
        "t_core": ("C","C"), "mu_oil": ("MU","MU"),
    }
    for c,(base,_) in col_defs.items():
        if c in g.columns:
            g[c] = g[c].apply(lambda x: scale_value(x, base, mode))
            labels[c] = unit_label(base, mode)
    return g, labels
