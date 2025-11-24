# core/shared/csv_robust.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import csv

EXPECTED_EVT = ["ts","site","equipment","level","code","msg","value","threshold","processed"]

def flex_read_csv(path: str | Path, expected: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=expected or [])
    # 1) Lecture standard
    try:
        df = pd.read_csv(p, dtype=str)
        return df
    except Exception:
        pass
    # 2) Sniffer + DictReader (tolère 7/9 colonnes, virgules, etc.)
    rows = []
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
            except Exception:
                class _D(csv.Dialect):
                    delimiter = ","
                    quotechar = '"'
                    doublequote = True
                    skipinitialspace = False
                    lineterminator = "\n"
                    quoting = csv.QUOTE_MINIMAL
                dialect = _D()
            reader = csv.DictReader(f, dialect=dialect)
            for r in reader:
                rows.append({(k or "").strip(): (v if v is not None else "") for k, v in (r or {}).items()})
        df = pd.DataFrame(rows, dtype=str)
    except Exception:
        # 3) Dernier recours : on skippe seulement les lignes corrompues
        try:
            df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=expected or [])
    # Colonnes attendues garanties
    if expected:
        for c in expected:
            if c not in df.columns:
                df[c] = ""
    return df

def safe_epoch_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convertit des epochs seconds ou milliseconds → datetime,
    sans overflow. On évite d’appeler unit='s' sur des valeurs > 9e9.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    sec_mask = s.abs() <= 9_000_000_000          # ~285 ans en secondes
    ms_mask  = (~sec_mask) & (s.abs() <= 9_000_000_000_000)

    if sec_mask.any():
        out.loc[sec_mask] = pd.to_datetime(s[sec_mask], unit="s",  errors="coerce")
    if ms_mask.any():
        out.loc[ms_mask]  = pd.to_datetime(s[ms_mask],  unit="ms", errors="coerce")

    return out
