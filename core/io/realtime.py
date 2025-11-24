# core/io/realtime.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

SENSORS_CSV = Path("data/realtime/sensors.csv")

def load_live_sensors(n_last: int | None = 500) -> pd.DataFrame:
    if not SENSORS_CSV.exists():
        return pd.DataFrame(columns=["ts_s","equipment_code","hours","load_pct","oil_temp_c","vibration_g","humidity_pct"])
    try:
        df = pd.read_csv(SENSORS_CSV)
        if n_last and len(df) > n_last:
            df = df.iloc[-n_last:]
        return df
    except Exception:
        return pd.DataFrame()

def clear_live_buffer():
    if SENSORS_CSV.exists():
        try:
            SENSORS_CSV.unlink()
        except Exception:
            pass
