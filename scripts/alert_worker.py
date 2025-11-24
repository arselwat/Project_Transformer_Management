# scripts/alert_worker.py
from __future__ import annotations
import time
import pandas as pd
from pathlib import Path
from core.notify.rt_alerts import notify_event

EVT_CSV = Path("data/realtime_events.csv")

def run(poll_s: float = 5.0):
    EVT_CSV.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            if EVT_CSV.exists():
                df = pd.read_csv(EVT_CSV, dtype=str)
                if not df.empty:
                    if "processed" not in df.columns:
                        df["processed"] = "0"
                    # traiter seulement les non traités
                    mask = df["processed"].fillna("0") == "0"
                    todo = df[mask].to_dict(orient="records")
                    for row in todo:
                        # caster value/threshold si possible (pas obligatoire)
                        try:
                            row["value"] = float(row.get("value",""))
                        except Exception:
                            pass
                        try:
                            row["threshold"] = float(row.get("threshold",""))
                        except Exception:
                            pass
                        res = notify_event(row)
                        # marquer comme traité (quel que soit le résultat, pour éviter loop)
                        # si tu veux re-tenter seulement en cas d'échec: adapte ici
                        ridx = df.index[(df["ts"] == row.get("ts")) & (df["code"] == row.get("code"))]
                        df.loc[ridx, "processed"] = "1"
                    # sauvegarder
                    df.to_csv(EVT_CSV, index=False)
        except Exception:
            # ne crash pas, boucle continue
            pass
        time.sleep(poll_s)

if __name__ == "__main__":
    run()
