from pathlib import Path
import json
import pandas as pd

# Racine projet = .../fiabilite_stock_project
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
EQUIP_JSON = CONFIG_DIR / "equipment.json"

def list_equipment_codes(max_from_df: int | None = None) -> list[str]:
    """
    Retourne la liste des codes équipement en combinant:
    - data/failures_saved.csv (dataset actif)
    - config/equipment.json   (plaques signalétiques)

    max_from_df: limite combien on prend depuis le CSV (utile si très gros).
    """
    codes = set()

    # 1) depuis le dataset actif (s'il existe)
    csv = DATA_DIR / "failures_saved.csv"
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            # heuristique simple pour détecter la colonne équipement
            candidates = [c for c in df.columns if "equip" in c.lower() or "code" in c.lower()]
            if candidates:
                col = candidates[0]
                vals = (df[col].dropna().astype(str).str.strip().unique().tolist())
                if max_from_df:
                    vals = vals[:max_from_df]
                codes.update(vals)
        except Exception:
            pass

    # 2) depuis equipment.json (plaques)
    if EQUIP_JSON.exists():
        try:
            eq = json.loads(EQUIP_JSON.read_text(encoding="utf-8"))
            codes.update(list(eq.keys()))
        except Exception:
            pass

    return sorted(list(codes))
