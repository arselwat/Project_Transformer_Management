# core/simulation/transformer_sim.py
from __future__ import annotations
import math, random, time
from pathlib import Path
import csv
from dataclasses import dataclass, asdict

DATA_RT_DIR = Path("data/realtime")
DATA_RT_DIR.mkdir(parents=True, exist_ok=True)
SENSORS_CSV = DATA_RT_DIR / "sensors.csv"
FAILURES_CSV = Path("data/failures_saved.csv")  # on réutilise le consolidé

@dataclass
class SimState:
    equipment_code: str = "TR-01"
    hour_counter: float = 0.0    # heures de fonctionnement (compteur “huile”)
    last_failure_h: float = 0.0  # date de la dernière panne (en heures)
    load_pct: float = 50.0       # charge %
    oil_temp_c: float = 45.0     # °C
    vib_g: float = 0.02          # pseudo-accélération
    humidity_pct: float = 35.0   # % HR
    beta: float = 2.2            # forme (Weibull simulée)
    eta: float = 1500.0          # échelle (heures)
    fail_flag: bool = False      # bascule interne (une panne a été créée)

def _append_csv(path: Path, row: dict, field_order: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order or list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def next_sample(state: SimState) -> dict:
    """
    Avance la simulation de 1 minute virtuelle ~ 1 seconde réelle (au choix de la page).
    Retourne un échantillon capteurs et l'écrit dans data/realtime/sensors.csv
    """
    # Avance “temps de service”
    state.hour_counter += 1.0/60.0  # +1 minute
    # Modèles très simples de dérive en fonction de la charge
    load = max(0.0, min(120.0, state.load_pct))
    state.oil_temp_c = 35.0 + 0.35*load + random.uniform(-0.8, 0.8)
    state.vib_g      = 0.015 + 0.0003*load + random.uniform(0.0, 0.01)
    state.humidity_pct = 30 + random.uniform(-3, 3)

    row = {
        "ts_s": int(time.time()),
        "equipment_code": state.equipment_code,
        "hours": round(state.hour_counter, 3),
        "load_pct": round(load, 1),
        "oil_temp_c": round(state.oil_temp_c, 2),
        "vibration_g": round(state.vib_g, 4),
        "humidity_pct": round(state.humidity_pct, 1),
    }
    _append_csv(SENSORS_CSV, row, [
        "ts_s","equipment_code","hours","load_pct","oil_temp_c","vibration_g","humidity_pct"
    ])
    return row

def maybe_failure(state: SimState, force: bool=False) -> dict | None:
    """
    Crée une panne simulée si probabilité atteinte (Weibull parallèle) ou forcée.
    Ajoute une ligne dans data/failures_saved.csv avec equipment_code + ttf_h.
    """
    # Probabilité instantanée ~ hazard Weibull discrétisée
    def hazard_weibull(t: float, beta: float, eta: float) -> float:
        if t <= 0 or eta <= 0 or beta <= 0:
            return 0.0
        # h(t) = (beta/eta)*(t/eta)^(beta-1)
        return (beta/eta)*((max(t, 1e-6)/eta)**(beta-1))

    t_since_last = state.hour_counter - state.last_failure_h
    p = hazard_weibull(max(t_since_last, 1e-6), state.beta, state.eta) / 60.0  # par minute
    trigger = force or (random.random() < p)

    if not trigger:
        return None

    # Enregistre TTF (heures depuis la dernière défaillance)
    ttf = max(0.1, t_since_last)
    row = {
        "equipment_code": state.equipment_code,
        "ttf_h": round(ttf, 3),
        "duree_rep_h": round(random.uniform(1.5, 6.0), 2)  # MTTR simulé
    }
    _append_csv(FAILURES_CSV, row, ["equipment_code","ttf_h","duree_rep_h"])
    # Reset point de référence
    state.last_failure_h = state.hour_counter
    state.fail_flag = True
    return row
