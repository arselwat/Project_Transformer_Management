# core/realtime/sim_internal.py
from __future__ import annotations
import threading, time, math, json
from pathlib import Path
from dataclasses import dataclass, asdict
import random
import csv

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
MEAS_CSV = DATA_DIR / "realtime_measures.csv"
EVTS_CSV = DATA_DIR / "realtime_events.csv"

@dataclass
class SimControls:
    load_pct: float = 40.0        # % de charge
    ambient_c: float = 28.0       # °C
    pf_target: float = 0.95       # facteur de puissance nominal
    force_overcurrent: bool = False
    force_overtemp: bool = False
    freq_hz: float = 50.0
    vprim_nom: float = 220.0
    vsec_nom: float = 20.0
    p_nom_w: float = 500.0        # maquette (≈ 0.5 kW)
    step_hz: float = 10.0         # cadence génération

class InternalTransformerSim:
    def __init__(self):
        self.controls = SimControls()
        self._th = None
        self._stop = threading.Event()
        self._t_core = 35.0   # °C
        self._energy_wh = 0.0

        # Fichiers CSV : header si absent
        if not MEAS_CSV.exists():
            with open(MEAS_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","v_sec","i_sec","p_sec","t_core","v_prim_rms","i_prim_rms","pf_prim","freq","status"])
        if not EVTS_CSV.exists():
            with open(EVTS_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","level","code","msg","value","threshold","processed"])

    def set_controls(self, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self.controls, k):
                setattr(self.controls, k, v)

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()

    def _emit_event(self, code: str, level: str, msg: str, value: float, threshold: float):
        ts = time.time()
        with open(EVTS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, level, code, msg, value, threshold, 0])

    def _run(self):
        # Modèle thermique simple 1er ordre
        k_heat = 0.08   # échauffement (°C/s) pour charge=100 %
        k_cool = 0.02   # refroidissement (°C/s) vers ambient
        overtemp_warn = 65.0
        overtemp_alarm = 85.0

        over_i_warn_factor = 1.00   # I >= 100% In  => WARN
        over_i_alarm_factor = 1.20  # I >= 120% In  => ALARM

        last = time.time()
        while not self._stop.is_set():
            dt = max(1.0/self.controls.step_hz, 0.05)
            time.sleep(dt)
            ts = time.time()

            # Charge & puissance
            load = max(0.0, min(self.controls.load_pct, 200.0)) / 100.0
            p_sec = self.controls.p_nom_w * load
            # variations réalistes
            p_sec *= (1.0 + 0.02*math.sin(ts/3.0) + random.uniform(-0.01, 0.01))

            # Secondaire
            v_sec = self.controls.vsec_nom * (1.0 - 0.02*load) + random.uniform(-0.05, 0.05)
            v_sec = max(0.1, v_sec)
            i_sec = p_sec / v_sec

            # Primaire RMS (approx) & PF
            v_prim = self.controls.vprim_nom + random.uniform(-0.4, 0.4)
            pf = max(0.5, min(1.0, self.controls.pf_target + random.uniform(-0.03, 0.03)))
            s_va = p_sec / max(0.05, pf)
            i_prim = s_va / max(1e-3, v_prim)

            # Thermique
            target = self.controls.ambient_c + 10.0 + 45.0*load
            self._t_core += (target - self._t_core) * k_heat * dt
            self._t_core += (self.controls.ambient_c - self._t_core) * k_cool * dt

            # Forçages
            if self.controls.force_overcurrent:
                i_prim *= 1.35
            if self.controls.force_overtemp:
                self._t_core = max(self._t_core, overtemp_alarm + 2.0)

            # Fréquence
            freq = self.controls.freq_hz + random.uniform(-0.05, 0.05)

            # Energy Wh (approx)
            self._energy_wh += (p_sec * dt) / 3600.0

            status = "OK"
            # Événements courants
            in_i_warn = i_prim >= over_i_warn_factor * (self.controls.p_nom_w / self.controls.vprim_nom / max(pf,0.5))
            in_i_alarm = i_prim >= over_i_alarm_factor * (self.controls.p_nom_w / self.controls.vprim_nom / max(pf,0.5))

            if self._t_core >= overtemp_alarm:
                status = "ALARM"
                self._emit_event("TEMP_HIGH", "ALARM",
                                 f"Température noyau >= {overtemp_alarm} C", round(self._t_core,2), overtemp_alarm)
            elif self._t_core >= overtemp_warn:
                status = "WARN"
                self._emit_event("TEMP_WARN", "WARN",
                                 f"Température noyau >= {overtemp_warn} C", round(self._t_core,2), overtemp_warn)

            if in_i_alarm:
                status = "ALARM"
                thr = over_i_alarm_factor
                self._emit_event("OVERCURRENT", "ALARM", "Courant primaire >= 120% In", round(i_prim,3), thr)
            elif in_i_warn:
                status = "WARN"
                thr = over_i_warn_factor
                self._emit_event("HIGH_CURRENT", "WARN", "Courant primaire >= 100% In", round(i_prim,3), thr)

            if pf < 0.85:
                status = "WARN"
                self._emit_event("PF_LOW", "WARN", "Facteur de puissance < 0.85", round(pf,3), 0.85)

            # Écriture mesures
            with open(MEAS_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts, round(v_sec,3), round(i_sec,3), round(p_sec,2), round(self._t_core,2),
                    round(v_prim,2), round(i_prim,3), round(pf,3), round(freq,2), status
                ])
