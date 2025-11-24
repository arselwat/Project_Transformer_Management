from __future__ import annotations
import json, time
from pathlib import Path
from collections import deque
from typing import Dict, Any, List

CFG_FILE = Path("config/realtime_thresholds.json")

def load_cfg() -> Dict[str, Any]:
    cfg = {
        "i_nom": 0.5, "i_sec_nom": 5.0,
        "warn": {"t_core":60, "i_prim_rms_factor":0.9, "i_sec_factor":0.9, "pf_prim":0.9, "freq_dev":0.4, "v_dev_pct":0.10, "thd_i":10},
        "alarm":{"t_core":70, "i_prim_rms_factor":1.0, "i_sec_factor":1.1, "pf_prim":0.8, "freq_dev":0.8, "v_dev_pct":0.15, "thd_i":15, "overcurrent_hold_s":60},
        "hysteresis":{"t_core":3, "i_pct":0.05, "pf":0.03, "freq":0.2, "v_pct":0.02},
        "windows":{"t_core_s":10, "i_s":3, "pf_s":10, "freq_s":5, "v_ms":500},
    }
    try:
        if CFG_FILE.exists():
            cfg.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg

class Debounce:
    def __init__(self, secs: float):
        self.secs = secs
        self.q: deque = deque()  # timestamps d’échantillons hors seuil
    def hit(self, ts: float):
        self.q.append(ts)
        while self.q and ts - self.q[0] > self.secs:
            self.q.popleft()
    def is_triggered(self) -> bool:
        return (self.secs <= 0) or (len(self.q) > 0 and (self.q[-1] - self.q[0]) >= self.secs)

class EventEvaluator:
    def __init__(self, site="bench1", equip="tr_220_20"):
        self.cfg = load_cfg()
        self.site = site
        self.equip = equip
        # debouncers
        w = self.cfg["windows"]
        self.db_t = Debounce(w.get("t_core_s",10))
        self.db_i = Debounce(w.get("i_s",3))
        self.db_pf= Debounce(w.get("pf_s",10))
        self.db_f = Debounce(w.get("freq_s",5))
        # mémoires “hold”
        self.oc_start: float|None = None

    def _evt(self, level:str, code:str, msg:str, value:float=None, threshold:float=None) -> Dict[str,Any]:
        return {
            "ts": time.time(),
            "level": level, "code": code, "msg": msg,
            "value": value, "threshold": threshold,
            "site": self.site, "equipment": self.equip
        }

    def evaluate(self, m: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        m = measures JSON (voir mqtt_stream). Retourne une liste d’événements (peut être vide).
        """
        evts: List[Dict[str, Any]] = []
        cfg = self.cfg; ts = float(m.get("ts", time.time()))
        # alias
        t_core = float(m.get("t_core", 0))
        ip = float(m.get("i_prim_rms", 0))
        isec = float(m.get("i_sec", 0))
        pf = float(m.get("pf_prim", 1.0))
        freq = float(m.get("freq", 50.0))
        vp = float(m.get("v_prim_rms", 230.0))
        vs = float(m.get("v_sec", 20.0))
        v_nom_p = 230.0; v_nom_s = 20.0  # tu peux les rendre configurables

        # TEMP
        if t_core > cfg["warn"]["t_core"]:
            self.db_t.hit(ts)
            if t_core > cfg["alarm"]["t_core"] and self.db_t.is_triggered():
                evts.append(self._evt("ALARM","TEMP_HIGH","Temp noyau > seuil critique", t_core, cfg["alarm"]["t_core"]))
            elif self.db_t.is_triggered():
                evts.append(self._evt("WARN","TEMP_HIGH","Temp noyau > seuil avert.", t_core, cfg["warn"]["t_core"]))

        # OVERCURRENT prim
        f_warn = cfg["warn"]["i_prim_rms_factor"] * cfg["i_nom"]
        f_alrm = cfg["alarm"]["i_prim_rms_factor"] * cfg["i_nom"]
        if ip > f_warn:
            self.db_i.hit(ts)
            if ip > f_alrm and self.db_i.is_triggered():
                # hold time for alarm
                if self.oc_start is None:
                    self.oc_start = ts
                if (ts - self.oc_start) >= cfg["alarm"]["overcurrent_hold_s"]:
                    evts.append(self._evt("ALARM","OVERCURRENT", "I_prim > In soutenu", ip, f_alrm))
            else:
                evts.append(self._evt("WARN","OVERCURRENT","I_prim > 0.9 In", ip, f_warn))
        else:
            self.oc_start = None  # reset

        # PF LOW
        if pf < cfg["warn"]["pf_prim"]:
            self.db_pf.hit(ts)
            if pf < cfg["alarm"]["pf_prim"] and self.db_pf.is_triggered():
                evts.append(self._evt("ALARM","PF_LOW","Facteur de puissance bas", pf, cfg["alarm"]["pf_prim"]))
            elif self.db_pf.is_triggered():
                evts.append(self._evt("WARN","PF_LOW","PF sous 0.9", pf, cfg["warn"]["pf_prim"]))

        # FREQ
        dev = abs(freq - 50.0)
        if dev > cfg["warn"]["freq_dev"]:
            self.db_f.hit(ts)
            if dev > cfg["alarm"]["freq_dev"] and self.db_f.is_triggered():
                evts.append(self._evt("ALARM","FREQ_DRIFT","Ecart de fréquence élevé", dev, cfg["alarm"]["freq_dev"]))
            elif self.db_f.is_triggered():
                evts.append(self._evt("WARN","FREQ_DRIFT","Ecart fréquence", dev, cfg["warn"]["freq_dev"]))

        # VOLTAGE dips/surges (instantanés)
        vdevp = abs(vp - v_nom_p) / v_nom_p
        vdevs = abs(vs - v_nom_s) / v_nom_s
        if vdevp > cfg["alarm"]["v_dev_pct"]:
            evts.append(self._evt("ALARM","VOLTAGE_SURGE" if vp>v_nom_p else "VOLTAGE_DIP",
                                  "Variation tension primaire > 15%", vdevp, cfg["alarm"]["v_dev_pct"]))
        elif vdevp > cfg["warn"]["v_dev_pct"]:
            evts.append(self._evt("WARN","VOLTAGE_SURGE" if vp>v_nom_p else "VOLTAGE_DIP",
                                  "Variation tension primaire > 10%", vdevp, cfg["warn"]["v_dev_pct"]))

        if vdevs > cfg["alarm"]["v_dev_pct"]:
            evts.append(self._evt("ALARM","VOLTAGE_SEC_DEV", "Variation tension secondaire > 15%", vdevs, cfg["alarm"]["v_dev_pct"]))
        elif vdevs > cfg["warn"]["v_dev_pct"]:
            evts.append(self._evt("WARN","VOLTAGE_SEC_DEV", "Variation tension secondaire > 10%", vdevs, cfg["warn"]["v_dev_pct"]))

        return evts
