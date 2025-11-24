# core/transformer/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math, time

# -------------------------------------------------------------------
# Paramètres de plaque + seuils + thermique + ventilation (central)
# -------------------------------------------------------------------
@dataclass
class TransformerParams:
    # Plaque (unités internes SI)
    S_rated: float = 25e6      # VA (25 MVA)
    V1n: float = 220e3         # V (220 kV)
    V2n: float = 20e3          # V (20 kV)
    f0: float = 50.0           # Hz
    vector_group: str = "Dyn5"

    # Seuils réalistes (sur hot-spot)
    TEMP_WARN: float = 95.0    # °C
    TEMP_ALARM: float = 110.0  # °C
    PF_WARN: float = 0.90
    I1_WARN_MULT: float = 1.20
    OVERLOAD_MULT: float = 1.10
    MU_MIN: float = 0.010      # Pa·s (info: huile trop fluide)

    # Modèle thermique (1er ordre)
    tau_hotspot_s: float = 900.0
    tau_oil_s: float = 1800.0
    amb_filter_tau_s: float = 300.0
    theta_rise_nom_C: float = 55.0     # élévation d’huile à charge nominale
    n_thermal: float = 1.6             # ∼ IEC (0.8–1.6)

    # Viscosité (Arrhenius)
    mu_ref_Pa_s: float = 0.02
    T_ref_C: float = 40.0
    mu_A: float = -0.033

    # Pertes (ordre de grandeur)
    cu_loss_ratio_at_rated: float = 0.012   # ~1.2% Sn à In
    fe_loss_ratio: float = 0.004            # ~0.4% Sn à vide

    # Ventilation (stades + hystérésis) — agit sur l’évacuation thermique
    fan1_on_C: float = 85.0
    fan1_off_C: float = 80.0
    fan2_on_C: float = 95.0
    fan2_off_C: float = 90.0
    k_fan1: float = 0.6     # gain de refroidissement stage 1
    k_fan2: float = 1.0     # gain de refroidissement stage 2

# Commandes venant de l’UI
@dataclass
class Controls:
    load_pct: float = 50.0
    pf_set: float = 0.95
    Tamb: float = 30.0
    freq_hz: float = 50.0
    force_temp_high: bool = False
    force_overcurrent: bool = False
    force_pf_low: bool = False
    force_dip: bool = False
    # Ventilation
    fan_mode: str = "AUTO"   # "AUTO" / "MAN"
    fan1_force: bool = False
    fan2_force: bool = False

# États internes
@dataclass
class TransformerState:
    t_last: float = field(default_factory=time.time)
    T_amb_f: float = 30.0
    T_oil: float = 35.0
    T_hotspot: float = 40.0
    life_consumption_h: float = 0.0
    energy_out_MWh: float = 0.0
    energy_loss_MWh: float = 0.0
    # Ventilation
    fan1_on: bool = False
    fan2_on: bool = False

class TransformerEngine:
    """
    Moteur unique: électrique -> pertes -> thermique + ventilation (auto/manuelle).
    Génère aussi des événements (WARN/ALARM/INFO) consommés par la page Temps réel.
    """
    def __init__(self, p: TransformerParams | None = None):
        self.p = p or TransformerParams()
        self.c = Controls()
        self.s = TransformerState()

    # -------- API publique --------
    def set_controls(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.c, k):
                setattr(self.c, k, v)

    def set_params_from_record(self, rec: Dict):
        """
        Prend un enregistrement CRUD (MVA/kV/Hz) et met à jour les unités internes SI.
        """
        p = self.p
        try:
            p.S_rated = float(rec.get("rated_mva", 25.0)) * 1e6
            p.V1n     = float(rec.get("V1n_kV", 220.0)) * 1e3
            p.V2n     = float(rec.get("V2n_kV", 20.0))  * 1e3
            p.f0      = float(rec.get("f_nominal", 50.0))
            vg = str(rec.get("vector_group", "")).strip()
            if vg: p.vector_group = vg
        except Exception:
            pass

    def reset(self):
        self.s = TransformerState()

    # -------- Step principal --------
    def step(self, dt: float) -> Tuple[Dict, List[Dict]]:  # (mesure, événements)
        now = time.time()
        p, c, s = self.p, self.c, self.s
        events: List[Dict] = []

        # 0) Garde-fous
        S_rated = max(1.0, p.S_rated)
        V1n = max(1.0, p.V1n); V2n = max(1.0, p.V2n)

        # 1) Ambiance filtrée (1er ordre)
        s.T_amb_f += self._alpha(dt, p.amb_filter_tau_s) * (c.Tamb - s.T_amb_f)

        # 2) Électrique
        load = max(0.0, min(2.0, c.load_pct/100.0))
        PF = max(0.1, min(1.0, c.pf_set))
        if c.force_pf_low:
            PF = min(PF, p.PF_WARN * 0.95)

        V1 = V1n * (0.96 if c.force_dip else 1.0)
        V2 = V2n * (0.99 if c.force_dip else 1.0)

        S = load * S_rated
        P_out = S * PF
        I1 = S / (math.sqrt(3) * V1)
        I2 = S / (math.sqrt(3) * V2)
        In1 = S_rated / (math.sqrt(3) * V1n)
        freq = c.freq_hz

        # 3) Pertes
        P_fe = p.fe_loss_ratio * S_rated
        P_cu_rated = p.cu_loss_ratio_at_rated * S_rated
        P_cu = P_cu_rated * (I1 / max(In1, 1e-9))**2
        P_total_loss = P_fe + P_cu
        P_net = max(0.0, P_out - P_total_loss)

        # 4) Ventilation (hystérésis + mode)
        T_preview = s.T_oil + 0.15 * p.theta_rise_nom_C * (load ** p.n_thermal)  # preview hot-spot
        self._ventilation_step(T_preview)

        # 5) Thermique (1er ordre) — l’évacuation est améliorée par les ventilos
        #    On réduit l’élévation d’huile en régime permanent en divisant par le gain "cooling"
        cooling_gain = 1.0 + (p.k_fan1 if s.fan1_on else 0.0) + (p.k_fan2 if s.fan2_on else 0.0)
        k = (S / S_rated) ** p.n_thermal
        dT_oil_target = (p.theta_rise_nom_C / max(1e-3, cooling_gain)) * k
        T_oil_target = s.T_amb_f + dT_oil_target
        T_hs_target  = T_oil_target + 0.15 * dT_oil_target  # gradient hot-spot

        s.T_oil     += self._alpha(dt, p.tau_oil_s)    * (T_oil_target - s.T_oil)
        s.T_hotspot += self._alpha(dt, p.tau_hotspot_s) * (T_hs_target  - s.T_hotspot)

        if c.force_temp_high:
            s.T_hotspot = max(s.T_hotspot, p.TEMP_ALARM + 2.0)

        # 6) Viscosité
        mu = p.mu_ref_Pa_s * math.exp(p.mu_A * (s.T_oil - p.T_ref_C))
        mu = max(mu, 5e-4)

        # 7) Vieillissement (IEC 60076-7 approchée)
        FAA = 2.0 ** ((s.T_hotspot - 98.0) / 6.0)   # facteur d'accélération
        FAA = max(0.0, min(128.0, FAA))
        s.life_consumption_h += (dt / 3600.0) * FAA

        # 8) Énergies
        s.energy_out_MWh  += (P_net / 1e6) * (dt / 3600.0)
        s.energy_loss_MWh += (P_total_loss / 1e6) * (dt / 3600.0)

        # 9) Événements
        if (S > p.OVERLOAD_MULT * S_rated) or c.force_overcurrent:
            events.append(self._evt("WARN", "OVERLOAD", f"S={S/S_rated:.2f} ×Sn", S, p.OVERLOAD_MULT*S_rated))
        if (I1 > p.I1_WARN_MULT * In1) or c.force_overcurrent:
            events.append(self._evt("WARN", "OVERCURRENT", f"I1={I1/In1:.2f} ×In", I1, p.I1_WARN_MULT*In1))
        if PF < p.PF_WARN:
            events.append(self._evt("WARN", "PF_LOW", f"PF={PF:.2f} < {p.PF_WARN:.2f}", PF, p.PF_WARN))
        if V1 < 0.95 * V1n or c.force_dip:
            events.append(self._evt("INFO", "V_DIP", f"V1={V1/V1n:.3f} ·Vn", V1, 0.95*V1n))

        if s.T_hotspot > p.TEMP_ALARM:
            events.append(self._evt("ALARM", "TEMP_HIGH", f"θhs={s.T_hotspot:.1f}°C > {p.TEMP_ALARM:.1f}°C", s.T_hotspot, p.TEMP_ALARM))
        elif s.T_hotspot > p.TEMP_WARN:
            events.append(self._evt("WARN", "TEMP_WARM", f"θhs={s.T_hotspot:.1f}°C > {p.TEMP_WARN:.1f}°C", s.T_hotspot, p.TEMP_WARN))

        if mu < p.MU_MIN:
            events.append(self._evt("INFO", "OIL_THIN", f"μ={mu:.3f} Pa·s < {p.MU_MIN:.3f}", mu, p.MU_MIN))

        # 10) Mesures retournées
        m = {
            "ts": now,
            "v_prim_rms": V1, "i_prim_rms": I1,
            "v_sec": V2,      "i_sec": I2,
            "p_sec": P_out,   # active brute
            "p_loss": P_total_loss,
            "p_net": P_net,
            "t_core": s.T_hotspot,
            "t_oil": s.T_oil,
            "mu_oil": mu,
            "pf_prim": PF,
            "freq": freq,
            "faa": FAA,
            "life_h": s.life_consumption_h,
            "e_out_MWh": s.energy_out_MWh,
            "e_loss_MWh": s.energy_loss_MWh,
            # Ventilation pour l'UI
            "fan_mode": self.c.fan_mode,
            "fan1": int(self.s.fan1_on),
            "fan2": int(self.s.fan2_on),
        }
        self.s.t_last = now
        return m, events

    # -------- internes --------
    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        tau = max(1e-6, float(tau)); dt = max(0.0, float(dt))
        return 1.0 - math.exp(-dt / tau)

    @staticmethod
    def _evt(level: str, code: str, msg: str, value=None, threshold=None) -> Dict:
        return {"ts": time.time(), "level": level, "code": code, "msg": msg,
                "value": value if value is not None else "", "threshold": threshold if threshold is not None else ""}

    def _ventilation_step(self, T_hs_preview: float) -> None:
        """AUTO: hystérésis; MAN: forcé. Met à jour s.fan1_on/s.fan2_on."""
        p, c, s = self.p, self.c, self.s
        if (c.fan_mode or "").upper() == "AUTO":
            # Stage 1
            if (not s.fan1_on) and (T_hs_preview >= p.fan1_on_C):
                s.fan1_on = True
            if s.fan1_on and (T_hs_preview <= p.fan1_off_C):
                s.fan1_on = False
            # Stage 2
            if (not s.fan2_on) and (T_hs_preview >= p.fan2_on_C):
                s.fan2_on = True
            if s.fan2_on and (T_hs_preview <= p.fan2_off_C):
                s.fan2_on = False
        else:
            s.fan1_on = bool(c.fan1_force)
            s.fan2_on = bool(c.fan2_force)
