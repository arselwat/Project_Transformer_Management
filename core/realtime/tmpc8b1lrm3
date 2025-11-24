# core/realtime/sim_transfo.py
from __future__ import annotations
import math, time, random

class TransformerParams:
    def __init__(self,
                 S_rated=25000.0, V1n=220.0, V2n=20.0, f0=50.0,
                 vector_group="",
                 # Seuils/limites (inchangés – UI existante)
                 TEMP_WARN=60.0, TEMP_ALARM=70.0,
                 PF_WARN=0.85, I1_WARN_MULT=1.2, OVERLOAD_MULT=1.0, MU_MIN=0.010,
                 # Constantes thermiques (nouveaux défauts raisonnables)
                 dtheta_to_u=55.0, dtheta_h_u=20.0, tau_to=900.0, tau_h=120.0,
                 n_oil=0.8, m_hot=0.8,
                 # Viscosité huile (modèle type Arrhenius)
                 mu_ref=0.02, T_ref_C=40.0, A_visc=1800.0):
        self.S_rated=float(S_rated)
        self.V1n=float(V1n); self.V2n=float(V2n); self.f0=float(f0)
        self.vector_group=vector_group

        self.TEMP_WARN=float(TEMP_WARN); self.TEMP_ALARM=float(TEMP_ALARM)
        self.PF_WARN=float(PF_WARN); self.I1_WARN_MULT=float(I1_WARN_MULT)
        self.OVERLOAD_MULT=float(OVERLOAD_MULT); self.MU_MIN=float(MU_MIN)

        self.dtheta_to_u=float(dtheta_to_u); self.dtheta_h_u=float(dtheta_h_u)
        self.tau_to=float(tau_to); self.tau_h=float(tau_h)
        self.n_oil=float(n_oil); self.m_hot=float(m_hot)

        self.mu_ref=float(mu_ref); self.T_ref_C=float(T_ref_C); self.A_visc=float(A_visc)

class TransformerSim:
    def __init__(self, params: TransformerParams):
        self.p=params
        self.reset()

    def reset(self):
        self._Tamb=28.0
        self._pf=0.95
        self._freq=self.p.f0
        self._load_pct=50.0
        self._force_temp=False
        self._force_i=False
        self._force_pf=False
        self._force_dip=False

        # États thermiques
        self._dtheta_to=10.0   # top-oil rise initial
        self._dtheta_h=5.0     # hot-spot rise initial

        # Courants nominaux (approx 1φ pour compat – mêmes unités)
        self._In  = max(1e-3, self.p.S_rated / max(1.0, self.p.V1n))
        self._I2n = max(1e-3, self.p.S_rated / max(1.0, self.p.V2n))

        self._last_ts = time.time()

    def set_controls(self, load_pct: float, pf_set: float, Tamb: float, freq_hz: float,
                     force_temp_high=False, force_overcurrent=False, force_pf_low=False, force_dip=False):
        self._load_pct=float(load_pct)
        self._pf=float(pf_set)
        self._Tamb=float(Tamb)
        self._freq=float(freq_hz)
        self._force_temp=bool(force_temp_high)
        self._force_i=bool(force_overcurrent)
        self._force_pf=bool(force_pf_low)
        self._force_dip=bool(force_dip)

    # --- utilitaires internes ---
    def _clip(self, v, lo, hi):
        try:
            return float(min(max(v, lo), hi))
        except Exception:
            return float(lo)

    def _oil_viscosity(self, T_C: float) -> float:
        # μ(T) = μref * exp(A*(1/T - 1/Tref)), clampé dans un range réaliste
        T_K = max(1.0, T_C + 273.15)
        Tref_K = max(1.0, self.p.T_ref_C + 273.15)
        mu = self.p.mu_ref * math.exp(self.p.A_visc * (1.0/T_K - 1.0/Tref_K))
        return self._clip(mu, self.p.MU_MIN*0.5, 10.0)

    def step(self, dt: float):
        # dt borné pour stabilité
        dt = float(max(0.001, min(dt, 5.0)))
        ts = time.time()

        # Facteur de charge et PF (PF forcé si demandé)
        K = max(0.0, self._load_pct) / 100.0
        pf = self._pf
        if self._force_pf:
            pf = min(pf, self.p.PF_WARN - 0.05)

        # Tensions avec un léger bruit (+ dip si forcé)
        V2 = self.p.V2n * (0.98 + 0.04*random.random())
        V1 = self.p.V1n * (0.98 + 0.04*random.random())
        if self._force_dip:
            V2 *= 0.80
            V1 *= 0.90

        # Courants (approx 1φ – garde compat noms/units)
        I2 = K * self._I2n * (1.0 + 0.02*(random.random()-0.5))
        I1 = K * self._In  * (1.0 + 0.02*(random.random()-0.5))
        S2 = V2 * abs(I2)
        P2 = S2 * max(0.0, min(1.0, pf))

        # Pertes très simples
        P_cu = (K**2) * 0.02 * self.p.S_rated   # cuivre ~K^2
        P_fe = 0.01 * self.p.S_rated            # fer ~constante
        _ = P2 + P_cu + P_fe                    # P_tot si besoin plus tard

        # Thermique simplifié (IEC/IEEE) : top-oil & hot-spot
        dtheta_to_u = self.p.dtheta_to_u
        dtheta_h_u  = self.p.dtheta_h_u
        n = self.p.n_oil; m = self.p.m_hot

        target_to = dtheta_to_u * (max(0.2, K)**n)
        target_h  = dtheta_h_u  * (max(0.2, K)**m)

        a_to = dt / max(1.0, self.p.tau_to)
        a_h  = dt / max(1.0, self.p.tau_h)

        self._dtheta_to += a_to * (target_to - self._dtheta_to)
        self._dtheta_h  += a_h  * (target_h  - self._dtheta_h)

        if self._force_temp:
            # pousse au-dessus du WARN/ALARM sans casser la dynamique
            self._dtheta_to = max(self._dtheta_to, self.p.TEMP_ALARM - self._Tamb - 1.0)

        t_core = self._Tamb + self._dtheta_to + self._dtheta_h
        mu = self._oil_viscosity(t_core)

        # Fréquence avec petite dérive bornée
        freq = self._clip(self._freq + 0.02*(random.random()-0.5), self.p.f0-1.0, self.p.f0+1.0)

        # Clamps de sécurité (anti overflow/NaN)
        V2 = self._clip(V2, 0.0, 1e6)
        I2 = self._clip(I2, 0.0, 1e6)
        P2 = self._clip(P2, 0.0, 1e9)
        V1 = self._clip(V1, 0.0, 1e6)
        I1 = self._clip(I1, 0.0, 1e6)
        t_core = self._clip(t_core, -40.0, 200.0)

        # Événements
        evts=[]
        def emit(level, code, msg, value=None, threshold=None):
            evts.append({
                "level": level, "code": code, "msg": msg,
                "value": value, "threshold": threshold
            })

        if t_core >= self.p.TEMP_ALARM:
            emit("ALARM","TEMP_HIGH","Température noyau haute", t_core, self.p.TEMP_ALARM)
        elif t_core >= self.p.TEMP_WARN:
            emit("WARN","TEMP_HIGH","Température noyau élevée", t_core, self.p.TEMP_WARN)

        if pf < self.p.PF_WARN:
            emit("WARN","PF_LOW","Facteur de puissance bas", pf, self.p.PF_WARN)

        if I1 > self.p.I1_WARN_MULT * self._In or self._force_i:
            emit("WARN","OVERCURRENT","Courant primaire élevé",
                 I1, self.p.I1_WARN_MULT*self._In)

        if P2 > self.p.OVERLOAD_MULT * self.p.S_rated * max(0.7, pf):
            emit("WARN","OVERLOAD","Surcharge apparente",
                 P2, self.p.OVERLOAD_MULT*self.p.S_rated)

        # Mesure renvoyée (mêmes clés que ton UI)
        m = {
            "ts": ts,
            "v_sec": V2, "i_sec": I2, "p_sec": P2,
            "v_prim_rms": V1, "i_prim_rms": I1,
            "pf_prim": pf, "freq": freq,
            "t_core": t_core, "mu_oil": mu,
            "status": "OK" if not evts else ("ALARM" if any(e["level"]=="ALARM" for e in evts) else "WARN")
        }
        return m, evts
