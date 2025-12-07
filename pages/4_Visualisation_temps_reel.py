from __future__ import annotations
import time, threading, os, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Bootstrapping import ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.transformer.engine import TransformerEngine, TransformerParams
from core.transformer.store import list_transformers, get_transformer

# Alerting centralisée (email/WhatsApp) — déjà gérée par ta page 7_Parametres_Alertes
try:
    from core.notify.rt_alerts import notify_event  # (evt: dict) -> dict
except Exception:
    def notify_event(e: dict):  # fallback no-op
        return

# ---------------- Layout / Paths ----------------

st.set_page_config(page_title="Temps réel — Transfo", page_icon="📡", layout="wide")
st.title("📡 Visualisation Temps Réel — Transformateur")

BASE_DIR = PROJECT_ROOT
DATA_DIR = Path(os.environ.get("FS_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVT_CSV  = DATA_DIR / "realtime_events.csv"
MEAS_CSV = DATA_DIR / "realtime_log.csv"

# ---------------- Etat session ----------------

if "engines" not in st.session_state:
    st.session_state.engines = {}      # code -> TransformerEngine
if "rt_data_map" not in st.session_state:
    st.session_state.rt_data_map = {}  # code -> list[measure dict]
if "events_map" not in st.session_state:
    st.session_state.events_map = {}   # code -> list[event dict]
if "running_map" not in st.session_state:
    st.session_state.running_map = {}  # code -> bool
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.monotonic()

# ---------------- Helpers généraux ----------------

def _safe_epoch_to_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    sec = s.abs() <= 9_000_000_000
    ms  = (~sec) & (s.abs() <= 9_000_000_000_000)
    if sec.any():
        out.loc[sec] = pd.to_datetime(s[sec], unit="s", errors="coerce")
    if ms.any():
        out.loc[ms]  = pd.to_datetime(s[ms], unit="ms", errors="coerce")
    return out

def _fmt(v, fmt="{}"):
    try:
        return fmt.format(float(v))
    except Exception:
        return "—"

def _fmt_scaled(v, scale, fmt):
    try:
        return fmt.format(float(v)/float(scale))
    except Exception:
        return "—"

def _append_events_to_csv(events: list[dict]) -> None:
    if not events:
        return
    EVT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # ajout des colonnes site/equipment/equipment_code dans le CSV             # <<< MODIF
    header = ["ts","level","code","msg","value","threshold","site","equipment","equipment_code"]  # <<< MODIF
    write_header = not EVT_CSV.exists() or EVT_CSV.stat().st_size == 0
    try:
        with open(EVT_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for e in events:
                row = {k: e.get(k, "") for k in header}                        # <<< MODIF
                w.writerow(row)
    except Exception as ex:
        st.warning(f"Journal d’événements non écrit: {ex}")

# ---------------- Sélection équipements ----------------

rows = list_transformers(include_retired=False)
if not rows:
    st.warning("Aucun transformateur actif. Va d’abord dans la page **Transformateurs** pour en créer un (MVA/kV).")
    st.stop()

codes = [r["equipment_code"] for r in rows if r.get("equipment_code")]
q = st.query_params
pre = str(q.get("tfm","")) if q else ""
default_index = codes.index(pre) if pre and pre in codes else 0

mode_multi = st.toggle("Activer le mode multi-transformateurs (max 3)", value=False, key="rt_multi")

# =====================================================================
# ============================== MONO ================================
# =====================================================================
if not mode_multi:
    code = st.selectbox("Transformateur", options=codes, index=default_index, key="rt_sel_one")
    sel_rec = get_transformer(code) or {}

    # métadonnées pour les alertes e-mail                                   # <<< MODIF
    site_name    = sel_rec.get("site") or ""                                # <<< MODIF
    equip_name   = sel_rec.get("name") or code                              # <<< MODIF
    equip_code   = code                                                     # <<< MODIF

    # Engine unique par code
    eng = st.session_state.engines.get(code) or TransformerEngine(TransformerParams())
    eng.set_params_from_record(sel_rec)
    st.session_state.engines[code] = eng
    st.session_state.rt_data_map.setdefault(code, [])
    st.session_state.events_map.setdefault(code, [])
    st.session_state.running_map.setdefault(code, False)

    # --------- Plaque signalétique ---------
    with st.container(border=True):
        st.subheader("Plaque signalétique")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.caption("Nom / Modèle"); st.write(sel_rec.get("name") or "—")
            st.caption("Site"); st.write(sel_rec.get("site") or "—")
            st.caption("Mise en service"); st.write(sel_rec.get("commissioned_on") or "—")
        with c2:
            st.caption("Puissance (MVA)"); st.write(f"{float(sel_rec.get('rated_mva',0) or 0):.2f}")
            st.caption("V1n (kV)"); st.write(f"{float(sel_rec.get('V1n_kV',0) or 0):.3f}")
            st.caption("V2n (kV)"); st.write(f"{float(sel_rec.get('V2n_kV',0) or 0):.3f}")
        with c3:
            st.caption("Fréquence (Hz)"); st.write(f"{float(sel_rec.get('f_nominal',0) or 0):.0f}")
            st.caption("Groupe vectoriel"); st.write(sel_rec.get("vector_group") or "—")
            st.caption("Code"); st.write(code)

    # --------- Contrôles top ---------
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Contrôle")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶️ Démarrer", type="primary", key=f"start_{code}", disabled=st.session_state.running_map[code]):
                st.session_state.running_map[code] = True
        with c2:
            if st.button("⏹️ Arrêter", key=f"stop_{code}", disabled=not st.session_state.running_map[code]):
                st.session_state.running_map[code] = False
        with c3:
            if st.button("♻️ Reset thermique (→ Tamb)", key=f"reset_{code}"):
                eng.s.T_oil = eng.c.Tamb
                eng.s.T_hotspot = eng.c.Tamb
                eng.s.fan1_on = eng.s.fan2_on = False
                st.info("Thermique ramenée à l’ambiante.")

        st.caption("Tick avant affichage → latence minimale.")

    with right:
        st.subheader("Cadence & fenêtre")
        hz = st.slider("Fréquence d’update (Hz)", 1, 10, 3, 1, key="rt_hz_one")
        max_pts = st.number_input("Points gardés", min_value=60, max_value=2000, value=300, step=60, key="rt_maxpts_one")

    # --------- Panneau latéral: Entrées + Ventilation + Seuils ---------
    with st.sidebar:
        st.markdown("### ⚙️ Entrées")
        load_pct = st.slider("Charge (%)", 0, 200, 50, 1, key=f"ld_{code}")
        pf_prim  = st.slider("Facteur de puissance PF (%)", 50, 100, 95, 1, key=f"pf_{code}")/100.0
        Tamb     = st.slider("Température ambiante (°C)", 0, 60, 30, 1, key=f"ta_{code}")
        freq_hz  = st.slider("Fréquence (Hz)", 49, 51, 50, 1, key=f"fq_{code}")

        st.markdown("### 🌀 Ventilation")
        fan_mode = st.radio("Mode", ["AUTO","MAN"], horizontal=True, key=f"fanmode_{code}")
        colf1, colf2 = st.columns(2)
        with colf1:
            fan1 = st.toggle("Stage 1", value=False, disabled=(fan_mode=="AUTO"), key=f"f1_{code}")
        with colf2:
            fan2 = st.toggle("Stage 2", value=False, disabled=(fan_mode=="AUTO"), key=f"f2_{code}")

        with st.expander("🔧 Seuils & tests avancés", expanded=False):
            p = eng.p
            c1s, c2s = st.columns(2)
            with c1s:
                p.TEMP_WARN  = st.number_input("TEMP_WARN (°C)", 60.0, 140.0, float(p.TEMP_WARN), 1.0, key=f"tw_{code}")
                p.PF_WARN    = st.number_input("PF_WARN", 0.5, 1.0, float(p.PF_WARN), 0.01, key=f"pfw_{code}", format="%.2f")
                p.I1_WARN_MULT = st.number_input("I1_WARN_MULT (×In)", 1.0, 3.0, float(p.I1_WARN_MULT), 0.05, key=f"i1_{code}")
            with c2s:
                p.TEMP_ALARM = st.number_input("TEMP_ALARM (°C)", 80.0, 160.0, float(p.TEMP_ALARM), 1.0, key=f"ta_{code}_al")
                p.OVERLOAD_MULT = st.number_input("OVERLOAD_MULT (×Sn)", 0.8, 2.0, float(p.OVERLOAD_MULT), 0.05, key=f"ov_{code}")
                p.MU_MIN = st.number_input("Viscosité min μ (Pa·s)", 0.001, 0.2, float(p.MU_MIN), 0.001, key=f"mu_{code}", format="%.3f")

            st.markdown("**Actions (tests)**")
            a1, a2 = st.columns(2)
            with a1:
                f_temp = st.checkbox("Forcer TEMP_HIGH", value=False, key=f"ft_{code}")
                f_i    = st.checkbox("Forcer OVERCURRENT", value=False, key=f"fi_{code}")
            with a2:
                f_pf   = st.checkbox("Forcer PF LOW", value=False, key=f"fp_{code}")
                f_dip  = st.checkbox("Forcer DIP VOLT", value=False, key=f"fd_{code}")

    # -------- Tick mono --------

    def _tick_one():
        if not st.session_state.running_map[code]:
            return

        base_load = float(load_pct)
        base_pf   = float(pf_prim)
        base_Tamb = float(Tamb)
        base_freq = float(freq_hz)

        base_load = np.clip(base_load * (1.0 + 0.03*np.random.randn()), 0.0, 200.0)
        base_pf   = np.clip(base_pf   + 0.01*np.random.randn(), 0.5, 1.0)
        base_Tamb = np.clip(base_Tamb + 1.0*np.random.randn(), 0.0, 60.0)
        base_freq = np.clip(base_freq + 0.02*np.random.randn(), 49.0, 51.0)

        eng.set_controls(
            load_pct=float(base_load),
            pf_set=float(base_pf),
            Tamb=float(base_Tamb),
            freq_hz=float(base_freq),
            force_temp_high=bool(st.session_state.get(f"ft_{code}", False)),
            force_overcurrent=bool(st.session_state.get(f"fi_{code}", False)),
            force_pf_low=bool(st.session_state.get(f"fp_{code}", False)),
            force_dip=bool(st.session_state.get(f"fd_{code}", False)),
            fan_mode=fan_mode,
            fan1_force=bool(fan1),
            fan2_force=bool(fan2),
        )

        target_dt = 1.0 / max(int(hz), 1)
        now_m = time.monotonic()
        if (now_m - st.session_state.last_tick) >= target_dt * 0.95:
            m, E = eng.step(target_dt)
            buf = st.session_state.rt_data_map[code]
            buf.append(m)
            keep = int(max_pts)
            if len(buf) > keep:
                st.session_state.rt_data_map[code] = buf[-keep:]

            if E:
                evbuf = st.session_state.events_map[code]
                enriched = []                                                # <<< MODIF
                for e in E:                                                  # <<< MODIF
                    e2 = dict(e)                                             # <<< MODIF
                    e2.setdefault("equipment", equip_name)                   # <<< MODIF
                    e2.setdefault("equipment_code", equip_code)             # <<< MODIF
                    e2.setdefault("site", site_name)                         # <<< MODIF
                    enriched.append(e2)                                      # <<< MODIF
                evbuf.extend(enriched)                                       # <<< MODIF
                st.session_state.events_map[code] = evbuf[-1000:]

                _append_events_to_csv(enriched)                              # <<< MODIF
                for e in enriched:                                           # <<< MODIF
                    try:
                        notify_event(e)                                      # <<< MODIF
                    except Exception:
                        pass

                for e in enriched:                                           # <<< MODIF
                    if e.get("level") == "ALARM":                            # <<< MODIF
                        st.error(f"🔴 {e.get('code')}: {e.get('msg')}")      # <<< MODIF
                    elif e.get("level") == "WARN":                           # <<< MODIF
                        st.warning(f"🟡 {e.get('code')}: {e.get('msg')}")    # <<< MODIF

            st.session_state.last_tick = now_m

    _tick_one()

    # -------- Export / journal --------
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("💾 Exporter mesures CSV", key="exp_meas_one"):
            df = pd.DataFrame(st.session_state.rt_data_map[code])
            if df.empty:
                st.warning("Aucune donnée.")
            else:
                MEAS_CSV.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(MEAS_CSV, index=False)
                st.success(f"Export → {MEAS_CSV}")
    with cB:
        if st.button("🧾 Ouvrir le journal d’événements", key="see_evt_one"):
            if EVT_CSV.exists():
                st.code(EVT_CSV.read_text(encoding="utf-8")[-4000:], language="text")
            else:
                st.info("Aucun évènement pour l’instant.")
    with cC:
        st.caption("Astuce: laisse tourner et reviens — tout est journalisé en CSV.")

    st.divider()

    # -------- Affichage mono --------
    df_view = pd.DataFrame(st.session_state.rt_data_map[code])
    if df_view.empty:
        st.info("En attente de données…")
        st.stop()

    df_view = df_view.sort_values("ts")
    df_view["t"] = _safe_epoch_to_datetime(df_view["ts"])
    df_view.loc[df_view["t"].isna(), "t"] = pd.Timestamp.now()

    df_ht = df_view.copy()
    if "v_sec" in df_ht:
        df_ht["v_sec_kV"]   = pd.to_numeric(df_ht["v_sec"], errors="coerce") / 1e3
    if "i_sec" in df_ht:
        df_ht["i_sec_kA"]   = pd.to_numeric(df_ht["i_sec"], errors="coerce") / 1e3
    if "p_sec" in df_ht:
        df_ht["p_sec_MW"]   = pd.to_numeric(df_ht["p_sec"], errors="coerce") / 1e6
    if "p_net" in df_ht:
        df_ht["p_net_MW"]  = pd.to_numeric(df_ht["p_net"], errors="coerce") / 1e6
    if "p_loss" in df_ht:
        df_ht["p_loss_MW"] = pd.to_numeric(df_ht["p_loss"], errors="coerce") / 1e6
    if "v_prim_rms" in df_ht:
        df_ht["v_prim_kV"]  = pd.to_numeric(df_ht["v_prim_rms"], errors="coerce") / 1e3
    if "i_prim_rms" in df_ht:
        df_ht["i_prim_kA"]  = pd.to_numeric(df_ht["i_prim_rms"], errors="coerce") / 1e3

    last = df_view.iloc[-1]
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric("V sec (kV)", _fmt_scaled(last.get('v_sec'), 1e3, "{:.3f}"))
    with m2: st.metric("I sec (kA)", _fmt_scaled(last.get('i_sec'), 1e3, "{:.3f}"))
    with m3: st.metric("P nette (MW)", _fmt_scaled(last.get('p_net'), 1e6, "{:.3f}"))
    with m4: st.metric("Hot-spot (°C)", _fmt(last.get('t_core'), "{:.1f}"))
    with m5: st.metric("PF prim", _fmt(last.get('pf_prim'), "{:.3f}"))
    with m6:
        fans = f"{'A' if last.get('fan_mode')=='AUTO' else 'M'}/{int(last.get('fan1'))+int(last.get('fan2'))}"
        st.metric("Ventilation", fans)

    tabs = st.tabs([
        "V/I secondaire (kV/kA)",
        "Puissance (MW)",
        "Prim. Vrms/Irms (kV/kA)",
        "Température & μ",
        "PF & Fréquence",
        "Analyse & KPI (pro)",
        "Journal d’événements",
    ])

    with tabs[0]:
        ycols = [c for c in ["v_sec_kV","i_sec_kA"] if c in df_ht.columns]
        if ycols:
            st.plotly_chart(px.line(df_ht, x="t", y=ycols, title="Secondaire — Tension (kV) & Courant (kA)"), use_container_width=True)

    with tabs[1]:
        ycols = [c for c in ["p_sec_MW","p_net_MW","p_loss_MW"] if c in df_ht.columns]
        if ycols:
            st.plotly_chart(px.line(df_ht, x="t", y=ycols, title="Puissance: brute/nettes/pertes (MW)"), use_container_width=True)

    with tabs[2]:
        ycols = [c for c in ["v_prim_kV","i_prim_kA"] if c in df_ht.columns]
        if ycols:
            st.plotly_chart(px.line(df_ht, x="t", y=ycols, title="Primaire — Vrms (kV) & Irms (kA)"), use_container_width=True)

    with tabs[3]:
        ycols = [c for c in ["t_core","t_oil","mu_oil"] if c in df_view.columns]
        st.plotly_chart(px.line(df_view, x="t", y=ycols, title="Températures & Viscosité"), use_container_width=True)

    with tabs[4]:
        ycols = [c for c in ["pf_prim","freq"] if c in df_view.columns]
        st.plotly_chart(px.line(df_view, x="t", y=ycols, title="Facteur de puissance & Fréquence (Hz)"), use_container_width=True)

    with tabs[5]:
        p = eng.p
        S_rated_MVA = p.S_rated / 1e6
        dfA = df_view.copy()
        for c in ["p_sec","p_net","p_loss","pf_prim","t_core","faa","e_out_MWh","e_loss_MWh","life_h"]:
            if c in dfA:
                dfA[c] = pd.to_numeric(dfA[c], errors="coerce")

        e_out = float(dfA["e_out_MWh"].iloc[-1]) if "e_out_MWh" in dfA else np.nan
        e_loss = float(dfA["e_loss_MWh"].iloc[-1]) if "e_loss_MWh" in dfA else np.nan

        if "p_net" in dfA and "pf_prim" in dfA and S_rated_MVA > 0:
            dfA["p_net_MW"] = dfA["p_net"] / 1e6
            with np.errstate(divide="ignore", invalid="ignore"):
                dfA["S_MVA_est"] = dfA["p_net_MW"] / dfA["pf_prim"].clip(lower=0.1)
                dfA["load_pct"]  = (dfA["S_MVA_est"] / S_rated_MVA) * 100.0

        if "p_net" in dfA and "p_loss" in dfA:
            num = dfA["p_net"].clip(lower=0.0)
            den = (dfA["p_net"] + dfA["p_loss"]).replace(0, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                dfA["eta"] = num / den
        else:
            dfA["eta"] = np.nan

        eta_last = float(dfA["eta"].iloc[-1]) if len(dfA) else np.nan
        eta_mean = float(dfA["eta"].mean()) if len(dfA) else np.nan

        def _dur(cond: pd.Series) -> float:
            if cond is None or cond.empty:
                return 0.0
            dt = dfA["t"].diff().dt.total_seconds().fillna(0.0)
            return float((dt * cond.fillna(False).astype(float)).sum())

        TEMP_WARN = p.TEMP_WARN; TEMP_ALARM = p.TEMP_ALARM
        dur_warn = _dur(dfA["t_core"] > TEMP_WARN) if "t_core" in dfA else 0.0
        dur_alarm = _dur(dfA["t_core"] > TEMP_ALARM) if "t_core" in dfA else 0.0

        life_h = float(dfA["life_h"].iloc[-1]) if "life_h" in dfA else np.nan

        PF_WARN = p.PF_WARN
        if "pf_prim" in dfA:
            n = dfA["pf_prim"].notna().sum()
            low = (dfA["pf_prim"] < PF_WARN).sum()
            pf_low_pct = (low/n*100.0) if n else 0.0
        else:
            pf_low_pct = np.nan

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1:
            st.metric("Charge pic (%)", _fmt(dfA.get("load_pct", pd.Series()).max(), "{:.1f}"))
        with c2:
            st.metric("θhs max (°C)", _fmt(dfA.get("t_core", pd.Series()).max(), "{:.1f}"))
        with c3:
            st.metric("Rendement inst. (%)", _fmt(eta_last*100.0 if not np.isnan(eta_last) else np.nan, "{:.1f}"))
        with c4:
            st.metric("Rendement moyen (%)", _fmt(eta_mean*100.0 if not np.isnan(eta_mean) else np.nan, "{:.1f}"))
        with c5:
            st.metric("Durée > WARN (s)", f"{dur_warn:.0f}")
        with c6:
            st.metric("Durée > ALARM (s)", f"{dur_alarm:.0f}")

        c7,c8 = st.columns(2)
        with c7:
            st.metric("Énergie délivrée (MWh)", _fmt(e_out, "{:.3f}"))
        with c8:
            st.metric("Énergie pertes (MWh)", _fmt(e_loss, "{:.3f}"))

        st.caption(f"Temps sous PF < seuil : {_fmt(pf_low_pct, '{:.1f}')} % des pas de temps.")
        st.caption(f"Vieillissement cumulé (h équiv.) : {_fmt(life_h, '{:.2f}')}")

        if "load_pct" in dfA.columns or "eta" in dfA.columns:
            ycols = []
            if "load_pct" in dfA.columns:
                ycols.append("load_pct")
            if "eta" in dfA.columns:
                dfA["eta_pct"] = dfA["eta"] * 100.0
                ycols.append("eta_pct")
            st.plotly_chart(
                px.line(dfA, x="t", y=ycols, title="Courbe de charge (%) & rendement (%)"),
                use_container_width=True
            )

    with tabs[6]:
        ev = pd.DataFrame(st.session_state.events_map.get(code, []))
        if not ev.empty:
            ev["t"] = pd.to_datetime(ev["ts"], unit="s", errors="coerce")
            ev = ev.sort_values("t").tail(500)
            # afficher aussi site/equipment/equipment_code si présents             # <<< MODIF
            cols = [c for c in ["t","level","code","msg","value","threshold","site","equipment","equipment_code"] if c in ev.columns]  # <<< MODIF
            st.dataframe(ev[cols], use_container_width=True, hide_index=True)      # <<< MODIF
            if st.button("💾 Exporter journal (append)", key="evt_export_one"):
                _append_events_to_csv(ev.to_dict("records"))
                st.success(f"Journal ajouté → {EVT_CSV}")
        else:
            st.info("Aucun événement en mémoire.")

    time.sleep(1.0 / max(int(hz), 1))
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()

# =====================================================================
# ============================== MULTI ===============================
# =====================================================================
else:
    sel_codes = st.multiselect(
        "Choisir jusqu’à 3 transformateurs",
        options=codes,
        default=codes[:min(3,len(codes))],
        max_selections=3,
        key="rt_sel_multi"
    )
    if not sel_codes:
        st.info("Sélectionne au moins un transformateur.")
        st.stop()

    # init engines + buffers
    tfm_meta = {}  # code -> dict(site, equipment, equipment_code)          # <<< MODIF
    for code in sel_codes:
        if code not in st.session_state.engines:
            st.session_state.engines[code] = TransformerEngine(TransformerParams())
        if code not in st.session_state.rt_data_map:
            st.session_state.rt_data_map[code] = []
        if code not in st.session_state.events_map:
            st.session_state.events_map[code] = []
        if code not in st.session_state.running_map:
            st.session_state.running_map[code] = False
        rec = get_transformer(code) or {}
        st.session_state.engines[code].set_params_from_record(rec)
        tfm_meta[code] = {                                                  # <<< MODIF
            "site": rec.get("site") or "",                                  # <<< MODIF
            "equipment": rec.get("name") or code,                           # <<< MODIF
            "equipment_code": code,                                         # <<< MODIF
        }                                                                   # <<< MODIF

    cTop = st.columns(len(sel_codes))
    for i, code in enumerate(sel_codes):
        with cTop[i]:
            running = st.session_state.running_map[code]
            st.markdown(f"### {code}")
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                if st.button("▶️", key=f"start_{code}", disabled=running):
                    st.session_state.running_map[code] = True
            with colb2:
                if st.button("⏹️", key=f"stop_{code}", disabled=not running):
                    st.session_state.running_map[code] = False
            with colb3:
                if st.button("♻️", key=f"reset_{code}"):
                    eng = st.session_state.engines[code]
                    eng.s.T_oil = eng.c.Tamb
                    eng.s.T_hotspot = eng.c.Tamb
                    eng.s.fan1_on = eng.s.fan2_on = False

            st.caption("Paramètres instantanés")
            load = st.slider(f"Charge {code} (%)", 0, 200, 50, 1, key=f"ld_{code}")
            pf   = st.slider(f"PF {code} (%)", 50, 100, 95, 1, key=f"pf_{code}")/100.0
            amb  = st.slider(f"T amb. {code} (°C)", 0, 60, 30, 1, key=f"ta_{code}")
            fan_mode = st.radio(f"Ventil {code}", ["AUTO","MAN"], horizontal=True, key=f"fanmode_{code}")
            f1 = st.toggle(f"F1 {code}", value=False, disabled=(fan_mode=="AUTO"), key=f"f1_{code}")
            f2 = st.toggle(f"F2 {code}", value=False, disabled=(fan_mode=="AUTO"), key=f"f2_{code}")

            eng = st.session_state.engines[code]

            base_load = float(load)
            base_pf   = float(pf)
            base_Tamb = float(amb)

            base_load = np.clip(base_load * (1.0 + 0.03*np.random.randn()), 0.0, 200.0)
            base_pf   = np.clip(base_pf   + 0.01*np.random.randn(), 0.5, 1.0)
            base_Tamb = np.clip(base_Tamb + 1.0*np.random.randn(), 0.0, 60.0)

            eng.set_controls(load_pct=float(base_load), pf_set=float(base_pf), Tamb=float(base_Tamb),
                             fan_mode=fan_mode, fan1_force=bool(f1), fan2_force=bool(f2))

    hz = st.slider("Fréquence d’update (Hz)", 1, 10, 2, 1, key="rt_hz_multi")
    max_pts = st.number_input("Points gardés", 60, 2000, 300, 60, key="rt_maxpts_multi")

    def _tick_multi():
        target_dt = 1.0 / max(int(hz), 1)
        now_m = time.monotonic()
        if (now_m - st.session_state.last_tick) >= target_dt * 0.95:
            for code in sel_codes:
                if not st.session_state.running_map[code]:
                    continue
                eng = st.session_state.engines[code]
                m, E = eng.step(target_dt)
                buf = st.session_state.rt_data_map[code]
                buf.append(m)
                st.session_state.rt_data_map[code] = buf[-int(max_pts):]
                if E:
                    evbuf = st.session_state.events_map[code]
                    enriched = []                                            # <<< MODIF
                    meta = tfm_meta.get(code, {})                            # <<< MODIF
                    for e in E:                                              # <<< MODIF
                        e2 = dict(e)                                         # <<< MODIF
                        e2.setdefault("equipment", meta.get("equipment", code))          # <<< MODIF
                        e2.setdefault("equipment_code", meta.get("equipment_code", code))# <<< MODIF
                        e2.setdefault("site", meta.get("site", ""))         # <<< MODIF
                        enriched.append(e2)                                  # <<< MODIF
                    evbuf.extend(enriched)                                   # <<< MODIF
                    st.session_state.events_map[code] = evbuf[-1000:]
                    _append_events_to_csv(enriched)                          # <<< MODIF
                    for e in enriched:                                       # <<< MODIF
                        try:
                            notify_event(e)                                  # <<< MODIF
                        except Exception:
                            pass
            st.session_state.last_tick = now_m

    _tick_multi()

    grid = st.columns(len(sel_codes))
    for i, code in enumerate(sel_codes):
        with grid[i]:
            df = pd.DataFrame(st.session_state.rt_data_map[code])
            if df.empty:
                st.info(f"{code}: en attente…")
                continue
            df = df.sort_values("ts")
            df["t"] = _safe_epoch_to_datetime(df["ts"])
            df["p_net_MW"] = (pd.to_numeric(df.get("p_net"), errors="coerce")/1e6)
            df["t_core"] = pd.to_numeric(df.get("t_core"), errors="coerce")
            st.plotly_chart(
                px.line(df, x="t", y=["p_net_MW","t_core"], title=f"{code} — P nette (MW) & θhs (°C)"),
                use_container_width=True
            )
            last = df.iloc[-1]
            colm1, colm2 = st.columns(2)
            with colm1:
                st.metric(
                    "P nette (MW)",
                    f"{last.get('p_net_MW', float('nan')):.3f}" if pd.notna(last.get('p_net_MW')) else "—"
                )
            with colm2:
                st.metric(
                    "θhs (°C)",
                    f"{last.get('t_core', float('nan')):.1f}" if pd.notna(last.get('t_core')) else "—"
                )

    time.sleep(1.0 / max(int(hz), 1))
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()
