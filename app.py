# app.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from core.security.auth import login_form  # login unique

# =========================
# CONFIG GÉNÉRALE
# =========================

st.set_page_config(
    page_title="Fiabilité & Gestion de stock — Transformateurs",
    page_icon="🛠️",
    layout="wide",
)

# ---- Authentification ----
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    # On affiche uniquement le formulaire de login, rien d'autre
    login_form()
    st.stop()

# =========================
# THÈME + LOGO (sidebar)
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR.mkdir(exist_ok=True, parents=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

with st.sidebar:
    st.markdown("### 🎨 Apparence")
    theme = st.radio(
        "Mode",
        ["Clair", "Sombre"],
        horizontal=True,
        key="__theme__",
        index=0,
    )

LIGHT = """
:root{
  --bg:#ffffff;--bg2:#f5f7fb;--text:#0b1221;--muted:#63708a;
  --card:#ffffff;--ring:rgba(31,119,180,.18);--primary:#1f77b4;
}
"""
DARK = """
:root{
  --bg:#0f1420;--bg2:#141b2c;--text:#e8eef8;--muted:#a8b2c2;
  --card:#0e1525;--ring:rgba(74,163,255,.22);--primary:#4aa3ff;
}
"""
BASE = """
html,body,[class*="css"]{background:var(--bg);color:var(--text);font-size:17px}
.block-container{padding-top:1.1rem}
h1{font-size:2.0rem} h2{font-size:1.55rem} h3{font-size:1.2rem}
.hero{
  display:flex;gap:18px;align-items:center;
  background:linear-gradient(180deg,var(--bg2),var(--bg));
  border:1px solid var(--ring);padding:18px;border-radius:18px;
  box-shadow:0 10px 32px -14px var(--ring)
}
.logo{
  width:64px;height:64px;border-radius:14px;border:1px solid var(--ring);
  background:var(--card);display:flex;align-items:center;
  justify-content:center;font-size:32px
}
.card-panel{
  background:var(--card);border-radius:16px;border:1px solid var(--ring);
  padding:14px 14px 10px 14px;margin-bottom:12px;
  box-shadow:0 8px 22px -14px var(--ring);
}
.card-panel h3{margin:0 0 4px 0;font-size:1.05rem}
.card-panel p{margin:0 0 6px 0;color:var(--muted);font-size:0.9rem}
.stButton>button{
  background:var(--primary)!important;color:#fff!important;border-radius:10px!important;
  border:none!important;padding:.35rem .8rem!important;font-size:0.85rem!important;
}
[data-testid="stMetricValue"]{font-size:1.6rem!important}
.footer{color:var(--muted);text-align:center;margin-top:24px;font-size:0.85rem}
"""
st.markdown(f"<style>{(DARK if theme=='Sombre' else LIGHT)+BASE}</style>", unsafe_allow_html=True)

# =========================
# HÉRO
# =========================
st.markdown(
    """
<div class="hero">
  <div class="logo">⚡</div>
  <div>
    <h1>Analyse de Fiabilité & Gestion de Stock — Transformateurs</h1>
    <p style="margin:6px 0 0 0;color:var(--muted)">
      MTBF/MTTR & Weibull • Optimisation des intervalles • Maintenance & kits pièces • Temps réel MQTT
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# MÉTRIQUES RAPIDES
# =========================
df = st.session_state.get("failures_df")
if not isinstance(df, pd.DataFrame):
    file_conso = DATA_DIR / "failures_saved.csv"
    if file_conso.exists():
        try:
            df = pd.read_csv(file_conso)
        except Exception:
            df = None

total_pannes = 0 if df is None else len(df)
nb_eq = 0 if df is None or "equipment_code" not in df.columns else df["equipment_code"].nunique()

mtbf = None
mttr = None
if isinstance(df, pd.DataFrame) and "ttf_h" in df.columns:
    vals = pd.to_numeric(df["ttf_h"], errors="coerce").dropna()
    mtbf = float(vals.mean()) if len(vals) else None
if isinstance(df, pd.DataFrame) and "duree_rep_h" in df.columns:
    vals = pd.to_numeric(df["duree_rep_h"], errors="coerce").dropna()
    mttr = float(vals.mean()) if len(vals) else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📄 Pannes", f"{total_pannes}")
with c2:
    st.metric("🔧 Équipements", f"{nb_eq}")
with c3:
    st.metric("⏳ MTBF (h)", f"{mtbf:.2f}" if mtbf else "n/a")
with c4:
    st.metric("🛠️ MTTR (h)", f"{mttr:.2f}" if mttr else "n/a")

st.markdown("---")

# =========================
# CARTES DE NAVIGATION (3 par ligne)
# =========================
st.markdown("### 🧭 Navigation principale")

# (tout le reste de ton app inchangé : lignes row1, row2, row3, footer…)
# ...
