# app.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
#from utils.auth import check_password

# =========================
# CONFIG GÉNÉRALE
# =========================

# ---- Authentification ----
if not check_password():
    # On arrête ici si pas loggé (seul le login est affiché)
    st.stop()

# ---- Suite de ton app (accueil, navigation, etc.) ----
st.title("📊 Tableau de bord – Fiabilité des transformateurs")

st.set_page_config(
    page_title="Fiabilité & Gestion de stock— Transformateurs",
    page_icon="🛠️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR.mkdir(exist_ok=True, parents=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# THÈME + LOGO (sidebar)
# =========================
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

# Ligne 1 : Sources / Indicateurs / Optimisation
row1 = st.columns(3)
with row1[0]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 📥 Sources de données")
    st.markdown(
        "<p>Importer les historiques de pannes (CSV TTF) ou configurer la source MQTT pour le temps réel.</p>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Sources_donnees.py", label="Ouvrir les sources", icon="📥")
    st.markdown("</div>", unsafe_allow_html=True)

with row1[1]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 📊 Indicateurs")
    st.markdown("<p>Courbes R/F/f/h, estimation Weibull, MTBF/MTTR détaillés.</p>", unsafe_allow_html=True)
    st.page_link("pages/2_Indicateurs.py", label="Ouvrir les indicateurs", icon="📊")
    st.markdown("</div>", unsafe_allow_html=True)

with row1[2]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Optimisation")
    st.markdown("<p>Calcul des intervalles optimaux et génération du rapport de fiabilité.</p>", unsafe_allow_html=True)
    st.page_link("pages/3_Optimisation.py", label="Ouvrir l’optimisation", icon="🧠")
    st.markdown("</div>", unsafe_allow_html=True)

# Ligne 2 : Temps réel / Maintenance / Stock
row2 = st.columns(3)
with row2[0]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 📡 Temps réel")
    st.markdown("<p>Simulation ou acquisition MQTT : tensions, courants, températures, alertes.</p>", unsafe_allow_html=True)
    st.page_link("pages/4_Visualisation_temps_reel.py", label="Ouvrir le temps réel", icon="📡")
    st.markdown("</div>", unsafe_allow_html=True)

with row2[1]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ Maintenance")
    st.markdown("<p>Tâches dues, plan PDF, envois par email/WhatsApp, lien avec le stock.</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Maintenance.py", label="Ouvrir la maintenance", icon="🛠️")
    st.markdown("</div>", unsafe_allow_html=True)

with row2[2]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 📦 Stock & pièces")
    st.markdown("<p>Pièces de rechange recommandées, mises à jour des quantités, alertes de seuil.</p>", unsafe_allow_html=True)
    st.page_link("pages/6_Stock.py", label="Ouvrir le stock", icon="📦")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Paramètres alertes
# =========================
row3 = st.columns(3)
with row3[0]:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("#### 🚨 Paramètres d’alertes")
    st.markdown("<p>Destinataires email/WhatsApp, seuils globaux et configuration des canaux.</p>", unsafe_allow_html=True)
    st.page_link("pages/7_Parametres_Alertes.py", label="Configurer les alertes", icon="🚨")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PIED DE PAGE
# =========================
st.markdown('<div class="footer">© 2025 mumputujeanbaptiste@gmail.com— Fiabilité et gestion de Stock • Interface</div>', unsafe_allow_html=True)
