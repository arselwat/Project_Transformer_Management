from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.security.auth import login_form
from core.datahub import (
    get_current_failures_df,
    get_failures_meta,
    get_project_meta,
)

st.set_page_config(
    page_title="Fiabilité Transformateurs",
    page_icon="⚡",
    layout="wide",
)

# =========================================================
# AUTH
# =========================================================
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    login_form()
    st.stop()

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR.mkdir(exist_ok=True, parents=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# HELPERS
# =========================================================
def _file_exists(rel_path: str) -> bool:
    return (BASE_DIR / rel_path).exists()

def _metric_fmt(v, nd: int = 2, fallback: str = "n/a") -> str:
    try:
        if v is None:
            return fallback
        x = float(v)
        if pd.isna(x):
            return fallback
        return f"{x:.{nd}f}"
    except Exception:
        return fallback

def _nav_link(rel_path: str, label: str, icon: str):
    if _file_exists(rel_path):
        st.page_link(rel_path, label=label, icon=icon)

def _render_nav_card(title: str, desc: str, rel_path: str, icon: str):
    st.markdown('<div class="nav-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{icon} {title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-desc'>{desc}</div>", unsafe_allow_html=True)
    if _file_exists(rel_path):
        st.page_link(rel_path, label="Ouvrir", icon=icon)
    else:
        st.caption("Page non disponible")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR THEME
# =========================================================
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
  --bg:#f7f9fc;
  --bg2:#ffffff;
  --text:#142033;
  --muted:#6b7a90;
  --card:#ffffff;
  --line:#dce6f2;
  --primary:#1f77b4;
  --primary-soft:#eaf4ff;
  --shadow:0 8px 24px rgba(31,119,180,.08);
}
"""

DARK = """
:root{
  --bg:#0f1724;
  --bg2:#141f31;
  --text:#edf3fb;
  --muted:#a7b4c8;
  --card:#162235;
  --line:#253752;
  --primary:#4aa3ff;
  --primary-soft:#18314f;
  --shadow:0 8px 24px rgba(0,0,0,.20);
}
"""

BASE = """
html, body, [class*="css"]{
  background:var(--bg);
  color:var(--text);
  font-size:15px;
}

/* cache le menu multipage par défaut de streamlit */
[data-testid="stSidebarNav"]{
  display:none;
}

.block-container{
  padding-top:1rem;
  padding-bottom:1.5rem;
  max-width:1280px;
}

h1{font-size:1.8rem !important;}
h2{font-size:1.3rem !important;}
h3{font-size:1.05rem !important;}

.hero{
  display:flex;
  align-items:center;
  gap:16px;
  padding:18px 20px;
  border-radius:20px;
  background:linear-gradient(180deg, var(--bg2), var(--primary-soft));
  border:1px solid var(--line);
  box-shadow:var(--shadow);
  margin-bottom:14px;
}

.hero-logo{
  width:64px;
  height:64px;
  border-radius:16px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:var(--card);
  border:1px solid var(--line);
  font-size:30px;
  flex-shrink:0;
}

.hero p{
  margin:6px 0 0 0;
  color:var(--muted);
  font-size:.95rem;
}

.kpi-box{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:16px;
  padding:6px 8px;
  box-shadow:var(--shadow);
}

.nav-card{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:16px;
  padding:12px;
  min-height:120px;
  box-shadow:var(--shadow);
  margin-bottom:10px;
}

.card-title{
  font-weight:700;
  font-size:1rem;
  margin-bottom:6px;
}

.card-desc{
  color:var(--muted);
  font-size:.87rem;
  line-height:1.4;
  margin-bottom:10px;
}

.section-title{
  font-size:1.05rem;
  font-weight:700;
  margin:10px 0 8px 0;
}

.hr-soft{
  height:1px;
  background:linear-gradient(90deg, transparent, var(--line), transparent);
  margin:12px 0 16px 0;
}

.stButton > button{
  background:var(--primary) !important;
  color:white !important;
  border:none !important;
  border-radius:10px !important;
  padding:.35rem .8rem !important;
  font-size:.82rem !important;
}

[data-testid="stMetricValue"]{
  font-size:1.35rem !important;
}

[data-testid="stMetricLabel"]{
  font-size:.82rem !important;
}

.sidebar-box{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:14px;
  padding:10px;
  margin-top:10px;
}

.sidebar-title{
  font-weight:700;
  margin-bottom:8px;
}

.sidebar-small{
  color:var(--muted);
  font-size:.82rem;
  line-height:1.4;
}

.footer{
  text-align:center;
  color:var(--muted);
  margin-top:24px;
  font-size:.82rem;
}
"""

st.markdown(f"<style>{(DARK if theme == 'Sombre' else LIGHT) + BASE}</style>", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
failures_df = get_current_failures_df()
fail_meta = get_failures_meta()
proj_meta = get_project_meta()

total_pannes = 0 if failures_df.empty else len(failures_df)
nb_eq = 0 if failures_df.empty or "equipment_code" not in failures_df.columns else failures_df["equipment_code"].nunique()

mtbf = None
mttr = None

if isinstance(failures_df, pd.DataFrame) and not failures_df.empty and "ttf_h" in failures_df.columns:
    vals = pd.to_numeric(failures_df["ttf_h"], errors="coerce").dropna()
    mtbf = float(vals.mean()) if len(vals) else None

if isinstance(failures_df, pd.DataFrame) and not failures_df.empty and "duree_rep_h" in failures_df.columns:
    vals = pd.to_numeric(failures_df["duree_rep_h"], errors="coerce").dropna()
    mttr = float(vals.mean()) if len(vals) else None

project_loaded = "Oui" if proj_meta.get("ok") else "Non"

# =========================================================
# CLEAN SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚡ Navigation</div>', unsafe_allow_html=True)

    _nav_link("pages/1_Sources_fully_linked_fixed.py", "Sources de données", "📥")
    _nav_link("pages/2_Indicateurs_verified.py", "Indicateurs", "📊")
    _nav_link("pages/3_Optimisation_verified.py", "Optimisation", "🧠")
    _nav_link("pages/4_Maintenance_verified.py", "Maintenance", "🛠️")
    _nav_link("pages/5_Resultat_analyse_optimisation_Maintenance_fixed.py", "Résultat global", "📋")
    _nav_link("pages/6_Stock.py", "Stock", "📦")
    _nav_link("pages/7_Transformateurs.py", "Transformateurs", "🔌")
    _nav_link("pages/8_Visualisation_temps_reel.py", "Temps réel", "📡")
    _nav_link("pages/9_Parametres_Alertes.py", "Alertes", "🚨")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📌 État</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='sidebar-small'>"
        f"TTF : {fail_meta.get('rows', 0)} lignes<br>"
        f"Projet : {project_loaded}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-logo">⚡</div>
      <div>
        <h1>Fiabilité et maintenance des transformateurs</h1>
        <p>Importation des données, analyse, optimisation, maintenance et décision finale.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Pannes", f"{total_pannes}")
with k2:
    st.metric("Équipements", f"{nb_eq}")
with k3:
    st.metric("MTBF (h)", _metric_fmt(mtbf))
with k4:
    st.metric("MTTR (h)", _metric_fmt(mttr))
with k5:
    st.metric("Projet chargé", project_loaded)

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================================================
# SHORT INFO
# =========================================================
a, b = st.columns(2)
with a:
    st.info(f"Dataset actif : {fail_meta.get('rows', 0)} lignes")
with b:
    st.info(f"Projet actif : {'Oui' if proj_meta.get('ok') else 'Non'}")

# =========================================================
# MAIN NAV
# =========================================================
st.markdown('<div class="section-title">🧭 Modules</div>', unsafe_allow_html=True)

row1 = st.columns(3)
with row1[0]:
    _render_nav_card(
        "Sources de données",
        "Importer le CSV ou le projet Excel.",
        "pages/1_Sources_fully_linked.py",
        "📥",
    )
with row1[1]:
    _render_nav_card(
        "Indicateurs",
        "Voir les tests, paramètres et courbes.",
        "pages/2_Indicateurs_corrected.py",
        "📊",
    )
with row1[2]:
    _render_nav_card(
        "Optimisation",
        "Calculer les intervalles et recommandations.",
        "pages/3_Optimisation_corrected.py",
        "🧠",
    )

row2 = st.columns(3)
with row2[0]:
    _render_nav_card(
        "Maintenance",
        "Afficher le planning et les tâches dues.",
        "pages/4_Maintenance_corrected.py",
        "🛠️",
    )
with row2[1]:
    _render_nav_card(
        "Résultat global",
        "Voir la synthèse complète et la décision finale.",
        "pages/5_Resultat_analyse_optimisation_Maintenance.py",
        "📋",
    )
with row2[2]:
    _render_nav_card(
        "Stock",
        "Gérer les pièces et niveaux de stock.",
        "pages/6_Stock.py",
        "📦",
    )

row3 = st.columns(3)
with row3[0]:
    _render_nav_card(
        "Transformateurs",
        "Consulter les fiches des équipements.",
        "pages/7_Transformateurs.py",
        "🔌",
    )
with row3[1]:
    _render_nav_card(
        "Temps réel",
        "Suivre les mesures et alertes en direct.",
        "pages/8_Visualisation_temps_reel.py",
        "📡",
    )
with row3[2]:
    _render_nav_card(
        "Alertes",
        "Configurer les notifications.",
        "pages/9_Parametres_Alertes.py",
        "🚨",
    )

# =========================================================
# SIMPLE FLOW
# =========================================================
st.markdown('<div class="section-title">✅ Ordre conseillé</div>', unsafe_allow_html=True)
st.success("Sources de données → Indicateurs → Optimisation → Maintenance → Résultat global")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">© 2026 — Fiabilité et maintenance des transformateurs</div>',
    unsafe_allow_html=True,
)