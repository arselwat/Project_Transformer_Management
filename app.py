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
    page_title="Fiabilité & Gestion de stock — Transformateurs",
    page_icon="🛠️",
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

def _render_nav_card(title: str, desc: str, rel_path: str, icon: str, compact: bool = False):
    exists = _file_exists(rel_path)
    size_class = "card-compact" if compact else "card-panel"
    st.markdown(f'<div class="{size_class}">', unsafe_allow_html=True)
    st.markdown(f"#### {icon} {title}")
    st.markdown(f"<p>{desc}</p>", unsafe_allow_html=True)
    if exists:
        st.page_link(rel_path, label="Ouvrir", icon=icon)
    else:
        st.warning(f"Fichier introuvable : {rel_path}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
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
  --bg:#ffffff;
  --bg2:#f6f8fc;
  --text:#0b1221;
  --muted:#60708b;
  --card:#ffffff;
  --ring:rgba(31,119,180,.16);
  --primary:#1f77b4;
  --accent:#eaf3ff;
  --shadow:0 10px 28px -18px rgba(31,119,180,.22);
}
"""

DARK = """
:root{
  --bg:#0f1420;
  --bg2:#141b2c;
  --text:#e8eef8;
  --muted:#a8b2c2;
  --card:#111a2c;
  --ring:rgba(74,163,255,.18);
  --primary:#4aa3ff;
  --accent:#13233b;
  --shadow:0 12px 28px -18px rgba(74,163,255,.18);
}
"""

BASE = """
html, body, [class*="css"]{
  background:var(--bg);
  color:var(--text);
  font-size:16px;
}

.block-container{
  padding-top:0.95rem;
  padding-bottom:1.5rem;
  max-width: 1320px;
}

h1{font-size:1.85rem}
h2{font-size:1.35rem}
h3{font-size:1.08rem}

.hero{
  display:flex;
  gap:16px;
  align-items:center;
  background:linear-gradient(180deg,var(--bg2),var(--bg));
  border:1px solid var(--ring);
  padding:16px 18px;
  border-radius:18px;
  box-shadow:var(--shadow);
  margin-bottom:.35rem;
}

.logo{
  width:58px;
  height:58px;
  border-radius:14px;
  border:1px solid var(--ring);
  background:var(--card);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:28px;
  flex-shrink:0;
}

.kpi-wrap{
  background:var(--card);
  border:1px solid var(--ring);
  border-radius:16px;
  padding:10px 12px;
  box-shadow:var(--shadow);
}

.card-panel, .card-compact{
  background:var(--card);
  border-radius:14px;
  border:1px solid var(--ring);
  box-shadow:var(--shadow);
  height:100%;
}

.card-panel{
  padding:12px 12px 8px 12px;
  min-height:150px;
}

.card-compact{
  padding:10px 10px 6px 10px;
  min-height:128px;
}

.card-panel h4, .card-compact h4{
  margin:0 0 6px 0;
}

.card-panel p, .card-compact p{
  margin:0 0 8px 0;
  color:var(--muted);
  font-size:0.88rem;
  line-height:1.35;
}

.section-title{
  font-size:1.08rem;
  font-weight:700;
  margin: .35rem 0 .55rem 0;
}

.hr-soft{
  height:1px;
  background:linear-gradient(90deg, transparent, var(--ring), transparent);
  margin: .75rem 0 1rem 0;
}

.stButton>button{
  background:var(--primary)!important;
  color:#fff!important;
  border-radius:9px!important;
  border:none!important;
  padding:.28rem .72rem!important;
  font-size:.82rem!important;
  box-shadow:none!important;
}

[data-testid="stMetricValue"]{
  font-size:1.35rem!important;
}

[data-testid="stMetricLabel"]{
  font-size:.84rem!important;
}

.quick-menu{
  background:var(--card);
  border:1px solid var(--ring);
  border-radius:14px;
  padding:10px 10px 6px 10px;
  margin-top:8px;
}

.quick-menu h4{
  margin:0 0 8px 0;
  font-size:.98rem;
}

.sidebar-note{
  color:var(--muted);
  font-size:.82rem;
  line-height:1.35;
}

.footer{
  color:var(--muted);
  text-align:center;
  margin-top:22px;
  font-size:.82rem;
}
"""

st.markdown(f"<style>{(DARK if theme == 'Sombre' else LIGHT) + BASE}</style>", unsafe_allow_html=True)

# =========================================================
# DATA SUMMARY
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
# CUSTOM SIDEBAR MENU
# =========================================================
with st.sidebar:
    st.markdown('<div class="quick-menu">', unsafe_allow_html=True)
    st.markdown("#### 🧭 Accès rapide")
    if _file_exists("pages/1_Sources_fully_linked.py"):
        st.page_link("pages/1_Sources_fully_linked.py", label="Sources", icon="📥")
    if _file_exists("pages/2_Indicateurs_corrected.py"):
        st.page_link("pages/2_Indicateurs_corrected.py", label="Indicateurs", icon="📊")
    if _file_exists("pages/3_Optimisation_corrected.py"):
        st.page_link("pages/3_Optimisation_corrected.py", label="Optimisation", icon="🧠")
    if _file_exists("pages/4_Maintenance_corrected.py"):
        st.page_link("pages/4_Maintenance_corrected.py", label="Maintenance", icon="🛠️")
    if _file_exists("pages/5_Resultat_analyse_optimisation_Maintenance.py"):
        st.page_link("pages/5_Resultat_analyse_optimisation_Maintenance.py", label="Résultat global", icon="📋")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="sidebar-note">Pour garder le menu automatique de Streamlit bien rangé, '
        'renomme les fichiers avec des préfixes 01, 02, 03... '
        'Le désordre du menu latéral vient surtout de l’ordre des noms de fichiers.</p>',
        unsafe_allow_html=True,
    )

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
      <div class="logo">⚡</div>
      <div>
        <h1>Analyse de fiabilité, optimisation et aide à la décision pour la maintenance des transformateurs de puissance</h1>
        <p style="margin:6px 0 0 0;color:var(--muted)">
          TTF / MTBF / MTTR • Weibull • RP / NHPP / BPP • Thermique • Optimisation • Maintenance • Résultat global
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("📄 Pannes", f"{total_pannes}")
with k2:
    st.metric("🔧 Équipements", f"{nb_eq}")
with k3:
    st.metric("⏳ MTBF (h)", _metric_fmt(mtbf))
with k4:
    st.metric("🛠️ MTTR (h)", _metric_fmt(mttr))
with k5:
    st.metric("📦 Projet chargé", project_loaded)

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================================================
# STATUS STRIP
# =========================================================
left, right = st.columns([1.25, 1])
with left:
    st.info(
        f"Dataset TTF actif : {fail_meta.get('rows', 0)} lignes | "
        f"hash={fail_meta.get('hash', '') or '—'} | "
        f"source={fail_meta.get('source', '') or '—'}"
    )
with right:
    st.info(
        f"Projet complet : {'chargé' if proj_meta.get('ok') else 'non chargé'} | "
        f"source={proj_meta.get('source', '') or '—'}"
    )

# =========================================================
# MAIN NAV
# =========================================================
st.markdown('<div class="section-title">🧭 Navigation principale</div>', unsafe_allow_html=True)

# Ligne 1
r1 = st.columns(4)
with r1[0]:
    _render_nav_card(
        "Sources de données",
        "Importer le CSV TTF ou le projet Excel complet qui alimentera toutes les pages.",
        "pages/1_Sources_fully_linked.py",
        "📥",
        compact=True,
    )
with r1[1]:
    _render_nav_card(
        "Indicateurs",
        "Tests de tendance/dépendance, processus retenu, courbes fiabilistes et thermique.",
        "pages/2_Indicateurs_corrected.py",
        "📊",
        compact=True,
    )
with r1[2]:
    _render_nav_card(
        "Optimisation",
        "Intervalles optimisés, coût, fiabilité cible et contraintes thermiques.",
        "pages/3_Optimisation_corrected.py",
        "🧠",
        compact=True,
    )
with r1[3]:
    _render_nav_card(
        "Maintenance",
        "Planning virtuel, tâches dues, commentaires maintenance et plan PDF.",
        "pages/4_Maintenance_corrected.py",
        "🛠️",
        compact=True,
    )

# Ligne 2
r2 = st.columns(4)
with r2[0]:
    _render_nav_card(
        "Résultat global",
        "De l’analyse aux recommandations finales, avec synthèse complète par équipement.",
        "pages/5_Resultat_analyse_optimisation_Maintenance.py",
        "📋",
        compact=True,
    )
with r2[1]:
    _render_nav_card(
        "Stock & pièces",
        "Pièces de rechange, seuils de stock et liaison avec la maintenance.",
        "pages/6_Stock.py",
        "📦",
        compact=True,
    )
with r2[2]:
    _render_nav_card(
        "Transformateurs",
        "Vue transformateurs, fiches ou registres dédiés si ton module est activé.",
        "pages/7_Transformateurs.py",
        "🔌",
        compact=True,
    )
with r2[3]:
    _render_nav_card(
        "Temps réel",
        "Visualisation MQTT / simulation des mesures et alertes temps réel.",
        "pages/8_Visualisation_temps_reel.py",
        "📡",
        compact=True,
    )

# Ligne 3
r3 = st.columns([1, 1, 2])
with r3[0]:
    _render_nav_card(
        "Paramètres alertes",
        "Email, WhatsApp, seuils et canaux de notification.",
        "pages/9_Parametres_Alertes.py",
        "🚨",
        compact=True,
    )

with r3[1]:
    st.markdown('<div class="card-compact">', unsafe_allow_html=True)
    st.markdown("#### ✅ Flux conseillé")
    st.markdown(
        "<p>1. Sources → 2. Indicateurs → 3. Optimisation → 4. Maintenance → 5. Résultat global</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r3[2]:
    st.markdown('<div class="card-compact">', unsafe_allow_html=True)
    st.markdown("#### ℹ️ Conseils d’organisation")
    st.markdown(
        "<p>Pour que le menu latéral automatique Streamlit soit propre, renomme tes fichiers avec des préfixes sur 2 chiffres : "
        "01_Sources.py, 02_Indicateurs.py, 03_Optimisation.py, etc. "
        "Ton désordre actuel vient surtout du fait que les fichiers commencent par 1, 2, 4, 5, 6, 7, 8, 9, 10 avec des noms différents.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">© 2026 mumputujeanbaptiste@gmail.com — Fiabilité, optimisation et gestion de stock des transformateurs</div>',
    unsafe_allow_html=True,
)
