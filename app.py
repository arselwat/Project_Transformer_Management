from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.security.auth import login_form
from core.datahub import get_current_failures_df, get_failures_meta, get_project_meta
from core.ui import render_shell, render_page_header


st.set_page_config(
    page_title="Fiabilité & maintenance des transformateurs",
    page_icon="⚡",
    layout="wide",
)

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    login_form()
    st.stop()

render_shell("app.py")
render_page_header(
    "Fiabilité et maintenance des transformateurs",
    "Importation des données, analyse, optimisation, maintenance et décision finale.",
    "⚡",
)

BASE_DIR = Path(__file__).resolve().parent


def _page_exists(rel_path: str) -> bool:
    return (BASE_DIR / rel_path).exists()


def _fmt(v, nd: int = 2, fallback: str = "n/a") -> str:
    try:
        if v is None:
            return fallback
        x = float(v)
        if pd.isna(x):
            return fallback
        return f"{x:.{nd}f}"
    except Exception:
        return fallback


def _module_card(title: str, desc: str, rel_path: str, icon: str):
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="module-title">{icon} {title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="module-desc">{desc}</div>', unsafe_allow_html=True)
    if _page_exists(rel_path):
        st.page_link(rel_path, label="Ouvrir", icon=icon)
    else:
        st.caption("Page non disponible")
    st.markdown("</div>", unsafe_allow_html=True)


fail_df = get_current_failures_df()
fail_meta = get_failures_meta()
proj_meta = get_project_meta()

total_pannes = 0 if fail_df.empty else len(fail_df)
nb_eq = 0 if fail_df.empty else fail_df["equipment_code"].astype(str).nunique()

mtbf = None
if isinstance(fail_df, pd.DataFrame) and not fail_df.empty and "ttf_h" in fail_df.columns:
    vals = pd.to_numeric(fail_df["ttf_h"], errors="coerce").dropna()
    mtbf = float(vals.mean()) if len(vals) else None

mttr = None
if isinstance(fail_df, pd.DataFrame) and not fail_df.empty and "duree_rep_h" in fail_df.columns:
    vals = pd.to_numeric(fail_df["duree_rep_h"], errors="coerce").dropna()
    mttr = float(vals.mean()) if len(vals) else None

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Pannes", total_pannes)
with k2:
    st.metric("Équipements", nb_eq)
with k3:
    st.metric("MTBF (h)", _fmt(mtbf))
with k4:
    st.metric("MTTR (h)", _fmt(mttr))
with k5:
    st.metric("Projet chargé", "Oui" if proj_meta.get("ok") else "Non")

st.markdown("")

s1, s2 = st.columns(2)
with s1:
    st.markdown(
        f'<div class="status-box">Dataset actif : {fail_meta.get("rows", 0)} lignes</div>',
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        f'<div class="status-box">Projet actif : {"Oui" if proj_meta.get("ok") else "Non"}</div>',
        unsafe_allow_html=True,
    )

st.markdown("### 🧩 Modules")

r1 = st.columns(3)
with r1[0]:
    _module_card(
        "Sources de données",
        "Importer le CSV ou le projet Excel.",
        "pages/1_Sources_fully_linked_fixed.py",
        "📥",
    )
with r1[1]:
    _module_card(
        "Indicateurs",
        "Voir les tests, paramètres et courbes.",
        "pages/2_Indicateurs_verified.py",
        "📊",
    )
with r1[2]:
    _module_card(
        "Optimisation",
        "Calculer les intervalles et recommandations.",
        "pages/3_Optimisation_verified.py",
        "🧠",
    )

r2 = st.columns(3)
with r2[0]:
    _module_card(
        "Maintenance",
        "Planning, échéances et commentaires.",
        "pages/4_Maintenance_verified.py",
        "🛠️",
    )
with r2[1]:
    _module_card(
        "Résultat global",
        "Traçabilité complète de la décision.",
        "pages/5_Resultat_analyse_optimisation_Maintenance.py",
        "📋",
    )
with r2[2]:
    _module_card(
        "Stock",
        "Pièces et seuils de réapprovisionnement.",
        "pages/6_Stock.py",
        "📦",
    )

r3 = st.columns(3)
with r3[0]:
    _module_card(
        "Transformateurs",
        "Fiches et vue équipement.",
        "pages/7_Transformateurs.py",
        "🔌",
    )
with r3[1]:
    _module_card(
        "Temps réel",
        "Supervision et mesures terrain.",
        "pages/8_Visualisation_temps_reel.py",
        "📡",
    )
with r3[2]:
    _module_card(
        "Alertes",
        "Destinataires et paramètres d’alerte.",
        "pages/9_Parametres_Alertes.py",
        "🚨",
    )

st.markdown(
    '<div class="footer">© 2026 — Fiabilité, optimisation et maintenance des transformateurs</div>',
    unsafe_allow_html=True,
)