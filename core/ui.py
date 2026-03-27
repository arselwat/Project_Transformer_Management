from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]

try:
    from core.datahub import get_current_failures_df, get_failures_meta, get_project_meta
except Exception:
    get_current_failures_df = None
    get_failures_meta = None
    get_project_meta = None


NAV_ITEMS = [
    ("app.py", "Accueil", "🏠"),
    ("pages/1_Sources_fully_linked_fixed.py", "Sources de données", "📥"),
    ("pages/2_Indicateurs_verified.py", "Indicateurs", "📊"),
    ("pages/3_Optimisation_verified.py", "Optimisation", "🧠"),
    ("pages/4_Maintenance_verified.py", "Maintenance", "🛠️"),
    ("pages/5_Resultat_analyse_optimisation_Maintenance.py", "Résultat global", "📋"),
    ("pages/6_Stock.py", "Stock", "📦"),
    ("pages/7_Transformateurs.py", "Transformateurs", "🔌"),
    ("pages/8_Visualisation_temps_reel.py", "Temps réel", "📡"),
    ("pages/9_Parametres_Alertes.py", "Alertes", "🚨"),
]


def _exists(rel_path: str) -> bool:
    return (BASE_DIR / rel_path).exists()


def _theme_css(theme: str) -> str:
    light = """
    :root{
      --bg:#f7f9fc;
      --bg2:#eef4fb;
      --text:#1f2a44;
      --muted:#6b7a96;
      --card:#ffffff;
      --line:#d8e2f0;
      --primary:#2f6fed;
      --soft:#edf4ff;
      --success:#eaf7ef;
      --shadow:0 10px 24px -18px rgba(47,111,237,.35);
    }
    """
    dark = """
    :root{
      --bg:#0f1724;
      --bg2:#131e31;
      --text:#edf2ff;
      --muted:#a8b5cf;
      --card:#111c2f;
      --line:#22314f;
      --primary:#5d9bff;
      --soft:#16253b;
      --success:#143122;
      --shadow:0 10px 24px -18px rgba(93,155,255,.28);
    }
    """
    base = """
    [data-testid="stSidebarNav"] {display:none;}
    section[data-testid="stSidebar"]{
      min-width: 285px !important;
      max-width: 285px !important;
      border-right: 1px solid var(--line);
      background: var(--bg);
    }
    html, body, [class*="css"]{
      background: var(--bg);
      color: var(--text);
      font-size: 16px;
    }
    .block-container{
      max-width: 1320px;
      padding-top: 1rem;
      padding-bottom: 1.5rem;
    }
    h1,h2,h3,h4{color:var(--text);}
    .hero-box{
      display:flex;
      align-items:center;
      gap:16px;
      background: linear-gradient(180deg, var(--bg2), var(--soft));
      border:1px solid var(--line);
      border-radius:20px;
      padding:18px 20px;
      box-shadow: var(--shadow);
      margin-bottom: 1rem;
    }
    .hero-icon{
      width:64px;
      height:64px;
      border-radius:16px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:30px;
      background: var(--card);
      border:1px solid var(--line);
      flex-shrink:0;
    }
    .hero-title{
      font-size:1.85rem;
      font-weight:800;
      line-height:1.1;
      margin:0;
    }
    .hero-sub{
      margin-top:4px;
      color:var(--muted);
      font-size:.96rem;
    }
    .kpi-card{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:10px 14px;
      box-shadow: var(--shadow);
    }
    .status-box{
      background: var(--soft);
      border:1px solid var(--line);
      border-radius:14px;
      padding:12px 14px;
      color:var(--text);
      margin-bottom: .5rem;
    }
    .sidebar-brand{
      display:flex;
      align-items:center;
      gap:10px;
      font-weight:800;
      font-size:1.15rem;
      margin:.25rem 0 .85rem 0;
    }
    .sidebar-brand-icon{
      width:42px;
      height:42px;
      border-radius:12px;
      display:flex;
      align-items:center;
      justify-content:center;
      background:var(--soft);
      border:1px solid var(--line);
      font-size:20px;
    }
    .side-chip{
      background: var(--success);
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 12px;
      margin:.4rem 0;
      font-size:.9rem;
    }
    .nav-title{
      margin-top:.8rem;
      margin-bottom:.35rem;
      font-weight:800;
      font-size:1rem;
    }
    .nav-current{
      background: var(--soft);
      border:1px solid var(--line);
      border-radius:12px;
      padding:10px 12px;
      font-weight:700;
      margin-bottom:.3rem;
    }
    .stPageLink{
      margin-bottom:.15rem;
    }
    .module-card{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:12px 14px;
      min-height:142px;
      box-shadow: var(--shadow);
    }
    .module-title{
      font-weight:800;
      font-size:1rem;
      margin-bottom:6px;
    }
    .module-desc{
      color:var(--muted);
      font-size:.9rem;
      line-height:1.35;
      margin-bottom:10px;
    }
    .small-note{
      color:var(--muted);
      font-size:.85rem;
    }
    .footer{
      color:var(--muted);
      text-align:center;
      margin-top:22px;
      font-size:.82rem;
    }
    .paper-caption{
      text-align:center;
      font-weight:700;
      margin:.9rem 0 .35rem 0;
      font-size:1rem;
    }
    .paper-table table{
      width:100%;
      border-collapse:collapse;
      background:var(--card);
      font-size:.92rem;
    }
    .paper-table th, .paper-table td{
      border:1.4px solid #111827;
      padding:8px 10px;
      vertical-align:top;
    }
    .paper-table th{
      background:#f3f4f6;
      color:#111827;
      text-align:center;
      font-weight:700;
    }
    """
    return f"<style>{(dark if theme == 'Sombre' else light) + base}</style>"


def render_shell(current_path: str) -> None:
    theme = st.session_state.get("__theme__", "Clair")

    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        theme = st.radio(
            "Mode",
            ["Clair", "Sombre"],
            index=0 if theme == "Clair" else 1,
            horizontal=True,
            key="__theme__",
        )

    st.markdown(_theme_css(theme), unsafe_allow_html=True)

    fail_rows = 0
    proj_ok = False
    user_name = st.session_state.get("username", "admin")

    try:
        if callable(get_current_failures_df):
            df = get_current_failures_df()
            if isinstance(df, pd.DataFrame):
                fail_rows = len(df)
    except Exception:
        pass

    try:
        if callable(get_project_meta):
            proj_ok = bool((get_project_meta() or {}).get("ok"))
    except Exception:
        proj_ok = False

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-brand-icon">⚡</div>
              <div>Fiabilité transfo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="side-chip">Dataset actif : {fail_rows} lignes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="side-chip">Projet actif : {"Oui" if proj_ok else "Non"}</div>', unsafe_allow_html=True)

        st.markdown('<div class="nav-title">🧭 Navigation</div>', unsafe_allow_html=True)

        for rel_path, label, icon in NAV_ITEMS:
            if not _exists(rel_path):
                continue
            if rel_path == current_path:
                st.markdown(f'<div class="nav-current">{icon} {label}</div>', unsafe_allow_html=True)
            else:
                st.page_link(rel_path, label=label, icon=icon)

        st.markdown("---")
        st.markdown(f'<div class="side-chip">Connecté : {user_name}</div>', unsafe_allow_html=True)

        if st.button("Se déconnecter", use_container_width=True):
            st.session_state["auth_ok"] = False
            st.rerun()


def render_page_header(title: str, subtitle: str, icon: str = "⚡") -> None:
    st.markdown(
        f"""
        <div class="hero-box">
          <div class="hero-icon">{icon}</div>
          <div>
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_paper_table(caption: str, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Aucune donnée à afficher.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(f'<div class="paper-caption">{caption}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="paper-table">{html}</div>', unsafe_allow_html=True)