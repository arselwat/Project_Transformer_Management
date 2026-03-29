from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


APP_TITLE = "Fiabilité et maintenance des transformateurs"

NAV_ITEMS = [
    ("app.py", "Accueil", "🏠"),
    ("pages/1_Sources_fully_linked_fixed.py", "Sources de données", "📥"),
    ("pages/2_Indicateurs_verified.py", "Indicateurs", "📊"),
    ("pages/3_Optimisation_verified.py", "Optimisation", "🧠"),
    ("pages/4_Maintenance_verified.py", "Maintenance", "🛠️"),
    ("pages/5_Resultat_analyse_optimisation_Maintenance_fixed.py", "Résultat global", "📋"),
    ("pages/6_Stock.py", "Stock", "📦"),
    ("pages/7_Transformateurs.py", "Transformateurs", "🔌"),
    ("pages/8_Visualisation_temps_reel.py", "Temps réel", "📡"),
    ("pages/9_Parametres_Alertes.py", "Alertes", "🔔"),
]


def _normalize_path(p: str) -> str:
    return str(p).replace("\\", "/").strip().lower()


def _inject_shell_css() -> None:
    st.markdown(
        """
        <style>
        .block-container{
            padding-top: 1rem;
            padding-bottom: 1.2rem;
            max-width: 1300px;
        }

        [data-testid="stSidebar"]{
            border-right: 1px solid #e8edf5;
        }

        .app-brand{
            display:flex;
            align-items:center;
            gap:12px;
            padding:10px 8px 14px 8px;
            margin-bottom:8px;
        }

        .app-brand-icon{
            width:42px;
            height:42px;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:22px;
            background:#eef4ff;
            border:1px solid #d8e4ff;
        }

        .app-brand-title{
            font-size:1.02rem;
            font-weight:700;
            line-height:1.2;
            margin:0;
        }

        .app-brand-sub{
            font-size:.82rem;
            color:#6b7280;
            margin-top:2px;
        }

        .side-group-title{
            font-size:.80rem;
            font-weight:700;
            color:#6b7280;
            margin:10px 0 6px 0;
            text-transform:uppercase;
            letter-spacing:.04em;
        }

        .page-header{
            display:flex;
            align-items:center;
            gap:14px;
            padding:16px 18px;
            border:1px solid #dbe7f5;
            border-radius:18px;
            background:linear-gradient(180deg,#f7fbff 0%, #eef5fc 100%);
            margin-bottom:16px;
        }

        .page-header-icon{
            width:56px;
            height:56px;
            border-radius:16px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:28px;
            background:#ffffff;
            border:1px solid #dbe7f5;
            flex-shrink:0;
        }

        .page-header-title{
            margin:0;
            font-size:1.95rem;
            font-weight:800;
            line-height:1.1;
        }

        .page-header-sub{
            margin-top:4px;
            color:#667085;
            font-size:.95rem;
        }

        .paper-title{
            text-align:center;
            font-weight:700;
            font-size:1.05rem;
            margin:18px 0 8px 0;
        }

        .status-box{
            padding:12px 14px;
            border-radius:14px;
            background:#edf5ff;
            border:1px solid #d8e7fb;
            font-weight:600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shell(current_page: Optional[str] = None) -> None:
    _inject_shell_css()

    current_norm = _normalize_path(current_page or "")

    with st.sidebar:
        st.markdown(
            f"""
            <div class="app-brand">
                <div class="app-brand-icon">⚡</div>
                <div>
                    <div class="app-brand-title">{APP_TITLE}</div>
                    <div class="app-brand-sub">Navigation de l'application</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-group-title">Navigation</div>', unsafe_allow_html=True)

        for path, label, icon in NAV_ITEMS:
            exists = Path(path).exists()
            if exists:
                st.page_link(path, label=label, icon=icon)
            else:
                st.caption(f"{icon} {label} — page non trouvée")

        st.divider()

        auth_user = (
            st.session_state.get("username")
            or st.session_state.get("user")
            or "admin"
        )
        st.success(f"Connecté : {auth_user}")


def render_page_header(title: str, subtitle: str = "", icon: str = "📄") -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-icon">{icon}</div>
            <div>
                <div class="page-header-title">{title}</div>
                <div class="page-header-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_paper_table(title: str, df: pd.DataFrame) -> None:
    st.markdown(f'<div class="paper-title">{title}</div>', unsafe_allow_html=True)
    if df is None or df.empty:
        st.info("Aucune donnée disponible.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)