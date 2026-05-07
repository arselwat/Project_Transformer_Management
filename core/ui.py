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


def _get_theme_type() -> str:
    try:
        theme_type = getattr(st.context.theme, "type", "light")
        if theme_type:
            return str(theme_type).lower()
    except Exception:
        pass
    return "light"


def _get_palette() -> dict:
    is_dark = _get_theme_type() == "dark"

    if is_dark:
        return {
            "sidebar_border": "rgba(255,255,255,0.10)",
            "brand_icon_bg": "rgba(255,255,255,0.08)",
            "brand_icon_border": "rgba(255,255,255,0.14)",
            "brand_title": "#F8FAFC",
            "brand_sub": "#CBD5E1",
            "muted": "#94A3B8",
            "header_bg_1": "rgba(255,255,255,0.08)",
            "header_bg_2": "rgba(255,255,255,0.05)",
            "header_border": "rgba(255,255,255,0.14)",
            "header_icon_bg": "rgba(255,255,255,0.08)",
            "header_icon_border": "rgba(255,255,255,0.14)",
            "header_title": "#F8FAFC",
            "header_sub": "#CBD5E1",
            "paper_title": "#F8FAFC",
            "status_bg": "rgba(255,255,255,0.06)",
            "status_border": "rgba(255,255,255,0.12)",
            "status_text": "#E2E8F0",
        }

    return {
        "sidebar_border": "#E8EDF5",
        "brand_icon_bg": "#EEF4FF",
        "brand_icon_border": "#D8E4FF",
        "brand_title": "#111827",
        "brand_sub": "#6B7280",
        "muted": "#6B7280",
        "header_bg_1": "#F7FBFF",
        "header_bg_2": "#EEF5FC",
        "header_border": "#DBE7F5",
        "header_icon_bg": "#FFFFFF",
        "header_icon_border": "#DBE7F5",
        "header_title": "#1F2937",
        "header_sub": "#667085",
        "paper_title": "#1F2937",
        "status_bg": "#EDF5FF",
        "status_border": "#D8E7FB",
        "status_text": "#1F2937",
    }


def _inject_shell_css() -> None:
    c = _get_palette()

    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1.2rem;
            max-width: 1300px;
        }}

        [data-testid="stSidebar"] {{
            border-right: 1px solid {c["sidebar_border"]};
        }}

        .app-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 8px 14px 8px;
            margin-bottom: 8px;
        }}

        .app-brand-icon {{
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: {c["brand_icon_bg"]};
            border: 1px solid {c["brand_icon_border"]};
        }}

        .app-brand-title {{
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
            color: {c["brand_title"]} !important;
        }}

        .app-brand-sub {{
            font-size: .82rem;
            color: {c["brand_sub"]} !important;
            margin-top: 2px;
        }}

        .side-group-title {{
            font-size: .80rem;
            font-weight: 700;
            color: {c["muted"]} !important;
            margin: 10px 0 6px 0;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}

        .page-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 16px 18px;
            border: 1px solid {c["header_border"]};
            border-radius: 18px;
            background: linear-gradient(180deg, {c["header_bg_1"]} 0%, {c["header_bg_2"]} 100%);
            margin-bottom: 16px;
        }}

        .page-header-icon {{
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            background: {c["header_icon_bg"]};
            border: 1px solid {c["header_icon_border"]};
            flex-shrink: 0;
        }}

        .page-header-title {{
            margin: 0;
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.1;
            color: {c["header_title"]} !important;
        }}

        .page-header-sub {{
            margin-top: 4px;
            color: {c["header_sub"]} !important;
            font-size: .95rem;
        }}

        .paper-title {{
            text-align: center;
            font-weight: 700;
            font-size: 1.05rem;
            margin: 18px 0 8px 0;
            color: {c["paper_title"]} !important;
        }}

        .status-box {{
            padding: 12px 14px;
            border-radius: 14px;
            background: {c["status_bg"]};
            border: 1px solid {c["status_border"]};
            font-weight: 600;
            color: {c["status_text"]} !important;
        }}
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