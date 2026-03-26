from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats as sst

from core.security.auth import require_login
from core.datahub import get_current_failures_df, get_failures_meta, get_pipeline_inputs
from core.reliability.organigram import analyze_ttf_pipeline

try:
    from core.reliability.reporting_merged import export_merged_report_pdf
except Exception as e:
    export_merged_report_pdf = None
    _REPORT_ERR = str(e)
else:
    _REPORT_ERR = None


st.set_page_config(page_title="Indicateurs", page_icon="📊", layout="wide")
require_login()

st.title("📊 Indicateurs")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def fnum(x: Any, nd: int = 2, default: str = "—") -> str:
    try:
        if x is None:
            return default
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return f"{x:.{nd}f}"
    except Exception:
        return default


def _series_to_list(s: pd.Series) -> Optional[list[float]]:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return None
    return vals.astype(float).tolist()


def _get_dist_and_params(reliability: Dict[str, Any]):
    model = reliability.get("model")
    if model != "RP":
        return None, None

    name = reliability.get("distribution")
    params = (reliability.get("params") or {}).get("raw")
    if not params:
        return None, None

    if name == "expon":
        return sst.expon, params
    if name == "norm":
        return sst.norm, params
    if name == "lognorm":
        return sst.lognorm, params
    if name in {"weibull_2p", "weibull_3p"}:
        return sst.weibull_min, params
    return None, None


def _compute_curve(reliability: Dict[str, Any], t: np.ndarray, curve: str) -> Optional[np.ndarray]:
    dist, params = _get_dist_and_params(reliability)
    if dist is None or params is None:
        return None

    try:
        if curve == "R":
            y = dist.sf(t, *params)
        elif curve == "F":
            y = dist.cdf(t, *params)
        elif curve == "pdf":
            y = dist.pdf(t, *params)
        elif curve == "hazard":
            sf = dist.sf(t, *params)
            yy = dist.pdf(t, *params)
            y = np.divide(yy, sf, out=np.full_like(yy, np.nan, dtype=float), where=sf > 1e-12)
        else:
            return None
        return np.asarray(y, dtype=float)
    except Exception:
        return None


def _export_tables_xlsx(result_by_eq: Dict[str, Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []
        for eq, result in result_by_eq.items():
            tables = result.get("tables") or {}
            for name, table in tables.items():
                if isinstance(table, pd.DataFrame) and not table.empty:
                    sheet = f"{eq}_{name}"[:31]
                    table.to_excel(writer, sheet_name=sheet, index=False)

            rel = result.get("reliability", {}) or {}
            ind = rel.get("indicators", {}) or {}
            summary_rows.append(
                {
                    "equipment_code": eq,
                    "model": rel.get("model"),
                    "distribution": rel.get("distribution"),
                    "MTTF_h": ind.get("theoretical_mttf_h") or ind.get("empirical_mttf_h"),
                    "MTBF_h": ind.get("mtbf_h"),
                    "MTTR_h": ind.get("mttr_h"),
                    "availability": ind.get("availability_intrinsic"),
                }
            )

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)

    buffer.seek(0)
    return buffer.getvalue()


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
meta = get_failures_meta()
df_src = get_current_failures_df()

if df_src.empty:
    st.error("Aucun dataset actif. Va sur la page Sources.")
    st.stop()

st.success(f"Dataset actif | rows={meta.get('rows')} | hash={meta.get('hash')} | source={meta.get('source')}")

if "equipment_code" not in df_src.columns or "ttf_h" not in df_src.columns:
    st.error("Le dataset doit contenir equipment_code et ttf_h.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    eqs_all = sorted(df_src["equipment_code"].astype(str).unique().tolist())
    selected_eqs = st.multiselect("Équipements", options=eqs_all, default=eqs_all[: min(5, len(eqs_all))])
with c2:
    alpha = st.number_input("Seuil alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.005)

if not selected_eqs:
    st.info("Sélectionne au moins un équipement.")
    st.stop()


# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------
results_by: Dict[str, Dict[str, Any]] = {}
summary_rows: list[dict[str, Any]] = []
curve_ready_eqs: list[str] = []
skipped_curve_eqs: list[str] = []

for eq in selected_eqs:
    g = df_src[df_src["equipment_code"].astype(str) == str(eq)].copy()
    ttf_list = _series_to_list(g["ttf_h"])
    if not ttf_list or len(ttf_list) < 3:
        continue

    repair_list = None
    if "duree_rep_h" in g.columns:
        repair_list = _series_to_list(g["duree_rep_h"])

    bundle = get_pipeline_inputs(asset_id=str(eq))
    thermal_df = bundle.get("thermal_df")
    thermal_cfg = bundle.get("thermal_config")

    try:
        result = analyze_ttf_pipeline(
            ttf_series=ttf_list,
            alpha=float(alpha),
            repair_series=repair_list,
            thermal_df=thermal_df,
            thermal_config=thermal_cfg,
        )
    except Exception as e:
        st.warning(f"{eq} : {e}")
        continue

    results_by[str(eq)] = result

    rel = result.get("reliability", {}) or {}
    ind = rel.get("indicators", {}) or {}
    therm = result.get("thermal")
    therm_summary = (therm or {}).get("summary", {}) if therm else {}

    summary_rows.append(
        {
            "equipment_code": eq,
            "model": rel.get("model"),
            "distribution": rel.get("distribution"),
            "MTTF_h": ind.get("theoretical_mttf_h") or ind.get("empirical_mttf_h"),
            "MTBF_h": ind.get("mtbf_h"),
            "MTTR_h": ind.get("mttr_h"),
            "availability_pct": None if ind.get("availability_intrinsic") is None else 100.0 * float(ind.get("availability_intrinsic")),
            "beta": (rel.get("params") or {}).get("beta"),
            "eta_h": (rel.get("params") or {}).get("eta"),
            "gamma_h": (rel.get("params") or {}).get("gamma"),
            "theta_HS_max": therm_summary.get("theta_hs_max"),
            "FAA_max": therm_summary.get("faa_max"),
            "loss_of_life_pct": therm_summary.get("loss_of_life_pct"),
        }
    )

    if rel.get("model") == "RP" and rel.get("distribution") in {"expon", "norm", "lognorm", "weibull_2p", "weibull_3p"}:
        curve_ready_eqs.append(str(eq))
    else:
        skipped_curve_eqs.append(str(eq))

if not results_by:
    st.error("Pas assez de TTF exploitables (≥3).")
    st.stop()

summary_df = pd.DataFrame(summary_rows).sort_values("equipment_code").reset_index(drop=True)
detail_eq = st.selectbox("Équipement à détailler", options=list(results_by.keys()), index=0)
detail_result = results_by[detail_eq]
detail_tables = detail_result.get("tables", {}) or {}
detail_therm = detail_result.get("thermal")


# -------------------------------------------------------------------
# Top summary
# -------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Équipements", len(results_by))
with m2:
    st.metric("TTF total", int(len(df_src[df_src["equipment_code"].astype(str).isin(list(results_by.keys()))])))
with m3:
    avail = summary_df["availability_pct"].dropna()
    st.metric("Disponibilité moy.", fnum(avail.mean(), 2) if not avail.empty else "—")
with m4:
    hs = summary_df["theta_HS_max"].dropna()
    st.metric("θHS max", fnum(hs.max(), 2) if not hs.empty else "—")

st.dataframe(summary_df, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tabs = st.tabs([
    "Tendance",
    "Dépendance",
    "Fiabilité",
    "Thermique",
    "Courbes",
    "Téléchargements",
])

with tabs[0]:
    st.subheader(f"Tendance — {detail_eq}")
    df_trend = detail_tables.get("trend_results", pd.DataFrame())
    if df_trend.empty:
        st.info("Aucun tableau de tendance.")
    else:
        st.dataframe(df_trend, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader(f"Dépendance — {detail_eq}")
    df_dep = detail_tables.get("dependence_results", pd.DataFrame())
    if df_dep.empty:
        st.info("Aucun tableau de dépendance.")
    else:
        st.dataframe(df_dep, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader(f"Fiabilité — {detail_eq}")

    df_process = detail_tables.get("process_choice", pd.DataFrame())
    if not df_process.empty:
        st.dataframe(df_process, use_container_width=True, hide_index=True)

    df_rel = detail_tables.get("reliability_summary", pd.DataFrame())
    if df_rel.empty:
        st.info("Aucun tableau de fiabilité.")
    else:
        st.dataframe(df_rel, use_container_width=True, hide_index=True)

    with st.expander("Voir les candidats / ajustements"):
        df_fit = detail_tables.get("fit_candidates", pd.DataFrame())
        if df_fit.empty:
            st.info("Aucun détail d’ajustement.")
        else:
            st.dataframe(df_fit, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader(f"Thermique — {detail_eq}")

    if detail_therm is None:
        st.info("Aucune donnée thermique disponible pour cet équipement.")
    else:
        df_therm = detail_tables.get("thermal_summary", pd.DataFrame())
        if not df_therm.empty:
            st.dataframe(df_therm, use_container_width=True, hide_index=True)

        df_ind = detail_tables.get("thermal_table_indicators", pd.DataFrame())
        if not df_ind.empty:
            st.dataframe(df_ind, use_container_width=True, hide_index=True)

        with st.expander("Voir plus de détails thermiques"):
            for key in ["thermal_table_dataset", "thermal_table_params", "thermal_daily", "thermal_top5_days"]:
                dfx = detail_tables.get(key, pd.DataFrame())
                if isinstance(dfx, pd.DataFrame) and not dfx.empty:
                    st.markdown(f"**{key}**")
                    st.dataframe(dfx, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Courbes")

    if skipped_curve_eqs:
        st.info("Pas de courbe analytique pour : " + ", ".join(skipped_curve_eqs))

    tmax = float(df_src[df_src["equipment_code"].astype(str).isin(list(results_by.keys()))]["ttf_h"].max())
    tmax = max(1000.0, tmax if np.isfinite(tmax) and tmax > 0 else 1000.0)
    t = np.linspace(1e-6, tmax, 300)

    def multi_plot(ax, curve: str, title: str, ylabel: str):
        plotted = 0
        for eq in curve_ready_eqs:
            rel = results_by[eq]["reliability"]
            y = _compute_curve(rel, t, curve)
            if y is None:
                continue
            ax.plot(t, y, label=f"{eq} ({rel.get('distribution')})", linewidth=2)
            plotted += 1

        ax.set_title(title)
        ax.set_xlabel("Temps (h)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Aucune courbe disponible", ha="center", va="center", transform=ax.transAxes)

    curve_tabs = st.tabs(["R(t)", "F(t)", "f(t)", "h(t)", "Thermique"])
    with curve_tabs[0]:
        fig, ax = plt.subplots()
        multi_plot(ax, "R", "Fiabilité R(t)", "R(t)")
        st.pyplot(fig, clear_figure=True)

    with curve_tabs[1]:
        fig, ax = plt.subplots()
        multi_plot(ax, "F", "Fonction de répartition F(t)", "F(t)")
        st.pyplot(fig, clear_figure=True)

    with curve_tabs[2]:
        fig, ax = plt.subplots()
        multi_plot(ax, "pdf", "Densité f(t)", "f(t)")
        st.pyplot(fig, clear_figure=True)

    with curve_tabs[3]:
        fig, ax = plt.subplots()
        multi_plot(ax, "hazard", "Taux de défaillance h(t)", "h(t)")
        st.pyplot(fig, clear_figure=True)

    with curve_tabs[4]:
        if detail_therm is None:
            st.info("Aucune courbe thermique.")
        else:
            ts = detail_therm.get("timeseries")
            if isinstance(ts, pd.DataFrame) and not ts.empty:
                ts = ts.copy()
                ts["timestamp"] = pd.to_datetime(ts["timestamp"])

                c1, c2 = st.columns(2)
                with c1:
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(ts["timestamp"], ts["theta_HS_est_C"], label="θHS")
                    ax1.plot(ts["timestamp"], ts["theta_TO_est_C"], label="θTO")
                    ax1.set_title("Températures estimées")
                    ax1.set_xlabel("Temps")
                    ax1.set_ylabel("°C")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    st.pyplot(fig1, clear_figure=True)

                with c2:
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(ts["timestamp"], ts["FAA"], label="FAA")
                    ax2.set_title("FAA")
                    ax2.set_xlabel("Temps")
                    ax2.set_ylabel("FAA")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    st.pyplot(fig2, clear_figure=True)

with tabs[5]:
    st.subheader("Téléchargements")

    xlsx_bytes = _export_tables_xlsx(results_by)
    st.download_button(
        "Télécharger Excel",
        data=xlsx_bytes,
        file_name="indicateurs_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_merged_report_pdf is None:
        st.info("Module PDF non disponible.")
        if _REPORT_ERR:
            st.caption(_REPORT_ERR)
    else:
        df_sel = df_src[df_src["equipment_code"].astype(str).isin(selected_eqs)].copy()

        if st.button("Générer le PDF", type="primary", use_container_width=True):
            try:
                try:
                    path = export_merged_report_pdf(
                        df=df_sel,
                        out_dir=str(BASE_DIR / "reports"),
                        title="Rapport — Indicateurs",
                        analysis_results=results_by,
                    )
                except TypeError:
                    path = export_merged_report_pdf(
                        df=df_sel,
                        out_dir=str(BASE_DIR / "reports"),
                        title="Rapport — Indicateurs",
                    )

                st.session_state["last_report_path"] = path
                st.success(f"PDF généré : {path}")
            except Exception as e:
                st.error(f"PDF : {e}")

        pdf_path = st.session_state.get("last_report_path")
        if pdf_path and Path(pdf_path).exists():
            pdf_bytes = Path(pdf_path).read_bytes()
            st.download_button(
                "Télécharger le PDF",
                data=pdf_bytes,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )