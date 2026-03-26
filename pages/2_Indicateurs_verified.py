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
from core.datahub import (
    get_current_failures_df,
    get_failures_meta,
    get_project_meta,
)

try:
    from core.datahub import get_pipeline_inputs  # type: ignore
except Exception:
    get_pipeline_inputs = None

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

st.title("📊 Indicateurs — Fiabilité & Thermique")
st.caption(
    "Tests de tendance et de dépendance, choix du processus, ajustement des lois, "
    "indicateurs fiabilistes et modélisation thermique si des données projet sont disponibles."
)

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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


def _sanitize_thermal_config(cfg: Any) -> Optional[Dict[str, Any]]:
    """
    Garde uniquement les clés réellement utiles au modèle thermique.
    Empêche l'erreur du type:
    simulate_thermal_dynamic() got an unexpected keyword argument 'asset_id'
    """
    if not isinstance(cfg, dict) or not cfg:
        return None

    allowed = {
        "sn_mva",
        "R",
        "delta_to_r",
        "delta_h_r",
        "tau_to_min",
        "tau_w_min",
        "n_exp",
        "m_exp",
        "forced_tau_to_factor",
        "forced_delta_to_factor",
        "forced_delta_h_factor",
        "normal_insulation_life_h",
        "faa_limit",
        "lol_limit_hours",
        "dt_hours",
    }

    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in allowed and pd.notna(v):
            out[k] = v
    return out or None


def _sanitize_thermal_df(df: Any) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # on supprime juste les colonnes clairement non utiles
    for c in ["asset_id", "equipment_code"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    return out if not out.empty else None


def _get_pipeline_bundle(eq: str) -> dict[str, Any]:
    if callable(get_pipeline_inputs):
        try:
            bundle = get_pipeline_inputs(asset_id=str(eq))
            if isinstance(bundle, dict):
                bundle["thermal_config"] = _sanitize_thermal_config(bundle.get("thermal_config"))
                bundle["thermal_df"] = _sanitize_thermal_df(bundle.get("thermal_df"))
                return bundle
        except Exception:
            pass

    return {
        "asset_id": str(eq),
        "ttf_series": [],
        "repair_series": [],
        "thermal_df": None,
        "thermal_config": None,
        "alpha": 0.05,
        "project_meta": get_project_meta(),
    }


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


def _pipeline_str(result: Dict[str, Any]) -> str:
    rel = result.get("reliability", {}) or {}
    tests = rel.get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}
    dep = tests.get("dependence", {}) or {}
    good = rel.get("goodness", {}) or {}
    dec = rel.get("decision", {}) or {}
    return (
        f"TTF>0 → MK(p={fnum(mk.get('p'),3)}, dir={mk.get('direction','none')}) "
        f"→ Laplace(p={fnum(lap.get('p'),3)}, dir={lap.get('direction','none')}) "
        f"→ Dep(Spearman r={fnum(dep.get('spearman_r'),3)}, p={fnum(dep.get('spearman_p'),3)}) "
        f"→ Process={rel.get('model','?')} ; Dist={rel.get('distribution','?')} "
        f"; KS p={fnum(good.get('ks_p'),3)} ; Chi2 p={fnum(good.get('chi2_p'),3)} ; "
        f"Décision={dec.get('reason','—')}"
    )


def _export_tables_xlsx(result_by_eq: Dict[str, Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []
        for eq, result in result_by_eq.items():
            for name, table in (result.get("tables") or {}).items():
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


# ---------------------------------------------------------------------
# Dataset actif
# ---------------------------------------------------------------------
meta = get_failures_meta()
df_src = get_current_failures_df()

if df_src.empty:
    st.error("Aucun dataset actif. Va sur « Sources de données » et synchronise un dataset.")
    st.stop()

st.success(
    f"Dataset actif ✅ | rows={meta.get('rows')} | hash={meta.get('hash')} | source={meta.get('source')}"
)

if "equipment_code" not in df_src.columns or "ttf_h" not in df_src.columns:
    st.error("Le dataset actif doit contenir au minimum les colonnes `equipment_code` et `ttf_h`.")
    st.stop()

controls = st.columns([2, 1])
with controls[0]:
    eqs_all = sorted(df_src["equipment_code"].astype(str).unique().tolist())
    sel = st.multiselect("Équipements", options=eqs_all, default=eqs_all[: min(5, len(eqs_all))])
with controls[1]:
    alpha = st.number_input("Seuil alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.005)

if not sel:
    st.info("Sélectionne au moins un équipement.")
    st.stop()


# ---------------------------------------------------------------------
# Analyse intégrée par équipement
# ---------------------------------------------------------------------
results_by: Dict[str, Dict[str, Any]] = {}
summary_rows: list[dict[str, Any]] = []
curve_ready_eqs: list[str] = []
skipped_curve_eqs: list[str] = []

for eq in sel:
    g = df_src[df_src["equipment_code"].astype(str) == str(eq)].copy()
    ttf_list = _series_to_list(g["ttf_h"])
    if not ttf_list or len(ttf_list) < 3:
        continue

    repair_list = None
    if "duree_rep_h" in g.columns:
        repair_list = _series_to_list(g["duree_rep_h"])

    bundle = _get_pipeline_bundle(str(eq))
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
        results_by[str(eq)] = result

        rel = result.get("reliability", {}) or {}
        ind = rel.get("indicators", {}) or {}
        therm = result.get("thermal")
        therm_summary = (therm or {}).get("summary", {}) if therm else {}

        summary_rows.append(
            {
                "equipment_code": eq,
                "n_ttf": int(rel.get("cleaned_n", 0)),
                "model": rel.get("model"),
                "distribution": rel.get("distribution"),
                "MTTF_h": ind.get("theoretical_mttf_h") or ind.get("empirical_mttf_h"),
                "MTBF_h": ind.get("mtbf_h"),
                "MTTR_h": ind.get("mttr_h"),
                "availability": ind.get("availability_intrinsic"),
                "AIC": (rel.get("goodness") or {}).get("aic"),
                "KS_p": (rel.get("goodness") or {}).get("ks_p"),
                "Chi2_p": (rel.get("goodness") or {}).get("chi2_p"),
                "CvM_p": (rel.get("goodness") or {}).get("cvm_p"),
                "beta": (rel.get("params") or {}).get("beta"),
                "eta": (rel.get("params") or {}).get("eta"),
                "gamma": (rel.get("params") or {}).get("gamma"),
                "theta_HS_max": therm_summary.get("theta_hs_max"),
                "FAA_max": therm_summary.get("faa_max"),
                "loss_of_life_pct": therm_summary.get("loss_of_life_pct"),
            }
        )

        if rel.get("model") == "RP" and rel.get("distribution") in {"expon", "norm", "lognorm", "weibull_2p", "weibull_3p"}:
            curve_ready_eqs.append(str(eq))
        else:
            skipped_curve_eqs.append(str(eq))
    except Exception as e:
        st.warning(f"Analyse impossible pour {eq}: {e}")

if not results_by:
    st.error("Pas assez de TTF exploitables (≥3) pour les équipements sélectionnés.")
    st.stop()

summary_df = pd.DataFrame(summary_rows).sort_values("equipment_code").reset_index(drop=True)
detail_eq = st.selectbox("Équipement à détailler", options=list(results_by.keys()), index=0)
detail_result = results_by[detail_eq]
detail_tables = detail_result.get("tables", {}) or {}
detail_therm = detail_result.get("thermal")

st.subheader("📋 Synthèse globale")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Équipements analysés", len(results_by))
with metric_cols[1]:
    st.metric("TTF total", int(summary_df["n_ttf"].sum()))
with metric_cols[2]:
    avail_mean = summary_df["availability"].dropna()
    st.metric("Disponibilité moyenne", fnum(avail_mean.mean(), 4) if not avail_mean.empty else "—")
with metric_cols[3]:
    theta_max = summary_df["theta_HS_max"].dropna()
    st.metric("θHS max observé", fnum(theta_max.max(), 2) if not theta_max.empty else "—")

st.dataframe(summary_df, use_container_width=True, hide_index=True)

main_tabs = st.tabs([
    "📈 Courbes fiabilistes",
    "🧭 Tests & organigramme",
    "🌡️ Thermique",
    "📄 Exports",
])

with main_tabs[0]:
    st.caption("Les courbes analytiques R(t), F(t), f(t), h(t) sont tracées pour les équipements RP avec loi paramétrique iid retenue.")
    if skipped_curve_eqs:
        st.info(
            "Courbes analytiques non tracées pour : "
            + ", ".join(skipped_curve_eqs)
            + " (processus NHPP/BPP ou loi non paramétrique iid)."
        )

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

    curve_tabs = st.tabs(["R(t)", "F(t)", "f(t)", "h(t)"])
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

with main_tabs[1]:
    st.subheader(f"Détail — {detail_eq}")
    st.code(_pipeline_str(detail_result), language="text")

    rel_tabs = st.tabs([
        "Tendance",
        "Dépendance",
        "Processus",
        "Ajustements",
        "Résumé fiabiliste",
        "JSON brut",
    ])
    with rel_tabs[0]:
        st.dataframe(detail_tables.get("trend_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    with rel_tabs[1]:
        st.dataframe(detail_tables.get("dependence_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    with rel_tabs[2]:
        st.dataframe(detail_tables.get("process_choice", pd.DataFrame()), use_container_width=True, hide_index=True)
    with rel_tabs[3]:
        st.dataframe(detail_tables.get("fit_candidates", pd.DataFrame()), use_container_width=True, hide_index=True)
    with rel_tabs[4]:
        st.dataframe(detail_tables.get("reliability_summary", pd.DataFrame()), use_container_width=True, hide_index=True)
    with rel_tabs[5]:
        st.json(detail_result)

with main_tabs[2]:
    if detail_therm is None:
        st.info(
            "Aucune série thermique disponible pour cet équipement. La partie fiabilité reste exploitable. "
            "La partie thermique apparaîtra dès que la page Sources synchronisera thermal_timeseries et thermal_params."
        )
    else:
        therm_tabs = st.tabs([
            "Synthèse thermique",
            "Tables thermique",
            "Courbes thermique",
            "Journaliers & top 5",
        ])
        with therm_tabs[0]:
            st.dataframe(detail_tables.get("thermal_summary", pd.DataFrame()), use_container_width=True, hide_index=True)
        with therm_tabs[1]:
            st.markdown("**Jeu de données**")
            st.dataframe(detail_tables.get("thermal_table_dataset", pd.DataFrame()), use_container_width=True, hide_index=True)
            st.markdown("**Paramètres**")
            st.dataframe(detail_tables.get("thermal_table_params", pd.DataFrame()), use_container_width=True, hide_index=True)
            st.markdown("**Indicateurs**")
            st.dataframe(detail_tables.get("thermal_table_indicators", pd.DataFrame()), use_container_width=True, hide_index=True)
        with therm_tabs[2]:
            ts = detail_therm["timeseries"].copy()
            ts["timestamp"] = pd.to_datetime(ts["timestamp"])

            fig1, ax1 = plt.subplots(figsize=(11, 4))
            ax1.plot(ts["timestamp"], ts["theta_HS_est_C"], label="θHS estimée")
            ax1.plot(ts["timestamp"], ts["theta_TO_est_C"], label="θTO estimée")
            ax1.set_title(f"Températures estimées — {detail_eq}")
            ax1.set_xlabel("Temps")
            ax1.set_ylabel("Température (°C)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            st.pyplot(fig1, clear_figure=True)

            fig2, ax2 = plt.subplots(figsize=(11, 4))
            ax2.plot(ts["timestamp"], ts["FAA"], label="FAA")
            ax2.set_title(f"Facteur d'accélération du vieillissement — {detail_eq}")
            ax2.set_xlabel("Temps")
            ax2.set_ylabel("FAA")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

            fig3, ax3 = plt.subplots(figsize=(11, 4))
            ax3.plot(ts["timestamp"], ts["life_consumed_pct_cum"], label="Vie consommée cumulée (%)")
            ax3.plot(ts["timestamp"], ts["remaining_life_pct"], label="Vie restante (%)")
            ax3.set_title(f"Perte de vie cumulée / vie restante — {detail_eq}")
            ax3.set_xlabel("Temps")
            ax3.set_ylabel("Pourcentage (%)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            st.pyplot(fig3, clear_figure=True)
        with therm_tabs[3]:
            st.markdown("**Résumé journalier**")
            st.dataframe(detail_tables.get("thermal_daily", pd.DataFrame()), use_container_width=True, hide_index=True)
            st.markdown("**Top 5 jours critiques**")
            st.dataframe(detail_tables.get("thermal_top5_days", pd.DataFrame()), use_container_width=True, hide_index=True)

with main_tabs[3]:
    st.subheader("Téléchargements")
    xlsx_bytes = _export_tables_xlsx(results_by)
    st.download_button(
        "📥 Télécharger les tableaux Excel",
        data=xlsx_bytes,
        file_name="indicateurs_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_merged_report_pdf is None:
        st.info("Module core.reliability.reporting_merged non détecté ou incompatible.")
        if _REPORT_ERR:
            st.caption(f"Détail import: {_REPORT_ERR}")
    else:
        df_sel = df_src[df_src["equipment_code"].astype(str).isin(sel)].copy()
        if st.button("📄 Générer rapport PDF indicateurs", type="primary"):
            try:
                try:
                    path = export_merged_report_pdf(
                        df=df_sel,
                        out_dir=str(BASE_DIR / "reports"),
                        title="Rapport complet — Indicateurs",
                        analysis_results=results_by,
                    )
                except TypeError:
                    path = export_merged_report_pdf(
                        df=df_sel,
                        out_dir=str(BASE_DIR / "reports"),
                        title="Rapport complet — Indicateurs",
                    )
                st.session_state["last_report_path"] = path
                st.success(f"PDF généré : {path}")
            except Exception as e:
                st.error(f"PDF : {e}")

        pdf_path = st.session_state.get("last_report_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Télécharger le rapport PDF",
                    data=f,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )