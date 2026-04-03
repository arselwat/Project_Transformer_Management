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
from core.datahub import get_current_failures_df, get_failures_meta
from core.reliability.organigram import analyze_ttf_pipeline
from core.ui import render_shell, render_page_header

try:
    from core.datahub import get_pipeline_inputs
except Exception:
    def get_pipeline_inputs(asset_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "asset_id": asset_id,
            "thermal_df": None,
            "thermal_config": None,
            "alpha": 0.05,
        }

try:
    from core.reliability.reporting_merged import export_merged_report_pdf
except Exception as error:
    export_merged_report_pdf = None
    report_error_message = str(error)
else:
    report_error_message = None


st.set_page_config(page_title="Indicateurs", page_icon="📊", layout="wide")
require_login()

render_shell("pages/2_Indicateurs_verified.py")
render_page_header(
    "Indicateurs",
    "Tests de tendance, dépendance, paramètres fiabilistes, résultats thermiques et courbes détaillées.",
    "📊",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def format_number(value: Any, decimals: int = 2, default: str = "—") -> str:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return default
        return f"{numeric_value:.{decimals}f}"
    except Exception:
        return default


def series_to_positive_list(series: pd.Series) -> Optional[list[float]]:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    numeric_values = numeric_values[numeric_values > 0]
    if numeric_values.empty:
        return None
    return numeric_values.astype(float).tolist()


def sanitize_thermal_config(configuration: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(configuration, dict) or not configuration:
        return None

    allowed_keys = {
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
        "dt_hours",
    }

    sanitized_configuration: Dict[str, Any] = {}
    for key, value in configuration.items():
        if key in allowed_keys and pd.notna(value):
            sanitized_configuration[key] = value

    return sanitized_configuration or None


def sanitize_thermal_dataframe(dataframe: Any) -> Optional[pd.DataFrame]:
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return None

    sanitized_dataframe = dataframe.copy()
    sanitized_dataframe.columns = [str(column).strip() for column in sanitized_dataframe.columns]

    for removable_column in ["asset_id", "equipment_code"]:
        if removable_column in sanitized_dataframe.columns:
            sanitized_dataframe = sanitized_dataframe.drop(columns=[removable_column])

    return sanitized_dataframe if not sanitized_dataframe.empty else None


def get_pipeline_bundle(equipment_code: str) -> Dict[str, Any]:
    try:
        bundle = get_pipeline_inputs(asset_id=str(equipment_code))
        if not isinstance(bundle, dict):
            bundle = {}
    except Exception:
        bundle = {}

    bundle["thermal_df"] = sanitize_thermal_dataframe(bundle.get("thermal_df"))
    bundle["thermal_config"] = sanitize_thermal_config(bundle.get("thermal_config"))
    bundle.setdefault("alpha", 0.05)
    bundle.setdefault("asset_id", str(equipment_code))
    return bundle


DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi de probabilité retenue",
    "MTTF_h": "Temps moyen avant défaillance (heures)",
    "MTBF_h": "Temps moyen entre défaillances (heures)",
    "MTTR_h": "Temps moyen de réparation (heures)",
    "availability_pct": "Disponibilité intrinsèque (%)",
    "beta": "Paramètre bêta",
    "eta_h": "Paramètre êta (heures)",
    "gamma_h": "Paramètre gamma (heures)",
    "theta_HS_max": "Température maximale du point chaud (°C)",
    "FAA_max": "Facteur maximal d’accélération du vieillissement",
    "loss_of_life_pct": "Perte de vie (%)",
    "timestamp": "Date et heure",
    "temp_amb_C": "Température ambiante (°C)",
    "charge_pct": "Charge (%)",
    "K": "Facteur de charge",
    "etat_ventilateurs": "État des ventilateurs",
    "Delta_theta_TO": "Élévation de température top-oil (°C)",
    "Delta_theta_H": "Élévation de température du point chaud (°C)",
    "theta_TO_est_C": "Température top-oil estimée (°C)",
    "theta_HS_est_C": "Température du point chaud estimée (°C)",
    "FAA": "Facteur d’accélération du vieillissement",
    "dt_h_step": "Pas de temps (heures)",
    "aging_hours_step": "Heures de vieillissement sur le pas",
    "aging_hours_cum": "Heures de vieillissement cumulées",
    "life_consumed_pct_cum": "Perte de vie cumulée (%)",
    "remaining_life_pct": "Vie résiduelle estimée (%)",
    "mk_p": "Valeur p du test de Mann-Kendall",
    "mk_direction": "Sens du test de Mann-Kendall",
    "laplace_p": "Valeur p du test de Laplace",
    "laplace_direction": "Sens du test de Laplace",
    "spearman_r": "Coefficient de Spearman",
    "spearman_p": "Valeur p du test de Spearman",
}


def rename_columns_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe
    renamed_dataframe = dataframe.copy()
    renamed_dataframe = renamed_dataframe.rename(columns={
        column: DISPLAY_COLUMN_NAMES.get(column, column)
        for column in renamed_dataframe.columns
    })
    return renamed_dataframe


DETAIL_TABLE_LABELS = {
    "trend_results": "Résultats des tests de tendance",
    "dependence_results": "Résultats des tests de dépendance",
    "process_choice": "Décision sur le processus fiabiliste",
    "fit_candidates": "Comparaison des lois candidates",
    "reliability_summary": "Synthèse fiabiliste",
    "thermal_summary": "Synthèse thermique",
    "thermal_table_dataset": "Résumé de la série thermique utilisée",
    "thermal_table_params": "Paramètres du modèle thermique",
    "thermal_table_indicators": "Indicateurs thermiques calculés",
    "thermal_daily": "Résumé journalier thermique",
    "thermal_top5_days": "Jours les plus critiques",
}


def get_distribution_and_parameters(reliability_result: Dict[str, Any]):
    if reliability_result.get("model") != "RP":
        return None, None

    distribution_name = reliability_result.get("distribution")
    raw_parameters = (reliability_result.get("params") or {}).get("raw")
    if not raw_parameters:
        return None, None

    if distribution_name == "expon":
        return sst.expon, raw_parameters
    if distribution_name == "norm":
        return sst.norm, raw_parameters
    if distribution_name == "lognorm":
        return sst.lognorm, raw_parameters
    if distribution_name in {"weibull_2p", "weibull_3p"}:
        return sst.weibull_min, raw_parameters

    return None, None


def compute_reliability_curve(
    reliability_result: Dict[str, Any],
    time_axis: np.ndarray,
    curve_name: str,
) -> Optional[np.ndarray]:
    distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
    if distribution_object is None or distribution_parameters is None:
        return None

    try:
        if curve_name == "reliability":
            values = distribution_object.sf(time_axis, *distribution_parameters)
        elif curve_name == "cumulative_distribution":
            values = distribution_object.cdf(time_axis, *distribution_parameters)
        elif curve_name == "density":
            values = distribution_object.pdf(time_axis, *distribution_parameters)
        elif curve_name == "failure_rate":
            survival_values = distribution_object.sf(time_axis, *distribution_parameters)
            density_values = distribution_object.pdf(time_axis, *distribution_parameters)
            values = np.divide(
                density_values,
                survival_values,
                out=np.full_like(density_values, np.nan, dtype=float),
                where=survival_values > 1e-12,
            )
        else:
            return None
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def export_tables_to_excel(results_by_equipment: Dict[str, Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []

        for equipment_code, result in results_by_equipment.items():
            tables = result.get("tables") or {}
            for table_name, table_dataframe in tables.items():
                if isinstance(table_dataframe, pd.DataFrame) and not table_dataframe.empty:
                    readable_name = DETAIL_TABLE_LABELS.get(table_name, table_name)
                    sheet_name = f"{str(equipment_code)[:14]}_{readable_name}"[:31]
                    rename_columns_for_display(table_dataframe).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                    )

            thermal_result = result.get("thermal")
            thermal_timeseries = (thermal_result or {}).get("timeseries")
            if isinstance(thermal_timeseries, pd.DataFrame) and not thermal_timeseries.empty:
                sheet_name = f"{str(equipment_code)[:14]}_serie_thermique"[:31]
                rename_columns_for_display(thermal_timeseries).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

            reliability_result = result.get("reliability", {}) or {}
            indicators = reliability_result.get("indicators", {}) or {}
            thermal_summary = ((thermal_result or {}).get("summary", {}) if thermal_result else {}) or {}

            summary_rows.append(
                {
                    "equipment_code": equipment_code,
                    "model": reliability_result.get("model"),
                    "process_variant": reliability_result.get("process_variant"),
                    "distribution": reliability_result.get("distribution"),
                    "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
                    "MTBF_h": indicators.get("mtbf_h"),
                    "MTTR_h": indicators.get("mttr_h"),
                    "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
                    "beta": (reliability_result.get("params") or {}).get("beta"),
                    "eta_h": (reliability_result.get("params") or {}).get("eta"),
                    "gamma_h": (reliability_result.get("params") or {}).get("gamma"),
                    "theta_HS_max": thermal_summary.get("theta_hs_max"),
                    "FAA_max": thermal_summary.get("faa_max"),
                    "loss_of_life_pct": thermal_summary.get("loss_of_life_pct"),
                }
            )

        if summary_rows:
            rename_columns_for_display(pd.DataFrame(summary_rows)).to_excel(
                writer,
                sheet_name="Synthese",
                index=False,
            )

    buffer.seek(0)
    return buffer.getvalue()


def plot_single_series(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    y_label: str,
):
    if y_column not in dataframe.columns:
        st.info(f"La variable « {y_column} » n’est pas disponible.")
        return

    plot_dataframe = dataframe[[x_column, y_column]].dropna().copy()
    if plot_dataframe.empty:
        st.info("Aucune donnée disponible pour cette courbe.")
        return

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(plot_dataframe[x_column], plot_dataframe[y_column], linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Temps")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    st.pyplot(figure, clear_figure=True)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
failures_meta = get_failures_meta()
source_dataframe = get_current_failures_df()

if source_dataframe.empty:
    st.error("Aucun jeu de données actif. Va sur la page Sources de données.")
    st.stop()

st.success(
    f"Jeu de données actif | lignes={failures_meta.get('rows')} | empreinte={failures_meta.get('hash')} | source={failures_meta.get('source')}"
)

if "equipment_code" not in source_dataframe.columns or "ttf_h" not in source_dataframe.columns:
    st.error("Le jeu de données doit contenir au moins les colonnes equipment_code et ttf_h.")
    st.stop()

select_col_1, select_col_2 = st.columns([2, 1])
with select_col_1:
    all_equipment_codes = sorted(source_dataframe["equipment_code"].astype(str).unique().tolist())
    selected_equipment_codes = st.multiselect(
        "Équipements analysés",
        options=all_equipment_codes,
        default=all_equipment_codes[: min(5, len(all_equipment_codes))],
    )

with select_col_2:
    alpha_value = st.number_input(
        "Seuil alpha",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.005,
    )

if not selected_equipment_codes:
    st.info("Sélectionne au moins un équipement.")
    st.stop()


# -------------------------------------------------------------------
# Analyse
# -------------------------------------------------------------------
results_by_equipment: Dict[str, Dict[str, Any]] = {}
summary_rows: list[dict[str, Any]] = []
equipment_ready_for_reliability_curves: list[str] = []
equipment_without_reliability_curves: list[str] = []

for equipment_code in selected_equipment_codes:
    equipment_dataframe = source_dataframe[source_dataframe["equipment_code"].astype(str) == str(equipment_code)].copy()
    time_to_failure_list = series_to_positive_list(equipment_dataframe["ttf_h"])
    if not time_to_failure_list or len(time_to_failure_list) < 3:
        continue

    repair_time_list = None
    if "duree_rep_h" in equipment_dataframe.columns:
        repair_time_list = series_to_positive_list(equipment_dataframe["duree_rep_h"])

    bundle = get_pipeline_bundle(str(equipment_code))
    thermal_dataframe = bundle.get("thermal_df")
    thermal_configuration = bundle.get("thermal_config")

    try:
        result = analyze_ttf_pipeline(
            ttf_series=time_to_failure_list,
            alpha=float(alpha_value),
            repair_series=repair_time_list,
            thermal_df=thermal_dataframe,
            thermal_config=thermal_configuration,
        )
    except Exception as error:
        st.warning(f"{equipment_code} : {error}")
        continue

    results_by_equipment[str(equipment_code)] = result

    reliability_result = result.get("reliability", {}) or {}
    indicators = reliability_result.get("indicators", {}) or {}
    thermal_result = result.get("thermal")
    thermal_summary = ((thermal_result or {}).get("summary", {}) if thermal_result else {}) or {}

    summary_rows.append(
        {
            "equipment_code": equipment_code,
            "model": reliability_result.get("model"),
            "process_variant": reliability_result.get("process_variant"),
            "distribution": reliability_result.get("distribution"),
            "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
            "MTBF_h": indicators.get("mtbf_h"),
            "MTTR_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
            "beta": (reliability_result.get("params") or {}).get("beta"),
            "eta_h": (reliability_result.get("params") or {}).get("eta"),
            "gamma_h": (reliability_result.get("params") or {}).get("gamma"),
            "theta_HS_max": thermal_summary.get("theta_hs_max"),
            "FAA_max": thermal_summary.get("faa_max"),
            "loss_of_life_pct": thermal_summary.get("loss_of_life_pct"),
        }
    )

    if reliability_result.get("model") == "RP" and reliability_result.get("distribution") in {"expon", "norm", "lognorm", "weibull_2p", "weibull_3p"}:
        equipment_ready_for_reliability_curves.append(str(equipment_code))
    else:
        equipment_without_reliability_curves.append(str(equipment_code))

if not results_by_equipment:
    st.error("Pas assez de temps entre défaillances exploitables (au moins 3 valeurs positives).")
    st.stop()

summary_dataframe = pd.DataFrame(summary_rows).sort_values("equipment_code").reset_index(drop=True)
selected_equipment_for_detail = st.selectbox(
    "Équipement détaillé",
    options=list(results_by_equipment.keys()),
    index=0,
)
selected_result = results_by_equipment[selected_equipment_for_detail]
selected_tables = selected_result.get("tables", {}) or {}
selected_thermal_result = selected_result.get("thermal")


# -------------------------------------------------------------------
# Top summary
# -------------------------------------------------------------------
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Nombre d’équipements analysés", len(results_by_equipment))
with metric_col_2:
    total_failure_count = int(len(source_dataframe[source_dataframe["equipment_code"].astype(str).isin(list(results_by_equipment.keys()))]))
    st.metric("Nombre total d’enregistrements de défaillance", total_failure_count)
with metric_col_3:
    availability_series = summary_dataframe["availability_pct"].dropna()
    st.metric(
        "Disponibilité moyenne",
        format_number(availability_series.mean(), 2) if not availability_series.empty else "—",
    )
with metric_col_4:
    hotspot_series = summary_dataframe["theta_HS_max"].dropna()
    st.metric(
        "Température maximale du point chaud",
        format_number(hotspot_series.max(), 2) if not hotspot_series.empty else "—",
    )


page_tabs = st.tabs([
    "Vue d’ensemble",
    "Tendance",
    "Dépendance",
    "Fiabilité",
    "Thermique",
    "Courbes",
    "Téléchargements",
])

with page_tabs[0]:
    st.subheader("Vue d’ensemble")
    st.dataframe(rename_columns_for_display(summary_dataframe), use_container_width=True, hide_index=True)

with page_tabs[1]:
    st.subheader(f"Tendance — {selected_equipment_for_detail}")
    trend_dataframe = selected_tables.get("trend_results", pd.DataFrame())
    if trend_dataframe.empty:
        st.info("Aucun tableau de tendance disponible.")
    else:
        st.dataframe(rename_columns_for_display(trend_dataframe), use_container_width=True, hide_index=True)

with page_tabs[2]:
    st.subheader(f"Dépendance — {selected_equipment_for_detail}")
    dependence_dataframe = selected_tables.get("dependence_results", pd.DataFrame())
    if dependence_dataframe.empty:
        st.info("Aucun tableau de dépendance disponible.")
    else:
        st.dataframe(rename_columns_for_display(dependence_dataframe), use_container_width=True, hide_index=True)

with page_tabs[3]:
    st.subheader(f"Fiabilité — {selected_equipment_for_detail}")

    reliability_view = summary_dataframe[
        summary_dataframe["equipment_code"] == selected_equipment_for_detail
    ][
        [
            "equipment_code",
            "model",
            "process_variant",
            "distribution",
            "MTTF_h",
            "MTBF_h",
            "MTTR_h",
            "availability_pct",
            "beta",
            "eta_h",
            "gamma_h",
        ]
    ].copy()
    st.dataframe(rename_columns_for_display(reliability_view), use_container_width=True, hide_index=True)

    with st.expander("Voir la décision sur le processus fiabiliste"):
        process_dataframe = selected_tables.get("process_choice", pd.DataFrame())
        if process_dataframe.empty:
            st.info("Aucun tableau de processus disponible.")
        else:
            st.dataframe(rename_columns_for_display(process_dataframe), use_container_width=True, hide_index=True)

    with st.expander("Voir la comparaison des lois candidates"):
        fit_dataframe = selected_tables.get("fit_candidates", pd.DataFrame())
        if fit_dataframe.empty:
            st.info("Aucun détail d’ajustement disponible.")
        else:
            st.dataframe(rename_columns_for_display(fit_dataframe), use_container_width=True, hide_index=True)

    with st.expander("Voir la synthèse fiabiliste détaillée"):
        reliability_summary_dataframe = selected_tables.get("reliability_summary", pd.DataFrame())
        if reliability_summary_dataframe.empty:
            st.info("Aucune synthèse fiabiliste disponible.")
        else:
            st.dataframe(rename_columns_for_display(reliability_summary_dataframe), use_container_width=True, hide_index=True)

with page_tabs[4]:
    st.subheader(f"Thermique — {selected_equipment_for_detail}")

    thermal_view = summary_dataframe[
        summary_dataframe["equipment_code"] == selected_equipment_for_detail
    ][
        [
            "equipment_code",
            "theta_HS_max",
            "FAA_max",
            "loss_of_life_pct",
        ]
    ].copy()

    if thermal_view[["theta_HS_max", "FAA_max", "loss_of_life_pct"]].isna().all().all():
        st.info("Aucune donnée thermique disponible pour cet équipement.")
    else:
        st.markdown("#### Vue synthétique")
        st.dataframe(rename_columns_for_display(thermal_view), use_container_width=True, hide_index=True)

        st.markdown("#### Tous les tableaux thermiques")
        for table_key in [
            "thermal_summary",
            "thermal_table_dataset",
            "thermal_table_params",
            "thermal_table_indicators",
            "thermal_daily",
            "thermal_top5_days",
        ]:
            thermal_table = selected_tables.get(table_key, pd.DataFrame())
            if isinstance(thermal_table, pd.DataFrame) and not thermal_table.empty:
                st.markdown(f"##### {DETAIL_TABLE_LABELS.get(table_key, table_key)}")
                st.dataframe(rename_columns_for_display(thermal_table), use_container_width=True, hide_index=True)

        thermal_timeseries = (selected_thermal_result or {}).get("timeseries")
        if isinstance(thermal_timeseries, pd.DataFrame) and not thermal_timeseries.empty:
            st.markdown("#### Série temporelle thermique calculée")
            st.dataframe(
                rename_columns_for_display(thermal_timeseries),
                use_container_width=True,
                hide_index=True,
            )

with page_tabs[5]:
    st.subheader("Courbes")

    if equipment_without_reliability_curves:
        st.info(
            "Les courbes analytiques de fiabilité ne sont pas disponibles pour : "
            + ", ".join(equipment_without_reliability_curves)
        )

    maximum_time_value = float(source_dataframe[source_dataframe["equipment_code"].astype(str).isin(list(results_by_equipment.keys()))]["ttf_h"].max())
    maximum_time_value = max(1000.0, maximum_time_value if np.isfinite(maximum_time_value) and maximum_time_value > 0 else 1000.0)
    time_axis = np.linspace(1e-6, maximum_time_value, 300)

    def plot_multiple_reliability_curves(axis, curve_name: str, title: str, y_label: str):
        plotted_count = 0
        for equipment_code in equipment_ready_for_reliability_curves:
            reliability_result = results_by_equipment[equipment_code]["reliability"]
            curve_values = compute_reliability_curve(reliability_result, time_axis, curve_name)
            if curve_values is None:
                continue
            axis.plot(
                time_axis,
                curve_values,
                label=f"{equipment_code} ({reliability_result.get('distribution')})",
                linewidth=2,
            )
            plotted_count += 1

        axis.set_title(title)
        axis.set_xlabel("Temps (heures)")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        if plotted_count:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "Aucune courbe disponible",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    curve_tabs = st.tabs([
        "Fiabilité",
        "Fonction de répartition",
        "Densité",
        "Taux de défaillance",
        "Température ambiante",
        "Charge",
        "État des ventilateurs",
        "Élévation top-oil",
        "Élévation du point chaud",
        "Température top-oil estimée",
        "Température du point chaud estimée",
        "Facteur d’accélération du vieillissement",
        "Vieillissement sur chaque pas",
        "Vieillissement cumulé",
        "Perte de vie cumulée et vie résiduelle",
    ])

    with curve_tabs[0]:
        figure, axis = plt.subplots()
        plot_multiple_reliability_curves(axis, "reliability", "Courbe de fiabilité", "Fiabilité")
        st.pyplot(figure, clear_figure=True)

    with curve_tabs[1]:
        figure, axis = plt.subplots()
        plot_multiple_reliability_curves(axis, "cumulative_distribution", "Fonction de répartition", "Probabilité cumulée")
        st.pyplot(figure, clear_figure=True)

    with curve_tabs[2]:
        figure, axis = plt.subplots()
        plot_multiple_reliability_curves(axis, "density", "Densité de probabilité", "Densité")
        st.pyplot(figure, clear_figure=True)

    with curve_tabs[3]:
        figure, axis = plt.subplots()
        plot_multiple_reliability_curves(axis, "failure_rate", "Taux instantané de défaillance", "Taux de défaillance")
        st.pyplot(figure, clear_figure=True)

    thermal_timeseries = (selected_thermal_result or {}).get("timeseries")
    if isinstance(thermal_timeseries, pd.DataFrame) and not thermal_timeseries.empty:
        thermal_timeseries = thermal_timeseries.copy()
        thermal_timeseries["timestamp"] = pd.to_datetime(thermal_timeseries["timestamp"])

    with curve_tabs[4]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "temp_amb_C",
                "Température ambiante au cours du temps",
                "Température (°C)",
            )

    with curve_tabs[5]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            if "charge_pct" in thermal_timeseries.columns:
                plot_single_series(
                    thermal_timeseries,
                    "timestamp",
                    "charge_pct",
                    "Charge du transformateur au cours du temps",
                    "Charge (%)",
                )
            else:
                plot_single_series(
                    thermal_timeseries,
                    "timestamp",
                    "K",
                    "Facteur de charge au cours du temps",
                    "Facteur de charge",
                )

    with curve_tabs[6]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "etat_ventilateurs",
                "État des ventilateurs au cours du temps",
                "État des ventilateurs",
            )

    with curve_tabs[7]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "Delta_theta_TO",
                "Élévation de température top-oil au cours du temps",
                "Élévation de température (°C)",
            )

    with curve_tabs[8]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "Delta_theta_H",
                "Élévation de température du point chaud au cours du temps",
                "Élévation de température (°C)",
            )

    with curve_tabs[9]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "theta_TO_est_C",
                "Température top-oil estimée au cours du temps",
                "Température (°C)",
            )

    with curve_tabs[10]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "theta_HS_est_C",
                "Température du point chaud estimée au cours du temps",
                "Température (°C)",
            )

    with curve_tabs[11]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "FAA",
                "Facteur d’accélération du vieillissement au cours du temps",
                "Facteur",
            )

    with curve_tabs[12]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "aging_hours_step",
                "Heures de vieillissement sur chaque pas de temps",
                "Heures de vieillissement",
            )

    with curve_tabs[13]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            plot_single_series(
                thermal_timeseries,
                "timestamp",
                "aging_hours_cum",
                "Heures de vieillissement cumulées",
                "Heures de vieillissement cumulées",
            )

    with curve_tabs[14]:
        if thermal_timeseries is None or thermal_timeseries.empty:
            st.info("Aucune courbe thermique disponible.")
        else:
            dual_figure, dual_axis = plt.subplots(figsize=(9, 4))

            plot_dataframe = thermal_timeseries[["timestamp", "life_consumed_pct_cum", "remaining_life_pct"]].dropna(how="all").copy()
            if plot_dataframe.empty:
                st.info("Aucune donnée disponible pour cette courbe.")
            else:
                if "life_consumed_pct_cum" in plot_dataframe.columns:
                    dual_axis.plot(
                        plot_dataframe["timestamp"],
                        plot_dataframe["life_consumed_pct_cum"],
                        linewidth=2,
                        label="Perte de vie cumulée (%)",
                    )
                if "remaining_life_pct" in plot_dataframe.columns:
                    dual_axis.plot(
                        plot_dataframe["timestamp"],
                        plot_dataframe["remaining_life_pct"],
                        linewidth=2,
                        label="Vie résiduelle estimée (%)",
                    )

                dual_axis.set_title("Perte de vie cumulée et vie résiduelle estimée")
                dual_axis.set_xlabel("Temps")
                dual_axis.set_ylabel("Pourcentage (%)")
                dual_axis.grid(True, alpha=0.3)
                dual_axis.legend()
                st.pyplot(dual_figure, clear_figure=True)

with page_tabs[6]:
    st.subheader("Téléchargements")

    excel_bytes = export_tables_to_excel(results_by_equipment)
    st.download_button(
        "Télécharger le fichier Excel des indicateurs",
        data=excel_bytes,
        file_name="indicateurs_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_merged_report_pdf is None:
        st.info("Le module PDF n’est pas disponible.")
        if report_error_message:
            st.caption(report_error_message)
    else:
        selected_source_dataframe = source_dataframe[source_dataframe["equipment_code"].astype(str).isin(selected_equipment_codes)].copy()

        if st.button("Générer le PDF", type="primary", use_container_width=True):
            try:
                try:
                    output_path = export_merged_report_pdf(
                        df=selected_source_dataframe,
                        out_dir=str(Path(__file__).resolve().parents[1] / "reports"),
                        title="Rapport — Indicateurs",
                        analysis_results=results_by_equipment,
                    )
                except TypeError:
                    output_path = export_merged_report_pdf(
                        df=selected_source_dataframe,
                        out_dir=str(Path(__file__).resolve().parents[1] / "reports"),
                        title="Rapport — Indicateurs",
                    )

                st.session_state["last_report_path"] = output_path
                st.success(f"PDF généré : {output_path}")
            except Exception as error:
                st.error(f"PDF : {error}")

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