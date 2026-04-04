
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
    "Tests de tendance, dépendance, paramètres fiabilistes et courbes détaillées.",
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


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return default
        return numeric_value
    except Exception:
        return default


def series_to_positive_list(series: pd.Series) -> Optional[list[float]]:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    numeric_values = numeric_values[numeric_values > 0]
    if numeric_values.empty:
        return None
    return numeric_values.astype(float).tolist()


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
    "mk_p": "Valeur p du test de Mann-Kendall",
    "mk_direction": "Sens du test de Mann-Kendall",
    "laplace_p": "Valeur p du test de Laplace",
    "laplace_direction": "Sens du test de Laplace",
    "spearman_r": "Coefficient de Spearman",
    "spearman_p": "Valeur p du test de Spearman",
    "Test": "Test",
    "Statistique": "Statistique du test",
    "p_value": "Valeur p",
    "Décision": "Décision",
    "Direction": "Sens",
    "Méthode": "Méthode",
    "r": "Coefficient",
    "Dépendance": "Dépendance détectée",
    "Tendance": "Tendance détectée",
    "Direction tendance": "Sens global de la tendance",
    "Processus retenu": "Processus retenu",
    "Variant": "Variant du processus",
    "Hypothèse entité": "Hypothèse sur l’entité",
    "Justification": "Justification",
    "Modèle": "Modèle candidat",
    "Paramètres": "Paramètres",
    "Méthode estimation": "Méthode d’estimation",
    "LogLik": "Log-vraisemblance",
    "AIC": "Critère d'information d’Akaike",
    "KS p": "Valeur p du test de Kolmogorov-Smirnov",
    "Chi2 p": "Valeur p du test du chi carré",
    "CvM p": "Valeur p du test de Cramér-von Mises",
    "Acceptée": "Ajustement accepté",
    "Retenue": "Modèle retenu",
    "Processus": "Processus retenu",
    "Distribution": "Loi retenue",
    "Eta": "Paramètre êta",
    "Gamma": "Paramètre gamma",
    "Lambda_HPP (1/h)": "Taux constant du processus homogène (1/h)",
    "Mu": "Taux de base",
    "Alpha": "Paramètre alpha",
    "Beta_kernel": "Paramètre bêta du noyau",
    "Branch_ratio": "Ratio de branchement",
    "Ajustement accepté": "Ajustement accepté",
    "MTTF (h)": "Temps moyen avant défaillance (heures)",
    "MTBF (h)": "Temps moyen entre défaillances (heures)",
    "MTTR (h)": "Temps moyen de réparation (heures)",
    "Disponibilité": "Disponibilité intrinsèque",
    "Taux de défaillance moyen (1/h)": "Taux moyen de défaillance (1/h)",
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


def compute_rp_curve(
    reliability_result: Dict[str, Any],
    time_axis: np.ndarray,
    curve_name: str,
) -> Optional[np.ndarray]:
    distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
    if distribution_object is None or distribution_parameters is None:
        return None

    try:
        if curve_name == "survival":
            values = distribution_object.sf(time_axis, *distribution_parameters)
        elif curve_name == "cdf":
            values = distribution_object.cdf(time_axis, *distribution_parameters)
        elif curve_name == "pdf":
            values = distribution_object.pdf(time_axis, *distribution_parameters)
        elif curve_name == "hazard":
            survival_values = distribution_object.sf(time_axis, *distribution_parameters)
            density_values = distribution_object.pdf(time_axis, *distribution_parameters)
            values = np.divide(
                density_values,
                survival_values,
                out=np.full_like(density_values, np.nan, dtype=float),
                where=survival_values > 1e-12,
            )
        elif curve_name == "cum_mean_events":
            indicators = reliability_result.get("indicators", {}) or {}
            mean_time_between_failures = safe_float(
                indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
                None,
            )
            if mean_time_between_failures is None or mean_time_between_failures <= 0:
                return None
            values = time_axis / mean_time_between_failures
        else:
            return None

        return np.asarray(values, dtype=float)
    except Exception:
        return None


def compute_nhpp_curve(
    reliability_result: Dict[str, Any],
    time_axis: np.ndarray,
    curve_name: str,
) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    beta_value = safe_float(parameters.get("beta"), None)
    eta_value = safe_float(parameters.get("eta"), None)

    if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
        return None

    safe_time_axis = np.maximum(time_axis, 1e-6)

    mean_cumulative_events = (safe_time_axis / eta_value) ** beta_value
    intensity = (beta_value / eta_value) * ((safe_time_axis / eta_value) ** (beta_value - 1.0))
    survival_like = np.exp(-mean_cumulative_events)
    cumulative_probability = 1.0 - survival_like
    density_like = intensity * survival_like

    if curve_name == "survival":
        return survival_like
    if curve_name == "cdf":
        return cumulative_probability
    if curve_name == "pdf":
        return density_like
    if curve_name == "hazard":
        return intensity
    if curve_name == "cum_mean_events":
        return mean_cumulative_events

    return None


def compute_bpp_curve(
    reliability_result: Dict[str, Any],
    time_axis: np.ndarray,
    ttf_series: list[float],
    curve_name: str,
) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    mu_value = safe_float(parameters.get("mu"), None)
    alpha_value = safe_float(parameters.get("alpha"), None)
    beta_kernel_value = safe_float(parameters.get("beta_kernel"), None)

    if mu_value is None or alpha_value is None or beta_kernel_value is None:
        return None
    if mu_value < 0 or alpha_value < 0 or beta_kernel_value <= 0:
        return None

    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    if event_times.size == 0:
        return None

    safe_time_axis = np.maximum(time_axis, 1e-6)
    intensity = np.full_like(safe_time_axis, fill_value=mu_value, dtype=float)

    for event_time in event_times:
        mask = safe_time_axis >= event_time
        if np.any(mask):
            intensity[mask] += alpha_value * np.exp(-beta_kernel_value * (safe_time_axis[mask] - event_time))

    cumulative_intensity = np.zeros_like(safe_time_axis, dtype=float)
    if len(safe_time_axis) > 1:
        delta = np.diff(safe_time_axis)
        trapezoids = 0.5 * (intensity[1:] + intensity[:-1]) * delta
        cumulative_intensity[1:] = np.cumsum(trapezoids)

    survival_like = np.exp(-cumulative_intensity)
    cumulative_probability = 1.0 - survival_like
    density_like = intensity * survival_like

    if curve_name == "survival":
        return survival_like
    if curve_name == "cdf":
        return cumulative_probability
    if curve_name == "pdf":
        return density_like
    if curve_name == "hazard":
        return intensity
    if curve_name == "cum_mean_events":
        return cumulative_intensity

    return None


def compute_model_based_curve(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
    time_axis: np.ndarray,
    curve_name: str,
) -> Optional[np.ndarray]:
    model_name = str(reliability_result.get("model") or "").upper()

    if model_name == "RP":
        return compute_rp_curve(reliability_result, time_axis, curve_name)
    if model_name == "NHPP":
        return compute_nhpp_curve(reliability_result, time_axis, curve_name)
    if model_name == "BPP":
        return compute_bpp_curve(reliability_result, time_axis, ttf_series, curve_name)

    return None


def build_curve_explanation(
    reliability_result: Dict[str, Any],
    curve_name: str,
) -> str:
    model_name = str(reliability_result.get("model") or "").upper()

    if curve_name == "survival":
        if model_name == "RP":
            return "R(t) représente la probabilité qu’aucune défaillance ne survienne avant le temps t."
        if model_name == "NHPP":
            return "Ici, la courbe joue le rôle d’une probabilité de non-événement jusqu’au temps t dans un processus non homogène."
        return "Ici, la courbe représente une probabilité conditionnelle de non-événement compte tenu de la dépendance entre événements."

    if curve_name == "cdf":
        if model_name == "RP":
            return "F(t) représente la probabilité cumulée qu’une défaillance soit déjà survenue avant le temps t."
        if model_name == "NHPP":
            return "Ici, F(t) représente la probabilité cumulée d’avoir observé au moins un événement avant le temps t."
        return "Ici, F(t) représente une probabilité cumulée conditionnelle d’apparition d’au moins un événement."

    if curve_name == "pdf":
        if model_name == "RP":
            return "f(t) montre dans quelles zones de temps les défaillances sont les plus probables."
        if model_name == "NHPP":
            return "Ici, f(t) représente une densité du premier événement dans le processus non homogène."
        return "Ici, f(t) représente une densité conditionnelle d’apparition d’un événement."

    if curve_name == "hazard":
        if model_name == "RP":
            return "h(t) est le taux instantané de défaillance : plus il est élevé, plus le risque immédiat de défaillance est fort."
        if model_name == "NHPP":
            return "λ(t) est l’intensité instantanée du processus : elle montre si l’arrivée des événements s’accélère ou ralentit."
        return "λ(t) est l’intensité conditionnelle du processus avec dépendance : elle dépend aussi des événements passés."

    if curve_name == "cum_mean_events":
        if model_name == "RP":
            return "Cette courbe donne une approximation du nombre cumulé moyen de défaillances."
        if model_name == "NHPP":
            return "m(t) représente le nombre cumulé moyen d’événements attendu dans le modèle non homogène."
        return "Cette courbe représente l’intensité cumulée ou le cumul moyen conditionnel des événements."

    return "Courbe non documentée."


def build_time_horizon_for_equipment(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
) -> float:
    model_name = str(reliability_result.get("model") or "").upper()
    parameters = reliability_result.get("params", {}) or {}

    values = np.asarray(ttf_series, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size == 0:
        return 100.0

    max_ttf = float(np.max(values))
    mean_ttf = float(np.mean(values))
    q90_ttf = float(np.quantile(values, 0.90))

    eta_value = safe_float(parameters.get("eta"), None)

    base_horizon = max(
        50.0,
        max_ttf * 2.0,
        mean_ttf * 6.0,
        q90_ttf * 4.0,
    )

    if model_name == "RP":
        if eta_value is not None and eta_value > 0:
            return max(base_horizon, eta_value * 1.5)
        return base_horizon

    if model_name == "NHPP":
        nhpp_horizon = base_horizon
        if eta_value is not None and eta_value > 0:
            nhpp_horizon = min(base_horizon, eta_value * 1.25)
        return max(25.0, nhpp_horizon)

    if model_name == "BPP":
        return base_horizon

    return base_horizon


def build_group_time_horizon(
    results_by_equipment: Dict[str, Dict[str, Any]],
    ttf_series_by_equipment: Dict[str, list[float]],
    equipment_codes: list[str],
) -> float:
    horizons = []
    for equipment_code in equipment_codes:
        result = results_by_equipment.get(equipment_code, {})
        reliability_result = result.get("reliability", {}) or {}
        ttf_series = ttf_series_by_equipment.get(equipment_code, [])
        horizons.append(build_time_horizon_for_equipment(reliability_result, ttf_series))

    if not horizons:
        return 100.0

    if len(horizons) == 1:
        return float(horizons[0])

    return float(np.quantile(np.asarray(horizons, dtype=float), 0.75))


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

            reliability_result = result.get("reliability", {}) or {}
            indicators = reliability_result.get("indicators", {}) or {}

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

with st.expander("Comprendre les principales variables affichées sur cette page", expanded=False):
    st.markdown(
        """
**Temps moyen avant défaillance** : durée moyenne attendue avant une panne.

**Temps moyen entre défaillances** : durée moyenne séparant deux défaillances successives.

**Temps moyen de réparation** : durée moyenne nécessaire pour remettre l’équipement en service.

**Disponibilité intrinsèque** : part du temps où l’équipement est disponible.

**Paramètre bêta** : décrit la forme du vieillissement.
- inférieur à 1 : défauts précoces
- proche de 1 : comportement aléatoire
- supérieur à 1 : usure

**Paramètre êta** : durée de vie caractéristique estimée.

**Paramètre gamma** : éventuel décalage temporel dans le modèle.

**Processus retenu** :
- RP : renouvellement avec durées supposées indépendantes
- NHPP : processus évolutif dans le temps
- BPP : processus avec dépendance entre événements
        """
    )

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
ttf_series_by_equipment: Dict[str, list[float]] = {}
equipment_without_curves: list[str] = []

for equipment_code in selected_equipment_codes:
    equipment_dataframe = source_dataframe[source_dataframe["equipment_code"].astype(str) == str(equipment_code)].copy()
    time_to_failure_list = series_to_positive_list(equipment_dataframe["ttf_h"])
    if not time_to_failure_list or len(time_to_failure_list) < 3:
        continue

    ttf_series_by_equipment[str(equipment_code)] = time_to_failure_list

    repair_time_list = None
    if "duree_rep_h" in equipment_dataframe.columns:
        repair_time_list = series_to_positive_list(equipment_dataframe["duree_rep_h"])

    try:
        result = analyze_ttf_pipeline(
            ttf_series=time_to_failure_list,
            alpha=float(alpha_value),
            repair_series=repair_time_list,
        )
    except Exception as error:
        st.warning(f"{equipment_code} : {error}")
        continue

    results_by_equipment[str(equipment_code)] = result

    reliability_result = result.get("reliability", {}) or {}
    indicators = reliability_result.get("indicators", {}) or {}

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
        }
    )

    if str(reliability_result.get("model") or "").upper() not in {"RP", "NHPP", "BPP"}:
        equipment_without_curves.append(str(equipment_code))

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


# -------------------------------------------------------------------
# Top summary
# -------------------------------------------------------------------
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Nombre d’équipements analysés", len(results_by_equipment))
with metric_col_2:
    total_failure_count = int(
        len(source_dataframe[source_dataframe["equipment_code"].astype(str).isin(list(results_by_equipment.keys()))])
    )
    st.metric("Nombre total d’enregistrements de défaillance", total_failure_count)
with metric_col_3:
    availability_series = summary_dataframe["availability_pct"].dropna()
    st.metric(
        "Disponibilité moyenne",
        format_number(availability_series.mean(), 2) if not availability_series.empty else "—",
    )
with metric_col_4:
    beta_series = summary_dataframe["beta"].dropna()
    st.metric(
        "Bêta maximal observé",
        format_number(beta_series.max(), 2) if not beta_series.empty else "—",
    )


page_tabs = st.tabs([
    "Vue d’ensemble",
    "Tendance",
    "Dépendance",
    "Fiabilité",
    "Courbes fiabilistes",
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
    st.subheader("Courbes fiabilistes")

    if equipment_without_curves:
        st.info(
            "Certaines courbes n’ont pas pu être construites pour : "
            + ", ".join(equipment_without_curves)
        )

    compare_all_equipment = st.toggle(
        "Comparer tous les équipements sur les courbes",
        value=False,
        help="Décoche pour lire plus clairement les courbes de l’équipement sélectionné.",
    )

    if compare_all_equipment:
        equipment_codes_for_plots = list(results_by_equipment.keys())
    else:
        equipment_codes_for_plots = [selected_equipment_for_detail]

    time_horizon = build_group_time_horizon(
        results_by_equipment=results_by_equipment,
        ttf_series_by_equipment=ttf_series_by_equipment,
        equipment_codes=equipment_codes_for_plots,
    )
    time_axis = np.linspace(1e-6, time_horizon, 400)

    selected_reliability_result = (results_by_equipment.get(selected_equipment_for_detail, {}) or {}).get("reliability", {}) or {}

    curve_definitions = [
        (
            "survival",
            "Fiabilité R(t) ou probabilité de non-événement",
            "Probabilité",
            "Pour un processus de renouvellement, cette courbe correspond à la fiabilité R(t). "
            "Pour NHPP et BPP, elle est interprétée comme une probabilité de non-événement avant le temps t.",
        ),
        (
            "cdf",
            "Fonction de répartition F(t)",
            "Probabilité cumulée",
            "Cette courbe montre la probabilité cumulée d’avoir déjà observé au moins un événement avant le temps t.",
        ),
        (
            "pdf",
            "Densité f(t)",
            "Densité",
            "Cette courbe montre dans quelles zones de temps les événements sont les plus concentrés.",
        ),
        (
            "hazard",
            "Intensité h(t) ou λ(t)",
            "Intensité",
            "Pour RP, la courbe correspond au taux instantané de défaillance h(t). "
            "Pour NHPP et BPP, elle correspond à l’intensité λ(t) du processus.",
        ),
        (
            "cum_mean_events",
            "Nombre cumulé moyen m(t)",
            "Cumul moyen",
            "Cette courbe montre le cumul moyen attendu des événements au cours du temps.",
        ),
    ]

    def plot_multiple_model_curves(
        axis,
        curve_name: str,
        title: str,
        y_label: str,
    ):
        plotted_count = 0

        for equipment_code in equipment_codes_for_plots:
            result = results_by_equipment.get(equipment_code, {})
            reliability_result = result.get("reliability", {}) or {}
            equipment_ttf_series = ttf_series_by_equipment.get(equipment_code, [])

            curve_values = compute_model_based_curve(
                reliability_result=reliability_result,
                ttf_series=equipment_ttf_series,
                time_axis=time_axis,
                curve_name=curve_name,
            )
            if curve_values is None:
                continue

            label = (
                f"{equipment_code} | "
                f"{reliability_result.get('model', '—')} | "
                f"{reliability_result.get('distribution', '—')}"
            )
            axis.plot(time_axis, curve_values, label=label, linewidth=2)
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

    reliability_curve_tabs = st.tabs([
        "Fiabilité R(t)",
        "Fonction de répartition F(t)",
        "Densité f(t)",
        "Intensité h(t) ou λ(t)",
        "Nombre cumulé moyen m(t)",
    ])

    for curve_tab, curve_definition in zip(reliability_curve_tabs, curve_definitions):
        curve_name, title, y_label, generic_explanation = curve_definition
        with curve_tab:
            figure, axis = plt.subplots(figsize=(10, 5))
            plot_multiple_model_curves(axis, curve_name, title, y_label)
            st.pyplot(figure, clear_figure=True)

            if str(selected_reliability_result.get("model") or "").upper() == "NHPP":
                st.caption(
                    "L’échelle de l’axe du temps a été volontairement réduite pour le NHPP afin de rester dans une zone de lecture utile."
                )

            st.caption(generic_explanation)
            st.caption(
                f"Lecture pour l’équipement sélectionné ({selected_equipment_for_detail}) : "
                + build_curve_explanation(selected_reliability_result, curve_name)
            )

with page_tabs[5]:
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
        selected_source_dataframe = source_dataframe[
            source_dataframe["equipment_code"].astype(str).isin(selected_equipment_codes)
        ].copy()

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
