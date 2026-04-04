from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
from typing import Any, Dict, Optional, List, Tuple

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
    "Lecture ciblée du processus retenu, de la loi choisie, de l’ajustement, des indicateurs calculés et des courbes fiabilistes.",
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
    "distribution": "Loi choisie",
    "MTTF_h": "MTTF (h)",
    "MTBF_h": "MTBF (h)",
    "MTTR_h": "MTTR (h)",
    "availability_pct": "Disponibilité (%)",
    "beta": "Bêta",
    "eta_h": "Êta (h)",
    "gamma_h": "Gamma (h)",
    "mk_p": "p Mann-Kendall",
    "mk_direction": "Sens Mann-Kendall",
    "laplace_p": "p Laplace",
    "laplace_direction": "Sens Laplace",
    "spearman_r": "Coeff. Spearman",
    "spearman_p": "p Spearman",
    "Test": "Test",
    "Statistique": "Statistique",
    "p_value": "Valeur p",
    "Décision": "Décision",
    "Direction": "Sens",
    "Méthode": "Méthode",
    "r": "Coefficient",
    "Dépendance": "Dépendance",
    "Tendance": "Tendance",
    "Direction tendance": "Sens global",
    "Processus retenu": "Processus retenu",
    "Variant": "Variant",
    "Hypothèse entité": "Hypothèse sur l’entité",
    "Justification": "Justification",
    "Modèle": "Modèle candidat",
    "Paramètres": "Paramètres",
    "Méthode estimation": "Méthode d’estimation",
    "LogLik": "Log-vraisemblance",
    "AIC": "AIC",
    "KS p": "p Kolmogorov-Smirnov",
    "Chi2 p": "p Chi carré",
    "CvM p": "p Cramér-von Mises",
    "Acceptée": "Acceptée",
    "Retenue": "Retenue",
    "Processus": "Processus",
    "Distribution": "Loi",
    "Eta": "Êta",
    "Gamma": "Gamma",
    "Lambda_HPP (1/h)": "Lambda_HPP (1/h)",
    "Mu": "Mu",
    "Alpha": "Alpha",
    "Beta_kernel": "Bêta noyau",
    "Branch_ratio": "Ratio de branchement",
    "Ajustement accepté": "Ajustement accepté",
    "MTTF (h)": "MTTF (h)",
    "MTBF (h)": "MTBF (h)",
    "MTTR (h)": "MTTR (h)",
    "Disponibilité": "Disponibilité",
    "Taux de défaillance moyen (1/h)": "Taux moyen (1/h)",
    "t_ref_h": "Temps de référence t* (h)",
    "R_t_ref": "Fiabilité R(t*)",
    "F_t_ref": "Répartition F(t*)",
    "f_t_ref": "Densité f(t*)",
    "lambda_t_ref": "Taux λ(t*)",
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
    if str(reliability_result.get("model") or "").upper() != "RP":
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


def build_time_horizon_for_equipment(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
) -> float:
    values = np.asarray(ttf_series, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size == 0:
        return 100.0

    max_ttf = float(np.max(values))
    mean_ttf = float(np.mean(values))
    q90_ttf = float(np.quantile(values, 0.90))
    eta_value = safe_float((reliability_result.get("params") or {}).get("eta"), None)

    base_horizon = max(50.0, max_ttf * 2.0, mean_ttf * 6.0, q90_ttf * 4.0)
    if eta_value is not None and eta_value > 0:
        base_horizon = max(base_horizon, eta_value * 1.5)
    return base_horizon


def build_reference_time(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
) -> float:
    indicators = reliability_result.get("indicators", {}) or {}
    candidates = [
        safe_float(indicators.get("mtbf_h"), None),
        safe_float(indicators.get("theoretical_mttf_h"), None),
        safe_float(indicators.get("empirical_mttf_h"), None),
    ]
    if ttf_series:
        series_np = np.asarray(ttf_series, dtype=float)
        series_np = series_np[np.isfinite(series_np)]
        series_np = series_np[series_np > 0]
        if series_np.size:
            candidates.extend([
                float(np.median(series_np)),
                float(np.mean(series_np)),
            ])
    positive_candidates = [float(value) for value in candidates if value is not None and value > 0]
    return positive_candidates[0] if positive_candidates else 1.0


def build_scalar_indicators_dataframe(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
) -> pd.DataFrame:
    indicators = reliability_result.get("indicators", {}) or {}
    params = reliability_result.get("params", {}) or {}
    availability = indicators.get("availability_intrinsic")

    t_ref = build_reference_time(reliability_result, ttf_series)
    evaluation_axis = np.asarray([max(t_ref, 1e-6)], dtype=float)

    survival_values = compute_model_based_curve(reliability_result, ttf_series, evaluation_axis, "survival")
    cdf_values = compute_model_based_curve(reliability_result, ttf_series, evaluation_axis, "cdf")
    density_values = compute_model_based_curve(reliability_result, ttf_series, evaluation_axis, "pdf")
    hazard_values = compute_model_based_curve(reliability_result, ttf_series, evaluation_axis, "hazard")

    return pd.DataFrame([
        {
            "t_ref_h": t_ref,
            "R_t_ref": None if survival_values is None else float(survival_values[0]),
            "F_t_ref": None if cdf_values is None else float(cdf_values[0]),
            "f_t_ref": None if density_values is None else float(density_values[0]),
            "lambda_t_ref": None if hazard_values is None else float(hazard_values[0]),
            "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
            "MTBF_h": indicators.get("mtbf_h"),
            "MTTR_h": indicators.get("mttr_h"),
            "availability_pct": None if availability is None else 100.0 * float(availability),
            "beta": params.get("beta"),
            "eta_h": params.get("eta"),
            "gamma_h": params.get("gamma"),
        }
    ])


def get_decisive_block(
    reliability_result: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
) -> Tuple[str, pd.DataFrame]:
    model_name = str(reliability_result.get("model") or "").upper()

    if model_name == "NHPP":
        return "Étape décisive : Tendance", tables.get("trend_results", pd.DataFrame())
    if model_name == "BPP":
        return "Étape décisive : Dépendance", tables.get("dependence_results", pd.DataFrame())

    process_choice_df = tables.get("process_choice", pd.DataFrame())
    if process_choice_df.empty:
        process_choice_df = pd.DataFrame([
            {
                "Processus retenu": reliability_result.get("model"),
                "Variant": reliability_result.get("process_variant"),
                "Hypothèse entité": "Renouvellement / comportement sans tendance ni dépendance dominante",
                "Justification": "Le chemin de décision aboutit au processus RP/HPP.",
            }
        ])
    return "Étape décisive : RP / HPP", process_choice_df


def get_selected_law_dataframe(
    reliability_result: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fit_candidates = tables.get("fit_candidates", pd.DataFrame())
    if isinstance(fit_candidates, pd.DataFrame) and not fit_candidates.empty:
        selected_rows = fit_candidates.copy()
        if "Retenue" in selected_rows.columns:
            retained_mask = selected_rows["Retenue"].astype(str).str.lower().isin(["true", "vrai", "oui", "1"])
            if retained_mask.any():
                selected_rows = selected_rows[retained_mask]
        if "Acceptée" in selected_rows.columns:
            accepted_mask = selected_rows["Acceptée"].astype(str).str.lower().isin(["true", "vrai", "oui", "1"])
            if accepted_mask.any():
                selected_rows = selected_rows[accepted_mask]
        if not selected_rows.empty:
            preferred_columns = [
                column
                for column in [
                    "Modèle",
                    "Paramètres",
                    "Méthode estimation",
                    "LogLik",
                    "AIC",
                    "KS p",
                    "Chi2 p",
                    "CvM p",
                    "Acceptée",
                    "Retenue",
                ]
                if column in selected_rows.columns
            ]
            if preferred_columns:
                return selected_rows[preferred_columns].head(1).reset_index(drop=True)
            return selected_rows.head(1).reset_index(drop=True)

    params = reliability_result.get("params", {}) or {}
    return pd.DataFrame([
        {
            "Processus": reliability_result.get("model"),
            "Distribution": reliability_result.get("distribution"),
            "Paramètres": f"beta={format_number(params.get('beta'), 3)} ; eta={format_number(params.get('eta'), 2)} ; gamma={format_number(params.get('gamma'), 2)}",
            "Méthode estimation": "MLE / pipeline interne",
        }
    ])


def build_adjustment_dataframe(
    reliability_result: Dict[str, Any],
) -> pd.DataFrame:
    goodness = reliability_result.get("goodness", {}) or {}
    return pd.DataFrame([
        {
            "AIC": goodness.get("aic"),
            "KS p": goodness.get("ks_p"),
            "Chi2 p": goodness.get("chi2_p"),
            "CvM p": goodness.get("cvm_p"),
            "Ajustement accepté": goodness.get("accepted"),
        }
    ])


def build_equipment_summary_row(
    equipment_code: str,
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
) -> dict[str, Any]:
    indicators = reliability_result.get("indicators", {}) or {}
    params = reliability_result.get("params", {}) or {}
    scalar_indicators = build_scalar_indicators_dataframe(reliability_result, ttf_series).iloc[0].to_dict()

    return {
        "equipment_code": equipment_code,
        "model": reliability_result.get("model"),
        "process_variant": reliability_result.get("process_variant"),
        "distribution": reliability_result.get("distribution"),
        "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
        "MTBF_h": indicators.get("mtbf_h"),
        "MTTR_h": indicators.get("mttr_h"),
        "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
        "beta": params.get("beta"),
        "eta_h": params.get("eta"),
        "gamma_h": params.get("gamma"),
        "R_t_ref": scalar_indicators.get("R_t_ref"),
        "lambda_t_ref": scalar_indicators.get("lambda_t_ref"),
    }


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
**Bêta** : décrit la forme du vieillissement.
- inférieur à 1 : défauts précoces
- proche de 1 : comportement aléatoire
- supérieur à 1 : usure

**Êta** : durée de vie caractéristique estimée.

**Gamma** : éventuel décalage temporel dans le modèle.

**R(t\*)** : probabilité de bon fonctionnement au temps de référence t\*.

**f(t\*)** : densité de probabilité au temps de référence t\*.

**λ(t\*)** : taux instantané de défaillance au temps de référence t\*.

**MTTF** : temps moyen avant défaillance.

**MTBF** : temps moyen entre défaillances.

**MTTR** : temps moyen de réparation.
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

    reliability_result = result.get("reliability", {}) or {}
    results_by_equipment[str(equipment_code)] = result
    summary_rows.append(build_equipment_summary_row(str(equipment_code), reliability_result, time_to_failure_list))

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
selected_reliability_result = selected_result.get("reliability", {}) or {}
selected_ttf_series = ttf_series_by_equipment.get(selected_equipment_for_detail, [])
selected_indicators_dataframe = build_scalar_indicators_dataframe(selected_reliability_result, selected_ttf_series)
decisive_title, decisive_dataframe = get_decisive_block(selected_reliability_result, selected_tables)
selected_law_dataframe = get_selected_law_dataframe(selected_reliability_result, selected_tables)
adjustment_dataframe = build_adjustment_dataframe(selected_reliability_result)


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
    st.metric("Nombre total de défaillances", total_failure_count)
with metric_col_3:
    availability_series = summary_dataframe["availability_pct"].dropna()
    st.metric(
        "Disponibilité moyenne (%)",
        format_number(availability_series.mean(), 2) if not availability_series.empty else "—",
    )
with metric_col_4:
    beta_series = summary_dataframe["beta"].dropna()
    st.metric(
        "Bêta maximal observé",
        format_number(beta_series.max(), 2) if not beta_series.empty else "—",
    )

st.subheader("Synthèse des équipements analysés")
overview_columns = [
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
st.dataframe(
    rename_columns_for_display(summary_dataframe[overview_columns]),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# -------------------------------------------------------------------
# Detailed reading
# -------------------------------------------------------------------
st.subheader(f"{decisive_title} — {selected_equipment_for_detail}")
if decisive_dataframe is None or decisive_dataframe.empty:
    st.info("Aucune table disponible pour cette étape.")
else:
    st.dataframe(rename_columns_for_display(decisive_dataframe), use_container_width=True, hide_index=True)

st.subheader("Loi choisie")
st.dataframe(rename_columns_for_display(selected_law_dataframe), use_container_width=True, hide_index=True)

st.subheader("Processus d’ajustement")
st.dataframe(rename_columns_for_display(adjustment_dataframe), use_container_width=True, hide_index=True)

st.subheader("Indicateurs calculés")
st.dataframe(rename_columns_for_display(selected_indicators_dataframe), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------
st.subheader("Graphiques")
time_horizon = build_time_horizon_for_equipment(selected_reliability_result, selected_ttf_series)
time_axis = np.linspace(1e-6, time_horizon, 400)

graph_tabs = st.tabs([
    "Fiabilité R(t)",
    "Répartition F(t)",
    "Densité f(t)",
    "Taux λ(t)",
])

curve_specs = [
    ("survival", "Fiabilité R(t)", "R(t)"),
    ("cdf", "Fonction de répartition F(t)", "F(t)"),
    ("pdf", "Densité de probabilité f(t)", "f(t)"),
    ("hazard", "Taux de défaillance λ(t)", "λ(t)"),
]

for graph_tab, (curve_name, title, y_label) in zip(graph_tabs, curve_specs):
    with graph_tab:
        figure, axis = plt.subplots(figsize=(10, 4.8))
        curve_values = compute_model_based_curve(
            reliability_result=selected_reliability_result,
            ttf_series=selected_ttf_series,
            time_axis=time_axis,
            curve_name=curve_name,
        )
        if curve_values is None:
            axis.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(time_axis, curve_values, linewidth=2)
        axis.set_title(f"{title} — {selected_equipment_for_detail}")
        axis.set_xlabel("Temps (heures)")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        st.pyplot(figure, clear_figure=True)

# -------------------------------------------------------------------
# Downloads
# -------------------------------------------------------------------
st.divider()
st.subheader("Téléchargement du rapport")

if export_merged_report_pdf is None:
    st.info("Le module PDF n’est pas disponible.")
    if report_error_message:
        st.caption(report_error_message)
else:
    selected_source_dataframe = source_dataframe[
        source_dataframe["equipment_code"].astype(str).isin(selected_equipment_codes)
    ].copy()

    if st.button("Générer le rapport PDF", type="primary", use_container_width=True):
        try:
            output_path = export_merged_report_pdf(
                df=selected_source_dataframe,
                out_dir=str(Path(__file__).resolve().parents[1] / "reports"),
                title="Rapport — Indicateurs fiabilistes",
                analysis_results=results_by_equipment,
                alpha=float(alpha_value),
            )
            st.session_state["last_indicators_report_path"] = output_path
            st.success(f"PDF généré : {output_path}")
        except Exception as error:
            st.error(f"PDF : {error}")

    pdf_path = st.session_state.get("last_indicators_report_path")
    if pdf_path and Path(pdf_path).exists():
        pdf_bytes = Path(pdf_path).read_bytes()
        st.download_button(
            "Télécharger le PDF",
            data=pdf_bytes,
            file_name=Path(pdf_path).name,
            mime="application/pdf",
            use_container_width=True,
        )
