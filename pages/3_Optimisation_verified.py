
from __future__ import annotations

from pathlib import Path
import io
import hashlib
import math
from typing import Any, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sst

from core.reliability.weibull import fit_weibull
from core.reliability.policy import suggested_actions
from core.reliability.organigram import analyze_ttf_pipeline
from core.reliability.optimize import propose_intervals_cost_and_reliability
from core.security.auth import require_login
from core.datahub import get_current_failures_df, get_failures_meta
from core.ui import render_shell, render_page_header

export_optimization_report_pdf = None
pdf_import_error_message = None
try:
    from core.reliability.reporting_optimize import export_optimization_report_pdf as _export_optimization_report_pdf
    export_optimization_report_pdf = _export_optimization_report_pdf
except Exception as error:
    pdf_import_error_message = str(error)
    export_optimization_report_pdf = None


st.set_page_config(page_title="Optimisation", page_icon="🧠", layout="wide")
require_login()

render_shell("pages/3_Optimisation_verified.py")
render_page_header(
    "Optimisation",
    "Reprise des indicateurs, calcul complet des intervalles optimaux et décision finale de maintenance.",
    "🧠",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else default
    except Exception:
        return default


def format_number(value: Any, decimals: int = 2, default: str = "—") -> str:
    numeric_value = safe_number(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{decimals}f}"


def series_to_positive_list(series: pd.Series) -> Optional[list[float]]:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    numeric_values = numeric_values[numeric_values > 0]
    if numeric_values.empty:
        return None
    return numeric_values.astype(float).tolist()


def is_positive_number(value: Any) -> bool:
    numeric_value = safe_number(value)
    return numeric_value is not None and numeric_value > 0


def dataframe_hash(dataframe: pd.DataFrame) -> str:
    if dataframe is None or dataframe.empty:
        return "empty"
    return hashlib.md5(dataframe.to_csv(index=False).encode("utf-8")).hexdigest()


def rename_columns_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe
    renamed_dataframe = dataframe.copy()
    renamed_dataframe = renamed_dataframe.rename(columns={
        column: DISPLAY_COLUMN_NAMES.get(column, column)
        for column in renamed_dataframe.columns
    })
    return renamed_dataframe


def recommend_maintenance_type(reliability_result: dict[str, Any]) -> str:
    process_name = str(reliability_result.get("model") or "").upper()
    process_variant = str(reliability_result.get("process_variant") or "").upper()
    decision = reliability_result.get("decision", {}) or {}
    trend_direction = str(decision.get("trend_direction") or "").lower()
    beta_value = safe_number((reliability_result.get("params") or {}).get("beta"))

    if process_name == "NHPP":
        if trend_direction == "up":
            return "Maintenance préventive planifiée"
        if trend_direction == "down":
            return "Surveillance conditionnelle renforcée"
        return "Maintenance préventive planifiée"

    if process_name == "BPP":
        return "Inspection conditionnelle renforcée"

    if process_variant == "HPP" or str(reliability_result.get("distribution") or "").lower() == "expon":
        return "Surveillance conditionnelle"

    if beta_value is not None:
        if beta_value < 0.9:
            return "Action corrective et fiabilisation"
        if beta_value <= 1.1:
            return "Maintenance conditionnelle"

    return "Maintenance préventive planifiée"


def choose_recommended_interval(
    reliability_interval_hours: Any,
    economic_interval_hours: Any,
    reliability_at_economic_interval: Any,
    reliability_floor: float,
) -> tuple[Optional[float], str]:
    tr = safe_number(reliability_interval_hours)
    tc = safe_number(economic_interval_hours)
    rtc = safe_number(reliability_at_economic_interval)

    if tr is None and tc is None:
        return None, "Aucun intervalle exploitable n’a pu être calculé."

    if tc is None and tr is not None:
        return tr, "T_cost n’est pas disponible. T_R est retenu automatiquement."

    if tr is None and tc is not None:
        return tc, "T_R n’est pas disponible. T_cost est retenu automatiquement."

    if rtc is not None and rtc < reliability_floor:
        return tr, (
            f"R(T_cost) = {rtc:.3f} est inférieur au seuil {reliability_floor:.2f}. "
            "T_R est retenu automatiquement."
        )

    if tc >= tr:
        return tc, (
            f"T_cost = {tc:.1f} h maximise le nombre de jours avant maintenance et respecte "
            f"le seuil de fiabilité ({reliability_floor:.2f})."
        )

    return tr, (
        f"T_R = {tr:.1f} h est plus grand ou plus prudent dans le cas présent. "
        "Il est retenu comme intervalle recommandé."
    )


def build_optimization_note(
    characteristic_life_hours: Any,
    economic_interval_hours: Any,
    reliability_interval_hours: Any,
    recommended_interval_hours: Any,
    reliability_at_economic_interval: Any,
    decision_rule_text: str,
) -> str:
    characteristic_life_value = safe_number(characteristic_life_hours)
    economic_interval_value = safe_number(economic_interval_hours)
    reliability_interval_value = safe_number(reliability_interval_hours)
    recommended_interval_value = safe_number(recommended_interval_hours)
    reliability_at_economic_value = safe_number(reliability_at_economic_interval)

    if recommended_interval_value is None:
        return "Aucun intervalle exploitable n’a pu être retenu automatiquement pour cet équipement."

    parts = [f"Intervalle retenu : {recommended_interval_value:.1f} heures."]
    if reliability_interval_value is not None:
        parts.append(f"T_R = {reliability_interval_value:.1f} heures.")
    if economic_interval_value is not None:
        parts.append(f"T_cost = {economic_interval_value:.1f} heures.")
    if reliability_at_economic_value is not None:
        parts.append(f"R(T_cost) = {reliability_at_economic_value:.3f}.")
    if characteristic_life_value is not None:
        parts.append(f"Vie caractéristique estimée : {characteristic_life_value:.1f} heures.")
    parts.append(decision_rule_text)
    return " ".join(parts)


def days_from_hours(hours: Any) -> Optional[float]:
    value = safe_number(hours)
    if value is None:
        return None
    return value / 24.0


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


def compute_rp_curve(reliability_result: Dict[str, Any], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
    if distribution_object is None or distribution_parameters is None:
        return None

    try:
        if curve_name == "survival":
            return np.asarray(distribution_object.sf(time_axis, *distribution_parameters), dtype=float)
        if curve_name == "cdf":
            return np.asarray(distribution_object.cdf(time_axis, *distribution_parameters), dtype=float)
        if curve_name == "pdf":
            return np.asarray(distribution_object.pdf(time_axis, *distribution_parameters), dtype=float)
        if curve_name == "hazard":
            survival_values = distribution_object.sf(time_axis, *distribution_parameters)
            density_values = distribution_object.pdf(time_axis, *distribution_parameters)
            return np.divide(
                density_values,
                survival_values,
                out=np.full_like(density_values, np.nan, dtype=float),
                where=survival_values > 1e-12,
            )
    except Exception:
        return None
    return None


def compute_nhpp_curve(reliability_result: Dict[str, Any], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    beta_value = safe_number(parameters.get("beta"))
    eta_value = safe_number(parameters.get("eta"))

    if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
        return None

    safe_time_axis = np.maximum(time_axis, 1e-6)
    cumulative_events = (safe_time_axis / eta_value) ** beta_value
    intensity = (beta_value / eta_value) * ((safe_time_axis / eta_value) ** (beta_value - 1.0))
    survival_like = np.exp(-cumulative_events)
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


def compute_bpp_curve(reliability_result: Dict[str, Any], ttf_series: list[float], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    mu_value = safe_number(parameters.get("mu"))
    alpha_value = safe_number(parameters.get("alpha"))
    beta_kernel_value = safe_number(parameters.get("beta_kernel"))

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


def compute_model_based_curve(reliability_result: Dict[str, Any], ttf_series: list[float], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    model_name = str(reliability_result.get("model") or "").upper()

    if model_name == "RP":
        return compute_rp_curve(reliability_result, time_axis, curve_name)
    if model_name == "NHPP":
        return compute_nhpp_curve(reliability_result, time_axis, curve_name)
    if model_name == "BPP":
        return compute_bpp_curve(reliability_result, ttf_series, time_axis, curve_name)
    return None


def build_time_horizon_for_equipment(reliability_result: Dict[str, Any], ttf_series: list[float], extra_values: Optional[list[float]] = None) -> float:
    values = np.asarray(ttf_series, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size == 0:
        return 100.0

    parameters = reliability_result.get("params", {}) or {}
    eta_value = safe_number(parameters.get("eta"))

    base_horizon = max(
        50.0,
        float(np.max(values)) * 2.0,
        float(np.mean(values)) * 6.0,
        float(np.quantile(values, 0.90)) * 4.0,
    )

    if eta_value is not None and eta_value > 0:
        base_horizon = max(base_horizon, eta_value * 1.5)

    extra_values = extra_values or []
    positive_extras = [float(value) for value in extra_values if is_positive_number(value)]
    if positive_extras:
        base_horizon = max(base_horizon, max(positive_extras) * 1.25)

    return base_horizon


def build_curve_figure(
    equipment_code: str,
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
    interval_tr: Any,
    interval_tc: Any,
    interval_rec: Any,
) -> plt.Figure:
    interval_tr_value = safe_number(interval_tr)
    interval_tc_value = safe_number(interval_tc)
    interval_rec_value = safe_number(interval_rec)

    time_horizon = build_time_horizon_for_equipment(
        reliability_result,
        ttf_series,
        extra_values=[interval_tr_value, interval_tc_value, interval_rec_value],
    )
    time_axis = np.linspace(1e-6, time_horizon, 400)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    curve_definitions = [
        ("survival", "Fiabilité R(t)", "R(t)"),
        ("cdf", "Fonction de répartition F(t)", "F(t)"),
        ("pdf", "Densité f(t)", "f(t)"),
        ("hazard", "Taux de défaillance λ(t)", "λ(t)"),
    ]

    for axis, (curve_name, title, y_label) in zip(axes, curve_definitions):
        curve_values = compute_model_based_curve(
            reliability_result=reliability_result,
            ttf_series=ttf_series,
            time_axis=time_axis,
            curve_name=curve_name,
        )
        if curve_values is None:
            axis.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(time_axis, curve_values, linewidth=2, label=f"{equipment_code} - {title}")

        if interval_rec_value is not None:
            axis.axvline(
                interval_rec_value,
                color="green",
                linestyle="-",
                linewidth=2.2,
                label=f"T_recommandé = {interval_rec_value:.1f} h",
            )

        if interval_tr_value is not None:
            color_tr = "green" if interval_rec_value is not None and abs(interval_rec_value - interval_tr_value) < 1e-9 else "red"
            label_tr = f"T_R = {interval_tr_value:.1f} h"
            if interval_rec_value is not None and abs(interval_rec_value - interval_tr_value) < 1e-9:
                label_tr += " (retenu)"
            axis.axvline(interval_tr_value, color=color_tr, linestyle="--", linewidth=1.6, label=label_tr)

        if interval_tc_value is not None:
            color_tc = "green" if interval_rec_value is not None and abs(interval_rec_value - interval_tc_value) < 1e-9 else "red"
            label_tc = f"T_cost = {interval_tc_value:.1f} h"
            if interval_rec_value is not None and abs(interval_rec_value - interval_tc_value) < 1e-9:
                label_tc += " (retenu)"
            axis.axvline(interval_tc_value, color=color_tc, linestyle=":", linewidth=1.8, label=label_tc)

        axis.set_title(title)
        axis.set_xlabel("Temps (heures)")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        if unique_labels:
            axis.legend(unique_handles, unique_labels, fontsize=8)

    figure.suptitle(f"Courbes fiabilistes et intervalles d’optimisation - {equipment_code}", fontsize=12)
    figure.tight_layout()
    return figure


def render_formula_block():
    st.markdown("### Étapes et formules appliquées")
    st.markdown(
        """
**Étape 1 — Reprise des indicateurs calculés précédemment**  
On reprend les résultats fiabilistes déjà calculés : processus retenu, loi choisie, paramètres, MTTF, MTBF, MTTR et disponibilité.

**Étape 2 — Calcul des deux intervalles candidats**  
- **T_R** : intervalle issu du critère de fiabilité
- **T_cost** : intervalle issu du critère économique

**Étape 3 — Règle de décision finale**  
On retient l’intervalle qui maximise les jours avant maintenance tout en respectant le seuil minimal de fiabilité.
        """
    )
    st.latex(r"R(T)=\exp\left(-\left(\frac{T-\gamma}{\eta}\right)^\beta\right)")
    st.latex(r"T_{recommand\acute{e}}=\max(T_{cost},T_R)")
    st.latex(r"\text{Si } R(T_{cost})<0.70,\ \text{alors } T_{recommand\acute{e}}=T_R")
    st.markdown(
        """
**Étape 4 — Lecture décisionnelle**  
- si **R(T_cost) < 0.70**, on refuse automatiquement **T_cost**
- sinon, on choisit la valeur la plus grande entre **T_cost** et **T_R**
- les jours avant maintenance sont calculés par : **jours = T / 24**
        """
    )


DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi choisie",
    "mk_p": "p de Mann-Kendall",
    "mk_direction": "Sens Mann-Kendall",
    "laplace_p": "p de Laplace",
    "laplace_direction": "Sens Laplace",
    "spearman_r": "Coefficient de Spearman",
    "spearman_p": "p de Spearman",
    "MTTF_h": "MTTF (h)",
    "MTBF_h": "MTBF (h)",
    "MTTR_h": "MTTR (h)",
    "availability_pct": "Disponibilité (%)",
    "beta": "Bêta",
    "eta_h": "Êta (h)",
    "gamma_h": "Gamma (h)",
    "beta_weibull_ref": "Bêta Weibull référence",
    "eta_weibull_ref_h": "Êta Weibull référence (h)",
    "gamma_weibull_ref_h": "Gamma Weibull référence (h)",
    "T_R_h": "T_R (h)",
    "T_cost_h": "T_cost (h)",
    "R(T_cost)": "R(T_cost)",
    "C_min_per_h": "C_min / h",
    "T_recommended_h": "T_recommandé (h)",
    "days_T_R": "Jours avant maintenance via T_R",
    "days_T_cost": "Jours avant maintenance via T_cost",
    "days_recommended": "Jours avant maintenance retenus",
    "recommended_source": "Source de l’intervalle retenu",
    "maintenance_type": "Type de maintenance recommandé",
    "decision_reason": "Justification fiabiliste",
    "optimization_decision": "Règle de décision d’optimisation",
    "optimization_note": "Note d’optimisation",
    "trend_direction": "Sens global de la tendance",
    "trend_confidence": "Niveau de confiance sur la tendance",
    "reliability_adjustment_accepted": "Ajustement accepté",
    "reliability_ok": "Conformité fiabiliste",
}

DETAIL_TABLE_LABELS = {
    "trend_results": "Résultats détaillés des tests de tendance",
    "dependence_results": "Résultats détaillés des tests de dépendance",
    "process_choice": "Décision sur le processus fiabiliste",
    "fit_candidates": "Comparaison des lois candidates",
    "reliability_summary": "Synthèse fiabiliste",
}


def build_excel_export(
    summary_dataframe: pd.DataFrame,
    detail_payload: dict[str, dict[str, pd.DataFrame]],
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        rename_columns_for_display(summary_dataframe).to_excel(
            writer,
            sheet_name="Synthese_optimisation",
            index=False,
        )

        for equipment_code, tables_by_name in detail_payload.items():
            equipment_prefix = str(equipment_code)[:16]
            for table_name, table_dataframe in tables_by_name.items():
                if isinstance(table_dataframe, pd.DataFrame) and not table_dataframe.empty:
                    readable_name = DETAIL_TABLE_LABELS.get(table_name, table_name)
                    safe_sheet_name = f"{equipment_prefix}_{readable_name}"[:31]
                    try:
                        rename_columns_for_display(table_dataframe).to_excel(
                            writer,
                            sheet_name=safe_sheet_name,
                            index=False,
                        )
                    except Exception:
                        pass

    buffer.seek(0)
    return buffer.getvalue()


# -------------------------------------------------------------------
# Source centrale
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
FALLBACK_OPTIMIZATION_FILE = DATA_DIR / "last_optimization.csv"

failures_meta = get_failures_meta()
source_dataframe = get_current_failures_df()

if source_dataframe.empty:
    st.error("Aucun jeu de données actif. Va d’abord dans la page Sources de données.")
    st.stop()

source_dataframe = source_dataframe.copy()
source_dataframe.columns = [str(column).strip() for column in source_dataframe.columns]
source_dataframe["equipment_code"] = source_dataframe["equipment_code"].astype(str)
source_dataframe["ttf_h"] = pd.to_numeric(source_dataframe["ttf_h"], errors="coerce")

if "duree_rep_h" in source_dataframe.columns:
    source_dataframe["duree_rep_h"] = pd.to_numeric(source_dataframe["duree_rep_h"], errors="coerce")
else:
    source_dataframe["duree_rep_h"] = np.nan

source_dataframe = source_dataframe.dropna(subset=["ttf_h"])
source_dataframe = source_dataframe[source_dataframe["ttf_h"] > 0].reset_index(drop=True)

st.success(
    f"Jeu de données actif | lignes={failures_meta.get('rows')} | empreinte={failures_meta.get('hash')}"
)


# -------------------------------------------------------------------
# Contrôles
# -------------------------------------------------------------------
all_equipment_codes = sorted(source_dataframe["equipment_code"].unique().tolist())
default_selected_equipment_codes = all_equipment_codes[: min(5, len(all_equipment_codes))] if all_equipment_codes else []

control_col_1, control_col_2 = st.columns([2, 1])
with control_col_1:
    selected_equipment_codes = st.multiselect(
        "Équipements analysés",
        all_equipment_codes,
        default=default_selected_equipment_codes if default_selected_equipment_codes else all_equipment_codes,
    )
with control_col_2:
    alpha_value = st.number_input(
        "Seuil alpha",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.001,
        format="%.3f",
    )

if not selected_equipment_codes:
    st.info("Sélectionne au moins un équipement.")
    st.stop()

economic_col_1, economic_col_2, economic_col_3, economic_col_4 = st.columns(4)
with economic_col_1:
    target_reliability = st.slider("Fiabilité cible R*", 0.50, 0.99, 0.80, 0.01)
with economic_col_2:
    preventive_cost = st.number_input("Coût préventif Cp", min_value=0.0, value=1.0, step=0.1)
with economic_col_3:
    corrective_cost = st.number_input("Coût correctif Cf", min_value=0.0, value=5.0, step=0.5)
with economic_col_4:
    minimum_reliability_for_economic_interval = st.slider(
        "Seuil minimal accepté pour R(T_cost)",
        0.50,
        0.99,
        0.70,
        0.01,
    )

economic_optimization_enabled = (preventive_cost > 0) and (corrective_cost > 0)
if not economic_optimization_enabled:
    st.warning("Le coût préventif et le coût correctif doivent être strictement positifs pour lancer l’optimisation économique.")


# -------------------------------------------------------------------
# Analyse + optimisation
# -------------------------------------------------------------------
weibull_reference_fits: dict[str, Any] = {}
pipeline_results_by_equipment: dict[str, dict[str, Any]] = {}
detail_tables_by_equipment: dict[str, dict[str, pd.DataFrame]] = {}
optimization_rows: list[dict[str, Any]] = {}
optimization_rows = []
ttf_by_equipment: dict[str, list[float]] = {}

for equipment_code in selected_equipment_codes:
    equipment_dataframe = source_dataframe[source_dataframe["equipment_code"] == equipment_code].copy()
    time_to_failure_series = series_to_positive_list(equipment_dataframe["ttf_h"])
    if not time_to_failure_series or len(time_to_failure_series) < 3:
        continue

    ttf_by_equipment[equipment_code] = time_to_failure_series

    repair_time_series = None
    if "duree_rep_h" in equipment_dataframe.columns:
        extracted_repair_time_series = series_to_positive_list(equipment_dataframe["duree_rep_h"])
        repair_time_series = extracted_repair_time_series if extracted_repair_time_series else None

    try:
        pipeline_result = analyze_ttf_pipeline(
            ttf_series=time_to_failure_series,
            alpha=float(alpha_value),
            repair_series=repair_time_series,
        )
    except Exception as error:
        pipeline_result = {
            "reliability": {
                "error": str(error),
                "model": "?",
                "process_variant": "?",
                "distribution": "?",
                "params": {},
                "goodness": {},
                "tests": {},
                "decision": {},
                "indicators": {},
            },
            "tables": {},
        }

    pipeline_results_by_equipment[equipment_code] = pipeline_result
    detail_tables_by_equipment[equipment_code] = pipeline_result.get("tables", {}) or {}

    try:
        weibull_reference_fits[equipment_code] = fit_weibull(np.array(time_to_failure_series, dtype=float))
    except Exception:
        continue

if not weibull_reference_fits:
    st.error("Pas assez de temps entre défaillances exploitables (au moins 3 valeurs positives) pour les équipements sélectionnés.")
    st.stop()

economic_results_by_equipment: dict[str, dict[str, Any]] = {}
if economic_optimization_enabled:
    try:
        economic_results_by_equipment = propose_intervals_cost_and_reliability(
            fits=weibull_reference_fits,
            C_prev=float(preventive_cost),
            C_corr=float(corrective_cost),
            R_target=float(target_reliability),
            R_min_cost=float(minimum_reliability_for_economic_interval),
        )
    except Exception:
        economic_results_by_equipment = {}

for equipment_code, weibull_fit in weibull_reference_fits.items():
    pipeline_result = pipeline_results_by_equipment.get(equipment_code, {}) or {}
    reliability_result = pipeline_result.get("reliability", {}) or {}
    indicators = reliability_result.get("indicators", {}) or {}
    parameters = reliability_result.get("params", {}) or {}
    tests = reliability_result.get("tests", {}) or {}
    decision = reliability_result.get("decision", {}) or {}
    goodness = reliability_result.get("goodness", {}) or {}

    weibull_beta_reference = safe_number(getattr(weibull_fit, "beta", None))
    weibull_eta_reference = safe_number(getattr(weibull_fit, "eta", None))
    weibull_gamma_reference = safe_number(getattr(weibull_fit, "gamma", 0.0))

    primary_beta = safe_number(parameters.get("beta"), weibull_beta_reference)
    primary_eta = safe_number(parameters.get("eta"), weibull_eta_reference)
    primary_gamma = safe_number(parameters.get("gamma"), weibull_gamma_reference)

    reliability_interval_hours = (economic_results_by_equipment.get(equipment_code) or {}).get("T_R")
    economic_interval_hours = (economic_results_by_equipment.get(equipment_code) or {}).get("T_cost")
    reliability_at_economic_interval = (economic_results_by_equipment.get(equipment_code) or {}).get("R_at_T")
    minimum_hourly_cost = (economic_results_by_equipment.get(equipment_code) or {}).get("C_min")

    recommended_interval_hours, optimization_decision = choose_recommended_interval(
        reliability_interval_hours=reliability_interval_hours,
        economic_interval_hours=economic_interval_hours,
        reliability_at_economic_interval=reliability_at_economic_interval,
        reliability_floor=float(minimum_reliability_for_economic_interval),
    )

    tr_value = safe_number(reliability_interval_hours)
    tc_value = safe_number(economic_interval_hours)
    rec_value = safe_number(recommended_interval_hours)

    if rec_value is not None and tr_value is not None and abs(rec_value - tr_value) < 1e-9:
        recommended_source = "T_R"
    elif rec_value is not None and tc_value is not None and abs(rec_value - tc_value) < 1e-9:
        recommended_source = "T_cost"
    else:
        recommended_source = "Indéterminé"

    recommended_maintenance_type = recommend_maintenance_type(reliability_result)

    mann_kendall_test_result = tests.get("trend_mk", {}) or {}
    laplace_test_result = tests.get("trend_laplace", {}) or {}
    dependence_test_result = tests.get("dependence", {}) or {}

    reliability_adjustment_accepted = goodness.get("accepted")
    reliability_is_ok = None if reliability_adjustment_accepted is None else bool(reliability_adjustment_accepted)

    optimization_rows.append(
        {
            "equipment_code": equipment_code,
            "model": reliability_result.get("model"),
            "process_variant": reliability_result.get("process_variant"),
            "distribution": reliability_result.get("distribution"),
            "mk_p": mann_kendall_test_result.get("p"),
            "mk_direction": mann_kendall_test_result.get("direction"),
            "laplace_p": laplace_test_result.get("p"),
            "laplace_direction": laplace_test_result.get("direction"),
            "spearman_r": dependence_test_result.get("spearman_r") or dependence_test_result.get("r"),
            "spearman_p": dependence_test_result.get("spearman_p") or dependence_test_result.get("p"),
            "trend_direction": decision.get("trend_direction"),
            "trend_confidence": decision.get("trend_confidence"),
            "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
            "MTBF_h": indicators.get("mtbf_h"),
            "MTTR_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
            "beta": primary_beta,
            "eta_h": primary_eta,
            "gamma_h": primary_gamma,
            "beta_weibull_ref": weibull_beta_reference,
            "eta_weibull_ref_h": weibull_eta_reference,
            "gamma_weibull_ref_h": weibull_gamma_reference,
            "reliability_adjustment_accepted": reliability_adjustment_accepted,
            "reliability_ok": reliability_is_ok,
            "T_R_h": tr_value,
            "T_cost_h": tc_value,
            "R(T_cost)": safe_number(reliability_at_economic_interval),
            "C_min_per_h": safe_number(minimum_hourly_cost),
            "T_recommended_h": rec_value,
            "days_T_R": days_from_hours(tr_value),
            "days_T_cost": days_from_hours(tc_value),
            "days_recommended": days_from_hours(rec_value),
            "recommended_source": recommended_source,
            "maintenance_type": recommended_maintenance_type,
            "decision_reason": decision.get("reason"),
            "optimization_decision": optimization_decision,
            "optimization_note": build_optimization_note(
                primary_eta,
                economic_interval_hours,
                reliability_interval_hours,
                recommended_interval_hours,
                reliability_at_economic_interval,
                optimization_decision,
            ),
        }
    )

optimization_dataframe = pd.DataFrame(optimization_rows).sort_values("equipment_code").reset_index(drop=True)
if optimization_dataframe.empty:
    st.error("Aucun résultat exploitable après analyse et optimisation.")
    st.stop()

st.session_state["optimization_df"] = optimization_dataframe.copy()
st.session_state["optimization_src"] = "optimisation_page"
st.session_state["opt_meta"] = {
    "hash": dataframe_hash(optimization_dataframe),
    "rows": int(len(optimization_dataframe)),
    "source": "optimisation_page",
}


# -------------------------------------------------------------------
# Affichages
# -------------------------------------------------------------------
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Équipements optimisés", len(optimization_dataframe))
with metric_col_2:
    st.metric(
        "Jours max avant maintenance",
        format_number(optimization_dataframe["days_recommended"].dropna().max() if "days_recommended" in optimization_dataframe.columns and optimization_dataframe["days_recommended"].notna().any() else None, 1),
    )
with metric_col_3:
    st.metric(
        "Fiabilité moyenne à T_cost",
        format_number(optimization_dataframe["R(T_cost)"].dropna().mean() if "R(T_cost)" in optimization_dataframe.columns and optimization_dataframe["R(T_cost)"].notna().any() else None, 3),
    )
with metric_col_4:
    st.metric(
        "Disponibilité moyenne",
        format_number(optimization_dataframe["availability_pct"].dropna().mean() if optimization_dataframe["availability_pct"].notna().any() else None, 2),
    )

page_tabs = st.tabs([
    "Indicateurs repris",
    "Phase d’optimisation",
    "Courbes",
    "Détail par équipement",
    "Exports",
])

with page_tabs[0]:
    st.subheader("Indicateurs repris depuis la phase précédente")

    indicators_view = optimization_dataframe[
        [
            "equipment_code",
            "model",
            "process_variant",
            "distribution",
            "mk_p",
            "mk_direction",
            "laplace_p",
            "laplace_direction",
            "spearman_r",
            "spearman_p",
            "MTTF_h",
            "MTBF_h",
            "MTTR_h",
            "availability_pct",
            "beta",
            "eta_h",
            "gamma_h",
        ]
    ].copy()
    st.dataframe(rename_columns_for_display(indicators_view), use_container_width=True, hide_index=True)

with page_tabs[1]:
    st.subheader("Phase d’optimisation complète")
    render_formula_block()

    optimization_view = optimization_dataframe[
        [
            "equipment_code",
            "model",
            "distribution",
            "beta",
            "eta_h",
            "gamma_h",
            "T_R_h",
            "T_cost_h",
            "R(T_cost)",
            "T_recommended_h",
            "days_T_R",
            "days_T_cost",
            "days_recommended",
            "recommended_source",
            "C_min_per_h",
            "maintenance_type",
            "optimization_decision",
            "optimization_note",
        ]
    ].copy()
    st.dataframe(rename_columns_for_display(optimization_view), use_container_width=True, hide_index=True)

    st.success(
        f"Résultat envoyé vers la future page Maintenance | lignes={len(optimization_dataframe)} | empreinte={dataframe_hash(optimization_dataframe)}"
    )

    if st.button("Enregistrer aussi le résultat dans le fichier de secours", use_container_width=True):
        optimization_dataframe.to_csv(FALLBACK_OPTIMIZATION_FILE, index=False, encoding="utf-8")
        st.success(f"Fichier écrit : {FALLBACK_OPTIMIZATION_FILE}")

with page_tabs[2]:
    st.subheader("Courbes fiabilistes avec intervalles d’optimisation")

    selected_equipment_for_curves = st.selectbox(
        "Équipement pour les courbes",
        options=optimization_dataframe["equipment_code"].tolist(),
        key="optimization_curve_equipment",
    )

    selected_row_for_curves = optimization_dataframe[
        optimization_dataframe["equipment_code"] == selected_equipment_for_curves
    ].iloc[0].to_dict()
    selected_reliability_for_curves = (pipeline_results_by_equipment.get(selected_equipment_for_curves, {}) or {}).get("reliability", {}) or {}
    selected_ttf_for_curves = ttf_by_equipment.get(selected_equipment_for_curves, [])

    figure = build_curve_figure(
        equipment_code=selected_equipment_for_curves,
        reliability_result=selected_reliability_for_curves,
        ttf_series=selected_ttf_for_curves,
        interval_tr=selected_row_for_curves.get("T_R_h"),
        interval_tc=selected_row_for_curves.get("T_cost_h"),
        interval_rec=selected_row_for_curves.get("T_recommended_h"),
    )
    st.pyplot(figure, clear_figure=True)

    st.info(
        f"T_recommandé = {format_number(selected_row_for_curves.get('T_recommended_h'), 1)} h "
        f"({format_number(selected_row_for_curves.get('days_recommended'), 1)} jours) | "
        f"source retenue : {selected_row_for_curves.get('recommended_source', '—')}."
    )
    st.caption(selected_row_for_curves.get("optimization_decision", "—"))

with page_tabs[3]:
    st.subheader("Détail par équipement")

    selected_equipment_for_detail = st.selectbox(
        "Équipement détaillé",
        options=optimization_dataframe["equipment_code"].tolist(),
        key="optimization_detail_equipment",
    )
    selected_row = optimization_dataframe[optimization_dataframe["equipment_code"] == selected_equipment_for_detail].iloc[0].to_dict()
    selected_pipeline_result = pipeline_results_by_equipment.get(selected_equipment_for_detail, {}) or {}
    selected_reliability_result = selected_pipeline_result.get("reliability", {}) or {}
    selected_detail_tables = selected_pipeline_result.get("tables", {}) or {}

    st.markdown(
        f"### {selected_equipment_for_detail}\n"
        f"- **Processus retenu** : **{selected_row.get('model', '—')}**\n"
        f"- **Variant du processus** : **{selected_row.get('process_variant', '—')}**\n"
        f"- **Loi choisie** : **{selected_row.get('distribution', '—')}**\n"
        f"- **MTTF** : **{format_number(selected_row.get('MTTF_h'), 1)} h**\n"
        f"- **MTBF** : **{format_number(selected_row.get('MTBF_h'), 1)} h**\n"
        f"- **MTTR** : **{format_number(selected_row.get('MTTR_h'), 1)} h**\n"
        f"- **Disponibilité** : **{format_number(selected_row.get('availability_pct'), 2)} %**\n"
        f"- **Bêta / Êta / Gamma** : **{format_number(selected_row.get('beta'), 2)} / {format_number(selected_row.get('eta_h'), 1)} / {format_number(selected_row.get('gamma_h'), 1)}**\n"
        f"- **T_R** : **{format_number(selected_row.get('T_R_h'), 1)} h**\n"
        f"- **T_cost** : **{format_number(selected_row.get('T_cost_h'), 1)} h**\n"
        f"- **R(T_cost)** : **{format_number(selected_row.get('R(T_cost)'), 3)}**\n"
        f"- **T_recommandé** : **{format_number(selected_row.get('T_recommended_h'), 1)} h**\n"
        f"- **Jours avant maintenance retenus** : **{format_number(selected_row.get('days_recommended'), 1)} jours**\n"
        f"- **Type de maintenance recommandé** : **{selected_row.get('maintenance_type', '—')}**"
    )

    st.info(selected_row.get("optimization_note", "—"))

    with st.expander("Voir la logique de décision détaillée", expanded=True):
        st.markdown(
            f"""
- **Règle appliquée** : {selected_row.get('optimization_decision', '—')}
- **Source retenue** : {selected_row.get('recommended_source', '—')}
- **Justification fiabiliste** : {selected_row.get('decision_reason', '—')}
            """
        )

    st.markdown("#### Actions suggérées")
    beta_reference = float(selected_row["beta_weibull_ref"]) if is_positive_number(selected_row.get("beta_weibull_ref")) else 1.0
    for action in suggested_actions(beta_reference):
        st.markdown(f"- {action}")

    detail_tabs = st.tabs(["Tableaux d’analyse", "Tableaux exportables"])

    with detail_tabs[0]:
        for table_key, section_title in [
            ("trend_results", "Résultats des tests de tendance"),
            ("dependence_results", "Résultats des tests de dépendance"),
            ("process_choice", "Choix du processus fiabiliste"),
            ("fit_candidates", "Comparaison des lois candidates"),
            ("reliability_summary", "Synthèse fiabiliste"),
        ]:
            table_dataframe = selected_detail_tables.get(table_key)
            if isinstance(table_dataframe, pd.DataFrame) and not table_dataframe.empty:
                st.markdown(f"##### {section_title}")
                st.dataframe(rename_columns_for_display(table_dataframe), use_container_width=True, hide_index=True)

    with detail_tabs[1]:
        exportable_rows = pd.DataFrame(
            [
                {
                    "equipment_code": selected_row.get("equipment_code"),
                    "model": selected_row.get("model"),
                    "process_variant": selected_row.get("process_variant"),
                    "distribution": selected_row.get("distribution"),
                    "MTTF_h": selected_row.get("MTTF_h"),
                    "MTBF_h": selected_row.get("MTBF_h"),
                    "MTTR_h": selected_row.get("MTTR_h"),
                    "availability_pct": selected_row.get("availability_pct"),
                    "beta": selected_row.get("beta"),
                    "eta_h": selected_row.get("eta_h"),
                    "gamma_h": selected_row.get("gamma_h"),
                    "T_R_h": selected_row.get("T_R_h"),
                    "T_cost_h": selected_row.get("T_cost_h"),
                    "R(T_cost)": selected_row.get("R(T_cost)"),
                    "T_recommended_h": selected_row.get("T_recommended_h"),
                    "days_recommended": selected_row.get("days_recommended"),
                    "recommended_source": selected_row.get("recommended_source"),
                    "maintenance_type": selected_row.get("maintenance_type"),
                    "optimization_decision": selected_row.get("optimization_decision"),
                }
            ]
        )
        st.dataframe(rename_columns_for_display(exportable_rows), use_container_width=True, hide_index=True)

with page_tabs[4]:
    st.subheader("Exports")

    csv_bytes = rename_columns_for_display(optimization_dataframe).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le fichier CSV d’optimisation",
        data=csv_bytes,
        file_name="optimisation_intervalles.csv",
        mime="text/csv",
        use_container_width=True,
    )

    excel_bytes = build_excel_export(optimization_dataframe, detail_tables_by_equipment)
    st.download_button(
        "Télécharger le fichier Excel d’optimisation et de détails",
        data=excel_bytes,
        file_name="optimisation_intervalles_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_optimization_report_pdf is None:
        st.info("Le module PDF d’optimisation n’est pas disponible.")
        if pdf_import_error_message:
            st.caption(pdf_import_error_message)
    else:
        if st.button("Générer le PDF d’optimisation", type="primary", use_container_width=True):
            try:
                intervals_by_equipment = {}
                for equipment_code in weibull_reference_fits.keys():
                    equipment_row = optimization_dataframe[optimization_dataframe["equipment_code"] == equipment_code]
                    if equipment_row.empty:
                        continue

                    current_row = equipment_row.iloc[0]
                    intervals_by_equipment[equipment_code] = {
                        "T_R": current_row.get("T_R_h"),
                        "T_cost": current_row.get("T_cost_h"),
                        "T_recommended": current_row.get("T_recommended_h"),
                        "R_at_T": current_row.get("R(T_cost)"),
                        "C_min": current_row.get("C_min_per_h"),
                        "recommended_source": current_row.get("recommended_source"),
                        "days_recommended": current_row.get("days_recommended"),
                        "decision_text": current_row.get("optimization_decision"),
                    }

                try:
                    output_path = export_optimization_report_pdf(
                        df=source_dataframe[source_dataframe["equipment_code"].isin(selected_equipment_codes)].copy(),
                        fits=weibull_reference_fits,
                        intervals=intervals_by_equipment,
                        organigram_by_eq=pipeline_results_by_equipment,
                        out_dir=str(BASE_DIR / "reports"),
                        df_out=optimization_dataframe,
                        meta={
                            "alpha": alpha_value,
                            "R_target": target_reliability,
                            "C_prev": preventive_cost,
                            "C_corr": corrective_cost,
                            "R_min_cost": minimum_reliability_for_economic_interval,
                        },
                    )
                except TypeError:
                    output_path = export_optimization_report_pdf(
                        source_dataframe[source_dataframe["equipment_code"].isin(selected_equipment_codes)].copy(),
                        weibull_reference_fits,
                        intervals_by_equipment,
                        pipeline_results_by_equipment,
                        out_dir=str(BASE_DIR / "reports"),
                    )

                st.session_state["opt_pdf_path"] = output_path
                st.success(f"PDF généré : {output_path}")
            except Exception as error:
                st.error(f"PDF : {error}")

        pdf_path = st.session_state.get("opt_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            pdf_bytes = Path(pdf_path).read_bytes()
            st.download_button(
                "Télécharger le PDF d’optimisation",
                data=pdf_bytes,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )
