
from __future__ import annotations

from pathlib import Path
import io
import hashlib
import math
from typing import Any, Optional

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
    "Intervalles recommandés, coûts, fiabilité et préparation du planning de maintenance.",
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


def recommend_maintenance_type(
    reliability_result: dict[str, Any],
) -> str:
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


def recommend_interval(
    maintenance_type: str,
    reliability_interval_hours: Any,
    economic_interval_hours: Any,
    reliability_at_economic_interval: Any,
    target_reliability: float,
) -> Optional[float]:
    """
    Règle cohérente avec la logique fiabiliste :
    - si l’horizon économique existe et respecte la cible de fiabilité, on le retient ;
    - sinon, on revient à l’horizon issu du critère de fiabilité ;
    - pour un cas explicitement correctif/fiabilisation, aucun intervalle automatique n’est imposé.
    """
    maintenance_type_lower = str(maintenance_type or "").lower()
    if "corrective" in maintenance_type_lower and "fiabilisation" in maintenance_type_lower:
        return None

    reliability_interval_value = safe_number(reliability_interval_hours)
    economic_interval_value = safe_number(economic_interval_hours)
    reliability_at_economic_value = safe_number(reliability_at_economic_interval)

    if (
        economic_interval_value is not None
        and economic_interval_value > 0
        and reliability_at_economic_value is not None
        and reliability_at_economic_value >= float(target_reliability)
    ):
        return float(economic_interval_value)

    if reliability_interval_value is not None and reliability_interval_value > 0:
        return float(reliability_interval_value)

    if economic_interval_value is not None and economic_interval_value > 0:
        return float(economic_interval_value)

    return None


def build_optimization_note(
    scale_parameter_hours: Any,
    economic_interval_hours: Any,
    reliability_interval_hours: Any,
    recommended_interval_hours: Any,
    reliability_at_economic_interval: Any,
    target_reliability: Any,
    optimization_model_source: str,
) -> str:
    scale_parameter_value = safe_number(scale_parameter_hours)
    economic_interval_value = safe_number(economic_interval_hours)
    reliability_interval_value = safe_number(reliability_interval_hours)
    recommended_interval_value = safe_number(recommended_interval_hours)
    reliability_at_economic_value = safe_number(reliability_at_economic_interval)
    target_reliability_value = safe_number(target_reliability)

    if recommended_interval_value is None:
        return "Aucun intervalle exploitable n’a pu être retenu automatiquement pour cet équipement."

    parts = [f"Intervalle retenu : {recommended_interval_value:.1f} heures."]
    parts.append(f"Optimisation calculée avec le modèle retenu par l’organigramme ({optimization_model_source}).")

    if reliability_interval_value is not None:
        parts.append(f"Intervalle issu du critère de fiabilité : {reliability_interval_value:.1f} heures.")

    if economic_interval_value is not None:
        parts.append(f"Intervalle issu du critère économique : {economic_interval_value:.1f} heures.")

    if reliability_at_economic_value is not None and target_reliability_value is not None:
        if reliability_at_economic_value >= target_reliability_value:
            parts.append(
                f"La fiabilité à l’intervalle économique est {reliability_at_economic_value:.4f}, "
                f"donc elle respecte la cible {target_reliability_value:.2f}."
            )
        else:
            parts.append(
                f"La fiabilité à l’intervalle économique est {reliability_at_economic_value:.4f}, "
                f"donc elle est inférieure à la cible {target_reliability_value:.2f}."
            )

    if scale_parameter_value is not None:
        parts.append(f"Paramètre d’échelle utilisé par le modèle : {scale_parameter_value:.1f} heures.")

    return " ".join(parts)




def hours_to_days(hours: Any) -> Optional[float]:
    value = safe_number(hours)
    if value is None:
        return None
    return value / 24.0


def reliability_from_selected_model(
    reliability_result: dict[str, Any],
    t: Any,
    ttf_series: Optional[list[float]] = None,
) -> Optional[float]:
    """Calcule R(t) avec le modèle réellement retenu par l’organigramme."""
    time_value = safe_number(t)
    if time_value is None or time_value < 0:
        return None

    model_name = str(reliability_result.get("model") or "").upper()
    parameters = reliability_result.get("params", {}) or {}

    if model_name == "NHPP":
        beta_value = safe_number(parameters.get("beta"))
        eta_value = safe_number(parameters.get("eta"))
        if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
            return None
        return float(math.exp(-((time_value / eta_value) ** beta_value)))

    if model_name == "RP":
        distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
        if distribution_object is None or distribution_parameters is None:
            return None
        try:
            return float(distribution_object.sf(time_value, *distribution_parameters))
        except Exception:
            return None

    if model_name == "BPP":
        # Approximation prudente : on utilise la partie de base mu lorsque disponible.
        # Le BPP est surtout exploité ici comme signal de surveillance renforcée.
        mu_value = safe_number(parameters.get("mu"))
        if mu_value is None or mu_value <= 0:
            return None
        return float(math.exp(-mu_value * time_value))

    return None


def reliability_interval_from_selected_model(
    reliability_result: dict[str, Any],
    target_reliability: float,
) -> Optional[float]:
    """Calcule T_R tel que R(T_R)=target_reliability avec le modèle retenu."""
    target_value = safe_number(target_reliability)
    if target_value is None or not (0 < target_value < 1):
        return None

    model_name = str(reliability_result.get("model") or "").upper()
    parameters = reliability_result.get("params", {}) or {}

    if model_name == "NHPP":
        beta_value = safe_number(parameters.get("beta"))
        eta_value = safe_number(parameters.get("eta"))
        if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
            return None
        return float(eta_value * ((-math.log(target_value)) ** (1.0 / beta_value)))

    if model_name == "RP":
        distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
        if distribution_object is None or distribution_parameters is None:
            return None
        try:
            interval_value = float(distribution_object.isf(target_value, *distribution_parameters))
            return interval_value if np.isfinite(interval_value) and interval_value > 0 else None
        except Exception:
            return None

    if model_name == "BPP":
        mu_value = safe_number(parameters.get("mu"))
        if mu_value is None or mu_value <= 0:
            return None
        return float(-math.log(target_value) / mu_value)

    return None


def economic_interval_from_selected_model(
    reliability_result: dict[str, Any],
    preventive_cost: float,
    corrective_cost: float,
    minimum_reliability: float,
) -> dict[str, Optional[float]]:
    """
    Recherche l’intervalle économique avec le même modèle que celui retenu
    par l’organigramme. L’intervalle est limité à la zone où R(t) reste au moins
    égale à minimum_reliability.
    """
    preventive_cost_value = safe_number(preventive_cost)
    corrective_cost_value = safe_number(corrective_cost)
    minimum_reliability_value = safe_number(minimum_reliability)

    if (
        preventive_cost_value is None
        or corrective_cost_value is None
        or preventive_cost_value <= 0
        or corrective_cost_value <= 0
        or minimum_reliability_value is None
        or not (0 < minimum_reliability_value < 1)
    ):
        return {"T_cost": None, "R_at_T": None, "C_min": None}

    upper_bound = reliability_interval_from_selected_model(
        reliability_result=reliability_result,
        target_reliability=minimum_reliability_value,
    )
    if upper_bound is None or upper_bound <= 1.0:
        return {"T_cost": None, "R_at_T": None, "C_min": None}

    # Grille suffisamment fine pour Streamlit, sans dépendance additionnelle.
    time_grid = np.linspace(1.0, float(upper_bound), 2500)

    best_time = None
    best_reliability = None
    best_cost = None

    for current_time in time_grid:
        reliability_value = reliability_from_selected_model(reliability_result, float(current_time))
        if reliability_value is None or not np.isfinite(reliability_value):
            continue

        cost_value = (
            preventive_cost_value + corrective_cost_value * (1.0 - reliability_value)
        ) / float(current_time)

        if best_cost is None or cost_value < best_cost:
            best_time = float(current_time)
            best_reliability = float(reliability_value)
            best_cost = float(cost_value)

    return {"T_cost": best_time, "R_at_T": best_reliability, "C_min": best_cost}


def selected_model_source_label(reliability_result: dict[str, Any]) -> str:
    model_name = str(reliability_result.get("model") or "?").upper()
    distribution_name = str(reliability_result.get("distribution") or "?")
    if model_name == "NHPP":
        return "NHPP / loi de puissance"
    if model_name == "RP":
        return f"RP / {distribution_name}"
    if model_name == "BPP":
        return "BPP / approximation de surveillance"
    return f"{model_name} / {distribution_name}"


def get_distribution_and_parameters(reliability_result: dict[str, Any]):
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


def compute_rp_curve(reliability_result: dict[str, Any], time_axis: np.ndarray) -> Optional[np.ndarray]:
    distribution_object, distribution_parameters = get_distribution_and_parameters(reliability_result)
    if distribution_object is None or distribution_parameters is None:
        return None
    try:
        return np.asarray(distribution_object.sf(time_axis, *distribution_parameters), dtype=float)
    except Exception:
        return None


def compute_nhpp_curve(reliability_result: dict[str, Any], time_axis: np.ndarray) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    beta_value = safe_number(parameters.get("beta"))
    eta_value = safe_number(parameters.get("eta"))
    if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
        return None
    safe_time_axis = np.maximum(time_axis, 1e-6)
    cumulative_events = (safe_time_axis / eta_value) ** beta_value
    return np.exp(-cumulative_events)


def compute_bpp_curve(reliability_result: dict[str, Any], ttf_series: list[float], time_axis: np.ndarray) -> Optional[np.ndarray]:
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
    return np.exp(-cumulative_intensity)


def compute_model_reliability_curve(reliability_result: dict[str, Any], ttf_series: list[float], time_axis: np.ndarray) -> Optional[np.ndarray]:
    model_name = str(reliability_result.get("model") or "").upper()
    if model_name == "RP":
        return compute_rp_curve(reliability_result, time_axis)
    if model_name == "NHPP":
        return compute_nhpp_curve(reliability_result, time_axis)
    if model_name == "BPP":
        return compute_bpp_curve(reliability_result, ttf_series, time_axis)
    return None


def build_time_horizon_for_equipment(reliability_result: dict[str, Any], ttf_series: list[float], extra_values: Optional[list[float]] = None) -> float:
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


def build_curve_figure(equipment_code: str, reliability_result: dict[str, Any], ttf_series: list[float], interval_tr: Any, interval_tc: Any, interval_rec: Any) -> plt.Figure:
    interval_tr_value = safe_number(interval_tr)
    interval_tc_value = safe_number(interval_tc)
    interval_rec_value = safe_number(interval_rec)
    time_horizon = build_time_horizon_for_equipment(reliability_result, ttf_series, [interval_tr_value, interval_tc_value, interval_rec_value])
    time_axis = np.linspace(1e-6, time_horizon, 400)
    reliability_curve = compute_model_reliability_curve(reliability_result, ttf_series, time_axis)

    figure, axis = plt.subplots(figsize=(10, 5.4))
    if reliability_curve is None:
        axis.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=axis.transAxes)
    else:
        axis.plot(
            time_axis,
            reliability_curve,
            linewidth=2.2,
            label=(
                f"{equipment_code} | {reliability_result.get('model', '?')} / {reliability_result.get('distribution', '?')} "
                f"| beta={format_number((reliability_result.get('params', {}) or {}).get('beta'), 2)}, "
                f"eta={format_number((reliability_result.get('params', {}) or {}).get('eta'), 1)}"
            ),
        )

    if interval_rec_value is not None:
        axis.axvline(interval_rec_value, color="green", linestyle="-", linewidth=2.2, label=f"T_recommandé = {interval_rec_value:.1f} h")
    if interval_tr_value is not None:
        color_tr = "green" if interval_rec_value is not None and abs(interval_rec_value - interval_tr_value) < 1e-9 else "red"
        suffix = " (retenu)" if interval_rec_value is not None and abs(interval_rec_value - interval_tr_value) < 1e-9 else ""
        axis.axvline(interval_tr_value, color=color_tr, linestyle="--", linewidth=1.7, label=f"T_R = {interval_tr_value:.1f} h{suffix}")
    if interval_tc_value is not None:
        color_tc = "green" if interval_rec_value is not None and abs(interval_rec_value - interval_tc_value) < 1e-9 else "red"
        suffix = " (retenu)" if interval_rec_value is not None and abs(interval_rec_value - interval_tc_value) < 1e-9 else ""
        axis.axvline(interval_tc_value, color=color_tc, linestyle=":", linewidth=1.9, label=f"T_cost = {interval_tc_value:.1f} h{suffix}")

    axis.grid(True, alpha=0.3)
    axis.set_xlabel("Temps (heures)")
    axis.set_ylabel("Fiabilité R(t)")
    axis.set_title("Courbe de fiabilité du modèle retenu")
    handles, labels = axis.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    if uniq_l:
        axis.legend(uniq_h, uniq_l, fontsize=8)
    figure.tight_layout()
    return figure

DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi de probabilité retenue",
    "mk_p": "Valeur p du test de Mann-Kendall",
    "mk_direction": "Sens du test de Mann-Kendall",
    "laplace_p": "Valeur p du test de Laplace",
    "laplace_direction": "Sens du test de Laplace",
    "spearman_r": "Coefficient de Spearman",
    "spearman_p": "Valeur p du test de Spearman",
    "MTTF_h": "Temps moyen avant défaillance (heures)",
    "MTBF_h": "Temps moyen entre défaillances (heures)",
    "MTTR_h": "Temps moyen de réparation (heures)",
    "availability_pct": "Disponibilité intrinsèque (%)",
    "beta": "Paramètre bêta",
    "eta_h": "Paramètre êta (heures)",
    "gamma_h": "Paramètre gamma (heures)",
    "beta_weibull_ref": "Bêta Weibull de référence",
    "eta_weibull_ref_h": "Êta Weibull de référence (heures)",
    "gamma_weibull_ref_h": "Gamma Weibull de référence (heures)",
    "T_R_h": "Intervalle issu du critère de fiabilité (heures)",
    "T_cost_h": "Intervalle issu du critère économique (heures)",
    "R(T_cost)": "Fiabilité au niveau de l’intervalle économique",
    "C_min_per_h": "Coût minimal par heure",
    "T_recommended_h": "Intervalle recommandé (heures)",
    "days_recommended": "Jours avant maintenance retenus",
    "maintenance_type": "Type de maintenance recommandé",
    "decision_reason": "Justification de la décision fiabiliste",
    "optimization_model_source": "Modèle utilisé pour l’optimisation",
    "optimization_note": "Note d’optimisation",
    "trend_direction": "Sens global de la tendance",
    "trend_confidence": "Niveau de confiance sur la tendance",
    "reliability_adjustment_accepted": "Ajustement fiabiliste accepté",
    "reliability_ok": "Conformité fiabiliste",
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
    target_reliability = st.slider("Fiabilité cible", 0.50, 0.99, 0.80, 0.01)
with economic_col_2:
    preventive_cost = st.number_input("Coût préventif", min_value=0.0, value=1.0, step=0.1)
with economic_col_3:
    corrective_cost = st.number_input("Coût correctif", min_value=0.0, value=5.0, step=0.5)
with economic_col_4:
    minimum_reliability_for_economic_interval = st.slider(
        "Fiabilité minimale pour le calcul économique",
        0.0,
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
optimization_rows: list[dict[str, Any]] = []

for equipment_code in selected_equipment_codes:
    equipment_dataframe = source_dataframe[source_dataframe["equipment_code"] == equipment_code].copy()
    time_to_failure_series = series_to_positive_list(equipment_dataframe["ttf_h"])
    if not time_to_failure_series or len(time_to_failure_series) < 3:
        continue

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

# Les fits Weibull restent calculés comme référence affichable et pour compatibilité
# avec certains exports historiques. Cependant, les intervalles T_R, T_cost et
# R(T_cost) sont désormais calculés avec le modèle réellement retenu par
# l’organigramme, afin d’éviter de mélanger paramètres NHPP et optimisation Weibull.

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

    reliability_interval_hours = reliability_interval_from_selected_model(
        reliability_result=reliability_result,
        target_reliability=float(target_reliability),
    )

    if economic_optimization_enabled:
        economic_result = economic_interval_from_selected_model(
            reliability_result=reliability_result,
            preventive_cost=float(preventive_cost),
            corrective_cost=float(corrective_cost),
            minimum_reliability=float(minimum_reliability_for_economic_interval),
        )
    else:
        economic_result = {"T_cost": None, "R_at_T": None, "C_min": None}

    economic_interval_hours = economic_result.get("T_cost")
    reliability_at_economic_interval = economic_result.get("R_at_T")
    minimum_hourly_cost = economic_result.get("C_min")

    recommended_maintenance_type = recommend_maintenance_type(reliability_result)
    recommended_interval_hours = recommend_interval(
        maintenance_type=recommended_maintenance_type,
        reliability_interval_hours=reliability_interval_hours,
        economic_interval_hours=economic_interval_hours,
        reliability_at_economic_interval=reliability_at_economic_interval,
        target_reliability=float(target_reliability),
    )
    optimization_model_source = selected_model_source_label(reliability_result)

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
            "spearman_r": dependence_test_result.get("spearman_r"),
            "spearman_p": dependence_test_result.get("spearman_p"),
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
            "T_R_h": safe_number(reliability_interval_hours),
            "T_cost_h": safe_number(economic_interval_hours),
            "R(T_cost)": safe_number(reliability_at_economic_interval),
            "C_min_per_h": safe_number(minimum_hourly_cost),
            "T_recommended_h": safe_number(recommended_interval_hours),
            "days_recommended": hours_to_days(recommended_interval_hours),
            "maintenance_type": recommended_maintenance_type,
            "decision_reason": decision.get("reason"),
            "optimization_model_source": optimization_model_source,
            "optimization_note": build_optimization_note(
                primary_eta,
                economic_interval_hours,
                reliability_interval_hours,
                recommended_interval_hours,
                reliability_at_economic_interval,
                float(target_reliability),
                optimization_model_source,
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
page_tabs = st.tabs([
    "Paramètres issus de l’analyse",
    "Résultat de l’optimisation",
    "Courbes",
    "Détail par équipement",
    "Exports",
])

with page_tabs[0]:
    st.subheader("Paramètres issus de l’analyse")

    trend_dataframe = optimization_dataframe[
        [
            "equipment_code",
            "mk_p",
            "mk_direction",
            "laplace_p",
            "laplace_direction",
            "trend_direction",
            "trend_confidence",
        ]
    ].copy()

    dependence_dataframe = optimization_dataframe[
        [
            "equipment_code",
            "spearman_r",
            "spearman_p",
        ]
    ].copy()

    reliability_dataframe = optimization_dataframe[
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
            "reliability_adjustment_accepted",
        ]
    ].copy()

    st.markdown("#### Tendance")
    st.dataframe(rename_columns_for_display(trend_dataframe), use_container_width=True, hide_index=True)

    st.markdown("#### Dépendance")
    st.dataframe(rename_columns_for_display(dependence_dataframe), use_container_width=True, hide_index=True)

    st.markdown("#### Fiabilité")
    st.dataframe(rename_columns_for_display(reliability_dataframe), use_container_width=True, hide_index=True)

with page_tabs[1]:
    st.subheader("Résultat de l’optimisation")

    optimization_view = optimization_dataframe[
        [
            "equipment_code",
            "model",
            "process_variant",
            "distribution",
            "optimization_model_source",
            "beta",
            "eta_h",
            "gamma_h",
            "T_R_h",
            "T_cost_h",
            "T_recommended_h",
            "R(T_cost)",
            "C_min_per_h",
            "maintenance_type",
            "reliability_ok",
            "optimization_note",
        ]
    ].copy()

    st.dataframe(rename_columns_for_display(optimization_view), use_container_width=True, hide_index=True)

    st.info(
        "Les intervalles T_R, T_cost et R(T_cost) sont calculés avec le modèle retenu par l’organigramme. "
        "Les paramètres Weibull de référence restent disponibles uniquement comme repères historiques ou pour compatibilité d’export."
    )

    st.success(
        f"Résultat envoyé vers la future page Maintenance | lignes={len(optimization_dataframe)} | empreinte={dataframe_hash(optimization_dataframe)}"
    )

    if st.button("Enregistrer aussi le résultat dans le fichier de secours", use_container_width=True):
        optimization_dataframe.to_csv(FALLBACK_OPTIMIZATION_FILE, index=False, encoding="utf-8")
        st.success(f"Fichier écrit : {FALLBACK_OPTIMIZATION_FILE}")

with page_tabs[2]:
    st.subheader("Courbes de fiabilité du modèle retenu")

    selected_equipment_for_curves = st.selectbox(
        "Équipement pour les courbes",
        options=optimization_dataframe["equipment_code"].tolist(),
        key="optimization_curve_equipment",
    )

    selected_row_for_curves = optimization_dataframe[
        optimization_dataframe["equipment_code"] == selected_equipment_for_curves
    ].iloc[0].to_dict()
    selected_reliability_for_curves = (pipeline_results_by_equipment.get(selected_equipment_for_curves, {}) or {}).get("reliability", {}) or {}
    selected_ttf_series = series_to_positive_list(
        source_dataframe[source_dataframe["equipment_code"] == selected_equipment_for_curves]["ttf_h"]
    ) or []

    figure = build_curve_figure(
        equipment_code=selected_equipment_for_curves,
        reliability_result=selected_reliability_for_curves,
        ttf_series=selected_ttf_series,
        interval_tr=selected_row_for_curves.get("T_R_h"),
        interval_tc=selected_row_for_curves.get("T_cost_h"),
        interval_rec=selected_row_for_curves.get("T_recommended_h"),
    )
    st.pyplot(figure, clear_figure=True)
    st.info(
        f"La courbe utilise maintenant les paramètres du modèle retenu affichés dans le tableau : "
        f"bêta={format_number(selected_row_for_curves.get('beta'), 2)}, "
        f"êta={format_number(selected_row_for_curves.get('eta_h'), 1)} h, "
        f"gamma={format_number(selected_row_for_curves.get('gamma_h'), 1)} h."
    )

with page_tabs[3]:
    st.subheader("Détail par équipement")

    selected_equipment_for_detail = st.selectbox(
        "Équipement détaillé",
        options=optimization_dataframe["equipment_code"].tolist(),
    )
    selected_row = optimization_dataframe[optimization_dataframe["equipment_code"] == selected_equipment_for_detail].iloc[0].to_dict()
    selected_pipeline_result = pipeline_results_by_equipment.get(selected_equipment_for_detail, {}) or {}
    selected_detail_tables = selected_pipeline_result.get("tables", {}) or {}

    st.markdown(
        f"### {selected_equipment_for_detail}\n"
        f"- **Processus retenu** : **{selected_row.get('model', '—')}**\n"
        f"- **Variant du processus** : **{selected_row.get('process_variant', '—')}**\n"
        f"- **Loi de probabilité retenue** : **{selected_row.get('distribution', '—')}**\n"
        f"- **Modèle utilisé pour l’optimisation** : **{selected_row.get('optimization_model_source', '—')}**\n"
        f"- **Paramètres principaux** : **bêta = {format_number(selected_row.get('beta'), 2)}**, "
        f"**êta = {format_number(selected_row.get('eta_h'), 1)} heures**, "
        f"**gamma = {format_number(selected_row.get('gamma_h'), 1)} heures**\n"
        f"- **Intervalle issu du critère économique** : **{format_number(selected_row.get('T_cost_h'), 1)} heures**\n"
        f"- **Intervalle issu du critère de fiabilité** : **{format_number(selected_row.get('T_R_h'), 1)} heures**\n"
        f"- **Intervalle recommandé** : **{format_number(selected_row.get('T_recommended_h'), 1)} heures** = **{format_number(selected_row.get('days_recommended'), 1)} jours**\n"
        f"- **Type de maintenance recommandé** : **{selected_row.get('maintenance_type', '—')}**"
    )

    if selected_row.get("decision_reason"):
        st.caption(str(selected_row["decision_reason"]))

    st.info(selected_row.get("optimization_note", "—"))

    st.markdown("#### Actions suggérées")
    for action in suggested_actions(
        float(selected_row["beta"]) if is_positive_number(selected_row.get("beta")) else 1.0
    ):
        st.markdown(f"- {action}")

    detail_tabs = st.tabs(["Fiabilité", "Tableaux exportables"])

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
        for table_key, table_dataframe in selected_detail_tables.items():
            if isinstance(table_dataframe, pd.DataFrame) and not table_dataframe.empty:
                st.markdown(f"##### {DETAIL_TABLE_LABELS.get(table_key, table_key)}")
                st.dataframe(rename_columns_for_display(table_dataframe), use_container_width=True, hide_index=True)

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
                        "R_at_T": current_row.get("R(T_cost)"),
                        "C_min": current_row.get("C_min_per_h"),
                    }

                compatibility_reliability_results = {
                    equipment_code: (pipeline_results_by_equipment.get(equipment_code, {}) or {}).get("reliability", {})
                    for equipment_code in pipeline_results_by_equipment.keys()
                }

                try:
                    output_path = export_optimization_report_pdf(
                        df=source_dataframe[source_dataframe["equipment_code"].isin(selected_equipment_codes)].copy(),
                        fits=weibull_reference_fits,
                        intervals=intervals_by_equipment,
                        organigram_by_eq=compatibility_reliability_results,
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
                        compatibility_reliability_results,
                        out_dir=str(BASE_DIR / "reports"),
                    )

                st.session_state["opt_pdf_path"] = output_path
                st.success(f"PDF généré : {output_path}")
            except Exception as error:
                st.error(f"PDF : {error}")

        pdf_path = st.session_state.get("opt_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as file:
                st.download_button(
                    "Télécharger le PDF d’optimisation",
                    data=file,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
