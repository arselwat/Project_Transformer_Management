
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sst

from core.security.auth import require_login
from core.reliability.organigram import analyze_ttf_pipeline
from core.ui import render_shell, render_page_header

try:
    from core.datahub import get_current_failures_df
except Exception:
    get_current_failures_df = None

try:
    from core.reliability.reporting_global import export_global_analysis_report_pdf
except Exception:
    export_global_analysis_report_pdf = None


st.set_page_config(page_title="Résultat global", page_icon="📋", layout="wide")
require_login()

render_shell("pages/5_Resultat_analyse_optimisation_Maintenance.py")
render_page_header(
    "Résultat global",
    "Pipeline complet : tendance, dépendance, choix du modèle, ajustement, paramètres, optimisation et décision finale.",
    "📋",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FAILURES_CSV = DATA_DIR / "failures_saved.csv"
OPTIMIZATION_CSV = DATA_DIR / "last_optimization.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def read_csv_flex(source) -> pd.DataFrame:
    def _try_read(obj, **kwargs):
        try:
            return pd.read_csv(obj, **kwargs)
        except Exception:
            return None

    dataframe = _try_read(source)
    if dataframe is None:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        dataframe = _try_read(source, engine="python", on_bad_lines="skip", sep=None)
    if dataframe is None:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        dataframe = _try_read(source, sep=";", engine="python", on_bad_lines="skip")
    if dataframe is None:
        return pd.DataFrame()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    return dataframe


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            return default
        return numeric_value
    except Exception:
        return default


def format_number(value: Any, decimals: int = 3, default: str = "—") -> str:
    numeric_value = safe_float(value, None)
    return default if numeric_value is None else f"{numeric_value:.{decimals}f}"


def series_to_positive_list(series: pd.Series) -> Optional[List[float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return values.astype(float).tolist()


def load_failures_dataframe(uploaded_csv=None) -> pd.DataFrame:
    if uploaded_csv is not None:
        dataframe = read_csv_flex(uploaded_csv)
    elif callable(get_current_failures_df):
        try:
            dataframe = get_current_failures_df()
        except Exception:
            dataframe = pd.DataFrame()
    elif FAILURES_CSV.exists():
        dataframe = read_csv_flex(FAILURES_CSV)
    else:
        dataframe = pd.DataFrame()

    if dataframe.empty:
        return dataframe

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if "equipment_code" not in dataframe.columns or "ttf_h" not in dataframe.columns:
        return pd.DataFrame()

    dataframe["equipment_code"] = dataframe["equipment_code"].astype(str)
    dataframe["ttf_h"] = pd.to_numeric(dataframe["ttf_h"], errors="coerce")

    if "duree_rep_h" in dataframe.columns:
        dataframe["duree_rep_h"] = pd.to_numeric(dataframe["duree_rep_h"], errors="coerce")
    else:
        dataframe["duree_rep_h"] = np.nan

    dataframe = dataframe.dropna(subset=["ttf_h"])
    dataframe = dataframe[dataframe["ttf_h"] > 0].reset_index(drop=True)
    return dataframe


def load_optimization_dataframe() -> pd.DataFrame:
    dataframe = st.session_state.get("optimization_df")
    if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
        output_dataframe = dataframe.copy()
    elif OPTIMIZATION_CSV.exists():
        output_dataframe = read_csv_flex(OPTIMIZATION_CSV)
    else:
        output_dataframe = pd.DataFrame()

    if output_dataframe.empty:
        return output_dataframe

    output_dataframe.columns = [str(column).strip() for column in output_dataframe.columns]
    return output_dataframe


def build_virtual_maintenance_plan_from_optimization(
    optimization_dataframe: pd.DataFrame,
    start_date: date,
    due_window_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if optimization_dataframe is None or optimization_dataframe.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, record in optimization_dataframe.iterrows():
        equipment_code = str(record.get("equipment_code", "")).strip()
        if not equipment_code:
            continue

        interval_hours = None
        interval_source = None
        for column_name in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
            value = safe_float(record.get(column_name), None)
            if value is not None and value > 0:
                interval_hours = value
                interval_source = column_name
                break

        if interval_hours is None:
            continue

        periodicity_days = max(1, int(round(interval_hours / 24.0)))
        next_due_date = start_date + timedelta(days=periodicity_days)
        days_left = int((next_due_date - start_date).days)

        rows.append(
            {
                "equipment_code": equipment_code,
                "maintenance_type": record.get("maintenance_type"),
                "interval_source": interval_source,
                "interval_h": interval_hours,
                "next_due_date": next_due_date.isoformat(),
                "days_left": days_left,
            }
        )

    all_rows_dataframe = pd.DataFrame(rows)
    if all_rows_dataframe.empty:
        return all_rows_dataframe, all_rows_dataframe

    due_rows_dataframe = all_rows_dataframe[all_rows_dataframe["days_left"] <= int(due_window_days)].copy()
    return all_rows_dataframe.reset_index(drop=True), due_rows_dataframe.reset_index(drop=True)


def process_score(process_name: str) -> int:
    normalized = (process_name or "").upper()
    if "NHPP" in normalized:
        return 3
    if "BPP" in normalized or "HAWKES" in normalized:
        return 2
    return 1


def compute_final_decision_row(row: pd.Series):
    score = 0
    process_name = str(row.get("model", "RP"))
    beta_value = safe_float(row.get("beta"), None)
    days_left = safe_float(row.get("days_left"), None)

    score += process_score(process_name)

    if beta_value is not None:
        if beta_value > 1.2:
            score += 3
        elif beta_value >= 1.0:
            score += 2
        elif beta_value < 0.8:
            score += 1

    if days_left is not None:
        if days_left <= 7:
            score += 3
        elif days_left <= 30:
            score += 2
        elif days_left <= 90:
            score += 1

    maintenance_type = str(row.get("maintenance_type", ""))

    if score >= 8:
        decision = "Intervention prioritaire"
        reason = (
            f"Le comportement fiabiliste et l’échéance imposent une action rapide. "
            f"Type d’action retenu : {maintenance_type or 'maintenance ciblée'}."
        )
        level = "Critique"
    elif score >= 5:
        decision = "Préventif renforcé"
        reason = "Le risque reste important. Une intervention planifiée rapide est recommandée."
        level = "Élevée"
    elif score >= 3:
        decision = "Surveillance active"
        reason = "La situation est intermédiaire. Il faut suivre le plan calculé et surveiller les dérives."
        level = "Modérée"
    else:
        decision = "Suivi nominal"
        reason = "Aucun signal critique immédiat n’est dominant. Le plan standard peut être appliqué."
        level = "Faible"

    return decision, reason, int(score), level


def format_param_string(reliability_result: Dict[str, Any]) -> str:
    params = reliability_result.get("params", {}) or {}
    model = str(reliability_result.get("model") or "").upper()
    distribution = str(reliability_result.get("distribution") or "")

    if model == "NHPP":
        alpha_v = params.get("alpha", params.get("eta"))
        return f"alpha = {format_number(alpha_v, 3)} ; beta = {format_number(params.get('beta'), 2)}"
    if model == "BPP":
        return (
            f"mu = {format_number(params.get('mu'), 3)} ; alpha = {format_number(params.get('alpha'), 3)} ; "
            f"beta_kernel = {format_number(params.get('beta_kernel'), 3)}"
        )
    if distribution.startswith("weibull"):
        return f"beta = {format_number(params.get('beta'), 2)} ; eta = {format_number(params.get('eta'), 1)} ; gamma = {format_number(params.get('gamma'), 1)}"
    if distribution == "norm":
        raw = params.get("raw") or []
        if len(raw) >= 2:
            return f"mu = {format_number(raw[0], 2)} ; sigma = {format_number(raw[1], 2)}"
    if distribution == "lognorm":
        raw = params.get("raw") or []
        if len(raw) >= 3:
            return f"sigma = {format_number(raw[0], 2)} ; loc = {format_number(raw[1], 2)} ; scale = {format_number(raw[2], 2)}"
    if distribution == "expon":
        return f"lambda = {format_number(params.get('lambda_hpp_h'), 5)}"
    raw_params = params.get("raw")
    if raw_params:
        return ", ".join(format_number(value, 3) for value in raw_params)
    return "—"


def simple_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def build_graphical_trend_plot(ttf_series: List[float], reliability_result: Dict[str, Any]):
    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    index = np.arange(1, len(event_times) + 1, dtype=float)
    graph = (reliability_result.get("tests", {}) or {}).get("trend_graphical", {}) or {}
    slope = safe_float(graph.get("slope_loglog"), 1.0)
    intercept = safe_float(graph.get("intercept_loglog"), 0.0)
    r2 = safe_float(graph.get("r2"), None)
    direction = str(graph.get("direction", "none"))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(event_times, index, s=32, label="Défaillances cumulées")
    if len(event_times) >= 2:
        fitted = np.exp(intercept + slope * np.log(event_times))
        ax.plot(event_times, fitted, linewidth=2.0, label=f"Ajustement log-log | pente={format_number(slope,2)} | R²={format_number(r2,3)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temps cumulé t (h)")
    ax.set_ylabel("Nombre cumulé N(t)")
    ax.set_title("Méthode graphique de tendance (Crow-AMSAA / log N(t) vs log t)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    legend_text = (
        f"Pente log-log = {format_number(slope,2)} ; direction = {direction}. "
        f"Si la pente est supérieure à 1, la tendance est croissante ; si elle est inférieure à 1, "
        f"la tendance est décroissante ; proche de 1, il n’y a pas de tendance nette."
    )
    return fig, legend_text


def build_graphical_dependence_plot(ttf_series: List[float], reliability_result: Dict[str, Any]):
    x = np.asarray(ttf_series[:-1], dtype=float)
    y = np.asarray(ttf_series[1:], dtype=float)
    graph = (reliability_result.get("tests", {}) or {}).get("dependence_graphical", {}) or {}
    slope = safe_float(graph.get("slope"), 0.0)
    intercept = safe_float(graph.get("intercept"), 0.0)
    r2 = safe_float(graph.get("r2"), None)
    lag1_r = safe_float(graph.get("lag1_r"), None)
    direction = str(graph.get("direction", "none"))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(x, y, s=32, label="Lag plot TTFᵢ vs TTFᵢ₊₁")
    if len(x) >= 2:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 150)
        ys = intercept + slope * xs
        ax.plot(xs, ys, linewidth=2.0, label=f"Droite ajustée | pente={format_number(slope,2)} | R²={format_number(r2,3)}")
    ax.set_xlabel("TTFᵢ (h)")
    ax.set_ylabel("TTFᵢ₊₁ (h)")
    ax.set_title("Méthode graphique de dépendance (lag plot)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    legend_text = (
        f"Corrélation lag-1 = {format_number(lag1_r,3)} ; direction = {direction}. "
        f"Une corrélation positive forte suggère une dépendance entre événements successifs."
    )
    return fig, legend_text


def build_reliability_curves_plot(reliability_result: Dict[str, Any]):
    curves = reliability_result.get("curves")
    if not isinstance(curves, pd.DataFrame) or curves.empty:
        return None, "Aucune courbe fiabiliste disponible."

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    axes = axes.ravel()
    defs = [("R_t", "Fiabilité R(t)"), ("F_t", "Défaillance F(t)"), ("f_t", "Densité f(t)"), ("h_t", "Taux λ(t) / h(t)")]
    for ax, (col, title) in zip(axes, defs):
        ax.plot(curves["t"], curves[col], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Temps (h)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    legend_text = (
        "Les quatre courbes résument le comportement du modèle retenu : R(t) pour la survie, F(t) pour la défaillance cumulée, "
        "f(t) pour la densité des défaillances et λ(t) / h(t) pour le risque instantané."
    )
    return fig, legend_text


def build_trend_tables(reliability_result: Dict[str, Any], alpha_value: float) -> Dict[str, pd.DataFrame]:
    tests = reliability_result.get("tests", {}) or {}
    graph = tests.get("trend_graphical", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    laplace = tests.get("trend_laplace", {}) or {}
    combined = tests.get("trend_combined", {}) or {}
    beta_graph = safe_float(graph.get("beta_graph"), None)
    mil_decision = "NHPP" if graph.get("has_trend") else "RP/HPP"

    return {
        "trend_graphical": simple_df([
            {
                "Méthode": "Graphique log N(t) vs log t",
                "Pente beta_graph": beta_graph,
                "R2": graph.get("r2"),
                "Direction": graph.get("direction"),
                "Signal": graph.get("graphical_signal"),
                "Décision": "Tendance détectée" if graph.get("has_trend") else "Pas de tendance détectée",
            }
        ]),
        "trend_mk": simple_df([
            {
                "Test": "Mann-Kendall",
                "Statistique Z": mk.get("z"),
                "Valeur p": mk.get("p"),
                "Direction": mk.get("direction"),
                "Seuil alpha": alpha_value,
                "Décision": "Rejet H0" if mk.get("has_trend") else "Non rejet H0",
            }
        ]),
        "trend_laplace": simple_df([
            {
                "Test": "Laplace",
                "Statistique U": laplace.get("u"),
                "Valeur p": laplace.get("p"),
                "Direction": laplace.get("direction"),
                "Seuil alpha": alpha_value,
                "Décision": "Rejet H0" if laplace.get("has_trend") else "Non rejet H0",
            }
        ]),
        "trend_mil": simple_df([
            {
                "Test": "MIL-HDBK-189 (interprétation graphique)",
                "Pente beta_graph": beta_graph,
                "R2": graph.get("r2"),
                "Interprétation": "Croissance" if (beta_graph is not None and beta_graph > 1.05) else "Décroissance" if (beta_graph is not None and beta_graph < 0.95) else "Stationnaire",
                "Décision": mil_decision,
            }
        ]),
        "trend_decision": simple_df([
            {
                "Synthèse tendance": "Tendance détectée" if combined.get("has_trend") else "Pas de tendance détectée",
                "Direction retenue": combined.get("direction"),
                "Niveau de confiance": combined.get("confidence"),
                "Processus orienté": "NHPP" if combined.get("has_trend") else "RP/HPP",
            }
        ]),
    }


def build_dependence_tables(reliability_result: Dict[str, Any], alpha_value: float) -> Dict[str, pd.DataFrame]:
    tests = reliability_result.get("tests", {}) or {}
    graph = tests.get("dependence_graphical", {}) or {}
    corr = tests.get("dependence_correlation", tests.get("dependence", {})) or {}
    dep = tests.get("dependence", {}) or {}
    return {
        "dep_graphical": simple_df([
            {
                "Méthode": "Lag plot",
                "Corrélation lag-1": graph.get("lag1_r"),
                "Pente": graph.get("slope"),
                "R2": graph.get("r2"),
                "Direction": graph.get("direction"),
                "Décision": "Dépendance détectée" if graph.get("has_dependence") else "Pas de dépendance graphique nette",
            }
        ]),
        "dep_pearson": simple_df([
            {
                "Test": "Pearson",
                "r": corr.get("pearson_r", corr.get("r")),
                "Valeur p": corr.get("pearson_p", corr.get("p")),
                "Seuil alpha": alpha_value,
                "Décision": "Rejet H0" if safe_float(corr.get("pearson_p", corr.get("p")), 1.0) < alpha_value else "Non rejet H0",
            }
        ]),
        "dep_spearman": simple_df([
            {
                "Test": "Spearman",
                "rho": corr.get("spearman_r", corr.get("r")),
                "Valeur p": corr.get("spearman_p", corr.get("p")),
                "Seuil alpha": alpha_value,
                "Décision": "Rejet H0" if safe_float(corr.get("spearman_p", corr.get("p")), 1.0) < alpha_value else "Non rejet H0",
            }
        ]),
        "dep_decision": simple_df([
            {
                "Synthèse dépendance": "Dépendance détectée" if dep.get("has_dep") else "Pas de dépendance détectée",
                "Coefficient retenu": dep.get("r"),
                "Valeur p retenue": dep.get("p"),
                "Processus orienté": "BPP" if dep.get("has_dep") else "RP/HPP",
            }
        ]),
    }


def build_process_table(reliability_result: Dict[str, Any]) -> pd.DataFrame:
    decision = reliability_result.get("decision", {}) or {}
    return simple_df([
        {
            "Processus retenu": reliability_result.get("model"),
            "Variant": reliability_result.get("process_variant"),
            "Loi retenue": reliability_result.get("distribution"),
            "Hypothèse entité": decision.get("entity_assumption", reliability_result.get("entity_assumption")),
            "Justification": decision.get("reason", reliability_result.get("reason")),
        }
    ])


def build_fit_tables(reliability_result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates = reliability_result.get("candidates", {}) or {}
    rows: List[Dict[str, Any]] = []
    for name, fit in candidates.items():
        if not isinstance(fit, dict):
            continue
        rows.append(
            {
                "Modèle candidat": name,
                "AIC": fit.get("aic"),
                "KS p": fit.get("ks_p"),
                "Chi2 p": fit.get("chi2_p"),
                "CvM p": fit.get("cvm_p"),
                "Ajustement accepté": fit.get("accepted"),
            }
        )
    if not rows:
        goodness = reliability_result.get("goodness", {}) or {}
        rows.append(
            {
                "Modèle candidat": reliability_result.get("distribution"),
                "AIC": goodness.get("aic"),
                "KS p": goodness.get("ks_p"),
                "Chi2 p": goodness.get("chi2_p"),
                "CvM p": goodness.get("cvm_p"),
                "Ajustement accepté": goodness.get("accepted"),
            }
        )
    selected = simple_df([
        {
            "Loi retenue": reliability_result.get("distribution"),
            "AIC": (reliability_result.get("goodness", {}) or {}).get("aic"),
            "KS p": (reliability_result.get("goodness", {}) or {}).get("ks_p"),
            "Chi2 p": (reliability_result.get("goodness", {}) or {}).get("chi2_p"),
            "CvM p": (reliability_result.get("goodness", {}) or {}).get("cvm_p"),
            "Ajustement accepté": (reliability_result.get("goodness", {}) or {}).get("accepted"),
        }
    ])
    return pd.DataFrame(rows), selected


def build_parameter_table(reliability_result: Dict[str, Any]) -> pd.DataFrame:
    params = reliability_result.get("params", {}) or {}
    indicators = reliability_result.get("indicators", {}) or {}
    availability = indicators.get("availability_intrinsic")
    return simple_df([
        {"Variable": "beta", "Valeur": params.get("beta"), "Lecture": "forme du vieillissement"},
        {"Variable": "eta_h", "Valeur": params.get("eta"), "Lecture": "durée de vie caractéristique"},
        {"Variable": "gamma_h", "Valeur": params.get("gamma"), "Lecture": "décalage temporel éventuel"},
        {"Variable": "MTTF_h", "Valeur": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"), "Lecture": "temps moyen avant défaillance"},
        {"Variable": "MTBF_h", "Valeur": indicators.get("mtbf_h"), "Lecture": "temps moyen entre défaillances"},
        {"Variable": "MTTR_h", "Valeur": indicators.get("mttr_h"), "Lecture": "temps moyen de réparation"},
        {"Variable": "Disponibilité_pct", "Valeur": None if availability is None else 100.0 * float(availability), "Lecture": "part du temps disponible"},
    ])


def build_optimization_table(summary_row: Dict[str, Any]) -> pd.DataFrame:
    return simple_df([
        {"Variable": "T_R_h", "Valeur": summary_row.get("T_R_h"), "Lecture": "intervalle issu de la fiabilité"},
        {"Variable": "T_cost_h", "Valeur": summary_row.get("T_cost_h"), "Lecture": "intervalle issu du coût"},
        {"Variable": "T_recommended_h", "Valeur": summary_row.get("T_recommended_h"), "Lecture": "intervalle recommandé"},
        {"Variable": "R(T_cost)", "Valeur": summary_row.get("R(T_cost)"), "Lecture": "fiabilité à l’intervalle économique"},
        {"Variable": "C_min_per_h", "Valeur": summary_row.get("C_min_per_h"), "Lecture": "coût minimal par heure"},
        {"Variable": "maintenance_type", "Valeur": summary_row.get("maintenance_type"), "Lecture": "type de maintenance proposé"},
        {"Variable": "days_left", "Valeur": summary_row.get("days_left"), "Lecture": "jours restants avant maintenance"},
    ])


def build_final_decision_table(summary_row: Dict[str, Any]) -> pd.DataFrame:
    return simple_df([
        {
            "Priorité": summary_row.get("priorite"),
            "Score": summary_row.get("priority_score"),
            "Décision finale": summary_row.get("decision_finale"),
            "Motif": summary_row.get("motif_decision"),
        }
    ])


def build_global_trend_table(summary_dataframe: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["equipment_code", "trend_detected", "trend_direction", "mk_p", "laplace_p", "model"] if c in summary_dataframe.columns]
    return summary_dataframe[cols].copy()


def build_global_dependence_table(summary_dataframe: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["equipment_code", "dependence_detected", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "model"] if c in summary_dataframe.columns]
    return summary_dataframe[cols].copy()


def build_global_fit_table(summary_dataframe: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["equipment_code", "distribution", "aic", "ks_p", "chi2_p", "cvm_p", "goodness_accepted"] if c in summary_dataframe.columns]
    return summary_dataframe[cols].copy()


def build_global_replacement_table(summary_dataframe: pd.DataFrame, current_pm_hours: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in summary_dataframe.iterrows():
        rows.append(
            {
                "Équipement": row.get("equipment_code"),
                "Loi": row.get("distribution"),
                "Paramètres": row.get("param_string"),
                "MTBF (h)": row.get("mtbf_h"),
                "MP actuelle (h)": current_pm_hours,
                "Remplacement fiabiliste (h)": row.get("T_R_h") or row.get("T_recommended_h"),
                "Remplacement économique (h)": row.get("T_cost_h"),
                "Remplacement recommandé (h)": row.get("T_recommended_h"),
            }
        )
    return pd.DataFrame(rows)


def build_excel_bytes(global_tables: Dict[str, pd.DataFrame], detail_tables_by_equipment: Dict[str, Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, dataframe in global_tables.items():
            if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                dataframe.to_excel(writer, sheet_name=name[:31], index=False)

        for equipment_code, tables in detail_tables_by_equipment.items():
            for table_name, dataframe in tables.items():
                if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                    try:
                        dataframe.to_excel(writer, sheet_name=f"{equipment_code}_{table_name}"[:31], index=False)
                    except Exception:
                        pass
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
control_col_1, control_col_2, control_col_3 = st.columns(3)
with control_col_1:
    uploaded_failures_csv = st.file_uploader("Fichier CSV des temps entre défaillances (optionnel)", type=["csv"], key="global_uploaded_failures")
with control_col_2:
    alpha_value = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
with control_col_3:
    due_window_days = st.slider("Fenêtre de maintenance (jours)", 7, 365, 30, 1)

reference_start_date = st.date_input("Date de référence", value=date.today())
current_pm_hours = st.number_input("Maintenance préventive actuelle (heures)", min_value=1.0, value=500.0, step=10.0)

with st.expander("Comprendre le contenu de cette page", expanded=False):
    st.markdown(
        """
Cette page déroule tout le pipeline, depuis les tests de **tendance** et de **dépendance** jusqu’au
calcul des **paramètres fiabilistes**, à l’**optimisation** et à la **décision finale**.

Dans le détail par équipement, vous verrez :
- la **méthode graphique** de tendance avec son graphique et sa légende détaillée ;
- les tableaux séparés du **test de Mann-Kendall**, du **test de Laplace** et de l’interprétation **MIL-HDBK-189** ;
- la **méthode graphique de dépendance** avec son graphique et sa légende ;
- les tableaux **Pearson**, **Spearman** et la décision de dépendance ;
- le **choix du modèle**, l’**ajustement**, les **paramètres**, l’**optimisation** et la **décision finale**.
        """
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
failures_dataframe = load_failures_dataframe(uploaded_failures_csv)
if failures_dataframe.empty:
    st.error("Aucun jeu de données de temps entre défaillances n’est disponible.")
    st.stop()

optimization_dataframe = load_optimization_dataframe()
maintenance_all_rows = st.session_state.get("pm_virtual_all")
maintenance_due_rows = st.session_state.get("pm_virtual_due")
if isinstance(maintenance_all_rows, list) and isinstance(maintenance_due_rows, list):
    maintenance_all_dataframe = pd.DataFrame(maintenance_all_rows)
    maintenance_due_dataframe = pd.DataFrame(maintenance_due_rows)
else:
    maintenance_all_dataframe, maintenance_due_dataframe = build_virtual_maintenance_plan_from_optimization(
        optimization_dataframe,
        reference_start_date,
        due_window_days,
    )


# -----------------------------------------------------------------------------
# Global analysis
# -----------------------------------------------------------------------------
with st.spinner("Analyse globale en cours..."):
    results_by_equipment: Dict[str, Dict[str, Any]] = {}
    detail_tables_by_equipment: Dict[str, Dict[str, Any]] = {}
    global_rows: List[Dict[str, Any]] = []

    equipment_codes = sorted(failures_dataframe["equipment_code"].astype(str).unique().tolist())

    for equipment_code in equipment_codes:
        equipment_dataframe = failures_dataframe[failures_dataframe["equipment_code"].astype(str) == str(equipment_code)].copy()
        ttf_series = series_to_positive_list(equipment_dataframe["ttf_h"])
        if not ttf_series or len(ttf_series) < 3:
            continue

        repair_series = None
        if "duree_rep_h" in equipment_dataframe.columns:
            repair_series = series_to_positive_list(equipment_dataframe["duree_rep_h"])

        try:
            result = analyze_ttf_pipeline(ttf_series=ttf_series, alpha=float(alpha_value), repair_series=repair_series)
        except Exception as error:
            st.warning(f"{equipment_code} : analyse impossible ({error})")
            continue

        reliability_result = result.get("reliability", {}) or {}
        tests = reliability_result.get("tests", {}) or {}
        indicators = reliability_result.get("indicators", {}) or {}
        goodness = reliability_result.get("goodness", {}) or {}
        decision = reliability_result.get("decision", {}) or {}
        params = reliability_result.get("params", {}) or {}

        optimization_row = {}
        if isinstance(optimization_dataframe, pd.DataFrame) and not optimization_dataframe.empty and "equipment_code" in optimization_dataframe.columns:
            matched = optimization_dataframe[optimization_dataframe["equipment_code"].astype(str) == str(equipment_code)]
            if not matched.empty:
                optimization_row = matched.iloc[0].to_dict()

        maintenance_row = {}
        if isinstance(maintenance_all_dataframe, pd.DataFrame) and not maintenance_all_dataframe.empty and "equipment_code" in maintenance_all_dataframe.columns:
            matched = maintenance_all_dataframe[maintenance_all_dataframe["equipment_code"].astype(str) == str(equipment_code)]
            if not matched.empty:
                maintenance_row = matched.iloc[0].to_dict()

        trend_tables = build_trend_tables(reliability_result, alpha_value)
        dependence_tables = build_dependence_tables(reliability_result, alpha_value)
        fit_candidates_df, fit_selected_df = build_fit_tables(reliability_result)
        process_df = build_process_table(reliability_result)
        parameter_df = build_parameter_table(reliability_result)

        row = {
            "equipment_code": equipment_code,
            "n_ttf": len(ttf_series),
            "trend_detected": "Oui" if decision.get("has_trend") else "Non",
            "trend_direction": decision.get("trend_direction"),
            "dependence_detected": "Oui" if decision.get("has_dependence") else "Non",
            "model": reliability_result.get("model"),
            "process_variant": reliability_result.get("process_variant"),
            "distribution": reliability_result.get("distribution"),
            "mk_p": (tests.get("trend_mk", {}) or {}).get("p"),
            "laplace_p": (tests.get("trend_laplace", {}) or {}).get("p"),
            "pearson_r": (tests.get("dependence_correlation", {}) or {}).get("pearson_r"),
            "pearson_p": (tests.get("dependence_correlation", {}) or {}).get("pearson_p"),
            "spearman_r": (tests.get("dependence_correlation", {}) or {}).get("spearman_r"),
            "spearman_p": (tests.get("dependence_correlation", {}) or {}).get("spearman_p"),
            "aic": goodness.get("aic"),
            "ks_p": goodness.get("ks_p"),
            "chi2_p": goodness.get("chi2_p"),
            "cvm_p": goodness.get("cvm_p"),
            "goodness_accepted": goodness.get("accepted"),
            "beta": params.get("beta", optimization_row.get("beta")),
            "eta_h": params.get("eta", optimization_row.get("eta_h")),
            "gamma_h": params.get("gamma", optimization_row.get("gamma_h")),
            "mtbf_h": indicators.get("mtbf_h"),
            "mttr_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
            "maintenance_type": optimization_row.get("maintenance_type"),
            "T_recommended_h": optimization_row.get("T_recommended_h"),
            "T_R_h": optimization_row.get("T_R_h"),
            "T_cost_h": optimization_row.get("T_cost_h"),
            "R(T_cost)": optimization_row.get("R(T_cost)"),
            "C_min_per_h": optimization_row.get("C_min_per_h"),
            "next_due_date": maintenance_row.get("next_due_date"),
            "days_left": maintenance_row.get("days_left"),
        }
        row["param_string"] = format_param_string(reliability_result)
        global_rows.append(row)

        results_by_equipment[equipment_code] = result
        detail_tables_by_equipment[equipment_code] = {
            **trend_tables,
            **dependence_tables,
            "process_choice": process_df,
            "fit_candidates": fit_candidates_df,
            "fit_selected": fit_selected_df,
            "parameter_table": parameter_df,
            "__payload__": {
                "ttf_series": ttf_series,
                "alpha": alpha_value,
                "reliability": reliability_result,
            },
        }

summary_dataframe = pd.DataFrame(global_rows)
if summary_dataframe.empty:
    st.error("Aucun équipement exploitable n’a pu être analysé.")
    st.stop()

final_decisions = summary_dataframe.apply(lambda row: compute_final_decision_row(row), axis=1)
summary_dataframe[["decision_finale", "motif_decision", "priority_score", "priorite"]] = pd.DataFrame(final_decisions.tolist(), index=summary_dataframe.index)
summary_dataframe = summary_dataframe.sort_values(["priority_score", "equipment_code"], ascending=[False, True]).reset_index(drop=True)

# enrich detail tables with optimization + final decision after summary exists
for equipment_code, detail in detail_tables_by_equipment.items():
    selected_row = summary_dataframe[summary_dataframe["equipment_code"].astype(str) == str(equipment_code)]
    if selected_row.empty:
        continue
    row_dict = selected_row.iloc[0].to_dict()
    detail["optimization_table"] = build_optimization_table(row_dict)
    detail["final_decision_table"] = build_final_decision_table(row_dict)

trend_overview_dataframe = build_global_trend_table(summary_dataframe)
dependence_overview_dataframe = build_global_dependence_table(summary_dataframe)
fit_overview_dataframe = build_global_fit_table(summary_dataframe)
replacement_overview_dataframe = build_global_replacement_table(summary_dataframe, current_pm_hours)
final_decision_dataframe = summary_dataframe[["equipment_code", "model", "distribution", "maintenance_type", "days_left", "priority_score", "priorite", "decision_finale", "motif_decision"]].copy()
due_tasks_dataframe = maintenance_due_dataframe.copy() if not maintenance_due_dataframe.empty else pd.DataFrame()

global_tables = {
    "Synthese_globale": summary_dataframe,
    "Tableau_tendance_global": trend_overview_dataframe,
    "Tableau_dependance_global": dependence_overview_dataframe,
    "Tableau_ajustement_global": fit_overview_dataframe,
    "Tableau_remplacement_global": replacement_overview_dataframe,
    "Tableau_decision_globale": final_decision_dataframe,
    "Tableau_taches_dues": due_tasks_dataframe,
}


# -----------------------------------------------------------------------------
# Top metrics
# -----------------------------------------------------------------------------
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Équipements analysés", len(summary_dataframe))
with metric_col_2:
    st.metric("Priorité critique", int((summary_dataframe["priorite"].astype(str) == "Critique").sum()))
with metric_col_3:
    st.metric("Tâches dues", len(due_tasks_dataframe))
with metric_col_4:
    st.metric("NHPP détectés", int((summary_dataframe["model"].astype(str).str.upper() == "NHPP").sum()))


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Vue globale", "Pipeline par équipement", "Synthèse mémoire", "Exports"])

with tab1:
    st.subheader("Résumé global")
    st.dataframe(summary_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Décision globale")
    st.dataframe(final_decision_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Remplacement / optimisation")
    st.dataframe(replacement_overview_dataframe, use_container_width=True, hide_index=True)

with tab2:
    selected_equipment_code = st.selectbox("Choisir un équipement", options=summary_dataframe["equipment_code"].tolist())
    selected_result = results_by_equipment[selected_equipment_code]
    selected_detail = detail_tables_by_equipment[selected_equipment_code]
    selected_row = summary_dataframe[summary_dataframe["equipment_code"] == selected_equipment_code].iloc[0].to_dict()
    payload = selected_detail.get("__payload__", {}) or {}
    reliability_result = payload.get("reliability", {}) or {}
    ttf_series = payload.get("ttf_series", []) or []

    st.markdown(f"## Équipement {selected_equipment_code}")

    st.markdown("### 1. Tendance")
    trend_fig, trend_legend = build_graphical_trend_plot(ttf_series, reliability_result)
    st.pyplot(trend_fig, clear_figure=True)
    st.caption(trend_legend)
    st.dataframe(selected_detail["trend_graphical"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["trend_mk"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["trend_laplace"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["trend_mil"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["trend_decision"], use_container_width=True, hide_index=True)

    st.markdown("### 2. Dépendance")
    if len(ttf_series) >= 3:
        dep_fig, dep_legend = build_graphical_dependence_plot(ttf_series, reliability_result)
        st.pyplot(dep_fig, clear_figure=True)
        st.caption(dep_legend)
    st.dataframe(selected_detail["dep_graphical"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["dep_pearson"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["dep_spearman"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["dep_decision"], use_container_width=True, hide_index=True)

    st.markdown("### 3. Choix du processus")
    st.dataframe(selected_detail["process_choice"], use_container_width=True, hide_index=True)

    st.markdown("### 4. Ajustement")
    st.dataframe(selected_detail["fit_candidates"], use_container_width=True, hide_index=True)
    st.dataframe(selected_detail["fit_selected"], use_container_width=True, hide_index=True)

    st.markdown("### 5. Paramètres calculés")
    st.dataframe(selected_detail["parameter_table"], use_container_width=True, hide_index=True)

    st.markdown("### 6. Courbes fiabilistes")
    rel_fig, rel_legend = build_reliability_curves_plot(reliability_result)
    if rel_fig is not None:
        st.pyplot(rel_fig, clear_figure=True)
    st.caption(rel_legend)

    st.markdown("### 7. Optimisation")
    st.dataframe(selected_detail["optimization_table"], use_container_width=True, hide_index=True)

    st.markdown("### 8. Décision finale")
    st.dataframe(selected_detail["final_decision_table"], use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Tableau global du test de tendance")
    st.dataframe(trend_overview_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau global du test de dépendance")
    st.dataframe(dependence_overview_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau global d’ajustement")
    st.dataframe(fit_overview_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau loi, paramètres et remplacement")
    st.dataframe(replacement_overview_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau de décision finale")
    st.dataframe(final_decision_dataframe, use_container_width=True, hide_index=True)

with tab4:
    excel_bytes = build_excel_bytes(global_tables, detail_tables_by_equipment)
    st.download_button(
        "Télécharger le pack Excel global",
        data=excel_bytes,
        file_name="resultat_global_pipeline.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_global_analysis_report_pdf is None:
        st.info("Le module PDF global n’est pas disponible.")
    else:
        if st.button("Générer le rapport PDF global", use_container_width=True):
            try:
                pdf_path = export_global_analysis_report_pdf(
                    summary_df=summary_dataframe,
                    global_tables=global_tables,
                    detail_tables_by_eq=detail_tables_by_equipment,
                    out_dir=str(BASE_DIR / "reports"),
                    title="Résultat global de l'analyse et de l'optimisation de maintenance",
                    meta={
                        "alpha": alpha_value,
                        "window_days": due_window_days,
                        "start_date": str(reference_start_date),
                        "number_of_equipment": len(summary_dataframe),
                    },
                )
                st.session_state["global_pdf_path"] = pdf_path
                st.success(f"PDF généré : {pdf_path}")
            except Exception as error:
                st.error(f"PDF : {error}")

        pdf_path = st.session_state.get("global_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as file:
                st.download_button(
                    "Télécharger le PDF global",
                    data=file,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
