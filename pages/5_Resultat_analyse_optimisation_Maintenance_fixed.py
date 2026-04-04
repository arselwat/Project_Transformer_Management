
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional

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


st.set_page_config(
    page_title="Résultat global",
    page_icon="📋",
    layout="wide",
)

require_login()

render_shell("pages/5_Resultat_analyse_optimisation_Maintenance.py")
render_page_header(
    "Résultat global",
    "Synthèse fiabiliste, optimisation, maintenance et décision finale, présentées sous forme de tableaux de type mémoire.",
    "📋",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FAILURES_CSV = DATA_DIR / "failures_saved.csv"
OPTIMIZATION_CSV = DATA_DIR / "last_optimization.csv"


# =========================================================
# Helpers
# =========================================================
def read_csv_flex(source) -> pd.DataFrame:
    def try_read(obj, **kwargs):
        try:
            return pd.read_csv(obj, **kwargs)
        except Exception:
            return None

    dataframe = try_read(source)
    if dataframe is None:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        dataframe = try_read(source, engine="python", on_bad_lines="skip", sep=None)

    if dataframe is None:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        dataframe = try_read(source, sep=";", engine="python", on_bad_lines="skip")

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


def format_number(value: Any, decimals: int = 2, default: str = "—") -> str:
    numeric_value = safe_float(value, None)
    return default if numeric_value is None else f"{numeric_value:.{decimals}f}"


def format_yes_no(value: Any) -> str:
    if value is True:
        return "Oui"
    if value is False:
        return "Non"
    return "—"


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def critical_component_from_row(row: Dict[str, Any]) -> str:
    aliases = [
        "critical_component",
        "composant_critique",
        "component_critical",
        "component",
        "subsystem",
        "sous_systeme",
        "sous-système",
    ]
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "Composant principal"


def format_param_string(reliability_result: Dict[str, Any]) -> str:
    params = reliability_result.get("params", {}) or {}
    model = str(reliability_result.get("model") or "").upper()
    distribution = str(reliability_result.get("distribution") or "")

    if model == "NHPP":
        beta_value = format_number(params.get("beta"), 2)
        eta_value = format_number(params.get("eta"), 1)
        return f"β = {beta_value} ; η = {eta_value}"
    if model == "BPP":
        mu_value = format_number(params.get("mu"), 3)
        alpha_value = format_number(params.get("alpha"), 3)
        beta_kernel_value = format_number(params.get("beta_kernel"), 3)
        return f"μ = {mu_value} ; α = {alpha_value} ; βk = {beta_kernel_value}"
    if distribution.startswith("weibull"):
        beta_value = format_number(params.get("beta"), 2)
        eta_value = format_number(params.get("eta"), 1)
        gamma_value = format_number(params.get("gamma"), 1)
        return f"β = {beta_value} ; η = {eta_value} ; γ = {gamma_value}"
    if distribution == "expon":
        lambda_value = format_number(params.get("lambda_hpp_h"), 5)
        return f"λ = {lambda_value}"
    raw_params = params.get("raw")
    if raw_params:
        return ", ".join(format_number(value, 3) for value in raw_params)
    return "—"


def find_optional_value(row: Dict[str, Any], aliases: List[str], default: Any = None) -> Any:
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def build_trend_global_table(results_by_equipment: Dict[str, Dict[str, Any]], alpha_value: float) -> pd.DataFrame:
    z_critical = float(sst.norm.ppf(1.0 - alpha_value / 2.0))
    rows: List[Dict[str, Any]] = []

    for equipment_code, result in results_by_equipment.items():
        reliability_result = result.get("reliability", {}) or {}
        tests = reliability_result.get("tests", {}) or {}
        laplace_result = tests.get("trend_laplace", {}) or {}
        mann_kendall_result = tests.get("trend_mk", {}) or {}
        graphical_result = tests.get("trend_graphical", {}) or tests.get("trend_mil_hdbk_189", {}) or {}
        decision = reliability_result.get("decision", {}) or {}
        n_value = reliability_result.get("cleaned_n")

        laplace_stat = safe_float(laplace_result.get("u"), 0.0)
        laplace_rejected = abs(laplace_stat) > z_critical
        laplace_decision = (
            f"rejetée : NHPP ({format_number(laplace_stat, 2)})"
            if laplace_rejected
            else f"non rejetée : RP/HPP ({format_number(laplace_stat, 2)})"
        )

        mk_p = safe_float(mann_kendall_result.get("p"), None)
        mk_decision = "rejetée : tendance" if (mk_p is not None and mk_p < alpha_value) else "non rejetée"

        graphical_direction = str(graphical_result.get("direction", graphical_result.get("interpreted_trend", "none")))
        graphical_decision = "rejetée : NHPP" if graphical_direction in {"up", "down"} else "non rejetée : RP/HPP"

        rows.append(
            {
                "Sous-système": f"{equipment_code} (n = {n_value})",
                "Test de Laplace (ZL)": format_number(laplace_stat, 2),
                "Seuil critique": f"{format_number(-z_critical, 3)} < ZL < {format_number(z_critical, 3)}",
                "Décision Laplace": laplace_decision,
                "Méthode graphique": f"pente = {format_number(graphical_result.get('beta_graph', graphical_result.get('slope_loglog')), 3)}",
                "Décision graphique": graphical_decision,
                "Mann-Kendall p-valeur": format_number(mk_p, 4),
                "Décision MK": mk_decision,
                "Décision finale": "rejetée : NHPP" if decision.get("has_trend") else "non rejetée : RP/HPP",
            }
        )

    return pd.DataFrame(rows)


def build_dependence_global_table(results_by_equipment: Dict[str, Dict[str, Any]], alpha_value: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for equipment_code, result in results_by_equipment.items():
        reliability_result = result.get("reliability", {}) or {}
        tests = reliability_result.get("tests", {}) or {}
        dependence_result = tests.get("dependence", {}) or {}
        graphical_dependence = tests.get("dependence_graphical", {}) or {}
        n_value = reliability_result.get("cleaned_n")

        pearson_p = safe_float(dependence_result.get("pearson_p"), None)
        spearman_p = safe_float(dependence_result.get("spearman_p"), None)
        final_decision = "rejetée : BPP" if dependence_result.get("has_dep") else "non rejetée : RP/HPP"

        rows.append(
            {
                "Sous-système": f"{equipment_code} (n = {n_value})",
                "r1 graphique": format_number(graphical_dependence.get("lag1_r"), 3),
                "Pearson": format_number(dependence_result.get("pearson_r"), 3),
                "p-valeur Pearson": format_number(pearson_p, 4),
                "Spearman": format_number(dependence_result.get("spearman_r"), 3),
                "p-valeur Spearman": format_number(spearman_p, 4),
                "Décision": final_decision,
            }
        )

    return pd.DataFrame(rows)


def build_synthesis_table(
    summary_dataframe: pd.DataFrame,
    failures_dataframe: pd.DataFrame,
    current_pm_hours: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in summary_dataframe.iterrows():
        equipment_code = str(row.get("equipment_code"))
        source_rows = failures_dataframe[failures_dataframe["equipment_code"].astype(str) == equipment_code]
        sample_row = source_rows.iloc[0].to_dict() if not source_rows.empty else {}

        replacement_age = find_optional_value(
            row.to_dict(),
            ["T_age_h", "age_replacement_h", "replacement_age_h", "T_cost_h"],
            default=None,
        )
        replacement_block = find_optional_value(
            row.to_dict(),
            ["T_block_h", "block_replacement_h", "replacement_block_h"],
            default=None,
        )
        fiability_interval = find_optional_value(
            row.to_dict(),
            ["T_R_h", "T_recommended_h"],
            default=None,
        )

        rows.append(
            {
                "Equipement": equipment_code,
                "Composant critique": critical_component_from_row(sample_row),
                "Loi": row.get("distribution"),
                "Paramètres": row.get("param_string"),
                "MTBF (h)": row.get("mtbf_h"),
                "MP actuelle (h)": current_pm_hours,
                "Remplacement optimisé approche fiabiliste (h)": fiability_interval,
                "Remplacement optimisé type âge (h)": replacement_age,
                "Remplacement optimisé type bloc (h)": replacement_block,
            }
        )

    return pd.DataFrame(rows)


def build_final_decision_table(summary_dataframe: pd.DataFrame) -> pd.DataFrame:
    return summary_dataframe[
        [
            "equipment_code",
            "model",
            "process_variant",
            "distribution",
            "maintenance_type",
            "T_recommended_h",
            "days_left",
            "priority_score",
            "priorite",
            "decision_finale",
            "motif_decision",
        ]
    ].copy()


def build_detail_excel_tables(result: Dict[str, Any], row: Dict[str, Any], alpha_value: float) -> Dict[str, pd.DataFrame]:
    reliability_result = result.get("reliability", {}) or {}
    tests = reliability_result.get("tests", {}) or {}
    tables = result.get("tables", {}) or {}

    z_critical = float(sst.norm.ppf(1.0 - alpha_value / 2.0))
    laplace_result = tests.get("trend_laplace", {}) or {}
    mk_result = tests.get("trend_mk", {}) or {}
    graphical_trend = tests.get("trend_graphical", {}) or tests.get("trend_mil_hdbk_189", {}) or {}
    dependence_result = tests.get("dependence", {}) or {}
    graphical_dependence = tests.get("dependence_graphical", {}) or {}

    table_trend = pd.DataFrame(
        [
            {
                "Test": "Laplace",
                "Statistique": format_number(laplace_result.get("u"), 3),
                "Seuil critique": f"{format_number(-z_critical, 3)} < ZL < {format_number(z_critical, 3)}",
                "Décision": "rejetée : NHPP" if abs(safe_float(laplace_result.get("u"), 0.0)) > z_critical else "non rejetée : RP/HPP",
            },
            {
                "Test": "Mann-Kendall",
                "Statistique": format_number(mk_result.get("z"), 3),
                "Seuil critique": f"p < {format_number(alpha_value, 3)}",
                "Décision": "rejetée : tendance" if safe_float(mk_result.get("p"), 1.0) < alpha_value else "non rejetée",
            },
            {
                "Test": "Méthode graphique",
                "Statistique": format_number(graphical_trend.get("beta_graph", graphical_trend.get("slope_loglog")), 3),
                "Seuil critique": "pente hors [0.95 ; 1.05]",
                "Décision": "rejetée : NHPP" if str(graphical_trend.get("direction", graphical_trend.get("interpreted_trend", "none"))) in {"up", "down"} else "non rejetée : RP/HPP",
            },
        ]
    )

    table_dependence = pd.DataFrame(
        [
            {
                "Méthode": "Graphique",
                "Coefficient": format_number(graphical_dependence.get("lag1_r"), 3),
                "p-valeur": "—",
                "Décision": "signal qualitatif",
            },
            {
                "Méthode": "Pearson",
                "Coefficient": format_number(dependence_result.get("pearson_r"), 3),
                "p-valeur": format_number(dependence_result.get("pearson_p"), 4),
                "Décision": "rejetée : BPP" if safe_float(dependence_result.get("pearson_p"), 1.0) < alpha_value else "non rejetée : RP/HPP",
            },
            {
                "Méthode": "Spearman",
                "Coefficient": format_number(dependence_result.get("spearman_r"), 3),
                "p-valeur": format_number(dependence_result.get("spearman_p"), 4),
                "Décision": "rejetée : BPP" if safe_float(dependence_result.get("spearman_p"), 1.0) < alpha_value else "non rejetée : RP/HPP",
            },
        ]
    )

    table_parameters = pd.DataFrame(
        [
            {
                "Variable": "Paramètre bêta",
                "Valeur": format_number(row.get("beta"), 3),
                "Lecture": "Usure" if safe_float(row.get("beta"), 1.0) > 1.2 else "Aléatoire" if safe_float(row.get("beta"), 1.0) >= 0.8 else "Défauts précoces",
            },
            {
                "Variable": "Paramètre êta (h)",
                "Valeur": format_number(row.get("eta_h"), 1),
                "Lecture": "Durée de vie caractéristique",
            },
            {
                "Variable": "Paramètre gamma (h)",
                "Valeur": format_number(row.get("gamma_h"), 1),
                "Lecture": "Décalage éventuel du modèle",
            },
            {
                "Variable": "MTBF (h)",
                "Valeur": format_number(row.get("mtbf_h"), 1),
                "Lecture": "Temps moyen entre défaillances",
            },
            {
                "Variable": "MTTR (h)",
                "Valeur": format_number(row.get("mttr_h"), 1),
                "Lecture": "Temps moyen de réparation",
            },
            {
                "Variable": "Disponibilité (%)",
                "Valeur": format_number(row.get("availability_pct"), 2),
                "Lecture": "Disponibilité intrinsèque",
            },
        ]
    )

    table_optimization = pd.DataFrame(
        [
            {
                "Variable": "Intervalle fiabiliste (h)",
                "Valeur": format_number(row.get("T_R_h"), 1),
                "Lecture": "Intervalle issu du critère de fiabilité",
            },
            {
                "Variable": "Intervalle économique (h)",
                "Valeur": format_number(row.get("T_cost_h"), 1),
                "Lecture": "Intervalle issu du critère économique",
            },
            {
                "Variable": "Intervalle recommandé (h)",
                "Valeur": format_number(row.get("T_recommended_h"), 1),
                "Lecture": "Intervalle retenu pour l’action",
            },
            {
                "Variable": "Type de maintenance",
                "Valeur": row.get("maintenance_type", "—"),
                "Lecture": "Traduction opérationnelle",
            },
            {
                "Variable": "Jours restants",
                "Valeur": row.get("days_left", "—"),
                "Lecture": "Échéance calculée",
            },
        ]
    )

    table_decision = pd.DataFrame(
        [
            {
                "Variable": "Score de priorité",
                "Valeur": row.get("priority_score"),
                "Lecture": "Score global de hiérarchisation",
            },
            {
                "Variable": "Niveau de priorité",
                "Valeur": row.get("priorite"),
                "Lecture": "Classement final",
            },
            {
                "Variable": "Décision finale",
                "Valeur": row.get("decision_finale"),
                "Lecture": row.get("motif_decision"),
            },
        ]
    )

    out = {}
    out.update(tables)
    out["tableau_tendance"] = table_trend
    out["tableau_dependance"] = table_dependence
    out["tableau_parametres"] = table_parameters
    out["tableau_optimisation"] = table_optimization
    out["tableau_decision_finale"] = table_decision
    return out


def build_excel_bytes(
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_equipment: Dict[str, Dict[str, pd.DataFrame]],
) -> bytes:
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


# =========================================================
# Contrôles page
# =========================================================
control_col_1, control_col_2, control_col_3, control_col_4 = st.columns([1, 1, 1, 1])
with control_col_1:
    uploaded_failures_csv = st.file_uploader(
        "Fichier CSV des temps entre défaillances (optionnel)",
        type=["csv"],
        key="global_uploaded_failures",
    )
with control_col_2:
    alpha_value = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
with control_col_3:
    due_window_days = st.slider("Fenêtre de maintenance (jours)", 7, 365, 30, 1)
with control_col_4:
    current_pm_hours = st.number_input("MP actuelle (h)", min_value=1.0, value=500.0, step=10.0)

reference_start_date = st.date_input("Date de référence", value=date.today())

with st.expander("Comprendre clairement les tableaux affichés sur cette page", expanded=False):
    st.markdown(
        """
Cette page présente les résultats sous forme de **tableaux proches de ceux d’un mémoire** :

- un tableau global de **validation des tests de tendance** ;
- un tableau global de **validation des tests de dépendance** ;
- un tableau de **loi, paramètres et temps de remplacement** ;
- un tableau de **décision finale hiérarchisée** ;
- puis, pour chaque équipement, les **tableaux détaillés** reprenant la logique complète du pipeline.
        """
    )


# =========================================================
# Chargement données
# =========================================================
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

info_col_1, info_col_2, info_col_3 = st.columns(3)
with info_col_1:
    st.info(f"Temps entre défaillances actifs : {len(failures_dataframe)} lignes")
with info_col_2:
    st.info(f"Optimisation disponible : {'Oui' if not optimization_dataframe.empty else 'Non'}")
with info_col_3:
    st.info(f"Planning maintenance disponible : {'Oui' if not maintenance_all_dataframe.empty else 'Non'}")


# =========================================================
# Analyse globale
# =========================================================
with st.spinner("Analyse globale en cours..."):
    results_by_equipment: Dict[str, Dict[str, Any]] = {}
    detail_tables_by_equipment: Dict[str, Dict[str, pd.DataFrame]] = {}
    global_rows: List[Dict[str, Any]] = []

    equipment_codes = sorted(failures_dataframe["equipment_code"].astype(str).unique().tolist())

    for equipment_code in equipment_codes:
        equipment_dataframe = failures_dataframe[failures_dataframe["equipment_code"].astype(str) == str(equipment_code)].copy()
        time_to_failure_series = equipment_dataframe["ttf_h"].dropna().astype(float).tolist()
        if len(time_to_failure_series) < 3:
            continue

        repair_time_series = None
        if "duree_rep_h" in equipment_dataframe.columns:
            extracted_repair_time_series = pd.to_numeric(
                equipment_dataframe["duree_rep_h"],
                errors="coerce",
            ).dropna().tolist()
            repair_time_series = extracted_repair_time_series if extracted_repair_time_series else None

        try:
            result = analyze_ttf_pipeline(
                ttf_series=time_to_failure_series,
                alpha=float(alpha_value),
                repair_series=repair_time_series,
            )
        except Exception as error:
            st.warning(f"{equipment_code} : analyse impossible ({error})")
            continue

        results_by_equipment[equipment_code] = result

        reliability_result = result.get("reliability", {}) or {}
        indicators = reliability_result.get("indicators", {}) or {}
        parameters = reliability_result.get("params", {}) or {}
        decision = reliability_result.get("decision", {}) or {}
        goodness = reliability_result.get("goodness", {}) or {}

        optimization_row = {}
        if isinstance(optimization_dataframe, pd.DataFrame) and not optimization_dataframe.empty and "equipment_code" in optimization_dataframe.columns:
            matched_optimization_rows = optimization_dataframe[optimization_dataframe["equipment_code"].astype(str) == str(equipment_code)]
            if not matched_optimization_rows.empty:
                optimization_row = matched_optimization_rows.iloc[0].to_dict()

        maintenance_row = {}
        if isinstance(maintenance_all_dataframe, pd.DataFrame) and not maintenance_all_dataframe.empty and "equipment_code" in maintenance_all_dataframe.columns:
            matched_maintenance_rows = maintenance_all_dataframe[maintenance_all_dataframe["equipment_code"].astype(str) == str(equipment_code)]
            if not matched_maintenance_rows.empty:
                maintenance_row = matched_maintenance_rows.iloc[0].to_dict()

        current_row = {
            "equipment_code": equipment_code,
            "n_ttf": len(time_to_failure_series),
            "trend_detected": decision.get("has_trend"),
            "trend_direction": decision.get("trend_direction"),
            "dependence_detected": decision.get("has_dependence"),
            "model": reliability_result.get("model"),
            "process_variant": reliability_result.get("process_variant"),
            "distribution": reliability_result.get("distribution"),
            "param_string": format_param_string(reliability_result),
            "aic": goodness.get("aic"),
            "ks_p": goodness.get("ks_p"),
            "chi2_p": goodness.get("chi2_p"),
            "cvm_p": goodness.get("cvm_p"),
            "goodness_accepted": goodness.get("accepted"),
            "beta": parameters.get("beta", optimization_row.get("beta")),
            "eta_h": parameters.get("eta", optimization_row.get("eta_h")),
            "gamma_h": parameters.get("gamma", optimization_row.get("gamma_h")),
            "mtbf_h": indicators.get("mtbf_h"),
            "mttr_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
            "maintenance_type": optimization_row.get("maintenance_type", maintenance_row.get("maintenance_type")),
            "T_recommended_h": optimization_row.get("T_recommended_h"),
            "T_R_h": optimization_row.get("T_R_h"),
            "T_cost_h": optimization_row.get("T_cost_h"),
            "R(T_cost)": optimization_row.get("R(T_cost)"),
            "C_min_per_h": optimization_row.get("C_min_per_h"),
            "next_due_date": maintenance_row.get("next_due_date"),
            "days_left": maintenance_row.get("days_left"),
        }
        global_rows.append(current_row)

    summary_dataframe = pd.DataFrame(global_rows)
    if summary_dataframe.empty:
        st.error("Aucun équipement exploitable n’a pu être analysé.")
        st.stop()

    final_decisions = summary_dataframe.apply(lambda row: compute_final_decision_row(row), axis=1)
    summary_dataframe[["decision_finale", "motif_decision", "priority_score", "priorite"]] = pd.DataFrame(
        final_decisions.tolist(),
        index=summary_dataframe.index,
    )

    summary_dataframe = summary_dataframe.sort_values(
        ["priority_score", "equipment_code"],
        ascending=[False, True],
    ).reset_index(drop=True)

    for equipment_code in results_by_equipment.keys():
        row_df = summary_dataframe[summary_dataframe["equipment_code"].astype(str) == str(equipment_code)]
        if row_df.empty:
            continue
        selected_row = row_df.iloc[0].to_dict()
        selected_result = results_by_equipment[equipment_code]


# =========================================================
# Construction des tableaux style mémoire
# =========================================================
trend_global_dataframe = build_trend_global_table(results_by_equipment, alpha_value)
dependence_global_dataframe = build_dependence_global_table(results_by_equipment, alpha_value)
synthesis_global_dataframe = build_synthesis_table(summary_dataframe, failures_dataframe, current_pm_hours)
final_decision_dataframe = build_final_decision_table(summary_dataframe)

global_tables = {
    "Tableau_tendance_global": trend_global_dataframe,
    "Tableau_dependance_global": dependence_global_dataframe,
    "Tableau_synthese_fiabiliste": synthesis_global_dataframe,
    "Tableau_decision_finale": final_decision_dataframe,
}


# =========================================================
# KPIs
# =========================================================
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Équipements analysés", len(summary_dataframe))
with metric_col_2:
    st.metric("Priorité critique", int((summary_dataframe["priorite"].astype(str) == "Critique").sum()))
with metric_col_3:
    st.metric("Tâches dues", len(maintenance_due_dataframe))
with metric_col_4:
    st.metric("Processus NHPP", int((summary_dataframe["model"].astype(str).str.upper() == "NHPP").sum()))


# =========================================================
# Onglets
# =========================================================
page_tab_1, page_tab_2, page_tab_3 = st.tabs(["Tableaux globaux", "Détail par équipement", "Exports"])

with page_tab_1:
    st.subheader("Tableaux globaux")

    st.markdown("### Tableau 1 : Calcul numérique du test de tendance")
    st.dataframe(trend_global_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 2 : Calcul numérique du test de dépendance")
    st.dataframe(dependence_global_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 3 : Loi, estimation des paramètres et temps de remplacement")
    st.dataframe(synthesis_global_dataframe, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 4 : Décision finale hiérarchisée")
    st.dataframe(final_decision_dataframe, use_container_width=True, hide_index=True)

with page_tab_2:
    selected_equipment_code = st.selectbox(
        "Choisir un équipement",
        options=summary_dataframe["equipment_code"].tolist(),
    )

    selected_result = results_by_equipment[selected_equipment_code]
    selected_row = summary_dataframe[summary_dataframe["equipment_code"] == selected_equipment_code].iloc[0].to_dict()
    selected_tables = detail_tables_by_equipment.get(selected_equipment_code, {})

    st.markdown("### Tableau 1 — Tests de tendance")
    trend_table = selected_tables.get("tableau_tendance", pd.DataFrame())
    st.dataframe(trend_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 2 — Tests de dépendance")
    dependence_table = selected_tables.get("tableau_dependance", pd.DataFrame())
    st.dataframe(dependence_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 3 — Choix du processus")
    process_table = selected_tables.get("process_choice", pd.DataFrame())
    if process_table.empty:
        st.info("Aucun tableau de choix du processus disponible.")
    else:
        st.dataframe(process_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 4 — Ajustement et lois candidates")
    fit_table = selected_tables.get("fit_candidates", pd.DataFrame())
    if fit_table.empty:
        st.info("Aucun tableau d’ajustement disponible.")
    else:
        st.dataframe(fit_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 5 — Paramètres fiabilistes")
    param_table = selected_tables.get("tableau_parametres", pd.DataFrame())
    st.dataframe(param_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 6 — Optimisation et maintenance")
    optimization_table = selected_tables.get("tableau_optimisation", pd.DataFrame())
    st.dataframe(optimization_table, use_container_width=True, hide_index=True)

    st.markdown("### Tableau 7 — Décision finale")
    decision_table = selected_tables.get("tableau_decision_finale", pd.DataFrame())
    st.dataframe(decision_table, use_container_width=True, hide_index=True)

    st.success(f"Décision finale — {selected_row.get('decision_finale', '—')}")
    st.info(selected_row.get("motif_decision", "Aucun motif disponible."))

with page_tab_3:
    excel_bytes = build_excel_bytes(global_tables, detail_tables_by_equipment)
    st.download_button(
        "Télécharger le pack Excel global",
        data=excel_bytes,
        file_name="resultat_analyse_optimisation_maintenance.xlsx",
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
