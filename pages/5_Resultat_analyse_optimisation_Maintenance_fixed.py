from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from io import BytesIO
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.security.auth import require_login
from core.reliability.organigram import analyze_ttf_pipeline
from core.ui import render_shell, render_page_header

try:
    from core.datahub import (
        get_current_failures_df,
        get_current_project_data,
    )
except Exception:
    get_current_failures_df = None
    get_current_project_data = None

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
    "Traçabilité complète : fiabilité, thermique, optimisation, maintenance et décision finale.",
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


def normalize_thermal_timeseries(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    normalized_dataframe = dataframe.copy()
    normalized_dataframe.columns = [str(column).strip() for column in normalized_dataframe.columns]

    aliases = {
        "ambient_temp_c": "temp_amb_C",
        "temp_ambiante_c": "temp_amb_C",
        "temperature_ambiante": "temp_amb_C",
        "fan_status": "etat_ventilateurs",
        "fans_status": "etat_ventilateurs",
        "ventilateurs": "etat_ventilateurs",
        "load_pct": "charge_pct",
    }
    lowercase_columns = {column.lower().strip(): column for column in normalized_dataframe.columns}
    rename_map = {}
    for alias_name, target_name in aliases.items():
        if alias_name in lowercase_columns and target_name not in normalized_dataframe.columns:
            rename_map[lowercase_columns[alias_name]] = target_name

    normalized_dataframe = normalized_dataframe.rename(columns=rename_map)

    if "timestamp" in normalized_dataframe.columns:
        normalized_dataframe["timestamp"] = pd.to_datetime(normalized_dataframe["timestamp"], errors="coerce")
        normalized_dataframe = normalized_dataframe.dropna(subset=["timestamp"])

    if "K" not in normalized_dataframe.columns:
        if "load_factor" in normalized_dataframe.columns:
            normalized_dataframe["K"] = pd.to_numeric(normalized_dataframe["load_factor"], errors="coerce")
        elif "charge_pct" in normalized_dataframe.columns:
            normalized_dataframe["K"] = pd.to_numeric(normalized_dataframe["charge_pct"], errors="coerce") / 100.0
        elif "load_mva" in normalized_dataframe.columns:
            normalized_dataframe["K"] = pd.to_numeric(normalized_dataframe["load_mva"], errors="coerce") / 100.0

    if "etat_ventilateurs" not in normalized_dataframe.columns:
        normalized_dataframe["etat_ventilateurs"] = 0

    for column_name in [
        "temp_amb_C",
        "K",
        "etat_ventilateurs",
        "top_oil_temp_c",
        "hotspot_temp_c",
        "charge_pct",
    ]:
        if column_name in normalized_dataframe.columns:
            normalized_dataframe[column_name] = pd.to_numeric(normalized_dataframe[column_name], errors="coerce")

    return normalized_dataframe.reset_index(drop=True)


def normalize_thermal_parameters(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    normalized_dataframe = dataframe.copy()
    normalized_dataframe.columns = [str(column).strip() for column in normalized_dataframe.columns]

    aliases = {
        "delta_theta_to_r": "delta_to_r",
        "delta_theta_h_r": "delta_h_r",
        "tau_to_hours": "tau_to_hours",
        "tau_h_hours": "tau_h_hours",
        "normal_life_hours": "normal_insulation_life_h",
        "rated_power_mva": "sn_mva",
    }
    lowercase_columns = {column.lower().strip(): column for column in normalized_dataframe.columns}
    rename_map = {}
    for alias_name, target_name in aliases.items():
        if alias_name in lowercase_columns and target_name not in normalized_dataframe.columns:
            rename_map[lowercase_columns[alias_name]] = target_name

    normalized_dataframe = normalized_dataframe.rename(columns=rename_map)

    for column_name in normalized_dataframe.columns:
        if column_name != "asset_id":
            normalized_dataframe[column_name] = pd.to_numeric(
                normalized_dataframe[column_name],
                errors="ignore",
            )

    if "tau_to_hours" in normalized_dataframe.columns and "tau_to_min" not in normalized_dataframe.columns:
        normalized_dataframe["tau_to_min"] = pd.to_numeric(normalized_dataframe["tau_to_hours"], errors="coerce") * 60.0

    if "tau_h_hours" in normalized_dataframe.columns and "tau_w_min" not in normalized_dataframe.columns:
        normalized_dataframe["tau_w_min"] = pd.to_numeric(normalized_dataframe["tau_h_hours"], errors="coerce") * 60.0

    return normalized_dataframe


def load_project_context(uploaded_xlsx=None) -> Dict[str, pd.DataFrame]:
    if uploaded_xlsx is not None:
        try:
            raw_bytes = uploaded_xlsx.read()
            excel_file = pd.ExcelFile(io.BytesIO(raw_bytes))
            sheets = {
                sheet_name: pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name)
                for sheet_name in excel_file.sheet_names
            }
        except Exception:
            sheets = {}
    elif callable(get_current_project_data):
        try:
            sheets = get_current_project_data() or {}
        except Exception:
            sheets = {}
    else:
        sheets = {}

    normalized_sheets: Dict[str, pd.DataFrame] = {}
    for key, value in sheets.items():
        if isinstance(value, pd.DataFrame):
            normalized_sheets[str(key).strip()] = value.copy()

    if "thermal_timeseries" in normalized_sheets:
        normalized_sheets["thermal_timeseries"] = normalize_thermal_timeseries(normalized_sheets["thermal_timeseries"])

    if "thermal_params" in normalized_sheets:
        normalized_sheets["thermal_params"] = normalize_thermal_parameters(normalized_sheets["thermal_params"])

    return normalized_sheets


def extract_thermal_for_equipment(
    sheets: Dict[str, pd.DataFrame],
    equipment_code: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if not sheets:
        return None, None

    thermal_dataframe = None
    thermal_configuration = None

    timeseries_dataframe = sheets.get("thermal_timeseries")
    if isinstance(timeseries_dataframe, pd.DataFrame) and not timeseries_dataframe.empty:
        temp_dataframe = timeseries_dataframe.copy()
        asset_column = "asset_id" if "asset_id" in temp_dataframe.columns else None
        if asset_column:
            temp_dataframe = temp_dataframe[temp_dataframe[asset_column].astype(str) == str(equipment_code)]
        if not temp_dataframe.empty:
            thermal_dataframe = temp_dataframe.reset_index(drop=True)

    thermal_parameters_dataframe = sheets.get("thermal_params")
    if isinstance(thermal_parameters_dataframe, pd.DataFrame) and not thermal_parameters_dataframe.empty:
        temp_parameters_dataframe = thermal_parameters_dataframe.copy()
        asset_column = "asset_id" if "asset_id" in temp_parameters_dataframe.columns else None
        if asset_column:
            temp_parameters_dataframe = temp_parameters_dataframe[temp_parameters_dataframe[asset_column].astype(str) == str(equipment_code)]
        if not temp_parameters_dataframe.empty:
            first_row = temp_parameters_dataframe.iloc[0].to_dict()
            thermal_configuration = {
                "sn_mva": safe_float(first_row.get("sn_mva"), 100.0) or 100.0,
                "R": safe_float(first_row.get("R"), 5.0) or 5.0,
                "delta_to_r": safe_float(first_row.get("delta_to_r"), 55.0) or 55.0,
                "delta_h_r": safe_float(first_row.get("delta_h_r"), 30.0) or 30.0,
                "tau_to_min": safe_float(first_row.get("tau_to_min"), 180.0) or 180.0,
                "tau_w_min": safe_float(first_row.get("tau_w_min"), 10.0) or 10.0,
                "n_exp": safe_float(first_row.get("n_exp"), 0.8) or 0.8,
                "m_exp": safe_float(first_row.get("m_exp"), 0.8) or 0.8,
                "forced_tau_to_factor": safe_float(first_row.get("forced_tau_to_factor"), 0.75) or 0.75,
                "forced_delta_to_factor": safe_float(first_row.get("forced_delta_to_factor"), 0.92) or 0.92,
                "forced_delta_h_factor": safe_float(first_row.get("forced_delta_h_factor"), 0.92) or 0.92,
                "normal_insulation_life_h": safe_float(first_row.get("normal_insulation_life_h"), 180000.0) or 180000.0,
            }

    return thermal_dataframe, thermal_configuration


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
    if "process_model" in output_dataframe.columns and "model" not in output_dataframe.columns:
        output_dataframe["model"] = output_dataframe["process_model"]
    if "beta_pipe" in output_dataframe.columns and "beta" not in output_dataframe.columns:
        output_dataframe["beta"] = output_dataframe["beta_pipe"]
    if "eta_pipe_h" in output_dataframe.columns and "eta_h" not in output_dataframe.columns:
        output_dataframe["eta_h"] = output_dataframe["eta_pipe_h"]
    if "gamma_pipe_h" in output_dataframe.columns and "gamma_h" not in output_dataframe.columns:
        output_dataframe["gamma_h"] = output_dataframe["gamma_pipe_h"]

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


def compute_thermal_status(
    maximum_hotspot_temperature: Any,
    maximum_ageing_acceleration_factor: Any,
    loss_of_life_percent: Any,
) -> str:
    hotspot_temperature = safe_float(maximum_hotspot_temperature, None)
    ageing_acceleration_factor = safe_float(maximum_ageing_acceleration_factor, None)
    loss_of_life_value = safe_float(loss_of_life_percent, None)

    if (
        (hotspot_temperature is not None and hotspot_temperature >= 130)
        or (ageing_acceleration_factor is not None and ageing_acceleration_factor >= 2.0)
        or (loss_of_life_value is not None and loss_of_life_value >= 1.0)
    ):
        return "Critique"

    if (
        (hotspot_temperature is not None and hotspot_temperature >= 110)
        or (ageing_acceleration_factor is not None and ageing_acceleration_factor >= 1.5)
        or (loss_of_life_value is not None and loss_of_life_value >= 0.3)
    ):
        return "Alerte"

    if hotspot_temperature is None and ageing_acceleration_factor is None and loss_of_life_value is None:
        return "Non disponible"

    return "Normal"


def process_score(process_name: str) -> int:
    normalized = (process_name or "").upper()
    if "NHPP" in normalized:
        return 3
    if "BPP" in normalized or "HAWKES" in normalized:
        return 2
    return 1


def build_priority_contributors(row: pd.Series | Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(row, pd.Series):
        row = row.to_dict()

    contributors: List[Dict[str, Any]] = []

    process_name = str(row.get("model", "RP"))
    process_points = process_score(process_name)
    contributors.append(
        {
            "Paramètre pris en compte": "Processus retenu",
            "Valeur observée": process_name,
            "Intervention dans la décision": (
                "NHPP augmente davantage la priorité, BPP augmente modérément, RP reste la situation la plus stable."
            ),
            "Points": process_points,
        }
    )

    beta_value = safe_float(row.get("beta"), None)
    if beta_value is None:
        beta_points = 0
        beta_comment = "Le paramètre bêta n'était pas disponible, donc il n'a pas été utilisé directement pour le score."
    elif beta_value > 1.2:
        beta_points = 3
        beta_comment = "Le paramètre bêta traduit une phase d'usure marquée, ce qui augmente fortement la priorité."
    elif beta_value >= 1.0:
        beta_points = 2
        beta_comment = "Le paramètre bêta traduit un comportement déjà orienté vers l'usure ou le régime aléatoire avancé."
    elif beta_value < 0.8:
        beta_points = 1
        beta_comment = "Le paramètre bêta suggère surtout des défauts précoces, ce qui pèse faiblement sur la priorité."
    else:
        beta_points = 0
        beta_comment = "Le paramètre bêta n'a pas ajouté de point spécifique dans cette zone intermédiaire."
    contributors.append(
        {
            "Paramètre pris en compte": "Paramètre bêta",
            "Valeur observée": format_number(beta_value, 3),
            "Intervention dans la décision": beta_comment,
            "Points": beta_points,
        }
    )

    thermal_status = str(row.get("thermal_status", "Non disponible"))
    if thermal_status == "Critique":
        thermal_points = 4
        thermal_comment = "Le statut thermique critique accélère fortement la maintenance."
    elif thermal_status == "Alerte":
        thermal_points = 2
        thermal_comment = "Le statut thermique en alerte renforce la prudence et rapproche l'intervention."
    else:
        thermal_points = 0
        thermal_comment = "Le statut thermique n'a pas majoré le score."
    contributors.append(
        {
            "Paramètre pris en compte": "Statut thermique",
            "Valeur observée": thermal_status,
            "Intervention dans la décision": thermal_comment,
            "Points": thermal_points,
        }
    )

    days_left = safe_float(row.get("days_left"), None)
    if days_left is None:
        due_points = 0
        due_comment = "Aucune échéance exploitable n'était disponible."
    elif days_left <= 7:
        due_points = 3
        due_comment = "L'échéance est très proche, ce qui augmente fortement la priorité."
    elif days_left <= 30:
        due_points = 2
        due_comment = "L'échéance est proche, ce qui augmente modérément la priorité."
    elif days_left <= 90:
        due_points = 1
        due_comment = "L'échéance reste à surveiller, avec un impact faible sur le score."
    else:
        due_points = 0
        due_comment = "L'échéance n'est pas suffisamment proche pour majorer le score."
    contributors.append(
        {
            "Paramètre pris en compte": "Nombre de jours restants",
            "Valeur observée": "Non disponible" if days_left is None else f"{int(days_left)} jours",
            "Intervention dans la décision": due_comment,
            "Points": due_points,
        }
    )

    contributors.append(
        {
            "Paramètre pris en compte": "Température maximale du point chaud",
            "Valeur observée": format_number(row.get("theta_hs_max"), 2),
            "Intervention dans la décision": "Cet indicateur thermique aide à qualifier le statut thermique global.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Facteur maximal d'accélération du vieillissement",
            "Valeur observée": format_number(row.get("faa_max"), 3),
            "Intervention dans la décision": "Il augmente la sévérité thermique lorsque la chaleur accélère le vieillissement.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Perte de vie estimée",
            "Valeur observée": format_number(row.get("loss_of_life_pct"), 3) + " %",
            "Intervention dans la décision": "Elle participe à la qualification thermique globale et à la prudence de maintenance.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Temps moyen entre défaillances",
            "Valeur observée": format_number(row.get("mtbf_h"), 1) + " h",
            "Intervention dans la décision": "Il renseigne sur l'espacement moyen des pannes et aide à interpréter le niveau de risque.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Temps moyen de réparation",
            "Valeur observée": format_number(row.get("mttr_h"), 1) + " h",
            "Intervention dans la décision": "Il renseigne sur la rapidité de remise en service et éclaire la disponibilité.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Intervalle recommandé",
            "Valeur observée": format_number(row.get("T_recommended_h"), 1) + " h",
            "Intervention dans la décision": "Il n'ajoute pas de point directement, mais structure la date d'intervention proposée.",
            "Points": 0,
        }
    )
    contributors.append(
        {
            "Paramètre pris en compte": "Type de maintenance recommandé",
            "Valeur observée": str(row.get("maintenance_type", "—")),
            "Intervention dans la décision": "C'est la traduction opérationnelle finale des résultats de fiabilité, de thermique et d'optimisation.",
            "Points": 0,
        }
    )

    return contributors


def compute_final_decision_row(row: pd.Series):
    contributors = build_priority_contributors(row)
    score = int(sum(int(item.get("Points", 0) or 0) for item in contributors))

    process_name = str(row.get("model", "RP"))
    maintenance_type = str(row.get("maintenance_type", "maintenance ciblée"))
    thermal_status = str(row.get("thermal_status", "Non disponible"))

    if score >= 10:
        decision = "Intervention prioritaire"
        base_reason = (
            f"Le niveau de risque global est élevé. Le processus {process_name}, le niveau thermique "
            f"et l'échéance imposent une action rapide. Type d'action retenu : {maintenance_type}."
        )
    elif score >= 7:
        decision = "Préventif renforcé"
        base_reason = (
            f"Le risque reste important. Une surveillance renforcée et une planification rapide d'une maintenance "
            f"de type {maintenance_type} sont recommandées."
        )
    elif score >= 4:
        decision = "Surveillance active"
        base_reason = (
            f"La situation est intermédiaire. Il faut suivre de près l'équipement, conserver le plan calculé "
            f"et appliquer le type {maintenance_type} au bon moment."
        )
    else:
        decision = "Suivi nominal"
        base_reason = (
            f"Aucun signal critique immédiat ne domine. Le plan standard peut être appliqué avec le type "
            f"de maintenance {maintenance_type}."
        )

    strong_drivers = [
        item["Paramètre pris en compte"]
        for item in contributors
        if int(item.get("Points", 0) or 0) >= 2
    ]
    if strong_drivers:
        reason = base_reason + " Paramètres les plus influents : " + ", ".join(strong_drivers) + "."
    else:
        reason = base_reason

    if thermal_status == "Critique":
        reason += " La thermique a clairement aggravé la décision finale."
    elif thermal_status == "Alerte":
        reason += " La thermique a contribué à renforcer la prudence."
    elif thermal_status == "Normal":
        reason += " La thermique n'a pas majoré le score."

    return decision, reason, score


DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "n_ttf": "Nombre de temps entre défaillances",
    "trend_detected": "Tendance détectée",
    "trend_direction": "Sens de la tendance",
    "dependence_detected": "Dépendance détectée",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi de probabilité retenue",
    "aic": "Critère d'information d'Akaike",
    "ks_p": "Valeur p du test de Kolmogorov-Smirnov",
    "chi2_p": "Valeur p du test du chi carré",
    "cvm_p": "Valeur p du test de Cramér-von Mises",
    "goodness_accepted": "Ajustement accepté",
    "beta": "Paramètre bêta",
    "eta_h": "Paramètre êta (heures)",
    "gamma_h": "Paramètre gamma (heures)",
    "mtbf_h": "Temps moyen entre défaillances (heures)",
    "mttr_h": "Temps moyen de réparation (heures)",
    "availability_pct": "Disponibilité intrinsèque (%)",
    "theta_hs_max": "Température maximale du point chaud (°C)",
    "faa_max": "Facteur maximal d’accélération du vieillissement",
    "loss_of_life_pct": "Perte de vie (%)",
    "thermal_status": "Statut thermique",
    "maintenance_type": "Type de maintenance recommandé",
    "T_recommended_h": "Intervalle recommandé (heures)",
    "T_R_h": "Intervalle issu du critère de fiabilité (heures)",
    "T_cost_h": "Intervalle issu du critère économique (heures)",
    "R(T_cost)": "Fiabilité au niveau de l’intervalle économique",
    "C_min_per_h": "Coût minimal par heure",
    "next_due_date": "Prochaine date d’échéance",
    "days_left": "Nombre de jours restants",
    "decision_finale": "Décision finale",
    "motif_decision": "Motif de la décision",
    "priority_score": "Score de priorité",
    "priorite": "Niveau de priorité",
    "Paramètre pris en compte": "Paramètre pris en compte",
    "Valeur observée": "Valeur observée",
    "Intervention dans la décision": "Intervention dans la décision",
    "Points": "Points",
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


def build_trend_trace_table(result: Dict[str, Any], alpha_value: float) -> pd.DataFrame:
    tests = (result.get("reliability", {}) or {}).get("tests", {}) or {}
    mann_kendall_result = tests.get("trend_mk", {}) or {}
    laplace_result = tests.get("trend_laplace", {}) or {}
    military_indicator = tests.get("trend_mil_hdbk_189", {}) or {}
    combined_trend = tests.get("trend_combined", {}) or {}

    rows = [
        {
            "Élément observé": "Test de Mann-Kendall",
            "Valeur": f"statistique = {format_number(mann_kendall_result.get('z'), 3)}, p-valeur = {format_number(mann_kendall_result.get('p'), 4)}",
            "Comment lire cette valeur": f"Si la p-valeur est inférieure à {alpha_value:.3f}, la tendance est considérée significative.",
            "Conclusion": "Tendance détectée" if mann_kendall_result.get("has_trend") else "Pas de tendance significative",
        },
        {
            "Élément observé": "Test de Laplace",
            "Valeur": f"statistique = {format_number(laplace_result.get('u'), 3)}, p-valeur = {format_number(laplace_result.get('p'), 4)}",
            "Comment lire cette valeur": f"Si la p-valeur est inférieure à {alpha_value:.3f}, la tendance est considérée significative.",
            "Conclusion": "Tendance détectée" if laplace_result.get("has_trend") else "Pas de tendance significative",
        },
        {
            "Élément observé": "Indicateur graphique inspiré de MIL-HDBK-189",
            "Valeur": f"pente log-log = {format_number(military_indicator.get('beta_graph'), 3)}",
            "Comment lire cette valeur": "Une pente supérieure à 1 suggère une dérive croissante, inférieure à 1 une dérive décroissante.",
            "Conclusion": str(military_indicator.get("interpreted_trend", "none")),
        },
        {
            "Élément observé": "Décision combinée sur la tendance",
            "Valeur": f"sens = {combined_trend.get('direction', 'none')}, confiance = {combined_trend.get('confidence', 'none')}",
            "Comment lire cette valeur": "Le pipeline combine les tests pour décider si une tendance globale existe réellement.",
            "Conclusion": "Tendance retenue" if combined_trend.get("has_trend") else "Aucune tendance retenue",
        },
    ]
    return pd.DataFrame(rows)


def build_dependence_trace_table(result: Dict[str, Any], alpha_value: float) -> pd.DataFrame:
    dependence_result = ((result.get("reliability", {}) or {}).get("tests", {}) or {}).get("dependence", {}) or {}

    rows = [
        {
            "Élément observé": "Corrélation de Pearson",
            "Valeur": f"coefficient = {format_number(dependence_result.get('pearson_r'), 3)}, p-valeur = {format_number(dependence_result.get('pearson_p'), 4)}",
            "Comment lire cette valeur": f"Si la p-valeur est inférieure à {alpha_value:.3f}, une dépendance linéaire est plausible.",
            "Conclusion": "Appui vers une dépendance" if safe_float(dependence_result.get("pearson_p"), 1.0) < alpha_value else "Pas de preuve forte de dépendance linéaire",
        },
        {
            "Élément observé": "Corrélation de Spearman",
            "Valeur": f"coefficient = {format_number(dependence_result.get('spearman_r'), 3)}, p-valeur = {format_number(dependence_result.get('spearman_p'), 4)}",
            "Comment lire cette valeur": f"Si la p-valeur est inférieure à {alpha_value:.3f}, une dépendance monotone est plausible.",
            "Conclusion": "Appui vers une dépendance" if dependence_result.get("has_dep") else "Pas de dépendance significative retenue",
        },
        {
            "Élément observé": "Décision finale sur la dépendance",
            "Valeur": f"force estimée = {dependence_result.get('strength', 'non précisée')}",
            "Comment lire cette valeur": "Cette information aide à savoir si les défaillances successives s’influencent.",
            "Conclusion": "Dépendance retenue" if dependence_result.get("has_dep") else "Indépendance privilégiée",
        },
    ]
    return pd.DataFrame(rows)


def build_model_trace_table(result: Dict[str, Any]) -> pd.DataFrame:
    reliability_result = result.get("reliability", {}) or {}
    decision = reliability_result.get("decision", {}) or {}

    rows = [
        {
            "Élément observé": "Processus retenu",
            "Valeur": reliability_result.get("model", "—"),
            "Comment lire cette valeur": "Il s’agit du type de comportement global retenu pour les défaillances.",
            "Conclusion": decision.get("reason", "—"),
        },
        {
            "Élément observé": "Variant du processus",
            "Valeur": reliability_result.get("process_variant", "—"),
            "Comment lire cette valeur": "Le variant précise si l’on reste dans un renouvellement simple ou dans un cas particulier.",
            "Conclusion": decision.get("entity_assumption", "—"),
        },
        {
            "Élément observé": "Loi de probabilité retenue",
            "Valeur": reliability_result.get("distribution", "—"),
            "Comment lire cette valeur": "La loi retenue décrit mathématiquement la durée entre deux défaillances.",
            "Conclusion": "Loi acceptée" if decision.get("law_accepted") is True else "Loi non validée strictement" if decision.get("law_accepted") is False else "Validation non disponible",
        },
    ]
    return pd.DataFrame(rows)


def build_goodness_of_fit_trace_table(result: Dict[str, Any]) -> pd.DataFrame:
    reliability_result = result.get("reliability", {}) or {}
    goodness = reliability_result.get("goodness", {}) or {}
    decision = reliability_result.get("decision", {}) or {}

    rows = [
        {
            "Élément observé": "Critère d'information d'Akaike",
            "Valeur": format_number(goodness.get("aic"), 3),
            "Comment lire cette valeur": "Plus cette valeur est faible, meilleur est le compromis entre qualité d'ajustement et complexité.",
            "Conclusion": "Utilisé dans le choix de la loi retenue",
        },
        {
            "Élément observé": "Test de Kolmogorov-Smirnov",
            "Valeur": format_number(goodness.get("ks_p"), 4),
            "Comment lire cette valeur": "Une valeur p élevée traduit un ajustement plus acceptable.",
            "Conclusion": "Compatible avec l'ajustement" if safe_float(goodness.get("ks_p"), 0.0) >= 0.05 else "Ajustement à surveiller",
        },
        {
            "Élément observé": "Test du chi carré",
            "Valeur": format_number(goodness.get("chi2_p"), 4),
            "Comment lire cette valeur": "Une valeur p élevée traduit un ajustement plus acceptable.",
            "Conclusion": "Compatible avec l'ajustement" if safe_float(goodness.get("chi2_p"), 0.0) >= 0.05 else "Ajustement à surveiller",
        },
        {
            "Élément observé": "Test de Cramér-von Mises",
            "Valeur": format_number(goodness.get("cvm_p"), 4),
            "Comment lire cette valeur": "Une valeur p élevée traduit un ajustement plus acceptable.",
            "Conclusion": "Compatible avec l'ajustement" if safe_float(goodness.get("cvm_p"), 0.0) >= 0.05 else "Ajustement à surveiller",
        },
        {
            "Élément observé": "Décision globale sur l'ajustement",
            "Valeur": str(goodness.get("accepted", "Non disponible")),
            "Comment lire cette valeur": "Cette décision résume si l'ajustement retenu est jugé acceptable par le pipeline.",
            "Conclusion": decision.get("law_selected", "—"),
        },
    ]
    return pd.DataFrame(rows)


def build_parameter_trace_table(row: Dict[str, Any]) -> pd.DataFrame:
    beta_value = safe_float(row.get("beta"), None)
    if beta_value is None:
        beta_explanation = "Le paramètre bêta n’est pas disponible."
    elif beta_value > 1:
        beta_explanation = "Le paramètre bêta supérieur à 1 évoque une phase d’usure."
    elif beta_value >= 0.9:
        beta_explanation = "Le paramètre bêta proche de 1 évoque un comportement aléatoire."
    else:
        beta_explanation = "Le paramètre bêta inférieur à 1 évoque des défauts précoces."

    rows = [
        {
            "Variable": "Paramètre bêta",
            "Valeur": format_number(row.get("beta"), 3),
            "Explication pour lecture": beta_explanation,
        },
        {
            "Variable": "Paramètre êta (heures)",
            "Valeur": format_number(row.get("eta_h"), 1),
            "Explication pour lecture": "Le paramètre êta représente une durée de vie caractéristique.",
        },
        {
            "Variable": "Paramètre gamma (heures)",
            "Valeur": format_number(row.get("gamma_h"), 1),
            "Explication pour lecture": "Le paramètre gamma traduit un éventuel décalage du modèle.",
        },
        {
            "Variable": "Temps moyen entre défaillances (heures)",
            "Valeur": format_number(row.get("mtbf_h"), 1),
            "Explication pour lecture": "Plus cette valeur est grande, plus les défaillances sont espacées.",
        },
        {
            "Variable": "Temps moyen de réparation (heures)",
            "Valeur": format_number(row.get("mttr_h"), 1),
            "Explication pour lecture": "Plus cette valeur est faible, plus la remise en état est rapide.",
        },
        {
            "Variable": "Disponibilité intrinsèque (%)",
            "Valeur": format_number(row.get("availability_pct"), 2),
            "Explication pour lecture": "Elle mesure la part du temps où l’équipement reste disponible.",
        },
        {
            "Variable": "Température maximale du point chaud (°C)",
            "Valeur": format_number(row.get("theta_hs_max"), 2),
            "Explication pour lecture": "C’est la température la plus sévère estimée dans l’équipement.",
        },
        {
            "Variable": "Facteur maximal d’accélération du vieillissement",
            "Valeur": format_number(row.get("faa_max"), 3),
            "Explication pour lecture": "Plus cette valeur est élevée, plus la chaleur accélère le vieillissement.",
        },
        {
            "Variable": "Perte de vie (%)",
            "Valeur": format_number(row.get("loss_of_life_pct"), 3),
            "Explication pour lecture": "Elle représente la part estimée de la durée de vie déjà consommée.",
        },
    ]
    return pd.DataFrame(rows)


def build_optimization_trace_table(row: Dict[str, Any]) -> pd.DataFrame:
    recommended_interval = safe_float(row.get("T_recommended_h"), None)
    reliability_interval = safe_float(row.get("T_R_h"), None)
    economic_interval = safe_float(row.get("T_cost_h"), None)

    if recommended_interval is not None and reliability_interval is not None and abs(recommended_interval - reliability_interval) < 1e-6:
        lecture = "L’intervalle recommandé correspond surtout au critère de fiabilité."
    elif recommended_interval is not None and economic_interval is not None and abs(recommended_interval - economic_interval) < 1e-6:
        lecture = "L’intervalle recommandé correspond surtout au critère économique."
    else:
        lecture = "L’intervalle recommandé correspond à un compromis entre coût et fiabilité."

    rows = [
        {
            "Variable": "Intervalle issu du critère de fiabilité (heures)",
            "Valeur": format_number(row.get("T_R_h"), 1),
            "Explication pour lecture": "C’est l’intervalle associé à l’objectif de fiabilité fixé.",
        },
        {
            "Variable": "Intervalle issu du critère économique (heures)",
            "Valeur": format_number(row.get("T_cost_h"), 1),
            "Explication pour lecture": "C’est l’intervalle qui réduit le coût moyen par heure.",
        },
        {
            "Variable": "Intervalle recommandé (heures)",
            "Valeur": format_number(row.get("T_recommended_h"), 1),
            "Explication pour lecture": lecture,
        },
        {
            "Variable": "Fiabilité au niveau de l’intervalle économique",
            "Valeur": format_number(row.get("R(T_cost)"), 3),
            "Explication pour lecture": "Cette valeur mesure le niveau de fiabilité attendu à l’intervalle économique.",
        },
        {
            "Variable": "Coût minimal par heure",
            "Valeur": format_number(row.get("C_min_per_h"), 4),
            "Explication pour lecture": "C’est le coût moyen minimal estimé après optimisation.",
        },
        {
            "Variable": "Type de maintenance recommandé",
            "Valeur": row.get("maintenance_type", "—"),
            "Explication pour lecture": "Il s’agit de l’action recommandée sur le terrain.",
        },
        {
            "Variable": "Prochaine date d’échéance",
            "Valeur": row.get("next_due_date", "—"),
            "Explication pour lecture": f"Il reste environ {row.get('days_left', '—')} jours avant cette échéance.",
        },
    ]
    return pd.DataFrame(rows)


def build_decision_trace_table(row: Dict[str, Any]) -> pd.DataFrame:
    contributors = build_priority_contributors(row)
    decision_rows = pd.DataFrame(contributors)

    score_row = pd.DataFrame(
        [
            {
                "Paramètre pris en compte": "Score final de priorité",
                "Valeur observée": row.get("priority_score", "—"),
                "Intervention dans la décision": f"Niveau de priorité final : {row.get('priorite', '—')}. Décision retenue : {row.get('decision_finale', '—')}. Motif : {row.get('motif_decision', '—')}",
                "Points": row.get("priority_score", "—"),
            }
        ]
    )

    return pd.concat([decision_rows, score_row], ignore_index=True)


def build_excel_bytes(
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_equipment: Dict[str, Dict[str, pd.DataFrame]],
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, dataframe in global_tables.items():
            if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                rename_columns_for_display(dataframe).to_excel(writer, sheet_name=name[:31], index=False)

        for equipment_code, tables in detail_tables_by_equipment.items():
            for table_name, dataframe in tables.items():
                if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
                    try:
                        rename_columns_for_display(dataframe).to_excel(
                            writer,
                            sheet_name=f"{equipment_code}_{table_name}"[:31],
                            index=False,
                        )
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
    uploaded_project_xlsx = st.file_uploader(
        "Fichier Excel du projet thermique (optionnel)",
        type=["xlsx"],
        key="global_uploaded_project",
    )
with control_col_3:
    alpha_value = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
with control_col_4:
    due_window_days = st.slider("Fenêtre de maintenance (jours)", 7, 365, 30, 1)

reference_start_date = st.date_input("Date de référence", value=date.today())

with st.expander("Comprendre clairement les variables affichées sur cette page", expanded=False):
    st.markdown(
        """
**Tendance détectée** : indique si les défaillances augmentent ou diminuent avec le temps.

**Dépendance détectée** : indique si une défaillance semble liée aux précédentes.

**Processus retenu** : décrit le comportement global du système de défaillance.

**Loi de probabilité retenue** : loi mathématique choisie pour représenter les temps entre défaillances.

**Critères d’ajustement** : ils servent à vérifier si la loi retenue représente correctement les données.

**Paramètre bêta** : forme du vieillissement.  
- inférieur à 1 : défauts précoces  
- proche de 1 : comportement aléatoire  
- supérieur à 1 : usure

**Paramètre êta** : durée de vie caractéristique estimée.

**Temps moyen entre défaillances** : durée moyenne séparant deux pannes.

**Temps moyen de réparation** : durée moyenne nécessaire pour remettre l’équipement en service.

**Disponibilité intrinsèque** : part du temps pendant laquelle l’équipement est disponible.

**Température maximale du point chaud** : niveau thermique le plus sévère estimé.

**Facteur maximal d’accélération du vieillissement** : indicateur montrant à quel point la chaleur accélère la dégradation.

**Perte de vie (%)** : part estimée de la durée de vie déjà consommée.

**Intervalle recommandé** : délai retenu pour agir en maintenance.

**Score de priorité** : score interne permettant de classer les équipements.

**Décision finale** : conclusion globale après synthèse de tous les résultats.
        """
    )


# =========================================================
# Chargement données
# =========================================================
failures_dataframe = load_failures_dataframe(uploaded_failures_csv)
if failures_dataframe.empty:
    st.error("Aucun jeu de données de temps entre défaillances n’est disponible.")
    st.stop()

project_sheets = load_project_context(uploaded_project_xlsx)
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
    st.info(f"Projet thermique disponible : {'Oui' if bool(project_sheets) else 'Non'}")
with info_col_3:
    st.info(f"Optimisation disponible : {'Oui' if not optimization_dataframe.empty else 'Non'}")


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

        thermal_dataframe_for_equipment, thermal_configuration_for_equipment = extract_thermal_for_equipment(
            project_sheets,
            equipment_code,
        )

        try:
            result = analyze_ttf_pipeline(
                ttf_series=time_to_failure_series,
                alpha=float(alpha_value),
                repair_series=repair_time_series,
                thermal_df=thermal_dataframe_for_equipment,
                thermal_config=thermal_configuration_for_equipment,
            )
        except Exception as error:
            st.warning(f"{equipment_code} : analyse impossible ({error})")
            continue

        results_by_equipment[equipment_code] = result
        detail_tables_by_equipment[equipment_code] = result.get("tables", {}) or {}

        reliability_result = result.get("reliability", {}) or {}
        indicators = reliability_result.get("indicators", {}) or {}
        parameters = reliability_result.get("params", {}) or {}
        decision = reliability_result.get("decision", {}) or {}
        goodness = reliability_result.get("goodness", {}) or {}
        thermal_result = result.get("thermal") or {}
        thermal_summary = (thermal_result.get("summary") or {}) if isinstance(thermal_result, dict) else {}

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

        global_rows.append(
            {
                "equipment_code": equipment_code,
                "n_ttf": len(time_to_failure_series),
                "trend_detected": "Oui" if decision.get("has_trend") else "Non",
                "trend_direction": decision.get("trend_direction"),
                "dependence_detected": "Oui" if decision.get("has_dependence") else "Non",
                "model": reliability_result.get("model"),
                "process_variant": reliability_result.get("process_variant"),
                "distribution": reliability_result.get("distribution"),
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
                "theta_hs_max": thermal_summary.get("theta_hs_max"),
                "faa_max": thermal_summary.get("faa_max"),
                "loss_of_life_pct": thermal_summary.get("loss_of_life_pct"),
                "thermal_status": compute_thermal_status(
                    thermal_summary.get("theta_hs_max"),
                    thermal_summary.get("faa_max"),
                    thermal_summary.get("loss_of_life_pct"),
                ),
                "maintenance_type": optimization_row.get("maintenance_type"),
                "T_recommended_h": optimization_row.get("T_recommended_h"),
                "T_R_h": optimization_row.get("T_R_h"),
                "T_cost_h": optimization_row.get("T_cost_h"),
                "R(T_cost)": optimization_row.get("R(T_cost)"),
                "C_min_per_h": optimization_row.get("C_min_per_h"),
                "next_due_date": maintenance_row.get("next_due_date"),
                "days_left": maintenance_row.get("days_left"),
            }
        )

summary_dataframe = pd.DataFrame(global_rows)
if summary_dataframe.empty:
    st.error("Aucun équipement exploitable n’a pu être analysé.")
    st.stop()

final_decisions = summary_dataframe.apply(lambda row: compute_final_decision_row(row), axis=1)
summary_dataframe[["decision_finale", "motif_decision", "priority_score"]] = pd.DataFrame(
    final_decisions.tolist(),
    index=summary_dataframe.index,
)

summary_dataframe["priorite"] = pd.cut(
    summary_dataframe["priority_score"],
    bins=[-1, 3, 6, 9, 100],
    labels=["Faible", "Modérée", "Élevée", "Critique"],
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

    detail_tables_by_equipment[equipment_code]["trace_trend"] = build_trend_trace_table(selected_result, alpha_value)
    detail_tables_by_equipment[equipment_code]["trace_dependence"] = build_dependence_trace_table(selected_result, alpha_value)
    detail_tables_by_equipment[equipment_code]["trace_model_choice"] = build_model_trace_table(selected_result)
    detail_tables_by_equipment[equipment_code]["trace_goodness_of_fit"] = build_goodness_of_fit_trace_table(selected_result)
    detail_tables_by_equipment[equipment_code]["trace_parameters"] = build_parameter_trace_table(selected_row)
    detail_tables_by_equipment[equipment_code]["trace_optimization"] = build_optimization_trace_table(selected_row)
    detail_tables_by_equipment[equipment_code]["trace_final_decision"] = build_decision_trace_table(selected_row)

trend_overview_dataframe = summary_dataframe[
    [
        "equipment_code",
        "n_ttf",
        "trend_detected",
        "trend_direction",
        "dependence_detected",
        "model",
        "process_variant",
        "distribution",
    ]
].copy()

goodness_overview_dataframe = summary_dataframe[
    [
        "equipment_code",
        "distribution",
        "aic",
        "ks_p",
        "chi2_p",
        "cvm_p",
        "goodness_accepted",
    ]
].copy()

risk_overview_dataframe = summary_dataframe[
    [
        "equipment_code",
        "beta",
        "eta_h",
        "mtbf_h",
        "mttr_h",
        "availability_pct",
        "theta_hs_max",
        "faa_max",
        "loss_of_life_pct",
        "thermal_status",
    ]
].copy()

optimization_overview_dataframe = summary_dataframe[
    [
        "equipment_code",
        "maintenance_type",
        "T_recommended_h",
        "T_R_h",
        "T_cost_h",
        "R(T_cost)",
        "C_min_per_h",
        "next_due_date",
        "days_left",
    ]
].copy()

final_decision_dataframe = summary_dataframe[
    [
        "equipment_code",
        "model",
        "process_variant",
        "distribution",
        "thermal_status",
        "maintenance_type",
        "days_left",
        "priority_score",
        "priorite",
        "decision_finale",
        "motif_decision",
    ]
].copy()

due_tasks_dataframe = maintenance_due_dataframe.copy() if not maintenance_due_dataframe.empty else pd.DataFrame()

global_tables = {
    "Synthese_globale": summary_dataframe,
    "Vue_tendance": trend_overview_dataframe,
    "Vue_ajustement": goodness_overview_dataframe,
    "Vue_risque": risk_overview_dataframe,
    "Vue_optimisation": optimization_overview_dataframe,
    "Taches_dues": due_tasks_dataframe,
    "Decision_finale": final_decision_dataframe,
}


# =========================================================
# KPIs
# =========================================================
metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
with metric_col_1:
    st.metric("Équipements analysés", len(summary_dataframe))
with metric_col_2:
    st.metric("Priorité critique", int((summary_dataframe["priorite"].astype(str) == "Critique").sum()))
with metric_col_3:
    st.metric("Tâches dues", len(due_tasks_dataframe))
with metric_col_4:
    st.metric("Cas thermiques critiques", int((summary_dataframe["thermal_status"] == "Critique").sum()))
with metric_col_5:
    st.metric("Processus non homogènes détectés", int((summary_dataframe["model"].astype(str).str.upper() == "NHPP").sum()))


# =========================================================
# Onglets
# =========================================================
page_tab_1, page_tab_2, page_tab_3 = st.tabs(["Vue synthèse", "Traçabilité par équipement", "Exports"])

with page_tab_1:
    st.subheader("Synthèse globale")
    st.dataframe(rename_columns_for_display(summary_dataframe), use_container_width=True, hide_index=True)

    st.markdown("### Tableau séparé — Tests d’ajustement")
    st.dataframe(rename_columns_for_display(goodness_overview_dataframe), use_container_width=True, hide_index=True)

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        figure_process, axis_process = plt.subplots(figsize=(8, 4))
        process_counts = summary_dataframe["model"].astype(str).value_counts()
        axis_process.bar(process_counts.index.tolist(), process_counts.values.tolist())
        axis_process.set_title("Répartition des processus retenus")
        axis_process.set_xlabel("Processus")
        axis_process.set_ylabel("Nombre d’équipements")
        axis_process.grid(True, alpha=0.25)
        st.pyplot(figure_process, clear_figure=True)

    with chart_col_2:
        figure_priority, axis_priority = plt.subplots(figsize=(8, 4))
        priority_dataframe = summary_dataframe[["equipment_code", "priority_score"]].sort_values("priority_score")
        axis_priority.barh(priority_dataframe["equipment_code"], priority_dataframe["priority_score"])
        axis_priority.set_title("Score de priorité par équipement")
        axis_priority.set_xlabel("Score de priorité")
        axis_priority.set_ylabel("Code équipement")
        axis_priority.grid(True, alpha=0.25)
        st.pyplot(figure_priority, clear_figure=True)

with page_tab_2:
    selected_equipment_code = st.selectbox(
        "Choisir un équipement",
        options=summary_dataframe["equipment_code"].tolist(),
    )

    selected_result = results_by_equipment[selected_equipment_code]
    selected_row = summary_dataframe[summary_dataframe["equipment_code"] == selected_equipment_code].iloc[0].to_dict()

    st.markdown("### Tableau 1 — Validation des tests de tendance")
    st.dataframe(
        rename_columns_for_display(build_trend_trace_table(selected_result, alpha_value)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 2 — Validation des tests de dépendance")
    st.dataframe(
        rename_columns_for_display(build_dependence_trace_table(selected_result, alpha_value)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 3 — Choix du processus et du modèle")
    st.dataframe(
        rename_columns_for_display(build_model_trace_table(selected_result)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 4 — Tests d’ajustement")
    st.dataframe(
        rename_columns_for_display(build_goodness_of_fit_trace_table(selected_result)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 5 — Paramètres fiabilistes et thermiques")
    st.dataframe(
        rename_columns_for_display(build_parameter_trace_table(selected_row)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 6 — Optimisation et maintenance retenue")
    st.dataframe(
        rename_columns_for_display(build_optimization_trace_table(selected_row)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tableau 7 — Traçabilité détaillée de la décision finale")
    st.dataframe(
        rename_columns_for_display(build_decision_trace_table(selected_row)),
        use_container_width=True,
        hide_index=True,
    )

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
                    title="Résultat global de l'analyse  et de l'optimisation de maintenance",
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