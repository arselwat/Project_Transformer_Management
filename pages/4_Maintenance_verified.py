from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.ui import render_shell, render_page_header

try:
    from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
except Exception as error:
    export_pm_plan_with_kits_pdf = None
    maintenance_report_error_message = str(error)
else:
    maintenance_report_error_message = None


st.set_page_config(page_title="Maintenance", page_icon="🛠️", layout="wide")
require_login()

render_shell("pages/4_Maintenance_verified.py")
render_page_header(
    "Maintenance",
    "Planning issu de l’optimisation, échéances, commentaires détaillés et recommandation finale.",
    "🛠️",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def hash_dataframe(dataframe: pd.DataFrame) -> str:
    if dataframe is None or dataframe.empty:
        return "empty"
    return hashlib.md5(dataframe.to_csv(index=False).encode("utf-8")).hexdigest()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if pd.isna(numeric_value) or math.isnan(numeric_value) or math.isinf(numeric_value):
            return default
        return numeric_value
    except Exception:
        return default


def format_number(value: Any, decimals: int = 2, default: str = "—") -> str:
    numeric_value = safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{decimals}f}"


def maintenance_label(maintenance_type: str) -> str:
    normalized = (maintenance_type or "").strip().lower()
    if not normalized:
        return "Type non défini"
    if "correct" in normalized:
        return "Maintenance corrective"
    if "condition" in normalized or "inspection" in normalized:
        return "Maintenance conditionnelle ou inspection"
    if "predict" in normalized or "prédict" in normalized:
        return "Maintenance prédictive"
    if "prévent" in normalized or "prevent" in normalized:
        return "Maintenance préventive planifiée"
    return maintenance_type


def readable_interval_source(source_name: Optional[str]) -> str:
    mapping = {
        "T_recommended_h": "intervalle recommandé",
        "T_R_h": "intervalle issu du critère de fiabilité",
        "T_cost_h": "intervalle issu du critère économique",
        "interval_opt_h": "intervalle optimisé",
        "interval_h": "intervalle générique",
    }
    return mapping.get(str(source_name or ""), "source non précisée")


def pick_interval_hours(row: dict) -> tuple[Optional[float], Optional[str]]:
    for column_name in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
        if column_name in row and row[column_name] is not None:
            interval_value = safe_float(row[column_name], None)
            if interval_value is not None and interval_value > 0:
                return interval_value, column_name
    return None, None


def beta_zone(beta_value: Optional[float]) -> str:
    if beta_value is None:
        return "unknown"
    if beta_value < 0.8:
        return "early"
    if beta_value <= 1.2:
        return "random"
    return "wear"


def beta_explanation(beta_value: Optional[float]) -> str:
    zone = beta_zone(beta_value)
    if zone == "early":
        return "Le paramètre bêta est inférieur à 1 : cela évoque surtout des défauts précoces ou des problèmes initiaux."
    if zone == "random":
        return "Le paramètre bêta est proche de 1 : les défaillances sont plutôt aléatoires."
    if zone == "wear":
        return "Le paramètre bêta est supérieur à 1 : cela évoque une phase d’usure ou de vieillissement."
    return "Le paramètre bêta n’est pas disponible."


def interval_intensity(interval_hours: Optional[float]) -> str:
    if interval_hours is None:
        return "unknown"
    if interval_hours <= 168:
        return "high"
    if interval_hours <= 720:
        return "medium"
    return "low"


def interval_intensity_explanation(interval_hours: Optional[float]) -> str:
    intensity = interval_intensity(interval_hours)
    if intensity == "high":
        return "L’intervalle est court : l’équipement demande des actions rapprochées."
    if intensity == "medium":
        return "L’intervalle est intermédiaire : il s’agit d’un compromis entre risque et coût."
    if intensity == "low":
        return "L’intervalle est long : la surveillance peut être plus espacée."
    return "L’intensité de planification n’a pas pu être déterminée."


def process_explanation(process_name: Optional[str], process_variant: Optional[str]) -> str:
    process_upper = str(process_name or "").upper()
    variant_upper = str(process_variant or "").upper()

    if process_upper == "NHPP":
        return "Le processus retenu est un processus de Poisson non homogène : le comportement évolue dans le temps."
    if process_upper == "BPP":
        return "Le processus retenu signale une dépendance entre événements : une panne peut influencer les suivantes."
    if variant_upper == "HPP":
        return "Le variant homogène du processus de Poisson est retenu : le taux de défaillance reste approximativement constant."
    return "Le processus retenu correspond à un renouvellement avec défaillances considérées comme indépendantes."


def thermal_status_explanation(thermal_status: Optional[str]) -> str:
    normalized = str(thermal_status or "").lower()
    if "alerte" in normalized or "critique" in normalized:
        return "Les indicateurs thermiques montrent une situation défavorable qui peut accélérer le vieillissement."
    if "conforme" in normalized or "normal" in normalized:
        return "Les indicateurs thermiques restent dans une zone acceptable."
    if "analyse thermique calculée" in normalized:
        return "Une analyse thermique est disponible, mais aucun seuil de conformité strict n’a été imposé ici."
    return "Aucune information thermique claire n’est disponible."


def build_maintenance_comment(
    *,
    beta: Optional[float],
    eta_h: Optional[float],
    interval_h: Optional[float],
    interval_source: Optional[str],
    maintenance_type_label: str,
    process_model: Optional[str] = None,
    process_variant: Optional[str] = None,
    thermal_status: Optional[str] = None,
    thermal_ageing_acceleration_factor_max: Optional[float] = None,
    thermal_loss_of_life_percent: Optional[float] = None,
) -> str:
    beta_value = safe_float(beta, None)
    eta_value = safe_float(eta_h, None)
    interval_value = safe_float(interval_h, None)
    ageing_acceleration_factor_value = safe_float(thermal_ageing_acceleration_factor_max, None)
    loss_of_life_value = safe_float(thermal_loss_of_life_percent, None)

    source_text = readable_interval_source(interval_source)
    parts: list[str] = []

    parts.append(f"Type recommandé : {maintenance_type_label}.")
    parts.append(process_explanation(process_model, process_variant))
    parts.append(beta_explanation(beta_value))
    parts.append(thermal_status_explanation(thermal_status))

    if interval_value is not None:
        parts.append(f"L’intervalle retenu est de {interval_value:.1f} heures et provient de la source suivante : {source_text}.")
        parts.append(interval_intensity_explanation(interval_value))
    else:
        parts.append("Aucun intervalle exploitable n’a été retenu automatiquement.")

    if eta_value is not None and interval_value is not None:
        ratio = interval_value / max(eta_value, 1e-9)
        parts.append(f"Le paramètre êta, qui représente une durée de vie caractéristique, vaut environ {eta_value:.1f} heures.")
        if ratio > 1.0:
            parts.append("L’intervalle choisi dépasse la durée de vie caractéristique estimée : une surveillance renforcée est recommandée.")
        elif ratio > 0.5:
            parts.append("L’intervalle choisi reste proche de la durée de vie caractéristique : il faut rester vigilant.")
        else:
            parts.append("L’intervalle choisi reste nettement inférieur à la durée de vie caractéristique : l’approche est prudente.")
    elif eta_value is not None:
        parts.append(f"Le paramètre êta, qui représente une durée de vie caractéristique, vaut environ {eta_value:.1f} heures.")

    if ageing_acceleration_factor_value is not None:
        parts.append(
            f"Le facteur maximal d’accélération du vieillissement est d’environ {ageing_acceleration_factor_value:.3f}."
        )

    if loss_of_life_value is not None:
        parts.append(
            f"La perte de vie estimée est d’environ {loss_of_life_value:.3f} %."
        )

    actions: list[str] = []
    zone = beta_zone(beta_value)
    if zone == "early":
        actions.extend([
            "vérifier la mise en service et la qualité des montages",
            "rechercher les causes racines des incidents répétés",
            "maintenir une surveillance rapprochée",
        ])
    elif zone == "random":
        actions.extend([
            "garder une surveillance régulière",
            "préparer les consommables pour réduire le temps de réparation",
            "contrôler les signes faibles avant toute dérive",
        ])
    elif zone == "wear":
        actions.extend([
            "planifier l’intervention avant la zone critique",
            "renforcer les contrôles ciblés",
            "préparer le remplacement des éléments vieillissants",
        ])

    if thermal_status and "alerte" in str(thermal_status).lower():
        actions.insert(0, "contrôler immédiatement l’échauffement et le système de refroidissement")
    elif thermal_status and ("conforme" in str(thermal_status).lower() or "normal" in str(thermal_status).lower()):
        actions.insert(0, "conserver la surveillance thermique actuelle")

    if actions:
        parts.append("Actions conseillées : " + "; ".join(actions) + ".")

    return " ".join(parts)


def load_optimization_fallback() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    fallback_file = base_dir / "data" / "last_optimization.csv"
    if fallback_file.exists():
        try:
            dataframe = pd.read_csv(fallback_file)
            dataframe.columns = [str(column).strip() for column in dataframe.columns]
            return dataframe
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def compute_priority_decision(row: Dict[str, Any]) -> tuple[int, str, str, str]:
    score = 0

    process_name = str(row.get("model", "RP")).upper()
    thermal_status = str(row.get("thermal_status", "Aucune donnée thermique disponible"))
    beta_value = safe_float(row.get("beta"), None)
    days_left = safe_float(row.get("days_left"), None)
    global_conformity = row.get("admissible_global")

    if process_name == "NHPP":
        score += 3
    elif process_name == "BPP":
        score += 2
    else:
        score += 1

    if beta_value is not None:
        if beta_value > 1.2:
            score += 3
        elif beta_value >= 1.0:
            score += 2
        elif beta_value < 0.8:
            score += 1

    normalized_thermal_status = thermal_status.lower()
    if "alerte" in normalized_thermal_status or "critique" in normalized_thermal_status:
        score += 4
    elif "conforme" in normalized_thermal_status or "normal" in normalized_thermal_status:
        score += 1

    if days_left is not None:
        if days_left <= 7:
            score += 3
        elif days_left <= 30:
            score += 2
        elif days_left <= 90:
            score += 1

    if global_conformity is False:
        score += 2

    if score >= 10:
        priority_level = "Critique"
        final_decision = "Intervention prioritaire"
        reason = "Les signaux fiabilistes, thermiques et calendaires imposent une action rapide."
    elif score >= 7:
        priority_level = "Élevée"
        final_decision = "Préventif renforcé"
        reason = "Le niveau de risque reste significatif : il faut maintenir une action planifiée rapide."
    elif score >= 4:
        priority_level = "Modérée"
        final_decision = "Surveillance active"
        reason = "La situation demande une surveillance structurée et le respect du plan calculé."
    else:
        priority_level = "Faible"
        final_decision = "Suivi nominal"
        reason = "Aucun signal critique immédiat n’est dominant : le plan standard peut être suivi."

    return score, priority_level, final_decision, reason


DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "title": "Titre du plan",
    "maintenance_type": "Type de maintenance recommandé",
    "maintenance_comment": "Commentaire détaillé",
    "interval_source": "Source brute de l’intervalle",
    "interval_source_readable": "Source lisible de l’intervalle",
    "interval_h": "Intervalle retenu (heures)",
    "periodicity_days": "Périodicité (jours)",
    "next_due_date": "Prochaine date d’échéance",
    "days_left": "Nombre de jours restants",
    "beta": "Paramètre bêta",
    "eta_h": "Paramètre êta (heures)",
    "gamma_h": "Paramètre gamma (heures)",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi de probabilité retenue",
    "T_recommended_h": "Intervalle recommandé (heures)",
    "T_R_h": "Intervalle issu du critère de fiabilité (heures)",
    "T_cost_h": "Intervalle issu du critère économique (heures)",
    "R_at_T": "Fiabilité au niveau de l’intervalle économique",
    "C_min_per_h": "Coût minimal par heure",
    "FAA_max": "Facteur maximal d’accélération du vieillissement",
    "loss_of_life_pct": "Perte de vie (%)",
    "thermal_status": "Statut thermique",
    "thermal_ok": "Conformité thermique",
    "reliability_ok": "Conformité fiabiliste",
    "admissible_global": "Conformité globale",
    "status": "Statut du plan",
    "priority_score": "Score de priorité",
    "priority_level": "Niveau de priorité",
    "final_decision": "Décision finale",
    "final_reason": "Motif de la décision",
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


def build_virtual_maintenance_plan_from_optimization(
    optimization_dataframe: pd.DataFrame,
    start_date: date,
    due_window_days: int,
    show_all: bool = True,
    only_preventive: bool = False,
    only_globally_conformant: bool = False,
    minimum_days: int = 1,
) -> Dict[str, Any]:
    if optimization_dataframe is None or optimization_dataframe.empty:
        return {"ok": False, "msg": "Le tableau d’optimisation est vide.", "rows": [], "due": []}

    dataframe = optimization_dataframe.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    if "equipment_code" not in dataframe.columns:
        return {"ok": False, "msg": "La colonne equipment_code est absente.", "rows": [], "due": []}

    rows: List[Dict[str, Any]] = []
    used_interval_columns = {
        "T_recommended_h": 0,
        "T_R_h": 0,
        "T_cost_h": 0,
        "interval_opt_h": 0,
        "interval_h": 0,
        "none": 0,
    }

    for _, record in dataframe.iterrows():
        row = record.to_dict()
        equipment_code = str(row.get("equipment_code") or "").strip()
        if not equipment_code:
            continue

        globally_conformant = row.get("admissible_global")
        if only_globally_conformant and globally_conformant is not None and not bool(globally_conformant):
            continue

        maintenance_type_readable = maintenance_label(str(row.get("maintenance_type") or "").strip())
        if only_preventive and "préventive" not in maintenance_type_readable.lower():
            continue

        interval_hours, interval_source = pick_interval_hours(row)
        if not interval_hours:
            used_interval_columns["none"] += 1
            continue

        if interval_source in used_interval_columns:
            used_interval_columns[interval_source] += 1

        periodicity_days = max(int(minimum_days), int(round(float(interval_hours) / 24.0)))
        next_due_date = start_date + timedelta(days=periodicity_days)
        days_left = (next_due_date - start_date).days

        beta_value = safe_float(row.get("beta"), None)
        eta_value = safe_float(row.get("eta_h"), None)

        maintenance_comment = build_maintenance_comment(
            beta=beta_value,
            eta_h=eta_value,
            interval_h=safe_float(interval_hours, None),
            interval_source=interval_source,
            maintenance_type_label=maintenance_type_readable,
            process_model=str(row.get("model") or ""),
            process_variant=str(row.get("process_variant") or ""),
            thermal_status=str(row.get("thermal_status") or ""),
            thermal_ageing_acceleration_factor_max=safe_float(row.get("FAA_max"), None),
            thermal_loss_of_life_percent=safe_float(row.get("loss_of_life_pct"), None),
        )

        task = {
            "equipment_code": equipment_code,
            "title": "Plan issu de l’optimisation",
            "maintenance_type": maintenance_type_readable,
            "maintenance_comment": maintenance_comment,
            "interval_source": interval_source,
            "interval_source_readable": readable_interval_source(interval_source),
            "interval_h": float(interval_hours),
            "periodicity_days": periodicity_days,
            "next_due_date": next_due_date.isoformat(),
            "days_left": int(days_left),
            "beta": beta_value,
            "eta_h": eta_value,
            "gamma_h": row.get("gamma_h"),
            "model": row.get("model"),
            "process_variant": row.get("process_variant"),
            "distribution": row.get("distribution"),
            "T_recommended_h": row.get("T_recommended_h"),
            "T_R_h": row.get("T_R_h"),
            "T_cost_h": row.get("T_cost_h"),
            "R_at_T": row.get("R(T_cost)"),
            "C_min_per_h": row.get("C_min_per_h"),
            "FAA_max": row.get("FAA_max"),
            "loss_of_life_pct": row.get("loss_of_life_pct"),
            "thermal_status": row.get("thermal_status"),
            "thermal_ok": row.get("thermal_ok"),
            "reliability_ok": row.get("reliability_ok"),
            "admissible_global": row.get("admissible_global"),
            "status": "Plan virtuel",
        }

        priority_score, priority_level, final_decision, final_reason = compute_priority_decision(task)
        task["priority_score"] = priority_score
        task["priority_level"] = priority_level
        task["final_decision"] = final_decision
        task["final_reason"] = final_reason

        if show_all or (days_left <= due_window_days):
            rows.append(task)

    rows = sorted(
        rows,
        key=lambda row: (
            -int(row.get("priority_score", 0)),
            int(row.get("days_left", 999999)),
            str(row.get("equipment_code", "")),
        ),
    )
    due_rows = [row for row in rows if int(row.get("days_left", 999999)) <= due_window_days]

    return {
        "ok": True,
        "rows": rows,
        "due": due_rows,
        "used_cols_stats": used_interval_columns,
        "interval_priority": ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"],
    }


# -------------------------------------------------------------------
# Chargement
# -------------------------------------------------------------------
optimization_dataframe = st.session_state.get("optimization_df")
if not isinstance(optimization_dataframe, pd.DataFrame) or optimization_dataframe.empty:
    optimization_dataframe = load_optimization_fallback()

if not isinstance(optimization_dataframe, pd.DataFrame) or optimization_dataframe.empty:
    st.info("Aucune optimisation disponible. Va d’abord dans la page Optimisation.")
    st.stop()

optimization_dataframe = optimization_dataframe.copy()
optimization_dataframe.columns = [str(column).strip() for column in optimization_dataframe.columns]

optimization_hash = hash_dataframe(optimization_dataframe)
st.success(f"Optimisation synchronisée | lignes={len(optimization_dataframe)} | empreinte={optimization_hash}")


# -------------------------------------------------------------------
# Explications
# -------------------------------------------------------------------
with st.expander("Comprendre clairement les variables utilisées sur cette page", expanded=False):
    st.markdown(
        """
**Paramètre bêta** : décrit la forme du vieillissement.  
- inférieur à 1 : défauts précoces ou problèmes initiaux  
- proche de 1 : comportement plutôt aléatoire  
- supérieur à 1 : phase d’usure

**Paramètre êta** : durée de vie caractéristique estimée. Plus il est grand, plus l’équipement peut tenir longtemps.

**Paramètre gamma** : éventuel décalage du modèle dans le temps.

**Intervalle recommandé** : temps retenu pour préparer l’action de maintenance.

**Intervalle issu du critère de fiabilité** : temps qui respecte le niveau de fiabilité visé.

**Intervalle issu du critère économique** : temps qui cherche à minimiser le coût moyen.

**Facteur maximal d’accélération du vieillissement** : indicateur thermique montrant à quel point la chaleur accélère le vieillissement de l’isolation.

**Perte de vie (%)** : part estimée de la durée de vie déjà consommée.

**Statut thermique** : indique si la situation thermique est normale, conforme, en alerte ou non disponible.

**Conformité fiabiliste** : indique si le modèle de fiabilité retenu est jugé acceptable.

**Conformité globale** : synthèse de la cohérence fiabilité + thermique.

**Score de priorité** : score interne utilisé pour classer les équipements à traiter en premier.

**Décision finale** : conclusion pratique retenue pour orienter l’action.
        """
    )


# -------------------------------------------------------------------
# Contrôles
# -------------------------------------------------------------------
control_col_1, control_col_2, control_col_3, control_col_4, control_col_5 = st.columns(5)
with control_col_1:
    due_window_days = st.slider("Fenêtre des tâches dues (jours)", 7, 365, 14, 1)
with control_col_2:
    show_all_rows = st.toggle("Afficher toutes les lignes", value=True)
with control_col_3:
    only_preventive_rows = st.toggle("Montrer seulement la maintenance préventive", value=False)
with control_col_4:
    only_globally_conformant_rows = st.toggle("Montrer seulement les cas globalement conformes", value=False)
with control_col_5:
    start_reference_date = st.date_input("Date de départ du planning", value=date.today())

maintenance_plan = build_virtual_maintenance_plan_from_optimization(
    optimization_dataframe=optimization_dataframe,
    start_date=start_reference_date,
    due_window_days=int(due_window_days),
    show_all=bool(show_all_rows),
    only_preventive=bool(only_preventive_rows),
    only_globally_conformant=bool(only_globally_conformant_rows),
    minimum_days=1,
)

if not maintenance_plan.get("ok"):
    st.error(maintenance_plan.get("msg", "Erreur lors de la construction du plan de maintenance."))
    st.stop()

all_planned_rows = maintenance_plan.get("rows", [])
due_rows = maintenance_plan.get("due", [])

st.caption(f"Ordre de priorité des colonnes d’intervalle : {', '.join(maintenance_plan.get('interval_priority', []))}")
st.caption(f"Nombre d’utilisations de chaque colonne d’intervalle : {maintenance_plan.get('used_cols_stats')}")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Équipements planifiés", len(all_planned_rows))
with metric_col_2:
    st.metric("Tâches dues dans la fenêtre", len(due_rows))
with metric_col_3:
    globally_conformant_count = int(
        pd.Series([row.get("admissible_global") for row in all_planned_rows]).fillna(False).astype(bool).sum()
    ) if all_planned_rows else 0
    st.metric("Plans globalement conformes", globally_conformant_count)
with metric_col_4:
    preventive_count = int(sum(1 for row in all_planned_rows if "préventive" in str(row.get("maintenance_type", "")).lower()))
    st.metric("Actions préventives", preventive_count)


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
page_tabs = st.tabs([
    "Commentaires et explications",
    "Planning complet",
    "Tâches dues",
    "Recommandation finale",
    "Exports",
])

with page_tabs[0]:
    st.subheader("Commentaires de maintenance")
    if all_planned_rows:
        all_rows_dataframe = pd.DataFrame(all_planned_rows)
        comment_dataframe = all_rows_dataframe[
            [
                column for column in [
                    "equipment_code",
                    "maintenance_type",
                    "interval_source_readable",
                    "interval_h",
                    "beta",
                    "eta_h",
                    "thermal_status",
                    "priority_level",
                    "maintenance_comment",
                ] if column in all_rows_dataframe.columns
            ]
        ].copy()
        comment_dataframe = comment_dataframe.drop_duplicates(subset=["equipment_code"]).sort_values("equipment_code")
        st.dataframe(rename_columns_for_display(comment_dataframe), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun commentaire disponible.")

with page_tabs[1]:
    st.subheader("Planning complet")
    if not all_planned_rows:
        st.warning("Aucune ligne exploitable.")
    else:
        full_planning_dataframe = pd.DataFrame(all_planned_rows)
        selected_columns = [
            "equipment_code",
            "maintenance_type",
            "interval_source_readable",
            "interval_h",
            "periodicity_days",
            "next_due_date",
            "days_left",
            "beta",
            "eta_h",
            "gamma_h",
            "model",
            "process_variant",
            "distribution",
            "FAA_max",
            "loss_of_life_pct",
            "thermal_status",
            "admissible_global",
            "priority_score",
            "priority_level",
            "final_decision",
            "maintenance_comment",
        ]
        selected_columns = [column for column in selected_columns if column in full_planning_dataframe.columns]
        st.dataframe(
            rename_columns_for_display(full_planning_dataframe[selected_columns]),
            use_container_width=True,
            hide_index=True,
        )

with page_tabs[2]:
    st.subheader("Tâches dues dans la fenêtre choisie")
    if not due_rows:
        st.info("Aucune tâche n’arrive à échéance dans la fenêtre choisie.")
    else:
        due_rows_dataframe = pd.DataFrame(due_rows)
        selected_columns = [
            "equipment_code",
            "maintenance_type",
            "interval_source_readable",
            "interval_h",
            "next_due_date",
            "days_left",
            "beta",
            "eta_h",
            "model",
            "process_variant",
            "distribution",
            "FAA_max",
            "loss_of_life_pct",
            "thermal_status",
            "admissible_global",
            "priority_score",
            "priority_level",
            "final_decision",
            "maintenance_comment",
        ]
        selected_columns = [column for column in selected_columns if column in due_rows_dataframe.columns]
        st.dataframe(
            rename_columns_for_display(due_rows_dataframe[selected_columns]),
            use_container_width=True,
            hide_index=True,
        )

with page_tabs[3]:
    st.subheader("Recommandation finale")
    if all_planned_rows:
        ranking_dataframe = pd.DataFrame(all_planned_rows).copy()
        if "priority_score" in ranking_dataframe.columns:
            ranking_dataframe["priority_score"] = pd.to_numeric(ranking_dataframe["priority_score"], errors="coerce").fillna(0)
        if "days_left" in ranking_dataframe.columns:
            ranking_dataframe["days_left"] = pd.to_numeric(ranking_dataframe["days_left"], errors="coerce").fillna(999999)

        ranking_dataframe = ranking_dataframe.sort_values(
            ["priority_score", "days_left"],
            ascending=[False, True],
        )

        best_row = ranking_dataframe.iloc[0].to_dict()

        st.success(
            f"Équipement prioritaire : {best_row.get('equipment_code', '—')} | "
            f"{best_row.get('maintenance_type', '—')} | "
            f"échéance prévue : {best_row.get('next_due_date', '—')}"
        )

        summary_rows = pd.DataFrame([
            {
                "equipment_code": best_row.get("equipment_code"),
                "maintenance_type": best_row.get("maintenance_type"),
                "model": best_row.get("model"),
                "process_variant": best_row.get("process_variant"),
                "distribution": best_row.get("distribution"),
                "interval_h": best_row.get("interval_h"),
                "days_left": best_row.get("days_left"),
                "thermal_status": best_row.get("thermal_status"),
                "priority_score": best_row.get("priority_score"),
                "priority_level": best_row.get("priority_level"),
                "final_decision": best_row.get("final_decision"),
                "final_reason": best_row.get("final_reason"),
            }
        ])
        st.dataframe(rename_columns_for_display(summary_rows), use_container_width=True, hide_index=True)

        st.write(best_row.get("maintenance_comment", "Aucun commentaire disponible."))
    else:
        st.info("Aucune recommandation disponible.")

with page_tabs[4]:
    st.subheader("Exports")

    st.session_state["pm_virtual_all"] = all_planned_rows
    st.session_state["pm_virtual_due"] = due_rows

    if all_planned_rows:
        all_rows_dataframe = pd.DataFrame(all_planned_rows)
        csv_bytes = rename_columns_for_display(all_rows_dataframe).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le planning complet au format CSV",
            data=csv_bytes,
            file_name="maintenance_virtual_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )

    include_full_planning_in_pdf = st.toggle("Inclure tout le planning dans le PDF", value=False)

    if export_pm_plan_with_kits_pdf is None:
        st.info("Le module PDF de maintenance n’est pas disponible.")
        if maintenance_report_error_message:
            st.caption(maintenance_report_error_message)
    else:
        if st.button("Générer le PDF de maintenance", use_container_width=True):
            tasks_for_pdf = all_planned_rows if include_full_planning_in_pdf else due_rows
            metrics_table = optimization_dataframe.to_dict("records")
            try:
                output_path = export_pm_plan_with_kits_pdf(
                    tasks_due=tasks_for_pdf,
                    kits_by_eq={},
                    metrics_table=metrics_table,
                    out_dir="reports",
                    title="Plan de maintenance issu de l’optimisation",
                    include_kits=False,
                    tools_checklist=None,
                )
                st.success("PDF généré.")
                with open(output_path, "rb") as file:
                    st.download_button(
                        "Télécharger le PDF de maintenance",
                        data=file,
                        file_name=Path(output_path).name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as error:
                st.error(f"PDF : {error}")