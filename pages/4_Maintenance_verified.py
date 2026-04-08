from __future__ import annotations

import hashlib
import html
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
    "Choix moderne de maintenance, hiérarchisation des cas critiques et génération du rapport.",
    "🛠️",
)


st.markdown(
    """
    <style>
    .maint-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px 18px;
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .maint-card-critical {
        border: 2px solid #dc2626;
        background: linear-gradient(180deg, #fff5f5 0%, #fff1f2 100%);
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.10);
    }
    .maint-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #111827;
    }
    .maint-title-critical {
        color: #b91c1c;
    }
    .maint-meta {
        font-size: 0.90rem;
        color: #374151;
        line-height: 1.55;
    }
    .maint-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #d1d5db;
        background: #f9fafb;
        color: #111827;
    }
    .maint-badge-critical {
        background: #dc2626;
        border-color: #dc2626;
        color: white;
    }
    .maint-badge-normal {
        background: #e5e7eb;
        border-color: #d1d5db;
        color: #111827;
    }
    .maint-section-title {
        font-size: 1rem;
        font-weight: 700;
        margin: 8px 0 10px 0;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def pick_first_present(row: Dict[str, Any], *keys: str):
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def availability_percent_from_row(row: Dict[str, Any]) -> Optional[float]:
    direct_value = pick_first_present(row, "availability_pct", "Disponibilite_pct", "Disponibilité_pct", "availability")
    direct_numeric = safe_float(direct_value, None)
    if direct_numeric is not None:
        return 100.0 * direct_numeric if direct_numeric <= 1.0 else direct_numeric

    intrinsic_value = pick_first_present(row, "availability_intrinsic")
    intrinsic_numeric = safe_float(intrinsic_value, None)
    if intrinsic_numeric is not None:
        return 100.0 * intrinsic_numeric if intrinsic_numeric <= 1.0 else intrinsic_numeric

    mtbf_value = safe_float(pick_first_present(row, "MTBF_h", "mtbf_h", "MTBF", "mtbf"), None)
    mttr_value = safe_float(pick_first_present(row, "MTTR_h", "mttr_h", "MTTR", "mttr"), None)
    if mtbf_value is not None and mttr_value is not None and (mtbf_value + mttr_value) > 0:
        return 100.0 * mtbf_value / (mtbf_value + mttr_value)
    return None


def maintenance_label(maintenance_type: str) -> str:
    normalized = (maintenance_type or "").strip().lower()
    if not normalized:
        return "Type non défini"
    if "correct" in normalized:
        return "Maintenance corrective"
    if "condition" in normalized or "inspection" in normalized:
        return "Maintenance conditionnelle"
    if "predict" in normalized or "prédict" in normalized:
        return "Maintenance prédictive"
    if "prévent" in normalized or "prevent" in normalized:
        return "Maintenance préventive planifiée"
    return maintenance_type


def readable_interval_source(source_name: Optional[str]) -> str:
    mapping = {
        "T_recommended_h": "intervalle recommandé",
        "T_R_h": "intervalle fiabiliste",
        "T_cost_h": "intervalle économique",
        "interval_opt_h": "intervalle optimisé",
        "interval_h": "intervalle générique",
    }
    return mapping.get(str(source_name or ""), "source non précisée")


def pick_interval_hours(row: dict) -> tuple[Optional[float], Optional[str]]:
    for column_name in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
        interval_value = safe_float(row.get(column_name), None)
        if interval_value is not None and interval_value > 0:
            return interval_value, column_name
    return None, None


def process_score(process_name: str) -> int:
    normalized = (process_name or "").upper()
    if "NHPP" in normalized:
        return 3
    if "BPP" in normalized or "HAWKES" in normalized:
        return 2
    return 1


def choose_maintenance_strategy(row: Dict[str, Any]) -> tuple[str, str]:
    raw_type = maintenance_label(str(row.get("maintenance_type", "")))
    process_name = str(row.get("model", "RP")).upper()
    process_variant = str(row.get("process_variant", "")).upper()
    beta_value = safe_float(row.get("beta"), None)
    days_left = safe_float(row.get("days_left"), None)
    interval_hours = safe_float(row.get("interval_h"), None)
    reliability_interval = safe_float(row.get("T_R_h"), None)
    economic_interval = safe_float(row.get("T_cost_h"), None)

    reasons: List[str] = []

    if days_left is not None and days_left <= 7:
        if beta_value is not None and beta_value > 1.0:
            choice = "Maintenance préventive immédiate"
        elif beta_value is not None and beta_value < 0.9:
            choice = "Maintenance corrective et fiabilisation"
        else:
            choice = "Maintenance conditionnelle prioritaire"
        reasons.append("échéance très proche")
    elif process_name == "BPP":
        choice = "Maintenance conditionnelle renforcée"
        reasons.append("dépendance entre événements")
    elif process_name == "NHPP":
        if beta_value is not None and beta_value > 1.1:
            choice = "Maintenance préventive planifiée"
            reasons.append("usure et tendance évolutive")
        elif beta_value is not None and beta_value < 0.9:
            choice = "Maintenance corrective et fiabilisation"
            reasons.append("défauts précoces dans un processus évolutif")
        else:
            choice = "Maintenance conditionnelle"
            reasons.append("tendance évolutive à surveiller")
    elif process_variant == "HPP":
        choice = "Maintenance conditionnelle"
        reasons.append("taux de défaillance proche du constant")
    elif beta_value is not None and beta_value < 0.9:
        choice = "Maintenance corrective et fiabilisation"
        reasons.append("bêta inférieur à 1")
    elif beta_value is not None and beta_value <= 1.1:
        choice = "Maintenance conditionnelle"
        reasons.append("bêta proche de 1")
    elif beta_value is not None and beta_value > 1.1:
        choice = "Maintenance préventive planifiée"
        reasons.append("bêta supérieur à 1")
    else:
        choice = raw_type if raw_type != "Type non défini" else "Maintenance à confirmer"
        reasons.append("peu d’éléments discriminants")

    if interval_hours is not None:
        if interval_hours <= 168:
            reasons.append("intervalle très court")
        elif interval_hours <= 720:
            reasons.append("intervalle intermédiaire")
        else:
            reasons.append("intervalle long")

    if reliability_interval is not None and economic_interval is not None:
        reasons.append("arbitrage entre fiabilité et économie")

    if raw_type != "Type non défini" and raw_type not in choice:
        reasons.append(f"suggestion initiale : {raw_type.lower()}")

    return choice, "; ".join(reasons)


def compute_priority_decision(row: Dict[str, Any]) -> tuple[int, str, str, str]:
    score = 0
    process_name = str(row.get("model", "RP"))
    beta_value = safe_float(row.get("beta"), None)
    days_left = safe_float(row.get("days_left"), None)
    maintenance_choice = str(row.get("maintenance_choice", ""))

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

    if "immédiate" in maintenance_choice.lower() or "prioritaire" in maintenance_choice.lower():
        score += 2

    if score >= 8:
        priority_level = "Critique"
        final_decision = "Intervention prioritaire"
        reason = (
            f"Le comportement fiabiliste, l’échéance et le type d’action retenu imposent une action rapide. "
            f"Action conseillée : {maintenance_choice or 'maintenance ciblée'}."
        )
    elif score >= 5:
        priority_level = "Élevée"
        final_decision = "Préventif renforcé"
        reason = "Le risque reste important. Une intervention planifiée rapide est recommandée."
    elif score >= 3:
        priority_level = "Modérée"
        final_decision = "Surveillance active"
        reason = "La situation est intermédiaire. Il faut suivre le plan et surveiller les dérives."
    else:
        priority_level = "Normale"
        final_decision = "Suivi nominal"
        reason = "Aucun signal critique immédiat n’est dominant. Le plan standard peut être appliqué."

    return score, priority_level, final_decision, reason


DISPLAY_COLUMN_NAMES = {
    "equipment_code": "Code équipement",
    "maintenance_type": "Type initial",
    "maintenance_choice": "Choix final de maintenance",
    "maintenance_choice_reason": "Justification du choix",
    "maintenance_comment": "Commentaire détaillé",
    "interval_source_readable": "Source de l’intervalle",
    "interval_h": "Intervalle retenu (heures)",
    "periodicity_days": "Périodicité (jours)",
    "next_due_date": "Prochaine échéance",
    "days_left": "Jours restants",
    "beta": "Paramètre bêta",
    "eta_h": "Paramètre êta (heures)",
    "gamma_h": "Paramètre gamma (heures)",
    "MTTF_h": "MTTF (heures)",
    "MTBF_h": "MTBF (heures)",
    "MTTR_h": "MTTR (heures)",
    "availability_pct": "Disponibilité (%)",
    "model": "Processus retenu",
    "process_variant": "Variant du processus",
    "distribution": "Loi retenue",
    "priority_score": "Score de priorité",
    "priority_level": "Niveau de priorité",
    "critical_case": "Cas critique",
    "final_decision": "Décision finale",
    "final_reason": "Motif de la décision",
}


def rename_columns_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe
    renamed_dataframe = dataframe.copy()
    return renamed_dataframe.rename(columns={
        column: DISPLAY_COLUMN_NAMES.get(column, column)
        for column in renamed_dataframe.columns
    })


def build_maintenance_comment(
    *,
    beta: Optional[float],
    eta_h: Optional[float],
    interval_h: Optional[float],
    interval_source: Optional[str],
    maintenance_choice: str,
    maintenance_choice_reason: str,
    process_model: Optional[str] = None,
    process_variant: Optional[str] = None,
) -> str:
    source_text = readable_interval_source(interval_source)
    parts: List[str] = []

    parts.append(f"Choix final : {maintenance_choice}.")
    if maintenance_choice_reason:
        parts.append(f"Base de choix : {maintenance_choice_reason}.")

    if process_model:
        process_upper = str(process_model).upper()
        variant_upper = str(process_variant or "").upper()
        if process_upper == "NHPP":
            parts.append("Le processus est évolutif dans le temps.")
        elif process_upper == "BPP":
            parts.append("Le processus montre une dépendance entre événements.")
        elif variant_upper == "HPP":
            parts.append("Le taux de défaillance reste proche du constant.")
        else:
            parts.append("Le processus retenu reste compatible avec une logique de renouvellement.")

    if beta is not None:
        if beta < 0.8:
            parts.append("Le paramètre bêta est inférieur à 1 : défauts précoces à traiter rapidement.")
        elif beta <= 1.2:
            parts.append("Le paramètre bêta est proche de 1 : surveillance régulière recommandée.")
        else:
            parts.append("Le paramètre bêta est supérieur à 1 : usure dominante et action planifiée souhaitable.")

    if interval_h is not None:
        parts.append(f"L’intervalle retenu est de {interval_h:.1f} heures ({source_text}).")

    if eta_h is not None:
        parts.append(f"La durée de vie caractéristique êta vaut environ {eta_h:.1f} heures.")

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


def build_virtual_maintenance_plan_from_optimization(
    optimization_dataframe: pd.DataFrame,
    start_date: date,
    due_window_days: int,
    show_all: bool = True,
    only_preventive: bool = False,
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

        interval_hours, interval_source = pick_interval_hours(row)
        if not interval_hours:
            used_interval_columns["none"] += 1
            continue

        if interval_source in used_interval_columns:
            used_interval_columns[interval_source] += 1

        periodicity_days = max(int(minimum_days), int(round(float(interval_hours) / 24.0)))
        next_due_date = start_date + timedelta(days=periodicity_days)
        days_left = (next_due_date - start_date).days

        base_maintenance_type = maintenance_label(str(row.get("maintenance_type") or "").strip())
        beta_value = safe_float(row.get("beta"), None)
        eta_value = safe_float(row.get("eta_h"), None)
        gamma_value = safe_float(row.get("gamma_h"), None)
        mttf_value = safe_float(pick_first_present(row, "MTTF_h", "mttf_h", "MTTF", "mttf"), None)
        mtbf_value = safe_float(pick_first_present(row, "MTBF_h", "mtbf_h", "MTBF", "mtbf"), None)
        mttr_value = safe_float(pick_first_present(row, "MTTR_h", "mttr_h", "MTTR", "mttr"), None)
        availability_pct_value = availability_percent_from_row(row)

        preview_task = {
            "equipment_code": equipment_code,
            "maintenance_type": base_maintenance_type,
            "model": row.get("model"),
            "process_variant": row.get("process_variant"),
            "beta": beta_value,
            "days_left": int(days_left),
            "interval_h": float(interval_hours),
            "T_R_h": row.get("T_R_h"),
            "T_cost_h": row.get("T_cost_h"),
        }
        maintenance_choice, maintenance_choice_reason = choose_maintenance_strategy(preview_task)

        if only_preventive and "préventive" not in maintenance_choice.lower():
            continue

        task = {
            "equipment_code": equipment_code,
            "maintenance_type": base_maintenance_type,
            "maintenance_choice": maintenance_choice,
            "maintenance_choice_reason": maintenance_choice_reason,
            "interval_source": interval_source,
            "interval_source_readable": readable_interval_source(interval_source),
            "interval_h": float(interval_hours),
            "periodicity_days": periodicity_days,
            "next_due_date": next_due_date.isoformat(),
            "days_left": int(days_left),
            "beta": beta_value,
            "eta_h": eta_value,
            "gamma_h": gamma_value,
            "MTTF_h": mttf_value,
            "MTBF_h": mtbf_value,
            "MTTR_h": mttr_value,
            "availability_pct": availability_pct_value,
            "model": row.get("model"),
            "process_variant": row.get("process_variant"),
            "distribution": row.get("distribution"),
            "T_recommended_h": row.get("T_recommended_h"),
            "T_R_h": row.get("T_R_h"),
            "T_cost_h": row.get("T_cost_h"),
            "R_at_T": row.get("R(T_cost)"),
            "C_min_per_h": row.get("C_min_per_h"),
            "status": "Plan virtuel",
        }

        task["maintenance_comment"] = build_maintenance_comment(
            beta=beta_value,
            eta_h=eta_value,
            interval_h=safe_float(interval_hours, None),
            interval_source=interval_source,
            maintenance_choice=maintenance_choice,
            maintenance_choice_reason=maintenance_choice_reason,
            process_model=str(row.get("model") or ""),
            process_variant=str(row.get("process_variant") or ""),
        )

        priority_score, priority_level, final_decision, final_reason = compute_priority_decision(task)
        task["priority_score"] = priority_score
        task["priority_level"] = priority_level
        task["critical_case"] = priority_level == "Critique"
        task["final_decision"] = final_decision
        task["final_reason"] = final_reason

        if show_all or (days_left <= due_window_days):
            rows.append(task)

    rows = sorted(
        rows,
        key=lambda current_row: (
            -int(current_row.get("priority_score", 0)),
            int(current_row.get("days_left", 999999)),
            str(current_row.get("equipment_code", "")),
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


def style_rows(dataframe: pd.DataFrame) -> pd.io.formats.style.Styler:
    work = rename_columns_for_display(dataframe.copy())

    def _row_style(row):
        is_critical = str(row.get("Niveau de priorité", "")) == "Critique" or str(row.get("Cas critique", "")).lower() == "true"
        if is_critical:
            return ["background-color: #fff1f2; color: #b91c1c; font-weight: 700;"] * len(row)
        return [""] * len(row)

    return work.style.apply(_row_style, axis=1)


def render_priority_cards(rows: List[Dict[str, Any]], max_cards: int = 8) -> None:
    if not rows:
        st.info("Aucune ligne disponible.")
        return

    st.markdown('<div class="maint-section-title">Vue rapide des priorités</div>', unsafe_allow_html=True)

    for row in rows[:max_cards]:
        is_critical = bool(row.get("critical_case"))
        badge_class = "maint-badge-critical" if is_critical else "maint-badge-normal"
        title_class = "maint-title maint-title-critical" if is_critical else "maint-title"
        card_class = "maint-card maint-card-critical" if is_critical else "maint-card"

        html_block = f"""
        <div class="{card_class}">
            <div class="{title_class}">{html.escape(str(row.get('equipment_code', '—')))}</div>
            <div>
                <span class="maint-badge {badge_class}">{html.escape(str(row.get('priority_level', 'Normale')))}</span>
                <span class="maint-badge maint-badge-normal">{html.escape(str(row.get('maintenance_choice', '—')))}</span>
                <span class="maint-badge maint-badge-normal">{html.escape(str(row.get('distribution', '—')))}</span>
            </div>
            <div class="maint-meta">
                <strong>Décision :</strong> {html.escape(str(row.get('final_decision', '—')))}<br/>
                <strong>Jours restants :</strong> {html.escape(str(row.get('days_left', '—')))}<br/>
                <strong>Intervalle :</strong> {html.escape(format_number(row.get('interval_h'), 1))} h<br/>
                <strong>Raison :</strong> {html.escape(str(row.get('maintenance_choice_reason', '—')))}
            </div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)


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


with st.expander("Comprendre les variables utilisées sur cette page", expanded=False):
    st.markdown(
        """
**Choix final de maintenance** : type d’action retenu après combinaison du processus, du paramètre bêta, des intervalles calculés et des jours restants.

**Cas critique** : équipement qui demande une action très rapide. Il apparaît en rouge dans l’interface et dans le rapport.

**Paramètre bêta** : décrit la forme du vieillissement.  
- inférieur à 1 : défauts précoces  
- proche de 1 : comportement aléatoire  
- supérieur à 1 : usure

**Paramètre êta** : durée de vie caractéristique estimée.

**Décision finale** : conclusion opérationnelle issue de la synthèse de tous les facteurs.
        """
    )


control_col_1, control_col_2, control_col_3, control_col_4 = st.columns(4)
with control_col_1:
    due_window_days = st.slider("Fenêtre des tâches dues (jours)", 7, 365, 14, 1)
with control_col_2:
    show_all_rows = st.toggle("Afficher toutes les lignes", value=True)
with control_col_3:
    only_preventive_rows = st.toggle("Montrer seulement la maintenance préventive", value=False)
with control_col_4:
    start_reference_date = st.date_input("Date de départ du planning", value=date.today())

maintenance_plan = build_virtual_maintenance_plan_from_optimization(
    optimization_dataframe=optimization_dataframe,
    start_date=start_reference_date,
    due_window_days=int(due_window_days),
    show_all=bool(show_all_rows),
    only_preventive=bool(only_preventive_rows),
    minimum_days=1,
)

if not maintenance_plan.get("ok"):
    st.error(maintenance_plan.get("msg", "Erreur lors de la construction du plan de maintenance."))
    st.stop()

all_planned_rows = maintenance_plan.get("rows", [])
due_rows = maintenance_plan.get("due", [])

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    st.metric("Équipements planifiés", len(all_planned_rows))
with metric_col_2:
    st.metric("Tâches dues", len(due_rows))
with metric_col_3:
    st.metric("Cas critiques", int(sum(1 for row in all_planned_rows if bool(row.get("critical_case")))))
with metric_col_4:
    st.metric("Choix préventifs", int(sum(1 for row in all_planned_rows if "préventive" in str(row.get("maintenance_choice", "")).lower())))


page_tabs = st.tabs([
    "Vue prioritaire",
    "Planning moderne",
    "Commentaires",
    "Exports",
])

with page_tabs[0]:
    st.subheader("Choix de maintenance et priorités")
    render_priority_cards(all_planned_rows, max_cards=10)

    if all_planned_rows:
        ranking_dataframe = pd.DataFrame(all_planned_rows)
        selected_columns = [
            "equipment_code",
            "maintenance_choice",
            "maintenance_choice_reason",
            "days_left",
            "priority_level",
            "critical_case",
            "final_decision",
        ]
        selected_columns = [column for column in selected_columns if column in ranking_dataframe.columns]
        st.markdown('<div class="maint-section-title">Synthèse hiérarchisée</div>', unsafe_allow_html=True)
        st.dataframe(
            style_rows(ranking_dataframe[selected_columns]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aucune recommandation disponible.")

with page_tabs[1]:
    st.subheader("Planning complet")
    if not all_planned_rows:
        st.warning("Aucune ligne exploitable.")
    else:
        full_planning_dataframe = pd.DataFrame(all_planned_rows)
        selected_columns = [
            "equipment_code",
            "maintenance_choice",
            "interval_source_readable",
            "interval_h",
            "periodicity_days",
            "next_due_date",
            "days_left",
            "beta",
            "eta_h",
            "MTTF_h",
            "MTBF_h",
            "MTTR_h",
            "availability_pct",
            "model",
            "process_variant",
            "distribution",
            "priority_score",
            "priority_level",
            "critical_case",
            "final_decision",
        ]
        selected_columns = [column for column in selected_columns if column in full_planning_dataframe.columns]
        st.dataframe(
            style_rows(full_planning_dataframe[selected_columns]),
            use_container_width=True,
            hide_index=True,
        )

with page_tabs[2]:
    st.subheader("Commentaires et justification détaillée")
    if all_planned_rows:
        comments_dataframe = pd.DataFrame(all_planned_rows)
        selected_columns = [
            "equipment_code",
            "maintenance_choice",
            "maintenance_choice_reason",
            "maintenance_comment",
            "priority_level",
            "critical_case",
        ]
        selected_columns = [column for column in selected_columns if column in comments_dataframe.columns]
        st.dataframe(
            style_rows(comments_dataframe[selected_columns]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aucun commentaire disponible.")

with page_tabs[3]:
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
            metrics_table = pd.DataFrame(all_planned_rows).to_dict("records") if all_planned_rows else []
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
