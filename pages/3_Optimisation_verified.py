
from __future__ import annotations

from pathlib import Path
import io
import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
) -> Optional[float]:
    candidate_values = [
        float(value)
        for value in [reliability_interval_hours, economic_interval_hours]
        if is_positive_number(value)
    ]
    if not candidate_values:
        return None

    maintenance_type_lower = maintenance_type.lower()

    if "corrective" in maintenance_type_lower and "fiabilisation" in maintenance_type_lower:
        return None

    return float(min(candidate_values))


def build_optimization_note(
    characteristic_life_hours: Any,
    economic_interval_hours: Any,
    reliability_interval_hours: Any,
    recommended_interval_hours: Any,
) -> str:
    characteristic_life_value = safe_number(characteristic_life_hours)
    economic_interval_value = safe_number(economic_interval_hours)
    reliability_interval_value = safe_number(reliability_interval_hours)
    recommended_interval_value = safe_number(recommended_interval_hours)

    if recommended_interval_value is None:
        return "Aucun intervalle exploitable n’a pu être retenu automatiquement pour cet équipement."

    parts = [f"Intervalle retenu : {recommended_interval_value:.1f} heures."]

    if reliability_interval_value is not None:
        parts.append(f"Intervalle issu du critère de fiabilité : {reliability_interval_value:.1f} heures.")

    if economic_interval_value is not None:
        parts.append(f"Intervalle issu du critère économique : {economic_interval_value:.1f} heures.")

    if characteristic_life_value is not None:
        parts.append(f"Vie caractéristique estimée : {characteristic_life_value:.1f} heures.")
        ratio = recommended_interval_value / max(characteristic_life_value, 1e-9)
        if ratio < 0.5:
            parts.append("La décision retenue reste prudente par rapport à la vie caractéristique.")
        elif ratio <= 1.0:
            parts.append("La décision retenue reste cohérente avec la vie caractéristique estimée.")
        else:
            parts.append("L’intervalle retenu dépasse la vie caractéristique : une vigilance renforcée est nécessaire.")

    return " ".join(parts)


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
    "maintenance_type": "Type de maintenance recommandé",
    "decision_reason": "Justification de la décision fiabiliste",
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


with st.expander("Comprendre les paramètres d’optimisation", expanded=False):
    st.markdown(
        """
**Paramètres principaux affichés**
- **bêta** : paramètre de forme du modèle retenu par le pipeline final ; il décrit la phase de vie de l’équipement.
- **êta** : paramètre d’échelle ou durée de vie caractéristique du modèle retenu.
- **gamma** : décalage temporel éventuel du modèle.

**Pourquoi il existait aussi une valeur de référence Weibull ?**
Le logiciel calcule en interne un ajustement Weibull de référence pour disposer d’un point d’appui économique, même lorsque le modèle final retenu n’est pas exactement Weibull.
Dans cette page, l’affichage met désormais l’accent sur les **paramètres principaux du modèle retenu** pour éviter la confusion.

**Jours avant maintenance**
Les jours avant maintenance sont simplement obtenus à partir de l’intervalle retenu : **jours = T_recommandé / 24**.
Ce n’est pas la même chose que le **MTBF** :
- le **MTBF** est une moyenne historique ou théorique entre défaillances ;
- les **jours avant maintenance** correspondent à une **échéance de planification** calculée pour la décision actuelle.

**Choix de l’intervalle recommandé**
- on compare **T_R** et **T_cost** ;
- si **R(T_cost) < seuil minimal**, alors **T_R** est retenu automatiquement ;
- sinon, on garde l’intervalle qui laisse le plus de temps avant intervention tout en restant acceptable sur le plan fiabiliste.
        """
    )

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

    recommended_maintenance_type = recommend_maintenance_type(reliability_result)
    recommended_interval_hours = recommend_interval(
        recommended_maintenance_type,
        reliability_interval_hours,
        economic_interval_hours,
    )

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
            "maintenance_type": recommended_maintenance_type,
            "decision_reason": decision.get("reason"),
            "optimization_note": build_optimization_note(
                primary_eta,
                economic_interval_hours,
                reliability_interval_hours,
                recommended_interval_hours,
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

    st.success(
        f"Résultat envoyé vers la future page Maintenance | lignes={len(optimization_dataframe)} | empreinte={dataframe_hash(optimization_dataframe)}"
    )

    if st.button("Enregistrer aussi le résultat dans le fichier de secours", use_container_width=True):
        optimization_dataframe.to_csv(FALLBACK_OPTIMIZATION_FILE, index=False, encoding="utf-8")
        st.success(f"Fichier écrit : {FALLBACK_OPTIMIZATION_FILE}")

with page_tabs[2]:
    st.subheader("Courbes de fiabilité de référence économique")

    eta_values = [float(getattr(fit, "eta", 1.0) or 1.0) for fit in weibull_reference_fits.values()]
    maximum_time = max(eta_values) * 1.6 if eta_values else 1000.0

    interval_candidates = []
    for _, row in optimization_dataframe.iterrows():
        if is_positive_number(row.get("T_R_h")):
            interval_candidates.append(float(row["T_R_h"]))
        if is_positive_number(row.get("T_cost_h")):
            interval_candidates.append(float(row["T_cost_h"]))

    if interval_candidates:
        maximum_time = max(maximum_time, max(interval_candidates) * 1.2)

    time_axis = np.linspace(0, max(maximum_time, 1.0), 350)
    figure, axis = plt.subplots()

    for equipment_code, fit in weibull_reference_fits.items():
        beta_value = float(getattr(fit, "beta", 1.0))
        eta_value = float(getattr(fit, "eta", 1.0))
        gamma_value = float(getattr(fit, "gamma", 0.0) or 0.0)

        reliability_curve = np.ones_like(time_axis, dtype=float)
        mask = time_axis > gamma_value
        reliability_curve[mask] = np.exp(-(((time_axis[mask] - gamma_value) / max(eta_value, 1e-9)) ** max(beta_value, 1e-9)))

        process_name = ((pipeline_results_by_equipment.get(equipment_code, {}) or {}).get("reliability", {}) or {}).get("model", "?")
        process_variant = ((pipeline_results_by_equipment.get(equipment_code, {}) or {}).get("reliability", {}) or {}).get("process_variant", "?")

        axis.plot(
            time_axis,
            reliability_curve,
            linewidth=2,
            label=f"{equipment_code} | {process_name} / {process_variant} | bêta={beta_value:.2f}, êta={eta_value:.1f}",
        )

    axis.grid(True, alpha=0.3)
    axis.set_xlabel("Temps (heures)")
    axis.set_ylabel("Fiabilité")
    axis.set_title("Courbe de fiabilité de référence")
    axis.legend(fontsize=8)
    st.pyplot(figure, clear_figure=True)

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
        f"- **Paramètres principaux du modèle retenu** : **bêta = {format_number(selected_row.get('beta'), 2)}**, "
        f"**êta = {format_number(selected_row.get('eta_h'), 1)} heures**, "
        f"**gamma = {format_number(selected_row.get('gamma_h'), 1)} heures**\n"
        f"- **MTBF** : **{format_number(selected_row.get('MTBF_h'), 1)} heures** ; **MTTR** : **{format_number(selected_row.get('MTTR_h'), 1)} heures** ; **Disponibilité** : **{format_number(selected_row.get('availability_pct'), 2)} %**\n"
        f"- **Intervalle issu du critère économique** : **{format_number(selected_row.get('T_cost_h'), 1)} heures**\n"
        f"- **Intervalle issu du critère de fiabilité** : **{format_number(selected_row.get('T_R_h'), 1)} heures**\n"
        f"- **Intervalle recommandé** : **{format_number(selected_row.get('T_recommended_h'), 1)} heures** = **{format_number(selected_row.get('days_recommended'), 1)} jours**\n"
        f"- **Type de maintenance recommandé** : **{selected_row.get('maintenance_type', '—')}**"
    )

    st.caption(
        "Lecture : le bêta affiché ici est le paramètre principal du modèle retenu par le pipeline. "
        "Les jours avant maintenance proviennent de l’intervalle recommandé converti en jours et ne doivent pas être confondus avec le MTBF, qui reste une moyenne de comportement."
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
