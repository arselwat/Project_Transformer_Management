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
from core.ui import render_shell, render_page_header, render_paper_table

try:
    from core.datahub import (
        get_current_failures_df,
        get_failures_meta,
        get_current_project_data,
        get_project_meta,
    )
except Exception:
    get_current_failures_df = None
    get_failures_meta = None
    get_current_project_data = None
    get_project_meta = None

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
    "Traçabilité complète : tests, modèle, thermique, optimisation, maintenance et décision.",
    "📋",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FAILURES_CSV = DATA_DIR / "failures_saved.csv"
OPTIM_CSV = DATA_DIR / "last_optimization.csv"


def _read_csv_flex(src) -> pd.DataFrame:
    def _try_read(s, **kw):
        try:
            return pd.read_csv(s, **kw)
        except Exception:
            return None

    df = _try_read(src)
    if df is None:
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        df = _try_read(src, engine="python", on_bad_lines="skip", sep=None)
    if df is None:
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        df = _try_read(src, sep=";", engine="python", on_bad_lines="skip")
    if df is None:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _fmt(x: Any, nd: int = 2, default: str = "—") -> str:
    v = _safe_float(x, None)
    return default if v is None else f"{v:.{nd}f}"


def _load_failures_df(uploaded_csv=None) -> pd.DataFrame:
    if uploaded_csv is not None:
        df = _read_csv_flex(uploaded_csv)
    elif callable(get_current_failures_df):
        try:
            df = get_current_failures_df()
        except Exception:
            df = pd.DataFrame()
    elif FAILURES_CSV.exists():
        df = _read_csv_flex(FAILURES_CSV)
    else:
        df = pd.DataFrame()

    if df.empty:
        return df

    df.columns = [str(c).strip() for c in df.columns]
    if "equipment_code" not in df.columns or "ttf_h" not in df.columns:
        return pd.DataFrame()

    df["equipment_code"] = df["equipment_code"].astype(str)
    df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
    if "duree_rep_h" in df.columns:
        df["duree_rep_h"] = pd.to_numeric(df["duree_rep_h"], errors="coerce")
    else:
        df["duree_rep_h"] = np.nan

    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0].reset_index(drop=True)
    return df


def _normalize_thermal_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    aliases = {
        "ambient_temp_c": "temp_amb_C",
        "temp_ambiante_c": "temp_amb_C",
        "temperature_ambiante": "temp_amb_C",
        "fan_status": "etat_ventilateurs",
        "fans_status": "etat_ventilateurs",
        "ventilateurs": "etat_ventilateurs",
        "load_pct": "charge_pct",
    }
    lower = {c.lower().strip(): c for c in out.columns}
    ren = {}
    for k, v in aliases.items():
        if k in lower and v not in out.columns:
            ren[lower[k]] = v
    out = out.rename(columns=ren)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"])

    if "K" not in out.columns:
        if "load_factor" in out.columns:
            out["K"] = pd.to_numeric(out["load_factor"], errors="coerce")
        elif "charge_pct" in out.columns:
            out["K"] = pd.to_numeric(out["charge_pct"], errors="coerce") / 100.0
        elif "load_mva" in out.columns:
            out["K"] = pd.to_numeric(out["load_mva"], errors="coerce") / 100.0

    if "etat_ventilateurs" not in out.columns:
        out["etat_ventilateurs"] = 0

    for c in ["temp_amb_C", "K", "etat_ventilateurs", "top_oil_temp_c", "hotspot_temp_c"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.reset_index(drop=True)


def _normalize_thermal_params(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    aliases = {
        "delta_theta_to_r": "delta_to_r",
        "delta_theta_h_r": "delta_h_r",
        "tau_to_hours": "tau_to_hours",
        "tau_h_hours": "tau_h_hours",
        "normal_life_hours": "normal_insulation_life_h",
        "rated_power_mva": "sn_mva",
    }
    lower = {c.lower().strip(): c for c in out.columns}
    ren = {}
    for k, v in aliases.items():
        if k in lower and v not in out.columns:
            ren[lower[k]] = v
    out = out.rename(columns=ren)

    for c in out.columns:
        if c != "asset_id":
            out[c] = pd.to_numeric(out[c], errors="ignore")

    if "tau_to_hours" in out.columns and "tau_to_min" not in out.columns:
        out["tau_to_min"] = pd.to_numeric(out["tau_to_hours"], errors="coerce") * 60.0
    if "tau_h_hours" in out.columns and "tau_w_min" not in out.columns:
        out["tau_w_min"] = pd.to_numeric(out["tau_h_hours"], errors="coerce") * 60.0

    return out


def _load_project_context(uploaded_xlsx=None) -> Dict[str, pd.DataFrame]:
    if uploaded_xlsx is not None:
        try:
            raw = uploaded_xlsx.read()
            xls = pd.ExcelFile(io.BytesIO(raw))
            sheets = {sheet: pd.read_excel(io.BytesIO(raw), sheet_name=sheet) for sheet in xls.sheet_names}
        except Exception:
            sheets = {}
    elif callable(get_current_project_data):
        try:
            sheets = get_current_project_data() or {}
        except Exception:
            sheets = {}
    else:
        sheets = {}

    out: Dict[str, pd.DataFrame] = {}
    for k, v in sheets.items():
        if isinstance(v, pd.DataFrame):
            out[str(k).strip()] = v.copy()

    if "thermal_timeseries" in out:
        out["thermal_timeseries"] = _normalize_thermal_timeseries(out["thermal_timeseries"])
    if "thermal_params" in out:
        out["thermal_params"] = _normalize_thermal_params(out["thermal_params"])

    return out


def _extract_thermal_for_eq(sheets: Dict[str, pd.DataFrame], eq: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if not sheets:
        return None, None

    thermal_df = None
    thermal_cfg = None

    ts = sheets.get("thermal_timeseries")
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        tmp = ts.copy()
        asset_col = "asset_id" if "asset_id" in tmp.columns else None
        if asset_col:
            tmp = tmp[tmp[asset_col].astype(str) == str(eq)]
        if not tmp.empty:
            thermal_df = tmp.reset_index(drop=True)

    params = sheets.get("thermal_params")
    if isinstance(params, pd.DataFrame) and not params.empty:
        tmp = params.copy()
        asset_col = "asset_id" if "asset_id" in tmp.columns else None
        if asset_col:
            tmp = tmp[tmp[asset_col].astype(str) == str(eq)]
        if not tmp.empty:
            r = tmp.iloc[0].to_dict()
            thermal_cfg = {
                "sn_mva": _safe_float(r.get("sn_mva"), 100.0) or 100.0,
                "R": _safe_float(r.get("R"), 5.0) or 5.0,
                "delta_to_r": _safe_float(r.get("delta_to_r"), 55.0) or 55.0,
                "delta_h_r": _safe_float(r.get("delta_h_r"), 30.0) or 30.0,
                "tau_to_min": _safe_float(r.get("tau_to_min"), 180.0) or 180.0,
                "tau_w_min": _safe_float(r.get("tau_w_min"), 10.0) or 10.0,
                "n_exp": _safe_float(r.get("n_exp"), 0.8) or 0.8,
                "m_exp": _safe_float(r.get("m_exp"), 0.8) or 0.8,
                "forced_tau_to_factor": _safe_float(r.get("forced_tau_to_factor"), 0.75) or 0.75,
                "forced_delta_to_factor": _safe_float(r.get("forced_delta_to_factor"), 0.92) or 0.92,
                "forced_delta_h_factor": _safe_float(r.get("forced_delta_h_factor"), 0.92) or 0.92,
                "normal_insulation_life_h": _safe_float(r.get("normal_insulation_life_h"), 180000.0) or 180000.0,
            }

    return thermal_df, thermal_cfg


def _load_optimization_df() -> pd.DataFrame:
    df = st.session_state.get("optimization_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
    elif OPTIM_CSV.exists():
        out = _read_csv_flex(OPTIM_CSV)
    else:
        out = pd.DataFrame()

    if out.empty:
        return out

    out.columns = [str(c).strip() for c in out.columns]
    if "process_model" in out.columns and "model" not in out.columns:
        out["model"] = out["process_model"]
    if "beta_pipe" in out.columns and "beta" not in out.columns:
        out["beta"] = out["beta_pipe"]
    if "eta_pipe_h" in out.columns and "eta_h" not in out.columns:
        out["eta_h"] = out["eta_pipe_h"]
    if "gamma_pipe_h" in out.columns and "gamma_h" not in out.columns:
        out["gamma_h"] = out["gamma_pipe_h"]
    return out


def _build_virtual_pm_plan_from_optimization(opt_df: pd.DataFrame, start_date: date, within_days: int):
    if opt_df is None or opt_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, r in opt_df.iterrows():
        eq = str(r.get("equipment_code", "")).strip()
        if not eq:
            continue

        interval_h = None
        src = None
        for c in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
            v = _safe_float(r.get(c), None)
            if v is not None and v > 0:
                interval_h = v
                src = c
                break

        if interval_h is None:
            continue

        periodicity_days = max(1, int(round(interval_h / 24.0)))
        next_due = start_date + timedelta(days=periodicity_days)
        days_left = int((next_due - start_date).days)

        rows.append(
            {
                "equipment_code": eq,
                "maintenance_type": r.get("maintenance_type"),
                "interval_source": src,
                "interval_h": interval_h,
                "next_due_date": next_due.isoformat(),
                "days_left": days_left,
            }
        )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df, all_df

    due_df = all_df[all_df["days_left"] <= int(within_days)].copy()
    return all_df.reset_index(drop=True), due_df.reset_index(drop=True)


def _thermal_status(theta_hs_max: Any, faa_max: Any, lol_pct: Any) -> str:
    th = _safe_float(theta_hs_max, None)
    faa = _safe_float(faa_max, None)
    lol = _safe_float(lol_pct, None)

    if (th is not None and th >= 130) or (faa is not None and faa >= 2.0) or (lol is not None and lol >= 1.0):
        return "Critique"
    if (th is not None and th >= 110) or (faa is not None and faa >= 1.5) or (lol is not None and lol >= 0.3):
        return "Alerte"
    if th is None and faa is None and lol is None:
        return "Non disponible"
    return "Normal"


def _process_score(model: str) -> int:
    m = (model or "").upper()
    if "NHPP" in m:
        return 3
    if "BPP" in m or "HAWKES" in m:
        return 2
    return 1


def _final_decision_row(row: pd.Series):
    score = 0
    model = str(row.get("model", "RP"))
    thermal_status = str(row.get("thermal_status", "Non disponible"))
    beta = _safe_float(row.get("beta"), None)
    days_left = _safe_float(row.get("days_left"), None)

    score += _process_score(model)

    if beta is not None:
        if beta > 1.2:
            score += 3
        elif beta >= 1.0:
            score += 2
        elif beta < 0.8:
            score += 1

    if thermal_status == "Critique":
        score += 4
    elif thermal_status == "Alerte":
        score += 2

    if days_left is not None:
        if days_left <= 7:
            score += 3
        elif days_left <= 30:
            score += 2
        elif days_left <= 90:
            score += 1

    maint_type = str(row.get("maintenance_type", ""))

    if score >= 10:
        decision = "Intervention prioritaire"
        reason = f"Processus {model}, état global défavorable et échéance proche. Action : {maint_type or 'maintenance ciblée'}."
    elif score >= 7:
        decision = "Préventif renforcé"
        reason = "Risque significatif détecté. Renforcer la surveillance et planifier l’intervention."
    elif score >= 4:
        decision = "Surveillance active"
        reason = "Situation intermédiaire. Conserver le plan optimisé et suivre les dérives."
    else:
        decision = "Suivi nominal"
        reason = "Pas de signal critique immédiat. Application du plan standard."

    return decision, reason, int(score)


def _trace_trend_table(result: Dict[str, Any], alpha: float) -> pd.DataFrame:
    tests = (result.get("reliability", {}) or {}).get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}

    rows = [
        {
            "Test": "Mann-Kendall",
            "Statistique": _fmt(mk.get("z"), 3),
            "p-valeur": _fmt(mk.get("p"), 4),
            "Règle": f"p < {alpha:.3f}",
            "Décision": "Tendance détectée" if mk.get("has_trend") else "Pas de tendance",
            "Interprétation": (
                f"Tendance {mk.get('direction', 'non précisée')}"
                if mk.get("has_trend") else "Le test ne met pas en évidence de tendance significative."
            ),
        },
        {
            "Test": "Laplace",
            "Statistique": _fmt(lap.get("z"), 3),
            "p-valeur": _fmt(lap.get("p"), 4),
            "Règle": f"p < {alpha:.3f}",
            "Décision": "Tendance détectée" if lap.get("has_trend") else "Pas de tendance",
            "Interprétation": (
                f"Tendance {lap.get('direction', 'non précisée')}"
                if lap.get("has_trend") else "Le test confirme une évolution non significative."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _trace_dependence_table(result: Dict[str, Any], alpha: float) -> pd.DataFrame:
    dep = ((result.get("reliability", {}) or {}).get("tests", {}) or {}).get("dependence", {}) or {}

    rho = dep.get("spearman_r")
    rho_txt = _fmt(rho, 3)
    rho_int = "Corrélation positive" if _safe_float(rho, 0) > 0 else "Corrélation négative ou nulle"

    tau = dep.get("kendall_tau")
    tau_txt = _fmt(tau, 3)
    tau_int = "Concordance positive" if _safe_float(tau, 0) > 0 else "Concordance négative ou nulle"

    rows = [
        {
            "Test": "Spearman",
            "Coefficient": rho_txt,
            "p-valeur": _fmt(dep.get("spearman_p"), 4),
            "Règle": f"p < {alpha:.3f}",
            "Décision": "Dépendance détectée" if dep.get("has_dep") else "Pas de dépendance significative",
            "Interprétation": rho_int,
        },
        {
            "Test": "Kendall",
            "Coefficient": tau_txt,
            "p-valeur": _fmt(dep.get("kendall_p"), 4),
            "Règle": f"p < {alpha:.3f}",
            "Décision": "Appui sur la dépendance" if dep.get("has_dep") else "Appui sur l’indépendance",
            "Interprétation": tau_int,
        },
    ]
    return pd.DataFrame(rows)


def _trace_model_table(result: Dict[str, Any]) -> pd.DataFrame:
    rel = result.get("reliability", {}) or {}
    good = rel.get("goodness", {}) or {}
    dec = rel.get("decision", {}) or {}

    rows = [
        {"Élément": "Processus retenu", "Valeur": rel.get("model", "—"), "Lecture": "Sortie finale du pipeline."},
        {"Élément": "Loi retenue", "Valeur": rel.get("distribution", "—"), "Lecture": "Loi choisie après ajustement."},
        {"Élément": "AIC", "Valeur": _fmt(good.get("aic"), 3), "Lecture": "Plus faible = meilleur compromis ajustement / complexité."},
        {"Élément": "KS p-valeur", "Valeur": _fmt(good.get("ks_p"), 4), "Lecture": "Plus la p-valeur est élevée, plus l’ajustement est acceptable."},
        {"Élément": "Chi² p-valeur", "Valeur": _fmt(good.get("chi2_p"), 4), "Lecture": "Validation complémentaire de l’ajustement."},
        {"Élément": "Motif de décision", "Valeur": dec.get("reason", "—"), "Lecture": "Justification textuelle du pipeline."},
    ]
    return pd.DataFrame(rows)


def _trace_parameter_table(row: Dict[str, Any]) -> pd.DataFrame:
    beta = _safe_float(row.get("beta"), None)
    beta_lecture = "Usure" if beta is not None and beta > 1 else "Aléatoire" if beta is not None and beta >= 0.9 else "Défauts précoces"

    rows = [
        {"Paramètre": "β", "Valeur": _fmt(row.get("beta"), 3), "Interprétation": beta_lecture},
        {"Paramètre": "η (h)", "Valeur": _fmt(row.get("eta_h"), 1), "Interprétation": "Échelle de durée de vie."},
        {"Paramètre": "γ (h)", "Valeur": _fmt(row.get("gamma_h"), 1), "Interprétation": "Décalage éventuel du modèle."},
        {"Paramètre": "MTBF (h)", "Valeur": _fmt(row.get("mtbf_h"), 1), "Interprétation": "Temps moyen entre défaillances."},
        {"Paramètre": "MTTR (h)", "Valeur": _fmt(row.get("mttr_h"), 1), "Interprétation": "Temps moyen de réparation."},
        {"Paramètre": "Disponibilité (%)", "Valeur": _fmt(row.get("availability_pct"), 2), "Interprétation": "Part du temps où l’équipement est disponible."},
        {"Paramètre": "θHS max (°C)", "Valeur": _fmt(row.get("theta_hs_max"), 2), "Interprétation": "Température maximale du point chaud."},
        {"Paramètre": "FAA max", "Valeur": _fmt(row.get("faa_max"), 3), "Interprétation": "Accélération maximale du vieillissement."},
        {"Paramètre": "Perte de vie (%)", "Valeur": _fmt(row.get("loss_of_life_pct"), 3), "Interprétation": "Consommation estimée de la vie d’isolation."},
    ]
    return pd.DataFrame(rows)


def _trace_optimization_table(row: Dict[str, Any]) -> pd.DataFrame:
    t_rec = _safe_float(row.get("T_recommended_h"), None)
    t_r = _safe_float(row.get("T_R_h"), None)
    t_cost = _safe_float(row.get("T_cost_h"), None)

    if t_rec is not None and t_r is not None and abs(t_rec - t_r) < 1e-6:
        lecture = "Le temps recommandé suit le critère fiabilité."
    elif t_rec is not None and t_cost is not None and abs(t_rec - t_cost) < 1e-6:
        lecture = "Le temps recommandé suit le critère économique."
    else:
        lecture = "Le temps recommandé est un compromis entre coût et fiabilité."

    rows = [
        {"Indicateur": "T_R (h)", "Valeur": _fmt(row.get("T_R_h"), 1), "Lecture": "Intervalle issu du critère de fiabilité cible."},
        {"Indicateur": "T_cost (h)", "Valeur": _fmt(row.get("T_cost_h"), 1), "Lecture": "Intervalle minimisant le coût moyen."},
        {"Indicateur": "T_recommended (h)", "Valeur": _fmt(row.get("T_recommended_h"), 1), "Lecture": lecture},
        {"Indicateur": "R(T_cost)", "Valeur": _fmt(row.get("R(T_cost)"), 3), "Lecture": "Fiabilité au temps économique."},
        {"Indicateur": "C_min / h", "Valeur": _fmt(row.get("C_min_per_h"), 4), "Lecture": "Coût minimal moyen par heure."},
        {"Indicateur": "Maintenance retenue", "Valeur": row.get("maintenance_type", "—"), "Lecture": "Type d’action recommandé."},
        {"Indicateur": "Échéance", "Valeur": row.get("next_due_date", "—"), "Lecture": f"J-{row.get('days_left', '—')}"},
    ]
    return pd.DataFrame(rows)


def _trace_decision_table(row: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"Critère": "Tendance", "Valeur": row.get("trend_detected", "—"), "Impact": "Augmente la vigilance si Oui."},
        {"Critère": "Dépendance", "Valeur": row.get("dependence_detected", "—"), "Impact": "Oriente vers processus plus complexe."},
        {"Critère": "Processus", "Valeur": row.get("model", "—"), "Impact": "Modifie le niveau de priorité."},
        {"Critère": "Statut thermique", "Valeur": row.get("thermal_status", "—"), "Impact": "Peut imposer une action plus rapide."},
        {"Critère": "Échéance", "Valeur": f"J-{row.get('days_left', '—')}", "Impact": "Plus l’échéance est proche, plus le score augmente."},
        {"Critère": "Score final", "Valeur": row.get("priority_score", "—"), "Impact": row.get("priorite", "—")},
        {"Critère": "Décision finale", "Valeur": row.get("decision_finale", "—"), "Impact": row.get("motif_decision", "—")},
    ]
    return pd.DataFrame(rows)


def _xlsx_bytes(global_tables: Dict[str, pd.DataFrame], detail_tables: Dict[str, Dict[str, pd.DataFrame]]) -> bytes:
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        for name, df in global_tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
        for eq, tables in detail_tables.items():
            for tname, df in tables.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        df.to_excel(writer, sheet_name=f"{eq}_{tname}"[:31], index=False)
                    except Exception:
                        pass
    buff.seek(0)
    return buff.getvalue()


with st.sidebar:
    st.markdown("### Options")
    up_fail = st.file_uploader("TTF CSV optionnel", type=["csv"])
    up_project = st.file_uploader("Projet Excel optionnel", type=["xlsx"])
    alpha = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
    within_days = st.slider("Fenêtre maintenance (jours)", 7, 365, 30, 1)
    start_dt = st.date_input("Date de référence", value=date.today())


df_fail = _load_failures_df(up_fail)
if df_fail.empty:
    st.error("Aucun dataset TTF disponible.")
    st.stop()

project_sheets = _load_project_context(up_project)
opt_df = _load_optimization_df()

pm_all = st.session_state.get("pm_virtual_all")
pm_due = st.session_state.get("pm_virtual_due")
if isinstance(pm_all, list) and isinstance(pm_due, list):
    pm_all_df = pd.DataFrame(pm_all)
    pm_due_df = pd.DataFrame(pm_due)
else:
    pm_all_df, pm_due_df = _build_virtual_pm_plan_from_optimization(opt_df, start_dt, within_days)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="status-box">TTF actifs : {len(df_fail)} lignes</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="status-box">Projet thermique : {"Oui" if bool(project_sheets) else "Non"}</div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="status-box">Optimisation : {"Oui" if not opt_df.empty else "Non"}</div>', unsafe_allow_html=True)

with st.spinner("Analyse globale..."):
    results_by_eq: Dict[str, Dict[str, Any]] = {}
    detail_tables_by_eq: Dict[str, Dict[str, pd.DataFrame]] = {}
    global_rows: List[Dict[str, Any]] = []

    eqs = sorted(df_fail["equipment_code"].astype(str).unique().tolist())

    for eq in eqs:
        g = df_fail[df_fail["equipment_code"].astype(str) == str(eq)].copy()
        ttf_series = g["ttf_h"].dropna().astype(float).tolist()
        if len(ttf_series) < 3:
            continue

        repair_series = None
        if "duree_rep_h" in g.columns:
            rr = pd.to_numeric(g["duree_rep_h"], errors="coerce").dropna().tolist()
            repair_series = rr if rr else None

        thermal_df_eq, thermal_cfg_eq = _extract_thermal_for_eq(project_sheets, eq)

        try:
            res = analyze_ttf_pipeline(
                ttf_series=ttf_series,
                alpha=float(alpha),
                repair_series=repair_series,
                thermal_df=thermal_df_eq,
                thermal_config=thermal_cfg_eq,
            )
        except Exception as e:
            st.warning(f"{eq} : analyse impossible ({e})")
            continue

        results_by_eq[eq] = res
        detail_tables_by_eq[eq] = res.get("tables", {}) or {}

        reliability = res.get("reliability", {}) or {}
        indicators = reliability.get("indicators", {}) or {}
        params = reliability.get("params", {}) or {}
        tests = reliability.get("tests", {}) or {}
        thermal = res.get("thermal") or {}
        thermal_summary = (thermal.get("summary") or {}) if isinstance(thermal, dict) else {}

        opt_row = {}
        if isinstance(opt_df, pd.DataFrame) and not opt_df.empty and "equipment_code" in opt_df.columns:
            m = opt_df[opt_df["equipment_code"].astype(str) == str(eq)]
            if not m.empty:
                opt_row = m.iloc[0].to_dict()

        pm_row = {}
        if isinstance(pm_all_df, pd.DataFrame) and not pm_all_df.empty and "equipment_code" in pm_all_df.columns:
            m = pm_all_df[pm_all_df["equipment_code"].astype(str) == str(eq)]
            if not m.empty:
                pm_row = m.iloc[0].to_dict()

        trend_mk = tests.get("trend_mk", {}) or {}
        trend_lap = tests.get("trend_laplace", {}) or {}

        beta_pref = params.get("beta", opt_row.get("beta"))
        eta_pref = params.get("eta", opt_row.get("eta_h"))
        gamma_pref = params.get("gamma", opt_row.get("gamma_h"))

        global_rows.append(
            {
                "equipment_code": eq,
                "n_ttf": len(ttf_series),
                "trend_detected": "Oui" if (trend_mk.get("has_trend") or trend_lap.get("has_trend")) else "Non",
                "trend_direction": trend_mk.get("direction") or trend_lap.get("direction"),
                "dependence_detected": "Oui" if ((tests.get("dependence", {}) or {}).get("has_dep")) else "Non",
                "model": reliability.get("model"),
                "distribution": reliability.get("distribution"),
                "beta": beta_pref,
                "eta_h": eta_pref,
                "gamma_h": gamma_pref,
                "mtbf_h": indicators.get("mtbf_h"),
                "mttr_h": indicators.get("mttr_h"),
                "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
                "theta_hs_max": thermal_summary.get("theta_hs_max"),
                "faa_max": thermal_summary.get("faa_max"),
                "loss_of_life_pct": thermal_summary.get("loss_of_life_pct"),
                "thermal_status": _thermal_status(
                    thermal_summary.get("theta_hs_max"),
                    thermal_summary.get("faa_max"),
                    thermal_summary.get("loss_of_life_pct"),
                ),
                "maintenance_type": opt_row.get("maintenance_type"),
                "T_recommended_h": opt_row.get("T_recommended_h"),
                "T_R_h": opt_row.get("T_R_h"),
                "T_cost_h": opt_row.get("T_cost_h"),
                "R(T_cost)": opt_row.get("R(T_cost)"),
                "C_min_per_h": opt_row.get("C_min_per_h"),
                "next_due_date": pm_row.get("next_due_date"),
                "days_left": pm_row.get("days_left"),
            }
        )

summary_df = pd.DataFrame(global_rows)
if summary_df.empty:
    st.error("Aucun équipement exploitable.")
    st.stop()

final_decisions = summary_df.apply(lambda r: _final_decision_row(r), axis=1)
summary_df[["decision_finale", "motif_decision", "priority_score"]] = pd.DataFrame(final_decisions.tolist(), index=summary_df.index)
summary_df["priorite"] = pd.cut(
    summary_df["priority_score"],
    bins=[-1, 3, 6, 9, 100],
    labels=["Faible", "Modérée", "Élevée", "Critique"],
)
summary_df = summary_df.sort_values(["priority_score", "equipment_code"], ascending=[False, True]).reset_index(drop=True)

trend_overview = summary_df[[
    "equipment_code", "n_ttf", "trend_detected", "trend_direction", "dependence_detected", "model", "distribution"
]].copy()

risk_overview = summary_df[[
    "equipment_code", "beta", "eta_h", "mtbf_h", "mttr_h", "availability_pct",
    "theta_hs_max", "faa_max", "loss_of_life_pct", "thermal_status"
]].copy()

optimization_overview = summary_df[[
    "equipment_code", "maintenance_type", "T_recommended_h", "T_R_h", "T_cost_h",
    "R(T_cost)", "C_min_per_h", "next_due_date", "days_left"
]].copy()

final_decision_df = summary_df[[
    "equipment_code", "model", "distribution", "thermal_status", "maintenance_type",
    "days_left", "priority_score", "priorite", "decision_finale", "motif_decision"
]].copy()

due_tasks_df = pm_due_df.copy() if not pm_due_df.empty else pd.DataFrame()

global_tables = {
    "global_summary": summary_df,
    "trend_overview": trend_overview,
    "risk_overview": risk_overview,
    "optimization_overview": optimization_overview,
    "due_tasks": due_tasks_df,
    "final_decision": final_decision_df,
}

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Équipements analysés", len(summary_df))
with k2:
    st.metric("Priorité critique", int((summary_df["priorite"].astype(str) == "Critique").sum()))
with k3:
    st.metric("Tâches dues", len(due_tasks_df))
with k4:
    st.metric("Thermique critique", int((summary_df["thermal_status"] == "Critique").sum()))
with k5:
    st.metric("NHPP détectés", int((summary_df["model"].astype(str).str.upper() == "NHPP").sum()))

tab1, tab2, tab3 = st.tabs(["Vue synthèse", "Traçabilité par équipement", "Exports"])

with tab1:
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(8, 4))
        vc = summary_df["model"].astype(str).value_counts()
        ax.bar(vc.index.tolist(), vc.values.tolist())
        ax.set_title("Répartition des processus")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with b:
        fig, ax = plt.subplots(figsize=(8, 4))
        tmp = summary_df[["equipment_code", "priority_score"]].sort_values("priority_score")
        ax.barh(tmp["equipment_code"], tmp["priority_score"])
        ax.set_title("Score de priorité")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

with tab2:
    eq = st.selectbox("Choisir un équipement", options=summary_df["equipment_code"].tolist())
    res = results_by_eq[eq]
    row = summary_df[summary_df["equipment_code"] == eq].iloc[0].to_dict()

    render_paper_table("Tableau 1 : Validation des tests de tendance", _trace_trend_table(res, alpha))
    render_paper_table("Tableau 2 : Validation des tests de dépendance", _trace_dependence_table(res, alpha))
    render_paper_table("Tableau 3 : Choix du processus et du modèle", _trace_model_table(res))
    render_paper_table("Tableau 4 : Paramètres fiabilistes et thermiques", _trace_parameter_table(row))
    render_paper_table("Tableau 5 : Optimisation et maintenance retenue", _trace_optimization_table(row))
    render_paper_table("Tableau 6 : Traçabilité de la décision finale", _trace_decision_table(row))

    st.success(f"Décision finale — {row.get('decision_finale', '—')}")
    st.info(row.get("motif_decision", "Aucun motif disponible."))

with tab3:
    excel_bytes = _xlsx_bytes(global_tables, detail_tables_by_eq)
    st.download_button(
        "⬇️ Télécharger le pack Excel global",
        data=excel_bytes,
        file_name="resultat_analyse_optimisation_maintenance.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_global_analysis_report_pdf is None:
        st.info("Module PDF global non détecté.")
    else:
        if st.button("📄 Générer le rapport PDF global", use_container_width=True):
            try:
                pdf_path = export_global_analysis_report_pdf(
                    summary_df=summary_df,
                    global_tables=global_tables,
                    detail_tables_by_eq=detail_tables_by_eq,
                    out_dir=str(BASE_DIR / "reports"),
                    title="Résultat analyse / optimisation / maintenance",
                    meta={
                        "alpha": alpha,
                        "window_days": within_days,
                        "start_date": str(start_dt),
                        "n_equipment": len(summary_df),
                    },
                )
                st.session_state["global_pdf_path"] = pdf_path
                st.success(f"PDF généré : {pdf_path}")
            except Exception as e:
                st.error(f"PDF : {e}")

        pdf_path = st.session_state.get("global_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Télécharger le PDF global",
                    data=f,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )