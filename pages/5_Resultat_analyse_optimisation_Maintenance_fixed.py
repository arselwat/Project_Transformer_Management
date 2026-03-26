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

from core.security.auth import require_login
from core.reliability.organigram import analyze_ttf_pipeline
from core.datahub import (
    get_current_failures_df,
    get_failures_meta,
    get_project_meta,
    get_pipeline_inputs,
)

try:
    from core.reliability.reporting_global import export_global_analysis_report_pdf
except Exception:
    export_global_analysis_report_pdf = None


st.set_page_config(
    page_title="Résultat analyse / optimisation / maintenance",
    page_icon="📊",
    layout="wide",
)
require_login()

st.title("📊 Résultat global")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
OPTIM_CSV = DATA_DIR / "last_optimization.csv"


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
    if v is None:
        return default
    return f"{v:.{nd}f}"


def _decision_badge(label: str) -> str:
    s = (label or "").strip().lower()
    if "crit" in s or "prior" in s:
        return "🔴 " + label
    if "alerte" in s or "renfor" in s:
        return "🟠 " + label
    if "surve" in s or "condition" in s:
        return "🟡 " + label
    return "🟢 " + label


def _series_to_list(s: pd.Series) -> Optional[list[float]]:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return None
    return vals.astype(float).tolist()


def _load_optimization_df() -> pd.DataFrame:
    df = st.session_state.get("optimization_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
    elif OPTIM_CSV.exists():
        try:
            out = pd.read_csv(OPTIM_CSV)
        except Exception:
            out = pd.DataFrame()
    else:
        out = pd.DataFrame()

    if out.empty:
        return out

    out.columns = [str(c).strip() for c in out.columns]
    return out


def _build_virtual_pm_plan_from_optimization(
    opt_df: pd.DataFrame,
    start_date: date,
    within_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                "periodicity_days": periodicity_days,
                "next_due_date": next_due.isoformat(),
                "days_left": days_left,
                "T_recommended_h": r.get("T_recommended_h"),
                "T_R_h": r.get("T_R_h"),
                "T_cost_h": r.get("T_cost_h"),
                "model": r.get("model"),
                "distribution": r.get("distribution"),
                "beta": r.get("beta"),
                "eta_h": r.get("eta_h"),
            }
        )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df, all_df

    due_df = all_df[all_df["days_left"] <= int(within_days)].copy()
    return (
        all_df.sort_values(["days_left", "equipment_code"]).reset_index(drop=True),
        due_df.sort_values(["days_left", "equipment_code"]).reset_index(drop=True),
    )


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


def _final_decision_row(row: pd.Series) -> Tuple[str, str, int]:
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
        reason = (
            f"Processus {model}, état défavorable et échéance proche. "
            f"Type recommandé : {maint_type or 'maintenance ciblée'}."
        )
    elif score >= 7:
        decision = "Préventif renforcé"
        reason = "Risque significatif détecté. Renforcer la surveillance et planifier une intervention."
    elif score >= 4:
        decision = "Surveillance active"
        reason = "Situation intermédiaire : conserver le plan optimisé et surveiller les dérives."
    else:
        decision = "Suivi nominal"
        reason = "Pas de signal critique immédiat : appliquer le calendrier recommandé."

    return decision, reason, int(score)


def _style_decision_df(df: pd.DataFrame):
    def _row_style(row):
        d = str(row.get("decision_finale", ""))
        if "prioritaire" in d.lower():
            return ["background-color: #fde2e2"] * len(row)
        if "renforcé" in d.lower():
            return ["background-color: #fff0cc"] * len(row)
        if "surveillance" in d.lower():
            return ["background-color: #fff9cc"] * len(row)
        return ["background-color: #e7f5ea"] * len(row)

    try:
        return df.style.apply(_row_style, axis=1)
    except Exception:
        return df


def _xlsx_bytes(global_tables: Dict[str, pd.DataFrame], detail_tables: Dict[str, Dict[str, pd.DataFrame]]) -> bytes:
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        for name, df in global_tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)

        for eq, tables in detail_tables.items():
            for tname, df in tables.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sheet = f"{eq}_{tname}"[:31]
                    try:
                        df.to_excel(writer, sheet_name=sheet, index=False)
                    except Exception:
                        pass
    buff.seek(0)
    return buff.getvalue()


def _explain_trend(eq_res: Dict[str, Any]) -> str:
    tests = (eq_res.get("reliability", {}) or {}).get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}

    mk_p = _safe_float(mk.get("p"), None)
    lap_p = _safe_float(lap.get("p"), None)
    mk_dir = mk.get("direction", "none")
    lap_dir = lap.get("direction", "none")

    if mk.get("has_trend") or lap.get("has_trend"):
        return (
            f"Tendance détectée. MK p={_fmt(mk_p,3)} ({mk_dir}) ; "
            f"Laplace p={_fmt(lap_p,3)} ({lap_dir})."
        )
    return (
        f"Pas de tendance significative. MK p={_fmt(mk_p,3)} ; "
        f"Laplace p={_fmt(lap_p,3)}."
    )


def _explain_dependence(eq_res: Dict[str, Any]) -> str:
    tests = (eq_res.get("reliability", {}) or {}).get("tests", {}) or {}
    dep = tests.get("dependence", {}) or {}
    r = _safe_float(dep.get("spearman_r"), None)
    p = _safe_float(dep.get("spearman_p"), None)

    if dep.get("has_dep"):
        return f"Dépendance détectée entre événements. Spearman r={_fmt(r,3)}, p={_fmt(p,3)}."
    return f"Pas de dépendance significative. Spearman r={_fmt(r,3)}, p={_fmt(p,3)}."


def _explain_process(eq_res: Dict[str, Any]) -> str:
    rel = eq_res.get("reliability", {}) or {}
    model = rel.get("model", "—")
    dist = rel.get("distribution", "—")
    reason = (rel.get("decision", {}) or {}).get("reason", "—")
    return f"Processus retenu : {model}. Distribution retenue : {dist}. Motif : {reason}"


def _explain_optimization(row: dict) -> str:
    return (
        f"T_recommended={_fmt(row.get('T_recommended_h'),1)} h ; "
        f"T_R={_fmt(row.get('T_R_h'),1)} h ; "
        f"T_cost={_fmt(row.get('T_cost_h'),1)} h ; "
        f"Maintenance={row.get('maintenance_type','—')}."
    )


def _explain_final(row: dict) -> str:
    return (
        f"Décision finale : {row.get('decision_finale','—')} | "
        f"Priorité {row.get('priorite','—')} | "
        f"Score {row.get('priority_score','—')}."
    )


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Options")
    alpha = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
    within_days = st.slider("Fenêtre maintenance (jours)", 7, 365, 30, 1)
    start_dt = st.date_input("Date de référence", value=date.today())


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
meta_fail = get_failures_meta()
meta_proj = get_project_meta()
df_fail = get_current_failures_df()

if df_fail.empty:
    st.error("Aucun dataset TTF disponible. Charge d’abord les sources.")
    st.stop()

df_fail = df_fail.copy()
df_fail.columns = [str(c).strip() for c in df_fail.columns]
df_fail["equipment_code"] = df_fail["equipment_code"].astype(str)
df_fail["ttf_h"] = pd.to_numeric(df_fail["ttf_h"], errors="coerce")
if "duree_rep_h" in df_fail.columns:
    df_fail["duree_rep_h"] = pd.to_numeric(df_fail["duree_rep_h"], errors="coerce")
else:
    df_fail["duree_rep_h"] = np.nan
df_fail = df_fail.dropna(subset=["ttf_h"])
df_fail = df_fail[df_fail["ttf_h"] > 0].reset_index(drop=True)

opt_df = _load_optimization_df()

pm_all = st.session_state.get("pm_virtual_all")
pm_due = st.session_state.get("pm_virtual_due")
if not isinstance(pm_all, list) or not isinstance(pm_due, list):
    pm_all_df, pm_due_df = _build_virtual_pm_plan_from_optimization(opt_df, start_dt, within_days)
else:
    pm_all_df = pd.DataFrame(pm_all)
    pm_due_df = pd.DataFrame(pm_due)

s1, s2, s3 = st.columns(3)
with s1:
    st.success(f"Dataset actif | rows={meta_fail.get('rows')} | hash={meta_fail.get('hash')}")
with s2:
    if meta_proj.get("ok"):
        st.success(f"Projet actif | hash={meta_proj.get('hash', '')}")
    else:
        st.info("Projet non disponible")
with s3:
    if isinstance(opt_df, pd.DataFrame) and not opt_df.empty:
        st.success(f"Optimisation dispo | rows={len(opt_df)}")
    else:
        st.info("Optimisation non disponible")


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------
with st.spinner("Analyse globale en cours..."):
    results_by_eq: Dict[str, Dict[str, Any]] = {}
    detail_tables_by_eq: Dict[str, Dict[str, pd.DataFrame]] = {}
    global_rows: List[Dict[str, Any]] = []

    eqs = sorted(df_fail["equipment_code"].astype(str).unique().tolist())

    for eq in eqs:
        g = df_fail[df_fail["equipment_code"] == eq].copy()
        ttf_series = _series_to_list(g["ttf_h"])
        if not ttf_series or len(ttf_series) < 3:
            continue

        repair_series = None
        rr = _series_to_list(g["duree_rep_h"]) if "duree_rep_h" in g.columns else None
        if rr:
            repair_series = rr

        bundle = get_pipeline_inputs(asset_id=str(eq))
        thermal_df_eq = bundle.get("thermal_df")
        thermal_cfg_eq = bundle.get("thermal_config")

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
        eq_tables = res.get("tables", {}) or {}
        detail_tables_by_eq[eq] = eq_tables

        rel = res.get("reliability", {}) or {}
        indicators = rel.get("indicators", {}) or {}
        params = rel.get("params", {}) or {}
        tests = rel.get("tests", {}) or {}
        thermal = res.get("thermal") or {}
        thermal_summary = (thermal.get("summary") or {}) if isinstance(thermal, dict) else {}

        opt_row = {}
        if isinstance(opt_df, pd.DataFrame) and not opt_df.empty and "equipment_code" in opt_df.columns:
            match = opt_df[opt_df["equipment_code"].astype(str) == str(eq)]
            if not match.empty:
                opt_row = match.iloc[0].to_dict()

        pm_row = {}
        if isinstance(pm_all_df, pd.DataFrame) and not pm_all_df.empty and "equipment_code" in pm_all_df.columns:
            match_pm = pm_all_df[pm_all_df["equipment_code"].astype(str) == str(eq)]
            if not match_pm.empty:
                pm_row = match_pm.iloc[0].to_dict()

        mk = tests.get("trend_mk", {}) or {}
        lap = tests.get("trend_laplace", {}) or {}
        dep = tests.get("dependence", {}) or {}

        beta_main = params.get("beta", opt_row.get("beta"))
        eta_main = params.get("eta", opt_row.get("eta_h"))
        gamma_main = params.get("gamma", opt_row.get("gamma_h"))

        row = {
            "equipment_code": eq,
            "n_ttf": len(ttf_series),
            "trend_detected": "Oui" if (mk.get("has_trend") or lap.get("has_trend")) else "Non",
            "trend_direction": mk.get("direction") or lap.get("direction"),
            "dependence_detected": "Oui" if dep.get("has_dep") else "Non",
            "model": rel.get("model"),
            "distribution": rel.get("distribution"),
            "beta": beta_main,
            "eta_h": eta_main,
            "gamma_h": gamma_main,
            "mtbf_h": indicators.get("mtbf_h"),
            "mttr_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),
            "mean_failure_rate_h": indicators.get("mean_failure_rate_h"),
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
        global_rows.append(row)

summary_df = pd.DataFrame(global_rows)
if summary_df.empty:
    st.error("Aucun équipement exploitable après analyse.")
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

if not pm_due_df.empty:
    due_tasks_df = pm_due_df.copy()
else:
    due_tasks_df = pd.DataFrame(columns=["equipment_code", "maintenance_type", "interval_h", "next_due_date", "days_left"])

global_tables = {
    "global_summary": summary_df,
    "trend_overview": trend_overview,
    "risk_overview": risk_overview,
    "optimization_overview": optimization_overview,
    "due_tasks": due_tasks_df,
    "final_decision": final_decision_df,
}

# -------------------------------------------------------------------
# KPI
# -------------------------------------------------------------------
critical_count = int((summary_df["priorite"].astype(str) == "Critique").sum())
due_count = int(len(due_tasks_df))
thermal_critical = int((summary_df["thermal_status"] == "Critique").sum())
nhpp_count = int((summary_df["model"].astype(str).str.upper() == "NHPP").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Équipements analysés", len(summary_df))
c2.metric("Priorité critique", critical_count)
c3.metric("Tâches dues", due_count)
c4.metric("Thermique critique", thermal_critical)
c5.metric("NHPP détectés", nhpp_count)

st.divider()

tabs = st.tabs(
    [
        "Vue globale",
        "Traçabilité",
        "Thermique",
        "Optimisation & maintenance",
        "Décision finale",
        "Exports",
    ]
)

with tabs[0]:
    st.subheader("Tableau de synthèse global")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(8, 4))
        proc = summary_df["model"].astype(str).value_counts()
        ax.bar(proc.index.tolist(), proc.values.tolist())
        ax.set_title("Répartition des processus retenus")
        ax.set_xlabel("Processus")
        ax.set_ylabel("Nombre d’équipements")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with cb:
        fig, ax = plt.subplots(figsize=(8, 4))
        dplot = summary_df[["equipment_code", "priority_score"]].sort_values("priority_score", ascending=True)
        ax.barh(dplot["equipment_code"], dplot["priority_score"])
        ax.set_title("Score de priorité globale")
        ax.set_xlabel("Score")
        ax.set_ylabel("Équipement")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

with tabs[1]:
    eq = st.selectbox("Équipement", options=summary_df["equipment_code"].tolist(), key="global_trace_eq")
    eq_res = results_by_eq[eq]
    eq_tables = detail_tables_by_eq[eq]
    row = summary_df[summary_df["equipment_code"] == eq].iloc[0].to_dict()

    st.markdown(f"### Traçabilité de calcul — {eq}")

    st.markdown("#### 1. Tendance")
    st.dataframe(eq_tables.get("trend_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.info(_explain_trend(eq_res))

    st.markdown("#### 2. Dépendance")
    st.dataframe(eq_tables.get("dependence_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.info(_explain_dependence(eq_res))

    st.markdown("#### 3. Choix du processus")
    st.dataframe(eq_tables.get("process_choice", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.info(_explain_process(eq_res))

    st.markdown("#### 4. Ajustement / fiabilité")
    st.dataframe(eq_tables.get("fit_candidates", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.dataframe(eq_tables.get("reliability_summary", pd.DataFrame()), use_container_width=True, hide_index=True)

    st.markdown("#### 5. Optimisation / maintenance")
    st.write(_explain_optimization(row))

    st.markdown("#### 6. Décision finale")
    st.write(_explain_final(row))
    st.info(row.get("motif_decision", "Aucun motif disponible."))

with tabs[2]:
    eq = st.selectbox("Équipement thermique", options=summary_df["equipment_code"].tolist(), key="global_eq_thermal")
    eq_res = results_by_eq[eq]
    eq_tables = detail_tables_by_eq[eq]
    thermal = eq_res.get("thermal")

    if not thermal:
        st.warning("Aucune donnée thermique disponible pour cet équipement.")
    else:
        for key in ["thermal_table_dataset", "thermal_table_params", "thermal_table_indicators", "thermal_summary", "thermal_top5_days"]:
            dfx = eq_tables.get(key, pd.DataFrame())
            if isinstance(dfx, pd.DataFrame) and not dfx.empty:
                st.dataframe(dfx, use_container_width=True, hide_index=True)

        ts = thermal.get("timeseries")
        if isinstance(ts, pd.DataFrame) and not ts.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(ts["timestamp"]), ts["theta_HS_est_C"])
                ax.set_title(f"θHS estimée — {eq}")
                ax.set_xlabel("Temps")
                ax.set_ylabel("°C")
                ax.grid(True, alpha=0.25)
                st.pyplot(fig, clear_figure=True)
            with c2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(ts["timestamp"]), ts["FAA"])
                ax.set_title(f"FAA — {eq}")
                ax.set_xlabel("Temps")
                ax.set_ylabel("p.u.")
                ax.grid(True, alpha=0.25)
                st.pyplot(fig, clear_figure=True)

with tabs[3]:
    st.subheader("Optimisation")
    if opt_df.empty:
        st.info("Aucune optimisation disponible pour le moment.")
    else:
        st.dataframe(optimization_overview, use_container_width=True, hide_index=True)

    st.subheader("Tâches de maintenance dues")
    if due_tasks_df.empty:
        st.info("Aucune tâche due dans la fenêtre sélectionnée.")
    else:
        st.dataframe(due_tasks_df, use_container_width=True, hide_index=True)

    if not opt_df.empty and "T_recommended_h" in opt_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = opt_df[["equipment_code", "T_recommended_h"]].copy()
        plot_df["T_recommended_h"] = pd.to_numeric(plot_df["T_recommended_h"], errors="coerce")
        plot_df = plot_df.dropna().sort_values("T_recommended_h", ascending=True)
        if not plot_df.empty:
            ax.barh(plot_df["equipment_code"], plot_df["T_recommended_h"])
            ax.set_title("Intervalles recommandés")
            ax.set_xlabel("Heures")
            ax.set_ylabel("Équipement")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig, clear_figure=True)

with tabs[4]:
    st.subheader("Décision finale hiérarchisée")
    st.dataframe(_style_decision_df(final_decision_df), use_container_width=True, hide_index=True)

    eq = st.selectbox("Équipement — décision finale", options=summary_df["equipment_code"].tolist(), key="global_eq_decision")
    row = summary_df[summary_df["equipment_code"] == eq].iloc[0].to_dict()

    st.markdown(f"### {_decision_badge(str(row['decision_finale']))} — {eq}")
    st.markdown(
        f"**Processus retenu** : {row.get('model','—')}  \n"
        f"**Distribution** : {row.get('distribution','—')}  \n"
        f"**Beta / Eta / Gamma** : {_fmt(row.get('beta'),2)} / {_fmt(row.get('eta_h'),1)} h / {_fmt(row.get('gamma_h'),1)} h  \n"
        f"**Statut thermique** : {row.get('thermal_status','—')}  \n"
        f"**Maintenance recommandée** : {row.get('maintenance_type','—')}  \n"
        f"**Échéance** : {row.get('next_due_date','—')} (J-{row.get('days_left','—')})  \n"
        f"**Score** : {row.get('priority_score','—')} ({row.get('priorite','—')})"
    )
    st.info(row.get("motif_decision", "Aucun motif disponible."))

    st.markdown("#### Chemin de décision")
    st.markdown(
        f"1. **Tendance** → {row.get('trend_detected','—')} ({row.get('trend_direction','—')})\n"
        f"2. **Dépendance** → {row.get('dependence_detected','—')}\n"
        f"3. **Processus** → {row.get('model','—')}\n"
        f"4. **Distribution** → {row.get('distribution','—')}\n"
        f"5. **Thermique** → {row.get('thermal_status','—')}\n"
        f"6. **Optimisation / maintenance** → {row.get('maintenance_type','—')} / T_recommended={_fmt(row.get('T_recommended_h'),1)} h\n"
        f"7. **Décision finale** → {row.get('decision_finale','—')}"
    )

with tabs[5]:
    st.subheader("Exports")

    excel_bytes = _xlsx_bytes(global_tables=global_tables, detail_tables=detail_tables_by_eq)
    st.download_button(
        "Télécharger le pack Excel global",
        data=excel_bytes,
        file_name="resultat_analyse_optimisation_maintenance.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_global_analysis_report_pdf is None:
        st.info("Module PDF global non détecté.")
    else:
        if st.button("Générer le rapport PDF global", use_container_width=True):
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
                        "source_rows": len(df_fail),
                        "source_hash": meta_fail.get("hash", ""),
                        "project_hash": meta_proj.get("hash", ""),
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
                    "Télécharger le PDF global",
                    data=f,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )