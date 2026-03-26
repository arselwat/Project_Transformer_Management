from __future__ import annotations

from pathlib import Path
import io
import math
import hashlib
from typing import Any, Optional, Dict, List

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
from core.datahub import (
    get_current_failures_df,
    get_failures_meta,
    get_project_meta,
    get_pipeline_inputs,
)

export_optimization_report_pdf = None
_pdf_import_error = None
try:
    from core.reliability.reporting_optimize import export_optimization_report_pdf as _export_opt_pdf
    export_optimization_report_pdf = _export_opt_pdf
except Exception as e:
    _pdf_import_error = str(e)
    export_optimization_report_pdf = None


st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()

st.title("🧠 Optimisation")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _safe_num(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _fmt(x: Any, nd: int = 2, default: str = "—") -> str:
    v = _safe_num(x)
    if v is None:
        return default
    return f"{v:.{nd}f}"


def _series_to_list(s: pd.Series) -> Optional[list[float]]:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return None
    return vals.astype(float).tolist()


def _is_pos_number(x: Any) -> bool:
    v = _safe_num(x)
    return v is not None and v > 0


def _df_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    return hashlib.md5(df.to_csv(index=False).encode("utf-8")).hexdigest()


def _thermal_status_label(
    thermal_result: Optional[dict],
    faa_limit: Optional[float],
    lol_limit_pct: Optional[float],
) -> tuple[Optional[bool], str, Optional[float], Optional[float], Optional[float]]:
    if not thermal_result:
        return None, "Pas de données thermiques", None, None, None

    summary = thermal_result.get("summary", {}) or {}
    theta_hs_max = _safe_num(summary.get("theta_hs_max"))
    faa_max = _safe_num(summary.get("faa_max"))
    lol_pct = _safe_num(summary.get("loss_of_life_pct"))

    checks = []
    if faa_limit is not None and faa_max is not None:
        checks.append(faa_max <= faa_limit)
    if lol_limit_pct is not None and lol_pct is not None:
        checks.append(lol_pct <= lol_limit_pct)

    if not checks:
        return None, "Thermique calculée", theta_hs_max, faa_max, lol_pct
    if all(checks):
        return True, "Conforme thermique", theta_hs_max, faa_max, lol_pct
    return False, "Alerte thermique", theta_hs_max, faa_max, lol_pct


def _recommend_maintenance(beta: float, model: Optional[str], thermal_status: str) -> str:
    model_s = (model or "").upper()

    if "ALERTE" in thermal_status.upper():
        return "Préventive immédiate + contrôle thermique"
    if "NHPP" in model_s:
        return "Préventive planifiée"
    if "BPP" in model_s or "HAWKES" in model_s:
        return "Conditionnelle / inspection renforcée"
    if beta < 0.9:
        return "Corrective + fiabilisation"
    if beta <= 1.1:
        return "Conditionnelle / inspection"
    return "Préventive planifiée"


def _recommend_interval(beta: float, model: Optional[str], t_cost: Any, t_r: Any) -> Optional[float]:
    model_s = (model or "").upper()
    vals = [float(v) for v in [t_cost, t_r] if _is_pos_number(v)]
    if not vals:
        return None
    if "BPP" in model_s or "HAWKES" in model_s:
        return float(min(vals))
    if beta <= 1.1 and "NHPP" not in model_s:
        return None
    return float(min(vals))


def _optimization_note(eta_h: Any, t_cost: Any, t_r: Any, t_rec: Any) -> str:
    eta_v = _safe_num(eta_h)
    cost_v = _safe_num(t_cost)
    r_v = _safe_num(t_r)
    rec_v = _safe_num(t_rec)

    if rec_v is None:
        return "Aucun intervalle recommandé calculé."

    parts = [f"Intervalle retenu = {rec_v:.1f} h."]
    if r_v is not None:
        parts.append(f"T_R = {r_v:.1f} h.")
    if cost_v is not None:
        parts.append(f"T_cost = {cost_v:.1f} h.")
    if eta_v is not None:
        parts.append(f"η = {eta_v:.1f} h.")

        ratio = rec_v / max(eta_v, 1e-9)
        if ratio < 0.5:
            parts.append("L’intervalle retenu est conservateur par rapport à η.")
        elif ratio <= 1.0:
            parts.append("L’intervalle retenu reste dans une zone acceptable par rapport à η.")
        else:
            parts.append("L’intervalle retenu dépasse η : vigilance nécessaire.")

    return " ".join(parts)


def _build_excel_export(summary_df: pd.DataFrame, detail_payload: dict[str, dict[str, pd.DataFrame]]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Synthese_Optimisation", index=False)
        for eq, tables in detail_payload.items():
            prefix = str(eq)[:20]
            for name, df in tables.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    safe_name = f"{prefix}_{name}"[:31]
                    try:
                        df.to_excel(writer, sheet_name=safe_name, index=False)
                    except Exception:
                        pass
    bio.seek(0)
    return bio.getvalue()


# -------------------------------------------------------------------
# Source centrale
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
FALLBACK_OPT = DATA_DIR / "last_optimization.csv"

meta_fail = get_failures_meta()
meta_proj = get_project_meta()
df_src = get_current_failures_df()

if df_src.empty:
    st.error("Aucun dataset actif. Va d’abord dans Sources de données.")
    st.stop()

df_src = df_src.copy()
df_src.columns = [str(c).strip() for c in df_src.columns]
df_src["equipment_code"] = df_src["equipment_code"].astype(str)
df_src["ttf_h"] = pd.to_numeric(df_src["ttf_h"], errors="coerce")
if "duree_rep_h" in df_src.columns:
    df_src["duree_rep_h"] = pd.to_numeric(df_src["duree_rep_h"], errors="coerce")
else:
    df_src["duree_rep_h"] = np.nan
df_src = df_src.dropna(subset=["ttf_h"])
df_src = df_src[df_src["ttf_h"] > 0].reset_index(drop=True)

s1, s2 = st.columns(2)
with s1:
    st.success(f"Dataset actif | rows={meta_fail.get('rows')} | hash={meta_fail.get('hash')}")
with s2:
    if meta_proj.get("ok"):
        st.success(f"Projet actif | hash={meta_proj.get('hash', '')}")
    else:
        st.info("Aucun projet thermique actif")


# -------------------------------------------------------------------
# Contrôles utilisateur
# -------------------------------------------------------------------
eqs_all = sorted(df_src["equipment_code"].unique().tolist())
default_eqs = eqs_all[: min(5, len(eqs_all))] if eqs_all else []

c0, c1 = st.columns([2, 1])
with c0:
    selected_eqs = st.multiselect("Équipements", eqs_all, default=default_eqs if default_eqs else eqs_all)
with c1:
    alpha_default = 0.05
    if meta_proj.get("ok") and selected_eqs:
        try:
            bundle0 = get_pipeline_inputs(asset_id=str(selected_eqs[0]))
            alpha_default = float(bundle0.get("alpha", 0.05) or 0.05)
        except Exception:
            alpha_default = 0.05
    alpha = st.number_input("Alpha", min_value=0.001, max_value=0.20, value=float(alpha_default), step=0.001, format="%.3f")

if not selected_eqs:
    st.info("Sélectionne au moins un équipement.")
    st.stop()

c2, c3, c4, c5 = st.columns(4)
with c2:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
with c3:
    C_prev = st.number_input("Coût préventif", min_value=0.0, value=1.0, step=0.1)
with c4:
    C_corr = st.number_input("Coût correctif", min_value=0.0, value=5.0, step=0.5)
with c5:
    R_min_cost = st.slider("Fiabilité min pour T_cost", 0.0, 0.99, 0.70, 0.01)

project_available = bool(meta_proj.get("ok"))
c6, c7, c8 = st.columns(3)
with c6:
    use_thermal_constraint = st.toggle("Contrainte thermique", value=project_available)
with c7:
    faa_limit = st.number_input("FAA max", min_value=0.0, value=1.50, step=0.05) if use_thermal_constraint else None
with c8:
    lol_limit_pct = st.number_input("Perte de vie max (%)", min_value=0.0, value=0.50, step=0.05) if use_thermal_constraint else None

econ_enabled = (C_prev > 0) and (C_corr > 0)
if not econ_enabled:
    st.warning("Renseigne des coûts préventif et correctif strictement positifs.")


# -------------------------------------------------------------------
# Analyse + Weibull + optimisation
# -------------------------------------------------------------------
fits: dict[str, Any] = {}
org_results: dict[str, dict[str, Any]] = {}
detail_tables: dict[str, dict[str, pd.DataFrame]] = {}
rows: list[dict[str, Any]] = []

for eq in selected_eqs:
    g = df_src[df_src["equipment_code"] == eq].copy()
    ttf_series = _series_to_list(g["ttf_h"])
    if not ttf_series or len(ttf_series) < 3:
        continue

    repair_series = None
    if "duree_rep_h" in g.columns:
        rr = _series_to_list(g["duree_rep_h"])
        repair_series = rr if rr else None

    bundle = get_pipeline_inputs(asset_id=str(eq))
    thermal_df = bundle.get("thermal_df")
    thermal_cfg = bundle.get("thermal_config")

    try:
        pipe = analyze_ttf_pipeline(
            ttf_series=ttf_series,
            alpha=float(alpha),
            repair_series=repair_series,
            thermal_df=thermal_df if use_thermal_constraint else None,
            thermal_config=thermal_cfg if use_thermal_constraint else None,
        )
    except Exception as e:
        pipe = {
            "reliability": {
                "error": str(e),
                "model": "?",
                "distribution": "?",
                "params": {},
                "goodness": {},
                "tests": {},
                "decision": {},
                "indicators": {},
            },
            "thermal": None,
            "tables": {},
        }

    org_results[eq] = pipe
    detail_tables[eq] = pipe.get("tables", {}) or {}

    try:
        fits[eq] = fit_weibull(np.array(ttf_series, dtype=float))
    except Exception:
        continue

if not fits:
    st.error("Pas assez de TTF exploitables (≥3) pour les équipements sélectionnés.")
    st.stop()

res_all: dict[str, dict[str, Any]] = {}
if econ_enabled:
    try:
        res_all = propose_intervals_cost_and_reliability(
            fits=fits,
            C_prev=float(C_prev),
            C_corr=float(C_corr),
            R_target=float(R_target),
            R_min_cost=float(R_min_cost),
        )
    except Exception:
        res_all = {}

for eq, ft in fits.items():
    pipe = org_results.get(eq, {}) or {}
    rel = pipe.get("reliability", {}) or {}
    therm = pipe.get("thermal")
    indicators = rel.get("indicators", {}) or {}
    params = rel.get("params", {}) or {}
    tests = rel.get("tests", {}) or {}
    decision = rel.get("decision", {}) or {}

    beta_weibull = _safe_num(getattr(ft, "beta", None))
    eta_weibull = _safe_num(getattr(ft, "eta", None))
    gamma_weibull = _safe_num(getattr(ft, "gamma", 0.0))

    beta_main = _safe_num(params.get("beta"), beta_weibull)
    eta_main = _safe_num(params.get("eta"), eta_weibull)
    gamma_main = _safe_num(params.get("gamma"), gamma_weibull)

    t_r = (res_all.get(eq) or {}).get("T_R")
    t_cost = (res_all.get(eq) or {}).get("T_cost")
    r_cost = (res_all.get(eq) or {}).get("R_at_T")
    c_min = (res_all.get(eq) or {}).get("C_min")

    thermal_ok, thermal_status, theta_hs_max, faa_max, lol_pct = _thermal_status_label(
        therm,
        faa_limit if use_thermal_constraint else None,
        lol_limit_pct if use_thermal_constraint else None,
    )

    maintenance_type = _recommend_maintenance(beta_main or 1.0, rel.get("model"), thermal_status)
    t_rec = _recommend_interval(beta_main or 1.0, rel.get("model"), t_cost, t_r)

    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}
    dep = tests.get("dependence", {}) or {}

    rows.append(
        {
            "equipment_code": eq,
            "model": rel.get("model"),
            "distribution": rel.get("distribution"),

            "mk_p": mk.get("p"),
            "mk_direction": mk.get("direction"),
            "laplace_p": lap.get("p"),
            "laplace_direction": lap.get("direction"),
            "spearman_r": dep.get("spearman_r"),
            "spearman_p": dep.get("spearman_p"),

            "MTTF_h": indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"),
            "MTBF_h": indicators.get("mtbf_h"),
            "MTTR_h": indicators.get("mttr_h"),
            "availability_pct": None if indicators.get("availability_intrinsic") is None else 100.0 * float(indicators.get("availability_intrinsic")),

            "beta": beta_main,
            "eta_h": eta_main,
            "gamma_h": gamma_main,

            "beta_weibull": beta_weibull,
            "eta_weibull_h": eta_weibull,
            "gamma_weibull_h": gamma_weibull,

            "theta_HS_max": theta_hs_max,
            "FAA_max": faa_max,
            "loss_of_life_pct": lol_pct,
            "thermal_status": thermal_status,

            "T_R_h": _safe_num(t_r),
            "T_cost_h": _safe_num(t_cost),
            "R(T_cost)": _safe_num(r_cost),
            "C_min_per_h": _safe_num(c_min),
            "T_recommended_h": _safe_num(t_rec),

            "maintenance_type": maintenance_type,
            "decision_reason": decision.get("reason"),
            "optimization_note": _optimization_note(eta_main, t_cost, t_r, t_rec),
        }
    )

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)
if df_out.empty:
    st.error("Aucun résultat exploitable après optimisation.")
    st.stop()

st.session_state["optimization_df"] = df_out.copy()
st.session_state["optimization_src"] = "optimisation_page"
st.session_state["opt_meta"] = {"hash": _df_hash(df_out), "rows": int(len(df_out)), "source": "optimisation_page"}

tabs = st.tabs([
    "Paramètres",
    "Optimisation",
    "Courbes",
    "Détail équipement",
    "Exports",
])

with tabs[0]:
    st.subheader("Paramètres issus de l’analyse")

    trend_df = df_out[[
        "equipment_code", "mk_p", "mk_direction", "laplace_p", "laplace_direction"
    ]].copy()
    dep_df = df_out[[
        "equipment_code", "spearman_r", "spearman_p"
    ]].copy()
    rel_df = df_out[[
        "equipment_code", "model", "distribution", "MTTF_h", "MTBF_h", "MTTR_h",
        "availability_pct", "beta", "eta_h", "gamma_h"
    ]].copy()
    therm_df = df_out[[
        "equipment_code", "theta_HS_max", "FAA_max", "loss_of_life_pct", "thermal_status"
    ]].copy()

    st.markdown("#### Tendance")
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

    st.markdown("#### Dépendance")
    st.dataframe(dep_df, use_container_width=True, hide_index=True)

    st.markdown("#### Fiabilité")
    st.dataframe(rel_df, use_container_width=True, hide_index=True)

    st.markdown("#### Thermique")
    st.dataframe(therm_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Résultat de l’optimisation")

    opt_view = df_out[[
        "equipment_code",
        "model",
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
        "thermal_status",
        "optimization_note",
    ]].copy()
    st.dataframe(opt_view, use_container_width=True, hide_index=True)

    st.success(f"Planning envoyé vers Maintenance | rows={len(df_out)} | hash={_df_hash(df_out)}")

    if st.button("Sauver aussi en fichier fallback", use_container_width=True):
        df_out.to_csv(FALLBACK_OPT, index=False, encoding="utf-8")
        st.success(f"Écrit : {FALLBACK_OPT}")

with tabs[2]:
    st.subheader("Courbes R(t) de référence économique")

    etas = [float(getattr(ft, "eta", 1.0) or 1.0) for ft in fits.values()]
    tmax = max(etas) * 1.6 if etas else 1000.0

    maybe_itv = []
    for eq in fits.keys():
        if _is_pos_number(df_out.loc[df_out["equipment_code"] == eq, "T_R_h"].iloc[0] if eq in df_out["equipment_code"].values else None):
            maybe_itv.append(float(df_out.loc[df_out["equipment_code"] == eq, "T_R_h"].iloc[0]))
        if _is_pos_number(df_out.loc[df_out["equipment_code"] == eq, "T_cost_h"].iloc[0] if eq in df_out["equipment_code"].values else None):
            maybe_itv.append(float(df_out.loc[df_out["equipment_code"] == eq, "T_cost_h"].iloc[0]))

    if maybe_itv:
        tmax = max(tmax, max(maybe_itv) * 1.2)

    t = np.linspace(0, max(tmax, 1.0), 350)
    fig, ax = plt.subplots()

    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", 1.0))
        eta = float(getattr(ft, "eta", 1.0))
        gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

        y = np.ones_like(t, dtype=float)
        mask = t > gamma
        y[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-9)) ** max(beta, 1e-9)))
        proc = ((org_results.get(eq, {}) or {}).get("reliability", {}) or {}).get("model", "?")
        ax.plot(t, y, linewidth=2, label=f"{eq} | {proc} | β={beta:.2f}, η={eta:.1f}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("R(t)")
    ax.set_title("Fiabilité R(t)")
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

with tabs[3]:
    st.subheader("Détail par équipement")

    sel_eq = st.selectbox("Équipement", options=df_out["equipment_code"].tolist())
    row = df_out[df_out["equipment_code"] == sel_eq].iloc[0].to_dict()
    pipe = org_results.get(sel_eq, {}) or {}
    rel = pipe.get("reliability", {}) or {}
    thermal = pipe.get("thermal")
    tables = pipe.get("tables", {}) or {}

    st.markdown(
        f"### {sel_eq}\n"
        f"- **Processus retenu** : **{row.get('model', '—')}**\n"
        f"- **Distribution retenue** : **{row.get('distribution', '—')}**\n"
        f"- **Beta / Eta / Gamma** : **{_fmt(row.get('beta'),2)} / {_fmt(row.get('eta_h'),1)} / {_fmt(row.get('gamma_h'),1)}**\n"
        f"- **T_cost** : **{_fmt(row.get('T_cost_h'),1)} h**\n"
        f"- **T_R** : **{_fmt(row.get('T_R_h'),1)} h**\n"
        f"- **T recommandé** : **{_fmt(row.get('T_recommended_h'),1)} h**\n"
        f"- **Maintenance recommandée** : **{row.get('maintenance_type', '—')}**\n"
        f"- **Statut thermique** : **{row.get('thermal_status', '—')}**"
    )

    if row.get("decision_reason"):
        st.caption(str(row["decision_reason"]))

    st.info(row.get("optimization_note", "—"))

    st.markdown("#### Actions suggérées")
    for a in suggested_actions(float(row["beta_weibull"]) if _is_pos_number(row.get("beta_weibull")) else 1.0):
        st.markdown(f"- {a}")

    subtabs = st.tabs(["Fiabilité", "Thermique", "Tableaux exportables"])

    with subtabs[0]:
        for key, label in [
            ("trend_results", "Tests de tendance"),
            ("dependence_results", "Tests de dépendance"),
            ("process_choice", "Choix du processus"),
            ("fit_candidates", "Candidats"),
            ("reliability_summary", "Synthèse fiabiliste"),
        ]:
            dfk = tables.get(key)
            if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                st.markdown(f"##### {label}")
                st.dataframe(dfk, use_container_width=True, hide_index=True)

    with subtabs[1]:
        if thermal is None:
            st.info("Pas de données thermiques disponibles pour cet équipement.")
        else:
            for key, label in [
                ("thermal_summary", "Synthèse thermique"),
                ("thermal_table_indicators", "Indicateurs thermiques"),
                ("thermal_top5_days", "Top 5 jours critiques"),
            ]:
                dfk = tables.get(key)
                if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                    st.markdown(f"##### {label}")
                    st.dataframe(dfk, use_container_width=True, hide_index=True)

            ts = thermal.get("timeseries")
            if isinstance(ts, pd.DataFrame) and not ts.empty:
                c1, c2 = st.columns(2)
                with c1:
                    fig1, ax1 = plt.subplots()
                    ax1.plot(pd.to_datetime(ts["timestamp"]), ts["theta_HS_est_C"])
                    ax1.set_title("Température point chaud estimée")
                    ax1.set_xlabel("Temps")
                    ax1.set_ylabel("°C")
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1, clear_figure=True)
                with c2:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(pd.to_datetime(ts["timestamp"]), ts["FAA"])
                    ax2.set_title("Facteur d’accélération du vieillissement")
                    ax2.set_xlabel("Temps")
                    ax2.set_ylabel("FAA")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2, clear_figure=True)

    with subtabs[2]:
        for key, dfk in tables.items():
            if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                st.markdown(f"##### {key}")
                st.dataframe(dfk, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Exports")

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger CSV optimisé",
        data=csv_bytes,
        file_name="optimisation_intervalles.csv",
        mime="text/csv",
        use_container_width=True,
    )

    xlsx_bytes = _build_excel_export(df_out, detail_tables)
    st.download_button(
        "Télécharger Excel optimisation + détails",
        data=xlsx_bytes,
        file_name="optimisation_intervalles_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if export_optimization_report_pdf is None:
        st.info("Module PDF optimisation indisponible.")
        if _pdf_import_error:
            st.caption(_pdf_import_error)
    else:
        if st.button("Générer le PDF optimisation", type="primary", use_container_width=True):
            try:
                intervals = {}
                for eq in fits.keys():
                    row_eq = df_out[df_out["equipment_code"] == eq]
                    if row_eq.empty:
                        continue
                    rr = row_eq.iloc[0]
                    intervals[eq] = {
                        "T_R": rr.get("T_R_h"),
                        "T_cost": rr.get("T_cost_h"),
                        "R_at_T": rr.get("R(T_cost)"),
                        "C_min": rr.get("C_min_per_h"),
                    }

                org_results_compat = {
                    eq: (org_results.get(eq, {}) or {}).get("reliability", {})
                    for eq in org_results.keys()
                }

                try:
                    path = export_optimization_report_pdf(
                        df=df_src[df_src["equipment_code"].isin(selected_eqs)].copy(),
                        fits=fits,
                        intervals=intervals,
                        organigram_by_eq=org_results_compat,
                        out_dir=str(BASE_DIR / "reports"),
                        df_out=df_out,
                        meta={
                            "alpha": alpha,
                            "R_target": R_target,
                            "C_prev": C_prev,
                            "C_corr": C_corr,
                            "R_min_cost": R_min_cost,
                        },
                    )
                except TypeError:
                    path = export_optimization_report_pdf(
                        df_src[df_src["equipment_code"].isin(selected_eqs)].copy(),
                        fits,
                        intervals,
                        org_results_compat,
                        out_dir=str(BASE_DIR / "reports"),
                    )

                st.session_state["opt_pdf_path"] = path
                st.success(f"PDF généré : {path}")
            except Exception as e:
                st.error(f"PDF : {e}")

        pdf_path = st.session_state.get("opt_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Télécharger le PDF optimisation",
                    data=f,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )