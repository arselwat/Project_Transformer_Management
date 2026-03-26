
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from io import BytesIO
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.security.auth import require_login
from core.reliability.organigram import analyze_ttf_pipeline

# Optional imports / fallbacks
try:
    from core.datahub import get_current_failures_df, get_failures_meta  # type: ignore
except Exception:
    get_current_failures_df = None
    get_failures_meta = None

try:
    from core.datahub import get_current_project_data, get_project_meta  # type: ignore
except Exception:
    get_current_project_data = None
    get_project_meta = None

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

st.title("📊 Résultat analyse / optimisation / maintenance")
st.caption(
    "De la détection statistique à la décision finale : tests de tendance, dépendance, choix du processus, "
    "ajustement, thermique, optimisation et recommandations maintenance."
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
FAILURES_CSV = DATA_DIR / "failures_saved.csv"
OPTIM_CSV = DATA_DIR / "last_optimization.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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


def _coerce_bool01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    mapping = {
        "1": 1, "0": 0, "true": 1, "false": 0,
        "yes": 1, "no": 0, "oui": 1, "non": 0, "y": 1, "n": 0,
    }
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(int)
    )


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

    # normalisation du facteur de charge
    if "K" not in out.columns:
        if "load_factor" in out.columns:
            out["K"] = pd.to_numeric(out["load_factor"], errors="coerce")
        elif "charge_pct" in out.columns:
            out["K"] = pd.to_numeric(out["charge_pct"], errors="coerce") / 100.0
        elif "load_mva" in out.columns:
            out["K"] = pd.to_numeric(out["load_mva"], errors="coerce") / 100.0  # fallback sur 100 MVA

    if "etat_ventilateurs" not in out.columns:
        out["etat_ventilateurs"] = 0

    for c in [
        "temp_amb_C", "K", "charge_pct", "load_factor", "load_mva",
        "etat_ventilateurs", "temp_cuve_C", "current_a",
        "top_oil_temp_c", "hotspot_temp_c",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.reset_index(drop=True)
    return out


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


def _load_failures_df(uploaded_csv=None) -> pd.DataFrame:
    if uploaded_csv is not None:
        df = _read_csv_flex(uploaded_csv)
    elif get_current_failures_df is not None:
        try:
            df = get_current_failures_df()
        except Exception:
            df = pd.DataFrame()
    elif FAILURES_CSV.exists():
        df = _read_csv_flex(FAILURES_CSV)
    else:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
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
    df = df[df["ttf_h"] > 0]
    return df.reset_index(drop=True)


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
    # normaliser quelques colonnes pour usage global
    if "process_model" in out.columns and "model" not in out.columns:
        out["model"] = out["process_model"]
    if "beta_pipe" in out.columns and "beta" not in out.columns:
        out["beta"] = out["beta_pipe"]
    if "eta_pipe_h" in out.columns and "eta_h" not in out.columns:
        out["eta_h"] = out["eta_pipe_h"]
    if "gamma_pipe_h" in out.columns and "gamma_h" not in out.columns:
        out["gamma_h"] = out["gamma_pipe_h"]

    for c in ["T_recommended_h", "T_R_h", "T_cost_h", "days_left", "beta", "eta_h", "gamma_h"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_project_context(uploaded_xlsx=None) -> Dict[str, pd.DataFrame]:
    # 1) upload manuel prioritaire
    if uploaded_xlsx is not None:
        try:
            raw = uploaded_xlsx.read()
            bio = io.BytesIO(raw)
            xls = pd.ExcelFile(bio)
            sheets = {sheet: pd.read_excel(io.BytesIO(raw), sheet_name=sheet) for sheet in xls.sheet_names}
        except Exception:
            sheets = {}
    # 2) projet actif dans datahub
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


def _extract_thermal_for_eq(
    sheets: Dict[str, pd.DataFrame],
    equipment_code: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if not sheets:
        return None, None

    thermal_df = None
    thermal_cfg = None

    ts = sheets.get("thermal_timeseries")
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        tmp = ts.copy()
        asset_col = "asset_id" if "asset_id" in tmp.columns else "equipment_code" if "equipment_code" in tmp.columns else None
        if asset_col:
            tmp = tmp[tmp[asset_col].astype(str) == str(equipment_code)]
        if not tmp.empty:
            thermal_df = tmp.reset_index(drop=True)

    params = sheets.get("thermal_params")
    if isinstance(params, pd.DataFrame) and not params.empty:
        tmp = params.copy()
        asset_col = "asset_id" if "asset_id" in tmp.columns else "equipment_code" if "equipment_code" in tmp.columns else None
        if asset_col:
            tmp = tmp[tmp[asset_col].astype(str) == str(equipment_code)]
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
                "eta_h": r.get("eta_h", r.get("eta")),
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
            f"Processus {model}, vieillissement/statut thermique défavorable et échéance proche. "
            f"Type recommandé : {maint_type or 'maintenance ciblée'}."
        )
    elif score >= 7:
        decision = "Préventif renforcé"
        reason = (
            "Risque significatif détecté. Renforcer la surveillance conditionnelle et planifier une "
            "intervention avant l’intervalle critique."
        )
    elif score >= 4:
        decision = "Surveillance active"
        reason = "Situation intermédiaire : conserver le plan optimisé, surveiller les tendances et les dérives thermiques."
    else:
        decision = "Suivi nominal"
        reason = "Pas de signal critique immédiat : appliquer le calendrier recommandé et une surveillance standard."

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


# ---------------------------------------------------------------------
# Sidebar / sources
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Sources & options")
    up_fail = st.file_uploader("TTF CSV optionnel", type=["csv"], help="Fallback si le dataset actif n’est pas chargé.")
    up_project = st.file_uploader(
        "Fichier projet Excel optionnel",
        type=["xlsx"],
        help="Pour enrichir la page avec la thermique (feuilles thermal_timeseries / thermal_params).",
    )
    alpha = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
    within_days = st.slider("Fenêtre maintenance (jours)", 7, 365, 30, 1)
    start_dt = st.date_input("Date de référence maintenance", value=date.today())


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df_fail = _load_failures_df(uploaded_csv=up_fail)
if df_fail.empty:
    st.error("Aucun dataset TTF disponible. Charge d’abord les sources ou importe un CSV equipment_code, ttf_h[, duree_rep_h].")
    st.stop()

project_sheets = _load_project_context(up_project)
opt_df = _load_optimization_df()

meta_fail = {}
if callable(get_failures_meta):
    try:
        meta_fail = get_failures_meta() or {}
    except Exception:
        meta_fail = {}

meta_project = {}
if callable(get_project_meta):
    try:
        meta_project = get_project_meta() or {}
    except Exception:
        meta_project = {}

pm_all = st.session_state.get("pm_virtual_all")
pm_due = st.session_state.get("pm_virtual_due")
if not isinstance(pm_all, list) or not isinstance(pm_due, list):
    pm_all_df, pm_due_df = _build_virtual_pm_plan_from_optimization(opt_df, start_dt, within_days)
else:
    pm_all_df = pd.DataFrame(pm_all)
    pm_due_df = pd.DataFrame(pm_due)

# small source strip
s1, s2, s3 = st.columns(3)
with s1:
    st.info(f"TTF actifs : {len(df_fail)} lignes")
with s2:
    st.info(f"Projet thermique : {'chargé' if bool(project_sheets) else 'non chargé'}")
with s3:
    st.info(f"Optimisation : {'disponible' if isinstance(opt_df, pd.DataFrame) and not opt_df.empty else 'non disponible'}")

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
with st.spinner("Analyse globale en cours..."):
    results_by_eq: Dict[str, Dict[str, Any]] = {}
    detail_tables_by_eq: Dict[str, Dict[str, pd.DataFrame]] = {}
    global_rows: List[Dict[str, Any]] = []

    eqs = sorted(df_fail["equipment_code"].astype(str).unique().tolist())

    for eq in eqs:
        g = df_fail[df_fail["equipment_code"].astype(str) == str(eq)].copy()
        ttf_series = g["ttf_h"].dropna().astype(float).tolist()
        if len(ttf_series) < 3:
            continue

        rr = pd.to_numeric(g.get("duree_rep_h"), errors="coerce").dropna().tolist() if "duree_rep_h" in g.columns else []
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
        tables = res.get("tables", {}) or {}
        detail_tables_by_eq[eq] = tables

        reliability = res.get("reliability", {}) or {}
        indicators = reliability.get("indicators", {}) or {}
        params = reliability.get("params", {}) or {}
        tests = reliability.get("tests", {}) or {}
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

        trend_mk = tests.get("trend_mk", {}) or {}
        trend_lap = tests.get("trend_laplace", {}) or {}
        dep = tests.get("dependence", {}) or {}
        direction = trend_mk.get("direction") or trend_lap.get("direction")

        beta_pref = params.get("beta")
        if beta_pref is None:
            beta_pref = opt_row.get("beta")
        eta_pref = params.get("eta")
        if eta_pref is None:
            eta_pref = opt_row.get("eta_h", opt_row.get("eta"))
        gamma_pref = params.get("gamma")
        if gamma_pref is None:
            gamma_pref = opt_row.get("gamma_h")

        row = {
            "equipment_code": eq,
            "n_ttf": len(ttf_series),
            "trend_detected": "Oui" if (trend_mk.get("has_trend") or trend_lap.get("has_trend")) else "Non",
            "trend_direction": direction,
            "dependence_detected": "Oui" if dep.get("has_dep") else "Non",
            "model": reliability.get("model"),
            "distribution": reliability.get("distribution"),
            "beta": beta_pref,
            "eta_h": eta_pref,
            "gamma_h": gamma_pref,
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

# ---------------------------------------------------------------------
# Global tables for exports
# ---------------------------------------------------------------------
trend_overview = summary_df[[
    "equipment_code", "n_ttf", "trend_detected", "trend_direction", "dependence_detected", "model", "distribution"
]].copy()

risk_overview = summary_df[[
    "equipment_code", "beta", "eta_h", "mtbf_h", "mttr_h", "availability_pct",
    "theta_hs_max", "faa_max", "loss_of_life_pct", "thermal_status"
]].copy()

optimization_overview = summary_df[[
    "equipment_code", "maintenance_type", "T_recommended_h", "T_R_h", "T_cost_h", "R(T_cost)",
    "C_min_per_h", "next_due_date", "days_left"
]].copy()

final_decision_df = summary_df[[
    "equipment_code", "model", "distribution", "thermal_status", "maintenance_type", "days_left",
    "priority_score", "priorite", "decision_finale", "motif_decision"
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

# ---------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
view_tab, tests_tab, thermal_tab, opt_tab, decision_tab = st.tabs(
    [
        "🌍 Vue globale",
        "🧪 Tests & modèles",
        "🌡️ Thermique",
        "🧠 Optimisation & maintenance",
        "✅ Décision finale",
    ]
)

with view_tab:
    st.subheader("Tableau de synthèse global")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        proc = summary_df["model"].astype(str).value_counts()
        ax.bar(proc.index.tolist(), proc.values.tolist())
        ax.set_title("Répartition des processus retenus")
        ax.set_xlabel("Processus")
        ax.set_ylabel("Nombre d’équipements")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4))
        dplot = summary_df[["equipment_code", "priority_score"]].sort_values("priority_score", ascending=True)
        ax.barh(dplot["equipment_code"], dplot["priority_score"])
        ax.set_title("Score de priorité globale")
        ax.set_xlabel("Score")
        ax.set_ylabel("Équipement")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Lecture rapide")
    st.markdown(
        "- **trend_detected** indique si les TTF évoluent significativement dans le temps.\n"
        "- **dependence_detected** indique si les pannes successives restent corrélées.\n"
        "- **thermal_status** synthétise θHS, FAA et perte de vie.\n"
        "- **priority_score / priorite** hiérarchise les actions."
    )

with tests_tab:
    eq = st.selectbox("Équipement", options=summary_df["equipment_code"].tolist(), key="global_eq_tests")
    eq_res = results_by_eq[eq]
    eq_tables = detail_tables_by_eq[eq]
    st.markdown(f"### Parcours d’analyse — {eq}")
    st.dataframe(eq_tables.get("trend_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.dataframe(eq_tables.get("dependence_results", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.dataframe(eq_tables.get("process_choice", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.dataframe(eq_tables.get("fit_candidates", pd.DataFrame()), use_container_width=True, hide_index=True)
    st.dataframe(eq_tables.get("reliability_summary", pd.DataFrame()), use_container_width=True, hide_index=True)

    rel = eq_res.get("reliability", {})
    ind = rel.get("indicators", {}) or {}
    avail = None if ind.get("availability_intrinsic") is None else 100 * float(ind.get("availability_intrinsic"))
    st.info(
        f"Équipement {eq} → Processus **{rel.get('model', '—')}**, loi **{rel.get('distribution', '—')}**, "
        f"MTBF={_fmt(ind.get('mtbf_h'),1)} h, MTTR={_fmt(ind.get('mttr_h'),1)} h, disponibilité={_fmt(avail,1)} %."
    )

with thermal_tab:
    eq = st.selectbox("Équipement thermique", options=summary_df["equipment_code"].tolist(), key="global_eq_thermal")
    eq_res = results_by_eq[eq]
    eq_tables = detail_tables_by_eq[eq]
    thermal = eq_res.get("thermal")

    if not thermal:
        st.warning("Aucune donnée thermique disponible pour cet équipement. Charge le fichier projet Excel ou un projet actif depuis Sources.")
    else:
        st.dataframe(eq_tables.get("thermal_table_dataset", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.dataframe(eq_tables.get("thermal_table_params", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.dataframe(eq_tables.get("thermal_table_indicators", pd.DataFrame()), use_container_width=True, hide_index=True)

        ts = thermal.get("timeseries")
        if isinstance(ts, pd.DataFrame) and not ts.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(ts["timestamp"]), ts["theta_HS_est_C"])
                ax.set_title(f"θHS estimée — {eq}")
                ax.set_xlabel("Temps")
                ax.set_ylabel("°C")
                ax.grid(True, alpha=0.25)
                st.pyplot(fig, clear_figure=True)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(ts["timestamp"]), ts["FAA"])
                ax.set_title(f"FAA — {eq}")
                ax.set_xlabel("Temps")
                ax.set_ylabel("p.u.")
                ax.grid(True, alpha=0.25)
                st.pyplot(fig, clear_figure=True)

        st.markdown("### Synthèses complémentaires")
        st.dataframe(eq_tables.get("thermal_summary", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.dataframe(eq_tables.get("thermal_top5_days", pd.DataFrame()), use_container_width=True, hide_index=True)

with opt_tab:
    st.subheader("Synthèse optimisation")
    if opt_df.empty:
        st.info("Aucune optimisation disponible pour le moment. Passe d’abord par la page Optimisation.")
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

with decision_tab:
    st.subheader("Décision finale hiérarchisée")
    st.dataframe(_style_decision_df(final_decision_df), use_container_width=True, hide_index=True)

    eq = st.selectbox("Équipement — décision finale", options=summary_df["equipment_code"].tolist(), key="global_eq_decision")
    row = summary_df[summary_df["equipment_code"] == eq].iloc[0].to_dict()

    st.markdown(f"### {_decision_badge(str(row['decision_finale']))} — {eq}")
    st.markdown(
        f"**Processus retenu** : {row.get('model','—')}  \n"
        f"**Distribution** : {row.get('distribution','—')}  \n"
        f"**Beta / Eta** : {_fmt(row.get('beta'),2)} / {_fmt(row.get('eta_h'),1)} h  \n"
        f"**Statut thermique** : {row.get('thermal_status','—')}  \n"
        f"**Maintenance recommandée** : {row.get('maintenance_type','—')}  \n"
        f"**Échéance** : {row.get('next_due_date','—')} (J-{row.get('days_left','—')})  \n"
        f"**Score de priorité** : {row.get('priority_score','—')} ({row.get('priorite','—')})"
    )
    st.info(row.get("motif_decision", "Aucun motif disponible."))

    st.markdown("#### Chemin de décision")
    st.markdown(
        f"1. **Test de tendance** → {row.get('trend_detected','—')} ({row.get('trend_direction','—')})\n"
        f"2. **Test de dépendance** → {row.get('dependence_detected','—')}\n"
        f"3. **Processus** → {row.get('model','—')}\n"
        f"4. **Modèle retenu** → {row.get('distribution','—')}\n"
        f"5. **Thermique** → {row.get('thermal_status','—')}\n"
        f"6. **Optimisation / maintenance** → {row.get('maintenance_type','—')} / T_recommended={_fmt(row.get('T_recommended_h'),1)} h\n"
        f"7. **Décision finale** → {row.get('decision_finale','—')}"
    )


# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
st.divider()
st.subheader("📦 Exports")

excel_bytes = _xlsx_bytes(global_tables=global_tables, detail_tables=detail_tables_by_eq)
st.download_button(
    "⬇️ Télécharger le pack Excel global",
    data=excel_bytes,
    file_name="resultat_analyse_optimisation_maintenance.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

if export_global_analysis_report_pdf is None:
    st.info("Module PDF global non détecté. Ajoute `core.reliability.reporting_global.py` pour le téléchargement PDF.")
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
                    "source_rows": len(df_fail),
                    "source_hash": meta_fail.get("hash", "") if isinstance(meta_fail, dict) else "",
                    "project_hash": meta_project.get("hash", "") if isinstance(meta_project, dict) else "",
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
