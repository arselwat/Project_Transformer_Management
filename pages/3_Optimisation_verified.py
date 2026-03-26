
from __future__ import annotations

from pathlib import Path
import io
import math
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
from core.datahub import get_current_failures_df, get_current_project_data


export_optimization_report_pdf = None
_pdf_import_error = None
try:
    from core.reliability.reporting_optimize import export_optimization_report_pdf as _export_opt_pdf
    export_optimization_report_pdf = _export_opt_pdf
except Exception as e:
    _pdf_import_error = e
    export_optimization_report_pdf = None


st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()

st.title("🧠 Optimisation — Intervalles, coût, fiabilité et thermique")
st.caption(
    "Le projet actif importé dans Sources est utilisé en priorité. "
    "Cette page calcule les intervalles économiques, contrôle les contraintes thermiques "
    "et prépare un tableau directement réutilisable dans Maintenance."
)


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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code",
        "equipement": "equipment_code",
        "code_equipement": "equipment_code",
        "asset_id": "equipment_code",
        "eqp": "equipment_code",
        "ttf": "ttf_h",
        "ttf_hours": "ttf_h",
        "mttr_h": "duree_rep_h",
        "repair_hours": "duree_rep_h",
        "repair_time_hours": "duree_rep_h",
    }
    cols = {str(c).lower().strip(): c for c in df.columns}
    ren = {}
    for k, v in mapping.items():
        if k in cols:
            ren[cols[k]] = v
    out = df.rename(columns=ren).copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _fnum(x: Any, nd: int = 2, default: str = "—") -> str:
    try:
        if x is None:
            return default
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return f"{x:.{nd}f}"
    except Exception:
        return default


def _is_pos_number(x: Any) -> bool:
    try:
        return float(x) > 0 and np.isfinite(float(x))
    except Exception:
        return False


def _df_hash(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def _safe_num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _load_excel_bytes(uploaded_file) -> bytes:
    uploaded_file.seek(0)
    return uploaded_file.read()


def _coerce_bool01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    mapping = {
        "1": 1, "0": 0, "true": 1, "false": 0,
        "yes": 1, "no": 0, "oui": 1, "non": 0, "y": 1, "n": 0,
    }
    return (
        s.astype(str).str.strip().str.lower().map(mapping)
        .fillna(pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)
    )


def _compute_ttf_from_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    df = events_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "asset_id" not in df.columns or "event_start" not in df.columns:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    if "is_failure" in df.columns:
        df["is_failure"] = _coerce_bool01(df["is_failure"])
        df = df[df["is_failure"] == 1].copy()
    elif "event_type" in df.columns:
        df = df[df["event_type"].astype(str).str.upper().eq("FAILURE")].copy()

    if df.empty:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df = df.dropna(subset=["event_start"]).sort_values(["asset_id", "event_start"])

    if "repair_time_hours" not in df.columns:
        df["repair_time_hours"] = np.nan
    df["repair_time_hours"] = pd.to_numeric(df["repair_time_hours"], errors="coerce")

    rows = []
    for eq, g in df.groupby("asset_id"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            ttf_h = (g.loc[i, "event_start"] - g.loc[i - 1, "event_start"]).total_seconds() / 3600.0
            if ttf_h > 0:
                rows.append(
                    {
                        "equipment_code": str(eq),
                        "ttf_h": float(ttf_h),
                        "duree_rep_h": _safe_num(g.loc[i, "repair_time_hours"]),
                    }
                )
    return pd.DataFrame(rows)


def _load_project_excel(src) -> dict[str, Any]:
    if isinstance(src, (str, Path)):
        xls = pd.ExcelFile(src)
    else:
        raw = _load_excel_bytes(src)
        xls = pd.ExcelFile(io.BytesIO(raw))

    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

    events = sheets.get("events_history", pd.DataFrame())
    thermal = sheets.get("thermal_timeseries", pd.DataFrame())
    thermal_params = sheets.get("thermal_params", pd.DataFrame())
    settings = sheets.get("analysis_settings", pd.DataFrame())
    policies = sheets.get("maintenance_policies", pd.DataFrame())
    asset_info = sheets.get("asset_info", pd.DataFrame())

    ttf_df = _compute_ttf_from_events(events)
    ttf_df = _normalize_columns(ttf_df)

    alpha_default = 0.05
    if not settings.empty and "alpha_significance" in settings.columns:
        alpha_default = _safe_num(settings.iloc[0]["alpha_significance"]) or 0.05

    thermal.columns = [str(c).strip() for c in thermal.columns]
    thermal_params.columns = [str(c).strip() for c in thermal_params.columns]
    policies.columns = [str(c).strip() for c in policies.columns]
    asset_info.columns = [str(c).strip() for c in asset_info.columns]

    thermal_map: dict[str, pd.DataFrame] = {}
    if not thermal.empty and "asset_id" in thermal.columns:
        for eq, g in thermal.groupby("asset_id"):
            thermal_map[str(eq)] = g.copy().reset_index(drop=True)

    thermal_cfg_map: dict[str, dict[str, Any]] = {}
    allowed_thermal_keys = {
        "sn_mva", "R", "delta_to_r", "delta_h_r", "tau_to_min", "tau_w_min",
        "n_exp", "m_exp", "forced_tau_to_factor", "forced_delta_to_factor",
        "forced_delta_h_factor", "normal_insulation_life_h",
    }
    if not thermal_params.empty and "asset_id" in thermal_params.columns:
        for _, row in thermal_params.iterrows():
            eq = str(row["asset_id"])
            cfg = {}
            for k in allowed_thermal_keys:
                if k in thermal_params.columns:
                    v = _safe_num(row.get(k))
                    if v is not None:
                        cfg[k] = v
            thermal_cfg_map[eq] = cfg

    return {
        "ttf_df": ttf_df,
        "events_history": events,
        "thermal_map": thermal_map,
        "thermal_cfg_map": thermal_cfg_map,
        "analysis_settings": settings,
        "maintenance_policies": policies,
        "asset_info": asset_info,
        "alpha_default": alpha_default,
        "sheet_names": list(sheets.keys()),
    }


def _recommend_maintenance(beta: float, model: Optional[str] = None, thermal_status: Optional[str] = None) -> str:
    model_s = (model or "").upper()
    if thermal_status == "Alerte thermique":
        return "Préventive immédiate + contrôle thermique"
    if "NHPP" in model_s:
        return "Préventive planifiée (vieillissement détecté)"
    if "BPP" in model_s:
        return "Conditionnelle + analyse causale / inspection rapprochée"
    if beta < 0.9:
        return "Corrective + fiabilisation (pannes de jeunesse)"
    if 0.9 <= beta <= 1.1:
        return "Conditionnelle / inspection (pannes aléatoires)"
    return "Préventive planifiée (âge / usure)"


def _recommend_interval(beta: float, model: Optional[str], t_cost: Any, t_r: Any) -> Optional[float]:
    model_s = (model or "").upper()
    vals = [float(v) for v in [t_cost, t_r] if _is_pos_number(v)]
    if not vals:
        return None
    if "BPP" in model_s:
        return float(min(vals))
    if beta <= 1.1 and "NHPP" not in model_s:
        return None
    return float(min(vals))


def _thermal_status_label(
    thermal_result: Optional[dict],
    faa_limit: Optional[float],
    lol_limit_pct: Optional[float],
) -> tuple[Optional[bool], str, Optional[float], Optional[float]]:
    if not thermal_result:
        return None, "Pas de données thermiques", None, None

    summary = thermal_result.get("summary", {}) or {}
    faa_max = _safe_num(summary.get("faa_max"))
    lol_pct = _safe_num(summary.get("loss_of_life_pct"))

    checks = []
    if faa_limit is not None and faa_max is not None:
        checks.append(faa_max <= faa_limit)
    if lol_limit_pct is not None and lol_pct is not None:
        checks.append(lol_pct <= lol_limit_pct)

    if not checks:
        return None, "Thermique calculée", faa_max, lol_pct
    if all(checks):
        return True, "Conforme thermique", faa_max, lol_pct
    return False, "Alerte thermique", faa_max, lol_pct


def _build_excel_export(summary_df: pd.DataFrame, detail_payload: dict[str, dict[str, pd.DataFrame]]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Synthese_Optimisation", index=False)
        for eq, tables in detail_payload.items():
            prefix = str(eq)[:20]
            for name, df in tables.items():
                safe_name = f"{prefix}_{name}"[:31]
                try:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=safe_name, index=False)
                except Exception:
                    pass
    bio.seek(0)
    return bio.getvalue()


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
FALLBACK_TTF = DATA_DIR / "failures_saved.csv"
FALLBACK_PROJECTS = sorted(DATA_DIR.glob("*.xlsx"))

project_data = get_current_project_data()
active_project_ok = isinstance(project_data, dict) and any(
    isinstance(project_data.get(k), pd.DataFrame) and not project_data.get(k).empty
    for k in ["asset_info", "events_history", "thermal_timeseries", "thermal_params", "analysis_settings"]
)

source_mode = st.radio(
    "Source des données",
    ["Projet actif (importé dans Sources)", "Fichier projet (.xlsx)", "CSV TTF simple"],
    horizontal=True,
)

project_payload: dict[str, Any] = {
    "ttf_df": pd.DataFrame(),
    "events_history": pd.DataFrame(),
    "thermal_map": {},
    "thermal_cfg_map": {},
    "analysis_settings": pd.DataFrame(),
    "maintenance_policies": pd.DataFrame(),
    "asset_info": pd.DataFrame(),
    "alpha_default": 0.05,
    "sheet_names": [],
}

ttf_df = pd.DataFrame()
auto_thermal_available = False

if source_mode == "Projet actif (importé dans Sources)":
    if not active_project_ok:
        st.warning("Aucun projet actif détecté dans Sources. Passe sur « Fichier projet (.xlsx) » ou recharge ton projet dans Sources.")
        st.stop()

    events = project_data.get("events_history", pd.DataFrame()).copy()
    thermal = project_data.get("thermal_timeseries", pd.DataFrame()).copy()
    thermal_params = project_data.get("thermal_params", pd.DataFrame()).copy()
    settings = project_data.get("analysis_settings", pd.DataFrame()).copy()
    policies = project_data.get("maintenance_policies", pd.DataFrame()).copy()
    asset_info = project_data.get("asset_info", pd.DataFrame()).copy()

    ttf_df = project_data.get("failures_ttf", pd.DataFrame()).copy()
    if ttf_df.empty:
        ttf_df = _compute_ttf_from_events(events)
    ttf_df = _normalize_columns(ttf_df)

    alpha_default = 0.05
    if not settings.empty and "alpha_significance" in settings.columns:
        alpha_default = _safe_num(settings.iloc[0]["alpha_significance"]) or 0.05

    thermal_map = {}
    if not thermal.empty and "asset_id" in thermal.columns:
        for eq, g in thermal.groupby("asset_id"):
            thermal_map[str(eq)] = g.copy().reset_index(drop=True)

    thermal_cfg_map = {}
    if not thermal_params.empty and "asset_id" in thermal_params.columns:
        for _, row in thermal_params.iterrows():
            eq = str(row["asset_id"])
            cfg = {}
            for k in ["sn_mva", "R", "delta_to_r", "delta_h_r", "tau_to_min", "tau_w_min", "n_exp", "m_exp",
                      "forced_tau_to_factor", "forced_delta_to_factor", "forced_delta_h_factor", "normal_insulation_life_h"]:
                if k in thermal_params.columns:
                    v = _safe_num(row.get(k))
                    if v is not None:
                        cfg[k] = v
            thermal_cfg_map[eq] = cfg

    project_payload = {
        "ttf_df": ttf_df,
        "events_history": events,
        "thermal_map": thermal_map,
        "thermal_cfg_map": thermal_cfg_map,
        "analysis_settings": settings,
        "maintenance_policies": policies,
        "asset_info": asset_info,
        "alpha_default": alpha_default,
        "sheet_names": [k for k, v in project_data.items() if isinstance(v, pd.DataFrame)],
    }
    auto_thermal_available = bool(thermal_map)

elif source_mode == "Fichier projet (.xlsx)":
    project_choice = st.radio("Mode de chargement", ["Fichier projet local", "Uploader Excel"], horizontal=True)

    if project_choice == "Fichier projet local":
        if not FALLBACK_PROJECTS:
            st.error("Aucun fichier .xlsx trouvé dans le dossier data/. Dépose un fichier projet ou passe par l’uploader.")
            st.stop()
        selected_path = st.selectbox("Choisir le fichier projet", options=[str(p.name) for p in FALLBACK_PROJECTS])
        project_path = DATA_DIR / selected_path
        try:
            project_payload = _load_project_excel(project_path)
        except Exception as e:
            st.error(f"Lecture Excel impossible: {e}")
            st.stop()
    else:
        up_xlsx = st.file_uploader("Uploader un fichier projet Excel", type=["xlsx"])
        if up_xlsx is None:
            st.stop()
        try:
            project_payload = _load_project_excel(up_xlsx)
        except Exception as e:
            st.error(f"Lecture Excel impossible: {e}")
            st.stop()

    ttf_df = project_payload["ttf_df"].copy()
    auto_thermal_available = bool(project_payload["thermal_map"])

    if ttf_df.empty:
        st.error("Aucun TTF construit depuis la feuille events_history. Vérifie les colonnes asset_id, event_start, is_failure.")
        st.stop()

    with st.expander("Voir les feuilles détectées"):
        st.write(project_payload.get("sheet_names", []))
        st.dataframe(ttf_df.head(20), use_container_width=True, hide_index=True)
else:
    csv_choice = st.radio("Mode CSV", ["Dataset actif synchronisé", "Fichier local", "Uploader CSV"], horizontal=True)
    if csv_choice == "Dataset actif synchronisé":
        ttf_df = get_current_failures_df().copy()
    elif csv_choice == "Fichier local":
        if not FALLBACK_TTF.exists():
            st.error("Aucun fichier data/failures_saved.csv — va d’abord dans « Sources de données ».")
            st.stop()
        ttf_df = _read_csv_flex(FALLBACK_TTF)
    else:
        up_csv = st.file_uploader("CSV (equipment_code, ttf_h[, duree_rep_h])", type=["csv"])
        if up_csv is None:
            st.stop()
        ttf_df = _read_csv_flex(up_csv)

    ttf_df = _normalize_columns(ttf_df)
    if ttf_df.empty:
        st.error("CSV vide ou illisible.")
        st.stop()

required = {"equipment_code", "ttf_h"}
if not required.issubset(set(ttf_df.columns)):
    st.error("Colonnes requises: equipment_code, ttf_h")
    st.stop()

ttf_df["equipment_code"] = ttf_df["equipment_code"].astype(str)
ttf_df["ttf_h"] = pd.to_numeric(ttf_df["ttf_h"], errors="coerce")
if "duree_rep_h" in ttf_df.columns:
    ttf_df["duree_rep_h"] = pd.to_numeric(ttf_df["duree_rep_h"], errors="coerce")
else:
    ttf_df["duree_rep_h"] = np.nan

ttf_df = ttf_df.dropna(subset=["ttf_h"])
ttf_df = ttf_df[ttf_df["ttf_h"] > 0].reset_index(drop=True)

if ttf_df.empty:
    st.error("Aucune donnée TTF valide après nettoyage.")
    st.stop()

eqs_all = sorted(ttf_df["equipment_code"].unique().tolist())
selected_eqs = st.multiselect("Équipements à optimiser", eqs_all, default=eqs_all)
if not selected_eqs:
    st.warning("Sélectionne au moins un équipement.")
    st.stop()

df = ttf_df[ttf_df["equipment_code"].isin(selected_eqs)].copy().reset_index(drop=True)

st.markdown("### Paramètres de fiabilité, coût et thermique")
alpha_default = float(project_payload.get("alpha_default", 0.05) or 0.05)

c_alpha, cR, cC1, cC2, cRmin = st.columns(5)
with c_alpha:
    alpha = st.number_input("Alpha", min_value=0.001, max_value=0.20, value=float(alpha_default), step=0.001, format="%.3f")
with cR:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
with cC1:
    C_prev = st.number_input("Coût préventive (C_prev)", min_value=0.0, value=1.0, step=0.1)
with cC2:
    C_corr = st.number_input("Coût corrective (C_corr)", min_value=0.0, value=5.0, step=0.5)
with cRmin:
    R_min_cost = st.slider("Fiabilité min. pour T_cost", 0.0, 0.99, 0.70, 0.01)

cTherm, cFAA, cLOL = st.columns(3)
with cTherm:
    use_thermal_constraint = st.toggle("Activer contrainte thermique", value=auto_thermal_available)
with cFAA:
    faa_limit = st.number_input("FAA max admissible", min_value=0.0, value=1.50, step=0.05) if use_thermal_constraint else None
with cLOL:
    lol_limit_pct = st.number_input("Perte de vie max (%)", min_value=0.0, value=0.50, step=0.05) if use_thermal_constraint else None

st.caption(
    "Base économique actuelle : politique âge sur Weibull. "
    "Le nouveau pipeline fiabiliste sert de décision principale, et Weibull reste la base de calcul de T_cost / T_R."
)

econ_enabled = (C_prev > 0) and (C_corr > 0)
if not econ_enabled:
    st.warning("Renseigne C_prev > 0 et C_corr > 0 pour activer T_cost.")

if use_thermal_constraint and not auto_thermal_available and source_mode == "CSV TTF simple":
    st.info("Aucune donnée thermique liée au CSV simple. La contrainte thermique sera ignorée pour ces équipements.")

fits: dict[str, Any] = {}
for eq in selected_eqs:
    x = df.loc[df["equipment_code"] == eq, "ttf_h"].values
    if len(x) >= 3:
        try:
            fits[eq] = fit_weibull(x)
        except Exception:
            pass

if not fits:
    st.error("Pas assez de TTF (≥3) pour estimer Weibull sur les équipements sélectionnés.")
    st.stop()

org_results: dict[str, dict[str, Any]] = {}
detail_tables: dict[str, dict[str, pd.DataFrame]] = {}

for eq in selected_eqs:
    ttf_series = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    rep_series = df.loc[df["equipment_code"] == eq, "duree_rep_h"].dropna().tolist()

    thermal_df = None
    thermal_cfg = None
    if source_mode != "CSV TTF simple":
        thermal_df = project_payload["thermal_map"].get(eq)
        thermal_cfg = project_payload["thermal_cfg_map"].get(eq, {}) or None

    try:
        pipe = analyze_ttf_pipeline(
            ttf_series=ttf_series,
            alpha=float(alpha),
            repair_series=rep_series if rep_series else None,
            thermal_df=thermal_df,
            thermal_config=thermal_cfg,
        )
    except Exception as e:
        pipe = {
            "reliability": {
                "error": str(e), "model": "?", "distribution": "?", "params": {},
                "goodness": {}, "tests": {}, "decision": {}, "indicators": {}, "candidates": {},
            },
            "thermal": None,
            "tables": {},
        }

    org_results[eq] = pipe
    detail_tables[eq] = pipe.get("tables", {}) or {}

res_all: dict[str, dict[str, Any]] = {}
if econ_enabled:
    res_all = propose_intervals_cost_and_reliability(
        fits=fits,
        C_prev=float(C_prev),
        C_corr=float(C_corr),
        R_target=float(R_target),
        R_min_cost=float(R_min_cost),
    )

intervals_R = {eq: (res_all.get(eq) or {}).get("T_R") for eq in fits.keys()}
intervals_cost = {eq: (res_all.get(eq) or {}).get("T_cost") for eq in fits.keys()}
R_at_cost = {eq: (res_all.get(eq) or {}).get("R_at_T") for eq in fits.keys()}
C_min_map = {eq: (res_all.get(eq) or {}).get("C_min") for eq in fits.keys()}

rows = []
for eq, ft in fits.items():
    pipe = org_results.get(eq, {}) or {}
    rel = pipe.get("reliability", {}) or {}
    thermal = pipe.get("thermal")
    indicators = rel.get("indicators", {}) or {}
    decision = rel.get("decision", {}) or {}
    params = rel.get("params", {}) or {}

    beta_weibull = float(getattr(ft, "beta", float("nan")))
    eta_weibull = float(getattr(ft, "eta", float("nan")))
    gamma_weibull = float(getattr(ft, "gamma", 0.0) or 0.0)

    t_r = intervals_R.get(eq)
    t_cost = intervals_cost.get(eq)
    r_cost = R_at_cost.get(eq)
    c_min = C_min_map.get(eq)

    thermal_ok, thermal_status, faa_max, lol_pct = _thermal_status_label(
        thermal,
        faa_limit if use_thermal_constraint else None,
        lol_limit_pct if use_thermal_constraint else None,
    )

    model = rel.get("model")
    distribution = rel.get("distribution")
    beta_pipe = params.get("beta")
    eta_pipe = params.get("eta")
    gamma_pipe = params.get("gamma")

    beta_for_policy = beta_pipe if _is_pos_number(beta_pipe) else beta_weibull
    maintenance_type = _recommend_maintenance(beta_for_policy, model, thermal_status)
    t_rec = _recommend_interval(beta_for_policy, model, t_cost, t_r)

    reliability_ok = None
    if _is_pos_number(t_cost) and _is_pos_number(r_cost):
        reliability_ok = float(r_cost) >= float(R_min_cost)
    elif _is_pos_number(t_r):
        reliability_ok = True

    admissible_global = None
    checks = []
    if reliability_ok is not None:
        checks.append(bool(reliability_ok))
    if thermal_ok is not None:
        checks.append(bool(thermal_ok))
    if checks:
        admissible_global = all(checks)

    rows.append(
        {
            "equipment_code": eq,
            "process_model": model,
            "distribution": distribution,
            "decision_reason": decision.get("reason"),
            "beta_opt": round(beta_weibull, 3) if np.isfinite(beta_weibull) else None,
            "eta_opt_h": round(eta_weibull, 1) if np.isfinite(eta_weibull) else None,
            "gamma_opt_h": round(gamma_weibull, 1) if np.isfinite(gamma_weibull) else 0.0,
            "beta_pipe": round(float(beta_pipe), 3) if _is_pos_number(beta_pipe) else None,
            "eta_pipe_h": round(float(eta_pipe), 1) if _is_pos_number(eta_pipe) else None,
            "gamma_pipe_h": round(float(gamma_pipe), 1) if gamma_pipe is not None and np.isfinite(float(gamma_pipe)) else None,
            "MTBF_h": round(float(indicators.get("mtbf_h")), 2) if _is_pos_number(indicators.get("mtbf_h")) else None,
            "MTTR_h": round(float(indicators.get("mttr_h")), 2) if _is_pos_number(indicators.get("mttr_h")) else None,
            "availability": round(float(indicators.get("availability_intrinsic")), 4) if indicators.get("availability_intrinsic") is not None else None,
            "T_cost_h": round(float(t_cost), 1) if _is_pos_number(t_cost) else None,
            "R(T_cost)": round(float(r_cost), 3) if _is_pos_number(r_cost) else None,
            "C_min_per_h": round(float(c_min), 4) if _is_pos_number(c_min) else None,
            "T_R_h": round(float(t_r), 1) if _is_pos_number(t_r) else None,
            "T_recommended_h": round(float(t_rec), 1) if _is_pos_number(t_rec) else None,
            "FAA_max": round(float(faa_max), 4) if faa_max is not None else None,
            "loss_of_life_pct": round(float(lol_pct), 4) if lol_pct is not None else None,
            "thermal_status": thermal_status,
            "thermal_ok": thermal_ok,
            "reliability_ok": reliability_ok,
            "admissible_global": admissible_global,
            "maintenance_type": maintenance_type,
        }
    )

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

st.session_state["optimization_df"] = df_out.copy()
st.session_state["optimization_src"] = "optimisation_page"
opt_hash = _df_hash(df_out)
st.session_state["opt_df_out"] = df_out.copy()
st.session_state["opt_meta"] = {"hash": opt_hash, "rows": int(len(df_out)), "source": "optimisation_page"}

st.divider()
st.subheader("🧩 Passerelle → Maintenance")
st.success(f"Planning envoyé automatiquement ✅ | rows={len(df_out)} | hash={opt_hash}")

FALLBACK_OPT = DATA_DIR / "last_optimization.csv"
if st.button("💾 Sauver aussi en fichier (fallback Streamlit Cloud)", use_container_width=True):
    df_out.to_csv(FALLBACK_OPT, index=False, encoding="utf-8")
    st.success(f"Écrit: {FALLBACK_OPT}")

st.caption(
    "La session peut disparaître après redémarrage sur Streamlit Cloud. "
    "Le fichier data/last_optimization.csv sert de fallback."
)

st.subheader("📋 Synthèse optimisation")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger CSV optimisé",
    data=csv_bytes,
    file_name="optimisation_intervalles.csv",
    mime="text/csv",
    use_container_width=True,
)

xlsx_bytes = _build_excel_export(df_out, detail_tables)
st.download_button(
    "⬇️ Télécharger Excel optimisation + détails",
    data=xlsx_bytes,
    file_name="optimisation_intervalles_detail.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.subheader("📈 Courbes R(t) — référence économique")
st.caption(
    "Ces courbes restent basées sur Weibull car le module économique T_cost / T_R utilise actuellement cette base. "
    "Le processus retenu par l’organigramme reste affiché séparément dans les tableaux et le détail."
)

etas = [float(getattr(ft, "eta", 1.0) or 1.0) for ft in fits.values()]
tmax = max(etas) * 1.6 if etas else 1000.0
maybe_itv = []
for eq in fits.keys():
    if _is_pos_number(intervals_R.get(eq)):
        maybe_itv.append(float(intervals_R[eq]))
    if _is_pos_number(intervals_cost.get(eq)):
        maybe_itv.append(float(intervals_cost[eq]))
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

st.subheader("🔎 Détails et tableaux du pipeline")
sel_eq = st.selectbox("Équipement", options=df_out["equipment_code"].tolist())
row = df_out[df_out["equipment_code"] == sel_eq].iloc[0].to_dict()
pipe = org_results.get(sel_eq, {}) or {}
rel = pipe.get("reliability", {}) or {}
thermal = pipe.get("thermal")
tables = pipe.get("tables", {}) or {}

st.markdown(
    f"### Résultats — **{sel_eq}**\n"
    f"- **Processus retenu** : **{row.get('process_model', '—')}**\n"
    f"- **Distribution retenue** : **{row.get('distribution', '—')}**\n"
    f"- **T_cost** = **{_fnum(row.get('T_cost_h'), 1)} h**\n"
    f"- **T_R** = **{_fnum(row.get('T_R_h'), 1)} h**\n"
    f"- **T recommandé** = **{_fnum(row.get('T_recommended_h'), 1)} h**\n"
    f"- **Maintenance recommandée** : **{row.get('maintenance_type', '—')}**\n"
    f"- **Statut thermique** : **{row.get('thermal_status', '—')}**\n"
)

if row.get("decision_reason"):
    st.caption(str(row["decision_reason"]))

st.markdown("#### Actions suggérées")
for a in suggested_actions(float(row["beta_opt"]) if _is_pos_number(row.get("beta_opt")) else 1.0):
    st.markdown(f"- {a}")

tab_a, tab_b, tab_c = st.tabs(["Fiabilité", "Thermique", "Tableaux exportables"])

with tab_a:
    if rel.get("error"):
        st.error(rel["error"])
    else:
        if "reliability_summary" in tables:
            st.markdown("##### Synthèse fiabiliste")
            st.dataframe(tables["reliability_summary"], use_container_width=True, hide_index=True)
        if "process_choice" in tables:
            st.markdown("##### Choix du processus")
            st.dataframe(tables["process_choice"], use_container_width=True, hide_index=True)
        if "trend_results" in tables:
            st.markdown("##### Tests de tendance")
            st.dataframe(tables["trend_results"], use_container_width=True, hide_index=True)
        if "dependence_results" in tables:
            st.markdown("##### Tests de dépendance")
            st.dataframe(tables["dependence_results"], use_container_width=True, hide_index=True)
        if "fit_candidates" in tables and isinstance(tables["fit_candidates"], pd.DataFrame) and not tables["fit_candidates"].empty:
            st.markdown("##### Candidats et ajustements")
            st.dataframe(tables["fit_candidates"], use_container_width=True, hide_index=True)

with tab_b:
    if thermal is None:
        st.info("Pas de données thermiques disponibles pour cet équipement.")
    else:
        if "thermal_summary" in tables:
            st.markdown("##### Synthèse thermique")
            st.dataframe(tables["thermal_summary"], use_container_width=True, hide_index=True)
        if "thermal_table_indicators" in tables:
            st.markdown("##### Indicateurs thermiques")
            st.dataframe(tables["thermal_table_indicators"], use_container_width=True, hide_index=True)
        if "thermal_top5_days" in tables:
            st.markdown("##### Top 5 jours critiques")
            st.dataframe(tables["thermal_top5_days"], use_container_width=True, hide_index=True)

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

with tab_c:
    for key in [
        "trend_results", "dependence_results", "process_choice", "fit_candidates",
        "reliability_summary", "thermal_table_dataset", "thermal_table_params",
        "thermal_table_indicators", "thermal_summary", "thermal_daily", "thermal_top5_days",
    ]:
        df_table = tables.get(key)
        if isinstance(df_table, pd.DataFrame) and not df_table.empty:
            st.markdown(f"##### {key}")
            st.dataframe(df_table, use_container_width=True, hide_index=True)

st.divider()
st.subheader("📄 Rapport PDF — Optimisation")

if export_optimization_report_pdf is None:
    st.info("Module core.reliability.reporting_optimize non détecté ou non compatible.")
    if _pdf_import_error is not None:
        st.caption(f"Détail import: {_pdf_import_error}")
else:
    if st.button("📄 Générer rapport optimisation (PDF)"):
        try:
            out_dir = str(BASE_DIR / "reports")
            intervals = {}
            for eq in fits.keys():
                intervals[eq] = {
                    "T_R": intervals_R.get(eq),
                    "T_cost": intervals_cost.get(eq),
                    "R_at_T": R_at_cost.get(eq),
                    "C_min": C_min_map.get(eq),
                }

            org_results_compat = {
                eq: (org_results.get(eq, {}) or {}).get("reliability", {})
                for eq in org_results.keys()
            }

            try:
                path = export_optimization_report_pdf(
                    df=df,
                    fits=fits,
                    intervals=intervals,
                    organigram_by_eq=org_results_compat,
                    out_dir=out_dir,
                    df_out=df_out,
                    meta={
                        "R_target": R_target,
                        "C_prev": C_prev,
                        "C_corr": C_corr,
                        "R_min_cost": R_min_cost,
                    },
                )
            except TypeError:
                path = export_optimization_report_pdf(
                    df,
                    fits,
                    intervals,
                    org_results_compat,
                    out_dir=out_dir,
                )

            st.session_state["opt_pdf_path"] = path
            st.success(f"PDF généré : {path}")
        except Exception as e:
            st.error(f"PDF : {e}")

    pdf_path = st.session_state.get("opt_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Télécharger le PDF optimisation",
                data=f,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )
