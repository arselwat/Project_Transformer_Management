from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAILURES_FILE = DATA_DIR / "failures_saved.csv"
PROJECT_DIR = DATA_DIR / "current_project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_META_FILE = PROJECT_DIR / "project_meta.json"

REQUIRED_COLS = {"equipment_code", "ttf_h"}
PROJECT_SHEETS = [
    "asset_info",
    "events_history",
    "thermal_timeseries",
    "thermal_params",
    "maintenance_policies",
    "analysis_settings",
    "failures_ttf",
]


# ============================================================
# Helpers
# ============================================================

def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out
    return pd.DataFrame()


def _coerce_bool01(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="int64")

    if s.dtype == bool:
        return s.astype(int)

    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "oui": 1,
        "non": 0,
        "y": 1,
        "n": 0,
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


def _dataset_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    keep = [c for c in ["equipment_code", "ttf_h", "duree_rep_h"] if c in x.columns]
    x = x[keep].copy()
    if "equipment_code" in x.columns:
        x["equipment_code"] = x["equipment_code"].astype(str)
    if "ttf_h" in x.columns:
        x["ttf_h"] = pd.to_numeric(x["ttf_h"], errors="coerce").round(6)
    if "duree_rep_h" in x.columns:
        x["duree_rep_h"] = pd.to_numeric(x["duree_rep_h"], errors="coerce").round(6)

    blob = x.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _project_hash(frames: Dict[str, pd.DataFrame]) -> str:
    chunks = []
    for name in sorted(frames.keys()):
        df = _safe_df(frames[name])
        chunks.append(f"##{name}\n")
        chunks.append(df.to_csv(index=False))
    blob = "".join(chunks).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ============================================================
# Failures dataset
# ============================================================

def _clean_failures_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if not REQUIRED_COLS.issubset(set(df.columns)):
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    if "duree_rep_h" not in df.columns:
        df["duree_rep_h"] = None

    df["equipment_code"] = df["equipment_code"].astype(str)
    df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
    df["duree_rep_h"] = pd.to_numeric(df["duree_rep_h"], errors="coerce")

    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0].reset_index(drop=True)
    return df


def set_current_failures_df(
    df: pd.DataFrame,
    source_name: str = "unknown",
    persist: bool = True,
) -> Dict[str, Any]:
    df2 = _clean_failures_df(df)
    if df2.empty:
        return {
            "ok": False,
            "msg": "Dataset vide ou invalide. Il faut equipment_code et ttf_h (>0).",
        }

    if persist:
        FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(FAILURES_FILE, index=False, encoding="utf-8")

    h = _dataset_hash(df2)
    st.session_state["failures_df"] = df2
    st.session_state["failures_hash"] = h
    st.session_state["failures_source"] = source_name

    return {
        "ok": True,
        "rows": int(len(df2)),
        "hash": h,
        "file": str(FAILURES_FILE),
    }


def get_current_failures_df() -> pd.DataFrame:
    if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
        df = _clean_failures_df(st.session_state["failures_df"])
        if not df.empty:
            if not st.session_state.get("failures_hash"):
                st.session_state["failures_hash"] = _dataset_hash(df)
            return df

    if not FAILURES_FILE.exists():
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    try:
        df = pd.read_csv(FAILURES_FILE)
    except Exception:
        try:
            df = pd.read_csv(FAILURES_FILE, engine="python", on_bad_lines="skip", sep=None)
        except Exception:
            return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])

    df = _clean_failures_df(df)
    if df.empty:
        return df

    st.session_state["failures_df"] = df
    st.session_state["failures_hash"] = _dataset_hash(df)
    st.session_state["failures_source"] = f"file:{FAILURES_FILE.name}"
    return df


def get_failures_meta() -> Dict[str, Any]:
    df = get_current_failures_df()
    if df.empty:
        return {
            "ok": False,
            "rows": 0,
            "hash": "",
            "source": "",
            "file": str(FAILURES_FILE),
        }

    return {
        "ok": True,
        "rows": int(len(df)),
        "hash": str(st.session_state.get("failures_hash", "")),
        "source": str(st.session_state.get("failures_source", "")),
        "file": str(FAILURES_FILE),
    }


# ============================================================
# Project normalization
# ============================================================

def _normalize_project_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    for raw_name, raw_df in (frames or {}).items():
        name = str(raw_name).strip()
        df = _safe_df(raw_df)
        lower = {str(c).lower().strip(): c for c in df.columns}
        ren = {}

        if name == "asset_info":
            aliases = {
                "assetid": "asset_id",
                "id_asset": "asset_id",
                "nom_actif": "asset_name",
                "nom": "asset_name",
                "rated_power_mva": "rated_power_mva",
                "sn_mva": "rated_power_mva",
            }
            for k, v in aliases.items():
                if k in lower and v not in df.columns:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)

        elif name == "events_history":
            aliases = {
                "date_panne": "event_start",
                "failure_time": "event_start",
                "failure_date": "event_start",
                "assetid": "asset_id",
                "repair_hours": "repair_time_hours",
                "mttr_h": "repair_time_hours",
                "downtime_h": "downtime_hours",
            }
            for k, v in aliases.items():
                if k in lower and v not in df.columns:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)

            for c in ["event_start", "event_end"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")

            for c in ["is_failure", "is_planned"]:
                if c in df.columns:
                    df[c] = _coerce_bool01(df[c])

            for c in ["repair_time_hours", "downtime_hours", "cost_corrective_usd"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        elif name == "thermal_timeseries":
            aliases = {
                "ambient_temp_c": "temp_amb_C",
                "temp_ambiante_c": "temp_amb_C",
                "temperature_ambiante": "temp_amb_C",
                "fan_status": "etat_ventilateurs",
                "fans_status": "etat_ventilateurs",
                "ventilateurs": "etat_ventilateurs",
                "load_pct": "charge_pct",
            }
            for k, v in aliases.items():
                if k in lower and v not in df.columns:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            for c in [
                "temp_amb_C",
                "K",
                "charge_pct",
                "load_factor",
                "load_mva",
                "etat_ventilateurs",
                "temp_cuve_C",
                "current_a",
                "top_oil_temp_c",
                "hotspot_temp_c",
            ]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        elif name == "thermal_params":
            aliases = {
                "delta_theta_to_r": "delta_to_r",
                "delta_theta_h_r": "delta_h_r",
                "tau_to_hours": "tau_to_hours",
                "tau_h_hours": "tau_h_hours",
                "normal_life_hours": "normal_insulation_life_h",
                "rated_power_mva": "sn_mva",
            }
            for k, v in aliases.items():
                if k in lower and v not in df.columns:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)

            for c in df.columns:
                if c != "asset_id":
                    try:
                        df[c] = pd.to_numeric(df[c], errors="ignore")
                    except Exception:
                        pass

            if "tau_to_hours" in df.columns and "tau_to_min" not in df.columns:
                df["tau_to_min"] = pd.to_numeric(df["tau_to_hours"], errors="coerce") * 60.0

            if "tau_h_hours" in df.columns and "tau_w_min" not in df.columns:
                df["tau_w_min"] = pd.to_numeric(df["tau_h_hours"], errors="coerce") * 60.0

        elif name == "maintenance_policies":
            for c in [
                "interval_days",
                "reliability_target",
                "cost_preventive_usd",
                "cost_corrective_usd",
                "cost_downtime_usd",
                "thermal_limit",
            ]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        elif name == "analysis_settings":
            aliases = {
                "alpha": "alpha_significance",
                "horizon_days": "analysis_horizon_days",
            }
            for k, v in aliases.items():
                if k in lower and v not in df.columns:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)

            if "alpha_significance" in df.columns:
                df["alpha_significance"] = pd.to_numeric(df["alpha_significance"], errors="coerce")
            if "analysis_horizon_days" in df.columns:
                df["analysis_horizon_days"] = pd.to_numeric(df["analysis_horizon_days"], errors="coerce")

        out[name] = df

    # enrichissement sn_mva depuis asset_info
    if "asset_info" in out and "thermal_params" in out:
        a = out["asset_info"].copy()
        t = out["thermal_params"].copy()
        if "asset_id" in a.columns and "asset_id" in t.columns:
            if ("sn_mva" not in t.columns or t["sn_mva"].isna().all()) and "rated_power_mva" in a.columns:
                merge = a[["asset_id", "rated_power_mva"]].copy().rename(columns={"rated_power_mva": "sn_mva_from_asset"})
                t = t.merge(merge, on="asset_id", how="left")
                if "sn_mva" not in t.columns:
                    t["sn_mva"] = t["sn_mva_from_asset"]
                else:
                    t["sn_mva"] = pd.to_numeric(t["sn_mva"], errors="coerce").fillna(t["sn_mva_from_asset"])
                t = t.drop(columns=[c for c in ["sn_mva_from_asset"] if c in t.columns])
                out["thermal_params"] = t

    return out


# ============================================================
# Public project builder
# ============================================================

def build_ttf_from_events(events: pd.DataFrame) -> pd.DataFrame:
    df = _safe_df(events)

    needed = {"event_id", "asset_id", "event_start", "is_failure"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h", "failure_time", "event_id"])

    if "repair_time_hours" not in df.columns:
        df["repair_time_hours"] = None

    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df = df.dropna(subset=["event_start"]).copy()

    df["asset_id"] = df["asset_id"].astype(str)
    df["is_failure"] = _coerce_bool01(df["is_failure"])
    df["repair_time_hours"] = pd.to_numeric(df["repair_time_hours"], errors="coerce")

    df = df[df["is_failure"] == 1].sort_values(["asset_id", "event_start"]).reset_index(drop=True)

    out = []
    for asset_id, g in df.groupby("asset_id"):
        g = g.sort_values("event_start").reset_index(drop=True)
        for i in range(1, len(g)):
            dt_h = (g.loc[i, "event_start"] - g.loc[i - 1, "event_start"]).total_seconds() / 3600.0
            if dt_h > 0:
                out.append(
                    {
                        "equipment_code": str(asset_id),
                        "ttf_h": float(dt_h),
                        "duree_rep_h": g.loc[i, "repair_time_hours"],
                        "failure_time": g.loc[i, "event_start"],
                        "event_id": g.loc[i, "event_id"],
                    }
                )

    if not out:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h", "failure_time", "event_id"])

    return pd.DataFrame(out)


def set_current_project_data(
    frames: Dict[str, pd.DataFrame],
    source_name: str = "unknown",
    persist: bool = True,
    sync_failures: bool = True,
) -> Dict[str, Any]:
    frames = _normalize_project_frames(frames)
    frames = {k: _safe_df(v) for k, v in frames.items()}

    for name in ["asset_info", "events_history", "thermal_timeseries", "thermal_params", "maintenance_policies", "analysis_settings"]:
        frames.setdefault(name, pd.DataFrame())

    failures_ttf = build_ttf_from_events(frames.get("events_history", pd.DataFrame()))
    frames["failures_ttf"] = failures_ttf

    h = _project_hash(frames)

    st.session_state["project_data"] = frames
    st.session_state["project_hash"] = h
    st.session_state["project_source"] = source_name

    if persist:
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_csv(PROJECT_DIR / f"{name}.csv", index=False, encoding="utf-8")

        meta = {
            "ok": True,
            "hash": h,
            "source": source_name,
            "sheets": list(frames.keys()),
            "rows": {k: int(len(v)) for k, v in frames.items()},
        }
        PROJECT_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if sync_failures and not failures_ttf.empty:
        set_current_failures_df(
            failures_ttf[["equipment_code", "ttf_h", "duree_rep_h"]],
            source_name=f"{source_name}:events_history",
            persist=persist,
        )

    return {
        "ok": True,
        "hash": h,
        "source": source_name,
        "sheets": list(frames.keys()),
        "rows": {k: int(len(v)) for k, v in frames.items()},
        "failures_rows": int(len(failures_ttf)),
        "meta_file": str(PROJECT_META_FILE),
    }


def _load_project_frames_from_disk() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    if not PROJECT_DIR.exists():
        return frames

    for name in PROJECT_SHEETS:
        p = PROJECT_DIR / f"{name}.csv"
        if p.exists():
            try:
                frames[name] = pd.read_csv(p)
            except Exception:
                frames[name] = pd.DataFrame()

    return _normalize_project_frames(frames)


def get_current_project_data() -> Dict[str, pd.DataFrame]:
    proj = st.session_state.get("project_data")
    if isinstance(proj, dict) and proj:
        return {k: _safe_df(v) for k, v in proj.items()}

    frames = _load_project_frames_from_disk()
    if frames:
        st.session_state["project_data"] = frames
        if not st.session_state.get("project_hash"):
            st.session_state["project_hash"] = _project_hash(frames)
        if not st.session_state.get("project_source"):
            st.session_state["project_source"] = "file:current_project"
        return frames

    return {name: pd.DataFrame() for name in PROJECT_SHEETS}


def get_project_meta() -> Dict[str, Any]:
    proj = get_current_project_data()
    has_any = any(isinstance(df, pd.DataFrame) and not df.empty for df in proj.values())

    if not has_any:
        return {
            "ok": False,
            "rows": {},
            "hash": "",
            "source": "",
            "dir": str(PROJECT_DIR),
        }

    return {
        "ok": True,
        "rows": {k: int(len(v)) for k, v in proj.items()},
        "hash": str(st.session_state.get("project_hash", "")),
        "source": str(st.session_state.get("project_source", "")),
        "dir": str(PROJECT_DIR),
    }


def clear_current_project_data(clear_failures: bool = False) -> None:
    for name in PROJECT_SHEETS:
        p = PROJECT_DIR / f"{name}.csv"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    if PROJECT_META_FILE.exists():
        try:
            PROJECT_META_FILE.unlink()
        except Exception:
            pass

    st.session_state.pop("project_data", None)
    st.session_state.pop("project_hash", None)
    st.session_state.pop("project_source", None)

    if clear_failures:
        if FAILURES_FILE.exists():
            try:
                FAILURES_FILE.unlink()
            except Exception:
                pass
        st.session_state.pop("failures_df", None)
        st.session_state.pop("failures_hash", None)
        st.session_state.pop("failures_source", None)


# ============================================================
# Unified pipeline bundle
# ============================================================

def get_pipeline_inputs(asset_id: Optional[str] = None) -> Dict[str, Any]:
    proj = get_current_project_data()
    failures_df = get_current_failures_df()

    asset_info = _safe_df(proj.get("asset_info"))
    events_history = _safe_df(proj.get("events_history"))
    thermal_timeseries = _safe_df(proj.get("thermal_timeseries"))
    thermal_params = _safe_df(proj.get("thermal_params"))
    maintenance_policies = _safe_df(proj.get("maintenance_policies"))
    analysis_settings = _safe_df(proj.get("analysis_settings"))
    failures_ttf = _clean_failures_df(_safe_df(proj.get("failures_ttf")))

    if failures_ttf.empty and not failures_df.empty:
        failures_ttf = failures_df.copy()

    selected_asset = asset_id
    if not selected_asset:
        if not asset_info.empty and "asset_id" in asset_info.columns:
            selected_asset = str(asset_info.iloc[0]["asset_id"])
        elif not failures_ttf.empty and "equipment_code" in failures_ttf.columns:
            selected_asset = str(failures_ttf.iloc[0]["equipment_code"])

    if selected_asset:
        if not asset_info.empty and "asset_id" in asset_info.columns:
            asset_info = asset_info[asset_info["asset_id"].astype(str) == str(selected_asset)].copy()

        if not events_history.empty and "asset_id" in events_history.columns:
            events_history = events_history[events_history["asset_id"].astype(str) == str(selected_asset)].copy()

        if not thermal_timeseries.empty and "asset_id" in thermal_timeseries.columns:
            thermal_timeseries = thermal_timeseries[
                thermal_timeseries["asset_id"].astype(str) == str(selected_asset)
            ].copy()

        if not thermal_params.empty and "asset_id" in thermal_params.columns:
            thermal_params = thermal_params[
                thermal_params["asset_id"].astype(str) == str(selected_asset)
            ].copy()

        if not maintenance_policies.empty and "asset_id" in maintenance_policies.columns:
            maintenance_policies = maintenance_policies[
                maintenance_policies["asset_id"].astype(str) == str(selected_asset)
            ].copy()

        if not analysis_settings.empty and "asset_id" in analysis_settings.columns:
            analysis_settings = analysis_settings[
                analysis_settings["asset_id"].astype(str) == str(selected_asset)
            ].copy()

        if not failures_ttf.empty and "equipment_code" in failures_ttf.columns:
            failures_ttf = failures_ttf[
                failures_ttf["equipment_code"].astype(str) == str(selected_asset)
            ].copy()

    ttf_series = []
    repair_series = []

    if not failures_ttf.empty:
        ttf_series = pd.to_numeric(failures_ttf["ttf_h"], errors="coerce").dropna().tolist()
        if "duree_rep_h" in failures_ttf.columns:
            repair_series = pd.to_numeric(failures_ttf["duree_rep_h"], errors="coerce").dropna().tolist()

    thermal_df = thermal_timeseries.copy()

    thermal_config: Dict[str, Any] = {}
    if not thermal_params.empty:
        row = thermal_params.iloc[0].to_dict()
        allowed_keys = {
            "sn_mva",
            "R",
            "delta_to_r",
            "delta_h_r",
            "tau_to_min",
            "tau_w_min",
            "n_exp",
            "m_exp",
            "forced_tau_to_factor",
            "forced_delta_to_factor",
            "forced_delta_h_factor",
            "normal_insulation_life_h",
        }
        for k, v in row.items():
            if k in allowed_keys and pd.notna(v):
                thermal_config[k] = v

    thermal_config.setdefault("sn_mva", 100.0)
    thermal_config.setdefault("R", 5.0)
    thermal_config.setdefault("delta_to_r", 55.0)
    thermal_config.setdefault("delta_h_r", 30.0)
    thermal_config.setdefault("tau_to_min", 180.0)
    thermal_config.setdefault("tau_w_min", 10.0)
    thermal_config.setdefault("n_exp", 0.8)
    thermal_config.setdefault("m_exp", 0.8)
    thermal_config.setdefault("forced_tau_to_factor", 0.75)
    thermal_config.setdefault("forced_delta_to_factor", 0.92)
    thermal_config.setdefault("forced_delta_h_factor", 0.92)
    thermal_config.setdefault("normal_insulation_life_h", 180000.0)

    alpha = 0.05
    if not analysis_settings.empty and "alpha_significance" in analysis_settings.columns:
        try:
            alpha = float(analysis_settings.iloc[0]["alpha_significance"])
        except Exception:
            alpha = 0.05

    return {
        "asset_id": selected_asset,
        "asset_info": asset_info,
        "events_history": events_history,
        "failures_ttf": failures_ttf,
        "ttf_series": ttf_series,
        "repair_series": repair_series,
        "thermal_df": thermal_df if not thermal_df.empty else None,
        "thermal_config": thermal_config if thermal_config else None,
        "maintenance_policies": maintenance_policies,
        "analysis_settings": analysis_settings,
        "alpha": alpha,
        "project_data": proj,
        "failures_meta": get_failures_meta(),
        "project_meta": get_project_meta(),
    }