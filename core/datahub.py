from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
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

DATAHUB_VERSION = "2.0"
FAILURES_META_FILE = DATA_DIR / "failures_meta.json"

REQUIRED_COLS = {"equipment_code", "ttf_h"}
PROJECT_SHEETS = [
    "asset_info",
    "events_history",
    "thermal_timeseries",
    "thermal_params",
    "maintenance_policies",
    "analysis_settings",
    "failures_ttf",
    "events_validated",
    "observation_windows",
    "data_quality_report",
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
    """Les codes inconnus restent manquants ; ils ne deviennent jamais 0."""
    if s is None:
        return pd.Series(dtype="Int64")
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0,
               "yes": 1, "no": 0, "oui": 1, "non": 0, "y": 1, "n": 0}
    out = s.astype("string").str.strip().str.lower().map(mapping)
    numeric = pd.to_numeric(s, errors="coerce")
    return out.fillna(numeric.where(numeric.isin([0, 1]))).astype("Int64")


def _invalidate_results() -> None:
    # Clés de résultats connues. Les widgets et les données métier sont conservés.
    for key in ("optimization_df", "optimization_src", "opt_meta", "opt_pdf_path"):
        st.session_state.pop(key, None)


def _finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))


def _dates(s: pd.Series) -> pd.Series:
    # Parsing scalaire pour accepter les formats ISO mixtes sans heuristique jour/mois.
    # Les dates sans fuseau sont supposées dans une même base locale.
    values = s.map(lambda v: pd.to_datetime(v, errors="coerce"))
    aware = values.map(lambda v: pd.notna(v) and getattr(v, "tzinfo", None) is not None)
    if aware.any() and not aware[values.notna()].all():
        raise ValueError("Dates avec et sans fuseau mélangées : uniformiser les horodatages.")
    if aware.any():
        return pd.to_datetime(values, utc=True).dt.tz_localize(None)
    return pd.to_datetime(values)


def _issue(log: list, row: Any, code: str, message: str, asset: Any = "") -> None:
    log.append({"source_row": str(row), "asset_id": str(asset),
                "code": code, "message": message})


def _dataset_hash(df: pd.DataFrame) -> str:
    """Inclut dates, conventions et contexte ; aucune réduction aux seuls TBF."""
    if df is None or df.empty:
        return ""
    blob = DATAHUB_VERSION + "\n" + df.to_csv(index=False)
    blob += json.dumps(df.attrs.get("analysis_context", {}), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _project_hash(frames: Dict[str, pd.DataFrame]) -> str:
    chunks = []
    for name in sorted(frames.keys()):
        df = _safe_df(frames[name])
        chunks.append(f"##{name}\n")
        chunks.append(df.to_csv(index=False))
    blob = (DATAHUB_VERSION + "".join(chunks)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ============================================================
# Failures dataset
# ============================================================

def _clean_failures_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _safe_df(df)
    log = list(out.attrs.get("quality_report", []))
    if out.empty:
        empty = pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])
        empty.attrs = dict(out.attrs)
        empty.attrs["quality_report"] = log
        return empty
    missing = REQUIRED_COLS - set(out.columns)
    if missing:
        empty = pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])
        _issue(log, "", "missing_columns", "Colonnes absentes : " + ", ".join(sorted(missing)))
        empty.attrs["quality_report"] = log
        return empty
    out["equipment_code"] = out["equipment_code"].astype("string").str.strip()
    valid_id = out["equipment_code"].notna() & out["equipment_code"].ne("")
    out["ttf_h"] = _finite_numeric(out["ttf_h"])
    valid_ttf = out["ttf_h"].notna() & out["ttf_h"].gt(0)
    for idx in out.index[~(valid_id & valid_ttf)]:
        _issue(log, idx, "excluded_interval", "Identifiant absent ou intervalle non fini/non positif.")
    if "duree_rep_h" not in out:
        out["duree_rep_h"] = float("nan")
    original = out["duree_rep_h"].copy()
    repair = _finite_numeric(original)
    bad = (original.notna() & repair.isna()) | repair.lt(0)
    for idx in out.index[bad]:
        _issue(log, idx, "invalid_repair", "Durée de réparation invalide remplacée par une valeur manquante.")
    out["duree_rep_h"] = repair.mask(bad)
    if "time_basis" not in out:
        out["time_basis"] = "unspecified"
    out = out.loc[valid_id & valid_ttf].reset_index(drop=True)
    out.attrs["quality_report"] = log
    return out


def set_current_failures_df(
    df: pd.DataFrame, source_name: str = "unknown", persist: bool = True,
) -> Dict[str, Any]:
    df2 = _clean_failures_df(df)
    log = df2.attrs.get("quality_report", [])
    context = df2.attrs.get("analysis_context", {})
    h = _dataset_hash(df2)
    if st.session_state.get("failures_hash") != h:
        _invalidate_results()
    meta = {"ok": not df2.empty, "rows": len(df2), "hash": h,
            "source": source_name, "file": str(FAILURES_FILE),
            "quality_report": log, "analysis_context": context,
            "datahub_version": DATAHUB_VERSION}
    if df2.empty:
        meta["msg"] = "Aucun intervalle exploitable. Consulter quality_report."
    if persist:
        FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(FAILURES_FILE, index=False, encoding="utf-8")
        FAILURES_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # Une importation vide remplace aussi le jeu actif : jamais de retour à un ancien jeu.
    st.session_state.update({"failures_df": df2, "failures_hash": h,
                             "failures_source": source_name, "failures_metadata": meta})
    return meta


def get_current_failures_df() -> pd.DataFrame:
    current = st.session_state.get("failures_df")
    if isinstance(current, pd.DataFrame):
        return current.copy()
    if not FAILURES_FILE.exists():
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])
    try:
        # Ne jamais ignorer silencieusement une ligne CSV mal formée.
        df = pd.read_csv(FAILURES_FILE, dtype={"equipment_code": "string"})
        meta = json.loads(FAILURES_META_FILE.read_text(encoding="utf-8")) if FAILURES_META_FILE.exists() else {}
        df.attrs["analysis_context"] = meta.get("analysis_context", {})
        df.attrs["quality_report"] = meta.get("quality_report", [])
        set_current_failures_df(df, meta.get("source", "file:failures_saved.csv"), persist=False)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        empty = pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h"])
        empty.attrs["quality_report"] = [{"code": "read_error", "message": str(exc)}]
        set_current_failures_df(empty, "file:failures_saved.csv", persist=False)
    return st.session_state["failures_df"].copy()


def get_failures_meta() -> Dict[str, Any]:
    df = get_current_failures_df()
    meta = dict(st.session_state.get("failures_metadata", {}))
    meta.update(ok=not df.empty, rows=len(df),
                hash=st.session_state.get("failures_hash", ""),
                source=st.session_state.get("failures_source", ""), file=str(FAILURES_FILE))
    return meta


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

            # Conversion et contrôle dans prepare_event_data ; historique brut conservé.

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

def prepare_event_data(events: pd.DataFrame, settings: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Prépare les événements sans inventer de première durée de fonctionnement.

    Référence par défaut : première panne enregistrée (pas la mise en service).
    Colonnes optionnelles de settings, par asset_id : observation_start,
    observation_end, decision_time. Sans bornes fournies, le début et la fin
    sont les événements extrêmes et cette convention est indiquée.
    ttf_h reste l'intervalle CALENDAIRE pour compatibilité des pages existantes.
    operating_time_h représente seulement la durée entre une remise en service
    documentée et la panne suivante ; d'autres arrêts peuvent s'y trouver.
    """
    df = _safe_df(events).reset_index(drop=True)
    log = []
    cols = ["equipment_code", "ttf_h", "duree_rep_h", "failure_time", "event_id",
            "previous_failure_time", "previous_event_id", "time_basis", "operating_time_h"]
    empty = pd.DataFrame(columns=cols)
    def result(intervals, valid, windows):
        report = pd.DataFrame(log, columns=["source_row", "asset_id", "code", "message"])
        intervals.attrs["quality_report"] = log
        return {"failures_ttf": intervals, "events_validated": valid,
                "observation_windows": windows, "data_quality_report": report}
    missing = {"asset_id", "event_start", "is_failure"} - set(df)
    if df.empty or missing:
        _issue(log, "", "missing_events", "Historique vide ou colonnes absentes : " + ", ".join(sorted(missing)))
        return result(empty, pd.DataFrame(), pd.DataFrame())
    df["source_row"] = df.index + 2  # ligne de fichier avec en-tête
    if "event_id" not in df:
        df["event_id"] = ["row_" + str(i) for i in df.source_row]
        _issue(log, "", "generated_ids", "Identifiants techniques générés à partir des lignes source.")
    df["asset_id"] = df.asset_id.astype("string").str.strip()
    try:
        df["event_start"] = _dates(df.event_start)
        if "event_end" in df:
            df["event_end"] = _dates(df.event_end)
    except ValueError as exc:
        _issue(log, "", "ambiguous_timezone", str(exc))
        return result(empty, pd.DataFrame(), pd.DataFrame())
    df["is_failure"] = _coerce_bool01(df.is_failure)
    planned_supplied = "is_planned" in df
    df["is_planned"] = _coerce_bool01(df.is_planned) if planned_supplied else 0
    for col in ("repair_time_hours", "downtime_hours"):
        if col not in df:
            df[col] = float("nan")
        original = df[col].copy()
        values = _finite_numeric(original)
        bad = (original.notna() & values.isna()) | values.lt(0)
        for i in df.index[bad]:
            _issue(log, df.at[i, "source_row"], "invalid_duration", col + " invalide ; conservé manquant.", df.at[i, "asset_id"])
        df[col] = values.mask(bad)
    valid_ids = df.asset_id.notna() & df.asset_id.ne("")
    for i in df.index[~valid_ids]:
        _issue(log, df.at[i, "source_row"], "missing_asset", "Ligne exclue : équipement non identifié.")
    df = df.loc[valid_ids].copy()
    output, validated, windows = [], [], []
    settings = _safe_df(settings)
    for asset, group in df.groupby("asset_id", sort=False):
        g = group.copy()
        invalid = g.event_start.isna() | g.is_failure.isna() | g.is_planned.isna()
        if invalid.any():
            for i in g.index[invalid]:
                _issue(log, g.at[i, "source_row"], "blocked_asset", "Date ou indicateur ambigu : équipement non analysé pour éviter de relier artificiellement des pannes.", asset)
            continue
        duplicate = g.drop(columns="source_row").duplicated()
        for i in g.index[duplicate]:
            _issue(log, g.at[i, "source_row"], "exact_duplicate", "Doublon exact exclu.", asset)
        g = g.loc[~duplicate].copy()
        known_id = g.event_id.notna() & g.event_id.astype("string").str.strip().ne("")
        if g.loc[known_id, "event_id"].duplicated().any():
            _issue(log, "", "conflicting_id", "Identifiant événement répété avec contenu différent : équipement bloqué.", asset)
            continue
        for i in g.index[~known_id]:
            g.at[i, "event_id"] = "row_" + str(g.at[i, "source_row"])
            _issue(log, g.at[i, "source_row"], "generated_id", "Identifiant technique généré.", asset)
        selected = g.is_failure.eq(1) & g.is_planned.eq(0)
        for i in g.index[~selected]:
            _issue(log, g.at[i, "source_row"], "excluded_non_failure", "Événement programmé ou non défaillant, hors comptage des pannes.", asset)
        g = g.loc[selected].sort_values("event_start", kind="stable").reset_index(drop=True)
        if g.empty:
            continue
        if g.event_start.duplicated().any():
            _issue(log, "", "simultaneous_events", "Plusieurs pannes au même instant : consolider l'historique avant analyse.", asset)
            continue
        settings_asset = settings
        if "asset_id" in settings:
            settings_asset = settings[settings.asset_id.astype("string").eq(str(asset))]
        elif len(settings) > 1:
            _issue(log, "", "ambiguous_settings", "Plusieurs réglages sans asset_id : équipement bloqué.", asset)
            continue
        if len(settings_asset) > 1:
            _issue(log, "", "ambiguous_settings", "Plusieurs réglages pour le même équipement.", asset)
            continue
        config = settings_asset.iloc[0].to_dict() if len(settings_asset) else {}
        first, last = g.event_start.iloc[0], g.event_start.iloc[-1]
        try:
            def boundary(key, default):
                value = config.get(key)
                if value is None or pd.isna(value) or str(value).strip() == "":
                    return default
                parsed = _dates(pd.Series([value])).iloc[0]
                if pd.isna(parsed):
                    raise ValueError("Borne temporelle invalide : " + key)
                return parsed
            start = boundary("observation_start", first)
            end = boundary("observation_end", last)
            decision = boundary("decision_time", end)
            if start > first or end < last or decision < end:
                raise ValueError("Exiger début <= première panne, fin >= dernière panne, décision >= fin.")
        except ValueError as exc:
            _issue(log, "", "invalid_window", str(exc), asset)
            continue
        explicit_start = pd.notna(config.get("observation_start")) and str(config.get("observation_start", "")).strip() != ""
        explicit_end = pd.notna(config.get("observation_end")) and str(config.get("observation_end", "")).strip() != ""
        window = {"asset_id": str(asset), "observation_start": start,
                  "observation_end": end, "decision_time": decision,
                  "first_failure_time": first, "last_failure_time": last,
                  "time_origin": first, "origin_convention": "first_recorded_failure",
                  "start_source": "provided" if explicit_start else "first_recorded_failure",
                  "end_source": "provided" if explicit_end else "last_recorded_failure",
                  "time_basis": "calendar", "unit": "hours",
                  "failure_count": len(g), "interval_count": max(0, len(g)-1),
                  "exposure_h": (end-start).total_seconds()/3600,
                  "history_time_h": (last-first).total_seconds()/3600,
                  "decision_time_h": (decision-first).total_seconds()/3600,
                  "right_censor_h": (end-last).total_seconds()/3600,
                  "pre_first_observation_h": (first-start).total_seconds()/3600,
                  "unobserved_after_end_h": (decision-end).total_seconds()/3600}
        windows.append(window)
        g["return_to_service_time"] = g.get("event_end", pd.Series(pd.NaT, index=g.index))
        for i in g.index:
            rt = g.at[i, "return_to_service_time"]
            if pd.notna(rt) and rt < g.at[i, "event_start"]:
                _issue(log, g.at[i, "source_row"], "invalid_return", "Remise en service antérieure à la panne ; date écartée.", asset)
                g.at[i, "return_to_service_time"] = pd.NaT
            elif pd.notna(rt):
                measured = (rt-g.at[i, "event_start"]).total_seconds()/3600
                given = g.at[i, "downtime_hours"]
                if pd.notna(given) and abs(given-measured) > 1e-6:
                    _issue(log, g.at[i, "source_row"], "downtime_mismatch", "Durée d'arrêt déclarée différente des horodatages ; les deux sont conservées.", asset)
        g["event_time_h"] = (g.event_start-first).dt.total_seconds()/3600
        validated.append(g)
        for i in range(1, len(g)):
            prev, row = g.iloc[i-1], g.iloc[i]
            gap = (row.event_start-prev.event_start).total_seconds()/3600
            uptime = float("nan")
            if pd.notna(prev.return_to_service_time):
                uptime = (row.event_start-prev.return_to_service_time).total_seconds()/3600
                if uptime < 0:
                    _issue(log, row.source_row, "overlapping_outage", "Panne avant remise en service précédente ; durée de fonctionnement non calculable.", asset)
                    uptime = float("nan")
            output.append({"equipment_code": str(asset), "ttf_h": gap,
                           "duree_rep_h": row.repair_time_hours,
                           "failure_time": row.event_start, "event_id": row.event_id,
                           "previous_failure_time": prev.event_start, "previous_event_id": prev.event_id,
                           "return_to_service_time": row.return_to_service_time,
                           "previous_return_to_service_time": prev.return_to_service_time,
                           "downtime_hours": row.downtime_hours,
                           "operating_time_h": uptime, "time_basis": "calendar",
                           "event_time_h": row.event_time_h, "time_origin": first,
                           "observation_start": start, "observation_end": end,
                           "decision_time": decision, "right_censor_h": window["right_censor_h"]})
    return result(pd.DataFrame(output) if output else empty,
                  pd.concat(validated, ignore_index=True) if validated else pd.DataFrame(),
                  pd.DataFrame(windows))


def build_ttf_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """API historique conservée ; le rapport de contrôle est dans DataFrame.attrs."""
    return prepare_event_data(events)["failures_ttf"]


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

    prepared = prepare_event_data(frames.get("events_history", pd.DataFrame()), frames.get("analysis_settings"))
    frames.update(prepared)
    failures_ttf = frames["failures_ttf"]

    h = _project_hash(frames)

    if st.session_state.get("project_hash") != h:
        _invalidate_results()
    st.session_state["project_data"] = frames
    st.session_state["project_hash"] = h
    st.session_state["project_source"] = source_name

    if persist:
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Nom de feuille invalide : " + name)
            df.to_csv(PROJECT_DIR / f"{name}.csv", index=False, encoding="utf-8")

        meta = {
            "ok": True,
            "hash": h,
            "source": source_name,
            "sheets": list(frames.keys()),
            "rows": {k: int(len(v)) for k, v in frames.items()},
        }
        PROJECT_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if sync_failures:
        failures_ttf.attrs["analysis_context"] = {"project_hash": h,
            "windows": frames["observation_windows"].to_dict(orient="records")}
        set_current_failures_df(
            failures_ttf,
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
                frames[name] = pd.read_csv(p, dtype={"asset_id": "string", "equipment_code": "string", "event_id": "string"})
            except Exception:
                frames[name] = pd.DataFrame()

    frames = _normalize_project_frames(frames)
    if not frames.get("events_history", pd.DataFrame()).empty:
        frames.update(prepare_event_data(frames["events_history"], frames.get("analysis_settings")))
    return frames


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

    _invalidate_results()
    if clear_failures:
        FAILURES_META_FILE.unlink(missing_ok=True)
        st.session_state.pop("failures_metadata", None)
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

    events_validated = _safe_df(proj.get("events_validated"))
    observation_windows = _safe_df(proj.get("observation_windows"))
    quality_report = _safe_df(proj.get("data_quality_report"))
    if selected_asset:
        if "asset_id" in events_validated:
            events_validated = events_validated[events_validated.asset_id.astype(str).eq(str(selected_asset))].copy()
        if "asset_id" in observation_windows:
            observation_windows = observation_windows[observation_windows.asset_id.astype(str).eq(str(selected_asset))].copy()
    # MTTR : toutes les réparations observées, y compris celle de l'événement de référence.
    # repair_series est une série descriptive indépendante, pas un vecteur apparié aux TBF.
    if not events_validated.empty and "repair_time_hours" in events_validated:
        repair_series = _finite_numeric(events_validated.repair_time_hours).dropna().tolist()
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
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        alpha = 0.05

    return {
        "asset_id": selected_asset,
        "asset_info": asset_info,
        "events_history": events_history,
        "failures_ttf": failures_ttf,
        "events_validated": events_validated,
        "observation_windows": observation_windows,
        "data_quality_report": quality_report,
        "time_basis": "calendar" if not events_validated.empty else "unspecified",
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
