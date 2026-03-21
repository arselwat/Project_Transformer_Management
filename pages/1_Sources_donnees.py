
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.datahub import set_current_failures_df, get_failures_meta, get_current_failures_df

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
require_login()

st.title("📥 Sources de données")
st.caption(
    "Charge soit un CSV simple de TTF, soit un fichier Excel projet (.xlsx) multi-feuilles. "
    "Les TTF dérivés sont synchronisés automatiquement pour les pages Indicateurs / Optimisation / Maintenance."
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

DB_PATH = DATA_DIR / "reliability.sqlite"
PROJECT_DIR = DATA_DIR / "current_project"
PROJECT_DIR.mkdir(exist_ok=True, parents=True)
PROJECT_META_PATH = PROJECT_DIR / "project_meta.json"

# ------------------------- DB helpers -------------------------
def _db_conn():
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _db_conn() as cx:
        cx.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              equipment_code TEXT NOT NULL,
              ttf_h REAL NOT NULL,
              duree_rep_h REAL,
              source TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cx.commit()


def bulk_insert_failures(df: pd.DataFrame, source: str):
    if df is None or df.empty:
        return 0
    df = df.copy()
    for c in ["equipment_code", "ttf_h", "duree_rep_h"]:
        if c not in df.columns:
            df[c] = None
    df["equipment_code"] = df["equipment_code"].astype(str)
    df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
    df["duree_rep_h"] = pd.to_numeric(df["duree_rep_h"], errors="coerce")
    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0]

    rows = df[["equipment_code", "ttf_h", "duree_rep_h"]].values.tolist()
    with _db_conn() as cx:
        cx.executemany(
            "INSERT INTO failures (equipment_code, ttf_h, duree_rep_h, source) VALUES (?,?,?,?)",
            [(*r, source) for r in rows],
        )
        cx.commit()
    return len(rows)


def clear_db():
    with _db_conn() as cx:
        cx.execute("DELETE FROM failures")
        cx.commit()


init_db()

# ------------------------- Specs projet -------------------------
REQUIRED_SIMPLE = ["equipment_code", "ttf_h"]
PROJECT_SHEETS = {
    "asset_info": ["asset_id", "asset_name"],
    "events_history": ["event_id", "asset_id", "event_start", "event_type", "is_failure", "is_planned"],
    "thermal_timeseries": ["timestamp", "asset_id", "ambient_temp_c"],
    "thermal_params": [
        "asset_id",
        "delta_theta_to_r",
        "delta_theta_h_r",
        "R",
        "n_exp",
        "m_exp",
        "tau_to_hours",
        "tau_h_hours",
        "normal_life_hours",
    ],
    "maintenance_policies": [
        "policy_id",
        "asset_id",
        "policy_name",
        "policy_type",
        "reliability_target",
        "cost_preventive_usd",
        "cost_corrective_usd",
        "cost_downtime_usd",
        "thermal_constraint_type",
        "thermal_limit",
    ],
    "analysis_settings": ["asset_id", "time_unit", "timezone", "alpha_significance", "analysis_horizon_days"],
}

OPTIONAL_PROJECT_COLS = {
    "events_history": ["event_end", "downtime_hours", "repair_time_hours", "subsystem", "cost_corrective_usd", "description"],
    "thermal_timeseries": [
        "load_factor",
        "top_oil_rise_c",
        "hotspot_gradient_c",
        "top_oil_temp_c",
        "hotspot_temp_c",
        "current_a",
        "load_mva",
        "fan_status",
        "pump_status",
    ],
    "thermal_params": ["faa_limit", "lol_limit_hours"],
    "analysis_settings": ["allow_nhpp_powerlaw", "allow_bpp_hawkes"],
}

# ------------------------- Utils -------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code",
        "equipement": "equipment_code",
        "code_equipement": "equipment_code",
        "eqp": "equipment_code",
        "ttf": "ttf_h",
        "ttf_hours": "ttf_h",
        "mttr_h": "duree_rep_h",
        "repair_hours": "duree_rep_h",
        "failure_time": "failure_time",
        "failure_date": "failure_time",
        "date_panne": "failure_time",
    }
    cols = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for k, v in mapping.items():
        if k in cols:
            ren[cols[k]] = v
    return df.rename(columns=ren)


def _coerce_bool01(s: pd.Series) -> pd.Series:
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
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)


def _compute_ttf_from_timestamps(df: pd.DataFrame, eq_col: str, ts_col: str) -> pd.DataFrame:
    tmp = df[[eq_col, ts_col]].dropna().copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp.dropna(subset=[ts_col]).sort_values([eq_col, ts_col])

    out = []
    for eq, g in tmp.groupby(eq_col):
        t = g[ts_col].tolist()
        for i in range(1, len(t)):
            dh = (t[i] - t[i - 1]).total_seconds() / 3600.0
            if dh > 0:
                out.append(
                    {
                        "equipment_code": str(eq),
                        "ttf_h": float(dh),
                        "duree_rep_h": None,
                        "failure_time": t[i],
                    }
                )
    return pd.DataFrame(out)


def _build_ttf_from_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    for c in ["event_id", "asset_id", "event_start", "event_type", "is_failure", "is_planned"]:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans events_history: {c}")

    if "repair_time_hours" not in df.columns:
        df["repair_time_hours"] = None

    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df = df.dropna(subset=["event_start"]).copy()
    df["asset_id"] = df["asset_id"].astype(str)
    df["is_failure"] = _coerce_bool01(df["is_failure"])
    df["is_planned"] = _coerce_bool01(df["is_planned"])
    df["repair_time_hours"] = pd.to_numeric(df["repair_time_hours"], errors="coerce")

    # Uniquement les vraies pannes
    df = df[df["is_failure"] == 1].copy()
    df = df.sort_values(["asset_id", "event_start"])

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

    ttf_df = pd.DataFrame(out)
    if ttf_df.empty:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h", "failure_time", "event_id"])
    return ttf_df


def _validate_project_sheets(frames: Dict[str, pd.DataFrame]) -> list[str]:
    errors = []

    missing_sheets = [s for s in PROJECT_SHEETS if s not in frames]
    if missing_sheets:
        errors.append(f"Feuilles manquantes: {missing_sheets}")
        return errors

    for sheet_name, required_cols in PROJECT_SHEETS.items():
        df = frames[sheet_name]
        df.columns = [str(c).strip() for c in df.columns]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"{sheet_name}: colonnes manquantes {missing_cols}")

    if errors:
        return errors

    # Validation events_history
    ev = frames["events_history"].copy()
    ev["event_start"] = pd.to_datetime(ev["event_start"], errors="coerce")
    if ev["event_start"].isna().any():
        errors.append("events_history: certaines dates event_start sont invalides.")
    if ev["event_id"].astype(str).duplicated().any():
        errors.append("events_history: event_id contient des doublons.")
    if "event_end" in ev.columns:
        ev["event_end"] = pd.to_datetime(ev["event_end"], errors="coerce")
        mask = ev["event_end"].notna() & ev["event_start"].notna() & (ev["event_end"] < ev["event_start"])
        if mask.any():
            errors.append("events_history: certaines lignes ont event_end < event_start.")

    # Validation thermal_timeseries
    th = frames["thermal_timeseries"].copy()
    th["timestamp"] = pd.to_datetime(th["timestamp"], errors="coerce")
    if th["timestamp"].isna().any():
        errors.append("thermal_timeseries: certaines dates timestamp sont invalides.")
    has_direct = {"top_oil_rise_c", "hotspot_gradient_c"}.issubset(set(th.columns))
    has_load = "load_factor" in th.columns
    if not (has_direct or has_load):
        errors.append(
            "thermal_timeseries: il faut soit [top_oil_rise_c + hotspot_gradient_c], soit [load_factor]."
        )

    # Cohérence asset_id
    assets = set(frames["asset_info"]["asset_id"].astype(str).dropna().unique())
    for sheet_name in ["events_history", "thermal_timeseries", "thermal_params", "maintenance_policies", "analysis_settings"]:
        current_assets = set(frames[sheet_name]["asset_id"].astype(str).dropna().unique())
        unknown = sorted(list(current_assets - assets))
        if unknown:
            errors.append(f"{sheet_name}: asset_id inconnus par rapport à asset_info: {unknown}")

    return errors


def _save_project_frames(frames: Dict[str, pd.DataFrame], source_name: str) -> dict:
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)
    saved = {}
    for name, df in frames.items():
        out = PROJECT_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        saved[name] = str(out)

    meta = {
        "ok": True,
        "source": source_name,
        "sheets": list(frames.keys()),
        "rows": {k: int(len(v)) for k, v in frames.items()},
    }
    PROJECT_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state["project_meta"] = meta
    st.session_state["project_frames"] = frames
    return meta


def _load_project_meta() -> dict:
    if PROJECT_META_PATH.exists():
        try:
            return json.loads(PROJECT_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"ok": False}


def _load_project_sheet(sheet_name: str) -> pd.DataFrame:
    p = PROJECT_DIR / f"{sheet_name}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _clear_project_cache():
    if PROJECT_DIR.exists():
        for p in PROJECT_DIR.glob("*.csv"):
            try:
                p.unlink()
            except Exception:
                pass
    if PROJECT_META_PATH.exists():
        try:
            PROJECT_META_PATH.unlink()
        except Exception:
            pass
    st.session_state.pop("project_meta", None)
    st.session_state.pop("project_frames", None)


def _read_uploaded_file(uploaded_file) -> tuple[str, object]:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.read()
    if suffix == ".csv":
        return "csv", pd.read_csv(io.BytesIO(raw))
    if suffix == ".xlsx":
        xls = pd.ExcelFile(io.BytesIO(raw))
        frames = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
        return "xlsx", frames
    raise ValueError("Format non supporté. Utilise .csv ou .xlsx")


# ------------------------- Header meta -------------------------
meta = get_failures_meta()
project_meta = _load_project_meta()

cmeta1, cmeta2 = st.columns(2)
with cmeta1:
    if meta.get("ok"):
        st.success(
            f"Dataset TTF actif ✅ | rows={meta['rows']} | hash={meta['hash']} | source={meta['source']}"
        )
    else:
        st.warning("Aucun dataset TTF actif pour le moment.")
with cmeta2:
    if project_meta.get("ok"):
        st.success(
            "Projet actif ✅ | "
            f"source={project_meta.get('source')} | "
            f"feuilles={', '.join(project_meta.get('sheets', []))}"
        )
    else:
        st.info("Aucun projet Excel complet persistant pour le moment.")

tab_upload, tab_current, tab_mqtt = st.tabs(
    ["📄 Import CSV / Excel", "🗂️ Dataset / projet actif", "📡 Données MQTT (réglages)"]
)

# ------------------------- Upload tab -------------------------
with tab_upload:
    st.subheader("Importer les données")
    st.caption(
        "Formats supportés : "
        "CSV simple (equipment_code, ttf_h[, duree_rep_h]) "
        "ou Excel projet complet (.xlsx) avec feuilles "
        "asset_info / events_history / thermal_timeseries / thermal_params / maintenance_policies / analysis_settings."
    )

    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Déposer un fichier", type=["csv", "xlsx"])
        has_timestamps = st.toggle(
            "Pour un CSV simple : mon fichier contient des horodatages (et pas ttf_h)",
            value=False,
        )
    with c2:
        st.markdown("**CSV minimal :**")
        st.code("equipment_code,ttf_h[,duree_rep_h]", language="text")
        st.markdown("**Excel projet :**")
        st.code(
            "asset_info / events_history / thermal_timeseries / thermal_params / maintenance_policies / analysis_settings",
            language="text",
        )

    if up is not None:
        try:
            kind, payload = _read_uploaded_file(up)
        except Exception as e:
            st.error(f"Lecture du fichier: {e}")
            kind, payload = None, None

        if kind == "csv" and isinstance(payload, pd.DataFrame):
            df_loaded = _normalize_columns(payload)

            if has_timestamps:
                existing = df_loaded.columns.tolist()
                if len(existing) < 2:
                    st.error("Il faut au moins 2 colonnes pour construire les TTF à partir des horodatages.")
                else:
                    eq_col = st.selectbox("Colonne équipement", options=existing, index=0)
                    ts_col = st.selectbox(
                        "Colonne horodatage panne",
                        options=existing,
                        index=min(1, len(existing) - 1),
                    )
                    if st.button("🧮 Construire ttf_h", type="primary", use_container_width=True):
                        try:
                            ttf_df = _compute_ttf_from_timestamps(df_loaded, eq_col, ts_col)
                            if ttf_df.empty:
                                st.error("Impossible de construire des TTF (vérifie les dates/format).")
                            else:
                                res = set_current_failures_df(
                                    ttf_df[["equipment_code", "ttf_h", "duree_rep_h"]],
                                    source_name=f"upload:{up.name}(timestamps)",
                                    persist=True,
                                )
                                st.success(
                                    f"Dataset TTF synchronisé ✅ | {res['rows']} lignes | hash={res['hash']}"
                                )
                                st.dataframe(ttf_df.head(50), use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.error(f"Construction TTF: {e}")
            else:
                missing = [c for c in REQUIRED_SIMPLE if c not in df_loaded.columns]
                if missing:
                    st.warning(
                        f"Colonnes manquantes: {missing}. Active l’option horodatage ou renomme tes colonnes."
                    )
                else:
                    if "duree_rep_h" not in df_loaded.columns:
                        df_loaded["duree_rep_h"] = None
                    st.dataframe(df_loaded.head(50), use_container_width=True, hide_index=True)
                    cL, cR = st.columns(2)
                    with cL:
                        if st.button("✅ Utiliser ce dataset CSV (session + fichier)", type="primary"):
                            res = set_current_failures_df(
                                df_loaded[["equipment_code", "ttf_h", "duree_rep_h"]],
                                source_name=f"upload:{up.name}",
                                persist=True,
                            )
                            st.success(
                                f"Dataset TTF synchronisé ✅ | {res['rows']} lignes | hash={res['hash']}"
                            )
                    with cR:
                        if st.button("⬆️ Envoyer dans SQLite (historique)", use_container_width=True):
                            n = bulk_insert_failures(df_loaded, source=f"upload:{up.name}")
                            st.success(f"{n} lignes insérées dans {DB_PATH.name}")

        elif kind == "xlsx" and isinstance(payload, dict):
            frames = {str(k).strip(): v.copy() for k, v in payload.items()}
            st.markdown("### Aperçu des feuilles détectées")
            st.write(sorted(frames.keys()))

            errors = _validate_project_sheets(frames)
            if errors:
                st.error("Le fichier projet n'est pas valide.")
                for msg in errors:
                    st.write(f"- {msg}")
            else:
                derived_ttf = _build_ttf_from_events(frames["events_history"])

                if derived_ttf.empty:
                    st.warning(
                        "Aucun TTF dérivé depuis events_history. "
                        "Vérifie qu'il y a au moins 2 pannes par asset_id avec is_failure=1."
                    )

                st.markdown("### Aperçu projet")
                preview_sheet = st.selectbox("Prévisualiser une feuille", options=list(frames.keys()))
                st.dataframe(frames[preview_sheet].head(50), use_container_width=True, hide_index=True)

                st.markdown("### TTF dérivés depuis events_history")
                st.dataframe(derived_ttf.head(50), use_container_width=True, hide_index=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Utiliser ce projet (persister + synchroniser TTF)", type="primary"):
                        try:
                            project_info = _save_project_frames(frames, source_name=f"upload:{up.name}")
                            res = set_current_failures_df(
                                derived_ttf[["equipment_code", "ttf_h", "duree_rep_h"]],
                                source_name=f"project:{up.name}:events_history",
                                persist=True,
                            )
                            st.success(
                                "Projet sauvegardé ✅ | "
                                f"feuilles={len(project_info['sheets'])} | "
                                f"TTF synchronisés={res['rows']} lignes | hash={res['hash']}"
                            )
                        except Exception as e:
                            st.error(f"Sauvegarde projet: {e}")
                with cc2:
                    if st.button("⬆️ Envoyer les TTF dérivés dans SQLite", use_container_width=True):
                        try:
                            n = bulk_insert_failures(
                                derived_ttf[["equipment_code", "ttf_h", "duree_rep_h"]],
                                source=f"project:{up.name}:events_history",
                            )
                            st.success(f"{n} lignes insérées dans {DB_PATH.name}")
                        except Exception as e:
                            st.error(f"SQLite: {e}")

# ------------------------- Current tab -------------------------
with tab_current:
    st.subheader("Dataset TTF actif")
    cur = get_current_failures_df()
    if cur.empty:
        st.info("Aucun dataset TTF actif.")
    else:
        st.dataframe(cur.head(50), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬆️ Synchroniser dataset TTF actif vers SQLite", use_container_width=True):
                n = bulk_insert_failures(cur, source=str(meta.get("source", "session")))
                st.success(f"{n} lignes insérées dans {DB_PATH.name}")
        with c2:
            if st.button("🗑️ Purge DB (failures) uniquement", use_container_width=True):
                clear_db()
                st.success("Table failures vidée.")

    st.divider()
    st.subheader("Projet Excel actif")
    project_meta = _load_project_meta()
    if not project_meta.get("ok"):
        st.info("Aucun projet Excel persistant.")
    else:
        st.json(project_meta)
        available_sheets = project_meta.get("sheets", [])
        if available_sheets:
            selected_sheet = st.selectbox("Voir une feuille persistée", options=available_sheets)
            project_df = _load_project_sheet(selected_sheet)
            if project_df.empty:
                st.warning("Feuille persistée introuvable ou vide.")
            else:
                st.dataframe(project_df.head(50), use_container_width=True, hide_index=True)

        if st.button("🗑️ Supprimer le projet Excel persistant", use_container_width=True):
            _clear_project_cache()
            st.success("Projet persistant supprimé.")

# ------------------------- MQTT tab -------------------------
with tab_mqtt:
    st.subheader("Paramètres MQTT (pour la page Temps réel)")
    mqtt_cfg_file = BASE_DIR / "config" / "mqtt.json"

    def load_mqtt():
        if mqtt_cfg_file.exists():
            try:
                return json.loads(mqtt_cfg_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "host": "localhost",
            "port": 1883,
            "site": "bench1",
            "equipement": "tr_230_20",
            "topic_base": "lab/transfo",
        }

    def save_mqtt(cfg: dict):
        mqtt_cfg_file.parent.mkdir(exist_ok=True, parents=True)
        mqtt_cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = load_mqtt()
    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input("Broker host", cfg.get("host", "localhost"))
        port = st.number_input("Broker port", min_value=1, value=int(cfg.get("port", 1883)), step=1)
        site = st.text_input("Site", cfg.get("site", "bench1"))
        eqp = st.text_input("Équipement", cfg.get("equipement", "tr_230_20"))
    with col2:
        topic_base = st.text_input("Topic base", cfg.get("topic_base", "lab/transfo"))
        st.caption("Ex: lab/transfo/{site}/{equipement}/measures")

    if st.button("💾 Enregistrer paramètres MQTT", type="primary"):
        new_cfg = {
            "host": host,
            "port": int(port),
            "site": site,
            "equipement": eqp,
            "topic_base": topic_base,
        }
        save_mqtt(new_cfg)
        st.success(f"Sauvegardé → {mqtt_cfg_file}")
        st.session_state["mqtt_cfg"] = new_cfg
