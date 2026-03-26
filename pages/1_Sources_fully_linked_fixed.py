from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.datahub import (
    build_ttf_from_events,
    clear_current_project_data,
    get_current_failures_df,
    get_current_project_data,
    get_failures_meta,
    get_project_meta,
    set_current_failures_df,
    set_current_project_data,
)

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
require_login()

st.title("📥 Sources de données")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = DATA_DIR / "reliability.sqlite"


# -------------------------------------------------------------------
# SQLite helpers
# -------------------------------------------------------------------
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


def bulk_insert_failures(df: pd.DataFrame, source: str) -> int:
    if df is None or df.empty:
        return 0

    work = df.copy()
    for c in ["equipment_code", "ttf_h", "duree_rep_h"]:
        if c not in work.columns:
            work[c] = None

    work["equipment_code"] = work["equipment_code"].astype(str)
    work["ttf_h"] = pd.to_numeric(work["ttf_h"], errors="coerce")
    work["duree_rep_h"] = pd.to_numeric(work["duree_rep_h"], errors="coerce")
    work = work.dropna(subset=["ttf_h"])
    work = work[work["ttf_h"] > 0]

    rows = work[["equipment_code", "ttf_h", "duree_rep_h"]].values.tolist()
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
REQUIRED_SIMPLE = ["equipment_code", "ttf_h"]
PROJECT_REQUIRED_SHEETS = [
    "asset_info",
    "events_history",
    "thermal_timeseries",
    "thermal_params",
    "maintenance_policies",
    "analysis_settings",
]


def _normalize_simple_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower = {str(c).lower().strip(): c for c in out.columns}
    ren = {}
    for k, v in mapping.items():
        if k in lower:
            ren[lower[k]] = v
    out = out.rename(columns=ren)
    out.columns = [str(c).strip() for c in out.columns]
    return out


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


def _compute_ttf_from_timestamps(df: pd.DataFrame, eq_col: str, ts_col: str) -> pd.DataFrame:
    tmp = df[[eq_col, ts_col]].dropna().copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp.dropna(subset=[ts_col]).sort_values([eq_col, ts_col])

    rows = []
    for eq, g in tmp.groupby(eq_col):
        times = g[ts_col].tolist()
        for i in range(1, len(times)):
            dh = (times[i] - times[i - 1]).total_seconds() / 3600.0
            if dh > 0:
                rows.append(
                    {
                        "equipment_code": str(eq),
                        "ttf_h": float(dh),
                        "duree_rep_h": None,
                    }
                )

    return pd.DataFrame(rows)


def _check_project_sheets(frames: Dict[str, pd.DataFrame]) -> list[str]:
    names = [str(k).strip() for k in frames.keys()]
    missing = [x for x in PROJECT_REQUIRED_SHEETS if x not in names]
    return missing


# -------------------------------------------------------------------
# Status header
# -------------------------------------------------------------------
meta = get_failures_meta()
project_meta = get_project_meta()

col1, col2 = st.columns(2)
with col1:
    if meta.get("ok"):
        st.success(f"Dataset actif | rows={meta['rows']} | hash={meta['hash']}")
    else:
        st.info("Aucun dataset actif")
with col2:
    if project_meta.get("ok"):
        st.success(f"Projet actif | hash={project_meta.get('hash', '')}")
    else:
        st.info("Aucun projet actif")

tab_import, tab_active, tab_mqtt = st.tabs(["Import", "Actif", "MQTT"])


# -------------------------------------------------------------------
# Import tab
# -------------------------------------------------------------------
with tab_import:
    up = st.file_uploader("Fichier CSV ou Excel", type=["csv", "xlsx"])
    has_timestamps = st.toggle("CSV avec horodatages", value=False)

    if up is not None:
        try:
            kind, payload = _read_uploaded_file(up)
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            kind, payload = None, None

        if kind == "csv" and isinstance(payload, pd.DataFrame):
            df_loaded = _normalize_simple_columns(payload)

            if has_timestamps:
                cols = df_loaded.columns.tolist()
                if len(cols) < 2:
                    st.error("Il faut au moins 2 colonnes.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        eq_col = st.selectbox("Colonne équipement", options=cols)
                    with c2:
                        ts_col = st.selectbox("Colonne date/heure", options=cols, index=min(1, len(cols) - 1))

                    if st.button("Construire les TTF", type="primary", use_container_width=True):
                        try:
                            ttf_df = _compute_ttf_from_timestamps(df_loaded, eq_col, ts_col)
                            if ttf_df.empty:
                                st.error("Aucun TTF valide construit.")
                            else:
                                res = set_current_failures_df(
                                    ttf_df,
                                    source_name=f"upload:{up.name}(timestamps)",
                                    persist=True,
                                )
                                st.success(f"Dataset synchronisé | rows={res['rows']} | hash={res['hash']}")
                                st.dataframe(ttf_df.head(30), use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.error(f"Construction impossible : {e}")
            else:
                missing = [c for c in REQUIRED_SIMPLE if c not in df_loaded.columns]
                if missing:
                    st.error(f"Colonnes manquantes : {missing}")
                else:
                    if "duree_rep_h" not in df_loaded.columns:
                        df_loaded["duree_rep_h"] = None

                    st.dataframe(df_loaded.head(30), use_container_width=True, hide_index=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Utiliser ce CSV", type="primary", use_container_width=True):
                            res = set_current_failures_df(
                                df_loaded[["equipment_code", "ttf_h", "duree_rep_h"]],
                                source_name=f"upload:{up.name}",
                                persist=True,
                            )
                            st.success(f"Dataset synchronisé | rows={res['rows']} | hash={res['hash']}")
                    with c2:
                        if st.button("Envoyer en historique SQLite", use_container_width=True):
                            n = bulk_insert_failures(df_loaded, source=f"upload:{up.name}")
                            st.success(f"{n} lignes insérées")

        elif kind == "xlsx" and isinstance(payload, dict):
            frames = {str(k).strip(): v.copy() for k, v in payload.items()}
            missing_sheets = _check_project_sheets(frames)

            if missing_sheets:
                st.error(f"Feuilles manquantes : {missing_sheets}")
            else:
                st.write(sorted(frames.keys()))

                preview_sheet = st.selectbox("Aperçu feuille", options=sorted(frames.keys()))
                st.dataframe(frames[preview_sheet].head(30), use_container_width=True, hide_index=True)

                if "events_history" in frames:
                    preview_ttf = build_ttf_from_events(frames["events_history"])
                    if not preview_ttf.empty:
                        st.dataframe(preview_ttf.head(30), use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Utiliser ce projet Excel", type="primary", use_container_width=True):
                        try:
                            res = set_current_project_data(
                                frames=frames,
                                source_name=f"upload:{up.name}",
                                persist=True,
                                sync_failures=True,
                            )
                            st.success(
                                f"Projet synchronisé | feuilles={len(res.get('sheets', []))} | "
                                f"TTF={res.get('failures_rows', 0)} | hash={res.get('hash', '')}"
                            )
                        except Exception as e:
                            st.error(f"Synchronisation impossible : {e}")
                with c2:
                    if st.button("Envoyer les TTF dérivés en SQLite", use_container_width=True):
                        try:
                            ttf_df = build_ttf_from_events(frames["events_history"])
                            n = bulk_insert_failures(ttf_df, source=f"project:{up.name}:events_history")
                            st.success(f"{n} lignes insérées")
                        except Exception as e:
                            st.error(f"SQLite : {e}")


# -------------------------------------------------------------------
# Active tab
# -------------------------------------------------------------------
with tab_active:
    st.subheader("Dataset TTF actif")
    cur = get_current_failures_df()
    if cur.empty:
        st.info("Aucun dataset actif.")
    else:
        st.dataframe(cur.head(30), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Envoyer le dataset actif en SQLite", use_container_width=True):
                n = bulk_insert_failures(cur, source=str(meta.get("source", "session")))
                st.success(f"{n} lignes insérées")
        with c2:
            if st.button("Vider SQLite (failures)", use_container_width=True):
                clear_db()
                st.success("Table failures vidée")

    st.divider()

    st.subheader("Projet actif")
    current_project = get_current_project_data()
    current_meta = get_project_meta()

    if not current_meta.get("ok"):
        st.info("Aucun projet actif.")
    else:
        st.json(current_meta)
        sheet_names = sorted(list(current_project.keys()))
        if sheet_names:
            selected = st.selectbox("Feuille active", options=sheet_names)
            df_sheet = current_project.get(selected, pd.DataFrame())
            if df_sheet.empty:
                st.info("Feuille vide.")
            else:
                st.dataframe(df_sheet.head(30), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Supprimer le projet actif", use_container_width=True):
                clear_current_project_data(clear_failures=False)
                st.success("Projet supprimé")
        with c2:
            if st.button("Supprimer projet + dataset actif", use_container_width=True):
                clear_current_project_data(clear_failures=True)
                st.success("Projet et dataset supprimés")


# -------------------------------------------------------------------
# MQTT tab
# -------------------------------------------------------------------
with tab_mqtt:
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

    c1, c2 = st.columns(2)
    with c1:
        host = st.text_input("Broker host", cfg.get("host", "localhost"))
        port = st.number_input("Broker port", min_value=1, value=int(cfg.get("port", 1883)), step=1)
        site = st.text_input("Site", cfg.get("site", "bench1"))
        eqp = st.text_input("Équipement", cfg.get("equipement", "tr_230_20"))
    with c2:
        topic_base = st.text_input("Topic base", cfg.get("topic_base", "lab/transfo"))

    if st.button("Enregistrer MQTT", type="primary", use_container_width=True):
        save_mqtt(
            {
                "host": host,
                "port": int(port),
                "site": site,
                "equipement": eqp,
                "topic_base": topic_base,
            }
        )
        st.session_state["mqtt_cfg"] = {
            "host": host,
            "port": int(port),
            "site": site,
            "equipement": eqp,
            "topic_base": topic_base,
        }
        st.success("Configuration enregistrée")