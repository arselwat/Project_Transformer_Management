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
    clear_current_project_data,
    get_current_failures_df,
    get_current_project_data,
    get_failures_meta,
    get_project_meta,
    set_current_failures_df,
    set_current_project_data,
)
from core.ui import render_shell, render_page_header

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
require_login()

render_shell("pages/1_Sources_fully_linked_fixed.py")
render_page_header(
    "Sources de données",
    "Importer un fichier simple à une seule feuille contenant les données essentielles de fiabilité et thermiques.",
    "📥",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = DATA_DIR / "reliability.sqlite"


# ============================================================
# SQLite
# ============================================================
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


# ============================================================
# Validation / normalisation
# ============================================================
REQUIRED_INPUT = [
    "equipment_code",
    "timestamp",
    "is_failure",
    "temp_amb_C",
    "charge_pct",
]

OPTIONAL_INPUT = [
    "repair_time_hours",
    "etat_ventilateurs",
]


def _read_csv_flex_from_bytes(raw: bytes) -> pd.DataFrame:
    attempts = [
        lambda: pd.read_csv(io.BytesIO(raw)),
        lambda: pd.read_csv(io.BytesIO(raw), engine="python", on_bad_lines="skip", sep=None),
        lambda: pd.read_csv(io.BytesIO(raw), sep=";", engine="python", on_bad_lines="skip"),
    ]
    for fn in attempts:
        try:
            df = fn()
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            continue
    raise ValueError("Impossible de lire le CSV.")


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.read()

    if suffix == ".csv":
        return _read_csv_flex_from_bytes(raw)

    if suffix == ".xlsx":
        xls = pd.ExcelFile(io.BytesIO(raw))
        if len(xls.sheet_names) != 1:
            raise ValueError(
                "Le fichier Excel doit contenir une seule feuille. "
                "Supprime les autres feuilles puis réessaie."
            )
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        df.columns = [str(c).strip() for c in df.columns]
        return df

    raise ValueError("Format non supporté. Utilise .csv ou .xlsx avec une seule feuille.")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code",
        "equipement": "equipment_code",
        "code_equipement": "equipment_code",
        "eqp": "equipment_code",
        "asset_id": "equipment_code",
        "assetid": "equipment_code",
        "horodatage": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "date_heure": "timestamp",
        "dateheure": "timestamp",
        "failure_time": "timestamp",
        "failure_date": "timestamp",
        "date_panne": "timestamp",
        "panne": "is_failure",
        "defaillance": "is_failure",
        "failure": "is_failure",
        "isfailure": "is_failure",
        "repair_hours": "repair_time_hours",
        "repair_time_hours": "repair_time_hours",
        "mttr_h": "repair_time_hours",
        "duree_rep_h": "repair_time_hours",
        "duree_reparation_h": "repair_time_hours",
        "ambient_temp_c": "temp_amb_C",
        "temp_ambiante_c": "temp_amb_C",
        "temperature_ambiante": "temp_amb_C",
        "temperature_ambiante_c": "temp_amb_C",
        "temp_ambiante": "temp_amb_C",
        "load_pct": "charge_pct",
        "load_percent": "charge_pct",
        "charge": "charge_pct",
        "charge_percent": "charge_pct",
        "charge_pourcent": "charge_pct",
        "fan_status": "etat_ventilateurs",
        "fans_status": "etat_ventilateurs",
        "ventilateurs": "etat_ventilateurs",
        "etat_ventilateur": "etat_ventilateurs",
    }

    cols = {str(c).lower().strip(): c for c in df.columns}
    ren = {}
    for k, v in mapping.items():
        if k in cols:
            ren[cols[k]] = v

    out = df.rename(columns=ren).copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


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

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(int)
    )


def _prepare_single_sheet(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = _normalize_columns(df)
    errors: list[str] = []

    for c in OPTIONAL_INPUT:
        if c not in work.columns:
            work[c] = None

    missing = [c for c in REQUIRED_INPUT if c not in work.columns]
    if missing:
        errors.append(f"Colonnes obligatoires manquantes : {missing}")
        return work, errors

    work["equipment_code"] = work["equipment_code"].astype(str).str.strip()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["is_failure"] = _coerce_bool01(work["is_failure"])
    work["repair_time_hours"] = pd.to_numeric(work["repair_time_hours"], errors="coerce")
    work["temp_amb_C"] = pd.to_numeric(work["temp_amb_C"], errors="coerce")
    work["charge_pct"] = pd.to_numeric(work["charge_pct"], errors="coerce")
    work["etat_ventilateurs"] = _coerce_bool01(work["etat_ventilateurs"])

    work = work[work["equipment_code"].notna()].copy()
    work = work[work["equipment_code"] != ""].copy()

    if work.empty:
        errors.append("Aucune ligne exploitable après nettoyage.")
        return work, errors

    if work["timestamp"].isna().all():
        errors.append("Toutes les dates de la colonne timestamp sont invalides.")

    if work["temp_amb_C"].isna().all():
        errors.append("La colonne temp_amb_C ne contient aucune valeur numérique valide.")

    if work["charge_pct"].isna().all():
        errors.append("La colonne charge_pct ne contient aucune valeur numérique valide.")

    if work["is_failure"].isin([0, 1]).sum() == 0:
        errors.append("La colonne is_failure ne contient aucune valeur exploitable.")

    work = work.sort_values(["equipment_code", "timestamp"]).reset_index(drop=True)

    ordered_cols = [
        "equipment_code",
        "timestamp",
        "is_failure",
        "repair_time_hours",
        "temp_amb_C",
        "charge_pct",
        "etat_ventilateurs",
    ]
    work = work[ordered_cols]

    return work, errors


def build_ttf_from_single_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h", "failure_time"])

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["is_failure"] = _coerce_bool01(work["is_failure"])
    work["repair_time_hours"] = pd.to_numeric(work["repair_time_hours"], errors="coerce")

    work = work.dropna(subset=["timestamp"]).copy()
    work = work[work["is_failure"] == 1].copy()
    work = work.sort_values(["equipment_code", "timestamp"]).reset_index(drop=True)

    out_rows = []
    for eq, g in work.groupby("equipment_code"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            dt_h = (g.loc[i, "timestamp"] - g.loc[i - 1, "timestamp"]).total_seconds() / 3600.0
            if dt_h > 0:
                out_rows.append(
                    {
                        "equipment_code": str(eq),
                        "ttf_h": float(dt_h),
                        "duree_rep_h": g.loc[i, "repair_time_hours"],
                        "failure_time": g.loc[i, "timestamp"],
                    }
                )

    if not out_rows:
        return pd.DataFrame(columns=["equipment_code", "ttf_h", "duree_rep_h", "failure_time"])

    return pd.DataFrame(out_rows)


def build_project_frames_from_single_sheet(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    source_data = df.copy()

    asset_codes = sorted(source_data["equipment_code"].astype(str).dropna().unique().tolist())
    asset_info = pd.DataFrame(
        {
            "asset_id": asset_codes,
            "asset_name": asset_codes,
        }
    )

    events_history = source_data[
        ["equipment_code", "timestamp", "is_failure", "repair_time_hours"]
    ].copy()
    events_history = events_history.rename(
        columns={
            "equipment_code": "asset_id",
            "timestamp": "event_start",
        }
    )
    events_history["event_id"] = [f"EVT_{i + 1:06d}" for i in range(len(events_history))]
    events_history["event_type"] = events_history["is_failure"].apply(
        lambda x: "failure" if int(x) == 1 else "observation"
    )
    events_history["is_planned"] = 0
    events_history = events_history[
        [
            "event_id",
            "asset_id",
            "event_start",
            "event_type",
            "is_failure",
            "is_planned",
            "repair_time_hours",
        ]
    ]

    thermal_timeseries = source_data[
        ["equipment_code", "timestamp", "temp_amb_C", "charge_pct", "etat_ventilateurs"]
    ].copy()
    thermal_timeseries = thermal_timeseries.rename(columns={"equipment_code": "asset_id"})

    failures_ttf = build_ttf_from_single_sheet(source_data)

    frames: Dict[str, pd.DataFrame] = {
        "source_data": source_data,
        "asset_info": asset_info,
        "events_history": events_history,
        "thermal_timeseries": thermal_timeseries,
        "failures_ttf": failures_ttf,
    }
    return frames


# ============================================================
# Résumé
# ============================================================
meta = get_failures_meta()
project_meta = get_project_meta()

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Lignes TTF actives", meta.get("rows", 0) if meta.get("ok") else 0)
with k2:
    st.metric("Jeu brut actif", "Oui" if project_meta.get("ok") else "Non")
with k3:
    st.metric("Base SQLite", "Prête")

tab_upload, tab_current, tab_mqtt = st.tabs(["Importer", "Données actives", "MQTT"])


# ============================================================
# Importer
# ============================================================
with tab_upload:
    st.subheader("Importer un fichier")
    st.caption(
        "Format attendu : une seule feuille avec les colonnes "
        "`equipment_code`, `timestamp`, `is_failure`, `temp_amb_C`, `charge_pct` "
        "et, si disponible, `repair_time_hours`, `etat_ventilateurs`."
    )
    st.info("La température du point chaud n'est pas demandée ici car elle est calculée par le logiciel.")

    up = st.file_uploader("Fichier CSV ou XLSX à une seule feuille", type=["csv", "xlsx"])

    if up is not None:
        try:
            raw_df = _read_uploaded_file(up)
            clean_df, errors = _prepare_single_sheet(raw_df)
        except Exception as e:
            st.error(f"Lecture du fichier : {e}")
            raw_df, clean_df, errors = None, None, [str(e)]

        if clean_df is not None:
            if errors:
                st.error("Le fichier importé n'est pas valide.")
                for msg in errors:
                    st.write(f"- {msg}")
            else:
                frames = build_project_frames_from_single_sheet(clean_df)
                derived_ttf = frames["failures_ttf"]

                st.markdown("#### Prévisualisation des données nettoyées")
                st.dataframe(clean_df.head(30), use_container_width=True, hide_index=True)

                st.markdown("#### TTF dérivés")
                if derived_ttf.empty:
                    st.warning(
                        "Aucun TTF n'a pu être dérivé. "
                        "Il faut au moins deux pannes (`is_failure = 1`) pour un même équipement."
                    )
                else:
                    st.dataframe(derived_ttf.head(30), use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)

                with c1:
                    if st.button("Utiliser ce fichier", type="primary", use_container_width=True):
                        try:
                            res_project = set_current_project_data(
                                frames=frames,
                                source_name=f"upload:{up.name}",
                                persist=True,
                                sync_failures=False,
                            )

                            ttf_to_sync = derived_ttf[["equipment_code", "ttf_h", "duree_rep_h"]].copy()
                            res_ttf = set_current_failures_df(
                                ttf_to_sync,
                                source_name=f"upload:{up.name}:ttf",
                                persist=True,
                            )

                            st.success(
                                "Source synchronisée | "
                                f"observations={len(clean_df)} | "
                                f"TTF={len(ttf_to_sync)} | "
                                f"hash_projet={res_project.get('hash', '')} | "
                                f"hash_ttf={res_ttf.get('hash', '')}"
                            )
                        except Exception as e:
                            st.error(f"Synchronisation : {e}")

                with c2:
                    if st.button("Envoyer les TTF dans SQLite", use_container_width=True):
                        try:
                            n = bulk_insert_failures(
                                derived_ttf[["equipment_code", "ttf_h", "duree_rep_h"]],
                                source=f"upload:{up.name}",
                            )
                            st.success(f"{n} lignes insérées")
                        except Exception as e:
                            st.error(f"SQLite : {e}")


# ============================================================
# Données actives
# ============================================================
with tab_current:
    st.subheader("Dataset TTF actif")
    cur = get_current_failures_df()

    if cur.empty:
        st.info("Aucun dataset TTF actif")
    else:
        st.dataframe(cur.head(30), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Synchroniser dataset actif vers SQLite", use_container_width=True):
                source_name = str(get_failures_meta().get("source", "session"))
                n = bulk_insert_failures(cur, source=source_name)
                st.success(f"{n} lignes insérées")

        with c2:
            if st.button("Vider SQLite (failures)", use_container_width=True):
                clear_db()
                st.success("Table failures vidée")

    st.divider()

    st.subheader("Jeu brut actif")
    current_project_meta = get_project_meta()
    frames = get_current_project_data()

    if not current_project_meta.get("ok") or not frames:
        st.info("Aucune source brute active")
    else:
        available_sheets = sorted(list(frames.keys()))
        selected_sheet = st.selectbox("Voir une table dérivée", options=available_sheets)
        project_df = frames.get(selected_sheet, pd.DataFrame())

        if project_df.empty:
            st.warning("Table vide")
        else:
            st.dataframe(project_df.head(30), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            if st.button("Supprimer la source brute active", use_container_width=True):
                clear_current_project_data(clear_failures=False)
                st.success("Source brute supprimée")

        with d2:
            if st.button("Supprimer source brute + dataset actif", use_container_width=True):
                clear_current_project_data(clear_failures=True)
                st.success("Source brute et dataset supprimés")


# ============================================================
# MQTT
# ============================================================
with tab_mqtt:
    st.subheader("Paramètres MQTT")
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
        st.caption("Exemple : lab/transfo/{site}/{equipement}/measures")

    if st.button("Enregistrer paramètres MQTT", type="primary", use_container_width=True):
        new_cfg = {
            "host": host,
            "port": int(port),
            "site": site,
            "equipement": eqp,
            "topic_base": topic_base,
        }
        save_mqtt(new_cfg)
        st.session_state["mqtt_cfg"] = new_cfg
        st.success(f"Sauvegardé → {mqtt_cfg_file}")