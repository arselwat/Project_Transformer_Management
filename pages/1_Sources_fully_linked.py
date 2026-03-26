from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional

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
st.caption(
    "Importe un CSV simple de TTF ou un fichier Excel projet. Une fois chargé ici, le dataset circule dans "
    "Indicateurs, Optimisation, Maintenance et Résultats globaux via core.datahub."
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = DATA_DIR / "reliability.sqlite"


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


# ------------------------- Specs / validation -------------------------
REQUIRED_SIMPLE = ["equipment_code", "ttf_h"]
PROJECT_REQUIRED = {
    "asset_info": ["asset_id", "asset_name"],
    "events_history": ["event_id", "asset_id", "event_start", "event_type", "is_failure", "is_planned"],
    "thermal_timeseries": ["timestamp", "asset_id", "temp_amb_C"],
    "thermal_params": ["asset_id", "R"],
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
    "analysis_settings": ["asset_id", "alpha_significance", "analysis_horizon_days"],
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
    out = df.rename(columns=ren).copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _normalize_project_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    for raw_name, raw_df in frames.items():
        name = str(raw_name).strip()
        df = raw_df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower().strip(): c for c in df.columns}
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
                if k in lower:
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
                if k in lower:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)
            for c in ["event_start", "event_end"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            for c in ["is_failure", "is_planned"]:
                if c in df.columns:
                    df[c] = (
                        df[c]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "oui": 1, "non": 0})
                        .fillna(pd.to_numeric(df[c], errors="coerce"))
                        .fillna(0)
                        .astype(int)
                    )
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
                if k in lower:
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
                "faa_limit": "faa_limit",
                "lol_limit_hours": "lol_limit_hours",
                "rated_power_mva": "sn_mva",
            }
            for k, v in aliases.items():
                if k in lower:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)
            for c in df.columns:
                if c != "asset_id":
                    df[c] = pd.to_numeric(df[c], errors="ignore")

            # Convertit éventuellement des constantes en heures vers minutes pour le moteur thermique
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
                if k in lower:
                    ren[lower[k]] = v
            df = df.rename(columns=ren)
            if "alpha_significance" in df.columns:
                df["alpha_significance"] = pd.to_numeric(df["alpha_significance"], errors="coerce")
            if "analysis_horizon_days" in df.columns:
                df["analysis_horizon_days"] = pd.to_numeric(df["analysis_horizon_days"], errors="coerce")

        out[name] = df

    # enrichit sn_mva depuis asset_info si manquant dans thermal_params
    if "asset_info" in out and "thermal_params" in out:
        a = out["asset_info"].copy()
        t = out["thermal_params"].copy()
        if "asset_id" in a.columns and "asset_id" in t.columns:
            if "sn_mva" not in t.columns or t["sn_mva"].isna().all():
                if "rated_power_mva" in a.columns:
                    merge = a[["asset_id", "rated_power_mva"]].copy()
                    merge = merge.rename(columns={"rated_power_mva": "sn_mva_from_asset"})
                    t = t.merge(merge, on="asset_id", how="left")
                    if "sn_mva" not in t.columns:
                        t["sn_mva"] = t["sn_mva_from_asset"]
                    else:
                        t["sn_mva"] = pd.to_numeric(t["sn_mva"], errors="coerce").fillna(t["sn_mva_from_asset"])
                    t = t.drop(columns=[c for c in ["sn_mva_from_asset"] if c in t.columns])
                    out["thermal_params"] = t

    return out


def _validate_project_sheets(frames: Dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []

    missing_sheets = [s for s in PROJECT_REQUIRED if s not in frames]
    if missing_sheets:
        errors.append(f"Feuilles manquantes: {missing_sheets}")
        return errors

    for sheet_name, required_cols in PROJECT_REQUIRED.items():
        df = frames[sheet_name]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"{sheet_name}: colonnes manquantes {missing_cols}")

    if errors:
        return errors

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

    th = frames["thermal_timeseries"].copy()
    th["timestamp"] = pd.to_datetime(th["timestamp"], errors="coerce")
    if th["timestamp"].isna().any():
        errors.append("thermal_timeseries: certaines dates timestamp sont invalides.")

    has_load_driver = any(c in th.columns for c in ["K", "charge_pct", "load_factor", "load_mva"])
    if not has_load_driver:
        errors.append(
            "thermal_timeseries: il faut au moins une colonne parmi K, charge_pct, load_factor ou load_mva."
        )

    assets = set(frames["asset_info"]["asset_id"].astype(str).dropna().unique())
    for sheet_name in ["events_history", "thermal_timeseries", "thermal_params", "maintenance_policies", "analysis_settings"]:
        current_assets = set(frames[sheet_name]["asset_id"].astype(str).dropna().unique())
        unknown = sorted(list(current_assets - assets))
        if unknown:
            errors.append(f"{sheet_name}: asset_id inconnus par rapport à asset_info: {unknown}")

    return errors


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
project_meta = get_project_meta()

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
        st.info("Aucun projet Excel actif pour le moment.")


tab_upload, tab_current, tab_mqtt = st.tabs(
    ["📄 Import CSV / Excel", "🗂️ Dataset / projet actif", "📡 Données MQTT (réglages)"]
)


# ------------------------- Upload tab -------------------------
with tab_upload:
    st.subheader("Importer les données")
    st.caption(
        "CSV simple : equipment_code, ttf_h[, duree_rep_h] ; "
        "Projet Excel : asset_info / events_history / thermal_timeseries / thermal_params / "
        "maintenance_policies / analysis_settings."
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
                            tmp = df_loaded[[eq_col, ts_col]].dropna().copy()
                            tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
                            tmp = tmp.dropna(subset=[ts_col]).sort_values([eq_col, ts_col])
                            out_rows = []
                            for eq, g in tmp.groupby(eq_col):
                                t = g[ts_col].tolist()
                                for i in range(1, len(t)):
                                    dh = (t[i] - t[i - 1]).total_seconds() / 3600.0
                                    if dh > 0:
                                        out_rows.append({"equipment_code": str(eq), "ttf_h": float(dh), "duree_rep_h": None})
                            ttf_df = pd.DataFrame(out_rows)

                            if ttf_df.empty:
                                st.error("Impossible de construire des TTF (vérifie les dates/format).")
                            else:
                                res = set_current_failures_df(
                                    ttf_df,
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
            frames = _normalize_project_frames({str(k).strip(): v.copy() for k, v in payload.items()})

            st.markdown("### Aperçu des feuilles détectées")
            st.write(sorted(frames.keys()))

            errors = _validate_project_sheets(frames)
            if errors:
                st.error("Le fichier projet n'est pas valide.")
                for msg in errors:
                    st.write(f"- {msg}")
            else:
                derived_ttf = build_ttf_from_events(frames["events_history"])
                frames["failures_ttf"] = derived_ttf

                if derived_ttf.empty:
                    st.warning(
                        "Aucun TTF dérivé depuis events_history. Vérifie qu'il y a au moins 2 pannes par asset_id avec is_failure=1."
                    )

                st.markdown("### Aperçu projet")
                preview_sheet = st.selectbox("Prévisualiser une feuille", options=list(frames.keys()))
                st.dataframe(frames[preview_sheet].head(50), use_container_width=True, hide_index=True)

                st.markdown("### TTF dérivés depuis events_history")
                st.dataframe(derived_ttf.head(50), use_container_width=True, hide_index=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Utiliser ce projet (persister + synchroniser tout)", type="primary"):
                        try:
                            res = set_current_project_data(
                                frames=frames,
                                source_name=f"upload:{up.name}",
                                persist=True,
                                sync_failures=True,
                            )
                            if res.get("ok"):
                                st.success(
                                    "Projet synchronisé ✅ | "
                                    f"feuilles={len(res.get('sheets', []))} | "
                                    f"TTF synchronisés={res.get('failures_rows', 0)} | "
                                    f"hash projet={res.get('hash')}"
                                )
                            else:
                                st.error(res.get("msg", "Erreur inconnue lors de la synchronisation projet."))
                        except Exception as e:
                            st.error(f"Synchronisation projet: {e}")
                with cc2:
                    if st.button("⬆️ Envoyer les TTF dérivés dans SQLite", use_container_width=True):
                        try:
                            n = bulk_insert_failures(
                                derived_ttf,
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
    st.subheader("Projet actif")
    project_meta = get_project_meta()
    frames = get_current_project_data()

    if not project_meta.get("ok") or not frames:
        st.info("Aucun projet complet actif.")
    else:
        st.json(project_meta)
        available_sheets = sorted(list(frames.keys()))
        if available_sheets:
            selected_sheet = st.selectbox("Voir une feuille active", options=available_sheets)
            project_df = frames.get(selected_sheet, pd.DataFrame())
            if project_df.empty:
                st.warning("Feuille active vide.")
            else:
                st.dataframe(project_df.head(50), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Supprimer le projet actif", use_container_width=True):
                clear_current_project_data(clear_failures=False)
                st.success("Projet actif supprimé.")
        with c2:
            if st.button("🗑️ Supprimer projet + TTF actif", use_container_width=True):
                clear_current_project_data(clear_failures=True)
                st.success("Projet actif et dataset TTF supprimés.")


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
