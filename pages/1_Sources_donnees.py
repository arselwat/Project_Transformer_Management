# pages/1_Sources.py  (ou ton nom réel de page)
from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.datahub import set_current_failures_df, get_failures_meta, get_current_failures_df

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
require_login()

st.title("📥 Sources de données")
st.caption("Ici tu charges/constructs les TTF. Ensuite **Indicateurs / Optimisation / Maintenance** utilisent ce même dataset automatiquement.")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = DATA_DIR / "reliability.sqlite"

# ------- DB helpers -------
def _db_conn():
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with _db_conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS failures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          equipment_code TEXT NOT NULL,
          ttf_h REAL NOT NULL,
          duree_rep_h REAL,
          source TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
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
    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0]

    rows = df[["equipment_code","ttf_h","duree_rep_h"]].values.tolist()
    with _db_conn() as cx:
        cx.executemany(
            "INSERT INTO failures (equipment_code, ttf_h, duree_rep_h, source) VALUES (?,?,?,?)",
            [(*r, source) for r in rows]
        )
        cx.commit()
    return len(rows)

def clear_db():
    with _db_conn() as cx:
        cx.execute("DELETE FROM failures")
        cx.commit()

init_db()

# ------------------------- Utils -------------------------
REQUIRED = ["equipment_code", "ttf_h"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code", "equipement": "equipment_code",
        "code_equipement": "equipment_code", "eqp": "equipment_code",
        "ttf": "ttf_h", "ttf_hours": "ttf_h",
        "mttr_h": "duree_rep_h",
        "repair_hours": "duree_rep_h",
        "failure_time": "failure_time", "failure_date": "failure_time",
        "date_panne": "failure_time",
    }
    cols = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for k, v in mapping.items():
        if k in cols:
            ren[cols[k]] = v
    return df.rename(columns=ren)

def _compute_ttf_from_timestamps(df: pd.DataFrame, eq_col: str, ts_col: str) -> pd.DataFrame:
    tmp = df[[eq_col, ts_col]].dropna().copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp.dropna(subset=[ts_col]).sort_values([eq_col, ts_col])
    out = []
    for eq, g in tmp.groupby(eq_col):
        t = g[ts_col].tolist()
        for i in range(1, len(t)):
            dh = (t[i] - t[i-1]).total_seconds() / 3600.0
            if dh > 0:
                out.append({"equipment_code": str(eq), "ttf_h": dh})
    return pd.DataFrame(out)

# ------------------------- Header meta -------------------------
meta = get_failures_meta()
if meta.get("ok"):
    st.success(f"Dataset actif ✅ | rows={meta['rows']} | hash={meta['hash']} | source={meta['source']}")
else:
    st.warning("Aucun dataset actif pour le moment. Charge un CSV ci-dessous.")

tab_csv, tab_mqtt = st.tabs(["📄 Fichier CSV / DB", "📡 Données MQTT (réglages)"])

with tab_csv:
    st.subheader("Importer ou construire les TTF (heures)")
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Déposer un CSV", type=["csv"])
        has_timestamps = st.toggle("Mon fichier contient des horodatages (et pas ttf_h)", value=False)
    with c2:
        st.markdown("**Format cible minimal :**")
        st.code("equipment_code, ttf_h[, duree_rep_h]", language="text")

    df_loaded: Optional[pd.DataFrame] = None
    if up is not None:
        try:
            content = up.read()
            df_loaded = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            st.error(f"Lecture CSV: {e}")

    if df_loaded is not None:
        df_loaded = _normalize_columns(df_loaded)

        if has_timestamps:
            existing = df_loaded.columns.tolist()
            eq_col = st.selectbox("Colonne équipement", options=existing, index=0)
            ts_col = st.selectbox("Colonne horodatage panne", options=existing, index=min(1, len(existing)-1))
            if st.button("🧮 Construire ttf_h", type="primary"):
                try:
                    ttf_df = _compute_ttf_from_timestamps(df_loaded, eq_col, ts_col)
                    if ttf_df.empty:
                        st.error("Impossible de construire des TTF (vérifie les dates/format).")
                    else:
                        res = set_current_failures_df(ttf_df, source_name=f"upload:{up.name}(timestamps)", persist=True)
                        st.success(f"Dataset synchronisé ✅ | {res['rows']} lignes | hash={res['hash']}")
                        st.dataframe(ttf_df.head(50), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Construction TTF: {e}")
        else:
            missing = [c for c in REQUIRED if c not in df_loaded.columns]
            if missing:
                st.warning(f"Colonnes manquantes: {missing}. Active l’option horodatage ou renomme tes colonnes.")
            else:
                st.dataframe(df_loaded.head(50), use_container_width=True, hide_index=True)
                cL, cR = st.columns(2)
                with cL:
                    if st.button("✅ Utiliser ce dataset (session + fichier)", type="primary"):
                        res = set_current_failures_df(df_loaded, source_name=f"upload:{up.name}", persist=True)
                        st.success(f"Dataset synchronisé ✅ | {res['rows']} lignes | hash={res['hash']}")
                with cR:
                    if st.button("⬆️ Envoyer dans SQLite (historique)", use_container_width=True):
                        n = bulk_insert_failures(df_loaded, source=f"upload:{up.name}")
                        st.success(f"{n} lignes insérées dans {DB_PATH.name}")

    st.divider()
    st.subheader("Dataset actuel (lecture)")
    cur = get_current_failures_df()
    if cur.empty:
        st.info("Aucun dataset actif.")
    else:
        st.dataframe(cur.head(50), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬆️ Synchroniser dataset actif vers SQLite", use_container_width=True):
                n = bulk_insert_failures(cur, source=str(meta.get("source", "session")))
                st.success(f"{n} lignes insérées dans {DB_PATH.name}")
        with c2:
            if st.button("🗑️ Purge DB (failures) uniquement", use_container_width=True):
                clear_db()
                st.success("Table failures vidée.")

with tab_mqtt:
    st.subheader("Paramètres MQTT (pour la page Temps réel)")
    import json
    mqtt_cfg_file = BASE_DIR / "config" / "mqtt.json"

    def load_mqtt():
        if mqtt_cfg_file.exists():
            try:
                return json.loads(mqtt_cfg_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"host": "localhost", "port": 1883, "site": "bench1",
                "equipement": "tr_230_20", "topic_base": "lab/transfo"}

    def save_mqtt(cfg: dict):
        mqtt_cfg_file.parent.mkdir(exist_ok=True, parents=True)
        mqtt_cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = load_mqtt()
    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input("Broker host", cfg.get("host", "localhost"))
        port = st.number_input("Broker port", min_value=1, value=int(cfg.get("port", 1883)), step=1)
        site = st.text_input("Site", cfg.get("site", "bench1"))
        eqp  = st.text_input("Équipement", cfg.get("equipement", "tr_230_20"))
    with col2:
        topic_base = st.text_input("Topic base", cfg.get("topic_base", "lab/transfo"))
        st.caption("Ex: lab/transfo/{site}/{equipement}/measures")

    if st.button("💾 Enregistrer paramètres MQTT", type="primary"):
        new_cfg = {"host": host, "port": int(port), "site": site,
                   "equipement": eqp, "topic_base": topic_base}
        save_mqtt(new_cfg)
        st.success(f"Sauvegardé → {mqtt_cfg_file}")
        st.session_state["mqtt_cfg"] = new_cfg
