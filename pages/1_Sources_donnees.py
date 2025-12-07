from __future__ import annotations
import io, sqlite3, time
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
import streamlit as st
from core.security.auth import require_login

st.set_page_config(page_title="Transformateurs", page_icon="🔌", layout="wide")

require_login()  # tant que auth_ok n’est pas True, cette page est bloquée

# ... le reste de ta page ...

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
st.title("📥 Sources de données")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
CONSOLIDATED = DATA_DIR / "failures_saved.csv"
DB_PATH = DATA_DIR / "reliability.sqlite"

# ------- DB helpers (autonomes, ne cassent rien) -------
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
    if df.empty: return 0
    cols = ["equipment_code","ttf_h","duree_rep_h"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    rows = df[cols].values.tolist()
    with _db_conn() as cx:
        cx.executemany("INSERT INTO failures (equipment_code, ttf_h, duree_rep_h, source) VALUES (?,?,?,?)",
                       [(*r, source) for r in rows])
        cx.commit()
    return len(rows)

def clear_db_and_csv():
    if CONSOLIDATED.exists():
        try: CONSOLIDATED.unlink()
        except Exception: pass
    with _db_conn() as cx:
        cx.execute("DELETE FROM failures"); cx.commit()

init_db()

# ------------------------- Utils -------------------------
REQUIRED = ["equipment_code", "ttf_h"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code", "equipement": "equipment_code",
        "code_equipement": "equipment_code", "eqp": "equipment_code",
        "ttf": "ttf_h", "ttf_hours": "ttf_h",
        "duree_rep_h": "duree_rep_h", "mttr_h": "duree_rep_h",
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
            if dh > 0: out.append({"equipment_code": str(eq), "ttf_h": dh})
    return pd.DataFrame(out)

def _put_session(df: pd.DataFrame, src_label: str):
    st.session_state["failures_df"] = df.copy()
    st.session_state["failures_src"] = src_label

# ------------------------- Onglets -------------------------
tab_csv, tab_mqtt = st.tabs(["Fichier CSV / DB", "Données MQTT (réglages)"])

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
        content = up.read()
        try:
            df_loaded = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            st.error(f"Lecture CSV: {e}")

    if df_loaded is not None:
        df_loaded = _normalize_columns(df_loaded)

        if has_timestamps:
            existing = df_loaded.columns.tolist()
            eq_col = st.selectbox("Colonne équipement", options=existing, index=0)
            ts_col = st.selectbox("Colonne horodatage panne", options=existing, index=min(1, len(existing)-1))
            if st.button("🧮 Construire ttf_h"):
                try:
                    ttf_df = _compute_ttf_from_timestamps(df_loaded, eq_col, ts_col)
                    st.success(f"{len(ttf_df)} TTF construits.")
                    st.dataframe(ttf_df.head(50), use_container_width=True, hide_index=True)
                    _put_session(ttf_df, src_label=f"upload:{up.name} (TTF construit)")
                except Exception as e:
                    st.error(f"Construction TTF: {e}")
        else:
            missing = [c for c in REQUIRED if c not in df_loaded.columns]
            if missing:
                st.warning(f"Colonnes manquantes: {missing}. Essaie l'option horodatage ou renomme tes colonnes.")
            else:
                st.success(f"Colonnes OK: {REQUIRED}")
                st.dataframe(df_loaded.head(50), use_container_width=True, hide_index=True)
                cL, cR = st.columns(2)
                with cL:
                    if st.button("📥 Charger en session"):
                        _put_session(df_loaded, src_label=f"upload:{up.name}")
                        st.toast("Jeu chargé en session ✅")
                with cR:
                    if st.button("💾 Enregistrer → CSV (data/failures_saved.csv)"):
                        try:
                            df_loaded.to_csv(CONSOLIDATED, index=False, encoding="utf-8")
                            st.success(f"Écrit: {CONSOLIDATED}")
                        except Exception as e:
                            st.error(f"Écriture: {e}")

    st.divider()
    st.subheader("Sauvegarde projet / Base SQLite")

    # Vue du jeu en session
    if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
        cur = st.session_state["failures_df"]
        st.caption(f"Source: {st.session_state.get('failures_src','')}")
        st.dataframe(cur.head(30), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 Sauver en CSV (consolidé)"):
                try:
                    cur.to_csv(CONSOLIDATED, index=False, encoding="utf-8")
                    st.success(f"Écrit: {CONSOLIDATED}")
                except Exception as e:
                    st.error(f"Écriture: {e}")
        with c2:
            if st.button("⬆️ Synchroniser vers SQLite"):
                try:
                    n = bulk_insert_failures(cur, source=st.session_state.get("failures_src","session"))
                    st.success(f"{n} lignes insérées dans {DB_PATH.name}")
                except Exception as e:
                    st.error(f"SQLite: {e}")
        with c3:
            if st.button("🗑️ Purge CSV + DB (irréversible)"):
                clear_db_and_csv()
                st.success("Consolidé effacé et table 'failures' vidée.")

    else:
        c0, c1 = st.columns([2,1])
        with c0:
            if CONSOLIDATED.exists():
                st.info("Aucun jeu en session, mais un fichier consolidé existe.")
                if st.button("Charger le consolidé en session"):
                    try:
                        df0 = pd.read_csv(CONSOLIDATED)
                        _put_session(df0, src_label=str(CONSOLIDATED))
                        st.success("Chargé en session ✅")
                    except Exception as e:
                        st.error(f"Lecture consolidé: {e}")
        with c1:
            if st.button("🗑️ Purge CSV + DB"):
                clear_db_and_csv()
                st.success("Consolidé effacé et table 'failures' vidée.")

    st.divider()
    st.subheader("Analyse rapide (organigramme + indicateurs)")

    try:
        from core.reliability.unify import compute_bundle, UnifyOptions
        if st.button("▶️ Lancer l’analyse maintenant"):
            src_df = st.session_state.get("failures_df")
            bundle = compute_bundle(session_df=src_df, options=UnifyOptions(force_weibull_2p=True))
            # export
            out_csv = DATA_DIR / "last_metrics.csv"
            bundle.metrics_df.to_csv(out_csv, index=False, encoding="utf-8")
            st.success(f"Rapport synthèse → {out_csv}")

            cA, cB = st.columns([2,1])
            with cA:
                st.dataframe(bundle.metrics_df, use_container_width=True, hide_index=True)
            with cB:
                st.json(bundle.pipeline_by_eq, expanded=False)
    except Exception as e:
        st.warning(f"Analyse rapide indisponible: {e}")

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

    if st.button("💾 Enregistrer paramètres MQTT"):
        new_cfg = {"host": host, "port": int(port), "site": site,
                   "equipement": eqp, "topic_base": topic_base}
        save_mqtt(new_cfg)
        st.success(f"Sauvegardé → {mqtt_cfg_file}")
        st.session_state["mqtt_cfg"] = new_cfg
