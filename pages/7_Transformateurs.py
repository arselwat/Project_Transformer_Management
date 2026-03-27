from __future__ import annotations

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.ui import render_shell, render_page_header
from core.transformer.store import (
    list_transformers,
    get_transformer,
    upsert_transformer,
    delete_transformer,
    set_status,
)

st.set_page_config(page_title="Transformateurs", page_icon="🔌", layout="wide")
require_login()

render_shell("pages/7_Transformateurs.py")
render_page_header(
    "Transformateurs",
    "Gestion centralisée des transformateurs : ajout, modification, activation et retrait.",
    "🔌",
)

for key, default in {
    "tfm_code": "",
    "tfm_name": "",
    "tfm_site": "",
    "tfm_comm": "",
    "tfm_mva": 25.0,
    "tfm_v1": 220.0,
    "tfm_v2": 20.0,
    "tfm_f": 50.0,
    "tfm_vg": "Dyn5",
    "tfm_status": "active",
    "tfm_notes": "",
}.items():
    st.session_state.setdefault(key, default)

rows = list_transformers(include_retired=True)

with st.container(border=True):
    st.subheader("Liste")
    if rows:
        cols = [
            "equipment_code",
            "name",
            "site",
            "rated_mva",
            "V1n_kV",
            "V2n_kV",
            "f_nominal",
            "vector_group",
            "status",
            "commissioned_on",
            "notes",
        ]
        view = [{k: r.get(k, "") for k in cols} for r in rows]
        st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun transformateur enregistré.")

with st.container(border=True):
    st.subheader("Charger un transformateur existant")

    codes = [r.get("equipment_code", "") for r in rows] if rows else []
    sel = st.selectbox("Choisir", options=[""] + codes)

    c_load1, c_load2 = st.columns([1, 1])
    with c_load1:
        if st.button("Charger", use_container_width=True):
            if sel:
                r = get_transformer(sel) or {}
                st.session_state["tfm_code"] = r.get("equipment_code", "")
                st.session_state["tfm_name"] = r.get("name", "")
                st.session_state["tfm_site"] = r.get("site", "")
                st.session_state["tfm_comm"] = r.get("commissioned_on", "")
                st.session_state["tfm_mva"] = r.get("rated_mva", 25.0)
                st.session_state["tfm_v1"] = r.get("V1n_kV", 220.0)
                st.session_state["tfm_v2"] = r.get("V2n_kV", 20.0)
                st.session_state["tfm_f"] = r.get("f_nominal", 50.0)
                st.session_state["tfm_vg"] = r.get("vector_group", "Dyn5")
                st.session_state["tfm_status"] = r.get("status", "active")
                st.session_state["tfm_notes"] = r.get("notes", "")
                st.success(f"Transformateur {sel} chargé.")
                st.rerun()

    with c_load2:
        if st.button("Vider le formulaire", use_container_width=True):
            st.session_state["tfm_code"] = ""
            st.session_state["tfm_name"] = ""
            st.session_state["tfm_site"] = ""
            st.session_state["tfm_comm"] = ""
            st.session_state["tfm_mva"] = 25.0
            st.session_state["tfm_v1"] = 220.0
            st.session_state["tfm_v2"] = 20.0
            st.session_state["tfm_f"] = 50.0
            st.session_state["tfm_vg"] = "Dyn5"
            st.session_state["tfm_status"] = "active"
            st.session_state["tfm_notes"] = ""
            st.rerun()

with st.container(border=True):
    st.subheader("Ajouter / Modifier")

    cA, cB, cC = st.columns(3)
    with cA:
        equipment_code = st.text_input("Code équipement*", key="tfm_code")
        name = st.text_input("Nom / Modèle", key="tfm_name")
        site = st.text_input("Site*", key="tfm_site")
        commissioned_on = st.text_input("Date mise en service (YYYY-MM-DD)", key="tfm_comm")

    with cB:
        rated_mva = st.number_input("Puissance (MVA)", min_value=0.0, step=0.1, key="tfm_mva")
        V1n_kV = st.number_input("V1n (kV)", min_value=0.001, step=0.1, format="%.3f", key="tfm_v1")
        V2n_kV = st.number_input("V2n (kV)", min_value=0.001, step=0.1, format="%.3f", key="tfm_v2")
        f_nominal = st.number_input("Fréquence (Hz)", min_value=1.0, step=1.0, key="tfm_f")

    with cC:
        vector_group = st.text_input("Groupe vectoriel", key="tfm_vg")
        status = st.selectbox("Statut", options=["active", "retired"], index=0 if st.session_state.get("tfm_status", "active") == "active" else 1)
        notes = st.text_input("Notes", key="tfm_notes")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("Enregistrer / MAJ", type="primary", use_container_width=True):
            rec = {
                "equipment_code": equipment_code.strip(),
                "name": name.strip(),
                "site": site.strip(),
                "rated_mva": rated_mva,
                "V1n_kV": V1n_kV,
                "V2n_kV": V2n_kV,
                "f_nominal": f_nominal,
                "vector_group": vector_group.strip(),
                "status": status.strip(),
                "commissioned_on": commissioned_on.strip(),
                "notes": notes.strip(),
            }
            ok, msg = upsert_transformer(rec)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with c2:
        if st.button("Supprimer", use_container_width=True):
            ok, msg = delete_transformer(equipment_code.strip())
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with c3:
        if st.button("Marquer retiré", use_container_width=True):
            ok, msg = set_status(equipment_code.strip(), active=False)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with c4:
        if st.button("Marquer actif", use_container_width=True):
            ok, msg = set_status(equipment_code.strip(), active=True)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with c5:
        if st.button("Utiliser ce transfo", use_container_width=True):
            code = equipment_code.strip()
            if code:
                st.query_params["tfm"] = code
                try:
                    st.switch_page("pages/8_Visualisation_temps_reel.py")
                except Exception:
                    st.success(f"Transformateur sélectionné : {code}. Va sur Temps réel.")
            else:
                st.warning("Renseigne d’abord un code équipement.")