# pages/3_Transformateurs.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from core.transformer.store import (
    list_transformers, get_transformer, upsert_transformer,
    delete_transformer, set_status
)

st.set_page_config(page_title="Transformateurs", page_icon="🔌", layout="wide")
st.title("🔌 Transformateurs — Gestion (CRUD centralisé)")

# ---------- Liste ----------
rows = list_transformers(include_retired=True)
with st.container(border=True):
    st.subheader("Liste")
    if rows:
        cols = ["equipment_code","name","site","rated_mva","V1n_kV","V2n_kV","f_nominal","vector_group","status","commissioned_on","notes"]
        view = [{k: r.get(k, "") for k in cols} for r in rows]
        st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun transformateur enregistré.")

# ---------- Formulaire ----------
with st.container(border=True):
    st.subheader("Ajouter / Modifier")
    cA, cB, cC = st.columns(3)
    with cA:
        equipment_code = st.text_input("Code équipement*", value=st.session_state.get("tfm_code",""), key="tfm_code_in")
        name = st.text_input("Nom/Modèle", value=st.session_state.get("tfm_name",""), key="tfm_name_in")
        site = st.text_input("Site*", value=st.session_state.get("tfm_site",""), key="tfm_site_in")
        commissioned_on = st.text_input("Date mise en service (YYYY-MM-DD)", value=st.session_state.get("tfm_comm",""), key="tfm_comm_in")
    with cB:
        rated_mva = st.number_input("Puissance (MVA)", min_value=0.0, value=float(st.session_state.get("tfm_mva", 25.0)), step=0.1, key="tfm_mva_in")
        V1n_kV = st.number_input("V1n (kV)", min_value=0.001, value=float(st.session_state.get("tfm_v1", 220.0)), step=0.1, format="%.3f", key="tfm_v1_in")
        V2n_kV = st.number_input("V2n (kV)", min_value=0.001, value=float(st.session_state.get("tfm_v2", 20.0)), step=0.1, format="%.3f", key="tfm_v2_in")
        f_nominal = st.number_input("Fréquence (Hz)", min_value=1.0, value=float(st.session_state.get("tfm_f", 50.0)), step=1.0, key="tfm_f_in")
    with cC:
        vector_group = st.text_input("Groupe vectoriel", value=st.session_state.get("tfm_vg","Dyn5"), key="tfm_vg_in")
        status = st.selectbox("Statut", options=["active","retired"], index=0, key="tfm_status_in")
        notes = st.text_input("Notes", value=st.session_state.get("tfm_notes",""), key="tfm_notes_in")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("✅ Enregistrer / MAJ", type="primary", use_container_width=True, key="btn_save"):
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
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.session_state.update({
                    "tfm_code": rec["equipment_code"],
                    "tfm_name": rec["name"],
                    "tfm_site": rec["site"],
                    "tfm_comm": rec["commissioned_on"],
                    "tfm_mva": rec["rated_mva"],
                    "tfm_v1": rec["V1n_kV"],
                    "tfm_v2": rec["V2n_kV"],
                    "tfm_f": rec["f_nominal"],
                    "tfm_vg": rec["vector_group"],
                    "tfm_notes": rec["notes"],
                })
            st.rerun()
    with c2:
        if st.button("🗑️ Supprimer définitivement", use_container_width=True, key="btn_del"):
            ok, msg = delete_transformer(equipment_code.strip())
            st.success(msg) if ok else st.error(msg)
            st.rerun()
    with c3:
        if st.button("🚫 Marquer retiré", use_container_width=True, key="btn_retire"):
            ok, msg = set_status(equipment_code.strip(), active=False)
            st.success(msg) if ok else st.error(msg)
            st.rerun()
    with c4:
        if st.button("✅ Marquer actif", use_container_width=True, key="btn_activate"):
            ok, msg = set_status(equipment_code.strip(), active=True)
            st.success(msg) if ok else st.error(msg)
            st.rerun()
    with c5:
        # -------- Bouton "Utiliser ce transfo" --------
        # On passe le code via query params, compatible partout.
        if st.button("▶️ Utiliser ce transfo", type="secondary", use_container_width=True, key="btn_use"):
            if equipment_code.strip():
                try:
                    # Streamlit >=1.25
                    st.switch_page("pages/4_Visualisation_temps_reel.py")
                except Exception:
                    pass
                st.query_params["tfm"] = equipment_code.strip()

# ---------- Charger un existant ----------
with st.container(border=True):
    st.subheader("Charger un existant")
    codes = [r.get("equipment_code","") for r in rows] if rows else []
    sel = st.selectbox("Choisir", options=[""]+codes, key="sel_exist")
    if sel:
        r = get_transformer(sel) or {}
        for k, v in {
            "tfm_code": r.get("equipment_code",""),
            "tfm_name": r.get("name",""),
            "tfm_site": r.get("site",""),
            "tfm_comm": r.get("commissioned_on",""),
            "tfm_mva": r.get("rated_mva", 25.0),
            "tfm_v1": r.get("V1n_kV", 220.0),
            "tfm_v2": r.get("V2n_kV", 20.0),
            "tfm_f": r.get("f_nominal", 50.0),
            "tfm_vg": r.get("vector_group","Dyn5"),
            "tfm_notes": r.get("notes",""),
        }.items():
            st.session_state[k] = v
        st.info(f"Chargé: {sel} (champs pré-remplis ci-dessus).")
