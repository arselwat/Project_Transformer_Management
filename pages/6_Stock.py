# pages/6_Stock.py
from __future__ import annotations
from pathlib import Path
import time as _t
import pandas as pd
import streamlit as st

from core.inventory import services as inv
from core.inventory.storage import load_movements
from core.maintenance.reporting_plus import SPARE_PARTS  # même liste que le PDF

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None

st.set_page_config(page_title="Stock — Simple", page_icon="📦", layout="wide")
st.title("📦 Gestion de Stock")

tab_list, tab_move, tab_alert = st.tabs(["🗂️ Articles (pièces recommandées)", "↕️ Mouvements", "🔔 Alertes seuil"])


# ============ 1) Articles ============

with tab_list:
    st.subheader("Pièces de rechange recommandées")

    # 1) on charge les pièces existantes
    parts = inv.list_parts_as_dicts() or []
    by_code = {str(p.get("code")): p for p in parts if p.get("code")}

    # 2) on s'assure que TOUTES les pièces SPARE_PARTS existent dans le fichier d'inventaire
    upserts_init = []
    for sp in SPARE_PARTS:
        code = sp.get("code") if isinstance(sp, dict) else None
        if not code:
            continue
        if code not in by_code:
            upserts_init.append({
                "code": code,
                "nom": sp.get("piece", ""),
                "famille": sp.get("categorie", ""),
                "quantite_dispo": 0,
                "seuil_min": 0,
                "localisation": "",
                "prix_unitaire": 0.0,
                "fournisseur": "",
            })
    if upserts_init:
        inv.upsert_parts(upserts_init)
        parts = inv.list_parts_as_dicts() or []
        by_code = {str(p.get("code")): p for p in parts if p.get("code")}

    # 3) tableau de synthèse (lecture seule)
    rows_disp = []
    for sp in SPARE_PARTS:
        code = sp.get("code") if isinstance(sp, dict) else None
        if not code:
            continue
        base = by_code.get(code, {})
        rows_disp.append({
            "Code": code,
            "Catégorie": sp.get("categorie", ""),
            "Pièce de rechange": sp.get("piece", ""),
            "Quantité recommandée": sp.get("qte_reco", 0),
            "Criticité": sp.get("criticite", ""),
            "Qté disponible": base.get("quantite_dispo", 0),
            "Seuil min": base.get("seuil_min", 0),
            "Remarques": sp.get("remarques", ""),
        })

    df_disp = pd.DataFrame(rows_disp)
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

    st.markdown("### ➕ Mettre à jour le stock des pièces recommandées")
    st.caption("Saisis uniquement les **quantités à ajouter** et ajuste les **seuils minimums** si nécessaire.")

    with st.form("maj_spares"):
        qty_inputs = {}
        seuil_inputs = {}
        for sp in SPARE_PARTS:
            code = sp.get("code") if isinstance(sp, dict) else None
            if not code:
                continue
            base = by_code.get(code, {})
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{code}** — {sp.get('piece','')}")
            with col2:
                qty_inputs[code] = st.number_input(
                    f"Qté à ajouter [{code}]",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"qadd_{code}",
                )
            with col3:
                seuil_inputs[code] = st.number_input(
                    f"Seuil min [{code}]",
                    min_value=0,
                    value=int(base.get("seuil_min", 0) or 0),
                    step=1,
                    key=f"seuil_{code}",
                )

        submitted = st.form_submit_button("✅ Enregistrer les mises à jour")
        if submitted:
            # mouvements IN + mise à jour des seuils
            upserts = []
            for sp in SPARE_PARTS:
                code = sp.get("code") if isinstance(sp, dict) else None
                if not code:
                    continue
                base = by_code.get(code, {})
                add = int(qty_inputs.get(code) or 0)
                seuil = int(seuil_inputs.get(code) or 0)

                if add > 0:
                    ok, msg = inv.move_in(
                        code,
                        add,
                        reason="Maj stock pièces recommandées",
                        ref="MAJ_STOCK_SPARES",
                        user="app",
                    )
                    if not ok:
                        st.error(f"{code}: {msg}")

                if (not base) or int(base.get("seuil_min", 0) or 0) != seuil:
                    rec = {"code": code, "seuil_min": seuil}
                    # garde aussi nom/famille la première fois
                    if not base:
                        rec.update({
                            "nom": sp.get("piece", ""),
                            "famille": sp.get("categorie", ""),
                        })
                    upserts.append(rec)

            if upserts:
                inv.upsert_parts(upserts)

            st.success("Stock des pièces recommandées mis à jour.")
            st.rerun()
