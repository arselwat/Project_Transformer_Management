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

tab_list, tab_move, tab_alert = st.tabs(
    ["🗂️ Articles (pièces recommandées)", "↕️ Mouvements", "🔔 Alertes seuil"]
)

# ============ 1) Articles ============
with tab_list:
    st.subheader("Pièces de rechange recommandées")

    # 1) on charge les pièces existantes
    parts = inv.list_parts_as_dicts() or []
    by_code = {str(p.get("code")): p for p in parts if p.get("code")}

    # 2) on s'assure que TOUTES les pièces SPARE_PARTS existent dans le fichier d'inventaire
    upserts_init = []
    for sp in SPARE_PARTS:
        if not isinstance(sp, dict):
            continue
        code = str(sp.get("code") or "").strip()
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
        if not isinstance(sp, dict):
            continue
        code = str(sp.get("code") or "").strip()
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
    st.caption("Saisis uniquement les quantités à ajouter et ajuste les seuils minimums si nécessaire.")

    with st.form("maj_spares"):
        qty_inputs = {}
        seuil_inputs = {}
        for sp in SPARE_PARTS:
            if not isinstance(sp, dict):
                continue
            code = str(sp.get("code") or "").strip()
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
            upserts = []
            for sp in SPARE_PARTS:
                if not isinstance(sp, dict):
                    continue
                code = str(sp.get("code") or "").strip()
                if not code:
                    continue
                base = by_code.get(code, {})
                add = int(qty_inputs.get(code, 0) or 0)
                seuil = int(seuil_inputs.get(code, 0) or 0)

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

    st.markdown("---")
    st.markdown("#### Articles supplémentaires (optionnel)")
    st.caption("Pour ajouter d’autres références hors liste recommandée, utilise ce formulaire.")

    c1, c2, c3 = st.columns(3)
    with c1:
        code_new = st.text_input("Code", key="code_extra")
        nom_new = st.text_input("Nom", key="nom_extra")
    with c2:
        fam_new = st.text_input("Famille", key="fam_extra")
        loc_new = st.text_input("Localisation", key="loc_extra")
    with c3:
        seuil_new = st.number_input("Seuil min", min_value=0, value=0, step=1, key="seuil_extra")
        prix_new = st.number_input("Prix unitaire", min_value=0.0, value=0.0, step=1.0, key="prix_extra")

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Enregistrer article supplémentaire"):
            if not code_new.strip():
                st.error("Code requis.")
            else:
                inv.upsert_parts([{
                    "code": code_new.strip(),
                    "nom": nom_new,
                    "famille": fam_new,
                    "localisation": loc_new,
                    "seuil_min": seuil_new,
                    "prix_unitaire": prix_new,
                    "quantite_dispo": 0,
                }])
                st.success("Article supplémentaire enregistré.")
                st.rerun()
    with colB:
        st.caption("Purge / suppression définitive à gérer directement dans le CSV si besoin.")

# ============ 2) Mouvements ============
with tab_move:
    st.subheader("Mouvements rapides")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        m_code = st.text_input("Code article*", key="mcode")
    with c2:
        action = st.selectbox("Type", ["IN", "OUT", "ADJUST"])
    with c3:
        m_qty = st.number_input("Quantité", min_value=1, value=1, step=1)
    with c4:
        who = st.text_input("Utilisateur", value="app")

    reason = st.text_input("Motif", value="")
    ref = st.text_input("Référence", value="")

    if st.button("Valider mouvement", type="primary"):
        ok = False
        msg = ""
        if action == "IN":
            ok, msg = inv.move_in(m_code, m_qty, reason=reason or "IN", ref=ref, user=who)
        elif action == "OUT":
            ok, msg = inv.move_out(m_code, m_qty, reason=reason or "OUT", ref=ref, user=who)
        else:
            q = m_qty
            if reason.strip().startswith("-"):
                q = -abs(m_qty)
            ok, msg = inv.adjust_stock(m_code, q, reason=reason or "ADJUST", ref=ref, user=who)

        st.success(msg) if ok else st.error(msg)
        st.rerun()

    st.markdown("### Historique récent")
    mv = load_movements(limit=400)
    dmv = pd.DataFrame(mv) if mv else pd.DataFrame(columns=["ts", "type", "code", "qty", "reason", "ref", "user"])
    if not dmv.empty:
        dmv["date"] = pd.to_datetime(dmv["ts"], unit="s")
        dmv = dmv.sort_values("ts", ascending=False)
        st.dataframe(
            dmv[["date", "type", "code", "qty", "reason", "ref", "user"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Aucun mouvement.")

# ============ 3) Alertes seuil ============
with tab_alert:
    st.subheader("Alertes seuil stock")

    auto = st.toggle(
        "Activer alertes auto sur cette page",
        value=st.session_state.get("stock_alert_enabled", False),
        key="stock_alert_enabled",
    )

    low = inv.low_stock(threshold_factor=1.0) or []
    if not low:
        st.success("RAS — aucun article sous le seuil.")
    else:
        st.dataframe(pd.DataFrame(low), use_container_width=True, hide_index=True)

        st.markdown("### Envoi manuel / export")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Envoyer alerte maintenant", use_container_width=True):
                if not notify_stock_alerts:
                    out = Path("data/low_stock_alert.csv")
                    pd.DataFrame(low).to_csv(out, index=False)
                    st.info(f"Module d’alerte indisponible — CSV généré : {out}")
                else:
                    try:
                        res = notify_stock_alerts(low)
                        if res.get("ok"):
                            st.success("Alerte stock envoyée (vérifie ta boîte et le spam).")
                        else:
                            err_txt = res.get("email", {}).get("error", "erreur inconnue")
                            st.error(
                                "Alerte non envoyée : "
                                f"{err_txt}.\n\n"
                                "Vérifie la configuration des alertes (alerts_config.json → smtp.host, "
                                "smtp.port, smtp.user, smtp.password, smtp.to_addrs)."
                            )
                    except Exception as e:
                        st.error(f"Alerte : {e}")
        with c2:
            if st.button("⬇️ Export CSV", use_container_width=True):
                out = Path("data/low_stock_alert.csv")
                pd.DataFrame(low).to_csv(out, index=False)
                st.success(f"Exporté → {out}")

    # Auto (anti-spam 10 min)
    if auto and low and notify_stock_alerts:
        last = st.session_state.get("_last_stock_alert_ts", 0.0)
        if _t.time() - last > 600:
            try:
                res = notify_stock_alerts(low)
                if res.get("ok"):
                    st.session_state["_last_stock_alert_ts"] = _t.time()
                    st.toast("Alerte stock auto envoyée.")
                else:
                    st.toast(
                        "Échec envoi alerte auto (voir alerts_config.json / section smtp.to_addrs et paramètres SMTP)."
                    )
            except Exception:
                pass
