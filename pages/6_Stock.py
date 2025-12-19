# pages/6_Stock.py
from __future__ import annotations

from pathlib import Path
import time as _t
import pandas as pd
import streamlit as st

from core.inventory import services as inv
from core.inventory.storage import load_movements
from core.maintenance.reporting_plus import SPARE_PARTS  # liste pièces de référence

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None


st.set_page_config(page_title="Stock — Simple", page_icon="📦", layout="wide")
st.title("📦 Gestion de Stock (simplifiée)")

# 2 onglets seulement : Historique + Stock/Alertes
tab_hist, tab_stock = st.tabs(["↕️ Historique mouvements", "🔔 Stock & Alertes seuil"])


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _ensure_spares_exist():
    """S’assure que toutes les pièces recommandées existent dans l’inventaire."""
    parts = inv.list_parts_as_dicts() or []
    by_code = {str(p.get("code")): p for p in parts if p.get("code")}

    upserts = []
    for sp in SPARE_PARTS:
        if not isinstance(sp, dict):
            continue
        code = str(sp.get("code") or "").strip()
        if not code:
            continue
        if code not in by_code:
            upserts.append({
                "code": code,
                "nom": sp.get("piece", ""),
                "famille": sp.get("categorie", ""),
                "quantite_dispo": 0,
                "seuil_min": 0,
                "localisation": "",
                "prix_unitaire": 0.0,
                "fournisseur": "",
            })

    if upserts:
        inv.upsert_parts(upserts)

    # reload
    parts = inv.list_parts_as_dicts() or []
    by_code = {str(p.get("code")): p for p in parts if p.get("code")}
    return parts, by_code


def _send_alert_if_needed():
    """
    Envoie automatiquement une alerte si des pièces sont sous seuil.
    Anti-spam: 10 minutes.
    """
    if not notify_stock_alerts:
        return

    low = inv.low_stock(threshold_factor=1.0) or []
    if not low:
        return

    last = st.session_state.get("_last_stock_alert_ts", 0.0)
    if _t.time() - last < 600:
        return

    try:
        res = notify_stock_alerts(low)
        if isinstance(res, dict) and res.get("ok"):
            st.session_state["_last_stock_alert_ts"] = _t.time()
            st.toast("📨 Alerte stock envoyée automatiquement (pièces sous seuil).")
        else:
            # on évite de spammer l’UI
            st.toast("⚠️ Alerte stock non envoyée (vérifier config SMTP).")
    except Exception:
        st.toast("⚠️ Erreur lors de l’envoi d’alerte stock.")


def _do_move(code: str, qty: int, direction: str):
    """
    direction = 'IN' ou 'OUT'
    """
    code = (code or "").strip()
    if not code:
        st.error("Code article invalide.")
        return

    if qty <= 0:
        st.error("Quantité doit être > 0.")
        return

    if direction == "IN":
        ok, msg = inv.move_in(
            code,
            int(qty),
            reason="Mise à jour stock (UI)",
            ref="UI_STOCK",
            user="app",
        )
    else:
        ok, msg = inv.move_out(
            code,
            int(qty),
            reason="Retrait stock (UI)",
            ref="UI_STOCK",
            user="app",
        )

    if ok:
        st.success(msg)
        _send_alert_if_needed()
        st.rerun()
    else:
        st.error(msg)


# ------------------------------------------------------------
# TAB 1 : Historique seulement
# ------------------------------------------------------------
with tab_hist:
    st.subheader("Historique récent")

    mv = load_movements(limit=600)
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
        st.caption("Aucun mouvement enregistré.")


# ------------------------------------------------------------
# TAB 2 : Stock + boutons Ajouter/Retirer + alertes auto
# ------------------------------------------------------------
with tab_stock:
    st.subheader("Stock des pièces (actions directes)")

    parts, by_code = _ensure_spares_exist()

    # 1) Construire rows
    rows = []

    # --- Priorité: SPARE_PARTS si disponible
    if isinstance(SPARE_PARTS, list) and len(SPARE_PARTS) > 0:
        for sp in SPARE_PARTS:
            if not isinstance(sp, dict):
                continue
            code = str(sp.get("code") or "").strip()
            if not code:
                continue
            base = by_code.get(code, {})
            rows.append({
                "code": code,
                "nom": sp.get("piece", "") or base.get("nom", ""),
                "famille": sp.get("categorie", "") or base.get("famille", ""),
                "quantite_dispo": int(base.get("quantite_dispo", 0) or 0),
                "seuil_min": int(base.get("seuil_min", 0) or 0),
                "criticite": sp.get("criticite", ""),
            })
    else:
        # --- Fallback: afficher tous les articles existants en inventaire
        for p in (parts or []):
            code = str(p.get("code") or "").strip()
            if not code:
                continue
            rows.append({
                "code": code,
                "nom": p.get("nom", ""),
                "famille": p.get("famille", ""),
                "quantite_dispo": int(p.get("quantite_dispo", 0) or 0),
                "seuil_min": int(p.get("seuil_min", 0) or 0),
                "criticite": "",
            })

    # ✅ Tri
    rows = sorted(rows, key=lambda x: x["code"])

    # 2) Tableau visible (même si on garde les boutons dessous)
    df_disp = pd.DataFrame(rows)
    if df_disp.empty:
        st.warning("Aucun article à afficher (SPARE_PARTS vide + inventaire vide).")
        st.stop()

    # bandeau alerte
    low_now = inv.low_stock(threshold_factor=1.0) or []
    if low_now:
        st.warning(f"⚠️ {len(low_now)} article(s) sous le seuil. Les alertes mail sont automatiques.")
    else:
        st.success("✅ RAS — aucun article sous le seuil.")

    st.caption("Clique sur Ajouter/Retirer à côté d’un article, saisis la quantité, puis valide.")

    st.dataframe(
        df_disp[["code", "nom", "famille", "quantite_dispo", "seuil_min", "criticite"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Actions rapides (Ajouter / Retirer)")
    st.caption("Les mouvements sont enregistrés automatiquement dans l’onglet Historique.")

    # 3) Boutons par ligne
    for r in rows:
        code = r["code"]
        nom = r["nom"]
        qty = r["quantite_dispo"]
        seuil = r["seuil_min"]

        c1, c2, c3, c4, c5 = st.columns([2.2, 3.5, 1.2, 1.2, 2.5])
        with c1:
            st.write(f"**{code}**")
        with c2:
            st.write(nom)
        with c3:
            st.write(f"Qté: **{qty}**")
        with c4:
            st.write(f"Seuil: **{seuil}**")
        with c5:
            colA, colB = st.columns(2)

            with colA:
                with st.popover("➕ Ajouter", use_container_width=True):
                    q_add = st.number_input(
                        "Quantité à ajouter",
                        min_value=1, value=1, step=1,
                        key=f"add_{code}"
                    )
                    if st.button("Valider ajout", key=f"btn_add_{code}", type="primary"):
                        _do_move(code, int(q_add), "IN")

            with colB:
                with st.popover("➖ Retirer", use_container_width=True):
                    q_out = st.number_input(
                        "Quantité à retirer",
                        min_value=1, value=1, step=1,
                        key=f"out_{code}"
                    )
                    if st.button("Valider retrait", key=f"btn_out_{code}"):
                        _do_move(code, int(q_out), "OUT")

        st.divider()

    # Auto alert (anti-spam 10 min)
    _send_alert_if_needed()
    st.caption("⚠️ Les actions de gestion de stock via cette interface sont simplifiées et ne remplacent pas un ERP complet.")