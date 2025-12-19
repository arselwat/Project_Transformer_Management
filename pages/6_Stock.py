# pages/6_Stock.py
from __future__ import annotations

from pathlib import Path
import time as _t
import pandas as pd
import streamlit as st

from core.inventory import services as inv
from core.inventory.storage import load_movements

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None


st.set_page_config(page_title="Stock — Simplifié", page_icon="📦", layout="wide")
st.title("📦 Gestion de Stock (simplifiée)")

tab_hist, tab_stock = st.tabs(["📑 Historique mouvements", "📦 Stock & Alertes seuil"])


# ---------------------------
# Utils
# ---------------------------
def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _low_stock_list() -> list[dict]:
    # low stock = quantite_dispo < seuil_min (threshold_factor=1.0)
    low = inv.low_stock(threshold_factor=1.0) or []
    # normalise
    out = []
    for r in low:
        if isinstance(r, dict):
            out.append(r)
    return out


def _send_low_stock_alert(low: list[dict], *, force: bool = False) -> None:
    """
    Envoi auto anti-spam: 10 minutes.
    - force=True => ignore anti-spam (à éviter, mais possible)
    """
    if not low:
        return
    if not notify_stock_alerts:
        return

    now = _t.time()
    last = st.session_state.get("_last_stock_alert_ts", 0.0)
    if (not force) and (now - last <= 600):
        return

    try:
        res = notify_stock_alerts(low)
        if isinstance(res, dict) and res.get("ok"):
            st.session_state["_last_stock_alert_ts"] = now
            st.toast("🔔 Alerte stock envoyée automatiquement.")
        else:
            st.toast("⚠️ Alerte auto non envoyée (vérifie la config SMTP).")
    except Exception:
        # on ne bloque pas la page
        pass


# ============================================================
# TAB 1 — HISTORIQUE
# ============================================================
with tab_hist:
    st.subheader("Historique récent des mouvements")

    mv = load_movements(limit=800)
    dmv = pd.DataFrame(mv) if mv else pd.DataFrame(columns=["ts", "type", "code", "qty", "reason", "ref", "user"])

    if not dmv.empty:
        dmv["date"] = pd.to_datetime(dmv["ts"], unit="s", errors="coerce")
        dmv = dmv.sort_values("ts", ascending=False)

        st.dataframe(
            dmv[["date", "type", "code", "qty", "reason", "ref", "user"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aucun mouvement enregistré pour le moment.")


# ============================================================
# TAB 2 — STOCK + ACTIONS + ALERTES AUTO
# ============================================================
with tab_stock:
    st.subheader("Stock des pièces (actions directes)")

    # ----------- Création d'articles (après achat / nouveau produit)
    st.markdown("### ➕ Ajouter un article au stock")
    st.caption("Crée une nouvelle référence dans le stock (avant de pouvoir ajouter/retirer des quantités).")

    with st.form("create_part"):
        c1, c2, c3 = st.columns(3)
        with c1:
            code_new = st.text_input("Code *")
            nom_new = st.text_input("Nom")
        with c2:
            famille_new = st.text_input("Famille")
            localisation_new = st.text_input("Localisation")
        with c3:
            seuil_new = st.number_input("Seuil minimum", min_value=0, value=0, step=1)
            prix_new = st.number_input("Prix unitaire", min_value=0.0, value=0.0, step=1.0)

        q_init = st.number_input("Quantité initiale", min_value=0, value=0, step=1)

        ok_create = st.form_submit_button("✅ Enregistrer l’article")
        if ok_create:
            code_new = (code_new or "").strip()
            if not code_new:
                st.error("Le champ Code est obligatoire.")
            else:
                # crée / met à jour l'article
                inv.upsert_parts([{
                    "code": code_new,
                    "nom": nom_new,
                    "famille": famille_new,
                    "localisation": localisation_new,
                    "prix_unitaire": float(prix_new),
                    "seuil_min": int(seuil_new),
                    "quantite_dispo": 0,
                }])

                # stock initial = mouvement IN (donc visible dans Historique)
                if int(q_init) > 0:
                    inv.move_in(
                        code_new,
                        int(q_init),
                        reason="Achat / Stock initial",
                        ref="INIT",
                        user="app",
                    )

                st.success("Article ajouté au stock.")
                st.rerun()

    st.divider()

    # ----------- Liste stock (avec boutons Ajouter/Retirer)
    parts = inv.list_parts_as_dicts() or []
    if not parts:
        st.warning("Aucun article dans le stock. Ajoute d’abord des articles avec le formulaire ci-dessus.")
        st.stop()

    # tri par code
    parts = sorted(parts, key=lambda x: str(x.get("code", "")))

    # banner alertes
    low = _low_stock_list()
    if low:
        st.warning(f"⚠️ {len(low)} article(s) sous le seuil. Les alertes mail sont automatiques.")
    else:
        st.success("✅ RAS — aucun article sous le seuil.")

    # auto email (si dispo)
    _send_low_stock_alert(low, force=False)

    st.caption("Clique sur Ajouter/Retirer à côté d’un article, saisis la quantité, puis valide.")

    # petite recherche
    q = st.text_input("Recherche (code/nom)", value="").strip().lower()
    if q:
        parts_view = []
        for p in parts:
            blob = f"{p.get('code','')} {p.get('nom','')}".lower()
            if q in blob:
                parts_view.append(p)
    else:
        parts_view = parts

    # tableau interactif “simple” : on affiche + actions
    for p in parts_view:
        code = str(p.get("code") or "").strip()
        if not code:
            continue

        nom = str(p.get("nom") or "")
        fam = str(p.get("famille") or "")
        loc = str(p.get("localisation") or "")
        qte = _as_int(p.get("quantite_dispo"), 0)
        seuil = _as_int(p.get("seuil_min"), 0)

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2.2, 2.6, 1.2, 2.0, 2.0])

            with c1:
                st.markdown(f"**{code}**")
                st.caption(nom)

            with c2:
                st.write(f"Famille: {fam or '—'}")
                st.write(f"Localisation: {loc or '—'}")

            with c3:
                st.metric("Qté", qte)
                st.caption(f"Seuil: {seuil}")

            # ---- Ajouter
            with c4:
                qty_add = st.number_input(
                    "Ajouter",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"add_{code}",
                )
                if st.button("➕ Valider", key=f"btn_add_{code}", use_container_width=True):
                    if qty_add <= 0:
                        st.info("Saisis une quantité > 0.")
                    else:
                        ok, msg = inv.move_in(
                            code,
                            int(qty_add),
                            reason="Achat / Ajout stock",
                            ref="IN_UI",
                            user="app",
                        )
                        if ok:
                            # après mouvement => re-check alertes (auto)
                            low2 = _low_stock_list()
                            _send_low_stock_alert(low2, force=False)
                            st.success("Ajout enregistré (mouvement IN).")
                            st.rerun()
                        else:
                            st.error(msg)

            # ---- Retirer
            with c5:
                qty_out = st.number_input(
                    "Retirer",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"out_{code}",
                )
                if st.button("➖ Valider", key=f"btn_out_{code}", use_container_width=True):
                    if qty_out <= 0:
                        st.info("Saisis une quantité > 0.")
                    else:
                        ok, msg = inv.move_out(
                            code,
                            int(qty_out),
                            reason="Consommation / Maintenance",
                            ref="OUT_UI",
                            user="app",
                        )
                        if ok:
                            # après retrait => si sous seuil => mail auto
                            low2 = _low_stock_list()
                            _send_low_stock_alert(low2, force=False)
                            st.success("Retrait enregistré (mouvement OUT).")
                            st.rerun()
                        else:
                            st.error(msg)

    st.divider()

    # ---- vue alertes (lecture)
    st.markdown("### 🔔 Articles sous seuil (lecture)")
    if low:
        st.dataframe(pd.DataFrame(low), use_container_width=True, hide_index=True)
        st.caption("Les alertes mail sont envoyées automatiquement (anti-spam 10 minutes).")
    else:
        st.caption("Aucun article sous seuil.")
