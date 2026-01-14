# pages/6_Stock.py
from __future__ import annotations

import time as _t
import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.inventory import services as inv
from core.inventory.storage import load_movements

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None

st.set_page_config(page_title="Stock", page_icon="📦", layout="wide")
require_login()

st.title("📦 Stock (simple)")
st.caption("Ici tu crées les articles, tu fais les entrées/sorties, et tu surveilles les seuils.")

tab_stock, tab_hist = st.tabs(["📦 Stock & Seuils", "📑 Historique mouvements"])

def _as_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def _low_stock_list() -> list[dict]:
    low = inv.low_stock(threshold_factor=1.0) or []
    return [r for r in low if isinstance(r, dict)]

def _send_low_stock_alert(low: list[dict], *, force: bool = False) -> tuple[bool, str]:
    """
    Envoi d'alerte stock.
    - mode auto : force=False avec cooldown 10 min
    - mode manuel : force=True possible
    """
    if not low:
        return False, "Aucun article sous seuil."
    if not notify_stock_alerts:
        return False, "Module notify_stock_alerts indisponible."

    now = _t.time()
    last = float(st.session_state.get("_last_stock_alert_ts", 0.0))

    if (not force) and (now - last <= 600):
        remain = int(600 - (now - last))
        return False, f"Cooldown actif (reste ~{remain}s)."

    try:
        res = notify_stock_alerts(low)
        if isinstance(res, dict) and res.get("ok"):
            st.session_state["_last_stock_alert_ts"] = now
            return True, "Alerte stock envoyée."
        # si le module renvoie autre chose
        st.session_state["_last_stock_alert_ts"] = now
        return True, "Alerte stock envoyée (réponse non standard)."
    except Exception as e:
        return False, f"Erreur envoi: {e}"

with tab_stock:
    st.subheader("➕ Ajouter un article (nouvelle référence)")
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

        q_init = st.number_input("Quantité initiale (si achat direct)", min_value=0, value=0, step=1)
        ok_create = st.form_submit_button("✅ Enregistrer")

        if ok_create:
            code_new = (code_new or "").strip()
            if not code_new:
                st.error("Le champ Code est obligatoire.")
            else:
                inv.upsert_parts([{
                    "code": code_new,
                    "nom": nom_new,
                    "famille": famille_new,
                    "localisation": localisation_new,
                    "prix_unitaire": float(prix_new),
                    "seuil_min": int(seuil_new),
                    "quantite_dispo": 0,
                }])
                if int(q_init) > 0:
                    inv.move_in(code_new, int(q_init), reason="Achat / Stock initial", ref="INIT", user="app")
                st.success("Article créé ✅")
                st.rerun()

    st.divider()
    st.subheader("📦 Articles en stock (entrées / sorties)")

    # ✅ Réduction de liste (par défaut ON)
    copt1, copt2 = st.columns([1.2, 2.0])
    with copt1:
        active_only = st.toggle("Afficher seulement articles actifs", value=True)
    with copt2:
        st.caption("Actif = stock>0 OU seuil>0 OU sous seuil. Désactive pour voir tout l’inventaire.")

    parts = inv.list_parts_as_dicts(active_only=active_only) or []
    parts = sorted(parts, key=lambda x: str(x.get("code", "")))

    q = st.text_input("Recherche (code/nom)", value="").strip().lower()
    if q:
        parts = [p for p in parts if q in f"{p.get('code','')} {p.get('nom','')}".lower()]

    low = _low_stock_list()
    if low:
        st.warning(f"⚠️ {len(low)} article(s) sous le seuil.")
    else:
        st.success("✅ RAS — aucun article sous seuil.")

    # ✅ Envoi auto (ne change pas)
    ok_auto, msg_auto = _send_low_stock_alert(low, force=False)
    if ok_auto:
        st.toast("🔔 Alerte stock envoyée (auto).")

    if not parts:
        st.info("Aucun article trouvé. Ajoute un article ci-dessus.")
    else:
        for p in parts:
            code = str(p.get("code") or "").strip()
            if not code:
                continue
            nom = str(p.get("nom") or "")
            qte = _as_int(p.get("quantite_dispo"), 0)
            seuil = _as_int(p.get("seuil_min"), 0)

            with st.container(border=True):
                c1, c2, c3 = st.columns([2.2, 1.2, 3.0])
                with c1:
                    st.markdown(f"**{code}** — {nom}")
                    st.caption(f"Seuil: {seuil}")
                with c2:
                    st.metric("Qté", qte)
                with c3:
                    a, b = st.columns(2)
                    with a:
                        qty_add = st.number_input("Entrée", min_value=0, value=0, step=1, key=f"add_{code}")
                        if st.button("➕ Valider", key=f"btn_add_{code}", use_container_width=True):
                            if qty_add <= 0:
                                st.info("Saisis une quantité > 0.")
                            else:
                                ok, msg = inv.move_in(code, int(qty_add), reason="Achat / Entrée stock", ref="IN_UI", user="app")
                                if ok:
                                    st.success("Entrée enregistrée ✅")
                                    st.rerun()
                                else:
                                    st.error(msg)
                    with b:
                        qty_out = st.number_input("Sortie", min_value=0, value=0, step=1, key=f"out_{code}")
                        if st.button("➖ Valider", key=f"btn_out_{code}", use_container_width=True):
                            if qty_out <= 0:
                                st.info("Saisis une quantité > 0.")
                            else:
                                ok, msg = inv.move_out(code, int(qty_out), reason="Consommation", ref="OUT_UI", user="app")
                                if ok:
                                    st.success("Sortie enregistrée ✅")
                                    st.rerun()
                                else:
                                    st.error(msg)

    st.divider()
    st.subheader("🔔 Articles sous seuil")
    if low:
        st.dataframe(pd.DataFrame(low), use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun article sous seuil.")

    # ✅ Bouton manuel en bas (nouveau)
    st.divider()
    st.subheader("📨 Envoi manuel des alertes (sans casser l'auto)")

    cman1, cman2 = st.columns([1.2, 2.0])
    with cman1:
        force_send = st.toggle("Forcer l’envoi (ignorer cooldown 10 min)", value=False)
    with cman2:
        st.caption("Si tu viens d’envoyer en auto, Streamlit met un cooldown. Ici tu peux forcer si nécessaire.")

    if st.button("📨 Envoyer alertes stock maintenant", use_container_width=True):
        ok, msg = _send_low_stock_alert(low, force=force_send)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

with tab_hist:
    st.subheader("Historique récent des mouvements")
    mv = load_movements(limit=800)

    dmv = pd.DataFrame(mv) if mv else pd.DataFrame(columns=["ts", "type", "code", "qty", "reason", "task_id", "ref", "user"])
    if not dmv.empty:
        dmv["date"] = pd.to_datetime(dmv["ts"], unit="s", errors="coerce")
        dmv = dmv.sort_values("ts", ascending=False)
        show_cols = [c for c in ["date", "type", "code", "qty", "reason", "task_id", "ref", "user"] if c in dmv.columns]
        st.dataframe(dmv[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Aucun mouvement enregistré.")
