# pages/5_Maintenance.py
from __future__ import annotations
from pathlib import Path
import time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# Services
from core.maintenance import services as pm
from core.inventory import services as inv
from core.reliability.unify import compute_bundle

# Recos & PDF / Alertes
from core.inventory.recommendations import build_pm_kit_for_equipment
from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.notify.alerts_plus import notify_pm_with_kits

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None

st.set_page_config(page_title="Maintenance — Simple", page_icon="🛠️", layout="wide")
st.title("🛠️ Maintenance")

DATA_CSV = Path("data/failures_saved.csv")


# ========= Helpers =========

def _read_csv_flex(src: Path) -> pd.DataFrame:
    def _try(**kw):
        try:
            return pd.read_csv(src, **kw)
        except Exception:
            return None

    if not src.exists():
        return pd.DataFrame()
    df = _try()
    if df is None:
        df = _try(engine="python", on_bad_lines="skip", sep=None)
    if df is None:
        df = _try(sep=";", engine="python", on_bad_lines="skip")
    if df is None:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ttf_df() -> pd.DataFrame:
    if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
        df = st.session_state["failures_df"].copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return _read_csv_flex(DATA_CSV)


def _build_kits_by_eq(due_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Construit les kits recommandés par équipement à partir des tâches dues.
    Utilise core.inventory.recommendations.build_pm_kit_for_equipment si dispo.
    """
    kits: Dict[str, List[Dict[str, Any]]] = {}
    if not due_list:
        return kits

    eqs = sorted(
        {str(d.get("equipment_code")) for d in due_list if d.get("equipment_code")}
    )
    for eq in eqs:
        try:
            kit = build_pm_kit_for_equipment(eq) or []
        except Exception:
            kit = []
        kits[eq] = kit
    return kits


def _build_parts_request(kits_by_eq: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Aggrège toutes les pièces des kits (par code) et ajoute les infos de stock actuelles.
    """
    if not kits_by_eq:
        return []

    # Stock actuel
    parts = inv.list_parts_as_dicts() or []
    stock_map = {str(p.get("code")): p for p in parts if p.get("code")}

    agg: Dict[str, Dict[str, Any]] = {}

    for eq, kit in (kits_by_eq or {}).items():
        for item in kit or []:
            code = str(item.get("code") or item.get("part_code") or "").strip()
            if not code:
                continue
            q_req = float(item.get("qty") or item.get("quantite") or 0)
            if q_req <= 0:
                continue

            if code not in agg:
                base = stock_map.get(code, {}) or {}
                agg[code] = {
                    "code": code,
                    "nom": base.get("nom") or item.get("nom") or item.get("name") or "",
                    "famille": base.get("famille", ""),
                    "localisation": base.get("localisation", ""),
                    "fournisseur": base.get("fournisseur", ""),
                    "prix_unitaire": base.get("prix_unitaire", 0.0),
                    "seuil_min": float(base.get("seuil_min") or 0),
                    "quantite_dispo": float(base.get("quantite_dispo") or 0),
                    "qte_requise": 0.0,
                }
            agg[code]["qte_requise"] += q_req

    # Calcule les champs dérivés
    rows: List[Dict[str, Any]] = []
    for code, row in agg.items():
        dispo = float(row.get("quantite_dispo") or 0)
        req = float(row.get("qte_requise") or 0)
        q_prelevable = min(dispo, req)
        q_manquante = max(0.0, req - dispo)
        stock_restant = max(0.0, dispo - q_prelevable)
        seuil_min = float(row.get("seuil_min") or 0)
        sous_seuil_apres = stock_restant <= seuil_min if seuil_min > 0 else False

        row["qte_prelevable"] = q_prelevable
        row["qte_manquante"] = q_manquante
        row["stock_restant"] = stock_restant
        row["sous_seuil_apres"] = sous_seuil_apres
        rows.append(row)

    return rows


def _apply_parts_request(parts_req: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applique la demande de pièces :
      - Déduit du stock (move_out) la quantité prélevable.
      - Construit un résumé de consommation pour le PDF.
    Retourne consumption_summary (liste de dict).
    """
    summary: List[Dict[str, Any]] = []
    if not parts_req:
        return summary

    for row in parts_req:
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        nom = row.get("nom", "")
        req = float(row.get("qte_requise") or 0)
        dispo = float(row.get("quantite_dispo") or 0)
        seuil_min = float(row.get("seuil_min") or 0)

        to_take = min(req, dispo)
        missing = max(0.0, req - to_take)
        before = dispo
        after = max(0.0, dispo - to_take)

        ok = True
        msg = ""
        if to_take > 0:
            ok, msg = inv.move_out(
                code,
                to_take,
                reason="Plan de maintenance (prélevé)",
                ref="PM_PLAN",
                user="system",
            )

        summary.append({
            "code": code,
            "nom": nom,
            "qte_requise": req,
            "qte_prelevee": to_take,
            "qte_manquante": missing,
            "stock_avant": before,
            "stock_apres": after,
            "seuil_min": seuil_min,
            "sous_seuil_apres": (after <= seuil_min) if seuil_min > 0 else False,
            "ok_move": ok,
            "msg_move": msg,
        })

    return summary


st.caption("Flux : **Scanner événements** ⟶ **Voir tâches dues** ⟶ **Demande de pièces & Validation** ⟶ **Plan PDF & Envoi**.")


# ========= 0) Événements → Propositions de tâches =========

with st.container(border=True):
    st.subheader("1) Événements temps réel → tâches")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔎 Scanner les événements non traités", use_container_width=True):
            try:
                from core.maintenance.ingest_events import ingest_events_to_tasks

                res = ingest_events_to_tasks("data/realtime_events.csv")
                st.session_state["ingested_tasks"] = res
                st.success(f"{res.get('marked_processed', 0)} événement(s) ingéré(s).")
            except Exception as e:
                st.error(f"Ingestion: {e}")
    with c2:
        if st.button("💾 Exporter les propositions (CSV)", use_container_width=True):
            res = st.session_state.get("ingested_tasks", {}) or {}
            rows = res.get("tasks", []) or []
            if not rows:
                st.info("Aucune tâche proposée.")
            else:
                out = Path("data/auto_tasks_pending.csv")
                pd.DataFrame(rows).to_csv(out, index=False)
                st.success(f"Exporté → {out}")

    res = st.session_state.get("ingested_tasks", {}) or {}
    tasks = res.get("tasks", []) or []
    if tasks:
        st.dataframe(pd.DataFrame(tasks), use_container_width=True, hide_index=True)
    else:
        st.caption("Aucune proposition (scanne les événements).")


# ========= 1) Tâches dues (période courte) =========

with st.container(border=True):
    st.subheader("2) Tâches dues (prochaines semaines)")
    within = st.slider("Fenêtre (jours)", 7, 60, 14, 1)
    try:
        due = pm.due_within(days=int(within)) or []
    except Exception as e:
        st.error(f"Lecture tâches dues : {e}")
        due = []

    if not due:
        st.info("Aucune tâche due dans l’intervalle.")
    else:
        df_due = pd.DataFrame(due)
        keep = [
            c
            for c in [
                "id",
                "equipment_code",
                "title",
                "priority",
                "next_due_date",
                "days_left",
                "status",
            ]
            if c in df_due.columns
        ]
        st.dataframe(
            df_due[keep] if not df_due.empty else df_due,
            use_container_width=True,
            hide_index=True,
        )

# Calculs de base pour la suite (kits + fiabilité)
bundle = compute_bundle(_ttf_df())
dfm = bundle.metrics_df.copy() if hasattr(bundle, "metrics_df") else pd.DataFrame()
kits_by_eq = _build_kits_by_eq(due)
parts_request = _build_parts_request(kits_by_eq)
st.session_state["parts_request"] = parts_request


# ========= 2) Demande de pièces & réservation (avant PDF) =========

with st.container(border=True):
    st.subheader("3) Demande de pièces (kits de maintenance)")

    if not parts_request:
        st.info(
            "Aucune pièce recommandée n'a été identifiée pour les tâches dues "
            "(kits vides ou non configurés)."
        )
    else:
        df_parts = pd.DataFrame(parts_request)
        cols_aff = [
            "code",
            "nom",
            "quantite_dispo",
            "qte_requise",
            "qte_prelevable",
            "qte_manquante",
            "stock_restant",
            "seuil_min",
            "sous_seuil_apres",
        ]
        cols_aff = [c for c in cols_aff if c in df_parts.columns]
        st.dataframe(
            df_parts[cols_aff],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Vérifie bien la demande : les quantités **prélevables** sont celles qui seront retirées du stock "
            "après validation ci-dessous."
        )


# ========= 3) Plan PDF & Envoi (avec déduction stock) =========

with st.container(border=True):
    st.subheader("4) Plan de maintenance — PDF, déduction stock & envoi")

    include_kits = st.checkbox(
        "Inclure les kits (résumé) dans le PDF",
        value=True,
        help="Le détail complet des kits reste géré dans le module Inventaire.",
    )

    colA, colB = st.columns([1, 1])

    # ---- Bouton simple : génération PDF sans déduction ----
    with colA:
        if st.button("📄 Générer le PDF (sans déduire le stock)", use_container_width=True):
            try:
                path_pdf = export_pm_plan_with_kits_pdf(
                    tasks_due=due or [],
                    kits_by_eq=kits_by_eq,
                    metrics_table=dfm.to_dict("records"),
                    out_dir="reports",
                    title="Plan de maintenance — Procédure, Tâches, Matériels",
                    procedure_docx=None,
                    include_kits=bool(include_kits),
                    tools_checklist=None,
                    consumption_summary=None,  # pas de déduction, juste analyse
                )
                st.success(f"PDF généré : {path_pdf}")
            except Exception as e:
                st.error(f"PDF : {e}")

    # ---- Bouton principal : validation demande + déduction + PDF + envois ----
    with colB:
        if st.button(
            "✅ Valider la demande, déduire le stock, générer & envoyer le plan",
            type="primary",
            use_container_width=True,
        ):
            try:
                # 1) Appliquer la consommation de stock
                parts_req = st.session_state.get("parts_request") or []
                consumption_summary = _apply_parts_request(parts_req)

                # 2) Générer le PDF avec la section 'Consommation de stock'
                path_pdf = export_pm_plan_with_kits_pdf(
                    tasks_due=due or [],
                    kits_by_eq=kits_by_eq,
                    metrics_table=dfm.to_dict("records"),
                    out_dir="reports",
                    title="Plan de maintenance — Procédure, Tâches, Matériels",
                    procedure_docx=None,
                    include_kits=bool(include_kits),
                    tools_checklist=None,
                    consumption_summary=consumption_summary,
                )

                # 3) Alertes stock (après mise à jour)
                low_after = inv.low_stock(threshold_factor=1.0) or []
                if low_after:
                    if notify_stock_alerts:
                        try:
                            notify_stock_alerts(low_after)
                            st.info("Alertes stock envoyées pour les articles sous seuil.")
                        except Exception as e_alert:
                            st.error(f"Envoi alertes stock : {e_alert}")
                    else:
                        out_low = Path("data/low_stock_alert_after_plan.csv")
                        pd.DataFrame(low_after).to_csv(out_low, index=False)
                        st.info(
                            f"Module d’alerte dédié indisponible — CSV généré : {out_low}"
                        )

                # 4) Notifications maintenance (plan + kits) via module dédié
                res = notify_pm_with_kits(due or [], kits_by_eq, dfm.to_dict("records")) or {}

                st.success("Plan de maintenance généré, stock mis à jour et notifications envoyées.")
                if path_pdf:
                    st.caption(f"PDF (Plan) : {path_pdf}")
                if res:
                    from pathlib import Path as _P

                    name_pdf = _P(res.get("pdf", "")).name or "—"
                    name_csv = _P(res.get("csv", "")).name or "—"
                    st.caption(f"Attachements (module notify_pm_with_kits) : PDF={name_pdf} • CSV={name_csv}")
            except Exception as e:
                st.error(f"Erreur lors de la validation & envoi : {e}")


# ========= 4) Alertes “Seuil stock” (rappel simple) =========

with st.container(border=True):
    st.subheader("🔔 Alertes stock — seuils (rappel)")

    st.caption("Alerte quand `quantite_dispo ≤ seuil_min`. Paramétrage des destinataires dans le module d'alertes.")

    auto = st.toggle(
        "Activer alertes automatiques sur cette page",
        value=st.session_state.get("stock_alert_enabled", False),
        key="stock_alert_enabled",
    )

    low = inv.low_stock(threshold_factor=1.0) or []
    if not low:
        st.success("Aucune alerte : tous les stocks sont au-dessus du seuil.")
    else:
        df_low = pd.DataFrame(low)
        st.dataframe(df_low, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("📤 Envoyer alerte stock maintenant", use_container_width=True):
                try:
                    if notify_stock_alerts:
                        r = notify_stock_alerts(low) or {}
                        st.success("Alerte stock envoyée.")
                    else:
                        out = Path("data/low_stock_alert.csv")
                        pd.DataFrame(low).to_csv(out, index=False)
                        st.info(
                            f"Module d’alerte dédié indisponible — CSV généré : {out}"
                        )
                except Exception as e:
                    st.error(f"Alerte stock : {e}")
        with c2:
            if st.button("⬇️ Export CSV", use_container_width=True):
                out = Path("data/low_stock_alert.csv")
                pd.DataFrame(low).to_csv(out, index=False)
                st.success(f"Exporté → {out}")

    # Auto (anti-spam : 10 min)
    if auto and low:
        last = st.session_state.get("_last_stock_alert_ts", 0.0)
        if time.time() - last > 600:  # 10 minutes
            try:
                if notify_stock_alerts:
                    notify_stock_alerts(low)
                    st.session_state["_last_stock_alert_ts"] = time.time()
                    st.toast("Alerte stock auto envoyée.")
            except Exception:
                pass
