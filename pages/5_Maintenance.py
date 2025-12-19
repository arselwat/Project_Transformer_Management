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

from core.security.auth import require_login


# =========================
# Page config + Auth
# =========================
st.set_page_config(page_title="Maintenance — Simple", page_icon="🛠️", layout="wide")
require_login()
st.title("🛠️ Maintenance")

DATA_CSV = Path("data/failures_saved.csv")


# =========================
# Helpers
# =========================
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
    kits: Dict[str, List[Dict[str, Any]]] = {}
    if not due_list:
        return kits

    eqs = sorted({str(d.get("equipment_code")) for d in due_list if d.get("equipment_code")})
    for eq in eqs:
        try:
            kit = build_pm_kit_for_equipment(eq) or []
        except Exception:
            kit = []
        kits[eq] = kit
    return kits


def _build_parts_request(kits_by_eq: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not kits_by_eq:
        return []

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
                    "prix_unitaire": float(base.get("prix_unitaire") or 0.0),
                    "seuil_min": float(base.get("seuil_min") or 0),
                    "quantite_dispo": float(base.get("quantite_dispo") or 0),
                    "qte_requise": 0.0,
                }

            agg[code]["qte_requise"] += q_req

    rows: List[Dict[str, Any]] = []
    for _, row in agg.items():
        dispo = float(row.get("quantite_dispo") or 0)
        req = float(row.get("qte_requise") or 0)

        q_prelevable = min(dispo, req)
        q_manquante = max(0.0, req - dispo)
        stock_restant = max(0.0, dispo - q_prelevable)

        seuil_min = float(row.get("seuil_min") or 0)
        sous_seuil_apres = (stock_restant <= seuil_min) if seuil_min > 0 else False

        row["qte_prelevable"] = q_prelevable
        row["qte_manquante"] = q_manquante
        row["stock_restant"] = stock_restant
        row["sous_seuil_apres"] = sous_seuil_apres
        rows.append(row)

    return rows


def _apply_parts_request(parts_req: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _is_opt_task(t: Dict[str, Any]) -> bool:
    return "(optimisation)" in str(t.get("title") or "").lower()


def _planning_from_optimization(within_days: int = 60) -> pd.DataFrame:
    try:
        tasks = pm.list_tasks() or []
    except Exception:
        tasks = []

    opt_tasks = [t for t in tasks if _is_opt_task(t)]
    if not opt_tasks:
        return pd.DataFrame(columns=[
            "id", "equipment_code", "title", "periodicity_days",
            "next_due_date", "last_done_date", "status", "days_left"
        ])

    dfp = pd.DataFrame(opt_tasks)
    try:
        today = pd.Timestamp.today().normalize()
        dfp["next_due_date_dt"] = pd.to_datetime(dfp.get("next_due_date"), errors="coerce")
        dfp["days_left"] = (dfp["next_due_date_dt"].dt.normalize() - today).dt.days
    except Exception:
        dfp["days_left"] = None

    if "days_left" in dfp.columns:
        dfp = dfp[(dfp["days_left"].isna()) | (dfp["days_left"] <= int(within_days))]

    keep = [c for c in [
        "id", "equipment_code", "title", "periodicity_days",
        "next_due_date", "last_done_date", "status", "days_left"
    ] if c in dfp.columns]

    if "next_due_date" in dfp.columns:
        dfp = dfp.sort_values(["next_due_date"], na_position="last")

    return dfp[keep].reset_index(drop=True)


def _build_purchase_list_from_parts_request(parts_req: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["code", "nom", "qte_manquante", "seuil_min", "stock_actuel", "suggestion_achat", "raison"]
    if not parts_req:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in parts_req:
        code = str(r.get("code") or "").strip()
        if not code:
            continue

        nom = str(r.get("nom") or "")
        dispo = float(r.get("quantite_dispo") or 0)
        missing = float(r.get("qte_manquante") or 0)
        seuil = float(r.get("seuil_min") or 0)
        restant = float(r.get("stock_restant") or 0)

        achat = 0.0
        raisons = []

        if missing > 0:
            achat = max(achat, missing)
            raisons.append("manquant pour exécuter le plan")

        if seuil > 0 and restant < seuil:
            topup = max(0.0, seuil - restant)
            achat = max(achat, topup)
            raisons.append("remise à niveau du seuil")

        if achat > 0:
            rows.append({
                "code": code,
                "nom": nom,
                "qte_manquante": round(missing, 2),
                "seuil_min": round(seuil, 2),
                "stock_actuel": round(dispo, 2),
                "suggestion_achat": round(achat, 2),
                "raison": " + ".join(raisons),
            })

    dfb = pd.DataFrame(rows)
    if dfb.empty:
        return pd.DataFrame(columns=cols)
    return dfb.sort_values(["suggestion_achat"], ascending=False).reset_index(drop=True)


# =========================
# UI
# =========================
st.caption("Flux : **Scanner événements** ⟶ **Voir tâches dues** ⟶ **Demande de pièces & Validation** ⟶ **Plan PDF & Envoi**.")

# -------- 0) Ingestion events --------
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

# -------- 1) Due tasks --------
with st.container(border=True):
    st.subheader("2) Tâches dues (prochaines semaines)")
    within = st.slider("Fenêtre (jours)", 7, 60, 14, 1, key="due_window_days")

    try:
        due = pm.due_within(days=int(within)) or []
    except Exception as e:
        st.error(f"Lecture tâches dues : {e}")
        due = []

    if not due:
        st.info("Aucune tâche due dans l’intervalle.")
    else:
        df_due = pd.DataFrame(due)
        keep = [c for c in [
            "id", "equipment_code", "title", "priority", "next_due_date", "days_left", "status"
        ] if c in df_due.columns]
        st.dataframe(df_due[keep] if not df_due.empty else df_due, use_container_width=True, hide_index=True)

# -------- Planning optimisation --------
with st.container(border=True):
    st.subheader("📌 Planning issu de l’optimisation")
    win_opt = st.slider("Fenêtre (jours) — planning optimisation", 7, 180, 60, 1, key="opt_plan_win")
    df_opt_plan = _planning_from_optimization(within_days=int(win_opt))

    if df_opt_plan.empty:
        st.info("Aucune tâche provenant de l’optimisation (titre contenant '(optimisation)').")
    else:
        st.dataframe(df_opt_plan, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export CSV (planning optimisation)",
            data=df_opt_plan.to_csv(index=False).encode("utf-8"),
            file_name="planning_optimisation_pm_task.csv",
            mime="text/csv",
            use_container_width=True,
        )

# -------- Bundle metrics --------
bundle = compute_bundle(_ttf_df())
dfm = bundle.metrics_df.copy() if hasattr(bundle, "metrics_df") else pd.DataFrame()

# -------- Kits + request --------
kits_by_eq = _build_kits_by_eq(due)
parts_request = _build_parts_request(kits_by_eq)
st.session_state["parts_request"] = parts_request

# -------- Purchase list --------
with st.container(border=True):
    st.subheader("🧾 Comparaison Stock vs Kits recommandés (liste d’achat)")

    if not parts_request:
        st.info("Aucune demande de pièces (kits vides ou aucune tâche due).")
    else:
        df_parts_req = pd.DataFrame(parts_request)
        show_cols = [
            "code", "nom",
            "quantite_dispo", "qte_requise",
            "qte_prelevable", "qte_manquante",
            "stock_restant", "seuil_min", "sous_seuil_apres",
        ]
        show_cols = [c for c in show_cols if c in df_parts_req.columns]

        st.markdown("#### 1) Détail besoin vs stock")
        st.dataframe(df_parts_req[show_cols], use_container_width=True, hide_index=True)

        st.markdown("#### 2) Liste d’achat proposée (manquants + remise à niveau du seuil)")
        df_buy = _build_purchase_list_from_parts_request(parts_request)

        if df_buy.empty:
            st.success("✅ Aucun achat nécessaire : tout est disponible et au-dessus des seuils.")
        else:
            st.dataframe(df_buy, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Télécharger la liste d’achat (CSV)",
                data=df_buy.to_csv(index=False).encode("utf-8"),
                file_name="liste_achat_pieces.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("💾 Enregistrer la liste d’achat dans data/purchase_list.csv", use_container_width=True):
                outp = Path("data") / "purchase_list.csv"
                outp.parent.mkdir(parents=True, exist_ok=True)
                df_buy.to_csv(outp, index=False)
                st.success(f"Enregistré → {outp}")

# -------- Parts request table --------
with st.container(border=True):
    st.subheader("3) Demande de pièces (kits de maintenance)")

    if not parts_request:
        st.info("Aucune pièce recommandée n'a été identifiée pour les tâches dues (kits vides ou non configurés).")
    else:
        df_parts = pd.DataFrame(parts_request)
        cols_aff = [
            "code", "nom", "quantite_dispo", "qte_requise",
            "qte_prelevable", "qte_manquante",
            "stock_restant", "seuil_min", "sous_seuil_apres",
        ]
        cols_aff = [c for c in cols_aff if c in df_parts.columns]
        st.dataframe(df_parts[cols_aff], use_container_width=True, hide_index=True)

# -------- PDF Plan + Download + Send --------
with st.container(border=True):
    st.subheader("4) Plan de maintenance — PDF, téléchargement, déduction stock & envoi")

    include_kits = st.checkbox(
        "Inclure les kits (résumé) dans le PDF",
        value=True,
        help="Le détail complet des kits reste géré dans le module Inventaire.",
    )

    colA, colB = st.columns([1, 1])

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
                    consumption_summary=None,
                )
                st.session_state["pm_plan_pdf_path"] = path_pdf
                st.success(f"PDF généré : {path_pdf}")
            except Exception as e:
                st.error(f"PDF : {e}")

    with colB:
        if st.button(
            "✅ Valider la demande, déduire le stock, générer & envoyer le plan",
            type="primary",
            use_container_width=True,
        ):
            try:
                parts_req = st.session_state.get("parts_request") or []
                consumption_summary = _apply_parts_request(parts_req)

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
                st.session_state["pm_plan_pdf_path"] = path_pdf

                # Alertes stock (après MAJ)
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
                        st.info(f"Module d’alerte indisponible — CSV généré : {out_low}")

                # Notifications maintenance
                res = notify_pm_with_kits(due or [], kits_by_eq, dfm.to_dict("records")) or {}

                st.success("Plan de maintenance généré, stock mis à jour et notifications envoyées.")
                if path_pdf:
                    st.caption(f"PDF (Plan) : {path_pdf}")
                if res:
                    name_pdf = Path(res.get("pdf", "")).name or "—"
                    name_csv = Path(res.get("csv", "")).name or "—"
                    st.caption(f"Attachements notify_pm_with_kits : PDF={name_pdf} • CSV={name_csv}")

            except Exception as e:
                st.error(f"Erreur lors de la validation & envoi : {e}")

    # ✅ Téléchargement du plan de maintenance (si dispo)
    pdf_path = st.session_state.get("pm_plan_pdf_path")
    if pdf_path and Path(str(pdf_path)).exists():
        st.markdown("### 📥 Télécharger le plan de maintenance (PDF)")
        with open(str(pdf_path), "rb") as f:
            st.download_button(
                "📥 Télécharger le PDF du plan",
                data=f,
                file_name=Path(str(pdf_path)).name,
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.caption("Génère d’abord un PDF pour activer le bouton de téléchargement.")


# -------- Stock alerts reminder --------
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
                        notify_stock_alerts(low)
                        st.success("Alerte stock envoyée.")
                    else:
                        out = Path("data/low_stock_alert.csv")
                        pd.DataFrame(low).to_csv(out, index=False)
                        st.info(f"Module d’alerte indisponible — CSV généré : {out}")
                except Exception as e:
                    st.error(f"Alerte stock : {e}")

        with c2:
            if st.button("⬇️ Export CSV", use_container_width=True):
                out = Path("data/low_stock_alert.csv")
                pd.DataFrame(low).to_csv(out, index=False)
                st.success(f"Exporté → {out}")

    if auto and low:
        last = st.session_state.get("_last_stock_alert_ts", 0.0)
        if time.time() - last > 600:
            try:
                if notify_stock_alerts:
                    notify_stock_alerts(low)
                    st.session_state["_last_stock_alert_ts"] = time.time()
                    st.toast("Alerte stock auto envoyée.")
            except Exception:
                pass
