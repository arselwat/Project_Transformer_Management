# pages/5_Maintenance.py
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# --- Auth ---
from core.security.auth import require_login

# --- Services ---
from core.maintenance import services as pm
from core.inventory import services as inv
from core.reliability.unify import compute_bundle

# --- Kits / PDF / Alerts ---
from core.inventory.recommendations import build_pm_kit_for_equipment
from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.notify.alerts_plus import notify_pm_with_kits

try:
    from core.notify.alerts_plus import notify_stock_alerts
except Exception:
    notify_stock_alerts = None

# Optional: ingest events -> tasks proposals
try:
    from core.maintenance.ingest_events import ingest_events_to_tasks
except Exception:
    ingest_events_to_tasks = None


# =========================================================
# Config + Auth
# =========================================================
st.set_page_config(page_title="Maintenance — Simple", page_icon="🛠️", layout="wide")
require_login()

st.title("🛠️ Maintenance")
st.caption(
    "Objectif : **Planning (optimisation)** → **Tâches dues (pm_task)** → **Kits recommandés** → "
    "**Comparaison Stock vs Kits** → **Liste d’achat** → **PDF + notifications**."
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_CSV = BASE_DIR / "data" / "failures_saved.csv"


# =========================================================
# Helpers
# =========================================================
def _read_csv_flex(path: Path) -> pd.DataFrame:
    def _try(**kw):
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            return None

    if not path.exists():
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


def _safe_int(x, default=0) -> int:
    try:
        v = int(float(x))
        return v
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


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
    """
    Agrège tous les articles demandés par les kits + jointure stock actuel.
    """
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
            q_req = _safe_float(item.get("qty") or item.get("quantite") or 0, 0.0)
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
                    "prix_unitaire": _safe_float(base.get("prix_unitaire", 0.0), 0.0),
                    "seuil_min": _safe_float(base.get("seuil_min", 0), 0.0),
                    "quantite_dispo": _safe_float(base.get("quantite_dispo", 0), 0.0),
                    "qte_requise": 0.0,
                    "equipements": set(),
                }

            agg[code]["qte_requise"] += q_req
            agg[code]["equipements"].add(eq)

    rows: List[Dict[str, Any]] = []
    for code, row in agg.items():
        dispo = _safe_float(row.get("quantite_dispo"), 0.0)
        req = _safe_float(row.get("qte_requise"), 0.0)

        q_prelevable = min(dispo, req)
        q_manquante = max(0.0, req - dispo)
        stock_restant = max(0.0, dispo - q_prelevable)
        seuil_min = _safe_float(row.get("seuil_min"), 0.0)
        sous_seuil_apres = (stock_restant <= seuil_min) if seuil_min > 0 else False

        row["qte_prelevable"] = q_prelevable
        row["qte_manquante"] = q_manquante
        row["stock_restant"] = stock_restant
        row["sous_seuil_apres"] = sous_seuil_apres
        row["equipements"] = ", ".join(sorted(list(row.get("equipements") or [])))
        rows.append(row)

    # tri: manquants d'abord, puis sous seuil
    rows = sorted(rows, key=lambda r: (-float(r.get("qte_manquante", 0.0)), -int(bool(r.get("sous_seuil_apres"))), str(r.get("code"))))
    return rows


def _build_purchase_list(parts_req: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Liste d'achat = tout ce qui manque (qte_manquante>0) + éventuellement ce qui passe sous seuil après prélèvement.
    """
    if not parts_req:
        return pd.DataFrame()

    rows = []
    for r in parts_req:
        missing = _safe_float(r.get("qte_manquante"), 0.0)
        if missing <= 0:
            continue
        rows.append({
            "code": r.get("code", ""),
            "nom": r.get("nom", ""),
            "equipements": r.get("equipements", ""),
            "qte_requise": _safe_float(r.get("qte_requise"), 0.0),
            "qte_dispo": _safe_float(r.get("quantite_dispo"), 0.0),
            "qte_a_acheter": missing,
            "seuil_min": _safe_float(r.get("seuil_min"), 0.0),
            "prix_unitaire": _safe_float(r.get("prix_unitaire"), 0.0),
            "cout_estime": missing * _safe_float(r.get("prix_unitaire"), 0.0),
            "fournisseur": r.get("fournisseur", ""),
            "localisation": r.get("localisation", ""),
        })

    df_buy = pd.DataFrame(rows)
    if df_buy.empty:
        return df_buy
    df_buy["cout_estime"] = df_buy["cout_estime"].round(2)
    return df_buy.sort_values(["cout_estime", "qte_a_acheter"], ascending=[False, False]).reset_index(drop=True)


def _apply_parts_request(parts_req: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Déduit le stock (move_out) sur la quantité prélevable.
    Retourne un résumé à inclure dans PDF.
    """
    summary: List[Dict[str, Any]] = []
    if not parts_req:
        return summary

    for row in parts_req:
        code = str(row.get("code") or "").strip()
        if not code:
            continue

        nom = row.get("nom", "")
        req = _safe_float(row.get("qte_requise"), 0.0)
        dispo = _safe_float(row.get("quantite_dispo"), 0.0)
        seuil_min = _safe_float(row.get("seuil_min"), 0.0)

        to_take = min(req, dispo)
        missing = max(0.0, req - to_take)
        before = dispo
        after = max(0.0, dispo - to_take)

        ok, msg = True, ""
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


def _make_or_update_optimized_tasks(
    df_opt: pd.DataFrame,
    task_title: str = "Maintenance préventive (issu de l’optimisation)",
    min_days: int = 7,
) -> Dict[str, Any]:
    """
    Crée/MAJ pm_task à partir d'un tableau d'optimisation.
    Attendu: equipment_code + une colonne intervalle en heures :
      - T_recommended_h (priorité)
      - sinon T_R_h
      - sinon T_cost_h
    """
    if df_opt is None or df_opt.empty:
        return {"ok": False, "msg": "Fichier optimisation vide."}

    # normaliser
    df = df_opt.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "equipment_code" not in df.columns:
        return {"ok": False, "msg": "Colonne equipment_code manquante."}

    # choisir la meilleure colonne d'intervalle
    interval_col = None
    for c in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
        if c in df.columns:
            interval_col = c
            break
    if interval_col is None:
        return {"ok": False, "msg": "Aucune colonne d’intervalle trouvée (T_recommended_h / T_R_h / T_cost_h)."}

    created = 0
    updated = 0
    skipped = 0

    today = date.today()

    # on va comparer avec l'existant pour savoir si update/insert
    existing = pm.list_tasks() or []
    key_map = {(str(t.get("equipment_code")), str(t.get("title"))): t for t in existing}

    for _, r in df.iterrows():
        eq = str(r.get("equipment_code") or "").strip()
        if not eq:
            skipped += 1
            continue

        interval_h = _safe_float(r.get(interval_col), 0.0)
        if interval_h <= 0:
            skipped += 1
            continue

        # conversion h -> jours (arrondi) et garde-fous
        per_days = max(min_days, int(round(interval_h / 24.0)))
        next_due = (today + timedelta(days=per_days)).isoformat()

        k = (eq, task_title)
        old = key_map.get(k)

        if old and old.get("id"):
            pm.upsert_task({
                "id": int(old["id"]),
                "equipment_code": eq,
                "title": task_title,
                "periodicity_days": per_days,
                "next_due_date": next_due,
                "last_done_date": old.get("last_done_date"),
                "status": old.get("status", "ACTIVE"),
            })
            updated += 1
        else:
            pm.upsert_task({
                "equipment_code": eq,
                "title": task_title,
                "periodicity_days": per_days,
                "next_due_date": next_due,
                "last_done_date": None,
                "status": "ACTIVE",
            })
            created += 1

    return {"ok": True, "created": created, "updated": updated, "skipped": skipped, "interval_col": interval_col}


# =========================================================
# UI — Tabs
# =========================================================
tab_opt, tab_due, tab_plan, tab_alert = st.tabs([
    "🧠 Planning issu de l’optimisation",
    "📅 Tâches dues (pm_task)",
    "🧾 Plan + Kits + Stock vs Kits",
    "🔔 Alertes stock",
])


# =========================================================
# TAB 1 — Planning issu de l’optimisation
# =========================================================
with tab_opt:
    st.subheader("🧠 Planning issu de l’optimisation")

    st.markdown(
        "- Cette section **crée / met à jour** les tâches `pm_task` à partir d’un fichier d’optimisation.\n"
        "- Le fichier d’optimisation peut venir de ta page **Optimisation** (CSV téléchargé), ou d’un export.\n"
        "- Conversion : **intervalle (heures)** → **périodicité (jours)** ≈ `round(h/24)`.\n"
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        up_opt = st.file_uploader(
            "Uploader le CSV d’optimisation (ex: optimisation_intervalles.csv)",
            type=["csv"],
            help="Doit contenir equipment_code et une colonne d’intervalle (T_recommended_h, T_R_h, T_cost_h...).",
        )
    with c2:
        min_days = st.number_input("Périodicité minimale (jours)", min_value=1, value=7, step=1)

    df_opt = pd.DataFrame()
    if up_opt is not None:
        try:
            df_opt = pd.read_csv(up_opt)
            df_opt.columns = [str(c).strip() for c in df_opt.columns]
            st.success(f"CSV chargé : {len(df_opt)} ligne(s).")
            st.dataframe(df_opt.head(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Lecture CSV optimisation : {e}")
            df_opt = pd.DataFrame()

    if st.button("✅ Créer / Mettre à jour le planning (pm_task) depuis optimisation", type="primary"):
        if df_opt.empty:
            st.error("Charge d’abord un CSV d’optimisation.")
        else:
            try:
                res = _make_or_update_optimized_tasks(df_opt=df_opt, min_days=int(min_days))
                if res.get("ok"):
                    st.success(
                        f"Planning MAJ ✅ | créées={res.get('created')} | mises à jour={res.get('updated')} | "
                        f"ignorées={res.get('skipped')} | colonne utilisée={res.get('interval_col')}"
                    )
                else:
                    st.error(res.get("msg", "Erreur inconnue"))
            except Exception as e:
                st.error(f"MAJ planning : {e}")

    st.divider()

    # (Optionnel) Ingestion événements -> propositions
    st.subheader("🛰️ Événements temps réel → propositions (optionnel)")
    st.caption(
        "Ici ce sont des **propositions**. Elles ne remplacent pas le planning `pm_task` tant que tu ne les intègres pas."
    )

    if ingest_events_to_tasks is None:
        st.info("Module ingest_events_to_tasks non disponible.")
    else:
        cA, cB = st.columns([1, 1])
        with cA:
            if st.button("🔎 Scanner les événements non traités", use_container_width=True):
                try:
                    res = ingest_events_to_tasks(str(BASE_DIR / "data" / "realtime_events.csv"))
                    st.session_state["ingested_tasks"] = res
                    st.success(f"{res.get('marked_processed', 0)} événement(s) ingéré(s).")
                except Exception as e:
                    st.error(f"Ingestion: {e}")
        with cB:
            if st.button("💾 Exporter les propositions (CSV)", use_container_width=True):
                res = st.session_state.get("ingested_tasks", {}) or {}
                rows = res.get("tasks", []) or []
                if not rows:
                    st.info("Aucune tâche proposée.")
                else:
                    out = BASE_DIR / "data" / "auto_tasks_pending.csv"
                    pd.DataFrame(rows).to_csv(out, index=False)
                    st.success(f"Exporté → {out}")

        res = st.session_state.get("ingested_tasks", {}) or {}
        tasks = res.get("tasks", []) or []
        if tasks:
            st.dataframe(pd.DataFrame(tasks), use_container_width=True, hide_index=True)
        else:
            st.caption("Aucune proposition (scanne les événements).")


# =========================================================
# TAB 2 — Tâches dues (pm_task)
# =========================================================
with tab_due:
    st.subheader("📅 Tâches dues (pm_task)")

    within = st.slider("Fenêtre (jours)", 7, 90, 14, 1)
    try:
        due = pm.due_within(days=int(within)) or []
    except Exception as e:
        st.error(f"Lecture tâches dues : {e}")
        due = []

    if not due:
        st.info("Aucune tâche due dans l’intervalle. (As-tu généré le planning depuis optimisation ?)")
    else:
        df_due = pd.DataFrame(due)
        df_due = df_due.copy()
        # Affiche seulement ce qui existe réellement dans ta DB
        pref_cols = ["id", "equipment_code", "title", "periodicity_days", "next_due_date", "days_left", "status"]
        keep = [c for c in pref_cols if c in df_due.columns]
        st.dataframe(df_due[keep] if keep else df_due, use_container_width=True, hide_index=True)

        st.caption("Astuce : quand une tâche est terminée, utilise `pm.mark_done(task_id)` depuis l’UI dédiée ou un bouton futur.")


# =========================================================
# TAB 3 — Plan + Kits + Comparaison Stock vs Kits + PDF + download
# =========================================================
with tab_plan:
    st.subheader("🧾 Plan de maintenance — Kits & comparaison Stock vs Kits")

    # reload due here too (avoid dependency between tabs)
    within2 = st.slider("Fenêtre (jours) pour le plan", 7, 90, 14, 1, key="within_plan")
    try:
        due2 = pm.due_within(days=int(within2)) or []
    except Exception as e:
        st.error(f"Lecture tâches dues : {e}")
        due2 = []

    if not due2:
        st.info("Aucune tâche due dans la fenêtre : impossible de construire un plan.")
        st.stop()

    # --- Reliability metrics (for PDF table) ---
    bundle = compute_bundle(_ttf_df())
    dfm = bundle.metrics_df.copy() if hasattr(bundle, "metrics_df") else pd.DataFrame()

    # --- kits + parts request ---
    kits_by_eq = _build_kits_by_eq(due2)
    parts_request = _build_parts_request(kits_by_eq)
    st.session_state["parts_request"] = parts_request

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ✅ Tâches incluses dans le plan")
        st.dataframe(pd.DataFrame(due2), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### 🧰 Kits par équipement (résumé)")
        # résumé: nb items par eq
        kit_rows = [{"equipment_code": eq, "nb_items": len(kit or [])} for eq, kit in (kits_by_eq or {}).items()]
        st.dataframe(pd.DataFrame(kit_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📦 Comparaison Stock vs Kits recommandés")

    if not parts_request:
        st.warning("Aucune pièce recommandée (kits vides ou non configurés).")
    else:
        df_parts = pd.DataFrame(parts_request)
        cols_aff = [
            "code",
            "nom",
            "equipements",
            "quantite_dispo",
            "qte_requise",
            "qte_prelevable",
            "qte_manquante",
            "stock_restant",
            "seuil_min",
            "sous_seuil_apres",
        ]
        cols_aff = [c for c in cols_aff if c in df_parts.columns]
        st.dataframe(df_parts[cols_aff], use_container_width=True, hide_index=True)

        st.caption(
            "Interprétation :\n"
            "- **qte_prelevable** = ce que tu peux réellement retirer du stock\n"
            "- **qte_manquante** = ce qu’il faut acheter\n"
        )

        df_buy = _build_purchase_list(parts_request)
        st.markdown("### 🛒 Liste d’achat proposée (manquants)")
        if df_buy.empty:
            st.success("RAS : aucune pièce manquante pour le plan.")
        else:
            st.dataframe(df_buy, use_container_width=True, hide_index=True)

            buy_csv = df_buy.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Télécharger la liste d’achat (CSV)",
                data=buy_csv,
                file_name="liste_achat_pieces.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.markdown("### 📄 PDF — Plan, kits, indicateurs + options")

    include_kits = st.checkbox(
        "Inclure les kits (résumé) dans le PDF",
        value=True,
        help="Le détail complet des kits reste géré dans le module Inventaire.",
    )

    colA, colB = st.columns([1, 1])

    # ---- PDF sans déduction ----
    with colA:
        if st.button("📄 Générer le PDF (sans déduire le stock)", use_container_width=True):
            try:
                path_pdf = export_pm_plan_with_kits_pdf(
                    tasks_due=due2 or [],
                    kits_by_eq=kits_by_eq,
                    metrics_table=dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else [],
                    out_dir=str(BASE_DIR / "reports"),
                    title="Plan de maintenance — Tâches, Matériels, Indicateurs",
                    procedure_docx=None,
                    include_kits=bool(include_kits),
                    tools_checklist=None,
                    consumption_summary=None,
                )
                st.session_state["pm_pdf_path"] = path_pdf
                st.success(f"PDF généré : {path_pdf}")
            except Exception as e:
                st.error(f"PDF : {e}")

    # ---- Validation (déduction + PDF + alertes + notif) ----
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
                    tasks_due=due2 or [],
                    kits_by_eq=kits_by_eq,
                    metrics_table=dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else [],
                    out_dir=str(BASE_DIR / "reports"),
                    title="Plan de maintenance — Tâches, Matériels, Indicateurs",
                    procedure_docx=None,
                    include_kits=bool(include_kits),
                    tools_checklist=None,
                    consumption_summary=consumption_summary,
                )

                st.session_state["pm_pdf_path"] = path_pdf

                # Alertes stock (après MAJ)
                low_after = inv.low_stock(threshold_factor=1.0) or []
                if low_after:
                    if notify_stock_alerts:
                        try:
                            notify_stock_alerts(low_after)
                            st.info("Alertes stock envoyées (articles sous seuil).")
                        except Exception as e_alert:
                            st.error(f"Envoi alertes stock : {e_alert}")
                    else:
                        out_low = BASE_DIR / "data" / "low_stock_alert_after_plan.csv"
                        pd.DataFrame(low_after).to_csv(out_low, index=False)
                        st.info(f"Module d’alerte indisponible — CSV généré : {out_low}")

                # Notifications maintenance (plan + kits)
                res = notify_pm_with_kits(due2 or [], kits_by_eq, dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else []) or {}

                st.success("Plan généré ✅ | Stock déduit ✅ | Notifications envoyées ✅")
                if path_pdf:
                    st.caption(f"PDF (Plan) : {path_pdf}")
                if res:
                    st.caption(f"Notify: {res}")
            except Exception as e:
                st.error(f"Erreur validation & envoi : {e}")

    # ---- Download PDF ----
    pdf_path = st.session_state.get("pm_pdf_path")
    if pdf_path and Path(str(pdf_path)).exists():
        st.markdown("### 📥 Télécharger le PDF du plan")
        try:
            with open(str(pdf_path), "rb") as f:
                st.download_button(
                    "📥 Télécharger le plan de maintenance (PDF)",
                    data=f,
                    file_name=Path(str(pdf_path)).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Téléchargement PDF : {e}")
    else:
        st.caption("Aucun PDF prêt au téléchargement pour l’instant.")


# =========================================================
# TAB 4 — Alertes stock
# =========================================================
with tab_alert:
    st.subheader("🔔 Alertes stock — seuils")

    st.caption("Alerte quand `quantite_dispo ≤ seuil_min`. Les destinataires SMTP se règlent dans alerts_config.json.")

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
                        out = BASE_DIR / "data" / "low_stock_alert.csv"
                        pd.DataFrame(low).to_csv(out, index=False)
                        st.info(f"Module d’alerte indisponible — CSV généré : {out}")
                except Exception as e:
                    st.error(f"Alerte stock : {e}")
        with c2:
            if st.button("⬇️ Export CSV", use_container_width=True):
                out = BASE_DIR / "data" / "low_stock_alert.csv"
                pd.DataFrame(low).to_csv(out, index=False)
                st.success(f"Exporté → {out}")

    # Auto anti-spam : 10 min
    if auto and low and notify_stock_alerts:
        last = float(st.session_state.get("_last_stock_alert_ts", 0.0))
        if time.time() - last > 600:
            try:
                notify_stock_alerts(low)
                st.session_state["_last_stock_alert_ts"] = time.time()
                st.toast("Alerte stock auto envoyée.")
            except Exception:
                pass
