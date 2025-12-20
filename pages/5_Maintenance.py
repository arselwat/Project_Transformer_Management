# pages/5_Maintenance.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.maintenance import services as pm
from core.maintenance.bridge import upsert_tasks_from_optimization, BridgeParams

# PDF / notif
from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.notify.alerts_plus import notify_pm_with_kits

from core.datahub import get_failures_meta  # traçabilité

st.set_page_config(page_title="Maintenance", page_icon="🛠️", layout="wide")
require_login()

st.title("🛠️ Maintenance (depuis Optimisation)")
st.caption("But : prendre le résultat d’Optimisation → créer le planning → générer le plan de maintenance.")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# par défaut, on s’attend à un export de la page Optimisation
DEFAULT_OPT_CSV = DATA_DIR / "optimisation_intervalles.csv"
ALT_OPT_CSV = DATA_DIR / "optimisation_last.csv"


# =========================
# 1) Charger optimisation
# =========================
st.subheader("1) Charger les résultats d’optimisation")

c1, c2 = st.columns([2, 1])
with c1:
    up = st.file_uploader("CSV d’optimisation (depuis la page Optimisation)", type=["csv"])
with c2:
    st.info("Astuce : tu peux aussi déposer le fichier dans /data.")

df_opt = pd.DataFrame()

if up is not None:
    df_opt = pd.read_csv(up)
else:
    if ALT_OPT_CSV.exists():
        df_opt = pd.read_csv(ALT_OPT_CSV)
    elif DEFAULT_OPT_CSV.exists():
        df_opt = pd.read_csv(DEFAULT_OPT_CSV)

if df_opt.empty:
    st.warning("Aucun fichier optimisation chargé.")
    st.stop()

df_opt.columns = [str(c).strip() for c in df_opt.columns]
st.success(f"Optimisation chargée ✅ ({len(df_opt)} lignes)")
st.dataframe(df_opt.head(20), use_container_width=True, hide_index=True)


# =========================
# 2) Synchroniser → pm_task
# =========================
st.subheader("2) Générer le planning de maintenance (pm_task) à partir d’optimisation")

colA, colB, colC = st.columns(3)
with colA:
    only_prev = st.toggle("Créer uniquement si Préventive", value=True)
with colB:
    min_days = st.number_input("Périodicité min (jours)", min_value=1, value=7, step=1)
with colC:
    start_dt = st.date_input("Date de départ", value=None)

if st.button("✅ Synchroniser planning", type="primary"):
    res = upsert_tasks_from_optimization(
        opt_df=df_opt,
        start_date=str(start_dt) if start_dt else None,
        params=BridgeParams(min_days=int(min_days), only_if_preventive=bool(only_prev)),
    )
    if res.get("ok"):
        st.success(f"Planning créé/MAJ ✅ | created={res['created']} | updated={res['updated']} | skipped={res['skipped']}")
    else:
        st.warning(f"Planning partiel | created={res['created']} | updated={res['updated']} | skipped={res['skipped']}")
        if res.get("errors"):
            st.error(" | ".join(res["errors"]))

    st.rerun()


# =========================
# 3) Plan de maintenance (issu OPT)
# =========================
st.subheader("3) Générer le plan (issu du planning OPT)")

within = st.slider("Fenêtre (jours) pour tâches dues", 7, 90, 14, 1)

# on récupère les tâches “due” puis on filtre celles issues optimisation
due = pm.due_within(days=int(within)) or []
due_opt = [t for t in due if str(t.get("source","")).upper() == "OPTIMISATION"]

if not due_opt:
    st.info("Aucune tâche due issue de l’optimisation dans cette fenêtre.")
    st.stop()

df_due = pd.DataFrame(due_opt)
# colonnes “simples” + maintenance_type visible
cols = [c for c in ["equipment_code","title","maintenance_type","periodicity_days","next_due_date","days_left","status"] if c in df_due.columns]
st.dataframe(df_due[cols] if cols else df_due, use_container_width=True, hide_index=True)

# PDF + download
st.divider()
st.markdown("### 📄 PDF — Plan de maintenance")

if st.button("📄 Générer le plan PDF", type="primary"):
    try:
        meta = get_failures_meta()
        ds_hash = meta.get("hash") if meta.get("ok") else "unknown"

        path_pdf = export_pm_plan_with_kits_pdf(
            tasks_due=due_opt,
            kits_by_eq={},  # ❌ stock retiré de maintenance
            metrics_table=[],  # tu peux laisser vide si tu veux un PDF “plan pur”
            out_dir=str(REPORTS_DIR),
            title=f"Plan de maintenance (issu optimisation) — dataset={ds_hash}",
            procedure_docx=None,
            include_kits=False,
            tools_checklist=None,
            consumption_summary=None,
        )
        st.session_state["pm_plan_pdf"] = path_pdf
        st.success(f"PDF généré : {path_pdf}")
    except Exception as e:
        st.error(f"PDF : {e}")

pdf_path = st.session_state.get("pm_plan_pdf")
if pdf_path and Path(str(pdf_path)).exists():
    with open(str(pdf_path), "rb") as f:
        st.download_button(
            "📥 Télécharger le plan PDF",
            data=f,
            file_name=Path(str(pdf_path)).name,
            mime="application/pdf",
            use_container_width=True,
        )

# Notif (optionnel)
st.divider()
st.markdown("### 🔔 Notifications (optionnel)")

if st.button("📤 Envoyer les notifications du plan"):
    try:
        res = notify_pm_with_kits(due_opt, {}, []) or {}
        st.success("Notifications envoyées ✅")
        st.json(res, expanded=False)
    except Exception as e:
        st.error(f"Notifications : {e}")
