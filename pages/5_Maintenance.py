# pages/5_Maintenance.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.datahub import get_failures_meta, get_current_failures_df

from core.maintenance import services as pm
from core.reliability.unify import compute_bundle

from core.inventory.recommendations import build_pm_kit_for_equipment
from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.notify.alerts_plus import notify_pm_with_kits

st.set_page_config(page_title="Maintenance", page_icon="🛠️", layout="wide")
require_login()

st.title("🛠️ Maintenance (simple)")
st.caption("Ici : **tâches dues** + **kits recommandés** + **PDF plan**. Le stock se gère uniquement dans la page **Stock**.")

BASE_DIR = Path(__file__).resolve().parents[1]

# --- dataset meta (juste pour rassurer que tout est aligné)
meta = get_failures_meta()
if meta.get("ok"):
    st.success(f"Dataset synchronisé ✅ | rows={meta['rows']} | hash={meta['hash']} | source={meta['source']}")
else:
    st.warning("Aucun dataset actif. Va sur « Sources de données ».")

# ----------------------------
# 1) tâches dues
# ----------------------------
st.subheader("1) Tâches dues (prochaines semaines)")
within = st.slider("Fenêtre (jours)", 7, 90, 14, 1)

try:
    due = pm.due_within(days=int(within)) or []
except Exception as e:
    st.error(f"Lecture tâches dues : {e}")
    due = []

if not due:
    st.info("Aucune tâche due dans l’intervalle. (As-tu synchronisé depuis Optimisation ?)")
    st.stop()

df_due = pd.DataFrame(due)
keep = [c for c in ["id","equipment_code","title","periodicity_days","next_due_date","days_left","status"] if c in df_due.columns]
st.dataframe(df_due[keep] if keep else df_due, use_container_width=True, hide_index=True)

# ----------------------------
# 2) kits recommandés (résumé)
# ----------------------------
st.subheader("2) Kits recommandés (résumé)")
eqs = sorted({str(d.get("equipment_code")) for d in due if d.get("equipment_code")})
kits_by_eq = {}
for eq in eqs:
    try:
        kits_by_eq[eq] = build_pm_kit_for_equipment(eq) or []
    except Exception:
        kits_by_eq[eq] = []

kit_rows = [{"equipment_code": eq, "nb_items": len(kits_by_eq.get(eq) or [])} for eq in eqs]
st.dataframe(pd.DataFrame(kit_rows), use_container_width=True, hide_index=True)

with st.expander("Voir le détail des kits (par équipement)", expanded=False):
    for eq in eqs:
        st.markdown(f"### {eq}")
        st.dataframe(pd.DataFrame(kits_by_eq.get(eq) or []), use_container_width=True, hide_index=True)

# ----------------------------
# 3) PDF plan + download
# ----------------------------
st.subheader("3) Générer le plan de maintenance (PDF)")

include_kits = st.checkbox("Inclure les kits dans le PDF", value=True)

# metrics table (pour enrichir le PDF)
df_ttf = get_current_failures_df()
try:
    bundle = compute_bundle(df_ttf)
    dfm = bundle.metrics_df.copy() if hasattr(bundle, "metrics_df") else pd.DataFrame()
except Exception:
    dfm = pd.DataFrame()

c1, c2 = st.columns(2)
with c1:
    if st.button("📄 Générer PDF (plan)", type="primary", use_container_width=True):
        try:
            path_pdf = export_pm_plan_with_kits_pdf(
                tasks_due=due,
                kits_by_eq=kits_by_eq,
                metrics_table=dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else [],
                out_dir=str(BASE_DIR / "reports"),
                title="Plan de maintenance — Tâches & Kits",
                procedure_docx=None,
                include_kits=bool(include_kits),
                tools_checklist=None,
                consumption_summary=None,
            )
            st.session_state["pm_plan_pdf"] = path_pdf
            st.success(f"PDF généré : {path_pdf}")
        except Exception as e:
            st.error(f"PDF : {e}")

with c2:
    if st.button("📤 Envoyer notifications (plan + kits)", use_container_width=True):
        try:
            res = notify_pm_with_kits(due, kits_by_eq, dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else []) or {}
            st.success("Notifications envoyées ✅")
            st.json(res, expanded=False)
        except Exception as e:
            st.error(f"Notify : {e}")

pdf_path = st.session_state.get("pm_plan_pdf")
if pdf_path and Path(str(pdf_path)).exists():
    with open(str(pdf_path), "rb") as f:
        st.download_button(
            "⬇️ Télécharger le plan de maintenance (PDF)",
            data=f,
            file_name=Path(str(pdf_path)).name,
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.caption("Génère d’abord le PDF pour activer le téléchargement.")
