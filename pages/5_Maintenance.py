# pages/5_Maintenance.py
from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import hashlib
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from core.security.auth import require_login
from core.reliability.unify import compute_bundle

from core.inventory.recommendations import build_pm_kit_for_equipment
from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.notify.alerts_plus import notify_pm_with_kits


# =========================================================
# Config + Auth
# =========================================================
st.set_page_config(page_title="Maintenance (simple)", page_icon="🛠️", layout="wide")
require_login()

st.title("🛠️ Maintenance (simple)")
st.caption("Ici : tâches dues + kits recommandés + PDF plan. Le stock se gère uniquement dans la page Stock.")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_CSV = BASE_DIR / "data" / "failures_saved.csv"
OPT_FALLBACK = BASE_DIR / "data" / "last_optimization.csv"


# =========================================================
# Helpers dataset (source/indicateurs)
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


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _df_hash(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


# =========================================================
# Helpers optimisation bridge (NO DB)
# =========================================================
def _load_opt_df() -> pd.DataFrame:
    df = st.session_state.get("opt_df_out")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()

    if OPT_FALLBACK.exists():
        try:
            d0 = pd.read_csv(OPT_FALLBACK)
            if not d0.empty:
                return d0
        except Exception:
            pass

    return pd.DataFrame()


def _interval_col(df: pd.DataFrame) -> str | None:
    for c in ["T_recommended_h", "T_R_h", "T_cost_h"]:
        if c in df.columns:
            return c
    return None


def _tasks_from_optimization(df_opt: pd.DataFrame, within_days: int) -> List[Dict[str, Any]]:
    """
    Fabrique des tâches 'virtuelles' depuis optimisation (sans BD pm_task).
    - periodicity_days = round(interval_h / 24)
    - next_due_date = today + periodicity_days
    """
    if df_opt is None or df_opt.empty:
        return []

    df = df_opt.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "equipment_code" not in df.columns:
        return []

    col = _interval_col(df)
    if not col:
        return []

    today = date.today()
    out: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        eq = str(r.get("equipment_code") or "").strip()
        if not eq:
            continue

        interval_h = _safe_float(r.get(col), 0.0)
        if interval_h <= 0:
            continue

        periodicity_days = max(1, int(round(interval_h / 24.0)))
        next_due = today + timedelta(days=periodicity_days)
        days_left = (next_due - today).days

        if days_left <= int(within_days):
            out.append({
                "id": None,
                "equipment_code": eq,
                "title": "Maintenance issue de l’optimisation",
                "maintenance_type": str(r.get("maintenance_type") or "").strip(),
                "periodicity_days": periodicity_days,
                "interval_h": interval_h,
                "next_due_date": next_due.isoformat(),
                "days_left": days_left,
                "status": "VIRTUAL",
            })

    return sorted(out, key=lambda x: x.get("days_left", 999999))


# =========================================================
# Kits
# =========================================================
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


# =========================================================
# Bandeau sync (debug clair)
# =========================================================
df_opt = _load_opt_df()
meta = st.session_state.get("opt_meta", {}) or {}

if df_opt.empty:
    st.warning(
        "Aucun résultat d’optimisation disponible. "
        "Va dans la page **Optimisation** puis clique **'Envoyer ce planning à Maintenance'** "
        "(ou sauvegarde le fallback `data/last_optimization.csv`)."
    )
else:
    h = meta.get("hash") or _df_hash(df_opt)
    rows = int(meta.get("rows") or len(df_opt))
    src = meta.get("source") or ("file:last_optimization.csv" if OPT_FALLBACK.exists() else "session")
    st.success(f"Dataset optimisation synchronisé ✅ | rows={rows} | hash={h} | source={src}")


# =========================================================
# UI
# =========================================================
st.divider()
st.subheader("1) Tâches dues (prochaines semaines)")

within = st.slider("Fenêtre (jours)", 7, 90, 14, 1)

due = _tasks_from_optimization(df_opt, within_days=int(within)) if not df_opt.empty else []

if not due:
    st.info("Aucune tâche due dans la fenêtre (sur base optimisation).")
    st.stop()

df_due = pd.DataFrame(due)
cols = [c for c in ["equipment_code", "title", "maintenance_type", "periodicity_days", "next_due_date", "days_left", "status"] if c in df_due.columns]
st.dataframe(df_due[cols] if cols else df_due, use_container_width=True, hide_index=True)


st.divider()
st.subheader("2) Kits recommandés (résumé)")

kits_by_eq = _build_kits_by_eq(due)
kit_rows = [{"equipment_code": eq, "nb_items": len(kit or [])} for eq, kit in kits_by_eq.items()]
st.dataframe(pd.DataFrame(kit_rows), use_container_width=True, hide_index=True)

with st.expander("Voir le détail des kits (par équipement)", expanded=False):
    for eq, kit in kits_by_eq.items():
        st.markdown(f"**{eq}** — {len(kit or [])} item(s)")
        if kit:
            st.dataframe(pd.DataFrame(kit), use_container_width=True, hide_index=True)
        else:
            st.caption("Aucun kit configuré pour cet équipement.")


# =========================================================
# PDF plan + download + notifications
# =========================================================
st.divider()
st.subheader("3) Générer le plan de maintenance (PDF)")

include_kits = st.checkbox("Inclure les kits dans le PDF", value=True)

# fiabilité (pour enrichir PDF)
try:
    bundle = compute_bundle(_ttf_df())
    dfm = bundle.metrics_df.copy() if hasattr(bundle, "metrics_df") else pd.DataFrame()
    metrics_table = dfm.to_dict("records") if isinstance(dfm, pd.DataFrame) else []
except Exception:
    metrics_table = []

colA, colB = st.columns([1, 1])

with colA:
    if st.button("📄 Générer PDF (plan)", type="primary", use_container_width=True):
        try:
            path_pdf = export_pm_plan_with_kits_pdf(
                tasks_due=due,
                kits_by_eq=kits_by_eq,
                metrics_table=metrics_table,
                out_dir=str(BASE_DIR / "reports"),
                title="Plan de maintenance — issu de l’optimisation",
                procedure_docx=None,
                include_kits=bool(include_kits),
                tools_checklist=None,
                consumption_summary=None,
            )
            st.session_state["pm_pdf_path"] = path_pdf
            st.success(f"PDF généré : {path_pdf}")
        except Exception as e:
            st.error(f"PDF : {e}")

with colB:
    if st.button("📨 Envoyer notifications (plan + kits)", use_container_width=True):
        try:
            res = notify_pm_with_kits(due, kits_by_eq, metrics_table) or {}
            st.success("Notifications envoyées ✅")
            if res:
                st.caption(f"Détails: {res}")
        except Exception as e:
            st.error(f"Notifications : {e}")


pdf_path = st.session_state.get("pm_pdf_path")
if pdf_path and Path(str(pdf_path)).exists():
    st.divider()
    st.subheader("📥 Télécharger le plan de maintenance (PDF)")
    with open(str(pdf_path), "rb") as f:
        st.download_button(
            "⬇️ Télécharger le plan (PDF)",
            data=f,
            file_name=Path(str(pdf_path)).name,
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.caption("Aucun PDF prêt au téléchargement pour l’instant.")
