from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf
from core.security.auth import require_login


# ============================================================
# Helpers
# ============================================================

def _hash_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if pd.isna(v) or math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _maintenance_label(mtype: str) -> str:
    s = (mtype or "").strip().lower()
    if not s:
        return "Non défini"
    if "correct" in s:
        return "Corrective"
    if "condition" in s or "inspection" in s:
        return "Conditionnelle / Inspection"
    if "predict" in s or "prédict" in s:
        return "Prédictive"
    if "prévent" in s or "prevent" in s:
        return "Préventive planifiée"
    return mtype


def _pick_interval_h(row: dict) -> tuple[Optional[float], Optional[str]]:
    """
    Priorité stricte des colonnes d’intervalle.
    """
    for c in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
        if c in row and row[c] is not None:
            v = _safe_float(row[c], None)
            if v is not None and v > 0:
                return v, c
    return None, None


def _beta_zone(beta: Optional[float]) -> str:
    if beta is None:
        return "unknown"
    if beta < 0.8:
        return "early"
    if beta <= 1.2:
        return "random"
    return "wear"


def _interval_intensity(interval_h: Optional[float]) -> str:
    if interval_h is None:
        return "unknown"
    if interval_h <= 168:
        return "high"
    if interval_h <= 720:
        return "medium"
    return "low"


def build_maintenance_comment(
    *,
    beta: Optional[float],
    eta_h: Optional[float],
    interval_h: Optional[float],
    interval_source: Optional[str],
    mtype_label: str,
    process_model: Optional[str] = None,
    thermal_status: Optional[str] = None,
    thermal_faa_max: Optional[float] = None,
    thermal_lol_pct: Optional[float] = None,
) -> str:
    """
    Génère un commentaire actionnable en combinant :
    - β / η
    - intervalle optimisé
    - type de maintenance
    - modèle global
    - statut thermique
    """
    b = _safe_float(beta, None)
    e = _safe_float(eta_h, None)
    itv = _safe_float(interval_h, None)
    faa = _safe_float(thermal_faa_max, None)
    lol = _safe_float(thermal_lol_pct, None)

    zone = _beta_zone(b)
    intensity = _interval_intensity(itv)
    model_s = (process_model or "").upper()
    src_txt = f"Intervalle basé sur {interval_source}" if interval_source else "Intervalle optimisé"

    risk_hint = ""
    if e is not None and itv is not None:
        ratio = itv / max(e, 1e-9)
        if ratio > 1.0:
            risk_hint = "Intervalle > η : risque de rater des signes avant panne, rapprocher la surveillance."
        elif ratio > 0.5:
            risk_hint = "Intervalle proche de η : vigilance accrue, tendance à la dégradation possible."
        else:
            risk_hint = "Intervalle << η : surveillance conservatrice, bon choix pour équipement critique."

    if zone == "early":
        main = (
            "β<1 : défauts précoces probables. Priorité aux contrôles de mise en service, "
            "à la correction des causes racines et aux inspections rapprochées."
        )
        actions = [
            "Vérifier montage, connexions, serrages et isolement.",
            "Tracer les incidents récurrents et corriger la cause racine.",
            "Maintenir une surveillance rapprochée à court terme.",
        ]
    elif zone == "random":
        main = (
            "β≈1 : régime plutôt aléatoire. Priorité à une maintenance préventive standard "
            "avec surveillance conditionnelle légère."
        )
        actions = [
            "Conserver un calendrier de visites périodiques.",
            "Surveiller température, isolement et signes faibles.",
            "Préparer les kits et consommables pour réduire le MTTR.",
        ]
    else:
        main = (
            "β>1 : régime d’usure. Priorité à une maintenance préventive ciblée, "
            "des inspections conditionnelles renforcées et des remplacements planifiés."
        )
        actions = [
            "Renforcer la fréquence des contrôles conditionnels.",
            "Planifier l’intervention avant le seuil critique.",
            "Vérifier refroidissement, huile, point chaud et organes sollicités.",
        ]

    if "NHPP" in model_s:
        main += " Le processus global signale aussi une tendance de vieillissement."
    elif "BPP" in model_s:
        main += " Le processus global suggère une dépendance entre événements : vérifier causes communes et enchaînements."

    if thermal_status and "alerte" in thermal_status.lower():
        actions.insert(0, "Contrôler immédiatement l’échauffement et le système de refroidissement.")
    elif thermal_status and "conforme" in thermal_status.lower():
        actions.insert(0, "Conserver la surveillance thermique actuelle.")

    if intensity == "high":
        freq = "Fréquence élevée (≤ 7 jours) : recommandée pour équipement à risque / critique."
    elif intensity == "medium":
        freq = "Fréquence moyenne (≤ 30 jours) : compromis coût / risque généralement acceptable."
    elif intensity == "low":
        freq = "Fréquence faible (> 30 jours) : acceptable si le risque reste faible et la thermique stable."
    else:
        freq = "Fréquence non déterminée (intervalle manquant)."

    parts = [
        f"Type recommandé : {mtype_label}.",
        main,
        f"{src_txt} : {itv:.1f} h." if itv is not None else f"{src_txt} : non disponible.",
        freq,
    ]
    if e is not None:
        parts.append(f"η ≈ {e:.1f} h.")
    if faa is not None:
        parts.append(f"FAA max ≈ {faa:.3f}.")
    if lol is not None:
        parts.append(f"Perte de vie ≈ {lol:.3f} %.")
    if risk_hint:
        parts.append(risk_hint)
    parts.append("Actions : " + " ".join(actions))
    return " ".join(parts)


def _load_optimization_fallback() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    fallback = base_dir / "data" / "last_optimization.csv"
    if fallback.exists():
        try:
            df = pd.read_csv(fallback)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ============================================================
# Core builder
# ============================================================

def build_virtual_pm_plan_from_optimization(
    opt_df: pd.DataFrame,
    start_date: date,
    within_days: int,
    show_all: bool = True,
    only_preventive: bool = False,
    only_admissible: bool = False,
    min_days: int = 1,
) -> Dict[str, Any]:
    """
    Construit un plan PM virtuel à partir du tableau d’optimisation.

    Règle :
      - interval_h = T_recommended_h > T_R_h > T_cost_h > interval_opt_h > interval_h
      - periodicity_days = max(min_days, round(interval_h/24))
      - next_due_date = start_date + periodicity_days
    """
    if opt_df is None or opt_df.empty:
        return {"ok": False, "msg": "Optimisation vide", "rows": [], "due": []}

    df = opt_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "equipment_code" not in df.columns:
        return {"ok": False, "msg": "Colonne equipment_code absente", "rows": [], "due": []}

    within_days = int(within_days or 0)
    within_days = max(within_days, 0)

    rows: List[Dict[str, Any]] = []
    used_cols = {"T_recommended_h": 0, "T_R_h": 0, "T_cost_h": 0, "interval_opt_h": 0, "interval_h": 0, "none": 0}

    for _, r in df.iterrows():
        row = r.to_dict()
        eq = str(row.get("equipment_code") or "").strip()
        if not eq:
            continue

        admissible = row.get("admissible_global")
        if only_admissible and admissible is not None and not bool(admissible):
            continue

        mtype = _maintenance_label(str(row.get("maintenance_type") or "").strip())
        if only_preventive and "Préventive" not in mtype:
            continue

        interval_h, src = _pick_interval_h(row)
        if not interval_h:
            used_cols["none"] += 1
            continue

        if src in used_cols:
            used_cols[src] += 1

        periodicity_days = max(int(min_days), int(round(float(interval_h) / 24.0)))
        next_due = start_date + timedelta(days=periodicity_days)
        days_left = (next_due - start_date).days

        beta = _safe_float(row.get("beta_pipe"), None)
        if beta is None:
            beta = _safe_float(row.get("beta_opt"), None)
        eta_h = _safe_float(row.get("eta_pipe_h"), None)
        if eta_h is None:
            eta_h = _safe_float(row.get("eta_opt_h"), None)

        comment = build_maintenance_comment(
            beta=beta,
            eta_h=eta_h,
            interval_h=_safe_float(interval_h, None),
            interval_source=src,
            mtype_label=mtype,
            process_model=str(row.get("process_model") or row.get("model") or ""),
            thermal_status=str(row.get("thermal_status") or ""),
            thermal_faa_max=_safe_float(row.get("FAA_max"), None),
            thermal_lol_pct=_safe_float(row.get("loss_of_life_pct"), None),
        )

        task = {
            "equipment_code": eq,
            "title": "Plan issu de l’optimisation",
            "maintenance_type": mtype,
            "maintenance_comment": comment,
            "interval_source": src,
            "interval_h": float(interval_h),
            "periodicity_days": periodicity_days,
            "next_due_date": next_due.isoformat(),
            "days_left": int(days_left),

            "beta": beta,
            "eta_h": eta_h,
            "gamma_h": row.get("gamma_pipe_h", row.get("gamma_opt_h")),
            "model": row.get("process_model", row.get("model")),
            "distribution": row.get("distribution"),

            "T_recommended_h": row.get("T_recommended_h"),
            "T_R_h": row.get("T_R_h"),
            "T_cost_h": row.get("T_cost_h"),
            "R_at_T": row.get("R_at_T", row.get("R(T_cost)")),
            "C_min_per_h": row.get("C_min_per_h", row.get("C_min")),

            "FAA_max": row.get("FAA_max"),
            "loss_of_life_pct": row.get("loss_of_life_pct"),
            "thermal_status": row.get("thermal_status"),
            "thermal_ok": row.get("thermal_ok"),
            "reliability_ok": row.get("reliability_ok"),
            "admissible_global": row.get("admissible_global"),

            "status": "VIRTUAL",
        }

        if show_all or (days_left <= within_days):
            rows.append(task)

    rows = sorted(rows, key=lambda x: (x.get("days_left", 999999), x.get("equipment_code", "")))
    due = [t for t in rows if int(t.get("days_left", 999999)) <= within_days]

    return {
        "ok": True,
        "rows": rows,
        "due": due,
        "used_cols_stats": used_cols,
        "interval_priority": ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"],
    }


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Maintenance", page_icon="🛠️", layout="wide")
require_login()

st.title("🛠️ Maintenance (basée sur l’optimisation)")
st.caption("Plan PM virtuel + recommandations + export PDF. Fonctionne avec la session Streamlit ou le fallback fichier.")

df_opt = st.session_state.get("optimization_df")
if not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
    df_opt = _load_optimization_fallback()

if not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
    st.info("Aucune optimisation disponible. Va d’abord sur la page Optimisation, puis reviens ici.")
    st.stop()

df_opt = df_opt.copy()
df_opt.columns = [str(c).strip() for c in df_opt.columns]

h = _hash_df(df_opt)
st.success(f"Dataset optimisation synchronisé ✅ | rows={len(df_opt)} | hash={h}")

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c1:
    within = st.slider("Fenêtre (jours) — tâches dues", 7, 365, 14, 1)
with c2:
    show_all = st.toggle("Afficher tout le planning", value=True)
with c3:
    only_prev = st.toggle("Seulement Préventive", value=False)
with c4:
    only_admissible = st.toggle("Seulement admissible", value=False)
with c5:
    start_dt = st.date_input("Date de départ", value=date.today())

plan = build_virtual_pm_plan_from_optimization(
    opt_df=df_opt,
    start_date=start_dt,
    within_days=int(within),
    show_all=bool(show_all),
    only_preventive=bool(only_prev),
    only_admissible=bool(only_admissible),
    min_days=1,
)

if not plan.get("ok"):
    st.error(plan.get("msg", "Erreur plan"))
    st.stop()

st.caption(f"Priorité intervalles : {', '.join(plan.get('interval_priority', []))}")
st.caption(f"Colonnes réellement utilisées : {plan.get('used_cols_stats')}")

rows_all = plan.get("rows", [])
rows_due = plan.get("due", [])

# ------------------------------------------------------------
# KPI
# ------------------------------------------------------------
colk1, colk2, colk3, colk4 = st.columns(4)
with colk1:
    st.metric("Équipements planifiés", len(rows_all))
with colk2:
    st.metric("Tâches dues", len(rows_due))
with colk3:
    admissibles = int(pd.Series([r.get("admissible_global") for r in rows_all]).fillna(False).astype(bool).sum()) if rows_all else 0
    st.metric("Plans admissibles", admissibles)
with colk4:
    preventive = int(sum(1 for r in rows_all if "Préventive" in str(r.get("maintenance_type", ""))))
    st.metric("Préventives", preventive)

# ------------------------------------------------------------
# Résumé commentaires
# ------------------------------------------------------------
st.markdown("## 0) Commentaires maintenance (résumé)")
if rows_all:
    df_comm = pd.DataFrame(rows_all)[
        [c for c in [
            "equipment_code",
            "maintenance_type",
            "interval_source",
            "interval_h",
            "beta",
            "eta_h",
            "thermal_status",
            "maintenance_comment",
        ] if c in pd.DataFrame(rows_all).columns]
    ].copy()
    df_comm = df_comm.drop_duplicates(subset=["equipment_code"]).sort_values("equipment_code")
    st.dataframe(df_comm, use_container_width=True, hide_index=True)
else:
    st.info("Aucun commentaire : aucune ligne exploitable.")

# ------------------------------------------------------------
# Planning complet
# ------------------------------------------------------------
st.markdown("## 1) Planning (issu de l’optimisation)")
if not rows_all:
    st.warning("Aucune ligne exploitable (intervalles manquants ou <=0).")
else:
    df_all = pd.DataFrame(rows_all)
    cols = [
        "equipment_code",
        "maintenance_type",
        "interval_source",
        "interval_h",
        "periodicity_days",
        "next_due_date",
        "days_left",
        "beta",
        "eta_h",
        "model",
        "distribution",
        "FAA_max",
        "loss_of_life_pct",
        "thermal_status",
        "admissible_global",
        "T_recommended_h",
        "T_R_h",
        "T_cost_h",
        "maintenance_comment",
    ]
    cols = [c for c in cols if c in df_all.columns]
    st.dataframe(df_all[cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Tâches dues
# ------------------------------------------------------------
st.markdown("## 2) Tâches dues (dans la fenêtre)")
if not rows_due:
    st.info("Aucune tâche due dans la fenêtre. Augmente la fenêtre ou change la date de départ.")
else:
    df_due = pd.DataFrame(rows_due)
    cols = [
        "equipment_code",
        "maintenance_type",
        "interval_source",
        "interval_h",
        "next_due_date",
        "days_left",
        "beta",
        "eta_h",
        "model",
        "distribution",
        "FAA_max",
        "loss_of_life_pct",
        "thermal_status",
        "admissible_global",
        "T_recommended_h",
        "T_R_h",
        "T_cost_h",
        "maintenance_comment",
    ]
    cols = [c for c in cols if c in df_due.columns]
    st.dataframe(df_due[cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Recommandation finale simple
# ------------------------------------------------------------
st.markdown("## 3) Recommandation finale")
if rows_all:
    df_rank = pd.DataFrame(rows_all).copy()
    if "admissible_global" in df_rank.columns:
        df_rank["admissible_global"] = df_rank["admissible_global"].fillna(False).astype(bool)
    if "days_left" in df_rank.columns:
        df_rank["days_left"] = pd.to_numeric(df_rank["days_left"], errors="coerce").fillna(999999)
    if "C_min_per_h" in df_rank.columns:
        df_rank["C_min_per_h"] = pd.to_numeric(df_rank["C_min_per_h"], errors="coerce")

    # priorité : admissible, dû tôt, coût plus bas
    sort_cols = [c for c in ["admissible_global", "days_left", "C_min_per_h"] if c in df_rank.columns]
    ascending = []
    for c in sort_cols:
        if c == "admissible_global":
            ascending.append(False)
        else:
            ascending.append(True)

    if sort_cols:
        df_rank = df_rank.sort_values(sort_cols, ascending=ascending)

    best = df_rank.iloc[0].to_dict()
    st.success(
        f"Équipement prioritaire : {best.get('equipment_code', '—')} | "
        f"{best.get('maintenance_type', '—')} | "
        f"échéance {best.get('next_due_date', '—')}"
    )
    st.write(best.get("maintenance_comment", "—"))

# ------------------------------------------------------------
# Exports
# ------------------------------------------------------------
st.session_state["pm_virtual_all"] = rows_all
st.session_state["pm_virtual_due"] = rows_due

st.markdown("## 4) Exports")
if rows_all:
    df_all = pd.DataFrame(rows_all)
    csv_all = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger le planning CSV",
        data=csv_all,
        file_name="maintenance_virtual_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

include_all_in_pdf = st.toggle("Inclure tout le planning dans le PDF (sinon seulement les tâches dues)", value=False)

if st.button("📄 Générer PDF (plan issu optimisation)", use_container_width=True):
    tasks_for_pdf = rows_all if include_all_in_pdf else rows_due
    metrics_table = df_opt.to_dict("records")

    out = export_pm_plan_with_kits_pdf(
        tasks_due=tasks_for_pdf,
        kits_by_eq={},
        metrics_table=metrics_table,
        out_dir="reports",
        title="Plan de maintenance — issu de l’optimisation",
        include_kits=False,
        tools_checklist=None,
    )
    st.success("PDF généré.")
    with open(out, "rb") as f:
        st.download_button(
            "⬇️ Télécharger le PDF",
            f,
            file_name=Path(out).name,
            mime="application/pdf",
            use_container_width=True,
        )
