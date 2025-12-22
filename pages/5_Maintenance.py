# pages/5_Maintenance.py
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import math
import pandas as pd
import streamlit as st

from core.maintenance.reporting_plus import export_pm_plan_with_kits_pdf


# ============================================================
# Helpers (Optimisation -> Plan PM virtuel)
# ============================================================

def _hash_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()

def _safe_float(x, default=None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if pd.isna(v) or math.isinf(v) or math.isnan(v):
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
    Priorité stricte:
      1) T_recommended_h
      2) T_R_h
      3) T_cost_h
      4) interval_opt_h
      5) interval_h
    """
    for c in ["T_recommended_h", "T_R_h", "T_cost_h", "interval_opt_h", "interval_h"]:
        if c in row and row[c] is not None:
            v = _safe_float(row[c], None)
            if v is not None and v > 0:
                return v, c
    return None, None


# ============================================================
# Commentaires maintenance (β, η, intervalle optimisé)
# ============================================================

def _beta_zone(beta: Optional[float]) -> str:
    if beta is None:
        return "unknown"
    if beta < 0.8:
        return "early"      # défauts précoces
    if beta <= 1.2:
        return "random"     # aléatoire ~ constant
    return "wear"           # usure

def _interval_intensity(interval_h: Optional[float]) -> str:
    """
    Classe intervalle (en heures) en intensité simple.
    """
    if interval_h is None:
        return "unknown"
    if interval_h <= 168:   # <= 7 jours
        return "high"
    if interval_h <= 720:   # <= 30 jours
        return "medium"
    return "low"

def build_maintenance_comment(
    *,
    beta: Optional[float],
    eta_h: Optional[float],
    interval_h: Optional[float],
    interval_source: Optional[str],
    mtype_label: str,
) -> str:
    """
    Génère un commentaire clair et actionnable à partir de :
      - β : forme (pannes précoces / aléatoire / usure)
      - η : échelle (niveau global de vie)
      - interval_h : intervalle optimisé
      - interval_source : T_recommended_h / T_R_h / T_cost_h / interval_opt_h ...
      - type de maintenance (label)
    """
    b = _safe_float(beta, None)
    e = _safe_float(eta_h, None)
    itv = _safe_float(interval_h, None)

    zone = _beta_zone(b)
    intensity = _interval_intensity(itv)

    # --- Lecture rapide η vs intervalle
    # (si eta est petit et intervalle petit => risque élevé)
    risk_hint = ""
    if e is not None and itv is not None:
        ratio = itv / max(e, 1e-9)
        if ratio > 1.0:
            risk_hint = "Intervalle > η : risque de rater des signes avant panne, rapprocher la surveillance."
        elif ratio > 0.5:
            risk_hint = "Intervalle proche de η : vigilance accrue, tendance à la dégradation possible."
        else:
            risk_hint = "Intervalle << η : surveillance conservatrice, bon pour équipements critiques."

    src_txt = f"Intervalle basé sur {interval_source}" if interval_source else "Intervalle optimisé"

    # --- Commentaire principal selon β
    if zone == "early":
        main = (
            "β<1 : défauts précoces probables. Priorité : contrôles de mise en service, "
            "qualité d’installation, resserrages, inspections rapprochées et correction des causes racines."
        )
        actions = [
            "Faire une inspection ciblée (connexions, échauffements, isolement).",
            "Vérifier conditions d’exploitation (surcharge, ventilation, humidité).",
            "Tracer chaque incident pour éliminer la cause répétitive.",
        ]
    elif zone == "random":
        main = (
            "β≈1 : pannes plutôt aléatoires. Priorité : maintenance préventive standard + surveillance CBM légère "
            "(contrôles périodiques + seuils)."
        )
        actions = [
            "Maintenir un calendrier de visite régulier.",
            "Surveiller indicateurs clés (T°, DGA/huile si dispo, isolement).",
            "Préparer kits minimaux pour réductions de MTTR.",
        ]
    else:  # wear
        main = (
            "β>1 : régime d’usure. Priorité : CBM + préventif ciblé + remplacements planifiés avant défaillance "
            "(inspection renforcée, tests diélectriques, contrôle refroidissement/OLTC)."
        )
        actions = [
            "Augmenter la fréquence des contrôles conditionnels (tendance).",
            "Planifier une intervention préventive (avant seuil critique).",
            "Vérifier huile, point chaud, refroidissement, OLTC (si présent).",
        ]

    # --- Ajuster selon intensité intervalle
    if intensity == "high":
        freq = "Fréquence élevée (≤ 7 jours) : recommandé pour équipement à risque / critique."
    elif intensity == "medium":
        freq = "Fréquence moyenne (≤ 30 jours) : bon compromis coût/risque."
    elif intensity == "low":
        freq = "Fréquence faible (> 30 jours) : acceptable si risque faible et indicateurs stables."
    else:
        freq = "Fréquence non déterminée (intervalle manquant)."

    # --- Clarifier le type de maintenance
    type_line = f"Type recommandé : {mtype_label}."

    # --- Construire texte final
    parts = [
        type_line,
        main,
        f"{src_txt} : {itv:.1f} h." if itv is not None else f"{src_txt} : (non disponible).",
        freq,
    ]
    if e is not None:
        parts.append(f"η ≈ {e:.1f} h (niveau global de vie / échelle).")
    if risk_hint:
        parts.append(risk_hint)
    parts.append("Actions : " + " ".join(actions))

    return " ".join(parts)


# ============================================================
# Core builder
# ============================================================

def build_virtual_pm_plan_from_optimization(
    opt_df: pd.DataFrame,
    start_date: date,
    within_days: int,
    show_all: bool = True,
    only_preventive: bool = False,
    min_days: int = 1,
) -> Dict[str, Any]:
    """
    Construit un plan PM VIRTUEL depuis l'optimisation (sans BD).

    Règle:
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
    if within_days < 0:
        within_days = 0

    rows: List[Dict[str, Any]] = []
    used_cols = {"T_recommended_h": 0, "T_R_h": 0, "T_cost_h": 0, "interval_opt_h": 0, "interval_h": 0, "none": 0}

    for _, r in df.iterrows():
        row = r.to_dict()
        eq = str(row.get("equipment_code") or "").strip()
        if not eq:
            continue

        mtype = _maintenance_label(str(row.get("maintenance_type") or "").strip())
        if only_preventive and ("Préventive" not in mtype):
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

        beta = row.get("beta")
        eta_h = row.get("eta_h", row.get("eta"))
        comment = build_maintenance_comment(
            beta=_safe_float(beta, None),
            eta_h=_safe_float(eta_h, None),
            interval_h=_safe_float(interval_h, None),
            interval_source=src,
            mtype_label=mtype,
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

            # garder infos optimisation utiles au PDF
            "beta": beta,
            "eta_h": eta_h,
            "gamma_h": row.get("gamma_h", row.get("gamma")),
            "model": row.get("model"),
            "distribution": row.get("distribution"),

            "T_recommended_h": row.get("T_recommended_h"),
            "T_R_h": row.get("T_R_h"),
            "T_cost_h": row.get("T_cost_h"),
            "T_cost": row.get("T_cost"),
            "R_at_T": row.get("R_at_T", row.get("R(T_cost)")),
            "C_min_per_h": row.get("C_min_per_h", row.get("C_min")),

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

st.title("🛠️ Maintenance (basée sur l’optimisation)")
st.caption("Ici : plan PM virtuel + PDF. Sans SQLite/BD (temporaire).")

df_opt = st.session_state.get("optimization_df")
if not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
    st.info("Aucune optimisation en mémoire. Va d’abord sur la page Optimisation, puis reviens ici.")
    st.stop()

h = _hash_df(df_opt)
st.success(f"Dataset optimisation synchronisé ✅ | rows={len(df_opt)} | hash={h} | source=optimisation_page")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    within = st.slider("Fenêtre (jours) — tâches dues", 7, 365, 14, 1)
with c2:
    show_all = st.toggle("Afficher tout le planning", value=True)
with c3:
    only_prev = st.toggle("Seulement Préventive", value=False)
with c4:
    start_dt = st.date_input("Date de départ", value=date.today())

plan = build_virtual_pm_plan_from_optimization(
    opt_df=df_opt,
    start_date=start_dt,
    within_days=int(within),
    show_all=bool(show_all),
    only_preventive=bool(only_prev),
    min_days=1,
)

if not plan.get("ok"):
    st.error(plan.get("msg", "Erreur plan"))
    st.stop()

st.caption(f"Priorité intervalles: {', '.join(plan.get('interval_priority', []))}")
st.caption(f"Colonnes réellement utilisées: {plan.get('used_cols_stats')}")

rows_all = plan.get("rows", [])
rows_due = plan.get("due", [])

st.markdown("## 0) Commentaires maintenance (résumé)")
if rows_all:
    # Résumé par équipement (1 ligne)
    df_comm = pd.DataFrame(rows_all)[
        [c for c in ["equipment_code","maintenance_type","interval_source","interval_h","beta","eta_h","maintenance_comment"] if c in pd.DataFrame(rows_all).columns]
    ].copy()
    df_comm = df_comm.drop_duplicates(subset=["equipment_code"]).sort_values("equipment_code")
    st.dataframe(df_comm, use_container_width=True, hide_index=True)
else:
    st.info("Pas de commentaires : aucune ligne exploitable.")

st.markdown("## 1) Planning (issu de l’optimisation)")
if not rows_all:
    st.warning("Aucune ligne exploitable (intervalles manquants ou <=0).")
else:
    df_all = pd.DataFrame(rows_all)
    cols = [
        "equipment_code", "maintenance_type",
        "interval_source", "interval_h", "periodicity_days",
        "next_due_date", "days_left",
        "beta", "eta_h", "model", "distribution",
        "T_recommended_h", "T_R_h", "T_cost_h",
        "maintenance_comment",
    ]
    cols = [c for c in cols if c in df_all.columns]
    st.dataframe(df_all[cols], use_container_width=True, hide_index=True)

st.markdown("## 2) Tâches dues (dans la fenêtre)")
if not rows_due:
    st.info("Aucune tâche due dans la fenêtre. Augmente la fenêtre (ex: 60/90 jours) ou change la date de départ.")
else:
    df_due = pd.DataFrame(rows_due)
    cols = [
        "equipment_code", "maintenance_type",
        "interval_source", "interval_h",
        "next_due_date", "days_left",
        "beta", "eta_h", "model", "distribution",
        "T_recommended_h", "T_R_h", "T_cost_h",
        "maintenance_comment",
    ]
    cols = [c for c in cols if c in df_due.columns]
    st.dataframe(df_due[cols], use_container_width=True, hide_index=True)

# garder pour PDF
st.session_state["pm_virtual_all"] = rows_all
st.session_state["pm_virtual_due"] = rows_due

st.markdown("## 3) Générer le plan de maintenance (PDF)")
include_all_in_pdf = st.toggle("Inclure TOUT le planning dans le PDF (sinon seulement les tâches dues)", value=False)

if st.button("📄 Générer PDF (plan issu optimisation)", use_container_width=True):
    tasks_for_pdf = rows_all if include_all_in_pdf else rows_due

    # IMPORTANT: metrics_table = optimisation (pour fiches par équipement)
    metrics_table = df_opt.to_dict("records")

    out = export_pm_plan_with_kits_pdf(
        tasks_due=tasks_for_pdf,
        kits_by_eq={},  # pas de BD/stock ici
        metrics_table=metrics_table,
        out_dir="reports",
        title="Plan de maintenance — issu de l’optimisation",
        include_kits=False,
        tools_checklist=None,
    )
    st.success("PDF généré.")
    with open(out, "rb") as f:
        st.download_button("⬇️ Télécharger le PDF", f, file_name=out.split("/")[-1], use_container_width=True)
