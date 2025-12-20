# pages/3_Optimisation.py (ou ton nom réel)
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from core.security.auth import require_login
from core.datahub import get_current_failures_df, set_current_failures_df, get_failures_meta

from core.reliability.weibull import fit_weibull
from core.reliability.policy import suggested_actions
from core.reliability.organigram import analyze_ttf_pipeline
from core.reliability.optimize import propose_intervals_cost_and_reliability

export_optimization_report_pdf = None
_pdf_import_error = None
try:
    from core.reliability.reporting_optimize import export_optimization_report_pdf as _export_opt_pdf
    export_optimization_report_pdf = _export_opt_pdf
except Exception as e:
    _pdf_import_error = e
    export_optimization_report_pdf = None

st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()
st.title("🧠 Optimisation — Intervalles, coût & fiabilité")

BASE_DIR = Path(__file__).resolve().parents[1]

def fnum(x, nd=2, default="—"):
    try:
        if x is None:
            return default
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return f"{x:.{nd}f}"
    except Exception:
        return default

def is_pos_number(x) -> bool:
    return isinstance(x, (int, float)) and float(x) > 0 and np.isfinite(float(x))

# -------------------------
# Dataset officiel + option upload (synchronise)
# -------------------------
meta0 = get_failures_meta()
df0 = get_current_failures_df()

st.info(f"Dataset projet | rows={meta0.get('rows')} | hash={meta0.get('hash')} | source={meta0.get('source')}")

with st.expander("📥 Option : remplacer le dataset du projet avec un upload (synchronise tout)", expanded=False):
    up = st.file_uploader("CSV (equipment_code, ttf_h)", type=["csv"])
    if up is not None and st.button("✅ Synchroniser ce CSV comme dataset officiel", type="primary"):
        try:
            df_up = pd.read_csv(up)
            res = set_current_failures_df(df_up, source_name=f"optimization_upload:{up.name}", persist=True)
            if res.get("ok"):
                st.success(f"Dataset remplacé ✅ | rows={res['rows']} | hash={res['hash']}")
                st.rerun()
            else:
                st.error(res.get("msg"))
        except Exception as e:
            st.error(f"Upload: {e}")

df = get_current_failures_df()
meta = get_failures_meta()

if df.empty:
    st.error("Aucun dataset actif. Va sur « Sources de données » et synchronise un CSV.")
    st.stop()

st.success(f"Dataset actif ✅ | rows={meta['rows']} | hash={meta['hash']} | source={meta['source']}")

# -------------------------
# Fit Weibull
# -------------------------
eqs = sorted(df["equipment_code"].unique().tolist())
fits = {}
for eq in eqs:
    x = df.loc[df["equipment_code"] == eq, "ttf_h"].values
    if len(x) >= 3:
        try:
            fits[eq] = fit_weibull(x)
        except Exception:
            pass

if not fits:
    st.error("Pas assez de TTF (≥3) pour estimer Weibull.")
    st.stop()

# -------------------------
# Paramètres
# -------------------------
st.markdown("### Paramètres de fiabilité et de coût")

colR, colC1, colC2, colRmin = st.columns(4)
with colR:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
with colC1:
    C_prev = st.number_input("Coût préventif (C_prev)", min_value=0.0, value=1.0, step=0.1)
with colC2:
    C_corr = st.number_input("Coût panne (C_corr)", min_value=0.0, value=5.0, step=0.5)
with colRmin:
    R_min_cost = st.slider("Fiabilité min. pour optimum coût", 0.0, 0.99, 0.70, 0.01)

econ_enabled = (C_prev > 0) and (C_corr > 0)
if not econ_enabled:
    st.warning("Renseigne C_prev > 0 et C_corr > 0 pour activer T_cost.")

# -------------------------
# Organigramme
# -------------------------
org_results: dict[str, dict] = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    try:
        org_results[eq] = analyze_ttf_pipeline(ttf)
    except Exception:
        org_results[eq] = {}

# -------------------------
# Intervalles
# -------------------------
res_all = {}
if econ_enabled:
    res_all = propose_intervals_cost_and_reliability(
        fits=fits,
        C_prev=float(C_prev),
        C_corr=float(C_corr),
        R_target=float(R_target),
        R_min_cost=float(R_min_cost),
    )

intervals_R = {eq: (res_all.get(eq) or {}).get("T_R") for eq in fits.keys()}
intervals_cost = {eq: (res_all.get(eq) or {}).get("T_cost") for eq in fits.keys()}
R_at_cost = {eq: (res_all.get(eq) or {}).get("R_at_T") for eq in fits.keys()}
C_min_map = {eq: (res_all.get(eq) or {}).get("C_min") for eq in fits.keys()}

def recommend_maintenance(beta: float, model: str | None = None) -> str:
    if beta < 0.9:
        return "Corrective + fiabilisation"
    if 0.9 <= beta <= 1.1:
        return "Conditionnelle / inspection"
    if model and "NHPP" in str(model).upper():
        return "Préventive planifiée"
    return "Préventive planifiée"

def recommend_interval(beta: float, T_cost: float | None, T_R: float | None) -> float | None:
    if beta <= 1.1:
        return None
    vals = [v for v in [T_cost, T_R] if is_pos_number(v)]
    return float(min(vals)) if vals else None

# -------------------------
# Tableau synthèse
# -------------------------
rows = []
for eq, ft in fits.items():
    org = org_results.get(eq, {}) or {}
    beta = float(getattr(ft, "beta", float("nan")))
    eta = float(getattr(ft, "eta", float("nan")))
    gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

    itv_R = intervals_R.get(eq)
    itv_C = intervals_cost.get(eq)

    maint_type = recommend_maintenance(beta, org.get("model"))
    T_rec = recommend_interval(beta, itv_C, itv_R)

    rows.append({
        "equipment_code": eq,
        "beta": round(beta, 3) if np.isfinite(beta) else None,
        "eta_h": round(eta, 1) if np.isfinite(eta) else None,
        "gamma_h": round(gamma, 1) if np.isfinite(gamma) else 0.0,
        "T_cost_h": round(float(itv_C), 1) if is_pos_number(itv_C) else None,
        "T_R_h": round(float(itv_R), 1) if is_pos_number(itv_R) else None,
        "T_recommended_h": round(float(T_rec), 1) if is_pos_number(T_rec) else None,
        "maintenance_type": maint_type,
        "model": org.get("model", "?"),
        "distribution": org.get("distribution", "?"),
    })

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

st.subheader("📋 Synthèse optimisation (à synchroniser vers Maintenance)")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger CSV optimisé",
    data=csv_bytes,
    file_name="optimisation_intervalles.csv",
    mime="text/csv",
    use_container_width=True,
)

# -------------------------
# Passerelle vers Maintenance (comme tu l’avais)
# -------------------------
st.divider()
st.subheader("🔁 Passerelle → Maintenance (création/MAJ pm_task)")

from core.maintenance.bridge import upsert_tasks_from_optimization, BridgeParams

col1, col2, col3 = st.columns(3)
with col1:
    only_prev = st.toggle("Créer tâches uniquement si Préventive", value=True)
with col2:
    min_days = st.number_input("Périodicité minimale (jours)", min_value=1, value=7, step=1)
with col3:
    start_dt = st.date_input("Date de départ planning", value=None)

if st.button("✅ Synchroniser les tâches vers Maintenance", type="primary"):
    cfg = BridgeParams(min_days=int(min_days), only_if_preventive=bool(only_prev))
    res = upsert_tasks_from_optimization(
        opt_df=df_out,
        start_date=str(start_dt) if start_dt else None,
        params=cfg,
    )
    if res.get("ok"):
        st.success(f"Tâches synchronisées ✅ | créées={res['created']} | MAJ={res['updated']} | ignorées={res['skipped']}")
    else:
        st.warning(f"Synchronisation partielle | créées={res['created']} | MAJ={res['updated']} | ignorées={res['skipped']}")
        if res.get("errors"):
            st.error("Erreurs: " + " | ".join(res["errors"]))

# -------------------------
# Courbe R(t) (simple)
# -------------------------
st.divider()
st.subheader("📈 Courbes R(t) (Weibull)")

etas = [float(getattr(ft, "eta", 1.0) or 1.0) for ft in fits.values()]
tmax = max(etas) * 1.6 if etas else 1000.0
t = np.linspace(0, max(tmax, 1.0), 350)

fig, ax = plt.subplots()
for eq, ft in fits.items():
    beta = float(getattr(ft, "beta", 1.0))
    eta = float(getattr(ft, "eta", 1.0))
    gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

    y = np.ones_like(t, dtype=float)
    mask = t > gamma
    y[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-9)) ** max(beta, 1e-9)))
    ax.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f})")

ax.grid(True, alpha=.3)
ax.set_xlabel("Temps (h)")
ax.set_ylabel("R(t)")
ax.set_title("Fiabilité R(t)")
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)
