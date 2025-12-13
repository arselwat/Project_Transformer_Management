from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.reliability.weibull import fit_weibull
from core.reliability.policy import suggested_actions
from core.reliability.organigram import analyze_ttf_pipeline
from core.reliability.reporting_optimize import export_optimization_report_pdf

# nouvelles fonctions que tu ajoutes dans core/reliability/optimize.py
try:
    from core.reliability.optimize import (
        propose_intervals_from_models,   # multi-modèles (Weibull, expo, …)
        optimize_interval_cost_weibull,  # option coût minimal
    )
except Exception:
    propose_intervals_from_models = None
    optimize_interval_cost_weibull = None

from core.security.auth import require_login

st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()

st.title("🧠 Optimisation — Intervalles, courbes & organigramme")

DATA_FILE = Path("data/failures_saved.csv")

# --- lecteur CSV robuste (fichier ou UploadedFile) ---
def _read_csv_flex(src):
    def _try_read(s, **kw):
        try:
            return pd.read_csv(s, **kw)
        except Exception:
            return None

    df = _try_read(src)
    if df is None:
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        df = _try_read(src, engine="python", on_bad_lines="skip", sep=None)
    if df is None:
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        df = _try_read(src, sep=";", engine="python", on_bad_lines="skip")
    if df is None:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------- Chargement données --------
src = st.radio("Source TTF", ["Fichier projet", "Uploader CSV"], horizontal=True)
if src == "Fichier projet":
    if not DATA_FILE.exists():
        st.error("Aucun data/failures_saved.csv — va dans « Sources de données » pour enregistrer.")
        st.stop()
    df = _read_csv_flex(DATA_FILE)
else:
    up = st.file_uploader("CSV (equipment_code, ttf_h)", type=["csv"])
    if up is None:
        st.stop()
    df = _read_csv_flex(up)

if "equipment_code" not in df.columns or "ttf_h" not in df.columns:
    st.error("Colonnes requises: equipment_code, ttf_h")
    st.stop()

df["equipment_code"] = df["equipment_code"].astype(str)
df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
df = df.dropna(subset=["ttf_h"])
df = df[df["ttf_h"] > 0]

eqs = sorted(df["equipment_code"].unique().tolist())
if not eqs:
    st.error("Aucun équipement valide.")
    st.stop()

# -------- Fit Weibull (baseline) --------
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

# -------- Utilitaires d’intervalle --------
def analytic_interval(beta: float, eta: float, R_target: float) -> float:
    if not (beta > 0 and eta > 0 and 0.0 < R_target < 1.0):
        return float("nan")
    return float(eta * ((-math.log(R_target)) ** (1.0 / beta)))

def normalize_intervals_weibull_only(fits: dict, R_target: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for eq, ft in fits.items():
        out[eq] = analytic_interval(float(ft.beta), float(ft.eta), R_target)
    return out

# -------- Paramètres utilisateur --------
st.markdown("### Paramètres de fiabilité et de coût")

colR, colC1, colC2 = st.columns(3)
with colR:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)

with colC1:
    C_prev = st.number_input("Coût maintenance préventive", min_value=0.0, value=1.0, step=0.1)

with colC2:
    C_corr = st.number_input("Coût panne (corrective)", min_value=0.0, value=5.0, step=0.5)

use_cost = st.checkbox("Calculer aussi l’intervalle à coût moyen minimal (Weibull)", value=False)

# -------- Organigramme + modèle global --------
org_results: dict[str, dict] = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    org_results[eq] = analyze_ttf_pipeline(ttf)

# Construire un dictionnaire “best_fits” utilisable par propose_intervals_from_models
best_fits: dict[str, dict] = {}
for eq, org in org_results.items():
    if "distribution" in org and org.get("distribution") is not None:
        best_fits[eq] = {
            "ok": True,
            "best_name": org.get("distribution"),
            "params": {},  # à remplir si tu exposes les params par loi dans ton pipeline
        }

# -------- Intervalles par fiabilité cible --------
intervals_R: dict[str, float] = {}

if callable(propose_intervals_from_models) and best_fits:
    try:
        raw = propose_intervals_from_models(best_fits, R_target=R_target)
        for eq, v in raw.items():
            val = v.get("interval_h")
            if isinstance(val, (int, float)) and val > 0:
                intervals_R[eq] = float(val)
    except Exception:
        intervals_R = {}

# fallback : si on n’a pas réussi, on revient à la formule Weibull seule
if not intervals_R:
    intervals_R = normalize_intervals_weibull_only(fits, R_target)

# -------- Intervalles à coût minimal (optionnel, Weibull) --------
intervals_cost: dict[str, float] = {}
if use_cost and callable(optimize_interval_cost_weibull) and C_prev > 0 and C_corr > 0:
    for eq, ft in fits.items():
        T_cost = optimize_interval_cost_weibull(
            beta=float(ft.beta),
            eta=float(ft.eta),
            gamma=0.0,
            C_prev=C_prev,
            C_corr=C_corr,
        )
        if isinstance(T_cost, (int, float)) and T_cost > 0:
            intervals_cost[eq] = float(T_cost)

# -------- Tableau synthèse + CSV --------
rows = []
for eq, ft in fits.items():
    org = org_results.get(eq, {})
    beta = float(ft.beta)
    eta = float(ft.eta)
    itv_R = intervals_R.get(eq)
    itv_C = intervals_cost.get(eq)

    rows.append({
        "equipment_code": eq,
        "beta": round(beta, 3),
        "eta_h": round(eta, 1),
        "interval_R_h": round(float(itv_R), 1) if isinstance(itv_R, (int, float)) else None,
        "interval_cost_h": round(float(itv_C), 1) if isinstance(itv_C, (int, float)) else None,
        "model": org.get("model", "?"),
        "distribution": org.get("distribution", "?"),
        "trend": bool(org.get("trend_mk", {}).get("has_trend", False)),
        "dep": bool(org.get("dependence", {}).get("has_dep", False)),
    })

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

st.subheader("📋 Synthèse optimisation")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger CSV optimisé",
    data=csv_bytes,
    file_name="optimisation_intervalles.csv",
    mime="text/csv",
)

# -------- Courbes R(t) (Weibull baseline) --------
st.subheader("📈 Courbes R(t) (Weibull)")
tmax = max([float(getattr(ft, "eta", 1.0)) for ft in fits.values()]) * 1.2
t = np.linspace(0, max(tmax, 1.0), 300)
fig, ax = plt.subplots()
for eq, ft in fits.items():
    beta = float(ft.beta)
    eta = float(ft.eta)
    y = np.exp(-((t / eta) ** beta))
    ax.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f})")
ax.grid(True, alpha=.3)
ax.set_xlabel("Temps (h)")
ax.set_ylabel("R(t)")
ax.set_title("Fiabilité R(t)")
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

# -------- Détails & actions --------
st.subheader("🔎 Détails par équipement")

sel = st.selectbox("Équipement", options=[r["equipment_code"] for r in rows])
ft = fits[sel]
beta = float(ft.beta)
eta = float(ft.eta)
itv_R = intervals_R.get(sel)
itv_C = intervals_cost.get(sel)

st.write(
    f"β = {beta:.3f} • η = {eta:.1f} h • "
    f"Intervalle fiabilité: {itv_R:.1f} h"
    + (f" • Intervalle coût: {itv_C:.1f} h" if isinstance(itv_C, (int, float)) else "")
)

org = org_results.get(sel, {})
st.write(
    f"Modèle global: {org.get('model','?')} • Distribution: {org.get('distribution','?')}"
)

st.markdown("**Actions suggérées (selon β)** :")
for a in suggested_actions(beta):
    st.markdown(f"- {a}")

# -------- Export PDF --------
st.divider()
if st.button("📄 Générer rapport optimisation (PDF)"):
    try:
        # tu peux faire évoluer la signature pour passer aussi intervals_cost, org_results, etc.
        path = export_optimization_report_pdf(df, fits, intervals_R, org_results, out_dir="reports")
        st.success(f"PDF généré : {path}")
    except Exception as e:
        st.error(f"PDF : {e}")
