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

from core.reliability.optimize import (
    propose_intervals_cost_and_reliability,
)

from core.security.auth import require_login

st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()

st.title("🧠 Optimisation — Intervalles, coût & fiabilité")

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

# -------- Paramètres utilisateur --------
st.markdown("### Paramètres de fiabilité et de coût")

colR, colC1, colC2, colRmin = st.columns(4)
with colR:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
with colC1:
    C_prev = st.number_input("Coût maintenance préventive", min_value=0.0, value=1.0, step=0.1)
with colC2:
    C_corr = st.number_input("Coût panne (corrective)", min_value=0.0, value=5.0, step=0.5)
with colRmin:
    R_min_cost = st.slider("Fiabilité min. pour l’optimum coût", 0.0, 0.99, 0.70, 0.01)

if C_prev <= 0 or C_corr <= 0:
    st.warning("Renseigne des coûts préventif/correctif > 0 pour l’optimisation économique.")

# -------- Organigramme + modèle global --------
org_results: dict[str, dict] = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    org_results[eq] = analyze_ttf_pipeline(ttf)

# -------- Calcul des intervalles coût + fiabilité --------
res_all = {}
if C_prev > 0 and C_corr > 0:
    res_all = propose_intervals_cost_and_reliability(
        fits=fits,
        C_prev=C_prev,
        C_corr=C_corr,
        R_target=R_target,
        R_min_cost=R_min_cost,
    )

intervals_R = {eq: d.get("T_R") for eq, d in res_all.items()}
intervals_cost = {eq: d.get("T_cost") for eq, d in res_all.items()}
R_at_cost = {eq: d.get("R_at_T") for eq, d in res_all.items()}
C_min_map = {eq: d.get("C_min") for eq, d in res_all.items()}

# -------- Tableau synthèse + CSV --------
rows = []
for eq, ft in fits.items():
    org = org_results.get(eq, {})
    beta = float(ft.beta)
    eta = float(ft.eta)

    itv_R = intervals_R.get(eq)
    itv_C = intervals_cost.get(eq)
    R_cost = R_at_cost.get(eq)
    C_min = C_min_map.get(eq)

    rows.append({
        "equipment_code": eq,
        "beta": round(beta, 3),
        "eta_h": round(eta, 1),
        "T_cost_h": round(float(itv_C), 1) if isinstance(itv_C, (int, float)) else None,
        "R(T_cost)": round(float(R_cost), 3) if isinstance(R_cost, (int, float)) else None,
        "C_min/heure": round(float(C_min), 4) if isinstance(C_min, (int, float)) else None,
        "T_R_h": round(float(itv_R), 1) if isinstance(itv_R, (int, float)) else None,
        "model": org.get("model", "?"),
        "distribution": org.get("distribution", "?"),
    })

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

st.subheader("📋 Synthèse optimisation (coût puis fiabilité)")
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
R_cost = R_at_cost.get(sel)
C_min = C_min_map.get(sel)

st.markdown(
    f"- β = **{beta:.3f}** • η = **{eta:.1f} h**\n"
    f"- Intervalle **coût minimal** T_cost : **{itv_C:.1f} h** "
    f"(R(T_cost) ≈ {R_cost:.3f}, coût moyen ≈ {C_min:.4f} / h)\n"
    f"- Intervalle **fiabilité cible** T_R : **{itv_R:.1f} h** (R_target = {R_target:.2f})"
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
        path = export_optimization_report_pdf(df, fits, intervals_R, org_results, out_dir="reports")
        st.success(f"PDF généré : {path}")
    except Exception as e:
        st.error(f"PDF : {e}")
