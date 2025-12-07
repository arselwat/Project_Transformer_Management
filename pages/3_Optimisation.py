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
try:
    from core.reliability.optimize import propose_intervals
except Exception:
    propose_intervals = None
from __future__ import annotations
import streamlit as st
from core.security.auth import require_login

st.set_page_config(page_title="Transformateurs", page_icon="🔌", layout="wide")

require_login()  # tant que auth_ok n’est pas True, cette page est bloquée

# ... le reste de ta page ...

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
            try: src.seek(0)
            except Exception: pass
        df = _try_read(src, engine="python", on_bad_lines="skip", sep=None)
    if df is None:
        if hasattr(src, "seek"):
            try: src.seek(0)
            except Exception: pass
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

# -------- Fit Weibull --------
fits = {}
for eq in eqs:
    x = df.loc[df["equipment_code"]==eq, "ttf_h"].values
    if len(x) >= 3:
        try:
            fits[eq] = fit_weibull(x)
        except Exception:
            pass

if not fits:
    st.error("Pas assez de TTF (≥3) pour estimer Weibull.")
    st.stop()

# -------- Intervalles (robuste aux formats) --------
def analytic_interval(beta, eta, R_target):
    return float(eta * ((-math.log(R_target)) ** (1.0/ beta))) if (beta>0 and eta>0 and 0<R_target<1) else float("nan")

def normalize_intervals(raw, fits, R_target):
    out = {}
    if raw is None:
        for eq, ft in fits.items():
            out[eq] = analytic_interval(float(ft.beta), float(ft.eta), R_target)
        return out
    if isinstance(raw, (int,float)):
        for eq in fits.keys():
            out[eq] = float(raw)
        return out
    if isinstance(raw, dict):
        for eq, ft in fits.items():
            v = raw.get(eq)
            if isinstance(v, dict):
                val = v.get("interval_opt_h") or v.get("interval_h") or v.get("interval")
                out[eq] = float(val) if isinstance(val,(int,float)) else analytic_interval(float(ft.beta), float(ft.eta), R_target)
            elif isinstance(v,(int,float)):
                out[eq] = float(v)
            else:
                out[eq] = analytic_interval(float(ft.beta), float(ft.eta), R_target)
        return out
    for eq, ft in fits.items():
        out[eq] = analytic_interval(float(ft.beta), float(ft.eta), R_target)
    return out

R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
raw = None
if callable(propose_intervals):
    try:
        raw = propose_intervals(fits, R_target=R_target)
    except Exception:
        raw = None
intervals = normalize_intervals(raw, fits, R_target)

# -------- Organigramme par équipement --------
org_results = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"]==eq, "ttf_h"].tolist()
    org_results[eq] = analyze_ttf_pipeline(ttf)

# -------- Tableau synthèse + CSV --------
rows = []
for eq, ft in fits.items():
    beta = float(ft.beta); eta = float(ft.eta)
    itv  = intervals.get(eq)
    rows.append({
        "equipment_code": eq,
        "beta": round(beta,3),
        "eta_h": round(eta,1),
        "interval_opt_h": round(float(itv),1) if isinstance(itv,(int,float)) else None,
        "model": org_results.get(eq,{}).get("model","?"),
        "distribution": org_results.get(eq,{}).get("distribution","?")
    })
df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

st.subheader("📋 Synthèse optimisation")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger CSV optimisé", data=csv_bytes, file_name="optimisation_intervalles.csv", mime="text/csv")

# -------- Courbes R(t) --------
st.subheader("📈 Courbes R(t)")
tmax = max([float(getattr(ft, "eta", 1.0)) for ft in fits.values()]) * 1.2
t = np.linspace(0, max(tmax, 1.0), 300)
fig, ax = plt.subplots()
for eq, ft in fits.items():
    beta = float(ft.beta); eta = float(ft.eta)
    y = np.exp(-((t/eta) ** beta))
    ax.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f})")
ax.grid(True, alpha=.3); ax.set_xlabel("Temps (h)"); ax.set_ylabel("R(t)")
ax.set_title("Fiabilité R(t)"); ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

# -------- Détails & actions --------
st.subheader("🔎 Détails par équipement")
sel = st.selectbox("Équipement", options=[r["equipment_code"] for r in rows])
ft  = fits[sel]
st.write(f"β={float(ft.beta):.3f} • η={float(ft.eta):.1f} h • Intervalle: {intervals.get(sel):.1f} h")
st.markdown("**Actions suggérées (selon β)** :")
for a in suggested_actions(float(ft.beta)):
    st.markdown(f"- {a}")

# -------- Export PDF --------
st.divider()
if st.button("📄 Générer rapport optimisation (PDF)"):
    try:
        path = export_optimization_report_pdf(df, fits, intervals, org_results, out_dir="reports")
        st.success(f"PDF généré : {path}")
    except Exception as e:
        st.error(f"PDF : {e}")
