from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from core.security.auth import require_login
from core.reliability.weibull import R, F, pdf, hazard, fit_weibull
from core.reliability.organigram import analyze_ttf_pipeline

# Export PDF (optionnel)
try:
    from core.reliability.reporting_merged import export_merged_report_pdf
except Exception:
    export_merged_report_pdf = None

st.set_page_config(page_title="Indicateurs", page_icon="📊", layout="wide")
require_login()

st.title("📊 Indicateurs — Fiabilité")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "failures_saved.csv"


# ---------- Helpers ----------
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

def _pipeline_str(pipe: dict) -> str:
    model = pipe.get("model", "RP")
    dist = pipe.get("distribution", "weibull_2p")
    mk = (pipe.get("tests", {}) or {}).get("trend_mk", {})
    dep = (pipe.get("tests", {}) or {}).get("dependence", {})
    good = pipe.get("goodness", {}) or {}

    return (
        f"TTF>0 → MK(p={fnum(mk.get('p'),3)}, dir={mk.get('direction','none')}) "
        f"→ Dep(r={fnum(dep.get('r'),3)}, p={fnum(dep.get('p'),3)}) "
        f"→ Model={model} ; Dist={dist} ; KS p={fnum(good.get('ks_p'),3)} ; Chi2 p={fnum(good.get('chi2_p'),3)}"
    )


# ---------- Chargement ----------
if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
    df_src = st.session_state["failures_df"].copy()
else:
    if not DATA_FILE.exists():
        st.error("Aucun fichier consolidé. Va d’abord sur « Sources de données » et enregistre.")
        st.stop()
    df_src = _read_csv_flex(DATA_FILE)

df_src.columns = [c.strip() for c in df_src.columns]
if "equipment_code" not in df_src.columns or "ttf_h" not in df_src.columns:
    st.error("Le jeu doit contenir: equipment_code, ttf_h.")
    st.stop()

df_src["equipment_code"] = df_src["equipment_code"].astype(str)
df_src["ttf_h"] = pd.to_numeric(df_src["ttf_h"], errors="coerce")
df_src = df_src.dropna(subset=["ttf_h"])
df_src = df_src[df_src["ttf_h"] > 0]

if df_src.empty:
    st.error("Pas de TTF valides (>0).")
    st.stop()

eqs_all = sorted(df_src["equipment_code"].unique().tolist())
sel = st.multiselect("Équipements", options=eqs_all, default=eqs_all[: min(5, len(eqs_all))])
if not sel:
    st.info("Sélectionne au moins un équipement.")
    st.stop()


# ---------- Fit + Pipeline ----------
class _WB:
    def __init__(self, beta, eta, gamma=0.0):
        self.beta = float(beta)
        self.eta = float(eta)
        self.gamma = float(gamma or 0.0)

fits: dict[str, _WB] = {}
pipe_by: dict[str, dict] = {}
metrics_rows: list[dict] = []

for eq in sel:
    ttfs = df_src.loc[df_src["equipment_code"] == eq, "ttf_h"].values
    if len(ttfs) < 3:
        continue
    try:
        wb = fit_weibull(ttfs)
        fits[eq] = _WB(wb.beta, wb.eta, getattr(wb, "gamma", 0.0))

        pipe = analyze_ttf_pipeline(ttfs.tolist())
        pipe_by[eq] = pipe

        mtbf = float(np.mean(ttfs))
        metrics_rows.append({
            "equipment_code": eq,
            "n_ttf": int(len(ttfs)),
            "MTBF": mtbf,
            "beta": float(fits[eq].beta),
            "eta": float(fits[eq].eta),
            "gamma": float(fits[eq].gamma),
            "model": pipe.get("model", "?"),
            "distribution": pipe.get("distribution", "?"),
            "ks_p": (pipe.get("goodness", {}) or {}).get("ks_p"),
            "chi2_p": (pipe.get("goodness", {}) or {}).get("chi2_p"),
        })
    except Exception:
        continue

if not fits:
    st.error("Pas assez de TTF (≥3) pour les équipements sélectionnés.")
    st.stop()


# ---------- Domaine temps ----------
tmax = float(df_src[df_src["equipment_code"].isin(sel)]["ttf_h"].max())
if not np.isfinite(tmax) or tmax <= 0:
    tmax = 1000.0
tmax = max(1000.0, tmax)
t = np.linspace(0, tmax, 300)


def multi_plot(ax, fun, title, ylabel):
    for eq, ft in fits.items():
        try:
            y = fun(t, ft)
            ax.plot(t, y, label=f"{eq} (β={ft.beta:.2f}, η={ft.eta:.1f}h)", linewidth=2)
        except Exception:
            continue
    ax.set_title(title)
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ---------- Graphiques ----------
tabR, tabF, tabf, tabh, tabG = st.tabs(["R(t)", "F(t)", "f(t)", "h(t)", "🧭 Organigramme"])

with tabR:
    fig, ax = plt.subplots()
    multi_plot(ax, R, "Fiabilité R(t)", "R(t)")
    st.pyplot(fig, clear_figure=True)

with tabF:
    fig, ax = plt.subplots()
    multi_plot(ax, F, "Répartition F(t)", "F(t)")
    st.pyplot(fig, clear_figure=True)

with tabf:
    fig, ax = plt.subplots()
    multi_plot(ax, pdf, "Densité f(t)", "f(t)")
    st.pyplot(fig, clear_figure=True)

with tabh:
    fig, ax = plt.subplots()
    multi_plot(ax, hazard, "Taux de défaillance h(t)", "h(t)")
    st.pyplot(fig, clear_figure=True)

with tabG:
    for eq in sel:
        with st.expander(f"Trace organigramme — {eq}", expanded=False):
            pipe = pipe_by.get(eq, {})
            if not pipe:
                st.info("Pas de trace disponible pour cet équipement.")
                continue

            st.write(f"- Modèle: **{pipe.get('model','?')}**")
            st.write(f"- Distribution: **{pipe.get('distribution','?')}**")

            mk = (pipe.get("tests", {}) or {}).get("trend_mk", {})
            dep = (pipe.get("tests", {}) or {}).get("dependence", {})
            good = pipe.get("goodness", {}) or {}
            prm = pipe.get("params", {}) or {}

            st.write(f"- MK: p={fnum(mk.get('p'),3)} • direction={mk.get('direction','none')}")
            st.write(f"- Dépendance: r={fnum(dep.get('r'),3)} • p={fnum(dep.get('p'),3)} • méthode={dep.get('method','?')}")
            st.write(f"- Goodness: KS p={fnum(good.get('ks_p'),3)} • Chi2 p={fnum(good.get('chi2_p'),3)} • AIC={fnum(good.get('aic'),2)}")

            if prm.get("beta") is not None:
                st.write(f"- Weibull: β={fnum(prm.get('beta'),3)} • η={fnum(prm.get('eta'),1)} h • γ={fnum(prm.get('gamma'),1)}")

            st.code(_pipeline_str(pipe), language="text")

            with st.expander("Détails bruts (JSON)", expanded=False):
                st.json(pipe)


# ---------- Tableau synthèse ----------
st.divider()
st.subheader("📋 Tableau synthèse MTBF + β/η/γ (+ modèle/loi)")

dfm = pd.DataFrame(metrics_rows).sort_values("equipment_code").reset_index(drop=True)
st.dataframe(dfm, use_container_width=True, hide_index=True)


# ---------- Export ----------
st.divider()
st.subheader("📄 Rapport complet (analyse + indicateurs + courbes)")

if export_merged_report_pdf is None:
    st.info("Module `core.reliability.reporting_merged` non détecté.")
else:
    df_sel = df_src[df_src["equipment_code"].isin(sel)].copy()
    if st.button("📄 Générer rapport complet"):
        try:
            path = export_merged_report_pdf(
                df=df_sel,
                out_dir=str(BASE_DIR / "reports"),
                title="Rapport complet — Analyse & Indicateurs",
            )
            st.success(f"PDF généré : {path}")
        except Exception as e:
            st.error(f"PDF : {e}")
