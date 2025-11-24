from __future__ import annotations
from pathlib import Path
import math
import io
import numpy as np
import pandas as pd
import streamlit as st

# Toujours en tout premier
st.set_page_config(page_title="Indicateurs", page_icon="📊", layout="wide")
st.title("📊 Indicateurs — Fiabilité ")

# === Imports core (source de vérité) ===
from core.reliability.unify import compute_bundle, UnifyOptions
from core.reliability.weibull import R, F, pdf, hazard

# Export PDF (optionnel)
try:
    from core.reliability.reporting_merged import export_merged_report_pdf
except Exception:
    export_merged_report_pdf = None

# Matplotlib (back-end non interactif)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "failures_saved.csv"

# ---------- Helpers robustes ----------
def _read_csv_flex(src):
    """
    Lecture CSV robuste: essaie pandas par défaut, puis engine=python avec sep=None,
    puis point-virgule. Retourne DataFrame (évent. vide).
    """
    def _try_read(s, **kw):
        try:
            return pd.read_csv(s, **kw)
        except Exception:
            return None

    # 1) essai standard
    df = _try_read(src)
    if df is None:
        # si file-like, reset le curseur
        if hasattr(src, "seek"):
            try: src.seek(0)
            except Exception: pass
        # 2) autodétection (python engine + on_bad_lines=skip)
        df = _try_read(src, engine="python", on_bad_lines="skip", sep=None)
    if df is None:
        if hasattr(src, "seek"):
            try: src.seek(0)
            except Exception: pass
        # 3) sep=';'
        df = _try_read(src, sep=";", engine="python", on_bad_lines="skip")
    if df is None:
        return pd.DataFrame()
    # nettoyage entêtes
    df.columns = [str(c).strip() for c in df.columns]
    return df

def fnum(x, nd=2, default="—"):
    try:
        if x is None:
            return default
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return default
        return f"{float(x):.{nd}f}"
    except Exception:
        return default

def _fmt(x, fmt="{:.2f}"):
    try:
        return fmt.format(float(x))
    except Exception:
        return "—"

def _safe_dict(x):
    return x if isinstance(x, dict) else {}

def _safe_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur

def _as_dist_dict(d):
    return d if isinstance(d, dict) else ({} if d is None else {"name": str(d)})

def _pipeline_path_str(pipe: dict) -> str:
    pipe = _safe_dict(pipe)
    mk  = _safe_dict(pipe.get("trend")) or _safe_dict(pipe.get("trend_mk"))
    model = pipe.get("model") or "RP"
    dist = _as_dist_dict(pipe.get("distribution")).get("name", "Weibull2P")
    ks_p = _safe_get(pipe, ["goodness", "ks_p"])
    chi2 = _safe_get(pipe, ["goodness", "chi2_p"])
    pval = mk.get("p_value", mk.get("p"))
    return f"TTF>0 → MK(p={fnum(pval,3)}) → {model} ; Dist={dist}, KS p={fnum(ks_p,3)}, Chi2 p={fnum(chi2,3)}"

def _get_interval_opt(optim_map: dict, eq: str):
    v = (optim_map or {}).get(eq)
    if isinstance(v, dict):
        for k in ("interval_opt_h", "interval", "t_opt_h", "MTBF_opt"):
            if v.get(k) is not None:
                try:
                    return float(v[k])
                except Exception:
                    pass
        return None
    try:
        return float(v)
    except Exception:
        return None

# ---------- Chargement des TTF via session / fichier ----------
if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
    df_src = st.session_state["failures_df"].copy()
else:
    if not DATA_FILE.exists():
        st.error("Aucun fichier consolidé. Va d’abord sur « Sources de données » et enregistre.")
        st.stop()
    df_src = _read_csv_flex(DATA_FILE)

# normalisation colonnes
df_src.columns = [c.strip() for c in df_src.columns]
if "equipment_code" not in df_src.columns or "ttf_h" not in df_src.columns:
    st.error("Le jeu doit contenir: equipment_code, ttf_h.")
    st.stop()

# ---------- Bundle unifié ----------
bundle = compute_bundle(session_df=df_src, options=UnifyOptions(force_weibull_2p=True))
df_ttf   = bundle.ttf
fits_df  = bundle.fits_df
metrics  = bundle.metrics_df
pipe_by  = bundle.pipeline_by_eq
optim    = bundle.optim

if df_ttf.empty or fits_df.empty:
    st.error("Pas assez de données (≥ 3 TTF par équipement).")
    st.stop()

# Sélection équipements
eqs_all = sorted(fits_df["equipment_code"].astype(str).unique().tolist())
sel = st.multiselect("Équipements", options=eqs_all, default=eqs_all[: min(5, len(eqs_all))])
if not sel:
    st.info("Sélectionne au moins un équipement.")
    st.stop()

# ---------- Objets “ft” pour R/F/pdf/hazard ----------
class _WB:
    def __init__(self, beta, eta, gamma=0.0):
        self.beta = float(beta)
        self.eta  = float(eta)
        self.gamma = float(gamma)

fits = {
    str(r["equipment_code"]): _WB(r["beta"], r["eta"], r.get("gamma", 0.0))
    for r in fits_df[fits_df["equipment_code"].isin(sel)].to_dict("records")
}

# Domaine temporel (bornes sûres)
try:
    tmax_src = df_ttf[df_ttf["equipment_code"].isin(sel)]["ttf_h"].max()
    tmax = float(tmax_src)
    if math.isnan(tmax) or tmax <= 0:
        tmax = 1000.0
    else:
        tmax = max(1000.0, tmax)
except Exception:
    tmax = 1000.0

t = np.linspace(0, tmax, 300)

def multi_plot(ax, fun, title, ylabel):
    for eq, ft in fits.items():
        try:
            y = fun(t, ft)
            label = f"{eq} (β={_fmt(ft.beta,'{:.2f}')}, η={_fmt(ft.eta,'{:.1f}')} h)"
            ax.plot(t, y, label=label, linewidth=2)
        except Exception:
            continue
    ax.set_title(title); ax.set_xlabel("Temps (h)"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.3); ax.legend()

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
            pipe = _safe_dict(pipe_by.get(eq, {}))
            if not pipe:
                st.info("Pas de trace disponible pour cet équipement.")
            else:
                dist_val = pipe.get("distribution")
                if isinstance(dist_val, dict):
                    dist = dist_val.get("name", "Weibull2P")
                elif isinstance(dist_val, str):
                    dist = dist_val
                else:
                    dist = "Weibull2P"

                ks_p  = _safe_get(pipe, ["goodness", "ks_p"])
                chi2  = _safe_get(pipe, ["goodness", "chi2_p"])
                trend = _safe_get(pipe, ["trend", "name"]) or _safe_get(pipe, ["trend_mk", "name"]) or "MK"
                trend_p = _safe_get(pipe, ["trend", "p_value"]) or _safe_get(pipe, ["trend_mk", "p_value"])

                itv = _get_interval_opt(optim, eq)

                st.write(f"- Distribution: **{dist}** • KS p={fnum(ks_p,3)} • Chi2 p={fnum(chi2,3)}")
                st.write(f"- Test de tendance: **{trend}** • p={fnum(trend_p,3)}")
                st.write(f"- **Intervalle optimisé**: {fnum(itv,1)} h")

                st.code(_pipeline_path_str(pipe), language="text")

                with st.expander("Détails bruts (JSON)", expanded=False):
                    st.json(pipe)

st.divider()
st.subheader("📋 Tableau cohérent — MTBF/MTTR + β/η/γ + interval_opt")

dfm = bundle.metrics_df.copy()
dfm = dfm[dfm["equipment_code"].isin(sel)]

# Ajout n_ttf si absent
if "n_ttf" not in dfm.columns:
    counts = (
        df_ttf[df_ttf["equipment_code"].isin(sel)]
        .groupby("equipment_code")["ttf_h"].count()
        .rename("n_ttf")
        .reset_index()
    )
    dfm = dfm.merge(counts, on="equipment_code", how="left")

# Fallback MTBF_opt sur interval_opt_h si absent
if "MTBF_opt" in dfm.columns and "interval_opt_h" in dfm.columns:
    dfm["MTBF_opt"] = dfm["MTBF_opt"].where(~dfm["MTBF_opt"].isna(), dfm["interval_opt_h"])

wanted = ["equipment_code","n_ttf","MTBF","MTTR","MTBF_opt","MTTR_opt","beta","eta","gamma","interval_opt_h"]
cols = [c for c in wanted if c in dfm.columns] + [c for c in dfm.columns if c not in wanted]
st.dataframe(dfm[cols].sort_values("equipment_code"), use_container_width=True, hide_index=True)

# ---------- Export rapport complet ----------
st.divider()
st.subheader("📄 Rapport complet (analyse + optimisation + organigramme + courbes)")
if export_merged_report_pdf is None:
    st.info("Module `core.reliability.reporting_merged` non détecté.")
else:
    df_sel = df_src[df_src["equipment_code"].isin(sel)].copy()
    if st.button("📄 Générer rapport complet"):
        try:
            path = export_merged_report_pdf(
                df=df_sel,
                out_dir=str(BASE_DIR / "reports"),
                title="Rapport complet — Analyse & Optimisation",
                options=UnifyOptions(force_weibull_2p=True, R_target=0.80),
            )
            st.success(f"PDF généré : {path}")
        except Exception as e:
            st.error(f"PDF : {e}")
