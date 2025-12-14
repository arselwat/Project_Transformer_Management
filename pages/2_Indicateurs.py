from __future__ import annotations

from pathlib import Path
import math
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from core.security.auth import require_login

# === Config page (une seule fois, tout en haut) ===
st.set_page_config(page_title="Indicateurs", page_icon="📊", layout="wide")
st.title("📊 Indicateurs — Fiabilité ")

# === Imports fiabilité (sans unify) ===
try:
    from core.reliability.weibull import R, F, pdf, hazard
    from core.reliability.organigram import analyze_ttf_pipeline
except ImportError:
    st.error(
        "Modules de fiabilité introuvables (`core.reliability.weibull` ou `organigram`). "
        "Vérifie que le dossier `core/` et `core/reliability/` sont bien présents "
        "dans l'environnement Streamlit."
    )
    st.stop()

# Export PDF (optionnel)
try:
    from core.reliability.reporting_merged import export_merged_report_pdf
except Exception:
    export_merged_report_pdf = None

# --- Auth obligatoire ---
require_login()

# === Constantes ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "failures_saved.csv"

# ---------- Helpers robustes ----------
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
    mk = _safe_dict(pipe.get("trend")) or _safe_dict(pipe.get("trend_mk"))
    model = pipe.get("model") or "RP"
    dist = _as_dist_dict(pipe.get("distribution")).get("name", "Weibull2P")
    ks_p = _safe_get(pipe, ["goodness", "ks_p"])
    chi2 = _safe_get(pipe, ["goodness", "chi2_p"])
    pval = mk.get("p_value", mk.get("p"))
    return (
        f"TTF>0 → MK(p={fnum(pval,3)}) → {model} ; "
        f"Dist={dist}, KS p={fnum(ks_p,3)}, Chi2 p={fnum(chi2,3)}"
    )

# ---------- Chargement des TTF via session / fichier ----------
if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
    df_src = st.session_state["failures_df"].copy()
else:
    if not DATA_FILE.exists():
        st.error(
            "Aucun fichier consolidé. "
            "Va d’abord sur « Sources de données » et enregistre."
        )
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

# ---------- Liste équipements & sélection ----------
eqs_all = sorted(df_src["equipment_code"].unique().tolist())
sel = st.multiselect(
    "Équipements",
    options=eqs_all,
    default=eqs_all[: min(5, len(eqs_all))]
)
if not sel:
    st.info("Sélectionne au moins un équipement.")
    st.stop()

# ---------- Fit Weibull simple pour indicateurs ----------
class _WB:
    def __init__(self, beta, eta, gamma=0.0):
        self.beta = float(beta)
        self.eta = float(eta)
        self.gamma = float(gamma)

from core.reliability.weibull import fit_weibull

fits: dict[str, _WB] = {}
pipe_by: dict[str, dict] = {}
metrics_rows: list[dict] = []

for eq in sel:
    ttfs = df_src.loc[df_src["equipment_code"] == eq, "ttf_h"].values
    if len(ttfs) < 3:
        continue
    try:
        wb = fit_weibull(ttfs)
        ft = _WB(wb.beta, wb.eta, getattr(wb, "gamma", 0.0))
        fits[eq] = ft

        # Organigramme complet (pipeline)
        pipe = analyze_ttf_pipeline(ttfs.tolist())
        pipe_by[eq] = pipe

        # Quelques métriques de base
        mtbf = float(np.mean(ttfs))
        metrics_rows.append({
            "equipment_code": eq,
            "n_ttf": len(ttfs),
            "MTBF": mtbf,
            "beta": float(ft.beta),
            "eta": float(ft.eta),
            "gamma": float(ft.gamma),
        })
    except Exception:
        continue

if not fits:
    st.error("Pas assez de TTF (≥3) pour les équipements sélectionnés.")
    st.stop()

# ---------- Domaine temporel ----------
try:
    tmax_src = df_src[df_src["equipment_code"].isin(sel)]["ttf_h"].max()
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
    ax.set_title(title)
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

# ---------- Graphiques ----------
tabR, tabF, tabf, tabh, tabG = st.tabs(
    ["R(t)", "F(t)", "f(t)", "h(t)", "🧭 Organigramme"]
)

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

                ks_p = _safe_get(pipe, ["distribution_full", "ks_p"])
                chi2 = _safe_get(pipe, ["distribution_full", "chi2_p"])
                trend = (
                    _safe_get(pipe, ["trend_mk", "name"])
                    or "MK"
                )
                trend_p = _safe_get(pipe, ["trend_mk", "p"])

                st.write(
                    f"- Distribution: **{dist}** • KS p={fnum(ks_p,3)} "
                    f"• Chi2 p={fnum(chi2,3)}"
                )
                st.write(
                    f"- Test de tendance: **{trend}** • p={fnum(trend_p,3)}"
                )

                st.code(_pipeline_path_str(pipe), language="text")

                with st.expander("Détails bruts (JSON)", expanded=False):
                    st.json(pipe)

st.divider()
st.subheader("📋 Tableau synthèse MTBF + β/η/γ")

dfm = pd.DataFrame(metrics_rows)
if not dfm.empty:
    dfm = dfm.sort_values("equipment_code").reset_index(drop=True)
    st.dataframe(
        dfm,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Aucune métrique calculable pour les équipements sélectionnés.")

# ---------- Export rapport complet ----------
st.divider()
st.subheader(
    "📄 Rapport complet (analyse + indicateurs + courbes)"
)
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
