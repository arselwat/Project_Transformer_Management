# pages/4_Optimisation.py
from __future__ import annotations

from pathlib import Path
import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.reliability.weibull import fit_weibull
from core.reliability.policy import suggested_actions
from core.reliability.organigram import analyze_ttf_pipeline
from core.reliability.optimize import propose_intervals_cost_and_reliability
from core.security.auth import require_login


# ==========================
# PDF export (SAFE IMPORT)
# ==========================
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
st.caption(
    "Cette page calcule un planning optimisé (par équipement) et peut **envoyer directement** le résultat à la page Maintenance "
    "sans passer par une BD (utile tant que la BD n'est pas prête)."
)

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


def is_pos_number(x) -> bool:
    try:
        return float(x) > 0 and np.isfinite(float(x))
    except Exception:
        return False


def _df_hash(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


# ==========================
# 1) Chargement données
# ==========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "failures_saved.csv"

src = st.radio("Source TTF", ["Fichier projet", "Uploader CSV"], horizontal=True)

if src == "Fichier projet":
    if not DATA_FILE.exists():
        st.error("Aucun fichier data/failures_saved.csv — va dans « Sources de données » pour enregistrer.")
        st.stop()
    df = _read_csv_flex(DATA_FILE)
else:
    up = st.file_uploader("CSV (equipment_code, ttf_h)", type=["csv"])
    if up is None:
        st.stop()
    df = _read_csv_flex(up)

if df.empty:
    st.error("CSV vide ou illisible.")
    st.stop()

required = {"equipment_code", "ttf_h"}
if not required.issubset(set(df.columns)):
    st.error("Colonnes requises: equipment_code, ttf_h")
    st.stop()

df["equipment_code"] = df["equipment_code"].astype(str)
df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
df = df.dropna(subset=["ttf_h"])
df = df[df["ttf_h"] > 0]

eqs = sorted(df["equipment_code"].unique().tolist())
if not eqs:
    st.error("Aucun équipement valide (TTF>0).")
    st.stop()


# ==========================
# 2) Fit Weibull (baseline)
# ==========================
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


# ==========================
# 3) Paramètres utilisateur
# ==========================
st.markdown("### Paramètres de fiabilité et de coût")

colR, colC1, colC2, colRmin = st.columns(4)
with colR:
    R_target = st.slider("Fiabilité cible R(t)", 0.50, 0.99, 0.80, 0.01)
with colC1:
    C_prev = st.number_input("Coût maintenance préventive (C_prev)", min_value=0.0, value=1.0, step=0.1)
with colC2:
    C_corr = st.number_input("Coût panne / corrective (C_corr)", min_value=0.0, value=5.0, step=0.5)
with colRmin:
    R_min_cost = st.slider("Fiabilité min. pour l’optimum coût", 0.0, 0.99, 0.70, 0.01)

st.caption(
    "Formule coût (politique âge) : "
    "C(T) = (C_prev·R(T) + C_corr·(1−R(T))) / ∫₀ᵀ R(t)dt. "
    "Le préventif est pondéré par R(T) (on ne paye pas si la panne survient avant T)."
)

econ_enabled = (C_prev > 0) and (C_corr > 0)
if not econ_enabled:
    st.warning("Renseigne C_prev > 0 et C_corr > 0 pour activer l’optimisation économique (T_cost).")


# ==========================
# 4) Organigramme (par équipement)
# ==========================
org_results: dict[str, dict] = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    try:
        org_results[eq] = analyze_ttf_pipeline(ttf)
    except Exception:
        org_results[eq] = {}


# ==========================
# 5) Intervalles coût + fiabilité
# ==========================
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


# ==========================
# 6) Recommandations
# ==========================
def recommend_maintenance(beta: float, model: str | None = None) -> str:
    if beta < 0.9:
        return "Corrective + fiabilisation (pannes de jeunesse)"
    if 0.9 <= beta <= 1.1:
        return "Conditionnelle / inspection (pannes aléatoires)"
    if model and "NHPP" in str(model).upper():
        return "Préventive planifiée (bloc/inspection) — vieillissement"
    return "Préventive planifiée (âge) — usure / vieillissement"


def recommend_interval(beta: float, T_cost: float | None, T_R: float | None) -> float | None:
    if beta <= 1.1:
        return None
    vals = [v for v in [T_cost, T_R] if is_pos_number(v)]
    return float(min(vals)) if vals else None


# ==========================
# 7) Tableau synthèse + CSV + PASSERELLE
# ==========================
rows = []
for eq, ft in fits.items():
    org = org_results.get(eq, {}) or {}
    beta = float(getattr(ft, "beta", float("nan")))
    eta = float(getattr(ft, "eta", float("nan")))
    gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

    itv_R = intervals_R.get(eq)
    itv_C = intervals_cost.get(eq)
    R_cost = R_at_cost.get(eq)
    C_min = C_min_map.get(eq)

    maint_type = recommend_maintenance(beta, org.get("model"))
    T_rec = recommend_interval(beta, itv_C, itv_R)

    rows.append({
        "equipment_code": eq,
        "beta": round(beta, 3) if np.isfinite(beta) else None,
        "eta_h": round(eta, 1) if np.isfinite(eta) else None,
        "gamma_h": round(gamma, 1) if np.isfinite(gamma) else 0.0,

        "T_cost_h": round(float(itv_C), 1) if is_pos_number(itv_C) else None,
        "R(T_cost)": round(float(R_cost), 3) if is_pos_number(R_cost) else None,
        "C_min_per_h": round(float(C_min), 4) if is_pos_number(C_min) else None,

        "T_R_h": round(float(itv_R), 1) if is_pos_number(itv_R) else None,

        "T_recommended_h": round(float(T_rec), 1) if is_pos_number(T_rec) else None,
        "maintenance_type": maint_type,

        "model": org.get("model", "?"),
        "distribution": org.get("distribution", "?"),
    })

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)
st.session_state["optimization_df"] = df_out.copy()
st.session_state["optimization_src"] = "optimisation_page"

st.divider()
st.subheader()

opt_hash = _df_hash(df_out)

# ✅ Envoi automatique vers Maintenance (session) — sans bouton
st.session_state["opt_df_out"] = df_out.copy()
st.session_state["opt_meta"] = {"hash": opt_hash, "rows": int(len(df_out)), "source": "optimisation_page"}

colB = st.columns([1])[0]

with colB:
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    FALLBACK_OPT = DATA_DIR / "last_optimization.csv"

    if st.button("💾 Sauver aussi en fichier (fallback Streamlit Cloud)", use_container_width=True):
        df_out.to_csv(FALLBACK_OPT, index=False, encoding="utf-8")
        st.success(f"Écrit: {FALLBACK_OPT}")

st.caption(
    "Astuce Streamlit Cloud : la session peut disparaître après redémarrage. "
    "Le fichier `data/last_optimization.csv` sert de fallback."
)

st.subheader("📋 Synthèse optimisation (coût, fiabilité, recommandation)")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger CSV optimisé",
    data=csv_bytes,
    file_name="optimisation_intervalles.csv",
    mime="text/csv",
    use_container_width=True,
)


# ==========================
# 8) Courbes R(t) (Weibull)
# ==========================
st.subheader("📈 Courbes R(t) (Weibull)")

etas = [float(getattr(ft, "eta", 1.0) or 1.0) for ft in fits.values()]
tmax = max(etas) * 1.6 if etas else 1000.0

maybe_itv = []
for eq in fits.keys():
    if is_pos_number(intervals_R.get(eq)):
        maybe_itv.append(float(intervals_R[eq]))
    if is_pos_number(intervals_cost.get(eq)):
        maybe_itv.append(float(intervals_cost[eq]))
if maybe_itv:
    tmax = max(tmax, max(maybe_itv) * 1.2)

t = np.linspace(0, max(tmax, 1.0), 350)

fig, ax = plt.subplots()
for eq, ft in fits.items():
    beta = float(getattr(ft, "beta", 1.0))
    eta = float(getattr(ft, "eta", 1.0))
    gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

    y = np.ones_like(t, dtype=float)
    mask = t > gamma
    y[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-9)) ** max(beta, 1e-9)))
    ax.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f}, γ={gamma:.1f})")

ax.grid(True, alpha=.3)
ax.set_xlabel("Temps (h)")
ax.set_ylabel("R(t)")
ax.set_title("Fiabilité R(t)")
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)


# ==========================
# 9) Détails & interprétation
# ==========================
st.subheader("🔎 Détails & interprétation")

sel_eq = st.selectbox("Équipement", options=df_out["equipment_code"].tolist())
row = df_out[df_out["equipment_code"] == sel_eq].iloc[0].to_dict()
ft = fits[sel_eq]
org = (org_results.get(sel_eq, {}) or {})

beta = float(getattr(ft, "beta", float("nan")))
eta = float(getattr(ft, "eta", float("nan")))
gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

itv_R = intervals_R.get(sel_eq)
itv_C = intervals_cost.get(sel_eq)
R_cost = R_at_cost.get(sel_eq)
C_min = C_min_map.get(sel_eq)

st.markdown(
    f"### Résultats — **{sel_eq}**\n"
    f"- **β (forme)** = **{fnum(beta,3)}**\n"
    f"- **η (échelle)** = **{fnum(eta,1)} h**\n"
    f"- **γ (décalage)** = **{fnum(gamma,1)} h**\n"
    f"- **T_cost** = **{fnum(itv_C,1)} h**\n"
    f"- **R(T_cost)** = **{fnum(R_cost,3)}**\n"
    f"- **C_min** = **{fnum(C_min,4)} /h**\n"
    f"- **T_R** = **{fnum(itv_R,1)} h** (R_target={R_target:.2f})\n"
    f"- **Maintenance recommandée** : **{row.get('maintenance_type','—')}**\n"
)

if is_pos_number(row.get("T_recommended_h")):
    st.markdown(f"- **Intervalle recommandé** : **{row['T_recommended_h']:.1f} h**")
else:
    st.markdown("- **Intervalle recommandé** : basé sur l’état/inspection (pas de périodicité stricte)")

st.caption(
    "Note: T_cost est économique, T_R est orienté fiabilité. "
    "Sur un équipement critique, on privilégie souvent T_R."
)

st.write(f"Modèle global (organigramme): **{org.get('model','?')}** • Distribution: **{org.get('distribution','?')}**")

st.markdown("#### Actions suggérées (selon β)")
for a in suggested_actions(beta if np.isfinite(beta) else 1.0):
    st.markdown(f"- {a}")


# ==========================
# 10) Export PDF + Download
# ==========================
st.divider()
st.subheader("📄 Rapport PDF — Optimisation")

if export_optimization_report_pdf is None:
    st.info("Module `core.reliability.reporting_optimize` non détecté.")
    if _pdf_import_error is not None:
        st.caption(f"Détail import: {_pdf_import_error}")
else:
    if st.button("📄 Générer rapport optimisation (PDF)"):
        try:
            out_dir = str(BASE_DIR / "reports")

            intervals = {}
            for eq in fits.keys():
                intervals[eq] = {
                    "T_R": intervals_R.get(eq),
                    "T_cost": intervals_cost.get(eq),
                    "R_at_T": R_at_cost.get(eq),
                    "C_min": C_min_map.get(eq),
                }

            path = export_optimization_report_pdf(
                df,
                fits,
                intervals,
                org_results,
                out_dir=out_dir,
            )

            st.session_state["opt_pdf_path"] = path
            st.success(f"PDF généré : {path}")
        except Exception as e:
            st.error(f"PDF : {e}")

    pdf_path = st.session_state.get("opt_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Télécharger le PDF optimisation",
                data=f,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )
