from __future__ import annotations
from pathlib import Path
from datetime import datetime
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

# ✅ On garde ton export existant (fallback)
from core.reliability.reporting_optimize import export_optimization_report_pdf


st.set_page_config(page_title="Optimisation maintenance", page_icon="🧠", layout="wide")
require_login()

st.title("🧠 Optimisation — Intervalles, coût & fiabilité")
st.caption(
    "Objectif : proposer, par équipement, un intervalle de maintenance basé sur "
    "l’optimum économique (T_cost) et/ou une contrainte de fiabilité (T_R)."
)

DATA_FILE = Path("data/failures_saved.csv")


# ---------------------------
# Utils
# ---------------------------
def _read_csv_flex(src):
    """Lecture CSV robuste (fichier local ou UploadedFile), tolère séparateurs différents."""
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


def _safe_float(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _fmt(x, nd=1, suffix=""):
    v = _safe_float(x)
    if v is None:
        return "—"
    return f"{v:.{nd}f}{suffix}"


def _badge_risk(R_at_cost: float | None, R_min_cost: float) -> str:
    """
    Retourne un badge risque simple basé sur R(T_cost).
    """
    v = _safe_float(R_at_cost)
    if v is None:
        return "⚪ Indéterminé"
    if v < max(0.5, R_min_cost - 0.05):
        return "🔴 Risque élevé"
    if v < R_min_cost:
        return "🟠 Risque modéré"
    return "🟢 Acceptable"


# ---------------------------
# Guide de lecture (pro)
# ---------------------------
with st.expander("📘 Guide de lecture (paramètres & signification)", expanded=False):
    st.markdown(
        """
### Données & modèle
- **TTF (ttf_h)** : *Time To Failure* (heures). Temps jusqu’à la panne (ou entre pannes, selon le jeu de données).
- **Weibull (β, η)** :
  - **β (forme)** : interprète la dynamique du taux de panne.
    - **β < 1** : pannes de jeunesse → corrective + fiabilisation (qualité, installation, rodage).
    - **β ≈ 1** : pannes aléatoires → conditionnelle / inspection (le temps est peu prédictif).
    - **β > 1** : usure → préventive planifiée (remplacement/révision avant usure).
  - **η (échelle)** : ordre de grandeur de durée de vie / temps caractéristique (en heures).

### Paramètres de décision
- **R_target** : fiabilité cible (ex: 0.80). On cherche **T_R** tel que **R(T_R)=R_target**.
- **C_prev** : coût de maintenance préventive (action planifiée).
- **C_corr** : coût de panne / corrective (réparation d’urgence + pertes + indisponibilité, etc.).
- **R_min_cost** : contrainte de fiabilité minimale lors de l’optimisation coût.

### Sorties principales
- **T_cost** : intervalle minimisant le **coût moyen** (politique type âge).
- **R(T_cost)** : fiabilité au moment de l’optimum économique.
- **C_min/h** : coût moyen minimal par heure à l’optimum.
- **T_R** : intervalle imposé par la fiabilité cible.
- **T_recommended** : intervalle recommandé (compromis prudent) quand β>1.
        """
    )


# ---------------------------
# 1) Chargement données
# ---------------------------
st.markdown("### 📥 Chargement des données")

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


# ---------------------------
# 2) Fit Weibull
# ---------------------------
st.markdown("### 🧪 Estimation Weibull (par équipement)")

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


# ---------------------------
# 3) Paramètres utilisateur
# ---------------------------
st.markdown("### ⚙️ Paramètres de fiabilité et de coût")

colR, colC1, colC2, colRmin = st.columns(4)
with colR:
    R_target = st.slider("Fiabilité cible R_target", 0.50, 0.99, 0.80, 0.01)
with colC1:
    C_prev = st.number_input("Coût maintenance préventive C_prev", min_value=0.0, value=1.0, step=0.1)
with colC2:
    C_corr = st.number_input("Coût panne / corrective C_corr", min_value=0.0, value=5.0, step=0.5)
with colRmin:
    R_min_cost = st.slider("Fiabilité min. pour l’optimum coût R_min_cost", 0.0, 0.99, 0.70, 0.01)

st.caption(
    "Rappel : la fonction coût utilisée correspond à la formulation “type âge” : "
    "C(T) = (C_prev·R(T) + C_corr·(1−R(T))) / ∫₀ᵀ R(t) dt."
)

if C_prev <= 0 or C_corr <= 0:
    st.warning("Renseigne des coûts préventif/correctif > 0 pour l’optimisation économique.")


# ---------------------------
# 4) Organigramme
# ---------------------------
st.markdown("### 🧭 Diagnostic (organigramme) — modèle global par équipement")
org_results: dict[str, dict] = {}
for eq in fits.keys():
    ttf = df.loc[df["equipment_code"] == eq, "ttf_h"].tolist()
    org_results[eq] = analyze_ttf_pipeline(ttf)


# ---------------------------
# 5) Optimisation intervalles
# ---------------------------
st.markdown("### 🧠 Optimisation des intervalles")
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


# ---------------------------
# 6) Recommandation maintenance + intervalle
# ---------------------------
def recommend_maintenance(beta: float, model: str | None = None) -> str:
    if beta < 0.9:
        return "Corrective + fiabilisation (pannes de jeunesse)"
    if 0.9 <= beta <= 1.1:
        return "Conditionnelle / inspection (pannes aléatoires)"
    if model and "NHPP" in str(model).upper():
        return "Préventive planifiée (bloc/inspection) — vieillissement"
    return "Préventive planifiée (âge) — usure / vieillissement"


def recommend_interval(beta: float, T_cost: float | None, T_R: float | None) -> float | None:
    # Si β <= 1.1 : éviter de recommander une périodicité stricte
    if beta <= 1.1:
        return None
    vals = [v for v in [T_cost, T_R] if isinstance(v, (int, float)) and v and v > 0]
    return float(min(vals)) if vals else None


# ---------------------------
# 7) Tableau synthèse + KPIs
# ---------------------------
rows = []
for eq, ft in fits.items():
    org = org_results.get(eq, {})
    beta = float(ft.beta)
    eta = float(ft.eta)

    itv_R = intervals_R.get(eq)
    itv_C = intervals_cost.get(eq)
    R_cost = R_at_cost.get(eq)
    C_min = C_min_map.get(eq)

    maint_type = recommend_maintenance(beta, org.get("model"))
    T_rec = recommend_interval(beta, itv_C, itv_R)
    risk_badge = _badge_risk(R_cost, R_min_cost)

    rows.append({
        "equipment_code": eq,
        "beta": round(beta, 3),
        "eta_h": round(eta, 1),

        "T_cost_h": round(float(itv_C), 1) if isinstance(itv_C, (int, float)) else None,
        "R(T_cost)": round(float(R_cost), 3) if isinstance(R_cost, (int, float)) else None,
        "C_min_per_h": round(float(C_min), 4) if isinstance(C_min, (int, float)) else None,

        "T_R_h": round(float(itv_R), 1) if isinstance(itv_R, (int, float)) else None,

        "T_recommended_h": round(float(T_rec), 1) if isinstance(T_rec, (int, float)) else None,
        "maintenance_type": maint_type,
        "risk": risk_badge,

        "model": org.get("model", "?"),
        "distribution": org.get("distribution", "?"),
    })

df_out = pd.DataFrame(rows).sort_values("equipment_code").reset_index(drop=True)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Équipements", len(df_out))
k2.metric("Observations TTF", len(df))
k3.metric("R_target", f"{R_target:.2f}")
nb_wear = int((df_out["beta"] > 1.1).sum()) if "beta" in df_out.columns else 0
k4.metric("β>1.1 (usure)", nb_wear)

st.subheader("📋 Synthèse optimisation (coût, fiabilité, recommandation)")
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger CSV optimisé",
    data=csv_bytes,
    file_name="optimisation_intervalles.csv",
    mime="text/csv",
)


# ---------------------------
# 8) Courbes R(t)
# ---------------------------
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


# ---------------------------
# 9) Détails par équipement
# ---------------------------
st.subheader("🔎 Détails & interprétation")

sel = st.selectbox("Équipement", options=df_out["equipment_code"].tolist())
row = df_out[df_out["equipment_code"] == sel].iloc[0].to_dict()
ft = fits[sel]
org = org_results.get(sel, {})

beta = float(ft.beta)
eta = float(ft.eta)
itv_R = intervals_R.get(sel)
itv_C = intervals_cost.get(sel)
R_cost = R_at_cost.get(sel)
C_min = C_min_map.get(sel)

# Bloc "fiche décision" professionnel
left, right = st.columns([1.3, 1])
with left:
    st.markdown(f"### 🧾 Fiche décision — **{sel}**")
    st.markdown(
        f"""
- **β (forme)** : **{beta:.3f}**  
  ↳ Interprétation : *{'jeunesse (β<1)' if beta < 0.9 else 'aléatoire (β≈1)' if beta <= 1.1 else 'usure (β>1)'}*
- **η (échelle)** : **{eta:.1f} h**
- **Modèle (organigramme)** : **{org.get('model','?')}** • **Distribution** : **{org.get('distribution','?')}**
- **T_cost** : **{_fmt(itv_C, 1, ' h')}** • **R(T_cost)** : **{_fmt(R_cost, 3)}** • **C_min** : **{_fmt(C_min, 4, ' /h')}**
- **T_R** : **{_fmt(itv_R, 1, ' h')}** *(R_target = {R_target:.2f})*
- **Risque (optimum coût)** : **{row.get('risk','—')}**
- **Maintenance recommandée** : **{row.get('maintenance_type','—')}**
        """
    )

    # Intervalle recommandé (évite "nan h")
    T_rec = _safe_float(row.get("T_recommended_h"))
    if T_rec is None:
        st.info("Intervalle recommandé : non applicable (β ≤ 1). Préférer inspection/CBM ou corrective + fiabilisation.")
    else:
        st.success(f"Intervalle recommandé (compromis prudent) : {T_rec:.1f} h")

with right:
    st.markdown("### ✅ Actions suggérées")
    for a in suggested_actions(beta):
        st.markdown(f"- {a}")

    st.markdown("### 🧠 Lecture rapide")
    st.markdown(
        """
- **T_cost** : meilleur coût moyen (économie).
- **T_R** : respecte une fiabilité cible (sécurité/qualité).
- Pour un équipement critique : privilégier **T_R** ou augmenter **R_min_cost**.
        """
    )


# ---------------------------
# 10) Export PDF (robuste)
# ---------------------------
st.divider()
st.subheader("📄 Rapport PDF")

st.caption(
    "Recommandation : générer le PDF en mémoire et le proposer en téléchargement. "
    "Si ReportLab n’est pas installé, ajoute `reportlab` dans requirements.txt."
)

if st.button("Générer le rapport optimisation (PDF)"):
    try:
        # ✅ 1) Tentative : export bytes (plus fiable sur Streamlit)
        try:
            from core.reliability.reporting_optimize import export_optimization_report_pdf_bytes  # à ajouter
            pdf_bytes = export_optimization_report_pdf_bytes(df=df, df_out=df_out)
            st.success("PDF généré avec succès ✅")
            st.download_button(
                "⬇️ Télécharger le PDF",
                data=pdf_bytes,
                file_name=f"rapport_optimisation_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        except Exception:
            # ✅ 2) Fallback : ton export existant qui retourne un path
            path = export_optimization_report_pdf(df, fits, intervals_R, org_results, out_dir="reports")
            st.success(f"PDF généré : {path}")
            # lecture binaire pour download
            p = Path(path)
            if p.exists():
                st.download_button(
                    "⬇️ Télécharger le PDF",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/pdf",
                )
            else:
                st.warning("Le fichier PDF n’a pas été trouvé sur disque (chemin invalide).")
    except Exception as e:
        st.error(f"Impossible de générer le PDF : {e}")
        st.info("Vérifie : (1) `reportlab` installé, (2) pas de fichier local nommé reportlab.py, (3) requirements.txt à jour.")
