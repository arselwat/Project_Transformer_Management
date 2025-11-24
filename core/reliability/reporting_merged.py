# core/reliability/reporting_merged.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

import math
import numpy as np
import pandas as pd

# Matplotlib headless pour générer les figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)

# "Vérité unique" (bundle cohérent) + fonctions de courbes
from core.reliability.unify import compute_bundle, UnifyOptions, UnifyBundle
from core.reliability.weibull import R, F, pdf, hazard

# --- Helpers robustes pour optim ---
def _opt_interval(optim: dict | None, eq: str) -> float | None:
    v = (optim or {}).get(eq)
    if isinstance(v, dict):
        return v.get("interval_opt_h")
    try:
        return float(v)
    except Exception:
        return None

def _opt_R_target(optim: dict | None, eq: str, default: float = 0.80) -> float:
    v = (optim or {}).get(eq)
    if isinstance(v, dict):
        return float(v.get("R_target", default))
    return float(default)

# ------------------------- UTILITAIRES -------------------------
def _fmt(x, nd=2, dash="—"):
    try:
        if x is None:
            return dash
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return dash
            return f"{x:.{nd}f}"
        return str(x)
    except Exception:
        return dash

def _mk_table(data: List[List[Any]], col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("TOPPADDING", (0,0), (-1,0), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    return t

class _WB:
    """Petit conteneur pour compatibilité R/F/pdf/hazard (attrs beta, eta, gamma)."""
    def __init__(self, beta: float, eta: float, gamma: float = 0.0):
        self.beta = float(beta)
        self.eta = float(eta)
        self.gamma = float(gamma or 0.0)

def _dist_name(d) -> str:
    if isinstance(d, dict):
        return d.get("name", "Weibull2P")
    if isinstance(d, str):
        return d
    return "Weibull2P"

def _pipe_field(pipe: dict, path: List[str], default=None):
    """Accès sûr aux champs dans les traces organigramme."""
    cur = pipe or {}
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return default
    return cur

def _equip_time_grid(eq: str, bundle: UnifyBundle, fit: _WB) -> np.ndarray:
    """Grille temporelle : couvre au moins les TTF réels et ~1.5*eta."""
    try:
        ttf_eq = bundle.ttf.loc[bundle.ttf["equipment_code"] == eq, "ttf_h"]
        tmax_data = float(ttf_eq.max()) if not ttf_eq.empty else 0.0
    except Exception:
        tmax_data = 0.0
    tmax = max(100.0, tmax_data * 1.2, fit.eta * 1.5)
    return np.linspace(0.0, tmax, 400)

def _fig_to_rl_image(fig, width_cm=16.0) -> Image:
    """Sauve une figure matplotlib en PNG mémoire puis retourne un Flowable Image."""
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    img = Image(bio)
    # Ajuste largeur, conserve le ratio
    w = width_cm * cm
    ratio = img.imageHeight / max(img.imageWidth, 1)
    img.drawWidth = w
    img.drawHeight = w * ratio
    return img

def _plot_panel_for_eq(eq: str, fit: _WB, t: np.ndarray) -> Optional[Image]:
    """Figure 2x2 : R(t), F(t), f(t), h(t). Retourne une Image reportlab (ou None)."""
    try:
        yR = R(t, fit)
        yF = F(t, fit)
        yf = pdf(t, fit)
        yh = hazard(t, fit)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.ravel()

        axes[0].plot(t, yR, linewidth=2)
        axes[0].set_title("Fiabilité R(t)"); axes[0].set_xlabel("Temps (h)"); axes[0].set_ylabel("R(t)")
        axes[0].grid(True, alpha=.3)

        axes[1].plot(t, yF, linewidth=2)
        axes[1].set_title("Répartition F(t)"); axes[1].set_xlabel("Temps (h)"); axes[1].set_ylabel("F(t)")
        axes[1].grid(True, alpha=.3)

        axes[2].plot(t, yf, linewidth=2)
        axes[2].set_title("Densité f(t)"); axes[2].set_xlabel("Temps (h)"); axes[2].set_ylabel("f(t)")
        axes[2].grid(True, alpha=.3)

        axes[3].plot(t, yh, linewidth=2)
        axes[3].set_title("Taux de défaillance h(t)"); axes[3].set_xlabel("Temps (h)"); axes[3].set_ylabel("h(t)")
        axes[3].grid(True, alpha=.3)

        fig.tight_layout()
        return _fig_to_rl_image(fig, width_cm=16.0)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ------------------------- SECTIONS PAR ÉQUIPEMENT -------------------------
def _per_eq_section(eq: str, metrics_df: pd.DataFrame, bundle: UnifyBundle) -> List[Any]:
    """
    Construit les blocs pour 1 équipement :
      - trace organigramme (résumé)
      - tableaux d'analyse (avant / après)
      - panneau 4 graphiques (R, F, f, h)
    """
    elems: List[Any] = []
    styles = getSampleStyleSheet()

    elems.append(Paragraph(f"Équipement : <b>{eq}</b>", styles["Heading3"]))

    row = metrics_df.loc[metrics_df["equipment_code"] == eq]
    if row.empty:
        elems.append(Paragraph("Aucune donnée exploitable pour cet équipement.", styles["Normal"]))
        elems.append(Spacer(1, 0.3*cm))
        return elems
    r = row.iloc[0].to_dict()

    # Résumé organigramme/pipeline (si dispo)
    pipe = (bundle.pipeline_by_eq or {}).get(eq, {}) or {}
    dist = _dist_name(pipe.get("distribution")) if pipe else "Weibull2P"
    ks_p = _pipe_field(pipe, ["goodness", "ks_p"])
    chi2_p = _pipe_field(pipe, ["goodness", "chi2_p"])
    trend_name = _pipe_field(pipe, ["trend", "name"], "MK")
    trend_p = _pipe_field(pipe, ["trend", "p_value"])

    trace_line = f"TTF>0 → {trend_name}(p={_fmt(trend_p,3)}) → RP/NHPP/BPP ; Dist={dist}"
    if ks_p is not None or chi2_p is not None:
        trace_line += f" ; KS p={_fmt(ks_p,3)}, Chi2 p={_fmt(chi2_p,3)}"

    elems.append(Paragraph(f"Chaîne d'exécution — Organigramme<br/>{trace_line}", styles["Normal"]))
    elems.append(Spacer(1, 0.2*cm))

    # Tableaux AVANT / APRES
    n_ttf = 0
    try:
        n_ttf = int(bundle.ttf.loc[bundle.ttf["equipment_code"] == eq, "ttf_h"].size)
    except Exception:
        pass

    avant = [
        ["Mesure", "Valeur"],
        ["n TTF", _fmt(n_ttf, 0)],
        ["MTBF empirique (h)", _fmt(r.get("MTBF"), 1)],
        ["MTTF théorique (h)", _fmt(r.get("MTTF_th"), 1)],
        ["β", _fmt(r.get("beta"), 2)],
        ["η (h)", _fmt(r.get("eta"), 1)],
        ["γ (h)", _fmt(r.get("gamma"), 1)],
    ]
    elems.append(_mk_table(avant, [8*cm, 8*cm]))
    elems.append(Spacer(1, 0.2*cm))

    # ...
    rstar = _opt_R_target(bundle.optim, eq, 0.80)
    mtbf_opt = r.get("MTBF_opt")
    if (mtbf_opt is None or (isinstance(mtbf_opt, float) and math.isnan(mtbf_opt))):
        # fallback possible : intervalle optimisé
        mtbf_opt = r.get("interval_opt_h", _opt_interval(bundle.optim, eq))
    
    apres = [
        ["Paramètre", "Valeur"],
        ["Intervalle optimisé (h)", _fmt(r.get("interval_opt_h", _opt_interval(bundle.optim, eq)), 1)],
        ["Fiabilité cible R*", _fmt(rstar, 2)],
        ["MTBF optimisé (h)", _fmt(mtbf_opt, 1)],
        ["MTTR optimisé (h)", _fmt(r.get("MTTR_opt"), 1)],
    ]
    # ...
    
    elems.append(_mk_table(apres, [8*cm, 8*cm]))
    elems.append(Spacer(1, 0.25*cm))

    # Graphiques (2x2)
    try:
        # fit = _WB depuis les β/η/γ du bundle (cohérence totale)
        beta = float(r.get("beta"))
        eta  = float(r.get("eta"))
        gamma = float(r.get("gamma") or 0.0)
        fit = _WB(beta, eta, gamma)
        t = _equip_time_grid(eq, bundle, fit)
        panel_img = _plot_panel_for_eq(eq, fit, t)
        if panel_img is not None:
            elems.append(panel_img)
            elems.append(Spacer(1, 0.35*cm))
    except Exception:
        pass

    return elems


# ------------------------- EXPORT PDF PRINCIPAL -------------------------
def export_merged_report_pdf(
    df: Optional[pd.DataFrame] = None,
    fits: Optional[Dict[str, Any]] = None,              # ignorés si df fourni
    pipeline_by_eq: Optional[Dict[str, Dict[str, Any]]] = None,  # idem
    optim_intervals: Optional[Dict[str, Dict[str, Any]]] = None, # idem
    metrics_table: Optional[List[Dict[str, Any]]] = None,        # idem
    out_dir: str = "reports",
    title: str = "Rapport complet — Analyse & Optimisation",
    options: Optional[UnifyOptions] = None,
) -> str:
    """
    Génére un PDF consolidé à partir de la vérité unique (compute_bundle):
      - β/η/γ + KPI (MTBF, MTTF_th) + interval_opt + MTBF_opt (fallback sur intervalle)
      - Organigramme résumé
      - Panneau 4 graphes (R, F, f, h) par équipement
      - Récap global
    Les arguments fits/pipeline/optim/metrics sont ignorés si 'df' est fourni :
    on reconstruit tout depuis compute_bundle pour la cohérence.
    """
    opt = options or UnifyOptions(force_weibull_2p=True, R_target=0.80)
    bundle = compute_bundle(session_df=df, options=opt)

    if bundle.metrics_df.empty:
        raise RuntimeError("Aucune métrique exploitable pour générer le rapport.")

    # Compléter MTTF_th si absent, et normaliser MTBF_opt
    mdf = bundle.metrics_df.copy()
    if "MTTF_th" not in mdf.columns:
        mdf["MTTF_th"] = None
    for i, row in mdf.iterrows():
        b = row.get("beta"); e = row.get("eta"); g = row.get("gamma", 0.0)
        try:
            mdf.at[i, "MTTF_th"] = float(g) + float(e) * math.gamma(1.0 + 1.0/float(b))
        except Exception:
            mdf.at[i, "MTTF_th"] = None
        if pd.isna(row.get("MTBF_opt")) and not pd.isna(row.get("interval_opt_h")):
            mdf.at[i, "MTBF_opt"] = row.get("interval_opt_h")

    # Création PDF
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"report_analyse_optim_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"
    fpath = out / fname

    doc = SimpleDocTemplate(str(fpath), pagesize=A4,
                            rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    # En-tête
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # Sections par équipement
    eqs = sorted(mdf["equipment_code"].dropna().astype(str).unique().tolist())
    for k, eq in enumerate(eqs):
        story.extend(_per_eq_section(eq, mdf, bundle))
        if k < len(eqs) - 1:
            story.append(PageBreak())

    # Récap global
    story.append(Paragraph("Récapitulatif global", styles["Heading2"]))
    head = ["Équipement", "β", "η (h)", "γ (h)", "Intervalle opt (h)",
            "n TTF", "MTBF mesuré (h)", "MTTF théorique (h)", "MTBF optimisé (h)"]
    rows = [head]
    for _, r in mdf.sort_values("equipment_code").iterrows():
        try:
            n_ttf = bundle.ttf.loc[bundle.ttf["equipment_code"] == r["equipment_code"], "ttf_h"].size
        except Exception:
            n_ttf = 0
        rows.append([
            str(r.get("equipment_code")),
            _fmt(r.get("beta"), 2),
            _fmt(r.get("eta"), 1),
            _fmt(r.get("gamma"), 1),
            _fmt(r.get("interval_opt_h"), 1),
            _fmt(n_ttf, 0),
            _fmt(r.get("MTBF"), 1),
            _fmt(r.get("MTTF_th"), 1),
            _fmt(r.get("MTBF_opt"), 1),
        ])
    story.append(_mk_table(rows, [3.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 3.0*cm, 1.7*cm, 3.2*cm, 3.0*cm, 3.2*cm]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Rapport généré automatiquement (bundle unifié : β/η/γ et KPI identiques partout).",
        styles["Italic"])
    )

    doc.build(story)
    return str(fpath)
