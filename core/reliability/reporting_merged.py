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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

from core.reliability.unify import compute_bundle, UnifyOptions, UnifyBundle
from core.reliability.weibull import R, F, pdf, hazard


def _fmt(x, nd=2, dash="—"):
    try:
        if x is None:
            return dash
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return dash
        return f"{float(x):.{nd}f}"
    except Exception:
        return dash


def _mk_table(data: List[List[Any]], col_widths=None, font_size=8, avail_width=None):
    """
    Table ReportLab robuste:
    - wrap automatique via Paragraph
    - scale automatique si col_widths dépasse la largeur dispo (A4)
    - repeatRows=1 pour répéter l’en-tête si la table déborde sur plusieurs pages
    """
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"]
    cell_style.fontName = "Helvetica"
    cell_style.fontSize = font_size
    cell_style.leading = font_size + 2

    # wrap cellules (sauf si déjà Flowable)
    wrapped: List[List[Any]] = []
    for row in data:
        new_row = []
        for c in row:
            if HAVE_REPORTLAB and not hasattr(c, "wrapOn"):
                txt = "" if c is None else str(c)
                new_row.append(Paragraph(txt, cell_style))
            else:
                new_row.append(c)
        wrapped.append(new_row)

    # largeur dispo (A4 - marges) si non fournie
    if avail_width is None:
        try:
            avail_width = A4[0] - (14 * mm + 14 * mm)  # mêmes marges que le doc
        except Exception:
            avail_width = None

    # colWidths
    if col_widths is None and avail_width:
        ncol = len(wrapped[0]) if wrapped else 1
        col_widths = [avail_width / max(ncol, 1)] * ncol

    # scale si trop large
    if avail_width and col_widths:
        total = float(sum(col_widths))
        if total > avail_width and total > 0:
            k = avail_width / total
            col_widths = [w * k for w in col_widths]

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),

                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


class _WB:
    def __init__(self, beta: float, eta: float, gamma: float = 0.0):
        self.beta = float(beta)
        self.eta = float(eta)
        self.gamma = float(gamma or 0.0)


def _fig_to_rl_image(fig, width_mm=170):
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    img = Image(bio)
    w = width_mm * mm
    ratio = img.imageHeight / max(img.imageWidth, 1)
    img.drawWidth = w
    img.drawHeight = w * ratio
    return img


def _equip_time_grid(eq: str, bundle: UnifyBundle, fit: _WB) -> np.ndarray:
    try:
        ttf_eq = bundle.ttf.loc[bundle.ttf["equipment_code"] == eq, "ttf_h"]
        tmax_data = float(ttf_eq.max()) if not ttf_eq.empty else 0.0
    except Exception:
        tmax_data = 0.0
    tmax = max(100.0, tmax_data * 1.2, fit.eta * 1.5)
    return np.linspace(0.0, tmax, 400)


def _plot_panel(eq: str, fit: _WB, t: np.ndarray):
    yR = R(t, fit)
    yF = F(t, fit)
    yf = pdf(t, fit)
    yh = hazard(t, fit)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()

    axes[0].plot(t, yR, linewidth=2)
    axes[0].set_title("Fiabilité R(t)")
    axes[0].set_xlabel("Temps (h)")
    axes[0].set_ylabel("R(t)")
    axes[0].grid(True, alpha=.3)

    axes[1].plot(t, yF, linewidth=2)
    axes[1].set_title("Répartition F(t)")
    axes[1].set_xlabel("Temps (h)")
    axes[1].set_ylabel("F(t)")
    axes[1].grid(True, alpha=.3)

    axes[2].plot(t, yf, linewidth=2)
    axes[2].set_title("Densité f(t)")
    axes[2].set_xlabel("Temps (h)")
    axes[2].set_ylabel("f(t)")
    axes[2].grid(True, alpha=.3)

    axes[3].plot(t, yh, linewidth=2)
    axes[3].set_title("Taux de défaillance h(t)")
    axes[3].set_xlabel("Temps (h)")
    axes[3].set_ylabel("h(t)")
    axes[3].grid(True, alpha=.3)

    fig.tight_layout()
    return _fig_to_rl_image(fig, width_mm=170)


def _maintenance_type(beta: float) -> str:
    if beta < 0.9:
        return "Corrective + fiabilisation (jeunesse)"
    if beta <= 1.1:
        return "Conditionnelle / inspection (aléatoire)"
    return "Préventive planifiée (âge) (usure)"


def _pipe_line(pipe: dict) -> str:
    if not isinstance(pipe, dict) or not pipe:
        return "Trace indisponible."
    model = pipe.get("model", "RP")
    dist = pipe.get("distribution", "weibull_2p")
    good = pipe.get("goodness", {}) or {}
    tests = pipe.get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    dep = tests.get("dependence", {}) or {}
    return (
        f"TTF>0 → MK(p={_fmt(mk.get('p'),3)}, dir={mk.get('direction','none')}) "
        f"→ Dep(r={_fmt(dep.get('r'),3)}, p={_fmt(dep.get('p'),3)}) "
        f"→ Model={model} ; Dist={dist} ; KS p={_fmt(good.get('ks_p'),3)} ; Chi2 p={_fmt(good.get('chi2_p'),3)}"
    )


def _per_eq_section(eq: str, mdf: pd.DataFrame, bundle: UnifyBundle) -> List[Any]:
    styles = getSampleStyleSheet()
    elems: List[Any] = []

    row = mdf.loc[mdf["equipment_code"] == eq]
    if row.empty:
        elems.append(Paragraph(f"Équipement : <b>{eq}</b>", styles["Heading3"]))
        elems.append(Paragraph("Aucune donnée exploitable.", styles["Normal"]))
        return elems

    r = row.iloc[0].to_dict()
    beta = float(r.get("beta")) if r.get("beta") is not None else float("nan")
    eta = float(r.get("eta")) if r.get("eta") is not None else float("nan")
    gamma = float(r.get("gamma") or 0.0)

    pipe = (bundle.pipeline_by_eq or {}).get(eq, {}) or {}

    # ---- Header
    elems.append(Paragraph(f"Équipement : <b>{eq}</b>", styles["Heading2"]))
    elems.append(Paragraph(_pipe_line(pipe), styles["Normal"]))
    elems.append(Spacer(1, 6))

    # ---- AVANT (analyse)
    try:
        n_ttf = int(bundle.ttf.loc[bundle.ttf["equipment_code"] == eq, "ttf_h"].size)
    except Exception:
        n_ttf = 0

    # MTTF théorique Weibull (γ + η Γ(1+1/β))
    mttf_th = None
    try:
        if np.isfinite(beta) and beta > 0 and np.isfinite(eta) and eta > 0:
            mttf_th = gamma + eta * math.gamma(1.0 + 1.0 / beta)
    except Exception:
        mttf_th = None

    avant = [
        ["Mesure (avant optimisation)", "Valeur"],
        ["n TTF", _fmt(n_ttf, 0)],
        ["MTBF empirique (h)", _fmt(r.get("MTBF"), 1)],
        ["MTTR (h)", _fmt(r.get("MTTR"), 1)],
        ["β", _fmt(beta, 3)],
        ["η (h)", _fmt(eta, 1)],
        ["γ (h)", _fmt(gamma, 1)],
        ["MTTF théorique (h)", _fmt(mttf_th, 1)],
        ["Type maintenance (β)", _maintenance_type(beta)],
    ]
    elems.append(_mk_table(avant, [8.5 * cm, 8.5 * cm]))
    elems.append(Spacer(1, 8))

    # ---- APRÈS (optimisation)
    interval_opt = r.get("interval_opt_h")
    if interval_opt is None or (isinstance(interval_opt, float) and math.isnan(interval_opt)):
        try:
            interval_opt = (bundle.optim or {}).get(eq, {}).get("interval_opt_h")
        except Exception:
            interval_opt = None

    apres = [
        ["Mesure (après optimisation)", "Valeur"],
        ["Intervalle optimisé (h)", _fmt(interval_opt, 1)],
        ["R* (fiabilité cible)", _fmt(getattr(bundle, "R_target", None), 2)],
        ["MTBF optimisé (h)", _fmt(r.get("MTBF_opt"), 1)],
        ["MTTR optimisé (h)", _fmt(r.get("MTTR_opt"), 1)],
    ]
    elems.append(_mk_table(apres, [8.5 * cm, 8.5 * cm]))
    elems.append(Spacer(1, 10))

    # ---- Courbes (R,F,f,h)
    try:
        fit = _WB(beta=float(beta), eta=float(eta), gamma=float(gamma))
        t = _equip_time_grid(eq, bundle, fit)
        elems.append(_plot_panel(eq, fit, t))
    except Exception:
        pass

    elems.append(Spacer(1, 10))
    return elems


def export_merged_report_pdf(
    df: Optional[pd.DataFrame] = None,
    out_dir: str = "reports",
    title: str = "Rapport complet — Analyse & Optimisation",
    options: Optional[UnifyOptions] = None,
) -> str:
    if not HAVE_REPORTLAB:
        raise RuntimeError("ReportLab non disponible. Installe: pip install reportlab")

    # options par défaut
    opt = options or UnifyOptions(force_weibull_2p=True, R_target=0.80)
    bundle = compute_bundle(session_df=df, options=opt)

    if bundle.metrics_df.empty:
        raise RuntimeError("Aucune métrique exploitable pour générer le rapport.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"report_analyse_optim_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    doc = SimpleDocTemplate(
        str(fpath),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    story: List[Any] = []

    # ---- Couverture
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 10))

    # ---- Récap synthèse global
    mdf = bundle.metrics_df.copy()
    eqs = sorted(mdf["equipment_code"].dropna().astype(str).unique().tolist())

    story.append(Paragraph("Synthèse globale", styles["Heading2"]))
    story.append(Paragraph(f"Nombre d’équipements : {len(eqs)}", styles["Normal"]))
    story.append(Paragraph(f"Nombre d’observations TTF : {int(bundle.ttf.shape[0])}", styles["Normal"]))
    story.append(Spacer(1, 8))

    # ---- Ajout : Bloc optimisation (hypothèses & paramètres)
    story.append(Paragraph("Optimisation — Hypothèses & paramètres", styles["Heading2"]))

    # On lit des attributs possibles (si tu les ajoutes dans UnifyOptions plus tard, ils s’afficheront automatiquement)
    R_target = getattr(opt, "R_target", None)
    C_prev = getattr(opt, "C_prev", None) or getattr(opt, "Cp", None) or getattr(opt, "cost_prev", None)
    C_corr = getattr(opt, "C_corr", None) or getattr(opt, "Cf", None) or getattr(opt, "cost_corr", None)
    R_min_cost = getattr(opt, "R_min_cost", None) or getattr(opt, "Rmin", None)

    opt_rows = [
        ["Paramètre", "Valeur"],
        ["Fiabilité cible R*", _fmt(R_target, 2)],
        ["Coût préventif Cp", _fmt(C_prev, 2)],
        ["Coût correctif Cf", _fmt(C_corr, 2)],
        ["Seuil R_min (optionnel)", _fmt(R_min_cost, 2)],
        ["Modèle économique", "C(T) = (Cp·R(T) + Cf·(1−R(T))) / ∫0..T R(t) dt"],
    ]
    story.append(_mk_table(opt_rows, [8.5 * cm, 8.5 * cm], font_size=8))
    story.append(Spacer(1, 10))

    # ---- Table synthèse globale (protégée A4 grâce à _mk_table)
    head = ["Équipement", "n TTF", "β", "η(h)", "γ(h)", "MTBF(h)", "Intervalle opt(h)", "Type maintenance", "Modèle", "Loi"]
    rows = [head]
    for eq in eqs:
        r = mdf.loc[mdf["equipment_code"] == eq].iloc[0].to_dict()
        try:
            n_ttf = int(bundle.ttf.loc[bundle.ttf["equipment_code"] == eq, "ttf_h"].size)
        except Exception:
            n_ttf = 0
        pipe = (bundle.pipeline_by_eq or {}).get(eq, {}) or {}
        rows.append([
            eq,
            _fmt(n_ttf, 0),
            _fmt(r.get("beta"), 3),
            _fmt(r.get("eta"), 1),
            _fmt(r.get("gamma"), 1),
            _fmt(r.get("MTBF"), 1),
            _fmt(r.get("interval_opt_h"), 1),
            _maintenance_type(float(r.get("beta")) if r.get("beta") is not None else float("nan")),
            str(pipe.get("model", "RP")),
            str(pipe.get("distribution", "weibull_2p")),
        ])

    story.append(
        _mk_table(
            rows,
            # large table → _mk_table va auto-scale si besoin
            [2.7 * cm, 1.4 * cm, 1.3 * cm, 1.5 * cm, 1.5 * cm, 1.8 * cm, 2.2 * cm, 3.0 * cm, 1.6 * cm, 2.0 * cm],
            font_size=7,  # un peu plus petit pour une synthèse large
        )
    )
    story.append(PageBreak())

    # ---- Pages par équipement
    for i, eq in enumerate(eqs):
        story.extend(_per_eq_section(eq, mdf, bundle))
        if i < len(eqs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(fpath)
