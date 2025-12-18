# core/reliability/reporting_optimize.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ReportLab (recommandé)
# ------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
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
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _san(s: Any) -> str:
    """Sanitize texte (évite certaines typos qui cassent certains PDF)."""
    s = "" if s is None else str(s)
    return (
        s.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00A0", " ")
    )
def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None
def _fmt(x: Any, nd: int = 2, dash: str = "—") -> str:
    v = _safe_float(x)
    if v is None:
        return dash
    return f"{v:.{nd}f}"
def _require_reportlab():
    if not HAVE_REPORTLAB:
        raise RuntimeError(
            "ReportLab n’est pas disponible. Ajoute `reportlab` dans requirements.txt "
            "ou installe-le: pip install reportlab. Ensuite relance l’application."
        )
def _plot_R_curves_png(fits: Dict[str, Any], out_png: Path) -> Optional[str]:
    """
    Courbes R(t).
    Supporte Weibull 2p et 3p (si gamma existe).
    """
    if not fits:
        return None

    etas = [float(getattr(ft, "eta", 0.0) or 0.0) for ft in fits.values()]
    tmax = max(etas) * 1.6 if etas and max(etas) > 0 else 1000.0
    t = np.linspace(0, max(tmax, 1.0), 350)

    plt.figure()
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", 1.0) or 1.0)
        eta = float(getattr(ft, "eta", 1.0) or 1.0)
        gamma = float(getattr(ft, "gamma", 0.0) or 0.0)
        if eta <= 0 or beta <= 0:
            continue

        y = np.ones_like(t, dtype=float)
        mask = t > gamma
        y[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-12)) ** beta))

        plt.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f}, γ={gamma:.1f})")

    plt.grid(True, alpha=0.3)
    plt.xlabel("Temps (h)")
    plt.ylabel("R(t)")
    plt.title("Courbes de fiabilité R(t) (Weibull)")
    plt.legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    return str(out_png)


def _mk_table(
    data: List[List[Any]],
    *,
    avail_width: Optional[float] = None,
    font_size: int = 8,
    repeat_header: bool = True,
) -> "Table":
    """
    Table robuste:
    - wrap automatique via Paragraph
    - scale automatique si la somme des colWidths dépasse la largeur dispo (A4)
    - repeatRows=1 pour répéter l'entête sur pages suivantes
    """
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"]
    cell_style.fontName = "Helvetica"
    cell_style.fontSize = font_size
    cell_style.leading = font_size + 2

    # largeur dispo: A4 - marges (doit correspondre à celles du doc)
    if avail_width is None:
        avail_width = A4[0] - (14 * mm + 14 * mm)

    wrapped: List[List[Any]] = []
    for r_i, row in enumerate(data):
        new_row = []
        for c in row:
            # si déjà un flowable (Image/Table/Paragraph), on garde
            if hasattr(c, "wrapOn"):
                new_row.append(c)
            else:
                new_row.append(Paragraph(_san(c), cell_style))
        wrapped.append(new_row)

    ncol = len(wrapped[0]) if wrapped else 1
    col_widths = [avail_width / max(ncol, 1)] * ncol

    tbl = Table(wrapped, colWidths=col_widths, repeatRows=1 if repeat_header else 0)

    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#96AEE4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),

        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),

        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def _maintenance_type(beta: float) -> str:
    if not np.isfinite(beta):
        return "—"
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


def _build_synthesis_table(
    *,
    fits: Dict[str, Any],
    intervals: Optional[Dict[str, Any]] = None,
    organigram_by_eq: Optional[Dict[str, Any]] = None,
    df_out: Optional[Any] = None,
) -> List[List[Any]]:
    """
    Table synthèse.
    - Si df_out fourni => on l'utilise directement (meilleur)
    - Sinon fallback sur fits + intervals dict
    intervals peut être:
      - dict[eq] = {"T_R":..., "T_cost":..., "R_at_T":..., "C_min":..., ...}
      - ou dict[eq] = valeur T_R (ancien format)
    """
    intervals = intervals or {}
    organigram_by_eq = organigram_by_eq or {}

    if df_out is not None:
        cols_pref = [
            "equipment_code",
            "beta",
            "eta_h",
            "gamma_h",
            "T_cost_h",
            "R(T_cost)",
            "C_min_per_h",
            "T_R_h",
            "T_recommended_h",
            "maintenance_type",
            "model",
            "distribution",
        ]
        cols = [c for c in cols_pref if c in getattr(df_out, "columns", [])]
        data: List[List[Any]] = [cols]
        for _, r in df_out.iterrows():
            row = []
            for c in cols:
                v = r.get(c, "")
                # format cohérent
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    if c in ("beta", "R(T_cost)"):
                        row.append(_fmt(v, 3))
                    elif c.endswith("_h") or "eta" in c or "gamma" in c:
                        row.append(_fmt(v, 1))
                    elif "C_min" in c:
                        row.append(_fmt(v, 4))
                    else:
                        row.append(_fmt(v, 2))
                else:
                    row.append(_san(v))
            data.append(row)
        return data

    # fallback
    head = ["Équipement", "β", "η(h)", "γ(h)", "T_R(h)", "T_cost(h)", "R(T_cost)", "C_min(/h)", "Type", "Modèle", "Loi"]
    data = [head]

    for eq, ft in fits.items():
        beta = _safe_float(getattr(ft, "beta", None))
        eta = _safe_float(getattr(ft, "eta", None))
        gamma = _safe_float(getattr(ft, "gamma", 0.0)) or 0.0

        itv = intervals.get(eq)
        # ancien format => itv est une valeur
        if isinstance(itv, (int, float)):
            T_R = itv
            T_cost = None
            R_at_T = None
            C_min = None
        # nouveau format => itv est dict
        elif isinstance(itv, dict):
            T_R = itv.get("T_R")
            T_cost = itv.get("T_cost")
            R_at_T = itv.get("R_at_T")
            C_min = itv.get("C_min")
        else:
            T_R = None
            T_cost = None
            R_at_T = None
            C_min = None

        og = organigram_by_eq.get(eq, {}) or {}
        model = og.get("model", "?")
        loi = og.get("distribution", "?")

        data.append([
            _san(eq),
            _fmt(beta, 3),
            _fmt(eta, 1),
            _fmt(gamma, 1),
            _fmt(T_R, 1),
            _fmt(T_cost, 1),
            _fmt(R_at_T, 3),
            _fmt(C_min, 4),
            _maintenance_type(beta if beta is not None else float("nan")),
            _san(model),
            _san(loi),
        ])

    return data


# ------------------------------------------------------------
# API 1 : Export PDF en mémoire (Streamlit download_button)
# ------------------------------------------------------------
def export_optimization_report_pdf_bytes(
    *,
    df,
    df_out=None,
    fits: Dict[str, Any] | None = None,
    intervals: Dict[str, Any] | None = None,
    organigram_by_eq: Dict[str, Any] | None = None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Rapport d’analyse & optimisation de maintenance",
) -> bytes:
    """
    Génère un PDF en mémoire.
    - df : données brutes (equipment_code, ttf_h)
    - df_out : dataframe synthèse UI (recommandé)
    - fits / intervals / organigram_by_eq : fallback si df_out non fourni
    - meta : paramètres (R_target, C_prev, C_corr, R_min_cost, etc.)
    """
    _require_reportlab()

    fits = fits or {}
    intervals = intervals or {}
    organigram_by_eq = organigram_by_eq or {}
    meta = meta or {}

    styles = getSampleStyleSheet()
    buff = BytesIO()

    doc = SimpleDocTemplate(
        buff,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        title=_san(title),
        author="Reliability Module",
    )

    story: List[Any] = []

    # --- Couverture
    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["Normal"]))
    story.append(Spacer(1, 10))

    # --- Résumé
    nb_obs = int(getattr(df, "shape", [0])[0]) if df is not None else 0
    nb_eq = int(getattr(df_out, "shape", [0])[0]) if df_out is not None else len(fits)

    story.append(Paragraph("Résumé", styles["Heading2"]))
    story.append(Paragraph(_san(f"- Nombre d’équipements analysés : {nb_eq}"), styles["Normal"]))
    story.append(Paragraph(_san(f"- Nombre d’observations TTF : {nb_obs}"), styles["Normal"]))
    story.append(Spacer(1, 8))

    # --- Paramètres d'optimisation (meta)
    if meta:
        story.append(Paragraph("Paramètres d’optimisation", styles["Heading2"]))
        lines = []
        if "R_target" in meta:
            lines.append(f"- Fiabilité cible R_target : {meta.get('R_target')}")
        if "C_prev" in meta:
            lines.append(f"- Coût préventif C_prev : {meta.get('C_prev')}")
        if "C_corr" in meta:
            lines.append(f"- Coût correctif C_corr : {meta.get('C_corr')}")
        if "R_min_cost" in meta:
            lines.append(f"- Fiabilité minimale pour l’optimum coût R_min_cost : {meta.get('R_min_cost')}")
        if lines:
            for l in lines:
                story.append(Paragraph(_san(l), styles["Normal"]))
            story.append(Spacer(1, 8))

    story.append(Paragraph(
        _san(
            "Rappel Weibull : β<1 (jeunesse), β≈1 (aléatoire), β>1 (usure). "
            "T_cost est orienté économie, T_R est orienté fiabilité. "
            "Sur un équipement critique, on privilégie souvent T_R ou on impose un R_min élevé."
        ),
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # --- Synthèse
    story.append(Paragraph("Synthèse paramètres & intervalles", styles["Heading2"]))
    data = _build_synthesis_table(
        fits=fits,
        intervals=intervals,
        organigram_by_eq=organigram_by_eq,
        df_out=df_out,
    )
    story.append(_mk_table(data, font_size=7))  # 7 => plus compact pour éviter dépassement A4
    story.append(Spacer(1, 12))

    # --- Courbes
    if fits:
        tmp_png = Path("reports") / f"rt_curves_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        png = _plot_R_curves_png(fits, tmp_png)
        if png and Path(png).exists():
            story.append(Paragraph("Courbes de fiabilité R(t)", styles["Heading2"]))
            story.append(Image(png, width=170 * mm, height=95 * mm))
            story.append(Spacer(1, 12))

    # --- Détails par équipement (optimisation)
    story.append(PageBreak())
    story.append(Paragraph("Détails — Optimisation par équipement", styles["Heading2"]))
    story.append(Spacer(1, 8))

    eqs = list(fits.keys())
    for eq in eqs:
        ft = fits.get(eq)
        og = (organigram_by_eq.get(eq, {}) or {})
        itv = intervals.get(eq)

        beta = _safe_float(getattr(ft, "beta", None))
        eta = _safe_float(getattr(ft, "eta", None))
        gamma = _safe_float(getattr(ft, "gamma", 0.0)) or 0.0

        # lecture intervalle (nouveau format dict / ancien float)
        if isinstance(itv, dict):
            T_R = itv.get("T_R")
            T_cost = itv.get("T_cost")
            R_at_T = itv.get("R_at_T")
            C_min = itv.get("C_min")
        else:
            T_R = itv if isinstance(itv, (int, float)) else None
            T_cost = None
            R_at_T = None
            C_min = None

        story.append(Paragraph(_san(f"{eq}"), styles["Heading3"]))
        story.append(Paragraph(_san(f"Paramètres Weibull : β={_fmt(beta,3)} ; η={_fmt(eta,1)} h ; γ={_fmt(gamma,1)} h"), styles["Normal"]))
        story.append(Paragraph(_san(f"Type maintenance (β) : {_maintenance_type(beta if beta is not None else float('nan'))}"), styles["Normal"]))
        story.append(Spacer(1, 4))

        opt_lines = [
            ["Indicateur", "Valeur"],
            ["T_R (fiabilité cible)", _fmt(T_R, 1)],
            ["T_cost (optimum économique)", _fmt(T_cost, 1)],
            ["R(T_cost)", _fmt(R_at_T, 3)],
            ["C_min (/h)", _fmt(C_min, 4)],
            ["Modèle (organigramme)", _san(og.get("model", "?"))],
            ["Loi (organigramme)", _san(og.get("distribution", "?"))],
        ]
        story.append(_mk_table(opt_lines, font_size=8))
        story.append(Spacer(1, 6))

        # Trace organigramme (format de analyze_ttf_pipeline)
        story.append(Paragraph(_san("Trace organigramme"), styles["Heading4"]))
        story.append(Paragraph(_san(_pipe_line(og)), styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buff.getvalue()


# ------------------------------------------------------------
# API 2 : Export PDF sur disque (compatibilité existante)
# ------------------------------------------------------------
def export_optimization_report_pdf(
    df,
    fits: Dict[str, Any],
    intervals: Dict[str, Any],
    organigram_by_eq: Dict[str, Any],
    out_dir: str = "reports",
    df_out=None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Rapport d’analyse & optimisation de maintenance",
    # alias acceptés (au cas où)
    org_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Écrit le PDF sur disque et retourne un chemin.

    ✅ Signature compatible avec ton appel Streamlit :
      export_optimization_report_pdf(df, fits, intervals, organigram_by_eq, out_dir="reports")

    ✅ Et accepte aussi org_results=... en alias (si besoin).
    """
    _require_reportlab()

    if organigram_by_eq is None and org_results is not None:
        organigram_by_eq = org_results
    organigram_by_eq = organigram_by_eq or {}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = Path(out_dir) / f"full_report_{now}.pdf"

    pdf_bytes = export_optimization_report_pdf_bytes(
        df=df,
        df_out=df_out,
        fits=fits,
        intervals=intervals,
        organigram_by_eq=organigram_by_eq,
        meta=meta,
        title=title,
    )
    out_path.write_bytes(pdf_bytes)
    return str(out_path)
