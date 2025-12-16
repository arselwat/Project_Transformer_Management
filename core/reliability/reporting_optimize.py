# core/reliability/reporting_optimize.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

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
    """
    Sanitize texte (évite les guillemets/typos qui cassent certaines polices PDF).
    """
    s = str(s)
    s = (
        s.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00A0", " ")
    )
    # reportlab supporte unicode, mais on garde léger (évite surprises sur environnements)
    return s


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _fmt(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def _plot_R_curves_png(fits: Dict[str, Any], out_png: Path) -> Optional[str]:
    """
    Trace des courbes R(t) Weibull 2p pour lisibilité : R(t)=exp(-(t/eta)^beta)
    (si tu utilises Weibull 3p avec gamma, tu peux l’ajouter, mais l’affichage est
    plus clair en 2p pour un rapport synthèse).
    """
    if not fits:
        return None

    etas = [float(getattr(ft, "eta", 0.0) or 0.0) for ft in fits.values()]
    tmax = max(etas) * 1.2 if etas and max(etas) > 0 else 1000.0
    t = np.linspace(0, max(tmax, 1.0), 300)

    plt.figure()
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", 1.0) or 1.0)
        eta = float(getattr(ft, "eta", 1.0) or 1.0)
        if eta <= 0:
            continue
        y = np.exp(-((t / eta) ** beta))
        plt.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f})")

    plt.grid(True, alpha=0.3)
    plt.xlabel("Temps (h)")
    plt.ylabel("R(t)")
    plt.title("Courbes de fiabilité R(t) (Weibull)")
    plt.legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    return str(out_png)


def _build_synthesis_table(
    *,
    fits: Dict[str, Any],
    intervals_R: Optional[Dict[str, Any]] = None,
    organigram_by_eq: Optional[Dict[str, Any]] = None,
    df_out: Optional[Any] = None,
) -> list:
    """
    Construit un tableau 'data' pour ReportLab.
    Priorité:
      - si df_out est fourni => on utilise df_out (recommandé)
      - sinon fallback sur fits + intervals_R (+ organigram)
    """
    intervals_R = intervals_R or {}
    organigram_by_eq = organigram_by_eq or {}

    # Si df_out (DataFrame) fourni : on reprend les colonnes les plus utiles
    if df_out is not None:
        cols_pref = [
            "equipment_code",
            "beta",
            "eta_h",
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
        data = [cols]
        # limiter pour lisibilité (les rapports longs peuvent aller sur plusieurs pages)
        for _, r in df_out.iterrows():
            row = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    if c in ("beta", "R(T_cost)"):
                        row.append(_fmt(v, 3))
                    elif c.endswith("_h") or "eta" in c:
                        row.append(_fmt(v, 1))
                    else:
                        row.append(_fmt(v, 4))
                else:
                    row.append(_san(v))
            data.append(row)
        return data

    # Fallback minimal : fits + intervals_R + organigram
    data = [["Équipement", "β", "η (h)", "T_R (h)", "Modèle", "Loi"]]
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", float("nan")))
        eta = float(getattr(ft, "eta", float("nan")))
        itv = intervals_R.get(eq)
        og = organigram_by_eq.get(eq, {})
        model = og.get("model", "?")
        loi = og.get("distribution", "?")

        data.append([
            _san(eq),
            f"{beta:.3f}" if np.isfinite(beta) else "",
            f"{eta:.1f}" if np.isfinite(eta) else "",
            f"{float(itv):.1f}" if isinstance(itv, (int, float)) else "",
            _san(model),
            _san(loi),
        ])
    return data


def _apply_table_style(tbl):
    """
    Style pro de tableau.
    """
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),

        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),

        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))


def _require_reportlab():
    if not HAVE_REPORTLAB:
        raise RuntimeError(
            "ReportLab n’est pas disponible. Ajoute `reportlab` dans requirements.txt "
            "ou installe-le: pip install reportlab. "
            "Ensuite relance l’application."
        )


# ------------------------------------------------------------
# API 1 : Export PDF en mémoire (pour Streamlit download_button)
# ------------------------------------------------------------
def export_optimization_report_pdf_bytes(
    *,
    df,
    df_out=None,
    fits: Dict[str, Any] | None = None,
    intervals_R: Dict[str, Any] | None = None,
    organigram_by_eq: Dict[str, Any] | None = None,
    title: str = "Rapport d’analyse & optimisation de maintenance",
) -> bytes:
    """
    Génère un PDF en mémoire.
    - df : données brutes (equipment_code, ttf_h)
    - df_out : (optionnel) dataframe synthèse affiché dans l’UI (recommandé)
    - fits / intervals_R / organigram : fallback si df_out non fourni
    """
    _require_reportlab()

    fits = fits or {}
    intervals_R = intervals_R or {}
    organigram_by_eq = organigram_by_eq or {}

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

    story = []

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

    story.append(Paragraph(
        _san(
            "Rappel interprétation Weibull : β<1 (jeunesse), β≈1 (aléatoire), β>1 (usure). "
            "T_cost est orienté économie, T_R est orienté fiabilité."
        ),
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # --- Synthèse
    story.append(Paragraph("Synthèse paramètres & intervalles", styles["Heading2"]))
    data = _build_synthesis_table(
        fits=fits,
        intervals_R=intervals_R,
        organigram_by_eq=organigram_by_eq,
        df_out=df_out,
    )
    tbl = Table(data, repeatRows=1)
    _apply_table_style(tbl)
    story.append(tbl)
    story.append(Spacer(1, 12))

    # --- Courbes
    if fits:
        tmp_png = Path("reports") / f"rt_curves_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        png = _plot_R_curves_png(fits, tmp_png)
        if png and Path(png).exists():
            story.append(Paragraph("Courbes de fiabilité R(t)", styles["Heading2"]))
            story.append(Image(png, width=170 * mm, height=95 * mm))
            story.append(Spacer(1, 12))

    # --- Détails organigramme
    if organigram_by_eq:
        story.append(PageBreak())
        story.append(Paragraph("Détail de l’organigramme par équipement", styles["Heading2"]))
        story.append(Spacer(1, 8))

        for eq, og in organigram_by_eq.items():
            story.append(Paragraph(_san(f"{eq}"), styles["Heading3"]))
            story.append(Paragraph(_san(f"Modèle retenu : {og.get('model','?')}"), styles["Normal"]))
            story.append(Paragraph(_san(f"Loi choisie : {og.get('distribution','?')}"), styles["Normal"]))

            det = og.get("details", {}) or {}
            story.append(Paragraph(_san(f"Mann-Kendall (tendance) : {det.get('mann_kendall', False)}"), styles["Normal"]))
            story.append(Paragraph(_san(f"Corrélation (dépendance) : {det.get('correlation', False)}"), styles["Normal"]))

            fit = og.get("fit", {}) or {}
            if fit:
                ks_p = fit.get("ks_p", "")
                chi2_p = fit.get("chi2_p", "")
                story.append(Paragraph(_san(f"KS p-value : {ks_p} | Chi2 p-value : {chi2_p}"), styles["Normal"]))

            # si params présents
            if "beta" in og and "eta" in og:
                story.append(Paragraph(_san(f"Paramètres : β={_fmt(og.get('beta'),3)} | η={_fmt(og.get('eta'),1)} h"), styles["Normal"]))

            story.append(Spacer(1, 6))

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
) -> str:
    """
    Conserve ta signature existante. Écrit le PDF sur disque et retourne un chemin.
    """
    _require_reportlab()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = Path(out_dir) / f"full_report_{now}.pdf"

    # On génère en mémoire puis on écrit sur disque (plus stable)
    pdf_bytes = export_optimization_report_pdf_bytes(
        df=df,
        fits=fits,
        intervals_R=intervals,
        organigram_by_eq=organigram_by_eq,
        title="Rapport d’analyse & optimisation de maintenance",
    )
    out_path.write_bytes(pdf_bytes)
    return str(out_path)
