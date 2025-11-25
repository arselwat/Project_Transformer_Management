# core/reliability/reporting_full.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import os
import math
import datetime

import numpy as np
import pandas as pd

# Matplotlib en mode "Agg" (pas d'UI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ReportLab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm  # selon ce qui était utilisé
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    )
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False
    A4 = (595.27, 841.89)
    colors = None
    cm = mm = 1.0
    def getSampleStyleSheet():
        raise RuntimeError("ReportLab non disponible")

try:
    from fpdf import FPDF
    HAVE_FPDF = True
except Exception:
    HAVE_FPDF = False

# ------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")

def _sanitize(s: Any) -> str:
    """
    Sanitize vers Latin-1/Helvetica: remplace quotes, tirets, puces,
    supprime emojis, garde ASCII/Latin-1 pour éviter les erreurs ReportLab.
    """
    s = str(s if s is not None else "")
    s = (
        s.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
         .replace("•", "-").replace("…", "...")
         .replace("\u00A0", " ")
    )
    s = "".join(ch if ord(ch) <= 255 else " " for ch in s)
    return s

def _compute_mtbf(ttf: np.ndarray) -> float | None:
    if ttf is None or len(ttf) == 0:
        return None
    return float(np.mean(ttf))

def _plot_and_save(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, out_file: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    ax.plot(x, y, linewidth=2)
    ax.set_title(_sanitize(title))
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel(_sanitize(ylabel))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_file))
    plt.close(fig)

def _plot_multi_curves(curves: Dict[str, np.ndarray], title: str, ylabel: str, out_file: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    for label, y in curves.items():
        ax.plot(y["t"], y["val"], linewidth=2, label=_sanitize(label))
    ax.set_title(_sanitize(title))
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel(_sanitize(ylabel))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_file))
    plt.close(fig)

# ------------------------------------------------------------
# API publique
# ------------------------------------------------------------

def export_full_report_pdf(
    df: pd.DataFrame,
    fits: Dict[str, Any],
    out_dir: str | Path = "reports",
    title: str = "Rapport — Indicateurs & Optimisation (Weibull, MTBF/MTTR)"
) -> str:
    """
    Génère un PDF complet à partir de:
      - df: DataFrame contenant au minimum 'equipment_code' et 'ttf_h'
      - fits: dict {equipment_code -> objet fit avec .beta, .eta, (optionnel .gamma)}
    Sortie: chemin absolu du PDF.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("export_full_report_pdf attend un DataFrame 'df' (et non un chemin).")

    if "equipment_code" not in df.columns or "ttf_h" not in df.columns:
        raise ValueError("df doit contenir les colonnes 'equipment_code' et 'ttf_h'.")

    out_dir = _ensure_dir(out_dir)
    img_dir = _ensure_dir(out_dir / f"imgs_{_ts()}")
    pdf_path = out_dir / f"full_report_{_ts()}.pdf"

    # ---------- Préparation données ----------
    # Table récap par équipement
    rows: List[List[str]] = [["Équipement", "N points", "MTBF (h)", "β (forme)", "η (échelle)", "γ (origine)"]]
    eq_list = []

    for eq, g in df.groupby("equipment_code"):
        ttf = pd.to_numeric(g["ttf_h"], errors="coerce").dropna().values
        n = len(ttf)
        if n == 0:
            continue
        mtbf = _compute_mtbf(ttf)
        ft = fits.get(eq)
        beta = float(getattr(ft, "beta", float("nan"))) if ft else float("nan")
        eta  = float(getattr(ft, "eta",  float("nan"))) if ft else float("nan")
        gamma = float(getattr(ft, "gamma", 0.0)) if ft else 0.0

        rows.append([
            _sanitize(eq),
            _sanitize(n),
            _sanitize(f"{mtbf:.1f}" if mtbf is not None else "n/a"),
            _sanitize(f"{beta:.2f}" if not math.isnan(beta) else "n/a"),
            _sanitize(f"{eta:.1f}" if not math.isnan(eta) else "n/a"),
            _sanitize(f"{gamma:.1f}")
        ])
        eq_list.append(str(eq))

    # ---------- Tracés multi-équipements ----------
    # t_max global (pour normaliser les échelles)
    ttf_all = pd.to_numeric(df["ttf_h"], errors="coerce").dropna()
    t_max = float(max(100.0, np.percentile(ttf_all, 95))) if len(ttf_all) else 100.0
    t = np.linspace(0.0, t_max, 400)

    curves_R, curves_F, curves_f, curves_h = {}, {}, {}, {}
    for eq in eq_list:
        ft = fits.get(eq)
        if not ft:
            continue
        beta = float(getattr(ft, "beta", float("nan")))
        eta  = float(getattr(ft, "eta",  float("nan")))
        gamma = float(getattr(ft, "gamma", 0.0))
        if not (beta > 0 and eta > 0):
            continue
        # Variables tronquées à t>=gamma
        tt = np.maximum(t - gamma, 0.0)
        R = np.exp(- (tt / eta) ** beta)
        F = 1.0 - R
        pdf = (beta / eta) * (tt / eta) ** (beta - 1.0) * np.exp(- (tt / eta) ** beta)
        # éviter division par zéro
        hazard = np.where(R > 1e-12, pdf / R, 0.0)

        label = f"{eq} (β={beta:.2f}, η={eta:.1f})"
        curves_R[label] = {"t": t, "val": R}
        curves_F[label] = {"t": t, "val": F}
        curves_f[label] = {"t": t, "val": pdf}
        curves_h[label] = {"t": t, "val": hazard}

    img_R = img_dir / "R.png"
    img_F = img_dir / "F.png"
    img_f = img_dir / "pdf.png"
    img_h = img_dir / "hazard.png"

    if curves_R:
        _plot_multi_curves(curves_R, "Fiabilité R(t)", "R(t)", img_R)
        _plot_multi_curves(curves_F, "Fonction de répartition F(t)", "F(t)", img_F)
        _plot_multi_curves(curves_f, "Densité f(t)", "f(t)", img_f)
        _plot_multi_curves(curves_h, "Taux de défaillance h(t)", "h(t)", img_h)

    # ---------- Construction PDF ----------
    styles = getSampleStyleSheet()
    story: List[Any] = []

    story.append(Paragraph(_sanitize(title), styles["Title"]))
    story.append(Spacer(1, 8))

    # Résumé
    story.append(Paragraph(_sanitize("Résumé des paramètres par équipement"), styles["Heading2"]))
    if len(rows) == 1:
        story.append(Paragraph(_sanitize("Aucune donnée suffisante pour le résumé."), styles["BodyText"]))
    else:
        tbl = Table(rows, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)

    story.append(Spacer(1, 10))

    # Courbes
    if curves_R:
        story.append(Paragraph(_sanitize("Courbes d’analyse"), styles["Heading2"]))
        # une image par ligne
        for path_img, legend in [
            (img_R, "Fiabilité R(t)"),
            (img_F, "Fonction de répartition F(t)"),
            (img_f, "Densité f(t)"),
            (img_h, "Taux de défaillance h(t)"),
        ]:
            if Path(path_img).exists():
                story.append(Paragraph(_sanitize(legend), styles["Heading3"]))
                story.append(Image(str(path_img), width=170*mm, height=90*mm))
                story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))
    story.append(Paragraph(_sanitize(f"Rapport généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["Normal"]))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        topMargin=20*mm, bottomMargin=15*mm, leftMargin=15*mm, rightMargin=15*mm
    )
    doc.build(story)

    return str(pdf_path)
