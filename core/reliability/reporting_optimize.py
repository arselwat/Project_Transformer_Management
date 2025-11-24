# core/reliability/reporting_optimize.py
from __future__ import annotations
from pathlib import Path
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet

# Sanitize pour polices Type1 par défaut (éviter apostrophes “ ’ ” etc.)
def SAN(s):
    s = str(s).replace("’","'").replace("“",'"').replace("”",'"').replace("–","-").replace("—","-").replace("\u00A0"," ")
    s = "".join(ch if ord(ch) <= 255 else " " for ch in s)
    return s

def _plot_R_curves(fits: dict, out_png: Path):
    """
    Trace R(t) = exp(-(t/eta)^beta) (gamma ignoré ici pour la lisibilité)
    Un seul PNG combiné pour tous les équipements.
    """
    if not fits:
        return None
    # borne temps: 0 → 1.2*max(eta)
    etas = [float(getattr(ft, "eta", 0.0)) for ft in fits.values()]
    tmax = max(etas) * 1.2 if etas else 1000.0
    t = np.linspace(0, max(tmax, 1.0), 300)

    plt.figure()
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", 1.0))
        eta  = float(getattr(ft, "eta", 1.0))
        y = np.exp(-((t/eta) ** beta))
        plt.plot(t, y, linewidth=2, label=f"{eq} (β={beta:.2f}, η={eta:.1f})")
    plt.grid(True, alpha=.3)
    plt.xlabel("Temps (h)")
    plt.ylabel("R(t)")
    plt.title("Courbes de fiabilité R(t)")
    plt.legend(fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    return str(out_png)

def export_optimization_report_pdf(df, fits: dict, intervals: dict, organigram_by_eq: dict, out_dir="reports") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = Path(out_dir) / f"full_report_{now}.pdf"

    # 1) tracer courbes
    png_path = Path(out_dir) / f"rt_curves_{now}.png"
    _plot_R_curves(fits, png_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=18*mm, bottomMargin=15*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []

    story.append(Paragraph(SAN("Rapport d'analyse & optimisation (organigramme de Ascher et Feingold)"), styles["Title"]))
    story.append(Paragraph(SAN(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["Normal"]))
    story.append(Spacer(1, 8))

    # TABLEAU synthèse
    story.append(Paragraph(SAN("Synthèse paramètres & intervalles"), styles["Heading2"]))
    data = [["Équipement","β","η (h)","Intervalle opt. (h)","Modèle","Loi","KS p","Chi2 p"]]
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", float("nan")))
        eta  = float(getattr(ft, "eta", float("nan")))
        itv  = intervals.get(eq)
        og   = organigram_by_eq.get(eq, {})
        model = og.get("model","?")
        loi   = og.get("distribution","?")
        fit   = og.get("fit", {})
        ks_p  = fit.get("ks_p","")
        chi_p = fit.get("chi2_p","")
        data.append([SAN(eq), f"{beta:.3f}", f"{eta:.1f}", f"{itv:.1f}" if isinstance(itv,(int,float)) else "", SAN(model), SAN(loi), str(ks_p), str(chi_p)])
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("GRID",(0,0),(-1,-1),0.3,colors.grey),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # COURBES
    if png_path.exists():
        story.append(Paragraph(SAN("Courbes de fiabilité R(t)"), styles["Heading2"]))
        story.append(Image(str(png_path), width=170*mm, height=95*mm))
        story.append(Spacer(1, 10))

    # DÉTAIL ORGANIGRAMME
    story.append(Paragraph(SAN("Détail de l'organigramme par équipement"), styles["Heading2"]))
    for eq, og in organigram_by_eq.items():
        story.append(Paragraph(SAN(f"• {eq}"), styles["Heading3"]))
        story.append(Paragraph(SAN(f"Modèle retenu : {og.get('model','?')}"), styles["Normal"]))
        story.append(Paragraph(SAN(f"Loi choisie : {og.get('distribution','?')}"), styles["Normal"]))
        det = og.get("details",{})
        story.append(Paragraph(SAN(f"Mann-Kendall (tendance) : {det.get('mann_kendall', False)}"), styles["Normal"]))
        story.append(Paragraph(SAN(f"Corrélation (dépendance) : {det.get('correlation', False)}"), styles["Normal"]))
        if og.get("distribution") == "weibull_min":
            story.append(Paragraph(SAN(f"β={og.get('beta'):.3f} | η={og.get('eta'):.1f} h | γ={og.get('gamma',0):.1f}"), styles["Normal"]))
        story.append(Spacer(1,5))

    doc.build(story)
    return str(out_path)
