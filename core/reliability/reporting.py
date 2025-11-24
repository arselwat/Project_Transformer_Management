from fpdf import FPDF
from datetime import datetime
import os
import pandas as pd
from typing import Optional, Tuple, List

def _fmt2(val) -> str:
    try:
        if val is None:
            return "n/a"
        return f"{float(val):.2f}"
    except Exception:
        return "n/a"

def export_reliability_pdf(
    df: pd.DataFrame,
    metrics: dict,
    weibull_fit: Optional[Tuple[float, float]],
    images: Optional[List[str]] = None
) -> str:
    os.makedirs("reports", exist_ok=True)
    out = f"reports/rapport_fiabilite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    mtbf = metrics["global"]["MTBF"]
    mttr = metrics["global"]["MTTR"]
    per_eq = metrics["per_equipment"]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Rapport de Fiabilité", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(3)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Lignes de pannes: {len(df)}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Indicateurs globaux:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"MTBF (h): {_fmt2(mtbf)}", ln=True)
    pdf.cell(0, 8, f"MTTR (h): {_fmt2(mttr)}", ln=True)
    pdf.ln(4)

    if weibull_fit:
        beta, eta = weibull_fit
        pdf.cell(0, 8, f"Ajustement Weibull: beta={_fmt2(beta)}, eta={_fmt2(eta)}", ln=True)
    else:
        pdf.cell(0, 8, "Ajustement Weibull: non disponible", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Par équipement:", ln=True)
    pdf.set_font("Arial", "", 11)
    if per_eq:
        for eq, vals in per_eq.items():
            line = f"- {eq}: MTBF={_fmt2(vals.get('MTBF'))} / MTTR={_fmt2(vals.get('MTTR'))}"
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "- Aucun détail équipement", ln=True)

    if images:
        for img in images:
            try:
                if not os.path.exists(img):
                    continue
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, os.path.basename(img), ln=True, align="C")
                pdf.ln(3)
                pdf.image(img, x=15, y=None, w=180)
            except Exception:
                pass

    pdf.output(out)
    return out
