# core/inventory/reporting.py
from fpdf import FPDF
from pathlib import Path
import datetime
import unicodedata

def SAN(s):
    s = str(s)
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-")
           .replace("•", "-").replace("…", "...")
           .replace("\u00A0", " "))
    s = "".join(ch if ord(ch) <= 255 else " " for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("latin-1", "ignore").decode("latin-1", "ignore")
    return s

def export_stock_pdf(parts: list, out_dir="reports", title="Rapport de stock (pieces)") -> str:
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    date_now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = Path(out_dir) / f"stock_report_{date_now}.pdf"

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(0, 8, SAN(title), ln=1, align="C")
    pdf.ln(4)

    if not parts:
        pdf.cell(0, 8, SAN("Aucune piece trouvee dans le stock."), ln=1)
    else:
        headers = ["Code", "Nom", "Famille", "Quantite", "Seuil min", "Localisation", "Fournisseur"]
        col_widths = [25, 50, 30, 20, 20, 25, 35]

        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, SAN(h), border=1, align="C")
        pdf.ln()

        for p in parts:
            pdf.cell(col_widths[0], 6, SAN(p.get("code", "")), 1)
            pdf.cell(col_widths[1], 6, SAN(p.get("nom", "")), 1)
            pdf.cell(col_widths[2], 6, SAN(p.get("famille", "")), 1)
            pdf.cell(col_widths[3], 6, SAN(str(p.get("quantite", p.get("quantite_dispo", "")))), 1, align="R")
            pdf.cell(col_widths[4], 6, SAN(str(p.get("seuil_min", ""))), 1, align="R")
            pdf.cell(col_widths[5], 6, SAN(p.get("localisation", "")), 1)
            pdf.cell(col_widths[6], 6, SAN(p.get("fournisseur", "")), 1)
            pdf.ln()

    pdf.ln(8)
    pdf.cell(0, 8, SAN(f"Rapport genere le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=1)
    pdf.output(str(out_path))
    return str(out_path)
