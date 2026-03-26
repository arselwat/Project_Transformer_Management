from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

try:
    from fpdf import FPDF
    HAVE_FPDF = True
except Exception:
    HAVE_FPDF = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _san(s: Any) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("°", " deg")
        .replace("\u00A0", " ")
    )



def _require_pdf():
    if not HAVE_REPORTLAB and not HAVE_FPDF:
        raise RuntimeError("Aucun moteur PDF disponible. Installe reportlab ou fpdf2.")



def _fmt(v: Any, nd: int = 2, dash: str = "—") -> str:
    try:
        if v is None:
            return dash
        fv = float(v)
        if pd.isna(fv):
            return dash
        return f"{fv:.{nd}f}"
    except Exception:
        return _san(v) if v not in (None, "") else dash



def _mk_table(data, col_widths=None, font_size=7):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        name="body_tbl",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.3)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head = ParagraphStyle(
        name="head_tbl",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(10, int(font_size * 1.35)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped = []
    for i, row in enumerate(data):
        style = head if i == 0 else body
        wrapped.append([Paragraph(_san(c).replace("\n", "<br/>"), style) for c in row])

    tbl = Table(wrapped, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#96AEE4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return tbl



def _df_to_table_data(df: pd.DataFrame, max_rows: Optional[int] = None):
    if df is None or df.empty:
        return [["Info"], ["Aucune donnée"]]
    df2 = df.copy()
    if max_rows is not None:
        df2 = df2.head(max_rows)
    data = [list(df2.columns)]
    for _, row in df2.iterrows():
        data.append([_fmt(v, 4) if isinstance(v, (int, float)) else _san(v) for v in row.tolist()])
    return data



def _fallback_fpdf(
    summary_df: pd.DataFrame,
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_eq: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: str | Path,
    title: str,
    meta: Optional[Dict[str, Any]],
) -> str:
    if not HAVE_FPDF:
        raise RuntimeError("FPDF indisponible et ReportLab indisponible.")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"global_analysis_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, _san(title))
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, _san(datetime.now().strftime("%d/%m/%Y %H:%M")), ln=1)
    pdf.ln(4)

    if meta:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Parametres de calcul", ln=1)
        pdf.set_font("Arial", "", 9)
        for k, v in meta.items():
            pdf.multi_cell(0, 5, f"- {_san(k)}: {_san(v)}")
        pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Decision finale - resume", ln=1)
    pdf.set_font("Arial", "", 9)
    if "final_decision" in global_tables and not global_tables["final_decision"].empty:
        for _, r in global_tables["final_decision"].head(20).iterrows():
            pdf.multi_cell(
                0,
                5,
                _san(
                    f"- {r.get('equipment_code','')} | priorite={r.get('priorite','')} | decision={r.get('decision_finale','')} | motif={r.get('motif_decision','')}"
                ),
            )
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Synthese globale", ln=1)
    pdf.set_font("Arial", "", 9)
    for _, r in summary_df.head(30).iterrows():
        pdf.multi_cell(
            0,
            5,
            _san(
                f"- {r.get('equipment_code','')} | model={r.get('model','')} | loi={r.get('distribution','')} | beta={_fmt(r.get('beta'),2)} | thermique={r.get('thermal_status','')} | score={r.get('priority_score','')}"
            ),
        )

    if detail_tables_by_eq:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Par equipement", ln=1)
        pdf.set_font("Arial", "", 9)
        for eq, tables in list(detail_tables_by_eq.items())[:15]:
            pdf.multi_cell(0, 5, _san(f"[{eq}]"))
            rs = tables.get("reliability_summary")
            if isinstance(rs, pd.DataFrame) and not rs.empty:
                rr = rs.iloc[0].to_dict()
                pdf.multi_cell(
                    0,
                    5,
                    _san(
                        f"Processus={rr.get('Processus','')} | Distribution={rr.get('Distribution','')} | MTBF={_fmt(rr.get('MTBF (h)'),1)} h | MTTR={_fmt(rr.get('MTTR (h)'),1)} h | Dispo={_fmt(rr.get('Disponibilité'),3)}"
                    ),
                )
            ts = tables.get("thermal_summary")
            if isinstance(ts, pd.DataFrame) and not ts.empty:
                tr = ts.iloc[0].to_dict()
                pdf.multi_cell(
                    0,
                    5,
                    _san(
                        f"Thermique: thetaHS max={_fmt(tr.get('θHS max'),1)} | FAA max={_fmt(tr.get('FAA max'),3)} | LOL={_fmt(tr.get('Loss of life (%)'),3)} %"
                    ),
                )
            pdf.ln(1)

    pdf.output(str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def export_global_analysis_report_pdf(
    summary_df: pd.DataFrame,
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_eq: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: str | Path = "reports",
    title: str = "Résultat analyse / optimisation / maintenance",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    _require_pdf()

    if not HAVE_REPORTLAB:
        return _fallback_fpdf(summary_df, global_tables, detail_tables_by_eq, out_dir, title, meta)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"global_analysis_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    styles = getSampleStyleSheet()
    story = []

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
    )

    # Cover / executive summary
    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 10))

    if meta:
        story.append(Paragraph("Paramètres de calcul", styles["Heading2"]))
        for k, v in meta.items():
            story.append(Paragraph(_san(f"- {k}: {v}"), styles["Normal"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Lecture exécutive", styles["Heading2"]))
    if summary_df is not None and not summary_df.empty:
        critical = int((summary_df["priorite"].astype(str) == "Critique").sum()) if "priorite" in summary_df.columns else 0
        nhpp = int((summary_df["model"].astype(str).str.upper() == "NHPP").sum()) if "model" in summary_df.columns else 0
        thermal_critical = int((summary_df["thermal_status"].astype(str) == "Critique").sum()) if "thermal_status" in summary_df.columns else 0
        story.append(Paragraph(_san(f"- Équipements analysés : {len(summary_df)}"), styles["Normal"]))
        story.append(Paragraph(_san(f"- Priorité critique : {critical}"), styles["Normal"]))
        story.append(Paragraph(_san(f"- NHPP détectés : {nhpp}"), styles["Normal"]))
        story.append(Paragraph(_san(f"- Statuts thermiques critiques : {thermal_critical}"), styles["Normal"]))
    story.append(Spacer(1, 8))

    # Global tables
    ordered_global = [
        ("final_decision", "Décision finale hiérarchisée"),
        ("global_summary", "Synthèse globale"),
        ("trend_overview", "Vue d'ensemble des tests"),
        ("risk_overview", "Vue d'ensemble risque & thermique"),
        ("optimization_overview", "Vue d'ensemble optimisation & maintenance"),
        ("due_tasks", "Tâches dues"),
    ]
    for key, label in ordered_global:
        df = global_tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            story.append(Paragraph(label, styles["Heading2"]))
            story.append(_mk_table(_df_to_table_data(df, max_rows=40), font_size=7))
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Detailed per equipment sections
    story.append(Paragraph("Détail par équipement", styles["Heading1"]))
    story.append(Spacer(1, 6))
    for eq, tables in detail_tables_by_eq.items():
        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))

        for key, label in [
            ("trend_results", "Tests de tendance"),
            ("dependence_results", "Tests de dépendance"),
            ("process_choice", "Choix du processus"),
            ("fit_candidates", "Modèles candidats"),
            ("reliability_summary", "Synthèse fiabiliste"),
            ("thermal_summary", "Synthèse thermique"),
            ("thermal_top5_days", "Top jours critiques"),
        ]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                story.append(Paragraph(label, styles["Heading3"]))
                max_rows = 20 if key not in ("fit_candidates", "thermal_top5_days") else 10
                story.append(_mk_table(_df_to_table_data(df, max_rows=max_rows), font_size=7))
                story.append(Spacer(1, 6))

        story.append(PageBreak())

    doc.build(story)
    return str(out_path)
