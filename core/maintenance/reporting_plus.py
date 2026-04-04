from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime
import math
import unicodedata

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False
    A4 = (595.27, 841.89)
    colors = None
    mm = 1.0
    cm = 10.0

try:
    from fpdf import FPDF
    HAVE_FPDF = True
except Exception:
    HAVE_FPDF = False


SPARE_PARTS: List[Dict[str, Any]] = [
    {"categorie": "Étanchéité & cuve", "piece": "Joints plats de cuve", "qte_reco": "1 jeu", "criticite": "Élevée", "remarques": "Pour interventions lourdes"},
    {"categorie": "Isolation & traversées", "piece": "Traversée HT", "qte_reco": "1 unité", "criticite": "Élevée", "remarques": "Pièce critique"},
    {"categorie": "OLTC", "piece": "Contacts mobiles OLTC", "qte_reco": "1 kit", "criticite": "Très élevée", "remarques": "Révision lourde"},
    {"categorie": "Protection & mesure", "piece": "Relais Buchholz", "qte_reco": "1 unité", "criticite": "Élevée", "remarques": "Pièce de sécurité"},
    {"categorie": "Consommables", "piece": "Huile isolante neuve", "qte_reco": "1 fût / IBC", "criticite": "Très élevée", "remarques": "Appoint ou traitement"},
]

DEFAULT_TOOLS = [
    {"categorie": "Mesure & essais", "outil": "Micro-ohmmètre", "description": "Résistance d’enroulements", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & essais", "outil": "Mégohmmètre 5 kV", "description": "Résistance d’isolement", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Outils", "outil": "Clés dynamométriques", "description": "Serrage au couple", "qte": 1, "unite": "set", "calibrage": "OK", "remarques": ""},
    {"categorie": "Sécurité", "outil": "EPI complets", "description": "Gants, casque, visière, tenue", "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
]


def SAN(value: Any) -> str:
    text = "" if value is None else str(value)
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("•", "-").replace("…", "...")
        .replace("≤", "<=").replace("≥", ">=")
        .replace("β", "beta").replace("η", "eta").replace("γ", "gamma")
        .replace("\u00A0", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1", "ignore")


def fnum(v: Any, nd: int = 1, default: str = "-") -> str:
    try:
        if v is None:
            return default
        numeric = float(v)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return f"{numeric:.{nd}f}"
    except Exception:
        return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        numeric = float(v)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def _compact(text: Any, max_len: int = 140) -> str:
    s = SAN(text).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _page_available_width(left_margin_mm: float = 16, right_margin_mm: float = 16) -> float:
    return A4[0] - (left_margin_mm * mm + right_margin_mm * mm)


def _scale_widths(widths, available_width):
    if not widths:
        return widths
    total = sum(widths)
    if total <= 0 or total <= available_width:
        return widths
    factor = available_width / total
    return [w * factor for w in widths]


def _auto_widths_from_data(data, available_width, min_col_mm=16, max_col_mm=70):
    if not data:
        return None
    ncols = max(len(r) for r in data)
    lengths = [8] * ncols
    for row in data[:40]:
        for idx in range(ncols):
            cell = row[idx] if idx < len(row) else ""
            lengths[idx] = max(lengths[idx], min(len(str(cell)), 60))
    total = sum(lengths) if sum(lengths) > 0 else ncols
    widths = [(length / total) * available_width for length in lengths]
    widths = [max(min_col_mm * mm, min(max_col_mm * mm, w)) for w in widths]
    return _scale_widths(widths, available_width)


def _mk_table(data, widths=None, font_size=8, available_width=None):
    if available_width is None:
        available_width = _page_available_width()

    body_style = ParagraphStyle(
        name=f"tbl_body_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.35)),
        alignment=TA_LEFT,
        wordWrap="CJK",
        splitLongWords=True,
    )
    head_style = ParagraphStyle(
        name=f"tbl_head_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=max(8, font_size),
        leading=max(10, int(font_size * 1.25)),
        alignment=TA_CENTER,
        wordWrap="CJK",
        splitLongWords=True,
    )

    def to_para(x, style):
        return Paragraph(SAN(x).replace("\n", "<br/>"), style)

    wrapped = []
    for row_idx, row in enumerate(data):
        style = head_style if row_idx == 0 else body_style
        wrapped.append([to_para(cell, style) for cell in row])

    if widths is None:
        widths = _auto_widths_from_data(data, available_width)
    else:
        widths = _scale_widths(widths, available_width)

    tbl = Table(wrapped, repeatRows=1, colWidths=widths, splitByRow=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("BOX", (0, 0), (-1, -1), 1.1, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    return tbl


def _title(story, styles, text: str, level: int = 2, space_after_pt: int = 6, color=None):
    style_name = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
    if color is None:
        story.append(Paragraph(SAN(text), styles[style_name]))
    else:
        custom = ParagraphStyle(
            name=f"custom_{style_name}_{text}",
            parent=styles[style_name],
            textColor=color,
        )
        story.append(Paragraph(SAN(text), custom))
    story.append(Spacer(1, space_after_pt))


def _build_choice_explanation(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("maintenance_choice"):
        parts.append(f"Choix final retenu : {SAN(row.get('maintenance_choice'))}.")
    if row.get("maintenance_choice_reason"):
        parts.append(f"Justification : {SAN(row.get('maintenance_choice_reason'))}.")
    if row.get("model") or row.get("distribution"):
        parts.append(f"Le processus est {SAN(row.get('model'))} avec la loi {SAN(row.get('distribution'))}.")
    if row.get("beta") is not None:
        parts.append(f"Le paramètre beta = {fnum(row.get('beta'), 2)} influence directement le type de maintenance.")
    if row.get("eta_h") is not None:
        parts.append(f"La durée caractéristique eta vaut {fnum(row.get('eta_h'), 1)} h.")
    if row.get("T_recommended_h") is not None:
        parts.append(f"L’intervalle recommandé est {fnum(row.get('T_recommended_h'), 1)} h.")
    if row.get("days_left") is not None:
        parts.append(f"Il reste {row.get('days_left')} jours avant l’échéance calculée.")
    if row.get("final_decision"):
        parts.append(f"Décision finale : {SAN(row.get('final_decision'))}.")
    if row.get("final_reason"):
        parts.append(f"Motif : {SAN(row.get('final_reason'))}.")
    return " ".join(parts) if parts else "Aucune explication détaillée disponible."


def _build_influence_table(row: Dict[str, Any]):
    return [
        ["Paramètre", "Valeur", "Impact sur le choix"],
        ["Choix final", SAN(row.get("maintenance_choice", "")), "Type de maintenance retenu après synthèse des facteurs."],
        ["Type initial", SAN(row.get("maintenance_type", "")), "Suggestion initiale provenant de l’optimisation."],
        ["Modèle", SAN(row.get("model", "")), "Le comportement global des défaillances influence la stratégie retenue."],
        ["Loi", SAN(row.get("distribution", "")), "La loi retenue structure l’estimation des durées et du risque."],
        ["beta", fnum(row.get("beta"), 2), "Indique défauts précoces, comportement aléatoire ou usure."],
        ["eta (h)", fnum(row.get("eta_h"), 1), "Référence de durée de vie caractéristique."],
        ["gamma (h)", fnum(row.get("gamma_h"), 1), "Décalage éventuel du modèle."],
        ["T_recommended (h)", fnum(row.get("T_recommended_h"), 1), "Intervalle principal proposé."],
        ["T_R (h)", fnum(row.get("T_R_h"), 1), "Intervalle issu du critère de fiabilité."],
        ["T_cost (h)", fnum(row.get("T_cost_h"), 1), "Intervalle issu du critère économique."],
        ["Jours restants", SAN(row.get("days_left", "")), "Plus l’échéance est proche, plus la priorité augmente."],
        ["Niveau de priorité", SAN(row.get("priority_level", "")), "Les cas critiques sont mis en évidence en rouge."],
    ]


def _apply_critical_style(table, data: List[List[Any]], priority_col: Optional[int] = None):
    if not HAVE_REPORTLAB or colors is None:
        return table
    styles = []
    for idx, row in enumerate(data[1:], start=1):
        try:
            target = str(row[priority_col]).strip().lower() if priority_col is not None and priority_col < len(row) else ""
            if target == "critique":
                styles.extend([
                    ("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#fff1f2")),
                    ("TEXTCOLOR", (0, idx), (-1, idx), colors.HexColor("#b91c1c")),
                    ("FONTNAME", (0, idx), (-1, idx), "Helvetica-Bold"),
                ])
        except Exception:
            pass
    if styles:
        table.setStyle(TableStyle(styles))
    return table


def _tasks_table(tasks_due: List[Dict[str, Any]], available_width: float):
    if not tasks_due:
        return None
    data = [["Équipement", "Choix final", "Priorité", "Échéance", "Jours restants", "Intervalle (h)", "Décision"]]
    for task in tasks_due:
        data.append([
            SAN(task.get("equipment_code", "")),
            SAN(task.get("maintenance_choice", task.get("maintenance_type", ""))),
            SAN(task.get("priority_level", "")),
            SAN(task.get("next_due_date", "")),
            SAN(task.get("days_left", "")),
            fnum(task.get("interval_h"), 1),
            SAN(task.get("final_decision", "")),
        ])
    table = _mk_table(data, available_width=available_width, font_size=7.5)
    return _apply_critical_style(table, data, priority_col=2)


def _summary_table(metrics_table: List[Dict[str, Any]], available_width: float):
    data = [["Équipement", "Choix final", "Modèle", "Loi", "beta", "eta (h)", "Intervalle (h)", "Jours", "Priorité"]]
    for row in metrics_table:
        data.append([
            SAN(row.get("equipment_code", "")),
            SAN(row.get("maintenance_choice", row.get("maintenance_type", ""))),
            SAN(row.get("model", "")),
            SAN(row.get("distribution", "")),
            fnum(row.get("beta"), 2),
            fnum(row.get("eta_h"), 1),
            fnum(row.get("interval_h", row.get("T_recommended_h")), 1),
            SAN(row.get("days_left", "")),
            SAN(row.get("priority_level", "")),
        ])
    table = _mk_table(data, available_width=available_width, font_size=7.3)
    return _apply_critical_style(table, data, priority_col=8)


def _add_reference_sections(story, styles, available_width: float, tools_checklist):
    _title(story, styles, "Références matérielles", level=2, space_after_pt=4)

    spare_data = [["Catégorie", "Pièce", "Qté recommandée", "Criticité", "Remarques"]]
    for item in SPARE_PARTS:
        spare_data.append([
            SAN(item["categorie"]),
            SAN(item["piece"]),
            SAN(item["qte_reco"]),
            SAN(item["criticite"]),
            SAN(item["remarques"]),
        ])
    story.append(_mk_table(spare_data, available_width=available_width, font_size=7))
    story.append(Spacer(1, 8))

    tools = tools_checklist if (isinstance(tools_checklist, list) and tools_checklist) else DEFAULT_TOOLS
    tools_data = [["Catégorie", "Outil", "Description", "Qté", "Unité", "État"]]
    for item in tools:
        tools_data.append([
            SAN(item.get("categorie", "")),
            SAN(item.get("outil", "")),
            SAN(item.get("description", "")),
            SAN(item.get("qte", "")),
            SAN(item.get("unite", "")),
            SAN(item.get("calibrage", "")),
        ])
    story.append(_mk_table(tools_data, available_width=available_width, font_size=7))
    story.append(Spacer(1, 8))


def _export_pm_plan_with_kits_pdf_fallback(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]],
    metrics_table: List[Dict[str, Any]],
    out_dir: str | Path,
    title: str,
    tools_checklist: List[Dict[str, Any]] | None,
) -> str:
    if not HAVE_FPDF:
        raise RuntimeError("Génération PDF non disponible : ReportLab et FPDF sont indisponibles.")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"pm_plan_kits_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, SAN(title))
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, SAN(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")), ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, SAN("Résumé"), ln=1)
    pdf.set_font("Arial", "", 9)
    for row in metrics_table[:25]:
        priority = str(row.get("priority_level", ""))
        label = "[CRITIQUE]" if priority == "Critique" else "-"
        pdf.multi_cell(
            0, 5,
            SAN(
                f"{label} {row.get('equipment_code', '')} | choix={row.get('maintenance_choice', row.get('maintenance_type', ''))} | "
                f"beta={fnum(row.get('beta'), 2)} | jours={row.get('days_left', '')}"
            ),
        )
    pdf.ln(2)

    if tasks_due:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, SAN("Tâches dues"), ln=1)
        pdf.set_font("Arial", "", 9)
        for task in tasks_due[:40]:
            priority = str(task.get("priority_level", ""))
            label = "[CRITIQUE]" if priority == "Critique" else "-"
            pdf.multi_cell(
                0,
                5,
                SAN(
                    f"{label} [{task.get('equipment_code', '')}] {task.get('maintenance_choice', task.get('maintenance_type', ''))} "
                    f"| échéance : {task.get('next_due_date', '')}"
                ),
            )

    pdf.output(str(out_path))
    return str(out_path)


def export_pm_plan_with_kits_pdf(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]],
    metrics_table: List[Dict[str, Any]],
    out_dir: str | Path = "reports",
    title: str = "Plan de maintenance - Procédure, tâches, matériels",
    procedure_docx: str | Path | None = None,
    *,
    include_kits: bool = False,
    tools_checklist: List[Dict[str, Any]] | None = None,
    consumption_summary=None,
) -> str:
    if not HAVE_REPORTLAB:
        return _export_pm_plan_with_kits_pdf_fallback(tasks_due, kits_by_eq, metrics_table, out_dir, title, tools_checklist)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"pm_plan_kits_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Justify",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            alignment=TA_JUSTIFY,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        title=SAN(title),
    )

    available_width = _page_available_width()
    story: List[Any] = []
    story.append(Paragraph(SAN(title), styles["Title"]))
    story.append(Paragraph(SAN(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")), styles["BodyText"]))
    story.append(Spacer(1, 10))

    _title(story, styles, "Résumé du plan de maintenance", level=2, space_after_pt=4)
    story.append(Paragraph(
        SAN(
            "Le choix final de maintenance est établi à partir du processus retenu, des paramètres fiabilistes, "
            "des intervalles calculés, des jours restants avant échéance et du niveau de priorité. "
            "Les cas critiques sont mis en évidence en rouge dans ce document."
        ),
        styles["Justify"],
    ))
    story.append(Spacer(1, 8))

    if metrics_table:
        story.append(_summary_table(metrics_table, available_width))
        story.append(Spacer(1, 10))

    if tasks_due:
        _title(story, styles, "Tâches de maintenance dues", level=2, space_after_pt=4, color=colors.HexColor("#b91c1c"))
        due_table = _tasks_table(tasks_due, available_width)
        if due_table is not None:
            story.append(due_table)
            story.append(Spacer(1, 10))

    for row in metrics_table:
        is_critical = str(row.get("priority_level", "")) == "Critique"
        title_color = colors.HexColor("#b91c1c") if is_critical else None
        _title(story, styles, f"Fiche équipement - {row.get('equipment_code', '')}", level=3, space_after_pt=3, color=title_color)

        summary = [
            ["Élément", "Valeur"],
            ["Choix final", SAN(row.get("maintenance_choice", row.get("maintenance_type", "")))],
            ["Type initial", SAN(row.get("maintenance_type", ""))],
            ["Modèle / Loi", f"{SAN(row.get('model', ''))} / {SAN(row.get('distribution', ''))}"],
            ["beta", fnum(row.get("beta"), 2)],
            ["eta (h)", fnum(row.get("eta_h"), 1)],
            ["gamma (h)", fnum(row.get("gamma_h"), 1)],
            ["Intervalle recommandé (h)", fnum(row.get("T_recommended_h"), 1)],
            ["Intervalle fiabiliste (h)", fnum(row.get("T_R_h"), 1)],
            ["Intervalle économique (h)", fnum(row.get("T_cost_h"), 1)],
            ["Jours restants", SAN(row.get("days_left", ""))],
            ["Niveau de priorité", SAN(row.get("priority_level", ""))],
            ["Décision finale", SAN(row.get("final_decision", ""))],
        ]
        equipment_table = _mk_table(summary, available_width=available_width, font_size=7.6)
        if is_critical:
            equipment_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7f7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#991b1b")),
            ]))
        story.append(equipment_table)
        story.append(Spacer(1, 4))

        story.append(Paragraph(SAN(_build_choice_explanation(row)), styles["Justify"]))
        story.append(Spacer(1, 5))

        story.append(_mk_table(_build_influence_table(row), available_width=available_width, font_size=7.2))
        story.append(Spacer(1, 8))

    _add_reference_sections(story, styles, available_width, tools_checklist)
    story.append(Paragraph(SAN(f"Rapport généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["BodyText"]))

    doc.build(story)
    return str(out_path)
