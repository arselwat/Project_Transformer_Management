from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Iterable

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
def _san(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("θ", "theta")
        .replace("λ", "lambda")
        .replace("β", "beta")
        .replace("η", "eta")
        .replace("γ", "gamma")
        .replace("°", " deg")
        .replace("\u00A0", " ")
    )


def _fmt(value: Any, decimals: int = 2, default: str = "—") -> str:
    try:
        if value is None:
            return default
        numeric = float(value)
        if pd.isna(numeric):
            return default
        return f"{numeric:.{decimals}f}"
    except Exception:
        text = _san(value)
        return text if text else default


def _require_pdf():
    if not HAVE_REPORTLAB and not HAVE_FPDF:
        raise RuntimeError("Aucun moteur PDF disponible. Installe reportlab ou fpdf2.")


def _safe_df(df: Any) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _get_table(global_tables: Dict[str, pd.DataFrame], *candidates: str) -> pd.DataFrame:
    for candidate in candidates:
        df = global_tables.get(candidate)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def _df_to_table_data(df: pd.DataFrame, max_rows: Optional[int] = None) -> list[list[str]]:
    if df is None or df.empty:
        return [["Information"], ["Aucune donnée disponible"]]

    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)

    headers = [str(col) for col in work.columns]
    data: list[list[str]] = [headers]

    for _, row in work.iterrows():
        line: list[str] = []
        for value in row.tolist():
            if isinstance(value, (int, float)):
                line.append(_fmt(value, 4))
            else:
                line.append(_san(value))
        data.append(line)

    return data


def _mk_table(data: list[list[str]], col_widths=None, font_size: int = 7) -> Table:
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        name="body_table",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.35)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        name="head_table",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(10, int(font_size * 1.4)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped_rows = []
    for row_index, row in enumerate(data):
        style = head_style if row_index == 0 else body_style
        wrapped_rows.append(
            [Paragraph(_san(cell).replace("\n", "<br/>"), style) for cell in row]
        )

    table = Table(wrapped_rows, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F6DB8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def _bool_label(value: Any) -> str:
    if value is True:
        return "Oui"
    if value is False:
        return "Non"
    return _san(value) if value not in (None, "") else "Non disponible"


def _first_row(df: pd.DataFrame) -> Dict[str, Any]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.iloc[0].to_dict()
    return {}


def _build_executive_summary(summary_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if summary_df is None or summary_df.empty:
        return ["Aucune donnée globale n'est disponible pour le rapport."]

    equipment_count = len(summary_df)
    critical_count = 0
    if "priorite" in summary_df.columns:
        critical_count = int((summary_df["priorite"].astype(str) == "Critique").sum())

    nhpp_count = 0
    if "model" in summary_df.columns:
        nhpp_count = int((summary_df["model"].astype(str).str.upper() == "NHPP").sum())

    thermal_critical = 0
    thermal_alert = 0
    if "thermal_status" in summary_df.columns:
        thermal_critical = int((summary_df["thermal_status"].astype(str) == "Critique").sum())
        thermal_alert = int((summary_df["thermal_status"].astype(str) == "Alerte").sum())

    lines.append(f"- Équipements analysés : {equipment_count}")
    lines.append(f"- Équipements en priorité critique : {critical_count}")
    lines.append(f"- Équipements classés NHPP : {nhpp_count}")
    lines.append(f"- Cas thermiques critiques : {thermal_critical}")
    lines.append(f"- Cas thermiques en alerte : {thermal_alert}")

    if "decision_finale" in summary_df.columns:
        top_decisions = summary_df["decision_finale"].astype(str).value_counts().head(3)
        for label, count in top_decisions.items():
            lines.append(f"- Décision fréquente : {label} ({int(count)})")

    return lines


def _thermal_influence_rules() -> list[str]:
    return [
        "- Statut thermique Critique : la maintenance doit être accélérée, même si la partie fiabiliste seule semblait acceptable.",
        "- Statut thermique Alerte : la maintenance recommandée reste souvent préventive renforcée ou avec surveillance active.",
        "- Statut thermique Normal : la décision repose surtout sur la fiabilité, le coût et l'échéance.",
        "- Température du point chaud élevée : elle traduit un échauffement sévère pouvant justifier un contrôle prioritaire.",
        "- Facteur d'accélération du vieillissement élevé : il montre que l'isolation vieillit plus vite et pousse vers une intervention plus précoce.",
        "- Perte de vie élevée : elle signale qu'une part significative de la durée de vie a déjà été consommée.",
        "- En pratique, la thermique ne remplace pas la fiabilité : elle agit comme facteur aggravant ou modérateur sur le choix du type de maintenance.",
    ]


def _thermal_influence_for_row(row: Dict[str, Any]) -> str:
    thermal_status = str(row.get("thermal_status", "Non disponible"))
    maintenance_type = str(row.get("maintenance_type", "Non précisé"))
    hotspot = _fmt(row.get("theta_hs_max"), 1)
    faa = _fmt(row.get("faa_max"), 3)
    lol = _fmt(row.get("loss_of_life_pct"), 3)

    if thermal_status == "Critique":
        return _san(
            f"Le statut thermique est Critique. La température de point chaud, le facteur d'accélération du vieillissement "
            f"ou la perte de vie sont jugés sévères (thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). "
            f"Cela pousse la décision vers une action plus rapide et renforce le besoin d'une maintenance de type {maintenance_type}."
        )

    if thermal_status == "Alerte":
        return _san(
            f"Le statut thermique est Alerte (thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). "
            f"La thermique ne rend pas toujours l'intervention immédiate, mais elle renforce la prudence et justifie souvent "
            f"une maintenance plus précoce ou un renforcement de la surveillance autour du type {maintenance_type}."
        )

    if thermal_status == "Normal":
        return _san(
            f"Le statut thermique est Normal. Les indicateurs thermiques restent compatibles avec le fonctionnement courant "
            f"(thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). Dans ce cas, le choix de maintenance dépend surtout "
            f"de la fiabilité, de l'optimisation économique et de l'échéance calculée. Le type retenu reste {maintenance_type}."
        )

    return _san(
        f"La thermique n'est pas disponible ou n'est pas exploitable pour cet équipement. Le type de maintenance retenu "
        f"({maintenance_type}) provient alors principalement de la fiabilité, du coût et de l'échéance."
    )


def _build_equipment_story_blocks(
    eq: str,
    summary_row: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    styles,
) -> list:
    story = []

    story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))

    story.append(
        Paragraph(
            _san(
                f"Décision finale : {summary_row.get('decision_finale', '—')} | "
                f"Priorité : {summary_row.get('priorite', '—')} | "
                f"Type de maintenance : {summary_row.get('maintenance_type', '—')}"
            ),
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            _san(f"Motif : {summary_row.get('motif_decision', 'Aucun motif disponible.')}"),
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            _san(_thermal_influence_for_row(summary_row)),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 6))

    ordered_tables = [
        ("trend_results", "Tests de tendance"),
        ("dependence_results", "Tests de dépendance"),
        ("process_choice", "Choix du processus"),
        ("fit_candidates", "Modèles candidats"),
        ("reliability_summary", "Synthèse fiabiliste"),
        ("thermal_summary", "Synthèse thermique"),
        ("thermal_table_indicators", "Indicateurs thermiques"),
        ("thermal_top5_days", "Jours thermiquement critiques"),
    ]

    for key, label in ordered_tables:
        df = tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            story.append(Paragraph(_san(label), styles["Heading3"]))
            max_rows = 20 if key not in ("fit_candidates", "thermal_top5_days") else 10
            story.append(_mk_table(_df_to_table_data(df, max_rows=max_rows), font_size=7))
            story.append(Spacer(1, 6))

    return story


# ---------------------------------------------------------------------
# Fallback FPDF
# ---------------------------------------------------------------------
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
    pdf.ln(3)

    if meta:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Paramètres de calcul", ln=1)
        pdf.set_font("Arial", "", 9)
        for key, value in meta.items():
            pdf.multi_cell(0, 5, _san(f"- {key}: {value}"))
        pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Résumé exécutif", ln=1)
    pdf.set_font("Arial", "", 9)
    for line in _build_executive_summary(summary_df):
        pdf.multi_cell(0, 5, _san(line))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Influence de la partie thermique sur la maintenance", ln=1)
    pdf.set_font("Arial", "", 9)
    for rule in _thermal_influence_rules():
        pdf.multi_cell(0, 5, _san(rule))
    pdf.ln(2)

    final_decision_df = _get_table(global_tables, "Decision_finale", "final_decision")
    if not final_decision_df.empty:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Décision finale hiérarchisée", ln=1)
        pdf.set_font("Arial", "", 9)
        for _, row in final_decision_df.head(25).iterrows():
            pdf.multi_cell(
                0,
                5,
                _san(
                    f"- {row.get('equipment_code', '')} | priorité={row.get('priorite', '')} | "
                    f"décision={row.get('decision_finale', '')} | motif={row.get('motif_decision', '')}"
                ),
            )
        pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Synthèse par équipement", ln=1)
    pdf.set_font("Arial", "", 9)
    for _, row in summary_df.head(30).iterrows():
        pdf.multi_cell(
            0,
            5,
            _san(
                f"- {row.get('equipment_code', '')} | processus={row.get('model', '')} | "
                f"loi={row.get('distribution', '')} | bêta={_fmt(row.get('beta'), 2)} | "
                f"thermique={row.get('thermal_status', '')} | maintenance={row.get('maintenance_type', '')} | "
                f"score={row.get('priority_score', '')}"
            ),
        )
    pdf.ln(2)

    if detail_tables_by_eq:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Détail par équipement", ln=1)
        pdf.set_font("Arial", "", 9)

        for eq, tables in detail_tables_by_eq.items():
            matching = summary_df[summary_df["equipment_code"].astype(str) == str(eq)]
            row = matching.iloc[0].to_dict() if not matching.empty else {}

            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 6, _san(f"[{eq}]"))
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(
                0,
                5,
                _san(
                    f"Décision={row.get('decision_finale', '—')} | "
                    f"Priorité={row.get('priorite', '—')} | "
                    f"Maintenance={row.get('maintenance_type', '—')}"
                ),
            )
            pdf.multi_cell(0, 5, _san(_thermal_influence_for_row(row)))

            reliability_summary = tables.get("reliability_summary")
            if isinstance(reliability_summary, pd.DataFrame) and not reliability_summary.empty:
                info = reliability_summary.iloc[0].to_dict()
                pdf.multi_cell(
                    0,
                    5,
                    _san(
                        f"Fiabilité: processus={info.get('Processus', '')} | distribution={info.get('Distribution', '')} | "
                        f"MTBF={_fmt(info.get('MTBF (h)'), 1)} h | MTTR={_fmt(info.get('MTTR (h)'), 1)} h | "
                        f"disponibilité={_fmt(info.get('Disponibilité'), 3)}"
                    ),
                )

            thermal_summary = tables.get("thermal_summary")
            if isinstance(thermal_summary, pd.DataFrame) and not thermal_summary.empty:
                info = thermal_summary.iloc[0].to_dict()
                pdf.multi_cell(
                    0,
                    5,
                    _san(
                        f"Thermique: thetaHS max={_fmt(info.get('θHS max') or info.get('theta_hs_max'), 1)} | "
                        f"FAA max={_fmt(info.get('FAA max') or info.get('faa_max'), 3)} | "
                        f"perte de vie={_fmt(info.get('Loss of life (%)') or info.get('loss_of_life_pct'), 3)} %"
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

    summary_df = _safe_df(summary_df)

    if not HAVE_REPORTLAB:
        return _fallback_fpdf(summary_df, global_tables, detail_tables_by_eq, out_dir, title, meta)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"global_analysis_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
        )
    )

    story = []

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
    )

    # -----------------------------------------------------------------
    # Cover / title
    # -----------------------------------------------------------------
    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 10))

    if meta:
        story.append(Paragraph("Paramètres de calcul", styles["Heading2"]))
        for key, value in meta.items():
            story.append(Paragraph(_san(f"- {key}: {value}"), styles["SmallBody"]))
        story.append(Spacer(1, 8))

    # -----------------------------------------------------------------
    # Executive summary
    # -----------------------------------------------------------------
    story.append(Paragraph("Résumé exécutif", styles["Heading2"]))
    for line in _build_executive_summary(summary_df):
        story.append(Paragraph(_san(line), styles["SmallBody"]))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------
    # Explain thermal influence
    # -----------------------------------------------------------------
    story.append(Paragraph("Influence de la partie thermique sur le choix du type de maintenance", styles["Heading2"]))
    story.append(
        Paragraph(
            _san(
                "La partie thermique n'agit pas seule, mais elle modifie la priorité et peut faire évoluer "
                "le type de maintenance recommandé. Lorsqu'un équipement présente un stress thermique marqué, "
                "la décision finale devient plus prudente même si les indicateurs fiabilistes restent acceptables."
            ),
            styles["SmallBody"],
        )
    )
    for line in _thermal_influence_rules():
        story.append(Paragraph(_san(line), styles["SmallBody"]))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------
    # Global tables
    # -----------------------------------------------------------------
    ordered_global = [
        (("Decision_finale", "final_decision"), "Décision finale hiérarchisée"),
        (("Synthese_globale", "global_summary"), "Synthèse globale"),
        (("Vue_tendance", "trend_overview"), "Vue d'ensemble des tendances et dépendances"),
        (("Vue_risque", "risk_overview"), "Vue d'ensemble fiabilité et thermique"),
        (("Vue_optimisation", "optimization_overview"), "Vue d'ensemble optimisation et maintenance"),
        (("Taches_dues", "due_tasks"), "Tâches dues"),
    ]

    for candidate_names, label in ordered_global:
        df = _get_table(global_tables, *candidate_names)
        if not df.empty:
            story.append(Paragraph(_san(label), styles["Heading2"]))
            story.append(_mk_table(_df_to_table_data(df, max_rows=40), font_size=7))
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # -----------------------------------------------------------------
    # Detailed equipment sections
    # -----------------------------------------------------------------
    story.append(Paragraph("Détail par équipement", styles["Heading1"]))
    story.append(Spacer(1, 6))

    for eq, tables in detail_tables_by_eq.items():
        matching = summary_df[summary_df["equipment_code"].astype(str) == str(eq)]
        summary_row = matching.iloc[0].to_dict() if not matching.empty else {}
        story.extend(_build_equipment_story_blocks(eq, summary_row, tables, styles))
        story.append(PageBreak())

    doc.build(story)
    return str(out_path)