from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4, landscape
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


def _compact_text(value: Any, max_len: int = 120) -> str:
    text = _san(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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
                line.append(_compact_text(value, 100))
        data.append(line)

    return data


def _auto_col_widths(data: list[list[str]], total_width: float) -> list[float]:
    if not data:
        return []

    ncols = max(len(row) for row in data)
    lengths = [8] * ncols

    for row in data[:50]:
        for idx in range(ncols):
            cell = row[idx] if idx < len(row) else ""
            lengths[idx] = max(lengths[idx], min(len(str(cell)), 50))

    weights = [max(length, 8) for length in lengths]
    weight_sum = sum(weights) if sum(weights) > 0 else ncols
    widths = [(weight / weight_sum) * total_width for weight in weights]

    if ncols <= 2:
        min_width = 35 * mm
        max_width = 120 * mm
    elif ncols <= 4:
        min_width = 28 * mm
        max_width = 90 * mm
    else:
        min_width = 18 * mm
        max_width = 65 * mm

    widths = [min(max(width, min_width), max_width) for width in widths]

    current_sum = sum(widths)
    if current_sum > total_width:
        ratio = total_width / current_sum
        widths = [width * ratio for width in widths]

    return widths


def _mk_table(data: list[list[str]], total_width: float, font_size: int = 7) -> Table:
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        name=f"body_table_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(8, int(font_size * 1.35)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        name=f"head_table_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(9, int(font_size * 1.35)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped_rows = []
    for row_index, row in enumerate(data):
        style = head_style if row_index == 0 else body_style
        wrapped_rows.append(
            [Paragraph(_san(cell).replace("\n", "<br/>"), style) for cell in row]
        )

    col_widths = _auto_col_widths(data, total_width=total_width)

    table = Table(
        wrapped_rows,
        repeatRows=1,
        colWidths=col_widths,
        splitByRow=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#355CBB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def _build_executive_summary(summary_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if summary_df is None or summary_df.empty:
        return ["Aucune donnée globale n'est disponible pour le rapport."]

    lines.append(f"- Équipements analysés : {len(summary_df)}")

    if "priorite" in summary_df.columns:
        lines.append(
            f"- Équipements en priorité critique : {int((summary_df['priorite'].astype(str) == 'Critique').sum())}"
        )

    if "model" in summary_df.columns:
        lines.append(
            f"- Équipements classés NHPP : {int((summary_df['model'].astype(str).str.upper() == 'NHPP').sum())}"
        )

    if "thermal_status" in summary_df.columns:
        lines.append(
            f"- Cas thermiques critiques : {int((summary_df['thermal_status'].astype(str) == 'Critique').sum())}"
        )
        lines.append(
            f"- Cas thermiques en alerte : {int((summary_df['thermal_status'].astype(str) == 'Alerte').sum())}"
        )

    if "decision_finale" in summary_df.columns:
        decisions = summary_df["decision_finale"].astype(str).value_counts().head(3)
        for label, count in decisions.items():
            lines.append(f"- Décision fréquente : {label} ({int(count)})")

    return lines


def _thermal_influence_rules() -> list[str]:
    return [
        "- Statut thermique Critique : la maintenance doit être accélérée, même si la partie fiabiliste seule semblait acceptable.",
        "- Statut thermique Alerte : la maintenance recommandée devient plus prudente et peut être avancée.",
        "- Statut thermique Normal : la décision dépend surtout de la fiabilité, du coût et de l'échéance.",
        "- Température du point chaud élevée : elle signale un échauffement sévère pouvant justifier un contrôle prioritaire.",
        "- Facteur d'accélération du vieillissement élevé : il montre que l'isolation se dégrade plus vite.",
        "- Perte de vie élevée : elle signale qu'une part importante de la durée de vie est déjà consommée.",
        "- En pratique, la thermique ne remplace pas la fiabilité : elle agit comme facteur aggravant ou modérateur.",
    ]


def _thermal_influence_for_row(row: Dict[str, Any]) -> str:
    thermal_status = str(row.get("thermal_status", "Non disponible"))
    maintenance_type = str(row.get("maintenance_type", "Non précisé"))
    hotspot = _fmt(row.get("theta_hs_max"), 1)
    faa = _fmt(row.get("faa_max"), 3)
    lol = _fmt(row.get("loss_of_life_pct"), 3)

    if thermal_status == "Critique":
        return _san(
            f"Le statut thermique est Critique. La température du point chaud, le facteur d'accélération du vieillissement "
            f"ou la perte de vie sont sévères (thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). "
            f"Cela accélère la décision et renforce le besoin d'une maintenance de type {maintenance_type}."
        )

    if thermal_status == "Alerte":
        return _san(
            f"Le statut thermique est Alerte (thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). "
            f"La thermique renforce la prudence et rapproche souvent l'intervention ou la surveillance autour du type {maintenance_type}."
        )

    if thermal_status == "Normal":
        return _san(
            f"Le statut thermique est Normal (thetaHS max={hotspot}, FAA max={faa}, perte de vie={lol} %). "
            f"Le choix de maintenance dépend alors surtout de la fiabilité, de l'optimisation économique et de l'échéance. "
            f"Le type retenu reste {maintenance_type}."
        )

    return _san(
        f"La thermique n'est pas disponible ou exploitable. Le type de maintenance retenu ({maintenance_type}) provient alors "
        f"principalement de la fiabilité, du coût et de l'échéance."
    )


def _table_explanation(label: str) -> str:
    explanations = {
        "Décision finale hiérarchisée": "Ce tableau classe les équipements selon la priorité finale de traitement et résume la décision retenue.",
        "Synthèse globale": "Ce tableau regroupe l'ensemble des résultats utiles à la lecture générale : fiabilité, thermique, optimisation et décision finale.",
        "Vue d'ensemble des tendances et dépendances": "Ce tableau montre si les données présentent une tendance et si les défaillances successives se comportent comme des événements dépendants.",
        "Vue d'ensemble des tests d'ajustement": "Ce tableau résume la qualité d'ajustement du modèle retenu à partir des critères statistiques.",
        "Vue d'ensemble fiabilité et thermique": "Ce tableau relie les paramètres fiabilistes aux indicateurs thermiques pour chaque équipement.",
        "Vue d'ensemble optimisation et maintenance": "Ce tableau présente les intervalles calculés et la recommandation de maintenance retenue.",
        "Tâches dues": "Ce tableau liste les interventions dont l'échéance est proche dans la fenêtre choisie.",
        "Validation des tests de tendance": "Ce tableau explique comment la présence ou non d'une tendance a été déterminée.",
        "Validation des tests de dépendance": "Ce tableau explique comment la dépendance entre défaillances successives a été évaluée.",
        "Choix du processus et du modèle": "Ce tableau justifie le processus fiabiliste retenu et la loi utilisée.",
        "Tests d'ajustement": "Ce tableau indique comment la qualité de l'ajustement a été jugée à l'aide des tests statistiques.",
        "Paramètres fiabilistes et thermiques": "Ce tableau donne les paramètres principaux utilisés pour interpréter le comportement de l'équipement.",
        "Optimisation et maintenance retenue": "Ce tableau explique l'intervalle choisi et le type de maintenance recommandé.",
        "Traçabilité détaillée de la décision finale": "Ce tableau montre quels paramètres ont pesé dans la décision finale et comment chacun est intervenu.",
        "Tableau technique des tests de tendance": "Ce tableau technique affiche les résultats bruts des tests de tendance.",
        "Tableau technique des tests de dépendance": "Ce tableau technique affiche les résultats bruts des tests de dépendance.",
        "Tableau technique du choix du processus": "Ce tableau technique récapitule la logique métier utilisée pour choisir le processus.",
        "Tableau technique des lois candidates": "Ce tableau compare les différentes lois candidates avant de retenir le meilleur ajustement.",
        "Synthèse fiabiliste": "Ce tableau résume les sorties principales du modèle de fiabilité retenu.",
        "Synthèse thermique": "Ce tableau résume les principaux indicateurs thermiques calculés.",
        "Indicateurs thermiques": "Ce tableau détaille les indicateurs thermiques les plus importants pour la décision.",
        "Jours thermiquement critiques": "Ce tableau met en évidence les jours où la contrainte thermique a été la plus élevée.",
    }
    return explanations.get(label, "Ce tableau présente les résultats associés à cette partie du rapport.")


def _to_vertical_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        prefix = f"Ligne {row_index} - " if len(df) > 1 else ""
        for column_name, value in row.items():
            rows.append(
                {
                    "Champ": f"{prefix}{column_name}",
                    "Valeur": _compact_text(value, 140),
                }
            )
    return pd.DataFrame(rows)


def _split_dataframe_columns(
    df: pd.DataFrame,
    fixed_columns: Optional[List[str]] = None,
    max_columns_per_part: int = 5,
) -> List[Tuple[str, pd.DataFrame]]:
    if df is None or df.empty:
        return [("Tableau", pd.DataFrame())]

    fixed_columns = [col for col in (fixed_columns or []) if col in df.columns]
    other_columns = [col for col in df.columns if col not in fixed_columns]

    if len(df.columns) <= max_columns_per_part:
        return [("Tableau complet", df)]

    available_for_other = max(1, max_columns_per_part - len(fixed_columns))
    chunks: List[Tuple[str, pd.DataFrame]] = []

    start = 0
    index = 1
    while start < len(other_columns):
        end = start + available_for_other
        current_columns = fixed_columns + other_columns[start:end]
        chunks.append((f"Partie {index}", df[current_columns].copy()))
        start = end
        index += 1

    return chunks


def _render_table_block(
    story: list,
    label: str,
    df: pd.DataFrame,
    styles,
    total_width: float,
    max_rows: int = 25,
    fixed_columns: Optional[List[str]] = None,
):
    if df is None or df.empty:
        return

    story.append(Paragraph(_san(label), styles["Heading2"]))

    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)

    # Cas très large -> vertical si peu de lignes
    if len(work.columns) > 8 and len(work) <= 5:
        work = _to_vertical_table(work)
        story.append(_mk_table(_df_to_table_data(work), total_width=total_width, font_size=7))
        story.append(Spacer(1, 4))
    # Cas large -> découpage
    elif len(work.columns) > 8:
        parts = _split_dataframe_columns(work, fixed_columns=fixed_columns, max_columns_per_part=5)
        for part_label, part_df in parts:
            story.append(Paragraph(_san(part_label), styles["Heading3"]))
            story.append(_mk_table(_df_to_table_data(part_df), total_width=total_width, font_size=6))
            story.append(Spacer(1, 4))
    else:
        font_size = 7 if len(work.columns) <= 5 else 6
        story.append(_mk_table(_df_to_table_data(work), total_width=total_width, font_size=font_size))
        story.append(Spacer(1, 4))

    story.append(
        Paragraph(
            _san("Explication : " + _table_explanation(label)),
            styles["SmallBody"],
        )
    )
    story.append(Spacer(1, 8))


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

    pdf = FPDF(orientation="L", unit="mm", format="A4")
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
        pdf.multi_cell(0, 5, _san("Explication : " + _table_explanation("Décision finale hiérarchisée")))
        pdf.ln(2)

    for eq, tables in detail_tables_by_eq.items():
        matching = summary_df[summary_df["equipment_code"].astype(str) == str(eq)]
        row = matching.iloc[0].to_dict() if not matching.empty else {}

        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, _san(f"Équipement {eq}"))
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(
            0,
            5,
            _san(
                f"Décision={row.get('decision_finale', '—')} | Priorité={row.get('priorite', '—')} | "
                f"Maintenance={row.get('maintenance_type', '—')}"
            ),
        )
        pdf.multi_cell(0, 5, _san(_thermal_influence_for_row(row)))

        for key, label in [
            ("trace_trend", "Validation des tests de tendance"),
            ("trace_dependence", "Validation des tests de dépendance"),
            ("trace_model_choice", "Choix du processus et du modèle"),
            ("trace_goodness_of_fit", "Tests d'ajustement"),
            ("trace_parameters", "Paramètres fiabilistes et thermiques"),
            ("trace_optimization", "Optimisation et maintenance retenue"),
            ("trace_final_decision", "Traçabilité détaillée de la décision finale"),
        ]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                pdf.ln(1)
                pdf.set_font("Arial", "B", 10)
                pdf.multi_cell(0, 5, _san(label))
                pdf.set_font("Arial", "", 9)
                for _, r in df.head(25).iterrows():
                    pdf.multi_cell(0, 5, _san(" | ".join([f"{c}={r[c]}" for c in df.columns])))
                pdf.multi_cell(0, 5, _san("Explication : " + _table_explanation(label)))

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
    title: str = "Résultat global de l'analyse et optimisation",
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

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=landscape(A4),
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
    )

    usable_width = landscape(A4)[0] - (doc.leftMargin + doc.rightMargin)

    story = []

    # -----------------------------------------------------------------
    # Titre
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
    # Résumé exécutif
    # -----------------------------------------------------------------
    story.append(Paragraph("Résumé exécutif", styles["Heading2"]))
    for line in _build_executive_summary(summary_df):
        story.append(Paragraph(_san(line), styles["SmallBody"]))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------
    # Influence thermique
    # -----------------------------------------------------------------
    story.append(Paragraph("Influence de la partie thermique sur le choix du type de maintenance", styles["Heading2"]))
    story.append(
        Paragraph(
            _san(
                "La partie thermique n'agit pas seule, mais elle modifie la priorité et peut faire évoluer le type de maintenance recommandé. "
                "Lorsqu'un équipement présente un stress thermique marqué, la décision finale devient plus prudente même si les indicateurs fiabilistes "
                "restent acceptables."
            ),
            styles["SmallBody"],
        )
    )
    for line in _thermal_influence_rules():
        story.append(Paragraph(_san(line), styles["SmallBody"]))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------
    # Tableaux globaux
    # -----------------------------------------------------------------
    ordered_global = [
        (("Decision_finale", "final_decision"), "Décision finale hiérarchisée", ["equipment_code"]),
        (("Synthese_globale", "global_summary"), "Synthèse globale", ["equipment_code"]),
        (("Vue_tendance", "trend_overview"), "Vue d'ensemble des tendances et dépendances", ["equipment_code"]),
        (("Vue_ajustement", "goodness_overview"), "Vue d'ensemble des tests d'ajustement", ["equipment_code"]),
        (("Vue_risque", "risk_overview"), "Vue d'ensemble fiabilité et thermique", ["equipment_code"]),
        (("Vue_optimisation", "optimization_overview"), "Vue d'ensemble optimisation et maintenance", ["equipment_code"]),
        (("Taches_dues", "due_tasks"), "Tâches dues", ["equipment_code"]),
    ]

    for candidate_names, label, fixed_columns in ordered_global:
        df = _get_table(global_tables, *candidate_names)
        if not df.empty:
            _render_table_block(
                story=story,
                label=label,
                df=df,
                styles=styles,
                total_width=usable_width,
                max_rows=40 if label != "Tâches dues" else 25,
                fixed_columns=fixed_columns,
            )

    story.append(PageBreak())

    # -----------------------------------------------------------------
    # Détail par équipement
    # -----------------------------------------------------------------
    story.append(Paragraph("Détail par équipement", styles["Heading1"]))
    story.append(Spacer(1, 6))

    ordered_trace_tables = [
        ("trace_trend", "Validation des tests de tendance", None),
        ("trace_dependence", "Validation des tests de dépendance", None),
        ("trace_model_choice", "Choix du processus et du modèle", None),
        ("trace_goodness_of_fit", "Tests d'ajustement", None),
        ("trace_parameters", "Paramètres fiabilistes et thermiques", None),
        ("trace_optimization", "Optimisation et maintenance retenue", None),
        ("trace_final_decision", "Traçabilité détaillée de la décision finale", None),
        ("trend_results", "Tableau technique des tests de tendance", None),
        ("dependence_results", "Tableau technique des tests de dépendance", None),
        ("process_choice", "Tableau technique du choix du processus", None),
        ("fit_candidates", "Tableau technique des lois candidates", ["Modèle"]),
        ("reliability_summary", "Synthèse fiabiliste", None),
        ("thermal_summary", "Synthèse thermique", None),
        ("thermal_table_indicators", "Indicateurs thermiques", None),
        ("thermal_top5_days", "Jours thermiquement critiques", ["date"]),
    ]

    equipment_codes = list(detail_tables_by_eq.keys())
    for index, eq in enumerate(equipment_codes):
        tables = detail_tables_by_eq[eq]
        matching = summary_df[summary_df["equipment_code"].astype(str) == str(eq)]
        summary_row = matching.iloc[0].to_dict() if not matching.empty else {}

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(
            Paragraph(
                _san(
                    f"Décision finale : {summary_row.get('decision_finale', '—')} | "
                    f"Priorité : {summary_row.get('priorite', '—')} | "
                    f"Type de maintenance : {summary_row.get('maintenance_type', '—')}"
                ),
                styles["SmallBody"],
            )
        )
        story.append(
            Paragraph(
                _san(f"Motif : {summary_row.get('motif_decision', 'Aucun motif disponible.')}"),
                styles["SmallBody"],
            )
        )
        story.append(Paragraph(_san(_thermal_influence_for_row(summary_row)), styles["SmallBody"]))
        story.append(Spacer(1, 6))

        for key, label, fixed_columns in ordered_trace_tables:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(
                    story=story,
                    label=label,
                    df=df,
                    styles=styles,
                    total_width=usable_width,
                    max_rows=25 if key != "fit_candidates" else 12,
                    fixed_columns=fixed_columns,
                )

        if index < len(equipment_codes) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(out_path)