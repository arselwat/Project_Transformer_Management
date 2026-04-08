
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
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


def _compact_text(value: Any, max_len: int = 140) -> str:
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
                line.append(_compact_text(value, 120))
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
            lengths[idx] = max(lengths[idx], min(len(str(cell)), 60))

    weights = [max(length, 8) for length in lengths]
    weight_sum = sum(weights) if sum(weights) > 0 else ncols
    widths = [(weight / weight_sum) * total_width for weight in weights]

    if ncols <= 2:
        min_width = 35 * mm
        max_width = 120 * mm
    elif ncols <= 4:
        min_width = 28 * mm
        max_width = 95 * mm
    elif ncols <= 7:
        min_width = 18 * mm
        max_width = 65 * mm
    else:
        min_width = 14 * mm
        max_width = 45 * mm

    widths = [min(max(width, min_width), max_width) for width in widths]

    current_sum = sum(widths)
    if current_sum > total_width:
        ratio = total_width / current_sum
        widths = [width * ratio for width in widths]

    return widths


def _mk_table(data: list[list[str]], total_width: float, font_size: int = 7) -> "Table":
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        name=f"body_table_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.35)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        name=f"head_table_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(10, int(font_size * 1.3)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped_rows = []
    for row_index, row in enumerate(data):
        style = head_style if row_index == 0 else body_style
        wrapped_rows.append([Paragraph(_san(cell).replace("\n", "<br/>"), style) for cell in row])

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
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def _fig_to_rl_image(fig, width_mm: float = 165):
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    img = Image(bio)
    width = width_mm * mm
    ratio = img.imageHeight / max(img.imageWidth, 1)
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def _set_plot_style(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)


def _build_graphical_trend_plot(ttf_series: List[float], reliability_result: Dict[str, Any]):
    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    index = np.arange(1, len(event_times) + 1, dtype=float)
    graph = (reliability_result.get("tests", {}) or {}).get("trend_graphical", {}) or {}
    slope = float(graph.get("slope_loglog", 1.0) or 1.0)
    intercept = float(graph.get("intercept_loglog", 0.0) or 0.0)
    r2 = graph.get("r2")
    direction = str(graph.get("direction", "none"))

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    _set_plot_style(fig, ax)
    ax.scatter(event_times, index, s=30, color="#1f77b4", edgecolor="#1f77b4", label="Défaillances cumulées", zorder=3)
    if len(event_times) >= 2:
        fitted = np.exp(intercept + slope * np.log(event_times))
        ax.plot(event_times, fitted, color="#d62728", linewidth=2.4, label=f"Ajustement log-log | pente={_fmt(slope,2)} | R²={_fmt(r2,3)}", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temps cumulé t (h)")
    ax.set_ylabel("Nombre cumulé N(t)")
    ax.set_title("Méthode graphique de tendance")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    legend = (
        f"Pente log-log = {_fmt(slope,2)} ; direction = {direction}. Une pente > 1 traduit une tendance croissante ; "
        f"une pente < 1 traduit une tendance décroissante ; proche de 1, il n’y a pas de tendance nette."
    )
    return fig, legend


def _build_graphical_dependence_plot(ttf_series: List[float], reliability_result: Dict[str, Any]):
    x = np.asarray(ttf_series[:-1], dtype=float)
    y = np.asarray(ttf_series[1:], dtype=float)
    graph = (reliability_result.get("tests", {}) or {}).get("dependence_graphical", {}) or {}
    slope = float(graph.get("slope", 0.0) or 0.0)
    intercept = float(graph.get("intercept", 0.0) or 0.0)
    r2 = graph.get("r2")
    lag1_r = graph.get("lag1_r")
    direction = str(graph.get("direction", "none"))

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    _set_plot_style(fig, ax)
    ax.scatter(x, y, s=30, color="#1f77b4", edgecolor="#1f77b4", label="Lag plot TTFᵢ vs TTFᵢ₊₁", zorder=3)
    if len(x) >= 2:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 150)
        ys = intercept + slope * xs
        ax.plot(xs, ys, color="#d62728", linewidth=2.4, label=f"Droite ajustée | pente={_fmt(slope,2)} | R²={_fmt(r2,3)}", zorder=2)
    ax.set_xlabel("TTFᵢ (h)")
    ax.set_ylabel("TTFᵢ₊₁ (h)")
    ax.set_title("Méthode graphique de dépendance")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    legend = (
        f"Corrélation lag-1 = {_fmt(lag1_r,3)} ; direction = {direction}. Une corrélation positive forte suggère une dépendance entre événements successifs."
    )
    return fig, legend


def _build_reliability_curves_plot(reliability_result: Dict[str, Any]):
    curves = reliability_result.get("curves")
    if not isinstance(curves, pd.DataFrame) or curves.empty:
        return None, "Aucune courbe fiabiliste disponible."
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), dpi=140)
    axes = axes.ravel()
    defs = [("R_t", "Fiabilité R(t)"), ("F_t", "Défaillance F(t)"), ("f_t", "Densité f(t)"), ("h_t", "Taux λ(t) / h(t)")]
    for ax, (col, title) in zip(axes, defs):
        _set_plot_style(fig, ax)
        ax.plot(curves["t"], curves[col], linewidth=2.0, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Temps (h)")
        ax.grid(True, alpha=0.28)
    fig.tight_layout()
    return fig, "Les quatre courbes résument la survie, la défaillance cumulée, la densité et le risque instantané du modèle retenu."


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
        lines.append(
            f"- Équipements classés BPP : {int((summary_df['model'].astype(str).str.upper() == 'BPP').sum())}"
        )

    if "decision_finale" in summary_df.columns:
        decisions = summary_df["decision_finale"].astype(str).value_counts().head(4)
        for label, count in decisions.items():
            lines.append(f"- Décision fréquente : {label} ({int(count)})")

    if "days_left" in summary_df.columns:
        due_series = pd.to_numeric(summary_df["days_left"], errors="coerce").dropna()
        if not due_series.empty:
            lines.append(f"- Échéance minimale observée : {int(due_series.min())} jour(s)")

    return lines


def _table_explanation(label: str) -> str:
    explanations = {
        "Décision finale hiérarchisée": "Ce tableau classe les équipements selon la priorité finale de traitement et résume la décision retenue.",
        "Synthèse globale": "Ce tableau regroupe l'ensemble des sorties utiles à la lecture générale : fiabilité, optimisation, maintenance et décision finale.",
        "Vue tendance": "Ce tableau montre si les données présentent une tendance et si les défaillances successives s'inscrivent dans une dérive identifiable.",
        "Vue ajustement": "Ce tableau résume la qualité d'ajustement du modèle retenu à l'aide des critères statistiques.",
        "Vue optimisation": "Ce tableau présente les intervalles calculés et la recommandation de maintenance retenue.",
        "Tâches dues": "Ce tableau liste les interventions dont l'échéance est proche dans la fenêtre choisie.",
        "Validation des tests de tendance": "Ce tableau explique comment la présence ou non d'une tendance a été déterminée.",
        "Validation des tests de dépendance": "Ce tableau explique comment la dépendance entre défaillances successives a été évaluée.",
        "Choix du processus et du modèle": "Ce tableau justifie le processus fiabiliste retenu et la loi utilisée.",
        "Tests d'ajustement": "Ce tableau indique comment la qualité de l'ajustement a été jugée.",
        "Paramètres fiabilistes": "Ce tableau regroupe les paramètres principaux utilisés pour interpréter le comportement de l'équipement.",
        "Optimisation et maintenance retenue": "Ce tableau explique l'intervalle choisi et le type de maintenance recommandé.",
        "Traçabilité détaillée de la décision finale": "Ce tableau montre quels paramètres ont pesé dans la décision finale et comment chacun est intervenu.",
        "Tableau technique des tests de tendance": "Ce tableau technique affiche les résultats bruts des tests de tendance.",
        "Tableau technique des tests de dépendance": "Ce tableau technique affiche les résultats bruts des tests de dépendance.",
        "Tableau technique du choix du processus": "Ce tableau technique récapitule la logique métier utilisée pour choisir le processus.",
        "Tableau technique des lois candidates": "Ce tableau compare les différentes lois candidates avant de retenir le meilleur ajustement.",
        "Synthèse fiabiliste": "Ce tableau résume les sorties principales du modèle de fiabilité retenu.",
    }
    return explanations.get(label, "Ce tableau présente les résultats associés à cette partie du rapport.")


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
    story.append(
        Paragraph(
            _san("Explication : " + _table_explanation(label)),
            styles["Justify"],
        )
    )
    story.append(Spacer(1, 5))

    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)

    if len(work.columns) > 8:
        parts = _split_dataframe_columns(work, fixed_columns=fixed_columns, max_columns_per_part=5)
        for part_label, part_df in parts:
            story.append(Paragraph(_san(part_label), styles["Heading3"]))
            story.append(_mk_table(_df_to_table_data(part_df), total_width=total_width, font_size=6))
            story.append(Spacer(1, 4))
    else:
        font_size = 7 if len(work.columns) <= 5 else 6
        story.append(_mk_table(_df_to_table_data(work), total_width=total_width, font_size=font_size))
        story.append(Spacer(1, 6))


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
        if row.get("motif_decision"):
            pdf.multi_cell(0, 5, _san(f"Motif : {row.get('motif_decision')}"))

        for key, label in [
            ("trace_trend", "Validation des tests de tendance"),
            ("trace_dependence", "Validation des tests de dépendance"),
            ("trace_model_choice", "Choix du processus et du modèle"),
            ("trace_goodness_of_fit", "Tests d'ajustement"),
            ("trace_parameters", "Paramètres fiabilistes"),
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
            name="Justify",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
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

    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 10))

    if meta:
        story.append(Paragraph("Paramètres de calcul", styles["Heading2"]))
        for key, value in meta.items():
            story.append(Paragraph(_san(f"- {key}: {value}"), styles["Justify"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Résumé exécutif", styles["Heading2"]))
    intro = (
        "Ce rapport présente la synthèse consolidée de l'analyse fiabiliste, de l'optimisation des "
        "intervalles de maintenance et de la hiérarchisation finale des décisions. La lecture suit "
        "l'ordre métier du logiciel : tests de tendance, tests de dépendance, choix du modèle, "
        "validation d'ajustement, optimisation, plan de maintenance et décision finale."
    )
    story.append(Paragraph(_san(intro), styles["Justify"]))
    for line in _build_executive_summary(summary_df):
        story.append(Paragraph(_san(line), styles["Justify"]))
    story.append(Spacer(1, 8))

    ordered_global = [
        (("Decision_finale", "final_decision"), "Décision finale hiérarchisée", ["equipment_code"]),
        (("Synthese_globale", "global_summary"), "Synthèse globale", ["equipment_code"]),
        (("Vue_tendance", "trend_overview"), "Vue tendance", ["equipment_code"]),
        (("Vue_ajustement", "goodness_overview"), "Vue ajustement", ["equipment_code"]),
        (("Vue_optimisation", "optimization_overview"), "Vue optimisation", ["equipment_code"]),
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
    story.append(Paragraph("Détail par équipement", styles["Heading1"]))
    story.append(
        Paragraph(
            _san(
                "Chaque équipement est présenté selon le même fil de lecture afin de garantir une "
                "traçabilité homogène : tendance, dépendance, choix du modèle, qualité d'ajustement, "
                "paramètres fiabilistes, optimisation et décision finale."
            ),
            styles["Justify"],
        )
    )
    story.append(Spacer(1, 8))

    ordered_trace_tables = [
        ("trace_trend", "Validation des tests de tendance", None),
        ("trace_dependence", "Validation des tests de dépendance", None),
        ("trace_model_choice", "Choix du processus et du modèle", None),
        ("trace_goodness_of_fit", "Tests d'ajustement", None),
        ("trace_parameters", "Paramètres fiabilistes", None),
        ("trace_optimization", "Optimisation et maintenance retenue", None),
        ("trace_final_decision", "Traçabilité détaillée de la décision finale", None),
        ("trend_results", "Tableau technique des tests de tendance", None),
        ("dependence_results", "Tableau technique des tests de dépendance", None),
        ("process_choice", "Tableau technique du choix du processus", None),
        ("fit_candidates", "Tableau technique des lois candidates", ["Modèle"]),
        ("reliability_summary", "Synthèse fiabiliste", None),
    ]

    equipment_codes = list(detail_tables_by_eq.keys())
    for index, eq in enumerate(equipment_codes):
        tables = detail_tables_by_eq[eq]
        payload = tables.get("__payload__", {}) or {}
        reliability_result = payload.get("reliability", {}) or {}
        ttf_series = payload.get("ttf_series", []) or []
        matching = summary_df[summary_df["equipment_code"].astype(str) == str(eq)]
        summary_row = matching.iloc[0].to_dict() if not matching.empty else {}

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(Paragraph(_san(f"Décision finale : {summary_row.get('decision_finale', '—')} | Priorité : {summary_row.get('priorite', '—')} | Type de maintenance : {summary_row.get('maintenance_type', '—')}"), styles["Justify"]))
        if summary_row.get("motif_decision"):
            story.append(Paragraph(_san(f"Motif : {summary_row.get('motif_decision')}"), styles["Justify"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("1. Tendance", styles["Heading3"]))
        if len(ttf_series) >= 3:
            fig, legend = _build_graphical_trend_plot(ttf_series, reliability_result)
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))
        for key, label in [("trend_graphical", "Tableau 1.1 Méthode graphique"), ("trend_mk", "Tableau 1.2 Test de Mann-Kendall"), ("trend_laplace", "Tableau 1.3 Test de Laplace"), ("trend_mil", "Tableau 1.4 MIL-HDBK-189"), ("trend_decision", "Tableau 1.5 Décision de tendance"), ("tableau_tendance", "Tableau 1.x Synthèse tendance")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=12, fixed_columns=None)

        story.append(Paragraph("2. Dépendance", styles["Heading3"]))
        if len(ttf_series) >= 3:
            fig, legend = _build_graphical_dependence_plot(ttf_series, reliability_result)
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))
        for key, label in [("dep_graphical", "Tableau 2.1 Méthode graphique"), ("dep_pearson", "Tableau 2.2 Test de Pearson"), ("dep_spearman", "Tableau 2.3 Test de Spearman"), ("dep_decision", "Tableau 2.4 Décision de dépendance"), ("tableau_dependance", "Tableau 2.x Synthèse dépendance")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=12, fixed_columns=None)

        story.append(Paragraph("3. Choix du modèle", styles["Heading3"]))
        for key, label in [("process_choice", "Tableau 3.1 Processus retenu")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=10, fixed_columns=None)

        story.append(Paragraph("4. Ajustement", styles["Heading3"]))
        for key, label in [("fit_candidates", "Tableau 4.1 Comparaison des lois candidates"), ("fit_selected", "Tableau 4.2 Ajustement retenu")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=14, fixed_columns=None)

        story.append(Paragraph("5. Paramètres fiabilistes", styles["Heading3"]))
        for key, label in [("parameter_table", "Tableau 5.1 Paramètres calculés"), ("tableau_parametres", "Tableau 5.2 Synthèse des paramètres")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=14, fixed_columns=None)
        fig, legend = _build_reliability_curves_plot(reliability_result)
        if fig is not None:
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))

        story.append(Paragraph("6. Optimisation", styles["Heading3"]))
        for key, label in [("optimization_table", "Tableau 6.1 Résultats d’optimisation"), ("tableau_optimisation", "Tableau 6.2 Intervalle retenu")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=12, fixed_columns=None)

        story.append(Paragraph("7. Décision finale", styles["Heading3"]))
        for key, label in [("final_decision_table", "Tableau 7.1 Décision finale"), ("tableau_decision_finale", "Tableau 7.2 Synthèse de la décision")]:
            df = tables.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _render_table_block(story, label, df, styles, usable_width, max_rows=10, fixed_columns=None)

        if index < len(equipment_codes) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(out_path)
