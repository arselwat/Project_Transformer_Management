
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

try:
    from fpdf import FPDF
    HAVE_FPDF = True
except Exception:
    HAVE_FPDF = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _require_pdf() -> None:
    if not HAVE_REPORTLAB and not HAVE_FPDF:
        raise RuntimeError("Aucun moteur PDF disponible. Installe reportlab ou fpdf2.")


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
        .replace("β", "beta")
        .replace("η", "eta")
        .replace("γ", "gamma")
        .replace("λ", "lambda")
        .replace("θ", "theta")
        .replace("\u00A0", " ")
    )


def _fmt(value: Any, digits: int = 3, default: str = "—") -> str:
    try:
        if value is None:
            return default
        numeric = float(value)
        if pd.isna(numeric):
            return default
        if digits <= 0:
            return str(int(round(numeric)))
        return f"{numeric:.{digits}f}"
    except Exception:
        text = _san(value).strip()
        return text if text else default


def _compact(value: Any, max_len: int = 120) -> str:
    text = _san(value).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _safe_df(df: Any) -> pd.DataFrame:
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _auto_widths(data: List[List[str]], total_width: float) -> List[float]:
    if not data:
        return []
    ncols = max(len(row) for row in data)
    lengths = [8] * ncols
    for row in data[:40]:
        for idx in range(ncols):
            cell = row[idx] if idx < len(row) else ""
            lengths[idx] = max(lengths[idx], min(len(str(cell)), 50))
    total = sum(lengths) if sum(lengths) > 0 else ncols
    widths = [(length / total) * total_width for length in lengths]
    widths = [max(18 * mm, min(65 * mm, width)) for width in widths]
    current = sum(widths)
    if current > total_width:
        ratio = total_width / current
        widths = [width * ratio for width in widths]
    return widths


def _df_to_data(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[List[str]]:
    if df is None or df.empty:
        return [["Information"], ["Aucune donnée disponible"]]
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    data = [[_san(col) for col in work.columns]]
    for _, row in work.iterrows():
        data.append([_compact(value, 120) for value in row.tolist()])
    return data


def _mk_table(data: List[List[str]], total_width: float, font_size: int = 8) -> "Table":
    styles = getSampleStyleSheet()
    head_style = ParagraphStyle(
        name=f"Head_{font_size}_{len(data)}",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=max(8, font_size),
        leading=max(10, int(font_size * 1.35)),
        alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        name=f"Body_{font_size}_{len(data)}",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.35)),
        alignment=TA_LEFT,
    )

    wrapped: List[List[Any]] = []
    for row_index, row in enumerate(data):
        style = head_style if row_index == 0 else body_style
        wrapped.append([Paragraph(_san(cell).replace("\n", "<br/>"), style) for cell in row])

    table = Table(wrapped, repeatRows=1, splitByRow=1, colWidths=_auto_widths(data, total_width))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.black),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.black),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
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


def _build_graphical_trend_plot(ttf_series: List[float], reliability_result: Dict[str, Any]):
    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    index = np.arange(1, len(event_times) + 1, dtype=float)
    graph = (reliability_result.get("tests", {}) or {}).get("trend_graphical", {}) or {}
    slope = float(graph.get("slope_loglog", 1.0) or 1.0)
    intercept = float(graph.get("intercept_loglog", 0.0) or 0.0)
    r2 = graph.get("r2")
    direction = str(graph.get("direction", "none"))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.scatter(event_times, index, s=28, label="Défaillances cumulées")
    if len(event_times) >= 2:
        fitted = np.exp(intercept + slope * np.log(event_times))
        ax.plot(event_times, fitted, linewidth=2.0, label=f"Ajustement | pente={_fmt(slope,2)} | R²={_fmt(r2,3)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temps cumulé t (h)")
    ax.set_ylabel("Nombre cumulé N(t)")
    ax.set_title("Méthode graphique de tendance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    legend = (
        f"Lecture : pente beta_graph = {_fmt(slope,2)}, direction = {direction}. "
        f"Une pente > 1 traduit une tendance croissante, une pente < 1 une tendance décroissante, "
        f"et une pente proche de 1 l'absence de tendance nette."
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

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.scatter(x, y, s=28, label="Lag plot TTFᵢ vs TTFᵢ₊₁")
    if len(x) >= 2:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 150)
        ys = intercept + slope * xs
        ax.plot(xs, ys, linewidth=2.0, label=f"Droite ajustée | pente={_fmt(slope,2)} | R²={_fmt(r2,3)}")
    ax.set_xlabel("TTFᵢ (h)")
    ax.set_ylabel("TTFᵢ₊₁ (h)")
    ax.set_title("Méthode graphique de dépendance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    legend = (
        f"Lecture : corrélation lag-1 = {_fmt(lag1_r,3)}, direction = {direction}. "
        f"Une corrélation positive notable suggère une dépendance entre événements successifs."
    )
    return fig, legend


def _build_reliability_curves_plot(reliability_result: Dict[str, Any]):
    curves = reliability_result.get("curves")
    if not isinstance(curves, pd.DataFrame) or curves.empty:
        return None, "Aucune courbe fiabiliste disponible."
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8))
    axes = axes.ravel()
    defs = [("R_t", "Fiabilité R(t)"), ("F_t", "Défaillance F(t)"), ("f_t", "Densité f(t)"), ("h_t", "Taux λ(t) / h(t)")]
    for ax, (col, title) in zip(axes, defs):
        ax.plot(curves["t"], curves[col], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Temps (h)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, "Ces courbes résument la survie, la défaillance cumulée, la densité et le risque instantané du modèle retenu."


def _add_caption(story: List[Any], styles, text: str) -> None:
    story.append(Paragraph(_san(text), styles["Caption"]))
    story.append(Spacer(1, 3))


def _add_table(story: List[Any], styles, caption: str, df: pd.DataFrame, total_width: float, max_rows: Optional[int] = None) -> None:
    if df is None or df.empty:
        return
    _add_caption(story, styles, caption)
    story.append(_mk_table(_df_to_data(df, max_rows=max_rows), total_width=total_width, font_size=7.2))
    story.append(Spacer(1, 8))


def _build_executive_summary(summary_df: pd.DataFrame) -> List[str]:
    if summary_df is None or summary_df.empty:
        return ["Aucune donnée globale n'est disponible pour ce rapport."]
    lines = [f"- Équipements analysés : {len(summary_df)}"]
    if "priorite" in summary_df.columns:
        counts = summary_df["priorite"].astype(str).value_counts()
        for label in ["Critique", "Élevée", "Modérée", "Faible"]:
            if label in counts:
                lines.append(f"- Priorité {label} : {int(counts[label])}")
    if "model" in summary_df.columns:
        models = summary_df["model"].astype(str).value_counts()
        for label, count in models.items():
            lines.append(f"- Processus {label} : {int(count)}")
    return lines


def _fallback_fpdf(
    summary_df: pd.DataFrame,
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_eq: Dict[str, Dict[str, Any]],
    out_dir: str | Path,
    title: str,
    meta: Optional[Dict[str, Any]],
) -> str:
    if not HAVE_FPDF:
        raise RuntimeError("Aucun moteur PDF disponible.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"global_analysis_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 13)
    pdf.multi_cell(0, 7, _san(title))
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 5, _san(datetime.now().strftime("%d/%m/%Y %H:%M")), ln=1)
    pdf.ln(2)

    if meta:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Paramètres", ln=1)
        pdf.set_font("Arial", "", 9)
        for key, value in meta.items():
            pdf.multi_cell(0, 5, _san(f"- {key}: {value}"))
        pdf.ln(2)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Résumé exécutif", ln=1)
    pdf.set_font("Arial", "", 9)
    for line in _build_executive_summary(summary_df):
        pdf.multi_cell(0, 5, _san(line))
    pdf.ln(2)

    preview = global_tables.get("Tableau_decision_globale", pd.DataFrame())
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Décision globale", ln=1)
        pdf.set_font("Arial", "", 8)
        pdf.multi_cell(0, 4.5, _san(preview.head(15).to_string(index=False)))
    pdf.output(str(out_path))
    return str(out_path)


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------
def export_global_analysis_report_pdf(
    summary_df: pd.DataFrame,
    global_tables: Dict[str, pd.DataFrame],
    detail_tables_by_eq: Dict[str, Dict[str, Any]],
    out_dir: str | Path = "reports",
    title: str = "Résultat global de l'analyse et de l'optimisation",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    _require_pdf()

    summary_df = _safe_df(summary_df)
    if not HAVE_REPORTLAB:
        return _fallback_fpdf(summary_df, global_tables, detail_tables_by_eq, out_dir, title, meta)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"global_analysis_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

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
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        title=_san(title),
    )
    usable_width = A4[0] - (doc.leftMargin + doc.rightMargin)
    story: List[Any] = []

    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 8))

    if meta:
        story.append(Paragraph("Paramètres de calcul", styles["Heading2"]))
        for key, value in meta.items():
            story.append(Paragraph(_san(f"- {key} : {value}"), styles["Justify"]))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Résumé exécutif", styles["Heading2"]))
    for line in _build_executive_summary(summary_df):
        story.append(Paragraph(_san(line), styles["Justify"]))
    story.append(Spacer(1, 8))

    ordered_global = [
        ("Tableau 1. Décision globale", global_tables.get("Tableau_decision_globale", pd.DataFrame())),
        ("Tableau 2. Tendance globale", global_tables.get("Tableau_tendance_global", pd.DataFrame())),
        ("Tableau 3. Dépendance globale", global_tables.get("Tableau_dependance_global", pd.DataFrame())),
        ("Tableau 4. Ajustement global", global_tables.get("Tableau_ajustement_global", pd.DataFrame())),
        ("Tableau 5. Loi, paramètres et remplacement", global_tables.get("Tableau_remplacement_global", pd.DataFrame())),
    ]
    for caption, df in ordered_global:
        _add_table(story, styles, caption, df, usable_width, max_rows=20)

    story.append(PageBreak())
    story.append(Paragraph("Détail du pipeline par équipement", styles["Heading1"]))
    story.append(Spacer(1, 6))

    for idx, (equipment_code, tables) in enumerate(detail_tables_by_eq.items()):
        payload = tables.get("__payload__", {}) or {}
        reliability_result = payload.get("reliability", {}) or {}
        ttf_series = payload.get("ttf_series", []) or []

        story.append(Paragraph(_san(f"Équipement {equipment_code}"), styles["Heading2"]))
        selected = summary_df[summary_df.get("equipment_code", pd.Series(dtype=str)).astype(str) == str(equipment_code)]
        if not selected.empty:
            row = selected.iloc[0].to_dict()
            story.append(Paragraph(_san(f"Décision retenue : {row.get('decision_finale', '—')}"), styles["Justify"]))
            story.append(Paragraph(_san(f"Motif : {row.get('motif_decision', '—')}"), styles["Justify"]))
            story.append(Spacer(1, 4))

        # Trend
        story.append(Paragraph("1. Tendance", styles["Heading3"]))
        if len(ttf_series) >= 3:
            fig, legend = _build_graphical_trend_plot(ttf_series, reliability_result)
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))
        _add_table(story, styles, "Tableau 1.1 Méthode graphique", tables.get("trend_graphical", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 1.2 Test de Mann-Kendall", tables.get("trend_mk", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 1.3 Test de Laplace", tables.get("trend_laplace", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 1.4 MIL-HDBK-189", tables.get("trend_mil", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 1.5 Décision de tendance", tables.get("trend_decision", pd.DataFrame()), usable_width, max_rows=6)

        # Dependence
        story.append(Paragraph("2. Dépendance", styles["Heading3"]))
        if len(ttf_series) >= 3:
            fig, legend = _build_graphical_dependence_plot(ttf_series, reliability_result)
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))
        _add_table(story, styles, "Tableau 2.1 Méthode graphique", tables.get("dep_graphical", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 2.2 Test de Pearson", tables.get("dep_pearson", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 2.3 Test de Spearman", tables.get("dep_spearman", pd.DataFrame()), usable_width, max_rows=6)
        _add_table(story, styles, "Tableau 2.4 Décision de dépendance", tables.get("dep_decision", pd.DataFrame()), usable_width, max_rows=6)

        # Model choice, fit, parameters, optimization, final decision
        story.append(Paragraph("3. Choix du modèle", styles["Heading3"]))
        _add_table(story, styles, "Tableau 3.1 Processus retenu", tables.get("process_choice", pd.DataFrame()), usable_width, max_rows=6)

        story.append(Paragraph("4. Ajustement", styles["Heading3"]))
        _add_table(story, styles, "Tableau 4.1 Comparaison des candidats", tables.get("fit_candidates", pd.DataFrame()), usable_width, max_rows=10)
        _add_table(story, styles, "Tableau 4.2 Ajustement retenu", tables.get("fit_selected", pd.DataFrame()), usable_width, max_rows=6)

        story.append(Paragraph("5. Paramètres et résultats fiabilistes", styles["Heading3"]))
        _add_table(story, styles, "Tableau 5.1 Paramètres calculés", tables.get("parameter_table", pd.DataFrame()), usable_width, max_rows=10)
        fig, legend = _build_reliability_curves_plot(reliability_result)
        if fig is not None:
            story.append(_fig_to_rl_image(fig, width_mm=165))
            story.append(Paragraph(_san(legend), styles["Justify"]))
            story.append(Spacer(1, 5))

        story.append(Paragraph("6. Optimisation", styles["Heading3"]))
        _add_table(story, styles, "Tableau 6.1 Résultats d’optimisation", tables.get("optimization_table", pd.DataFrame()), usable_width, max_rows=10)

        story.append(Paragraph("7. Décision finale", styles["Heading3"]))
        _add_table(story, styles, "Tableau 7.1 Synthèse finale", tables.get("final_decision_table", pd.DataFrame()), usable_width, max_rows=6)

        if idx < len(detail_tables_by_eq) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(out_path)
