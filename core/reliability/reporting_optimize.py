
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False


def _san(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("≤", "<=").replace("≥", ">=")
        .replace("β", "beta").replace("η", "eta").replace("γ", "gamma")
        .replace("θ", "theta").replace("\u00A0", " ")
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    except Exception:
        return None


def _fmt(value: Any, nd: int = 2, dash: str = "—") -> str:
    numeric = _safe_float(value)
    return dash if numeric is None else f"{numeric:.{nd}f}"


def _compact(value: Any, max_len: int = 160) -> str:
    text = _san(value).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _require_reportlab():
    if not HAVE_REPORTLAB:
        raise RuntimeError("ReportLab n'est pas disponible. Installe reportlab puis relance.")


def _auto_widths(data: List[List[Any]], total_width: float) -> List[float]:
    if not data:
        return []
    ncols = max(len(row) for row in data)
    lengths = [8] * ncols
    for row in data[:40]:
        for idx in range(ncols):
            cell = row[idx] if idx < len(row) else ""
            lengths[idx] = max(lengths[idx], min(len(str(cell)), 60))
    total = sum(lengths) if sum(lengths) > 0 else ncols
    widths = [(length / total) * total_width for length in lengths]
    widths = [max(16 * mm, min(55 * mm, w)) for w in widths]
    current = sum(widths)
    if current > total_width:
        ratio = total_width / current
        widths = [w * ratio for w in widths]
    return widths


def _mk_table(data: List[List[Any]], total_width: float, font_size: int = 8):
    styles = getSampleStyleSheet()
    head_style = ParagraphStyle(
        name=f"Head_{font_size}_{len(data)}",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=max(8, font_size),
        leading=max(10, int(font_size * 1.3)),
        alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        name=f"Body_{font_size}_{len(data)}",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.3)),
        alignment=TA_LEFT,
    )
    wrapped = []
    for row_index, row in enumerate(data):
        style = head_style if row_index == 0 else body_style
        wrapped.append([Paragraph(_san(cell).replace("\n", "<br/>"), style) for cell in row])
    tbl = Table(wrapped, repeatRows=1, splitByRow=1, colWidths=_auto_widths(data, total_width))
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BOX", (0, 0), (-1, -1), 1.1, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ])
    )
    return tbl


def _fig_to_rl_image(fig, width_mm: float = 180):
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


def _maintenance_type(beta: Optional[float], explicit: Any = None) -> str:
    if explicit not in (None, ""):
        return _san(explicit)
    if beta is None:
        return "Maintenance à confirmer"
    if beta < 0.9:
        return "Maintenance corrective / fiabilisation"
    if beta <= 1.1:
        return "Maintenance conditionnelle"
    return "Maintenance préventive planifiée"


def _maintenance_explanation(label: str) -> str:
    lowered = label.lower()
    if "correct" in lowered:
        return "Ce type est retenu lorsqu'il faut corriger une situation dégradée ou des défauts précoces."
    if "condition" in lowered:
        return "Ce type est retenu lorsque l'état réel de l'équipement doit guider l'intervention."
    if "prévent" in lowered or "prevent" in lowered:
        return "Ce type est retenu lorsqu'une intervention planifiée permet de limiter le risque futur."
    if "prédict" in lowered or "predict" in lowered:
        return "Ce type est retenu lorsqu'on peut anticiper le bon moment d'intervention."
    return "Type de maintenance retenu par synthèse des paramètres disponibles."


def _pipe_line(pipe: dict) -> str:
    if not isinstance(pipe, dict) or not pipe:
        return "Trace indisponible."
    rel = pipe.get("reliability", pipe)
    tests = rel.get("tests", {}) or {}
    goodness = rel.get("goodness", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    dep = tests.get("dependence", {}) or {}
    return (
        f"TTF > 0 -> MK(p = {_fmt(mk.get('p'), 3)}, dir = {_san(mk.get('direction', 'none'))}) "
        f"-> Dep(r = {_fmt(dep.get('r'), 3)}, p = {_fmt(dep.get('p'), 3)}) "
        f"-> Modèle = {_san(rel.get('model', 'RP'))} ; Loi = {_san(rel.get('distribution', '—'))} ; "
        f"KS p = {_fmt(goodness.get('ks_p'), 3)} ; Chi carré p = {_fmt(goodness.get('chi2_p'), 3)}"
    )


def _fig_R_curves(fits: Dict[str, Any], intervals: Dict[str, Any]):
    if not fits:
        return None
    etas = [float(getattr(fit, "eta", 0.0) or 0.0) for fit in fits.values()]
    tmax = max(etas) * 1.6 if etas and max(etas) > 0 else 1000.0
    maybe = []
    for value in (intervals or {}).values():
        if isinstance(value, dict):
            for key in ["T_R", "T_cost"]:
                current = _safe_float(value.get(key))
                if current is not None:
                    maybe.append(current)
    if maybe:
        tmax = max(tmax, max(maybe) * 1.2)
    t = np.linspace(0, max(tmax, 1.0), 350)
    fig, ax = plt.subplots(figsize=(10, 5))
    for eq, fit in fits.items():
        beta = float(getattr(fit, "beta", 1.0) or 1.0)
        eta = float(getattr(fit, "eta", 1.0) or 1.0)
        gamma = float(getattr(fit, "gamma", 0.0) or 0.0)
        R = np.ones_like(t)
        mask = t > gamma
        R[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-12)) ** max(beta, 1e-12)))
        ax.plot(t, R, linewidth=2, label=f"{eq} (beta={beta:.2f}, eta={eta:.1f})")
        current = intervals.get(eq, {})
        if isinstance(current, dict):
            tr = _safe_float(current.get("T_R"))
            tc = _safe_float(current.get("T_cost"))
            if tr is not None:
                ax.axvline(tr, linestyle="--", linewidth=1)
            if tc is not None:
                ax.axvline(tc, linestyle=":", linewidth=1)
    ax.set_xlabel("Temps (heures)")
    ax.set_ylabel("R(t)")
    ax.set_title("Courbes de fiabilité utilisées pour l'optimisation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_rl_image(fig, width_mm=180)


def _row_from_df_out(df_out, eq: str) -> Dict[str, Any]:
    if df_out is None or getattr(df_out, "empty", True):
        return {}
    if "equipment_code" not in df_out.columns:
        return {}
    matched = df_out[df_out["equipment_code"].astype(str) == str(eq)]
    return {} if matched.empty else matched.iloc[0].to_dict()


def _summary_table_data(fits: Dict[str, Any], intervals: Dict[str, Any], organigram_by_eq: Dict[str, Any], df_out=None) -> List[List[Any]]:
    data = [[
        "Équipement", "Modèle", "Loi", "beta", "eta (h)", "gamma (h)",
        "T_R (h)", "T_cost (h)", "T_recommandé (h)", "R(T_cost)", "C_min / h", "Maintenance retenue"
    ]]
    eqs = sorted(set(list(fits.keys()) + list((organigram_by_eq or {}).keys())))
    for eq in eqs:
        row = _row_from_df_out(df_out, eq)
        fit = fits.get(eq)
        rel = (organigram_by_eq.get(eq) or {}).get("reliability", organigram_by_eq.get(eq, {}) or {})
        beta = _safe_float(row.get("beta") if row else None)
        eta = _safe_float(row.get("eta_h") if row else None)
        gamma = _safe_float(row.get("gamma_h") if row else None)
        if fit is not None:
            beta = beta if beta is not None else _safe_float(getattr(fit, "beta", None))
            eta = eta if eta is not None else _safe_float(getattr(fit, "eta", None))
            gamma = gamma if gamma is not None else _safe_float(getattr(fit, "gamma", None))
        current_itv = intervals.get(eq, {}) if isinstance(intervals.get(eq, {}), dict) else {}
        data.append([
            eq,
            _san(row.get("model", rel.get("model", "—"))),
            _san(row.get("distribution", rel.get("distribution", "—"))),
            _fmt(beta, 3),
            _fmt(eta, 1),
            _fmt(gamma, 1),
            _fmt(row.get("T_R_h") if row else current_itv.get("T_R"), 1),
            _fmt(row.get("T_cost_h") if row else current_itv.get("T_cost"), 1),
            _fmt(row.get("T_recommended_h"), 1),
            _fmt(row.get("R(T_cost)") if row else current_itv.get("R_at_T"), 3),
            _fmt(row.get("C_min_per_h") if row else current_itv.get("C_min"), 4),
            _maintenance_type(beta, row.get("maintenance_type") if row else None),
        ])
    return data


def _parameter_influence_table(row: Dict[str, Any], pipe: Dict[str, Any]) -> List[List[Any]]:
    beta = _safe_float(row.get("beta"))
    model = _san(row.get("model", pipe.get("model", "—")))
    distribution = _san(row.get("distribution", pipe.get("distribution", "—")))
    maintenance = _maintenance_type(beta, row.get("maintenance_type"))
    return [
        ["Paramètre", "Valeur", "Influence sur le choix"],
        ["Type de maintenance retenu", maintenance, _maintenance_explanation(maintenance)],
        ["Paramètre beta", _fmt(beta, 3), "Décrit la phase de vie : défauts précoces, aléatoire ou usure."],
        ["Processus retenu", model, "Le type de processus influence le niveau de prudence."],
        ["Loi retenue", distribution, "Cadre mathématique retenu pour l'ajustement."],
        ["Intervalle de fiabilité T_R", _fmt(row.get("T_R_h"), 1), "Repère issu de l'objectif de fiabilité."],
        ["Intervalle économique T_cost", _fmt(row.get("T_cost_h"), 1), "Repère minimisant le coût moyen."],
        ["Intervalle recommandé", _fmt(row.get("T_recommended_h"), 1), "Compromis final appliqué."],
        ["Fiabilité à T_cost", _fmt(row.get("R(T_cost)"), 3), "Niveau de fiabilité conservé à l'intervalle économique."],
        ["Coût minimal par heure", _fmt(row.get("C_min_per_h"), 4), "Contribution économique dans le choix."],
        ["Jours restants", _san(row.get("days_left", "—")), "Permet de hiérarchiser l'intervention dans le temps."],
    ]


def export_optimization_report_pdf_bytes(
    *,
    df,
    df_out=None,
    fits: Dict[str, Any] | None = None,
    intervals: Dict[str, Any] | None = None,
    organigram_by_eq: Dict[str, Any] | None = None,
    detail_tables_by_eq: Optional[Dict[str, Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Rapport d'analyse et optimisation",
) -> bytes:
    _require_reportlab()

    fits = fits or {}
    intervals = intervals or {}
    organigram_by_eq = organigram_by_eq or {}
    detail_tables_by_eq = detail_tables_by_eq or {}
    meta = meta or {}

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
        )
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        title=_san(title),
    )
    usable_width = landscape(A4)[0] - (doc.leftMargin + doc.rightMargin)
    story: List[Any] = []

    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["Normal"]))
    story.append(Spacer(1, 8))

    nb_obs = int(getattr(df, "shape", [0])[0]) if df is not None else 0
    nb_eq = int(getattr(df_out, "shape", [0])[0]) if df_out is not None else len(fits)

    story.append(Paragraph("Résumé", styles["Heading2"]))
    story.append(Paragraph(_san(f"- Équipements analysés : {nb_eq}"), styles["Justify"]))
    story.append(Paragraph(_san(f"- Observations TTF : {nb_obs}"), styles["Justify"]))
    story.append(Spacer(1, 6))

    if meta:
        story.append(Paragraph("Paramètres utilisés", styles["Heading2"]))
        rows = [["Paramètre", "Valeur"]]
        for key in ["alpha", "R_target", "C_prev", "C_corr", "R_min_cost"]:
            if key in meta:
                rows.append([_san(key), _san(meta.get(key))])
        story.append(_mk_table(rows, total_width=usable_width, font_size=8))
        story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Ce rapport présente uniquement les éléments utiles à l'optimisation : paramètres fiabilistes, "
        "intervalles calculés, coût minimal et type de maintenance retenu.",
        styles["Justify"],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Tableau 1. Synthèse globale de l'optimisation", styles["Caption"]))
    story.append(_mk_table(
        _summary_table_data(fits=fits, intervals=intervals, organigram_by_eq=organigram_by_eq, df_out=df_out),
        total_width=usable_width,
        font_size=7,
    ))
    story.append(Spacer(1, 8))

    chart = _fig_R_curves(fits, intervals)
    if chart is not None:
        story.append(Paragraph("Figure. Courbes de fiabilité utilisées pour l'optimisation", styles["Caption"]))
        story.append(chart)
        story.append(Spacer(1, 8))

    eqs = sorted(set(list(fits.keys()) + list((organigram_by_eq or {}).keys())))
    if eqs:
        story.append(PageBreak())

    for idx, eq in enumerate(eqs):
        pipe = organigram_by_eq.get(eq, {}) or {}
        rel = pipe.get("reliability", pipe)
        row = _row_from_df_out(df_out, eq)

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(Paragraph(_compact(_pipe_line(pipe), 220), styles["Justify"]))
        story.append(Spacer(1, 4))

        maintenance_type = _maintenance_type(_safe_float(row.get("beta")), row.get("maintenance_type"))
        story.append(Paragraph(
            _san(
                f"Choix retenu dans le cas d'étude : {maintenance_type}. "
                f"Ce choix résulte de la combinaison entre les paramètres fiabilistes, les intervalles calculés et la logique d'optimisation."
            ),
            styles["Justify"],
        ))
        story.append(Spacer(1, 4))

        story.append(Paragraph("Tableau 2. Paramètres qui ont influencé le choix", styles["Caption"]))
        story.append(_mk_table(_parameter_influence_table(row, rel), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 6))

        if idx < len(eqs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()


def export_optimization_report_pdf(
    df,
    fits: Dict[str, Any],
    intervals: Dict[str, Any],
    organigram_by_eq: Dict[str, Any],
    out_dir: str = "reports",
    df_out=None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Rapport d'analyse et optimisation",
    org_results: Optional[Dict[str, Any]] = None,
    detail_tables_by_eq: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    _require_reportlab()

    if organigram_by_eq is None and org_results is not None:
        organigram_by_eq = org_results
    organigram_by_eq = organigram_by_eq or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_optimisation_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    pdf_bytes = export_optimization_report_pdf_bytes(
        df=df,
        df_out=df_out,
        fits=fits,
        intervals=intervals,
        organigram_by_eq=organigram_by_eq,
        detail_tables_by_eq=detail_tables_by_eq,
        meta=meta,
        title=title,
    )
    out_path.write_bytes(pdf_bytes)
    return str(out_path)
