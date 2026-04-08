
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import math

import numpy as np
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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
        .replace("θ", "theta")
        .replace("°", " deg")
        .replace("\u00A0", " ")
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _fmt(value: Any, nd: int = 2, dash: str = "—") -> str:
    v = _safe_float(value)
    if v is None:
        return dash
    return f"{v:.{nd}f}"


def _compact(value: Any, max_len: int = 160) -> str:
    text = _san(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _require_reportlab():
    if not HAVE_REPORTLAB:
        raise RuntimeError("ReportLab n’est pas disponible. Installe reportlab puis relance.")


def _auto_col_widths(data: List[List[Any]], total_width: float) -> List[float]:
    if not data:
        return []
    ncols = max(len(r) for r in data)
    lengths = [8] * ncols

    for row in data[:40]:
        for i in range(ncols):
            cell = row[i] if i < len(row) else ""
            lengths[i] = max(lengths[i], min(len(str(cell)), 60))

    weights = [max(x, 8) for x in lengths]
    s = sum(weights) if sum(weights) > 0 else ncols
    widths = [(w / s) * total_width for w in weights]

    if ncols <= 2:
        min_w, max_w = 40 * mm, 120 * mm
    elif ncols <= 6:
        min_w, max_w = 18 * mm, 55 * mm
    else:
        min_w, max_w = 14 * mm, 42 * mm

    widths = [min(max(w, min_w), max_w) for w in widths]
    total = sum(widths)
    if total > total_width:
        ratio = total_width / total
        widths = [w * ratio for w in widths]
    return widths


def _mk_table(data: List[List[Any]], total_width: float, font_size: int = 8):
    styles = getSampleStyleSheet()

    body = ParagraphStyle(
        name=f"body_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.3)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    head = ParagraphStyle(
        name=f"head_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(10, int(font_size * 1.25)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped = []
    for i, row in enumerate(data):
        style = head if i == 0 else body
        wrapped.append([Paragraph(_san(c).replace("\n", "<br/>"), style) for c in row])

    tbl = Table(
        wrapped,
        repeatRows=1,
        splitByRow=1,
        colWidths=_auto_col_widths(data, total_width=total_width),
    )
    tbl.setStyle(
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
    text = label.lower()
    if "correct" in text:
        return "Retenue quand les résultats suggèrent des défauts précoces ou un besoin de correction."
    if "condition" in text:
        return "Retenue quand la surveillance de l’état doit guider l’intervention."
    if "prévent" in text or "prevent" in text:
        return "Retenue quand l’usure ou l’intervalle optimal justifient une action planifiée."
    if "predict" in text or "prédict" in text:
        return "Retenue quand l’évolution des données permet d’anticiper le meilleur moment d’intervention."
    return "Type de maintenance retenu par synthèse des paramètres disponibles."


def _pipe_line(pipe: dict) -> str:
    if not isinstance(pipe, dict) or not pipe:
        return "Trace indisponible."

    rel = pipe.get("reliability", pipe)
    model = rel.get("model", "RP")
    dist = rel.get("distribution", "weibull_2p")
    good = rel.get("goodness", {}) or {}
    tests = rel.get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    dep = tests.get("dependence", {}) or {}

    return (
        f"TTF>0 -> MK(p={_fmt(mk.get('p'),3)}, dir={mk.get('direction','none')}) "
        f"-> Dep(r={_fmt(dep.get('r'),3)}, p={_fmt(dep.get('p'),3)}) "
        f"-> Modèle={model} ; Loi={dist} ; KS p={_fmt(good.get('ks_p'),3)} ; Chi carré p={_fmt(good.get('chi2_p'),3)}"
    )


def _plot_R_curves(fits: Dict[str, Any], intervals: Dict[str, Any]):
    if not fits:
        return None

    etas = [float(getattr(ft, "eta", 0.0) or 0.0) for ft in fits.values()]
    tmax = max(etas) * 1.6 if etas and max(etas) > 0 else 1000.0

    maybe = []
    for _, item in (intervals or {}).items():
        if isinstance(item, dict):
            for key in ["T_R", "T_cost"]:
                v = _safe_float(item.get(key))
                if v is not None:
                    maybe.append(v)
    if maybe:
        tmax = max(tmax, max(maybe) * 1.2)

    t = np.linspace(0, max(tmax, 1.0), 350)

    fig, ax = plt.subplots(figsize=(10, 5))
    for eq, ft in fits.items():
        beta = float(getattr(ft, "beta", 1.0) or 1.0)
        eta = float(getattr(ft, "eta", 1.0) or 1.0)
        gamma = float(getattr(ft, "gamma", 0.0) or 0.0)

        y = np.ones_like(t, dtype=float)
        mask = t > gamma
        y[mask] = np.exp(-(((t[mask] - gamma) / max(eta, 1e-12)) ** max(beta, 1e-12)))
        ax.plot(t, y, linewidth=2, label=f"{eq} (beta={beta:.2f}, eta={eta:.1f})")

        current = intervals.get(eq, {})
        if isinstance(current, dict):
            tr = _safe_float(current.get("T_R"))
            tc = _safe_float(current.get("T_cost"))
            if tr is not None:
                ax.axvline(tr, linestyle="--", linewidth=1)
            if tc is not None:
                ax.axvline(tc, linestyle=":", linewidth=1)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Temps (heures)")
    ax.set_ylabel("R(t)")
    ax.set_title("Courbes de fiabilité utilisées pour l’optimisation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_rl_image(fig, width_mm=180)


def _row_from_df_out(df_out, eq: str) -> Dict[str, Any]:
    if df_out is None or getattr(df_out, "empty", True):
        return {}
    if "equipment_code" not in df_out.columns:
        return {}

    matched = df_out[df_out["equipment_code"].astype(str) == str(eq)]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _summary_table_data(
    fits: Dict[str, Any],
    intervals: Dict[str, Any],
    organigram_by_eq: Dict[str, Any],
    df_out=None,
) -> List[List[Any]]:
    data = [[
        "Équipement",
        "beta",
        "eta (h)",
        "gamma (h)",
        "T_R (h)",
        "T_cost (h)",
        "T_recommandé (h)",
        "R(T_cost)",
        "C_min / h",
        "Maintenance retenue",
    ]]

    eqs = sorted(set(list(fits.keys()) + list((organigram_by_eq or {}).keys())))
    for eq in eqs:
        row = _row_from_df_out(df_out, eq)
        fit = fits.get(eq)
        itv = intervals.get(eq, {}) if isinstance(intervals.get(eq, {}), dict) else {}
        beta = _safe_float(row.get("beta") if row else None)
        eta = _safe_float(row.get("eta_h") if row else None)
        gamma = _safe_float(row.get("gamma_h") if row else None)

        if fit is not None:
            beta = beta if beta is not None else _safe_float(getattr(fit, "beta", None))
            eta = eta if eta is not None else _safe_float(getattr(fit, "eta", None))
            gamma = gamma if gamma is not None else _safe_float(getattr(fit, "gamma", None))

        data.append([
            _san(eq),
            _fmt(beta, 3),
            _fmt(eta, 1),
            _fmt(gamma, 1),
            _fmt(row.get("T_R_h") if row else itv.get("T_R"), 1),
            _fmt(row.get("T_cost_h") if row else itv.get("T_cost"), 1),
            _fmt(row.get("T_recommended_h") if row else None, 1),
            _fmt(row.get("R(T_cost)") if row else itv.get("R_at_T"), 3),
            _fmt(row.get("C_min_per_h") if row else itv.get("C_min"), 4),
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
        ["Processus retenu", model, "Le type de processus influence le niveau de prudence du plan proposé."],
        ["Loi retenue", distribution, "Cadre mathématique retenu pour l’évaluation fiabiliste."],
        ["Intervalle de fiabilité T_R", _fmt(row.get("T_R_h"), 1), "Repère issu de l’objectif de fiabilité."],
        ["Intervalle économique T_cost", _fmt(row.get("T_cost_h"), 1), "Repère minimisant le coût moyen."],
        ["Intervalle recommandé", _fmt(row.get("T_recommended_h"), 1), "Compromis final appliqué au planning."],
        ["Fiabilité à T_cost", _fmt(row.get("R(T_cost)"), 3), "Niveau de fiabilité conservé à l’intervalle économique."],
        ["Coût minimal par heure", _fmt(row.get("C_min_per_h"), 4), "Contribution économique dans le choix."],
        ["Jours avant maintenance", _fmt(row.get("days_recommended"), 1), "Échéance de planification issue de l’intervalle recommandé."],
        ["MTBF (h)", _fmt(row.get("MTBF_h"), 1), "Moyenne entre défaillances, différente de l’échéance planifiée."],
    ]


# ------------------------------------------------------------
# API 1 : PDF en mémoire
# ------------------------------------------------------------
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
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            wordWrap="CJK",
        )
    )

    buff = BytesIO()

    doc = SimpleDocTemplate(
        buff,
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
    story.append(Spacer(1, 10))

    nb_obs = int(getattr(df, "shape", [0])[0]) if df is not None else 0
    nb_eq = int(getattr(df_out, "shape", [0])[0]) if df_out is not None else len(fits)

    story.append(Paragraph("Résumé", styles["Heading2"]))
    story.append(Paragraph(_san(f"- Équipements analysés : {nb_eq}"), styles["Justify"]))
    story.append(Paragraph(_san(f"- Observations TTF : {nb_obs}"), styles["Justify"]))
    story.append(
        Paragraph(
            _san(
                "Le rapport d’optimisation présente uniquement les éléments utiles à la décision "
                "économique et à la planification de maintenance : paramètres Weibull, intervalles "
                "calculés, coût minimal et type de maintenance retenu. Aucune donnée thermique "
                "dynamique n’est injectée dans cette version."
            ),
            styles["Justify"],
        )
    )
    story.append(Spacer(1, 8))

    if meta:
        story.append(Paragraph("Paramètres utilisés", styles["Heading2"]))
        meta_rows = [["Paramètre", "Valeur"]]
        for key in ["alpha", "R_target", "C_prev", "C_corr", "R_min_cost"]:
            if key in meta:
                meta_rows.append([_san(key), _san(meta.get(key))])
        story.append(_mk_table(meta_rows, total_width=usable_width, font_size=7))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Synthèse globale de l’optimisation", styles["Heading2"]))
    story.append(
        _mk_table(
            _summary_table_data(
                fits=fits,
                intervals=intervals,
                organigram_by_eq=organigram_by_eq,
                df_out=df_out,
            ),
            total_width=usable_width,
            font_size=7,
        )
    )
    story.append(Spacer(1, 8))

    chart = _plot_R_curves(fits, intervals)
    if chart is not None:
        story.append(Paragraph("Courbes de fiabilité utilisées", styles["Heading2"]))
        story.append(chart)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    story.append(Paragraph("Rappel rapide sur les types de maintenance", styles["Heading2"]))
    data_types = [
        ["Type", "Définition courte", "Quand il est retenu"],
        ["Maintenance corrective", "Intervention après défaut ou pour corriger un comportement dégradé.", "Souvent liée aux défauts précoces ou à la fiabilisation."],
        ["Maintenance conditionnelle", "Intervention guidée par l’état observé.", "Quand la surveillance doit rester forte avant intervention."],
        ["Maintenance préventive planifiée", "Intervention programmée avant la panne.", "Quand l’usure ou l’intervalle optimal justifient une action planifiée."],
        ["Maintenance prédictive", "Intervention anticipée à partir des tendances de données.", "Quand les données permettent d’anticiper le bon moment d’action."],
    ]
    story.append(_mk_table(data_types, total_width=usable_width, font_size=7))
    story.append(Spacer(1, 10))

    eqs = sorted(set(list(fits.keys()) + list((organigram_by_eq or {}).keys())))
    for idx, eq in enumerate(eqs):
        pipe = organigram_by_eq.get(eq, {}) or {}
        row = _row_from_df_out(df_out, eq)

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(Paragraph(_compact(_pipe_line(pipe), 240), styles["Justify"]))
        story.append(Spacer(1, 5))

        maintenance_type = _maintenance_type(_safe_float(row.get("beta")), row.get("maintenance_type"))
        story.append(
            Paragraph(
                _san(
                    f"Choix retenu dans le cas d’étude : {maintenance_type}. "
                    f"Ce choix résulte de la combinaison entre les paramètres fiabilistes, "
                    f"l’intervalle optimisé et la contrainte de coût."
                ),
                styles["Justify"],
            )
        )
        story.append(Spacer(1, 5))

        story.append(Paragraph("Paramètres qui ont influencé le choix", styles["Heading3"]))
        story.append(_mk_table(_parameter_influence_table(row, pipe), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 6))

        if idx < len(eqs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buff.getvalue()


# ------------------------------------------------------------
# API 2 : Export sur disque
# ------------------------------------------------------------
def export_optimization_report_pdf(
    df,
    fits: Dict[str, Any],
    intervals: Dict[str, Any],
    organigram_by_eq: Dict[str, Any] | None = None,
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

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"report_optimisation_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

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
