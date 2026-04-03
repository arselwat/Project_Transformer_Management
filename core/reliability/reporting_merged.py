from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import numpy as np
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
    from core.reliability.unify import compute_bundle, UnifyOptions
except Exception:
    compute_bundle = None
    UnifyOptions = None


# ============================================================
# Helpers
# ============================================================
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


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _fmt(value: Any, nd: int = 2, dash: str = "—") -> str:
    v = _safe_float(value, None)
    if v is None:
        return dash
    return f"{v:.{nd}f}"


def _compact(value: Any, max_len: int = 120) -> str:
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
            lengths[i] = max(lengths[i], min(len(str(cell)), 50))

    weights = [max(x, 8) for x in lengths]
    s = sum(weights) if sum(weights) > 0 else ncols
    widths = [(w / s) * total_width for w in weights]

    min_w = 18 * mm
    max_w = 85 * mm
    widths = [min(max(w, min_w), max_w) for w in widths]

    total = sum(widths)
    if total > total_width:
        ratio = total_width / total
        widths = [w * ratio for w in widths]

    return widths


def _mk_table(data: List[List[Any]], total_width: float, font_size: int = 8) -> "Table":
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        name=f"body_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(9, int(font_size * 1.3)),
        alignment=TA_LEFT,
        wordWrap="CJK",
    )

    head_style = ParagraphStyle(
        name=f"head_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=max(font_size, 8),
        leading=max(10, int(font_size * 1.3)),
        alignment=TA_CENTER,
        wordWrap="CJK",
    )

    wrapped = []
    for row_idx, row in enumerate(data):
        style = head_style if row_idx == 0 else body_style
        wrapped.append([Paragraph(_san(c).replace("\n", "<br/>"), style) for c in row])

    tbl = Table(
        wrapped,
        repeatRows=1,
        colWidths=_auto_col_widths(data, total_width=total_width),
        splitByRow=1,
    )
    tbl.setStyle(
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
    return tbl


def _get_rel(result: Dict[str, Any]) -> Dict[str, Any]:
    return (result.get("reliability", {}) if isinstance(result, dict) else {}) or {}


def _get_thermal(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    thermal = result.get("thermal") if isinstance(result, dict) else None
    return thermal if isinstance(thermal, dict) else None


def _get_summary_rows(analysis_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    data = [[
        "Équipement",
        "Processus",
        "Loi",
        "Tendance",
        "Dépendance",
        "beta",
        "eta (h)",
        "gamma (h)",
        "MTBF (h)",
        "MTTR (h)",
        "Disponibilité (%)",
    ]]

    for eq, result in sorted(analysis_results.items()):
        rel = _get_rel(result)
        indicators = rel.get("indicators", {}) or {}
        tests = rel.get("tests", {}) or {}
        decision = rel.get("decision", {}) or {}
        params = rel.get("params", {}) or {}

        availability = None
        if indicators.get("availability_intrinsic") is not None:
            availability = 100.0 * float(indicators.get("availability_intrinsic"))

        data.append([
            _san(eq),
            _san(rel.get("model", "—")),
            _san(rel.get("distribution", "—")),
            "Oui" if decision.get("has_trend") else "Non",
            "Oui" if decision.get("has_dependence") else "Non",
            _fmt(params.get("beta"), 3),
            _fmt(params.get("eta"), 1),
            _fmt(params.get("gamma"), 1),
            _fmt(indicators.get("mtbf_h"), 1),
            _fmt(indicators.get("mttr_h"), 1),
            _fmt(availability, 2),
        ])

    return data


def _get_thermal_rows(analysis_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    data = [[
        "Équipement",
        "Température max du point chaud (degC)",
        "FAA max",
        "Perte de vie (%)",
    ]]

    found = False
    for eq, result in sorted(analysis_results.items()):
        thermal = _get_thermal(result)
        summary = (thermal or {}).get("summary", {}) if thermal else {}
        if not summary:
            continue

        found = True
        data.append([
            _san(eq),
            _fmt(summary.get("theta_hs_max"), 2),
            _fmt(summary.get("faa_max"), 3),
            _fmt(summary.get("loss_of_life_pct"), 3),
        ])

    if not found:
        return [["Information"], ["Aucune donnée thermique exploitable"]]
    return data


def _build_trend_table(result: Dict[str, Any]) -> List[List[Any]]:
    rel = _get_rel(result)
    tests = rel.get("tests", {}) or {}
    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}
    combined = tests.get("trend_combined", {}) or {}

    return [
        ["Élément", "Valeur", "Lecture"],
        ["Mann-Kendall", f"z={_fmt(mk.get('z'),3)} ; p={_fmt(mk.get('p'),4)}", "Détecte une tendance globale."],
        ["Laplace", f"u={_fmt(lap.get('u'),3)} ; p={_fmt(lap.get('p'),4)}", "Confirme ou non une évolution dans le temps."],
        ["Décision finale", "Oui" if combined.get("has_trend") else "Non", f"Sens : {_san(combined.get('direction', 'none'))}"],
    ]


def _build_dependence_table(result: Dict[str, Any]) -> List[List[Any]]:
    rel = _get_rel(result)
    dep = (rel.get("tests", {}) or {}).get("dependence", {}) or {}

    return [
        ["Élément", "Valeur", "Lecture"],
        ["Pearson", f"r={_fmt(dep.get('pearson_r'),3)} ; p={_fmt(dep.get('pearson_p'),4)}", "Dépendance linéaire."],
        ["Spearman", f"r={_fmt(dep.get('spearman_r'),3)} ; p={_fmt(dep.get('spearman_p'),4)}", "Dépendance monotone."],
        ["Décision finale", "Oui" if dep.get("has_dep") else "Non", f"Force : {_san(dep.get('strength', '—'))}"],
    ]


def _build_model_table(result: Dict[str, Any]) -> List[List[Any]]:
    rel = _get_rel(result)
    decision = rel.get("decision", {}) or {}

    return [
        ["Paramètre", "Valeur"],
        ["Processus retenu", _san(rel.get("model", "—"))],
        ["Variant", _san(rel.get("process_variant", "—"))],
        ["Loi retenue", _san(rel.get("distribution", "—"))],
        ["Justification", _compact(decision.get("reason", "—"), 120)],
    ]


def _build_fit_table(result: Dict[str, Any]) -> List[List[Any]]:
    rel = _get_rel(result)
    goodness = rel.get("goodness", {}) or {}

    return [
        ["Indicateur", "Valeur", "Lecture"],
        ["AIC", _fmt(goodness.get("aic"), 3), "Plus faible = meilleur compromis ajustement / complexité."],
        ["Kolmogorov-Smirnov p", _fmt(goodness.get("ks_p"), 4), "Plus la p-valeur est élevée, plus l’ajustement est acceptable."],
        ["Chi carré p", _fmt(goodness.get("chi2_p"), 4), "Vérification complémentaire de l’ajustement."],
        ["Cramér-von Mises p", _fmt(goodness.get("cvm_p"), 4), "Vérification complémentaire de la qualité d’ajustement."],
        ["Ajustement accepté", _san(goodness.get("accepted", "—")), "Décision finale sur l’acceptabilité statistique."],
    ]


def _build_parameter_table(result: Dict[str, Any]) -> List[List[Any]]:
    rel = _get_rel(result)
    indicators = rel.get("indicators", {}) or {}
    params = rel.get("params", {}) or {}

    availability = None
    if indicators.get("availability_intrinsic") is not None:
        availability = 100.0 * float(indicators.get("availability_intrinsic"))

    return [
        ["Paramètre", "Valeur", "Lecture"],
        ["beta", _fmt(params.get("beta"), 3), "Décrit la phase de vie du système."],
        ["eta (h)", _fmt(params.get("eta"), 1), "Durée de vie caractéristique."],
        ["gamma (h)", _fmt(params.get("gamma"), 1), "Décalage éventuel du modèle."],
        ["MTTF (h)", _fmt(indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"), 1), "Temps moyen avant défaillance."],
        ["MTBF (h)", _fmt(indicators.get("mtbf_h"), 1), "Temps moyen entre défaillances."],
        ["MTTR (h)", _fmt(indicators.get("mttr_h"), 1), "Temps moyen de réparation."],
        ["Disponibilité (%)", _fmt(availability, 2), "Part du temps où l’équipement reste disponible."],
    ]


def _build_thermal_table(result: Dict[str, Any]) -> List[List[Any]]:
    thermal = _get_thermal(result)
    summary = (thermal or {}).get("summary", {}) if thermal else {}

    if not summary:
        return [["Information"], ["Aucune donnée thermique exploitable"]]

    return [
        ["Paramètre thermique", "Valeur", "Lecture"],
        ["Température max du point chaud (degC)", _fmt(summary.get("theta_hs_max"), 2), "Température maximale estimée dans la zone la plus chaude."],
        ["FAA max", _fmt(summary.get("faa_max"), 3), "Accélération maximale du vieillissement thermique."],
        ["Perte de vie (%)", _fmt(summary.get("loss_of_life_pct"), 3), "Part estimée de vie déjà consommée."],
    ]


def _analysis_results_from_bundle(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if compute_bundle is None or UnifyOptions is None:
        raise RuntimeError("Impossible de reconstruire les résultats : module unify indisponible.")

    bundle = compute_bundle(session_df=df, options=UnifyOptions(force_weibull_2p=True, R_target=0.80))
    out: Dict[str, Dict[str, Any]] = {}
    for eq, pipe in (bundle.pipeline_by_eq or {}).items():
        if isinstance(pipe, dict):
            out[str(eq)] = pipe
    return out


def export_merged_report_pdf(
    df: Optional[pd.DataFrame] = None,
    out_dir: str = "reports",
    title: str = "Rapport des indicateurs",
    analysis_results: Optional[Dict[str, Dict[str, Any]]] = None,
    options=None,
) -> str:
    _require_reportlab()

    if analysis_results is None:
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("Aucune donnée disponible pour générer le rapport des indicateurs.")
        analysis_results = _analysis_results_from_bundle(df)

    if not analysis_results:
        raise RuntimeError("Aucun résultat exploitable pour générer le rapport des indicateurs.")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"report_indicateurs_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=landscape(A4),
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        title=_san(title),
    )
    usable_width = landscape(A4)[0] - (doc.leftMargin + doc.rightMargin)

    story: List[Any] = []

    # --------------------------------------------------------
    # Couverture
    # --------------------------------------------------------
    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Ce rapport présente uniquement les indicateurs utiles à la lecture des résultats : tendance, dépendance, modèle retenu, paramètres fiabilistes et synthèse thermique.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # Synthèse globale
    # --------------------------------------------------------
    story.append(Paragraph("Synthèse fiabiliste", styles["Heading2"]))
    story.append(_mk_table(_get_summary_rows(analysis_results), total_width=usable_width, font_size=7))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Synthèse thermique", styles["Heading2"]))
    story.append(_mk_table(_get_thermal_rows(analysis_results), total_width=usable_width, font_size=7))
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # --------------------------------------------------------
    # Détail par équipement
    # --------------------------------------------------------
    for idx, (eq, result) in enumerate(sorted(analysis_results.items())):
        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))

        rel = _get_rel(result)
        decision = rel.get("decision", {}) or {}
        if decision.get("reason"):
            story.append(Paragraph(_compact(decision.get("reason"), 200), styles["BodyText"]))
            story.append(Spacer(1, 5))

        story.append(Paragraph("Tests de tendance", styles["Heading3"]))
        story.append(_mk_table(_build_trend_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 5))

        story.append(Paragraph("Tests de dépendance", styles["Heading3"]))
        story.append(_mk_table(_build_dependence_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 5))

        story.append(Paragraph("Choix du processus", styles["Heading3"]))
        story.append(_mk_table(_build_model_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 5))

        story.append(Paragraph("Qualité d’ajustement", styles["Heading3"]))
        story.append(_mk_table(_build_fit_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 5))

        story.append(Paragraph("Paramètres fiabilistes", styles["Heading3"]))
        story.append(_mk_table(_build_parameter_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 5))

        story.append(Paragraph("Paramètres thermiques", styles["Heading3"]))
        story.append(_mk_table(_build_thermal_table(result), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 8))

        if idx < len(analysis_results) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(out_path)