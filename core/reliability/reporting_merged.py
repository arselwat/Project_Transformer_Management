
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List

import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm, cm
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

from scipy import stats as sst

from core.reliability.organigram import analyze_ttf_pipeline


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
        if math.isnan(v) or math.isinf(v):
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


def _auto_col_widths(data: List[List[Any]], total_width: float) -> List[float]:
    if not data:
        return []

    ncols = max(len(r) for r in data)
    lengths = [8] * ncols

    for row in data[:50]:
        for i in range(ncols):
            cell = row[i] if i < len(row) else ""
            lengths[i] = max(lengths[i], min(len(str(cell)), 60))

    weights = [max(x, 8) for x in lengths]
    s = sum(weights) if sum(weights) > 0 else ncols
    widths = [(w / s) * total_width for w in weights]

    if ncols <= 2:
        min_w, max_w = 35 * mm, 110 * mm
    elif ncols <= 5:
        min_w, max_w = 20 * mm, 60 * mm
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


def _fig_to_rl_image(fig, width_mm: float = 170):
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


def _series_to_positive_list(series: pd.Series) -> Optional[list[float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return values.astype(float).tolist()


def _recommend_maintenance_from_beta(beta_value: Any) -> str:
    beta = _safe_float(beta_value)
    if beta is None:
        return "Maintenance à confirmer"
    if beta < 0.9:
        return "Maintenance corrective / fiabilisation"
    if beta <= 1.1:
        return "Maintenance conditionnelle"
    return "Maintenance préventive planifiée"


def _build_pipeline_text(reliability_result: Dict[str, Any]) -> str:
    tests = reliability_result.get("tests", {}) or {}
    decision = reliability_result.get("decision", {}) or {}
    goodness = reliability_result.get("goodness", {}) or {}

    mk = tests.get("trend_mk", {}) or {}
    lap = tests.get("trend_laplace", {}) or {}
    dep = tests.get("dependence", {}) or {}

    return (
        f"Pipeline : tendance par Mann-Kendall (p={_fmt(mk.get('p'), 3)}) et Laplace "
        f"(p={_fmt(lap.get('p'), 3)}), puis dépendance (r={_fmt(dep.get('r'), 3)}, "
        f"p={_fmt(dep.get('p'), 3)}), puis choix du modèle {reliability_result.get('model', '—')} "
        f"avec la loi {reliability_result.get('distribution', '—')}. Qualité d'ajustement : "
        f"KS p={_fmt(goodness.get('ks_p'), 3)}, Chi carré p={_fmt(goodness.get('chi2_p'), 3)}, "
        f"Cramér-von Mises p={_fmt(goodness.get('cvm_p'), 3)}. "
        f"Conclusion métier : {_compact(decision.get('reason', '—'), 180)}"
    )


def _build_recommendation_text(reliability_result: Dict[str, Any]) -> str:
    params = reliability_result.get("params", {}) or {}
    indicators = reliability_result.get("indicators", {}) or {}
    beta = _safe_float(params.get("beta"))
    mttf = indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h")
    mtbf = indicators.get("mtbf_h")
    mttr = indicators.get("mttr_h")
    availability = indicators.get("availability_intrinsic")
    maintenance = _recommend_maintenance_from_beta(beta)

    text = (
        f"Le type de maintenance proposé est : {maintenance}. "
        f"Le paramètre beta vaut {_fmt(beta, 3)}, le paramètre eta vaut {_fmt(params.get('eta'), 1)} h "
        f"et le paramètre gamma vaut {_fmt(params.get('gamma'), 1)} h. "
        f"Le MTTF est estimé à {_fmt(mttf, 1)} h, le MTBF à {_fmt(mtbf, 1)} h, "
        f"le MTTR à {_fmt(mttr, 1)} h et la disponibilité intrinsèque à "
        f"{_fmt(None if availability is None else 100.0 * float(availability), 2)} %. "
        f"Cette recommandation est donc déduite uniquement des résultats fiabilistes."
    )
    return text


def _get_distribution_and_parameters(reliability_result: Dict[str, Any]):
    if str(reliability_result.get("model") or "").upper() != "RP":
        return None, None

    distribution_name = reliability_result.get("distribution")
    raw_parameters = (reliability_result.get("params") or {}).get("raw")
    if not raw_parameters:
        return None, None

    if distribution_name == "expon":
        return sst.expon, raw_parameters
    if distribution_name == "norm":
        return sst.norm, raw_parameters
    if distribution_name == "lognorm":
        return sst.lognorm, raw_parameters
    if distribution_name in {"weibull_2p", "weibull_3p"}:
        return sst.weibull_min, raw_parameters
    return None, None


def _compute_rp_curve(reliability_result: Dict[str, Any], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    distribution_object, distribution_parameters = _get_distribution_and_parameters(reliability_result)
    if distribution_object is None or distribution_parameters is None:
        return None

    try:
        if curve_name == "survival":
            values = distribution_object.sf(time_axis, *distribution_parameters)
        elif curve_name == "cdf":
            values = distribution_object.cdf(time_axis, *distribution_parameters)
        elif curve_name == "pdf":
            values = distribution_object.pdf(time_axis, *distribution_parameters)
        elif curve_name == "hazard":
            survival_values = distribution_object.sf(time_axis, *distribution_parameters)
            density_values = distribution_object.pdf(time_axis, *distribution_parameters)
            values = np.divide(
                density_values,
                survival_values,
                out=np.full_like(density_values, np.nan, dtype=float),
                where=survival_values > 1e-12,
            )
        else:
            return None
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def _compute_nhpp_curve(reliability_result: Dict[str, Any], time_axis: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    beta_value = _safe_float(parameters.get("beta"))
    eta_value = _safe_float(parameters.get("eta"))

    if beta_value is None or eta_value is None or beta_value <= 0 or eta_value <= 0:
        return None

    safe_time_axis = np.maximum(time_axis, 1e-6)
    mean_cumulative_events = (safe_time_axis / eta_value) ** beta_value
    intensity = (beta_value / eta_value) * ((safe_time_axis / eta_value) ** (beta_value - 1.0))
    survival_like = np.exp(-mean_cumulative_events)
    cumulative_probability = 1.0 - survival_like
    density_like = intensity * survival_like

    if curve_name == "survival":
        return survival_like
    if curve_name == "cdf":
        return cumulative_probability
    if curve_name == "pdf":
        return density_like
    if curve_name == "hazard":
        return intensity
    return None


def _compute_bpp_curve(
    reliability_result: Dict[str, Any],
    time_axis: np.ndarray,
    ttf_series: list[float],
    curve_name: str,
) -> Optional[np.ndarray]:
    parameters = reliability_result.get("params", {}) or {}
    mu_value = _safe_float(parameters.get("mu"))
    alpha_value = _safe_float(parameters.get("alpha"))
    beta_kernel_value = _safe_float(parameters.get("beta_kernel"))

    if mu_value is None or alpha_value is None or beta_kernel_value is None:
        return None
    if mu_value < 0 or alpha_value < 0 or beta_kernel_value <= 0:
        return None

    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    if event_times.size == 0:
        return None

    safe_time_axis = np.maximum(time_axis, 1e-6)
    intensity = np.full_like(safe_time_axis, fill_value=mu_value, dtype=float)

    for event_time in event_times:
        mask = safe_time_axis >= event_time
        if np.any(mask):
            intensity[mask] += alpha_value * np.exp(-beta_kernel_value * (safe_time_axis[mask] - event_time))

    cumulative_intensity = np.zeros_like(safe_time_axis, dtype=float)
    if len(safe_time_axis) > 1:
        delta = np.diff(safe_time_axis)
        trapezoids = 0.5 * (intensity[1:] + intensity[:-1]) * delta
        cumulative_intensity[1:] = np.cumsum(trapezoids)

    survival_like = np.exp(-cumulative_intensity)
    cumulative_probability = 1.0 - survival_like
    density_like = intensity * survival_like

    if curve_name == "survival":
        return survival_like
    if curve_name == "cdf":
        return cumulative_probability
    if curve_name == "pdf":
        return density_like
    if curve_name == "hazard":
        return intensity
    return None


def _compute_model_curve(
    reliability_result: Dict[str, Any],
    ttf_series: list[float],
    time_axis: np.ndarray,
    curve_name: str,
) -> Optional[np.ndarray]:
    model_name = str(reliability_result.get("model") or "").upper()
    if model_name == "RP":
        return _compute_rp_curve(reliability_result, time_axis, curve_name)
    if model_name == "NHPP":
        return _compute_nhpp_curve(reliability_result, time_axis, curve_name)
    if model_name == "BPP":
        return _compute_bpp_curve(reliability_result, time_axis, ttf_series, curve_name)
    return None


def _build_time_horizon_for_equipment(reliability_result: Dict[str, Any], ttf_series: list[float]) -> float:
    parameters = reliability_result.get("params", {}) or {}
    values = np.asarray(ttf_series, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size == 0:
        return 100.0

    max_ttf = float(np.max(values))
    mean_ttf = float(np.mean(values))
    q90_ttf = float(np.quantile(values, 0.90))
    eta_value = _safe_float(parameters.get("eta"))

    base_horizon = max(50.0, max_ttf * 2.0, mean_ttf * 6.0, q90_ttf * 4.0)
    if eta_value is not None and eta_value > 0:
        return max(base_horizon, eta_value * 1.5)
    return base_horizon


def _plot_equipment_curves(eq: str, reliability_result: Dict[str, Any], ttf_series: list[float]):
    time_horizon = _build_time_horizon_for_equipment(reliability_result, ttf_series)
    time_axis = np.linspace(1e-6, time_horizon, 400)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()

    definitions = [
        ("survival", "Fiabilité R(t)", "R(t)"),
        ("cdf", "Répartition F(t)", "F(t)"),
        ("pdf", "Densité f(t)", "f(t)"),
        ("hazard", "Taux de défaillance h(t)", "h(t) ou λ(t)"),
    ]

    for ax, (curve_name, title, ylabel) in zip(axes, definitions):
        values = _compute_model_curve(reliability_result, ttf_series, time_axis, curve_name)
        if values is None:
            ax.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(time_axis, values, linewidth=2)
        ax.set_title(f"{title} - {eq}")
        ax.set_xlabel("Temps (heures)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_rl_image(fig, width_mm=170)


def _summary_table(results_by_equipment: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    data = [[
        "Équipement",
        "Processus",
        "Variant",
        "Loi",
        "beta",
        "eta (h)",
        "gamma (h)",
        "MTTF (h)",
        "MTBF (h)",
        "MTTR (h)",
        "Disponibilité (%)",
        "Maintenance recommandée",
    ]]

    for eq, result in results_by_equipment.items():
        reliability_result = result.get("reliability", {}) or {}
        indicators = reliability_result.get("indicators", {}) or {}
        params = reliability_result.get("params", {}) or {}
        availability = indicators.get("availability_intrinsic")

        data.append([
            _san(eq),
            reliability_result.get("model", "—"),
            reliability_result.get("process_variant", "—"),
            reliability_result.get("distribution", "—"),
            _fmt(params.get("beta"), 3),
            _fmt(params.get("eta"), 1),
            _fmt(params.get("gamma"), 1),
            _fmt(indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"), 1),
            _fmt(indicators.get("mtbf_h"), 1),
            _fmt(indicators.get("mttr_h"), 1),
            _fmt(None if availability is None else 100.0 * float(availability), 2),
            _recommend_maintenance_from_beta(params.get("beta")),
        ])
    return data


def _parameter_table(reliability_result: Dict[str, Any]) -> List[List[Any]]:
    indicators = reliability_result.get("indicators", {}) or {}
    params = reliability_result.get("params", {}) or {}
    availability = indicators.get("availability_intrinsic")

    return [
        ["Paramètre", "Valeur", "Lecture"],
        ["Processus retenu", reliability_result.get("model", "—"), "Type de comportement global retenu."],
        ["Variant du processus", reliability_result.get("process_variant", "—"), "Précision du cadre retenu."],
        ["Loi retenue", reliability_result.get("distribution", "—"), "Loi de probabilité utilisée pour l'analyse."],
        ["beta", _fmt(params.get("beta"), 3), "Défauts précoces, aléatoire ou usure."],
        ["eta (h)", _fmt(params.get("eta"), 1), "Durée de vie caractéristique."],
        ["gamma (h)", _fmt(params.get("gamma"), 1), "Décalage éventuel du modèle."],
        ["MTTF (h)", _fmt(indicators.get("theoretical_mttf_h") or indicators.get("empirical_mttf_h"), 1), "Temps moyen avant défaillance."],
        ["MTBF (h)", _fmt(indicators.get("mtbf_h"), 1), "Temps moyen entre défaillances."],
        ["MTTR (h)", _fmt(indicators.get("mttr_h"), 1), "Temps moyen de réparation."],
        ["Disponibilité (%)", _fmt(None if availability is None else 100.0 * float(availability), 2), "Part du temps où l'équipement reste disponible."],
        ["Type de maintenance", _recommend_maintenance_from_beta(params.get("beta")), "Recommandation déduite du paramètre beta."],
    ]


def _fit_table(reliability_result: Dict[str, Any]) -> List[List[Any]]:
    goodness = reliability_result.get("goodness", {}) or {}
    return [
        ["Indicateur", "Valeur", "Lecture"],
        ["AIC", _fmt(goodness.get("aic"), 3), "Plus il est faible, plus le compromis ajustement/complexité est favorable."],
        ["KS p", _fmt(goodness.get("ks_p"), 4), "Valeur p du test de Kolmogorov-Smirnov."],
        ["Chi carré p", _fmt(goodness.get("chi2_p"), 4), "Valeur p du test du chi carré."],
        ["CvM p", _fmt(goodness.get("cvm_p"), 4), "Valeur p du test de Cramér-von Mises."],
        ["Ajustement accepté", _san(goodness.get("accepted", "—")), "Décision globale sur l'acceptabilité du modèle."],
    ]


def _analyze_from_dataframe(df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty:
        return {}

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    if "equipment_code" not in work.columns or "ttf_h" not in work.columns:
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for eq in sorted(work["equipment_code"].astype(str).unique().tolist()):
        eq_df = work[work["equipment_code"].astype(str) == str(eq)].copy()
        ttf_list = _series_to_positive_list(eq_df["ttf_h"])
        if not ttf_list or len(ttf_list) < 3:
            continue

        repair_list = None
        if "duree_rep_h" in eq_df.columns:
            repair_list = _series_to_positive_list(eq_df["duree_rep_h"])

        try:
            results[str(eq)] = analyze_ttf_pipeline(
                ttf_series=ttf_list,
                alpha=float(alpha),
                repair_series=repair_list,
            )
        except Exception:
            continue
    return results


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
def export_merged_report_pdf(
    df: Optional[pd.DataFrame] = None,
    out_dir: str = "reports",
    title: str = "Rapport - Indicateurs fiabilistes",
    analysis_results: Optional[Dict[str, Dict[str, Any]]] = None,
    alpha: float = 0.05,
) -> str:
    if not HAVE_REPORTLAB:
        raise RuntimeError("ReportLab non disponible. Installe reportlab puis relance.")

    results_by_equipment = analysis_results or _analyze_from_dataframe(df, alpha=alpha)
    if not results_by_equipment:
        raise RuntimeError("Aucun résultat exploitable pour générer le rapport des indicateurs.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"report_indicateurs_{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"

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

    doc = SimpleDocTemplate(
        str(fpath),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
    )
    usable_width = A4[0] - (doc.leftMargin + doc.rightMargin)

    story: List[Any] = []

    story.append(Paragraph(_san(title), styles["Title"]))
    story.append(Paragraph(_san(datetime.now().strftime("%d/%m/%Y %H:%M")), styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Résumé du rapport", styles["Heading2"]))
    story.append(
        Paragraph(
            _san(
                "Ce document regroupe exclusivement les indicateurs fiabilistes utilisés par la page Indicateurs. "
                "Il présente la synthèse fiabiliste, les paramètres beta, eta et gamma, les grandeurs MTTF, MTBF et MTTR, "
                "la disponibilité, le pipeline de décision, ainsi que les courbes R(t), F(t), f(t) et h(t). "
                "Il est destiné à être partagé avec les équipes de maintenance pour faciliter la compréhension des résultats fiabilistes et la prise de décision."
                
            ),
            styles["Justify"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Synthèse globale", styles["Heading2"]))
    story.append(_mk_table(_summary_table(results_by_equipment), total_width=usable_width, font_size=7))
    story.append(Spacer(1, 10))

    eqs = sorted(results_by_equipment.keys())
    for idx, eq in enumerate(eqs):
        result = results_by_equipment[eq]
        reliability_result = result.get("reliability", {}) or {}

        ttf_series = []
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty and "equipment_code" in df.columns and "ttf_h" in df.columns:
            eq_df = df[df["equipment_code"].astype(str) == str(eq)].copy()
            ttf_series = _series_to_positive_list(eq_df["ttf_h"]) or []

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(Paragraph(_san(_build_pipeline_text(reliability_result)), styles["Justify"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Paramètres et indicateurs", styles["Heading3"]))
        story.append(_mk_table(_parameter_table(reliability_result), total_width=usable_width, font_size=8))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Qualité d'ajustement", styles["Heading3"]))
        story.append(_mk_table(_fit_table(reliability_result), total_width=usable_width, font_size=8))
        story.append(Spacer(1, 6))

        if ttf_series:
            story.append(Paragraph("Courbes fiabilistes", styles["Heading3"]))
            story.append(_plot_equipment_curves(eq, reliability_result, ttf_series))
            story.append(Spacer(1, 6))

        story.append(Paragraph("Recommandation de maintenance basée sur beta", styles["Heading3"]))
        story.append(Paragraph(_san(_build_recommendation_text(reliability_result)), styles["Justify"]))
        story.append(Spacer(1, 8))

        if idx < len(eqs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(fpath)
