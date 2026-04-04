
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import math
import numpy as np
import pandas as pd
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

from scipy import stats as sst


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


def _compact(value: Any, max_len: int = 180) -> str:
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


def _distribution_object_and_params(reliability_result: Dict[str, Any]):
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


def _compute_rp_curve(reliability_result: Dict[str, Any], t: np.ndarray, kind: str) -> Optional[np.ndarray]:
    dist_obj, params = _distribution_object_and_params(reliability_result)
    if dist_obj is None:
        return None
    try:
        if kind == "R":
            return np.asarray(dist_obj.sf(t, *params), dtype=float)
        if kind == "F":
            return np.asarray(dist_obj.cdf(t, *params), dtype=float)
        if kind == "f":
            return np.asarray(dist_obj.pdf(t, *params), dtype=float)
        if kind == "h":
            sf = dist_obj.sf(t, *params)
            pdf = dist_obj.pdf(t, *params)
            return np.divide(pdf, sf, out=np.full_like(pdf, np.nan, dtype=float), where=sf > 1e-12)
    except Exception:
        return None
    return None


def _compute_nhpp_curve(reliability_result: Dict[str, Any], t: np.ndarray, kind: str) -> Optional[np.ndarray]:
    params = reliability_result.get("params", {}) or {}
    beta = _safe_float(params.get("beta"))
    eta = _safe_float(params.get("eta"))
    if beta is None or eta is None or beta <= 0 or eta <= 0:
        return None
    safe_t = np.maximum(t, 1e-6)
    cumulative = (safe_t / eta) ** beta
    intensity = (beta / eta) * ((safe_t / eta) ** (beta - 1.0))
    R = np.exp(-cumulative)
    F = 1.0 - R
    f = intensity * R
    if kind == "R":
        return R
    if kind == "F":
        return F
    if kind == "f":
        return f
    if kind == "h":
        return intensity
    return None


def _compute_bpp_curve(reliability_result: Dict[str, Any], ttf_series: List[float], t: np.ndarray, kind: str) -> Optional[np.ndarray]:
    params = reliability_result.get("params", {}) or {}
    mu = _safe_float(params.get("mu"))
    alpha = _safe_float(params.get("alpha"))
    beta_kernel = _safe_float(params.get("beta_kernel"))
    if mu is None or alpha is None or beta_kernel is None:
        return None
    if mu < 0 or alpha < 0 or beta_kernel <= 0:
        return None
    event_times = np.cumsum(np.asarray(ttf_series, dtype=float))
    if event_times.size == 0:
        return None
    safe_t = np.maximum(t, 1e-6)
    intensity = np.full_like(safe_t, fill_value=mu, dtype=float)
    for event_time in event_times:
        mask = safe_t >= event_time
        if np.any(mask):
            intensity[mask] += alpha * np.exp(-beta_kernel * (safe_t[mask] - event_time))
    cumulative = np.zeros_like(safe_t)
    if len(safe_t) > 1:
        dt = np.diff(safe_t)
        cumulative[1:] = np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * dt)
    R = np.exp(-cumulative)
    F = 1.0 - R
    f = intensity * R
    if kind == "R":
        return R
    if kind == "F":
        return F
    if kind == "f":
        return f
    if kind == "h":
        return intensity
    return None


def _compute_curve(reliability_result: Dict[str, Any], ttf_series: List[float], t: np.ndarray, kind: str) -> Optional[np.ndarray]:
    model = str(reliability_result.get("model") or "").upper()
    if model == "RP":
        return _compute_rp_curve(reliability_result, t, kind)
    if model == "NHPP":
        return _compute_nhpp_curve(reliability_result, t, kind)
    if model == "BPP":
        return _compute_bpp_curve(reliability_result, ttf_series, t, kind)
    return None


def _time_horizon(reliability_result: Dict[str, Any], ttf_series: List[float], extra_values: Optional[List[float]] = None) -> float:
    values = np.asarray(ttf_series, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return 100.0
    params = reliability_result.get("params", {}) or {}
    eta = _safe_float(params.get("eta"))
    horizon = max(float(values.max()) * 2.0, float(values.mean()) * 6.0, 50.0)
    if eta is not None and eta > 0:
        horizon = max(horizon, eta * 1.5)
    extra_values = extra_values or []
    extras = [float(v) for v in extra_values if v is not None and v > 0]
    if extras:
        horizon = max(horizon, max(extras) * 1.25)
    return horizon


def _plot_optimization_curves(eq: str, reliability_result: Dict[str, Any], ttf_series: List[float], row: Dict[str, Any]):
    tr = _safe_float(row.get("T_R_h"))
    tc = _safe_float(row.get("T_cost_h"))
    rec = _safe_float(row.get("T_recommended_h"))
    horizon = _time_horizon(reliability_result, ttf_series, extra_values=[tr, tc, rec])
    t = np.linspace(1e-6, horizon, 400)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    definitions = [
        ("R", "Fiabilité R(t)", "R(t)"),
        ("F", "Répartition F(t)", "F(t)"),
        ("f", "Densité f(t)", "f(t)"),
        ("h", "Taux de défaillance λ(t)", "λ(t)"),
    ]
    for ax, (kind, title, ylabel) in zip(axes, definitions):
        values = _compute_curve(reliability_result, ttf_series, t, kind)
        if values is None:
            ax.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(t, values, linewidth=2, label=f"{eq} - {title}")

        if rec is not None:
            ax.axvline(rec, color="green", linestyle="-", linewidth=2.2, label=f"T_recommandé = {rec:.1f} h")
        if tr is not None:
            color_tr = "green" if rec is not None and abs(rec - tr) < 1e-9 else "red"
            label_tr = f"T_R = {tr:.1f} h"
            if rec is not None and abs(rec - tr) < 1e-9:
                label_tr += " (retenu)"
            ax.axvline(tr, color=color_tr, linestyle="--", linewidth=1.6, label=label_tr)
        if tc is not None:
            color_tc = "green" if rec is not None and abs(rec - tc) < 1e-9 else "red"
            label_tc = f"T_cost = {tc:.1f} h"
            if rec is not None and abs(rec - tc) < 1e-9:
                label_tc += " (retenu)"
            ax.axvline(tc, color=color_tc, linestyle=":", linewidth=1.8, label=label_tc)

        ax.set_xlabel("Temps (heures)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        if unique_labels:
            ax.legend(unique_handles, unique_labels, fontsize=8)

    fig.suptitle(f"Courbes fiabilistes et intervalles d’optimisation - {eq}", fontsize=12)
    fig.tight_layout()
    return _fig_to_rl_image(fig, width_mm=180)


def _summary_table_data(df_out: pd.DataFrame) -> List[List[Any]]:
    data = [[
        "Équipement", "Processus", "Loi", "beta", "eta (h)", "gamma (h)",
        "MTTF (h)", "MTBF (h)", "MTTR (h)", "Disponibilité (%)",
        "T_R (h)", "T_cost (h)", "R(T_cost)", "T_recommandé (h)",
        "Jours retenus", "Source retenue", "Maintenance retenue"
    ]]
    for _, row in df_out.iterrows():
        data.append([
            _san(row.get("equipment_code")),
            _san(row.get("model", "—")),
            _san(row.get("distribution", "—")),
            _fmt(row.get("beta"), 3),
            _fmt(row.get("eta_h"), 1),
            _fmt(row.get("gamma_h"), 1),
            _fmt(row.get("MTTF_h"), 1),
            _fmt(row.get("MTBF_h"), 1),
            _fmt(row.get("MTTR_h"), 1),
            _fmt(row.get("availability_pct"), 2),
            _fmt(row.get("T_R_h"), 1),
            _fmt(row.get("T_cost_h"), 1),
            _fmt(row.get("R(T_cost)"), 3),
            _fmt(row.get("T_recommended_h"), 1),
            _fmt(row.get("days_recommended"), 1),
            _san(row.get("recommended_source", "—")),
            _maintenance_type(_safe_float(row.get("beta")), row.get("maintenance_type")),
        ])
    return data


def _indicator_table(row: Dict[str, Any]) -> List[List[Any]]:
    return [
        ["Indicateur", "Valeur", "Lecture"],
        ["Processus retenu", _san(row.get("model", "—")), "Processus issu de l'analyse fiabiliste."],
        ["Variant du processus", _san(row.get("process_variant", "—")), "Précision sur le comportement retenu."],
        ["Loi choisie", _san(row.get("distribution", "—")), "Loi utilisée pour l'analyse ou la lecture principale."],
        ["MTTF (h)", _fmt(row.get("MTTF_h"), 1), "Temps moyen avant défaillance."],
        ["MTBF (h)", _fmt(row.get("MTBF_h"), 1), "Temps moyen entre défaillances."],
        ["MTTR (h)", _fmt(row.get("MTTR_h"), 1), "Temps moyen de réparation."],
        ["Disponibilité (%)", _fmt(row.get("availability_pct"), 2), "Part du temps disponible."],
        ["beta", _fmt(row.get("beta"), 3), "Indique jeunesse, aléatoire ou usure."],
        ["eta (h)", _fmt(row.get("eta_h"), 1), "Durée caractéristique."],
        ["gamma (h)", _fmt(row.get("gamma_h"), 1), "Décalage éventuel du modèle."],
    ]


def _optimization_table(row: Dict[str, Any], reliability_floor: Optional[float]) -> List[List[Any]]:
    floor_txt = f"{reliability_floor:.2f}" if reliability_floor is not None else "0.70"
    return [
        ["Étape / Formule", "Valeur", "Interprétation"],
        ["T_R", _fmt(row.get("T_R_h"), 1), "Intervalle issu du critère de fiabilité."],
        ["T_cost", _fmt(row.get("T_cost_h"), 1), "Intervalle issu du critère économique."],
        ["R(T_cost)", _fmt(row.get("R(T_cost)"), 3), f"Doit rester >= {floor_txt} pour accepter T_cost."],
        ["T_recommandé = max(T_cost, T_R)", _fmt(row.get("T_recommended_h"), 1), "Règle de décision appliquée après contrôle du seuil de fiabilité."],
        ["Jours avant maintenance retenus", _fmt(row.get("days_recommended"), 1), "Durée lisible par l'exploitant."],
        ["Source retenue", _san(row.get("recommended_source", "—")), "Indique si la décision finale vient de T_R ou T_cost."],
        ["C_min / h", _fmt(row.get("C_min_per_h"), 4), "Coût minimal moyen par heure."],
        ["Type de maintenance retenu", _maintenance_type(_safe_float(row.get("beta")), row.get("maintenance_type")), _maintenance_explanation(_maintenance_type(_safe_float(row.get("beta")), row.get("maintenance_type")))],
    ]


def _build_formula_text(reliability_floor: Optional[float]) -> str:
    floor_txt = f"{reliability_floor:.2f}" if reliability_floor is not None else "0.70"
    return (
        "La phase d'optimisation reprend d'abord les indicateurs calculés précédemment, puis calcule les deux "
        "intervalles candidats T_R et T_cost. La règle finale appliquée est la suivante : on choisit l'intervalle "
        "qui maximise les jours avant maintenance tout en respectant le seuil minimal de fiabilité. En particulier, "
        f"si R(T_cost) < {floor_txt}, alors T_R est retenu automatiquement. Sinon, on retient max(T_cost, T_R)."
    )


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
    df_out = df_out if isinstance(df_out, pd.DataFrame) else pd.DataFrame()

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
    nb_eq = int(getattr(df_out, "shape", [0])[0]) if df_out is not None and not df_out.empty else len(fits)

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

    story.append(Paragraph("Logique d’optimisation", styles["Heading2"]))
    story.append(Paragraph(_build_formula_text(_safe_float(meta.get("R_min_cost")) if meta else None), styles["Justify"]))
    story.append(Spacer(1, 8))

    if not df_out.empty:
        story.append(Paragraph("Tableau 1. Synthèse globale de l’optimisation", styles["Caption"]))
        story.append(_mk_table(_summary_table_data(df_out), total_width=usable_width, font_size=7))
        story.append(Spacer(1, 10))

    eqs = df_out["equipment_code"].astype(str).tolist() if not df_out.empty else sorted(set(list(fits.keys()) + list(organigram_by_eq.keys())))
    for idx, eq in enumerate(eqs):
        row_dict = {}
        if not df_out.empty:
            matched = df_out[df_out["equipment_code"].astype(str) == str(eq)]
            if not matched.empty:
                row_dict = matched.iloc[0].to_dict()

        pipe = organigram_by_eq.get(eq, {}) or {}
        rel = pipe.get("reliability", pipe)

        ttf_series: List[float] = []
        if isinstance(df, pd.DataFrame) and not df.empty and "equipment_code" in df.columns and "ttf_h" in df.columns:
            eq_df = df[df["equipment_code"].astype(str) == str(eq)].copy()
            values = pd.to_numeric(eq_df["ttf_h"], errors="coerce").dropna()
            values = values[values > 0]
            ttf_series = values.astype(float).tolist()

        story.append(Paragraph(_san(f"Équipement {eq}"), styles["Heading2"]))
        story.append(Paragraph(_compact(_pipe_line(pipe), 260), styles["Justify"]))
        story.append(Spacer(1, 4))

        story.append(Paragraph("Tableau 2. Indicateurs repris", styles["Caption"]))
        story.append(_mk_table(_indicator_table(row_dict), total_width=usable_width, font_size=8))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Tableau 3. Phase d’optimisation", styles["Caption"]))
        story.append(_mk_table(
            _optimization_table(row_dict, _safe_float(meta.get("R_min_cost")) if meta else None),
            total_width=usable_width,
            font_size=8,
        ))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Décision finale", styles["Heading3"]))
        story.append(Paragraph(_san(row_dict.get("optimization_note", "—")), styles["Justify"]))
        story.append(Spacer(1, 6))

        if ttf_series:
            story.append(Paragraph("Figure. Courbes fiabilistes avec intervalles retenus", styles["Caption"]))
            story.append(_plot_optimization_curves(eq, rel, ttf_series, row_dict))
            story.append(Spacer(1, 8))

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
