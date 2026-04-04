
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
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
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


def SAN(value: Any) -> str:
    text = "" if value is None else str(value)
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("•", "-").replace("…", "...")
        .replace("≤", "<=").replace("≥", ">=")
        .replace("β", "beta").replace("η", "eta").replace("γ", "gamma")
        .replace("θ", "theta").replace("\u00A0", " ")
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


def _pick_first(row: Dict[str, Any], *keys: str):
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _metric_mtbf(row: Dict[str, Any]):
    return _pick_first(row, "MTBF", "MTBF_h", "mtbf_h", "mtbf")


def _metric_mttr(row: Dict[str, Any]):
    return _pick_first(row, "MTTR", "MTTR_h", "mttr_h", "mttr")


def _metric_availability_pct(row: Dict[str, Any]):
    direct = _pick_first(row, "Disponibilite_pct", "Disponibilité_pct", "availability_pct", "availability")
    if direct not in (None, ""):
        return direct
    intrinsic = _pick_first(row, "availability_intrinsic", "Disponibilite_intrinseque", "Disponibilité_intrinsèque")
    try:
        if intrinsic is not None:
            val = float(intrinsic)
            return 100.0 * val if val <= 1.0 else val
    except Exception:
        pass
    mtbf = safe_float(_metric_mtbf(row))
    mttr = safe_float(_metric_mttr(row))
    if mtbf is not None and mttr is not None and (mtbf + mttr) > 0:
        return 100.0 * mtbf / (mtbf + mttr)
    return None


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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
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


SPARE_PARTS: List[Dict[str, Any]] = [
    {"categorie": "Étanchéité & cuve", "piece": "Joints plats de cuve", "qte_reco": "1 jeu", "criticite": "Élevée", "remarques": "Pour interventions lourdes"},
    {"categorie": "Étanchéité & cuve", "piece": "Joints de brides", "qte_reco": "Assortiment", "criticite": "Élevée", "remarques": "Prévenir les fuites"},
    {"categorie": "Isolation & traversées", "piece": "Traversée HT", "qte_reco": "1 unité", "criticite": "Élevée", "remarques": "Pièce critique"},
    {"categorie": "Isolation & traversées", "piece": "Traversée MT", "qte_reco": "1 unité", "criticite": "Élevée", "remarques": "À adapter au poste"},
    {"categorie": "OLTC", "piece": "Contacts mobiles OLTC", "qte_reco": "1 kit", "criticite": "Très élevée", "remarques": "Révision lourde"},
    {"categorie": "OLTC", "piece": "Contacts fixes OLTC", "qte_reco": "1 kit", "criticite": "Très élevée", "remarques": "Souvent remplacés ensemble"},
    {"categorie": "OLTC", "piece": "Ressorts de contacts", "qte_reco": "1 kit", "criticite": "Élevée", "remarques": "Assurer la pression de contact"},
    {"categorie": "Protection & mesure", "piece": "Relais Buchholz", "qte_reco": "1 unité", "criticite": "Élevée", "remarques": "Pièce de sécurité"},
    {"categorie": "Protection & mesure", "piece": "Indicateur de niveau d’huile", "qte_reco": "1 unité", "criticite": "Moyenne", "remarques": "Contrôle du niveau"},
    {"categorie": "Protection & mesure", "piece": "Fusibles auxiliaires", "qte_reco": "Lot complet", "criticite": "Élevée", "remarques": "Toutes intensités utilisées"},
    {"categorie": "Instrumentation & contrôle", "piece": "Relais auxiliaires", "qte_reco": "5 à 10", "criticite": "Moyenne", "remarques": "Circuits de commande"},
    {"categorie": "Instrumentation & contrôle", "piece": "Borniers de raccordement", "qte_reco": "Assortiment", "criticite": "Faible", "remarques": "Raccordements auxiliaires"},
    {"categorie": "Consommables", "piece": "Silice dessiccante", "qte_reco": "Plusieurs recharges", "criticite": "Élevée", "remarques": "Maintenir l’huile au sec"},
    {"categorie": "Consommables", "piece": "Huile isolante neuve", "qte_reco": "1 fût / IBC", "criticite": "Très élevée", "remarques": "Appoint ou traitement"},
    {"categorie": "Consommables", "piece": "Boulonnerie inox/galva", "qte_reco": "Assortiment", "criticite": "Moyenne", "remarques": "Remplacements divers"},
]

DEFAULT_TOOLS = [
    {"categorie": "Mesure & essais", "outil": "Micro-ohmmètre", "description": "Résistance d’enroulements", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & essais", "outil": "Mégohmmètre 5 kV", "description": "Résistance d’isolement", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & essais", "outil": "Multimètre", "description": "Mesures électriques générales", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & essais", "outil": "Pince ampèremétrique", "description": "Mesures de courant", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Échantillonnage", "outil": "Kit de prélèvement d’huile", "description": "Prélèvements pour analyse labo", "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
    {"categorie": "Outils", "outil": "Clés dynamométriques", "description": "Serrage au couple", "qte": 1, "unite": "set", "calibrage": "OK", "remarques": ""},
    {"categorie": "Outils", "outil": "Pompe à huile", "description": "Transfert / vidange", "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Sécurité", "outil": "EPI complets", "description": "Gants, casque, visière, tenue", "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
]


def _title(story, styles, text: str, level: int = 2, space_after_pt: int = 6):
    style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
    story.append(Paragraph(SAN(text), styles[style]))
    story.append(Spacer(1, space_after_pt))


def _add_cahier_maintenance(story, styles):
    _title(story, styles, "Cahier de maintenance", level=1, space_after_pt=4)
    story.append(Paragraph(
        SAN(
            "Ce cahier sert de document de référence général pour le suivi de la maintenance d’un transformateur "
            "de puissance. Il regroupe des rappels sur les types de maintenance, les contrôles électriques, "
            "les inspections visuelles, les fréquences recommandées, les tableaux de suivi et les matériels à prévoir."
        ),
        styles["Justify"],
    ))
    story.append(Spacer(1, 8))

    _title(story, styles, "1. Types de maintenance", level=3, space_after_pt=3)
    story.append(Paragraph(
        SAN(
            "La maintenance préventive regroupe les actions planifiées destinées à limiter la probabilité de panne. "
            "La maintenance conditionnelle repose sur l’observation de l’état réel de l’équipement. La maintenance "
            "corrective intervient après apparition d’un défaut ou lorsqu’un écart significatif est observé. "
            "La maintenance prédictive vise à anticiper le bon moment d’intervention à partir des données disponibles."
        ),
        styles["Justify"],
    ))
    story.append(Spacer(1, 6))

    _title(story, styles, "2. Tableau – Essais électriques de référence", level=3, space_after_pt=4)
    tbl1 = [
        ["Test", "Méthode / appareil", "Valeur de référence", "Alerte", "Référence"],
        ["Résistance d’isolement", "Mégohmmètre 5 kV", "> 1000 MΩ", "< 600 MΩ", "IEC 60076-3"],
        ["Rapport de transformation", "TTR", "± 0,5 %", "> 0,5 %", "IEC 60076-1"],
        ["Résistance d’enroulements", "Micro-ohmmètre", "Écart < 2 %", "> 2 %", "IEC 60076-1"],
        ["Impédance de court-circuit", "Essai dédié", "10 à 12 %", "Δ > 3 %", "IEC 60076-5"],
        ["Courant d’excitation", "Essai à vide", "≤ 0,5 % In", "> 1 % In", "Constructeur"],
        ["Pertes à vide", "Essai", "Selon plaque", "Écart significatif", "Constructeur"],
        ["Pertes en charge", "Essai", "Selon plaque", "Écart significatif", "Constructeur"],
    ]
    story.append(_mk_table(tbl1, widths=[4.5*cm, 4.4*cm, 3.0*cm, 3.0*cm, 3.2*cm], font_size=7))
    story.append(Spacer(1, 6))

    _title(story, styles, "3. Tableau – Inspection mécanique et visuelle", level=3, space_after_pt=4)
    tbl2 = [
        ["Contrôle", "Critère normal", "Alerte", "Critique"],
        ["Niveau d’huile", "Niveau nominal", "Baisse lente", "Baisse rapide / fuite"],
        ["État des isolateurs", "Propres et intacts", "Dépôts / traces", "Fissure / casse"],
        ["Joints et brides", "Étanches", "Suintement", "Fuite active"],
        ["Relais Buchholz", "RAS", "Présence de gaz", "Déclenchement / défaut"],
        ["Bruit", "Stable", "Augmentation notable", "Claquements anormaux"],
    ]
    story.append(_mk_table(tbl2, widths=[4.6*cm, 4.5*cm, 4.0*cm, 4.0*cm], font_size=7))
    story.append(Spacer(1, 6))

    _title(story, styles, "4. Tableau – Fréquences recommandées des contrôles", level=3, space_after_pt=4)
    tbl3 = [
        ["Type de contrôle", "Fréquence recommandée"],
        ["Inspection visuelle complète", "Mensuelle"],
        ["Analyse d’huile", "Semestrielle"],
        ["Essais électriques complets", "Annuelle"],
        ["Contrôle OLTC", "Annuelle"],
        ["Filtration / traitement d’huile si besoin", "Selon constat"],
        ["Révision majeure", "Selon historique et criticité"],
    ]
    story.append(_mk_table(tbl3, widths=[8.4*cm, 8.4*cm], font_size=7))
    story.append(Spacer(1, 6))

    _title(story, styles, "5. Tableau de suivi de maintenance", level=3, space_after_pt=4)
    header = ["Date", "Agent", "Paramètre contrôlé", "Valeur de référence", "Résultat", "État", "Observations"]
    rows = [
        ["", "", "Résistance d’isolement", "> 1000 MΩ", "", "", ""],
        ["", "", "Rapport de transformation", "± 0,5 %", "", "", ""],
        ["", "", "Résistance d’enroulements", "< 2 % diff. phases", "", "", ""],
        ["", "", "Impédance de court-circuit", "± 2 % nominal", "", "", ""],
        ["", "", "Courant d’excitation", "≤ 0,5 % In", "", "", ""],
        ["", "", "Pertes à vide", "Selon plaque", "", "", ""],
        ["", "", "Pertes en charge", "Selon plaque", "", "", ""],
        ["", "", "Niveau d’huile", "Niveau nominal", "", "", ""],
        ["", "", "État des isolateurs", "Propres / intacts", "", "", ""],
        ["", "", "Relais Buchholz", "RAS", "", "", ""],
    ]
    story.append(_mk_table([header] + rows, widths=[1.5*cm, 2.2*cm, 4.4*cm, 4.3*cm, 2.1*cm, 1.6*cm, 2.7*cm], font_size=6.5))
    story.append(Spacer(1, 8))

    _title(story, styles, "6. Tableau – Pièces de rechange", level=3, space_after_pt=4)
    data_sp = [["Catégorie", "Pièce de rechange", "Quantité recommandée", "Criticité", "Remarques"]]
    for sp in SPARE_PARTS:
        data_sp.append([SAN(sp["categorie"]), SAN(sp["piece"]), SAN(sp["qte_reco"]), SAN(sp["criticite"]), SAN(sp["remarques"])])
    story.append(_mk_table(data_sp, widths=[3.2*cm, 6.0*cm, 2.8*cm, 2.3*cm, 3.5*cm], font_size=7))
    story.append(Spacer(1, 10))


def _policy_from_beta(beta: Any) -> str:
    b = safe_float(beta)
    if b is None:
        return "Politique non déterminée."
    if b > 1.0:
        return "Usure dominante : maintenance préventive ou conditionnelle renforcée."
    if b < 1.0:
        return "Défauts précoces : contrôles rapprochés, fiabilisation et inspection ciblée."
    return "Comportement proche du constant : surveillance standard et planification régulière."


def _maintenance_type_explanation(mtype: str, beta: Any) -> str:
    label = SAN(mtype).strip().lower()
    if "correct" in label:
        return "Ce type a été retenu car une correction rapide est nécessaire."
    if "condition" in label:
        return "Ce type a été retenu car l’état réel doit guider le moment d’intervention."
    if "prévent" in label or "prevent" in label:
        return "Ce type a été retenu car l’intervention doit être planifiée avant la panne."
    if "predict" in label or "prédict" in label:
        return "Ce type a été retenu car l’évolution des données permet d’anticiper l’intervention."
    return _policy_from_beta(beta)


def _build_choice_explanation(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("maintenance_type"):
        parts.append(f"Type retenu : {SAN(row.get('maintenance_type'))}.")
    if row.get("model") or row.get("distribution"):
        parts.append(f"Le modèle retenu est {SAN(row.get('model'))} avec la loi {SAN(row.get('distribution'))}.")
    if row.get("beta") is not None:
        parts.append(f"Le paramètre beta = {fnum(row.get('beta'), 2)} a orienté la stratégie de maintenance.")
    if row.get("eta_h", row.get("eta")) is not None:
        parts.append(f"La durée caractéristique eta vaut {fnum(row.get('eta_h', row.get('eta')), 1)} h.")
    if row.get("gamma_h", row.get("gamma")) is not None:
        parts.append(f"Le paramètre gamma vaut {fnum(row.get('gamma_h', row.get('gamma')), 1)} h.")
    if row.get("T_recommended_h") is not None:
        parts.append(f"L’intervalle recommandé est {fnum(row.get('T_recommended_h'), 1)} h.")
    if row.get("T_R_h") is not None:
        parts.append(f"L’intervalle fiabiliste vaut {fnum(row.get('T_R_h'), 1)} h.")
    if row.get("T_cost_h") is not None:
        parts.append(f"L’intervalle économique vaut {fnum(row.get('T_cost_h'), 1)} h.")
    if row.get("MTBF") is not None:
        parts.append(f"Le MTBF est de {fnum(row.get('MTBF'), 1)} h.")
    if row.get("MTTR") is not None:
        parts.append(f"Le MTTR est de {fnum(row.get('MTTR'), 1)} h.")
    if row.get("decision_finale"):
        parts.append(f"Décision finale : {SAN(row.get('decision_finale'))}.")
    if row.get("motif_decision"):
        parts.append(f"Motif : {SAN(row.get('motif_decision'))}")
    return " ".join(parts) if parts else "Aucune explication détaillée disponible."


def _build_influence_table(row: Dict[str, Any]):
    return [
        ["Paramètre", "Valeur", "Impact sur le choix"],
        ["Type de maintenance retenu", SAN(row.get("maintenance_type", "")), _maintenance_type_explanation(SAN(row.get("maintenance_type", "")), row.get("beta"))],
        ["Modèle", SAN(row.get("model", "")), "Le comportement global des défaillances influence la stratégie retenue."],
        ["Loi", SAN(row.get("distribution", "")), "La loi retenue structure l’estimation des durées et du risque."],
        ["beta", fnum(row.get("beta"), 2), "Indique défauts précoces, comportement aléatoire ou usure."],
        ["eta (h)", fnum(row.get("eta_h", row.get("eta")), 1), "Référence de durée de vie caractéristique."],
        ["gamma (h)", fnum(row.get("gamma_h", row.get("gamma")), 1), "Décalage éventuel du modèle."],
        ["T_recommended (h)", fnum(row.get("T_recommended_h"), 1), "Intervalle principal proposé."],
        ["T_R (h)", fnum(row.get("T_R_h"), 1), "Intervalle issu du critère de fiabilité."],
        ["T_cost (h)", fnum(row.get("T_cost_h"), 1), "Intervalle issu du critère économique."],
        ["MTBF (h)", fnum(row.get("MTBF"), 1), "Renseigne l’espacement moyen des pannes."],
        ["MTTR (h)", fnum(row.get("MTTR"), 1), "Renseigne le temps moyen de remise en état."],
        ["Jours restants", SAN(row.get("days_left", "")), "Plus l’échéance est proche, plus la priorité augmente."],
        ["Décision finale", SAN(row.get("decision_finale", "")), "Conclusion synthétique issue des paramètres disponibles."],
    ]


def _add_per_equipment_summaries(story, styles, metrics_table: List[Dict[str, Any]], kits_by_eq: Dict[str, List[Dict[str, Any]]] | None):
    if not metrics_table:
        return
    _title(story, styles, "Résultats d’analyse par équipement", level=2, space_after_pt=4)

    for row in metrics_table:
        row = row if isinstance(row, dict) else {}
        original_eq = row.get("equipment_code", "")
        eq_disp = SAN(original_eq)

        beta = row.get("beta")
        eta = row.get("eta_h", row.get("eta"))
        gamma = row.get("gamma_h", row.get("gamma"))
        T_rec = row.get("T_recommended_h")
        T_R = row.get("T_R_h")
        T_cost = row.get("T_cost_h")
        interval_opt = row.get("interval_opt_h", row.get("interval_h"))
        model = row.get("model")
        distribution = row.get("distribution")
        maintenance_type = row.get("maintenance_type")
        mtbf = row.get("MTBF")
        mttr = row.get("MTTR")
        decision = row.get("decision_finale")
        motif = row.get("motif_decision")

        availability = _metric_availability_pct(row)

        _title(story, styles, f"Fiche intervention – {eq_disp}", level=3, space_after_pt=2)

        data_param = [
            ["Élément", "Valeur"],
            ["Type de maintenance", SAN(maintenance_type)],
            ["Modèle / Loi", f"{SAN(model)} / {SAN(distribution)}"],
            ["beta (forme)", fnum(beta, 2)],
            ["eta (échelle, h)", fnum(eta, 1)],
            ["gamma (décalage, h)", fnum(gamma, 1)],
            ["T_recommended (h)", fnum(T_rec, 1)],
            ["T_R (h)", fnum(T_R, 1)],
            ["T_cost (h)", fnum(T_cost, 1)],
            ["Intervalle optimisé (fallback)", fnum(interval_opt, 1)],
            ["MTBF (h)", fnum(mtbf, 1)],
            ["MTTR (h)", fnum(mttr, 1)],
            ["Disponibilité (%)", fnum(availability, 1)],
            ["Décision finale", SAN(decision)],
        ]
        story.append(_mk_table(data_param, font_size=8))
        story.append(Spacer(1, 4))

        _title(story, styles, "Choix retenu pour le cas d’étude", level=3, space_after_pt=2)
        story.append(Paragraph(SAN(_build_choice_explanation(row)), styles["Justify"]))
        story.append(Spacer(1, 5))

        if motif:
            story.append(Paragraph(SAN(f"Motif : {motif}"), styles["Justify"]))
            story.append(Spacer(1, 5))

        _title(story, styles, "Paramètres ayant influencé le choix", level=3, space_after_pt=2)
        story.append(_mk_table(_build_influence_table(row), font_size=7.3))
        story.append(Spacer(1, 6))

        _title(story, styles, "Politique recommandée", level=3, space_after_pt=2)
        policy = ""
        if maintenance_type:
            policy += f"Type recommandé : {SAN(maintenance_type)}. "
        policy += _policy_from_beta(beta)
        story.append(Paragraph(SAN(policy), styles["Justify"]))
        story.append(Spacer(1, 6))

        _title(story, styles, "Kit de pièces recommandé", level=3, space_after_pt=2)
        kit_list = (kits_by_eq or {}).get(str(original_eq), [])
        if kit_list:
            kit_data = [["Pièce", "Quantité", "Criticité", "Remarques"]]
            for item in kit_list[:12]:
                item = item if isinstance(item, dict) else {}
                kit_data.append([
                    SAN(item.get("piece", "")),
                    SAN(item.get("qte_reco", item.get("qte", ""))),
                    SAN(item.get("criticite", "")),
                    _compact(item.get("remarques", ""), 70),
                ])
            story.append(_mk_table(kit_data, font_size=7))
        else:
            story.append(Paragraph(SAN("Aucune pièce recommandée détectée (ou stock non activé)."), styles["Justify"]))
        story.append(Spacer(1, 8))


def _table_from_tasks_due(tasks_due: List[Dict[str, Any]]):
    if not tasks_due:
        return None
    data = [[
        "Équipement", "Type maint.", "Intervalle (h)", "Source", "Échéance", "Jours restants",
        "T_rec (h)", "T_R (h)", "T_cost (h)", "Statut"
    ]]
    for task in tasks_due:
        task = task if isinstance(task, dict) else {}
        data.append([
            SAN(task.get("equipment_code", "")),
            SAN(task.get("maintenance_type", "")),
            fnum(task.get("interval_h"), 1),
            SAN(task.get("interval_source", "")),
            SAN(task.get("next_due_date", "")),
            SAN(task.get("days_left", "")),
            fnum(task.get("T_recommended_h"), 1),
            fnum(task.get("T_R_h"), 1),
            fnum(task.get("T_cost_h"), 1),
            SAN(task.get("status", "")),
        ])
    return _mk_table(data, widths=[2.2*cm, 2.8*cm, 2.0*cm, 1.9*cm, 2.4*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.9*cm], font_size=7.2)


def _add_tools_section(story, styles, tools_checklist: List[Dict[str, Any]] | None):
    _title(story, styles, "Matériels à prévoir pour l’entretien", level=2, space_after_pt=2)
    tools = tools_checklist if (isinstance(tools_checklist, list) and tools_checklist) else DEFAULT_TOOLS
    data = [["Catégorie", "Outil / instrument", "Description", "Qté", "Unité", "Calibrage / état", "Remarques"]]
    for item in tools:
        item = item if isinstance(item, dict) else {}
        data.append([
            SAN(item.get("categorie", "")),
            SAN(item.get("outil", "")),
            SAN(item.get("description", "")),
            SAN(item.get("qte", "")),
            SAN(item.get("unite", "")),
            SAN(item.get("calibrage", "")),
            SAN(item.get("remarques", "")),
        ])
    story.append(_mk_table(data, widths=[2.8*cm, 3.0*cm, 4.1*cm, 1.1*cm, 1.2*cm, 2.0*cm, 2.2*cm], font_size=7.5))
    story.append(Spacer(1, 6))


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
    pdf.cell(0, 6, SAN("Résumé (mode simplifié)"), ln=1)
    pdf.set_font("Arial", "", 9)
    for row in metrics_table[:20]:
        pdf.multi_cell(
            0, 5,
            SAN(
                f"- {row.get('equipment_code', '')} | type={row.get('maintenance_type', '')} | "
                f"beta={fnum(row.get('beta'), 2)} | intervalle recommandé={fnum(row.get('T_recommended_h'), 1)} h"
            ),
        )
    pdf.ln(2)

    if tasks_due:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, SAN("Tâches dues"), ln=1)
        pdf.set_font("Arial", "", 9)
        for task in tasks_due[:40]:
            pdf.multi_cell(0, 5, SAN(f"- [{task.get('equipment_code', '')}] {task.get('maintenance_type', '')} | échéance : {task.get('next_due_date', '')}"))

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

    story: List[Any] = []
    story.append(Paragraph(SAN(title), styles["Title"]))
    story.append(Paragraph(SAN(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")), styles["BodyText"]))
    story.append(Spacer(1, 10))

    _add_cahier_maintenance(story, styles)
    story.append(PageBreak())

    _add_per_equipment_summaries(story, styles, metrics_table, kits_by_eq)

    _title(story, styles, "Tâches de maintenance dues (tableau global)", level=2, space_after_pt=3)
    due_table = _table_from_tasks_due(tasks_due)
    if due_table is None:
        story.append(Paragraph(SAN("Aucune tâche due actuellement."), styles["Justify"]))
    else:
        story.append(due_table)
    story.append(Spacer(1, 8))

    _add_tools_section(story, styles, tools_checklist)

    story.append(Spacer(1, 8))
    story.append(Paragraph(SAN(f"Rapport généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["BodyText"]))

    doc.build(story)
    return str(out_path)
