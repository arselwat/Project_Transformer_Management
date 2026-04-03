from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import math
import unicodedata

# ============================================================
# Imports PDF : ReportLab (préféré) + fallback FPDF
# ============================================================
try:
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import getSampleStyleSheet
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
    A4 = (595.27, 841.89)
    colors = None
    mm = 1.0
    cm = 10.0

try:
    from fpdf import FPDF
    HAVE_FPDF = True
except Exception:
    HAVE_FPDF = False


# =========================
# Helpers
# =========================

def SAN(s: Any) -> str:
    """Normalise le texte pour PDF (Latin-1 safe)."""
    s = "" if s is None else str(s)
    s = (
        s.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
         .replace("•", "-").replace("…", "...")
         .replace("\u00A0", " ")
    )
    s = (
        s.replace("≤", "<=").replace("≥", ">=").replace("±", "+/-")
         .replace("Ω", "Ohm").replace("δ", "delta")
         .replace("β", "beta").replace("η", "eta").replace("γ", "gamma")
         .replace("θ", "theta")
         .replace("°C", " degC")
    )
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "ignore").decode("latin-1", "ignore")


def fnum(v: Any, nd: int = 1, default: str = "-") -> str:
    """Format numérique tolérant (None/NaN/inf)."""
    try:
        if v is None:
            return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return f"{float(v):.{nd}f}"
    except Exception:
        return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _compact(text: Any, max_len: int = 140) -> str:
    s = SAN(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _title(story, styles, text: str, level: int = 3, space_after_pt: int = 6):
    h = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
    story.append(Paragraph(SAN(text), styles[h]))
    story.append(Spacer(1, space_after_pt))


def _page_available_width(left_margin_mm: float = 16, right_margin_mm: float = 16) -> float:
    return A4[0] - (left_margin_mm * mm + right_margin_mm * mm)


def _scale_widths(widths, available_width):
    if not widths:
        return widths
    total = sum(widths)
    if total <= 0:
        return widths
    if total <= available_width:
        return widths
    factor = available_width / total
    return [w * factor for w in widths]


def _auto_widths_from_data(data, available_width, min_col_mm=16, max_col_mm=65):
    if not data:
        return None
    ncols = max(len(r) for r in data)
    lengths = [8] * ncols

    for row in data[:40]:
        for i in range(ncols):
            cell = row[i] if i < len(row) else ""
            lengths[i] = max(lengths[i], min(len(str(cell)), 60))

    total_len = sum(lengths) if sum(lengths) > 0 else ncols
    widths = [(length / total_len) * available_width for length in lengths]
    min_w = min_col_mm * mm
    max_w = max_col_mm * mm
    widths = [min(max(w, min_w), max_w) for w in widths]
    return _scale_widths(widths, available_width)


def _mk_table(
    data,
    widths=None,
    font_size=8,
    header_font_size=None,
    header_center=True,
    available_width=None,
):
    """
    Améliore la lisibilité des tableaux :
      - wrap automatique dans chaque cellule
      - padding + leading adaptés
      - header stylé
      - réduction auto si la largeur dépasse la page
    IMPORTANT: ne change PAS le contenu, seulement le rendu.
    """
    if available_width is None:
        available_width = _page_available_width()

    header_font_size = header_font_size or max(font_size, 8)

    body_style = ParagraphStyle(
        name=f"tbl_body_{font_size}_{len(data)}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=max(10, int(font_size * 1.35)),
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_LEFT,
        wordWrap="CJK",
        splitLongWords=True,
    )

    head_style = ParagraphStyle(
        name=f"tbl_head_{font_size}_{len(data)}",
        fontName="Helvetica-Bold",
        fontSize=header_font_size,
        leading=max(10, int(header_font_size * 1.2)),
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_CENTER if header_center else TA_LEFT,
        wordWrap="CJK",
        splitLongWords=True,
    )

    def to_para(x, style):
        s = SAN(x).replace("\n", "<br/>")
        return Paragraph(s, style)

    wrapped = []
    for i, row in enumerate(data):
        style = head_style if i == 0 else body_style
        wrapped.append([to_para(cell, style) for cell in row])

    if widths is not None:
        widths = _scale_widths(widths, available_width)
    else:
        widths = _auto_widths_from_data(data, available_width)

    t = Table(
        wrapped,
        repeatRows=1,
        colWidths=widths,
        splitByRow=1,
    )

    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),

        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),

        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.grey),
    ]))

    return t


# ============================================================
# Liste de pièces de rechange (pour le grand tableau matériel)
# ============================================================

SPARE_PARTS: List[Dict[str, Any]] = [
    # Étanchéité & cuve
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Joints plats de cuve (couvercle)",
        "qte_reco": "1 jeu complet",
        "criticite": "Élevée",
        "remarques": "Pour interventions lourdes (ouverture de cuve)",
    },
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Joints de brides de radiateurs",
        "qte_reco": "Assortiment (selon nb de brides)",
        "criticite": "Élevée",
        "remarques": "Très utile en cas de fuite d’huile sur radiateurs",
    },
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Joints du conservateur d’huile",
        "qte_reco": "1 jeu",
        "criticite": "Moyenne",
        "remarques": "Prévoir modèle conforme constructeur",
    },
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Joints toriques de vannes et bouchons de purge",
        "qte_reco": "Assortiment",
        "criticite": "Moyenne",
        "remarques": "Souvent source de petites fuites",
    },
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Presse-étoupes et passe-cloisons",
        "qte_reco": "Quelques unités de chaque type",
        "criticite": "Moyenne",
        "remarques": "Pour câbles d’instrumentation et auxiliaires",
    },
    {
        "categorie": "Étanchéité & cuve",
        "piece": "Boulonnerie HT/MT (inox / galvanisé HR)",
        "qte_reco": "Assortiment (M8, M10, M12, etc.)",
        "criticite": "Moyenne",
        "remarques": "Pour remplacement de vis corrodées / cassées",
    },

    # Isolation & traversées
    {
        "categorie": "Isolation & traversées",
        "piece": "Isolateurs de traversée HT",
        "qte_reco": "1 unité par type critique",
        "criticite": "Élevée",
        "remarques": "Si un isolateur casse, arrêt de longue durée sans spare",
    },
    {
        "categorie": "Isolation & traversées",
        "piece": "Isolateurs de traversée MT",
        "qte_reco": "1–2 unités",
        "criticite": "Élevée",
        "remarques": "À adapter au schéma de raccordement",
    },
    {
        "categorie": "Isolation & traversées",
        "piece": "Bagues d’isolement / bagues en bakélite ou composite",
        "qte_reco": "1 kit",
        "criticite": "Moyenne",
        "remarques": "Pour remplacements locaux sur raccordements",
    },

    # OLTC
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Jeu de contacts mobiles OLTC",
        "qte_reco": "1 kit complet",
        "criticite": "Très élevée",
        "remarques": "Pièce critique, à remplacer en révision lourde",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Jeu de contacts fixes OLTC",
        "qte_reco": "1 kit complet",
        "criticite": "Très élevée",
        "remarques": "Souvent remplacé avec les contacts mobiles",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Ressorts de contacts OLTC",
        "qte_reco": "1 kit",
        "criticite": "Élevée",
        "remarques": "Perte de pression = mauvais contact = échauffement",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Joints spécifiques OLTC",
        "qte_reco": "1 jeu complet",
        "criticite": "Élevée",
        "remarques": "Pour interventions sur cuve OLTC",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Cartouche(s) filtre huile OLTC",
        "qte_reco": "1–2 pcs",
        "criticite": "Moyenne",
        "remarques": "Si OLTC dispose d’un circuit d’huile dédié",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Silice dessiccateur OLTC",
        "qte_reco": "1–2 recharges",
        "criticite": "Moyenne",
        "remarques": "Pour garder l’OLTC au sec",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Moteur d’entraînement OLTC",
        "qte_reco": "1 moteur complet (si budget)",
        "criticite": "Élevée",
        "remarques": "Garanti une remise en service rapide en cas de panne",
    },
    {
        "categorie": "OLTC (Changeur de prises en charge)",
        "piece": "Fin de course / capteur de position OLTC",
        "qte_reco": "1 kit",
        "criticite": "Élevée",
        "remarques": "Indispensables pour la commande automatique",
    },

    # Refroidissement
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Ventilateurs complets",
        "qte_reco": "1–2 pcs",
        "criticite": "Élevée",
        "remarques": "En cas de panne, surtout en forte charge",
    },
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Moteurs de ventilateurs seuls",
        "qte_reco": "1–2 pcs",
        "criticite": "Moyenne",
        "remarques": "Si ventilateurs démontables",
    },
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Pompes à huile de circulation",
        "qte_reco": "1 pompe en spare",
        "criticite": "Très élevée",
        "remarques": "Critique pour ONAF en forte charge",
    },
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Joints de pompe (kit garnitures)",
        "qte_reco": "1 kit",
        "criticite": "Moyenne",
        "remarques": "Pour stopper fuites d’huile sur corps de pompe",
    },
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Thermostats / capteurs de température (huile / enroulements)",
        "qte_reco": "2–3 pcs",
        "criticite": "Élevée",
        "remarques": "Utilisés pour déclencher alarmes/ventilation",
    },
    {
        "categorie": "Refroidissement (ONAN / ONAF)",
        "piece": "Contacteurs / relais de puissance de commande ventilateurs",
        "qte_reco": "2–3 pcs",
        "criticite": "Moyenne",
        "remarques": "Pour circuits de commande auxiliaires",
    },

    # Protection & mesure
    {
        "categorie": "Protection & mesure",
        "piece": "Relais Buchholz (complet)",
        "qte_reco": "1 unité (optionnel mais conseillé)",
        "criticite": "Élevée",
        "remarques": "Si relais spécifique constructeur, utile en spare",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Kit flotteurs / contacts internes Buchholz",
        "qte_reco": "1 kit",
        "criticite": "Moyenne",
        "remarques": "Permet de réparer sans remplacer tout le relais",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Indicateurs de niveau d’huile (conservateur)",
        "qte_reco": "1 unité",
        "criticite": "Moyenne",
        "remarques": "En cas de casse ou blocage de flotteur",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Thermomètres d’huile (analogiques)",
        "qte_reco": "1–2 pcs",
        "criticite": "Moyenne",
        "remarques": "Pour affichage local T° huile",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Transformateurs de courant (TC) de protection",
        "qte_reco": "1 unité par calibre critique (optionnel)",
        "criticite": "Moyenne",
        "remarques": "Selon politique de stock et criticité du poste",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Transformateurs de tension (TT)",
        "qte_reco": "1–2 pcs (optionnel)",
        "criticite": "Moyenne",
        "remarques": "Utiles si délais d’approvisionnement longs",
    },
    {
        "categorie": "Protection & mesure",
        "piece": "Fusibles HT/MT (toutes intensités utilisées)",
        "qte_reco": "Lot complet",
        "criticite": "Élevée",
        "remarques": "Fusibles de protection auxiliaires et TT/TC",
    },

    # Instrumentation & contrôle
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Sondes PT100 / PT1000",
        "qte_reco": "3–5 pcs",
        "criticite": "Élevée",
        "remarques": "Pour mesure T° enroulements / huile",
    },
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Relais auxiliaires (24/48/110 V)",
        "qte_reco": "5–10 pcs",
        "criticite": "Moyenne",
        "remarques": "Très utilisés dans circuits de commande",
    },
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Voyants lumineux de signalisation",
        "qte_reco": "Assortiment couleurs (rouge/vert/orange)",
        "criticite": "Faible",
        "remarques": "Pour défaut, alarme, marche/arrêt",
    },
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Boutons poussoirs (marche/arrêt)",
        "qte_reco": "2–3 jeux",
        "criticite": "Faible / Moyenne",
        "remarques": "Usure mécanique fréquente",
    },
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Sélecteurs de commande locale / distante",
        "qte_reco": "1–2 pcs",
        "criticite": "Moyenne",
        "remarques": "Important pour manœuvres local/remote",
    },
    {
        "categorie": "Instrumentation & contrôle",
        "piece": "Borniers de raccordement",
        "qte_reco": "Assortiment",
        "criticite": "Faible",
        "remarques": "Pour remplacements sur circuits auxiliaires",
    },

    # Consommables & annexes
    {
        "categorie": "Consommables & annexes",
        "piece": "Silice déshydratante dessiccateur principal",
        "qte_reco": "Plusieurs recharges",
        "criticite": "Élevée",
        "remarques": "Indispensable pour limiter humidité dans l’huile",
    },
    {
        "categorie": "Consommables & annexes",
        "piece": "Huile isolante neuve",
        "qte_reco": "1 fût / IBC selon politique",
        "criticite": "Très élevée",
        "remarques": "Pour appoints après fuite ou traitement",
    },
    {
        "categorie": "Consommables & annexes",
        "piece": "Ruban d’étanchéité compatible huile",
        "qte_reco": "1–2 rouleaux",
        "criticite": "Moyenne",
        "remarques": "Pour petits raccords filetés",
    },
    {
        "categorie": "Consommables & annexes",
        "piece": "Joints standards (fibre, nitrile, viton)",
        "qte_reco": "Assortiment",
        "criticite": "Moyenne",
        "remarques": "Pour petites interventions de terrain",
    },
    {
        "categorie": "Consommables & annexes",
        "piece": "Peinture anticorrosion (cuve, radiateurs)",
        "qte_reco": "1 kit",
        "criticite": "Faible / Moyenne",
        "remarques": "Pour protection à long terme de la cuve",
    },

    # Éléments structurels
    {
        "categorie": "Éléments structurels (si stock avancé)",
        "piece": "Radiateur complet",
        "qte_reco": "1 unité (si budget & criticité réseau)",
        "criticite": "Moyenne",
        "remarques": "Permet un remplacement rapide en cas de fuite sévère",
    },
    {
        "categorie": "Éléments structurels (si stock avancé)",
        "piece": "Conservateur d’huile complet",
        "qte_reco": "1 unité (optionnel)",
        "criticite": "Moyenne",
        "remarques": "Utile si le modèle est spécifique et long à recevoir",
    },
    {
        "categorie": "Éléments structurels (si stock avancé)",
        "piece": "OLTC complet (bloc)",
        "qte_reco": "1 unité (grands postes très critiques)",
        "criticite": "Très élevée",
        "remarques": "Réservé aux sites stratégiques à forte disponibilité",
    },
    {
        "categorie": "Éléments structurels (si stock avancé)",
        "piece": "Transformateur de secours (plus petite puissance)",
        "qte_reco": "1 unité (si politique réseau)",
        "criticite": "Très élevée",
        "remarques": "Utilisé comme secours temporaire en cas de panne grave",
    },
]

# ============================================================
# Cahier de maintenance statique (texte + tableaux)
# ============================================================

def _add_cahier_maintenance(story, styles):
    _title(story, styles, "Cahier de Maintenance", level=1, space_after_pt=4)
    _title(story, styles, "Transformateur 220/20 kV – 100 MVA", level=2, space_after_pt=6)

    intro = (
        "Ce cahier regroupe les principaux tableaux de valeurs de référence et les consignes de suivi "
        "pour la maintenance d’un transformateur de puissance 220/20 kV – 100 MVA, conformément aux "
        "références IEC et CIGRE. La maintenance appliquée à un transformateur de puissance se divise "
        "en plusieurs catégories complémentaires, chacune ayant un rôle précis dans la fiabilité et la "
        "durée de vie de l’équipement."
    )
    story.append(Paragraph(SAN(intro), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "1. Maintenance préventive", level=3, space_after_pt=3)
    txt = (
        "La maintenance préventive regroupe toutes les actions planifiées visant à éviter l’apparition de pannes. "
        "Elle est réalisée à des intervalles réguliers (mensuel, trimestriel, annuel) et comprend les inspections "
        "visuelles, les mesures électriques, les analyses d’huile et le contrôle du système de refroidissement. "
        "Son objectif principal est de maintenir le transformateur dans un état optimal en détectant les défauts "
        "mineurs avant qu’ils ne deviennent critiques."
    )
    story.append(Paragraph(SAN(txt), styles["BodyText"]))
    story.append(Spacer(1, 6))

    _title(story, styles, "2. Maintenance conditionnelle (CBM – Condition Based Maintenance)", level=3, space_after_pt=3)
    txt = (
        "La maintenance conditionnelle repose sur l’analyse continue de l’état réel du transformateur. "
        "Elle utilise des mesures et des capteurs (température, vibrations, humidité, DGA, thermographie) "
        "permettant de suivre l’évolution des paramètres critiques. Cette méthode permet d’intervenir uniquement "
        "lorsque les indicateurs montrent un début de dérive, ce qui optimise les coûts et augmente la disponibilité "
        "de l’équipement."
    )
    story.append(Paragraph(SAN(txt), styles["BodyText"]))
    story.append(Spacer(1, 6))

    _title(story, styles, "3. Maintenance corrective", level=3, space_after_pt=3)
    txt = (
        "La maintenance corrective intervient après l’apparition d’un défaut ou lorsqu’une mesure dépasse les seuils "
        "critiques fixés par les normes IEC, IEEE ou CIGRE. Elle consiste à réparer ou remplacer les éléments défectueux "
        "(joints, ventilateurs, isolateurs, OLTC, huile, etc.). Dans le cas d’un transformateur, la maintenance corrective "
        "peut inclure la filtration ou le remplacement de l’huile, l’intervention sur les enroulements, ou le changement "
        "des composants mécaniques endommagés."
    )
    story.append(Paragraph(SAN(txt), styles["BodyText"]))
    story.append(Spacer(1, 6))

    _title(story, styles, "4. Maintenance prédictive", level=3, space_after_pt=3)
    txt = (
        "La maintenance prédictive utilise des méthodes avancées de diagnostic (analyse de tendances DGA, modélisation "
        "thermique, capteurs IoT, IA) pour prédire les pannes avant qu’elles ne se produisent. Basée sur l’analyse "
        "statistique des données historiques et en temps réel, elle permet d’anticiper précisément le moment optimal "
        "pour intervenir. Ce type de maintenance est particulièrement adapté aux transformateurs critiques des réseaux HT/MT."
    )
    story.append(Paragraph(SAN(txt), styles["BodyText"]))
    story.append(Spacer(1, 10))

    _title(story, styles, "1. Tableau – Huile isolante (valeurs normalisées)", level=3, space_after_pt=4)
    tbl1 = [
        ["Paramètre", "Méthode", "Normale", "Alerte", "Critique", "Référence"],
        ["Rigidité diélectrique BDV (2,5 mm)", "Test BDV", "≥ 70 kV", "50–69 kV", "< 50 kV", "IEC 60156"],
        ["Teneur en eau", "Karl Fischer", "≤ 20 ppm", "20–35 ppm", "> 35 ppm", "IEC 60814"],
        ["Indice d’acidité", "Titrage (mg KOH/g)", "< 0.03", "0.03–0.15", "> 0.15", "IEC 62021"],
        ["Tan δ à 90°C", "Mesure", "≤ 0.005", "0.005–0.02", "> 0.02", "IEC 60247"],
        ["Viscosité (40°C)", "Viscosimètre", "8–12 cSt", "12–15 cSt", "> 15 cSt", "IEC 3104"],
        ["Teneur en furane (2-FAL)", "Chromatographie", "< 0.1 ppm", "0.1–1 ppm", "> 1 ppm", "CIGRE"],
    ]
    widths1 = [3.4*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.6*cm]
    story.append(_mk_table(tbl1, widths=widths1, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "L’huile isolante est un élément essentiel de l’isolation et du refroidissement du transformateur. Lorsque sa "
        "rigidité diélectrique ou son tan δ se dégradent, le risque de claquage interne augmente fortement. Une forte "
        "teneur en eau, en acides ou en furanes indique un vieillissement avancé de l’huile ou du papier isolant. "
        "Une huile en zone critique nécessite une filtration immédiate, voire un remplacement complet."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    abbrev = (
        "Abréviations :\n"
        "• BDV : Breakdown Voltage (rigidité diélectrique).\n"
        "• ppm : parties par million (concentration).\n"
        "• mg KOH/g : indice d’acidité en milligrammes d’hydroxyde de potassium par gramme d’huile.\n"
        "• tan δ : facteur de dissipation diélectrique.\n"
        "• 2-FAL : 2-Furfural, indicateur de vieillissement du papier isolant.\n"
        "• cSt : centistokes, unité de viscosité."
    )
    for line in abbrev.split("\n"):
        story.append(Paragraph(SAN(line), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "2. Tableau – Analyse des gaz dissous (DGA – IEC 60599)", level=3, space_after_pt=4)
    tbl2 = [
        ["Gaz", "Normale", "Alerte", "Critique", "Signification"],
        ["H₂", "< 150 ppm", "150–700 ppm", "> 700 ppm", "Décharges partielles"],
        ["CH₄", "< 80 ppm", "80–120 ppm", "> 120 ppm", "Surchauffe légère"],
        ["C₂H₆", "< 65 ppm", "65–100 ppm", "> 100 ppm", "Surchauffe de l’huile"],
        ["C₂H₄", "< 50 ppm", "50–200 ppm", "> 200 ppm", "Surchauffe élevée"],
        ["C₂H₂", "< 3 ppm", "3–35 ppm", "> 35 ppm", "Arc / défaut grave"],
        ["CO", "< 350 ppm", "350–1500 ppm", "> 1500 ppm", "Dégradation cellulose"],
        ["CO₂", "< 2000 ppm", "2000–10000 ppm", "> 10000 ppm", "Vieillissement papier isolant"],
    ]
    widths2 = [1.4*cm, 2.4*cm, 2.6*cm, 2.6*cm, 7.0*cm]
    story.append(_mk_table(tbl2, widths=widths2, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "L’analyse des gaz dissous (DGA) est la méthode la plus fiable pour détecter des défauts internes non visibles. "
        "Chaque gaz correspond à un type de défaut électrique ou thermique spécifique à l’intérieur du transformateur. "
        "L’acétylène (C₂H₂), même en faible concentration, est souvent lié à des arcs internes et doit alerter immédiatement. "
        "Une dérive progressive des valeurs dans le temps indique un début de défaillance et nécessite une analyse de tendance."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    abbrev2 = (
        "Abréviations :\n"
        "• DGA : Dissolved Gas Analysis (analyse des gaz dissous).\n"
        "• H₂ : Hydrogène ; CH₄ : Méthane ; C₂H₆ : Éthane ; C₂H₄ : Éthylène ; C₂H₂ : Acétylène.\n"
        "• CO / CO₂ : Monoxyde / Dioxyde de carbone."
    )
    for line in abbrev2.split("\n"):
        story.append(Paragraph(SAN(line), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "3. Tableau – Essais électriques (IEC 60076)", level=3, space_after_pt=4)
    tbl3 = [
        ["Test", "Méthode / Appareil", "Valeur ref.", "Alerte", "Référence"],
        ["Résistance d’isolement", "Mégohmmètre 5 kV", "> 1000 MΩ", "< 600 MΩ", "IEC 60076-3"],
        ["Rapport de transfo (TTR)", "Rapporteur", "± 0.5 %", "> 0.5 %", "IEC 60076-1"],
        ["Résistance d’enroulements", "Micro-ohmmètre", "Écart < 2 % phases", "> 2 %", "IEC 60076-1"],
        ["Impédance de court-circuit", "Essai en charge", "10–12 %", "Δ > 3 %", "IEC 60076-5"],
        ["Courant d’excitation", "Essai à vide", "≤ 0.5 % In", "> 1 % In", "—"],
        ["Perte à vide", "Essai", "25–35 kW", "> 40 kW", "Plaque constructeur"],
        ["Perte en charge", "Essai", "300–500 kW", "> 550 kW", "Plaque constructeur"],
        ["Tenue diélectrique", "Essai IEC", "Conforme", "Non conforme", "IEC 60076-3"],
    ]
    widths3 = [4.0*cm, 4.2*cm, 3.0*cm, 3.0*cm, 3.0*cm]
    story.append(_mk_table(tbl3, widths=widths3, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "Les essais électriques permettent de vérifier l’état des enroulements, de l’isolation et du circuit magnétique. "
        "Une variation anormale du rapport de transformation ou des impédances indique un déplacement d’enroulement ou "
        "un court-circuit interne. Le courant d’excitation renseigne sur l’état du noyau magnétique et sur d’éventuelles "
        "saturations. Les pertes à vide et en charge sont essentielles pour évaluer le rendement et la santé globale du transformateur."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    abbrev3 = (
        "Abréviations :\n"
        "• TTR : Transformer Turns Ratio (rapport de transformation).\n"
        "• MΩ : Mégaohms ; In : Courant nominal ; kW : Kilowatts.\n"
        "• IEC : International Electrotechnical Commission."
    )
    for line in abbrev3.split("\n"):
        story.append(Paragraph(SAN(line), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "4. Tableau – Paramètres thermiques et refroidissement", level=3, space_after_pt=4)
    tbl4 = [
        ["Paramètre", "Normale", "Alerte", "Critique"],
        ["Température huile haut", "≤ 90°C", "90–105°C", "> 105°C"],
        ["Point chaud enroulements", "≤ 110°C", "110–130°C", "> 130°C"],
        ["Gradient haut-bas réservoir", "≤ 15°C", "15–25°C", "> 25°C"],
        ["Ventilateurs / Pompes ONAF", "100 % fonctionnels", "1 panne", "≥ 2 pannes"],
    ]
    widths4 = [6.0*cm, 3.6*cm, 3.6*cm, 3.6*cm]
    story.append(_mk_table(tbl4, widths=widths4, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "Le comportement thermique reflète directement la charge et la capacité de refroidissement du transformateur. "
        "Une augmentation excessive du point chaud accélère fortement le vieillissement du papier isolant interne. "
        "Un gradient de température trop élevé entre le haut et le bas du réservoir indique un mauvais transfert thermique. "
        "La défaillance de plusieurs ventilateurs ou pompes de refroidissement est critique, surtout pour les transformateurs ONAF."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations : ONAF : Oil Natural Air Forced (huile naturelle, air forcé)."), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "5. Tableau – Inspection mécanique & visuelle", level=3, space_after_pt=4)
    tbl5 = [
        ["Contrôle", "Critère normal", "Alerte", "Critique"],
        ["Niveau d’huile", "Niveau nominal", "Baisse lente", "Baisse rapide / fuite"],
        ["Isolateurs", "Propres, sans fissures", "Dépôts / effluves", "Fissure / casse"],
        ["Joints et brides", "Étanches", "Suintement léger", "Fuite active"],
        ["Bruit transformateur", "Bruit stable", "Bruit accru / vibration", "Claquements anormaux"],
        ["Relais Buchholz", "Aucun gaz / RAS", "Présence de gaz", "Déclenchement alarme / défaut"],
    ]
    widths5 = [4.0*cm, 4.6*cm, 4.2*cm, 4.2*cm]
    story.append(_mk_table(tbl5, widths=widths5, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "L’inspection mécanique permet de détecter les défauts visibles avant qu’ils ne deviennent critiques. "
        "Les isolateurs, les joints et les brides sont des points sensibles soumis à la chaleur, à l’humidité et aux "
        "contraintes mécaniques. Le relais Buchholz est un indicateur important de dégagement de gaz à l’intérieur du transformateur. "
        "Le bruit et les vibrations fournissent de précieux indices sur des défauts magnétiques ou mécaniques internes."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    abbrev5 = (
        "Abréviations :\n"
        "• RAS : Rien À Signaler.\n"
        "• Buchholz : Relais détecteur de gaz et de défauts internes.\n"
        "• HT/MT : Haute Tension / Moyenne Tension."
    )
    for line in abbrev5.split("\n"):
        story.append(Paragraph(SAN(line), styles["BodyText"]))
    story.append(Spacer(1, 8))

    _title(story, styles, "6. Tableau – Fréquences recommandées des contrôles", level=3, space_after_pt=4)
    tbl6 = [
        ["Type de contrôle", "Fréquence recommandée"],
        ["Inspection visuelle complète", "Mensuelle"],
        ["Analyse d’huile (BDV, eau, acidité)", "Semestrielle"],
        ["DGA (gaz dissous)", "Trimestrielle à semestrielle (selon criticité)"],
        ["Essais électriques complets", "Annuelle"],
        ["Thermographie infrarouge", "Annuelle"],
        ["Contrôle OLTC", "Annuelle"],
        ["Filtration / traitement huile (si besoin)", "Tous les 2 ans"],
        ["Mesure des furanes (papier isolant)", "Tous les 3 ans"],
    ]
    widths6 = [8.8*cm, 8.2*cm]
    story.append(_mk_table(tbl6, widths=widths6, font_size=7))
    story.append(Spacer(1, 4))

    txt = (
        "La fréquence de maintenance dépend de la charge, de l’environnement et de l’âge du transformateur. "
        "Un transformateur fortement sollicité ou exposé à un environnement sévère nécessite une surveillance plus rapprochée. "
        "Les fréquences indiquées servent de base et peuvent être ajustées selon le retour d’expérience et les tendances mesurées. "
        "Un bon calendrier de maintenance augmente la fiabilité et prolonge la durée de vie de l’équipement."
    )
    story.append(Paragraph(SAN("Explication : " + txt), styles["BodyText"]))
    story.append(Spacer(1, 4))

    abbrev6 = (
        "Abréviations : OLTC : On-Load Tap Changer (changeur de prises en charge). "
        "BDV : Breakdown Voltage. DGA : Dissolved Gas Analysis."
    )
    story.append(Paragraph(SAN(abbrev6), styles["BodyText"]))
    story.append(Spacer(1, 10))

    _title(story, styles, "7. Tableau de suivi de maintenance – Transformateur 220/20 kV – 100 MVA", level=3, space_after_pt=4)

    header = [
        "Date",
        "Nom de l’Agent",
        "Paramètre contrôlé",
        "Valeur de référence",
        "Résultat mesuré",
        "État (OK/NOK)",
        "Observations",
    ]
    rows = [
        ["", "", "Rigidité diélectrique", "≥ 50 kV (IEC 60156)", "", "", ""],
        ["", "", "Teneur en eau", "< 20 ppm (IEC 60814)", "", "", ""],
        ["", "", "Indice d’acidité", "< 0.03 mg KOH/g (IEC 62021)", "", "", ""],
        ["", "", "Tan δ à 90°C", "< 0.01 (IEC 60247)", "", "", ""],
        ["", "", "Viscosité huile (40°C)", "8–12 cSt (IEC 3104)", "", "", ""],
        ["", "", "2-FAL (furanes)", "< 0.1 ppm (CIGRE TB 771)", "", "", ""],
        ["", "", "H₂", "< 100 ppm (IEC 60599)", "", "", ""],
        ["", "", "CH₄", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₆", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₄", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₂", "< 1 ppm (IEC 60599)", "", "", ""],
        ["", "", "CO", "< 350 ppm (IEC 60599)", "", "", ""],
        ["", "", "CO₂", "< 2500 ppm (IEC 60599)", "", "", ""],
        ["", "", "Résistance d’isolement 5 kV", "> 1000 MΩ (IEC 60076-3)", "", "", ""],
        ["", "", "TTR (rapport de transformation)", "± 0.5 % (IEC 60076-1)", "", "", ""],
        ["", "", "Résistance d’enroulement", "< 2 % diff. phases (IEC 60076-1)", "", "", ""],
        ["", "", "Impédance de court-circuit", "± 2 % nominal (IEC 60076-5)", "", "", ""],
        ["", "", "Courant d’excitation", "≤ 0.5 % In (Constructeur)", "", "", ""],
        ["", "", "Pertes à vide", "Selon plaque constructeur", "", "", ""],
        ["", "", "Pertes en charge", "Selon plaque constructeur", "", "", ""],
        ["", "", "Température huile haut", "< 95°C (IEC 60076-2)", "", "", ""],
        ["", "", "Point chaud enroulements", "< 120°C (IEC 60076-2)", "", "", ""],
        ["", "", "Gradient vertical", "< 15°C (IEC)", "", "", ""],
        ["", "", "Ventilateurs / Pompes ONAF", "100 % fonctionnels", "", "", ""],
        ["", "", "Niveau d’huile", "Niveau nominal", "", "", ""],
        ["", "", "État isolateurs", "Propres / intacts", "", "", ""],
        ["", "", "Relais Buchholz", "RAS / Aucun gaz", "", "", ""],
        ["", "", "Bruit transformateur", "Régulier / stable", "", "", ""],
    ]
    data = [header] + rows
    widths7 = [1.6*cm, 2.4*cm, 4.2*cm, 4.8*cm, 2.5*cm, 1.7*cm, 2.8*cm]
    story.append(_mk_table(data, widths=widths7, font_size=6))
    story.append(Spacer(1, 6))

    exp = (
        "Ce tableau sert à consigner chaque intervention de maintenance préventive, conditionnelle ou corrective. "
        "Il regroupe les principaux paramètres critiques pour la santé du transformateur, avec les valeurs de référence "
        "associées. L’objectif est d’assurer la traçabilité, la régularité des inspections et la détection précoce des dérives."
    )
    story.append(Paragraph(SAN("Explication : " + exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    how = (
        "Comment remplir les colonnes :\n"
        "• Date : indiquer la date de la mesure (format JJ/MM/AAAA).\n"
        "• Technicien : nom de la personne ayant réalisé les mesures.\n"
        "• Paramètre contrôlé : BDV, H₂, C₂H₂, TTR, isolement, température, etc.\n"
        "• Valeur de référence : valeur normative tirée des tableaux précédents.\n"
        "• Résultat mesuré : valeur réellement mesurée sur le terrain.\n"
        "• État (OK/NOK) : OK si conforme, NOK si dépasse les seuils d’alerte ou critiques.\n"
        "• Observations / Actions : anomalies constatées et actions prévues (filtration, resserrage, réparation, etc.)."
    )
    for line in how.split("\n"):
        story.append(Paragraph(SAN(line), styles["BodyText"]))
    story.append(Spacer(1, 10))

    _title(story, styles, "8. Matériel à prévoir pour la maintenance du transfo 220/20 kV – 100 MVA", level=3, space_after_pt=4)
    header_sp = ["Catégorie", "Pièce de rechange", "Quantité recommandée", "Criticité", "Remarques"]
    data_sp = [header_sp]
    for sp in SPARE_PARTS:
        data_sp.append([
            SAN(sp.get("categorie", "")),
            SAN(sp.get("piece", "")),
            SAN(sp.get("qte_reco", "")),
            SAN(sp.get("criticite", "")),
            SAN(sp.get("remarques", "")),
        ])
    widths_sp = [3.0*cm, 6.2*cm, 2.6*cm, 2.2*cm, 3.8*cm]
    story.append(_mk_table(data_sp, widths=widths_sp, font_size=7))
    story.append(Spacer(1, 10))


# ============================================================
# 2) Annexes dynamiques : résultats d’analyse, tâches, matériels
# ============================================================

def _policy_from_beta(beta: float) -> str:
    try:
        b = float(beta)
    except Exception:
        b = None
    if b is None:
        return "Non déterminée."
    if b > 1.0:
        return "Usure (>1) : maintenance préventive / conditionnelle renforcée avec remplacements ciblés."
    if b < 1.0:
        return "Défauts précoces (<1) : contrôles précoces, fiabilisation, qualité d’installation et inspection rapprochée."
    return "Taux de panne proche du constant (=1) : préventif calendaire ou surveillance standard."


def _maintenance_type_explanation(mtype: str, beta: Any) -> str:
    label = SAN(mtype).strip().lower()
    if "correct" in label:
        return "Ce type a été retenu car il faut corriger un défaut observé ou probable avant stabilisation."
    if "condition" in label:
        return "Ce type a été retenu car l’état de l’équipement doit guider le moment de l’intervention."
    if "prévent" in label or "prevent" in label:
        return "Ce type a été retenu car l’intervention doit être planifiée avant la panne pour limiter le risque."
    if "predict" in label or "prédict" in label:
        return "Ce type a été retenu car les tendances permettent d’anticiper l’intervention."
    return _policy_from_beta(beta)


def _build_choice_explanation(r: Dict[str, Any]) -> str:
    beta = r.get("beta")
    eta = r.get("eta_h", r.get("eta"))
    gamma = r.get("gamma_h", r.get("gamma"))
    model = r.get("model")
    dist = r.get("distribution")
    mtype = r.get("maintenance_type")
    T_rec = r.get("T_recommended_h")
    T_R = r.get("T_R_h")
    T_cost = r.get("T_cost_h")
    mtbf = r.get("MTBF")
    mttr = r.get("MTTR")
    thermal_status = r.get("thermal_status")
    faa = r.get("FAA_max", r.get("faa_max"))
    lol = r.get("loss_of_life_pct")
    decision = r.get("decision_finale")
    motif = r.get("motif_decision")

    parts = []

    if mtype:
        parts.append(f"Type retenu : {SAN(mtype)}.")
    if model or dist:
        parts.append(f"Le modèle retenu est {SAN(model)} avec la loi {SAN(dist)}.")
    if beta is not None:
        parts.append(f"Le paramètre beta = {fnum(beta, 2)} a servi à orienter la politique de maintenance.")
    if eta is not None:
        parts.append(f"La durée caractéristique eta est estimée à {fnum(eta, 1)} h.")
    if gamma is not None:
        parts.append(f"Le décalage gamma vaut {fnum(gamma, 1)} h.")
    if T_rec is not None:
        parts.append(f"L’intervalle recommandé est {fnum(T_rec, 1)} h.")
    if T_R is not None:
        parts.append(f"L’intervalle fiabiliste vaut {fnum(T_R, 1)} h.")
    if T_cost is not None:
        parts.append(f"L’intervalle économique vaut {fnum(T_cost, 1)} h.")
    if mtbf is not None:
        parts.append(f"Le MTBF est de {fnum(mtbf, 1)} h.")
    if mttr is not None:
        parts.append(f"Le MTTR est de {fnum(mttr, 1)} h.")
    if thermal_status not in (None, ""):
        parts.append(f"Le statut thermique est {SAN(thermal_status)}.")
    if faa is not None:
        parts.append(f"Le FAA maximal est {fnum(faa, 3)}.")
    if lol is not None:
        parts.append(f"La perte de vie estimée est {fnum(lol, 3)} %.")
    if decision:
        parts.append(f"Décision finale : {SAN(decision)}.")
    if motif:
        parts.append(f"Motif : {SAN(motif)}")

    return " ".join(parts) if parts else "Aucune explication détaillée disponible."


def _build_influence_table(r: Dict[str, Any]):
    beta = r.get("beta")
    eta = r.get("eta_h", r.get("eta"))
    gamma = r.get("gamma_h", r.get("gamma"))
    model = r.get("model")
    dist = r.get("distribution")
    mtype = r.get("maintenance_type")
    T_rec = r.get("T_recommended_h")
    T_R = r.get("T_R_h")
    T_cost = r.get("T_cost_h")
    mtbf = r.get("MTBF")
    mttr = r.get("MTTR")
    thermal_status = r.get("thermal_status")
    faa = r.get("FAA_max", r.get("faa_max"))
    lol = r.get("loss_of_life_pct")
    days_left = r.get("days_left")
    decision = r.get("decision_finale")

    return [
        ["Paramètre", "Valeur", "Impact sur le choix"],
        ["Type de maintenance retenu", SAN(mtype), _maintenance_type_explanation(SAN(mtype), beta)],
        ["Modèle", SAN(model), "Le comportement global des défaillances influence la stratégie retenue."],
        ["Loi", SAN(dist), "La loi de probabilité retenue structure l’estimation des durées et du risque."],
        ["beta", fnum(beta, 2), "Indique défauts précoces, comportement aléatoire ou usure."],
        ["eta (h)", fnum(eta, 1), "Donne une référence de durée de vie caractéristique."],
        ["gamma (h)", fnum(gamma, 1), "Décalage éventuel du modèle."],
        ["T_recommended (h)", fnum(T_rec, 1), "Intervalle principal proposé pour agir."],
        ["T_R (h)", fnum(T_R, 1), "Intervalle issu du critère de fiabilité."],
        ["T_cost (h)", fnum(T_cost, 1), "Intervalle issu du critère économique."],
        ["MTBF (h)", fnum(mtbf, 1), "Renseigne l’espacement moyen des pannes."],
        ["MTTR (h)", fnum(mttr, 1), "Renseigne le temps moyen de remise en état."],
        ["Statut thermique", SAN(thermal_status), "Peut accélérer ou renforcer l’intervention."],
        ["FAA max", fnum(faa, 3), "Indique l’accélération du vieillissement thermique."],
        ["Perte de vie (%)", fnum(lol, 3), "Indique la consommation estimée de durée de vie."],
        ["Jours restants", SAN(days_left), "Plus l’échéance est proche, plus la priorité augmente."],
        ["Décision finale", SAN(decision), "Conclusion synthétique issue des paramètres disponibles."],
    ]


def _add_per_equipment_summaries(
    story,
    styles,
    metrics_table: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]] | None,
):
    """Section : résultats d’analyse par équipement."""
    if not metrics_table:
        return

    _title(story, styles, "Résultats d’analyse par équipement (issus optimisation)", level=2, space_after_pt=4)

    for r in metrics_table:
        r = r if isinstance(r, dict) else {}
        orig_eq = (r or {}).get("equipment_code", "")
        eq_disp = SAN(orig_eq)

        beta = (r or {}).get("beta")
        eta = (r or {}).get("eta_h", (r or {}).get("eta"))
        gamma = (r or {}).get("gamma_h", (r or {}).get("gamma"))
        T_rec = (r or {}).get("T_recommended_h")
        T_R = (r or {}).get("T_R_h")
        T_cost = (r or {}).get("T_cost_h")
        itv_opt = (r or {}).get("interval_opt_h", (r or {}).get("interval_h"))
        model = (r or {}).get("model")
        dist = (r or {}).get("distribution")
        mtype = (r or {}).get("maintenance_type")
        mtbf = (r or {}).get("MTBF")
        mttr = (r or {}).get("MTTR")
        thermal_status = (r or {}).get("thermal_status")
        faa = (r or {}).get("FAA_max", (r or {}).get("faa_max"))
        lol = (r or {}).get("loss_of_life_pct")

        avail = None
        try:
            if mtbf is not None and mttr is not None and (float(mtbf) + float(mttr)) > 0:
                avail = 100.0 * float(mtbf) / (float(mtbf) + float(mttr))
        except Exception:
            avail = None

        _title(story, styles, f"Fiche intervention – {eq_disp}", level=3, space_after_pt=2)

        data_param = [
            ["Élément", "Valeur"],
            ["Type de maintenance (optimisation)", SAN(mtype)],
            ["Modèle / Loi", f"{SAN(model)} / {SAN(dist)}"],
            ["beta (forme)", fnum(beta, 2)],
            ["eta (échelle, h)", fnum(eta, 1)],
            ["gamma (décalage, h)", fnum(gamma, 1)],
            ["T_recommended (h)", fnum(T_rec, 1)],
            ["T_R (h)", fnum(T_R, 1)],
            ["T_cost (h)", fnum(T_cost, 1)],
            ["Intervalle opt (h) (fallback)", fnum(itv_opt, 1)],
            ["MTBF (h)", fnum(mtbf, 1)],
            ["MTTR (h)", fnum(mttr, 1)],
            ["Disponibilité (%)", fnum(avail, 1)],
            ["Statut thermique", SAN(thermal_status)],
            ["FAA max", fnum(faa, 3)],
            ["Perte de vie (%)", fnum(lol, 3)],
        ]
        story.append(_mk_table(data_param, font_size=8))
        story.append(Spacer(1, 4))

        _title(story, styles, "Choix retenu pour le cas d’étude", level=3, space_after_pt=2)
        story.append(Paragraph(SAN(_build_choice_explanation(r)), styles["BodyText"]))
        story.append(Spacer(1, 6))

        _title(story, styles, "Paramètres ayant influencé le choix", level=3, space_after_pt=2)
        story.append(_mk_table(_build_influence_table(r), font_size=7.5))
        story.append(Spacer(1, 6))

        _title(story, styles, "Politique recommandée", level=3, space_after_pt=2)
        pol = ""
        if mtype:
            pol += f"Type recommandé (optimisation) : {SAN(mtype)}. "
        pol += _policy_from_beta(beta)
        story.append(Paragraph(SAN(pol), styles["BodyText"]))
        story.append(Spacer(1, 6))

        _title(story, styles, "Kit de pièces recommandé", level=3, space_after_pt=2)
        kit_list = (kits_by_eq or {}).get(str(orig_eq), [])
        if kit_list:
            story.append(Paragraph(SAN("Les pièces suivantes sont recommandées pour cet équipement :"), styles["BodyText"]))
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
            story.append(Paragraph(SAN("Aucune pièce recommandée détectée (ou stock non activé)."), styles["BodyText"]))
        story.append(Spacer(1, 8))


def _table_from_tasks_due(tasks_due: List[Dict[str, Any]]):
    if not tasks_due:
        return None

    data = [[
        "Équipement",
        "Type maint.",
        "Intervalle (h)",
        "Source",
        "Échéance",
        "Jours restants",
        "T_rec (h)",
        "T_R (h)",
        "T_cost (h)",
        "Statut",
    ]]

    for t in tasks_due:
        td = t if isinstance(t, dict) else {}
        data.append([
            SAN(td.get("equipment_code", "")),
            SAN(td.get("maintenance_type", "")),
            fnum(td.get("interval_h"), 1),
            SAN(td.get("interval_source", "")),
            SAN(td.get("next_due_date", "")),
            SAN(td.get("days_left", "")),
            fnum(td.get("T_recommended_h"), 1),
            fnum(td.get("T_R_h"), 1),
            fnum(td.get("T_cost_h"), 1),
            SAN(td.get("status", "")),
        ])

    widths = [
        2.2*cm,
        2.8*cm,
        2.0*cm,
        1.7*cm,
        2.3*cm,
        1.6*cm,
        1.9*cm,
        1.7*cm,
        1.8*cm,
        1.7*cm,
    ]

    return _mk_table(data, widths=widths, font_size=7.5)


DEFAULT_TOOLS = [
    {"categorie": "Mesure & Essais", "outil": "Viscosimètre huile", "description": "Mesure viscosité huile diélectrique",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & Essais", "outil": "Micro-ohmmètre", "description": "Résistance d’enroulements",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": "200 A si possible"},
    {"categorie": "Mesure & Essais", "outil": "Mégohmmètre 5 kV", "description": "Résistance d’isolement",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & Essais", "outil": "Testeur BDV huile", "description": "Rigidité diélectrique",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & Essais", "outil": "Caméra thermographique", "description": "Inspection échauffements",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & Essais", "outil": "Multimètre + Pince ampèremétrique",
     "description": "Tensions, courants, continuité",
     "qte": 1, "unite": "set", "calibrage": "OK", "remarques": ""},
    {"categorie": "Mesure & Essais", "outil": "Thermomètre IR", "description": "Points chauds",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Échantillonnage huile", "outil": "Bouteilles + seringues",
     "description": "Prélèvements huile (analyse labo / DGA externe)",
     "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
    {"categorie": "Traitement huile", "outil": "Unité de filtration",
     "description": "Déshydratation/filtration huile sur site",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": "si intervention prévue"},
    {"categorie": "Consommables", "outil": "Joints & garnitures", "description": "Remplacement étanchéité",
     "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
    {"categorie": "Outils", "outil": "Clés dynamométriques", "description": "Serrage au couple",
     "qte": 1, "unite": "set", "calibrage": "OK", "remarques": ""},
    {"categorie": "Outils", "outil": "Pompe à huile", "description": "Transfert/vidange huile",
     "qte": 1, "unite": "pcs", "calibrage": "OK", "remarques": ""},
    {"categorie": "Sécurité (EPI)", "outil": "Gants, visière, tenue arc-flash",
     "description": "Protection intervention BT/HTA",
     "qte": 1, "unite": "set", "calibrage": "N/A", "remarques": ""},
    {"categorie": "Divers", "outil": "Silice déshydratante", "description": "Dessiccateur (renouvellement)",
     "qte": 1, "unite": "pcs", "calibrage": "N/A", "remarques": ""},
]


def _add_tools_section(story, styles, tools_checklist: List[Dict[str, Any]] | None):
    """Liste d’outils/EPI en plus du grand tableau des pièces."""
    _title(story, styles, "Matériels à prévoir pour l’entretien (instruments et EPI)", level=2, space_after_pt=2)
    tools = tools_checklist if (isinstance(tools_checklist, list) and tools_checklist) else DEFAULT_TOOLS
    data = [["Catégorie", "Outil/Instrument", "Description", "Qté", "Unité", "Calibrage/État", "Remarques"]]
    for it in tools:
        it = it if isinstance(it, dict) else {}
        data.append([
            SAN(it.get("categorie", "")),
            SAN(it.get("outil", "")),
            SAN(it.get("description", "")),
            SAN(it.get("qte", "")),
            SAN(it.get("unite", "")),
            SAN(it.get("calibrage", "")),
            SAN(it.get("remarques", "")),
        ])
    widths = [2.8*cm, 3.0*cm, 4.0*cm, 1.1*cm, 1.1*cm, 2.0*cm, 2.3*cm]
    story.append(_mk_table(data, widths=widths, font_size=8))
    story.append(Spacer(1, 6))


def _export_pm_plan_with_kits_pdf_fallback(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]],
    metrics_table: List[Dict[str, Any]],
    out_dir: str | Path,
    title: str,
    tools_checklist: List[Dict[str, Any]] | None,
) -> str:
    """
    Version simplifiée pour plateformes sans ReportLab.
    """
    if not HAVE_FPDF:
        raise RuntimeError(
            "Génération PDF non disponible (ReportLab et FPDF indisponibles). "
            "Installez soit reportlab, soit fpdf2 dans l'environnement."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    date_now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = out_dir / f"pm_plan_kits_{date_now}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, SAN(title))
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, SAN(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")), ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, SAN("Résumé (mode simplifié – ReportLab indisponible)"), ln=1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, SAN(
        "Ce document a été généré dans un environnement où ReportLab n'est pas disponible "
        "(par exemple Streamlit Cloud). Le contenu détaillé avec mise en page complète reste "
        "disponible lorsque l'application est exécutée en local avec ReportLab installé."
    ))
    pdf.ln(4)

    if metrics_table:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, SAN("Synthèse paramètres par équipement :"), ln=1)
        pdf.set_font("Arial", "", 9)
        for r in metrics_table:
            eq = SAN(str(r.get("equipment_code", "")))
            beta = fnum(r.get("beta"), 2)
            eta = fnum(r.get("eta_h", r.get("eta")), 1)
            itv = fnum(r.get("T_recommended_h", r.get("interval_opt_h")), 1)
            mtype = SAN(r.get("maintenance_type", ""))
            pdf.multi_cell(
                0,
                5,
                SAN(f"- {eq} | type={mtype} | beta={beta}, eta={eta} h, intervalle recommandé ≈ {itv} h")
            )
        pdf.ln(3)

    if tasks_due:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, SAN("Tâches de maintenance dues (résumé) :"), ln=1)
        pdf.set_font("Arial", "", 9)
        for t in tasks_due[:40]:
            eq = SAN(str(t.get("equipment_code", "")))
            title_t = SAN(str(t.get("title", t.get("maintenance_type", ""))))
            due = SAN(str(t.get("next_due_date", "")))
            pdf.multi_cell(0, 5, SAN(f"- [{eq}] {title_t} (échéance : {due})"))
        pdf.ln(3)

    if tools_checklist:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, SAN("Matériels / Outils à prévoir (résumé) :"), ln=1)
        pdf.set_font("Arial", "", 9)
        for it in tools_checklist[:40]:
            cat = SAN(str(it.get("categorie", "")))
            outil = SAN(str(it.get("outil", "")))
            pdf.multi_cell(0, 5, SAN(f"- [{cat}] {outil}"))
        pdf.ln(3)

    pdf.output(str(out_path))
    return str(out_path)


# ============================================================
# Export principal
# ============================================================

def export_pm_plan_with_kits_pdf(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]],
    metrics_table: List[Dict[str, Any]],
    out_dir: str | Path = "reports",
    title: str = "Plan de maintenance - Procédure, Tâches, Matériels",
    procedure_docx: str | Path | None = None,
    *,
    include_kits: bool = False,
    tools_checklist: List[Dict[str, Any]] | None = None,
    consumption_summary=None,
) -> str:
    """
    Ordre final :
      1) Cahier de maintenance complet (texte + tableaux statiques)
      2) Résultats d’analyse par équipement
      3) Tâches de maintenance dues (tableau global)
      4) Matériels à prévoir (instruments, EPI…)
    """
    if not HAVE_REPORTLAB:
        return _export_pm_plan_with_kits_pdf_fallback(
            tasks_due, kits_by_eq, metrics_table, out_dir, title, tools_checklist
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    date_now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = out_dir / f"pm_plan_kits_{date_now}.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
    )
    styles = getSampleStyleSheet()

    story: List[Any] = []
    story.append(Paragraph(SAN(title), styles["Title"]))
    story.append(Paragraph(SAN(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # 1) Cahier de maintenance complet
    _add_cahier_maintenance(story, styles)
    story.append(PageBreak())

    # 2) Résultats par équipement
    _add_per_equipment_summaries(story, styles, metrics_table, kits_by_eq)

    # 3) Tâches de maintenance dues
    _title(story, styles, "Tâches de maintenance dues (tableau global)", level=2, space_after_pt=3)
    tbl_due = _table_from_tasks_due(tasks_due)
    if tbl_due is None:
        story.append(Paragraph(SAN("Aucune tâche due actuellement."), styles["BodyText"]))
    else:
        story.append(tbl_due)
    story.append(Spacer(1, 8))

    # 4) Matériels / outils / EPI
    _add_tools_section(story, styles, tools_checklist)

    story.append(Spacer(1, 8))
    story.append(Paragraph(SAN(f"Rapport généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["BodyText"]))

    doc.build(story)
    return str(out_path)