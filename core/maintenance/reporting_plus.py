from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import datetime
import math
import unicodedata

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

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


def _title(story, styles, text: str, level: int = 3, space_after_pt: int = 6):
    h = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(level, "Heading3")
    story.append(Paragraph(SAN(text), styles[h]))
    story.append(Spacer(1, space_after_pt))


def _mk_table(data, widths=None, font_size=8):
    t = Table(data, repeatRows=1, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
    # ---------------- En-tête ----------------
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

    # ---------------- 1. Maintenance préventive ----------------
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

    # ---------------- 2. Maintenance conditionnelle ----------------
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

    # ---------------- 3. Maintenance corrective ----------------
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

    # ---------------- 4. Maintenance prédictive ----------------
    _title(story, styles, "4. Maintenance prédictive", level=3, space_after_pt=3)
    txt = (
        "La maintenance prédictive utilise des méthodes avancées de diagnostic (analyse de tendances DGA, modélisation "
        "thermique, capteurs IoT, IA) pour prédire les pannes avant qu’elles ne se produisent. Basée sur l’analyse "
        "statistique des données historiques et en temps réel, elle permet d’anticiper précisément le moment optimal "
        "pour intervenir. Ce type de maintenance est particulièrement adapté aux transformateurs critiques des réseaux HT/MT."
    )
    story.append(Paragraph(SAN(txt), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------
    # 1) Tableau – Huile isolante
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 2) Tableau – DGA
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 3) Tableau – Essais électriques
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 4) Tableau – Paramètres thermiques
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 5) Tableau – Inspection mécanique
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 6) Tableau – Fréquences recommandées
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 7) Tableau de suivi de maintenance
    # -----------------------------------------------------------
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
    widths7 = [1.6*cm, 2.1*cm, 3.8*cm, 3.4*cm, 2.3*cm, 1.5*cm, 2.5*cm]
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

    # -----------------------------------------------------------
    # 8) Matériel à prévoir (liste de pièces de rechange)
    # -----------------------------------------------------------
    _title(story, styles, "8. Matériel à prévoir pour la maintenance du transfo 220/20 kV – 100 MVA", level=3, space_after_pt=4)

    # On réutilise SPARE_PARTS : la liste détaillée des pièces
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
# 1) Cahier de maintenance écrit à la main dans le code
# ============================================================

def _add_cahier_section(story, styles):
    # Titre + intro
    _title(story, styles, "Cahier de Maintenance", level=1, space_after_pt=4)
    _title(story, styles, "Transformateur 220/20 kV – 100 MVA", level=2, space_after_pt=6)

    intro = (
        "Ce cahier regroupe les principaux tableaux de valeurs de référence et les consignes de suivi "
        "pour la maintenance d’un transformateur de puissance 220/20 kV – 100 MVA, conformément aux "
        "références IEC et CIGRE."
    )
    story.append(Paragraph(SAN(intro), styles["BodyText"]))
    story.append(Spacer(1, 6))

    para = (
        "La maintenance appliquée à un transformateur de puissance se divise en plusieurs catégories "
        "complémentaires, chacune ayant un rôle précis dans la fiabilité et la durée de vie de l’équipement."
    )
    story.append(Paragraph(SAN(para), styles["BodyText"]))
    story.append(Spacer(1, 6))

    # 1. Maintenance préventive
    _title(story, styles, "1. Maintenance préventive", level=3, space_after_pt=2)
    p = (
        "La maintenance préventive regroupe toutes les actions planifiées visant à éviter l’apparition de pannes. "
        "Elle est réalisée à des intervalles réguliers (mensuel, trimestriel, annuel) et comprend les inspections "
        "visuelles, les mesures électriques, les analyses d’huile et le contrôle du système de refroidissement. "
        "Son objectif principal est de maintenir le transformateur dans un état optimal en détectant les défauts "
        "mineurs avant qu’ils ne deviennent critiques."
    )
    story.append(Paragraph(SAN(p), styles["BodyText"]))
    story.append(Spacer(1, 6))

    # 2. Maintenance conditionnelle
    _title(story, styles, "2. Maintenance conditionnelle (CBM – Condition Based Maintenance)", level=3, space_after_pt=2)
    p = (
        "La maintenance conditionnelle repose sur l'analyse continue de l’état réel du transformateur. Elle utilise "
        "des mesures et des capteurs (température, vibrations, humidité, DGA, thermographie) permettant de suivre "
        "l’évolution des paramètres critiques. Cette méthode permet d'intervenir uniquement lorsque les indicateurs "
        "montrent un début de dérive, ce qui optimise les coûts et augmente la disponibilité de l’équipement."
    )
    story.append(Paragraph(SAN(p), styles["BodyText"]))
    story.append(Spacer(1, 6))

    # 3. Maintenance corrective
    _title(story, styles, "3. Maintenance corrective", level=3, space_after_pt=2)
    p = (
        "La maintenance corrective intervient après l’apparition d’un défaut ou lorsqu’une mesure dépasse les seuils "
        "critiques fixés par les normes IEC, IEEE ou CIGRE. Elle consiste à réparer ou remplacer les éléments "
        "défectueux (joints, ventilateurs, isolateurs, OLTC, huile, etc.). Dans le cas d’un transformateur, la "
        "maintenance corrective peut inclure la filtration ou le remplacement de l’huile, l’intervention sur les "
        "enroulements, ou le changement des composants mécaniques endommagés."
    )
    story.append(Paragraph(SAN(p), styles["BodyText"]))
    story.append(Spacer(1, 6))

    # 4. Maintenance prédictive
    _title(story, styles, "4. Maintenance prédictive", level=3, space_after_pt=2)
    p = (
        "La maintenance prédictive utilise des méthodes avancées de diagnostic (analyse de tendances DGA, modélisation "
        "thermique, capteurs IoT, IA) pour prédire les pannes avant qu’elles ne se produisent. Basée sur l’analyse "
        "statistique des données historiques et en temps réel, elle permet d’anticiper précisément le moment optimal "
        "pour intervenir. Ce type de maintenance est particulièrement adapté aux transformateurs critiques des réseaux HT/MT."
    )
    story.append(Paragraph(SAN(p), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 1 – Huile isolante
    # ===============================
    _title(story, styles, "1. Tableau – Huile isolante (valeurs normalisées)", level=3, space_after_pt=4)
    data_oil = [
        ["Paramètre", "Méthode", "Valeur normale", "Alerte", "Critique", "Référence"],
        ["Rigidité diélectrique BDV (2,5 mm)", "Test BDV", "≥ 70 kV", "50–69 kV", "< 50 kV", "IEC 60156"],
        ["Teneur en eau", "Karl Fischer", "≤ 20 ppm", "20–35 ppm", "> 35 ppm", "IEC 60814"],
        ["Indice d’acidité", "Titrage (mg KOH/g)", "< 0.03", "0.03–0.15", "> 0.15", "IEC 62021"],
        ["Tan δ à 90°C", "Mesure", "≤ 0.005", "0.005–0.02", "> 0.02", "IEC 60247"],
        ["Viscosité (40°C)", "Viscosimètre", "8–12 cSt", "12–15 cSt", "> 15 cSt", "IEC 3104"],
        ["Teneur en furane (2-FAL)", "Chromatographie", "< 0.1 ppm", "0.1–1 ppm", "> 1 ppm", "CIGRE"],
    ]
    widths_oil = [4.0*cm, 3.0*cm, 2.6*cm, 2.6*cm, 2.6*cm, 3.0*cm]
    story.append(_mk_table(data_oil, widths=widths_oil, font_size=7))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : L’huile isolante est un élément essentiel de l’isolation et du refroidissement du transformateur. "
        "Lorsque sa rigidité diélectrique ou son tan δ se dégradent, le risque de claquage interne augmente fortement. "
        "Une forte teneur en eau, en acides ou en furanes indique un vieillissement avancé de l’huile ou du papier isolant. "
        "Une huile en zone critique nécessite une filtration immédiate, voire un remplacement complet."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- BDV : Breakdown Voltage (rigidité diélectrique)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- ppm : parties par million (concentration)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- mg KOH/g : indice d’acidité mesuré en milligrammes d’hydroxyde de potassium par gramme d’huile."), styles["BodyText"]))
    story.append(Paragraph(SAN("- tan δ : facteur de dissipation diélectrique (pertes dans l’isolant)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- 2-FAL : 2-Furfural, indicateur de vieillissement du papier isolant."), styles["BodyText"]))
    story.append(Paragraph(SAN("- cSt : centistokes, unité de viscosité."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 2 – DGA
    # ===============================
    _title(story, styles, "2. Tableau – Analyse des gaz dissous (DGA – IEC 60599)", level=3, space_after_pt=4)
    data_dga = [
        ["Gaz", "Valeur normale", "Alerte", "Critique", "Signification"],
        ["H₂", "< 150 ppm", "150–700 ppm", "> 700 ppm", "Décharges partielles"],
        ["CH₄", "< 80 ppm", "80–120 ppm", "> 120 ppm", "Surchauffe légère"],
        ["C₂H₆", "< 65 ppm", "65–100 ppm", "> 100 ppm", "Surchauffe de l’huile"],
        ["C₂H₄", "< 50 ppm", "50–200 ppm", "> 200 ppm", "Surchauffe élevée"],
        ["C₂H₂", "< 3 ppm", "3–35 ppm", "> 35 ppm", "Arc / défaut grave"],
        ["CO", "< 350 ppm", "350–1500 ppm", "> 1500 ppm", "Dégradation cellulose"],
        ["CO₂", "< 2000 ppm", "2000–10000 ppm", "> 10000 ppm", "Vieillissement papier isolant"],
    ]
    widths_dga = [2.0*cm, 3.0*cm, 3.2*cm, 3.2*cm, 6.0*cm]
    story.append(_mk_table(data_dga, widths=widths_dga, font_size=7))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : L’analyse des gaz dissous (DGA) est la méthode la plus fiable pour détecter des défauts internes "
        "non visibles. Chaque gaz correspond à un type de défaut électrique ou thermique spécifique à l’intérieur du "
        "transformateur. L’acétylène (C₂H₂), même en faible concentration, est souvent lié à des arcs internes et doit "
        "alerter immédiatement. Une dérive progressive des valeurs dans le temps indique un début de défaillance et "
        "nécessite une analyse de tendance."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- DGA : Dissolved Gas Analysis (analyse des gaz dissous)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- H₂ : Hydrogène."), styles["BodyText"]))
    story.append(Paragraph(SAN("- CH₄ : Méthane."), styles["BodyText"]))
    story.append(Paragraph(SAN("- C₂H₆ : Éthane."), styles["BodyText"]))
    story.append(Paragraph(SAN("- C₂H₄ : Éthylène."), styles["BodyText"]))
    story.append(Paragraph(SAN("- C₂H₂ : Acétylène."), styles["BodyText"]))
    story.append(Paragraph(SAN("- CO / CO₂ : Monoxyde / Dioxyde de carbone."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 3 – Essais électriques
    # ===============================
    _title(story, styles, "3. Tableau – Essais électriques (IEC 60076)", level=3, space_after_pt=4)
    data_elec = [
        ["Test", "Méthode / Appareil", "Valeur de référence", "Seuil d’alerte", "Référence"],
        ["Résistance d’isolement", "Mégaohmmètre 5 kV", "> 1000 MΩ", "< 600 MΩ", "IEC 60076-3"],
        ["Rapport de transformation (TTR)", "Rapporteur", "± 0.5 %", "> 0.5 %", "IEC 60076-1"],
        ["Résistance d’enroulements", "Micro-ohmmètre", "Écart < 2 % entre phases", "> 2 %", "IEC 60076-1"],
        ["Impédance de court-circuit", "Essai en charge", "10–12 %", "Variation > 3 %", "IEC 60076-5"],
        ["Courant d’excitation", "Essai à vide", "≤ 0.5 % In", "> 1 % In", "—"],
        ["Perte à vide", "Essai", "25–35 kW", "> 40 kW", "Plaque constructeur"],
        ["Perte en charge", "Essai", "300–500 kW", "> 550 kW", "Plaque constructeur"],
        ["Tenue diélectrique", "Essai IEC", "Conforme", "Non conforme", "IEC 60076-3"],
    ]
    widths_elec = [4.0*cm, 4.0*cm, 4.0*cm, 3.0*cm, 3.0*cm]
    story.append(_mk_table(data_elec, widths=widths_elec, font_size=7))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : Les essais électriques permettent de vérifier l’état des enroulements, de l’isolation et du circuit "
        "magnétique. Une variation anormale du rapport de transformation ou des impédances indique un déplacement "
        "d’enroulement ou un court-circuit interne. Le courant d’excitation renseigne sur l’état du noyau magnétique et "
        "sur d’éventuelles saturations. Les pertes à vide et en charge sont essentielles pour évaluer le rendement et la "
        "santé globale du transformateur."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- TTR : Transformer Turns Ratio (rapport de transformation)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- MΩ : Mégaohms."), styles["BodyText"]))
    story.append(Paragraph(SAN("- In : Courant nominal."), styles["BodyText"]))
    story.append(Paragraph(SAN("- kW : Kilowatts (pertes)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- IEC : International Electrotechnical Commission."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 4 – Thermique
    # ===============================
    _title(story, styles, "4. Tableau – Paramètres thermiques et refroidissement", level=3, space_after_pt=4)
    data_th = [
        ["Paramètre", "Valeur normale", "Alerte", "Critique"],
        ["Température maximale huile (haut)", "≤ 90°C", "90–105°C", "> 105°C"],
        ["Point chaud enroulements", "≤ 110°C", "110–130°C", "> 130°C"],
        ["Gradient haut-bas réservoir", "≤ 15°C", "15–25°C", "> 25°C"],
        ["Ventilateurs / Pompes ONAF", "100 % fonctionnels", "1 panne", "2+ pannes"],
    ]
    widths_th = [5.0*cm, 3.0*cm, 3.0*cm, 3.0*cm]
    story.append(_mk_table(data_th, widths=widths_th, font_size=8))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : Le comportement thermique reflète directement la charge et la capacité de refroidissement du "
        "transformateur. Une augmentation excessive du point chaud accélère fortement le vieillissement du papier isolant "
        "interne. Un gradient de température trop élevé entre le haut et le bas du réservoir indique un mauvais transfert "
        "thermique. La défaillance de plusieurs ventilateurs ou pompes de refroidissement est critique, surtout pour les "
        "transformateurs ONAF."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- ONAF : Oil Natural Air Forced (huile naturelle, air forcé)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- °C : Degré Celsius."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 5 – Inspection mécanique & visuelle
    # ===============================
    _title(story, styles, "5. Tableau – Inspection mécanique & visuelle", level=3, space_after_pt=4)
    data_mech = [
        ["Contrôle", "Critère normal", "Alerte", "Critique"],
        ["Niveau d’huile", "Niveau nominal", "Baisse lente", "Baisse rapide / fuite"],
        ["Isolateurs", "Propres, sans fissures", "Dépôts / traces d’effluve", "Fissure / casse"],
        ["Joints et brides", "Étanches", "Suintement léger", "Fuite active"],
        ["Bruit transformateur", "Bruit stable", "Bruit accru / vibration", "Claquements anormaux"],
        ["Relais Buchholz", "Aucun gaz / RAS", "Présence de gaz", "Déclenchement alarme / défaut"],
    ]
    widths_mech = [4.0*cm, 4.5*cm, 4.0*cm, 4.0*cm]
    story.append(_mk_table(data_mech, widths=widths_mech, font_size=8))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : L’inspection mécanique permet de détecter les défauts visibles avant qu’ils ne deviennent critiques. "
        "Les isolateurs, les joints et les brides sont des points sensibles soumis à la chaleur, à l’humidité et aux "
        "contraintes mécaniques. Le relais Buchholz est un indicateur important de dégagement de gaz à l’intérieur du "
        "transformateur. Le bruit et les vibrations du transformateur fournissent de précieux indices sur des défauts "
        "magnétiques ou mécaniques internes."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- RAS : Rien À Signaler."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Buchholz : Relais détecteur de gaz et de défauts internes."), styles["BodyText"]))
    story.append(Paragraph(SAN("- HT/MT : Haute Tension / Moyenne Tension."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 6 – Fréquences recommandées
    # ===============================
    _title(story, styles, "6. Tableau – Fréquences recommandées des contrôles", level=3, space_after_pt=4)
    data_freq = [
        ["Type de contrôle", "Fréquence recommandée"],
        ["Inspection visuelle complète", "Mensuelle"],
        ["Analyse d’huile (BDV, eau, acidité)", "Semestrielle"],
        ["DGA (gaz dissous)", "Trimestrielle à semestrielle selon criticité"],
        ["Essais électriques complets", "Annuelle"],
        ["Thermographie infrarouge", "Annuelle"],
        ["Contrôle OLTC", "Annuelle"],
        ["Filtration / traitement huile (si besoin)", "Tous les 2 ans"],
        ["Mesure des furanes (papier isolant)", "Tous les 3 ans"],
    ]
    story.append(_mk_table(data_freq, font_size=8))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : La fréquence de maintenance dépend de la charge, de l’environnement et de l’âge du transformateur. "
        "Un transformateur fortement sollicité ou exposé à un environnement sévère nécessite une surveillance plus "
        "rapprochée. Les fréquences indiquées servent de base et peuvent être ajustées selon le retour d’expérience et "
        "les tendances mesurées. Un bon calendrier de maintenance augmente la fiabilité et prolonge la durée de vie de "
        "l’équipement."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Abréviations :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- OLTC : On-Load Tap Changer (changeur de prises en charge)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- BDV : Breakdown Voltage."), styles["BodyText"]))
    story.append(Paragraph(SAN("- DGA : Dissolved Gas Analysis."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Tableau 7 – Suivi de maintenance (vierge)
    # ===============================
    _title(story, styles, "7. Tableau de suivi de maintenance – Transformateur 220/20 kV – 100 MVA", level=3, space_after_pt=4)
    header = ["Date", "Nom de l’Agent", "Paramètre contrôlé", "Valeur de référence",
              "Résultat mesuré", "État (OK/NOK)", "Observations"]
    rows = [
        ["", "", "Rigidité diélectrique", "≥ 50 kV (IEC 60156)", "", "", ""],
        ["", "", "Teneur en eau", "< 20 ppm (IEC 60814)", "", "", ""],
        ["", "", "Indice d’acidité", "< 0.03 mg KOH/g (IEC 62021)", "", "", ""],
        ["", "", "Tan δ à 90°C", "< 0.01 (IEC 60247)", "", "", ""],
        ["", "", "Viscosité huile (40°C)", "8–12 cSt (IEC 3104)", "", "", ""],
        ["", "", "2-FAL (furannes)", "< 0.1 ppm (CIGRE TB 771)", "", "", ""],
        ["", "", "H₂", "< 100 ppm (IEC 60599)", "", "", ""],
        ["", "", "CH₄", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₆", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₄", "< 50 ppm (IEC 60599)", "", "", ""],
        ["", "", "C₂H₂", "< 1 ppm (IEC 60599)", "", "", ""],
        ["", "", "CO", "< 350 ppm (IEC 60599)", "", "", ""],
        ["", "", "CO₂", "< 2500 ppm (IEC 60599)", "", "", ""],
        ["", "", "Résistance d’isolement 5 kV", "> 1000 MΩ (IEC 60076-3)", "", "", ""],
        ["", "", "TTR (rapport de transformation)", "± 0.5 % (IEC 60076-1)", "", "", ""],
        ["", "", "Résistance d’enroulement", "< 2 % diff entre phases (IEC 60076-1)", "", "", ""],
        ["", "", "Impédance de court-circuit", "± 2 % nominal (IEC 60076-5)", "", "", ""],
        ["", "", "Courant d'excitation", "≤ 0.5 % In (Constructeur)", "", "", ""],
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
    data_suivi = [header] + rows
    widths_suivi = [2.0*cm, 3.2*cm, 5.5*cm, 5.5*cm, 3.0*cm, 2.0*cm, 4.0*cm]
    story.append(_mk_table(data_suivi, widths=widths_suivi, font_size=6.8))
    story.append(Spacer(1, 6))

    exp = (
        "Explication : Ce tableau sert à consigner chaque intervention de maintenance préventive, conditionnelle ou "
        "corrective. Il regroupe les principaux paramètres critiques pour la santé du transformateur, avec les valeurs "
        "de référence associées. L’objectif est d’assurer la traçabilité, la régularité des inspections et la détection "
        "précoce des dérives. Un tableau bien rempli constitue un élément essentiel du carnet de vie du transformateur."
    )
    story.append(Paragraph(SAN(exp), styles["BodyText"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(SAN("Comment remplir les colonnes :"), styles["BodyText"]))
    story.append(Paragraph(SAN("- Date : indiquer la date de la mesure (format JJ/MM/AAAA)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Technicien : mentionner le nom de la personne ayant réalisé les mesures."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Paramètre contrôlé : indiquer le paramètre (BDV, H₂, C₂H₂, TTR, isolement, température, etc.)."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Valeur de référence : recopier la valeur normative tirée des tableaux précédents."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Résultat mesuré : inscrire la valeur réellement mesurée sur le terrain."), styles["BodyText"]))
    story.append(Paragraph(SAN("- État (OK/NOK) : cocher OK si conforme, NOK si la valeur dépasse les seuils d’alerte ou critiques."), styles["BodyText"]))
    story.append(Paragraph(SAN("- Observations / Actions : noter les anomalies constatées et les actions prévues (filtration, resserrage, réparation, etc.)."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    # ===============================
    # Matériel à prévoir (grande table)
    # ===============================
    _title(story, styles, "6. Matériel à prévoir pour la maintenance du transfo 220/20 kV – 100 MVA", level=3, space_after_pt=4)
    data_mat = [["Catégorie", "Pièce de rechange", "Quantité recommandée", "Criticité", "Remarques"]]
    for sp in SPARE_PARTS:
        data_mat.append([
            SAN(sp["categorie"]),
            SAN(sp["piece"]),
            SAN(sp["qte_reco"]),
            SAN(sp["criticite"]),
            SAN(sp["remarques"]),
        ])
    widths_mat = [4.0*cm, 5.5*cm, 3.0*cm, 2.5*cm, 4.0*cm]
    story.append(_mk_table(data_mat, widths=widths_mat, font_size=7))
    story.append(Spacer(1, 8))

    # Fin cahier
    story.append(PageBreak())


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
        return "Usure (>1): CBM + remplacements préventifs ciblés, contrôle diélectrique, nettoyage/refroidissement."
    if b < 1.0:
        return "Défauts précoces (<1): contrôles précoces, qualité d’installation, resserrage/inspection rapprochés."
    return "Taux de panne ~constant (=1): préventif calendaire/opportuniste, surveillance standard."


def _add_per_equipment_summaries(
    story,
    styles,
    metrics_table: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]] | None,
):
    """Section : 'Résultats d’analyse par équipement' (partie dynamique)."""
    if not metrics_table:
        return
    _title(story, styles, "Résultats d’analyse par équipement", level=2, space_after_pt=4)

    for r in metrics_table:
        r = r if isinstance(r, dict) else {}
        orig_eq = (r or {}).get("equipment_code", "")
        eq_disp = SAN(orig_eq)
        beta = (r or {}).get("beta")
        eta = (r or {}).get("eta")
        gamma = (r or {}).get("gamma")
        itv = (r or {}).get("interval_opt_h")
        mtbf = (r or {}).get("MTBF")
        mttr = (r or {}).get("MTTR")
        avail = None
        try:
            if mtbf is not None and mttr is not None and (float(mtbf) + float(mttr)) > 0:
                avail = 100.0 * float(mtbf) / (float(mtbf) + float(mttr))
        except Exception:
            avail = None

        _title(story, styles, f"Fiche intervention – {eq_disp}", level=3, space_after_pt=2)

        data_param = [
            ["Élément", "Valeur"],
            ["β (forme)", fnum(beta, 2)],
            ["η (échelle, h)", fnum(eta, 1)],
            ["γ (décalage, h)", fnum(gamma, 1)],
            ["Intervalle de visite optimisé (h)", fnum(itv, 1)],
            ["MTBF (h)", fnum(mtbf, 1)],
            ["MTTR (h)", fnum(mttr, 1)],
            ["Disponibilité (%)", fnum(avail, 1)],
        ]
        story.append(_mk_table(data_param, font_size=8))
        story.append(Spacer(1, 4))

        _title(story, styles, "Politique recommandée (d’après β)", level=3, space_after_pt=2)
        story.append(Paragraph(SAN(_policy_from_beta(beta)), styles["BodyText"]))
        story.append(Spacer(1, 4))

        _title(story, styles, "Tâches planifiées / dues (résumé)", level=3, space_after_pt=2)
        story.append(Paragraph(SAN("Voir le tableau global des tâches dues en fin de document."), styles["BodyText"]))
        story.append(Spacer(1, 4))

        _title(story, styles, "Kit de pièces recommandé (résumé)", level=3, space_after_pt=2)
        has_kit = bool((kits_by_eq or {}).get(str(orig_eq)))
        if has_kit:
            story.append(Paragraph(SAN("Un kit est recommandé. Voir module Inventaire pour la liste à jour."), styles["BodyText"]))
        else:
            story.append(Paragraph(SAN("Aucune pièce recommandée détectée."), styles["BodyText"]))
        story.append(Spacer(1, 6))


def _table_from_tasks_due(tasks_due: List[Dict[str, Any]]):
    if not tasks_due:
        return None
    data = [["Équipement", "Titre", "Priorité", "Échéance", "Jours restants", "Statut"]]
    for t in tasks_due:
        td = t if isinstance(t, dict) else {}
        data.append([
            SAN(td.get("equipment_code", "")),
            SAN(td.get("title", "")),
            SAN(td.get("priority", "")),
            SAN(td.get("next_due_date", "")),
            SAN(td.get("days_left", "")),
            SAN(td.get("status", "")),
        ])
    return _mk_table(
        data,
        widths=[3.2*cm, 5.5*cm, 2.0*cm, 2.5*cm, 2.5*cm, 2.5*cm],
        font_size=9,
    )


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
    """Liste courte d’outils/EPI (en plus du grand tableau de pièces)."""
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


# ============================================================
# Export principal
# ============================================================

def export_pm_plan_with_kits_pdf(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, List[Dict[str, Any]]],
    metrics_table: List[Dict[str, Any]],
    out_dir: str | Path = "reports",
    title: str = "Plan de maintenance - Procédure, Tâches, Matériels",
    procedure_docx: str | Path | None = None,   # plus utilisé, juste pour compat
    *,
    include_kits: bool = False,                 # compat ancien code
    tools_checklist: List[Dict[str, Any]] | None = None,
    consumption_summary=None,                   # compat appelant
) -> str:
    """
    Ordre final :
      1) Cahier de maintenance complet (texte + tableaux statiques)
      2) Résultats d’analyse par équipement (β, η, γ, MTBF, MTTR, politique)
      3) Tâches de maintenance dues (tableau global)
      4) Matériels à prévoir (instruments, EPI…)
    """
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

    # 1) Cahier de maintenance (statique, reconstitué à la main)
    _add_cahier_maintenance(story, styles)
    story.append(PageBreak())

    # 2) Résultats par équipement
    _add_per_equipment_summaries(story, styles, metrics_table, kits_by_eq)

    # 3) Tâches de maintenance dues (tableau global)
    _title(story, styles, "Tâches de maintenance dues (tableau global)", level=2, space_after_pt=3)
    tbl_due = _table_from_tasks_due(tasks_due)
    if tbl_due is None:
        story.append(Paragraph(SAN("Aucune tâche due actuellement."), styles["BodyText"]))
    else:
        story.append(tbl_due)
    story.append(Spacer(1, 8))

    # 4) Matériels (outillage, EPI…)
    _add_tools_section(story, styles, tools_checklist)

    story.append(Spacer(1, 8))
    story.append(Paragraph(SAN(f"Rapport généré le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), styles["BodyText"]))

    doc.build(story)
    return str(out_path)
