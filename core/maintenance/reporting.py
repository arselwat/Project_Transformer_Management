# core/maintenance/reporting.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from datetime import datetime
import os
from typing import List, Dict, Any
# --- Anti-caractères “exotiques” pour PDF basique ---
def SAN(s):
    s = str(s)
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-")
             .replace("•", "-").replace("…", "...")
             .replace("\u00A0", " "))

# ---------------------------
# 1) Modèle de contenu (issu de "PROCEDURE DE MAINTENANCE 2.docx")
# ---------------------------

def _modele_procedure_preventive() -> List[str]:
    return [
        "Inspection visuelle (mensuelle) : connexions, propreté, fuites d’huile, corrosion, isolateurs, câbles, accessoires.",
        "Analyse de l’huile (semestre/annuel) : teneur en eau, gaz dissous (DGA), acidité.",
        "Test d’isolement (annuel) : mégohmmètre, résistance enroulements ↔ masse.",
        "Test rapport de transformation (annuel) : comparer aux valeurs nominales.",
        "Contrôle accessoires (semestre) : refroidissement, régulateurs de tension, protections.",
        "Surveillance température (continue) : capteurs et tendances.",
    ]

def _modele_procedure_predictive() -> List[str]:
    return [
        "Capteurs en continu : température, vibrations, humidité.",
        "Analyse des tendances (logiciel) pour anticiper les défauts internes.",
    ]

def _modele_procedure_corrective() -> List[str]:
    return [
        "Remplacement des pièces défectueuses, réparation connexions/accessoires.",
        "Reconditionnement / remplacement huile isolante si nécessaire.",
    ]

def _modele_valeurs_reference_table() -> List[List[str]]:
    # colonnes : Paramètre | Méthode/Appareil | Valeur de référence
    return [
        ["Résistance d’isolement", "Mégohmmètre 5 kV", "> 100 MΩ entre enroulements et masse"],
        ["Rapport de transformation", "Rapporteur", "± 0.5 % de la valeur nominale"],
        ["Tension de claquage huile", "Test diélectrique", "> 60 kV"],
        ["Teneur en eau (huile)", "Analyse chimique", "< 30 ppm"],
        ["Gaz dissous (DGA)", "Chromatographie", "H₂ < 150 ppm, CH₄ < 100 ppm, C₂H₂ < 35 ppm"],
        ["Température fonctionnement", "Capteur thermique", "< 65 °C (normal), > 85 °C (alarme)"],
        ["Résistance des enroulements", "Micro-ohmmètre", "Variation entre phases < 2 %"],
        ["Teneur en furanes", "Chromatographie", "< 0.1 ppm (normal), > 1 ppm (critique)"],
        ["Indice de neutralisation (huile)", "Titrage chimique", "< 0.3 mg KOH/g"],
    ]

def _modele_checklist_table_vierge() -> List[List[str]]:
    # Tableau de suivi « vierge » pour saisie terrain
    return [
        ["Date", "Technicien", "Élément contrôlé", "Type de test", "Valeur réf.", "Résultat", "État (OK/NOK)", "Observations / Actions"],
        ["", "", "Résistance d’isolement", "Mégohmmètre 5 kV", "> 100 MΩ", "", "", ""],
        ["", "", "Rapport de transformation", "Rapporteur", "± 0.5 %", "", "", ""],
        ["", "", "Tension de claquage de l’huile", "Test diélectrique", "> 60 kV", "", "", ""],
        ["", "", "Teneur en eau (huile)", "Analyse chimique", "< 30 ppm", "", "", ""],
        ["", "", "Gaz dissous (DGA)", "Chromatographie", "H₂<150 ppm, CH₄<100 ppm", "", "", ""],
        ["", "", "Température de fonctionnement", "Capteur thermique", "< 65 °C / >85 °C alarme", "", "", ""],
        ["", "", "Résistance des enroulements", "Micro-ohmmètre", "Δ < 2 %", "", "", ""],
        ["", "", "Teneur en furanes", "Chromatographie", "< 0.1 ppm", "", "", ""],
        ["", "", "Indice de neutralisation (huile)", "Titrage", "< 0.3 mg KOH/g", "", "", ""],
        ["", "", "Inspection visuelle", "Observation", "Aucun défaut visible", "", "", ""],
        ["", "", "Système de refroidissement", "Fonctionnement", "Fluide/ventilateur actif", "", "", ""],
        ["", "", "Dispositifs de protection", "Test fonctionnel", "Déclenchement correct", "", "", ""],
    ]

# ---------------------------
# 2) Génération du PDF
# ---------------------------

def export_pm_plan_pdf(
    tasks: List[Dict[str, Any]],
    out_dir: str = "reports",
    title: str = "Plan de maintenance",
) -> str:
    """
    Génère un PDF complet :
      - Couverture
      - Synthèse des tâches dues (équipement, titre, échéance, priorité, SLA)
      - Procédures : Préventive, Prédictive, Corrective (selon modèle fourni)
      - Tableau « Valeurs de référence » transformateur
      - Checklist de suivi « vierge » à imprimer
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"plan_maintenance_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")

    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, textColor=colors.HexColor("#0d47a1"), spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, textColor=colors.HexColor("#1a237e"), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.grey))

    flow = []

    # --- Couverture ---
    flow.append(Paragraph(f"{title} – Transformateurs", styles["H1"]))
    flow.append(Paragraph(f"Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Small"]))
    flow.append(Spacer(1, 0.4*cm))
    flow.append(Paragraph("Ce document regroupe les interventions à réaliser (préventif), les recommandations prédictives et les valeurs de référence pour contrôle. Il peut être imprimé et complété sur site.", styles["Body"]))
    flow.append(PageBreak())

    # --- Synthèse des tâches dues (si fournies) ---
    flow.append(Paragraph("1) Synthèse – Tâches de maintenance dues", styles["H2"]))
    if tasks:
        header = ["Équipement", "Tâche", "Échéance", "J-?", "Priorité", "SLA (h)", "Statut"]
        rows = [header]
        for t in tasks:
            rows.append([
                str(t.get("equipment_code", "")),
                str(t.get("title","")),
                str(t.get("next_due_date","")),
                str(t.get("days_left","")),
                str(t.get("priority","")),
                str(t.get("sla_hours","")),
                str(t.get("status","")),
            ])
        tbl = Table(rows, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E3F2FD")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0d47a1")),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.Color(0.98,0.98,1)]),
        ]))
        flow.append(tbl)
    else:
        flow.append(Paragraph("Aucune tâche due dans l’intervalle défini.", styles["Body"]))
    flow.append(Spacer(1, 0.4*cm))

    # --- Procédures (modèle) ---
    flow.append(Paragraph("2) Procédure de maintenance – Modèle", styles["H2"]))
    flow.append(Paragraph("<b>a) Maintenance préventive</b>", styles["Body"]))
    for li in _modele_procedure_preventive():
        flow.append(Paragraph(f"• {li}", styles["Body"]))
    flow.append(Spacer(1, 0.2*cm))

    flow.append(Paragraph("<b>b) Maintenance prédictive</b>", styles["Body"]))
    for li in _modele_procedure_predictive():
        flow.append(Paragraph(f"• {li}", styles["Body"]))
    flow.append(Spacer(1, 0.2*cm))

    flow.append(Paragraph("<b>c) Maintenance corrective</b>", styles["Body"]))
    for li in _modele_procedure_corrective():
        flow.append(Paragraph(f"• {li}", styles["Body"]))
    flow.append(PageBreak())

    # --- Valeurs de référence ---
    flow.append(Paragraph("3) Valeurs de référence — Transformateur", styles["H2"]))
    ref_rows = [["Paramètre", "Méthode / Appareil", "Valeur de référence"]]
    ref_rows += _modele_valeurs_reference_table()
    ref_tbl = Table(ref_rows, repeatRows=1, hAlign="LEFT", colWidths=[5*cm, 5*cm, 7*cm])
    ref_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8EAF6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#1a237e")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.Color(0.96,0.98,1)]),
    ]))
    flow.append(ref_tbl)
    flow.append(PageBreak())

    # --- Checklist vierge (à imprimer) ---
    flow.append(Paragraph("4) Tableau de suivi — à compléter lors de l’intervention", styles["H2"]))
    ck_rows = _modele_checklist_table_vierge()
    ck_tbl = Table(ck_rows, repeatRows=1, hAlign="LEFT", colWidths=[2*cm, 2.7*cm, 3.5*cm, 2.7*cm, 3.0*cm, 2.2*cm, 2.2*cm, 4.0*cm])
    ck_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E1F5FE")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#01579B")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    flow.append(ck_tbl)

    doc.build(flow)
    return out_path
