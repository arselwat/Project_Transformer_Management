# core/notify/alerts.py
import os, ssl, smtplib, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from core.inventory import services as inv
from core.maintenance import services as pm
from core.maintenance.reporting import export_pm_plan_pdf
from core.reliability.predict import (
    load_failures_csv_for_scheduler,
    failure_risk_by_equipment,
)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
RECIP_FILE = CONFIG_DIR / "alerts.json"

DEFAULT_CONFIG = {
    "emails": [],
    "whatsapp_numbers": [],
    # STOCK
    "low_stock_factor": 1.0,
    # PREDICTIF
    "risk_horizon_h": 720.0,    # 30 jours ~ 720 h
    "risk_threshold": 0.35,     # alerte si P(faille d'ici horizon) >= 35%
    "top_n_risky": 5,
    # MAINTENANCE
    "pm_due_within_days": 7,
    # Optionnel: base URL publique pour joindre un PDF via WhatsApp (Twilio media_url)
    "whatsapp_media_base": ""   # ex: "https://mon-domaine/public/reports/"
}

def load_recipients() -> dict:
    if RECIP_FILE.exists():
        try:
            cfg = json.loads(RECIP_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_recipients(cfg: dict):
    RECIP_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------ E-MAIL ------------------

def _smtp_config_ok() -> bool:
    return all([
        os.getenv("SMTP_HOST"),
        os.getenv("SMTP_USER"),
        os.getenv("SMTP_PASS"),
    ])

def send_email(subject: str, body: str, to_list: List[str], attachments: List[str] | None = None) -> Optional[str]:
    if not to_list:
        return "Aucun destinataire e-mail."
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    if not (host and user and pwd):
        return "SMTP non configuré (.env)."

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # pièces jointes
    for fp in attachments or []:
        try:
            with open(fp, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(fp))
            msg.attach(part)
        except Exception as e:
            # on ne bloque pas l'envoi si une PJ échoue
            print(f"[EMAIL] Impossible de joindre {fp}: {e}")

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx) as server:
            server.login(user, pwd)
            server.sendmail(user, to_list, msg.as_string())
        return None
    except Exception as e:
        return f"Erreur SMTP: {e}"

# ------------------ WhatsApp (Twilio) ------------------

def _twilio_config_ok() -> bool:
    return all([
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN"),
        os.getenv("TWILIO_WHATSAPP_FROM"),
    ])

def send_whatsapp_text(body: str, numbers: List[str]) -> Optional[str]:
    if not numbers:
        return "Aucun numéro WhatsApp."
    if not _twilio_config_ok():
        return "Twilio non configuré (.env)."
    try:
        from twilio.rest import Client
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        tok = os.getenv("TWILIO_AUTH_TOKEN")
        wa_from = os.getenv("TWILIO_WHATSAPP_FROM")
        client = Client(sid, tok)
        for n in numbers:
            to = f"whatsapp:{n}" if not str(n).startswith("whatsapp:") else n
            client.messages.create(from_=wa_from, to=to, body=body)
        return None
    except Exception as e:
        return f"Erreur Twilio: {e}"

def send_whatsapp_with_media(body: str, numbers: List[str], media_url: str) -> Optional[str]:
    """
    Twilio WhatsApp n'accepte que des URLs publiques (pas de fichier local).
    - Fournis un "whatsapp_media_base" dans config/alerts.json si tu héberges tes PDFs.
    - Sinon, on envoie un message texte seulement.
    """
    if not numbers:
        return "Aucun numéro WhatsApp."
    if not _twilio_config_ok():
        return "Twilio non configuré (.env)."
    if not media_url:
        return send_whatsapp_text(body, numbers)

    try:
        from twilio.rest import Client
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        tok = os.getenv("TWILIO_AUTH_TOKEN")
        wa_from = os.getenv("TWILIO_WHATSAPP_FROM")
        client = Client(sid, tok)
        for n in numbers:
            to = f"whatsapp:{n}" if not str(n).startswith("whatsapp:") else n
            client.messages.create(from_=wa_from, to=to, body=body, media_url=[media_url])
        return None
    except Exception as e:
        return f"Erreur Twilio: {e}"

# ------------------ FORMATTEURS ------------------

def fmt_low_stock(items: list) -> str:
    if not items:
        return "Aucune pièce sous seuil."
    lines = ["ALERTE STOCK — pièces sous seuil:"]
    for it in items:
        lines.append(f"- {it.get('code','?')} | {it.get('nom','?')} | qte={it.get('quantite','?')} | seuil={it.get('seuil_min','?')}")
    return "\n".join(lines)

def fmt_risky(risks: List[Dict], thr: float, horizon: float) -> str:
    if not risks:
        return "Aucun équipement à risque élevé."
    lines = [f"ALERTE PREDICTIVE — proba de défaut d'ici {int(horizon)} h (seuil {int(thr*100)}%):"]
    for r in risks:
        lines.append(f"- {r.get('equipment_code','?')} | P(faille)={r.get('risk',0)*100:.1f}% (β={r.get('beta',0):.2f}, η={r.get('eta',0):.1f})")
    return "\n".join(lines)

def fmt_pm(due_list: List[Dict], within: int) -> str:
    if not due_list:
        return f"Aucune tâche de maintenance due dans {within} jours."
    lines = [f"ALERTE MAINTENANCE — tâches dues dans {within} jours:"]
    for t in due_list:
        lines.append(
            f"- {t.get('equipment_code','?')} | {t.get('title','?')} "
            f"| échéance={t.get('next_due_date','?')} | J-{t.get('days_left','?')}"
        )
    return "\n".join(lines)

# ------------------ CHECKERS ------------------

def check_low_stock_and_notify() -> dict:
    cfg = load_recipients()
    factor = float(cfg.get("low_stock_factor", 1.0))
    try:
        under = inv.low_stock(threshold_factor=factor)
    except TypeError:
        # fallback si ton service n'accepte pas threshold_factor
        under = inv.low_stock()
    result = {"count": len(under), "email": None, "whatsapp": None}
    if not under:
        return result

    msg = fmt_low_stock(under)
    if cfg.get("emails") and _smtp_config_ok():
        result["email"] = send_email("Alerte stock de maintenance", msg, cfg["emails"])
    if cfg.get("whatsapp_numbers") and _twilio_config_ok():
        result["whatsapp"] = send_whatsapp_text(msg, cfg["whatsapp_numbers"])
    return result

def check_predictive_risk_and_notify() -> dict:
    cfg = load_recipients()
    horizon = float(cfg.get("risk_horizon_h", 720.0))
    thr = float(cfg.get("risk_threshold", 0.35))
    topn = int(cfg.get("top_n_risky", 5))
    df = load_failures_csv_for_scheduler()
    risks = []
    if not df.empty:
        all_risks = failure_risk_by_equipment(df, horizon_h=horizon)
        risks = [r for r in all_risks if r.get("risk", 0) >= thr][:topn]

    result = {"count": len(risks), "email": None, "whatsapp": None}
    if not risks:
        return result

    msg = fmt_risky(risks, thr, horizon)
    if cfg.get("emails") and _smtp_config_ok():
        result["email"] = send_email("Alerte prédictive défaut transfo", msg, cfg["emails"])
    if cfg.get("whatsapp_numbers") and _twilio_config_ok():
        result["whatsapp"] = send_whatsapp_text(msg, cfg["whatsapp_numbers"])
    return result

def check_pm_due_and_notify() -> dict:
    """
    - Récupère les tâches dues via pm.due_within(days=?)
    - Génère le PDF « plan de maintenance » conforme au modèle
    - Envoie e-mail avec PDF en pièce jointe
    - Envoie WhatsApp (texte) + optionnellement media_url si "whatsapp_media_base" est fourni
    """
    cfg = load_recipients()
    within = int(cfg.get("pm_due_within_days", 7))
    due = pm.due_within(days=within)  # ↔ assure-toi que cette fonction renvoie days_left, etc.

    result = {"count": len(due), "email": None, "whatsapp": None, "pdf": None}
    if not due:
        return result

    # Générer le PDF conforme au modèle
    pdf_path = export_pm_plan_pdf(due, out_dir="reports", title="Plan de maintenance (tâches dues)")
    result["pdf"] = pdf_path

    # Corps de message
    msg = fmt_pm(due, within)

    # Envoi e-mail avec PDF attaché
    if cfg.get("emails") and _smtp_config_ok():
        result["email"] = send_email(
            subject="Alerte maintenance préventive — Plan en PJ",
            body=msg + "\n\n(Plan détaillé en pièce jointe)",
            to_list=cfg["emails"],
            attachments=[pdf_path]
        )

    # WhatsApp : texte + (option média si URL publique possible)
    if cfg.get("whatsapp_numbers") and _twilio_config_ok():
        media_url = ""
        base = str(cfg.get("whatsapp_media_base","")).strip()
        if base:
            # si tu sais publier ton PDF via une URL publique (CDN/HTTP)
            # concatène base + nom du fichier
            media_url = base.rstrip("/") + "/" + os.path.basename(pdf_path)
        # si pas d’URL publique, on envoie uniquement le texte
        result["whatsapp"] = send_whatsapp_with_media(msg, cfg["whatsapp_numbers"], media_url)

    return result
# ------------------ Compatibilité descendante (anciens imports) ------------------

# Ancien nom attendu par certaines pages : `send_whatsapp`
def send_whatsapp(body, numbers):
    """Alias vers l’envoi WhatsApp texte (ancien nom conservé)."""
    return send_whatsapp_text(body, numbers)
def check_pm_due_and_notify() -> dict:
    """
    Récupère les tâches dues, génère le PDF complet (fiabilité+optimisation+plan+kits),
    envoie e-mail avec le PDF en PJ et WhatsApp (texte / media_url si dispo).
    """
    from core.maintenance.reporting_full import export_full_report_pdf
    try:
        from core.inventory import services as inv
        parts = inv.list_parts_as_dicts()
    except Exception:
        parts = []

    cfg = load_recipients()
    within = int(cfg.get("pm_due_within_days", 7))
    due = pm.due_within(days=within)

    result = {"count": len(due), "email": None, "whatsapp": None, "pdf": None}
    if not due:
        return result

    # PDF complet
    pdf_path = export_full_report_pdf(
        failures_ttf_csv="data/failures_saved.csv",
        tasks_due=due,
        inv_parts=parts,
        title=f"Plan de maintenance + Kits (Jusqu’à J+{within})"
    )
    result["pdf"] = pdf_path

    # Corps de message court
    msg = fmt_pm(due, within)

    # E-mail
    if cfg.get("emails") and _smtp_config_ok():
        result["email"] = send_email(
            subject="Alerte maintenance — Plan + Kits en PJ",
            body=msg + "\n\nVoir la pièce jointe pour la synthèse fiabilité, graphes R/F/f/h, plan et kits proposés.",
            to_list=cfg["emails"],
            attachments=[pdf_path]
        )

    # WhatsApp
    if cfg.get("whatsapp_numbers") and _twilio_config_ok():
        base = str(cfg.get("whatsapp_media_base","")).strip()
        media_url = base.rstrip("/") + "/" + os.path.basename(pdf_path) if base else ""
        result["whatsapp"] = send_whatsapp_with_media(msg, cfg["whatsapp_numbers"], media_url)

    return result

# Anciens formatteurs (si ton ancien code importait ces noms)
format_low_stock_message = fmt_low_stock
format_predictive_message = fmt_risky
format_pm_message = fmt_pm
