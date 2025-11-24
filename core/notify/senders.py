# core/notify/senders.py
from __future__ import annotations
from typing import Dict
from .config import SMTPConfig, TwilioConfig
from .emailer import SMTPSettings, send_email_smtp

def send_email_alert(subject: str, body: str, smtp: SMTPConfig) -> Dict:
    settings = SMTPSettings(
        host=smtp.host, port=int(smtp.port),
        username=smtp.user, password=smtp.password,
        use_tls=bool(smtp.use_starttls), use_ssl=bool(smtp.use_ssl),
        sender=smtp.from_addr or "SIMCO Monitoring <alerts@simco.cd>"
    )
    return send_email_smtp(settings, smtp.to_addrs or [], subject, body, None)

def send_whatsapp_alert(body: str, tw: TwilioConfig) -> Dict:
    """
    Envoi WhatsApp via Twilio (si la lib est dispo).
    Utilise le format "whatsapp:+243..." attendu par Twilio.
    """
    try:
        from twilio.rest import Client
    except Exception:
        return {"ok": False, "error": "twilio non installé (pip install twilio)"}

    if not (tw.sid and tw.token and tw.whatsapp_from and tw.whatsapp_to):
        return {"ok": False, "error": "config Twilio incomplète"}

    try:
        client = Client(tw.sid, tw.token)
        msg = client.messages.create(
            from_=tw.whatsapp_from if tw.whatsapp_from.startswith("whatsapp:") else f"whatsapp:{tw.whatsapp_from}",
            to=tw.whatsapp_to if tw.whatsapp_to.startswith("whatsapp:") else f"whatsapp:{tw.whatsapp_to}",
            body=body
        )
        return {"ok": True, "sid": msg.sid}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}
