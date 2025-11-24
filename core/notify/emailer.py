# core/notify/emailer.py
from __future__ import annotations
from typing import List, Optional, Dict
import smtplib, ssl, traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SMTPSettings:
    def __init__(self, host: str, port: int, username: str, password: str,
                 use_tls: bool = True, use_ssl: bool = False, sender: str = "Alerts <alerts@example.com>"):
        self.host = host; self.port = int(port)
        self.username = username; self.password = password
        self.use_tls = bool(use_tls); self.use_ssl = bool(use_ssl)
        self.sender = sender

def send_email_smtp(settings: SMTPSettings, recipients: List[str],
                    subject: str, body_text: str, body_html: Optional[str] = None) -> Dict:
    if not recipients:
        return {"ok": False, "error": "No recipients"}
    msg = MIMEMultipart("alternative")
    msg["From"] = settings.sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text or "", "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        if settings.use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(settings.host, settings.port, context=context) as server:
                if settings.username: server.login(settings.username, settings.password)
                server.sendmail(settings.sender, recipients, msg.as_string())
        else:
            with smtplib.SMTP(settings.host, settings.port, timeout=25) as server:
                server.ehlo()
                if settings.use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                    server.ehlo()
                if settings.username:
                    server.login(settings.username, settings.password)
                server.sendmail(settings.sender, recipients, msg.as_string())
        return {"ok": True}
    except Exception as ex:
        return {"ok": False, "error": f"{ex.__class__.__name__}: {ex}\n{traceback.format_exc()}"}
