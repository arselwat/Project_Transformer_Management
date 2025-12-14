# core/notify/alerts_plus.py
from __future__ import annotations

import smtplib
import ssl
import socket
import time
import tempfile
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from core.notify.config import load_alerts_config


# ========== CONFIG SMTP COMMUNE (même que temps réel) ==========

def _smtp_settings() -> Dict[str, Any]:
    """
    Lit alerts_config.json via load_alerts_config() pour récupérer
    exactement les mêmes paramètres SMTP que les alertes temps réel.
    """
    cfg = load_alerts_config()
    return {
        "host": cfg.smtp.host,
        "port": cfg.smtp.port,
        "user": cfg.smtp.user,
        "password": cfg.smtp.password,
        "secure": "ssl" if cfg.smtp.use_ssl else ("starttls" if cfg.smtp.use_starttls else "plain"),
        "from_addr": cfg.smtp.from_addr or cfg.smtp.user,
        "to_default": list(cfg.smtp.to_addrs or []),
        "timeout": 30,
        "debug": False,
    }


# ========== ENVOI E-MAIL BAS NIVEAU ==========

def send_email(
    subject: str,
    body_text: str,
    to: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    Envoie un e-mail simple (texte + PJ) en utilisant la config SMTP centrale.
    Utilisé par notify_stock_alerts et d'autres modules.
    """
    cfg = _smtp_settings()
    to = to or cfg["to_default"]
    cc = cc or []
    bcc = bcc or []
    attachments = attachments or []

    if not to:
        return {"ok": False, "error": "No recipients (smtp.to_addrs empty and no 'to' provided)"}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg["from_addr"].split("@")[-1])
    msg.set_content(body_text or "")

    # pièces jointes
    for p in attachments:
        p = Path(p)
        if not p.exists():
            continue
        data = p.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=p.name,
        )

    all_rcpts = list(to) + list(cc) + list(bcc)

    try:
        # Connexion SMTP (STARTTLS / SSL / plain comme pour les alertes temps réel)
        if cfg["secure"] == "ssl":
            server = smtplib.SMTP_SSL(
                cfg["host"],
                cfg["port"],
                timeout=cfg["timeout"],
                context=ssl.create_default_context(),
            )
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"])

        with server:
            if cfg["debug"]:
                server.set_debuglevel(1)
            server.ehlo()
            if cfg["secure"] == "starttls":
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if cfg["user"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg, from_addr=cfg["from_addr"], to_addrs=all_rcpts)

        return {"ok": True, "to": all_rcpts, "message_id": msg["Message-ID"]}

    except (smtplib.SMTPException, socket.gaierror, ConnectionError, TimeoutError) as ex:
        return {"ok": False, "error": f"SMTP error: {ex}"}
    except Exception as ex:
        return {"ok": False, "error": f"Unknown error: {ex}"}


# ========== ALERTES STOCK (page Stock) ==========

def notify_stock_alerts(
    low_items: List[Dict[str, Any]],
    extra_recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Envoie un e-mail d'alerte stock avec un CSV en pièce jointe listant
    les articles sous le seuil. Utilise les MÊMES destinataires que les
    alertes temps réel (alerts_config.json → smtp.to_addrs), sauf si
    extra_recipients est fourni.
    """
    if not low_items:
        return {"ok": True, "email": {"ok": True, "note": "No low items"}, "csv": ""}

    # Génération CSV temporaire
    df = pd.DataFrame(low_items)
    tmp = Path(tempfile.gettempdir()) / f"low_stock_{int(time.time())}.csv"
    df.to_csv(tmp, index=False, encoding="utf-8")

    # Sujet et corps proches des alertes transfo, mais spécifiques au stock
    subject = "[ALERTE TRANSFO] Stock sous seuil"
    body = (
        "Bonjour,\n\n"
        "Veuillez trouver en pièce jointe la liste des articles sous le seuil.\n"
        "— Généré automatiquement par l'application Fiabilité & Stock.\n"
    )

    # Destinataires : même liste que temps réel si rien n'est passé
    if not extra_recipients:
        cfg = load_alerts_config()
        extra_recipients = list(cfg.smtp.to_addrs or [])

    res = send_email(subject, body, to=extra_recipients or None, attachments=[tmp])
    return {"ok": bool(res.get("ok")), "email": res, "csv": str(tmp)}


# ========== (OPTIONNEL) AUTRES NOTIFICATIONS LIÉES AU STOCK / MAINTENANCE ==========

def notify_pm_with_kits(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, Any],
    metrics_table: List[Dict[str, Any]],
    pdf_path: Optional[str] = None,
    extra_recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Exemple : envoi du plan de maintenance avec pièces nécessaires.
    Réutilise la même config SMTP et les mêmes destinataires par défaut.
    """
    attachments: List[Path] = []

    if pdf_path:
        p = Path(pdf_path)
        if p.exists():
            attachments.append(p)

    tmp_csv = Path(tempfile.gettempdir()) / f"pm_due_{int(time.time())}.csv"
    try:
        pd.DataFrame(tasks_due or []).to_csv(tmp_csv, index=False, encoding="utf-8")
        attachments.append(tmp_csv)
    except Exception:
        pass

    subject = "[ALERTE TRANSFO] Plan de maintenance"
    body = (
        "Bonjour,\n\n"
        "Veuillez trouver ci-joint le plan de maintenance et les tâches dues.\n"
        "— Généré automatiquement par l'application Fiabilité & Stock.\n"
    )

    if not extra_recipients:
        cfg = load_alerts_config()
        extra_recipients = list(cfg.smtp.to_addrs or [])

    res = send_email(subject, body, to=extra_recipients or None, attachments=attachments)
    return {
        "ok": bool(res.get("ok")),
        "email": res,
        "pdf": str(pdf_path or ""),
        "csv": str(tmp_csv),
    }
