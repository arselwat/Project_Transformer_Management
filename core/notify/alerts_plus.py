from __future__ import annotations
import os, smtplib, ssl, socket, time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import pandas as pd

# ---------- SMTP Settings ----------
def _smtp_settings() -> Dict[str, Any]:
    return {
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASS", ""),
        "secure": os.environ.get("SMTP_SECURE", "starttls"),  # starttls | ssl | plain
        "from_addr": os.environ.get("MAIL_FROM", os.environ.get("SMTP_USER", "no-reply@example.com")),
        "to_default": [e.strip() for e in os.environ.get("MAIL_TO", "").split(",") if e.strip()],
        "timeout": int(os.environ.get("SMTP_TIMEOUT", "30")),
        "debug": os.environ.get("MAIL_DEBUG", "0") == "1",
    }

# ---------- Low-level sender ----------
def send_email(
    subject: str,
    body_text: str,
    to: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    cfg = _smtp_settings()
    to = to or cfg["to_default"]
    cc = cc or []
    bcc = bcc or []
    attachments = attachments or []

    if not to:
        return {"ok": False, "error": "No recipients (MAIL_TO empty and no 'to' provided)"}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg["from_addr"].split("@")[-1])
    msg.set_content(body_text or "")

    for p in attachments:
        try:
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
        except Exception as ex:
            return {"ok": False, "error": f"Attachment error for {p}: {ex}"}

    all_rcpts = list(to) + list(cc) + list(bcc)

    try:
        if cfg["secure"].lower() == "ssl":
            server = smtplib.SMTP_SSL(
                cfg["host"], cfg["port"], timeout=cfg["timeout"], context=ssl.create_default_context()
            )
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"])
        with server:
            if cfg["debug"]:
                server.set_debuglevel(1)
            server.ehlo()
            if cfg["secure"].lower() == "starttls":
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

# ---------- STOCK ALERTS ----------
def notify_stock_alerts(
    low_items: List[Dict[str, Any]],
    extra_recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Envoie un email avec un CSV en PJ listant les articles sous le seuil.
    Retourne toujours un dict {ok: bool, email: {...}, csv: path}.
    """
    if not low_items:
        return {"ok": True, "email": {"ok": True, "note": "No low items"}, "csv": ""}

    df = pd.DataFrame(low_items)
    tmp = Path(tempfile.gettempdir()) / f"low_stock_{int(time.time())}.csv"
    df.to_csv(tmp, index=False, encoding="utf-8")

    subject = f"[Stock] {len(df)} article(s) sous le seuil"
    body = (
        "Bonjour,\n\n"
        "Veuillez trouver en pièce jointe la liste des articles sous le seuil.\n"
        "— Généré automatiquement par l'application Fiabilité & Stock.\n"
    )
    res = send_email(subject, body, to=extra_recipients, attachments=[tmp])
    return {"ok": bool(res.get("ok")), "email": res, "csv": str(tmp)}

# ---------- MAINTENANCE PLAN ----------
def notify_pm_with_kits(
    tasks_due: List[Dict[str, Any]],
    kits_by_eq: Dict[str, Any],
    metrics_table: List[Dict[str, Any]],
    pdf_path: Optional[str] = None,
    extra_recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
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

    subject = f"[Maintenance] Plan et tâches ({len(tasks_due or [])} due)"
    body = (
        "Bonjour,\n\n"
        "Veuillez trouver ci-joint le plan de maintenance et le récapitulatif des tâches dues.\n"
        "— Généré automatiquement par l'application Fiabilité & Stock.\n"
    )

    res = send_email(subject, body, to=extra_recipients, attachments=attachments)
    return {
        "ok": bool(res.get("ok")),
        "email": res,
        "pdf": str(pdf_path or ""),
        "csv": str(tmp_csv),
    }
