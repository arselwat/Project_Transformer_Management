# pages/7_Parametres_Alertes.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from core.notify.config import (
    AlertsConfig, SMTPConfig, TwilioConfig,
    load_alerts_config, save_alerts_config
)
from core.notify.senders import send_email_alert, send_whatsapp_alert
from core.notify.rt_alerts import notify_event  # test pipeline end-to-end
import time
import streamlit as st
from core.security.auth import require_login

st.set_page_config(page_title="Transformateurs", page_icon="🔌", layout="wide")

require_login()  # tant que auth_ok n’est pas True, cette page est bloquée

# ... le reste de ta page ...

st.set_page_config(page_title="Paramètres alertes", page_icon="🔔", layout="centered")
st.title("🔔 Paramètres d’alertes (SMTP / WhatsApp)")

cfg = load_alerts_config()

def safe_port(p: int | None) -> int:
    try:
        p = int(p or 0)
    except Exception:
        p = 0
    return 587 if p < 1 else p

st.subheader("Général")
cfg.enable_email    = st.checkbox("Activer e-mail", value=bool(cfg.enable_email))
cfg.enable_whatsapp = st.checkbox("Activer WhatsApp (Twilio)", value=bool(cfg.enable_whatsapp))
cfg.subject_prefix  = st.text_input("Préfixe sujet", value=cfg.subject_prefix or "[ALERT] ")
colG1, colG2 = st.columns(2)
with colG1:
    cfg.cooldown_minutes = st.number_input("Anti-spam e-mail (cooldown, minutes)", min_value=1, step=1, value=int(cfg.cooldown_minutes or 15))
with colG2:
    cfg.min_period_s = st.number_input("Compatibilité legacy (min période en s)", min_value=0.0, step=5.0, value=float(cfg.min_period_s or 30.0))

levels_all = ["INFO","WARN","ALARM"]
cfg.levels = st.multiselect("Niveaux qui déclenchent l’e-mail", options=levels_all, default=cfg.levels or ["WARN","ALARM"])
sites_txt = st.text_input("Filtre sites (liste séparée par virgules, vide = tous)", value=",".join(cfg.site_filter or []))
cfg.site_filter = [s.strip() for s in sites_txt.split(",") if s.strip()]

st.subheader("SMTP")
c1, c2 = st.columns(2)
with c1:
    cfg.smtp.host = st.text_input("SMTP_HOST", value=cfg.smtp.host or "")
    cfg.smtp.port = st.number_input("SMTP_PORT", min_value=1, step=1, value=safe_port(cfg.smtp.port))
    cfg.smtp.user = st.text_input("SMTP_USER", value=cfg.smtp.user or "")
with c2:
    cfg.smtp.from_addr = st.text_input("FROM", value=cfg.smtp.from_addr or "")
    cfg.smtp.to_addrs  = [s.strip() for s in st.text_input("TO (séparés par virgules)", value=",".join(cfg.smtp.to_addrs or [])).split(",") if s.strip()]
    cfg.smtp.password  = st.text_input("SMTP_PASS", type="password", value=cfg.smtp.password or "")

c3, c4 = st.columns(2)
with c3:
    cfg.smtp.use_ssl = st.toggle("SSL (465)", value=bool(cfg.smtp.use_ssl))
with c4:
    cfg.smtp.use_starttls = st.toggle("STARTTLS (587)", value=bool(cfg.smtp.use_starttls))

st.subheader("Twilio (WhatsApp)")
c5, c6 = st.columns(2)
with c5:
    cfg.twilio.sid  = st.text_input("TWILIO_SID", value=cfg.twilio.sid or "")
    cfg.twilio.whatsapp_from = st.text_input("WHATSAPP_FROM (whatsapp:+1415...)", value=cfg.twilio.whatsapp_from or "")
with c6:
    cfg.twilio.token = st.text_input("TWILIO_TOKEN", type="password", value=cfg.twilio.token or "")
    cfg.twilio.whatsapp_to   = st.text_input("WHATSAPP_TO (whatsapp:+243...)", value=cfg.twilio.whatsapp_to or "")

st.markdown("---")
b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("💾 Enregistrer"):
        path = save_alerts_config(cfg)
        st.success(f"Configuration enregistrée → {path}")

with b2:
    if st.button("✉️ Test e-mail (direct SMTP)"):
        cfg_now = load_alerts_config()
        res = send_email_alert(
            subject=f"{cfg_now.subject_prefix} TEST",
            body="Ceci est un message de test (Paramètres alertes).",
            smtp=cfg_now.smtp
        )
        st.info(str(res))

with b3:
    if st.button("🟢 Test WhatsApp (direct)"):
        cfg_now = load_alerts_config()
        res = send_whatsapp_alert(
            body="Test WhatsApp – Paramètres alertes",
            tw=cfg_now.twilio
        )
        st.info(str(res))

with b4:
    if st.button("🚨 Test pipeline (ALARM)"):
        # Simule un événement ALARM qui doit passer par notify_event -> SMTP
        ev = {
            "ts": time.time(),
            "site": (cfg.site_filter[0] if cfg.site_filter else "bench1"),
            "equipment": "tr_demo_220_20",
            "level": "ALARM",
            "code": "TEMP_HIGH",
            "msg": "Température hotspot au-dessus du seuil",
            "value": 92.3,
            "threshold": 80.0
        }
        res = notify_event(ev)
        st.info(str(res))

st.caption("Les paramètres sont stockés dans data/alerts_config.json • Logs e-mail : data/alerts_email.log")
