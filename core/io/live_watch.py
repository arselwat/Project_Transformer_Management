# core/notify/live_watch.py
from __future__ import annotations
import os
from typing import Optional
from core.notify.alerts import send_email, send_whatsapp_text, load_recipients

def check_live_thresholds(sample: dict) -> dict:
    """
    Exemple simple : si T° > 95°C ou vib > 0.08 g → alerte.
    Retourne {email:err|None, whatsapp:err|None} si envoi.
    """
    oil = float(sample.get("oil_temp_c", 0) or 0)
    vib = float(sample.get("vibration_g", 0) or 0)
    if oil < 95 and vib < 0.08:
        return {}

    cfg = load_recipients()
    eq = str(sample.get("equipment_code", "?"))
    msg = (
        f"ALERTE TEMPS RÉEL\n"
        f"Équipement: {eq}\n"
        f"T° huile: {oil:.1f} °C (seuil 95)\n"
        f"Vibration: {vib:.3f} g (seuil 0.08)\n"
        f"Charge: {float(sample.get('load_pct',0)):.1f}%\n"
    )
    out = {"email": None, "whatsapp": None}
    if cfg.get("emails"):
        out["email"] = send_email("Alerte capteurs en temps réel", msg, cfg["emails"])
    if cfg.get("whatsapp_numbers"):
        out["whatsapp"] = send_whatsapp_text(msg, cfg["whatsapp_numbers"])
    return out
