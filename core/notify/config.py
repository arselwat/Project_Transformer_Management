# core/notify/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import os, json
from typing import List, Dict, Any

BASE_DIR = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = (BASE_DIR / "alerts_config.json").resolve()

@dataclass
class SMTPConfig:
    host: str = "smtp.gmail.com"
    port: int = 587
    user: str = ""
    password: str = ""
    from_addr: str = "SIMCO Monitoring <alerts@simco.cd>"
    to_addrs: List[str] = field(default_factory=lambda: ["ops@simco.cd"])
    use_ssl: bool = False
    use_starttls: bool = True

@dataclass
class TwilioConfig:
    sid: str = ""
    token: str = ""
    whatsapp_from: str = ""  # format "whatsapp:+1415..."
    whatsapp_to: str = ""    # format "whatsapp:+243..."

@dataclass
class AlertsConfig:
    # Activation par canal
    enable_email: bool = True
    enable_whatsapp: bool = False

    # Sujet / anti-spam (legacy) + pipeline
    subject_prefix: str = "[ALERT] "
    min_period_s: float = 30.0              # hérité de ta page (on le map sur cooldown)
    cooldown_minutes: int = 15              # pipeline e-mail (anti-spam {equip|code|level})
    levels: List[str] = field(default_factory=lambda: ["WARN", "ALARM"])
    site_filter: List[str] = field(default_factory=list)

    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)

def _coerce_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, list): return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return []

def _merge_dict(d: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    for k, v in (d or {}).items():
        out[k] = v
    return out

def load_alerts_config() -> AlertsConfig:
    if not CFG_PATH.exists():
        cfg = AlertsConfig()
        save_alerts_config(cfg)
        return cfg
    try:
        raw = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = {}

    smtp_raw   = _merge_dict(raw.get("smtp", {}), asdict(SMTPConfig()))
    twilio_raw = _merge_dict(raw.get("twilio", {}), asdict(TwilioConfig()))

    # normalisation
    smtp_raw["to_addrs"] = _coerce_list(smtp_raw.get("to_addrs"))
    levels = [s.upper() for s in _coerce_list(raw.get("levels", ["WARN","ALARM"]))]
    site_filter = _coerce_list(raw.get("site_filter", []))

    cfg = AlertsConfig(
        enable_email=bool(raw.get("enabled", raw.get("enable_email", True))),
        enable_whatsapp=bool(raw.get("enable_whatsapp", False)),
        subject_prefix=str(raw.get("subject_prefix", raw.get("subject_prefix", "[ALERT] "))),
        min_period_s=float(raw.get("min_period_s", 30.0)),
        cooldown_minutes=int(raw.get("cooldown_minutes", 15)),
        levels=levels or ["WARN","ALARM"],
        site_filter=site_filter,
        smtp=SMTPConfig(**smtp_raw),
        twilio=TwilioConfig(**twilio_raw),
    )
    return cfg

def save_alerts_config(cfg: AlertsConfig) -> Path:
    # on conserve aussi la clé "enabled" pour compatibilité
    payload = asdict(cfg)
    payload["enabled"] = bool(cfg.enable_email)
    CFG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CFG_PATH
