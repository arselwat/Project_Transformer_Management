# core/notify/rt_alerts.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import os, json, time, datetime
from .config import load_alerts_config
from .emailer import SMTPSettings, send_email_smtp

BASE_DIR = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = (BASE_DIR / "alerts_state.json").resolve()
LOG_PATH   = (BASE_DIR / "alerts_email.log").resolve()

def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_sent": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_sent": {}}

def _save_state(state: Dict[str, Any]):
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass

def _log(line: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {line}\n")
    except Exception:
        pass

def _format_subject(prefix: str, equipment: str, level: str, code: str) -> str:
    icon = "🔴" if level=="ALARM" else ("🟡" if level=="WARN" else "ℹ️")
    return f"{prefix}{icon} [{level}] {equipment} — {code}"

def _format_body_text(e: Dict[str, Any]) -> str:
    ts = float(e.get("ts", time.time()))
    tstr = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Date/Heure : {tstr}",
        f"Site       : {e.get('site','')}",
        f"Equipement : {e.get('equipment','')}",
        f"Niveau     : {e.get('level','')}",
        f"Code       : {e.get('code','')}",
        f"Message    : {e.get('msg','')}",
    ]
    if e.get("value") not in (None, ""):     lines.append(f"Valeur     : {e.get('value')}")
    if e.get("threshold") not in (None, ""): lines.append(f"Seuil      : {e.get('threshold')}")
    lines.append("")
    lines.append("— Notification automatique —")
    return "\n".join(lines)

def _cooldown_ok(state: Dict[str, Any], key: str, minutes: int) -> bool:
    last = float(state.get("last_sent", {}).get(key, 0.0))
    return (time.time() - last) >= minutes * 60.0

def _remember_sent(state: Dict[str, Any], key: str):
    state.setdefault("last_sent", {})[key] = time.time()
    _save_state(state)

def notify_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Appelé par le bus. Envoie e-mail si:
      - enable_email = True
      - level ∈ levels
      - (site_filter vide ou contient le site)
      - cooldown respecté
    """
    cfg = load_alerts_config()
    if not cfg.enable_email:
        return {"ok": False, "skipped": "email-disabled"}

    level = str(event.get("level","")).upper()
    if cfg.levels and level not in [x.upper() for x in cfg.levels]:
        return {"ok": False, "skipped": "level-filter"}

    if cfg.site_filter:
        if str(event.get("site","")) not in cfg.site_filter:
            return {"ok": False, "skipped": "site-filter"}

    recipients = cfg.smtp.to_addrs or []
    if not recipients:
        return {"ok": False, "error": "No recipients configured"}

    equip = str(event.get("equipment",""))
    code  = str(event.get("code",""))
    key = f"{equip}|{code}|{level}"

    state = _load_state()
    minutes = int(cfg.cooldown_minutes or max(1, int(cfg.min_period_s/60.0)))
    if not _cooldown_ok(state, key, minutes):
        return {"ok": False, "skipped": "cooldown"}

    settings = SMTPSettings(
        host=cfg.smtp.host, port=int(cfg.smtp.port),
        username=cfg.smtp.user, password=cfg.smtp.password,
        use_tls=bool(cfg.smtp.use_starttls), use_ssl=bool(cfg.smtp.use_ssl),
        sender=cfg.smtp.from_addr or "SIMCO Monitoring <alerts@simco.cd>"
    )

    subj = _format_subject(cfg.subject_prefix or "", equip or "?", level, code or "?")
    txt  = _format_body_text(event)
    res  = send_email_smtp(settings, recipients, subj, txt, None)

    if res.get("ok"):
        _remember_sent(state, key)
        _log(f"OK sent {key} -> {recipients}")
    else:
        _log(f"ERR send {key}: {res.get('error')}")
    return res
