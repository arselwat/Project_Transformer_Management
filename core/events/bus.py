# core/events/bus.py (extrait)
from core.notify.rt_alerts import notify_event  # <-- ajoute l'import

def emit_event(equipment: str, event: dict, site: str | None = None, extra: dict | None = None):
    """
    event attendu: {ts, level, code, msg, value?, threshold?}
    Ici tu journalises -> stock -> maintenance (si tu veux), puis :
    """
    rec = {
        "ts": event.get("ts"),
        "equipment": equipment,
        "site": site or "",
        "level": event.get("level","INFO"),
        "code": event.get("code",""),
        "msg": event.get("msg",""),
        "value": event.get("value",""),
        "threshold": event.get("threshold","")
    }
    # … écriture CSV/DB (si existant) …
    try:
        notify_event(rec)  # <-- déclenche l’e-mail (cooldown + filtres + prefix)
    except Exception:
        pass
