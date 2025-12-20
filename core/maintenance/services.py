# core/maintenance/services.py
from __future__ import annotations

from datetime import date, timedelta, datetime
from typing import Any, Dict, List, Optional

from .models import get_conn


# ============================================================
# Utils
# ============================================================

def _rows_to_dicts(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _iso_date(d: Any) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        s = d.strip()
        if not s:
            return None
        try:
            date.fromisoformat(s)
            return s
        except Exception:
            return None
    return None


def _parse_iso_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date):
        return s
    if isinstance(s, str):
        st = s.strip()
        if not st:
            return None
        try:
            return date.fromisoformat(st)
        except Exception:
            return None
    return None


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# 1) Tâches PM (préventif)
# ============================================================

def upsert_task(task: Dict[str, Any]) -> int:
    """
    Upsert:
    - si task["id"] présent => update par id
    - sinon => upsert par clé fonctionnelle (equipment_code + title)
    Champs attendus :
      equipment_code, title, periodicity_days, next_due_date, last_done_date, status
    + (optionnel) maintenance_type, interval_h, source_hash
    """
    equipment_code = str(task.get("equipment_code", "")).strip()
    title = str(task.get("title", "")).strip()
    if not equipment_code or not title:
        raise ValueError("equipment_code et title sont requis.")

    periodicity_days = _safe_int(task.get("periodicity_days", 0), 0)
    next_due_date = _iso_date(task.get("next_due_date"))
    last_done_date = _iso_date(task.get("last_done_date"))
    status = str(task.get("status", "ACTIVE") or "ACTIVE").strip().upper()

    maintenance_type = (task.get("maintenance_type") or None)
    if maintenance_type is not None:
        maintenance_type = str(maintenance_type).strip() or None

    interval_h = task.get("interval_h")
    interval_h = _safe_float(interval_h, 0.0) if interval_h is not None else None
    if interval_h is not None and interval_h <= 0:
        interval_h = None

    source_hash = (task.get("source_hash") or None)
    if source_hash is not None:
        source_hash = str(source_hash).strip() or None

    conn = get_conn()
    with conn:
        if task.get("id"):
            tid = int(task["id"])
            conn.execute(
                """
                UPDATE pm_task
                   SET equipment_code=?,
                       title=?,
                       periodicity_days=?,
                       next_due_date=?,
                       last_done_date=?,
                       status=?,
                       maintenance_type=?,
                       interval_h=?,
                       source_hash=?,
                       updated_at=?
                 WHERE id=?
                """,
                (
                    equipment_code,
                    title,
                    periodicity_days,
                    next_due_date,
                    last_done_date,
                    status,
                    maintenance_type,
                    interval_h,
                    source_hash,
                    _now_iso(),
                    tid,
                ),
            )
            return tid

        # upsert par clé (equipment_code, title)
        cur = conn.cursor()
        cur.execute("SELECT id FROM pm_task WHERE equipment_code=? AND title=?", (equipment_code, title))
        row = cur.fetchone()

        if row:
            tid = int(row["id"])
            conn.execute(
                """
                UPDATE pm_task
                   SET periodicity_days=?,
                       next_due_date=?,
                       last_done_date=COALESCE(?, last_done_date),
                       status=?,
                       maintenance_type=?,
                       interval_h=?,
                       source_hash=?,
                       updated_at=?
                 WHERE id=?
                """,
                (
                    periodicity_days,
                    next_due_date,
                    last_done_date,
                    status,
                    maintenance_type,
                    interval_h,
                    source_hash,
                    _now_iso(),
                    tid,
                ),
            )
            return tid

        cur.execute(
            """
            INSERT INTO pm_task(
                equipment_code, title, periodicity_days, next_due_date,
                last_done_date, status, maintenance_type, interval_h, source_hash, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                equipment_code,
                title,
                periodicity_days,
                next_due_date,
                last_done_date,
                status,
                maintenance_type,
                interval_h,
                source_hash,
                _now_iso(),
            ),
        )
        cur.execute("SELECT last_insert_rowid() AS id")
        return int(cur.fetchone()["id"])


def upsert_task_by_key(
    equipment_code: str,
    title: str,
    periodicity_days: int,
    next_due_date: str,
    status: str = "ACTIVE",
    maintenance_type: Optional[str] = None,
    interval_h: Optional[float] = None,
    source_hash: Optional[str] = None,
) -> int:
    return upsert_task({
        "equipment_code": equipment_code,
        "title": title,
        "periodicity_days": int(periodicity_days),
        "next_due_date": next_due_date,
        "status": status,
        "maintenance_type": maintenance_type,
        "interval_h": interval_h,
        "source_hash": source_hash,
    })


def list_tasks() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, equipment_code, title, periodicity_days,
               next_due_date, last_done_date, status,
               maintenance_type, interval_h, source_hash, updated_at
          FROM pm_task
         ORDER BY CASE WHEN next_due_date IS NULL THEN 1 ELSE 0 END,
                  next_due_date
        """
    )
    return _rows_to_dicts(cur)


def due_within(days: int = 7, include_overdue: bool = True) -> List[Dict[str, Any]]:
    days = int(days or 0)
    if days < 0:
        days = 0

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, equipment_code, title, periodicity_days, next_due_date,
               last_done_date, status, maintenance_type, interval_h, source_hash, updated_at
          FROM pm_task
         WHERE status='ACTIVE'
           AND next_due_date IS NOT NULL
         ORDER BY next_due_date
        """
    )
    rows = _rows_to_dicts(cur)

    today = date.today()
    out: List[Dict[str, Any]] = []

    for r in rows:
        nd = _parse_iso_date(r.get("next_due_date"))
        if not nd:
            continue
        delta = (nd - today).days

        if include_overdue:
            if delta <= days:
                out.append({**r, "days_left": delta})
        else:
            if 0 <= delta <= days:
                out.append({**r, "days_left": delta})

    return out


def mark_done(task_id: int) -> None:
    tid = int(task_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT periodicity_days FROM pm_task WHERE id=?", (tid,))
    row = cur.fetchone()
    if not row:
        return

    per = int(row["periodicity_days"] or 0)
    today_iso = date.today().isoformat()
    nextd = (date.today() + timedelta(days=per)).isoformat() if per > 0 else None

    with conn:
        conn.execute(
            "UPDATE pm_task SET last_done_date=?, next_due_date=?, updated_at=? WHERE id=?",
            (today_iso, nextd, _now_iso(), tid),
        )


# ============================================================
# 2) Templates PM
# ============================================================

DEFAULT_TEMPLATES: List[Dict[str, Any]] = [
    {"title": "Inspection visuelle", "group_name": "Préventif", "periodicity_days": 30},
    {"title": "Contrôle accessoires (refroidissement/protections)", "group_name": "Préventif", "periodicity_days": 180},
    {"title": "Analyse huile isolante (eau/gaz/acide)", "group_name": "Préventif", "periodicity_days": 180},
    {"title": "Test d’isolement (mégohmmètre)", "group_name": "Préventif", "periodicity_days": 365},
    {"title": "Test rapport de transformation", "group_name": "Préventif", "periodicity_days": 365},
    {"title": "Surveillance température (continu)", "group_name": "Prédictif", "periodicity_days": 0},
    {"title": "Surveillance vibrations/humidité (capteurs)", "group_name": "Prédictif", "periodicity_days": 0},
]


def list_templates() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, group_name, periodicity_days, default_enabled
          FROM pm_template
         ORDER BY group_name, title
        """
    )
    rows = _rows_to_dicts(cur)

    # seed si table vide
    if not rows:
        with conn:
            for t in DEFAULT_TEMPLATES:
                conn.execute(
                    """
                    INSERT INTO pm_template(title, group_name, periodicity_days, default_enabled)
                    VALUES(?,?,?,1)
                    """,
                    (t["title"], t["group_name"], int(t["periodicity_days"])),
                )
        return list_templates()

    return rows


def upsert_template(tpl: Dict[str, Any]) -> None:
    title = str(tpl.get("title", "")).strip()
    group_name = str(tpl.get("group_name", "")).strip() or "Préventif"
    periodicity_days = _safe_int(tpl.get("periodicity_days", 0), 0)
    default_enabled = _safe_int(tpl.get("default_enabled", 1), 1)

    if not title:
        raise ValueError("title requis pour template")

    conn = get_conn()
    with conn:
        if tpl.get("id"):
            conn.execute(
                """
                UPDATE pm_template
                   SET title=?,
                       group_name=?,
                       periodicity_days=?,
                       default_enabled=?
                 WHERE id=?
                """,
                (title, group_name, periodicity_days, default_enabled, int(tpl["id"])),
            )
        else:
            conn.execute(
                """
                INSERT INTO pm_template(title, group_name, periodicity_days, default_enabled)
                VALUES(?,?,?,?)
                """,
                (title, group_name, periodicity_days, default_enabled),
            )


def sync_tasks_from_templates(equipment_code: str, start_date: Optional[str] = None) -> None:
    """
    Crée les tâches PM à partir des templates (si elles n’existent pas déjà) pour un équipement.
    - Ne crée que les templates periodicity_days > 0
    - Démarre à start_date (ISO) sinon aujourd’hui
    """
    eq = str(equipment_code).strip()
    if not eq:
        raise ValueError("equipment_code requis")

    base_dt = _parse_iso_date(start_date) or date.today()
    tpls = list_templates()

    for t in tpls:
        per = int(t.get("periodicity_days", 0) or 0)
        if per <= 0:
            continue

        title = str(t.get("title", "")).strip()
        if not title:
            continue

        next_due = (base_dt + timedelta(days=per)).isoformat()

        # upsert par clé, pas de doublons
        upsert_task_by_key(
            equipment_code=eq,
            title=title,
            periodicity_days=per,
            next_due_date=next_due,
            status="ACTIVE",
        )


# ============================================================
# 3) Seuils & Résultats (CBM / contrôles)
# ============================================================

DEFAULT_THRESHOLDS = [
    ("resistance_isolement", "MΩ", ">=", 100, None, "> 100 MΩ entre enroulements et masse"),
    ("rapport_transformation", "%", "range", -0.5, 0.5, "± 0.5 % de la valeur nominale"),
    ("tension_claquage_huile", "kV", ">=", 60, None, "> 60 kV"),
    ("teneur_eau_huile", "ppm", "<=", 30, None, "< 30 ppm"),
    ("gaz_h2", "ppm", "<=", 150, None, "H₂ < 150 ppm"),
    ("gaz_ch4", "ppm", "<=", 100, None, "CH₄ < 100 ppm"),
    ("gaz_c2h2", "ppm", "<=", 35, None, "C₂H₂ < 35 ppm"),
    ("temperature", "°C", "<=", 85, None, "< 65 °C normal, > 85 °C alarme"),
    ("resistance_enroulement_delta", "%", "<=", 2, None, "Variation entre phases < 2 %"),
    ("furanes", "ppm", "<=", 0.1, None, "< 0.1 ppm (normal)"),
    ("indice_neutralisation", "mg KOH/g", "<=", 0.3, None, "< 0.3 mg KOH/g"),
]


def list_thresholds() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, test_key, unit, rule, min_val, max_val, ref_text FROM pm_threshold")
    rows = _rows_to_dicts(cur)

    if not rows:
        with conn:
            for t in DEFAULT_THRESHOLDS:
                conn.execute(
                    """
                    INSERT INTO pm_threshold(test_key, unit, rule, min_val, max_val, ref_text)
                    VALUES(?,?,?,?,?,?)
                    """,
                    t,
                )
        return list_thresholds()

    return rows


def upsert_threshold(th: Dict[str, Any]) -> None:
    test_key = str(th.get("test_key", "")).strip()
    if not test_key:
        raise ValueError("test_key requis")

    unit = str(th.get("unit", "") or "")
    rule = str(th.get("rule", ">=") or ">=").strip()
    min_val = th.get("min_val")
    max_val = th.get("max_val")
    ref_text = str(th.get("ref_text", "") or "")

    conn = get_conn()
    with conn:
        conn.execute(
            """
            INSERT INTO pm_threshold(test_key, unit, rule, min_val, max_val, ref_text)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(test_key) DO UPDATE SET
                unit=excluded.unit,
                rule=excluded.rule,
                min_val=excluded.min_val,
                max_val=excluded.max_val,
                ref_text=excluded.ref_text
            """,
            (test_key, unit, rule, min_val, max_val, ref_text),
        )


def evaluate(measured: float, rule: str, min_val: Optional[float], max_val: Optional[float]) -> str:
    try:
        x = float(measured)
    except Exception:
        return "NOK"

    if rule == ">=":
        return "OK" if (min_val is not None and x >= float(min_val)) else "NOK"
    if rule == "<=":
        return "OK" if (min_val is not None and x <= float(min_val)) else "NOK"
    if rule == "range":
        return "OK" if (min_val is not None and max_val is not None and float(min_val) <= x <= float(max_val)) else "NOK"
    return "NOK"


def record_result(
    equipment_code: str,
    test_key: str,
    measured: float,
    operator: str = "",
    comment: str = "",
) -> Dict[str, Any]:
    eq = str(equipment_code).strip()
    tk = str(test_key).strip()
    if not eq or not tk:
        raise ValueError("equipment_code et test_key sont requis")

    ths = {t["test_key"]: t for t in list_thresholds()}
    th = ths.get(tk)
    if not th:
        raise ValueError("test_key inconnu. Configure-le dans les seuils.")

    status = evaluate(measured, th["rule"], th["min_val"], th["max_val"])

    conn = get_conn()
    with conn:
        conn.execute(
            """
            INSERT INTO pm_result(date, equipment_code, test_key, measured, unit, status, comment, operator)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                date.today().isoformat(),
                eq,
                tk,
                float(measured),
                th.get("unit", ""),
                status,
                str(comment or ""),
                str(operator or ""),
            ),
        )

    return {
        "date": date.today().isoformat(),
        "equipment_code": eq,
        "test_key": tk,
        "measured": float(measured),
        "unit": th.get("unit", ""),
        "status": status,
        "comment": str(comment or ""),
        "operator": str(operator or ""),
    }


def list_results(equipment_code: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()

    lim = int(limit or 200)
    if lim <= 0:
        lim = 200

    if equipment_code:
        eq = str(equipment_code).strip()
        cur.execute(
            """
            SELECT id, date, equipment_code, test_key, measured, unit, status, comment, operator
              FROM pm_result
             WHERE equipment_code=?
             ORDER BY date DESC, id DESC
             LIMIT ?
            """,
            (eq, lim),
        )
    else:
        cur.execute(
            """
            SELECT id, date, equipment_code, test_key, measured, unit, status, comment, operator
              FROM pm_result
             ORDER BY date DESC, id DESC
             LIMIT ?
            """,
            (lim,),
        )

    return _rows_to_dicts(cur)


# ============================================================
# 4) KPI
# ============================================================

def kpi_maintenance() -> Dict[str, int]:
    tasks = list_tasks()
    today = date.today()
    total = len(tasks)
    overdue = 0
    next7 = 0

    for t in tasks:
        nd = _parse_iso_date(t.get("next_due_date"))
        if not nd:
            continue
        delta = (nd - today).days
        if delta < 0:
            overdue += 1
        elif delta <= 7:
            next7 += 1

    return {"tasks_total": total, "overdue": overdue, "due_7d": next7}


def bulk_sync_templates(equipment_codes: List[str]) -> int:
    total = 0
    for eq in equipment_codes or []:
        try:
            sync_tasks_from_templates(equipment_code=str(eq))
            total += 1
        except Exception:
            pass
    return total
