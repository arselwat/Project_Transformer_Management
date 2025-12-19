# core/maintenance/services.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from .models import get_conn


# ============================================================
# Utils
# ============================================================

def _rows_to_dicts(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _iso_date(d: Any) -> Optional[str]:
    """
    Convertit vers YYYY-MM-DD si possible.
    Accepte date / str ISO.
    """
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, str):
        s = d.strip()
        if not s:
            return None
        try:
            # valide ISO
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


# ============================================================
# 1) Tâches PM (préventif)
# ============================================================

def upsert_task(task: Dict[str, Any]) -> None:
    """
    Upsert simple :
    - si task["id"] présent => update
    - sinon insert
    Champs attendus :
      equipment_code, title, periodicity_days, next_due_date, last_done_date, status
    """
    equipment_code = str(task.get("equipment_code", "")).strip()
    title = str(task.get("title", "")).strip()
    if not equipment_code or not title:
        raise ValueError("equipment_code et title sont requis.")

    periodicity_days = int(task.get("periodicity_days", 0) or 0)
    next_due_date = _iso_date(task.get("next_due_date"))
    last_done_date = _iso_date(task.get("last_done_date"))
    status = str(task.get("status", "ACTIVE") or "ACTIVE").strip().upper()

    conn = get_conn()
    with conn:
        if task.get("id"):
            conn.execute(
                """
                UPDATE pm_task
                   SET equipment_code=?,
                       title=?,
                       periodicity_days=?,
                       next_due_date=?,
                       last_done_date=?,
                       status=?
                 WHERE id=?
                """,
                (
                    equipment_code,
                    title,
                    periodicity_days,
                    next_due_date,
                    last_done_date,
                    status,
                    int(task["id"]),
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO pm_task(equipment_code, title, periodicity_days, next_due_date, last_done_date, status)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    equipment_code,
                    title,
                    periodicity_days,
                    next_due_date,
                    last_done_date,
                    status,
                ),
            )


def list_tasks() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, equipment_code, title, periodicity_days, next_due_date, last_done_date, status
          FROM pm_task
         ORDER BY CASE WHEN next_due_date IS NULL THEN 1 ELSE 0 END,
                  next_due_date
        """
    )
    return _rows_to_dicts(cur)


def due_within(days: int = 7, include_overdue: bool = True) -> List[Dict[str, Any]]:
    """
    Renvoie les tâches ACTIVE dont next_due_date est:
      - dans les prochains 'days' jours
      - et (optionnel) inclut aussi celles en retard (delta < 0) si include_overdue=True
    """
    days = int(days or 0)
    if days < 0:
        days = 0

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, equipment_code, title, periodicity_days, next_due_date, last_done_date, status
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
    """
    Marque la tâche comme faite aujourd'hui:
    - last_done_date = today
    - next_due_date = today + periodicity_days (si periodicity_days > 0) sinon NULL
    """
    tid = int(task_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT periodicity_days FROM pm_task WHERE id=?", (tid,))
    row = cur.fetchone()
    if not row:
        return

    per = int(row[0] or 0)
    today_iso = date.today().isoformat()
    nextd = (date.today() + timedelta(days=per)).isoformat() if per > 0 else None

    with conn:
        conn.execute(
            "UPDATE pm_task SET last_done_date=?, next_due_date=? WHERE id=?",
            (today_iso, nextd, tid),
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
    periodicity_days = int(tpl.get("periodicity_days", 0) or 0)
    default_enabled = int(tpl.get("default_enabled", 1) or 1)

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

    conn = get_conn()
    cur = conn.cursor()

    for t in tpls:
        per = int(t.get("periodicity_days", 0) or 0)
        if per <= 0:
            continue

        title = str(t.get("title", "")).strip()
        if not title:
            continue

        # existe déjà ?
        cur.execute(
            "SELECT 1 FROM pm_task WHERE equipment_code=? AND title=?",
            (eq, title),
        )
        if cur.fetchone():
            continue

        next_due = (base_dt + timedelta(days=per)).isoformat()
        upsert_task({
            "equipment_code": eq,
            "title": title,
            "periodicity_days": per,
            "next_due_date": next_due,
            "status": "ACTIVE",
        })


# ============================================================
# 3) Seuils & Résultats (CBM / contrôles)
# ============================================================

DEFAULT_THRESHOLDS = [
    # test_key, unit, rule, min, max, ref_text
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
    """
    Nécessite que pm_threshold.test_key soit UNIQUE pour ON CONFLICT(test_key).
    """
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
    """
    Enregistre une mesure dans pm_result, calcule OK/NOK selon pm_threshold.
    """
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
# 4) KPI / Dashboard
# ============================================================

def kpi_maintenance() -> Dict[str, int]:
    """
    KPI simples :
      - total tasks
      - overdue
      - due next 7 days
    """
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


def list_tasks_due(within_days: int = 7, include_overdue: bool = True) -> List[Dict[str, Any]]:
    """
    Alternative SQL-friendly : renvoie ACTIVE et next_due_date dans fenêtre.
    """
    days = int(within_days or 0)
    if days < 0:
        days = 0

    # SQLite date('now', '+7 day') marche si next_due_date est ISO YYYY-MM-DD
    conn = get_conn()
    cur = conn.cursor()

    if include_overdue:
        cur.execute(
            """
            SELECT id, equipment_code, title, next_due_date, priority, sla_hours, status, procedure_id
              FROM pm_task
             WHERE status='ACTIVE'
               AND next_due_date IS NOT NULL
               AND date(next_due_date) <= date('now', ?)
             ORDER BY next_due_date
            """,
            (f"+{days} day",),
        )
    else:
        cur.execute(
            """
            SELECT id, equipment_code, title, next_due_date, priority, sla_hours, status, procedure_id
              FROM pm_task
             WHERE status='ACTIVE'
               AND next_due_date IS NOT NULL
               AND date(next_due_date) BETWEEN date('now') AND date('now', ?)
             ORDER BY next_due_date
            """,
            (f"+{days} day",),
        )

    return _rows_to_dicts(cur)


def list_tasks_sla_over() -> List[Dict[str, Any]]:
    """
    Placeholder simplifié: retourne les tâches avec sla_hours > 0.
    (Tu peux raffiner si tu stockes des timestamps exacts.)
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, equipment_code, title, sla_hours, status
          FROM pm_task
         WHERE status='ACTIVE' AND IFNULL(sla_hours,0) > 0
        """
    )
    out = _rows_to_dicts(cur)
    return [t for t in out if float(t.get("sla_hours") or 0) > 0]


def bulk_sync_templates(equipment_codes: List[str]) -> int:
    """
    Crée/MAJ les tâches pour chaque equipment_code à partir des templates.
    Retourne le nombre d'équipements traités.
    """
    total = 0
    for eq in equipment_codes or []:
        try:
            sync_tasks_from_templates(equipment_code=str(eq))
            total += 1
        except Exception:
            pass
    return total
def upsert_task_by_key(equipment_code: str, title: str, periodicity_days: int, next_due_date: str, status: str = "ACTIVE"):
    """
    Upsert stable par clé fonctionnelle (equipment_code + title).
    Évite les doublons quand l'optimisation est relancée.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pm_task WHERE equipment_code=? AND title=?", (equipment_code, title))
    row = cur.fetchone()

    if row:
        task_id = int(row[0])
        with conn:
            conn.execute(
                """UPDATE pm_task
                   SET periodicity_days=?, next_due_date=?, status=?
                   WHERE id=?""",
                (int(periodicity_days), next_due_date, status, task_id),
            )
        return task_id

    with conn:
        conn.execute(
            """INSERT INTO pm_task(equipment_code,title,periodicity_days,next_due_date,last_done_date,status)
               VALUES(?,?,?,?,?,?)""",
            (equipment_code, title, int(periodicity_days), next_due_date, None, status),
        )
    # récupérer id
    cur = conn.cursor()
    cur.execute("SELECT last_insert_rowid()")
    return int(cur.fetchone()[0])
