from .models import get_conn
from datetime import date, timedelta, datetime
from typing import List, Dict, Optional

# -------- Tâches (préventif) --------
def upsert_task(task: Dict):
    conn = get_conn()
    with conn:
        if task.get("id"):
            conn.execute("""UPDATE pm_task
                            SET equipment_code=?, title=?, periodicity_days=?, next_due_date=?, last_done_date=?, status=?
                            WHERE id=?""",
                         (task["equipment_code"], task["title"], int(task.get("periodicity_days",0)),
                          task.get("next_due_date"), task.get("last_done_date"), task.get("status","ACTIVE"),
                          int(task["id"])))
        else:
            conn.execute("""INSERT INTO pm_task(equipment_code,title,periodicity_days,next_due_date,last_done_date,status)
                            VALUES(?,?,?,?,?,?)""",
                         (task["equipment_code"], task["title"], int(task.get("periodicity_days",0)),
                          task.get("next_due_date"), task.get("last_done_date"), task.get("status","ACTIVE")))

def list_tasks() -> List[Dict]:
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,equipment_code,title,periodicity_days,next_due_date,last_done_date,status
                   FROM pm_task ORDER BY next_due_date""")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def due_within(days: int = 7) -> List[Dict]:
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,equipment_code,title,periodicity_days,next_due_date,last_done_date,status
                   FROM pm_task WHERE status='ACTIVE' AND next_due_date IS NOT NULL
                   ORDER BY next_due_date""")
    cols = [d[0] for d in cur.description]
    res = []
    today = date.today()
    for r in [dict(zip(cols,row)) for row in cur.fetchall()]:
        try:
            y,m,d = [int(x) for x in r["next_due_date"].split("-")]
            delta = (date(y,m,d) - today).days
            if delta <= days:
                res.append({**r, "days_left": delta})
        except Exception:
            pass
    return res

def mark_done(task_id: int):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT periodicity_days FROM pm_task WHERE id=?", (task_id,))
    row = cur.fetchone()
    if not row: return
    per = int(row[0] or 0)
    today = date.today().isoformat()
    nextd = (date.today() + timedelta(days=per)).isoformat() if per > 0 else None
    with conn:
        conn.execute("UPDATE pm_task SET last_done_date=?, next_due_date=? WHERE id=?", (today, nextd, task_id))

# -------- Templates --------
DEFAULT_TEMPLATES = [
    {"title":"Inspection visuelle","group_name":"Préventif","periodicity_days":30},
    {"title":"Contrôle accessoires (refroidissement/protections)","group_name":"Préventif","periodicity_days":180},
    {"title":"Analyse huile isolante (eau/gaz/acide)","group_name":"Préventif","periodicity_days":180},
    {"title":"Test d’isolement (mégohmmètre)","group_name":"Préventif","periodicity_days":365},
    {"title":"Test rapport de transformation","group_name":"Préventif","periodicity_days":365},
    {"title":"Surveillance température (continu)","group_name":"Prédictif","periodicity_days":0},
    {"title":"Surveillance vibrations/humidité (capteurs)","group_name":"Prédictif","periodicity_days":0},
]

def list_templates() -> List[Dict]:
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,title,group_name,periodicity_days,default_enabled
                   FROM pm_template ORDER BY group_name, title""")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols,row)) for row in cur.fetchall()]
    if not rows:
        # seed
        with conn:
            for t in DEFAULT_TEMPLATES:
                conn.execute("""INSERT INTO pm_template(title,group_name,periodicity_days,default_enabled)
                                VALUES(?,?,?,1)""",
                             (t["title"], t["group_name"], t["periodicity_days"]))
        return list_templates()
    return rows

def upsert_template(tpl: Dict):
    conn = get_conn()
    with conn:
        if tpl.get("id"):
            conn.execute("""UPDATE pm_template SET title=?, group_name=?, periodicity_days=?, default_enabled=?
                            WHERE id=?""",
                         (tpl["title"], tpl["group_name"], int(tpl["periodicity_days"]),
                          int(tpl.get("default_enabled",1)), int(tpl["id"])))
        else:
            conn.execute("""INSERT INTO pm_template(title,group_name,periodicity_days,default_enabled)
                            VALUES(?,?,?,?)""",
                         (tpl["title"], tpl["group_name"], int(tpl["periodicity_days"]),
                          int(tpl.get("default_enabled",1))))

def sync_tasks_from_templates(equipment_code: str, start_date: Optional[str] = None):
    """Crée les tâches PM à partir des templates (si elles n’existent pas déjà) pour un équipement."""
    conn = get_conn(); tpls = list_templates()
    today = date.today().isoformat()
    base = start_date or today
    cur = conn.cursor()
    for t in tpls:
        if int(t["periodicity_days"]) <= 0:
            continue
        # vérifier existence
        cur.execute("""SELECT 1 FROM pm_task WHERE equipment_code=? AND title=?""",
                    (equipment_code, t["title"]))
        if cur.fetchone():
            continue
        next_due = (date.fromisoformat(base) + timedelta(days=int(t["periodicity_days"]))).isoformat()
        upsert_task({
            "equipment_code": equipment_code,
            "title": t["title"],
            "periodicity_days": int(t["periodicity_days"]),
            "next_due_date": next_due,
            "status": "ACTIVE"
        })

# -------- Seuils & Résultats --------
DEFAULT_THRESHOLDS = [
    # test_key, unit, rule, min, max, ref_text
    ("resistance_isolement","MΩ",">=",100,None,"> 100 MΩ entre enroulements et masse"),
    ("rapport_transformation","%", "range", -0.5, 0.5, "± 0.5 % de la valeur nominale"),
    ("tension_claquage_huile","kV",">=",60,None,"> 60 kV"),
    ("teneur_eau_huile","ppm","<=",30,None,"< 30 ppm"),
    ("gaz_h2","ppm","<=",150,None,"H₂ < 150 ppm"),
    ("gaz_ch4","ppm","<=",100,None,"CH₄ < 100 ppm"),
    ("gaz_c2h2","ppm","<=",35,None,"C₂H₂ < 35 ppm"),
    ("temperature","°C","<=",85,None,"< 65 °C normal, > 85 °C alarme"),
    ("resistance_enroulement_delta","%", "<=", 2, None, "Variation entre phases < 2 %"),
    ("furanes","ppm","<=",0.1,None,"< 0.1 ppm (normal)"),
    ("indice_neutralisation","mg KOH/g","<=",0.3,None,"< 0.3 mg KOH/g"),
]

def list_thresholds() -> List[Dict]:
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,test_key,unit,rule,min_val,max_val,ref_text FROM pm_threshold""")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols,row)) for row in cur.fetchall()]
    if not rows:
        with conn:
            for t in DEFAULT_THRESHOLDS:
                conn.execute("""INSERT INTO pm_threshold(test_key,unit,rule,min_val,max_val,ref_text)
                                VALUES(?,?,?,?,?,?)""", t)
        return list_thresholds()
    return rows

def upsert_threshold(th: Dict):
    conn = get_conn()
    with conn:
        conn.execute("""INSERT INTO pm_threshold(test_key,unit,rule,min_val,max_val,ref_text)
                        VALUES(?,?,?,?,?,?)
                        ON CONFLICT(test_key) DO UPDATE SET
                          unit=excluded.unit,
                          rule=excluded.rule,
                          min_val=excluded.min_val,
                          max_val=excluded.max_val,
                          ref_text=excluded.ref_text
                     """,
                     (th["test_key"], th.get("unit",""), th.get("rule",">="),
                      th.get("min_val"), th.get("max_val"), th.get("ref_text","")))

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

def record_result(equipment_code: str, test_key: str, measured: float, operator: str = "", comment: str = "") -> Dict:
    """Enregistre une mesure, calcule OK/NOK selon pm_threshold, retourne le dict enregistré."""
    # récupérer seuils
    ths = {t["test_key"]: t for t in list_thresholds()}
    th = ths.get(test_key)
    if not th:
        raise ValueError("test_key inconnu. Configurez-le dans les seuils.")
    status = evaluate(measured, th["rule"], th["min_val"], th["max_val"])
    conn = get_conn()
    with conn:
        conn.execute("""INSERT INTO pm_result(date,equipment_code,test_key,measured,unit,status,comment,operator)
                        VALUES(?,?,?,?,?,?,?,?)""",
                     (date.today().isoformat(), equipment_code, test_key, float(measured),
                      th.get("unit",""), status, comment, operator))
    return {
        "date": date.today().isoformat(),
        "equipment_code": equipment_code, "test_key": test_key,
        "measured": measured, "unit": th.get("unit",""),
        "status": status, "comment": comment, "operator": operator
    }

def list_results(equipment_code: Optional[str] = None, limit: int = 200) -> List[Dict]:
    conn = get_conn(); cur = conn.cursor()
    if equipment_code:
        cur.execute("""SELECT id,date,equipment_code,test_key,measured,unit,status,comment,operator
                       FROM pm_result WHERE equipment_code=? ORDER BY date DESC, id DESC LIMIT ?""",
                    (equipment_code, limit))
    else:
        cur.execute("""SELECT id,date,equipment_code,test_key,measured,unit,status,comment,operator
                       FROM pm_result ORDER BY date DESC, id DESC LIMIT ?""",
                    (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

# -------- KPI / Dashboard --------
def kpi_maintenance() -> Dict:
    """Renvoie quelques KPI simples."""
    tasks = list_tasks()
    today = date.today()
    total = len(tasks)
    overdue = 0
    next7 = 0
    for t in tasks:
        nd = t.get("next_due_date")
        if not nd: continue
        y,m,d = [int(x) for x in nd.split("-")]
        delta = (date(y,m,d) - today).days
        if delta < 0:
            overdue += 1
        elif delta <= 7:
            next7 += 1
    return {"tasks_total": total, "overdue": overdue, "due_7d": next7}
def list_tasks_due(within_days: int = 7):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,equipment_code,title,next_due_date,priority,sla_hours,status,procedure_id
                   FROM pm_task
                   WHERE status='ACTIVE' AND date(next_due_date) <= date('now', ?)
                   ORDER BY next_due_date""", (f"+{within_days} day",))
    cols=[d[0] for d in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def list_tasks_sla_over():
    # ici, on considère 'over SLA' si ACTIVE et (now - last_done) > sla_hours (simpli.)
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""SELECT id,equipment_code,title,sla_hours,status FROM pm_task
                   WHERE status='ACTIVE' AND IFNULL(sla_hours,0) > 0""")
    cols=[d[0] for d in cur.description]
    out = [dict(zip(cols,row)) for row in cur.fetchall()]
    return [t for t in out if t.get("sla_hours",0) > 0]  # simplifié, adapte si tu as timestamps exacts

def bulk_sync_templates(equipment_codes: list[str]) -> int:
    """
    Crée/MAJ les tâches pour chaque equipment_code à partir des templates actifs.
    Retourne le nombre d'équipements traités.
    """
    total = 0
    for eq in equipment_codes:
        try:
            sync_tasks_from_templates(equipment_code=eq)
            total += 1
        except Exception:
            # on n'arrête pas la boucle si un code pose problème
            pass
    return total
