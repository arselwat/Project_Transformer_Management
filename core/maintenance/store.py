from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import time, csv, os

DATA_DIR = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
TASKS_CSV = DATA_DIR / "maintenance_tasks.csv"

@dataclass
class Task:
    id: str
    created_ts: float
    transformer_code: str
    severity: str      # LOW / MEDIUM / HIGH / CRITICAL
    title: str
    description: str
    rule: str
    status: str        # OPEN / ACK / IN_PROGRESS / DONE
    due_date: str = ""
    spare_suggestion: str = ""
    notes: str = ""

def _next_id() -> str:
    ts = int(time.time())
    return f"TASK-{ts}"

def _ensure_header(path: Path, header: list[str]):
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

def create_task(t: Task) -> Tuple[bool, str]:
    header = [f.name for f in Task.__dataclass_fields__.values()]
    _ensure_header(TASKS_CSV, header)
    try:
        with open(TASKS_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow(asdict(t))
        return True, t.id
    except Exception as e:
        return False, str(e)

def new_task(transformer_code: str, severity: str, title: str, description: str, rule: str,
             due_date: str = "", spare_suggestion: str = "") -> Tuple[bool, str]:
    t = Task(
        id=_next_id(), created_ts=time.time(), transformer_code=transformer_code,
        severity=severity, title=title, description=description, rule=rule,
        status="OPEN", due_date=due_date, spare_suggestion=spare_suggestion, notes=""
    )
    return create_task(t)

def list_tasks() -> List[dict]:
    if not TASKS_CSV.exists():
        return []
    rows = []
    with open(TASKS_CSV, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def update_task_status(task_id: str, status: str) -> Tuple[bool, str]:
    rows = list_tasks()
    if not rows:
        return False, "Aucune tâche"
    found = False
    for r in rows:
        if r.get("id") == task_id:
            r["status"] = status
            found = True
            break
    if not found:
        return False, "ID introuvable"
    header = rows[0].keys()
    with open(TASKS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    return True, "OK"
