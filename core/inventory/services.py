# core/inventory/services.py
from __future__ import annotations

from pathlib import Path
import os, csv, time, argparse
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd  # pyright: ignore[reportMissingModuleSource]

# ========= Emplacements =========
DATA_DIR = Path(os.environ.get("FS_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARTS_CSV = (DATA_DIR / "inventory_parts.csv").resolve()
MOVES_CSV = (DATA_DIR / "stock_movements.csv").resolve()  # ✅ même nom partout
BOM_CSV   = (DATA_DIR / "bom_tasks.csv").resolve()

# Schéma canonique des pièces (colonnes du CSV d’inventaire)
CANON = ["code","nom","famille","quantite_dispo","seuil_min","localisation","prix_unitaire","fournisseur"]

# Schéma canonique du journal des mouvements
MOVES_HEADER = ["ts","type","code","qty","reason","task_id","ref","user"]

# ========= Helpers I/O =========
def _ensure_csv(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()

def _flush_csv(path: Path, rows: List[Dict], header: List[str]) -> None:
    _ensure_csv(path, header)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
        f.flush()
        os.fsync(f.fileno())

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")

# ========= Normalisation inventaire =========
def normalize_parts_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CANON)

    d = df.copy()

    aliases = {
        "name":"nom","family":"famille","qty":"quantite_dispo","quantite":"quantite_dispo",
        "unit_price":"prix_unitaire","price":"prix_unitaire","location":"localisation"
    }
    for src, dst in aliases.items():
        if src in d.columns and dst not in d.columns:
            d[dst] = d[src]

    for col in CANON:
        if col not in d.columns:
            d[col] = ""

    d["code"] = d["code"].astype(str).str.strip()

    d["quantite_dispo"] = pd.to_numeric(d["quantite_dispo"], errors="coerce").fillna(0).astype(int)
    d["seuil_min"]      = pd.to_numeric(d["seuil_min"], errors="coerce").fillna(0).astype(int)
    d["prix_unitaire"]  = pd.to_numeric(d["prix_unitaire"], errors="coerce").fillna(0.0).astype(float)

    d = d[d["code"].astype(str).str.len() > 0]
    d = d[CANON].drop_duplicates(subset=["code"]).reset_index(drop=True)
    return d

# ========= Réduction de liste (filtrage “actifs”) =========
def _is_active_part_row(row: pd.Series) -> bool:
    """
    Définit si un article doit apparaître par défaut :
    - stock > 0 OU seuil_min > 0 OU sous seuil (low stock)
    """
    try:
        q = int(row.get("quantite_dispo") or 0)
        s = int(row.get("seuil_min") or 0)
        return (q > 0) or (s > 0) or (q <= s and s > 0)
    except Exception:
        return True

def list_parts_as_dicts(active_only: bool = False) -> List[Dict]:
    _ensure_csv(PARTS_CSV, CANON)
    df = _read_csv(PARTS_CSV)
    if df.empty:
        return []
    df = normalize_parts_df(df)
    if active_only:
        if not df.empty:
            df = df[df.apply(_is_active_part_row, axis=1)]
    return df.to_dict("records")

def list_parts(active_only: bool = False) -> List[Dict]:
    return list_parts_as_dicts(active_only=active_only)

def get_part(code: str) -> Dict | None:
    if not code:
        return None
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return None
    m = df[df["code"] == str(code).strip()]
    return (m.iloc[0].to_dict() if not m.empty else None)

def upsert_parts(rows: List[Dict]) -> int:
    """Ajoute ou met à jour des pièces (par code)."""
    _ensure_csv(PARTS_CSV, CANON)
    base = normalize_parts_df(_read_csv(PARTS_CSV))
    add  = normalize_parts_df(pd.DataFrame(rows))

    if base.empty:
        out = add
    else:
        out = pd.concat([base[~base["code"].isin(add["code"])], add], ignore_index=True)

    out.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    return len(add)

def delete_part(code: str) -> bool:
    _ensure_csv(PARTS_CSV, CANON)
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return False
    before = len(df)
    df = df[df["code"] != str(code).strip()]
    df.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    return len(df) < before

# ========= Mouvements bas-niveau (journal) =========
def _append_move(move: Dict) -> None:
    row = {
        "ts": time.time(),
        "type": move.get("type",""),
        "code": move.get("code",""),
        "qty":  move.get("qty",""),
        "reason": move.get("reason",""),
        "task_id": move.get("task_id",""),
        "ref": move.get("ref",""),
        "user": move.get("user",""),
    }
    _flush_csv(MOVES_CSV, [row], header=MOVES_HEADER)

def list_movements(limit: int = 500) -> List[Dict]:
    _ensure_csv(MOVES_CSV, MOVES_HEADER)
    df = _read_csv(MOVES_CSV)
    if df.empty:
        return []
    df = df.tail(int(limit))
    for c in ("ts","qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.to_dict("records")

# ========= Opérations de stock (compatibles pages/6_Stock.py) =========
def move_in(code: str, qty: float, reason: str = "IN", ref: str = "", user: str = "") -> Tuple[bool, str]:
    if not code:
        return False, "code requis"
    try:
        q = float(qty)
    except Exception:
        return False, "qty invalide"
    if q <= 0:
        return False, "qty doit être > 0"

    code = str(code).strip()

    _ensure_csv(PARTS_CSV, CANON)
    df = normalize_parts_df(_read_csv(PARTS_CSV))

    if df.empty or code not in df["code"].values:
        new_row = {
            "code": code, "nom": code, "famille": "",
            "quantite_dispo": int(q), "seuil_min": 0,
            "localisation":"", "prix_unitaire": 0.0, "fournisseur":""
        }
        df = pd.concat([df, normalize_parts_df(pd.DataFrame([new_row]))], ignore_index=True)
        stock = int(q)
    else:
        idx = df.index[df["code"] == code][0]
        stock = int(df.at[idx, "quantite_dispo"]) + int(q)
        df.at[idx, "quantite_dispo"] = stock

    df.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    _append_move({"type":"IN","code":code,"qty":int(q),"reason":reason,"task_id":"","ref":ref,"user":user})
    return True, f"+{int(q)} sur {code} (stock={stock})"

def move_out(code: str, qty: float, reason: str = "OUT", ref: str = "", user: str = "") -> Tuple[bool, str]:
    if not code:
        return False, "code requis"
    try:
        q = float(qty)
    except Exception:
        return False, "qty invalide"
    if q <= 0:
        return False, "qty doit être > 0"

    code = str(code).strip()

    _ensure_csv(PARTS_CSV, CANON)
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty or code not in df["code"].values:
        return False, f"{code} introuvable"

    idx = df.index[df["code"] == code][0]
    curr = int(df.at[idx, "quantite_dispo"])
    if int(q) > curr:
        return False, f"stock insuffisant ({curr}) pour {code}"

    stock = curr - int(q)
    df.at[idx, "quantite_dispo"] = stock
    df.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    _append_move({"type":"OUT","code":code,"qty":int(q),"reason":reason,"task_id":"","ref":ref,"user":user})
    return True, f"-{int(q)} sur {code} (stock={stock})"

# ========= Réservation / Annulation / Consommation (flux maintenance) =========
def reserve_parts(req: List[Dict]) -> Dict:
    _ensure_csv(PARTS_CSV, CANON)
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return {"done": 0, "errors": ["Inventaire vide"]}

    done = 0
    errors = []
    moves = []

    for r in (req or []):
        code = str(r.get("code","")).strip()
        qty  = int(r.get("qty",0) or 0)
        if not code or qty <= 0:
            errors.append({"code":code,"error":"code/qty invalide"}); continue
        if code not in df["code"].values:
            errors.append({"code":code,"error":"inconnu"}); continue
        idx = df.index[df["code"]==code][0]
        dispo = int(df.at[idx, "quantite_dispo"])
        if dispo < qty:
            errors.append({"code":code,"error":f"disponible={dispo} < {qty}"}); continue

        df.at[idx, "quantite_dispo"] = dispo - qty
        done += 1
        moves.append({
            "ts": time.time(), "type":"RESERVE", "code":code, "qty":qty,
            "reason": r.get("reason",""), "task_id": r.get("task_id",""),
            "ref":"", "user":""
        })

    df.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    if moves:
        _flush_csv(MOVES_CSV, moves, header=MOVES_HEADER)

    return {"done": done, "errors": errors}

def release_parts(req: List[Dict]) -> Dict:
    _ensure_csv(PARTS_CSV, CANON)
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return {"done": 0, "errors": ["Inventaire vide"]}

    done = 0
    errors = []
    moves = []

    for r in (req or []):
        code = str(r.get("code","")).strip()
        qty  = int(r.get("qty",0) or 0)
        if not code or qty <= 0:
            errors.append({"code":code,"error":"code/qty invalide"}); continue
        if code not in df["code"].values:
            errors.append({"code":code,"error":"inconnu"}); continue
        idx = df.index[df["code"]==code][0]
        df.at[idx, "quantite_dispo"] = int(df.at[idx, "quantite_dispo"]) + qty
        done += 1
        moves.append({
            "ts": time.time(), "type":"RELEASE", "code":code, "qty":qty,
            "reason": r.get("reason",""), "task_id": r.get("task_id",""),
            "ref":"", "user":""
        })

    df.to_csv(PARTS_CSV, index=False, encoding="utf-8")
    if moves:
        _flush_csv(MOVES_CSV, moves, header=MOVES_HEADER)

    return {"done": done, "errors": errors}

def consume_parts(req: List[Dict]) -> Dict:
    moves = []
    for r in (req or []):
        code = str(r.get("code","")).strip()
        qty  = int(r.get("qty",0) or 0)
        if not code or qty <= 0:
            continue
        moves.append({
            "ts": time.time(), "type":"CONSUME", "code":code, "qty":qty,
            "reason": r.get("reason",""), "task_id": r.get("task_id",""),
            "ref":"", "user":""
        })
    if moves:
        _flush_csv(MOVES_CSV, moves, header=MOVES_HEADER)
    return {"done": len(moves), "errors": []}

def stock_snapshot() -> Dict[str, Any]:
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return {"total_articles": 0, "total_quantite": 0, "valeur_totale": 0.0}
    total_articles = len(df)
    total_qte = int(df["quantite_dispo"].sum())
    valeur = float((df["quantite_dispo"] * df["prix_unitaire"]).sum())
    return {"total_articles": total_articles, "total_quantite": total_qte, "valeur_totale": round(valeur, 4)}

def low_stock(threshold_factor: float = 1.0) -> List[Dict]:
    df = normalize_parts_df(_read_csv(PARTS_CSV))
    if df.empty:
        return []
    cond = df["quantite_dispo"] <= (threshold_factor * df["seuil_min"])
    return df[cond].to_dict("records")

# ========= BOM par tâche =========
def attach_bom_to_task(task_id: str, items: List[Dict]) -> int:
    if not task_id:
        return 0
    rows = []
    for it in (items or []):
        rows.append({"task_id": task_id, "code": it.get("code",""), "qty": int(it.get("qty",0) or 0)})
    if rows:
        _flush_csv(BOM_CSV, rows, header=["task_id","code","qty"])
    return len(rows)

def bom_for_task(task_id: str) -> List[Dict]:
    df = _read_csv(BOM_CSV)
    if df.empty:
        return []
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce").fillna(0).astype(int)
    out = df[df.get("task_id")==task_id][["code","qty"]] if "task_id" in df.columns else df[["code","qty"]]
    return out.to_dict("records")

# ========= CLI facultative =========
def _cli():
    p = argparse.ArgumentParser(description="Inventory services CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_in  = sub.add_parser("in",  help="Entrée stock")
    p_in.add_argument("code"); p_in.add_argument("qty", type=float)
    p_in.add_argument("--reason", default="IN"); p_in.add_argument("--ref", default=""); p_in.add_argument("--user", default="")

    p_out = sub.add_parser("out", help="Sortie stock")
    p_out.add_argument("code"); p_out.add_argument("qty", type=float)
    p_out.add_argument("--reason", default="OUT"); p_out.add_argument("--ref", default=""); p_out.add_argument("--user", default="")

    sub.add_parser("list", help="Lister pièces")
    p_mv  = sub.add_parser("moves", help="Lister mouvements"); p_mv.add_argument("--limit", type=int, default=100)
    p_low = sub.add_parser("low", help="Seuils bas"); p_low.add_argument("--factor", type=float, default=1.0)

    args = p.parse_args()
    if args.cmd == "in":
        ok, msg = move_in(args.code, args.qty, reason=args.reason, ref=args.ref, user=args.user)
        print(("OK " if ok else "ERR ") + msg); return 0 if ok else 1
    if args.cmd == "out":
        ok, msg = move_out(args.code, args.qty, reason=args.reason, ref=args.ref, user=args.user)
        print(("OK " if ok else "ERR ") + msg); return 0 if ok else 1
    if args.cmd == "list":
        for r in list_parts():
            print(r)
        return 0
    if args.cmd == "moves":
        for r in list_movements(limit=args.limit):
            print(r)
        return 0
    if args.cmd == "low":
        for r in low_stock(args.factor):
            print(r)
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(_cli())
