import pandas as pd
import unicodedata
import difflib
from typing import Union, List, Dict, Tuple, Optional

# Colonnes canoniques requises
REQUIRED_COLS = {"equipment_code", "date_panne", "heures_fonct", "duree_rep_h"}

# Variantes usuelles (élargies)
SYNONYMS = {
    "equipment_code": {
        "equipment_code","equipement_code","equipement","equipment","asset","machine",
        "code_equipement","code_equip","code","transformateur","transformer_id",
        "id_equipement","id","eq_code","eq id","eqid","no_equipement","num_equipement",
    },
    "date_panne": {
        "date_panne","date panne","date","date defaut","date_defaut","date_defaillance",
        "failure_date","incident_date","dateincident","datepanne","date de panne",
        "date de defaut","date d incident","date d'incident","date d’arrêt","date arret",
        "date outage","outage_date",
    },
    "heures_fonct": {
        "heures_fonct","heures","heures_fonctionnement","heures fonctionnement",
        "run_hours","operating_hours","service_hours","heures_service","heures_cumulees",
        "heurescumul","heures avant panne","hours to failure","time to failure","ttf",
    },
    "duree_rep_h": {
        "duree_rep_h","duree_reparation_h","duree_reparation","temps_reparation",
        "repair_time_h","repair_time","mttr","duree","tempsreparation","dureereparation",
        "downtime_h","downtime","temps d arret","temps d'arrêt","time to repair","ttr",
    },
}

# ----------------------------------------------------
# Normalisation & utilitaires
# ----------------------------------------------------
def _normalize(s: str) -> str:
    """minuscule, sans accents, espaces compressés, remplace -_ par espace."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    for ch in ["\n", "\t", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def _build_inverse(cols: List[str]) -> Dict[str, List[str]]:
    """map: nom_normalisé -> [noms_originaux...]"""
    inv: Dict[str, List[str]] = {}
    for c in cols:
        n = _normalize(c)
        inv.setdefault(n, []).append(c)
    return inv

def _best_fuzzy_match(target_norm: str, candidates_norm: List[str], min_ratio: float = 0.75) -> Optional[str]:
    """Retourne le candidat normalisé le plus proche (>= min_ratio), sinon None."""
    if not candidates_norm:
        return None
    # difflib renvoie une liste des plus proches
    best = difflib.get_close_matches(target_norm, candidates_norm, n=1, cutoff=min_ratio)
    return best[0] if best else None

def _try_find_header_row(df: pd.DataFrame) -> Optional[int]:
    """
    Si le fichier a des lignes d'en-tête avant les vraies colonnes,
    essaie de trouver l'index de la ligne qui ressemble à un header (>=2 colonnes plausibles).
    """
    # On essaie sur les 10 premières lignes
    max_scan = min(10, len(df))
    for i in range(max_scan):
        row = df.iloc[i].tolist()
        # Convertir en strings et normaliser
        norm = [_normalize(x) for x in row]
        # Combien matchent des synonymes connus ?
        hits = 0
        synonyms_flat = set().union(*SYNONYMS.values())
        synonyms_norm = {_normalize(x) for x in synonyms_flat}
        for x in norm:
            if x in synonyms_norm:
                hits += 1
        if hits >= 2:
            return i
    return None

# ----------------------------------------------------
# Auto-mapping des colonnes
# ----------------------------------------------------
def _auto_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Essaie de renommer automatiquement vers equipment_code, date_panne, heures_fonct, duree_rep_h.
    Retourne (df_renommé, dict_renames) pour debug.
    """
    if df is None or df.empty:
        return df, {}

    inv = _build_inverse(list(df.columns))
    candidates_norm = list(inv.keys())
    renames: Dict[str, str] = {}

    for target, variants in SYNONYMS.items():
        target_norm = _normalize(target)
        # 1) match exact du nom canonique
        if target_norm in inv:
            orig = inv[target_norm][0]
            renames[orig] = target
            continue
        # 2) match exact d'une variante
        found_orig = None
        for v in variants:
            vn = _normalize(v)
            if vn in inv:
                found_orig = inv[vn][0]
                break
        if found_orig:
            renames[found_orig] = target
            continue
        # 3) fuzzy matching sur les normalisés
        best_norm = _best_fuzzy_match(target_norm, candidates_norm, min_ratio=0.78)
        if best_norm:
            # éviter de prendre des "unnamed: 0" etc.
            if not best_norm.startswith("unnamed"):
                orig = inv[best_norm][0]
                renames[orig] = target

    if renames:
        df = df.rename(columns=renames)

    return df, renames

# ----------------------------------------------------
# Lectures robustes
# ----------------------------------------------------
def _read_any(path_or_buffer) -> pd.DataFrame:
    """Lit CSV/Excel de manière tolérante : essaie successivement ; , \t | espace."""
    if hasattr(path_or_buffer, "name"):
        name = path_or_buffer.name.lower()
    else:
        name = str(path_or_buffer).lower()

    # --- Cas Excel ---
    if name.endswith((".xls", ".xlsx")):
        try:
            return pd.read_excel(path_or_buffer)
        except Exception:
            # fallback brut
            return pd.read_excel(path_or_buffer, header=None)

    # --- Cas CSV texte ---
    separators = [",", ";", "\t", "|", " "]
    for sep in separators:
        try:
            df = pd.read_csv(path_or_buffer, sep=sep, engine="python")
            # S’il n’y a qu’une seule colonne, continuer à tester
            if len(df.columns) < 2:
                continue
            return df
        except Exception:
            continue

    # Dernier recours : lire brut puis re-split
    try:
        text = path_or_buffer.read().decode("utf-8", errors="ignore") if hasattr(path_or_buffer, "read") else open(path_or_buffer, "r", encoding="utf-8", errors="ignore").read()
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            raise ValueError("Fichier vide.")
        first_line = lines[0]
        # détection simple : compter les séparateurs dominants
        counts = {sep: first_line.count(sep) for sep in separators}
        best_sep = max(counts, key=counts.get)
        df = pd.read_csv(pd.compat.StringIO(text), sep=best_sep, engine="python")
        return df
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier CSV: {e}")


# ----------------------------------------------------
# API principale
# ----------------------------------------------------
def load_failures_csv(path_or_buffer_or_df: Union[str, "pd.DataFrame"]) -> pd.DataFrame:
    """
    Lit un CSV/Excel/DF, détecte l'en-tête, renomme automatiquement les colonnes
    vers: equipment_code, date_panne, heures_fonct, duree_rep_h.
    Convertit types, valide, et retourne un DataFrame propre.
    """
    # Lire de façon robuste
    if isinstance(path_or_buffer_or_df, pd.DataFrame):
        df0 = path_or_buffer_or_df.copy()
    else:
        df0 = _read_any(path_or_buffer_or_df)

    # Auto-rename tolérant (synonymes + fuzzy)
    df1, ren = _auto_map_columns(df0)

    # Diagnostic utile (colonnes présentes)
    present = list(df1.columns)

    # Vérifier colonnes manquantes
    missing = REQUIRED_COLS - set(df1.columns)
    if missing:
        raise ValueError(
            "Colonnes manquantes: "
            f"{sorted(missing)}\nColonnes présentes détectées: {present}\n"
            f"Renommages appliqués: {ren}"
        )

    # Types & nettoyage
    df1["date_panne"] = pd.to_datetime(df1["date_panne"], errors="coerce")
    df1["heures_fonct"] = pd.to_numeric(df1["heures_fonct"], errors="coerce")
    df1["duree_rep_h"] = pd.to_numeric(df1["duree_rep_h"], errors="coerce")

    df1 = df1.dropna(subset=["equipment_code", "date_panne", "heures_fonct", "duree_rep_h"])

    return df1
