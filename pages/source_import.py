"""Compatibilité avec les fichiers historiques ; aucune ligne ignorée silencieusement."""
import io
import pandas as pd
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "equipment": "equipment_code",
        "equipement": "equipment_code",
        "code_equipement": "equipment_code",
        "eqp": "equipment_code",
        "asset_id": "equipment_code",
        "assetid": "equipment_code",
        "horodatage": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "date_heure": "timestamp",
        "dateheure": "timestamp",
        "failure_time": "timestamp",
        "failure_date": "timestamp",
        "date_panne": "timestamp",
        "panne": "is_failure",
        "defaillance": "is_failure",
        "failure": "is_failure",
        "isfailure": "is_failure",
        "repair_hours": "repair_time_hours",
        "repair_time_hours": "repair_time_hours",
        "mttr_h": "repair_time_hours",
        "duree_rep_h": "repair_time_hours",
        "duree_reparation_h": "repair_time_hours",
        "time_repair": "repair_time_hours",
        "repair_time": "repair_time_hours",
    }

    cols = {str(c).lower().strip(): c for c in df.columns}
    ren = {}
    for k, v in mapping.items():
        if k in cols:
            ren[cols[k]] = v

    out = df.rename(columns=ren).copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def normalize_input(df):
    # La normalisation historique s'applique AVANT la détection du format.
    df=_normalize_columns(df)
    if df.columns.duplicated().any():
        raise ValueError('Plusieurs colonnes correspondent au même champ. Vérifiez les en-têtes.')
    if 'event_start' in df or 'timestamp' in df:
        df=df.rename(columns={'equipment_code':'asset_id','timestamp':'event_start'})
        if 'asset_id' not in df:
            raise ValueError('Identifiant équipement absent.')
        if 'is_failure' not in df:
            raise ValueError('Colonne is_failure absente : ajouter 1 pour les pannes et 0 pour les autres événements.')
        return 'events_history',df
    if {'equipment_code','ttf_h'}.issubset(df.columns):
        return 'intervals',df.rename(columns={'repair_time_hours':'duree_rep_h'})
    raise ValueError('Format non reconnu. Utilisez equipment_code, timestamp, is_failure et repair_time_hours ; ou equipment_code et ttf_h.')

def read_source(raw,filename):
    if filename.lower().endswith('.xlsx'):
        frames=pd.read_excel(io.BytesIO(raw),sheet_name=None,dtype={'equipment_code':str,'asset_id':str})
        frames={str(k).strip():v for k,v in frames.items()}
        if 'events_history' in frames:
            _,frames['events_history']=normalize_input(frames['events_history'])
            return frames
        if len(frames)!=1:
            raise ValueError('Classeur multifeuille : nommez la feuille des événements events_history pour conserver aussi les réglages.')
        df=next(iter(frames.values()))
    else:
        for encoding in ('utf-8-sig','cp1252'):
            try:
                df=pd.read_csv(io.StringIO(raw.decode(encoding)),sep=None,engine='python',dtype={'equipment_code':str,'asset_id':str})
                break
            except UnicodeDecodeError:
                continue
    kind,df=normalize_input(df)
    return {kind:df}
