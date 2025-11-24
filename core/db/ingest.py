import time
from datetime import datetime
from sqlalchemy import delete
from .engine import SessionLocal
from .schema import Measurement, Event, Failure, AFResult
from ..analytics.organigram_af import run_af

def _iso(ts): 
    try: return datetime.fromtimestamp(float(ts)).isoformat(sep=' ', timespec='seconds')
    except: return datetime.now().isoformat(sep=' ', timespec='seconds')

def persist_measurement(m: dict):
    with SessionLocal() as s:
        row = Measurement(
            ts=float(m.get("ts", time.time())), t_iso=_iso(m.get("ts")),
            site=m.get("site",""), equipment=m.get("equipment",""),
            v_sec=m.get("v_sec"), i_sec=m.get("i_sec"), p_sec=m.get("p_sec"),
            v_prim_rms=m.get("v_prim_rms"), i_prim_rms=m.get("i_prim_rms"),
            pf_prim=m.get("pf_prim"), freq=m.get("freq"),
            t_core=m.get("t_core"), mu_oil=m.get("mu_oil"),
            status=m.get("status","OK")
        ); s.add(row); s.commit()

def persist_event(e: dict):
    with SessionLocal() as s:
        row = Event(
            ts=float(e.get("ts", time.time())), t_iso=_iso(e.get("ts")),
            site=e.get("site",""), equipment=e.get("equipment",""),
            level=e.get("level","INFO"), code=e.get("code",""), msg=e.get("msg",""),
            value=str(e.get("value","")), threshold=str(e.get("threshold","")), processed=0
        )
        s.add(row); s.commit()
        return row.id

def record_failure(ts_fail: float, site: str, equipment: str, src_event_id=None, reason="", note=""):
    with SessionLocal() as s:
        s.add(Failure(t_fail=ts_fail, site=site, equipment=equipment,
                      source_event_id=src_event_id, reason=reason, note=note))
        s.commit()

def run_af_and_store(site: str, equipment: str):
    """recalcule AF sur toutes les défaillances du couple site/equipment"""
    from sqlalchemy import select
    with SessionLocal() as s:
        ts = [r.t_fail for r in s.execute(
            select(Failure).where(Failure.site==site, Failure.equipment==equipment)
            .order_by(Failure.t_fail)
        ).scalars()]
        if len(ts) < 6: return None
        res = run_af(ts)
        s.add(AFResult(site=site, equipment=equipment, n=res["n"],
                       z_laplace=res["z"], tau_mk=res["tau"], 
                       classification=res["class"], ts_generated=time.time()))
        s.commit(); return res

def reset_tables(tables=("measurements","events","failures","af_results")):
    with SessionLocal() as s:
        for T in (Measurement, Event, Failure, AFResult):
            if T.__tablename__ in tables:
                s.execute(delete(T))
        s.commit()
