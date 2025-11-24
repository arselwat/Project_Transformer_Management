from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from .engine import Base, ENGINE

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True)
    ts = Column(Float, index=True)
    t_iso = Column(String, index=True)
    site = Column(String); equipment = Column(String)
    v_sec = Column(Float); i_sec = Column(Float); p_sec = Column(Float)
    v_prim_rms = Column(Float); i_prim_rms = Column(Float)
    pf_prim = Column(Float); freq = Column(Float)
    t_core = Column(Float); mu_oil = Column(Float); status = Column(String)
    created_at = Column(DateTime, server_default=func.now())

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    ts = Column(Float, index=True)
    t_iso = Column(String, index=True)
    site = Column(String); equipment = Column(String)
    level = Column(String); code = Column(String); msg = Column(String)
    value = Column(String); threshold = Column(String)
    processed = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

class Failure(Base):
    __tablename__ = "failures"
    id = Column(Integer, primary_key=True)
    t_fail = Column(Float, index=True)
    site = Column(String); equipment = Column(String)
    source_event_id = Column(Integer); reason = Column(String); note = Column(String)

class AFResult(Base):
    __tablename__ = "af_results"
    id = Column(Integer, primary_key=True)
    site = Column(String); equipment = Column(String)
    n = Column(Integer); z_laplace = Column(Float); tau_mk = Column(Float)
    classification = Column(String)  # RP/NHPP/BPP
    ts_generated = Column(Float)

def init_db():
    Base.metadata.create_all(bind=ENGINE)
