# core/datahub.py
from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAILURES_FILE = DATA_DIR / "failures_saved.csv"

REQUIRED_COLS = {"equipment_code", "ttf_h"}


def _clean_failures_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if not REQUIRED_COLS.issubset(set(df.columns)):
        return pd.DataFrame()

    df["equipment_code"] = df["equipment_code"].astype(str)
    df["ttf_h"] = pd.to_numeric(df["ttf_h"], errors="coerce")
    df = df.dropna(subset=["ttf_h"])
    df = df[df["ttf_h"] > 0]
    return df


def _dataset_hash(df: pd.DataFrame) -> str:
    x = df[["equipment_code", "ttf_h"]].copy()
    x["equipment_code"] = x["equipment_code"].astype(str)
    x["ttf_h"] = x["ttf_h"].astype(float).round(6)
    b = x.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def set_current_failures_df(df: pd.DataFrame, source_name: str = "unknown", persist: bool = True) -> dict:
    df2 = _clean_failures_df(df)
    if df2.empty:
        return {"ok": False, "msg": "Dataset vide ou invalide. Il faut equipment_code et ttf_h (>0)."}

    if persist:
        FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(FAILURES_FILE, index=False, encoding="utf-8")

    h = _dataset_hash(df2)
    st.session_state["failures_df"] = df2
    st.session_state["failures_hash"] = h
    st.session_state["failures_source"] = source_name
    return {"ok": True, "rows": int(len(df2)), "hash": h, "file": str(FAILURES_FILE)}


def get_current_failures_df() -> pd.DataFrame:
    # 1) session d'abord
    if isinstance(st.session_state.get("failures_df"), pd.DataFrame):
        df = _clean_failures_df(st.session_state["failures_df"])
        if not df.empty:
            if not st.session_state.get("failures_hash"):
                st.session_state["failures_hash"] = _dataset_hash(df)
            return df

    # 2) fallback fichier
    if not FAILURES_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(FAILURES_FILE)
    except Exception:
        try:
            df = pd.read_csv(FAILURES_FILE, engine="python", on_bad_lines="skip", sep=None)
        except Exception:
            return pd.DataFrame()

    df = _clean_failures_df(df)
    if df.empty:
        return df

    st.session_state["failures_df"] = df
    st.session_state["failures_hash"] = _dataset_hash(df)
    st.session_state["failures_source"] = "file:data/failures_saved.csv"
    return df


def get_failures_meta() -> dict:
    df = get_current_failures_df()
    if df.empty:
        return {"ok": False, "rows": 0, "hash": "", "source": "", "file": str(FAILURES_FILE)}
    return {
        "ok": True,
        "rows": int(len(df)),
        "hash": str(st.session_state.get("failures_hash", "")),
        "source": str(st.session_state.get("failures_source", "")),
        "file": str(FAILURES_FILE),
    }
