"""Résultats partagés par les cinq pages du parcours scientifique.

Les résultats sont conservés dans la session. Toute modification des données
invalide analyses, optimisations et propositions. Les exports n'ajustent aucun modèle.
"""
from __future__ import annotations
import io
import json
import hashlib
from datetime import datetime
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import streamlit as st
from core.datahub import get_current_project_data, get_current_failures_df, get_pipeline_inputs, get_failures_meta, get_project_meta
from core.reliability.organigram import analyze_project_inputs, build_reliability_tables
from core.reliability.optimize import optimize_maintenance
from core.security.auth import require_login
from core.ui import render_shell, render_page_header

LABELS={"expon":"HPP / exponentielle", "power_law_nhpp":"PLP-NHPP", "weibull_2p":"Renouvellement Weibull", "lognorm":"Renouvellement lognormal", "grp_kijima_i":"GRP Kijima I"}

def page_header(path,title,subtitle,icon):
    st.set_page_config(page_title=title,page_icon=icon,layout="wide")
    require_login()
    render_shell(path)
    render_page_header(title,subtitle,icon)
    sync_state()

def sync_state():
    signature=str(get_failures_meta().get("hash",""))+":"+str(get_project_meta().get("hash",""))
    if st.session_state.get("workflow_signature")!=signature:
        st.session_state["workflow_signature"]=signature
        for name in ("analyses_v2","optimizations_v2","plans_v2"):
            st.session_state[name]={}
    for name in ("analyses_v2","optimizations_v2","plans_v2"):
        st.session_state.setdefault(name,{})

def choose_asset(key):
    df=get_current_failures_df()
    if df.empty:
        st.info("Importez un historique exploitable dans Sources de données.")
        st.stop()
    assets=sorted(df.equipment_code.astype(str).unique())
    selected=st.selectbox("Équipement",assets,key=key)
    return selected,get_pipeline_inputs(selected)

def calculate(asset,inputs,options):
    window=inputs.get("observation_windows",pd.DataFrame())
    options=dict(options)
    if len(window)==1:
        options["observation_scheme"]="time_terminated" if window.iloc[0].get("end_source")=="provided" else "event_terminated"
    result=analyze_project_inputs(inputs,**options)
    r=result["reliability"]
    # Une tentative échouée remplace aussi les résultats précédents.
    st.session_state["analyses_v2"][asset]=result
    st.session_state["optimizations_v2"].pop(asset,None)
    st.session_state["plans_v2"].pop(asset,None)
    return result

def current_analysis(asset):
    value=st.session_state["analyses_v2"].get(asset)
    if not value or value["reliability"].get("status")!="computed":
        st.info("Calculez d'abord l'analyse de cet équipement dans Indicateurs.")
        st.stop()
    return value

def optimize(asset,analysis,options):
    value=optimize_maintenance(analysis,**options)
    value["optimization_id"]=hashlib.sha256(json.dumps(json_safe(value),sort_keys=True).encode()).hexdigest()
    st.session_state["optimizations_v2"][asset]=value
    st.session_state["plans_v2"].pop(asset,None)
    return value

def current_optimization(asset,analysis):
    value=st.session_state["optimizations_v2"].get(asset)
    if not value or value.get("analysis_hash")!=analysis["reliability"].get("analysis_hash"):
        st.info("Calculez les horizons dans Optimisation pour cette analyse.")
        st.stop()
    return value

def validation_message(r):
    accepted=r.get("goodness",{}).get("accepted")
    if accepted is False:
        st.warning("Le modèle sélectionné est rejeté par au moins un test bootstrap. Les horizons restent des scénarios exploratoires.")
    elif accepted is None:
        st.info("Adéquation non conclue : le bootstrap n'a pas été exécuté ou est incomplet. Les tests nominaux ne suffisent pas à valider le modèle.")
    else:
        st.success("Les tests bootstrap exécutés ne rejettent pas le modèle au seuil choisi. Cela ne prouve pas sa validité physique.")

def number(value,digits=1):
    if value is None: return "Non disponible"
    try:
        return f"{float(value):,.{digits}f}".replace(","," ") if np.isfinite(float(value)) else "Non disponible"
    except (ValueError,TypeError): return str(value)

def json_safe(value):
    if isinstance(value,pd.DataFrame): return json_safe(value.to_dict(orient="records"))
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime)): return value.isoformat()
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT: return None
    return value

def excel_bytes(tables):
    target=io.BytesIO()
    with pd.ExcelWriter(target,engine="openpyxl") as writer:
        for i,(name,frame) in enumerate(tables.items()):
            df=frame.copy()
            for col in df:
                if df[col].dtype==object:
                    df[col]=df[col].map(lambda v:json.dumps(json_safe(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v)
            df.to_excel(writer,sheet_name=(str(i+1)+"_"+name)[:31],index=False)
    return target.getvalue()

def export_tables(asset,inputs,analysis,optimization=None,plan=None):
    tables=dict(analysis["tables"])
    tables["observations"]=inputs.get("events_validated",pd.DataFrame())
    tables["intervalles"]=inputs.get("failures_ttf",pd.DataFrame())
    tables["fenetre"]=inputs.get("observation_windows",pd.DataFrame())
    tables["controle_donnees"]=inputs.get("data_quality_report",pd.DataFrame())
    r=analysis["reliability"]
    tables["tracabilite"]=pd.DataFrame([{"equipment":asset,"analysis_hash":r.get("analysis_hash"),**r.get("context",{}),**r.get("selection",{})}])
    if optimization: tables["optimisation"]=pd.DataFrame([{k:v for k,v in optimization.items() if k not in ("economic_result","warnings")}])
    if plan: tables["proposition_maintenance"]=pd.DataFrame([plan])
    tables["limites"]=pd.DataFrame({"limite":r.get("warnings",[])+(optimization or {}).get("warnings",[])})
    for name,b in r.get("bootstrap",{}).items():
        tables["bootstrap_"+name]=pd.DataFrame([{k:v for k,v in b.items() if k not in ("draws","errors")}])
    return tables

def pdf_bytes(asset,tables):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    target=io.BytesIO();styles=getSampleStyleSheet()
    document=SimpleDocTemplate(target,pagesize=landscape(A4),rightMargin=26,leftMargin=26,topMargin=26,bottomMargin=26)
    flow=[Paragraph("Analyse de fiabilité — "+escape(str(asset)),styles["Title"]),Spacer(1,10)]
    # Tableaux longs et résultats complets disponibles en Excel/JSON ; PDF de synthèse.
    for name in ("tracabilite","reliability_summary","fit_candidates","predictive_validation","optimisation","proposition_maintenance","limites"):
        frame=tables.get(name,pd.DataFrame())
        if frame.empty: continue
        flow.append(Paragraph(escape(name.replace("_"," ")),styles["Heading2"]))
        # Présentation verticale pour garder tous les champs lisibles.
        for _,row in frame.iterrows():
            pairs=[[Paragraph(escape(str(k)),styles["BodyText"]),Paragraph(escape(str(json_safe(v))),styles["BodyText"])] for k,v in row.items()]
            table=Table(pairs,colWidths=[215,550],hAlign="LEFT")
            table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.25,colors.lightgrey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef4ff"))]))
            flow.extend([table,Spacer(1,8)])
        flow.append(PageBreak())
    document.build(flow)
    return target.getvalue()
