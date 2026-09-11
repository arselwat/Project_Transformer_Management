import io,json
from datetime import datetime
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
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
                if isinstance(df[col].dtype,pd.DatetimeTZDtype): df[col]=df[col].astype(str)
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