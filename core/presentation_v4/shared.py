"""Parcours scientifique commun : une analyse et une optimisation par jeu de données."""
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from core.datahub import get_current_failures_df, get_current_project_data, get_pipeline_inputs
from .organigram import analyze_project_inputs
from .optimize import optimize_maintenance

@st.cache_data(max_entries=8, ttl=3600, show_spinner=False)
def cached_analysis(inputs, **options):
    """Réutilise un résultat uniquement pour les mêmes données et options."""
    return analyze_project_inputs(inputs, **options)

LABELS = {'expon':'HPP', 'power_law_nhpp':'PLP-NHPP', 'weibull_2p':'Renouvellement Weibull', 'lognorm':'Renouvellement lognormal', 'grp_kijima_i':'GRP Kijima I'}

def start(title, path):
    from core.security.auth import require_login
    from core.ui import render_shell, render_page_header
    st.set_page_config(page_title=title, layout='wide')
    require_login()
    render_shell(path)
    render_page_header(title, 'Analyse commune, prévision conditionnelle et traçabilité.', '📊')

def invalidate():
    for key in ('science_analysis','science_optimization','science_input_hash','science_inputs','exports_v4','maintenance_proposal_v4'):
        st.session_state.pop(key, None)

def input_context():
    frames = get_current_project_data()
    ids = set()
    for frame, column in [(frames.get('events_history',pd.DataFrame()),'asset_id'), (get_current_failures_df(),'equipment_code')]:
        if column in frame:
            ids.update(frame[column].dropna().astype(str))
    if not ids:
        st.info('Importez les données dans Sources pour commencer.')
        st.stop()
    asset = st.selectbox('Équipement', sorted(ids), key='_science_asset_widget', index=sorted(ids).index(st.session_state.get('chosen_asset')) if st.session_state.get('chosen_asset') in ids else 0)
    st.session_state['chosen_asset']=asset
    inputs = get_pipeline_inputs(asset)
    signature = hashlib.sha256(repr((asset, inputs['ttf_series'], inputs['repair_series'], inputs['observation_windows'].to_dict(), inputs['project_meta'], inputs['failures_meta'])).encode()).hexdigest()
    if st.session_state.get('science_input_hash') != signature:
        invalidate()
        st.session_state['science_input_hash'] = signature
    st.session_state['science_inputs'] = inputs
    return inputs

def analysis_required():
    inputs = input_context()
    result = st.session_state.get('science_analysis')
    if not result:
        st.info('Lancez le calcul dans la page Indicateurs pour cet équipement.')
        st.stop()
    return inputs, result

def notices(r):
    for warning in r.get('warnings',[]):
        if not warning.startswith('Les anciennes pages'):
            st.warning(warning)
    st.caption('La sélection statistique ne démontre ni une cause physique ni une opération de maintenance nécessaire.')

def show_optimization(o):
    st.dataframe(pd.DataFrame([{'Horizon fiabiliste (h)':o.get('T_R'), 'Horizon économique (h)':o.get('T_cost'), 'Horizon retenu (h)':o.get('T_recommended'), 'Fiabilité retenue':o.get('reliability_at_recommended'), 'Statut':o.get('status')}]), hide_index=True)
    st.write(o.get('selection_reason', o.get('reason','')))
    if o.get('economic_status') == 'unsupported_model':
        st.info('Modèle de renouvellement ou GRP : horizon fiabiliste uniquement ; optimisation économique indisponible.')
    notices(o)

def plan(inputs, o, today=None):
    windows = inputs.get('observation_windows',pd.DataFrame())
    horizon = o.get('T_recommended')
    if windows.empty or horizon is None or not np.isfinite(horizon):
        return {'statut':'Date indisponible : référence calendaire ou horizon absent.'}
    reference = pd.to_datetime(windows.iloc[0]['decision_time'], utc=True)
    due = reference + pd.Timedelta(hours=float(horizon))
    now = pd.Timestamp.now(tz='UTC') if today is None else pd.to_datetime(today,utc=True)
    remaining = (due-now).total_seconds()/86400
    return {'référence':reference.isoformat(), 'échéance indicative':due.isoformat(), 'jours restants':remaining,
            'statut':'Échéance dépassée : actualiser les données et réexaminer le plan.' if remaining<0 else 'Échéance indicative à examiner avec le diagnostic technique.',
            'qualification':o.get('status')}

def export_bundle(inputs, analysis, optimization):
    def default(x):
        if isinstance(x,pd.DataFrame): return x.to_dict(orient='records')
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        return str(x)
    payload = {'asset_id':inputs['asset_id'], 'analysis':analysis, 'optimization':optimization,
               'plan':plan(inputs,optimization) if optimization else None,
               'observation_windows':inputs['observation_windows'], 'quality_report':inputs['data_quality_report']}
    b = io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('resultats.json',json.dumps(payload,ensure_ascii=False,indent=2,default=default))
        for name, df in analysis['tables'].items():
            z.writestr(name+'.csv',df.to_csv(index=False))
        z.writestr('LISEZ_MOI.txt','Les CSV et le JSON proviennent du même calcul. Voir analysis_hash et les options de validation. Une échéance est indicative, non une date certaine de panne.')
    return b.getvalue()


def scientific_charts(inputs, result, optimization=None):
    """Graphiques issus des observations et sorties calculées, sans réajustement."""
    x=np.asarray(inputs['ttf_series'],dtype=float)
    if len(x):
        st.subheader('Historique et diagnostics graphiques')
        a,b=st.columns(2)
        with a:
            st.write('Nombre cumulé de pannes après la panne de référence')
            st.line_chart(pd.DataFrame({'Temps calendaire (h)':np.r_[0,np.cumsum(x)],'Pannes cumulées':np.arange(len(x)+1)}).set_index('Temps calendaire (h)'))
        with b:
            st.write('Intervalles calendaires selon le rang')
            st.line_chart(pd.DataFrame({'Rang':np.arange(1,len(x)+1),'Intervalle (h)':x}).set_index('Rang'))
        if len(x)>1:
            st.write('Dépendance entre intervalles successifs')
            st.scatter_chart(pd.DataFrame({'Intervalle i (h)':x[:-1],'Intervalle i+1 (h)':x[1:]}),x='Intervalle i (h)',y='Intervalle i+1 (h)')
    r=result['reliability']
    st.subheader('Comparaison des modèles par AICc')
    scores=[{'Modèle':LABELS.get(k,k),'AICc':v.get('aicc')} for k,v in r.get('candidates',{}).items() if v.get('aicc') is not None]
    if scores:
        st.bar_chart(pd.DataFrame(scores).set_index('Modèle'))
    curves=r.get('curves',pd.DataFrame()).replace([np.inf,-np.inf],np.nan)
    st.subheader('Courbes conditionnelles depuis la référence')
    for col,title in [('R_t','Fiabilité R(u | s)'),('F_t','Probabilité de panne avant u'),('f_t','Densité du délai avant la prochaine panne (1/h)'),('h_t','Risque conditionnel / intensité NHPP (1/h)')]:
        if col in curves and not curves.empty:
            st.write(title)
            st.line_chart(curves[['t',col]].rename(columns={'t':'Horizon supplémentaire (h)',col:title}).set_index('Horizon supplémentaire (h)'))
    st.caption('Les courbes sont conditionnées par l’historique et le modèle retenu. Elles ne repartent pas de l’état neuf. Les valeurs infinies ne sont pas tracées.')
    if optimization and optimization.get('T_recommended') is not None:
        from .organigram import conditional_reliability
        from .optimize import cost_rate_nhpp
        o=optimization;p=r['params'];name=r['distribution']
        top=max(o['T_recommended'],o.get('T_R') or 0,o.get('T_cost') or 0)*1.3
        u=np.linspace(max(top/300,1e-6),top,250)
        st.subheader('Fiabilité et cibles de planification')
        rr=[conditional_reliability(name,p,float(t)) for t in u]
        st.line_chart(pd.DataFrame({'Horizon (h)':u,'Fiabilité':rr,'Cible principale':o['R_target'],'Cible économique':o['R_min_cost']}).set_index('Horizon (h)'))
        if name in ('power_law_nhpp','expon'):
            beta=1. if name=='expon' else p['beta']
            eta=1/p['lambda_hpp_h'] if name=='expon' else p['eta']
            costs=[cost_rate_nhpp(float(t),beta,eta,o['C_prev'],o['C_corr'],p['reference_time_h'],p.get('origin_offset_h',0.)) for t in u]
            st.write('Coût moyen prospectif par heure')
            st.line_chart(pd.DataFrame({'Horizon (h)':u,'Coût / h':costs}).set_index('Horizon (h)'))
        st.caption('Horizon retenu : %.2f h. Le tracé explore aussi des horizons hors des cibles ; ils ne sont pas des recommandations.' % o['T_recommended'])
