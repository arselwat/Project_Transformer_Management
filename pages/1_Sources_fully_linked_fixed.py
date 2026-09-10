import pandas as pd
import streamlit as st
from core.scientific_ui import start, invalidate
from core.datahub import set_current_project_data, set_current_failures_df, clear_current_project_data, get_current_project_data, get_current_failures_df, get_failures_meta
start('Sources de données', 'pages/1_Sources_fully_linked_fixed.py')
st.write('Importez un classeur Excel avec events_history et, si disponibles, analysis_settings, asset_info et les autres feuilles du projet ; ou un CSV d’événements ou d’intervalles.')
st.caption('Événements : asset_id, event_start, repair_time_hours (optionnel). Intervalles : equipment_code, ttf_h, duree_rep_h (optionnel). Réglages : asset_id, observation_start, observation_end, decision_time. Dates ISO recommandées. Les intervalles seuls ne permettent pas de dater une échéance.')
f = st.file_uploader('CSV ou Excel', type=['csv','xlsx'])
if f is not None:
    try:
        if f.name.lower().endswith('.xlsx'):
            frames = pd.read_excel(f,sheet_name=None)
            if 'events_history' not in frames:
                df = next(iter(frames.values()))
                frames = {'events_history':df} if 'event_start' in df or 'date_panne' in df else {'intervals':df}
        else:
            df = pd.read_csv(f,sep=None,engine='python')
            frames = {'events_history':df} if any(c in df for c in ['event_start','date_panne','failure_date','failure_time']) else {'intervals':df}
        for name,df in frames.items():
            st.write(name)
            st.dataframe(df.head(30))
        if st.button('Utiliser ces données',type='primary'):
            if 'intervals' in frames:
                df=frames['intervals']
                if not {'equipment_code','ttf_h'}.issubset(df.columns):
                    raise ValueError('Colonnes requises : equipment_code et ttf_h.')
                clear_current_project_data(clear_failures=True)
                info=set_current_failures_df(df,source_name=f.name)
            else:
                info=set_current_project_data(frames,source_name=f.name)
            invalidate()
            st.success('Import traité. Vérifiez les anomalies ci-dessous avant l’analyse.')
            st.json(info)
    except Exception as exc:
        st.error('Import impossible : '+str(exc))
project=get_current_project_data()
for name in ['observation_windows','data_quality_report']:
    if not project.get(name,pd.DataFrame()).empty:
        st.write(name)
        st.dataframe(project[name])
st.subheader('Intervalles actifs')
st.dataframe(get_current_failures_df())
with st.expander('Contrôles de l’import des intervalles'):
    st.json(get_failures_meta())
