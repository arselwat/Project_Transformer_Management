from core.scientific_ui import scientific_charts
import os
import streamlit as st
from core.scientific_ui import start,input_context,notices,LABELS,cached_analysis
from core.reliability.organigram import analyze_project_inputs
start('Indicateurs','pages/2_Indicateurs_verified.py')
inputs=input_context()
advanced = os.environ.get('SCIENTIFIC_HEAVY_COMPUTE', '0') == '1'
if not advanced:
    st.info('Mode économe : comparaison AICc et scénarios disponibles. Bootstrap et validation séquentielle réservés à une exécution locale activée explicitement.')
st.dataframe(inputs['observation_windows'])
with st.form('science_calculation'):
    rule=st.selectbox('Sélection', (['AICc','Prévision séquentielle','Scénario explicite'] if advanced else ['AICc','Scénario explicite']))
    model=st.selectbox('Modèle du scénario explicite',list(LABELS),format_func=LABELS.get)
    sequential=st.checkbox('Calculer la validation prédictive séquentielle',value=False, disabled=not advanced)
    initial=st.number_input('Intervalles d’apprentissage initial',min_value=3,value=8,step=1)
    boot=st.number_input('Réplications bootstrap (0 : non exécuté)',min_value=0,max_value=10000,value=0,step=100,disabled=not advanced)
    seed=st.number_input('Graine de simulation',min_value=0,value=20260823,step=1)
    alpha=st.number_input('Seuil des tests',min_value=.001,max_value=.2,value=.05,format='%.3f')
    st.caption('Le bootstrap porte sur le modèle retenu. Un nombre élevé de réplications peut prendre plusieurs minutes.')
    run=st.form_submit_button('Calculer',type='primary')
if run:
    st.session_state.pop('science_analysis',None)
    st.session_state.pop('science_optimization',None)
    try:
        with st.spinner('Estimation des cinq modèles et validations demandées…'):
            result=cached_analysis(inputs,alpha=alpha,selection_rule='predictive' if rule=='Prévision séquentielle' else 'aicc',selected_model=model if rule=='Scénario explicite' else None,run_sequential=advanced and (sequential or rule=='Prévision séquentielle'),initial_train=int(initial),n_boot=int(boot) if advanced else 0,seed=int(seed))
        if result['reliability']['status']!='computed':
            st.error(result['reliability'].get('error','Analyse indisponible'))
        else:
            st.session_state['science_analysis']=result
    except Exception as exc:
        st.error(str(exc))
result=st.session_state.get('science_analysis')
if result:
    r=result['reliability']
    st.success('Modèle retenu : '+LABELS.get(r['distribution'],r['distribution']))
    st.json(r['selection'])
    notices(r)
    for name in ['fit_candidates','trend_results','dependence_results','predictive_validation','reliability_summary']:
        st.subheader(name.replace('_',' ').capitalize())
        st.dataframe(result['tables'][name],hide_index=True)
    st.caption('Les p-valeurs KS/CvM nominales ne remplacent pas les p-valeurs bootstrap avec réestimation. Un non-rejet ne prouve pas le modèle.')
    scientific_charts(inputs,result)
    with st.expander('Contexte et résultats bootstrap complets'):
        st.write(r['context'])
        st.write(r.get('bootstrap',{}))
