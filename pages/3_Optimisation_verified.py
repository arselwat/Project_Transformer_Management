from core.scientific_ui import scientific_charts
import streamlit as st
from core.scientific_ui import start,analysis_required,show_optimization
from core.reliability.optimize import optimize_maintenance
start('Optimisation','pages/3_Optimisation_verified.py')
inputs,result=analysis_required()
st.write('Modèle :',result['reliability']['distribution'])
st.caption('Coûts dans la même unité. Les valeurs 1 et 5 sont un scénario normalisé. Le critère PLP/HPP est (Cp + Cc × nombre attendu de pannes) / horizon.')
with st.form('optimization'):
    cp=st.number_input('Coût préventif Cp',min_value=.001,value=1.)
    cc=st.number_input('Coût correctif Cc',min_value=0.,value=5.)
    target=st.number_input('Cible principale',min_value=.01,max_value=.999,value=.8)
    minimum=st.number_input('Cible du scénario économique',min_value=0.,max_value=.999,value=.7)
    low=st.number_input('Horizon minimal (h)',min_value=0.,value=0.)
    high=st.number_input('Horizon maximal (h ; 0 = sans borne métier)',min_value=0.,value=0.)
    run=st.form_submit_button('Calculer les horizons',type='primary')
if run:
    st.session_state.pop('science_optimization',None)
    try:
        st.session_state['science_optimization']=optimize_maintenance(result,cp,cc,target,minimum,min_horizon_h=low,max_horizon_h=high or None)
    except Exception as exc:
        st.error(str(exc))
o=st.session_state.get('science_optimization')
if o:
    show_optimization(o)
    with st.expander('Détail du calcul économique et contraintes'):
        st.json(o)

if o:
    scientific_charts(inputs,result,o)
