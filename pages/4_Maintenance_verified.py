import streamlit as st
from core.scientific_ui import start,analysis_required,show_optimization,plan
start('Maintenance','pages/4_Maintenance_verified.py')
inputs,result=analysis_required()
o=st.session_state.get('science_optimization')
if not o:
    st.info('Calculez les horizons dans Optimisation.')
    st.stop()
show_optimization(o)
st.subheader('Planification indicative')
p=plan(inputs,o)
st.write(p)
st.caption('Les jours restants sont calculés à la consultation en UTC. L’échéance reste rattachée à la fin d’observation documentée. Une échéance dépassée ne signifie pas qu’une panne aurait dû survenir.')
st.write('Avant de retenir une intervention, examiner les anomalies techniques, la criticité, les possibilités de secours et les délais de mobilisation. Aucun score arbitraire ni diagnostic d’usure automatique n’est appliqué.')
