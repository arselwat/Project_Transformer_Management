import streamlit as st
from core.scientific_ui import start,analysis_required,show_optimization,export_bundle,plan,notices
start('Résultats et exports','pages/5_Resultat_analyse_optimisation_Maintenance_fixed.py')
inputs,result=analysis_required()
r=result['reliability']
st.caption('Identifiant de l’analyse : '+r['analysis_hash'])
notices(r)
for name,df in result['tables'].items():
    with st.expander(name.replace('_',' ').capitalize(),expanded=name=='fit_candidates'):
        st.dataframe(df)
o=st.session_state.get('science_optimization')
if o:
    show_optimization(o)
    st.write(plan(inputs,o))
else:
    st.info('Export de l’analyse seule : optimisation non calculée.')
st.download_button('Télécharger les résultats (CSV + JSON)',export_bundle(inputs,result,o),'resultats_scientifiques.zip','application/zip')
st.caption('Cet export utilise directement les tableaux du moteur commun. Il remplace les anciens exports scientifiques qui recalculaient leurs propres valeurs.')
