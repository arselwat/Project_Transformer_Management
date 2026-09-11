"""Parcours scientifique de présentation, version cohérente 4.0."""
import io,json,os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from .interpretation import confidence,confidence_band,model_table,recommendation,GLOSSARY
from .shared import start,input_context,analysis_required,cached_analysis,notices,plan,export_bundle,LABELS,invalidate
from .source_import import read_source
from .organigram import conditional_reliability
from .optimize import optimize_maintenance,cost_rate_nhpp
from .exports import export_tables,excel_bytes,pdf_bytes,json_safe
from core.datahub import get_current_project_data,get_current_failures_df,set_current_project_data,set_current_failures_df,clear_current_project_data,prepare_event_data
PATHS=['1_Sources_fully_linked_fixed.py','2_Indicateurs_verified.py','3_Optimisation_verified.py','4_Maintenance_verified.py','5_Resultat_analyse_optimisation_Maintenance_fixed.py']
TITLES=['Sources de données','Analyse de fiabilité','Optimisation de maintenance','Plan de maintenance','Rapport et résultats']
COLORS=['#35b9b1','#6495ed','#efa858','#ca8eff','#ef7384']
def notices(r):
    accepted=r.get('goodness',{}).get('accepted',r.get('validation_accepted'))
    if accepted is False:
        st.warning('Les tests d’adéquation rejettent l’ajustement. Les prévisions doivent être confrontées à l’état technique du transformateur.')
    elif accepted is None:
        st.info('Adéquation non établie : consulter la validation statistique avant de retenir le calendrier.')

def fmt(x):
    try: return f'{float(x):,.1f}'.replace(',',' ') if np.isfinite(float(x)) else '—'
    except (TypeError,ValueError): return '—'
def cards(items):
    for col,(label,value) in zip(st.columns(len(items)),items):
        with col: st.metric(label,value)
def chart(title,x,ys,xlabel='Horizon supplémentaire (h)',ylabel='',scatter=False):
    fig=go.Figure()
    for n,(name,y) in enumerate(ys.items()):
        fig.add_trace(go.Scatter(x=x,y=y,name=name,mode='markers' if scatter else 'lines',line=dict(color=COLORS[n%len(COLORS)],width=2.5),marker=dict(size=7)))
    fig.update_layout(title=title,height=330,margin=dict(l=25,r=20,t=60,b=45),xaxis_title=xlabel,yaxis_title=ylabel,legend=dict(orientation='h'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
def data_table(df):
    clean=df.copy();clean.attrs={}
    for c in clean:
        if clean[c].dtype=='object': clean[c]=clean[c].map(lambda x:json.dumps(json_safe(x),ensure_ascii=False) if isinstance(x,(list,dict,tuple)) else x)
    st.dataframe(clean,use_container_width=True,hide_index=True)
def figures(inputs,r):
    x=np.asarray(inputs['ttf_series'],float)
    left,right=st.columns(2)
    with left: chart('Chronologie des pannes',np.r_[0,np.cumsum(x)],{'Nombre cumulé':np.arange(len(x)+1)},'Temps depuis la première panne (h)','Pannes')
    with right: chart('Espacement des événements',np.arange(1,len(x)+1),{'Intervalles':x},'Rang de l’intervalle','Heures')
    chart('Dépendance entre intervalles',x[:-1],{'Couples successifs':x[1:]},'Intervalle i (h)','Intervalle i+1 (h)',True)
def reliability_plots(r):
    c=r['curves'].replace([np.inf,-np.inf],np.nan)
    for row in [[('R_t','Fiabilité conditionnelle','Probabilité'),('F_t','Probabilité de panne','Probabilité')],[('f_t','Densité du prochain délai','1/h'),('h_t','Risque conditionnel / intensité NHPP','1/h')]]:
        for col,(key,title,unit) in zip(st.columns(2),row):
            with col: chart(title,c['t'],{title:c[key]},ylabel=unit)
    st.caption('La référence est la fin documentée de l’observation. Les courbes représentent le futur conditionnel, pas un redémarrage à neuf.')
def source():
    df=get_current_failures_df();project=get_current_project_data()
    cards([('Équipements',str(df.equipment_code.nunique()) if not df.empty else '0'),('Intervalles actifs',str(len(df))),('Origine des durées','Dates ou intervalles')])
    tabs=st.tabs(['Importer un historique','Données et qualité','Période d’observation'])
    with tabs[0]:
        st.write('Importez vos fichiers habituels. Les dates sont automatiquement converties en intervalles calendaires.')
        st.code('equipment_code,timestamp,is_failure,repair_time_hours\nTR001,2024-01-10 08:00,1,0\nTR001,2024-06-15 09:30,1,4',language='text')
        up=st.file_uploader('Fichier CSV ou Excel',type=['csv','xlsx'])
        if up:
            try:
                frames=read_source(up.getvalue(),up.name)
                for name,frame in frames.items():
                    st.write(name);data_table(frame.head(100))
                if st.button('Utiliser ces données',type='primary'):
                    if 'intervals' in frames:
                        d=frames['intervals'];values=pd.to_numeric(d.ttf_h,errors='coerce')
                        if not (values.notna() & values.gt(0)).any(): raise ValueError('Aucun intervalle positif valide. Import annulé.')
                        clear_current_project_data(clear_failures=True);set_current_failures_df(d,up.name)
                    else:
                        preview=prepare_event_data(frames['events_history'],frames.get('analysis_settings'))
                        if preview['failures_ttf'].empty:
                            data_table(preview['data_quality_report']);raise ValueError('Aucun intervalle exploitable. Corrigez les anomalies signalées.')
                        set_current_project_data(frames,up.name)
                    invalidate();st.success('Import enregistré. Ouvrez Analyse de fiabilité pour lancer le calcul.');st.rerun()
            except Exception as exc: st.error(str(exc))
    with tabs[1]:
        st.subheader('Intervalles retenus');data_table(df)
        st.download_button('Télécharger les intervalles CSV',df.to_csv(index=False).encode('utf-8-sig'),'intervalles.csv','text/csv')
        for name in ['events_validated','data_quality_report']:
            if name in project: st.write('Événements validés' if name=='events_validated' else 'Contrôles de qualité');data_table(project[name])
    with tabs[2]:
        windows=project.get('observation_windows',pd.DataFrame());data_table(windows)
        if not windows.empty:
            asset=st.selectbox('Équipement à actualiser',windows.asset_id.astype(str).tolist())
            row=windows[windows.asset_id.astype(str)==asset].iloc[0]
            with st.form('window'):
                end=st.text_input('Fin documentée',str(row.observation_end))
                st.caption('La date doit correspondre à une période réellement observée. Elle ne peut précéder la dernière panne.')
                apply=st.form_submit_button('Enregistrer cette période')
            if apply:
                try:
                    end=pd.Timestamp(end)
                    settings=project.get('analysis_settings',pd.DataFrame()).copy()
                    if 'asset_id' in settings: settings=settings[settings.asset_id.astype(str)!=asset]
                    new={'asset_id':asset,'observation_end':end,'decision_time':end}
                    settings=pd.concat([settings,pd.DataFrame([new])],ignore_index=True)
                    check=prepare_event_data(project['events_history'],settings)
                    if asset not in check['observation_windows'].get('asset_id',pd.Series(dtype=str)).astype(str).tolist(): raise ValueError('Fenêtre non valide pour cet équipement.')
                    project['analysis_settings']=settings;set_current_project_data(project,'mise à jour période');invalidate();st.success('Période actualisée. Relancez l’analyse.');st.rerun()
                except Exception as exc: st.error(str(exc))
def indicators():
    inputs=input_context()
    with st.expander('Paramètres de calcul',expanded=not bool(st.session_state.get('science_analysis'))):
        with st.form('analysis_v4'):
            a,b=st.columns(2)
            with a:
                rule=st.selectbox('Critère de sélection',['NHPP — réparation minimale','AICc (comparaison)','Scénario explicite','Prévision séquentielle'])
                model=st.selectbox('Modèle du scénario',list(LABELS),index=1,format_func=LABELS.get)
            with b:
                alpha=st.number_input('Seuil des tests',.001,.2,.05,format='%.3f')
                heavy=st.checkbox('Autoriser la validation prédictive ou le bootstrap des modèles alternatifs',value=False)
                boot=st.number_input('Réplications pour les intervalles de confiance',0,3000,300,100)
            st.caption('Le NHPP représente la réparation minimale retenue dans cette étude : le fonctionnement est rétabli sans remise à zéro de l’âge. Les autres modèles servent à discuter cette hypothèse. Les intervalles de confiance sont calculés par bootstrap avec réestimation.')
            run=st.form_submit_button('Calculer l’analyse',type='primary')
        if run:
            if (rule=='Prévision séquentielle' or (boot and rule!='NHPP — réparation minimale')) and not heavy:
                st.error('Autorisez les validations coûteuses ou choisissez AICc avec 0 réplication.')
            else:
                st.session_state.pop('science_analysis',None);st.session_state.pop('science_optimization',None);st.session_state.pop('maintenance_proposal_v4',None);st.session_state.pop('exports_v4',None)
                with st.spinner('Comparaison des cinq modèles…'):
                    result=cached_analysis(inputs,alpha=alpha,selection_rule='predictive' if rule=='Prévision séquentielle' else 'aicc',selected_model='power_law_nhpp' if rule=='NHPP — réparation minimale' else model if rule=='Scénario explicite' else None,run_sequential=rule=='Prévision séquentielle',n_boot=int(boot))
                if result['reliability']['status']=='computed': st.session_state['science_analysis']=result
                else: st.error(result['reliability'].get('error','Calcul indisponible'))
    result=st.session_state.get('science_analysis')
    if not result: st.info('Les données sont prêtes. Cliquez sur Calculer l’analyse pour afficher les résultats.');return
    r=result['reliability'];ind=r['indicators']
    st.write('**Hypothèse principale :** remise en service sans retour à l’état neuf. Le choix du NHPP est déclaré comme hypothèse de l’étude, distincte du classement AICc.')
    cards([('Intervalles',str(len(inputs['ttf_series']))),('Intervalle moyen (h)',fmt(np.mean(inputs['ttf_series']))),('MTTR renseigné (h)',fmt(ind.get('mttr_h'))),('Horizon à 80 % (h)',fmt(r['horizon']['hours']))])
    st.subheader(LABELS[r['distribution']]);notices(r)
    tabs=st.tabs(['Vue d’ensemble','Diagnostics','Modèles et paramètres','Courbes fiabilistes','Validation'])
    with tabs[0]:
        figures(inputs,r)
    with tabs[1]:
        st.subheader('Tendance');data_table(result['tables']['trend_results'])
        st.subheader('Dépendance');data_table(result['tables']['dependence_results'])
        st.caption('Une tendance des intervalles n’a pas le même sens qu’une tendance de l’intensité. Une corrélation non significative ne démontre pas l’indépendance.')
    with tabs[2]:
        data_table(model_table(r));st.write('Paramètres et intervalles de confiance');ic,b=confidence(r);data_table(ic)
        st.caption('Les renouvellements supposent une restauration statistique des cycles. Ils ne sont pas utilisés comme hypothèse physique principale du transformateur.')
        rows=[(LABELS[k],v['aicc']) for k,v in r['candidates'].items() if v.get('aicc') is not None]
        if rows:
            f=go.Figure(go.Bar(x=[x[0] for x in rows],y=[x[1] for x in rows],marker_color=COLORS));f.update_layout(title='Comparaison AICc — plus faible = meilleur classement relatif',height=330);st.plotly_chart(f,use_container_width=True)
    with tabs[3]: reliability_plots(r); confidence_display(r)
    with tabs[4]:
        data_table(result['tables']['predictive_validation']);confidence_display(r,draw_band=False)
        st.subheader('Comprendre les indicateurs et les tests');data_table(GLOSSARY)
        st.caption('Les tests KS/CvM nominaux ne remplacent pas le bootstrap avec réestimation. Classement, adéquation et prévision sont trois évaluations distinctes.')
def optimization_charts(r,o):
    if o.get('T_recommended') is None:return
    upper=max(o['T_R'],o.get('T_cost') or 0,o['T_recommended'])*1.25
    u=np.linspace(max(upper/300,1e-6),upper,220);p=r['params'];name=r['distribution']
    a,b=st.columns(2)
    with a: chart('Fiabilité et exigences',u,{'Fiabilité':[conditional_reliability(name,p,float(t)) for t in u],'Cible principale':np.full_like(u,o['R_target']),'Cible économique':np.full_like(u,o['R_min_cost'])},ylabel='Probabilité')
    with b:
        if name in ('expon','power_law_nhpp'):
            beta=1 if name=='expon' else p['beta'];eta=1/p['lambda_hpp_h'] if name=='expon' else p['eta']
            costs=[cost_rate_nhpp(t,beta,eta,o['C_prev'],o['C_corr'],p['reference_time_h'],p.get('origin_offset_h',0)) for t in u]
            chart('Coût moyen prospectif',u,{'Coût / heure':costs},ylabel='Unité de coût / h')
        else:st.info('Renouvellement ou GRP : horizon fiabiliste disponible ; optimisation économique non implémentée pour cette famille.')
    st.caption('La courbe explore aussi des horizons non admissibles. Seul l’horizon retenu respecte les contraintes de ce scénario.')
def optimization():
    inputs,result=analysis_required();r=result['reliability']
    st.write('Modèle utilisé : **'+LABELS[r['distribution']]+'**')
    with st.form('opt_v4'):
        a,b,c=st.columns(3)
        with a: cp=st.number_input('Coût préventif Cp',min_value=.001,value=1.);cc=st.number_input('Coût correctif Cc',min_value=0.,value=5.)
        with b: rt=st.number_input('Cible principale',.01,.999,.8);rc=st.number_input('Cible économique',0.,.999,.7)
        with c: low=st.number_input('Horizon minimal (h)',min_value=0.,value=0.);high=st.number_input('Horizon maximal (0 = libre)',min_value=0.,value=0.)
        run=st.form_submit_button('Calculer les horizons',type='primary')
    if run:
        st.session_state.pop('science_optimization',None);st.session_state.pop('maintenance_proposal_v4',None);st.session_state.pop('exports_v4',None)
        try:st.session_state['science_optimization']=optimize_maintenance(result,cp,cc,rt,rc,min_horizon_h=low,max_horizon_h=high or None)
        except Exception as exc:st.error(str(exc))
    o=st.session_state.get('science_optimization')
    if o:
        cards([('Horizon fiabiliste (h)',fmt(o['T_R'])),('Horizon économique (h)',fmt(o['T_cost'])),('Horizon retenu (h)',fmt(o['T_recommended'])),('Fiabilité retenue',fmt(100*o['reliability_at_recommended'])+' %' if o['reliability_at_recommended'] is not None else '—')]);notices(o)
        optimization_charts(r,o)
        st.write(o.get('selection_reason',o.get('reason','')))
        with st.expander('Détails et traçabilité'):st.json(json_safe(o))
def maintenance():
    inputs,result=analysis_required();o=st.session_state.get('science_optimization')
    if not o:st.info('Calculez les horizons dans Optimisation.');return
    p=plan(inputs,o)
    cards([('Équipement',inputs['asset_id']),('Horizon (jours)',fmt(o['T_recommended']/24) if o.get('T_recommended') is not None else '—'),('Jours restants',fmt(p.get('jours restants')))])
    proposed=recommendation(result['reliability'],o,p)
    st.subheader(proposed['Mode proposé']);st.info(proposed['Action recommandée'])
    st.write('**Priorité :** '+proposed['Priorité proposée']);st.write(proposed['Justification'])
    data_table(pd.DataFrame([p]).drop(columns=['qualification'],errors='ignore'))
    confidence_display(result['reliability'])
    with st.form('proposal'):
        a,b=st.columns(2)
        with a: action=st.selectbox('Action proposée',[proposed['Mode proposé'],'Maintenance conditionnelle avec inspections périodiques','Maintenance préventive planifiée','Diagnostic complémentaire']);owner=st.text_input('Responsable / service')
        with b: priority=st.selectbox('Priorité appréciée par l’exploitant',[proposed['Priorité proposée'],'Normale','Élevée','Urgente']);status=st.selectbox('État de la proposition',['À examiner','À planifier','Planifiée'])
        notes=st.text_area('Justification technique et observations',value=proposed['Justification'])
        save=st.form_submit_button('Enregistrer la proposition',type='primary')
    if save:
        st.session_state.pop('exports_v4',None)
        if not notes.strip():st.error('Renseignez la justification technique.')
        else:
            st.session_state['maintenance_proposal_v4']={'asset':inputs['asset_id'],'analysis_hash':result['reliability']['analysis_hash'],'proposition_automatique':proposed,'action':action,'responsable':owner,'priorite':priority,'etat':status,'justification':notes,'enregistre_le':datetime.now().isoformat(),**p};st.success('Proposition conservée dans cette session. Téléchargez-la pour la conserver après déconnexion.')
    proposal=st.session_state.get('maintenance_proposal_v4')
    if proposal and proposal['analysis_hash']==result['reliability']['analysis_hash']:
        data_table(pd.DataFrame([proposal]));st.download_button('Télécharger la fiche de maintenance',json.dumps(json_safe(proposal),ensure_ascii=False,indent=2),'fiche_maintenance.json')
def reports():
    inputs,result=analysis_required();o=st.session_state.get('science_optimization');r=result['reliability'];proposal=st.session_state.get('maintenance_proposal_v4')
    if proposal and proposal.get('analysis_hash')!=r['analysis_hash']:proposal=None
    st.subheader('Synthèse — '+str(inputs['asset_id']));cards([('Modèle retenu',LABELS[r['distribution']]),('Intervalles analysés',str(len(inputs['ttf_series']))),('Horizon retenu (h)',fmt(o.get('T_recommended')) if o else 'Non calculé')]);notices(r)
    tabs=st.tabs(['Rapport visuel','Tableaux détaillés','Téléchargements'])
    with tabs[0]:reliability_plots(r);optimization_charts(r,o) if o else None
    tables=export_tables(inputs['asset_id'],inputs,result,o,proposal)
    tables['intervalles_confiance']=confidence(r)[0]
    tables['guide_indicateurs']=GLOSSARY
    tables['comparaison_modeles']=model_table(r)
    with tabs[1]:
        for title,df in tables.items():
            with st.expander(title.replace('_',' ').capitalize()):data_table(df)
    with tabs[2]:
        st.write('Les exports utilisent les mêmes résultats que les pages. Aucun modèle n’est réajusté.')
        st.download_button('Dossier CSV + JSON',export_bundle(inputs,result,o),'analyse.zip','application/zip')
        # Génération à la demande, pas à chaque changement de page.
        if st.button('Préparer le rapport PDF et Excel'):
            with st.spinner('Préparation des rapports…'):
                try:st.session_state['exports_v4']={'hash':r['analysis_hash'],'pdf':pdf_bytes(inputs['asset_id'],tables),'xlsx':excel_bytes(tables)}
                except Exception as exc:st.error('Export impossible : '+str(exc))
        exports=st.session_state.get('exports_v4')
        if exports and exports['hash']==r['analysis_hash']:
            st.download_button('Rapport PDF',exports['pdf'],'rapport_fiabilite.pdf','application/pdf');st.download_button('Classeur Excel complet',exports['xlsx'],'resultats.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.caption('Le PDF est une synthèse tabulaire. Les graphiques interactifs sont dans le rapport visuel ; toutes les données des courbes figurent dans les exports.')

def confidence_display(r,draw_band=True):
    table,b=confidence(r)
    level=100*(1-r['context']['alpha'])
    st.subheader('Intervalles de confiance à %g %%'%level)
    if table.empty:
        st.info('Intervalles non disponibles : lancer le calcul avec des réplications bootstrap et vérifier que toutes les simulations aboutissent.')
        return
    data_table(table)
    st.caption('%s simulations réussies sur %s. IC percentiles conditionnels au modèle ; ils ne sont pas des intervalles de prédiction de la date de panne.'%(b.get('successful'),b.get('requested')))
    axis=r['curves']['t'].to_numpy();band=confidence_band(r,axis)
    if band is not None and draw_band:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=axis,y=band[1],mode='lines',line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=axis,y=band[0],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(53,185,177,.22)',name='IC ponctuel à %g %%'%level))
        fig.add_trace(go.Scatter(x=axis,y=r['curves']['R_t'],name='Fiabilité estimée',line=dict(color='#35b9b1',width=3)))
        fig.update_layout(title='Fiabilité future et incertitude paramétrique',xaxis_title='Horizon supplémentaire (h)',yaxis_title='Probabilité de fonctionnement sans panne',height=380)
        st.plotly_chart(fig,use_container_width=True)
    if b.get('accepted') is False:st.warning('L’ajustement est rejeté par au moins un test bootstrap. Les IC restent conditionnels à ce modèle et demandent une interprétation prudente.')

def render(number):
    start(TITLES[number-1],'pages/'+PATHS[number-1])

    [source,indicators,optimization,maintenance,reports][number-1]()
