"""Interprétation des résultats, incertitude et règles explicites de proposition."""
import numpy as np
import pandas as pd
from .organigram import conditional_reliability
NAMES={'expon':'Poisson homogène (HPP)','power_law_nhpp':'Poisson non homogène (PLP-NHPP)','weibull_2p':'Renouvellement Weibull','lognorm':'Renouvellement lognormal','grp_kijima_i':'Réparation imparfaite (GRP Kijima I)'}
PARAMS={'beta':('β — paramètre de forme','sans unité'),'eta':('η — paramètre d’échelle','h'),'q':('q — facteur de réparation','sans unité'),'mu':('μ — moyenne du logarithme des durées','log(h)'),'sigma':('σ — dispersion logarithmique','sans unité'),'lambda_hpp_h':('λ — intensité constante','1/h'),'horizon_h':('Horizon fiabiliste','h')}
def confidence(r):
    b=r.get('bootstrap',{}).get(r['distribution'],{})
    intervals=b.get('parameter_intervals',{})
    rows=[]
    for key,values in intervals.items():
        label,unit=PARAMS.get(key,(key,''))
        if key=='horizon_h':label+=' à %.0f %% de fiabilité'%(100*b.get('target',.8))
        point=r['horizon']['hours'] if key=='horizon_h' else r['params'].get(key)
        rows.append({'Indicateur':label,'Estimation':point,'Borne inférieure':values[0],'Médiane bootstrap':values[1],'Borne supérieure':values[2],'Unité':unit})
    return pd.DataFrame(rows),b

def confidence_band(r,axis):
    b=r.get('bootstrap',{}).get(r['distribution'],{})
    if b.get('status')!='completed' or r['distribution']!='power_law_nhpp': return None
    draws=b.get('draws',pd.DataFrame())
    if draws.empty:return None
    params=r['params'];pred=[]
    for row in draws.itertuples():
        p=dict(params,beta=row.beta,eta=row.eta)
        pred.append([conditional_reliability('power_law_nhpp',p,float(t)) for t in axis])
    alpha=r['context']['alpha']
    return np.quantile(np.asarray(pred),[alpha/2,1-alpha/2],axis=0)

def model_table(r):
    rows=[]
    for name,f in r.get('candidates',{}).items():
        rows.append({'Modèle':NAMES[name],'Log-vraisemblance':f.get('loglik'),'Paramètres libres':f.get('n_params'),'AICc':f.get('aicc'),'BIC':f.get('bic'),'Estimation':'Réussie' if f.get('status')=='fitted' else 'Échec','Usage':'Modèle principal' if name==r['distribution'] else 'Comparaison'})
    return pd.DataFrame(rows)

def recommendation(r,o,p):
    days=p.get('jours restants');beta=r['params'].get('beta');tab,b=confidence(r)
    ci=b.get('parameter_intervals',{}).get('beta')
    reasons=[]
    if r['distribution']=='power_law_nhpp' and beta is not None and beta>1:
        mode='Maintenance préventive planifiée avec surveillance conditionnelle'
        reasons.append('L’intensité estimée des pannes augmente avec le temps (β = %.3f).'%beta)
        if ci:reasons.append('IC de β : [%.3f ; %.3f]. %s'%(ci[0],ci[2],'La borne inférieure dépasse 1.' if ci[0]>1 else 'L’intervalle inclut des valeurs compatibles avec une intensité non croissante.'))
    else:
        mode='Maintenance conditionnelle avec inspections périodiques'
        reasons.append('Le modèle ne suffit pas à établir une augmentation certaine de l’intensité ; le suivi de l’état reste nécessaire.')
    priority='Planification à examiner'
    action='Programmer une inspection et préparer le plan d’intervention dans l’horizon calculé.'
    if days is not None and days<0:
        priority='Réexamen prioritaire'
        action='Actualiser l’historique et réaliser une inspection de l’état avant de fixer une nouvelle échéance.'
        reasons.append('L’échéance calculée sur l’archive est dépassée : le délai négatif n’est pas une nouvelle prévision de panne.')
    elif days is not None:
        reasons.append('Il reste %.1f jours avant l’échéance indicative.'%days)
    if o.get('T_recommended') is None:
        priority='Analyse à compléter';action='Résoudre les contraintes de planification avant de fixer une date.'
    if r.get('goodness',{}).get('accepted') is not True:
        reasons.append('L’adéquation est non établie ou rejetée : confirmer le calendrier par le diagnostic technique.')
    return {'Mode proposé':mode,'Priorité proposée':priority,'Action recommandée':action,'Justification':' '.join(reasons)}

GLOSSARY=pd.DataFrame([
('Intervalle moyen','Moyenne des durées calendaires entre pannes observées.','h','Ce n’est pas nécessairement le temps réel de fonctionnement.'),
('MTTR','Moyenne des durées de réparation renseignées, y compris les durées nulles.','h','Les valeurs manquantes sont exclues ; vérifier la couverture des données.'),
('Horizon fiabiliste','Durée future maximale respectant la cible de fiabilité à la référence.','h','Ce n’est ni une durée de vie restante certaine ni une date de panne.'),
('β du PLP','Forme de l’évolution temporelle de l’intensité.','—','β > 1 : intensité croissante ; β = 1 : constante ; β < 1 : décroissante. Consulter l’IC.'),
('η du PLP','Échelle de la fonction moyenne cumulée Λ(t) = (t/η)^β.','h','η n’est pas le temps moyen entre les pannes.'),
('q du GRP','Coefficient d’accumulation d’âge virtuel dans Kijima I.','—','q proche de 1 : réparation minimale ; q proche de 0 : forte restauration statistique, pas un rajeunissement physique.'),
('Log-vraisemblance','Mesure de l’ajustement du modèle aux observations.','—','Plus élevée : meilleur ajustement brut ; ne pénalise pas le nombre de paramètres.'),
('Paramètres libres','Nombre de paramètres estimés.','—','Entre dans la pénalisation AICc et BIC.'),
('AICc','Compromis ajustement/complexité corrigé pour petit échantillon.','—','Plus faible : meilleur classement relatif ; ne prouve pas l’adéquation.'),
('BIC','Critère de comparaison pénalisant la complexité.','—','Plus faible : classement plus favorable selon ce critère.'),
('Mann–Kendall','Diagnostic de tendance des intervalles selon leur rang.','—','Tendance négative : intervalles qui se raccourcissent.'),
('Laplace','Diagnostic de tendance des instants de panne.','—','Dépend de la convention de fin d’observation.'),
('Pearson / Spearman','Association entre intervalles successifs.','—','Une absence de significativité ne prouve pas l’indépendance.'),
('p-valeur','Compatibilité statistique avec l’hypothèse testée.','—','p < α : rejet au seuil choisi ; p ≥ α : non-rejet, pas preuve.'),
('KS / Cramér–von Mises','Diagnostics d’écart des résidus à la distribution attendue.','—','Après estimation des paramètres, utiliser la calibration bootstrap.'),
('IC bootstrap','Variabilité des estimations par simulation et réestimation.','Unité du paramètre','Conditionnel au modèle et au protocole ; ne couvre pas toute l’incertitude physique.'),
('Log-score prédictif','Qualité de la densité prédite pour les événements suivants.','—','Plus élevé : meilleure prévision selon le protocole temporel utilisé.'),
('Disponibilité conventionnelle','Ratio calculable seulement avec les hypothèses et durées adaptées.','%','Ne pas confondre avec une disponibilité réellement mesurée sur toute la période.'),
],columns=['Indicateur','Définition','Unité','Lecture et limites'])
