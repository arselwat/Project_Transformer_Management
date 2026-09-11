"""Moteur de fiabilité des systèmes réparables — version 2.0.

Installation : remplacer core/reliability/organigram.py par ce fichier.
Dépendances : numpy, pandas, scipy. Aucun import Streamlit, aucun accès disque.

Conventions
-----------
Les x_i sont des intervalles strictement positifs, dans l'ordre chronologique.
Le premier événement de l'archive est la référence t=0, conditionnée : il ne
constitue pas un intervalle supplémentaire et n'entre pas dans la vraisemblance.
observation_end_h est mesuré depuis cette référence et inclut, s'il existe,
le temps censuré après la dernière panne. Aucun intervalle invalide n'est retiré
silencieusement. Sans contexte explicite, le temps est déclaré « unspecified ».

Candidats : HPP, PLP-NHPP, renouvellement Weibull et lognormal, GRP Kijima I
de base Weibull, V_i=V_(i-1)+q*x_i, V_0=0, 0<=q<=1.
La sélection AICc, l'adéquation et la performance prédictive sont distinctes.
Les tests KS/CvM nominaux après estimation ne valident pas un modèle.
Le bootstrap avec réestimation et la comparaison séquentielle sont optionnels
car coûteux : n_boot=0 et run_sequential=False par défaut.

Les signatures principales et les noms de tables historiques sont conservés.
Les pages qui recalculent leurs propres courbes/horizons devront être raccordées
aux fonctions conditionnelles ci-dessous ; ce fichier ne modifie pas ces pages.

Exemple :
    out = analyze_ttf_pipeline(x, repair_series=rep,
        time_basis='calendar', run_sequential=True, initial_train=8,
        selection_rule='predictive', n_boot=3000, seed=20260823)
    r = out['reliability']
    h = conditional_horizon(r['distribution'], r['params'], 0.8)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy.optimize import minimize, minimize_scalar, brentq

ENGINE_VERSION = "2.0"
MODEL_NAMES = ("expon", "power_law_nhpp", "weibull_2p", "lognorm", "grp_kijima_i")
SHAPE_BOUNDS = (0.05, 30.0)


def _clean_positive(series):
    """Nom historique ; validation stricte, sans supprimer ou trier les données."""
    if series is None:
        return np.array([], dtype=float)
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("Les intervalles doivent être un vecteur fini, strictement positif et chronologique.")
    return x.copy()


def _to_event_times(series):
    return np.cumsum(_clean_positive(series))


def _safe_float(value, default=None):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _round_df(df, digits=6):
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(digits)
    return out


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement compris entre 0 et 1.")


def _window(x, observation_end_h=None):
    last = float(np.sum(x))
    end = last if observation_end_h is None else float(observation_end_h)
    if not np.isfinite(end) or end <= 0 or (end < last and not math.isclose(end,last,rel_tol=1e-12,abs_tol=1e-9)):
        raise ValueError("La fin d'observation doit être finie et >= à la dernière panne.")
    if math.isclose(end,last,rel_tol=1e-12,abs_tol=1e-9):
        end=last
    return last, end, end-last


def _strength_label(r):
    if r is None:
        return "unavailable"
    return "very_low" if abs(r)<.2 else "low" if abs(r)<.4 else "medium" if abs(r)<.6 else "high" if abs(r)<.8 else "very_high"


def graphical_trend_test(ttf_series):
    t = _to_event_times(ttf_series)
    if len(t)<3:
        return dict(beta_graph=None, r2=None, has_trend=None, direction="unavailable")
    slope, intercept, r, _, _ = sst.linregress(np.log(t), np.log(np.arange(1,len(t)+1)))
    direction = "up" if slope>1.05 else "down" if slope<.95 else "none"
    return dict(beta_graph=float(slope), slope_loglog=float(slope), intercept_loglog=float(intercept),
                r2=float(r*r), direction=direction, graphical_signal=direction,
                has_trend=direction!="none", method="descriptive_log_count",
                note="Les seuils graphiques sont descriptifs, sans test d'hypothèse.")


def mann_kendall_test(series, alpha=.05):
    _alpha(alpha)
    x=_clean_positive(series); n=len(x)
    if n<3:
        return dict(z=None,p=None,has_trend=None,direction="unavailable")
    S=sum(float(np.sign(x[i+1:]-x[i]).sum()) for i in range(n-1))
    counts=np.unique(x,return_counts=True)[1]
    var=(n*(n-1)*(2*n+5)-np.sum(counts*(counts-1)*(2*counts+5)))/18
    if var<=0:
        return dict(z=None,p=None,has_trend=None,direction="unavailable",reason="Série constante.")
    z=(S-np.sign(S))/math.sqrt(var)
    p=float(2*sst.norm.sf(abs(z)))
    direction="up" if p<alpha and z>0 else "down" if p<alpha and z<0 else "none"
    return dict(z=float(z),p=p,has_trend=p<alpha,direction=direction,
                direction_basis="interval_length", calibration="normal_approximation")


def laplace_trend_test(ttf_series, alpha=.05, *, observation_end_h=None,
                       observation_scheme="auto"):
    _alpha(alpha); x=_clean_positive(ttf_series); t=np.cumsum(x)
    if len(t)<3:
        return dict(u=None,p=None,has_trend=None,direction="unavailable")
    _,end,_=_window(x,observation_end_h)
    scheme=_scheme(observation_scheme,end,float(t[-1]))
    # Arrêt à la n-ième panne : les n-1 instants intérieurs sont utilisés.
    interior=t[:-1] if scheme=="event_terminated" else t
    u=math.sqrt(12*len(interior))*(float(interior.mean())/end-.5)
    p=float(2*sst.norm.sf(abs(u)))
    legacy=math.sqrt(12*len(t))*(float(t.mean())/end-.5)
    return dict(u=u,p=p,has_trend=p<alpha,
                direction="up" if p<alpha and u>0 else "down" if p<alpha and u<0 else "none",
                observation_scheme=scheme,calibration="normal_approximation",
                legacy_endpoint_included_u=legacy,
                legacy_endpoint_included_p=float(2*sst.norm.sf(abs(legacy))))


def combine_trend_evidence(graphical,mk,lap,alpha=.05):
    _alpha(alpha)
    votes=[]
    if mk.get("has_trend"):
        # Des intervalles décroissants correspondent à un rythme croissant.
        votes.append({"up":"down","down":"up"}.get(mk.get("direction"),"none"))
    if lap.get("has_trend"):
        votes.append(lap["direction"])
    available=mk.get("p") is not None or lap.get("p") is not None
    direction=(votes[0] if len(set(votes))==1 else "conflicting") if votes else "none"
    return dict(has_trend=bool(votes) if available else None,direction=direction,
                confidence="diagnostic_only",graphical_direction=graphical.get("direction"),
                mk_sig=mk.get("has_trend"),lap_sig=lap.get("has_trend"),
                reason="Diagnostics exploratoires : ils ne sélectionnent pas automatiquement une famille de modèles.")


def graphical_dependence_test(series):
    x=_clean_positive(series)
    if len(x)<4 or np.ptp(x[:-1])==0 or np.ptp(x[1:])==0:
        return dict(lag1_r=None,slope=None,intercept=None,r2=None,direction="unavailable",has_dependence=None)
    slope,intercept,r,_,_=sst.linregress(x[:-1],x[1:])
    return dict(lag1_r=float(r),slope=float(slope),intercept=float(intercept),r2=float(r*r),
                direction="positive" if r>.2 else "negative" if r<-.2 else "none",
                has_dependence=abs(r)>=.2,strength=_strength_label(r),method="descriptive_lag_plot")


def dependence_correlation_test(series,alpha=.05):
    _alpha(alpha); x=_clean_positive(series)
    if len(x)<4 or np.ptp(x[:-1])==0 or np.ptp(x[1:])==0:
        return dict(r=None,p=None,has_dep=None,pearson_r=None,pearson_p=None,
                    spearman_r=None,spearman_p=None,strength="unavailable",method="spearman")
    pr,pp=sst.pearsonr(x[:-1],x[1:]); sr,sp=sst.spearmanr(x[:-1],x[1:])
    # Spearman prédéfini : pas de choix opportuniste du plus grand coefficient.
    return dict(r=float(sr),p=float(sp),has_dep=bool(sp<alpha),method="spearman",
                pearson_r=float(pr),pearson_p=float(pp),spearman_r=float(sr),spearman_p=float(sp),
                strength=_strength_label(sr),calibration="nominal_exploratory",
                note="Paires de retard chevauchantes ; absence de rejet ne prouve pas l'indépendance.")


def combine_dependence_evidence(graphical,corr,alpha=.05):
    return {**corr,"graphical_direction":graphical.get("direction"),
            "graphical_r":graphical.get("lag1_r"),
            "reason":"Une corrélation ne démontre pas un processus de branchement."}


def _scheme(value,end,last):
    same_end=math.isclose(end,last,rel_tol=1e-12,abs_tol=1e-9)
    if value=="auto":
        return "event_terminated" if same_end else "time_terminated"
    if value not in ("event_terminated","time_terminated"):
        raise ValueError("observation_scheme : auto, event_terminated ou time_terminated.")
    if value=="event_terminated" and not same_end:
        raise ValueError("Un arrêt sur événement ne peut inclure de censure finale.")
    return value


def _power_delta(a,b,k):
    """b**k-a**k avec a>=0, b>=a ; protège la soustraction de valeurs proches."""
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        value=np.where(a>0,np.exp(k*np.log(a))*np.expm1(k*np.log1p((b-a)/a)),b**k)
    return value


def _blank_fit(name,error=None):
    return dict(name=name,params=None,loglik=None,aic=None,aicc=None,bic=None,
                n_params=None,ks_p=None,cvm_p=None,chi2_p=None,accepted=None,
                status="failed",error=error,estimation_method="MLE",
                validation_status="not_calibrated",warnings=[])


def _finish_fit(fit,x,k,ll,residuals):
    n=len(x)
    if not np.isfinite(ll) or not np.all(np.isfinite(residuals)):
        raise ValueError("Vraisemblance ou résidus non finis.")
    aic=2*k-2*ll
    fit.update(status="fitted",loglik=float(ll),n_params=k,n=n,aic=float(aic),
               aicc=float(aic+2*k*(k+1)/(n-k-1)) if n>k+1 else None,
               bic=float(k*np.log(n)-2*ll),residuals=np.asarray(residuals,float).tolist())
    if n>=3:
        ks=sst.kstest(residuals,"expon"); cvm=sst.cramervonmises(residuals,"expon")
        fit.update(ks_stat=float(ks.statistic),ks_p=float(ks.pvalue),
                   cvm_stat=float(cvm.statistic),cvm_p=float(cvm.pvalue))
    fit["nominal_test_note"]="Résidus transformés ; p-valeurs nominales après estimation, non calibrées."
    fit["chi2_note"]="Non calculé : aucune p-valeur fictive pour un nombre insuffisant de classes."
    return fit


def _profile_weibull(x,censor,q,k):
    prev=q*np.r_[0.,np.cumsum(x)[:-1]]
    age=prev+x
    S=float(np.sum(_power_delta(prev,age,k)))
    if censor>0:
        v=q*float(x.sum()); S+=float(_power_delta(v,v+censor,k))
    if S<=0 or not np.isfinite(S):
        return -np.inf,None,None
    eta=(S/len(x))**(1/k)
    ll=len(x)*np.log(k)-len(x)*np.log(S/len(x))+(k-1)*np.log(age).sum()-len(x)
    return float(ll),float(eta),_power_delta(prev/eta,age/eta,k)


def fit_model(name,ttf_series,*,observation_end_h=None,origin_offset_h=0.0):
    """Vraisemblances comparables, incluant la survie de l'intervalle final censuré.

    origin_offset_h déplace uniquement l'âge du PLP, conditionnellement au début
    d'observation. Il ne suppose pas l'absence de pannes avant l'archive.
    Les modèles RP et GRP supposent respectivement âge initial nul et V0=0.
    """
    fit=_blank_fit(name)
    try:
        x=_clean_positive(ttf_series); n=len(x)
        if n<2:
            raise ValueError("Au moins deux intervalles sont requis pour l'ajustement.")
        last,end,censor=_window(x,observation_end_h)
        if not np.isfinite(origin_offset_h) or origin_offset_h<0:
            raise ValueError("Le décalage d'origine doit être fini et positif ou nul.")
        fit.update(T_end=end,history_time_h=last,origin_offset_h=float(origin_offset_h))
        scale=float(x.mean()); y=x/scale; c=censor/scale; t=np.cumsum(x)
        if name=="expon":
            rate=n/end; raw=(0.,1/rate)
            fit.update(params=raw,lambda_h=rate,lambda_hpp_h=rate)
            return _finish_fit(fit,x,1,n*np.log(rate)-rate*end,rate*x)
        if name=="power_law_nhpp":
            off=origin_offset_h/scale; end_s=end/scale; times=t/scale+off
            def profile(k):
                S=float(_power_delta(off,off+end_s,k))
                if not np.isfinite(S) or S<=0:
                    return -np.inf,None
                eta=(S/n)**(1/k)
                ll=n*np.log(k)-n*np.log(S/n)+(k-1)*np.log(times).sum()-n
                return float(ll),eta
            if off==0:
                den=float(np.log(end/t).sum())
                if den<=0: raise ValueError("PLP non identifiable.")
                k=n/den
            else:
                opt=minimize_scalar(lambda z:-profile(np.exp(z))[0],bounds=np.log(SHAPE_BOUNDS),method="bounded")
                if not opt.success: raise ValueError("Échec d'estimation PLP.")
                k=float(np.exp(opt.x))
            ll,eta_s=profile(k); eta=eta_s*scale
            z=_power_delta((origin_offset_h+np.r_[0,t[:-1]])/eta,(origin_offset_h+t)/eta,k)
            fit.update(params=(k,eta),beta=k,eta=eta)
            if k<=SHAPE_BOUNDS[0]*1.001 or k>=SHAPE_BOUNDS[1]*.999:
                fit["warnings"].append("Paramètre de forme extrême ou proche de la borne numérique.")
            return _finish_fit(fit,x,2,ll-n*np.log(scale),z)
        if name in ("weibull_2p","grp_kijima_i"):
            if np.ptp(y)<1e-12:
                raise ValueError("Intervalles constants : maximum Weibull non fini en forme.")
            if name=="weibull_2p":
                opt=minimize_scalar(lambda z:-_profile_weibull(y,c,0.,np.exp(z))[0],
                                    bounds=np.log(SHAPE_BOUNDS),method="bounded")
                if not opt.success: raise ValueError("Échec Weibull.")
                k=float(np.exp(opt.x)); q=0.; npar=2
            else:
                # Multidépart sur q : indispensable lorsque plusieurs extrema existent.
                trials=[]
                def objective(theta):
                    ll=_profile_weibull(y,c,float(theta[1]),float(np.exp(theta[0])))[0]
                    return -ll if np.isfinite(ll) else 1e100
                for k0 in (1.,3.,6.):
                    for q0 in (0.,.05,.3,1.):
                        opt=minimize(objective,[np.log(k0),q0],method="L-BFGS-B",
                                     bounds=[tuple(np.log(SHAPE_BOUNDS)),(0.,1.)],
                                     options={"maxiter":1000,"ftol":1e-12})
                        if opt.success and np.isfinite(opt.fun): trials.append(opt)
                if not trials: raise ValueError("Aucun multidépart GRP n'a convergé.")
                best=min(trials,key=lambda v:v.fun); k=float(np.exp(best.x[0])); q=float(best.x[1]); npar=3
                fit["successful_starts"]=len(trials)
            ll,eta_s,z=_profile_weibull(y,c,q,k); eta=eta_s*scale
            raw=(k,0.,eta) if name=="weibull_2p" else (k,eta,q)
            fit.update(params=raw,beta=k,eta=eta,gamma=0.,q=q,
                       virtual_age_h=q*last if name=="grp_kijima_i" else 0.)
            if k<=SHAPE_BOUNDS[0]*1.001 or k>=SHAPE_BOUNDS[1]*.999:
                fit["warnings"].append("Forme proche d'une borne ; vérifier l'identifiabilité et les bornes.")
            if name=="grp_kijima_i" and (q<1e-6 or q>1-1e-6):
                fit["warnings"].append("q sur une frontière : AICc et incertitude demandent prudence.")
            return _finish_fit(fit,x,npar,ll-n*np.log(scale),z)
        if name=="lognorm":
            logs=np.log(y); mu=float(logs.mean()); sigma=float(logs.std(ddof=0))
            if sigma<1e-10: raise ValueError("Lognormale non identifiable sur une série constante.")
            if c>0:
                def objective(v):
                    sig=np.exp(v[1]); ll=sst.lognorm.logpdf(y,sig,scale=np.exp(v[0])).sum()
                    ll+=sst.lognorm.logsf(c,sig,scale=np.exp(v[0]))
                    return -float(ll) if np.isfinite(ll) else 1e100
                opt=minimize(objective,[mu,np.log(sigma)],method="L-BFGS-B",
                             bounds=[(-30.,30.),(-8.,4.)])
                if not opt.success: raise ValueError("Échec lognormal censuré.")
                mu=float(opt.x[0]); sigma=float(np.exp(opt.x[1]))
            raw=(sigma,0.,float(np.exp(mu)*scale))
            ll=float(sst.lognorm.logpdf(x,*raw).sum())
            if censor>0: ll+=float(sst.lognorm.logsf(censor,*raw))
            fit.update(params=raw,mu=float(mu+np.log(scale)),sigma=sigma)
            return _finish_fit(fit,x,2,ll,-sst.lognorm.logsf(x,*raw))
        raise ValueError("Famille non prise en charge : "+str(name))
    except (ValueError,TypeError,FloatingPointError,OverflowError) as exc:
        fit.update(status="failed",error=str(exc))
        return fit


def fit_power_law_nhpp(ttf_series,alpha=.05,**kwargs):
    _alpha(alpha)
    return fit_model("power_law_nhpp",ttf_series,**kwargs)


def fit_grp_kijima_i(ttf_series,**kwargs):
    return fit_model("grp_kijima_i",ttf_series,**kwargs)


def _fit_distribution(name,data,alpha=.05):
    _alpha(alpha)
    return fit_model(name,data)


def fit_hawkes_bpp(ttf_series,alpha=.05):
    """Ancien point d'entrée conservé ; BPP retiré du parcours validé."""
    return _blank_fit("hawkes_exp_bpp","Branche BPP désactivée : une corrélation ne justifie pas ce modèle. Utiliser la comparaison des cinq candidats.")


def weibull_probability_plot_ls(data):
    x=np.sort(_clean_positive(data)); n=len(x)
    if n<3 or np.ptp(x)==0:
        return dict(beta_ls=None,eta_ls=None,r2=None)
    p=(np.arange(1,n+1)-.3)/(n+.4)
    slope,intercept,r,_,_=sst.linregress(np.log(x),np.log(-np.log1p(-p)))
    return dict(beta_ls=float(slope),eta_ls=float(np.exp(-intercept/slope)),r2=float(r*r))


def fit_and_compare_distributions(data,alpha=.05):
    _alpha(alpha)
    fits={name:fit_model(name,data) for name in ("expon","weibull_2p","lognorm")}
    valid={name:f for name,f in fits.items() if f["status"]=="fitted" and f["aicc"] is not None}
    if not valid:
        return dict(best_name=None,best={},all=fits,selected_by=None,weibull=None,weibull_ls=None,hpp=None)
    name=min(valid,key=lambda v:valid[v]["aicc"]); f=valid[name]
    return dict(best_name=name,best=f,all=fits,selected_by="min_aicc",
                weibull={"beta":f["beta"],"eta":f["eta"],"gamma":0.} if name=="weibull_2p" else None,
                weibull_ls=weibull_probability_plot_ls(data) if name=="weibull_2p" else None,
                hpp={"lambda_h":f["lambda_h"]} if name=="expon" else None)


def _parameters(fit,x,reference_time_h=None):
    last=float(np.sum(x)); end=fit.get("T_end",last)
    ref=end if reference_time_h is None else float(reference_time_h)
    if not np.isfinite(ref) or ref<last:
        raise ValueError("Le temps de référence doit être fini et >= à la dernière panne.")
    return {"raw":fit.get("params"),"beta":fit.get("beta"),"eta":fit.get("eta"),
            "gamma":fit.get("gamma",0.),"mu":fit.get("mu"),"sigma":fit.get("sigma"),
            "q":fit.get("q"),"lambda_hpp_h":fit.get("lambda_hpp_h"),
            "alpha":None,"beta_kernel":None,"branch_ratio":None,"beta_ls":None,"eta_ls":None,
            "history_time_h":last,"observation_end_h":end,"reference_time_h":ref,
            "origin_offset_h":fit.get("origin_offset_h",0.),
            "age_since_last_failure_h":ref-last,
            "virtual_age_h":fit.get("virtual_age_h",0.)}


def _conditional_logsurvival(name,params,u):
    u=np.asarray(u,float)
    if np.any(~np.isfinite(u)) or np.any(u<0): raise ValueError("Horizon non négatif fini requis.")
    if name=="expon":
        rate=params.get("lambda_hpp_h")
        if rate is None: rate=1/params["raw"][1]
        return -float(rate)*u
    if name=="power_law_nhpp":
        age=float(params.get("reference_time_h",0.))+float(params.get("origin_offset_h",0.))
        return -_power_delta(age/params["eta"],(age+u)/params["eta"],params["beta"])
    if name in ("weibull_2p","grp_kijima_i"):
        age=float(params.get("age_since_last_failure_h",0.))
        if name=="grp_kijima_i": age+=float(params.get("virtual_age_h",0.))
        return -_power_delta(age/params["eta"],(age+u)/params["eta"],params["beta"])
    if name=="lognorm":
        age=float(params.get("age_since_last_failure_h",0.)); raw=params["raw"]
        return sst.lognorm.logsf(age+u,*raw)-sst.lognorm.logsf(age,*raw)
    raise ValueError("Prédiction non prise en charge pour "+str(name))


def conditional_reliability(distribution,params,horizon_h):
    """Probabilité de zéro panne future, conditionnelle à l'histoire fournie."""
    value=np.exp(_conditional_logsurvival(distribution,params,horizon_h))
    return float(value) if np.ndim(value)==0 else value


def conditional_horizon(distribution,params,target=.8):
    if not np.isfinite(target) or not 0<target<1:
        raise ValueError("La fiabilité cible doit être strictement comprise entre 0 et 1.")
    goal=-np.log(target)
    if distribution=="expon":
        rate=params.get("lambda_hpp_h") or 1/params["raw"][1]
        return float(goal/rate)
    if distribution in ("power_law_nhpp","weibull_2p","grp_kijima_i"):
        if distribution=="power_law_nhpp":
            age=params.get("reference_time_h",0.)+params.get("origin_offset_h",0.)
        else:
            age=params.get("age_since_last_failure_h",0.)
            if distribution=="grp_kijima_i": age+=params.get("virtual_age_h",0.)
        k=params["beta"]; eta=params["eta"]
        if age==0: return float(eta*goal**(1/k))
        return float(age*np.expm1(np.log1p(goal/(age/eta)**k)/k))
    if distribution=="lognorm":
        hi=max(float(params["raw"][-1]),1.)
        for _ in range(80):
            if -float(_conditional_logsurvival(distribution,params,hi))>=goal:
                return float(brentq(lambda u:-float(_conditional_logsurvival(distribution,params,u))-goal,0.,hi))
            hi*=2
        raise ValueError("Horizon lognormal hors domaine numérique.")
    raise ValueError("Modèle non pris en charge.")


def _future_hazard(name,params,u):
    u=np.asarray(u,float)
    if name=="expon":
        rate=params.get("lambda_hpp_h") or 1/params["raw"][1]
        return np.full_like(u,rate)
    if name=="power_law_nhpp":
        age=params.get("reference_time_h",0.)+params.get("origin_offset_h",0.)
    else:
        age=params.get("age_since_last_failure_h",0.)
        if name=="grp_kijima_i": age+=params.get("virtual_age_h",0.)
    if name=="lognorm":
        return np.exp(sst.lognorm.logpdf(age+u,*params["raw"])-sst.lognorm.logsf(age+u,*params["raw"]))
    k=params["beta"]; eta=params["eta"]
    with np.errstate(divide="ignore",over="ignore"):
        return (k/eta)*((age+u)/eta)**(k-1)


def build_reliability_curves(ttf_series,model,distribution,params,points=200):
    x=_clean_positive(ttf_series)
    cols=["t","R_t","F_t","f_t","h_t"]
    if len(x)==0 or distribution not in MODEL_NAMES or not params.get("raw"):
        return pd.DataFrame(columns=cols)
    if points<2: raise ValueError("Au moins deux points de courbe sont requis.")
    p=dict(params)
    p.setdefault("reference_time_h",float(x.sum()))
    p.setdefault("age_since_last_failure_h",max(0.,p["reference_time_h"]-float(x.sum())))
    if distribution=="grp_kijima_i": p.setdefault("virtual_age_h",p["q"]*float(x.sum()))
    max_u=max(conditional_horizon(distribution,p,.1),float(x.mean()))
    grid=np.linspace(0.,max_u,points); R=conditional_reliability(distribution,p,grid)
    h=_future_hazard(distribution,p,grid)
    with np.errstate(invalid="ignore"):
        density=h*R
    out=pd.DataFrame(dict(t=grid,R_t=R,F_t=1-R,f_t=density,h_t=h))
    out.attrs.update(curve_kind="conditional_next_failure",reference_time_h=p["reference_time_h"],
                     note="t est une durée future ; h_t est le risque de la prochaine panne, pas un comptage moyen GRP/RP.")
    return out


def _maintenance_recommendation(model,process_variant,distribution,params):
    return dict(maintenance_type="À examiner avec le diagnostic technique",priority="Non déterminée",
                reason="Le modèle statistique seul ne détermine ni l'urgence ni le mécanisme physique. Examiner l'adéquation, la criticité et les contraintes d'exploitation.")


def _distribution_mean(name,params):
    if params is None: return None
    dist={"expon":sst.expon,"weibull_2p":sst.weibull_min,"lognorm":sst.lognorm}.get(name)
    return _safe_float(dist.mean(*params)) if dist else None


def compute_reliability_indicators(ttf_series,repair_series=None,*,model,distribution,
                                   process_variant,params,time_basis="unspecified",
                                   matched_repair_cycles=False):
    x=_clean_positive(ttf_series)
    raw=[] if repair_series is None else list(repair_series)
    rep=np.array([np.nan if v is None or pd.isna(v) else float(v) for v in raw],float)
    invalid=(np.isinf(rep)| (rep<0))
    missing=np.isnan(rep); valid=rep[~invalid & ~missing]
    mean=float(x.mean()) if len(x) else None
    mttr=float(valid.mean()) if len(valid) else None
    availability=None
    # Cette approximation n'est autorisée que pour des cycles appariés en temps de fonctionnement.
    if time_basis=="operating" and matched_repair_cycles and len(rep)==len(x) and not (invalid|missing).any() and len(x):
        availability=float(x.sum()/(x.sum()+rep.sum()))
    theoretical=_distribution_mean(distribution,params.get("raw"))
    intensity=None
    if distribution in MODEL_NAMES and params.get("raw"):
        intensity=_safe_float(_future_hazard(distribution,params,0.))
    return dict(cleaned_failures_n=len(x),interval_count=len(x),cleaned_repairs_n=len(valid),
                missing_repairs_n=int(missing.sum()),invalid_repairs_n=int(invalid.sum()),
                mean_interval_h=mean,empirical_mttf_h=None,theoretical_mttf_h=theoretical,
                mtbf_h=mean,mtbf_definition="Moyenne descriptive des intervalles observés ; voir time_basis.",
                mttr_h=mttr,availability_intrinsic=availability,time_basis=time_basis,
                availability_note="Non calculable sans temps de fonctionnement et réparations de cycles appariés." if availability is None else "Rapport des durées des cycles appariés, sans délais supplémentaires.",
                sample_std_ttf_h=float(np.std(x,ddof=1)) if len(x)>1 else None,
                mean_failure_rate_h=None,intensity_at_reference_h=intensity,
                empirical_failure_rate_h=len(x)/float(x.sum()) if len(x) else None,
                **_maintenance_recommendation(model,process_variant,distribution,params))


def sequential_predictive_validation(ttf_series,initial_train=8,*,origin_offset_h=0.):
    """Réestimation sur le passé, puis log-densité conditionnelle de la panne suivante."""
    x=_clean_positive(ttf_series)
    if not isinstance(initial_train,int) or initial_train<4 or initial_train>=len(x):
        raise ValueError("initial_train doit être entier, >=4 et < au nombre d'intervalles.")
    rows=[]; scores={}
    for name in MODEL_NAMES:
        total=0.; success=0
        for i in range(initial_train,len(x)):
            fit=fit_model(name,x[:i],origin_offset_h=origin_offset_h)
            score=None
            if fit["status"]=="fitted":
                p=_parameters(fit,x[:i]); u=float(x[i])
                hazard=float(_future_hazard(name,p,u))
                if hazard>0 and np.isfinite(hazard):
                    score=float(np.log(hazard)+_conditional_logsurvival(name,p,u))
                    if not np.isfinite(score): score=None
            if score is not None: total+=score; success+=1
            rows.append(dict(model=name,train_n=i,test_interval=i+1,log_score=score,
                             status="ok" if score is not None else "failed"))
        scores[name]=dict(total_log_score=total if success==len(x)-initial_train else None,
                          successful_predictions=success,expected_predictions=len(x)-initial_train)
    complete={n:s["total_log_score"] for n,s in scores.items() if s["total_log_score"] is not None}
    return dict(status="completed",initial_train=initial_train,scores=scores,
                best_name=max(complete,key=complete.get) if complete else None,details=pd.DataFrame(rows))


def _simulate(fit,n,end,scheme,rng,max_events=100000):
    name=fit["name"]; elapsed=0.; virtual=0.; out=[]
    for _ in range(n if scheme=="event_terminated" else max_events):
        e=float(rng.exponential()); eta=fit.get("eta"); k=fit.get("beta")
        if name=="expon": gap=e/fit["lambda_hpp_h"]
        elif name=="lognorm": gap=float(rng.lognormal(fit["mu"],fit["sigma"]))
        elif name=="weibull_2p": gap=eta*e**(1/k)
        else:
            age=elapsed+fit.get("origin_offset_h",0.) if name=="power_law_nhpp" else virtual
            gap=eta*e**(1/k) if age==0 else age*np.expm1(np.log1p(e/(age/eta)**k)/k)
        if not np.isfinite(gap) or gap<=0: raise ValueError("Simulation numériquement invalide.")
        if scheme=="time_terminated" and elapsed+gap>end:
            return np.array(out,float),end
        elapsed+=gap; out.append(gap)
        if name=="grp_kijima_i": virtual+=fit["q"]*gap
    if scheme=="time_terminated": raise ValueError("Nombre maximal d'événements simulés atteint.")
    return np.array(out,float),elapsed


def bootstrap_model(ttf_series,model_name="power_law_nhpp",*,n_boot=1000,seed=20260823,
                    alpha=.05,observation_end_h=None,observation_scheme="auto",
                    origin_offset_h=0.,reference_time_h=None,target=.8):
    """Bootstrap paramétrique avec réestimation, même règle d'arrêt que l'échantillon.

    Les quantiles d'horizon réévaluent les paramètres réestimés à l'histoire réelle,
    pas à l'âge final aléatoire de chaque historique simulé. Ils décrivent
    l'incertitude paramétrique sous le modèle, pas l'incertitude de choix du modèle.
    Une simulation/réestimation échouée est comptée ; aucune n'est remplacée.
    """
    _alpha(alpha)
    if not isinstance(n_boot,int) or n_boot<1: raise ValueError("n_boot doit être un entier positif.")
    x=_clean_positive(ttf_series); last,end,_=_window(x,observation_end_h)
    scheme=_scheme(observation_scheme,end,last)
    fit=fit_model(model_name,x,observation_end_h=end,origin_offset_h=origin_offset_h)
    if fit["status"]!="fitted" or fit.get("ks_stat") is None:
        return dict(status="unavailable",error=fit.get("error") or "Résidus insuffisants.",requested=n_boot)
    reference=end if reference_time_h is None else float(reference_time_h)
    conditional_horizon(model_name,_parameters(fit,x,reference),target)
    rng=np.random.default_rng(seed); stats=[]; draws=[]; errors=[]
    for b in range(n_boot):
        try:
            sim,sim_end=_simulate(fit,len(x),end,scheme,rng)
            f=fit_model(model_name,sim,observation_end_h=sim_end,origin_offset_h=origin_offset_h)
            if f["status"]!="fitted" or f.get("ks_stat") is None:
                raise ValueError(f.get("error") or "Trop peu de résidus simulés.")
            # État GRP recalculé sur l'histoire réelle avec q* réestimé.
            f_actual=dict(f,T_end=end,virtual_age_h=f.get("q",0.)*last)
            p=_parameters(f_actual,x,reference)
            horizon=conditional_horizon(model_name,p,target)
            stats.append((f["ks_stat"],f["cvm_stat"]))
            draws.append({"replicate":b+1,"beta":f.get("beta"),"eta":f.get("eta"),
                          "q":f.get("q"),"mu":f.get("mu"),"sigma":f.get("sigma"),
                          "lambda_hpp_h":f.get("lambda_hpp_h"),"horizon_h":horizon})
        except (ValueError,OverflowError,FloatingPointError) as exc:
            errors.append({"replicate":b+1,"error":str(exc)})
    valid=len(stats); complete=valid==n_boot
    result=dict(status="completed" if complete else "incomplete",requested=n_boot,successful=valid,
                failed=len(errors),seed=seed,observation_scheme=scheme,reference_time_h=reference,
                target=target,errors=errors,draws=pd.DataFrame(draws),accepted=None,
                ks_p_bootstrap=None,cvm_p_bootstrap=None,parameter_intervals={})
    if complete:
        array=np.array(stats); pk=(1+int(np.sum(array[:,0]>=fit["ks_stat"]))) /(n_boot+1)
        pc=(1+int(np.sum(array[:,1]>=fit["cvm_stat"]))) /(n_boot+1)
        result.update(ks_p_bootstrap=pk,cvm_p_bootstrap=pc,accepted=bool(pk>=alpha and pc>=alpha))
        for col in ("beta","eta","q","mu","sigma","lambda_hpp_h","horizon_h"):
            values=pd.to_numeric(result["draws"][col],errors="coerce").dropna()
            if len(values): result["parameter_intervals"][col]=np.quantile(values,[alpha/2,.5,1-alpha/2]).tolist()
    else:
        result["note"]="Calibration incomplète : aucune p-valeur ni conclusion d'acceptation publiée. Examiner les échecs."
    return result


def origin_sensitivity(ttf_series,offsets_h,*,observation_end_h=None,target=.8):
    x=_clean_positive(ttf_series); rows=[]
    for offset in offsets_h:
        fit=fit_model("power_law_nhpp",x,observation_end_h=observation_end_h,origin_offset_h=float(offset))
        h=None
        if fit["status"]=="fitted": h=conditional_horizon("power_law_nhpp",_parameters(fit,x),target)
        rows.append(dict(origin_offset_h=offset,status=fit["status"],beta=fit.get("beta"),
                         eta=fit.get("eta"),loglik=fit.get("loglik"),aicc=fit.get("aicc"),horizon_h=h,error=fit.get("error")))
    return pd.DataFrame(rows)


def analyze_reliability_only(ttf_series,repair_series=None,alpha=.05,*,
        observation_end_h=None,reference_time_h=None,origin_offset_h=0.,
        observation_scheme="auto",time_basis="unspecified",matched_repair_cycles=False,
        selection_rule="aicc",selected_model=None,run_sequential=False,initial_train=8,
        n_boot=0,seed=20260823,bootstrap_models=None,reliability_target=.8):
    """Analyse pure. Les options nouvelles sont nommées ; anciens appels conservés.

    selection_rule='aicc' par défaut ; 'predictive' exige run_sequential=True.
    selected_model permet un scénario explicite, toujours identifié comme tel.
    n_boot=0 ne signifie jamais « modèle validé ».
    reference_time_h ne peut dépasser la fin d'observation : ajouter les nouvelles
    données avant de calculer une prévision à une date ultérieure non documentée.
    """
    base=dict(engine_version=ENGINE_VERSION,status="unavailable",model=None,process_variant=None,
              distribution=None,params={},decision={},tests={},goodness={},candidates={},
              curves=pd.DataFrame(columns=["t","R_t","F_t","f_t","h_t"]),indicators={},warnings=[])
    try:
        _alpha(alpha); x=_clean_positive(ttf_series); base["cleaned_n"]=len(x)
        if len(x)<3: raise ValueError("Au moins trois intervalles sont requis ; aucun modèle par défaut n'est imposé.")
        if time_basis not in ("calendar","operating","unspecified"): raise ValueError("time_basis invalide.")
        if not isinstance(n_boot,int) or n_boot<0: raise ValueError("n_boot doit être un entier >=0.")
        if not 0<reliability_target<1: raise ValueError("Fiabilité cible invalide.")
        last,end,_=_window(x,observation_end_h); scheme=_scheme(observation_scheme,end,last)
        reference=end if reference_time_h is None else float(reference_time_h)
        if math.isclose(reference,end,rel_tol=1e-12,abs_tol=1e-9): reference=end
        if math.isclose(reference,last,rel_tol=1e-12,abs_tol=1e-9): reference=last
        if not np.isfinite(reference) or reference<last or reference>end:
            raise ValueError("Le temps de référence doit être entre dernière panne et fin observée.")
        if selection_rule not in ("aicc","predictive"): raise ValueError("selection_rule : aicc ou predictive.")
        if selected_model is not None and selected_model not in MODEL_NAMES: raise ValueError("Modèle demandé inconnu.")
        fits={name:fit_model(name,x,observation_end_h=end,origin_offset_h=origin_offset_h) for name in MODEL_NAMES}
        base["candidates"]=fits
        viable={name:f for name,f in fits.items() if f["status"]=="fitted" and f["aicc"] is not None}
        aicc_best=min(viable,key=lambda n:viable[n]["aicc"]) if viable else None
        sequential=sequential_predictive_validation(x,initial_train,origin_offset_h=origin_offset_h) if run_sequential else {"status":"not_requested"}
        if selected_model is not None:
            chosen=selected_model; rule="explicit_scenario"
        elif selection_rule=="predictive":
            if not run_sequential: raise ValueError("La sélection prédictive exige run_sequential=True.")
            chosen=sequential.get("best_name"); rule="sequential_log_score"
        else:
            chosen=aicc_best; rule="min_aicc"
        if chosen is None or fits[chosen]["status"]!="fitted":
            raise ValueError("Aucun modèle sélectionnable pour le critère demandé (AICc peut être indéfini sur petit échantillon).")
        bootstrap={}
        names=[chosen] if bootstrap_models is None else list(bootstrap_models)
        if any(n not in MODEL_NAMES for n in names): raise ValueError("bootstrap_models contient une famille inconnue.")
        if n_boot:
            for name in names:
                b=bootstrap_model(x,name,n_boot=n_boot,seed=seed,alpha=alpha,observation_end_h=end,
                                  observation_scheme=scheme,origin_offset_h=origin_offset_h,
                                  reference_time_h=reference,target=reliability_target)
                bootstrap[name]=b
                fits[name].update(ks_p_bootstrap=b.get("ks_p_bootstrap"),cvm_p_bootstrap=b.get("cvm_p_bootstrap"),
                                  accepted=b.get("accepted"),validation_status=b["status"])
        fit=fits[chosen]; params=_parameters(fit,x,reference)
        model="NHPP" if chosen=="power_law_nhpp" else "GRP" if chosen=="grp_kijima_i" else "RP"
        variant="HPP" if chosen=="expon" else model
        graph=graphical_trend_test(x); mk=mann_kendall_test(x,alpha)
        lap=laplace_trend_test(x,alpha,observation_end_h=end,observation_scheme=scheme)
        trend=combine_trend_evidence(graph,mk,lap,alpha)
        dg=graphical_dependence_test(x); dc=dependence_correlation_test(x,alpha); dep=combine_dependence_evidence(dg,dc,alpha)
        indicators=compute_reliability_indicators(x,repair_series,model=model,distribution=chosen,
                process_variant=variant,params=params,time_basis=time_basis,matched_repair_cycles=matched_repair_cycles)
        assumptions={"expon":"Intensité constante / renouvellement exponentiel",
                     "power_law_nhpp":"PLP conditionnel à l'origine observée / réparation minimale",
                     "weibull_2p":"Renouvellement Weibull, remise à neuf statistique",
                     "lognorm":"Renouvellement lognormal, remise à neuf statistique",
                     "grp_kijima_i":"Kijima I, V0=0, q entre 0 et 1"}
        good={key:fit.get(key) for key in ("aic","aicc","bic","loglik","ks_p","cvm_p","chi2_p","ks_p_bootstrap","cvm_p_bootstrap","accepted","validation_status")}
        context=dict(time_basis=time_basis,unit="hours",observation_end_h=end,reference_time_h=reference,
                     history_time_h=last,right_censor_h=end-last,origin_offset_h=origin_offset_h,
                     observation_scheme=scheme,initial_event="conditioned_reference",alpha=alpha)
        fingerprint=dict(engine=ENGINE_VERSION,x=x.tolist(),repairs=None if repair_series is None else list(repair_series),
                         context=context,selection_rule=rule,chosen=chosen,n_boot=n_boot,seed=seed,
                         bootstrap_models=names,initial_train=initial_train,run_sequential=run_sequential,target=reliability_target)
        base.update(status="computed",model=model,process_variant=variant,distribution=chosen,params=params,
                    goodness=good,indicators=indicators,context=context,
                    analysis_hash=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,default=str).encode()).hexdigest(),
                    curves=build_reliability_curves(x,model,chosen,params),bootstrap=bootstrap,sequential=sequential,
                    selection=dict(aicc_best=aicc_best,predictive_best=sequential.get("best_name"),selected=chosen,rule=rule),
                    decision=dict(has_trend=trend["has_trend"],trend_direction=trend["direction"],trend_confidence=trend["confidence"],
                                  has_dependence=dep["has_dep"],dependence_strength=dep.get("strength"),selected_process=model,
                                  selected_variant=variant,entity_assumption=assumptions[chosen],law_selected=chosen,
                                  law_accepted=fit.get("accepted"),selection_rule=rule,
                                  reason="Sélection relative selon "+rule+" ; distincte de l'adéquation et du diagnostic physique."),
                    tests=dict(trend_graphical=graph,trend_mil_hdbk_189=graph,trend_mk=mk,trend_laplace=lap,
                               trend_combined=trend,dependence_graphical=dg,dependence_correlation=dc,dependence=dep),
                    horizon=dict(target=reliability_target,hours=conditional_horizon(chosen,params,reliability_target),
                                 reference_time_h=reference,kind="conditional_next_failure"))
        base["warnings"]=["Les anciennes pages peuvent recalculer des courbes depuis zéro : utiliser les sorties de ce moteur."]
        if fit.get("accepted") is None: base["warnings"].append("Adéquation non conclue : bootstrap absent ou incomplet.")
        if fit.get("accepted") is False: base["warnings"].append("Modèle rejeté par au moins un test bootstrap ; prévision à présenter comme scénario exploratoire.")
        if time_basis=="unspecified": base["warnings"].append("Base de temps non fournie ; aucune disponibilité calculée.")
        base["warnings"].extend(fit.get("warnings",[]))
    except (ValueError,TypeError,OverflowError,FloatingPointError) as exc:
        base["error"]=str(exc)
    return base


def _yes_no(value):
    return "Non calculé" if value is None else "Oui" if value else "Non"


def build_reliability_tables(reliability_result):
    r=reliability_result; tests=r.get("tests",{}); d=r.get("decision",{}); p=r.get("params",{}); ind=r.get("indicators",{})
    trends=[]
    for key,label,stat in (("trend_graphical","Graphique descriptif","beta_graph"),("trend_mk","Mann–Kendall (intervalles)","z"),("trend_laplace","Laplace (comptage)","u")):
        test=tests.get(key,{})
        trends.append({"Test":label,"Statistique":test.get(stat),"p_value":test.get("p"),
                       "Décision":_yes_no(test.get("has_trend")),"Direction":test.get("direction"),"R2":test.get("r2")})
    dep=tests.get("dependence",{})
    deps=[{"Méthode":name,"r":dep.get(prefix+"_r"),"p_value":dep.get(prefix+"_p"),
           "Dépendance":_yes_no(dep.get("has_dep")) if prefix=="spearman" else "Descriptif complémentaire"}
          for name,prefix in (("Pearson","pearson"),("Spearman nominal","spearman"))]
    rows=[]
    for name,f in r.get("candidates",{}).items():
        rows.append({"Modèle":name,"Paramètres":str(f.get("params")),"Méthode estimation":f.get("estimation_method"),
                     "Statut":f.get("status"),"Erreur":f.get("error"),"Log-vraisemblance":f.get("loglik"),
                     "k":f.get("n_params"),"AIC":f.get("aic"),"AICc":f.get("aicc"),"BIC":f.get("bic"),
                     "KS p":f.get("ks_p"),"Chi2 p":None,"CvM p":f.get("cvm_p"),
                     "KS bootstrap p":f.get("ks_p_bootstrap"),"CvM bootstrap p":f.get("cvm_p_bootstrap"),
                     "Acceptée":f.get("accepted"),"Retenue":"Oui" if name==r.get("distribution") else "Non"})
    summary={"Processus":r.get("model"),"Variant":r.get("process_variant"),"Distribution":r.get("distribution"),
             "Beta":p.get("beta"),"Eta":p.get("eta"),"q":p.get("q"),"Gamma":p.get("gamma"),
             "Lambda_HPP (1/h)":p.get("lambda_hpp_h"),"MTTF (h)":ind.get("theoretical_mttf_h"),
             "MTBF (h)":ind.get("mtbf_h"),"MTTR (h)":ind.get("mttr_h"),"Base de temps":ind.get("time_basis"),
             "Disponibilité":ind.get("availability_intrinsic"),"Intensité/risque à la référence (1/h)":ind.get("intensity_at_reference_h"),
             "Référence (h)":p.get("reference_time_h"),"Horizon conditionnel (h)":r.get("horizon",{}).get("hours"),
             "Ajustement accepté":r.get("goodness",{}).get("accepted"),"Erreur":r.get("error")}
    return {"trend_results":pd.DataFrame(trends),"dependence_results":pd.DataFrame(deps),
            "process_choice":pd.DataFrame([{"Processus retenu":r.get("model"),"Variant":r.get("process_variant"),
                                          "Hypothèse entité":d.get("entity_assumption"),"Justification":d.get("reason"),
                                          "Règle":d.get("selection_rule")}]),
            "fit_candidates":pd.DataFrame(rows),"reliability_summary":pd.DataFrame([summary]),
            "maintenance_recommendation":pd.DataFrame([{"Type de maintenance":ind.get("maintenance_type"),
                                                       "Priorité":ind.get("priority"),"Raison":ind.get("reason")}]),
            "reliability_curves":r.get("curves",pd.DataFrame()),
            "predictive_validation":pd.DataFrame.from_dict(r.get("sequential",{}).get("scores",{}),orient="index").rename_axis("model").reset_index()}


def build_global_result_tables(reliability_result):
    return build_reliability_tables(reliability_result)


def analyze_ttf_pipeline(ttf_series,alpha=.05,repair_series=None,**kwargs):
    reliability=analyze_reliability_only(ttf_series,repair_series=repair_series,alpha=alpha,**kwargs)
    return {"reliability":reliability,"tables":build_global_result_tables(reliability)}


def analyze_project_inputs(inputs,**kwargs):
    """Adaptateur pour le dictionnaire fourni par le datahub corrigé.

    Les anciennes pages doivent appeler cet adaptateur pour transmettre la fenêtre.
    Le segment avant la panne de référence n'est pas traité comme un âge initial.
    """
    windows=inputs.get("observation_windows")
    options=dict(kwargs)
    if isinstance(windows,pd.DataFrame) and len(windows):
        if len(windows)!=1: raise ValueError("Sélectionner un seul équipement avant l'analyse.")
        w=windows.iloc[0]
        end=float(w["history_time_h"])+float(w["right_censor_h"])
        options.setdefault("observation_end_h",end)
        options.setdefault("reference_time_h",float(w["decision_time_h"]))
        options.setdefault("time_basis",str(w["time_basis"]))
    options.setdefault("time_basis",inputs.get("time_basis","unspecified"))
    options.setdefault("alpha",inputs.get("alpha",.05))
    return analyze_ttf_pipeline(inputs.get("ttf_series",[]),repair_series=inputs.get("repair_series"),**options)


analyze_integrated_pipeline=analyze_ttf_pipeline
