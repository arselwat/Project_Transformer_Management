"""Optimisation des horizons de maintenance — version 2.0.

À installer dans core/reliability/optimize.py, avec organigram.py corrigé.

API principale : optimize_maintenance(reliability_result, C_prev, C_corr).
Elle reçoit le résultat de analyze_reliability_only ou le dictionnaire complet
de analyze_ttf_pipeline. Elle ne réajuste pas de loi et conserve le temps de
référence du moteur. Un rejet statistique reste visible dans les résultats.

PLP/HPP : coût prospectif sur un horizon fini
    [Cp + Cc * E(N(s+u)-N(s))] / u.
La réparation minimale ne remet pas l'âge du PLP à zéro. Ce critère ne décrit
pas un renouvellement parfait après chaque intervention. Cp et Cc doivent
utiliser la même unité monétaire ou la même normalisation.

RP/GRP : horizon conditionnel fourni par le moteur ; aucune conversion abusive
de 1-R(u) en nombre moyen de pannes. L'optimisation économique correspondante
n'est pas implémentée dans l'API principale et reste explicitement indisponible.

Les anciennes fonctions Weibull restent disponibles pour les anciens appelants.
Elles décrivent une AUTRE politique : remplacement à l'âge T ou à la première
panne, avec remise à neuf. Les résultats indiquent leur politique et leur domaine
de recherche. Les pages doivent appeler l'API principale pour exploiter le PLP.

Exemple :
    analysis = analyze_ttf_pipeline(x, time_basis='calendar',
        run_sequential=True, selection_rule='predictive')
    result = optimize_maintenance(analysis, 1.0, 5.0,
        R_target=0.80, R_min_cost=0.70)
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammainc, gamma as gamma_function

from .organigram import conditional_horizon, conditional_reliability

OPTIMIZER_VERSION = "2.0"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        value = float(x)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _number(value, label, *, positive=False):
    number = _safe_float(value)
    if number is None or (number <= 0 if positive else number < 0):
        raise ValueError(label + (" doit être fini et > 0." if positive else " doit être fini et >= 0."))
    return number


def _probability(value, label, *, allow_zero=False):
    p = _safe_float(value)
    if p is None or not (0 <= p < 1 if allow_zero else 0 < p < 1):
        raise ValueError(label + (" doit appartenir à [0,1[." if allow_zero else " doit appartenir à ]0,1[."))
    return p


def _weibull_parameters(beta, eta, gamma=0.):
    return (_number(beta,"beta",positive=True), _number(eta,"eta",positive=True),
            _number(gamma,"gamma"))


def _weibull_survival_3p(t, beta, eta, gamma=0.):
    beta,eta,gamma = _weibull_parameters(beta,eta,gamma)
    t = _number(t,"t")
    if t <= gamma:
        return 1.
    with np.errstate(over="ignore"):
        cumulative = np.power((t-gamma)/eta,beta)
    return float(np.exp(-cumulative))


def _integral_survival_trapz(beta, eta, gamma, T, integ_steps=400):
    """Nom historique conservé ; intégrale analytique via la Gamma incomplète.

    integ_steps est conservé pour compatibilité, sans effet sur l'intégrale.
    """
    beta,eta,gamma = _weibull_parameters(beta,eta,gamma)
    T = _number(T,"T")
    if T <= gamma:
        return T
    shape = 1./beta
    with np.errstate(over="ignore"):
        z = np.power((T-gamma)/eta,beta)
    result = gamma + eta*gamma_function(1+shape)*gammainc(shape,z)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Intégrale de survie hors domaine numérique.")
    return float(result)


def interval_weibull_target(beta, eta, gamma, R_target):
    """Quantile depuis l'état neuf. Ce n'est pas un horizon PLP conditionnel."""
    beta,eta,gamma = _weibull_parameters(beta,eta,gamma)
    target = _probability(R_target,"R_target")
    return float(gamma + eta*(-math.log(target))**(1/beta))


def expected_failures_nhpp(horizon_h, beta, eta, reference_time_h=0., origin_offset_h=0.):
    """Incrément de fonction moyenne PLP, stable lorsque u est petit devant s."""
    beta,eta,_ = _weibull_parameters(beta,eta,0.)
    u = _number(horizon_h,"horizon_h")
    s = _number(reference_time_h,"reference_time_h") + _number(origin_offset_h,"origin_offset_h")
    with np.errstate(over="ignore",invalid="ignore"):
        value = np.power(u/eta,beta) if s == 0 else np.power(s/eta,beta)*np.expm1(beta*np.log1p(u/s))
    if not np.isfinite(value):
        raise ValueError("Nombre moyen de pannes hors domaine numérique.")
    return float(value)


def cost_rate_nhpp(horizon_h, beta, eta, C_prev, C_corr,
                   reference_time_h=0., origin_offset_h=0.):
    u = _number(horizon_h,"horizon_h",positive=True)
    cp = _number(C_prev,"C_prev",positive=True)
    cc = _number(C_corr,"C_corr")
    return (cp + cc*expected_failures_nhpp(u,beta,eta,reference_time_h,origin_offset_h))/u


def optimize_interval_cost_nhpp(beta, eta, C_prev, C_corr, *, reference_time_h,
                                origin_offset_h=0., R_min=0., max_horizon_h=None):
    """Minimum du critère prospectif, avec ou sans contrainte de fiabilité.

    beta>1 et Cc>0 : minimum intérieur unique, recherché par la dérivée.
    beta<=1 ou Cc=0 : coût décroissant ; sans borne il n'existe pas de minimum
    fini. Une borne de fiabilité ou d'exploitation produit un optimum de frontière.
    Aucun plafond numérique n'est présenté comme un optimum non contraint.
    """
    beta,eta,_ = _weibull_parameters(beta,eta)
    cp = _number(C_prev,"C_prev",positive=True); cc = _number(C_corr,"C_corr")
    s = _number(reference_time_h,"reference_time_h")
    offset = _number(origin_offset_h,"origin_offset_h")
    minimum = _probability(R_min,"R_min",allow_zero=True)
    params = dict(beta=beta,eta=eta,reference_time_h=s,origin_offset_h=offset)
    bound = conditional_horizon("power_law_nhpp",params,minimum) if minimum>0 else None
    if max_horizon_h is not None:
        limit = _number(max_horizon_h,"max_horizon_h",positive=True)
        bound = limit if bound is None else min(bound,limit)
    unconstrained = None
    if beta>1 and cc>0:
        age = s+offset
        # u*M'(s+u)-[M(s+u)-M(s)] = Cp/Cc.
        def derivative_numerator(u):
            increment = expected_failures_nhpp(u,beta,eta,s,offset)
            intensity = (beta/eta)*((age+u)/eta)**(beta-1)
            return cc*(u*intensity-increment)-cp
        high = max(eta,age,1.)
        for _ in range(100):
            if derivative_numerator(high)>=0:
                unconstrained = float(brentq(derivative_numerator,0.,high,xtol=1e-8,rtol=1e-12))
                break
            high *= 2
        else:
            raise ValueError("Impossible d'encadrer le minimum économique dans le domaine numérique.")
    if unconstrained is None and bound is None:
        return dict(status="no_finite_minimum",T_cost=None,C_min=None,R_at_T=None,
                    T_cost_unconstrained=None,search_upper_h=None,at_boundary=False,
                    asymptotic_cost_rate=cc/eta if beta==1 else 0.,
                    policy="finite_horizon_minimal_repair",
                    note="Coût décroissant sans minimum fini : fournir une contrainte de fiabilité ou d'exploitation.")
    optimum = bound if unconstrained is None else unconstrained if bound is None else min(unconstrained,bound)
    risk = conditional_reliability("power_law_nhpp",params,optimum)
    return dict(status="computed",T_cost=float(optimum),
                C_min=cost_rate_nhpp(optimum,beta,eta,cp,cc,s,offset),R_at_T=risk,
                T_cost_unconstrained=unconstrained,search_upper_h=bound,
                at_boundary=bound is not None and math.isclose(optimum,bound,rel_tol=1e-9),
                expected_failures=expected_failures_nhpp(optimum,beta,eta,s,offset),
                policy="finite_horizon_minimal_repair",reference_time_h=s,origin_offset_h=offset)


def optimize_maintenance(reliability_result, C_prev, C_corr, R_target=.80, R_min_cost=.70,
                         *, max_horizon_h=None, min_horizon_h=0.):
    """API commune destinée aux pages Optimisation, Maintenance et Résultats.

    T_R : horizon maximal lié à R_target depuis la référence du moteur.
    T_cost : horizon économique sous R_min_cost et l'éventuelle borne métier.
    T_recommended : optimum économique admissible sous les DEUX cibles.
    Pour RP/GRP : T_recommended reste un horizon fiabiliste, sans optimum économique.
    Le statut exploratory signale une adéquation absente/incomplète/rejetée.
    """
    target = _probability(R_target,"R_target")
    rmin = _probability(R_min_cost,"R_min_cost",allow_zero=True)
    cp = _number(C_prev,"C_prev",positive=True); cc = _number(C_corr,"C_corr")
    lower = _number(min_horizon_h,"min_horizon_h")
    upper = None if max_horizon_h is None else _number(max_horizon_h,"max_horizon_h",positive=True)
    r = reliability_result.get("reliability",reliability_result)
    if r.get("status") != "computed" or not r.get("params"):
        raise ValueError("Résultat du moteur absent ou non exploitable : "+str(r.get("error","")))
    name = r.get("distribution"); p = r["params"]
    reference = _number(p.get("reference_time_h"),"reference_time_h")
    horizon = conditional_horizon(name,p,target)
    # La contrainte économique peut être plus stricte que la cible principale.
    feasible_upper = min(horizon,conditional_horizon(name,p,rmin)) if rmin>target else horizon
    if upper is not None: feasible_upper = min(feasible_upper,upper)
    accepted = r.get("goodness",{}).get("accepted")
    warnings = list(r.get("warnings",[]))
    out = dict(optimizer_version=OPTIMIZER_VERSION,analysis_hash=r.get("analysis_hash"),
               distribution=name,reference_time_h=reference,R_target=target,R_min_cost=rmin,
               C_prev=cp,C_corr=cc,T_R=horizon,T_cost=None,C_min=None,R_at_T=None,
               T_recommended=None,interval_opt_h=None,reliability_at_recommended=None,
               validation_accepted=accepted,warnings=warnings,
               status="computed" if accepted is True else "exploratory",
               economic_status="unsupported_model",policy="reliability_only",
               min_horizon_h=lower,max_horizon_h=upper)
    if lower>feasible_upper:
        out.update(status="infeasible",reason="L'horizon minimal dépasse l'horizon admissible ; aucune échéance conforme.")
        return out
    if name in ("power_law_nhpp","expon"):
        beta = 1. if name=="expon" else p["beta"]
        eta = 1./p["lambda_hpp_h"] if name=="expon" else p["eta"]
        offset = p.get("origin_offset_h",0.) if name=="power_law_nhpp" else 0.
        cost = optimize_interval_cost_nhpp(beta,eta,cp,cc,reference_time_h=reference,
                    origin_offset_h=offset,R_min=rmin,max_horizon_h=upper)
        out.update(T_cost=cost["T_cost"],C_min=cost["C_min"],R_at_T=cost["R_at_T"],
                   economic_status=cost["status"],economic_result=cost,policy=cost["policy"])
        economic_optimum = cost["T_cost"]
        chosen = feasible_upper if economic_optimum is None else min(economic_optimum,feasible_upper)
        chosen = max(chosen,lower)
        out["cost_rate_at_recommended"] = cost_rate_nhpp(chosen,beta,eta,cp,cc,reference,offset)
        out["expected_failures_at_recommended"] = expected_failures_nhpp(chosen,beta,eta,reference,offset)
        out["selection_reason"] = "Minimum économique sur le domaine satisfaisant les cibles et les bornes d'exploitation."
    else:
        chosen = feasible_upper
        out["selection_reason"] = "Horizon fiabiliste ; coût RP/GRP non optimisé par cette API."
        out["warnings"].append("L'espérance du nombre de pannes RP/GRP ne se déduit pas de 1-R ; aucun coût PLP n'est appliqué à ce modèle.")
    reliability = conditional_reliability(name,p,chosen)
    if reliability+1e-10 < max(target,rmin):
        raise ValueError("Échec numérique de la vérification finale de fiabilité.")
    out.update(T_recommended=float(chosen),interval_opt_h=float(chosen),
               reliability_at_recommended=float(reliability),days_from_reference=float(chosen/24))
    if accepted is not True:
        out["warnings"].append("Horizon exploratoire : il ne constitue pas une instruction automatique d'intervention.")
    return out


def optimize_interval_cost_weibull(beta, eta, gamma, C_prev, C_corr, R_min=0.,
                                  t_max_mult=3., steps=200, integ_steps=400):
    """Compatibilité : remplacement par âge depuis l'état neuf.

    [Cp*R(T)+Cc*(1-R(T))] / intégrale_0^T R(t)dt.
    Recherche sur un domaine FINI déclaré ; jamais appelé pour un PLP ou un GRP.
    """
    beta,eta,gamma = _weibull_parameters(beta,eta,gamma)
    cp = _number(C_prev,"C_prev",positive=True); cc = _number(C_corr,"C_corr")
    rmin = _probability(R_min,"R_min",allow_zero=True)
    multiplier = _number(t_max_mult,"t_max_mult",positive=True)
    if int(steps)<3: raise ValueError("steps doit être >=3.")
    upper = gamma+eta*multiplier
    if rmin>0: upper=min(upper,interval_weibull_target(beta,eta,gamma,rmin))
    lower = min(eta*1e-8,upper*1e-6)
    def cost(t):
        R = _weibull_survival_3p(t,beta,eta,gamma)
        integral = _integral_survival_trapz(beta,eta,gamma,t,integ_steps)
        return (cp*R+cc*(1-R))/integral
    grid = np.geomspace(lower,upper,int(steps))
    if 0<gamma<upper: grid=np.unique(np.r_[grid,gamma])
    values = np.array([cost(t) for t in grid])
    candidates = [(float(grid[0]),float(values[0])),(float(grid[-1]),float(values[-1]))]
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1]:
            opt=minimize_scalar(cost,bounds=(grid[i-1],grid[i+1]),method="bounded")
            if opt.success: candidates.append((float(opt.x),float(opt.fun)))
    T,C=min(candidates,key=lambda item:item[1])
    return dict(status="computed_on_finite_domain",T_cost=T,C_min=C,
                R_at_T=_weibull_survival_3p(T,beta,eta,gamma),
                policy="age_replacement_as_good_as_new",reference_time_h=0.,
                search_lower_h=lower,search_upper_h=upper,
                at_boundary=math.isclose(T,upper,rel_tol=1e-7) or math.isclose(T,lower,rel_tol=1e-7),
                note="Minimum sur le domaine indiqué ; un optimum de frontière n'est pas un minimum intérieur démontré.")


def _legacy_fit_values(fit):
    getter = fit.get if isinstance(fit,dict) else lambda key,default=None:getattr(fit,key,default)
    return _weibull_parameters(getter("beta"),getter("eta"),getter("gamma",0.) or 0.)


def propose_intervals_cost_and_reliability(fits, C_prev, C_corr, R_target=.80, R_min_cost=0.):
    """Accepte les nouveaux résultats moteur ou les anciens objets Weibull.

    Les anciens objets sont clairement étiquetés age_replacement_as_good_as_new.
    Une erreur par équipement reste visible ; elle n'est pas supprimée du retour.
    """
    out={}
    for equipment,fit in (fits or {}).items():
        try:
            if isinstance(fit,dict) and ("reliability" in fit or "distribution" in fit):
                value=optimize_maintenance(fit,C_prev,C_corr,R_target,R_min_cost)
            else:
                beta,eta,gamma=_legacy_fit_values(fit)
                tr=interval_weibull_target(beta,eta,gamma,R_target)
                value=optimize_interval_cost_weibull(beta,eta,gamma,C_prev,C_corr,R_min_cost)
                recommended=min(tr,value["T_cost"])
                value.update(T_R=tr,T_recommended=recommended,interval_opt_h=recommended,
                             reliability_at_recommended=_weibull_survival_3p(recommended,beta,eta,gamma))
            out[str(equipment)]=value
        except (ValueError,TypeError,OverflowError,KeyError) as exc:
            out[str(equipment)]=dict(status="invalid",error=str(exc),T_R=None,T_cost=None,
                                    C_min=None,R_at_T=None,T_recommended=None,interval_opt_h=None)
    return out


def propose_intervals(fits, R_target=.80, t_min=0.):
    """Compatibilité : horizon de fiabilité, sans imposer un modèle économique.

    t_min ne peut pas allonger l'horizon au-delà du niveau de fiabilité demandé.
    """
    out={}
    for equipment,fit in (fits or {}).items():
        try:
            minimum=_number(t_min,"t_min")
            _probability(R_target,"R_target")
            if isinstance(fit,dict) and ("reliability" in fit or "distribution" in fit):
                r=fit.get("reliability",fit)
                if r.get("status")!="computed": raise ValueError("Résultat moteur indisponible.")
                horizon=conditional_horizon(r["distribution"],r["params"],R_target)
                reference=r["params"].get("reference_time_h")
            else:
                beta,eta,gamma=_legacy_fit_values(fit)
                horizon=interval_weibull_target(beta,eta,gamma,R_target); reference=0.
            out[str(equipment)]=dict(interval_opt_h=horizon if horizon>=minimum else None,
                                    status="computed" if horizon>=minimum else "infeasible",
                                    T_R=horizon,reference_time_h=reference)
        except (ValueError,TypeError,OverflowError,KeyError) as exc:
            out[str(equipment)]=dict(status="invalid",interval_opt_h=None,error=str(exc))
    return out
