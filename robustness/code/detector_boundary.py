#!/usr/bin/env python3
"""Protected coarse detectors and exact bounded-hazard response certificates.

Mathematical certificates are conditional on supplied support/rate/reset facts.
They cannot establish those facts about an uncharacterized physical apparatus.
The exact helpers accept int, Fraction, or rational strings, never floats/bools.
Numerical illustrations are isolated in make_results().
"""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import argparse
import json


def rational(value):
    if type(value) not in (int, F, str):
        raise TypeError('Use an integer, Fraction, or exact rational string')
    return F(value)


def _exp_negative(x: F, terms=64):
    if x < 0 or type(terms) is not int or terms < 0 or x >= terms + 2:
        raise ValueError('Require x>=0 and a valid Taylor-tail order')
    term = total = F(1)
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    upper = total + term * x / (terms + 1) / (1 - x / (terms + 2))
    return 1 / upper, 1 / total


def _a_interval(rate, time, terms):
    if rate == 0:
        return time, time
    lo, hi = _exp_negative(rate * time, terms)
    return (1-hi)/rate, (1-lo)/rate


def _g_interval(a, b, time, terms):
    ea = _exp_negative(a*time, terms)
    if a == b:
        return time*ea[0], time*ea[1]
    eb = _exp_negative(b*time, terms)
    values = [(x-y)/(b-a) for x in ea for y in eb]
    return min(values), max(values)


def bounded_hazard_response(entrance_cap, exit_cap, time, reset_defect=1, terms=64):
    """Enclose L=G-eta K: every declared classical model has H>=L.

    H=integral[P_nuB(B,s)-P_nuA(B,s)]ds. nuA starts outside B;
    nuB starts inside. Entrance conditional laws at every entrance time are
    within TV eta of nuB. Outside-to-B hazards are <=a; B-to-outside <=b.
    The theorem's lower bound is sharp over finite CTMC kernels.
    """
    a, b, t, eta = map(rational, (entrance_cap, exit_cap, time, reset_defect))
    if min(a,b) < 0 or t <= 0 or not 0 <= eta <= 1:
        raise ValueError('Require nonnegative caps, positive time, eta in [0,1]')
    aa, ab = _a_interval(a,t,terms), _a_interval(b,t,terms)
    g = _g_interval(a,b,t,terms)
    unrestricted = (aa[0]+ab[0]-t, aa[1]+ab[1]-t)
    lower = ((1-eta)*g[0]+eta*unrestricted[0],
             (1-eta)*g[1]+eta*unrestricted[1])
    k = ((F(0),F(0)) if a==0 or b==0 else
         (max(F(0),t-aa[1]-ab[1]+g[0]),
          max(F(0),t-aa[0]-ab[0]+g[1])))
    return {'entrance_cap':a, 'exit_cap':b, 'time':t, 'reset_defect':eta,
            'A_entrance':aa, 'A_exit':ab, 'G':g, 'K':k,
            'lower_bound_interval':lower,
            'certified_strictly_positive':lower[0]>0,
            'terms':terms}


def mixed_preparation_bound(entrance_cap, exit_cap, time, initial_inside,
                            initial_reset_defect=1, entrance_reset_defect=1, terms=64):
    """Sharp extension when comparison preparation already has mass p inside B.

    rho=p*nu_in+(1-p)*nu_out; TV(nu_in,nuB)<=eta0. The outside component's
    first-entrance reset defect is <=eta1. H=U_nuB-U_rho. Neither initial
    occupancy nor quantum coherence is inferred from this classical contract.
    """
    p,e0=map(rational,(initial_inside,initial_reset_defect))
    if not 0<=p<=1 or not 0<=e0<=1:
        raise ValueError('Initial occupancy and reset defect must be in [0,1]')
    result=bounded_hazard_response(entrance_cap,exit_cap,time,entrance_reset_defect,terms)
    t=result['time'];lo,hi=result['lower_bound_interval'];ab=result['A_exit']
    bound=((1-p)*lo-p*e0*(t-ab[0]),(1-p)*hi-p*e0*(t-ab[1]))
    return {'initial_inside':p,'initial_reset_defect':e0,
            'entrance_reset_defect':result['reset_defect'],
            'lower_bound_interval':bound,
            'certified_strictly_positive':bound[0]>0}


def reset_defect_lower_bound(response_upper, entrance_cap, exit_cap, time, terms=64):
    """Conservative eta lower bound from an already calibrated H upper bound.

    An output >1 contradicts the declared bounded-hazard classical contract.
    It is not a device-independent certification of quantum physics.
    """
    upper = rational(response_upper)
    result = bounded_hazard_response(entrance_cap,exit_cap,time,1,terms)
    numerator = result['G'][0] - upper
    if numerator <= 0:
        defect = F(0)
    elif result['K'][1] == 0:
        return {'minimum_reset_defect':None, 'status':'incompatible_hazard_contract'}
    else:
        defect = numerator/result['K'][1]
    return {'minimum_reset_defect':defect,
            'status':'incompatible_hazard_contract' if defect>1 else 'conditional_defect_bound'}


def classical_hazard_lower_bound(contrast_upper, time, bits=48):
    """Invert the sharp eta=1 law using exact rational interval arithmetic.

    contrast_upper is a calibrated upper bound on H/time.  The returned lower
    bound applies to max(entrance hazard, exit hazard) in every classical model
    with finite conditional hazards reproducing that contrast.  It is an
    explanatory activity requirement, not unconditional quantum certification.
    """
    c,t=map(rational,(contrast_upper,time))
    if t<=0 or type(bits) is not int or bits<1:
        raise ValueError('Positive time and positive integer bisection count required')
    if c>=1:
        return {'hazard_lower_bound':F(0),'status':'no_positive_activity_bound'}
    if c<=-1:
        return {'hazard_lower_bound':None,'status':'incompatible_finite_hazard_contract'}
    def f_interval(z):
        if z==0:return F(1),F(1)
        if z>=64:
            # exp(z)>=(1+z/64)^64; enough even for extremely large roots.
            lo,hi=F(0),(F(64)/(64+z))**64
        else:
            lo,hi=_exp_negative(z)
        return 2*(1-hi)/z-1,2*(1-lo)/z-1
    lo=F(0);hi=2/(1+c);completed=0
    for _ in range(bits):
        mid=(lo+hi)/2;lower,upper=f_interval(mid)
        if lower>c:lo=mid
        elif upper<c:hi=mid
        else:break
        completed+=1
    return {'hazard_lower_bound':lo/t,'root_hazard_upper_bound':hi/t,
            'contrast_upper':c,'time':t,'bisections':completed,
            'lower_endpoint_response_interval':f_interval(lo),
            'status':'conditional_classical_activity_lower_bound'}


def three_state_exclusion(contrast_upper, initial_inside_lower=1, initial_outside_upper=0):
    """Certify a >=4-state requirement for a common homogeneous classical CTMC.

    The binary detector response f(x) may be stochastic but must be fixed.
    contrast_upper bounds the uniform averaged contrast, or a mixture of
    protected uniform clocks. The two initial bounds concern
    nuB f and nuA f. Their confidence/calibration coverage is external.
    The conclusion concerns state count, not quantum certification or the
    exact minimal model of a complete record law.
    """
    c,mb,ma=map(rational,(contrast_upper,initial_inside_lower,initial_outside_upper))
    if not 0<=mb<=1 or not 0<=ma<=1:
        raise ValueError('Initial binary mean bounds must lie in [0,1]')
    lower=max(F(-1),mb-ma-1)
    if c < -1:
        return {'classical_three_state_lower_bound':lower,
                'minimum_classical_states':None,
                'status':'incompatible_binary_probability_contract'}
    excluded=c<lower
    return {'classical_three_state_lower_bound':lower,
            'contrast_drop_lower_bound':mb-ma-c,
            'three_state_absolute_drop_bound':F(1),
            'minimum_classical_states':4 if excluded else None,
            'status':'excludes_at_most_three_homogeneous_classical_states' if excluded else 'inconclusive'}


def _graph(n, edges, detector):
    if type(n) is not int or n<1:
        raise ValueError('Positive number of states required')
    b = set(detector)
    if not b <= set(range(n)):
        raise ValueError('Detector states out of range')
    adj = [set() for _ in range(n)]
    for x,y in edges:
        if type(x) is not int or type(y) is not int or not (0<=x<n and 0<=y<n) or x==y:
            raise ValueError('Directed edges must join distinct valid states')
        adj[x].add(y)
    return adj,b


def _reachable(adj, start, deleted):
    if start==deleted:
        return set()
    seen={start}; todo=[start]
    while todo:
        for target in adj[todo.pop()]:
            if target != deleted and target not in seen:
                seen.add(target); todo.append(target)
    return seen


def gate_dominance(n, edges, detector):
    """Sufficient, not claimed complete, rate-uniform occupation preorder.

    x dominates y means U_x(t)>=U_y(t) for every t and every supported CTMC.
    Each primitive is a first-hitting certificate; inequalities compose.
    """
    adj,b = _graph(n,edges,detector)
    witnesses = {(x,x):[x] for x in range(n)}
    primitive = []
    for x in range(n):
        for y in range(n):
            if x==y:
                continue
            kind=None
            if x in b and _reachable(adj,x,y) <= b:
                kind='exit_gateway'
            elif y not in b and not (_reachable(adj,y,x) & b):
                kind='entrance_gateway'
            if kind is not None:
                witnesses[x,y]=[x,y]
                primitive.append({'higher':x,'lower':y,'reason':kind})
    for k in range(n):
        for x in range(n):
            for y in range(n):
                if (x,y) not in witnesses and (x,k) in witnesses and (k,y) in witnesses:
                    witnesses[x,y]=witnesses[x,k]+witnesses[k,y][1:]
    return {'primitive':primitive,
            'relations':[{'higher':x,'lower':y,'path':p} for (x,y),p in sorted(witnesses.items())],
            'complete_characterization':False}


def contamination_envelope(exit_perturbation_cap, record_times):
    """TV envelope from adding transitions of total intensity <=epsilon.

    Every record's horizon is a pathwise upper bound and controls preserve
    the intensity bound. Includes adaptive within-record protocols by coupling.
    Independent resetting records add; no bound on statistical power is hidden.
    """
    epsilon=rational(exit_perturbation_cap)
    times=[rational(t) for t in record_times]
    if epsilon<0 or any(t<0 for t in times):
        raise ValueError('Rates and horizon bounds must be nonnegative')
    return min(F(1),epsilon*sum(times,F(0)))


def _jsonable(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonable(v) for v in x]
    return x


def make_results():
    import numpy as np
    from scipy.linalg import expm
    from scipy.optimize import brentq
    def generator(n,edges):
        q=np.zeros((n,n))
        for i,j,r in edges:q[i,j]+=r;q[i,i]-=r
        return q
    def integrated(q,t):
        block=np.zeros((5,5));block[:4,:4]=q;block[:4,4]=[0,1,1,0]
        u=expm(t*block)[:4,4]
        return u[1]-u[0]
    def lower(a,b,t,eta):
        aa=-np.expm1(-a*t)/a;ab=-np.expm1(-b*t)/b
        g=t*np.exp(-a*t) if a==b else (np.exp(-a*t)-np.exp(-b*t))/(b-a)
        return (1-eta)*g+eta*(aa+ab-t)
    rng=np.random.default_rng(20261012)
    rows=[];max_error=0.0
    for a,b,t,eta in [(1.,1.,2.,.5),(1.,1.,2.,1.),(2.,.7,3.,.3)]+[
        tuple(rng.uniform([.1,.1,.05,0],[3,3,4,1])) for _ in range(80)]:
        q=generator(4,[(0,1,a*(1-eta)),(0,2,a*eta),(1,3,b)])
        actual=integrated(q,t);expected=lower(a,b,t,eta)
        max_error=max(max_error,abs(actual-expected))
        if len(rows)<3:rows.append({'a':a,'b':b,'t':t,'eta':eta,'H':actual,'sharp_lower_bound':expected})
    family=[]
    a=b=1.;eta=.8;t=3.
    for eps in [.1,.01,.001,.0001]:
        qr=generator(4,[(0,1,a*(1-eta)),(0,2,a*eta),(1,0,eps),(1,3,b-eps),(2,1,eps),(3,0,eps)])
        qnative=qr+generator(4,[(0,1,a*(1-eta)),(1,0,eps)])
        system=qnative.T.copy();system[-1,:]=1;rhs=np.array([0.,0.,0.,1.])
        pi=np.linalg.solve(system,rhs)
        current=2*pi[0]*a*(1-eta)-2*pi[1]*eps
        response=(pi@expm(t*qr))[1:3].sum()-pi[1:3].sum()
        h=integrated(qr,t)
        family.append({'epsilon':eps,'H':h,'limit':lower(a,b,t,eta),
                       'native_current':current,'response':response,
                       'response_identity_error':abs(response+current*h/2)})
    phase=brentq(lambda z:2*(-np.expm1(-z))-z,1,2,xtol=1e-14)
    closure=gate_dominance(4,[(0,2),(2,3),(1,2),(3,2)],{1,2})
    output={'evidence':'mathematical derivations and synthetic numerical verification; no empirical data',
            'seed':20261012,'sharp_extremizers_checked':83,'maximum_numerical_residual':max_error,
            'sharp_examples':rows,'irreducible_stationary_channel_approximation':family,
            'equal_hazard_unrestricted_zero_numerical':phase,
            'inverse_activity_example':classical_hazard_lower_bound('-1/10',1),
            'state_count_example':three_state_exclusion('-1/10','49/50','1/50'),
            'phase_bracket_certificates':{
                'positive_1593_over_1000':bounded_hazard_response(1,1,'1593/1000'),
                'negative_1594_over_1000':bounded_hazard_response(1,1,'797/500')},
            'composed_gate_example':closure,
            'scope':'Graph preorder is sufficient only; hazard and reset caps require external calibration.'}
    return _jsonable(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=make_results()
    serialized=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(serialized)
    else:print(serialized)
