"""Fixed-contrast confidence and entropy certificates for virtual/physical records.

Only the mathematical contract is checked. Physical stationarity, reversal
parity, detector noninvasiveness, conditional bias and time calibration are
external premises. A contrast must be frozen before its test records are seen.
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import json
from constructive_access.frontier import energy_certificate as ec


def sqrt_upper(value, bits=90):
    value=ec.exact(value)
    if value<0:raise ValueError('nonnegative square-root argument required')
    if type(bits) is not int or not 16<=bits<=256:raise ValueError('bits outside16..256')
    scale=1<<bits
    floor=isqrt(value.numerator*scale*scale//value.denominator)
    if F(floor*floor,scale*scale)==value:return F(floor,scale)
    return F(floor+1,scale)


def support(lags,coefficients,mean_interval=(-1,1),depth=6):
    """Exact upper support over the continuous reversible moment comparison set."""
    a=list(map(ec.exact,coefficients));m=len(lags)
    if not m or len(a)!=m or not any(a):raise ValueError('nonempty matching nonzero contrast required')
    ec.validate(lags,1,[F(1,m)]*m,[(0,1)]*m,mean_interval,[(1,1)]*m)
    lo,hi=ec.mean_square_interval(mean_interval)
    baselines=sorted(set(((1+lo)/2,(1+hi)/2)))
    polynomials=[];bounds=[]
    for r in baselines:
        power=[F(0)]*(max(lags)+1);power[0]=r*sum(a)
        for k,c in zip(lags,a):power[k]+=(1-r)*c
        polynomials.append(power);bounds.append(ec.polynomial_upper(power,depth))
    return dict(lags=list(lags),coefficients=a,mean_interval=tuple(map(ec.exact,mean_interval)),
                depth=depth,polynomials=polynomials,polynomial_bounds=bounds,
                reversible_support_upper=max(bounds))


def certify_counts(lags,coefficients,records,successes,durations,alpha=F(1,20),
                   conditional_radii=None,mean_interval=(-1,1),
                   selection='predeclared',depth=6,reference_mean_calibrated=False):
    """One-sided fixed-contrast certificate with conditional-mean calibration.

    For each setting k and each record, conditioned on the full preceding
    history, the mean agreement bit must lie in p_k +/- conditional_radii[k]
    for ONE fixed target record law p. Counts are fixed quotas (possibly
    predictably routed), not data-selected stopping counts. Independent records
    are the zero-radius special case. Coefficients/support/calibration must be
    fixed before test data, or chosen using a separately independent pilot.

    durations are positive calibrated upper expected FULL path durations for
    the corresponding target laws; not merely an integration-free nominal lag.
    Inference weights are mathematical mixtures and need not equal sample quotas.
    """
    if selection not in ('predeclared','independent_pilot'):
        raise ValueError('freeze contrast before test data; arbitrary postselection is unsupported')
    sup=support(lags,coefficients,mean_interval,depth);a=sup['coefficients'];m=len(a)
    if sup['mean_interval']!=(F(-1),F(1)) and not reference_mean_calibrated:
        raise ValueError('narrow stationary reference mean requires separate calibration')
    alpha=ec.exact(alpha);times=list(map(ec.exact,durations))
    if not 0<alpha<1 or not len(records)==len(successes)==len(times)==m:
        raise ValueError('invalid confidence allocation or menu lengths')
    if any(type(n) is not int or n<=0 for n in records):raise ValueError('positive fixed integer quotas required')
    if any(type(s) is not int or not 0<=s<=n for s,n in zip(successes,records)):
        raise ValueError('success count outside quota')
    if any(t<=0 for t in times):raise ValueError('positive full-duration expectations required')
    delta=[F(0)]*m if conditional_radii is None else list(map(ec.exact,conditional_radii))
    if len(delta)!=m or any(not 0<=d<=1 for d in delta):raise ValueError('invalid conditional probability radii')
    variance_proxy=sum((c*c/n for c,n in zip(a,records)),F(0))
    log_upper=ec.log_interval(1/alpha)[1]
    radius=sqrt_upper(log_upper*variance_proxy/2)
    calibration=sum((abs(c)*d for c,d in zip(a,delta)),F(0))
    empirical=sum((c*F(s,n) for c,s,n in zip(a,successes,records)),F(0))
    signed_lower=empirical-sup['reversible_support_upper']-radius-calibration
    gap=max(F(0),signed_lower)
    denominator_upper=sum((abs(c)*sqrt_upper(t) for c,t in zip(a,times)),F(0))**2
    rate=8*gap*gap/denominator_upper
    # A rational amplitude-weighted mixture maximizes this record-KL lower bound.
    active=[i for i,c in enumerate(a) if c]
    weights=[abs(c)/sum(map(abs,a)) for c in a]
    norm=sum((a[i]**2/weights[i] for i in active),F(0))
    kl=2*gap*gap/norm
    return dict(status='positive_certificate' if gap else 'no_positive_certificate',
                support=sup,lags=list(lags),coefficients=a,records=list(records),successes=list(successes),
                durations=times,alpha=alpha,conditional_radii=delta,selection=selection,
                mean_interval=tuple(map(ec.exact,mean_interval)),depth=depth,
                reference_mean_calibrated=bool(reference_mean_calibrated),
                empirical_contrast=empirical,concentration_variance_proxy=variance_proxy,
                sampling_radius_upper=radius,conditional_bias_penalty=calibration,
                signed_contrast_lower=signed_lower,contrast_lower=gap,
                entropy_denominator_upper=denominator_upper,entropy_production_rate_lower=rate,
                resource_mixture_weights=weights,record_kl_lower=kl,
                preparation_plus_half_housekeeping_lower=kl,
                confidence_failure_upper=alpha,
                contract='fixed contrast and quotas; declared conditional record means; reversible moment reference; stationary ordinary-reversal CTMC for rate; calibrated full path durations')


def verify(candidate,**contract):
    try:return candidate==certify_counts(**contract)
    except (ValueError,TypeError,ZeroDivisionError,KeyError):return False


def allocation(coefficients,costs,budget):
    """Numerical continuous optimum plus rounded feasible fixed integer quotas.

    Minimize sum a_k^2/n_k subject to sum c_k n_k<=budget. The result is a
    proposal, not a global integer optimum. Exact verification of its cost
    and subsequent confidence bound supplies the usable guarantee.
    """
    import math
    a=list(map(ec.exact,coefficients));cost=list(map(ec.exact,costs));budget=ec.exact(budget)
    if not a or len(a)!=len(cost) or any(c==0 for c in a) or any(c<=0 for c in cost):
        raise ValueError('matching nonzero coefficients and positive costs required')
    if budget<sum(cost):raise ValueError('budget cannot fund one record per setting')
    norm=sum(abs(float(c))*math.sqrt(float(t)) for c,t in zip(a,cost))
    proposed=[max(1,math.floor(float(budget)*abs(float(c))/(math.sqrt(float(t))*norm))) for c,t in zip(a,cost)]
    def total():return sum((n*c for n,c in zip(proposed,cost)),F(0))
    while total()>budget:
        available=[i for i,n in enumerate(proposed) if n>1]
        if not available:raise ArithmeticError('feasibility correction failed')
        i=min(available,key=lambda i:float(a[i]**2/F(proposed[i]*(proposed[i]-1))/cost[i]))
        proposed[i]-=1
    return dict(records=proposed,costs=cost,budget=budget,actual_cost=total(),
                variance_proxy=sum((c*c/n for c,n in zip(a,proposed)),F(0)),
                integer_optimality_claimed=False)


def reversible_box_witness(lags,intervals,grid=100):
    """Exact two-state equilibrium point inside a supplied probability box.

    The searched curve is q_k=(1+A*x**k)/2, with 0<x<1 and0<=A<=1.
    It is realized at deterministic k*Delta by a symmetric two-state CTMC
    with jump rate -log(x)/(2*Delta), stationary preparation and readout means
    +/-sqrt(A). For general compatible clock families this is a moment-cone
    point, not automatically a realization under that particular clock law.
    Failure of this limited rational-grid search proves no infeasibility.
    """
    if type(grid) is not int or not 2<=grid<=1000:raise ValueError('grid must lie in2..1000')
    m=len(lags);boxes=[tuple(map(ec.exact,z)) for z in intervals]
    ec.validate(lags,1,[F(1,m)]*m,boxes,(-1,1),[(1,1)]*m)
    for j in range(1,grid):
        x=F(j,grid)
        lo=max([F(0)]+[(2*a-1)/x**k for k,(a,b) in zip(lags,boxes)])
        hi=min([F(1)]+[(2*b-1)/x**k for k,(a,b) in zip(lags,boxes)])
        if lo<=hi:
            amplitude=(lo+hi)/2
            return dict(status='feasible_equilibrium_point',lags=list(lags),intervals=boxes,
                        decay_per_step=x,squared_detector_mean=amplitude,
                        probabilities=[(1+amplitude*x**k)/2 for k in lags],
                        physical_interpretation='symmetric two-state equilibrium at deterministic indexed times; for other clock laws only comparison-cone membership is asserted')
    return dict(status='search_inconclusive',lags=list(lags),intervals=boxes,grid=grid)


def verify_box_witness(candidate,lags,intervals):
    """Check membership directly, independently of the grid search."""
    try:
        boxes=[tuple(map(ec.exact,z)) for z in intervals]
        m=len(lags);ec.validate(lags,1,[F(1,m)]*m,boxes,(-1,1),[(1,1)]*m)
        if candidate['status']!='feasible_equilibrium_point' or candidate['lags']!=list(lags) or candidate['intervals']!=boxes:return False
        x=ec.exact(candidate['decay_per_step']);a=ec.exact(candidate['squared_detector_mean'])
        if not 0<x<1 or not 0<=a<=1:return False
        q=[(1+a*x**k)/2 for k in lags]
        return candidate['probabilities']==q and all(lo<=v<=hi for v,(lo,hi) in zip(q,boxes))
    except (TypeError,ValueError,KeyError,ZeroDivisionError):return False


def jsonable(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def example():
    contract=dict(lags=[1,2,3],coefficients=[-1,2,-1],records=[50000]*3,
                  successes=[35849,32265,27572],durations=[F(3,20),F(3,10),F(9,20)],alpha=F(1,20))
    cert=certify_counts(**contract);assert verify(cert,**contract)
    return dict(evidence='illustrative synthetic counts; not a laboratory dataset',certificate=cert,
                equal_record_proposal=allocation([-1,2,-1],[1,1,1],150000),
                equal_time_proposal=allocation([-1,2,-1],[F(3,20),F(3,10),F(9,20)],45000))


if __name__=='__main__':
    out=example();path=Path(__file__).resolve().parents[1]/'results/contrast.json'
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(jsonable(out),sort_keys=True,indent=2)+'\n')
    print(json.dumps({'entropy_rate_lower':float(out['certificate']['entropy_production_rate_lower'])}))
