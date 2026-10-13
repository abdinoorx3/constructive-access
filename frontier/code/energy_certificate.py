"""Exact dual certificates for stationary classical entropy-production bounds.

The physical bridge and moment-cone premise are separate theorems. This module
checks a supplied probability box and rational positive test scores against the
complete reversible spectral-mixture class. It does not infer stationarity,
noninvasive observation, physical time-reversal parity, or local detailed balance.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb,ceil
from pathlib import Path
import json


def exact(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,F)):
        raise TypeError('exact integer, rational string or Fraction required')
    return F(x)


@lru_cache(maxsize=512,typed=True)
def log_interval(value):
    """Rational atanh-series enclosure with dyadic outward rounding."""
    x=exact(value)
    if x<=0:raise ValueError('positive logarithm argument required')
    if x==1:return F(0),F(0)
    if x<1:
        lo,hi=log_interval(1/x);return -hi,-lo
    k=0
    while x>2:x/=2;k+=1
    z=(x-1)/(x+1);terms=80
    lo=2*sum((z**(2*j+1)/(2*j+1) for j in range(terms)),F(0))
    hi=lo+2*z**(2*terms+1)/((2*terms+1)*(1-z*z))
    if k:
        a,b=log_interval(F(2));lo+=k*a;hi+=k*b
    scale=1<<90
    return F(lo.numerator*scale//lo.denominator,scale),F(ceil(hi*scale),scale)


def mean_square_interval(mean_interval):
    lo,hi=map(exact,mean_interval)
    if not -1<=lo<=hi<=1:raise ValueError('mean interval outside [-1,1]')
    lower=F(0) if lo<=0<=hi else min(lo*lo,hi*hi)
    return lower,max(lo*lo,hi*hi)


def bernstein_coefficients(power):
    """Power coefficients on [0,1] -> same-degree Bernstein coefficients."""
    power=list(map(exact,power));degree=len(power)-1
    if degree<0:raise ValueError('nonempty polynomial required')
    return [sum((power[j]*F(comb(k,j),comb(degree,j)) for j in range(k+1)),F(0)) for k in range(degree+1)]


def bisect_bernstein(coefficients):
    row=list(coefficients);left=[row[0]];right=[row[-1]]
    while len(row)>1:
        row=[(a+b)/2 for a,b in zip(row,row[1:])]
        left.append(row[0]);right.append(row[-1])
    return left,list(reversed(right))


def polynomial_upper(power,depth=4):
    """Certified upper bound using convex hulls on 2**depth equal cells."""
    if type(depth) is not int or not 0<=depth<=12:raise ValueError('depth must lie in 0..12')
    leaves=[bernstein_coefficients(power)]
    for _ in range(depth):
        leaves=[part for row in leaves for part in bisect_bernstein(row)]
    return max(max(row) for row in leaves)


def validate(lags,step,weights,intervals,mean_interval,scores):
    if not lags or any(type(k) is not int or k<=0 for k in lags):raise ValueError('positive integer lags required')
    if len(set(lags))!=len(lags):raise ValueError('distinct lag indices required')
    if max(lags)>256:raise ValueError('certificate degree exceeds declared computational cap256')
    step=exact(step);weights=list(map(exact,weights))
    intervals=[tuple(map(exact,p)) for p in intervals]
    scores=[tuple(map(exact,p)) for p in scores]
    if not len(lags)==len(weights)==len(intervals)==len(scores):raise ValueError('inconsistent menu lengths')
    if step<=0 or any(w<=0 for w in weights) or sum(weights)!=1:raise ValueError('positive time and normalized positive weights required')
    if any(len(p)!=2 or not 0<=p[0]<=p[1]<=1 for p in intervals):raise ValueError('probability intervals invalid')
    if any(len(p)!=2 or min(p)<=0 for p in scores):raise ValueError('strictly positive zero/one scores required')
    ms=mean_square_interval(mean_interval)
    return step,weights,intervals,scores,((1+ms[0])/2,(1+ms[1])/2)


def certify(lags,step,weights,intervals,mean_interval,scores,depth=4):
    """Certify a lower bound for sigma in natural entropy units per time.

    intervals bound same-outcome probabilities at lags*step simultaneously;
    mean_interval bounds the stationary +/-1 readout mean on the same event.
    Scores may be selected after constructing that simultaneous confidence box:
    minimization over the entire box preserves its joint coverage. Confidence
    failure is external to this deterministic certificate, never silently zero.
    """
    step,weights,intervals,scores,radii=validate(lags,step,weights,intervals,mean_interval,scores)
    polynomials=[];bounds=[]
    for base in sorted(set(radii)):
        coeff=[F(0)]*(max(lags)+1)
        for k,w,(h0,h1) in zip(lags,weights,scores):
            coeff[0]+=w*(h0+(h1-h0)*base)
            coeff[k]+=w*(h1-h0)*(1-base)
        polynomials.append(coeff);bounds.append(polynomial_upper(coeff,depth))
    normalizer=max(bounds)
    if normalizer<=0:raise ArithmeticError('positive scores imply positive normalizer')
    expected_log=F(0)
    for w,(lo,hi),(h0,h1) in zip(weights,intervals,scores):
        log0=log_interval(h0)[0];log1=log_interval(h1)[0]
        expected_log+=w*min((1-lo)*log0+lo*log1,(1-hi)*log0+hi*log1)
    divergence=max(F(0),expected_log-log_interval(normalizer)[1])
    mean_time=step*sum((w*k for w,k in zip(weights,lags)),F(0))
    rate=4*divergence/mean_time
    return dict(status='positive_certificate' if rate>0 else 'no_positive_certificate',
                lags=list(lags),step=step,weights=weights,intervals=intervals,
                mean_interval=tuple(map(exact,mean_interval)),scores=scores,depth=depth,
                reversible_baseline_interval=radii,normalizer_upper=normalizer,
                polynomial_coefficients=polynomials,polynomial_upper_bounds=bounds,
                expectation_log_lower=expected_log,record_kl_lower=divergence,
                expected_lag=mean_time,entropy_production_rate_lower=rate,
                contract='stationary ordinary-reversal CTMC; same noninvasive fixed binary detector; independent randomized lag; joint confidence box supplied externally')


def verify(candidate,lags,step,weights,intervals,mean_interval,scores,depth=4):
    """Bind certificate to independently supplied data and physical menu."""
    try:return candidate==certify(lags,step,weights,intervals,mean_interval,scores,depth)
    except (TypeError,ValueError,KeyError,ZeroDivisionError):return False


def resource_frontier(certificate,preparation_kl_cap=None,reference_mean_calibrated=False):
    """General nonstationary preparation/housekeeping consequence.

    Requires the same homogeneous generator with positive stationary pi, not
    stationary initialization. Reference mean is pi's detector mean, which a
    transient observed mean does not calibrate. A nontrivial reference-mean
    interval requires a separate calibration assertion; otherwise use [-1,1].
    This function does not turn such an assertion into empirical evidence.
    """
    fields=('lags','step','weights','intervals','mean_interval','scores','depth')
    if not verify(certificate,**{k:certificate[k] for k in fields}):
        raise ValueError('invalid underlying dual certificate')
    if certificate['mean_interval']!=(F(-1),F(1)) and not reference_mean_calibrated:
        raise ValueError('stationary reference mean must be independently calibrated, or use [-1,1]')
    bound=certificate['record_kl_lower']
    result=dict(preparation_plus_half_housekeeping_lower=bound,
                detailed_balance_preparation_kl_lower=bound,
                reference_mean_calibrated=bool(reference_mean_calibrated),
                contract='D(nu||pi) + one half expected integrated housekeeping entropy >= record KL bound; fixed Q, positive stationary pi, common noninvasive detector')
    if preparation_kl_cap is not None:
        cap=exact(preparation_kl_cap)
        if cap<0:raise ValueError('nonnegative preparation KL cap required')
        result.update(preparation_kl_cap=cap,integrated_housekeeping_lower=2*max(F(0),bound-cap))
    return result


def design(lags,step,weights,intervals,mean_interval=(-1,1),grid=65,depth=5):
    """Numerical score proposal followed by a global exact dual certificate.

    The optimization grid proposes a mixture and has no authority to certify a
    bound. Rational score rounding and the continuous polynomial verifier supply
    validity even if the numerical solver is inaccurate or its grid is coarse.
    No global optimality or experimental calibration is claimed.
    """
    import numpy as np
    from scipy.optimize import minimize
    if type(grid) is not int or not 3<=grid<=257:raise ValueError('grid must lie in 3..257')
    dummy=[(F(1),F(1)) for _ in lags]
    step,w,boxes,_,r=validate(lags,step,weights,intervals,mean_interval,dummy)
    wf=np.array(list(map(float,w)));mid=np.array([float((a+b)/2) for a,b in boxes])
    points=np.linspace(0,1,grid);base=float(r[0])
    A=base+(1-base)*np.array([points**k for k in lags])
    def objective(mass):
        q=np.clip(A@mass,1e-12,1-1e-12)
        return float(-np.sum(wf*(mid*np.log(q)+(1-mid)*np.log1p(-q))))
    def gradient(mass):
        q=np.clip(A@mass,1e-12,1-1e-12)
        return A.T@(wf*(q-mid)/(q*(1-q)))
    fit=minimize(objective,np.ones(grid)/grid,jac=gradient,method='SLSQP',
                 bounds=[(0,1)]*grid,constraints={'type':'eq','fun':lambda x:np.sum(x)-1,'jac':lambda x:np.ones(grid)},
                 options={'ftol':1e-12,'maxiter':1000})
    mass=np.maximum(0,fit.x)
    mass=mass/mass.sum() if mass.sum()>0 else np.ones(grid)/grid
    q=np.clip(A@mass,1e-8,1-1e-8);p=np.clip(mid,1e-8,1-1e-8)
    scores=[(F(float((1-a)/(1-b))).limit_denominator(10**8),F(float(a/b)).limit_denominator(10**8)) for a,b in zip(p,q)]
    cert=certify(lags,step,w,boxes,mean_interval,scores,depth)
    return dict(certificate=cert,proposal_status=str(fit.message),proposal_success=bool(fit.success),
                proposal_grid=grid,proposal_objective=float(fit.fun),
                validity='Only the exact continuous-domain certificate is authoritative; numerical optimization is a proposal.')

def example():
    # Synthetic confidence box containing the 40-ring model's same-outcome laws.
    # Reversible q1>=q2. Scores fitted to a violating revival, then rounded exactly.
    contract=dict(lags=[16,21],step=F(1,10),weights=[F(1,2),F(1,2)],
                  intervals=[(F(501,1000),F(523,1000)),(F(725,1000),F(747,1000))],
                  mean_interval=(F(-1),F(1)),
                  scores=[(F(488,376),F(512,624)),(F(264,376),F(736,624))],depth=4)
    certificate=certify(**contract)
    assert verify(certificate,**contract)
    return dict(evidence='synthetic probability confidence box; not an empirical dataset',
                certificate=certificate,resource_frontier=resource_frontier(certificate),
                entropy_rate_lower_float=float(certificate['entropy_production_rate_lower']),
                rate_units='natural entropy per declared model time; heat conversion needs separate physical assumptions')


def from_counts(lags,step,weights,records,successes,radius,alpha,scores,
                calibration_radii=None,depth=4):
    """Fixed independent pair samples -> simultaneous box -> dual bound.

    Each record contains a same/not-same pair at one lag. These must be
    independently prepared stationary runs for the stationary interpretation,
    not overlapping time-series pairs treated as independent. Statistical
    coverage is conditional on the supplied whole-record calibration bounds.
    The reference stationary mean remains unrestricted, so the same box also
    supports the preparation/housekeeping tradeoff without stationarity.
    """
    radius,alpha=map(exact,(radius,alpha));m=len(lags)
    if not m or len(records)!=m or len(successes)!=m or not 0<radius<1 or not 0<alpha<1:
        raise ValueError('matching counts and valid error parameters required')
    if any(type(n) is not int or n<=0 for n in records):raise ValueError('positive integer counts required')
    if any(type(s) is not int or not 0<=s<=n for s,n in zip(successes,records)):
        raise ValueError('success count outside record count')
    if 2*min(records)*radius*radius < log_interval(2*m/alpha)[1]:
        raise ValueError('supplied radius does not certify the requested joint coverage')
    errors=[F(0)]*m if calibration_radii is None else list(map(exact,calibration_radii))
    if len(errors)!=m or any(not 0<=e<=1 for e in errors):raise ValueError('invalid calibration radii')
    intervals=[(max(F(0),F(s,n)-radius-e),min(F(1),F(s,n)+radius+e))
               for n,s,e in zip(records,successes,errors)]
    cert=certify(lags,step,weights,intervals,(-1,1),scores,depth)
    return dict(records=list(records),successes=list(successes),radius=radius,
                statistical_failure_upper=alpha,calibration_radii=errors,
                certificate=cert,resource_frontier=resource_frontier(cert),
                evidence='input counts are user supplied; code does not establish their physical origin or calibration')


def curvature_example():
    # The following rational detector values define the model exactly; they are
    # chosen to resemble a cosine but no exact sinusoidal identity is assumed.
    import math
    try:from .thermodynamic import ring_same_bounds,outward_decimal
    except ImportError:from thermodynamic import ring_same_bounds,outward_decimal
    detector=[F(round(10**6*math.cos(2*math.pi*j/40)),10**6) for j in range(40)]
    exact_model=[ring_same_bounds(40,F(20),F(1),F(3*k,20),detector=detector) for k in [1,2,3]]
    model=[outward_decimal(interval,10**10) for interval in exact_model]
    # A wider rational box illustrates a finite uncertainty envelope.
    boxes=[(lo-F(1,5000),hi+F(1,5000)) for lo,hi in model]
    v=[-1,2,-1];weights=[F(1,3)]*3;q=F(2,3);kappa=F(1,200)
    scores=[(1-kappa*a*q/w,1+kappa*a*(1-q)/w) for a,w in zip(v,weights)]
    cert=certify([1,2,3],F(3,20),weights,boxes,(-1,1),scores,depth=4)
    assert cert['entropy_production_rate_lower']>0
    assert boxes[0][0]>boxes[1][1]>boxes[2][1]>F(1,2)
    return dict(evidence='exact classical model prediction plus synthetic uncertainty box; no measured data',
                model=dict(states=40,forward_rate=20,backward_rate=1,detector_conditional_means=detector),
                exact_probability_intervals=model,certificate=cert,
                entropy_rate_lower_float=float(cert['entropy_production_rate_lower']),
                negative_correlation_bound=0,monotonic_revival_bound=0,
                resource_frontier=resource_frontier(cert,preparation_kl_cap=F(0)))


if __name__=='__main__':
    result=dict(revival=example(),curvature=curvature_example())
    path=Path(__file__).resolve().parents[1]/'results/energy_certificate.json'
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,default=str,sort_keys=True,indent=2)+'\n')
    print(json.dumps({name:row['entropy_rate_lower_float'] for name,row in result.items()}))
