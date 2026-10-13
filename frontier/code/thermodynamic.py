#!/usr/bin/env python3
"""Finite-lag stationary entropy certificates, using rational arithmetic.

The physical interpretation assumes an even-state, noninvasively observed,
stationary classical Markov jump process with local detailed balance. These
are not entropy bounds for arbitrary quantum or underdamped dynamics.
"""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import json


def _fraction(x):
    if type(x) is not F:
        raise TypeError('All real inputs must be fractions.Fraction')
    return x


def log_bounds(x: F, terms=80):
    """Enclose log(x) by the atanh series with a geometric tail."""
    _fraction(x)
    if x <= 0 or type(terms) is not int or terms < 1:
        raise ValueError('Positive x and positive integer terms required')
    z = (x-1)/(x+1)
    total = sum((2*z**(2*k+1)/F(2*k+1) for k in range(terms)), F(0))
    error = 2*abs(z)**(2*terms+1)/((2*terms+1)*(1-z*z))
    return total-error, total+error


def binary_kl_bounds(p: F, q: F, terms=80):
    """Enclose natural-log Bernoulli KL; infinite cases are rejected."""
    for x in (p, q): _fraction(x)
    if not 0 <= p <= 1 or not 0 <= q <= 1:
        raise ValueError('Probabilities must lie in [0,1]')
    if p == q: return F(0), F(0)
    if q in (0, 1): raise ValueError('KL is infinite at this endpoint')
    lo = hi = F(0)
    for mass, reference in ((p,q), (1-p,1-q)):
        if mass:
            a,b = log_bounds(mass/reference, terms)
            lo += mass*a
            hi += mass*b
    return max(F(0),lo), hi


def single_lag_entropy_lower(same_probability_upper: F, lag: F,
                             reversible_floor: F=F(1,2), terms=80):
    """Certified rate lower bound from an upper confidence bound on P(same)."""
    for x in (same_probability_upper,lag,reversible_floor): _fraction(x)
    p,q = same_probability_upper,reversible_floor
    if not 0 <= p <= 1 or not F(1,2) <= q < 1 or lag <= 0:
        raise ValueError('Invalid probability, floor or lag')
    return F(0) if p >= q else 4*binary_kl_bounds(p,q,terms)[0]/lag


def two_lag_entropy_lower(first_upper: F, second_lower: F,
                          first_lag: F, second_lag: F,
                          first_weight: F=F(1,2), terms=80):
    """Robust increasing-correlation certificate.

    Inputs are simultaneous bounds p1 <= first_upper, p2 >= second_lower.
    No stationary-mean or detector-gain calibration is needed if the same
    fixed noninvasive detector channel is used, with independent readout noise.
    """
    for x in (first_upper,second_lower,first_lag,second_lag,first_weight):
        _fraction(x)
    p1,p2,w = first_upper,second_lower,first_weight
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 < first_lag < second_lag
            and 0 < w < 1):
        raise ValueError('Invalid probabilities, ordered lags or weight')
    if p1 >= p2: return F(0)
    pooled = w*p1+(1-w)*p2
    information = (w*binary_kl_bounds(p1,pooled,terms)[0]
                   +(1-w)*binary_kl_bounds(p2,pooled,terms)[0])
    return 4*information/(w*first_lag+(1-w)*second_lag)


def two_lag_resource_frontier(first_upper: F, second_lower: F,
                              first_lag: F, second_lag: F,
                              preparation_upper: F,
                              first_weight: F=F(1,2), terms=80):
    """Preparation-or-housekeeping certificate without initial stationarity.

    preparation_upper bounds D(initial || stationary). Returned housekeeping
    is E[integral_0^U sigma_hk(s) ds], not stationary heat or total entropy.
    The detector remains fixed, noninvasive, and independently noisy.
    """
    _fraction(preparation_upper)
    if preparation_upper < 0: raise ValueError('Preparation budget must be nonnegative')
    mean_lag = first_weight*first_lag+(1-first_weight)*second_lag
    information_lower = two_lag_entropy_lower(first_upper,second_lower,
        first_lag,second_lag,first_weight,terms)*mean_lag/4
    return {'record_information_lower':information_lower,
            'preparation_divergence_upper':preparation_upper,
            'integrated_housekeeping_lower':max(F(0),2*(information_lower-preparation_upper)),
            'stationary_rate_lower_if_stationary':4*information_lower/mean_lag}


def decreasing_kl_projection(probabilities, weights, floor=F(1,2)):
    """Exact binomial PAVA onto q1 >= ... >= qK >= floor."""
    for x in [*probabilities,*weights,floor]: _fraction(x)
    if not probabilities or len(probabilities) != len(weights):
        raise ValueError('Equal nonempty input lengths required')
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probabilities')
    if any(w <= 0 for w in weights) or sum(weights) != 1:
        raise ValueError('Strictly positive weights must sum to one')
    if not 0 < floor < 1: raise ValueError('Interior floor required')
    blocks = []
    for i,(p,w) in enumerate(zip(probabilities,weights)):
        blocks.append([i,i+1,w,w*p])
        while len(blocks) >= 2 and blocks[-2][3]/blocks[-2][2] < blocks[-1][3]/blocks[-1][2]:
            b = blocks.pop(); a = blocks.pop()
            blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    result = [F(0)]*len(probabilities)
    for start,end,w,mass in blocks:
        for i in range(start,end): result[i] = max(floor,mass/w)
    return result


def order_entropy_lower(probabilities, lags, weights, floor=F(1,2), terms=80):
    """Safe relaxation of the full reversible autocorrelation moment cone.

    Here probabilities are exact model values, not unqualified sample means.
    Use two_lag_entropy_lower for a simultaneous-confidence-interval result.
    """
    for t in lags: _fraction(t)
    if len(lags) != len(probabilities) or any(t <= 0 for t in lags):
        raise ValueError('Positive matching lags required')
    if any(a >= b for a,b in zip(lags,lags[1:])):
        raise ValueError('Lags must be strictly increasing')
    q = decreasing_kl_projection(probabilities,weights,floor)
    divergence = sum((w*binary_kl_bounds(p,z,terms)[0]
                      for p,z,w in zip(probabilities,q,weights)), F(0))
    return {'projection':q, 'rate_lower':4*divergence/sum(t*w for t,w in zip(lags,weights))}


def from_counts(n1: int, successes1: int, n2: int, successes2: int,
                radius: F, alpha: F, first_lag: F, second_lag: F,
                first_weight: F=F(1,2), terms=80):
    """Two independent stationary-pair samples, certified Hoeffding radius.

    Two one-sided failure probabilities are union bounded by alpha.
    This does not certify consecutive overlapping pairs as independent.
    """
    for x in (radius,alpha,first_lag,second_lag,first_weight): _fraction(x)
    if any(type(x) is not int for x in (n1,successes1,n2,successes2)):
        raise TypeError('Counts must be integers')
    if not (n1 > 0 and n2 > 0 and 0 <= successes1 <= n1 and
            0 <= successes2 <= n2 and 0 < radius < 1 and 0 < alpha < 1):
        raise ValueError('Invalid counts, radius or error probability')
    log_upper = log_bounds(2/alpha,terms)[1]
    if 2*min(n1,n2)*radius**2 < log_upper:
        raise ValueError('Radius is not certified at the requested error level')
    first_upper = min(F(1), F(successes1,n1)+radius)
    second_lower = max(F(0), F(successes2,n2)-radius)
    return {'first_upper':first_upper,'second_lower':second_lower,
            'confidence_failure_upper':alpha,
            'rate_lower':two_lag_entropy_lower(first_upper,second_lower,
                first_lag,second_lag,first_weight,terms)}


def _exp_negative_bounds(x: F, terms: int):
    if x < 0 or x >= terms+2: raise ValueError('Need 0 <= x < terms+2')
    term = partial = F(1)
    for k in range(1,terms+1):
        term *= x/k; partial += term
    tail = term*x/(terms+1)/(1-x/(terms+2))
    return 1/(partial+tail),1/partial


def ring_same_bounds(states: int, forward: F, backward: F, lag: F, terms=180,
                     detector=None):
    """Exact uniformization enclosure for a stationary biased ring.

    The default detector is +1 on the first half of an even ring and -1 on
    the rest. Optional detector values are conditional means in [-1,1] for
    independent binary readouts. Uniform stationary distribution is exact.
    """
    for x in (forward,backward,lag): _fraction(x)
    if type(states) is not int or states < 4 or states % 2:
        raise ValueError('An even state count >=4 is required')
    if forward <= 0 or backward <= 0 or lag <= 0:
        raise ValueError('Positive rates and lag required')
    total_rate = forward+backward
    x = total_rate*lag
    if x >= terms+2: raise ValueError('Increase truncation terms')
    if detector is None:
        f = [F(1) if i < states//2 else F(-1) for i in range(states)]
    else:
        if len(detector) != states: raise ValueError('Detector length must match state count')
        for z in detector:
            _fraction(z)
            if not -1 <= z <= 1: raise ValueError('Detector means must lie in [-1,1]')
        f = list(detector)
    v = f[:]; weight = F(1); polynomial = sum(z*z for z in f)/states
    for k in range(1,terms+1):
        v = [(forward*v[(i+1)%states]+backward*v[(i-1)%states])/total_rate
             for i in range(states)]
        weight *= x/k
        polynomial += weight*sum(a*b for a,b in zip(f,v))/states
    exp_lo,exp_hi = _exp_negative_bounds(x,terms)
    tail_upper = exp_hi*weight*x/(terms+1)/(1-x/(terms+2))
    products = [exp_lo*polynomial,exp_hi*polynomial]
    return ((1+min(products)-tail_upper)/2,(1+max(products)+tail_upper)/2)


def outward_decimal(interval, denominator=10**12):
    lo,hi = interval
    return (F(lo.numerator*denominator//lo.denominator,denominator),
            F(-((-hi.numerator*denominator)//hi.denominator),denominator))


def make_results():
    t1,t2 = F(8,5),F(21,10)
    p1 = outward_decimal(ring_same_bounds(40,F(20),F(1),t1))
    p2 = outward_decimal(ring_same_bounds(40,F(20),F(1),t2))
    certified = two_lag_entropy_lower(p1[1],p2[0],t1,t2)
    stationary_rate = [19*x for x in log_bounds(F(20),200)]
    # Deliberately labelled synthetic counts; they are not empirical data.
    sample = from_counts(5000,2560,5000,3680,F(1,50),F(1,20),t1,t2)
    frontier = two_lag_resource_frontier(p1[1],p2[0],t1,t2,F(1,100))
    return {'model':'40-state stationary classical ring; rates 20 and 1; binary half-ring detector',
        'status':'exact rational certificates for a specified model; no laboratory data',
        'physical_contract':'ordinary even-state time reversal and local detailed balance',
        'lags':[str(t1),str(t2)],
        'same_probability_bounds':[[str(z) for z in p1],[str(z) for z in p2]],
        'individual_nonnegative_correlation_bounds':[0,0],
        'two_lag_rate_lower':str(outward_decimal((certified,certified))[0]),
        'two_lag_rate_lower_decimal':float(certified),
        'true_rate_bounds_decimal':[float(x) for x in stationary_rate],
        'synthetic_counts':[2560,3680], 'records_per_lag':5000,
        'synthetic_confidence_rate_lower':float(sample['rate_lower']),
        'synthetic_confidence_failure_upper':str(sample['confidence_failure_upper']),
        'preparation_housekeeping_frontier':{
            'record_information_lower_decimal':float(frontier['record_information_lower']),
            'preparation_divergence_upper':'1/100',
            'integrated_housekeeping_lower_decimal':float(frontier['integrated_housekeeping_lower']),
            'meaning':'same temporal record; initial stationarity may be replaced by the stated D(initial||stationary) budget'},
        'factor_four':'sharp for the path-divergence comparison near detailed balance; final observed bound need not be tight',
        'limitations':['stationarity','noninvasive observation','fixed detector channel and independent readout noise',
            'independent random lag selection','even-state physical time reversal','no claim of quantum entropy production']}


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1]/'results'/'thermodynamic.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(make_results(),indent=2)+'\n')
    print(destination)
