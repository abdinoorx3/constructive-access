#!/usr/bin/env python3
"""Certified preparation and randomized-readout comparison.

The classical statement is conditional on a singleton or regenerative reset
readout. It is not a device-independent witness. Numerical examples are
simulations, not measurements. Exact certificate helpers use Fraction inputs.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import factorial, isqrt
from pathlib import Path
import argparse
import json


def _rational(x, name='value'):
    if type(x) is not F:
        raise TypeError(f'{name} must be fractions.Fraction')
    return x


def _mul(x, y):
    endpoints = [a*b for a in x for b in y]
    return min(endpoints), max(endpoints)


def _add(x, y):
    return x[0]+y[0], x[1]+y[1]


def exp_negative_bounds(x: F, terms=64):
    """Enclose exp(-x) using positive Taylor series and geometric tail."""
    _rational(x)
    if x < 0 or type(terms) is not int or terms < 0 or x >= terms+2:
        raise ValueError('Need x >= 0, integer terms >=0, and x < terms+2')
    term = partial = F(1)
    for k in range(1, terms+1):
        term *= x/k
        partial += term
    remainder = term*x/(terms+1)/(1-x/(terms+2))
    return 1/(partial+remainder), 1/partial


def trig_bounds(x: F, kind: str, degree=80):
    """Enclose sin/cos by Taylor polynomial and Lagrange remainder."""
    _rational(x)
    if kind not in ('sin', 'cos') or type(degree) is not int or degree < 1:
        raise ValueError('Require sin/cos and integer degree >=1')
    parity = 1 if kind == 'sin' else 0
    partial = sum(((-1)**((k-parity)//2)*x**k/F(factorial(k))
                   for k in range(parity, degree+1, 2)), F(0))
    remainder = abs(x)**(degree+1)/F(factorial(degree+1))
    return partial-remainder, partial+remainder


def depolarizing_contrast_bounds(gamma: F, omega: F, horizon: F,
                                detector_gain: F=F(1), terms=80):
    """Rational enclosure of observed p(j|j)-p(j|i), uniform readout time.

    L rho = -i*(omega/2)[X,rho] + gamma*(Tr(rho) I/2-rho).
    A shared affine detector of positive gain preserves the classical sign.
    No detector offset appears because it cancels between preparations.
    """
    for x, name in ((gamma,'gamma'),(omega,'omega'),(horizon,'horizon'),
                    (detector_gain,'detector_gain')):
        _rational(x,name)
    if gamma < 0 or omega <= 0 or horizon <= 0 or not 0 <= detector_gain <= 1:
        raise ValueError('Require gamma >=0, omega,horizon >0, gain in [0,1]')
    sine = trig_bounds(omega*horizon, 'sin', terms)
    cosine = trig_bounds(omega*horizon, 'cos', terms)
    exponential = exp_negative_bounds(gamma*horizon, terms)
    bracket = _add(_mul((-gamma,-gamma), cosine),
                   _mul((omega,omega), sine))
    numerator = _add((gamma,gamma), _mul(exponential, bracket))
    scale = detector_gain/((gamma*gamma+omega*omega)*horizon)
    return numerator[0]*scale, numerator[1]*scale


def exponential_clock_contrast(gamma: F, omega: F, clock_rate: F):
    """Exact contrast for an independent exponential readout clock.

    It is positive even for the coherent model; not every protected clock has
    power against the same alternative.
    """
    for x in (gamma,omega,clock_rate):
        _rational(x)
    if gamma < 0 or omega <= 0 or clock_rate <= 0:
        raise ValueError('Require gamma>=0 and omega,clock_rate>0')
    return clock_rate*(clock_rate+gamma)/((clock_rate+gamma)**2+omega**2)


def sqrt_bounds(x: F, digits=60):
    """Rational square-root enclosure by an integer square-root certificate."""
    _rational(x)
    if x < 0 or type(digits) is not int or digits < 1:
        raise ValueError('Require x >=0 and positive integer digits')
    scale=10**digits
    floor=isqrt(x.numerator*scale*scale//x.denominator)
    return F(floor,scale), F(floor+1,scale)


def optimized_preparation_gap_bounds(gamma: F, omega: F, horizon: F,
                                     detector_gain: F=F(1), terms=80):
    """Best comparison-state gap relative to the trusted reference |0>.

    Assumes arbitrary coherent state preparation. This optimizes a specified
    quantum model, not a device-independent classical/quantum distinction.
    """
    c=depolarizing_contrast_bounds(gamma,omega,horizon,F(1),terms)
    sine=trig_bounds(omega*horizon,'sin',terms)
    cosine=trig_bounds(omega*horizon,'cos',terms)
    decay=exp_negative_bounds(gamma*horizon,terms)
    bracket=_add(_mul((omega,omega),cosine),_mul((gamma,gamma),sine))
    product=_mul(decay,bracket)
    denominator=(gamma*gamma+omega*omega)*horizon
    s=((omega-product[1])/denominator,(omega-product[0])/denominator)
    def square(interval):
        a,b=interval
        return (F(0) if a<=0<=b else min(a*a,b*b),max(a*a,b*b))
    norm2=_add(square(c),square(s))
    norm_lo=sqrt_bounds(norm2[0])[0]
    norm_hi=sqrt_bounds(norm2[1])[1]
    _rational(detector_gain,'detector_gain')
    if not 0<=detector_gain<=1:
        raise ValueError('Require detector gain in [0,1]')
    return detector_gain*(norm_lo-c[1])/2,detector_gain*(norm_hi-c[0])/2


def quadrature_preparation_gap_bounds(gamma: F, omega: F, horizon: F,
                                      detector_gain: F=F(1), terms=80):
    """Gap for a fixed Y eigenstate comparison preparation.

    Choose the sign of Y to match the positive sine term in the adjoint effect.
    The reference remains |0>; no model-dependent eigenvector optimization is
    needed for this preparation.
    """
    c=depolarizing_contrast_bounds(gamma,omega,horizon,F(1),terms)
    sine=trig_bounds(omega*horizon,'sin',terms)
    cosine=trig_bounds(omega*horizon,'cos',terms)
    bracket=_add(_mul((omega,omega),cosine),_mul((gamma,gamma),sine))
    product=_mul(exp_negative_bounds(gamma*horizon,terms),bracket)
    denominator=(gamma*gamma+omega*omega)*horizon
    s=((omega-product[1])/denominator,(omega-product[0])/denominator)
    _rational(detector_gain,'detector_gain')
    if not 0<=detector_gain<=1:
        raise ValueError('Require detector gain in [0,1]')
    return detector_gain*(s[0]-c[1])/2,detector_gain*(s[1]-c[0])/2


def baseline_subtracted_witness_bounds(gamma: F, omega: F, horizon: F,
                                       bloch_y: F, bloch_z: F,
                                       detector_gain: F=F(1), terms=80):
    """Exact C(0)-C(T) for antipodal Bloch preparations (0,y,z), -(0,y,z).

    A separate classical theorem supplies the <=3-hidden-state null bound 1.
    That null permits arbitrary preparations and stochastic detector f, and
    does not assume that this quantum model or detector gain describes the null.
    """
    for x in (bloch_y,bloch_z,detector_gain):
        _rational(x)
    if bloch_y* bloch_y+bloch_z*bloch_z>1 or not 0<=detector_gain<=1:
        raise ValueError('Need valid antipodal Bloch vectors and gain in [0,1]')
    c=depolarizing_contrast_bounds(gamma,omega,horizon,F(1),terms)
    q=quadrature_preparation_gap_bounds(gamma,omega,horizon,F(1),terms)
    s=(2*q[0]+c[0],2*q[1]+c[1])
    drift=(1-c[1],1-c[0])
    value=_add(_mul((bloch_z,bloch_z),drift),_mul((-bloch_y,-bloch_y),s))
    return detector_gain*value[0],detector_gain*value[1]


def comparison_benchmark():
    """Numerical optimal Hoeffding proxies, under common ideal readout.

    These are sufficient-bound design comparisons, not exact statistical minima.
    All methods share fixed gamma=.1, omega=2, independent records, zero
    calibration error, and free preparation. The quadrature/optimized methods
    additionally use coherent comparison preparation. Evolution-time costs are
    expectations for randomized clocks. Search results are numerical.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    gamma=.1;omega=2.
    def f(t): return np.exp(-gamma*t)*np.cos(omega*t)
    def c(t):
        return (gamma+np.exp(-gamma*t)*(-gamma*np.cos(omega*t)+omega*np.sin(omega*t)))/((gamma**2+omega**2)*t)
    def s(t):
        return (omega-np.exp(-gamma*t)*(omega*np.cos(omega*t)+gamma*np.sin(omega*t)))/((gamma**2+omega**2)*t)
    gaps={'uniform_orthogonal':lambda t:-c(t),
          'uniform_quadrature':lambda t:(s(t)-c(t))/2,
          'uniform_optimized':lambda t:(np.hypot(c(t),s(t))-c(t))/2,
          'Lambert_linear':lambda t:f(t)-.5*f(2*t)-.5,
          'survival_nonlinear':lambda t:((1+f(t))/2)**2-(1+f(2*t))/2}
    rows=[]
    for name,gap in gaps.items():
        for objective in ('physical_records','expected_evolution_time'):
            def score(t):
                signal=gap(t)
                if signal<=0: return 1e100
                if name.startswith('uniform'):
                    coefficient=4 if objective=='physical_records' else 2*t
                elif name=='Lambert_linear':
                    coefficient=9 if objective=='physical_records' else (2+2**.5)**2*t
                else:
                    coefficient=(1+2**.5)**2 if objective=='physical_records' else 8*t
                return coefficient/signal**2
            grid=np.linspace(.001,5,10000)
            k=int(np.argmin([score(t) for t in grid]))
            optimum=minimize_scalar(score,bounds=(grid[max(0,k-2)],grid[min(len(grid)-1,k+2)]),method='bounded')
            rows.append({'method':name,'objective':objective,
                         'time_parameter':float(optimum.x),'gap':float(gap(optimum.x)),
                         'hoeffding_proxy':float(optimum.fun)})
    return {'assumptions':'Ideal known readout; gamma=.1, omega=2; independent records; no preparation overhead or calibration errors',
            'formula':'proxy=(sum_k |a_k| sqrt(cost_k))^2 / gap^2; common Hoeffding confidence factor omitted',
            'interpretation':'No universal dominance; conventional nonlinear survival beats orthogonal uniform preparation. Coherent preparation improves the modeled uniform assay but is an additional resource.',
            'optimization':'Numerical grid and bounded minimization on time parameter [.001,5]; not certified global optimizer',
            'rows':rows}


def error_envelope(preparation_i: F, preparation_j: F,
                   common_clock_tv: F, detector_i: F, detector_j: F):
    """Sufficient contrast error, under separate fixed calibration bounds.

    All inputs are total-variation probability bounds. Positive shared affine
    detector gain is contracted conservatively by at most one. Overlapping
    physical errors must not be counted twice. The returned bound can exceed
    one; this is a conservative envelope, not an attainable-distance claim.
    """
    values=(preparation_i,preparation_j,common_clock_tv,detector_i,detector_j)
    for x in values:
        _rational(x)
        if not 0 <= x <= 1:
            raise ValueError('TV probability bounds must lie in [0,1]')
    return preparation_i+preparation_j+2*common_clock_tv+detector_i+detector_j


def make_results():
    import numpy as np
    import scipy
    from scipy.linalg import expm
    from scipy.optimize import brentq, minimize_scalar
    import mpmath as mp
    seed=73109
    rng=np.random.default_rng(seed)
    checks=[]
    for n in (2,3,4,7,12):
        for repetition in range(12):
            rates=rng.uniform(0,2,(n,n)); np.fill_diagonal(rates,0)
            rates *= rng.uniform(0,1,(n,n))>.35
            q=rates-np.diag(rates.sum(axis=1))
            horizon=float(rng.uniform(.04,1.2))
            i,j=rng.choice(n,2,replace=False)
            block=np.zeros((2*n,2*n));block[:n,:n]=q;block[:n,n:]=np.eye(n)
            integral=expm(horizon*block)[:n,n:]
            contrast=(integral[j,j]-integral[i,j])/horizon
            cap=max(-q[i,i],-q[j,j])
            lower=np.exp(-cap*horizon)
            assert contrast+1e-12>=lower
            checks.append({'n':n,'contrast':float(contrast),'lower':float(lower)})
    # A fixed-time detector can reverse sign in a wholly classical cycle.
    q=np.array([[-1.,1,0],[0,-1.,1],[1,0,-1.]])
    p=expm(2*q)
    fixed=float(p[1,1]-p[0,1])
    assert fixed<0
    # Coarse readout destroys the singleton theorem, even for exact micro-preps.
    qc=np.zeros((4,4));qc[0,3]=qc[1,2]=2
    qc-=np.diag(qc.sum(axis=1))
    block=np.zeros((8,8));block[:4,:4]=qc;block[:4,4:]=np.eye(4)
    integral=expm(block)[:4,4:]
    detector=np.array([1,0,1,0])
    coarse=float((integral[0]-integral[1])@detector)
    assert abs(coarse+np.exp(-2))<1e-12 and coarse<0
    mp.mp.dps=70
    critical=mp.exp(mp.lambertw(3*mp.pi/2))
    gamma=.1;omega=2.
    def contrast(t):
        return (gamma+np.exp(-gamma*t)*(-gamma*np.cos(omega*t)+omega*np.sin(omega*t)))/((gamma**2+omega**2)*t)
    optimum=minimize_scalar(contrast,bounds=(np.pi/omega,1.5*np.pi/omega),method='bounded',options={'xatol':1e-13})
    lower,upper=depolarizing_contrast_bounds(F(1,10),F(2),F(9,4),F(9,10))
    assert upper < -F(18,125)
    return {'status':'mathematical and simulated; no empirical physical data',
      'seed':seed,'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
      'classical_checks':len(checks),'classical_min_margin':min(c['contrast']-c['lower'] for c in checks),
      'critical_omega_over_gamma':str(critical),
      'critical_definition':'exp(W(3*pi/2)); reversal iff omega/gamma > critical',
      'quantum_example':{'gamma':'1/10','omega':'2','horizon':'9/4','detector_gain':'9/10',
        'mean_evolution_time':'9/8','contrast_bounds':[str(lower),str(upper)],
        'contrast_bounds_float':[float(lower),float(upper)],
        'certified_reversed_gap_lower':'18/125','certificate_degree':80},
      'coherent_preparation_example':{
        'gamma':'1/10','omega':'2','horizon':'7/4','detector_gain':'9/10',
        'mean_evolution_time':'7/8','preparation':'Y eigenstate with positive adjoint-effect quadrature',
        'gap_bounds':[str(x) for x in quadrature_preparation_gap_bounds(F(1,10),F(2),F(7,4),F(9,10))],
        'gap_bounds_float':[float(x) for x in quadrature_preparation_gap_bounds(F(1,10),F(2),F(7,4),F(9,10))]},
      'baseline_subtracted_example':{
        'interpretation':'Conditional dimension witness; invokes the separately proved <=3-state bound',
        'gamma':'1/10','omega':'2','horizon':'2','detector_gain':'9/10',
        'bloch_positive_arm':['0','-12/37','35/37'],
        'bloch_negative_arm':['0','12/37','-35/37'],
        'witness_bounds':[str(x) for x in baseline_subtracted_witness_bounds(F(1,10),F(2),F(2),F(-12,37),F(35,37),F(9,10))],
        'witness_bounds_float':[float(x) for x in baseline_subtracted_witness_bounds(F(1,10),F(2),F(2),F(-12,37),F(35,37),F(9,10))],
        'certified_witness_lower':'27/25',
        'mean_evolution_random_time_arm':'1','baseline_arm_evolution':'0',
        'null_detector_assumption':'Arbitrary fixed stochastic detector; the modeled gain is used only for alternative power'},
      'prior_comparison':comparison_benchmark(),
      'clock_optimization':{'assumptions':'All nonincreasing probability densities; no extra clock resource constraint',
        'uniform_horizon_numerical':float(optimum.x),'contrast_numerical':float(optimum.fun),
        'theorem':'Every decreasing-density clock is a mixture of uniform clocks; a best uniform clock attains the unconstrained optimum'},
      'negative_controls':{'classical_fixed_time_contrast_at_2':fixed,
        'classical_coarse_uniform_contrast_at_1':coarse,
        'quantum_exponential_clock_contrast':str(exponential_clock_contrast(F(1,10),F(2),F(1)))},
      'limitations':['Requires singleton detector or certified common entry-reset law',
        'Clock independent of trajectory and identical between preparations',
        'Not a device-independent quantum or coherence witness',
        'No claim of historical priority or actual laboratory validation']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=make_results()
    output=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    else: print(output,end='')
