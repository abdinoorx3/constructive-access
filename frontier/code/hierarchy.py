"""Finite-record positive-realization dimension certificates.

Exact antichain certificates and binomial tail bounds use Fraction/integer
arithmetic. SciPy, when used, proposes sample designs; exact arithmetic checks
all reported error guarantees. No experimental data are included.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import comb, factorial
from pathlib import Path
import argparse
import json


def frac(x):
    if isinstance(x, float):
        raise TypeError('Use Fraction or integer, not an inexact float.')
    return F(x)


def rational_interval(lo, hi, digits=12):
    d = 10 ** digits
    return (F((lo*d).numerator//(lo*d).denominator, d),
            F(-((-hi*d).numerator//(-hi*d).denominator), d))


def pi_interval(terms=30):
    """Machin identity with alternating-series rational bounds."""
    def atan(x):
        s = sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)), F(0))
        tail = (-1)**terms*x**(2*terms+1)/F(2*terms+1)
        return min(s, s+tail), max(s, s+tail)
    a, b = atan(F(1,5)), atan(F(1,239))
    return 16*a[0]-4*b[1], 16*a[1]-4*b[0]


def _exp_negative_interval(xlo, xhi, terms=32):
    def point(x):
        s = sum(((-x)**k/F(factorial(k)) for k in range(terms+1)), F(0))
        rem = x**(terms+1)/F(factorial(terms+1))
        # The derivative of exp(-x) is at most one in absolute value for x>=0.
        return max(F(0),s-rem), min(F(1),s+rem)
    return point(xhi)[0], point(xlo)[1]


def _cos_interval(xlo, xhi, terms=24):
    # Inputs lie in [0, pi]. Taylor's remainder is bounded by |x|^(2m+2)/(2m+2)!.
    def point(x):
        s = sum(((-1)**k*x**(2*k)/F(factorial(2*k)) for k in range(terms+1)),F(0))
        rem = x**(2*terms+2)/F(factorial(2*terms+2))
        return max(F(-1),s-rem), min(F(1),s+rem)
    return point(xhi)[0],point(xlo)[1]


def qubit_probabilities(grid_size, coherence_ratio, digits=12):
    """Exact outward intervals for p_k=(1-e^(-2pi*k/(r*N))*cos(2pi*k/N))/2.

    One qubit, one fixed preparation, one fixed binary detector, 2N-1 wait
    times. r=omega/gamma. The dephasing here is isotropic depolarization.
    """
    N, r = int(grid_size), frac(coherence_ratio)
    if N < 2 or r < 7:
        raise ValueError('Require N>=2 and coherence ratio>=7 for this implementation.')
    plo, phi = rational_interval(*pi_interval(), digits=20)
    values = []
    for k in range(1,2*N):
        a,b = 2*plo*k/(r*N),2*phi*k/(r*N)
        elo,ehi = _exp_negative_interval(a,b)
        phase = min(k%N, N-(k%N))
        if phase == 0:
            clo,chi = F(1),F(1)
        elif 2*phase == N:
            clo,chi = F(-1),F(-1)
        else:
            clo,chi = _cos_interval(2*plo*phase/N,2*phi*phase/N)
        products = [elo*clo,elo*chi,ehi*clo,ehi*chi]
        low,high = (1-max(products))/2,(1-min(products))/2
        values.append(rational_interval(low,high,digits))
    return values


def minimum_states(grid_size):
    N = int(grid_size)
    if N < 1:
        raise ValueError('Positive grid size required.')
    n = 1
    while comb(n,n//2) < N:
        n += 1
    return n


def certify_matrix_bounds(diagonal_upper, offdiagonal_lower, grid_size, threshold=None):
    eps,beta = frac(diagonal_upper),frac(offdiagonal_lower)
    if not 0 <= eps <= 1 or not 0 <= beta <= 1 or grid_size < 2:
        raise ValueError('Probability bounds and grid size are invalid.')
    if eps == 0:
        good = beta > 0
        return {'certified':good,'minimum_states':minimum_states(grid_size) if good else None,
                'grid_size':grid_size,'threshold':None,'classical_comparison_bound':F(0),
                'margin':beta}
    if threshold is None:
        # A rational proposal avoids floating-point comparisons in certificates.
        from math import isqrt
        d = 10**9
        threshold = F(max(1,isqrt(eps.numerator*d*d//eps.denominator)),d)
    tau = frac(threshold)
    if not 0 < tau < 1:
        raise ValueError('Threshold must lie strictly between zero and one.')
    bound = tau+(1-tau)*eps/tau
    good = beta > bound
    return {'certified':good,'minimum_states':minimum_states(grid_size) if good else None,
            'grid_size':grid_size,'threshold':tau,'classical_comparison_bound':bound,
            'margin':beta-bound}


def certify_grid(probability_intervals, threshold=None):
    if len(probability_intervals)%2 != 1:
        raise ValueError('The grid must contain 2N-1 probabilities.')
    N = (len(probability_intervals)+1)//2
    for lo,hi in probability_intervals:
        if not 0<=lo<=hi<=1:
            raise ValueError('Invalid interval.')
    eps = probability_intervals[N-1][1]
    beta = min(lo for k,(lo,_) in enumerate(probability_intervals) if k != N-1)
    out = certify_matrix_bounds(eps,beta,N,threshold)
    out.update(diagonal_upper=eps,offdiagonal_lower=beta)
    return out


def binomial_tail_upper(n,k,p,tail):
    """Exact rational geometric bound on Pr[S<=k] or Pr[S>=k].

    This is an upper bound, not an approximate binomial CDF. An inconclusive
    bound is one. The ratio estimate follows from monotonic adjacent masses.
    """
    p = frac(p)
    if n < 0 or not 0<=p<=1 or tail not in ('lower','upper'):
        raise ValueError('Invalid binomial parameters.')
    if tail == 'lower':
        if k < 0: return F(0)
        if k >= n: return F(1)
    else:
        if k <= 0: return F(1)
        if k > n: return F(0)
    if p == 0: return F(int(tail=='lower'))
    if p == 1: return F(int(tail=='upper'))
    a,b = p.numerator,p.denominator
    mass = F(comb(n,k)*a**k*(b-a)**(n-k),b**n)
    if tail=='lower':
        ratio = F(k*(b-a),(n-k+1)*a)
    else:
        ratio = F((n-k)*a,(k+1)*(b-a))
    if ratio >= 1:
        return F(1)
    return min(F(1),mass/(1-ratio))


def _candidate_test(n,pnull,palt,direction,alpha,beta):
    """SciPy proposes a critical count; exact integer arithmetic verifies it."""
    from scipy.stats import binom
    if direction=='low':
        k = int(binom.ppf(float(alpha),n,float(pnull)))
        # ppf can return the first count above alpha. Check both nearby counts.
        for candidate in range(k+1,max(-1,k-8),-1):
            a = binomial_tail_upper(n,candidate,pnull,'lower')
            if a <= alpha:
                b = binomial_tail_upper(n,candidate+1,palt,'upper')
                return {'n':n,'critical':candidate,'direction':direction,
                        'false_rejection_bound':a,'miss_bound':b,
                        'valid':b<=beta}
    else:
        k = int(binom.isf(float(alpha),n,float(pnull)))+1
        for candidate in range(max(0,k-1),k+9):
            a = binomial_tail_upper(n,candidate,pnull,'upper')
            if a <= alpha:
                b = binomial_tail_upper(n,candidate-1,palt,'lower')
                return {'n':n,'critical':candidate,'direction':direction,
                        'false_rejection_bound':a,'miss_bound':b,
                        'valid':b<=beta}
    return {'n':n,'valid':False}


def plan_one(pnull,palt,direction,alpha=F(1,20),beta=F(1,260),block=100,max_records=200000):
    """Find a checked sufficient fixed-record design (not a minimum proof)."""
    pnull,palt,alpha,beta = map(frac,(pnull,palt,alpha,beta))
    if (direction=='low' and not palt < pnull) or (direction=='high' and not palt>pnull):
        raise ValueError('The alternative must be separated from the null.')
    n=block
    while n<=max_records:
        result = _candidate_test(n,pnull,palt,direction,alpha,beta)
        if result['valid']:
            result.update(null_boundary=pnull,alternative_bound=palt,alpha=alpha,beta=beta)
            return result
        n+=block
    raise RuntimeError('No checked design within the stated search range.')


def plan_grid(grid_size=7,coherence_ratio=F(1000),epsilon_null=F(3,500),
              beta_null=F(3,20),threshold=F(2,25),alpha=F(1,20),miss=F(1,20),
              probability_bias=F(0),block=100):
    """Intersection-union test for an entire forbidden classical probability region.

    Type-I error is max of individual error bounds, not their sum: every
    classical null violates at least one of the declared constraints. Power
    uses the sum of miss bounds. Settings may be dependent across trials, but
    the reported binomial calculation requires IID records within each setting.
    Bias is an absolute bound relating each recorded probability to the common
    preparation/transition/readout model and is paid on both sides.
    """
    eps,bet,tau,alpha,miss,bias=map(frac,(epsilon_null,beta_null,threshold,alpha,miss,probability_bias))
    witness = certify_matrix_bounds(eps,bet,grid_size,tau)
    if not witness['certified']:
        raise ValueError('The proposed probability region does not exclude the null.')
    ps=qubit_probabilities(grid_size,coherence_ratio)
    allocated=miss/F(len(ps))
    plans=[]
    for index,(lo,hi) in enumerate(ps):
        diagonal=index==grid_size-1
        pnull=eps-bias if diagonal else bet+bias
        palt=hi+bias if diagonal else lo-bias
        if not 0<=pnull<=1 or not 0<=palt<=1:
            raise ValueError('Bias pushes probability bounds outside [0,1].')
        plan=plan_one(pnull,palt,'low' if diagonal else 'high',alpha,allocated,block)
        plan.update(step=index+1,probability_interval=(lo,hi))
        plans.append(plan)
    type_i=max(p['false_rejection_bound'] for p in plans)
    type_ii=sum((p['miss_bound'] for p in plans),F(0))
    return {'grid_size':grid_size,'coherence_ratio':frac(coherence_ratio),
            'probability_bias':bias,'witness':witness,'settings':plans,
            'total_records':sum(p['n'] for p in plans),
            'total_steps':sum(p['n']*p['step'] for p in plans),
            'false_rejection_bound':type_i,'miss_bound':type_ii,
            'independent_trials_per_setting':True,'laboratory_data':False,
            'design_is_globally_minimal':False}


def _json_value(x):
    if isinstance(x,F):
        # Tail rationals may have millions of digits. Only outward rational
        # decimal bounds are exported; exact values are independently rebuilt.
        lo,hi=rational_interval(x,x,12)
        return {'lower':str(lo),'upper':str(hi),'approx':float(x)}
    if isinstance(x,dict): return {k:_json_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json_value(v) for v in x]
    return x


def make_results():
    dims=[]
    for N,r in [(7,1000),(11,10000),(21,100000)]:
        ps=qubit_probabilities(N,F(r))
        dims.append({'grid_size':N,'settings':2*N-1,'coherence_ratio':r,
                     'certificate':certify_grid(ps)})
    plan=plan_grid()
    return {'status':'mathematical/synthetic; no laboratory data',
            'foundations':'Sperner antichain bound and positive-realization factorization are established.',
            'hierarchy':dims,'five_state_design':plan,
            'assumptions':['One common prepared distribution and one fixed stochastic binary detector',
                           'One common homogeneous Markov transition applied repeatedly',
                           'Every hidden controller or clock state is included in the state count',
                           'The statistical design uses independent reset records within each setting'],
            'limitations':['Not a universal classical exclusion','No thermodynamic energy bound',
                           'No assertion of historical priority','No globally optimal sample-count claim']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'results'/'hierarchy.json'))
    args=parser.parse_args()
    result=make_results()
    Path(args.output).write_text(json.dumps(_json_value(result),indent=2)+'\n')
    plan=result['five_state_design']
    print(json.dumps({'minimum_states':plan['witness']['minimum_states'],
                      'total_records':plan['total_records'],
                      'type_i_bound':float(plan['false_rejection_bound']),
                      'type_ii_bound':float(plan['miss_bound'])}))
