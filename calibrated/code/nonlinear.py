"""Certified nonlinear record-KL separation from calibrated reversible moments.

Numerical optimization only proposes scores and feasible relaxation points.
Rational logarithm bounds, an exact linear-program solver, and continuous-domain
Bernstein bounds certify the result. An upper bracket bounds the record-KL
RELAXATION, never the physical entropy production of an unknown system.
"""
from fractions import Fraction as F
from pathlib import Path
from math import ceil
import json
from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper


def _inputs(lags, weights, vertices, coefficients, lower, boxes=None):
    m = len(lags)
    w = list(map(ec.exact, weights))
    a = list(map(ec.exact, coefficients))
    ell = ec.exact(lower)
    vertices = [tuple(map(ec.exact, z)) for z in vertices]
    boxes = [(F(0), F(1))] * m if boxes is None else [tuple(map(ec.exact, z)) for z in boxes]
    ec.validate(lags, 1, w, boxes, (-1, 1), [(1, 1)] * m)
    if len(a) != m or not vertices or any(len(z) != 2 or not 0 <= z[0] <= z[1] <= 1 for z in vertices):
        raise ValueError('matching contrast and valid nonempty (mean-square, second-moment) vertices required')
    if max_linear(a, boxes) < ell:
        raise ValueError('empty confidence region')
    return w, vertices, a, ell, boxes


def max_linear(coefficients, boxes):
    return sum((c * (hi if c >= 0 else lo) for c, (lo, hi) in zip(coefficients, boxes)), F(0))


def linear_minimum(objective, coefficients, lower, boxes):
    """Exact fractional-knapsack LP on a box intersected with one halfspace."""
    b = list(map(ec.exact, objective)); a = list(map(ec.exact, coefficients)); ell = ec.exact(lower)
    boxes = [tuple(map(ec.exact, z)) for z in boxes]
    if not len(a) == len(b) == len(boxes) or any(not 0 <= lo <= hi <= 1 for lo, hi in boxes):
        raise ValueError('invalid linear-program input')
    if max_linear(a, boxes) < ell:
        raise ValueError('empty confidence region')
    p = [lo if c >= 0 else hi for c, (lo, hi) in zip(b, boxes)]
    deficit = ell - sum((c * x for c, x in zip(a, p)), F(0))
    moves = []
    for j, (c, d, (lo, hi)) in enumerate(zip(a, b, boxes)):
        if not c:
            continue
        target = hi if c > 0 else lo
        gain = c * (target - p[j])
        if gain > 0:
            price = d * (target - p[j]) / gain
            moves.append((price, j, gain, target))
    for _, j, gain, target in sorted(moves):
        if deficit <= 0:
            break
        taken = min(deficit, gain)
        p[j] += taken / a[j]
        deficit -= taken
    if deficit > 0:
        raise ArithmeticError('feasible linear program was not completed')
    return dict(value=sum((c * x for c, x in zip(b, p)), F(0)), minimizer=p)


def calibrated_normalizer(lags, weights, vertices, scores, depth=7):
    """Continuous-domain upper bound over every calibration vertex and x in [0,1]."""
    polynomials = []; bounds = []
    for u, second in vertices:
        power = [F(0)] * (max(lags) + 1)
        for k, w, (h0, h1) in zip(lags, weights, scores):
            power[0] += w * (h0 + (h1 - h0) * (1 + u) / 2)
            power[k] += w * (h1 - h0) * (second - u) / 2
        polynomials.append(power)
        bounds.append(ec.polynomial_upper(power, depth))
    return dict(normalizer_upper=max(bounds), polynomials=polynomials, polynomial_bounds=bounds)


def certify(lags, weights, vertices, coefficients, lower, durations, scores,
            boxes=None, depth=7):
    """Certified lower bound on inf_{p in P,q in K} sum_k w_k KL_Ber(p_k||q_k).

    P is boxes intersected with coefficients dot p >= lower. K is the convex
    hull of calibrated reversible moment curves at the supplied (m^2,A)
    polygon vertices. Scores may be selected using the same data after P has
    been constructed with valid simultaneous coverage. The contrast defining P
    itself must be fixed before its records, or independently selected.
    """
    w, vertices, a, ell, boxes = _inputs(lags, weights, vertices, coefficients, lower, boxes)
    scores = [tuple(map(ec.exact, z)) for z in scores]
    times = list(map(ec.exact, durations))
    if len(scores) != len(lags) or any(len(z) != 2 or min(z) <= 0 for z in scores):
        raise ValueError('matching strictly positive rational scores required')
    if len(times) != len(lags) or min(times) <= 0:
        raise ValueError('positive full expected path durations required')
    norm = calibrated_normalizer(lags, w, vertices, scores, depth)
    log0 = [ec.log_interval(z[0])[0] for z in scores]
    log1 = [ec.log_interval(z[1])[0] for z in scores]
    lp = linear_minimum([v * (b - c) for v, b, c in zip(w, log1, log0)], a, ell, boxes)
    expectation = sum((v * c for v, c in zip(w, log0)), F(0)) + lp['value']
    kl = max(F(0), expectation - ec.log_interval(norm['normalizer_upper'])[1])
    duration = sum((v * t for v, t in zip(w, times)), F(0))
    return dict(status='positive_certificate' if kl else 'no_positive_certificate',
                lags=list(lags), weights=w, vertices=vertices, coefficients=a,
                lower=ell, boxes=boxes, durations=times, scores=scores, depth=depth,
                normalizer=norm, log_expectation_lower=expectation,
                expectation_minimizer=lp['minimizer'], record_kl_lower=kl,
                expected_duration=duration, entropy_production_rate_lower=4 * kl / duration,
                contract='conditional record-KL relaxation; stationary ordinary-reversal CTMC and calibrated full duration needed for entropy rate; no physical optimality claim')


def verify(candidate, **contract):
    try:
        return candidate == certify(**contract)
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False


def kl_interval(p, q):
    """Exact rational enclosure of Bernoulli KL, including finite endpoints."""
    p, q = ec.exact(p), ec.exact(q)
    if not 0 <= p <= 1 or not 0 <= q <= 1:
        raise ValueError('probabilities required')
    if q in (0, 1):
        if p == q:
            return F(0), F(0)
        raise ValueError('infinite relative entropy')
    lo = hi = F(0)
    for mass, ref in ((p, q), (1-p, 1-q)):
        if mass:
            left, right = ec.log_interval(mass/ref)
            lo += mass * left; hi += mass * right
    return max(F(0), lo), max(F(0), hi)


def primal_bound(lags, weights, vertices, coefficients, lower, probabilities,
                 atoms, boxes=None):
    """Upper bound on the relaxation from feasible rational p and mixture q.

    atoms are (vertex_index, decay_x, mass), sum mass=1. x=0 belongs to the
    closed comparison class and need not have a finite-rate CTMC realization.
    Even an interior finite realization is NOT an upper bound on unknown EPR.
    """
    w, vertices, a, ell, boxes = _inputs(lags, weights, vertices, coefficients, lower, boxes)
    p = list(map(ec.exact, probabilities))
    if len(p) != len(lags) or any(not lo <= x <= hi for x, (lo, hi) in zip(p, boxes)):
        raise ValueError('primal probabilities outside confidence region')
    if sum((c*x for c,x in zip(a,p)), F(0)) < ell:
        raise ValueError('primal probabilities violate halfspace')
    checked = []
    for vertex, x, mass in atoms:
        x, mass = ec.exact(x), ec.exact(mass)
        if type(vertex) is not int or not 0 <= vertex < len(vertices) or not 0 <= x <= 1 or mass < 0:
            raise ValueError('invalid comparison mixture atom')
        checked.append((vertex, x, mass))
    if not checked or sum((mass for _,_,mass in checked), F(0)) != 1:
        raise ValueError('mixture masses must sum exactly to one')
    q = [sum((mass * (1+vertices[v][0]+(vertices[v][1]-vertices[v][0])*x**k)/2
                  for v,x,mass in checked), F(0)) for k in lags]
    intervals = [kl_interval(x,y) for x,y in zip(p,q)]
    upper = sum((v*z[1] for v,z in zip(w,intervals)), F(0))
    return dict(probabilities=p, comparison_probabilities=q, atoms=checked,
                bernoulli_kl_intervals=intervals, record_kl_relaxation_upper=upper,
                interpretation='upper bound on convex record-KL relaxation only; not an upper bound on physical entropy production')


def confidence_region(coefficients, records, successes, alpha=F(1,20),
                      conditional_radii=None, selection='predeclared'):
    """One-sided Hoeffding/Azuma halfspace for fixed quotas and fixed contrast."""
    if selection not in ('predeclared','independent_pilot'):
        raise ValueError('contrast must be predeclared or independently selected')
    a = list(map(ec.exact, coefficients)); alpha = ec.exact(alpha); m=len(a)
    delta = [F(0)]*m if conditional_radii is None else list(map(ec.exact,conditional_radii))
    if not m or not len(records)==len(successes)==len(delta)==m or not 0 < alpha < 1:
        raise ValueError('invalid confidence inputs')
    if any(type(n) is not int or n<=0 for n in records) or any(type(s) is not int or not 0<=s<=n for s,n in zip(successes,records)):
        raise ValueError('valid fixed integer quotas and counts required')
    if any(not 0<=d<=1 for d in delta):
        raise ValueError('conditional radii must lie in [0,1]')
    empirical=sum((c*F(s,n) for c,s,n in zip(a,successes,records)),F(0))
    variance=sum((c*c/n for c,n in zip(a,records)),F(0))
    radius=sqrt_upper(ec.log_interval(1/alpha)[1]*variance/2)
    bias=sum((abs(c)*d for c,d in zip(a,delta)),F(0))
    return dict(coefficients=a,lower=empirical-radius-bias,records=list(records),
                successes=list(successes),alpha=alpha,conditional_radii=delta,
                empirical_contrast=empirical,sampling_radius_upper=radius,
                conditional_bias_penalty=bias,selection=selection,
                interpretation='coverage conditional on declared record means; calibration-region failure must be added separately')


def design(lags, weights, vertices, coefficients, lower, durations,
           boxes=None, grid=41, depth=7, maxiter=600):
    """Convex numerical proposal, followed by independently checkable brackets.

    Scores and feasible mixture probabilities are rounded to rational values.
    Neither grid density nor solver success establishes certificate validity.
    The continuous dual and rational primal are authoritative.
    """
    import numpy as np
    from scipy.optimize import minimize
    w, vertices, a, ell, boxes = _inputs(lags,weights,vertices,coefficients,lower,boxes)
    if type(grid) is not int or not 3<=grid<=257:
        raise ValueError('grid must lie in 3..257')
    m=len(lags); wf=np.array(list(map(float,w))); af=np.array(list(map(float,a)))
    points=[F(j,grid-1) for j in range(grid)]
    labels=[(v,x) for v in range(len(vertices)) for x in points]
    matrix=np.array([[float((1+vertices[v][0]+(vertices[v][1]-vertices[v][0])*x**k)/2)
                      for v,x in labels] for k in lags])
    nf=len(labels); epsilon=1e-12
    pstart=np.array([float((lo+hi)/2) for lo,hi in boxes])
    if af@pstart < float(ell):
        pstart=np.array(list(map(float,linear_minimum([0]*m,a,ell,boxes)['minimizer'])))
    start=np.r_[pstart,np.full(nf,1/nf)]
    def objective(z):
        p=np.clip(z[:m],epsilon,1-epsilon);q=np.clip(matrix@z[m:],epsilon,1-epsilon)
        return float(np.sum(wf*(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)))))
    def gradient(z):
        p=np.clip(z[:m],epsilon,1-epsilon);q=np.clip(matrix@z[m:],epsilon,1-epsilon)
        return np.r_[wf*np.log(p*(1-q)/(q*(1-p))), matrix.T@(wf*(q-p)/(q*(1-q)))]
    constraints=[{'type':'eq','fun':lambda z:np.sum(z[m:])-1,
                  'jac':lambda z:np.r_[np.zeros(m),np.ones(nf)]},
                 {'type':'ineq','fun':lambda z:af@z[:m]-float(ell),
                  'jac':lambda z:np.r_[af,np.zeros(nf)]}]
    fit=minimize(objective,start,jac=gradient,method='SLSQP',bounds=[tuple(map(float,z)) for z in boxes]+[(0,1)]*nf,
                 constraints=constraints,options={'ftol':2e-13,'maxiter':maxiter})
    # A common integer grid prevents unrelated rational denominators from
    # multiplying into an enormous primal logarithm argument. Feasibility and
    # the final dual/primal checks, not this proposal rounding, give validity.
    mass=[max(0,round(float(x)*10**12)) for x in fit.x[m:]]
    total=sum(mass)
    if not total:
        mass=[F(1,nf)]*nf
    else:
        mass=[F(x,total) for x in mass]
    atoms=[(v,x,z) for (v,x),z in zip(labels,mass) if z]
    q=[sum((z*(1+vertices[v][0]+(vertices[v][1]-vertices[v][0])*x**k)/2 for v,x,z in atoms),F(0)) for k in lags]
    p=[min(hi,max(lo,F(float(x)).limit_denominator(10**9))) for x,(lo,hi) in zip(fit.x[:m],boxes)]
    deficit=ell-sum((c*x for c,x in zip(a,p)),F(0))
    if deficit>0:
        for j,(c,(lo,hi)) in enumerate(zip(a,boxes)):
            if not c:continue
            target=hi if c>0 else lo;gain=c*(target-p[j]);taken=min(deficit,gain)
            p[j]+=taken/c;deficit-=taken
            if deficit<=0:break
    # Strict positivity is necessary for the variational certificate. Boundary
    # clipping affects proposal quality only; all final inputs are rechecked.
    clipped_p=[min(F(10**10-1,10**10),max(F(1,10**10),x)) for x in p]
    clipped_q=[min(F(10**10-1,10**10),max(F(1,10**10),x)) for x in q]
    scores=[(F(float((1-x)/(1-y))).limit_denominator(10**9),F(float(x/y)).limit_denominator(10**9))
            for x,y in zip(clipped_p,clipped_q)]
    scores=[(max(F(1,10**12),x),max(F(1,10**12),y)) for x,y in scores]
    cert=certify(lags,w,vertices,a,ell,durations,scores,boxes,depth)
    primal=primal_bound(lags,w,vertices,a,ell,p,atoms,boxes)
    gap=primal['record_kl_relaxation_upper']-cert['record_kl_lower']
    if gap<0:
        raise ArithmeticError('inconsistent certified primal and dual bounds')
    return dict(certificate=cert,primal=primal,certified_relaxation_gap_upper=gap,
                proposal_success=bool(fit.success),proposal_status=str(fit.message),
                proposal_iterations=int(fit.nit),proposal_grid=grid,
                interpretation='exact bracket on a convex observational relaxation; no physical optimality or empirical validation claim')


def jsonable(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def activity_bridge(record_kl_lower, expected_duration, activity_cap, bits=75):
    """Nonlinear path-KL to entropy bridge with separately calibrated activity.

    The cap bounds total stationary microscopic jump activity, not observed
    detector changes. Inputs and returned bound are exact rationals. Positive
    logarithm arguments are enclosed by the inherited atanh-series backend.
    This extra access can be scientifically expensive and is never inferred by
    this function from endpoint records.
    """
    kl, duration, cap=map(ec.exact,(record_kl_lower,expected_duration,activity_cap))
    if kl<0 or duration<=0 or cap<=0 or type(bits) is not int or not 16<=bits<=90:
        raise ValueError('nonnegative KL, positive duration/activity and bits16..90 required')
    ratio=kl/(duration*cap); log2lo,log2hi=ec.log_interval(2)
    if ratio>log2hi:
        return dict(status='incompatible_activity_and_record_bounds',record_kl_lower=kl,
                    expected_duration=duration,activity_cap=cap,kl_per_activity=ratio,
                    interpretation='under the joint model/calibration event record KL cannot exceed activity times duration times log2; no physical estimate returned')
    if ratio>=log2lo:
        return dict(status='unresolved_log2_boundary',record_kl_lower=kl,
                    expected_duration=duration,activity_cap=cap,kl_per_activity=ratio)
    lo=F(0);hi=F(1)
    def j_interval(x):
        left=ec.log_interval(1+x);right=ec.log_interval(1-x)
        return tuple(((1+x)*left[k]+(1-x)*right[k])/2 for k in (0,1))
    for _ in range(bits):
        mid=(lo+hi)/2;jlo,jhi=j_interval(mid)
        if jhi<=ratio:lo=mid
        elif jlo>=ratio:hi=mid
        else:break
    e_lower=lo*ec.log_interval((1+lo)/(1-lo))[0]
    linear=4*kl/duration
    return dict(status='certified_activity_bridge',record_kl_lower=kl,
                expected_duration=duration,activity_cap=cap,kl_per_activity=ratio,
                asymmetry_lower=lo,asymmetry_upper=hi,
                entropy_production_rate_lower=max(linear,cap*e_lower),
                linear_bridge_lower=linear,bits=bits,
                contract='stationary ordinary-reversal CTMC; separately calibrated upper bound on total microscopic jump activity; no inference of activity from detector changes')


def diffusion_example():
    """Certified scalar check for a stationary rotational Ornstein--Uhlenbeck model.

    dX=(-I+omega*J)X dt+sqrt(2)dW, detector conditional mean sin(X_1).
    Gaussian calculation gives C(t)=exp(-1)*sinh(exp(-t)*cos(omega*t)),
    mean0, A=(1-exp(-2))/2, entropy rate2*omega**2. No sampled data.
    The finite-jump activity envelope does not apply to this diffusion.
    """
    from constructive_access.robust.preparation import exp_negative_bounds, trig_bounds
    def outward(interval,scale=10**12):
        lo,hi=interval
        return F(lo.numerator*scale//lo.denominator,scale),F(ceil(hi*scale),scale)
    def agreement(omega):
        e=exp_negative_bounds(F(1));c=trig_bounds(F(omega),'cos')
        products=[a*b for a in e for b in c];rho=outward((min(products),max(products)))
        # C(rho)=[exp(-(1-rho))-exp(-(1+rho))]/2 is increasing in rho.
        left_a=exp_negative_bounds(1-rho[0]);left_b=exp_negative_bounds(1+rho[0])
        right_a=exp_negative_bounds(1-rho[1]);right_b=exp_negative_bounds(1+rho[1])
        corr=outward(((left_a[0]-left_b[1])/2,(right_a[1]-right_b[0])/2))
        return dict(underlying_gaussian_correlation_interval=rho,
                    detector_correlation_interval=corr,
                    agreement_interval=outward(((1+corr[0])/2,(1+corr[1])/2)))
    driven=agreement(2);equilibrium=agreement(0)
    e2=exp_negative_bounds(F(2));A=outward(((1-e2[1])/2,(1-e2[0])/2))
    p=sum(driven['agreement_interval'])/2;scores=[(2*(1-p),2*p)]
    common=dict(lags=[1],weights=[1],vertices=[(0,0),(0,1),(1,1)],
                coefficients=[0],lower=0,durations=[1],scores=scores)
    return dict(evidence='exact Gaussian model calculation with rational transcendental enclosures; no sampled or empirical records',
        parameters={'gamma':F(1),'diffusion_constant':F(1),'angular_rate':F(2),'duration':F(1)},
        detector='conditional binary mean sin(X_1); same noninvasive channel with independent readout noise',
        mean=F(0),second_moment_interval=A,
        driven=dict(driven,certificate=certify(boxes=[driven['agreement_interval']],**common),
                    exact_stationary_entropy_rate=F(8),exact_full_path_kl_to_reversible_reference=F(2)),
        equilibrium=dict(equilibrium,certificate=certify(boxes=[equilibrium['agreement_interval']],**common),
                         exact_stationary_entropy_rate=F(0)),
        scope='diffusion transfer uses path likelihood and spectral theorem; jump-activity inversion is inapplicable')


def example():
    common=dict(lags=[1,2],weights=[F(1,2)]*2,coefficients=[F(38,25),-1],
                lower=F(1147,2500),durations=[1,2])
    unrestricted=design(vertices=[(0,0),(0,1),(1,1)],**common)
    calibrated=design(vertices=[(0,F(1,2))],**common)
    exact=design(lags=[1],weights=[1],vertices=[(0,0)],coefficients=[1],lower=F(4,5),durations=[1])
    return dict(evidence='deterministic mathematical examples; not measured data',
                uncalibrated=unrestricted,calibrated=calibrated,analytic_bernoulli=exact,
                activity=activity_bridge(F(1,2),1,1),diffusion=diffusion_example())


if __name__=='__main__':
    result=example();out=Path(__file__).resolve().parents[1]/'results/nonlinear.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(jsonable(result),sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:{'lower':float(v['certificate']['record_kl_lower']),
                         'upper':float(v['primal']['record_kl_relaxation_upper'])}
                      for k,v in result.items() if isinstance(v,dict) and 'certificate' in v}))
