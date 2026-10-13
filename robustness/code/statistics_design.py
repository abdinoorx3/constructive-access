"""Calibrated randomized contrast tests, with exact rational certificates.

Statistical machinery is established Chernoff / likelihood-ratio theory.
The physical null is supplied separately and is never inferred by this module.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import ceil
from pathlib import Path
import json


def rational(x):
    if isinstance(x, bool) or isinstance(x, float):
        raise TypeError("use an integer, Fraction, or exact rational string")
    if not isinstance(x, (int, F, str)):
        raise TypeError("unsupported exact scalar")
    return F(x)


def _outward(lo, hi, bits=70):
    scale = 1 << bits
    return F(lo.numerator * scale // lo.denominator, scale), F(ceil(hi * scale), scale)


@lru_cache(maxsize=512, typed=True)
def log_bounds(x):
    """Certified rational natural-log enclosure; no floating point."""
    x = rational(x)
    if x <= 0:
        raise ValueError("positive logarithm argument required")
    if x == 1:
        return F(0), F(0)
    if x < 1:
        lo, hi = log_bounds(1 / x)
        return -hi, -lo
    count = 0
    while x > 2:
        x /= 2
        count += 1
    z = (x - 1) / (x + 1)
    terms = 64
    value = 2 * sum((z ** (2*k+1) / (2*k+1) for k in range(terms)), F(0))
    remainder = 2*z**(2*terms+1) / ((2*terms+1)*(1-z*z))
    lo, hi = value, value + remainder
    if count:
        a, b = log_bounds(F(2))
        lo += count * a
        hi += count * b
    return _outward(lo, hi)


def kl_bounds(p, q):
    p, q = map(rational, (p, q))
    if not 0 < p < 1 or not 0 < q < 1:
        raise ValueError("interior Bernoulli probabilities required")
    a, b = log_bounds(p/q)
    c, d = log_bounds((1-p)/(1-q))
    return p*a + (1-p)*c, p*b + (1-p)*d


def contrast_contract(gap, radius=0, alpha=F(1,20), beta=F(1,20)):
    """gap is ideal p_i-p_j; radius bounds its setting-dependent error.

    Under H0 actual contrast <= radius. Under the stated alternative its
    contrast >= gap-radius. A shared positive affine detector gain belongs
    in gap. Calibration failures are accounted for outside this contract.
    """
    gap, radius, alpha, beta = map(rational, (gap, radius, alpha, beta))
    if not 0 < gap <= 1 or not 0 <= radius < 1:
        raise ValueError("gap in (0,1], radius in [0,1) required")
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("error probabilities must lie in (0,1)")
    theta0 = (1+radius)/2
    theta1 = (1+gap-radius)/2
    if theta1 >= 1:
        raise ValueError("nondegenerate alternative required")
    return dict(gap=gap, radius=radius, alpha=alpha, beta=beta,
                null_success_upper=theta0, alternative_success_lower=theta1)


def encode(arm, outcome):
    """The arm must have been selected by a fresh independent fair coin.

    arm 0 is preparation i; arm 1 is preparation j. This encoding does not
    validate the physical reset, clock, readout, or randomization contract.
    """
    if type(arm) is not int or type(outcome) is not int or arm not in (0,1) or outcome not in (0,1):
        raise ValueError("arm and outcome must be integer bits")
    return outcome if arm == 0 else 1-outcome


def fixed_plan(gap, radius=0, alpha=F(1,20), beta=F(1,20), grid=1024):
    """Chernoff-certified fixed-size design over a declared threshold grid.

    Selects a threshold near balanced certified error requirements. It does
    not claim the actual sample-size optimum. Fresh randomized assignment
    is required for every record.
    """
    contract = contrast_contract(gap, radius, alpha, beta)
    p0, p1 = contract["null_success_upper"], contract["alternative_success_lower"]
    if p1 <= p0:
        return dict(status="calibration_limited_design", contract=contract)
    if type(grid) is not int or grid < 2:
        raise ValueError("threshold grid must have at least two divisions")
    la = log_bounds(1/contract["alpha"])[1]
    lb = log_bounds(1/contract["beta"])[1]
    # Continuous KL ratio is monotone on (p0,p1). Binary search locates the
    # two adjacent grid points bracketing equality of the required counts.
    low, high = 1, grid-1
    while low < high:
        mid = (low+high)//2
        c = p0+(p1-p0)*F(mid,grid)
        d0, d1 = kl_bounds(c,p0)[0], kl_bounds(c,p1)[0]
        if la*d1 > lb*d0:
            low = mid+1
        else:
            high = mid
    candidates = []
    for index in sorted({max(1,low-1),low,min(grid-1,low+1)}):
        c = p0+(p1-p0)*F(index,grid)
        d0, d1 = kl_bounds(c,p0)[0], kl_bounds(c,p1)[0]
        if d0 > 0 and d1 > 0:
            n = ceil(max(la/d0, lb/d1))
            candidates.append((n,c,d0,d1))
    if not candidates:
        return dict(status="arithmetic_precision_limited_design",contract=contract)
    n,c,d0,d1 = min(candidates)
    result = dict(status="certified_sufficient_design", contract=contract,
                  records=n, success_fraction_threshold=c,
                  null_kl_lower=d0, alternative_kl_lower=d1,
                  threshold_grid=grid, null_log_upper=la, power_log_upper=lb)
    if not verify_fixed(result, **{k:contract[k] for k in ("gap","radius","alpha","beta")}):
        raise ArithmeticError("certificate failed")
    return result


def verify_fixed(candidate, gap, radius=0, alpha=F(1,20), beta=F(1,20)):
    """Verify sufficient bounds against an independently supplied contract."""
    try:
        contract=contrast_contract(gap,radius,alpha,beta)
        if candidate["contract"] != contract or candidate["status"] != "certified_sufficient_design":
            return False
        n=candidate["records"]
        if type(n) is not int or n <= 0:
            return False
        c=rational(candidate["success_fraction_threshold"])
        p0,p1=contract["null_success_upper"],contract["alternative_success_lower"]
        if not p0 < c < p1:
            return False
        d0,d1=kl_bounds(c,p0)[0],kl_bounds(c,p1)[0]
        la,lb=log_bounds(1/contract["alpha"])[1],log_bounds(1/contract["beta"])[1]
        return (n*d0>=la and n*d1>=lb and candidate["null_kl_lower"]==d0
                and candidate["alternative_kl_lower"]==d1
                and candidate["null_log_upper"]==la and candidate["power_log_upper"]==lb)
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def fixed_decision(plan, successes):
    c=plan["contract"]
    if not verify_fixed(plan,**{k:c[k] for k in ("gap","radius","alpha","beta")}):
        raise ValueError("invalid plan")
    n=plan["records"]
    if type(successes) is not int or not 0<=successes<=n:
        raise ValueError("invalid success count")
    return "reject_declared_classical_contract" if F(successes,n)>=plan["success_fraction_threshold"] else "inconclusive"


@lru_cache(maxsize=256, typed=True)
def binomial_upper_bounds(n, k, p):
    """Outward-rounded exact binomial upper tail, using integer arithmetic.

    The unrounded sum is an exact rational polynomial, not a floating-point
    CDF. Complexity grows with n and the rational denominator of p.
    """
    p=rational(p)
    if type(n) is not int or type(k) is not int or n<0 or not 0<p<1:
        raise ValueError("nonnegative integer n and interior rational p required")
    if k<=0:return F(1),F(1)
    if k>n:return F(0),F(0)
    a,b=p.numerator,p.denominator
    c=b-a
    term=c**n
    total=0
    for s in range(n+1):
        if s>=k:total+=term
        if s<n:
            numerator=term*(n-s)*a
            divisor=(s+1)*c
            term,remainder=divmod(numerator,divisor)
            if remainder:
                raise ArithmeticError("binomial integer recurrence failed")
    tail=F(total,b**n)
    return _outward(tail,tail)


def exact_fixed_plan(gap, radius=0, alpha=F(1,20), beta=F(1,20), records=600):
    """Exact binomial certificate for a supplied record budget.

    Finds the least integer success threshold certified at alpha. A power
    failure is reported, never relabeled as a successful design. This does
    not claim globally minimum records. The binomial bounds remain valid
    for adapted Bernoulli observations with the declared conditional bounds.
    """
    contract=contrast_contract(gap,radius,alpha,beta)
    if type(records) is not int or records<=0:
        raise ValueError("positive integer record budget required")
    p0,p1=contract["null_success_upper"],contract["alternative_success_lower"]
    if p1<=p0:return dict(status="calibration_limited_design",contract=contract)
    lo,hi=0,records+1
    while lo<hi:
        mid=(lo+hi)//2
        if binomial_upper_bounds(records,mid,p0)[1]<=contract["alpha"]:hi=mid
        else:lo=mid+1
    k=lo
    null_tail=binomial_upper_bounds(records,k,p0)
    alternative_tail=binomial_upper_bounds(records,k,p1)
    miss=(1-alternative_tail[1],1-alternative_tail[0])
    return dict(status="certified_sufficient_design" if miss[1]<=contract["beta"] else "power_not_certified",
                contract=contract,records=records,successes_threshold=k,
                false_rejection_interval=null_tail,miss_interval=miss,
                method="exact integer binomial tails with outward rational rounding")


def verify_exact_fixed(candidate,gap,radius=0,alpha=F(1,20),beta=F(1,20)):
    try:
        rebuilt=exact_fixed_plan(gap,radius,alpha,beta,candidate["records"])
        return rebuilt["status"]=="certified_sufficient_design" and candidate==rebuilt
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def bernoulli_exact_plan(null_upper,alternative_lower,records,successes_threshold,
                         alpha=F(1,20),beta=F(1,20)):
    """Exact tail certificate for a supplied generic one-sided design.

    Conditional success probabilities must be uniformly <= null_upper under
    the null, or >= alternative_lower for the stated power guarantee.
    This checks a supplied budget and threshold, without optimizing either.
    """
    p0,p1,alpha,beta=map(rational,(null_upper,alternative_lower,alpha,beta))
    if not 0<p0<p1<1 or not 0<alpha<1 or not 0<beta<1:
        raise ValueError("ordered interior probabilities and error levels required")
    if type(records) is not int or records<=0 or type(successes_threshold) is not int or not 0<=successes_threshold<=records+1:
        raise ValueError("valid integer record budget and rejection threshold required")
    null_tail=binomial_upper_bounds(records,successes_threshold,p0)
    alt_tail=binomial_upper_bounds(records,successes_threshold,p1)
    miss=(1-alt_tail[1],1-alt_tail[0])
    return dict(status="certified_sufficient_design" if null_tail[1]<=alpha and miss[1]<=beta else "error_bounds_not_certified",
                contract=dict(null_success_upper=p0,alternative_success_lower=p1,alpha=alpha,beta=beta),
                records=records,successes_threshold=successes_threshold,
                false_rejection_interval=null_tail,miss_interval=miss,
                method="exact integer binomial tails with outward rational rounding")


def verify_bernoulli_exact(candidate,null_upper,alternative_lower,alpha=F(1,20),beta=F(1,20)):
    try:
        rebuilt=bernoulli_exact_plan(null_upper,alternative_lower,candidate["records"],
                                     candidate["successes_threshold"],alpha,beta)
        return rebuilt["status"]=="certified_sufficient_design" and rebuilt==candidate
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def encode_dimension(arm,outcome):
    """Fresh uniform arm: 0=mu at0,1=nu at0,2=mu atU,3=nu atU.

    Witness W=mu*f-nu*f-average(mu*P*f-nu*P*f). Successes preserve positive
    coefficients (arms0,3), complement negative coefficients (arms1,2).
    The current arm/clock must not select the current bet or physical law.
    """
    if type(arm) is not int or arm not in (0,1,2,3) or type(outcome) is not int or outcome not in (0,1):
        raise ValueError("uniform four-arm index and binary outcome required")
    return outcome if arm in (0,3) else 1-outcome


def dimension_plan(witness_lower=F(261,250),radius=F(1,1000),records=20000,
                    successes_threshold=15107,alpha=F(1,20),beta=F(1,20)):
    """Four-arm fixed-record certificate for the <=3-state CTMC null.

    radius bounds absolute systematic error in the complete four-term W,
    not per-arm probability error. The physical dimension theorem is a
    separate requirement; rejection excludes its full declared contract.
    """
    w,b=map(rational,(witness_lower,radius))
    if not 1<w<=2 or not 0<=b<1:
        raise ValueError("witness lower bound in (1,2] and radius in [0,1) required")
    p0,p1=(3+b)/4,(2+w-b)/4
    if p1<=p0:
        return dict(status="calibration_limited_design",witness_lower=w,witness_radius=b)
    result=bernoulli_exact_plan(p0,p1,records,successes_threshold,alpha,beta)
    result["witness_lower"]=w
    result["witness_radius"]=b
    result["physical_null"]="common homogeneous CTMC with at most three states, arbitrary fixed preparations, fixed stochastic binary detector"
    return result


def verify_dimension(candidate,witness_lower=F(261,250),radius=F(1,1000),alpha=F(1,20),beta=F(1,20)):
    try:
        rebuilt=dimension_plan(witness_lower,radius,candidate["records"],candidate["successes_threshold"],alpha,beta)
        return rebuilt["status"]=="certified_sufficient_design" and rebuilt==candidate
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def fixed_lower_bound(records,successes,alpha=F(1,20),bits=24):
    """Exact conservative one-sided Bernoulli / encoded-contrast bound.

    With a stable common success probability this is a lower confidence
    bound for it. With adapted trials, it instead certifies rejection of
    every model with all conditional probabilities below the stated bound.
    The record count must be fixed independently of these observations.
    """
    alpha=rational(alpha)
    if type(records) is not int or records<=0 or type(successes) is not int or not 0<=successes<=records:
        raise ValueError("valid fixed record and success counts required")
    if not 0<alpha<1 or type(bits) is not int or bits<1:
        raise ValueError("alpha in (0,1) and positive integer precision required")
    lo,hi=F(0),F(1)
    if successes:
        for _ in range(bits):
            mid=(lo+hi)/2
            if binomial_upper_bounds(records,successes,mid)[1]<=alpha:lo=mid
            else:hi=mid
    return dict(records=records,successes=successes,alpha=alpha,bits=bits,
                success_probability_lower=lo,encoded_contrast_lower=2*lo-1,
                validity="fixed record count; stable mean or exclusion of uniform conditional upper bounds")


def anytime_lower_bound(records,successes,alpha=F(1,20),fraction=F(1,7),bits=24):
    """Time-uniform inversion of a fixed fractional likelihood bet.

    For each candidate theta the bet is q=theta+fraction*(1-theta).
    The fraction is chosen before observing this experiment and remains
    fixed; choosing it from current counts invalidates this API's guarantee.
    """
    alpha,fraction=map(rational,(alpha,fraction))
    if type(records) is not int or records<0 or type(successes) is not int or not 0<=successes<=records:
        raise ValueError("valid record and success counts required")
    if not 0<alpha<1 or not 0<fraction<1 or type(bits) is not int or bits<1:
        raise ValueError("valid alpha, predeclared fraction, and precision required")
    lo,hi=F(0),F(1)
    for _ in range(bits):
        theta=(lo+hi)/2
        capital=(1+fraction*(1-theta)/theta)**successes*(1-fraction)**(records-successes)
        if capital>=1/alpha:lo=theta
        else:hi=theta
    return dict(records=records,successes=successes,alpha=alpha,bits=bits,fraction=fraction,
                success_probability_lower=lo,encoded_contrast_lower=2*lo-1,
                validity="time-uniform for predeclared fractional bet; stable mean or uniform conditional upper bounds")


class AnytimeContrast:
    """Exact likelihood-ratio mixture for a calibrated one-sided null.

    Every conditional Bernoulli success probability must be <= (1+radius)/2.
    Radius can change predictably; components bet the same fractions of the
    remaining interval above the null. Predictions never use the next record.
    A mixture weight costs log(1/weight) in the finite-power certificate.
    Predictable means BEFORE the current arm and randomized clock draws.
    In the occupation experiment the null holds after averaging that clock;
    a bet selected from the current realized time is not covered.
    """
    def __init__(self, alpha=F(1,20), fractions=(F(1,32),F(1,16),F(1,8),F(1,4),F(1,2)), weights=None):
        self.alpha=rational(alpha)
        if not 0<self.alpha<1:
            raise ValueError("alpha must lie in (0,1)")
        self.fractions=tuple(map(rational,fractions))
        if not self.fractions or any(not 0<f<1 for f in self.fractions):
            raise ValueError("nonempty betting fractions in (0,1) required")
        self.weights=tuple(map(rational,weights)) if weights is not None else (F(1,len(self.fractions)),)*len(self.fractions)
        if len(self.weights)!=len(self.fractions) or any(w<=0 for w in self.weights) or sum(self.weights)!=1:
            raise ValueError("positive matching weights summing to one required")
        self.capitals=[F(1)]*len(self.fractions)
        self.records=0
        self.crossed=False

    @property
    def e_value(self):
        return sum((w*m for w,m in zip(self.weights,self.capitals)),F(0))

    def update(self, success, radius=0):
        if type(success) is not int or success not in (0,1):
            raise ValueError("success must be an integer bit")
        radius=rational(radius)
        if not 0<=radius<1:
            raise ValueError("predictable radius must lie in [0,1)")
        p0=(1+radius)/2
        for j,f in enumerate(self.fractions):
            p=p0+f*(1-p0)
            self.capitals[j] *= p/p0 if success else (1-p)/(1-p0)
        self.records+=1
        self.crossed |= self.e_value>=1/self.alpha
        return self.crossed


def anytime_power_plan(gap, radius=0, alpha=F(1,20), beta=F(1,20), fraction=None, weight=1):
    """Sufficient horizon for one component of an anytime mixture.

    Uses a constant radius and a uniform conditional alternative probability
    bound; does not require identical or independent successes. Certificate
    is Hoeffding's bounded-increment inequality, not an expected stopping time.
    """
    contract=contrast_contract(gap,radius,alpha,beta)
    p0,p1=contract["null_success_upper"],contract["alternative_success_lower"]
    if p1<=p0:
        return dict(status="calibration_limited_design",contract=contract)
    weight=rational(weight)
    fraction=(p1-p0)/(1-p0) if fraction is None else rational(fraction)
    if not 0<weight<=1 or not 0<fraction<1:
        raise ValueError("weight in (0,1], fraction in (0,1) required")
    p=p0+fraction*(1-p0)
    l1lo,l1hi=log_bounds(p/p0)
    l0lo,l0hi=log_bounds((1-p)/(1-p0))
    drift=p1*l1lo+(1-p1)*l0lo
    if drift<=0:
        return dict(status="uncertified_bet",contract=contract)
    width=l1hi-l0lo
    la=log_bounds(1/(contract["alpha"]*weight))[1]
    lb=log_bounds(1/contract["beta"])[1]
    def sufficient(n):
        return n*drift>la and (n*drift-la)**2>=n*width**2*lb/2
    hi=1
    while not sufficient(hi):
        hi*=2
    lo=1
    while lo<hi:
        mid=(lo+hi)//2
        if sufficient(mid):hi=mid
        else:lo=mid+1
    return dict(status="certified_sufficient_horizon",contract=contract,records=lo,
                betting_fraction=fraction,component_weight=weight,
                log_drift_lower=drift,log_increment_width_upper=width,
                null_log_upper=la,power_log_upper=lb)


def verify_anytime(candidate,gap,radius=0,alpha=F(1,20),beta=F(1,20)):
    try:
        contract=contrast_contract(gap,radius,alpha,beta)
        if candidate["contract"]!=contract or candidate["status"]!="certified_sufficient_horizon":
            return False
        # Reconstruction checks every bound; accepted record counts can exceed
        # the sufficient minimum without trusting a supplied drift or width.
        rebuilt=anytime_power_plan(gap,radius,alpha,beta,candidate["betting_fraction"],candidate["component_weight"])
        n=candidate["records"]
        return type(n) is int and n>=rebuilt["records"] and all(candidate[k]==v for k,v in rebuilt.items() if k!="records")
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def exact_anytime_power_plan(gap,radius=0,alpha=F(1,20),beta=F(1,20),records=1100,
                             fraction=None,weight=1):
    """Exact finite-horizon power bound for an anytime likelihood-ratio test.

    At the supplied horizon, one component alone suffices for rejection if
    its weight times its capital exceeds 1/alpha. The actual crossing event
    can have larger power; this certificate only uses terminal capital.
    """
    contract=contrast_contract(gap,radius,alpha,beta)
    if type(records) is not int or records<=0:
        raise ValueError("positive integer horizon required")
    p0,p1=contract["null_success_upper"],contract["alternative_success_lower"]
    if p1<=p0:return dict(status="calibration_limited_design",contract=contract)
    fraction=(p1-p0)/(1-p0) if fraction is None else rational(fraction)
    weight=rational(weight)
    if not 0<fraction<1 or not 0<weight<=1:
        raise ValueError("invalid betting fraction or component weight")
    p=p0+fraction*(1-p0)
    target=1/(contract["alpha"]*weight)
    def terminal_capital(k):
        return (p/p0)**k*((1-p)/(1-p0))**(records-k)
    lo,hi=0,records+1
    while lo<hi:
        mid=(lo+hi)//2
        if terminal_capital(mid)>=target:hi=mid
        else:lo=mid+1
    k=lo
    tails=binomial_upper_bounds(records,k,p1)
    miss=(1-tails[1],1-tails[0])
    return dict(status="certified_sufficient_horizon" if miss[1]<=contract["beta"] else "power_not_certified",
                contract=contract,records=records,terminal_successes_threshold=k,
                betting_fraction=fraction,component_weight=weight,
                miss_interval=miss,anytime_false_rejection_bound=contract["alpha"],
                method="Ville validity and exact binomial terminal-power certificate")


def verify_exact_anytime(candidate,gap,radius=0,alpha=F(1,20),beta=F(1,20)):
    try:
        rebuilt=exact_anytime_power_plan(gap,radius,alpha,beta,candidate["records"],
                                        candidate["betting_fraction"],candidate["component_weight"])
        return rebuilt["status"]=="certified_sufficient_horizon" and rebuilt==candidate
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return False


def examples():
    rows=[]
    for gap in (F(1,100),F(1,20),F(1,10),F(3,20)):
        rows.append(dict(gap=gap,calibration_radius=F(1,1000),
                         fixed=fixed_plan(gap,F(1,1000)),
                         anytime=anytime_power_plan(gap,F(1,1000))))
    main=dict(gap=F(18,125),radius=F(1,1000),
              fixed_chernoff=fixed_plan(F(18,125),F(1,1000)),
              fixed_exact=exact_fixed_plan(F(18,125),F(1,1000)),
              anytime=anytime_power_plan(F(18,125),F(1,1000)),
              anytime_exact=exact_anytime_power_plan(F(18,125),F(1,1000)),
              expected_evolution_time_per_record=F(9,8),
              quantum_note="gap requires the independent physical contrast certificate and common detector gain 9/10")
    dimension=dict(plan=dimension_plan(),expected_evolution_time_per_record=F(9,16),
                   expected_total_evolution_time=F(11250),
                   calibration_note="radius .001 is the full four-term witness error; four independent per-arm bounds .00025 suffice",
                   comparison_note="broader preparation/readout contract than the600-record test; smaller signal and larger record budget")
    dimension_tilted=dict(plan=dimension_plan(F(27,25),F(1,500),6000,4560),
                          expected_evolution_time_per_record=F(1,2),
                          expected_total_evolution_time=F(3000),
                          horizon=F(2),quantum_bloch_axis=(F(0),F(-12,37),F(35,37)),
                          calibration_note="full witness radius .002; four per-arm probability errors .0005 suffice",
                          physical_note="independent quantum certificate establishes W>=1.08; no exact preparation support required by the classical null")
    return dict(evidence="proved sufficient designs; example gaps are declared contracts, not measured data", main=main, dimension=dimension, dimension_tilted=dimension_tilted, examples=rows,
                physical_note="one independently randomized preparation and one endpoint readout per record; add preparation/readout energy and sampled evolution time",
                validity="calibration failure probability is added once; no claim of quantumness without the physical null contract")


if __name__=="__main__":
    path=Path(__file__).resolve().parents[1]/"results"/"statistics.json"
    path.write_text(json.dumps(examples(),default=str,indent=2)+"\n")
    print("wrote",path)
