"""Conditional-mean confidence boxes for dependent physical records.

Established conditional Hoeffding bounds, with explicitly supplied conditional
bias and apparatus envelopes.  The module checks arithmetic, not whether an
unknown apparatus truly obeys its declared mixing or calibration contract.
"""
from fractions import Fraction as F
from math import ceil, isqrt
from pathlib import Path
import hashlib
import json

from constructive_access.frontier.energy_certificate import (
    exact, log_interval, certify, resource_frontier,
)


def _rational_sqrt_upper(value, bits=32):
    value = exact(value)
    if value < 0 or type(bits) is not int or not 1 <= bits <= 128:
        raise ValueError("Nonnegative value and bit count1..128 required")
    scale = 1 << bits
    integer = ceil(value*scale*scale)
    root = isqrt(integer)
    if root*root < integer:
        root += 1
    return F(root,scale)


def confidence_radius(records, settings, alpha, bits=32):
    """Outward rational radius for equal Bonferroni allocation."""
    alpha = exact(alpha)
    if type(records) is not int or records <= 0 or type(settings) is not int or settings <= 0:
        raise ValueError("Positive record and setting counts required")
    if not 0 < alpha < 1:
        raise ValueError("0<alpha<1 required")
    return _rational_sqrt_upper(log_interval(F(2*settings)/alpha)[1]/(2*records),bits)


def _radii(values, count, name):
    if isinstance(values,(int,str,F)) and not isinstance(values,bool):
        values = [values]*count
    values = list(map(exact,values))
    if len(values) != count or any(x < 0 for x in values):
        raise ValueError("Invalid "+name)
    return values


def confidence_box(records, successes, statistical_radii, alpha,
                   conditional_bias_radii, calibration_radii=None,
                   calibration_failure=0, sampling_design="fixed_quotas_predictable"):
    """Counts -> a simultaneous box without an independence assumption.

    Each setting has a deterministic quota.  The next unfinished setting may
    depend on past records, but must be selected before the current outcome.
    Its whole-record conditional mean must stay within conditional_bias_radii
    of a fixed target probability (plus the declared apparatus correction).
    The caller supplies this physical contract; observed counts do not prove it.
    Data-dependent stopping or outcome-dependent inclusion is not supported.
    calibration_failure bounds failure of the global envelope for the entire
    run, not a separate per-record failure probability.
    """
    alpha, failure = map(exact,(alpha,calibration_failure))
    m = len(records)
    if not m or len(successes) != m or not 0 < alpha < 1:
        raise ValueError("Matching nonempty counts and0<alpha<1 required")
    if not 0 <= failure < 1 or alpha+failure >= 1:
        raise ValueError("Invalid total confidence failure budget")
    if sampling_design != "fixed_quotas_predictable":
        raise ValueError("Only deterministic quotas with predictable routing are supported")
    if any(type(n) is not int or n <= 0 for n in records):
        raise ValueError("Positive integer deterministic quotas required")
    if any(type(s) is not int or not 0 <= s <= n for n,s in zip(records,successes)):
        raise ValueError("Invalid binary success counts")
    statistical = _radii(statistical_radii,m,"statistical radii")
    bias = _radii(conditional_bias_radii,m,"conditional bias radii")
    calibration = _radii([F(0)]*m if calibration_radii is None else calibration_radii,
                         m,"calibration radii")
    if any(x > 1 for x in bias+calibration):
        raise ValueError("Probability bias radii must not exceed one")
    log_bound = log_interval(F(2*m)/alpha)[1]
    if any(2*n*r*r < log_bound for n,r in zip(records,statistical)):
        raise ValueError("Statistical radius does not certify the stated joint coverage")
    boxes = [(max(F(0),F(s,n)-r-d-e),min(F(1),F(s,n)+r+d+e))
             for n,s,r,d,e in zip(records,successes,statistical,bias,calibration)]
    return dict(records=list(records),successes=list(successes),intervals=boxes,
                statistical_radii=statistical,conditional_bias_radii=bias,
                calibration_radii=calibration,statistical_failure_upper=alpha,
                external_contract_failure_upper=failure,
                total_failure_upper=alpha+failure,
                log_threshold_upper=log_bound,sampling_design=sampling_design,
                required_contract="Fixed target probability per setting; whole-record conditional means bounded before each outcome; apparatus and mixing assertions supplied externally",
                limitation="Marginal stationarity, fitted effective sample size, or overlapping timestamps do not establish the conditional-mean bound")


def certify_counts(lags,step,weights,records,successes,statistical_radii,alpha,
                   conditional_bias_radii,scores,calibration_radii=None,
                   calibration_failure=0,depth=4,
                   sampling_design="fixed_quotas_predictable"):
    """Dependent-record confidence box -> continuous reversible-moment bound."""
    box = confidence_box(records,successes,statistical_radii,alpha,
                         conditional_bias_radii,calibration_radii,
                         calibration_failure,sampling_design)
    if len(lags) != len(records):
        raise ValueError("One count quota per lag is required")
    certificate = certify(lags,step,weights,box["intervals"],(-1,1),scores,depth)
    return dict(confidence=box,certificate=certificate,
                resource_frontier=resource_frontier(certificate),
                evidence="Supplied counts; dependence handled only under the declared conditional-mean contract; physical stationary/ordinary-reversal hypotheses remain separate")


def doeblin_mix_bound(minorization, blocks):
    """P_h(x,.)>=epsilon nu(.) for every hidden x => TV<= (1-epsilon)^j."""
    epsilon = exact(minorization)
    if not 0 <= epsilon <= 1 or type(blocks) is not int or blocks < 0:
        raise ValueError("0<=minorization<=1 and nonnegative integer blocks required")
    return (1-epsilon)**blocks


def exp_negative_upper(value, terms=80):
    """Exact e^(-x) upper bound from a positive Taylor lower bound for e^x."""
    x = exact(value)
    if x < 0 or type(terms) is not int or terms < 1:
        raise ValueError("Nonnegative x and positive integer terms required")
    term = total = F(1)
    for k in range(1,terms+1):
        term *= x/k
        total += term
    return 1/total


def spectral_mix_bound(spectral_gap_lower, stationary_minimum, blind_gap,
                       terms=80,bits=32):
    """Uniform TV envelope using the additive-reversibilization spectral gap.

    Requires finite homogeneous Q, positive stationary pi, known
    gap((Q+Q^dagger)/2)>=gap_lower and min_x pi_x>=stationary_minimum.
    It also applies to reversible Q, for which Q=Q^dagger.  This is not an
    estimator of those assumptions. Detector/controller memory must be
    included or reset. Q^dagger is the stationary-pi adjoint.
    """
    rate,pmin,time = map(exact,(spectral_gap_lower,stationary_minimum,blind_gap))
    if rate <= 0 or not 0 < pmin <= 1 or time < 0:
        raise ValueError("Positive gap,0<pmin<=1 and nonnegative time required")
    prefactor = _rational_sqrt_upper((1-pmin)/(4*pmin),bits)
    decay = exp_negative_upper(rate*time,terms)
    return dict(conditional_bias_upper=min(F(1),prefactor*decay),
                prefactor_upper=prefactor,decay_upper=decay,
                spectral_gap_lower=rate,stationary_minimum=pmin,blind_gap=time,
                required_contract="Uniform additive-reversibilization gap and stationary-mass bounds for the full hidden state, independently justified")


def _wilson(successes, trials):
    from scipy.stats import norm
    z = float(norm.ppf(.975)); p = successes/trials
    denominator = 1+z*z/trials
    center = (p+z*z/(2*trials))/denominator
    radius = z*((p*(1-p)/trials+z*z/(4*trials*trials))**.5)/denominator
    return [0. if successes==0 else max(0.,center-radius),
            1. if successes==trials else min(1.,center+radius)]


def make_results():
    import numpy as np
    from scipy.linalg import expm,eigh
    from scipy.stats import poisson
    root = Path(__file__).resolve().parents[1]
    contract_path = root/"contracts"/"sampling.json"
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    if not contract.get("frozen_before_production"):
        raise ValueError("A frozen simulation contract is required")
    design = contract["design"]; model = contract["model"]
    n = design["records_per_lag"]; lags=design["lags"]
    reps=design["replicates"]; seed=design["seed"]
    alpha=F(design["statistical_alpha"]); radius=F(design["statistical_radius"])
    confidence_box(n,[0]*len(n),radius,alpha,[F(0)]*len(n))
    kapp = float(F(model["sector_flip_rate"]))
    rates = list(map(lambda x:float(F(x)),model["bit_flip_rates_by_sector"]))
    Q=np.zeros((4,4))
    for sector in range(2):
        for bit in range(2):
            i=2*sector+bit
            Q[i,2*sector+1-bit]=rates[sector]
            Q[i,2*(1-sector)+bit]=kapp
            Q[i,i]=-rates[sector]-kapp
    detector=np.array([0,1,0,1])
    pi=np.full(4,.25)
    transition=[expm(Q*lag) for lag in lags]
    targets=np.array([sum(pi[i]*P[i,j] for i in range(4) for j in range(4)
                          if detector[i]==detector[j]) for P in transition])
    # Independent uniformization check for each pair semigroup.
    rate=max(-np.diag(Q)); U=np.eye(4)+Q/rate
    uniformization_error=[]
    for lag,P in zip(lags,transition):
        power=np.eye(4); approximation=np.zeros((4,4))
        for j in range(81):
            approximation += poisson.pmf(j,rate*lag)*power
            power=power@U
        uniformization_error.append(float(np.max(abs(P-approximation))))
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    rows=[]
    for scenario,rng in zip(contract["scenarios"],rngs):
        name=scenario["name"]; gap=F(scenario["inter_record_gap"])
        Pgap=expm(Q*float(gap))
        Pgap=np.maximum(Pgap,0); Pgap/=Pgap.sum(axis=1,keepdims=True)
        cdfgap=Pgap.cumsum(axis=1); cdfgap[:,-1]=1
        state=(rng.integers(0,4,size=reps) if name=="shared_endpoint_pairs"
               else np.zeros(reps,dtype=int))
        counts=np.zeros((reps,len(lags)),dtype=int)
        for k,(quota,P) in enumerate(zip(n,transition)):
            P=np.maximum(P,0); P/=P.sum(axis=1,keepdims=True); cdf=P.cumsum(axis=1); cdf[:,-1]=1
            for _ in range(quota):
                if name=="independent_stationary_pairs":
                    state=rng.integers(0,4,size=reps)
                elif gap>0:
                    state=(rng.random(reps)[:,None]>cdfgap[state]).sum(axis=1)
                before=detector[state]
                state=(rng.random(reps)[:,None]>cdf[state]).sum(axis=1)
                counts[:,k] += before==detector[state]
        estimates=counts/np.array(n)
        errors=abs(estimates-targets)
        analysis=[]
        if name=="shared_endpoint_pairs":
            choices=[("unqualified_iid",F(0)),("honest_conditional_bound",F(3,4))]
        else:
            choices=[("declared_corrected",F(scenario["declared_conditional_bias"]))]
        for label,bias in choices:
            halfwidth=float(radius+bias)
            cover=np.all(errors<=halfwidth,axis=1)
            reject=estimates[:,1]-estimates[:,0]>2*halfwidth
            analysis.append(dict(label=label,conditional_bias=str(bias),
                                 coverage=float(np.mean(cover)),
                                 coverage_mc95=_wilson(int(sum(cover)),reps),
                                 false_reversible_monotonicity_exclusions=float(np.mean(reject)),
                                 false_exclusion_mc95=_wilson(int(sum(reject)),reps)))
        # Data are retained as histograms with exact multiplicities for replay.
        unique,multiplicity=np.unique(counts,axis=0,return_counts=True)
        rows.append(dict(name=name,replicates=reps,analyses=analysis,
                         physical_model_time_per_replicate=float(sum(q*(lag+gap) for q,lag in zip(n,lags))),
                         count_histogram=[[int(x) for x in pair]+[int(mult)] for pair,mult in zip(unique,multiplicity)],
                         counts_sha256=hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest()))
    mixing=spectral_mix_bound(F(model["certified_spectral_gap_lower"]),F(1,4),F(3000))
    eigenvalues,vectors=eigh(Q)
    gap_eigen= vectors@np.diag(np.exp(eigenvalues*3000))@vectors.T
    solver_gap_error=float(np.max(abs(expm(Q*3000)-gap_eigen)))
    return dict(contract_sha256=hashlib.sha256(raw).hexdigest(),seed=seed,
                probability_targets=targets.tolist(),generator=Q.tolist(),
                stationary_entropy_production=0,uniformization_max_errors=uniformization_error,
                gap_semigroup_crosscheck_error=solver_gap_error,mixing_certificate=mixing,
                declared_gap_bias_verified=mixing["conditional_bias_upper"]<=F(1,400),
                scenarios=rows,
                evidence="Independent synthetic replicate trajectories; no measured apparatus or inferred universal mixing rate")


if __name__ == "__main__":
    result=make_results()
    destination=Path(__file__).resolve().parents[1]/"results"/"sampling.json"
    destination.write_text(json.dumps(result,default=str,sort_keys=True,indent=2)+"\n")
    print(json.dumps({row["name"]:row["analyses"] for row in result["scenarios"]},indent=2))
