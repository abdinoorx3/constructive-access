"""Finite-window classical realization certificates for reset experiments.

The approximation operator is the classical Szasz--Mirakjan Poisson operator.
This module supplies explicit finite CTMC realizations, including a reversible
lift with deterministic readout.  Certificates concern state/rate resources;
they do not establish an experimentally efficient classical implementation.
"""
from fractions import Fraction as F
from math import ceil, isqrt, log
from pathlib import Path
import json


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, F)):
        raise TypeError("Use an integer, rational string, or Fraction")
    return F(x)


def poisson_tail_bound(mean, threshold):
    """Exact factorial-moment bound on P(Poisson(mean) >= threshold).

    For integer s, E[(N)_s]=mean**s and (N)_s >= (threshold)_s
    on N>=threshold.  All factors <=1 are included, giving the best bound
    in this elementary falling-factorial family.  This is not an exact tail.
    """
    mean = rational(mean)
    if mean < 0 or not isinstance(threshold, int) or threshold < 0:
        raise ValueError("Nonnegative mean and integer threshold required")
    if threshold == 0:
        return F(1)
    if mean == 0:
        return F(0)
    result = F(1)
    for denominator in range(threshold, 0, -1):
        if denominator < mean:
            break
        result *= mean / denominator
    return result


def _tail_threshold(mean, tolerance):
    lo = max(1, ceil(mean))
    hi = lo
    while poisson_tail_bound(mean, hi) > tolerance:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if poisson_tail_bound(mean, mid) <= tolerance:
            hi = mid
        else:
            lo = mid + 1
    return lo


def lipschitz_certificate(lipschitz, horizon, tolerance, preparations=1,
                          reversible=False):
    """Constructive uniform error certificate on a finite time window.

    Input functions lie in [0,1] and share the stated Lipschitz constant.
    The certificate assumes exact samples; add any uniform sampling error.
    Reversible construction splits error equally between clock dispersion
    and backwards motion.  Disconnected preparation sectors are counted.
    """
    L, T, eps = map(rational, (lipschitz, horizon, tolerance))
    if L <= 0 or T <= 0 or not 0 < eps < 1:
        raise ValueError("Positive L,T and 0<tolerance<1 required")
    if not isinstance(preparations, int) or preparations < 1:
        raise ValueError("Positive integer preparation count required")
    clock_error = eps / 2 if reversible else eps
    rate = L * L * T / (clock_error * clock_error)
    phases = ceil(rate * T)
    backwards = eps / (2 * T) if reversible else F(0)
    return dict(kind="Lipschitz", rate=rate, backward_rate=backwards,
                horizon=T, last_phase=phases, preparations=preparations,
                deterministic_state_bound=2 * preparations * (phases + 1),
                peak_exit_rate_bound=rate + backwards,
                uniform_error_bound=clock_error + backwards * T,
                poisson_clock_error=clock_error,
                reversibility_error=backwards * T)


def smooth_certificate(curvature, horizon, tolerance, preparations=1,
                       reversible=False, tail_method="factorial"):
    """Certificate for all-time bounded probabilities with |p''|<=curvature.

    Unlike the Lipschitz construction, samples beyond the observation window
    are used.  A global second-derivative bound on t>=0 is required.
    """
    M, T, eps = map(rational, (curvature, horizon, tolerance))
    if M <= 0 or T <= 0 or not 0 < eps < 1:
        raise ValueError("Positive M,T and 0<tolerance<1 required")
    if not isinstance(preparations, int) or preparations < 1:
        raise ValueError("Positive integer preparation count required")
    share = eps / (3 if reversible else 2)
    rate = M * T / (2 * share)
    mean = rate * T
    if tail_method == "factorial":
        phases = _tail_threshold(mean, share)
        tail = poisson_tail_bound(mean, phases)
    elif tail_method == "cantelli":
        variance_allowance = mean*(1-share)/share
        integer_allowance = ceil(variance_allowance)
        root = isqrt(integer_allowance)
        if root*root < integer_allowance:
            root += 1
        phases = ceil(mean) + root
        tail = mean/(mean+(phases-mean)**2)
    else:
        raise ValueError("tail_method must be factorial or cantelli")
    backwards = share / T if reversible else F(0)
    return dict(kind="smooth", tail_method=tail_method, rate=rate, backward_rate=backwards,
                horizon=T, last_phase=phases, preparations=preparations,
                deterministic_state_bound=2 * preparations * (phases + 1),
                peak_exit_rate_bound=rate + backwards,
                uniform_error_bound=share + tail + backwards * T,
                poisson_clock_error=share, truncation_error=tail,
                reversibility_error=backwards * T)


def transcript_bound(per_record_error, records):
    eps = rational(per_record_error)
    if not 0 <= eps <= 1 or not isinstance(records, int) or records < 0:
        raise ValueError("0<=error<=1 and a nonnegative record count required")
    return 1 - (1 - eps) ** records


def _atan_bounds(x, terms=60):
    x = rational(x)
    if not 0 <= x <= 1 or not isinstance(terms, int) or terms < 1:
        raise ValueError("0<=x<=1 and a positive term count required")
    total = sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)), F(0))
    adjacent = total + (-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(total, adjacent), max(total, adjacent)


def cycle_admissibility_certificate(gamma, omega, states, terms=60):
    """Rational certificate for n atan(gamma/omega) >= pi.

    The rate construction itself uses algebraic/trigonometric values.
    This certificate proves its nonnegative-rate condition without accepting
    floating-point rounding at the spectral boundary.
    """
    gamma, omega = map(rational, (gamma, omega))
    if gamma <= 0 or omega <= 0 or not isinstance(states, int) or states < 3:
        raise ValueError("Positive gamma,omega and at least three states required")
    a5, b5 = _atan_bounds(F(1,5), terms)
    a239, b239 = _atan_bounds(F(1,239), terms)
    pi_lo, pi_hi = 16*a5-4*b239, 16*b5-4*a239
    ratio = gamma/omega
    if ratio <= 1:
        angle_lo, angle_hi = _atan_bounds(ratio, terms)
    else:
        u, v = _atan_bounds(1/ratio, terms)
        angle_lo, angle_hi = pi_lo/2-v, pi_hi/2-u
    lower, upper = states*angle_lo-pi_hi, states*angle_hi-pi_lo
    return dict(states=states, margin_lower=lower, margin_upper=upper,
                admissible=lower >= 0, inadmissible=upper < 0,
                decision="admissible" if lower >= 0 else "inadmissible" if upper < 0 else "unresolved")


def cyclic_realization(gamma, omega, states, antipodal=False):
    """Exact-form biased cycle; numerical matrices are an illustration.

    Fixed stochastic detector f_k=(1+cos(2*pi*k/n))/2.  The first preparation
    yields (1+exp(-gamma*t)*cos(omega*t))/2 for every t>=0.  For even n,
    antipodal=True also supplies its complement from the opposite phase.
    A deterministic detector lift requires at most 2n states, with zero-weight
    states deleted; its existence is proved in the accompanying manuscript.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    certificate = cycle_admissibility_certificate(gamma, omega, states)
    if not certificate["admissible"]:
        raise ValueError("Nonnegative cyclic rates not certified")
    if antipodal and states % 2:
        raise ValueError("This antipodal construction needs an even cycle")
    theta = 2*np.pi/states
    total = float(rational(gamma))/(1-np.cos(theta))
    difference = float(rational(omega))/np.sin(theta)
    forward, backward = (total+difference)/2, (total-difference)/2
    rows, cols, values = [], [], []
    for k in range(states):
        rows.extend([k,k,k]); cols.extend([(k+1)%states,(k-1)%states,k])
        values.extend([forward,backward,-total])
    Q = coo_matrix((values,(rows,cols)),shape=(states,states)).tocsr()
    detector = (1+np.cos(theta*np.arange(states)))/2
    initial = np.zeros((2 if antipodal else 1,states))
    initial[0,0] = 1
    if antipodal:
        initial[1,states//2] = 1
    epr = (forward-backward)*np.log(forward/backward) if backward > 0 else float("inf")
    return dict(generator=Q, detector=detector, initial=initial,
                forward_rate=forward, backward_rate=backward,
                stationary_entropy_production=epr, preparation_kl=log(states),
                deterministic_lift_state_bound=2*states-(2 if states%2==0 else 1),
                admissibility=certificate)


def realization(samples, rate, backward_rate=0):
    """Numerical sparse CTMC realizing a finite table of phase probabilities.

    samples[a][k] is an exact rational detector probability.  The detector
    in the returned model is deterministic.  Zero stationary-weight bit
    states are deleted.  Every preparation starts in its own phase zero.
    Backward_rate>0 gives detailed balance in each preparation sector.
    The initial KL divergence from the specified equally weighted stationary
    sector mixture is log(m*Z), with Z=sum((r/d)**k).
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.special import logsumexp
    r, d = map(rational, (rate, backward_rate))
    if r <= 0 or d < 0 or not samples:
        raise ValueError("Positive forward rate, nonnegative backward rate required")
    table = [[rational(p) for p in row] for row in samples]
    K = len(table[0]) - 1
    if K < 1 or any(len(row) != K + 1 for row in table):
        raise ValueError("At least two phases and equal table lengths required")
    if any(not 0 <= p <= 1 for row in table for p in row):
        raise ValueError("Detector probabilities must lie in [0,1]")
    states = [(a, k, y) for a, row in enumerate(table)
              for k, p in enumerate(row) for y in (0, 1)
              if (p if y else 1 - p) > 0]
    index = {state: i for i, state in enumerate(states)}
    rows, cols, data = [], [], []
    for i, (a, k, y) in enumerate(states):
        exit_rate = F(0)
        destinations = ([(k + 1, r)] if k < K else [])
        if k > 0 and d > 0:
            destinations.append((k - 1, d))
        for ell, charge in destinations:
            for z in (0, 1):
                probability = table[a][ell] if z else 1 - table[a][ell]
                if probability:
                    rows.append(i); cols.append(index[(a, ell, z)])
                    data.append(float(charge * probability))
                    exit_rate += charge * probability
        rows.append(i); cols.append(i); data.append(-float(exit_rate))
    Q = coo_matrix((data, (rows, cols)), shape=(len(states), len(states))).tocsr()
    initial = np.zeros((len(table), len(states)))
    detector = np.array([y for _, _, y in states], dtype=float)
    for a, row in enumerate(table):
        for y in (0, 1):
            if (a, 0, y) in index:
                initial[a, index[(a, 0, y)]] = float(row[0] if y else 1-row[0])
    stationary = None
    log_stationary = None
    preparation_kl = None
    if d > 0:
        log_weights = np.arange(K + 1) * log(float(r / d))
        log_Z = float(logsumexp(log_weights))
        phase_weights = np.exp(log_weights - log_Z)
        log_stationary = np.array([log_weights[k]-log_Z-log(len(table))+
                                   log(float(table[a][k] if y else 1-table[a][k]))
                                   for a, k, y in states])
        stationary = np.exp(log_stationary)
        preparation_kl = log(len(table)) + log_Z
    return dict(generator=Q, detector=detector, initial=initial,
                stationary=stationary, log_stationary=log_stationary,
                preparation_kl=preparation_kl,
                states=states, table=table)


def detailed_balance_certificate(samples, rate, backward_rate):
    """Exact edge-flow equality without normalizing enormous weights."""
    r, d = map(rational, (rate, backward_rate))
    if r <= 0 or d <= 0:
        raise ValueError("Positive rates required")
    checked = 0
    for row in samples:
        row = [rational(p) for p in row]
        if any(not 0 <= p <= 1 for p in row):
            raise ValueError("Invalid probability")
        for k in range(len(row) - 1):
            for y in (0, 1):
                for z in (0, 1):
                    py = row[k] if y else 1-row[k]
                    pz = row[k+1] if z else 1-row[k+1]
                    forward_flow = (r/d)**k * py * r * pz
                    reverse_flow = (r/d)**(k+1) * pz * d * py
                    if forward_flow != reverse_flow:
                        return dict(valid=False, checked=checked)
                    checked += 1
    return dict(valid=True, checked=checked)


def _jsonable(value):
    if isinstance(value, F):
        return dict(exact=str(value), decimal=float(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_results():
    import numpy as np
    from scipy.sparse.linalg import expm_multiply
    gamma, omega, T = F(1, 10), F(2), F(9, 4)
    M = (gamma * gamma + omega * omega) / 2
    certificate = smooth_certificate(M, T, F(1, 20), preparations=2,
                                     reversible=True)
    r, d, K = (certificate[k] for k in ("rate", "backward_rate", "last_phase"))
    t_phase = np.arange(K+1) / float(r)
    p = (1 + np.exp(-float(gamma)*t_phase)*np.cos(float(omega)*t_phase))/2
    # Decimal samples define exact rational CTMC parameters; sampling roundoff
    # is <1e-14 for this numerical illustration, not part of the symbolic proof.
    rows = [[F(str(float(x))) for x in p], [F(str(float(1-x))) for x in p]]
    model = realization(rows, r, d)
    times = np.linspace(0, float(T), 101)
    trajectories = expm_multiply(model["generator"], model["detector"],
                                 start=0, stop=float(T), num=len(times))
    predicted = trajectories @ model["initial"].T
    target = (1+np.exp(-float(gamma)*times)*np.cos(float(omega)*times))/2
    error = max(float(np.max(abs(predicted[:, 0]-target))),
                float(np.max(abs(predicted[:, 1]-(1-target)))))
    exact_db = detailed_balance_certificate(rows, r, d)
    stats = dict(certificate=certificate, actual_states=len(model["states"]),
                 sampled_uniform_error=error, sampled_points=len(times),
                 preparation_kl_nats=model["preparation_kl"],
                 stationary_entropy_production=0,
                 exact_detailed_balance=exact_db,
                 transient_entropy_production_may_be_positive=True,
                 benchmark="synthetic damped qubit; no physical measurements")
    no_go = smooth_certificate(M, T, F(1, 1000), preparations=2,
                               reversible=True)
    finite_records = smooth_certificate(M, T, F(1,120000), preparations=2,
                                         reversible=True, tail_method="cantelli")
    cycles = []
    for size in (63,64):
        cycle = cyclic_realization(gamma, omega, size, antipodal=(size==64))
        values = expm_multiply(cycle["generator"],cycle["detector"],
                               start=0,stop=10,num=101) @ cycle["initial"].T
        tt = np.linspace(0,10,101)
        expected = (1+np.exp(-float(gamma)*tt)*np.cos(float(omega)*tt))/2
        error = float(np.max(abs(values[:,0]-expected)))
        if size==64:
            error=max(error,float(np.max(abs(values[:,1]-(1-expected)))))
        cycles.append(dict(states=size, prepares_complement=(size==64),
                           sampled_error=error,
                           forward_rate=cycle["forward_rate"],
                           backward_rate=cycle["backward_rate"],
                           stationary_entropy_production=cycle["stationary_entropy_production"],
                           preparation_kl=cycle["preparation_kl"],
                           deterministic_lift_state_bound=cycle["deterministic_lift_state_bound"],
                           admissibility=cycle["admissibility"]))
    return dict(reversible_benchmark=stats,
                exact_form_cycle_benchmarks=cycles,
                sixty_two_state_cycle=cycle_admissibility_certificate(gamma,omega,62),
                tighter_uniform_certificate=no_go,
                six_thousand_record_certificate=dict(
                    records=6000, transcript_tv_bound=F(1,20),
                    construction=finite_records,
                    qualifier="A sufficient upper resource cost, not a minimum or practical hardware design"),
                transcript_example=dict(per_record_error=F(1, 1000), records=100,
                                        total_variation_bound=transcript_bound(F(1, 1000),100)),
                scope="Fresh reset trials; finite preparations; fixed binary readout; retained timestamps allowed; no intermediate instruments")


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "results" / "limits.json"
    destination.parent.mkdir(exist_ok=True)
    result = _jsonable(make_results())
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(destination=str(destination),
                         sampled_error=result["reversible_benchmark"]["sampled_uniform_error"],
                         states=result["reversible_benchmark"]["actual_states"])))
