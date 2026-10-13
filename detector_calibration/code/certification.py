"""Exact finite-data certificates for replicated boxcar observations.

All public numerical inputs are integers, rational strings or Fractions. Physical
premises (stationarity, noninvasiveness, replica independence conditional on the
latent path, calibrated clocks and generator bounds) cannot be checked from a
count table. The supported contract is stated explicitly and unsupported labels
are rejected. No laboratory measurements or numerical simulator live here.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb, factorial, isqrt


def exact(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, F)):
        raise TypeError("Use exact integers, rational strings or Fractions")
    return F(value)


@lru_cache(maxsize=128)
def log_interval(value):
    """Rational outward enclosure, inherited from observed_geometry/finite_resolution."""
    x = exact(value)
    if x <= 0:
        raise ValueError("Positive logarithm argument required")
    if x == 1:
        return F(0), F(0)
    if x < 1:
        lo, hi = log_interval(1/x)
        return -hi, -lo
    powers = 0
    while x > 2:
        x /= 2
        powers += 1
    z = (x-1)/(x+1)
    terms = 80
    lo = 2*sum((z**(2*j+1)/(2*j+1) for j in range(terms)), F(0))
    hi = lo+2*z**(2*terms+1)/((2*terms+1)*(1-z*z))
    if powers:
        l2, h2 = log_interval(F(2))
        lo, hi = lo+powers*l2, hi+powers*h2
    scale = 1 << 90
    return F(lo.numerator*scale//lo.denominator, scale), F(-((-hi.numerator*scale)//hi.denominator), scale)


def sqrt_upper(value, bits=90):
    x = exact(value)
    if x < 0 or type(bits) is not int or not 16 <= bits <= 256:
        raise ValueError("Nonnegative argument and precision 16..256 required")
    scale = 1 << bits
    root = isqrt(x.numerator*scale*scale//x.denominator)
    return F(root if F(root*root, scale*scale) == x else root+1, scale)


def _order(m):
    if type(m) is not int or m < 0:
        raise ValueError("Nonnegative integer hierarchy order required")
    return 2*m+2


def category_order(m):
    """Binary sufficient categories, preserving endpoint order."""
    k = _order(m)
    return tuple(product((-1, 1), range(k+1), (-1, 1)))


def elementary_vertices(replicas, plus):
    """Normalized elementary products e_j for a binary central record."""
    if type(replicas) is not int or replicas < 1 or type(plus) is not int or not 0 <= plus <= replicas:
        raise ValueError("Invalid binary replica count")
    minus = replicas-plus
    return tuple(F(sum((-1)**r*comb(minus, r)*comb(plus, j-r)
                       for r in range(max(0, j-plus), min(j, minus)+1)), comb(replicas, j))
                 for j in range(replicas+1))


def elementary(values):
    """Normalized elementary products for exact bounded continuous readouts."""
    values = tuple(map(exact, values))
    if not values or any(abs(x) > 1 for x in values):
        raise ValueError("Nonempty readouts bounded in [-1,1] required")
    p = [F(1)]+[F(0)]*len(values)
    for seen, value in enumerate(values, 1):
        for j in range(seen, 0, -1):
            p[j] += value*p[j-1]
    return tuple(x/comb(len(values), j) for j, x in enumerate(p))


def _score_values(left, e, right, m, h, lam, nominal_gain=None):
    b = 1-lam*h
    values = {f"S{j}": (-left*right*e[2*j]+b*(left+right)*e[2*j+1]-b*b*e[2*j+2])/h**2
              for j in range(m+1)}
    values.update({f"U{j}": e[2*j] for j in range(1, max(m, 1)+1)})
    values["V"] = (e[2]-(left+right)*e[1]/2)/h
    if nominal_gain is not None:
        nominal_gain = exact(nominal_gain)
        if not 0 < nominal_gain <= 1:
            raise ValueError("Positive predeclared nominal gain in (0,1] required")
        projection = F(0)
        for j in range(m+1):
            projection += values[f"S{j}"]/nominal_gain**(2*j+2)
            values[f"R{j}"] = projection
    return values


def scores(category, m, h, lam, nominal_gain=None):
    """S_j retains cancellation before division by the common gain power."""
    h, lam = exact(h), exact(lam)
    if h <= 0:
        raise ValueError("Positive fixed lag required")
    k = _order(m)
    if not isinstance(category, (tuple, list)) or len(category) != 3:
        raise ValueError("Category must be (left sign, central plus count, right sign)")
    left, plus, right = category
    if type(left) is not int or type(right) is not int or left not in (-1, 1) or right not in (-1, 1):
        raise ValueError("Binary endpoint signs required")
    return _score_values(F(left), elementary_vertices(k, plus), F(right), m, h, lam, nominal_gain)


def continuous_scores(left, centers, right, m, h, lam, nominal_gain=None):
    """Same multilinear scores for rational continuous bounded records."""
    left, right, h, lam = map(exact, (left, right, h, lam))
    centers = tuple(map(exact, centers))
    if len(centers) != _order(m) or abs(left) > 1 or abs(right) > 1 or h <= 0:
        raise ValueError("Wrong replica count, range or lag")
    return _score_values(left, elementary(centers), right, m, h, lam, nominal_gain)


def score_ranges(m, h, lam, nominal_gain=None):
    """Exact scalar extrema on [-1,1]^(2m+4), achieved at binary vertices.

    Every score is multiaffine in the distinct readouts, so extrema are at
    vertices. Central exchangeability reduces the vertex enumeration to k+1
    plus counts; it is not an independence assumption on unconditioned data.
    """
    h, lam = exact(h), exact(lam)
    rows = tuple(scores(c, m, h, lam, nominal_gain) for c in category_order(m))
    return {key: (min(r[key] for r in rows), max(r[key] for r in rows)) for key in rows[0]}


def aggregate_ordered_counts(counts, m):
    """Convert lexicographic (-1,+1)^(2m+4) outcome counts to categories."""
    k = _order(m)
    counts = tuple(counts)
    if len(counts) != 2**(k+2) or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Nonnegative integer count for every ordered binary outcome required")
    result = dict.fromkeys(category_order(m), 0)
    for bits, n in zip(product((-1, 1), repeat=k+2), counts):
        result[(bits[0], sum(x == 1 for x in bits[1:-1]), bits[-1])] += n
    return result


def summarize_counts(counts, m, h, lam, nominal_gain=None):
    """Exact sums and sums of squares from fixed independent record counts."""
    h, lam = exact(h), exact(lam)
    nominal_gain = None if nominal_gain is None else exact(nominal_gain)
    ranges = score_ranges(m, h, lam, nominal_gain)
    allowed = set(category_order(m))
    if not isinstance(counts, dict) or any(c not in allowed for c in counts):
        raise ValueError("Count mapping must use supported binary categories")
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Counts must be nonnegative integers")
    n = sum(counts.values())
    if n < 2:
        raise ValueError("At least two independent records required")
    stats = {key: {"sum": F(0), "sum_squares": F(0), "range": ran} for key, ran in ranges.items()}
    for cat, count in counts.items():
        for key, value in scores(cat, m, h, lam, nominal_gain).items():
            stats[key]["sum"] += count*value
            stats[key]["sum_squares"] += count*value*value
    return {"records": n, "order": m, "h": h, "lambda": lam, "statistics": stats,
            "readouts_per_record": 2*m+4, "nominal_gain": nominal_gain}


def empirical_bernstein(records, total, sum_squares, bounds, alpha):
    """Two-sided Maurer--Pontil interval, including union over its two tails."""
    total, sum_squares, alpha = map(exact, (total, sum_squares, alpha))
    lo, hi = map(exact, bounds)
    if type(records) is not int or records < 2 or not 0 < alpha < 1 or lo > hi:
        raise ValueError("Invalid quota, confidence allocation or scalar range")
    mean, second = total/records, sum_squares/records
    if not lo <= mean <= hi or not mean*mean <= second <= (lo+hi)*mean-lo*hi:
        raise ValueError("Summary incompatible with the declared bounded observations")
    variance = F(records, records-1)*(second-mean*mean)
    ell = log_interval(4/alpha)[1]
    radius = sqrt_upper(2*variance*ell/records)+7*(hi-lo)*ell/(3*(records-1))
    return {"mean": mean, "variance": variance, "radius": radius,
            "interval": (max(lo, mean-radius), min(hi, mean+radius)), "alpha": alpha}


def confidence_statistics(summary, alpha, *, record_contract="independent_stationary_records",
                          replica_contract="conditionally_independent_common_gain",
                          selection="predeclared"):
    """Simultaneous intervals; correlated scores within one record are allowed."""
    if record_contract != "independent_stationary_records":
        raise ValueError("Dependent trials and data-selected stopping are unsupported")
    if replica_contract != "conditionally_independent_common_gain":
        raise ValueError("Correlated, invasive or unequal-gain replica channels are unsupported")
    if selection not in ("predeclared", "independent_pilot"):
        raise ValueError("Order, lag, lambda and nominal gain must be fixed independently of inference records")
    alpha = exact(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Invalid confidence budget")
    stats = summary["statistics"]
    nominal_gain = summary.get("nominal_gain")
    expected = score_ranges(summary["order"], summary["h"], summary["lambda"], nominal_gain)
    if stats.keys() != expected.keys() or any(tuple(stats[key]["range"]) != expected[key] for key in stats):
        raise ValueError("Statistics must match the fixed hierarchy and its exact ranges")
    event_count = len(stats)-(1 if nominal_gain is not None else 0)
    intervals = {key: empirical_bernstein(summary["records"], item["sum"], item["sum_squares"], item["range"], alpha/event_count)
                 for key, item in stats.items() if key != "R0"}
    if nominal_gain is not None:
        # R0 is an exact positive rescaling, so its event is already covered by
        # S0. Reusing that interval avoids charging an identical event twice.
        scale = 1/exact(nominal_gain)**2
        base = intervals["S0"]
        intervals["R0"] = {"mean": scale*base["mean"], "variance": scale*scale*base["variance"],
                           "radius": scale*base["radius"], "interval": tuple(scale*x for x in base["interval"]),
                           "alpha": F(0), "derived_from": "S0"}
    return {"records": summary["records"], "order": summary["order"], "h": summary["h"],
            "lambda": summary["lambda"], "statistics": intervals, "alpha": alpha,
            "record_contract": record_contract, "replica_contract": replica_contract, "selection": selection,
            "nominal_gain": nominal_gain, "confidence_event_count": event_count}


def binary_mean_interval(records, plus, alpha):
    """Fixed-quota two-sided Hoeffding interval for a [-1,+1] bit mean."""
    alpha = exact(alpha)
    if type(records) is not int or records <= 0 or type(plus) is not int or not 0 <= plus <= records or not 0 < alpha < 1:
        raise ValueError("Invalid calibration counts or failure allocation")
    mean = F(2*plus-records, records)
    radius = sqrt_upper(2*log_interval(2/alpha)[1]/records)
    return max(F(-1), mean-radius), min(F(1), mean+radius)


def calibrate_references(gain_records, gain_plus, step_records, step_plus,
                         step_time, width_prior, alpha, *, gain_prior=(0, 1),
                         reference_contract="independent_binary_constant_one_step_zero_to_one"):
    """Covered rectangle for gain kappa and causal-boxcar width tau.

    The constant reference is exactly 1. The other switches from 0 to 1 at
    calibrated time zero and is read at s, with 0<s<=prior tau_L. Its mean is
    kappa*s/tau. Reference bits have the same gain and width as science bits;
    quotas are fixed and trials independent within each batch. Independence
    between reference batches and science data is unnecessary for the union
    bound, provided science design was not selected using its own records.
    """
    if reference_contract != "independent_binary_constant_one_step_zero_to_one":
        raise ValueError("Unsupported reference waveform or record dependence")
    s, alpha = exact(step_time), exact(alpha)
    tlo, thi = map(exact, width_prior)
    klo, khi = map(exact, gain_prior)
    if not 0 < s <= tlo <= thi or not 0 <= klo <= khi <= 1 or not 0 < alpha < 1:
        raise ValueError("Positive physical width prior with s<=tau_L and gain prior in [0,1] required")
    gi = binary_mean_interval(gain_records, gain_plus, alpha/2)
    si = binary_mean_interval(step_records, step_plus, alpha/2)
    gain = max(klo, gi[0]), min(khi, gi[1])
    step = max(F(0), si[0]), min(F(1), si[1])
    if gain[0] > gain[1] or step[0] > step[1] or step[1] <= 0:
        return {"status": "incompatible_reference_intervals", "alpha": alpha}
    width = max(tlo, gain[0]*s/step[1]), min(thi, gain[1]*s/step[0] if step[0] > 0 else thi)
    if width[0] > width[1]:
        return {"status": "incompatible_reference_intervals", "alpha": alpha}
    return {"status": "covered_region" if gain[0] > 0 else "gain_not_separated_from_zero",
            "gain": gain, "width": width, "alpha": alpha, "gain_mean_interval": gi,
            "step_mean_interval": si, "step_time": s, "width_prior": (tlo, thi),
            "gain_prior": (klo, khi), "gain_records": gain_records, "step_records": step_records,
            "contract": reference_contract}


def _bounds(bounds):
    bounds = tuple(map(exact, bounds))
    if len(bounds) != 4 or any(x < 0 for x in bounds):
        raise ValueError("Nonnegative bounds (M1+,M1-,M2+,M2-) required")
    return bounds


def monomial_blur(cluster_counts, tau, generator_bounds):
    """Uniform causal-boxcar monomial error for clusters at distinct h-spaced times."""
    tau = exact(tau)
    mp, mm, rp, rm = _bounds(generator_bounds)
    cluster_counts = tuple(cluster_counts)
    if any(type(x) is not int or x < 0 for x in cluster_counts) or tau < 0:
        raise ValueError("Nonnegative integer cluster degrees and width required")
    p = tuple(x for x in cluster_counts if x)
    d = sum(p)
    if d < 2:
        return F(0)
    if d == 2:
        return min(mp, mm)*tau/3 if len(p) == 1 else min(rp, rm)*tau*tau/12
    total, previous = F(0), 0
    for i, size in enumerate(p):
        for offset in range(1, size):
            k = previous+offset
            total += min(k*k*mm, (d-k)**2*mp)*tau/(size+1)
        previous += size
        if i+1 < len(p):
            following = p[i+1]
            gap = 1-F(1, size+1)-F(1, following+1)+F(2*factorial(size)*factorial(following), factorial(size+following+1))
            total += min(previous**2*mm, (d-previous)**2*mp)*tau*gap
    return total


def bias_bounds(m, h, lam, tau_upper, generator_bounds):
    """Finite-lag and exposure biases under the declared structural theorem."""
    _order(m)
    h, lam, tau = map(exact, (h, lam, tau_upper))
    mp, mm, rp, rm = _bounds(generator_bounds)
    if h <= 0 or not 0 <= tau <= h:
        raise ValueError("A common nonoverlapping exposure width 0<=tau<=h is required")
    bg = h*(rp+rm)/4
    bk = h*(mm*rp+mp*rm)/2
    beta = 2*abs(lam)*bg+bk
    b = 1-lam*h
    delta = F(0)
    for j in range(m+1):
        delta += (monomial_blur((1, 2*j, 1), tau, (mp, mm, rp, rm))
                  +abs(b)*(monomial_blur((1, 2*j+1), tau, (mp, mm, rp, rm))
                           +monomial_blur((2*j+1, 1), tau, (mp, mm, rp, rm)))
                  +b*b*monomial_blur((2*j+2,), tau, (mp, mm, rp, rm)))/h**2
        delta += beta*monomial_blur((2*j,), tau, (mp, mm, rp, rm))
    delta0, deltah = min(mp, mm)*tau/3, min(rp, rm)*tau*tau/12
    return {"g_bias": bg, "K_bias": bk, "beta": beta, "delta_N": delta,
            "delta_0": delta0, "delta_h": deltah, "delta_cal": (delta0+deltah)/h,
            "M2_star": min(rp, rm)}


def normalized_interval(interval, power, gain):
    """Outward division by a positive shared gain; sign-aware endpoint extrema."""
    lo, hi = map(exact, interval)
    gl, gu = map(exact, gain)
    if lo > hi or not 0 < gl <= gu <= 1 or type(power) is not int or power < 0:
        raise ValueError("Ordered interval, positive gain region and integer degree required")
    return min(lo/gl**power, lo/gu**power), max(hi/gl**power, hi/gu**power)


def projection_lower(projection_interval, score_intervals, nominal_gain, gain):
    """Covered common-gain correction to a fixed nominal linear projection.

    Each residual coefficient and each raw score mean is enclosed separately;
    the minimum of their product is a four-corner calculation. The sum is a
    conservative shared-gain lower bound, not a coupled polynomial optimum.
    """
    nominal_gain = exact(nominal_gain)
    gl, gu = map(exact, gain)
    if not 0 < nominal_gain <= 1 or not 0 < gl <= gu <= 1:
        raise ValueError("Positive nominal gain and gain interval required")
    plo, phi = map(exact, projection_interval)
    if plo > phi or not score_intervals:
        raise ValueError("Ordered projection and nonempty raw score intervals required")
    lower = plo
    for j, interval in enumerate(score_intervals):
        lo, hi = map(exact, interval)
        if lo > hi:
            raise ValueError("Ordered raw score intervals required")
        power = 2*j+2
        dl, du = 1/gu**power-1/nominal_gain**power, 1/gl**power-1/nominal_gain**power
        lower += min(dl*lo, dl*hi, du*lo, du*hi)
    return lower


def certify(confidence, calibration, generator_bounds, *,
            geometry_contract="Gamma_f_equals_gamma_times_one_minus_f_squared",
            reversal_contract="ordinary_stationary_compact_diffusion"):
    """Joint lower entropy certificate; gamma is an unknown positive constant.

    Requires smooth |f|<=1 and Gamma f=gamma*(1-f^2) pointwise, common causal
    boxcar, conditionally independent common-gain bounded replicas, independent
    stationary records, supplied uniform generator bounds, and ordinary full
    hidden-state reversal. Neither these physical premises nor gain reference
    correctness can be established by this function. Gain minimization is
    termwise conservative; no claim of solving the coupled nuisance optimum.
    """
    if geometry_contract != "Gamma_f_equals_gamma_times_one_minus_f_squared" or reversal_contract != "ordinary_stationary_compact_diffusion":
        raise ValueError("Unsupported local geometry or time-reversal contract")
    if confidence.get("record_contract") != "independent_stationary_records" or confidence.get("replica_contract") != "conditionally_independent_common_gain" or confidence.get("selection") not in ("predeclared", "independent_pilot"):
        raise ValueError("A supported fixed-design independent-record confidence contract is required")
    allocations = exact(confidence["alpha"]), exact(calibration["alpha"])
    alpha = sum(allocations, F(0))
    if any(not 0 < x < 1 for x in allocations) or not 0 < alpha < 1:
        raise ValueError("Combined confidence failure must lie in (0,1)")
    if calibration["status"] != "covered_region":
        return {"status": "calibration_abstention", "entropy_lower": F(0), "alpha": alpha}
    if calibration.get("contract") != "independent_binary_constant_one_step_zero_to_one":
        raise ValueError("Supported constant and step reference calibration contract required")
    m, h, lam = confidence["order"], exact(confidence["h"]), exact(confidence["lambda"])
    gain = tuple(map(exact, calibration["gain"]))
    tl, tau = tuple(map(exact, calibration["width"]))
    if not 0 < tl <= tau:
        raise ValueError("Positive ordered exposure calibration interval required")
    bias = bias_bounds(m, h, lam, tau, generator_bounds)
    stats = confidence["statistics"]
    normalized = {}
    for j in range(m+1):
        key = f"S{j}"
        normalized[key] = normalized_interval(stats[key]["interval"], 2*j+2, gain)
    normalized["U0"] = F(1), F(1)
    for j in range(1, max(m, 1)+1):
        key = f"U{j}"
        lo, hi = stats[key]["interval"]
        lo, hi = max(F(0), exact(lo)), min(F(1), exact(hi))
        if lo > hi:
            return {"status": "incompatible_moment_region", "entropy_lower": F(0), "alpha": alpha}
        normalized[key] = normalized_interval((lo, hi), 2*j, gain)
    normalized["V"] = normalized_interval(stats["V"]["interval"], 2, gain)
    a = 1-normalized["U1"][1]-bias["delta_0"]
    upper = normalized["V"][1]+bias["delta_cal"]+bias["M2_star"]*h/2
    nominal_gain = confidence.get("nominal_gain")
    prefixes = []
    for order in range(m+1):
        order_bias = bias_bounds(order, h, lam, tau, generator_bounds)
        penalty = order_bias["beta"]*sum((normalized[f"U{j}"][1] for j in range(order+1)), F(0))+order_bias["delta_N"]
        termwise = sum((normalized[f"S{j}"][0] for j in range(order+1)), F(0))
        projected = None
        if nominal_gain is not None:
            projected = projection_lower(stats[f"R{order}"]["interval"],
                                         [stats[f"S{j}"]["interval"] for j in range(order+1)], nominal_gain, gain)
        contrast = max(termwise, projected) if projected is not None else termwise
        p = contrast-penalty
        entropy = max(F(0), p)*max(F(0), a)/upper if upper > 0 else F(0)
        prefixes.append({"order": order, "P_lower": p, "entropy_lower": entropy,
                         "termwise_contrast_lower": termwise, "projected_contrast_lower": projected,
                         "penalty_upper": penalty, "biases": order_bias})
    best = max(prefixes, key=lambda row: row["P_lower"])
    p, entropy = best["P_lower"], best["entropy_lower"]
    return {"status": "positive_certificate" if entropy else "no_positive_certificate",
            "entropy_lower": entropy, "P_lower": p, "A_lower": a, "H_upper": upper,
            "gamma_upper": upper/a if a > 0 and upper > 0 else None,
            "normalized_intervals": normalized, "biases": best["biases"], "alpha": alpha,
            "gain_region": gain, "width_region": calibration["width"],
            "geometry_contract": geometry_contract, "reversal_contract": reversal_contract,
            "nuisance_optimization": "termwise extrema and optional fixed-projection correction",
            "order_selected": best["order"], "order_results": prefixes, "nominal_gain": nominal_gain}


def jsonable(value):
    """Stable exact-rational JSON representation for audit reports."""
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value
