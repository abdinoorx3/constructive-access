"""Covered finite-resolution dissipation inference for ordinary diffusions.

The input contract is physical calibration, not something these statistics
verify.  All certifying arithmetic uses rational outward enclosures.  Scores
use a fixed, state-measurable selection/weight/drift rule and independent
stationary triples.  Same-data tuning and weights chosen from noisy central
readings are not covered.  No simulation or physical measurement lives here.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt


def exact(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, F)):
        raise TypeError("Use an integer, Fraction, or rational string")
    return F(value)


@lru_cache(maxsize=128)
def log_interval(value):
    """Rational logarithm enclosure; atanh series plus dyadic rounding."""
    x = exact(value)
    if x <= 0:
        raise ValueError("The logarithm argument must be positive")
    if x == 1:
        return F(0), F(0)
    if x < 1:
        low, high = log_interval(1/x)
        return -high, -low
    powers = 0
    while x > 2:
        x /= 2
        powers += 1
    z = (x-1)/(x+1)
    terms = 80
    low = 2*sum((z**(2*j+1)/(2*j+1) for j in range(terms)), F(0))
    high = low + 2*z**(2*terms+1)/((2*terms+1)*(1-z*z))
    if powers:
        l2, h2 = log_interval(F(2))
        low += powers*l2
        high += powers*h2
    scale = 1 << 90
    return (F(low.numerator*scale//low.denominator, scale),
            F(-((-high.numerator*scale)//high.denominator), scale))


def sqrt_upper(value, bits=90):
    x = exact(value)
    if x < 0 or type(bits) is not int or not 16 <= bits <= 256:
        raise ValueError("Nonnegative argument and precision 16..256 required")
    scale = 1 << bits
    root = isqrt(x.numerator*scale*scale//x.denominator)
    if F(root*root, scale*scale) == x:
        return F(root, scale)
    return F(root+1, scale)


def semigroup_bias(h, generator_bounds):
    """Uniform biases of g_h and K_h using [M1+, M1-, M2+, M2-].

    M1+ >= ||Lf||_infty, M1- >= ||Ldagger f||_infty,
    M2+ >= ||L^2 f||_infty, M2- >= ||(Ldagger)^2 f||_infty.
    The contract includes the relevant generator domains and sup-norm
    contraction semigroups.  Bounds also hold after state-measurable conditioning.
    """
    h = exact(h)
    bounds = tuple(map(exact, generator_bounds))
    if h <= 0 or len(bounds) != 4 or any(x < 0 for x in bounds):
        raise ValueError("Positive lag and four nonnegative generator bounds required")
    mp, mm, rp, rm = bounds
    return {"g_bias_upper": h*(rp+rm)/4,
            "K_bias_upper": h*(mm*rp+mp*rm)/2,
            "generator_bounds": bounds, "h": h}


def make_protocol(h, q_abs_max, envelope_floor, generator_bounds,
                  readout_bounds=(-1, 1), global_gamma_upper=None,
                  selection="predeclared",
                  detector_contract="noiseless_state_function",
                  coordinate_readouts_per_record=0):
    """Freeze a bounded weighted score and its physical/calibration contract.

    For retained central states U, the functions q(U), a(U) obey |q|<=Q,
    a>=envelope_floor>0 and Gamma_f(X0)<=a(U) POINTWISE.  Exclusion is a
    predeclared function of U.  The floor bounds the envelope, not Gamma.
    U and the rules must be measurable functions of the complete central state.
    With noisy readouts U must be separately available without using their noise.
    Noisy detector values must be conditionally independent given the complete
    hidden triple, or satisfy the equivalent zero cross-covariance assumptions,
    and conditionally unbiased with mean f(X_t).
    global_gamma_upper bounds Gamma_f at ALL hidden states when supplied; it
    enables the noiseless known-variance Bernstein option.
    """
    Q, floor = map(exact, (q_abs_max, envelope_floor))
    sensor = tuple(map(exact, readout_bounds))
    if Q < 0 or floor <= 0 or len(sensor) != 2 or sensor[0] >= sensor[1]:
        raise ValueError("Nonnegative Q, positive envelope floor and ordered sensor range required")
    if selection not in ("predeclared", "independent_pilot"):
        raise ValueError("Select the complete rule before its test records")
    if detector_contract not in (
            "noiseless_state_function", "conditionally_unbiased_noise_external_center_rule"):
        raise ValueError("Detector contract does not cover noisy-center selection or sensor memory")
    if type(coordinate_readouts_per_record) is not int or coordinate_readouts_per_record < 0:
        raise ValueError("Declare a nonnegative integer additional coordinate readout count")
    bias = semigroup_bias(h, generator_bounds)
    h = bias["h"]
    beta = 2*Q*bias["g_bias_upper"] + bias["K_bias_upper"]
    span = sensor[1]-sensor[0]
    lower = -((span/h+Q)**2 + beta)/floor
    upper = span*span/(4*h*h*floor)
    result = dict(bias, q_abs_max=Q, envelope_floor=floor,
                  readout_bounds=sensor, maximum_weighted_bias=beta/floor,
                  corrected_score_bounds=(lower, upper), selection=selection,
                  detector_contract=detector_contract,
                  coordinate_readouts_per_record=coordinate_readouts_per_record,
                  global_gamma_upper=None, second_moment_upper=None)
    if global_gamma_upper is not None:
        G = exact(global_gamma_upper)
        if G < 0:
            raise ValueError("A nonnegative global Gamma upper bound is required")
        result["global_gamma_upper"] = G
        if detector_contract == "noiseless_state_function":
            mp, mm, _, _ = bias["generator_bounds"]
            noise = sqrt_upper(2*G/h)
            raw_l2 = (Q+mp+noise)*(Q+mm+noise)/floor
            # The second bound follows directly from boundedness and is useful
            # if global Gamma or generator bounds are deliberately loose.
            result["second_moment_upper"] = min(
                (raw_l2+beta/floor)**2, max(lower*lower, upper*upper))
    return result


def validate_protocol(protocol):
    """Reject a modified/tampered derived protocol dictionary."""
    rebuilt = make_protocol(**{key: protocol[key] for key in (
        "h", "q_abs_max", "envelope_floor", "generator_bounds", "readout_bounds",
        "global_gamma_upper", "selection", "detector_contract",
        "coordinate_readouts_per_record")})
    if rebuilt != protocol:
        raise ValueError("Protocol does not match its exact derived bounds")
    return rebuilt


def corrected_score(readings, q, envelope, protocol, retained=True):
    """One exact triple score; exclusion returns zero but still costs a record.

    readings are (Y_-h, Y_0, Y_h), or their bounded unbiased detector values.
    q and envelope are outputs of the previously frozen central-state rule.
    This function checks numerical bounds, not the rule's measurability or the
    physical pointwise-envelope premise.
    """
    if type(retained) is not bool:
        raise TypeError("retained must be Boolean")
    values = tuple(map(exact, readings))
    if len(values) != 3 or any(not protocol["readout_bounds"][0] <= y <= protocol["readout_bounds"][1]
                               for y in values):
        raise ValueError("Three readings within the declared sensor range required")
    if not retained:
        return F(0)
    q, a = map(exact, (q, envelope))
    if abs(q) > protocol["q_abs_max"] or a < protocol["envelope_floor"]:
        raise ValueError("Central rule exceeds its declared bounds")
    past, center, future = values
    h = protocol["h"]
    raw = -(q-(past-center)/h)*(q-(future-center)/h)
    correction = 2*abs(q)*protocol["g_bias_upper"] + protocol["K_bias_upper"]
    return (raw-correction)/a


def certify_summary(records, score_mean, protocol, alpha=F(1, 20),
                    mean_error=0, conditional_mean_radius=0,
                    external_failure=0, concentration="hoeffding",
                    score_second_moment=None, quantization_radius=0,
                    record_contract="independent_stationary_triples"):
    """Lower confidence bound on sigma from a fixed quota of triple scores.

    score_mean summarizes CORRECTED scores, including zeros for excluded
    centers.  quantization_radius bounds each score's uniform rounding error;
    score_second_moment must be the exact empirical second moment of those same
    rounded scores.  The range is expanded and the final lower bound loses
    quantization_radius.  Empirical Bernstein is Maurer--Pontil (2009), Theorem
    4, applied to the reflected affine-rescaled score, not a plug-in substitution
    into the known-variance Bernstein inequality.  mean_error encloses numerical summary error; it is not a sensor
    calibration.  conditional_mean_radius concerns score units.  A declared
    bounded-conditional-mean contract permits Hoeffding with that penalty;
    Bernstein requires independent stationary triples and the calibrated
    noiseless global-Gamma second-moment bound.
    """
    p = validate_protocol(protocol)
    if type(records) is not int or records <= 0:
        raise ValueError("Use a positive fixed integer quota")
    mean, alpha, numeric, drift, eta, quant = map(exact, (
        score_mean, alpha, mean_error, conditional_mean_radius, external_failure,
        quantization_radius))
    if not 0 < alpha < 1 or not 0 <= eta < 1 or alpha+eta >= 1 or min(numeric, drift, quant) < 0:
        raise ValueError("Invalid confidence or nonnegative error allocation")
    if record_contract not in ("independent_stationary_triples", "bounded_conditional_mean_error"):
        raise ValueError("Independent records or a bounded conditional-mean contract required")
    if concentration not in ("hoeffding", "bernstein", "empirical_bernstein"):
        raise ValueError("Use a predeclared supported concentration rule")
    lo, hi = p["corrected_score_bounds"]
    lo -= quant
    hi += quant
    if mean+numeric < lo or mean-numeric > hi:
        raise ValueError("Empirical mean lies outside its score range")
    width = hi-lo
    logarithm = log_interval(1/alpha)[1]
    if concentration == "hoeffding":
        radius = width*sqrt_upper(logarithm/(2*records))
    elif concentration == "bernstein":
        if record_contract != "independent_stationary_triples" or drift:
            raise ValueError("Bernstein requires independent stationary triples")
        second = p["second_moment_upper"]
        if second is None:
            raise ValueError("Bernstein needs noiseless readouts and a global Gamma bound")
        second = (sqrt_upper(second)+quant)**2
        linear = width*logarithm/(3*records)
        radius = linear+sqrt_upper(2*second*logarithm/records+linear*linear)
    else:
        if records < 2 or record_contract != "independent_stationary_triples" or drift or numeric:
            raise ValueError("Empirical Bernstein requires n>=2, iid triples and exact quantized summaries")
        if score_second_moment is None:
            raise ValueError("An exact second moment of the quantized scores is required")
        second = exact(score_second_moment)
        if not mean*mean <= second <= (lo+hi)*mean-lo*hi:
            raise ValueError("Second moment is inconsistent with its bounded empirical mean")
        variance = F(records, records-1)*(second-mean*mean)
        logarithm = log_interval(2/alpha)[1]
        radius = (sqrt_upper(2*variance*logarithm/records)
                  + 7*width*logarithm/(3*(records-1)))
    signed = mean-radius-numeric-drift-quant
    rate = max(F(0), signed)
    return {"status": "positive_certificate" if rate else "no_positive_certificate",
            "records": records, "score_mean": mean,
            "corrected_score_bounds": (lo, hi),
            "concentration": concentration, "sampling_radius_upper": radius,
            "numerical_mean_penalty": numeric, "conditional_mean_penalty": drift,
            "uniform_score_quantization_penalty": quant,
            "empirical_variance": variance if concentration == "empirical_bernstein" else None,
            "signed_rate_lower": signed, "entropy_production_rate_lower": rate,
            "confidence_failure_upper": alpha+eta,
            "external_failure_upper": eta, "selection": p["selection"],
            "record_contract": record_contract,
            "detector_contract": p["detector_contract"],
            "independent_preparations": records,
            "triple_detector_readouts": 3*records,
            "coordinate_detector_readouts": p["coordinate_readouts_per_record"]*records,
            "declared_evolution_time": 2*p["h"]*records,
            "resources_not_included": "Stationary preparation, resets, calibrated geometry, clock, sensor exposure, energy and independent pilot costs must be reported separately; they are not zero.",
            "physical_contract": "Stationary smooth ordinary-reversal hidden diffusion; bounded generator derivatives; state-measurable frozen weights and pointwise Gamma upper envelope; calibrated time. Sigma is entropy in nats per time, not uncalibrated heat.",
            "protocol": p}


def certify_scores(scores, protocol, **kwargs):
    p = validate_protocol(protocol)
    values = tuple(map(exact, scores))
    low, high = p["corrected_score_bounds"]
    quant = exact(kwargs.get("quantization_radius", 0))
    low -= quant
    high += quant
    if not values or any(not low <= x <= high for x in values):
        raise ValueError("A nonempty score list within its declared bounds is required")
    if kwargs.get("concentration") == "empirical_bernstein":
        kwargs.setdefault("score_second_moment", sum((x*x for x in values), F(0))/len(values))
    return certify_summary(len(values), sum(values, F(0))/len(values), p, **kwargs)


def certify_totals(records, score_total, score_square_total, protocol, **kwargs):
    """Certify rational aggregate sums, avoiding per-record Fraction objects.

    For integer quantizer codes k_i at denominator d, supply total=sum k_i/d
    and square_total=sum k_i^2/d^2.  Use the exact uniform quantization radius
    1/(2d) only if rounding to nearest is indeed applied to the true score;
    additional floating numerical error also needs a justified uniform radius.
    """
    if type(records) is not int or records <= 0:
        raise ValueError("A positive integer record quota is required")
    total, squares = map(exact, (score_total, score_square_total))
    kwargs.setdefault("score_second_moment", squares/records)
    return certify_summary(records, total/records, protocol, **kwargs)


def bounded_readout_score_error(readout_error, protocol):
    """Uniform score perturbation from bounded errors in all three readings.

    Both ideal and stored readings lie in the same declared interval and differ
    by at most delta.  The IDEAL score uses exactly the same retained flag, q
    and envelope as the approximate score.  In particular, a rule defined via
    deterministic rounding of the true center is a legitimate state rule; a
    rule fitted to the same data, or an uncalibrated noisy-center envelope, is
    not justified here.  The pointwise envelope must cover the true f geometry.
    Add output-rounding/numerical radii before passing quantization_radius.
    """
    p = validate_protocol(protocol)
    delta = exact(readout_error)
    if delta < 0:
        raise ValueError("A nonnegative uniform reading-error calibration is required")
    span = p["readout_bounds"][1]-p["readout_bounds"][0]
    return 4*(span/p["h"]+p["q_abs_max"])*delta/(p["h"]*p["envelope_floor"])
