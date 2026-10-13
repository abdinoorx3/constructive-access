"""An exact-rational, vanishing-radius consistency design for cosine observations.

This is a deterministic design calculation, not a simulator or a substitute
for physical calibration. It uses the finite_resolution primitives only to
enclose fixed constants. All m-dependent divisions are exact, avoiding the
fixed absolute grid in the existing final-square-root certificate.
"""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path

try:
    from .finite_resolution import exact, log_interval, sqrt_upper
except ImportError:
    from finite_resolution import exact, log_interval, sqrt_upper


def design(m, gamma=F(1), omega_abs_upper=F(1), reference_time=F(1),
           alpha=F(1, 20)):
    """Freeze one independent stationary-triple experiment.

    gamma>0 and |omega|<=omega_abs_upper are calibrated in physical clock
    units. The observed process is the stationary cosine-circle law, readings
    are noiseless, q(y)=-gamma*y and a(y)=gamma*(1-y*y). Retain a>=gamma/m².
    alpha is allocated equally to two Bernstein tails. The returned radius
    is calculated from separately proved moment constants, not from an
    unverified comparison with a rounded per-m finite_resolution bound.
    """
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    gamma, omega, h0, alpha = map(exact, (
        gamma, omega_abs_upper, reference_time, alpha))
    if gamma <= 0 or omega < 0 or h0 <= 0 or not 0 < alpha < 1:
        raise ValueError("Positive gamma/time, nonnegative drift bound, and alpha in (0,1) required")
    M = gamma+omega
    R = gamma*gamma+omega*omega
    C = R*(2*gamma+omega)
    root = sqrt_upper(2*gamma/h0)
    H = ((2*gamma+omega+root)**2+C*h0)/gamma
    B0 = (1/(h0*h0)+(2/h0+gamma)**2+C*h0)/gamma
    ell = log_interval(2/alpha)[1]
    root_ell = sqrt_upper(2*ell)
    first = H*root_ell
    fourth = 2*B0*ell/3
    radius = first/m+fourth/(m**4)
    h = h0/F(m**3)
    floor = gamma/F(m*m)
    population_gap = (omega*omega/gamma+2*C*h0/gamma)/m
    lower = -((2/h+gamma)**2+C*h)/floor
    upper = 1/(h*h*floor)
    return {
        "m": m, "gamma": gamma, "omega_abs_upper": omega,
        "reference_time": h0, "alpha": alpha,
        "records": m**12, "h": h, "envelope_floor": floor,
        "generator_bounds": (M, M, R, R), "q_abs_max": gamma,
        "global_gamma_upper": gamma,
        "sqrt_diffusion_constant_upper": root,
        "second_moment_coefficient_upper": H,
        "range_coefficient_upper": B0,
        "logarithm_upper": ell, "sqrt_logarithm_upper": root_ell,
        "radius_first_order_coefficient": first,
        "radius_fourth_order_coefficient": fourth,
        "sampling_radius_upper": radius,
        "population_gap_upper": population_gap,
        "joint_accuracy_gap_upper": population_gap+2*radius,
        "corrected_score_bounds": (lower, upper),
        "one_sided_failure_upper": alpha/2,
        "two_sided_failure_upper": alpha,
        "independent_preparations": m**12,
        "triple_detector_readouts": 3*m**12,
        "declared_evolution_time": 2*h0*m**9,
        "detector_contract": "Exact noiseless state-function readings, or a separately proved uniform error in the already defined fixed-score mean.",
        "physical_contract": "Stationary ordinary-reversal cosine-circle law; calibrated gamma, drift envelope and clock; independent stationary triples; fixed q, pointwise geometry and selector.",
        "omitted_resources": "Stationary preparation, resetting, sensor bandwidth, geometric calibration, numerical score precision and energy costs are additional.",
    }


def certify_mean(score_mean, plan, numerical_mean_radius=0):
    """Certify a summary of the specified ideal fixed scores.

    numerical_mean_radius is a proved deterministic bound on the difference
    between the supplied summary and the empirical mean of the ideal scores.
    It must include any actual changes in selection/weight/drift; an error
    bound on raw readings alone does not establish this premise.
    """
    rebuilt = design(**{key: plan[key] for key in (
        "m", "gamma", "omega_abs_upper", "reference_time", "alpha")})
    if rebuilt != plan:
        raise ValueError("Plan differs from its exact derived design")
    mean, error = map(exact, (score_mean, numerical_mean_radius))
    if error < 0:
        raise ValueError("Numerical mean radius must be nonnegative")
    lower, upper = plan["corrected_score_bounds"]
    if mean+error < lower or mean-error > upper:
        raise ValueError("Mean enclosure is outside the calibrated score range")
    signed = mean-plan["sampling_radius_upper"]-error
    return {
        "status": "positive_certificate" if signed > 0 else "no_positive_certificate",
        "entropy_production_rate_lower": max(F(0), signed),
        "signed_rate_lower": signed,
        "score_mean": mean,
        "numerical_mean_radius": error,
        "sampling_radius_upper": plan["sampling_radius_upper"],
        "confidence_failure_upper": plan["one_sided_failure_upper"],
        "joint_validity_and_accuracy_failure_upper": plan["two_sided_failure_upper"],
        "joint_accuracy_gap_upper": plan["joint_accuracy_gap_upper"]+2*error,
        "records": plan["records"],
        "physical_contract": plan["physical_contract"],
        "numerical_contract": "The supplied mean encloses the empirical mean of the fixed ideal scores; numerical_mean_radius must tend to zero for consistency.",
    }


def _json(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def deterministic_examples():
    """Return design bounds only: no fabricated records or detection claims."""
    levels = (2, 4, 8, 16, 32, 64, 128, 1024, 2**120)
    plans = [design(m) for m in levels]
    floor = sqrt_upper(F(1, 2**400))
    assert floor == F(1, 2**90)
    assert plans[-1]["sampling_radius_upper"] < floor
    return {
        "kind": "deterministic_design_calculation_no_simulated_or_measured_records",
        "parameters": {"gamma": "1", "omega_abs_upper": "1", "reference_time": "1", "two_sided_alpha": "1/20"},
        "plans": plans,
        "fixed_precision_diagnostic": {
            "tiny_argument": F(1, 2**400),
            "existing_sqrt_upper_result": floor,
            "consistent_radius_at_largest_m": plans[-1]["sampling_radius_upper"],
            "consistent_radius_below_fixed_grid": True,
        },
        "claim": "Covered finite experiments approach the circle minimum under the stated calibration contract; displayed plans are bounds and costs, not experimental successes.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(_json(deterministic_examples()), indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
