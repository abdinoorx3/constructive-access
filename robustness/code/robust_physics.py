#!/usr/bin/env python3
"""Exact conditional robustness certificates for finite-pulse CTMC response.

No routine certifies that a physical apparatus satisfies its input bounds.
Public arithmetic accepts exact Fraction inputs only. Numerical illustrations
are confined to make_results(). All generator conventions are row conventions.
"""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import json


def rational(x: object, name: str = "value") -> F:
    if type(x) is not F:
        raise TypeError(f"{name} must be fractions.Fraction")
    return x


def nonnegative(x: F, name: str) -> F:
    rational(x, name)
    if x < 0:
        raise ValueError(f"{name} must be nonnegative")
    return x


def probability(x: F, name: str) -> F:
    nonnegative(x, name)
    if x > 1:
        raise ValueError(f"{name} must be at most one")
    return x


def interval(x: tuple[F, F], name: str) -> tuple[F, F]:
    if type(x) is not tuple or len(x) != 2:
        raise TypeError(f"{name} must be a pair of Fractions")
    low, high = x
    rational(low, name); rational(high, name)
    if low > high:
        raise ValueError(f"{name} is reversed")
    return low, high


def exp_negative_bounds(x: F, terms: int = 64) -> tuple[F, F]:
    """Rational enclosure of exp(-x) using positive Taylor and a tail bound."""
    nonnegative(x, "exponent")
    if type(terms) is not int or terms < 0 or x >= terms + 2:
        raise ValueError("Require integer terms >= 0 and x < terms + 2")
    term = partial = F(1)
    for k in range(1, terms + 1):
        term *= x / k
        partial += term
    remainder = term * x / (terms + 1) / (1 - x / (terms + 2))
    return 1 / (partial + remainder), 1 / partial


def one_minus_exp_upper(x: F) -> F:
    """Upper bound on 1-exp(-x), tightened also by x."""
    nonnegative(x, "exponent")
    terms = max(64, 2 * (x.numerator // x.denominator + 1))
    return min(F(1), x, 1 - exp_negative_bounds(x, terms)[0])


def check_generator(q: tuple[tuple[F, ...], ...]) -> int:
    if type(q) is not tuple or not q:
        raise TypeError("Generator must be a nonempty tuple of tuple rows")
    n = len(q)
    for i, row in enumerate(q):
        if type(row) is not tuple or len(row) != n:
            raise ValueError("Generator must be square")
        for j, value in enumerate(row):
            rational(value, "generator entry")
            if i != j and value < 0:
                raise ValueError("Negative off-diagonal generator entry")
        if sum(row) != 0:
            raise ValueError("Generator row does not sum to zero")
    return n


def generator_coupling_rate(q: tuple[tuple[F, ...], ...],
                            r: tuple[tuple[F, ...], ...]) -> F:
    """max row TV(Q-R), the maximal same-state separation hazard.

    Couple equal jumps at their minimum rates. Couple as many unequal residual
    jumps simultaneously as possible; the remaining jumps occur unilaterally.
    This gives hazard max(sum positive offdiagonal differences,
                           sum negative offdiagonal differences).
    """
    n = check_generator(q)
    if check_generator(r) != n:
        raise ValueError("Generator dimensions differ")
    return max(sum(abs(q[i][j] - r[i][j]) for j in range(n)) / 2
               for i in range(n))


def matched_preparation_bias(eta: F, duration: F, suppression: F,
                             forward_rate: F, reverse_rate: F) -> F:
    """Response bias from a COMMON preparation within TV eta of stationarity.

    suppression = 1-r, not the retained fraction r. The matched native and
    controlled arms must have the same preparation distribution. A common
    stochastic postprocessing channel preserves this bound.
    """
    probability(eta, "preparation TV"); nonnegative(duration, "duration")
    probability(suppression, "suppression")
    nonnegative(forward_rate, "forward rate"); nonnegative(reverse_rate, "reverse rate")
    kappa = suppression * max(forward_rate, reverse_rate)
    return min(2 * eta,
               eta * suppression * (forward_rate + reverse_rate) * duration,
               2 * eta * one_minus_exp_upper(kappa * duration))


def delay_oscillation_bound(delay: F, endpoint_exit_cap: F,
                            endpoint_inflow_cap: F) -> F:
    """Bound osc((exp(dQ)-I)1_j) by endpoint exit and maximal entry hazards."""
    nonnegative(delay, "delay")
    nonnegative(endpoint_exit_cap, "endpoint exit cap")
    nonnegative(endpoint_inflow_cap, "endpoint inflow cap")
    return (one_minus_exp_upper(delay * endpoint_exit_cap) +
            one_minus_exp_upper(delay * endpoint_inflow_cap))


def shared_generator_drift_bias(duration: F, suppression: F, forward_rate: F,
                                reverse_rate: F, shared_error_rate: F) -> F:
    """O(t^2) response bound when the SAME additive generator error hits both arms.

    Compare Q+E versus Qr+E with Q versus Qr, for a common arbitrary initial
    distribution and a common terminal effect in [0,1]. All four matrices must
    be valid generators and max row TV(E)<=shared_error_rate. The rate E must
    be identical between arms. Common TERMINAL postprocessing is allowed;
    separately changing the reference readout-delay semigroup is not included.
    """
    nonnegative(duration, "duration"); probability(suppression, "suppression")
    nonnegative(forward_rate, "forward rate"); nonnegative(reverse_rate, "reverse rate")
    nonnegative(shared_error_rate, "shared error rate")
    if not duration or not shared_error_rate:
        return F(0)
    x = shared_error_rate * duration
    terms = max(64, 2 * (x.numerator // x.denominator + 1))
    integral_upper = min(duration, shared_error_rate * duration**2 / 2,
        duration - (1-exp_negative_bounds(x, terms)[1])/shared_error_rate)
    return suppression * (forward_rate + reverse_rate + 2*max(forward_rate,reverse_rate)) * integral_upper


def delayed_kernel_interval(duration: tuple[F, F], pulse_exit_cap: F,
                             delay_oscillation: F = F(0)) -> tuple[F, F]:
    """Kernel interval valid also for independent clock jitter in duration.

    The interval can have a nonpositive lower endpoint. In that case this
    certificate does not protect the response sign. The upper bound is t_hi,
    since each propagated endpoint indicator difference is at most one.
    """
    low, high = interval(duration, "duration")
    if low <= 0:
        raise ValueError("Pulse durations must be strictly positive")
    nonnegative(pulse_exit_cap, "pulse exit cap")
    nonnegative(delay_oscillation, "delay oscillation")
    lower_candidates = []
    for t in (low, high):
        x = pulse_exit_cap * t
        terms = max(64, 2 * (x.numerator // x.denominator + 1))
        lower_candidates.append(t * (exp_negative_bounds(x, terms)[0] - delay_oscillation))
    # t(exp(-Lambda*t)-omega) has at most one interior turning point, a maximum.
    return max(-high, min(lower_candidates)), high


def robust_current_interval(response: tuple[F, F], *,
                            duration: tuple[F, F], retained_fraction: F,
                            forward_rate_cap: F, reverse_rate_cap: F,
                            pulse_exit_cap: F, preparation_tv: F = F(0),
                            preparation_mismatch_tv: F = F(0),
                            native_generator_error_integral: F = F(0),
                            controlled_generator_error_integral: F = F(0),
                            delay: F = F(0), delay_exit_cap: F = F(0),
                            delay_inflow_cap: F = F(0),
                            detector_contrast_error: F = F(0),
                            detector_gain: tuple[F, F] = (F(1), F(1))) -> dict:
    """Enclose native current from matched NATIVE minus CONTROLLED readouts.

    Generator-error integrals bound the integral of the maximal row-TV
    generator discrepancy in each arm through its entire actual history.
    Delay caps describe the IDEAL common native postpulse evolution. Clock
    jitter is independent of hidden state and matched between the two arms.
    Detector gain must be positive; its common offset cancels. The response
    interval already includes sampling uncertainty. All calibrations are a
    joint external contract; this function is not a physical calibration tool.
    """
    dlow, dhigh = interval(response, "response")
    tlo, thi = interval(duration, "duration")
    gain_low, gain_high = interval(detector_gain, "detector gain")
    if not 0 < gain_low <= gain_high <= 1:
        raise ValueError("Require detector gain in (0,1]")
    probability(retained_fraction, "retained fraction")
    if retained_fraction == 1:
        raise ValueError("A nonzero channel suppression is required")
    probability(preparation_mismatch_tv, "preparation mismatch TV")
    nonnegative(detector_contrast_error, "detector contrast error")
    s = 1 - retained_fraction
    prep = matched_preparation_bias(preparation_tv, thi, s,
                                    forward_rate_cap, reverse_rate_cap)
    generator = (one_minus_exp_upper(native_generator_error_integral) +
                 one_minus_exp_upper(controlled_generator_error_integral))
    bias = (gain_high * (prep + preparation_mismatch_tv + generator) +
            detector_contrast_error)
    omega = delay_oscillation_bound(delay, delay_exit_cap, delay_inflow_cap)
    klo, khi = delayed_kernel_interval(duration, pulse_exit_cap, omega)
    result = {"orientation": "native minus controlled endpoint readout",
              "preparation_bias": str(prep), "generator_bias": str(generator),
              "total_response_bias": str(bias), "delay_oscillation_bound": str(omega),
              "kernel_interval": [str(klo), str(khi)],
              "scope": "Conditional finite-state CTMC certificate; no empirical calibration claim"}
    if klo <= 0:
        result["status"] = "delay_or_duration_contract_does_not_certify_sign"
        result["current_interval"] = None
        return result
    denominator = (gain_low * s * klo, gain_high * s * khi)
    widened = (max(-gain_high, dlow - bias), min(gain_high, dhigh + bias))
    if widened[0] > widened[1]:
        result["status"] = "inconsistent_response_and_calibration_contract"
        result["current_interval"] = None
        return result
    ratios = [x / y for x in widened for y in denominator]
    current = min(ratios), max(ratios)
    # The physical channel rate caps also give a valid native-current interval.
    current = max(-reverse_rate_cap, current[0]), min(forward_rate_cap, current[1])
    if current[0] > current[1]:
        result["status"] = "inconsistent_response_and_rate_contract"
        result["current_interval"] = None
        return result
    result["current_interval"] = [str(x) for x in current]
    result["status"] = ("positive_current_certified" if current[0] > 0 else
                        "negative_current_certified" if current[1] < 0 else "unresolved")
    return result


def nonstationary_current_bias(duration: F, forward_rate: F, reverse_rate: F,
                               controlled_membership_cap_i: F,
                               controlled_membership_cap_j: F,
                               native_exit_cap_j: F, native_inflow_cap_j: F,
                               current_cap: F) -> F:
    """Short-pulse bias for arbitrary COMMON preparation, no stationarity.

    |(p_native_j(t)-p_controlled_j(t))/((1-r)*t)-J_rho| <= this bound.
    Membership caps bound both departure from and arrival into each endpoint.
    This is an O(t) sufficient bias bound, not an all-time sign theorem.
    """
    entries = {"duration": duration, "forward rate": forward_rate,
               "reverse rate": reverse_rate, "controlled i cap": controlled_membership_cap_i,
               "controlled j cap": controlled_membership_cap_j,
               "native exit cap": native_exit_cap_j, "native inflow cap": native_inflow_cap_j,
               "current cap": current_cap}
    for name, value in entries.items():
        nonnegative(value, name)
    return duration * (forward_rate * controlled_membership_cap_i +
                       reverse_rate * controlled_membership_cap_j +
                       current_cap * (native_exit_cap_j + native_inflow_cap_j)) / 2


def make_results() -> dict:
    # Constructed rational examples; they are not measurements.
    import numpy as np
    from scipy.linalg import expm
    rng = np.random.default_rng(71293)
    trials = 180
    worst_ratios = {"coupling": 0.0, "preparation": 0.0, "delay": 0.0,
                    "nonstationary_bias": 0.0, "shared_generator_drift": 0.0}
    for index in range(trials):
        n = 2 + index % 7
        off = rng.uniform(.03, 2.0, (n, n))
        np.fill_diagonal(off, 0)
        q = off - np.diag(off.sum(axis=1))
        i, j = 0, 1
        a, b = off[i, j] * .6, off[j, i] * .7
        s = .6
        l = np.zeros((n, n)); l[i, j] = a; l[j, i] = b
        l[i, i] = -a; l[j, j] = -b
        qr = q - s * l
        m = q.T.copy(); m[-1] = 1
        rhs = np.zeros(n); rhs[-1] = 1
        pi = np.linalg.solve(m, rhs)
        rho = rng.dirichlet(np.ones(n))
        eta = abs(rho - pi).sum() / 2
        t = float(rng.uniform(.001, 1.2))
        p, pr = expm(t*q), expm(t*qr)
        delta = max(abs(p-pr).sum(axis=1)/2)
        cap = -np.expm1(-s*max(a,b)*t)
        worst_ratios["coupling"] = max(worst_ratios["coupling"], delta/cap)
        err = abs((rho-pi) @ (pr-p)[:, j])
        prep_cap = min(eta*s*(a+b)*t, 2*eta*cap)
        worst_ratios["preparation"] = max(worst_ratios["preparation"], err/prep_cap)
        d = float(rng.uniform(0, .3)); rd = expm(d*q)
        native_j = pi[i]*a-pi[j]*b
        delay_error = abs(pi @ (pr-p) @ (rd[:, j] - np.eye(n)[:, j]))
        omega = -np.expm1(q[j,j]*d) - np.expm1(-max(off[:,j])*d)
        bound = s*abs(native_j)*t*omega
        if bound > 1e-13:
            worst_ratios["delay"] = max(worst_ratios["delay"], delay_error/bound)
        rate_i = max(-qr[i,i], max(qr[k,i] for k in range(n) if k != i))
        rate_j = max(-qr[j,j], max(qr[k,j] for k in range(n) if k != j))
        current0 = rho[i]*a-rho[j]*b
        empirical_bias = abs((rho @ (p-pr)[:,j])/(s*t)-current0)
        bias_bound = t*(a*rate_i+b*rate_j+max(a,b)*(-q[j,j]+max(off[:,j])))/2
        worst_ratios["nonstationary_bias"] = max(worst_ratios["nonstationary_bias"], empirical_bias/bias_bound)
        # A shared extra physical generator is a constructed common drift.
        extra_off = rng.uniform(0,.1,(n,n)); np.fill_diagonal(extra_off,0)
        extra = extra_off - np.diag(extra_off.sum(axis=1))
        error_rate = max(abs(extra).sum(axis=1)/2)
        actual_contrast = rho @ (expm(t*(q+extra))-expm(t*(qr+extra)))[:,j]
        reference_contrast = rho @ (p-pr)[:,j]
        drift_bound = s*(a+b+2*max(a,b))*error_rate*t*t/2
        worst_ratios["shared_generator_drift"] = max(worst_ratios["shared_generator_drift"],
                abs(actual_contrast-reference_contrast)/drift_bound)
    assert max(worst_ratios.values()) < 1+1e-9
    # Compare matched-preparation and old unmatched bound in response units.
    comparison = []
    for t in (F(1,100), F(1,10), F(1)):
        bound = matched_preparation_bias(F(1,100), t, F(1,2), F(3), F(1))
        comparison.append({"duration": str(t), "old_preparation_response_bound": "1/50",
                           "matched_preparation_response_bound": str(bound),
                           "matched_current_bias_bound": str(bound/(F(1,2)*t))})
    certificate = robust_current_interval((F(9,1000), F(11,1000)), duration=(F(1,10),F(1,10)),
        retained_fraction=F(1,2), forward_rate_cap=F(3), reverse_rate_cap=F(1),
        pulse_exit_cap=F(4), preparation_tv=F(1,1000), delay=F(1,100),
        delay_exit_cap=F(4), delay_inflow_cap=F(4),
        native_generator_error_integral=F(1,10000), controlled_generator_error_integral=F(1,10000),
        detector_contrast_error=F(1,10000))
    return {"status": "Proved conditional bounds with synthetic numerical checks; no empirical data",
            "seed": 71293, "random_network_checks": trials, "state_dimensions": [2,3,4,5,6,7,8],
            "largest_observed_error_over_bound": worst_ratios,
            "matched_preparation_comparison": comparison,
            "illustrative_exact_current_certificate": certificate}


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "results" / "robust_physics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(make_results(), indent=2) + "\n")
    print(target)
