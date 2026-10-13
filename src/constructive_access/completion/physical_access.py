#!/usr/bin/env python3
"""Physical realizations of calibrated current probes in finite CTMCs.

The principal result is the finite-pulse endpoint response theorem.  The
arithmetic certificates below certify a conditional mathematical model, not a
laboratory's channel selectivity, state preparation, or detector calibration.
Public inference helpers use exact Fraction inputs; floating point is confined
to explicitly labelled numerical illustrations in make_results().
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json


def rational(x: object, name: str = "value") -> F:
    if type(x) is not F:
        raise TypeError(f"{name} must be an exact fractions.Fraction")
    return x


def interval(x: object, name: str = "interval") -> tuple[F, F]:
    if type(x) is not tuple or len(x) != 2:
        raise TypeError(f"{name} must be a pair of exact Fractions")
    lo, hi = x
    rational(lo, name); rational(hi, name)
    if lo > hi:
        raise ValueError(f"{name} is reversed")
    return lo, hi


def ceil_fraction(x: F) -> int:
    return -((-x.numerator) // x.denominator)


def exp_bounds(x: F, terms: int = 48) -> tuple[F, F]:
    """Rational enclosure of exp(x), x >= 0, by Taylor plus geometric tail."""
    rational(x)
    if x < 0 or type(terms) is not int or terms < 0:
        raise ValueError("Require x >= 0 and a nonnegative integer term count")
    if x >= terms + 2:
        raise ValueError("Taylor tail ratio must be strictly below one")
    term = F(1)
    partial = term
    for k in range(1, terms + 1):
        term *= x / k
        partial += term
    next_term = term * x / (terms + 1)
    upper = partial + next_term / (1 - x / (terms + 2))
    return partial, upper


def exp_negative_bounds(x: F, terms: int = 48) -> tuple[F, F]:
    lo, hi = exp_bounds(x, terms)
    return 1 / hi, 1 / lo


def certified_exponent(alpha: F) -> tuple[int, int]:
    """Find integer m with a rational certificate 2 exp(-m) <= alpha."""
    rational(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("Require 0 < alpha < 1")
    m = 1
    while True:
        terms = max(48, 2 * m)
        if exp_bounds(F(m), terms)[0] >= 2 / alpha:
            return m, terms
        m += 1


def hoeffding_plan(error: F, alpha: F) -> dict:
    """Independent paired endpoint differences Z in [-1,1].

    P(|Zbar-EZ| >= error) <= 2 exp(-N error**2/2).
    This is a sufficient count, not a claim of minimum statistical cost.
    """
    rational(error, "error")
    if not 0 < error < 2:
        raise ValueError("Require 0 < error < 2")
    m, terms = certified_exponent(alpha)
    records = ceil_fraction(2 * m / error**2)
    exponent = F(records) * error**2 / 2
    terms = max(terms, 2 * ceil_fraction(exponent))
    lower = exp_bounds(exponent, terms)[0]
    assert lower >= 2 / alpha
    return {"records": records, "error": str(error), "failure_probability": str(alpha),
            "exponent": str(exponent), "exponential_taylor_terms": terms,
            "certificate": "Taylor lower bound for exp(N*error^2/2) >= 2/alpha",
            "assumption": "Independent native-stationary preparations and noninvasive paired endpoint readouts"}


def poisson_exposure_plan(rate_cap: F, error: F, alpha: F) -> dict:
    """Fixed accumulated source-state exposure; NOT conditional fixed-walltime.

    N_ij(tau_i(T)) ~ Poisson(q_ij*T), where tau_i(T) is the inverse source
    occupation time in a fixed finite irreducible CTMC.  Bernstein's bound and
    q_ij <= rate_cap give the stated interval around N_ij/T.
    """
    rational(rate_cap, "rate_cap"); rational(error, "error")
    if rate_cap <= 0 or error <= 0:
        raise ValueError("Require positive rate cap and rate error")
    m, terms = certified_exponent(alpha)
    exposure = 2 * m * (rate_cap + error / 3) / error**2
    exponent = exposure * error**2 / (2 * (rate_cap + error / 3))
    assert exponent == m and exp_bounds(exponent, terms)[0] >= 2 / alpha
    return {"source_exposure": str(exposure), "rate_cap": str(rate_cap),
            "error": str(error), "failure_probability": str(alpha),
            "exponent": str(exponent), "exponential_taylor_terms": terms,
            "stopping_rule": "Stop on reaching deterministic accumulated source-state exposure",
            "wall_time": "Random and not bounded by the exposure; incomplete deadline-limited acquisitions abstain"}


def short_pulse_current_plan(error: F, current_cap: F, endpoint_exit_cap: F,
                            factor: F, alpha: F, stationary_target_cap: F = F(1)) -> dict:
    """Certified magnitude plan with O(error^-3) sufficient paired records.

    This plan assumes exact native-stationary preparation and exactly known
    selective factor. It is not a minimax lower bound. The deterministic bias
    of Zbar/((1-r)t) is <= current_cap*endpoint_exit_cap*t. For a singleton
    readout, E Z^2 <= [2*pi_j*Lambda+(1-r)*current_cap]*t. Bernstein therefore
    improves the range-only Hoeffding record exponent from four to three.
    """
    for name, x in (("error", error), ("current_cap", current_cap),
                    ("endpoint_exit_cap", endpoint_exit_cap), ("factor", factor),
                    ("stationary_target_cap", stationary_target_cap)):
        rational(x, name)
    if error <= 0 or current_cap <= 0 or endpoint_exit_cap <= 0 or not 0 < factor < 1:
        raise ValueError("Positive error/current/rate caps and factor in (0,1) required")
    if not 0 < stationary_target_cap <= 1:
        raise ValueError("Stationary target probability cap must lie in (0,1]")
    m, terms = certified_exponent(alpha)
    duration = error / (2 * current_cap * endpoint_exit_cap)
    response_error = (1 - factor) * duration * error / 2
    variance_cap = min(F(1), (2 * stationary_target_cap * endpoint_exit_cap +
                             (1 - factor) * current_cap) * duration)
    records = ceil_fraction(2 * m * (variance_cap + 2 * response_error / 3) / response_error**2)
    exponent = F(records) * response_error**2 / (2 * (variance_cap + 2 * response_error / 3))
    terms = max(terms, 2 * ceil_fraction(exponent))
    assert exp_bounds(exponent, terms)[0] >= 2 / alpha
    return {"records": records, "duration": str(duration), "current_error": str(error),
            "response_sampling_error": str(response_error), "variance_cap": str(variance_cap),
            "current_bias_bound": str(current_cap * endpoint_exit_cap * duration),
            "current_cap": str(current_cap), "endpoint_exit_cap": str(endpoint_exit_cap),
            "factor": str(factor), "stationary_target_cap": str(stationary_target_cap),
            "failure_probability": str(alpha), "bernstein_exponent": str(exponent),
            "exponential_taylor_terms": terms, "total_active_pulse_time": str(records * duration),
            "precision_scope": "O(error^-3) sufficient records at fixed nonzero parameters; no optimality lower bound",
            "physical_assumptions": "Exact stationary preparation and selective factor; independent paired singleton records; known current and endpoint exit caps"}


def pulse_current_interval(response: tuple[F, F], factor: tuple[F, F],
                           duration: F, endpoint_exit_cap: F,
                           preparation_tv: F = F(0), detector_error: F = F(0),
                           other_response_error: F = F(0)) -> dict:
    """Enclose native i->j current from D=E[1{X0=j}-1{Xt=j}].

    Factor r suppresses only the named bidirectional channel; all other
    generator entries remain unchanged. Both endpoint exit rates under the
    perturbed generator are <= endpoint_exit_cap. Preparation is within TV eta
    of the native stationary law; paired-readout bias is <= detector_error.
    The response interval already includes sampling error. The additional
    other_response_error must bound the whole expectation discrepancy, not an
    unpropagated per-operation or per-time error.
    """
    low, high = interval(response, "response")
    rlo, rhi = interval(factor, "factor")
    for name, x in (("duration", duration), ("endpoint_exit_cap", endpoint_exit_cap),
                    ("preparation_tv", preparation_tv), ("detector_error", detector_error),
                    ("other_response_error", other_response_error)):
        rational(x, name)
    if not 0 < rlo <= rhi < 1:
        raise ValueError("Require strictly positive channel suppression factors below one")
    if duration <= 0 or endpoint_exit_cap <= 0:
        raise ValueError("Require positive duration and endpoint exit cap")
    if not 0 <= preparation_tv <= 1 or detector_error < 0 or other_response_error < 0:
        raise ValueError("Invalid calibration errors")
    if low < -1 or high > 1:
        raise ValueError("The observed paired response lies in [-1,1]")
    bias = 2 * preparation_tv + detector_error + other_response_error
    adjusted = (max(F(-1), low - bias), min(F(1), high + bias))
    x = endpoint_exit_cap * duration
    terms = max(48, 2 * ceil_fraction(x))
    exp_lower, exp_upper = exp_negative_bounds(x, terms)
    hlow, hhigh = duration * exp_lower, duration
    gain = ((1 - rhi) * hlow, (1 - rlo) * hhigh)
    values = [v / g for v, g in product(adjusted, gain)]
    lo, hi = min(values), max(values)
    sign = "positive" if lo > 0 else "negative" if hi < 0 else "unresolved"
    return {"native_current_interval": [str(lo), str(hi)], "orientation": sign,
            "adjusted_response_interval": list(map(str, adjusted)),
            "positive_kernel_interval": [str(hlow), str(hhigh)],
            "positive_gain_interval": list(map(str, gain)),
            "duration": str(duration), "factor_interval": list(map(str, factor)),
            "endpoint_exit_cap": str(endpoint_exit_cap), "preparation_tv": str(preparation_tv),
            "detector_error": str(detector_error), "other_response_error": str(other_response_error),
            "exp_negative_interval": [str(exp_lower), str(exp_upper)],
            "exponential_taylor_terms": terms,
            "scope": "Conditional calibrated enclosure; conservative, not a sharp network-constrained minimax interval"}


def selective_rate_current_interval(bounds: tuple[tuple[F, F], ...]) -> dict:
    """Exact box envelope for native current inferred from rate differences.

    Coordinates: pi_i, pi_j, q0_ij, q0_ji, qr_ij, qr_ji, r.
    Native stationary probabilities must be used, not perturbed ones.  The box
    is allowed to include physically inconsistent points; this is conservative.
    """
    if type(bounds) is not tuple or len(bounds) != 7:
        raise TypeError("Seven rational intervals are required")
    for k, b in enumerate(bounds):
        lo, hi = interval(b, f"coordinate {k}")
        if lo < 0:
            raise ValueError("Probabilities, rates, and factors must be nonnegative")
        if k < 2 and hi > 1:
            raise ValueError("Probability exceeds one")
    if not 0 < bounds[-1][0] <= bounds[-1][1] < 1:
        raise ValueError("Suppression interval must lie strictly inside (0,1)")
    vals = []
    for pi, pj, f0, b0, fr, br, r in product(*bounds):
        vals.append((pi * (f0 - fr) - pj * (b0 - br)) / (1 - r))
    return {"native_current_interval": [str(min(vals)), str(max(vals))],
            "vertices": len(vals),
            "proof": "Coordinatewise affine numerator and positive linear denominator imply extrema at box vertices",
            "scope": "Sharp for the declared rectangular algebraic box, potentially conservative for physical constraints"}


@dataclass(frozen=True)
class Channel:
    source: int
    target: int
    forward: F
    reverse: F

    def __post_init__(self):
        if type(self.source) is not int or type(self.target) is not int or self.source < 0 or self.target < 0 or self.source == self.target:
            raise ValueError("Distinct nonnegative integer endpoints required")
        if rational(self.forward, "forward") <= 0 or rational(self.reverse, "reverse") <= 0:
            raise ValueError("Both channel rates must be strictly positive")


def generator(states: int, channels: tuple[Channel, ...]) -> list[list[F]]:
    if type(states) is not int or states < 2:
        raise ValueError("At least two states required")
    q = [[F(0) for _ in range(states)] for _ in range(states)]
    for c in channels:
        if max(c.source, c.target) >= states:
            raise ValueError("Channel endpoint outside state space")
        i, j = c.source, c.target
        q[i][j] += c.forward; q[i][i] -= c.forward
        q[j][i] += c.reverse; q[j][j] -= c.reverse
    return q


def suppress(channels: tuple[Channel, ...], index: int, factor: F) -> tuple[Channel, ...]:
    rational(factor)
    if type(index) is not int or not 0 <= index < len(channels) or not 0 < factor < 1:
        raise ValueError("Valid channel index and suppression factor in (0,1) required")
    out = list(channels)
    c = out[index]
    out[index] = Channel(c.source, c.target, c.forward * factor, c.reverse * factor)
    return tuple(out)


def solve_linear(a: list[list[F]], b: list[F]) -> list[F]:
    n = len(b)
    if len(a) != n or any(len(row) != n for row in a):
        raise ValueError("Square linear system required")
    m = [[F(x) for x in row] + [F(y)] for row, y in zip(a, b)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if m[i][k]), None)
        if pivot is None:
            raise ValueError("Singular linear system")
        m[k], m[pivot] = m[pivot], m[k]
        scale = m[k][k]
        m[k] = [x / scale for x in m[k]]
        for i in range(n):
            if i != k and m[i][k]:
                scale = m[i][k]
                m[i] = [x - scale * y for x, y in zip(m[i], m[k])]
    return [row[-1] for row in m]


def stationary(q: list[list[F]]) -> tuple[F, ...]:
    n = len(q)
    if n < 2 or any(len(row) != n for row in q):
        raise ValueError("Square generator with at least two states required")
    for i, row in enumerate(q):
        if sum(row) != 0 or any(x < 0 for j, x in enumerate(row) if i != j):
            raise ValueError("Invalid row generator")
    a = [[q[j][i] for j in range(n)] for i in range(n - 1)] + [[F(1)] * n]
    p = tuple(solve_linear(a, [F(0)] * (n - 1) + [F(1)]))
    if min(p) <= 0:
        raise ValueError("Require a unique strictly positive stationary distribution")
    assert all(sum(p[i] * q[i][j] for i in range(n)) == 0 for j in range(n))
    return p


def current(channel: Channel, pi: tuple[F, ...]) -> F:
    return pi[channel.source] * channel.forward - pi[channel.target] * channel.reverse


def initial_response(qr: list[list[F]], pi: tuple[F, ...]) -> tuple[F, ...]:
    return tuple(sum(pi[i] * qr[i][j] for i in range(len(pi))) for j in range(len(pi)))


def two_endpoint_certificate(q: list[list[F]], pi: tuple[F, ...],
                             qp: list[list[F]], source: int, target: int) -> dict:
    """Check algebraic premises for the general finite-amplitude sign theorem.

    No irreducibility is required. Both finite row generators must be valid,
    pi must be stationary for q, and only the source-target rate pair and its
    compensating diagonal entries may change. This checks a supplied model;
    it does not assert that an unknown apparatus has been calibrated.
    """
    n = len(pi)
    if (type(source) is not int or type(target) is not int or source == target or
        not 0 <= source < n or not 0 <= target < n):
        raise ValueError("Distinct valid endpoints required")
    for matrix in (q, qp):
        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ValueError("Inconsistent generator dimensions")
        for i, row in enumerate(matrix):
            for x in row:
                rational(x)
            if sum(row) or any(x < 0 for j, x in enumerate(row) if i != j):
                raise ValueError("Invalid generator")
    for p in pi:
        rational(p)
    if min(pi) < 0 or sum(pi) != 1 or any(initial_response(q, pi)):
        raise ValueError("pi must be a stationary probability distribution for q")
    allowed = {(source, source), (source, target), (target, source), (target, target)}
    if any(q[i][j] != qp[i][j] for i in range(n) for j in range(n) if (i, j) not in allowed):
        raise ValueError("The perturbation changes a nontarget rate")
    da, db = qp[source][target] - q[source][target], qp[target][source] - q[target][source]
    coefficient = pi[source] * da - pi[target] * db
    slope = initial_response(qp, pi)
    expected = tuple(-coefficient if k == source else coefficient if k == target else F(0) for k in range(n))
    assert slope == expected
    return {"coefficient": str(coefficient), "source": source, "target": target,
            "initial_response": list(map(str, slope)),
            "endpoint_exit_rates": [str(-qp[source][source]), str(-qp[target][target])],
            "conclusion": "Destination occupancy displacement has the coefficient's sign for every finite t>0; it vanishes identically iff the coefficient is zero",
            "proof": "Stationarity gives pi*qp=coefficient*(e_target-e_source); first-hitting survival convolution makes the integrated response kernel strictly positive",
            "irreducibility_required": False}


def gateway_detector_certificate(qp: list[list[F]], source: int, gateway: int,
                                 observed_states: tuple[int, ...]) -> dict:
    """Certify a sufficient coarse-readout extension by graph reachability.

    Every directed path from source into the observed set must first hit the
    gateway. It is not enough that the detector set contains the gateway.
    """
    n = len(qp)
    if n < 2 or any(len(row) != n for row in qp):
        raise ValueError("Square generator required")
    if type(source) is not int or type(gateway) is not int or not 0 <= source < n or not 0 <= gateway < n:
        raise ValueError("Invalid endpoints")
    if type(observed_states) is not tuple or any(type(k) is not int or not 0 <= k < n for k in observed_states):
        raise ValueError("Observed states must be a tuple of valid indices")
    observed = set(observed_states)
    if source in observed or gateway not in observed:
        raise ValueError("Source must be outside and gateway inside the observed set")
    for i, row in enumerate(qp):
        for x in row:
            rational(x)
        if sum(row) or any(x < 0 for j, x in enumerate(row) if i != j):
            raise ValueError("Invalid generator")
    reached = {source}
    frontier = [source]
    while frontier:
        i = frontier.pop()
        for j in range(n):
            if qp[i][j] > 0 and j != gateway and j not in reached:
                reached.add(j)
                frontier.append(j)
    forbidden = reached & observed
    return {"certified": not forbidden, "source": source, "gateway": gateway,
            "observed_states": sorted(observed), "reachable_without_gateway": sorted(reached),
            "counterexample_reachable_observed_states": sorted(forbidden),
            "conclusion": "If certified, the coarse response kernel is strictly positive by first hitting the gateway; if false, this sufficient certificate makes no sign claim"}


def fixed_current_family(states: int) -> tuple[tuple[Channel, ...], tuple[F, ...]]:
    """Increasing, nonidentical hidden leaf kinetics; fixed core current 1/8.

    This constructed family demonstrates size-independent access assumptions,
    not that arbitrary large systems have a size-independent target current or
    cheap stationary preparations.
    """
    if type(states) is not int or states < 4:
        raise ValueError("At least four states required")
    pi = (F(1, 6),) * 3 + (F(1, 2 * (states - 3)),) * (states - 3)
    channels = [Channel(i, (i + 1) % 3, F(9, 8), F(3, 8)) for i in range(3)]
    for k in range(3, states):
        leaf_rate = F(1, 4) + F((k - 3) % 5, 8)
        channels.append(Channel(0, k, pi[k] * leaf_rate / pi[0], leaf_rate))
    return tuple(channels), pi


def make_results(output: Path) -> dict:
    import numpy as np
    from scipy.linalg import expm

    # Two indistinguishable unperturbed generators, incompatible channel currents.
    plus = (Channel(0, 1, F(3), F(1)), Channel(0, 1, F(1), F(3)))
    zero = (Channel(0, 1, F(2), F(2)), Channel(0, 1, F(2), F(2)))
    minus = tuple(Channel(c.source, c.target, c.reverse, c.forward) for c in plus)
    assert generator(2, plus) == generator(2, zero) == generator(2, minus)
    same_generator = []
    for label, channels in (("positive", plus), ("zero", zero), ("negative", minus)):
        q = generator(2, channels)
        pi = stationary(q)
        qr = generator(2, suppress(channels, 0, F(1, 2)))
        j = current(channels[0], pi)
        slope = initial_response(qr, pi)
        assert slope == (j / 2, -j / 2)
        coefficient_certificate = two_endpoint_certificate(q, pi, qr, 0, 1)
        assert F(coefficient_certificate["coefficient"]) == -j / 2
        pr = stationary(qr)
        mfpt = 1 / qr[0][1]
        assert pr[1] - pi[1] == -j / 2 * pr[1] * mfpt
        response = float(pi[1] - (np.array(pi, dtype=float) @ expm(np.array(qr, dtype=float) / 4))[1])
        same_generator.append({"label": label, "native_current": str(j),
                               "stationary": list(map(str, pi)), "perturbed_stationary": list(map(str, pr)),
                               "initial_destination_slope": str(slope[1]),
                               "algebraic_response_certificate": coefficient_certificate,
                               "paired_response_at_t_1_4_numerical": response})

    # The local graph can grow without entering the endpoint response bound.
    family = []
    numerical_checks = 0
    for n in (4, 5, 8, 16, 32, 64):
        channels, pi = fixed_current_family(n)
        q = generator(n, channels)
        assert sum(pi) == 1 and all(sum(pi[i] * q[i][k] for i in range(n)) == 0 for k in range(n))
        j = current(channels[0], pi)
        assert j == F(1, 8)
        qr = generator(n, suppress(channels, 0, F(1, 2)))
        slope = initial_response(qr, pi)
        assert slope == (j / 2, -j / 2) + (F(0),) * (n - 2)
        localcap = max(-qr[0][0], -qr[1][1])
        assert localcap <= 4
        qr_float = np.array(qr, dtype=float)
        pi_float = np.array(pi, dtype=float)
        records = []
        for t in (F(1, 100), F(1, 4), F(1), F(4), F(16)):
            p = expm(qr_float * float(t))
            d = float(pi_float[1] - (pi_float @ p)[1])
            h = d / float(j / 2)
            lower = float(t) * np.exp(-float(localcap * t))
            assert h >= lower - 1e-12 and h <= float(t) + 1e-12
            assert d > 0
            numerical_checks += 1
            records.append({"duration": str(t), "paired_response": d,
                            "kernel_from_response": h, "local_cap_kernel_lower": lower})
        family.append({"states": n, "channels": len(channels), "native_current": str(j),
                       "endpoint_exit_cap": str(localcap), "numerical_pulses": records})

    calibration = pulse_current_interval((F(1, 100), F(3, 200)), (F(49, 100), F(51, 100)),
                                        F(1, 4), F(4), preparation_tv=F(1, 2000),
                                        detector_error=F(1, 2000))
    assert calibration["orientation"] == "positive"
    rejected = pulse_current_interval((F(1, 100), F(3, 200)), (F(49, 100), F(51, 100)),
                                      F(1, 4), F(4), preparation_tv=F(1, 100))
    assert rejected["orientation"] == "unresolved"
    rate_box = selective_rate_current_interval(((F(499, 1000), F(501, 1000)),) * 2 +
                ((F(399, 100), F(401, 100)),) * 2 +
                ((F(249, 100), F(251, 100)), (F(349, 100), F(351, 100)),
                 (F(49, 100), F(51, 100))))

    # Positivity is an integrated first-hitting fact, not pointwise positivity.
    cycle = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], dtype=float)
    cycle_transition = expm(2 * cycle)
    instantaneous_kernel = float(cycle_transition[1, 1] - cycle_transition[0, 1])
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = cycle
    augmented[1, 3] = 1
    integrated = expm(2 * augmented)[:3, 3]
    integrated_kernel = float(integrated[1] - integrated[0])
    assert instantaneous_kernel < 0 < integrated_kernel

    coarse_channels = (Channel(0, 1, F(2), F(1)), Channel(0, 2, F(10), F(1)),
                       Channel(1, 3, F(10), F(1)), Channel(2, 3, F(1), F(3)))
    coarse_pi = stationary(generator(4, coarse_channels))
    coarse_qr = generator(4, suppress(coarse_channels, 0, F(1, 2)))
    coarse_pir = stationary(coarse_qr)
    coarse_slope = initial_response(coarse_qr, coarse_pi)
    slope_a = coarse_slope[1] + coarse_slope[2]
    stationary_a = coarse_pir[1] + coarse_pir[2] - coarse_pi[1] - coarse_pi[2]
    assert slope_a == -F(5, 121) and stationary_a == F(10, 2299)
    gateway_failure = gateway_detector_certificate(coarse_qr, 0, 1, (1, 2))
    assert not gateway_failure["certified"]

    result = {
        "status": "Mathematical results and constructed numerical demonstrations; no laboratory measurements",
        "row_generator_convention": True,
        "general_two_endpoint_theorem": {
            "coefficient": "gamma=pi_i*(qprime_ij-q_ij)-pi_j*(qprime_ji-q_ji)",
            "response": "pprime_j(t)-pi_j=gamma*H_ij^prime(t), with H>0 for all finite t>0",
            "scope": "Any finite valid row generators differing only on the two directed rates and compensating diagonals, with native stationary pi; irreducibility is unnecessary",
            "testable_constraint": "A nonzero response cannot cross the native baseline as pulse duration varies while the same step perturbation is held fixed",
            "not_claimed": "The response magnitude need not be monotone; arbitrary coarse endpoint sets need not preserve the sign"},
        "finite_pulse_theorem": {
            "response": "E_pi[1{X0=j}-1{Xt=j}]=(1-r) J_e H_ij^r(t)",
            "kernel": "H_ij^r(t)=integral_0^t P_jj^r(s) Pr_i^r(tau_j>t-s) ds > 0 for every t>0",
            "bound": "t exp(-Lambda*t) <= H_ij^r(t) <= t; only endpoint exit caps are needed",
            "meaning": "Sign of a single physical-channel current from singleton endpoint occupancy response, conditional on selective kinetic control and native stationarity",
            "priority": "Finite-amplitude static response and linear transient response have close published precedents; priority of this exact packaged theorem unresolved"},
        "same_passive_generator_opposite_currents": same_generator,
        "instantaneous_kernel_counterexample": {"generator": cycle.tolist(), "duration": 2,
             "P_jj_minus_P_ij": instantaneous_kernel, "integrated_kernel": integrated_kernel,
             "meaning": "The proof must use the positive first-hitting convolution; instantaneous transition contrast can be negative"},
        "coarse_detector_counterexample": {
            "channels_source_target_forward_reverse": [[c.source, c.target, str(c.forward), str(c.reverse)] for c in coarse_channels],
            "native_stationary": list(map(str, coarse_pi)), "perturbed_stationary": list(map(str, coarse_pir)),
            "suppressed_channel": 0, "factor": "1/2", "detector_states": [1, 2],
            "initial_coarse_response_derivative": str(slope_a),
            "stationary_coarse_response": str(stationary_a),
            "crossing_proof": "Negative initial derivative and strictly positive stationary limit imply a finite-time baseline crossing by continuity",
            "gateway_check": gateway_failure,
            "implication": "Classical hidden-state coarse readout can cross; crossing alone is not a device-independent quantum-coherence witness"},
        "increasing_state_family": family, "numerical_pulse_checks": numerical_checks,
        "calibrated_pulse_enclosure": calibration, "calibration_failure_abstention": rejected,
        "paired_difference_sampling_plan": hoeffding_plan(F(1, 500), F(1, 20)),
        "short_pulse_magnitude_plans": [short_pulse_current_plan(e, F(1, 8), F(4), F(1, 2), F(1, 20), F(1, 6))
                                         for e in (F(1, 10), F(1, 20), F(1, 40))],
        "fixed_source_exposure_plan": poisson_exposure_plan(F(4), F(1, 100), F(1, 100)),
        "selective_rate_rectangle": rate_box,
        "energy_accounting": {"kinetic_factor_is_not_energy": True,
             "required_terms": ["native fuel during baseline and pulse acquisition", "stationary preparation and reset",
                                "selective control implementation", "detector and recording", "calibration"],
             "pulse_time": "N*t counts only active pulse duration; it omits preparation and calibration",
             "barrier_shift": "If Arrhenius calibration is justified, r=exp(-DeltaB/kBT); DeltaB is a barrier parameter, not total consumed work"},
        "limitations": ["Singleton endpoint must be resolved; a coarse aggregate can reverse the sign",
                        "Native stationary preparation and unchanged nontarget rates are substantive physical assumptions",
                        "Independent repeated preparations are required by the supplied Hoeffding plan",
                        "Exact sign identifiability does not imply useful magnitude precision or low energy cost",
                        "Source exposure is not elapsed wall time; fixed-walltime counts are not conditionally independent Poisson",
                        "Constant current and endpoint rate caps in the size benchmark are imposed, not universal"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "physical_access.json")
    args = parser.parse_args()
    result = make_results(args.output)
    print(json.dumps({"output": str(args.output), "numerical_pulse_checks": result["numerical_pulse_checks"],
                      "status": result["status"]}))
