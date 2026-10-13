#!/usr/bin/env python3
"""Exact-rational, anytime candidate adequacy and abstention.

A candidate must supply bounds on P(Y=1 | ALL retained past, selected action),
not stationary marginal bounds. Call prepare() BEFORE obtaining the observation.
The software validates algebra; it cannot validate that these scientific bounds
hold. Fixed mixtures of predictable test-supermartingale factors are standard
betting constructions, not a novel statistical principle.

Run this file to execute deterministic adversarial checks and write JSON.
Python >=3.10; standard library only. No random simulation is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path
from typing import Mapping, Sequence
import json

Rational = int | F


def rational(value: Rational) -> F:
    """Do not silently translate approximate floats into exact scientific input."""
    if not isinstance(value, (int, F)) or isinstance(value, bool):
        raise TypeError("use an integer or fractions.Fraction, not a float")
    return F(value)


@dataclass(frozen=True)
class Interval:
    lower: F
    upper: F

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", rational(self.lower))
        object.__setattr__(self, "upper", rational(self.upper))
        if not 0 <= self.lower <= self.upper <= 1:
            raise ValueError("conditional probability interval must lie in [0,1]")


@dataclass(frozen=True)
class Factor:
    zero: F
    one: F

    def __post_init__(self) -> None:
        object.__setattr__(self, "zero", rational(self.zero))
        object.__setattr__(self, "one", rational(self.one))
        if min(self.zero, self.one) < 0:
            raise ValueError("test factors must be nonnegative")

    def expected(self, p: Rational) -> F:
        p = rational(p)
        return (1 - p) * self.zero + p * self.one

    def valid_for(self, interval: Interval) -> bool:
        return max(self.expected(interval.lower), self.expected(interval.upper)) <= 1

    def observed(self, y: int) -> F:
        if type(y) is not int or y not in (0, 1):
            raise ValueError("observation must be integer 0 or 1")
        return self.one if y else self.zero


def interval_bet(interval: Interval, stake: Rational) -> Factor:
    """Affine factor 1+stake*(Y-bound), choosing the valid signed endpoint.

    stake >=0 centers at upper; stake<0 centers at lower. Nonnegativity
    is checked, including degenerate intervals at zero or one.
    """
    stake = rational(stake)
    bound = interval.upper if stake >= 0 else interval.lower
    factor = Factor(1 - stake * bound, 1 + stake * (1 - bound))
    if not factor.valid_for(interval):
        raise ValueError("factor is not valid for the supplied interval")
    return factor


def default_stakes(interval: Interval) -> tuple[F, F]:
    """Two directional components, each risking half its current wealth.

    At degenerate support endpoints use a finite stake. Observing an impossible
    outcome can also support a separately specified support test; no infinite
    floating-point values are introduced here.
    """
    up = F(1, 2) / interval.upper if interval.upper else F(1)
    down = -F(1, 2) / (1 - interval.lower) if interval.lower < 1 else -F(1)
    return up, down


def predictive_stake(interval: Interval, alternative: Rational) -> F:
    """Choose a valid stake from a PREDICTABLE alternative success forecast.

    In an interior point null, the resulting factor is exactly the likelihood
    ratio Bernoulli(alternative)/Bernoulli(null). A Laplace forecast based only
    on earlier bits yields the standard uniform-Beta mixture likelihood ratio.
    For an interval, use its nearest endpoint when the forecast lies outside;
    otherwise do not bet. Endpoint support cases use finite, valid stakes.
    """
    q = rational(alternative)
    if not 0 <= q <= 1:
        raise ValueError("alternative forecast must lie in [0,1]")
    if q > interval.upper:
        b = interval.upper
        return (q - b) / (b * (1 - b)) if b else q
    if q < interval.lower:
        b = interval.lower
        return (q - b) / (b * (1 - b)) if b < 1 else q - 1
    return F(0)


@dataclass(frozen=True)
class Decision:
    status: str
    target: str | None
    survivors: tuple[str, ...]
    excluded: tuple[str, ...]


class AdequacyMonitor:
    """Anytime survivor set with a strict prediction/observation lifecycle.

    targets maps each candidate to its asserted, constant scientific target.
    Every candidate must be nonempty and target-homogeneous. A correct target
    guarantee requires at least one fixed true candidate to be included.

    protection='union': each candidate uses alpha. If ANY one candidate is
    conditionally valid throughout, ever-empty survival or ever-wrong target
    commitment has probability <=alpha. No candidate-count correction is needed.

    protection='simultaneous': each candidate uses alpha/K. The probability of
    ever excluding ANY conditionally valid candidate is <=alpha by union bound.

    weights are fixed, nonnegative, sum one; stakes may depend on retained past.
    Default two-sided weights are (1/2,1/2). No assertion of power is implicit.
    """
    def __init__(self, targets: Mapping[str, str], alpha: Rational = F(1, 20),
                 protection: str = "union", weights: Sequence[Rational] = (F(1, 2), F(1, 2))) -> None:
        self.targets = dict(targets)
        if not self.targets or any(not isinstance(k, str) or not isinstance(v, str)
                                   for k, v in self.targets.items()):
            raise ValueError("provide a nonempty string-to-string target mapping")
        self.alpha = rational(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if protection not in ("union", "simultaneous"):
            raise ValueError("unknown protection type")
        self.protection = protection
        self.level = self.alpha if protection == "union" else self.alpha / len(self.targets)
        self.weights = tuple(rational(w) for w in weights)
        if not self.weights or min(self.weights) < 0 or sum(self.weights) != 1:
            raise ValueError("fixed mixture weights must be nonnegative and sum to one")
        self.wealth = {j: [F(1)] * len(self.weights) for j in self.targets}
        self.max_e = {j: F(1) for j in self.targets}
        self.excluded: set[str] = set()
        self.rounds = 0
        self.history: list[tuple[str, int]] = []
        self._pending: tuple[str, dict[str, tuple[Factor, ...]]] | None = None

    def prepare(self, action: str, intervals: Mapping[str, Interval],
                stakes: Mapping[str, Sequence[Rational]] | None = None) -> None:
        """Declare all conditional predictions and bets before reading Y."""
        if self._pending is not None:
            raise RuntimeError("observe the pending round before preparing another")
        if not isinstance(action, str):
            raise TypeError("action must be a string")
        if set(intervals) != set(self.targets):
            raise ValueError("supply an interval for every candidate, with no extra candidates")
        if stakes is not None and set(stakes) != set(self.targets):
            raise ValueError("supply stakes for every candidate")
        factors: dict[str, tuple[Factor, ...]] = {}
        for j in self.targets:
            if not isinstance(intervals[j], Interval):
                raise TypeError("predictions must be Interval instances")
            sj = default_stakes(intervals[j]) if stakes is None else tuple(stakes[j])
            if len(sj) != len(self.weights):
                raise ValueError("number of stakes must match fixed mixture weights")
            factors[j] = tuple(interval_bet(intervals[j], x) for x in sj)
        self._pending = (action, factors)

    def observe(self, y: int) -> Decision:
        if self._pending is None:
            raise RuntimeError("prepare predictions and bets before observing")
        if type(y) is not int or y not in (0, 1):
            raise ValueError("observation must be integer 0 or 1")
        action, factors = self._pending
        for j in self.targets:
            self.wealth[j] = [w * f.observed(y) for w, f in zip(self.wealth[j], factors[j])]
            self.max_e[j] = max(self.max_e[j], self.e_value(j))
            if self.max_e[j] * self.level >= 1:
                self.excluded.add(j)
        self.rounds += 1
        self.history.append((action, y))
        self._pending = None
        return self.decision()

    def e_value(self, candidate: str) -> F:
        return sum((q * w for q, w in zip(self.weights, self.wealth[candidate])), F(0))

    def decision(self) -> Decision:
        survivors = tuple(j for j in self.targets if j not in self.excluded)
        excluded = tuple(j for j in self.targets if j in self.excluded)
        labels = {self.targets[j] for j in survivors}
        if not survivors:
            return Decision("abstain_all_rejected", None, survivors, excluded)
        if len(labels) != 1:
            return Decision("abstain_ambiguous", None, survivors, excluded)
        return Decision("commit", next(iter(labels)), survivors, excluded)


def exact_static_e(p: F, n: int, k: int) -> F:
    interval = Interval(p, p)
    factors = [interval_bet(interval, x) for x in default_stakes(interval)]
    return sum((f.one ** k * f.zero ** (n - k) for f in factors), F(0)) / 2


def both_wrong_final_power(n: int, alpha: F = F(1, 20)) -> dict[str, object]:
    """Exact iid Bernoulli(1/2) probability BOTH point candidates reject at n.

    This is a lower bound on rejection by n, because actual exclusions persist.
    The alternatives 1/4 and 3/4 are wrong; these are synthetic distributions.
    """
    counts = [k for k in range(n + 1)
              if exact_static_e(F(1, 4), n, k) >= 1 / alpha
              and exact_static_e(F(3, 4), n, k) >= 1 / alpha]
    power = sum((F(comb(n, k), 2 ** n) for k in counts), F(0))
    return {"records": n, "true_conditional_p": "1/2", "candidate_p": ["1/4", "3/4"],
            "rejecting_one_counts": counts, "power_exact": str(power), "power_float": float(power),
            "interpretation": "Exact final-time power; lower bound for sticky rejection by this time."}


def both_wrong_predictive_power(n: int, alpha: F = F(1, 20)) -> dict[str, object]:
    """Exact final-time power of a standard Beta(1,1) predictive mixture."""
    counts = [k for k in range(n + 1)
              if all(F(1, (n + 1) * comb(n, k)) / (p ** k * (1 - p) ** (n - k)) >= 1 / alpha
                     for p in (F(1, 4), F(3, 4)))]
    power = sum((F(comb(n, k), 2 ** n) for k in counts), F(0))
    # Verify sequential predictable implementation against closed form for
    # EVERY complete record up to eight rounds, not just selected final counts.
    checked = 0
    for horizon in range(1, 9):
        for bits in product((0, 1), repeat=horizon):
            monitor = AdequacyMonitor({"a": "A", "b": "B"}, weights=(F(1),))
            intervals = {"a": Interval(F(1, 4), F(1, 4)), "b": Interval(F(3, 4), F(3, 4))}
            successes = 0
            for t, y in enumerate(bits):
                q = F(successes + 1, t + 2)  # No current/future outcome is used.
                monitor.prepare("read", intervals,
                                {j: (predictive_stake(interval, q),) for j, interval in intervals.items()})
                monitor.observe(y)
                successes += y
            numerator = F(1, (horizon + 1) * comb(horizon, successes))
            for j, p in (("a", F(1, 4)), ("b", F(3, 4))):
                assert monitor.e_value(j) == numerator / (p ** successes * (1 - p) ** (horizon - successes))
            checked += 1
    return {"records": n, "true_conditional_p": "1/2", "candidate_p": ["1/4", "3/4"],
            "strategy": "Laplace forecast from earlier bits; standard uniform-Beta predictive mixture.",
            "power_exact": str(power), "power_float": float(power),
            "sequential_closed_form_record_checks": checked,
            "interpretation": "Exact final-time power; lower bound for sticky rejection by this time. No optimality claim."}


def adaptive_tree_check(horizon: int = 10) -> dict[str, str | int | float]:
    """Exhaustive conditional-null calculation with feedback action choices.

    A is selected after a zero (and initially), B after a one. Conditional p is
    1/4 under A and 3/4 under B. Predictions are these point values. Both betting
    components and action sequence therefore depend on the same retained record.
    """
    alpha = F(1, 4)
    final_mean = F(0)
    crossed_probability = F(0)
    checked_edges = 0
    for bits in product((0, 1), repeat=horizon):
        monitor = AdequacyMonitor({"valid": "correct"}, alpha)
        probability = F(1)
        for t, y in enumerate(bits):
            action = "B" if t and bits[t - 1] else "A"
            p = F(3, 4) if action == "B" else F(1, 4)
            interval = Interval(p, p)
            for stake in default_stakes(interval):
                factor = interval_bet(interval, stake)
                assert factor.expected(p) == 1
            checked_edges += 1
            monitor.prepare(action, {"valid": interval})
            monitor.observe(y)
            probability *= p if y else 1 - p
        final_mean += probability * monitor.e_value("valid")
        if monitor.excluded:
            crossed_probability += probability
    assert final_mean == 1
    assert 0 < crossed_probability <= alpha
    return {"horizon": horizon, "full_records_enumerated": 2 ** horizon,
            "repeated_edge_checks": checked_edges, "expected_terminal_e": str(final_mean),
            "ever_reject_probability": str(crossed_probability),
            "ever_reject_float": float(crossed_probability), "alpha": str(alpha)}


def check_factor_algebra() -> dict[str, int]:
    count = 0
    # Includes all interval endpoints on a quarter grid and nonstationary p.
    for lo in range(5):
        for hi in range(lo, 5):
            interval = Interval(F(lo, 4), F(hi, 4))
            for stake in default_stakes(interval) + (F(0),):
                factor = interval_bet(interval, stake)
                for p in {interval.lower, interval.upper, (interval.lower + interval.upper) / 2}:
                    assert factor.expected(p) <= 1
                    count += 1
            for q in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
                factor = interval_bet(interval, predictive_stake(interval, q))
                assert factor.valid_for(interval)
                count += 1
    try:
        interval_bet(Interval(F(1, 4), F(3, 4)), F(2))
        raise AssertionError("inadmissible negative wealth was accepted")
    except ValueError:
        pass
    return {"conditional_factor_checks": count, "negative_wealth_rejected": 1}


def check_lifecycle_and_targets() -> dict[str, object]:
    monitor = AdequacyMonitor({"a": "clockwise", "b": "counterclockwise"})
    assert monitor.decision().status == "abstain_ambiguous"
    try:
        monitor.observe(1)
        raise AssertionError("read-before-predict was accepted")
    except RuntimeError:
        pass
    intervals = {"a": Interval(F(1, 4), F(1, 4)), "b": Interval(F(3, 4), F(3, 4))}
    decisions = []
    for _ in range(40):
        monitor.prepare("read", intervals)
        decisions.append(monitor.observe(1))
    assert monitor.decision().status == "abstain_all_rejected"
    assert any(d.status == "commit" and d.target == "counterclockwise" for d in decisions)
    # The data here are all ones, incompatible with BOTH point candidates.
    # A label may be provisionally selected before further evidence rejects it;
    # the coverage theorem needs a true candidate, absent in this counterexample.
    same = AdequacyMonitor({"a": "same", "b": "same"})
    assert same.decision().status == "commit"
    simul = AdequacyMonitor({"a": "same", "b": "same"}, protection="simultaneous")
    assert simul.level == F(1, 40)
    try:
        Interval(0.1, F(1, 2))
        raise AssertionError("float was accepted as exact input")
    except TypeError:
        pass
    return {"initial_ambiguous": True, "prediction_lifecycle_enforced": True,
            "same_target_commit": True, "simultaneous_alpha_per_candidate": str(simul.level),
            "absent_true_candidate_can_create_temporary_wrong_commitment": True,
            "float_input_rejected": True}


def marginal_dependence_counterexample() -> dict[str, object]:
    # Y_t=Z for one fair latent bit Z. Every one-time marginal is 1/2.
    # Treating the marginal as a conditional probability violates assumptions.
    crossing_rounds = []
    for z in (0, 1):
        monitor = AdequacyMonitor({"marginal_only": "claimed_null"})
        for t in range(1, 21):
            monitor.prepare("read", {"marginal_only": Interval(F(1, 2), F(1, 2))})
            monitor.observe(z)
            if monitor.excluded:
                crossing_rounds.append(t)
                break
    assert len(crossing_rounds) == 2
    return {"construction": "One fair latent bit Z; every Y_t equals Z.",
            "each_stationary_marginal": "1/2", "claimed_alpha": "1/20",
            "rejection_probability_if_marginal_bounds_misused": "1",
            "crossing_rounds_for_Z_0_and_1": crossing_rounds,
            "reason": "After the first observation, the conditional probability is Z, not 1/2."}


def invisible_misspecification() -> dict[str, object]:
    # T is a rational, irreversible three-state ring; the sensor emits a fresh
    # fair bit independently of the hidden state. Candidate is one-state iid.
    T = [[F(0), F(3, 4), F(1, 4)],
         [F(1, 4), F(0), F(3, 4)],
         [F(3, 4), F(1, 4), F(0)]]
    initial = [F(1, 3)] * 3
    def probability(word: tuple[int, ...]) -> F:
        v = initial[:]
        for _y in word:
            v = [sum((v[i] * T[i][j] / 2 for i in range(3)), F(0)) for j in range(3)]
        return sum(v, F(0))
    checked = 0
    for n in range(9):
        for word in product((0, 1), repeat=n):
            assert probability(word) == F(1, 2) ** n
            checked += 1
    assert initial[0] * T[0][1] - initial[1] * T[1][0] == F(1, 6)
    return {"hidden_transition_matrix": [[str(x) for x in row] for row in T],
            "hidden_stationary_edge_current": "1/6", "checked_words": checked,
            "all_length_identity": "K_y=T/2, so mu K_word 1 = 2^(-n) for every word.",
            "conclusion": "A one-state iid sensor model and an irreversible hidden ring have identical complete records. No test can distinguish these accessible laws."}


def main() -> None:
    output = {"method_status": "Established betting theory, implemented as a calibrated candidate-adequacy and abstention interface.",
              "factor_algebra": check_factor_algebra(),
              "adaptive_conditional_null": adaptive_tree_check(),
              "lifecycle_and_target_checks": check_lifecycle_and_targets(),
              "both_wrong_models": both_wrong_final_power(120),
              "both_wrong_predictive_models": both_wrong_predictive_power(120),
              "stationary_marginal_failure": marginal_dependence_counterexample(),
              "invisible_misspecification": invisible_misspecification(),
              "limitations": ["Scientific validity of input conditional envelopes is external to the algebra checker.",
                              "No universal detection of omitted physics or guarantee of eventual commitment.",
                              "No finite-energy discrimination guarantee without a separate physical cost model.",
                              "A transient target commitment can be wrong when the true process is absent.",
                              "Synthetic exact calculations are not physical data or independent external replication."]}
    destination = Path(__file__).resolve().parents[1] / "results" / "model_adequacy.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"checks": "PASS", "output": str(destination),
                      "adaptive_null_rejection_probability": output["adaptive_conditional_null"]["ever_reject_float"],
                      "both_wrong_power_n120": output["both_wrong_models"]["power_float"]}, indent=2))


if __name__ == "__main__":
    main()
