"""Certified rational finite-time approximation of classical jump dynamics.

Uniformization and its tail bounds are established numerical tools. This file
does not extend them to arbitrary quantum generators. A nominal equality of
approximations is not an exact equality certificate for the underlying CTMCs.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import sympy as sp

from .accesskit import matrix, rational

Q = Fraction


@dataclass(frozen=True)
class WaitCertificate:
    transition: sp.ImmutableMatrix
    row_tv_error: Q
    duration: Q
    terms: int
    uniformization_rate: Q

    def to_dict(self):
        return {"duration": str(self.duration), "terms": self.terms,
                "uniformization_rate": str(self.uniformization_rate),
                "row_tv_error_upper": str(self.row_tv_error),
                "transition": [[str(x) for x in self.transition.row(i)] for i in range(self.transition.rows)],
                "scope": "exact rational stochastic approximation; row TV enclosure of the true finite-time transition"}


def certified_wait(generator, duration, terms=24):
    """Return stochastic E_N and exact max-row TV bound to exp(duration*Q)."""
    g = matrix(generator)
    t = rational(duration)
    n = g.rows
    if n != g.cols or n == 0 or t < 0:
        raise ValueError("Require square nonempty generator and nonnegative duration")
    if not isinstance(terms, int) or isinstance(terms, bool) or terms < 0:
        raise ValueError("terms must be a nonnegative integer")
    if g*sp.ones(n, 1) != sp.zeros(n, 1):
        raise ValueError("Generator rows must sum to zero")
    if any(g[i,j] < 0 for i in range(n) for j in range(n) if i != j):
        raise ValueError("Off-diagonal generator entries must be nonnegative")
    lam = max(rational(-g[i,i]) for i in range(n))
    if not lam or not t:
        return WaitCertificate(sp.ImmutableMatrix(sp.eye(n)), Q(0), t, terms, lam)
    x = lam*t
    if x >= terms+2:
        raise ValueError("Increase terms: geometric scalar tail requires lambda*t < N+2")
    p = sp.eye(n)+g/sp.Rational(lam.numerator, lam.denominator)
    power = sp.eye(n)
    scalar_term = Q(1)
    scalar_sum = Q(1)
    numerator = sp.eye(n)
    for k in range(1, terms+1):
        scalar_term *= x/k
        scalar_sum += scalar_term
        power = power*p
        numerator += sp.Rational(scalar_term.numerator, scalar_term.denominator)*power
    tail = (scalar_term*x/(terms+1))/(1-x/(terms+2))
    radius = tail/(scalar_sum+tail)
    transition = sp.ImmutableMatrix(numerator/sp.Rational(scalar_sum.numerator, scalar_sum.denominator))
    assert min(transition) >= 0 and transition*sp.ones(n,1) == sp.ones(n,1)
    return WaitCertificate(transition, radius, t, terms, lam)


def protocol_tv_radius(action_names, certificates, preparation_tv=Q(0)):
    """Coupling upper bound for a fixed action sequence; observations contract TV."""
    radius = rational(preparation_tv)
    if not 0 <= radius <= 1:
        raise ValueError("preparation_tv must be in [0,1]")
    for a in action_names:
        if a in certificates:
            radius += certificates[a].row_tv_error
    return min(Q(1), radius)


def robust_word_gap(model0, model1, word, waits0, waits1):
    actions = [a for a,y in word]
    nominal = abs(model0.word_probability(word)-model1.word_probability(word))
    error0 = protocol_tv_radius(actions, waits0)
    error1 = protocol_tv_radius(actions, waits1)
    return {"nominal_word_gap": str(nominal), "approximation_error0": str(error0),
            "approximation_error1": str(error1),
            "physical_gap_lower": str(max(Q(0), nominal-error0-error1)),
            "physical_non_equivalence_certified": nominal > error0+error1}
