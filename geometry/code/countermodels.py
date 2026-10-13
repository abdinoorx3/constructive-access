"""Exact scope countermodels for observed local diffusion geometry.

All entropy rates use ordinary reversal unless explicitly marked generalized.
Numerical quadratures are checks, not certificates. Rational inequalities carry
separate analytic certificates documented in research/countermodels.txt.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import i0, i1


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def averaged_gamma_counterexample(c: float = 8.0, amplitude: float = 0.5) -> dict:
    """Connected uniformly elliptic torus, d=1+amplitude*cos(z), v_x=c*d.

    The advertised quantity is explicitly INVALID as an entropy lower bound:
    it divides by E[Gamma|Y] instead of retaining Gamma or an upper envelope.
    """
    c = _finite(c, "c")
    amplitude = _finite(amplitude, "amplitude")
    if not 0 <= abs(amplitude) < 1:
        raise ValueError("|amplitude| must be below one")
    variance = amplitude * amplitude / 2
    numerator = c * c * (1 + variance)
    if variance == 0:
        naive = numerator
    elif numerator == 0:
        naive = 0.0
    else:
        naive = 2 / math.pi * (
            (numerator + variance) * math.atan(math.sqrt(numerator / variance))
            - math.sqrt(numerator * variance)
        )
    return {
        "true_entropy_rate": c * c,
        "invalid_conditional_mean_substitution": naive,
        "valid_retained_volatility_bound": c * c,
        "diffusion_minimum_eigenvalue": min(1 - abs(amplitude), 1),
        "relative_overestimate": None if c == 0 else naive / (c * c) - 1,
    }


def averaged_gamma_certificate() -> dict:
    # atan(24)>pi/2-1/24 and pi>3; see accompanying analytic proof.
    lower = Fraction(577, 8) - Fraction(1153, 288)
    return {
        "invalid_bound_strictly_exceeds": str(lower),
        "true_entropy_rate": "64",
        "strict_violation_margin_lower": str(lower - 64),
        "proof_uses": ["atan(u)<u for u>0", "pi>3"],
    }


def local_volatility(cosine: float) -> float:
    c = _finite(cosine, "cosine")
    if not -1 <= c <= 1:
        raise ValueError("cosine must lie in [-1,1]")
    return (1 - c * c) * (1 + 2 * c) ** 2


def finite_h_naive_ratio(cosine: float, h: float = 1.0) -> float:
    """INVALID finite-h entropy integrand for f=cos(x)+cos(2x)/2.

    The formula is its continuous extension at x=0,pi; its pole at cosine=-1/2
    is genuine. The asymptotic theorem does not license this finite-h ratio.
    """
    c = _finite(cosine, "cosine")
    h = _finite(h, "h")
    if not -1 <= c <= 1 or h <= 0:
        raise ValueError("require cosine in [-1,1] and h>0")
    a = math.exp(-h) * math.sin(h)
    b = math.exp(-4 * h) * math.sin(2 * h)
    numerator = a + b * c
    denominator = h * (1 + 2 * c)
    if denominator == 0:
        return math.inf if numerator != 0 else math.nan
    return (numerator / denominator) ** 2


def finite_h_truncated_quadrature(h: float = 1.0, epsilon: float = 1 / 15625) -> dict:
    """Numerical, not interval-certified, integration on {Gamma>=epsilon}."""
    h = _finite(h, "h")
    epsilon = _finite(epsilon, "epsilon")
    if h <= 0 or epsilon <= 0:
        raise ValueError("h and epsilon must be positive")
    roots = np.roots([-4.0, -4.0, 3.0, 4.0, 1.0 - epsilon])
    cuts = sorted([-1.0, 1.0] + [float(r.real) for r in roots
                  if abs(r.imag) < 1e-10 and -1 < r.real < 1])
    value, estimated_error = 0.0, 0.0
    intervals = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if local_volatility((left + right) / 2) < epsilon:
            continue
        integral, error = quad(
            lambda c: finite_h_naive_ratio(c, h) / (math.pi * math.sqrt(1-c*c)),
            left, right, epsabs=1e-10, epsrel=1e-10, limit=250,
        )
        value += integral
        estimated_error += error
        intervals.append([left, right])
    return {
        "h": h, "epsilon": epsilon, "naive_truncated_value": value,
        "quadrature_estimated_error": estimated_error,
        "true_entropy_rate": 1.0, "included_intervals": intervals,
    }


def finite_h_certificate() -> dict:
    # On 1/200 <= |c+1/2| <= 1/10, Gamma>=1/15625.
    # a=e^-1 sin(1)>3/10, |b|=e^-4 |sin(2)|<1/50.
    # Therefore numerator>7/25; density >=1/pi; pi<22/7.
    lower = Fraction(7, 25) ** 2 * Fraction(190, 2) * Fraction(7, 22)
    return {
        "h": "1", "epsilon": "1/15625",
        "naive_truncated_value_strictly_exceeds": str(lower),
        "true_entropy_rate": "1",
        "strict_violation_margin_lower": str(lower - 1),
        "untruncated_expectation": "positive_infinity",
        "proof_uses": ["8/3<e<11/4", "sin(1)>5/6", "pi<22/7"],
    }


def invertible_recalibration_counterexample(kappa: float = 0.49, h: float = 1.0) -> dict:
    """Finite-lag failure despite an invertible detector and Gamma=a(Y).

    Y=cos(x)+kappa*cos(x)^2, |kappa|<1/2. The stationary phase is uniform,
    gamma=omega=1. The output is an invalid uncorrected finite-h functional.
    """
    kappa = _finite(kappa, "kappa")
    h = _finite(h, "h")
    if abs(kappa) >= .5 or h <= 0:
        raise ValueError("require |kappa|<1/2 and h>0")
    r = 2*kappa
    alpha = .5*math.exp(-4*h)*math.sin(2*h)
    beta = math.exp(-h)*math.sin(h)-alpha
    scale = math.sqrt(1-r*r)
    invalid = (alpha*alpha+2*alpha*beta/scale+beta*beta/scale**3)/h**2
    return {
        "kappa": kappa, "h": h, "invalid_finite_h_functional": invalid,
        "true_entropy_rate": 1.0, "minimum_recalibration_derivative": 1-abs(r),
        "pointwise_volatility_closure": True,
    }


def recalibration_certificate() -> dict:
    # At kappa=49/100,h=1: beta>29/100, alpha>0, sqrt(99)<10.
    lower = Fraction(29,100)**2 * Fraction(12500,99)
    return {
        "kappa": "49/100", "h": "1", "true_entropy_rate": "1",
        "invalid_finite_h_value_strictly_exceeds": str(lower),
        "strict_violation_margin_lower": str(lower-1),
        "proof_uses": ["e^-1 sin(1)>3/10", "0<e^-4 sin(2)/2<1/100", "sqrt(99)<10"],
    }


def parity_counterexample(speed: float = 1.0) -> dict:
    """Torus L=Delta-sin(p)*d_p+speed*sin(p)*d_q, pi proportional exp(cos p).

    Theta(q,p)=(q,-p) gives generalized detailed balance. The even detector is
    sin(q). A positive ordinary-reversal bound cannot be called physical entropy
    if Theta is instead the specified physical reversal.
    """
    speed = _finite(speed, "speed")
    expectation = float(i1(1.0) / i0(1.0))
    return {
        "ordinary_entropy_rate": speed * speed * expectation,
        "ordinary_local_defect_bound": speed * speed * expectation,
        "generalized_entropy_rate": 0.0,
        "physical_parity": "Theta(q,p)=(q,-p)",
        "observable_is_even_under_parity": True,
    }


def noisy_realized_qv(n: int, total_time: float = 1.0, noise_variance: float = 0.01,
                      gamma: float = 1.0, omega: float = 1.0) -> dict:
    """Exact expectation for sampled noisy stationary cos-circle data.

    Z_k=cos(phi(k*T/n))+eta_k, centered independent sensor noise Var(eta)=nu^2.
    This is a measurement-model obstruction, not a recommended estimator.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    total_time = _finite(total_time, "total_time")
    noise_variance = _finite(noise_variance, "noise_variance")
    gamma = _finite(gamma, "gamma")
    omega = _finite(omega, "omega")
    if total_time <= 0 or noise_variance < 0 or gamma <= 0:
        raise ValueError("require time,gamma>0 and noise_variance>=0")
    step = total_time / n
    clean = n * (1 - math.exp(-gamma * step) * math.cos(omega * step))
    sensor = 2 * n * noise_variance
    return {
        "n": n, "expected_realized_qv": clean + sensor,
        "clean_expected_realized_qv": clean, "sensor_contribution": sensor,
        "clean_continuous_qv_expectation": gamma * total_time,
    }


def report() -> dict:
    return {
        "status": "analytic countermodels with numerical checks; no physical experiment",
        "averaged_gamma": averaged_gamma_counterexample(),
        "averaged_gamma_rational_certificate": averaged_gamma_certificate(),
        "finite_h_rational_certificate": finite_h_certificate(),
        "finite_h_quadrature": finite_h_truncated_quadrature(),
        "invertible_recalibration": invertible_recalibration_counterexample(),
        "recalibration_rational_certificate": recalibration_certificate(),
        "generalized_parity": parity_counterexample(),
        "noisy_quadratic_variation": [noisy_realized_qv(n) for n in (100,1000,10000)],
        "equilibrium_negative_control": "g_h^2-K_h=-Var((P_h f-f)/h|U)<=0 exactly",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(report(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
