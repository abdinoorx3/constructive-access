"""Physical realizations and aliasing in a declared circulant/circle model class.

Floating-point illustrations accompany analytical proofs; these routines are
not interval-certified entropy estimators for arbitrary hidden processes.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def ring_minimum(n, gamma, omega):
    """Exact-form minimum over all circulant n-state CTMC displacement rates.

    Fix the FIRST Fourier eigenvalue -gamma+i*omega, ordinary reversal and
    uniform stationarity. Returns a finite-entropy nearest-neighbor minimizer.
    This is not a minimum over arbitrary hidden CTMCs with one observed moment.
    """
    if type(n) is not int or n < 3:
        raise ValueError('n must be an integer at least 3')
    gamma = _positive(gamma, 'gamma')
    omega = float(omega)
    if not math.isfinite(omega):
        raise ValueError('omega must be finite')
    theta = 2 * math.pi / n
    # 2 sin(theta/2)^2 is more accurate than 1-cos(theta) on fine rings.
    damping_factor = 2 * math.sin(theta / 2) ** 2
    traffic = gamma / damping_factor
    current = omega / math.sin(theta)
    bias = omega / gamma * math.tan(theta / 2)
    if abs(bias) >= 1:
        raise ValueError('no finite-entropy circulant realization at this n')
    forward, backward = (traffic + current) / 2, (traffic - current) / 2
    entropy = 2 * current * math.atanh(bias)
    return dict(n=n, gamma=gamma, omega=omega, forward=forward,
                backward=backward, entropy=entropy, circle_entropy=omega**2/gamma,
                model_class='circulant CTMC; fixed first Fourier eigenvalue')


def circulant_statistics(rates):
    """Uniform stationary EPR and first-mode eigenvalue for nonnegative rates.

    rates[j-1] is rate of displacement j mod n. Zero reverse rates produce
    infinite EPR. Disconnected supports are rejected, even though uniform
    stationarity and the algebraic inequality would still make sense.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or len(rates) < 2 or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError('at least two finite nonnegative displacement rates required')
    n = len(rates) + 1
    gcd = n
    for j, rate in enumerate(rates, 1):
        if rate > 0:
            gcd = math.gcd(gcd, j)
    if gcd != 1:
        raise ValueError('displacement support is not irreducible')
    theta = 2 * math.pi * np.arange(1, n) / n
    gamma = float(np.dot(rates, 2 * np.sin(theta/2)**2))
    omega = float(np.dot(rates, np.sin(theta)))
    entropy = 0.0
    for j in range(1, (n-1)//2 + 1):
        a, b = rates[j-1], rates[n-j-1]
        if a == b:
            continue
        if min(a, b) == 0:
            entropy = math.inf
            break
        entropy += (a-b) * math.log(a/b)
    return dict(n=n, gamma=gamma, omega=omega, entropy=float(entropy))


def ring_generator(n, gamma, omega):
    params = ring_minimum(n, gamma, omega)
    q = np.zeros((n, n))
    for j in range(n):
        q[j, (j+1) % n] = params['forward']
        q[j, (j-1) % n] = params['backward']
        q[j, j] = -params['forward']-params['backward']
    return q


def gated_ring_countermodel(amplitude=0.1, gamma=1.0, omega=0.5, exit_rate=0.01):
    """Five-state exact-covariance countermodel to an unrestricted circle minimum.

    Four active states have f=(1,1,-1,-1); a reversible hub has f=0.
    Active stationary mass A gives C(t)=A exp(-gamma*t) cos(omega*t).
    This architecture is noncirculant. It need not obey sigma>=omega^2/gamma.
    """
    amplitude = float(amplitude)
    gamma, exit_rate = _positive(gamma, 'gamma'), _positive(exit_rate, 'exit_rate')
    if not math.isfinite(amplitude) or not 0 < amplitude < 1 or exit_rate >= gamma:
        raise ValueError('0<amplitude<1 and 0<exit_rate<gamma required')
    ring = ring_minimum(4, gamma-exit_rate, omega)
    q = np.zeros((5, 5))
    q[:4,:4] = ring_generator(4, gamma-exit_rate, omega)
    q[np.arange(4),np.arange(4)] -= exit_rate
    q[:4,4] = exit_rate
    entry = amplitude*exit_rate/(1-amplitude)
    q[4,:4] = entry/4
    q[4,4] = -entry
    pi = np.array([amplitude/4]*4+[1-amplitude])
    detector = np.array([1.,1.,-1.,-1.,0.])
    return dict(generator=q, stationary=pi, detector=detector,
                entropy=amplitude*ring['entropy'], circle_entropy=float(omega)**2/gamma,
                amplitude=amplitude, gamma=gamma, omega=float(omega), exit_rate=exit_rate)


def sampled_mode(c0, c1, c2, delta=1.0):
    """Infer damping and PRINCIPAL frequency in the single-mode model only.

    Requires C(k delta)=A exp(-gamma k delta) cos(omega k delta), known A=C0.
    Higher moments do not automatically validate this architecture. Frequencies
    ±omega+2*pi*k/delta are aliases, with unbounded physical entropy.
    """
    amplitude = _positive(c0, 'c0')
    delta = _positive(delta, 'delta')
    if amplitude > 0.5:
        raise ValueError('bounded cosine realization requires 0<c0<=1/2')
    z, w = float(c1)/amplitude, float(c2)/amplitude
    if not all(math.isfinite(v) for v in (z, w)):
        raise ValueError('finite correlations required')
    r2 = 2*z*z-w
    if not 0 < r2 < 1 or z*z > r2:
        raise ValueError('not a positively damped real single-mode correlation')
    gamma = -math.log(r2)/(2*delta)
    theta = math.acos(max(-1.0, min(1.0, z/math.sqrt(r2))))
    omega = theta/delta
    return dict(amplitude=amplitude, delta=delta, gamma=gamma,
                principal_omega=omega, circle_entropy_minimum=omega**2/gamma,
                scope='single-mode circulant/circle class, not an unrestricted physical minimum')


def circle_covariance(times, gamma, omega, amplitude=0.5):
    gamma = _positive(gamma, 'gamma')
    omega, amplitude = float(omega), float(amplitude)
    if not math.isfinite(omega) or not 0 < amplitude <= 0.5:
        raise ValueError('finite frequency and amplitude in (0,1/2] required')
    t = np.asarray(times, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError('finite nonnegative times required')
    return amplitude*np.exp(-gamma*t)*np.cos(omega*t)


def phase_record_tv_bound(gaps, gamma, omega_difference, repeats=1):
    """Upper bound on whole sampled-state record TV for circle drift diffusion.

    Identical stationary starts, same gamma; gaps describe consecutive positive
    intervals. Wrapped normal TV <= unwrapped shifted normal TV <=
    |wrapped phase shift|/(2 sqrt(pi gamma t)); sequential coupling adds these.
    A repeat is an independent replication of this whole record. Numerical only.
    """
    gamma = _positive(gamma, 'gamma')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    gaps = np.asarray(gaps, dtype=float)
    if gaps.ndim != 1 or len(gaps) == 0 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError('positive finite observation gaps required')
    if not math.isfinite(float(omega_difference)):
        raise ValueError('finite frequency difference required')
    phases = np.remainder(float(omega_difference)*gaps + math.pi, 2*math.pi)-math.pi
    bound = repeats * np.sum(np.abs(phases)/(2*np.sqrt(math.pi*gamma*gaps)))
    return float(min(1, bound))


def pell_aliases(gamma=1.0, omega=0.5, count=8):
    """Near aliases at gaps 1 and sqrt(2), via p^2-2q^2=1.

    Difference 2*pi*q is an exact alias at gap 1; p-q*sqrt(2) is tiny at sqrt(2).
    The error is evaluated as 1/(p+sqrt(2)*q), avoiding cancellation.
    """
    gamma = _positive(gamma, 'gamma')
    if type(count) is not int or count < 1 or not math.isfinite(float(omega)):
        raise ValueError('positive count and finite frequency required')
    p, q = 1, 0
    rows = []
    for _ in range(count):
        p, q = 3*p+4*q, 2*p+3*q
        phase_error = 2*math.pi/(p+math.sqrt(2)*q)
        alias = omega+2*math.pi*q
        # Only the sqrt(2) interval differs, since the unit interval is exact.
        tv = min(1.0, phase_error/(2*math.sqrt(math.pi*gamma*math.sqrt(2))))
        rows.append(dict(p=p, q=q, omega=alias, entropy=alias**2/gamma,
                         second_gap_phase_error=phase_error,
                         two_interval_state_record_tv_upper=tv))
    return rows


def example(seed=6121013, trials=1000):
    """Reproducible algebra checks and architecture-scoped physical brackets."""
    mode = sampled_mode(0.5, 0.38, 0.18)
    gamma, omega = mode['gamma'], mode['principal_omega']
    rings = [ring_minimum(n, gamma, omega) for n in (16,20,40,100,1000)]
    q = ring_generator(20, gamma, omega)
    f = np.cos(2*math.pi*np.arange(20)/20)
    exact_cov = circle_covariance([0,1,2,3], gamma, omega)
    matrix_cov = [float(f@expm(t*q)@f/20) for t in (0,1,2,3)]
    rng = np.random.default_rng(seed)
    min_margin, max_error = math.inf, 0.0
    for _ in range(trials):
        n = int(rng.integers(3,31))
        rates = np.exp(rng.uniform(-4,4,n-1))
        stats = circulant_statistics(rates)
        lower = ring_minimum(n, stats['gamma'], stats['omega'])['entropy']
        min_margin = min(min_margin, stats['entropy']-lower)
        if n == 3:
            max_error = max(max_error, abs(stats['entropy']-lower))
    aliases = []
    for k in (0,1,2,10):
        frequency = omega+2*math.pi*k
        aliases.append(dict(k=k, omega=frequency, entropy=frequency**2/gamma,
                            max_grid_covariance_error=float(np.max(np.abs(
                                circle_covariance(np.arange(11),gamma,frequency)-
                                circle_covariance(np.arange(11),gamma,omega))))))
    # Exact two-gap counterexample to recovering |omega| from two cosine values.
    w1, w2 = math.pi*(math.sqrt(2)-1), math.pi*(math.sqrt(2)+1)
    two_gap = dict(omega1=w1, omega2=w2, gaps=[1,math.sqrt(2)],
                  cosine1=np.cos(w1*np.array([1,math.sqrt(2)])).tolist(),
                  cosine2=np.cos(w2*np.array([1,math.sqrt(2)])).tolist())
    counter = gated_ring_countermodel()
    cq, cp, cf = counter['generator'], counter['stationary'], counter['detector']
    counter_times = [0,0.2,1,2,5]
    counter_covariance = [float((cp*cf)@expm(t*cq)@cf) for t in counter_times]
    counter_summary = {key:(value.tolist() if isinstance(value,np.ndarray) else value)
                       for key,value in counter.items()}
    counter_summary['times'] = counter_times
    counter_summary['matrix_covariance'] = counter_covariance
    counter_summary['analytic_covariance'] = circle_covariance(
        counter_times,counter['gamma'],counter['omega'],counter['amplitude']).tolist()
    return dict(schema='physical-attainability-circulant-v1', seed=seed,
                model=mode, physical_minimizers=rings, noncirculant_countermodel=counter_summary, direct_matrix_covariance=matrix_cov,
                analytic_covariance=exact_cov.tolist(),
                max_matrix_covariance_error=float(np.max(np.abs(exact_cov-matrix_cov))),
                random_circulant_checks=dict(trials=trials,min_entropy_margin=min_margin,
                                             max_three_state_equality_error=max_error),
                exact_grid_aliases=aliases, two_incommensurate_cosine_counterexample=two_gap,
                pell_near_aliases=pell_aliases(),
                evidence='analytic model-class theorem plus floating-point checks; no universal physical minimum or measured-data certificate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(example(), indent=2, sort_keys=True)+'\n')

if __name__ == '__main__':
    main()
