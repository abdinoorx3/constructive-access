"""Independent checks of frontier arguments; no producer implementation used.

Run: python -m unittest discover -s tests -p test_independent.py -v
These checks challenge implementations and small cases; proofs are in redteam.txt.
"""
from fractions import Fraction as F
from itertools import combinations, product
import math
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.integrate import quad
from scipy.special import gammainc
from scipy.stats import poisson


def stationary(q):
    a = q.T.copy()
    a[-1] = 1
    rhs = np.zeros(len(q)); rhs[-1] = 1
    return np.linalg.solve(a, rhs)


def cycle(n, gamma, omega):
    angle = 2 * math.pi / n
    total = gamma / (1 - math.cos(angle))
    difference = omega / math.sin(angle)
    if total + 1e-12 < abs(difference):
        raise ValueError("negative transition rate")
    a, b = (total + difference) / 2, (total - difference) / 2
    q = np.zeros((n, n))
    for i in range(n):
        q[i, (i + 1) % n] += a
        q[i, (i - 1) % n] += b
        q[i, i] -= a + b
    return q


def kl(p, q):
    if p == 0: return -math.log1p(-q)
    if p == 1: return -math.log(q)
    return p * math.log(p / q) + (1-p) * math.log((1-p)/(1-q))


def producer(name):
    path=Path(__file__).resolve().parents[1]/"code"/(name+".py")
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("review_"+name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class IndependentFrontierTests(unittest.TestCase):
    def test_antichain_bound_is_tight_for_general_positive_factorizations(self):
        for n in range(2, 9):
            k = n // 2
            sets = list(map(set, combinations(range(n), k)))
            self.assertEqual(len(sets), math.comb(n, k))
            for a in sets:
                for b in sets:
                    value = F(len(a-b), k)
                    self.assertEqual(value == 0, a == b)
                    if a != b: self.assertGreaterEqual(value, F(1, k))

    def test_threshold_inclusion_bound_and_optimized_constant(self):
        rng = np.random.default_rng(487112)
        for _ in range(300):
            n = 8
            eps = rng.uniform(.0001, .2)
            tau = math.sqrt(eps)
            b_i = rng.uniform(0, 1, n)
            s_i = b_i < tau
            b_j = rng.uniform(0, 1, n)
            b_j[s_i] *= tau
            # Maximize alpha b_j subject to alpha b_i <= epsilon.
            result = linprog(-b_j, A_ub=[b_i], b_ub=[eps],
                             A_eq=[np.ones(n)], b_eq=[1], bounds=(0, None),
                             method="highs")
            if result.success:
                self.assertLessEqual(-result.fun, 2 * math.sqrt(eps)-eps + 1e-10)

    def test_quantum_grid_has_claimed_zero_pattern(self):
        for number in range(3, 14):
            phases = np.arange(number) * 2 * math.pi / number
            matrix = np.sin((phases[:, None] - phases[None, :]) / 2) ** 2
            self.assertLess(np.max(np.diag(matrix)), 1e-25)
            self.assertAlmostEqual(np.min(matrix[~np.eye(number, dtype=bool)]),
                                   math.sin(math.pi/number)**2, places=13)

    def test_exact_cycle_realizes_damped_qubit_occupation(self):
        for n in (4, 6, 10, 20, 64):
            omega = 1.7
            gamma = omega * math.tan(math.pi/n) * 1.05
            q = cycle(n, gamma, omega)
            detector = (1 + np.cos(2*math.pi*np.arange(n)/n))/2
            for time in (0, .1, .9, 2.3, 5):
                observed = expm(q*time) @ detector
                target = (1 + math.exp(-gamma*time)*math.cos(omega*time))/2
                self.assertAlmostEqual(observed[0], target, places=12)
                self.assertAlmostEqual(observed[n//2], 1-target, places=12)

    def test_cycle_rate_positivity_boundary_is_necessary(self):
        for n in (4, 6, 10, 20):
            with self.assertRaises(ValueError):
                cycle(n, .99*math.tan(math.pi/n), 1)

    def test_resolvent_wedge_transform_and_rational_distance(self):
        rng = np.random.default_rng(8831)
        for n in range(3, 11):
            slope = math.tan(math.pi/n)
            for _ in range(50):
                a = rng.uniform(.001, 10)
                b = rng.uniform(-a/slope, a/slope)
                z = 1/(1+a-1j*b)
                self.assertLessEqual(abs(z)**2-z.real+slope*abs(z.imag), 1e-12)
            mu = 1/(1 + slope/2 - 1j)
            margin = abs(mu)**2-mu.real+slope*mu.imag
            d = margin/(3+slope)
            self.assertGreater(d, 0)
            for _ in range(50):
                a = rng.uniform(.001, 10)
                b = rng.uniform(-a/slope, a/slope)
                z = 1/(1+a-1j*b)
                self.assertGreaterEqual(abs(mu-z)+1e-13, d)

    def test_erlang_moments_satisfy_characteristic_recurrence(self):
        rng = np.random.default_rng(45681)
        for n in range(2, 9):
            for _ in range(8):
                q = rng.uniform(.01, 3, (n, n)); np.fill_diagonal(q, 0)
                q -= np.diag(q.sum(axis=1))
                r = np.linalg.inv(np.eye(n)-q)
                p = np.poly(r)[::-1]
                alpha = rng.dirichlet(np.ones(n)); detector = rng.uniform(0, 1, n)
                m = [alpha @ np.linalg.matrix_power(r, k) @ detector
                     for k in range(n+2)]
                self.assertLess(abs(sum(p[j]*m[j] for j in range(n+1))), 1e-8)
                self.assertLess(abs(sum(p[j]*m[j+1] for j in range(n+1))), 1e-8)
                self.assertLessEqual(sum(abs(p)), 2**n+1e-8)

    def test_conjugate_pair_root_distance_improvement(self):
        rng = np.random.default_rng(644199)
        for _ in range(1000):
            mu = complex(rng.uniform(.1, .8), rng.uniform(.05, .45))
            root = complex(rng.uniform(-.5, .8), rng.uniform(0, .8))
            d = min(mu.imag, abs(mu-root), abs(mu-root.conjugate()))
            self.assertGreaterEqual(abs(mu-root)*abs(mu-root.conjugate())+1e-15,
                                    d*mu.imag)

    def test_erlang_tail_identity(self):
        for shape in range(1, 15):
            for x in (.1, 1, 5, 20):
                survival = math.exp(-x)*sum(x**j/math.factorial(j)
                                           for j in range(shape))
                self.assertAlmostEqual(survival, 1-gammainc(shape, x), places=13)

    def test_poisson_clock_generator_matches_analytic_law(self):
        rate, horizon = 3., 2.
        phases = math.ceil(rate*horizon)+1
        functions = [lambda t: (1+math.cos(2*t))/2,
                     lambda t: min(.9, .1+.4*t)]
        dim = 2*len(functions)*phases
        q = np.zeros((dim, dim)); detector = np.arange(dim)%2
        for arm, function in enumerate(functions):
            for k in range(phases-1):
                prob = function(min((k+1)/rate, horizon))
                for y in range(2):
                    row = 2*(arm*phases+k)+y
                    q[row, row] = -rate
                    q[row, 2*(arm*phases+k+1)] = rate*(1-prob)
                    q[row, 2*(arm*phases+k+1)+1] = rate*prob
        for arm, function in enumerate(functions):
            alpha = np.zeros(dim)
            alpha[2*arm*phases] = 1-function(0)
            alpha[2*arm*phases+1] = function(0)
            for time in (0, .2, 1, 2):
                pmf = poisson.pmf(np.arange(phases-1), rate*time)
                direct = sum(pmf[k]*function(k/rate) for k in range(phases-1))
                direct += poisson.sf(phases-2, rate*time)*function(horizon)
                self.assertAlmostEqual(alpha @ expm(q*time) @ detector, direct, places=13)
                self.assertLessEqual(abs(direct-function(time)), math.sqrt(time/rate)+1e-12)

    def test_poisson_lipschitz_bound_for_nonsmooth_trace(self):
        horizon = 2.
        function = lambda t: abs(t-1)/2
        lipschitz = .5
        for rate in (2, 5, 10, 40):
            kmax = math.ceil(rate*horizon)
            for time in np.linspace(0, horizon, 51):
                weights = poisson.pmf(np.arange(kmax), rate*time)
                value = sum(weights[k]*function(k/rate) for k in range(kmax))
                value += poisson.sf(kmax-1, rate*time)*function(horizon)
                self.assertLessEqual(abs(value-function(time)),
                                     lipschitz*math.sqrt(time/rate)+1e-12)

    def test_adaptive_fresh_trial_tv_coupling(self):
        epsilon = .07
        for trials in range(1, 8):
            total = 0.
            for history in product((0, 1), repeat=trials):
                p, q = 1., 1.
                for k, y in enumerate(history):
                    parameter = .2 if sum(history[:k]) % 2 else .7
                    shifted = parameter + (epsilon if k % 2 else -epsilon)
                    p *= parameter if y else 1-parameter
                    q *= shifted if y else 1-shifted
                total += abs(p-q)/2
            self.assertLessEqual(total, 1-(1-epsilon)**trials+1e-12)

    def test_entropy_symmetrization_constant(self):
        for x in np.concatenate(([0], np.geomspace(1e-6, .999999, 100))):
            if x == 0: continue
            path_rate = .5*((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))
            sigma = x*math.log((1+x)/(1-x))
            self.assertLessEqual(path_rate, sigma/4+1e-14)
        x = 1e-4
        path_rate = .5*((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))
        sigma = x*math.log((1+x)/(1-x))
        self.assertAlmostEqual(path_rate/sigma, .25, places=8)

    def test_symmetrization_preserves_stationarity_and_exit_rates(self):
        rng = np.random.default_rng(98123)
        for n in range(2, 10):
            q = rng.uniform(.01, 3, (n, n)); np.fill_diagonal(q, 0)
            q -= np.diag(q.sum(axis=1)); pi = stationary(q)
            reverse = (q.T*pi[None, :])/pi[:, None]
            sym = (q+reverse)/2
            self.assertTrue(np.allclose(np.diag(sym), np.diag(q)))
            self.assertTrue(np.allclose(sym.sum(axis=1), 0))
            self.assertTrue(np.allclose(pi @ sym, 0))
            self.assertTrue(np.allclose(pi[:, None]*sym, (pi[:, None]*sym).T))

    def test_reversible_correlation_is_hausdorff_moment_sequence(self):
        rng = np.random.default_rng(98124)
        for n in range(2, 10):
            pi = rng.dirichlet(np.ones(n))
            edge = rng.uniform(.01, .1, (n, n)); edge = (edge+edge.T)/2
            np.fill_diagonal(edge, 0)
            q = edge/pi[:, None]; q -= np.diag(q.sum(axis=1))
            detector = rng.uniform(-1, 1, n)
            correlations = np.array([pi @ (detector*(expm(q*k*.05) @ detector))
                                     for k in range(8)])
            for order in range(7):
                self.assertGreaterEqual(float(np.min((-1)**order*np.diff(correlations, n=order))),
                                        -1e-11)

    def test_binary_correlation_entropy_bound_on_random_networks(self):
        rng = np.random.default_rng(710239)
        for n in range(3, 10):
            for _ in range(20):
                q = rng.lognormal(0, 1.5, (n, n)); np.fill_diagonal(q, 0)
                q -= np.diag(q.sum(axis=1)); pi = stationary(q)
                rev = (q.T*pi[None, :])/pi[:, None]; sym = (q+rev)/2
                flow = pi[:, None]*q
                sigma = sum((flow[i,j]-flow[j,i])*math.log(flow[i,j]/flow[j,i])
                            for i in range(n) for j in range(i+1,n))
                detector = rng.uniform(-1, 1, n)
                time = rng.uniform(.01, 1)
                p = (1+pi @ (detector*(expm(q*time)@detector)))/2
                ref = (1+pi @ (detector*(expm(sym*time)@detector)))/2
                self.assertLessEqual(4*kl(p, ref), sigma*time+1e-10)

    def test_reversible_poisson_lift_has_exact_detailed_balance(self):
        samples = [[F(0), F(1,3), F(2,5), F(1)],
                   [F(1), F(2,3), F(1,5), F(0)]]
        rate, back = F(7,3), F(1,19)
        for row in samples:
            for k in range(len(row)-1):
                for y, z in product((0,1), repeat=2):
                    py = row[k] if y else 1-row[k]
                    pz = row[k+1] if z else 1-row[k+1]
                    self.assertEqual((rate/back)**k*py*rate*pz,
                                     (rate/back)**(k+1)*pz*back*py)

    def test_reversible_poisson_lift_backward_coupling_bound(self):
        rng = np.random.default_rng(993115)
        for phases in (5, 10, 20):
            r, d = 3., .07
            forward = np.zeros((phases, phases))
            backward = np.zeros((phases, phases))
            for k in range(phases):
                if k+1 < phases: forward[k,k+1] = r
                if k > 0: backward[k,k-1] = d
            q = forward-np.diag(forward.sum(axis=1))
            rev = q+backward-np.diag(backward.sum(axis=1))
            detector = rng.uniform(0,1,phases)
            for t in (0,.1,1,2.5):
                difference = abs(((expm(q*t)-expm(rev*t))@detector)[0])
                self.assertLessEqual(difference, 1-math.exp(-d*t)+1e-12)

    def test_entropy_dual_polynomial_majorant(self):
        rng = np.random.default_rng(417298)
        for n in range(3,8):
            for _ in range(10):
                q = rng.lognormal(0,1,(n,n)); np.fill_diagonal(q,0)
                q -= np.diag(q.sum(axis=1)); pi = stationary(q)
                reverse = (q.T*pi[None,:])/pi[:,None]; sym=(q+reverse)/2
                detector = rng.uniform(-1,1,n)
                r = (1+(pi@detector)**2)/2
                lags = np.arange(1,6)*.08
                weights = rng.dirichlet(np.ones(5))
                p = np.array([(1+pi@(detector*(expm(q*t)@detector)))/2
                              for t in lags])
                ref = np.array([(1+pi@(detector*(expm(sym*t)@detector)))/2
                                for t in lags])
                h1, h0 = p/ref, (1-p)/(1-ref)
                coeff = np.r_[sum(weights*(h0+r*(h1-h0))),
                              weights*(1-r)*(h1-h0)]
                candidates = [0.,1.]
                derivative = np.arange(1,len(coeff))*coeff[1:]
                for root in np.roots(derivative[::-1]):
                    if abs(root.imag)<1e-8 and 0<root.real<1:
                        candidates.append(root.real)
                upper = max(np.polynomial.polynomial.polyval(x,coeff)
                            for x in candidates)
                self.assertGreaterEqual(upper+1e-11, sum(weights*(ref*h1+(1-ref)*h0)))
                dual = sum(weights*(p*np.log(h1)+(1-p)*np.log(h0)))-math.log(upper)
                flow=pi[:,None]*q
                sigma=sum((flow[i,j]-flow[j,i])*math.log(flow[i,j]/flow[j,i])
                          for i in range(n) for j in range(i+1,n))
                self.assertLessEqual(4*dual, sigma*sum(weights*lags)+1e-10)

    def test_reversible_refresh_makes_clock_irreducible_at_arbitrary_small_cost(self):
        pi = np.array([.1,.2,.3,.4])
        q = np.array([[-2.,2,0,0],[1,-1,0,0],
                      [0,0,-4/3,4/3],[0,0,1,-1]])
        self.assertTrue(np.allclose(pi@q,0))
        projection = np.ones((4,1))*pi[None,:]
        refresh = .013*(projection-np.eye(4))
        combined = q+refresh
        self.assertGreater(np.min(combined[~np.eye(4,dtype=bool)]),0)
        self.assertTrue(np.allclose(pi[:,None]*combined,(pi[:,None]*combined).T))
        detector=np.array([0.,1,.3,.7])
        for time in (0,.2,1,8):
            expected=math.exp(-.013*time)*expm(q*time)+(1-math.exp(-.013*time))*projection
            self.assertTrue(np.allclose(expm(combined*time),expected))
            discrepancy=max(abs((expm(combined*time)-expm(q*time))@detector))
            self.assertLessEqual(discrepancy,1-math.exp(-.013*time)+1e-12)

    def test_nonstationary_housekeeping_bound_and_sharp_half_factor(self):
        ratios=[]
        for eps in (1e-2,1e-4,1e-6,1e-8):
            delta=eps*eps
            q=np.array([[0,1-delta,delta],[1-eps,0,1],[eps,1-eps+delta,0.]])
            q-=np.diag(q.sum(axis=1))
            self.assertTrue(np.allclose(np.ones(3)@q,0))
            reverse=q.T; sym=(q+reverse)/2
            hk=sum(q[0,j]*math.log(q[0,j]/reverse[0,j]) for j in (1,2))
            path=sum(q[0,j]*math.log(q[0,j]/sym[0,j]) for j in (1,2))
            self.assertLessEqual(path,hk/2+1e-14)
            ratios.append(path/hk)
        self.assertGreater(ratios[-1],.4999)
        self.assertLess(ratios[-1],.5001)

    def test_nonstationary_initial_entropy_plus_housekeeping_controls_record(self):
        rng=np.random.default_rng(371618)
        for n in range(3,7):
            for _ in range(8):
                q=rng.lognormal(0,1,(n,n)); np.fill_diagonal(q,0)
                q-=np.diag(q.sum(axis=1)); pi=stationary(q)
                rev=(q.T*pi[None,:])/pi[:,None]; sym=(q+rev)/2
                hk=np.zeros(n)
                for i in range(n):
                    hk[i]=sum(q[i,j]*math.log(q[i,j]/rev[i,j]) for j in range(n) if j!=i)
                self.assertGreaterEqual(min(hk),-1e-12)
                nu=rng.dirichlet(np.ones(n)); f=rng.uniform(-1,1,n); t=.4
                p=(1+nu@(f*(expm(q*t)@f)))/2
                ref=(1+pi@(f*(expm(sym*t)@f)))/2
                initial=sum(nu*np.log(nu/pi))
                integrated=quad(lambda s: nu@expm(q*s)@hk,0,t,epsabs=1e-11)[0]
                self.assertLessEqual(kl(p,ref),initial+integrated/2+1e-11)

    def test_exact_dual_log_enclosures_and_polynomial_upper(self):
        module=producer("energy_certificate")
        import mpmath as mp
        mp.mp.dps=100
        for x in (F(1,123),F(999,1000),F(2),F(31,7),F(10**20,3)):
            low,high=module.log_interval(x)
            value=mp.log(mp.mpf(x.numerator)/x.denominator)
            self.assertLessEqual(mp.mpf(low.numerator)/low.denominator,value)
            self.assertGreaterEqual(mp.mpf(high.numerator)/high.denominator,value)
        rng=np.random.default_rng(736516)
        for degree in range(1,11):
            coeff=[F(int(x),13) for x in rng.integers(-30,31,degree+1)]
            upper=module.polynomial_upper(coeff,depth=3)
            derivative=np.arange(1,len(coeff))*np.array(list(map(float,coeff[1:])))
            points=[0.,1.]
            for z in np.roots(derivative[::-1]):
                if abs(z.imag)<1e-10 and 0<z.real<1: points.append(z.real)
            numeric=max(np.polynomial.polynomial.polyval(x,list(map(float,coeff))) for x in points)
            self.assertGreaterEqual(float(upper)+1e-12,numeric)

    def test_reference_mean_guard_and_curvature_counterexample(self):
        module=producer("energy_certificate")
        curve=module.curvature_example()
        self.assertGreater(curve["entropy_rate_lower_float"],0)
        probs=[float((lo+hi)/2) for lo,hi in curve["exact_probability_intervals"]]
        self.assertTrue(all(p>.5 for p in probs))
        self.assertTrue(probs[0]>probs[1]>probs[2])
        self.assertGreater(2*probs[1]-probs[0]-probs[2],0)
        cert=curve["certificate"]
        fields=("lags","step","weights","intervals","mean_interval","scores","depth")
        inputs={k:cert[k] for k in fields}; inputs["mean_interval"]=(F(-1,10),F(1,10))
        narrow=module.certify(**inputs)
        with self.assertRaises(ValueError):module.resource_frontier(narrow)
        self.assertIn("preparation_plus_half_housekeeping_lower",
                      module.resource_frontier(narrow,reference_mean_calibrated=True))

    def test_hierarchy_qubit_intervals_against_high_precision(self):
        module=producer("hierarchy")
        import mpmath as mp
        mp.mp.dps=90
        for size in (4,7,8,13):
            intervals=module.qubit_probabilities(size,F(1000))
            for k,(lo,hi) in enumerate(intervals,1):
                phase=2*mp.pi*k/size
                value=(1-mp.exp(-phase/1000)*mp.cos(phase))/2
                self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,value)
                self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,value)

    def test_exact_binomial_geometric_bounds_against_scipy(self):
        module=producer("hierarchy")
        from scipy.stats import binom
        for n in (10,100,1000):
            for p in (F(1,100),F(1,4),F(3,4),F(99,100)):
                for k in set((0,n//8,n//2,7*n//8,n)):
                    lower=module.binomial_tail_upper(n,k,p,"lower")
                    upper=module.binomial_tail_upper(n,k,p,"upper")
                    self.assertGreaterEqual(float(lower)+1e-12,binom.cdf(k,n,float(p)))
                    self.assertGreaterEqual(float(upper)+1e-12,binom.sf(k-1,n,float(p)))

    def test_normalized_spectral_characteristic_and_pair_bound(self):
        rng=np.random.default_rng(215467)
        for n in range(3,10):
            q=rng.uniform(.01,3,(n,n)); np.fill_diagonal(q,0)
            q-=np.diag(q.sum(axis=1)); resolvent=np.linalg.inv(np.eye(n)-q)
            polynomial=np.poly(resolvent)
            self.assertAlmostEqual(sum(abs(polynomial)),abs(np.polyval(polynomial,-1)),places=9)
        for _ in range(1000):
            mu=complex(rng.uniform(.1,.7),rng.uniform(.05,.4))
            nu=(1-mu)/(1+mu); b=abs(nu.imag)
            z=complex(rng.uniform(.01,.8),rng.uniform(.01,.5))
            w=(1-z)/(1+z)
            d=min(b,abs(nu-w),abs(nu-w.conjugate()))
            product=abs(nu-w)*abs(nu-w.conjugate())
            self.assertGreaterEqual(product+1e-14,d*(2*b-d))
            self.assertAlmostEqual(abs(mu-z)/abs(1+z),abs(1+mu)*abs(nu-w)/2,places=13)

    def test_thermodynamic_exact_model_and_count_certificate(self):
        module=producer("thermodynamic")
        n=12; forward=F(4); backward=F(1); t=F(7,10)
        q=np.zeros((n,n))
        for i in range(n):
            q[i,(i+1)%n]=float(forward); q[i,(i-1)%n]=float(backward)
            q[i,i]=-float(forward+backward)
        f=np.r_[np.ones(n//2),-np.ones(n//2)]
        true=(1+f@expm(q*float(t))@f/n)/2
        low,high=module.ring_same_bounds(n,forward,backward,t)
        self.assertLessEqual(float(low)-1e-12,true)
        self.assertGreaterEqual(float(high)+1e-12,true)
        result=module.from_counts(5000,2560,5000,3680,F(1,50),F(1,20),F(8,5),F(21,10))
        self.assertGreater(result["rate_lower"],0)
        p1,p2=.512+.02,.736-.02; pooled=(p1+p2)/2
        expected=4*(kl(p1,pooled)+kl(p2,pooled))/2/1.85
        self.assertAlmostEqual(float(result["rate_lower"]),expected,places=12)

    def test_score_design_and_count_box_remain_exactly_certified(self):
        module=producer("energy_certificate")
        seed=module.example()["certificate"]
        kwargs=dict(lags=seed["lags"],step=seed["step"],weights=seed["weights"],
                    records=[5000,5000],successes=[2560,3680],radius=F(1,40),
                    alpha=F(1,20),scores=seed["scores"])
        clean=module.from_counts(**kwargs)
        enlarged=module.from_counts(**kwargs,calibration_radii=[F(1,100)]*2)
        self.assertGreater(clean["certificate"]["record_kl_lower"],0)
        self.assertLessEqual(enlarged["certificate"]["record_kl_lower"],
                             clean["certificate"]["record_kl_lower"])
        kwargs["radius"]=F(1,100)
        with self.assertRaises(ValueError): module.from_counts(**kwargs)
        proposal=module.design(seed["lags"],seed["step"],seed["weights"],
                               seed["intervals"],grid=3,depth=4)
        cert=proposal["certificate"]
        fields=("lags","step","weights","intervals","mean_interval","scores","depth")
        self.assertTrue(module.verify(cert,**{k:cert[k] for k in fields}))
        for base in cert["reversible_baseline_interval"]:
            for x in np.linspace(0,1,1001):
                total=sum(float(w)*(float(h0)+(float(h1)-float(h0))*(float(base)+(1-float(base))*x**k))
                          for k,w,(h0,h1) in zip(cert["lags"],cert["weights"],cert["scores"]))
                self.assertLessEqual(total,float(cert["normalizer_upper"])+1e-12)


if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentFrontierTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={
        "scope":"independent mathematical and computational review; no laboratory data",
        "tests_run":result.testsRun,
        "failures":len(result.failures),
        "errors":len(result.errors),
        "passed":result.wasSuccessful(),
        "reproduction":"python tests/test_independent.py",
        "review_document":"research/redteam.txt",
        "checks":[
            "robust antichain inclusion bound and exact support sharpness",
            "classical cycle matching damped quantum populations",
            "resolvent moments, spectral lens, normalized characteristic factors",
            "finite Poisson clock and adaptive fresh-record transcript coupling",
            "exact reversible lift and irreducible refresh strengthening",
            "stationary entropy comparison and reversible Hausdorff moments",
            "nonstationary preparation/housekeeping budget and sharp half factor",
            "exact rational logarithms and Bernstein polynomial upper bounds",
            "reference-mean calibration guard and positive-monotone curvature witness",
            "high-precision quantum-grid and exact binomial-tail cross-checks",
            "independent matrix-exponential check of thermodynamic count certificate"
        ],
        "limitations":["internal review only","historical priority unresolved",
                       "synthetic models and counts","physical assumptions require experimental validation"]
    }
    path=Path(__file__).resolve().parents[1]/"results"/"independent.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    sys.exit(not result.wasSuccessful())
