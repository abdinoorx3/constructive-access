"""Executable adversarial checks for finite-window classical realizations."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import limits
import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson
from scipy.sparse.linalg import expm_multiply


class RealizationLimitsTests(unittest.TestCase):
    def test_exact_poisson_tail_certificate(self):
        for mu in [F(0), F(1, 10), F(3), F(21, 2), F(100)]:
            for k in [0, 1, 3, 10, 20, 50, 140]:
                bound = limits.poisson_tail_bound(mu, k)
                actual = 1.0 if k == 0 else poisson.sf(k-1, float(mu))
                self.assertLessEqual(actual, float(bound) + 2e-14)
                self.assertTrue(0 <= bound <= 1)

    def test_lipschitz_certificate_contract(self):
        for reversible in (False, True):
            cert = limits.lipschitz_certificate(F(3, 2), F(2), F(1, 10),
                                                3, reversible)
            self.assertLessEqual(cert["uniform_error_bound"], F(1, 10))
            self.assertGreaterEqual(cert["last_phase"], cert["rate"]*cert["horizon"])
            self.assertEqual(cert["deterministic_state_bound"],
                             6*(cert["last_phase"]+1))

    def test_smooth_certificate_contract(self):
        cert = limits.smooth_certificate(F(2), F(3), F(1, 25), 2, True)
        self.assertLessEqual(cert["uniform_error_bound"], F(1, 25))
        self.assertEqual(cert["poisson_clock_error"],
                         F(2)*F(3)/(2*cert["rate"]))
        self.assertLessEqual(poisson.sf(cert["last_phase"]-1,
                                       float(cert["rate"]*cert["horizon"])),
                             float(cert["truncation_error"]))
        large = limits.smooth_certificate(F(401,200), F(9,4), F(1,120000),
                                           2, True, tail_method="cantelli")
        self.assertLessEqual(large["uniform_error_bound"], F(1,120000))
        self.assertLessEqual(6000*large["uniform_error_bound"], F(1,20))

    def test_deterministic_lift_is_ctmc_and_reversible(self):
        table = [[F(0), F(1, 4), F(1)], [F(1), F(2, 3), F(0)]]
        model = limits.realization(table, F(3), F(1, 2))
        Q = model["generator"].toarray()
        self.assertTrue(np.allclose(Q.sum(axis=1), 0))
        self.assertTrue(np.all((Q-np.diag(np.diag(Q))) >= 0))
        pi = model["stationary"]
        self.assertTrue(np.allclose(pi@Q, 0))
        self.assertTrue(np.allclose(pi[:, None]*Q, pi[None, :]*Q.T))
        self.assertTrue(limits.detailed_balance_certificate(table,F(3),F(1,2))["valid"])
        self.assertTrue(set(model["detector"]) <= {0, 1})
        self.assertEqual(len(model["states"]), 8)
        self.assertAlmostEqual(model["preparation_kl"], np.log(2*(1+6+36)))
        for initial in model["initial"]:
            mask = initial > 0
            kl = np.sum(initial[mask]*np.log(initial[mask]/pi[mask]))
            self.assertAlmostEqual(kl, model["preparation_kl"])

    def test_poisson_expectation_matches_generator(self):
        row = [F(1), F(2, 3), F(1, 3), F(0), F(2, 5)]
        model = limits.realization([row], F(7, 3))
        for t in [0, .1, .7, 1.6]:
            actual = model["initial"][0] @ expm(model["generator"].toarray()*t) @ model["detector"]
            weights = poisson.pmf(np.arange(4), (7/3)*t)
            expected = np.dot(weights, [float(x) for x in row[:4]]) + poisson.sf(3,(7/3)*t)*float(row[4])
            self.assertAlmostEqual(actual, expected, places=13)

    def test_lipschitz_bound_for_nonsmooth_and_ramp_targets(self):
        T, r = F(2), F(40)
        times = np.linspace(0, 2, 91)
        for function, L in [(lambda t: abs(t-1)/2, F(1,2)),
                            (lambda t: min(1, max(0, 2*t-F(1,2))), F(2))]:
            K = int(r*T)
            row = [F(str(float(function(F(k)/r)))) for k in range(K+1)]
            model = limits.realization([row], r)
            values = expm_multiply(model["generator"], model["detector"],
                                   start=0, stop=2, num=len(times)) @ model["initial"][0]
            target = np.array([float(function(t)) for t in times])
            self.assertLessEqual(np.max(abs(values-target)), float(L)*np.sqrt(float(T/r))+1e-12)

    def test_reversible_coupling_error(self):
        rng = np.random.default_rng(65091)
        for _ in range(8):
            row = [F(int(x),100) for x in rng.integers(0,101,10)]
            base = limits.realization([row], F(4))
            reversible = limits.realization([row], F(4), F(1,3))
            for t in [.1, .5, 2.0]:
                x = base["initial"][0] @ expm(base["generator"].toarray()*t) @ base["detector"]
                y = reversible["initial"][0] @ expm(reversible["generator"].toarray()*t) @ reversible["detector"]
                self.assertLessEqual(abs(x-y), 1-np.exp(-t/3)+1e-12)

    def test_adaptive_transcript_bound(self):
        # A depth-five adaptive Bernoulli tree attains differing local errors.
        rng = np.random.default_rng(23097)
        depth, eps = 5, F(1, 20)
        p_law, q_law = {(): 1.0}, {(): 1.0}
        for _ in range(depth):
            new_p, new_q = {}, {}
            for history in p_law:
                p = float(rng.uniform())
                q = np.clip(p+float(eps)*rng.choice([-1,1]), 0, 1)
                for y in (0,1):
                    new_p[history+(y,)] = p_law[history]*(p if y else 1-p)
                    new_q[history+(y,)] = q_law[history]*(q if y else 1-q)
            p_law, q_law = new_p, new_q
        tv = sum(abs(p_law[h]-q_law[h]) for h in p_law)/2
        self.assertLessEqual(tv, float(limits.transcript_bound(eps,depth))+1e-13)

    def test_unprotected_no_reset_path_is_outside_contract(self):
        # Endpoint marginal imitation cannot reproduce arbitrary repeated
        # measurements: it makes no claim about their backaction or joint law.
        # A static hidden fair bit has the same one-time marginal as IID fair
        # quantum resets, but its two-record TV from them is 1/2.
        static = np.array([.5, 0, 0, .5])
        independent = np.full(4, .25)
        self.assertEqual(np.sum(abs(static-independent))/2, .5)

    def test_invalid_contracts_rejected(self):
        with self.assertRaises(TypeError):
            limits.smooth_certificate(.2, 1, F(1,10))
        with self.assertRaises(ValueError):
            limits.realization([[F(3,2),F(0)]], 1)
        with self.assertRaises(ValueError):
            limits.transcript_bound(F(1,10), -1)

    def test_cycle_dimension_threshold_is_rationally_certified(self):
        rejected = limits.cycle_admissibility_certificate(F(1,10),F(2),62)
        accepted = limits.cycle_admissibility_certificate(F(1,10),F(2),63)
        self.assertTrue(rejected["inadmissible"])
        self.assertTrue(accepted["admissible"])
        self.assertLess(rejected["margin_upper"],0)
        self.assertGreater(accepted["margin_lower"],0)

    def test_exact_form_cyclic_realization(self):
        for n in (63,64):
            model = limits.cyclic_realization(F(1,10),F(2),n,antipodal=(n==64))
            times = np.linspace(0,10,101)
            values = expm_multiply(model["generator"],model["detector"],
                                   start=0,stop=10,num=len(times)) @ model["initial"].T
            expected = (1+np.exp(-.1*times)*np.cos(2*times))/2
            self.assertTrue(np.allclose(values[:,0],expected,rtol=0,atol=3e-13))
            if n==64:
                self.assertTrue(np.allclose(values[:,1],1-expected,rtol=0,atol=3e-13))
            self.assertGreater(model["backward_rate"],0)
        with self.assertRaises(ValueError):
            limits.cyclic_realization(F(1,10),F(2),62)


if __name__ == "__main__":
    unittest.main()
