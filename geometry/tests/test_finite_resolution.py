"""Executable checks of finite-h enclosures, exact confidence and scope guards."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from decimal import Decimal, localcontext
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import finite_resolution as fr


class FiniteResolutionTests(unittest.TestCase):
    def protocol(self, **kwargs):
        args = dict(h=F(1, 20), q_abs_max=F(4, 5), envelope_floor=F(9, 25),
                    generator_bounds=(F(19, 6), F(19, 6), 10, 10),
                    global_gamma_upper=1)
        args.update(kwargs)
        return fr.make_protocol(**args)

    def test_exact_inputs_and_outward_elementary_functions(self):
        for bad in (True, 0.5, object()):
            with self.assertRaises(TypeError):
                fr.exact(bad)
        with localcontext() as c:
            c.prec = 100
            for x in (F(1, 73), F(1), F(2), F(20), F(1234, 7)):
                lo, hi = fr.log_interval(x)
                true = (Decimal(x.numerator)/Decimal(x.denominator)).ln()
                self.assertLessEqual(Decimal(lo.numerator)/Decimal(lo.denominator), true)
                self.assertGreaterEqual(Decimal(hi.numerator)/Decimal(hi.denominator), true)
                root = fr.sqrt_upper(x)
                self.assertGreaterEqual(root*root, x)
                self.assertLess(root-fr.exact(1)/(1 << 90),
                                F(str((Decimal(x.numerator)/Decimal(x.denominator)).sqrt())))

    def test_bias_against_random_markov_semigroups(self):
        import numpy as np
        from scipy.linalg import expm
        rng = np.random.default_rng(711)
        for n in (3, 5, 8):
            rates = rng.uniform(.05, 2, (n, n))
            np.fill_diagonal(rates, 0)
            L = rates-np.diag(rates.sum(axis=1))
            A = L.T.copy(); b = np.zeros(n)
            A[-1] = 1; b[-1] = 1
            pi = np.linalg.solve(A, b)
            reverse = (L.T*pi[None, :])/pi[:, None]
            f = rng.uniform(-1, 1, n)
            forward = L@f; backward = reverse@f
            bounds = [max(abs(forward)), max(abs(backward)),
                      max(abs(L@forward)), max(abs(reverse@backward))]
            bounds = [F(str(x+1e-10)) for x in bounds]
            for h in (F(1, 1000), F(1, 20), F(1, 2)):
                P = expm(float(h)*L); R = expm(float(h)*reverse)
                dp = (P@f-f)/float(h); dm = (R@f-f)/float(h)
                bias = fr.semigroup_bias(h, bounds)
                self.assertLessEqual(max(abs((dp+dm-forward-backward)/2)),
                                     float(bias['g_bias_upper'])+1e-9)
                self.assertLessEqual(max(abs(dp*dm-forward*backward)),
                                     float(bias['K_bias_upper'])+1e-9)

    def test_asymmetric_product_bias_has_no_unneeded_quadratic_term(self):
        b = fr.semigroup_bias(F(2, 7), (2, 3, 5, 11))
        self.assertEqual(b['g_bias_upper'], F(8, 7))
        self.assertEqual(b['K_bias_upper'], F(37, 7))

    def test_score_factorization_and_exact_bounds(self):
        p = self.protocol(h=F(1, 2), q_abs_max=1, envelope_floor=F(1, 3))
        lo, hi = p['corrected_score_bounds']
        for triple in product((-1, F(-1, 2), 0, F(1, 2), 1), repeat=3):
            for q in (-1, 0, 1):
                for a in (F(1, 3), 1, 3):
                    s = fr.corrected_score(triple, q, a, p)
                    self.assertLessEqual(lo, s)
                    self.assertLessEqual(s, hi)
                    dm, dp = (triple[0]-triple[1])/p['h'], (triple[2]-triple[1])/p['h']
                    raw = q*(dm+dp)-q*q-dm*dp
                    correction = 2*abs(q)*p['g_bias_upper']+p['K_bias_upper']
                    self.assertEqual(s, (raw-correction)/a)

    def test_excluded_centers_still_cost_records(self):
        p = self.protocol(coordinate_readouts_per_record=7)
        s = fr.corrected_score((-1, 0, 1), 1000, 0, p, retained=False)
        self.assertEqual(s, 0)
        cert = fr.certify_scores([0]*100, p)
        self.assertEqual(cert['triple_detector_readouts'], 300)
        self.assertEqual(cert['coordinate_detector_readouts'], 700)
        self.assertEqual(cert['declared_evolution_time'], 10)
        self.assertEqual(cert['independent_preparations'], 100)

    def test_shared_center_unbiased_noise_lowers_the_score(self):
        p = self.protocol(h=F(1, 2), q_abs_max=1, envelope_floor=1,
                          readout_bounds=(-2, 2),
                          detector_contract='conditionally_unbiased_noise_external_center_rule')
        base = (F(-1, 4), F(1, 3), F(2, 5))
        q, a = F(1, 4), F(3, 2)
        true_score = fr.corrected_score(base, q, a, p)
        noisy = [fr.corrected_score(tuple(x+e for x, e in zip(base, eps)), q, a, p)
                 for eps in product((F(-1, 5), F(1, 5)), repeat=3)]
        self.assertEqual(sum(noisy, F(0))/len(noisy), true_score-F(1, 25)/(p['h']**2*a))
        self.assertIsNone(p['second_moment_upper'])

    def test_reject_unsupported_detector_selection_and_tampering(self):
        with self.assertRaises(ValueError):
            self.protocol(detector_contract='same_noisy_center_regression')
        with self.assertRaises(ValueError):
            self.protocol(selection='same_data_fit')
        p = self.protocol()
        p['K_bias_upper'] = 0
        with self.assertRaises(ValueError):
            fr.certify_summary(100, 0, p)

    def test_bad_q_and_envelope_are_rejected(self):
        p = self.protocol()
        with self.assertRaises(ValueError):
            fr.corrected_score((0, 0, 0), 1, 1, p)
        with self.assertRaises(ValueError):
            fr.corrected_score((0, 0, 0), 0, F(1, 10), p)
        with self.assertRaises(ValueError):
            fr.corrected_score((0, 0, 2), 0, 1, p)

    def test_bounded_readout_error_with_rounded_center_geometry(self):
        p = self.protocol(h=F(1, 10))
        delta = F(1, 100)
        certified = fr.bounded_readout_score_error(delta, p)
        for triple in product((F(-1, 2), 0, F(1, 2)), repeat=3):
            for signs in product((-1, 1), repeat=3):
                stored = tuple(x+sign*delta for x, sign in zip(triple, signs))
                z = stored[1]; q = -z
                envelope = 1-max(F(0), abs(z)-delta)**2
                self.assertGreaterEqual(envelope, 1-triple[1]**2)
                ideal = fr.corrected_score(triple, q, envelope, p)
                approximate = fr.corrected_score(stored, q, envelope, p)
                self.assertLessEqual(abs(ideal-approximate), certified)

    def test_rounding_defines_rule_even_at_retention_boundary(self):
        p = self.protocol(h=F(1, 25))
        delta = F(1, 1024)
        true = (F(1, 3), F(4, 5)+delta/2, F(1, 4))
        stored = (true[0], F(4, 5), true[2])
        q = -stored[1]
        envelope = 1-(abs(stored[1])-delta)**2
        self.assertGreaterEqual(envelope, 1-true[1]**2)
        # Actual selector retains by stored center; both ideal and approximate
        # score use it, even though the unrounded |Y|<=.8 rule would exclude.
        ideal = fr.corrected_score(true, q, envelope, p, retained=True)
        approximate = fr.corrected_score(stored, q, envelope, p, retained=True)
        self.assertLessEqual(abs(ideal-approximate), fr.bounded_readout_score_error(delta, p))

    def test_empirical_bernstein_matches_primary_theorem_and_aggregate(self):
        p = self.protocol(h=1, q_abs_max=0, envelope_floor=1,
                          generator_bounds=(0, 0, 0, 0))
        samples = [F(-1, 3), F(1, 7), F(1, 2)]*20
        a = fr.certify_scores(samples, p, concentration='empirical_bernstein')
        b = fr.certify_totals(len(samples), sum(samples, F(0)),
                              sum((x*x for x in samples), F(0)), p,
                              concentration='empirical_bernstein')
        self.assertEqual(a, b)
        n = len(samples); mean = sum(samples, F(0))/n
        variance = sum(((x-mean)**2 for x in samples), F(0))/(n-1)
        self.assertEqual(a['empirical_variance'], variance)
        width = a['corrected_score_bounds'][1]-a['corrected_score_bounds'][0]
        radius = math.sqrt(2*float(variance)*math.log(40)/n)+7*float(width)*math.log(40)/(3*(n-1))
        self.assertGreaterEqual(float(a['sampling_radius_upper'])+1e-14, radius)
        self.assertAlmostEqual(float(a['sampling_radius_upper']), radius, places=12)

    def test_empirical_bernstein_exact_binomial_coverage(self):
        # Checks the concentration operation itself, independently of physical
        # interpretation of the synthetic score law.
        p = self.protocol(h=1, q_abs_max=0, envelope_floor=1,
                          generator_bounds=(0, 0, 0, 0))
        lo, hi = p['corrected_score_bounds']; width = hi-lo
        n, success, alpha = 80, F(1, 3), F(1, 10)
        truth = lo+width*success; failure = F(0)
        for k in range(n+1):
            cert = fr.certify_totals(n, k*hi+(n-k)*lo,
                                    k*hi*hi+(n-k)*lo*lo, p, alpha=alpha,
                                    concentration='empirical_bernstein')
            if cert['signed_rate_lower'] > truth:
                failure += math.comb(n, k)*success**k*(1-success)**(n-k)
        self.assertLessEqual(failure, alpha)

    def test_uniform_quantization_expands_range_and_pays_bias(self):
        p = self.protocol(h=1, q_abs_max=0, envelope_floor=1,
                          generator_bounds=(0, 0, 0, 0))
        delta = F(1, 1000); values = [F(1)+delta]*100
        cert = fr.certify_totals(100, sum(values), sum(x*x for x in values), p,
                                quantization_radius=delta, concentration='empirical_bernstein')
        self.assertEqual(cert['corrected_score_bounds'][1], 1+delta)
        self.assertEqual(cert['uniform_score_quantization_penalty'], delta)
        self.assertEqual(cert['empirical_variance'], 0)
        self.assertEqual(cert['signed_rate_lower'], F(1)-cert['sampling_radius_upper'])

    def test_empirical_bernstein_rejects_invalid_summaries(self):
        p = self.protocol()
        for kwargs in (dict(score_second_moment=-1),
                       dict(score_second_moment=0, mean_error=F(1, 10)),
                       dict(score_second_moment=0, conditional_mean_radius=F(1, 10)),
                       dict(score_second_moment=0, record_contract='bounded_conditional_mean_error')):
            with self.assertRaises(ValueError):
                fr.certify_summary(100, 0, p, concentration='empirical_bernstein', **kwargs)
        with self.assertRaises(ValueError):
            fr.certify_summary(1, 0, p, concentration='empirical_bernstein', score_second_moment=0)

    def test_known_variance_bernstein_requires_additional_contract(self):
        p = self.protocol(global_gamma_upper=None)
        with self.assertRaises(ValueError):
            fr.certify_summary(100, 0, p, concentration='bernstein')
        p = self.protocol()
        cert = fr.certify_summary(100, 0, p, concentration='bernstein')
        self.assertGreater(cert['sampling_radius_upper'], 0)
        with self.assertRaises(ValueError):
            fr.certify_summary(100, 0, p, concentration='bernstein', conditional_mean_radius=1)

    def test_coverage_allocation_and_conditional_mean_penalties(self):
        p = self.protocol()
        plain = fr.certify_summary(100, 0, p)
        penalized = fr.certify_summary(100, 0, p, mean_error=F(1, 7),
                                      conditional_mean_radius=F(2, 7), external_failure=F(1, 10),
                                      record_contract='bounded_conditional_mean_error')
        self.assertEqual(penalized['confidence_failure_upper'], F(3, 20))
        self.assertEqual(penalized['signed_rate_lower'], plain['signed_rate_lower']-F(3, 7))


if __name__ == '__main__':
    unittest.main()
