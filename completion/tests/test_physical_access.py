"""Physical-contract tests; independent semigroup checks complement proofs."""
from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import physical_access as pa


class ExactPhysicalTests(unittest.TestCase):
    def test_exp_bounds_and_zero(self):
        import math
        self.assertEqual(pa.exp_bounds(F(0)), (F(1), F(1)))
        for x in (F(1, 100), F(1), F(5), F(20)):
            lo, hi = pa.exp_bounds(x, 64)
            # Floating comparison is only a check of enclosure implementation.
            self.assertLessEqual(float(lo), math.exp(float(x)) * (1 + 2e-15))
            self.assertGreaterEqual(float(hi), math.exp(float(x)) * (1 - 2e-15))
            self.assertLessEqual(lo, hi)
            nl, nh = pa.exp_negative_bounds(x, 64)
            self.assertEqual(nl, 1 / hi)
            self.assertEqual(nh, 1 / lo)

    def test_invalid_exp_inputs(self):
        for x in (0.5, True, "1/2"):
            with self.assertRaises(TypeError):
                pa.exp_bounds(x)
        with self.assertRaises(ValueError):
            pa.exp_bounds(F(-1))
        with self.assertRaises(ValueError):
            pa.exp_bounds(F(10), 3)

    def test_same_generator_and_different_channels(self):
        plus = (pa.Channel(0, 1, F(3), F(1)), pa.Channel(0, 1, F(1), F(3)))
        zero = (pa.Channel(0, 1, F(2), F(2)), pa.Channel(0, 1, F(2), F(2)))
        self.assertEqual(pa.generator(2, plus), pa.generator(2, zero))
        pi = pa.stationary(pa.generator(2, plus))
        self.assertEqual(pa.current(plus[0], pi), 1)
        self.assertEqual(pa.current(zero[0], pi), 0)
        self.assertNotEqual(pa.generator(2, pa.suppress(plus, 0, F(1, 2))),
                            pa.generator(2, pa.suppress(zero, 0, F(1, 2))))

    def test_native_not_perturbed_current(self):
        channels = (pa.Channel(0, 1, F(3), F(1)), pa.Channel(0, 1, F(1), F(3)))
        pi = pa.stationary(pa.generator(2, channels))
        perturbed = pa.suppress(channels, 0, F(1, 2))
        pir = pa.stationary(pa.generator(2, perturbed))
        self.assertEqual(pi, (F(1, 2), F(1, 2)))
        self.assertEqual(pir, (F(7, 12), F(5, 12)))
        self.assertEqual(pa.current(channels[0], pi), 1)
        self.assertEqual(pa.current(perturbed[0], pir), F(2, 3))

    def test_general_response_includes_reducible_chains(self):
        q = [[F(-1), F(1), F(0)], [F(1), F(-1), F(0)], [F(0), F(0), F(0)]]
        p = (F(1, 4), F(1, 4), F(1, 2))
        qp = [[F(-2), F(2), F(0)], [F(1), F(-1), F(0)], [F(0), F(0), F(0)]]
        cert = pa.two_endpoint_certificate(q, p, qp, 0, 1)
        self.assertEqual(cert["coefficient"], "1/4")
        self.assertFalse(cert["irreducibility_required"])

    def test_general_response_allows_complete_edge_removal(self):
        q = [[F(-1), F(1)], [F(1), F(-1)]]
        qp = [[F(0), F(0)], [F(0), F(0)]]
        cert = pa.two_endpoint_certificate(q, (F(1, 2), F(1, 2)), qp, 0, 1)
        self.assertEqual(cert["coefficient"], "0")

    def test_certificate_rejects_nontarget_perturbation(self):
        channels, p = pa.fixed_current_family(4)
        q = pa.generator(4, channels)
        qp = pa.generator(4, pa.suppress(channels, 1, F(1, 2)))
        with self.assertRaises(ValueError):
            pa.two_endpoint_certificate(q, p, qp, 0, 1)

    def test_certificate_rejects_nonstationary_preparation(self):
        q = [[F(-1), F(1)], [F(1), F(-1)]]
        with self.assertRaises(ValueError):
            pa.two_endpoint_certificate(q, (F(1), F(0)), q, 0, 1)

    def test_stationary_source_exposure_bernstein(self):
        plan = pa.poisson_exposure_plan(F(4), F(1, 100), F(1, 100))
        exposure = F(plan["source_exposure"])
        exponent = exposure * F(1, 100)**2 / (2 * (F(4) + F(1, 300)))
        self.assertEqual(exponent, F(plan["exponent"]))
        self.assertGreaterEqual(pa.exp_bounds(exponent, plan["exponential_taylor_terms"])[0], 200)
        self.assertEqual(exposure, 480400)

    def test_paired_hoeffding_certificate(self):
        plan = pa.hoeffding_plan(F(1, 500), F(1, 20))
        exponent = F(plan["records"]) * F(1, 500)**2 / 2
        self.assertGreaterEqual(pa.exp_bounds(exponent, plan["exponential_taylor_terms"])[0], 40)
        self.assertEqual(plan["records"], 2000000)

    def test_short_pulse_magnitude_bernstein_certificate(self):
        plans = [pa.short_pulse_current_plan(e, F(1, 8), F(4), F(1, 2), F(1, 20), F(1, 6))
                 for e in (F(1, 10), F(1, 20), F(1, 40))]
        for p in plans:
            self.assertEqual(F(p["current_bias_bound"]), F(p["current_error"]) / 2)
            t, v, s = F(p["duration"]), F(p["variance_cap"]), F(p["response_sampling_error"])
            exponent = F(p["records"]) * s**2 / (2 * (v + 2 * s / 3))
            self.assertGreaterEqual(pa.exp_bounds(exponent, p["exponential_taylor_terms"])[0], 40)
            self.assertEqual(s / ((1 - F(p["factor"])) * t), F(p["current_error"]) / 2)
        # Ceiling and the lower-order Bernstein term prevent exact cubic ratios.
        self.assertLess(plans[1]["records"], 8 * plans[0]["records"])
        self.assertGreater(plans[1]["records"], 7 * plans[0]["records"])

    def test_calibration_can_force_abstention(self):
        args = ((F(1, 100), F(3, 200)), (F(49, 100), F(51, 100)), F(1, 4), F(4))
        clean = pa.pulse_current_interval(*args)
        uncertain = pa.pulse_current_interval(*args, preparation_tv=F(1, 100))
        self.assertEqual(clean["orientation"], "positive")
        self.assertEqual(uncertain["orientation"], "unresolved")
        lo, hi = map(F, uncertain["native_current_interval"])
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)

    def test_current_interval_preserves_negative_sign(self):
        result = pa.pulse_current_interval((F(-3, 200), F(-1, 100)),
                                          (F(1, 2), F(1, 2)), F(1, 4), F(4))
        lo, hi = map(F, result["native_current_interval"])
        self.assertLess(lo, hi)
        self.assertLess(hi, 0)
        self.assertEqual(result["orientation"], "negative")

    def test_factor_uncertainty_crossing_one_rejected(self):
        for factors in ((F(1, 2), F(1)), (F(0), F(1, 2)), (F(9, 10), F(11, 10))):
            with self.assertRaises(ValueError):
                pa.pulse_current_interval((F(0), F(1, 100)), factors, F(1), F(1))

    def test_selective_rate_box_contains_exact_native_current(self):
        bounds = ((F(1, 2), F(1, 2)),) * 2 + ((F(4), F(4)),) * 2 + (
            (F(5, 2), F(5, 2)), (F(7, 2), F(7, 2)), (F(1, 2), F(1, 2)))
        result = pa.selective_rate_current_interval(bounds)
        self.assertEqual(result["native_current_interval"], ["1", "1"])
        self.assertEqual(result["vertices"], 128)

    def test_fixed_current_family_has_bounded_endpoint_rates(self):
        for n in (4, 5, 8, 16, 32, 64):
            channels, pi = pa.fixed_current_family(n)
            q = pa.generator(n, channels)
            self.assertEqual(pa.current(channels[0], pi), F(1, 8))
            self.assertEqual(pa.initial_response(q, pi), (F(0),) * n)
            qr = pa.generator(n, pa.suppress(channels, 0, F(1, 2)))
            self.assertLessEqual(max(-qr[0][0], -qr[1][1]), 4)

    def test_coarse_detector_crossing_has_exact_certificate(self):
        channels = (pa.Channel(0, 1, F(2), F(1)), pa.Channel(0, 2, F(10), F(1)),
                    pa.Channel(1, 3, F(10), F(1)), pa.Channel(2, 3, F(1), F(3)))
        pi = pa.stationary(pa.generator(4, channels))
        qr = pa.generator(4, pa.suppress(channels, 0, F(1, 2)))
        pir = pa.stationary(qr)
        self.assertEqual(pi, (F(7, 121), F(4, 121), F(80, 121), F(30, 121)))
        self.assertEqual(pa.current(channels[0], pi), F(10, 121))
        slope = pa.initial_response(qr, pi)
        self.assertLess(slope[1] + slope[2], 0)
        self.assertEqual(pir[1] + pir[2] - pi[1] - pi[2], F(10, 2299))
        self.assertFalse(pa.gateway_detector_certificate(qr, 0, 1, (1, 2))["certified"])
        # Negative initial derivative and positive infinite-time limit prove
        # crossing by continuity, independently of any sampled time grid.

    def test_gateway_coarse_detector_is_certified(self):
        channels = (pa.Channel(0, 1, F(2), F(1)), pa.Channel(1, 2, F(3), F(1)),
                    pa.Channel(1, 3, F(1), F(2)), pa.Channel(3, 0, F(1), F(1)))
        q = pa.generator(4, channels)
        result = pa.gateway_detector_certificate(q, 0, 1, (1, 2))
        self.assertTrue(result["certified"])
        self.assertEqual(result["reachable_without_gateway"], [0, 3])
        self.assertTrue(pa.gateway_detector_certificate(q, 0, 1, (1,))["certified"])


class IndependentSemigroupTests(unittest.TestCase):
    def test_random_finite_pulse_identity(self):
        import numpy as np
        from scipy.linalg import expm
        rng = random.Random(730019)
        for model in range(20):
            n = 3 + model % 4
            channels = tuple(pa.Channel(i, j, F(rng.randrange(1, 8), rng.randrange(1, 5)),
                                       F(rng.randrange(1, 8), rng.randrange(1, 5)))
                             for i in range(n) for j in range(i + 1, n))
            q = pa.generator(n, channels)
            pi = pa.stationary(q)
            qr = pa.generator(n, pa.suppress(channels, 0, F(1, 3)))
            j = pa.current(channels[0], pi)
            qrf = np.array(qr, dtype=float)
            # Augmented matrix independently computes the integrated semigroup.
            augmented = np.zeros((n + 1, n + 1))
            augmented[:n, :n] = qrf
            augmented[1, n] = 1
            cap = max(-qr[0][0], -qr[1][1])
            for t in (0.001, 0.1, 1.0, 8.0):
                semigroup = expm(qrf * t)
                occupation = expm(augmented * t)[:n, n]
                h = occupation[1] - occupation[0]
                delta = (np.array(pi, dtype=float) @ semigroup)[1] - float(pi[1])
                self.assertAlmostEqual(delta, -float(F(2, 3) * j) * h, delta=5e-12)
                self.assertGreater(h, 0)
                self.assertGreaterEqual(h + 5e-12, t * np.exp(-float(cap) * t))
                self.assertLessEqual(h, t + 5e-12)

    def test_preparation_failure_can_reverse_sign(self):
        import numpy as np
        from scipy.linalg import expm
        channels = (pa.Channel(0, 1, F(3), F(1)), pa.Channel(0, 1, F(1), F(3)))
        qr = pa.generator(2, pa.suppress(channels, 0, F(1, 2)))
        bad = np.array([0.9, 0.1])
        self.assertGreater((bad @ expm(np.array(qr, float)))[1] - bad[1], 0)
        self.assertGreater(pa.current(channels[0], (F(1, 2), F(1, 2))), 0)

    def test_instantaneous_kernel_need_not_be_positive(self):
        import numpy as np
        from scipy.linalg import expm
        q = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], float)
        p = expm(2 * q)
        self.assertLess(p[1, 1] - p[0, 1], 0)
        a = np.zeros((4, 4)); a[:3, :3] = q; a[1, 3] = 1
        f = expm(2 * a)[:3, 3]
        self.assertGreater(f[1] - f[0], 0)


if __name__ == "__main__":
    unittest.main()
