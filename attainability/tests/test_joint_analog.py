from fractions import Fraction as F
from pathlib import Path
import json
import sys
import unittest

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import joint_analog as ja


class JointAnalogTests(unittest.TestCase):
    def test_population_identity_and_noise_penalty(self):
        s = F(3, 4); A = F(1, 2); C1 = F(3, 8); C2 = F(1, 8)
        bare = ja.population_gap(A, C1, C2, s)
        noisy = ja.population_gap(A+F(1, 8), C1, C2, s)
        self.assertEqual(bare, F(5, 64))
        self.assertEqual(bare-noisy, s*s/16)
        sup = ja.support(s)
        p = [(1+A)/2, (1+C1)/2, (1+C2)/2]
        self.assertEqual(sum(a*x for a, x in zip(sup["coefficients"], p))-sup["reversible_support"], bare)

    def test_reversible_mixture_support_exactly(self):
        # Includes nonzero mean and a finite mixture of independently reset modes.
        for s in [F(0), F(1, 5), F(3, 4), F(1)]:
            u = F(1, 16); masses = [(F(1, 3), F(1, 5)), (F(4, 5), F(1, 2))]
            A = u+sum(w for _, w in masses)
            C1 = u+sum(w*x for x, w in masses)
            C2 = u+sum(w*x*x for x, w in masses)
            direct = ja.population_gap(A, C1, C2, s)
            spectral = -(u*(1-s)**2+sum(w*(x-s)**2 for x, w in masses))/2
            self.assertEqual(direct, spectral)
            self.assertLessEqual(direct, 0)

    def test_same_state_twins_are_unnecessary_but_noise_is_conservative(self):
        # State-independent fair +/-1 detector: A=0 but B=1, so squaring is not A.
        self.assertEqual(ja.population_gap(F(1), F(0), F(0), F(1, 2)), F(-1, 8))
        self.assertEqual(ja.population_gap(F(0), F(0), F(0), F(1, 2)), 0)

    def test_single_zero_lag_readout_uncertainty_must_remain(self):
        # Reversible reset generator, f in {0,+1,-1}, stationary (1/2,1/4,1/4).
        # A=1/2, decay=1/2. Z=0 (probability1/2) gives observed W0=1/2.
        p = [F(1, 2), F(5, 8), F(9, 16)]
        false_population_gap = sum(a*x for a, x in zip(ja.support()["coefficients"], p))-ja.support()["reversible_support"]
        self.assertEqual(false_population_gap, F(1, 16))
        safe = ja.certify_means([1, 10**12, 10**12], p)
        self.assertEqual(safe["entropy_production_rate_lower"], 0)
        self.assertGreater(safe["sampling_radius_upper"], false_population_gap)

    def test_radius_uses_half_range_and_costs_single_zero_lag_reading(self):
        out = ja.certify_means([4000, 3000, 3000], [F(3, 4), F(69, 100), F(59, 100)], lag_unit=F(1, 10))
        self.assertEqual(out["concentration_variance_proxy"], F(1, 16)/(4*4000)+F(1, 3000)+F(1, 3000))
        self.assertEqual(out["detector_readouts"], 16000)
        self.assertEqual(out["acquisitions"], 10000)
        self.assertEqual(out["declared_evolution_time"], 900)
        self.assertGreater(out["entropy_production_rate_lower"], 0)

    def test_sums_preserve_noninteger_scores(self):
        means = [F(750001, 10**6), F(690001, 10**6), F(590003, 10**6)]
        n = [4000, 3000, 3000]
        a = ja.certify_means(n, means)
        b = ja.certify_sums(n, [v*N for v, N in zip(means, n)])
        self.assertEqual(a, b)
        self.assertNotEqual((means[1]*n[1]).denominator, 1)

    def test_quantization_square_encloses_exact_stored_inputs(self):
        data = [F(1, 3), .1, -.7, F(4, 7)]
        out = ja.quantized_square_summary(data, 1001)
        true = sum((1+F(z)**2)/2 for z in data)/len(data)
        self.assertLessEqual(abs(out["mean"]-true), out["mean_quantization_radius"])
        self.assertGreaterEqual(out["quantized_empirical_variance"], 0)

    def test_declared_errors_subtracted_once_and_external_failures_added(self):
        args = dict(records=[4000, 3000, 3000], means=[F(3, 4), F(69, 100), F(59, 100)], parameter=F(3, 4))
        clean = ja.certify_means(**args)
        dirty = ja.certify_means(**args, mean_errors=[F(1, 1000)]*3, conditional_radii=[F(1, 500)]*3,
                                 external_failure=F(1, 100), record_contract="bounded_conditional_mean_error")
        total = sum(abs(x) for x in clean["support"]["coefficients"])*F(3, 1000)
        self.assertEqual(clean["signed_gap_lower"]-dirty["signed_gap_lower"], total)
        self.assertEqual(dirty["confidence_failure_upper"], F(3, 50))

    def test_many_random_reversible_generators_respect_analog_constraint(self):
        rng = np.random.default_rng(638441)
        for _ in range(25):
            pi = rng.dirichlet(np.ones(5)); c = rng.random((5, 5)); c = (c+c.T)/2
            np.fill_diagonal(c, 0); Q = c/pi[:, None]; np.fill_diagonal(Q, -Q.sum(axis=1))
            P = expm(.2*Q); f = rng.uniform(-.8, .8, 5)
            A = np.dot(pi, f*f); B = A+(1-A)*rng.random()
            C1 = np.dot(pi*f, P@f); C2 = np.dot(pi*f, P@P@f)
            for s in (0, .25, .5, .75, 1):
                self.assertLessEqual(-s*s*B+2*s*C1-C2, 1e-12)

    def test_validation_rejects_unsupported_contracts_and_same_data_selection(self):
        args = dict(records=[1000]*3, means=[F(3, 4), F(3, 5), F(1, 2)])
        for kwargs in ({"selection": "fit_same_data"}, {"lag_unit": 0}, {"alpha": 1},
                       {"detector_contract": "correlated_sensor_memory"},
                       {"record_contract": "dependent_without_bound"},
                       {"record_contract": "bounded_conditional_mean_error"},
                       {"mean_errors": [F(-1), 0, 0]}):
            with self.assertRaises(ValueError): ja.certify_means(**args, **kwargs)
        with self.assertRaises(TypeError): ja.certify_means([1000]*3, [.75, F(3, 5), F(1, 2)])
        with self.assertRaises(ValueError): ja.certify_means([1000]*3, [F(1, 4), F(3, 5), F(1, 2)])
        with self.assertRaises(ValueError): ja.quantized_square_summary([float("nan")])
        with self.assertRaises(ValueError): ja.quantized_square_summary([2])

    def test_exact_noisy_population_obstruction_covers_every_real_parameter(self):
        result = ja.noisy_population_obstruction()
        self.assertTrue(result["no_population_witness_in_entire_quadratic_family"])
        self.assertLess(result["max_gap_all_real_s_upper"], F(-7, 1000))
        sensor = np.rint(4096*np.cos(2*np.pi*np.arange(20)/20)).astype(int)
        self.assertEqual(sensor.tolist(), result["detector_integers"])
        Q = np.zeros((20, 20))
        for j in range(20):
            Q[j, (j+1) % 20] = 20; Q[j, (j-1) % 20] = 1; Q[j, j] = -21
        f = sensor/8192
        for lag, bounds in zip((.1, .2), result["correlation_intervals"]):
            numeric = np.dot(f, expm(lag*Q)@f)/20
            self.assertAlmostEqual(numeric, float(sum(bounds)/2), places=14)

    def test_frozen_contract_and_actual_old_api_comparison(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root/"contracts/joint_analog.json").read_text())
        stat = {"calibration_records": 4000, "calibration_sum": 0,
                "calibration_sum_squares": 2000, "calibration_denominator": 1,
                "science": [{"records": 3000, "product_sum": 1140, "product_denominator": 1, "encoded_successes": 2070},
                            {"records": 3000, "product_sum": 540, "product_denominator": 1, "encoded_successes": 1770}]}
        out, _, repair = ja.analyze_statistics(stat, spec)
        self.assertEqual(set(out), set(spec["methods"]))
        self.assertLess(out["joint_analog"]["sampling_radius_upper"], out["split_analog"]["sampling_radius_upper"])
        self.assertGreater(out["joint_analog"]["entropy_production_rate_lower"], out["split_analog"]["entropy_production_rate_lower"])
        self.assertEqual(repair["entropy_production_rate_lower"], 0)


if __name__ == "__main__": unittest.main()
