from fractions import Fraction as F
from pathlib import Path
import json
import sys
import unittest

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import retiming as rt
import joint_analog as ja
from spectral_dynamics import from_gap


class RetimingTests(unittest.TestCase):
    def test_negative_correlation_uses_half_lag_without_half_lag_data(self):
        out = rt.certify_mean(1800, F(9, 20), lag=F(1, 2))
        self.assertGreater(out["negative_correlation_gap_lower"], 0)
        expected = from_gap(out["negative_correlation_gap_lower"], lag_unit=F(1, 4))
        self.assertEqual(out["direct_bridge"], expected)
        self.assertEqual(expected["linear_rate_lower"], 16*out["negative_correlation_gap_lower"])
        self.assertEqual(out["calibration_acquisitions"], 0)
        self.assertEqual(out["stationary_pair_preparations"], 1800)
        self.assertEqual(out["detector_readouts"], 3600)
        self.assertEqual(out["declared_evolution_time"], 900)

    def test_pair_product_sums_reproduce_noninteger_means(self):
        out = rt.certify_product_sum(1800, -200003, 1000)
        expected = rt.certify_mean(1800, F(1800000-200003, 3600000))
        self.assertEqual(out, expected)
        self.assertNotEqual((out["mean"]*1800).denominator, 1)

    def test_zero_and_positive_correlations_do_not_give_population_witness(self):
        for mean in (F(1, 2), F(3, 5), F(1)):
            out = rt.certify_mean(10**12, mean)
            self.assertEqual(out["entropy_production_rate_lower"], 0)

    def test_exact_population_change_repairs_the_prior_family_obstruction(self):
        old = ja.noisy_population_obstruction(); new = rt.population_audit()
        self.assertLess(old["max_gap_all_real_s_upper"], F(-7, 1000))
        self.assertLess(new["new_correlation_interval"][1], F(-7, 100))
        self.assertGreater(new["population_bridge"]["entropy_production_rate_lower"], F(1, 2))
        self.assertLess(new["population_bridge"]["entropy_production_rate_lower"], new["actual_entropy_rate_interval"][0])

    def test_uniformization_matches_independent_matrix_exponential(self):
        for forward, backward, noisy in [(20, 1, True), (20, 1, False), (F(21, 2), F(21, 2), True)]:
            bounds = rt.correlation_interval(forward=forward, backward=backward, noisy=noisy)
            Q = np.zeros((20, 20))
            for j in range(20):
                Q[j, (j+1) % 20] = float(forward); Q[j, (j-1) % 20] = float(backward)
                Q[j, j] = -float(forward+backward)
            f = np.asarray(rt.SENSOR)/(8192 if noisy else 4096)
            actual = np.dot(f, expm(.5*Q)@f)/20
            self.assertAlmostEqual(float(sum(bounds)/2), actual, places=14)
            self.assertLess(float(bounds[1]-bounds[0]), 1e-25)

    def test_range_error_and_conditional_bias_are_not_free(self):
        clean = rt.certify_mean(1800, F(9, 20))
        dirty = rt.certify_mean(1800, F(9, 20), mean_error=F(1, 1000), conditional_radius=F(1, 500), external_failure=F(1, 100))
        self.assertEqual(clean["signed_gap_lower"]-dirty["signed_gap_lower"], F(3, 1000))
        self.assertEqual(dirty["confidence_failure_upper"], F(3, 50))
        self.assertLess(dirty["entropy_production_rate_lower"], clean["entropy_production_rate_lower"])

    def test_same_data_timing_and_unbounded_sensor_contracts_rejected(self):
        for kwargs in ({"selection": "max_same_bank"}, {"lag": 0}, {"alpha": 1},
                       {"detector_contract": "arbitrary_sensor_memory"},
                       {"record_contract": "uncontrolled_dependence"}, {"conditional_radius": -1}):
            with self.assertRaises(ValueError): rt.certify_mean(1800, F(9, 20), **kwargs)
        with self.assertRaises(TypeError): rt.certify_mean(1800, .45)
        with self.assertRaises(ValueError): rt.certify_product_sum(1800, 1800001, 1000)

    def test_actual_baseline_api_and_frozen_costs(self):
        root = Path(__file__).resolve().parents[1]
        design = json.loads((root/"contracts/retiming.json").read_text())
        stat = {"baseline_calibration": {"records": 4000, "sum_squares": 1500, "denominator": 1},
                "baseline_pairs": [{"records": 3000, "product_sum": 282, "product_denominator": 1},
                                   {"records": 3000, "product_sum": 118, "product_denominator": 1}],
                "retimed_pairs": {"records": 1800, "product_sum": -132, "product_denominator": 1}}
        baseline, new = rt.analyze_statistics(stat, design)
        self.assertEqual(set(baseline), {"1/2", "3/4"})
        self.assertEqual(baseline["1/2"]["declared_evolution_time"], new["declared_evolution_time"])
        self.assertEqual(baseline["1/2"]["detector_readouts"], 16000)
        self.assertEqual(new["detector_readouts"], 3600)
        self.assertEqual(baseline["1/2"]["best_entropy_production_rate_lower"], 0)
        self.assertGreater(new["entropy_production_rate_lower"], 0)


if __name__ == "__main__": unittest.main()
