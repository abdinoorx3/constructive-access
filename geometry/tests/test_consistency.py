"""Exact and high-precision checks of the vanishing-radius design."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import consistency as co
import finite_resolution as fr


def mpc(value):
    return mp.mpf(value.numerator)/value.denominator


class ConsistencyTests(unittest.TestCase):
    def test_exact_resources_and_calibration(self):
        p = co.design(3, F(3, 10), F(7, 10), F(1, 4))
        self.assertEqual(p["records"], 3**12)
        self.assertEqual(p["h"], F(1, 108))
        self.assertEqual(p["envelope_floor"], F(1, 30))
        self.assertEqual(p["generator_bounds"], (F(1), F(1), F(29, 50), F(29, 50)))
        self.assertEqual(p["triple_detector_readouts"], 3**13)
        self.assertEqual(p["declared_evolution_time"], F(3**9, 2))

    def test_radius_decreases_without_absolute_precision_floor(self):
        first = co.design(2)
        second = co.design(4)
        tiny = co.design(2**120)
        self.assertLessEqual(second["sampling_radius_upper"], first["sampling_radius_upper"]/2)
        fixed_floor = fr.sqrt_upper(F(1, 2**400))
        self.assertEqual(fixed_floor, F(1, 2**90))
        self.assertLess(tiny["sampling_radius_upper"], fixed_floor)
        self.assertGreater(tiny["sampling_radius_upper"], 0)

    def test_high_precision_domination_of_real_bernstein_radius(self):
        cases = ((F(1), F(2), F(1)),
                 (F(3, 10), F(7, 10), F(1, 4)),
                 (F(1, 10**80), F(0), F(10**80)),
                 (F(10**30), F(1, 10**20), F(1, 10**30)))
        with mp.workdps(180):
            for gamma, omega, h0 in cases:
                for m in (1, 2, 3, 20, 2**20):
                    p = co.design(m, gamma, omega, h0)
                    gg, oo, hh = map(mpc, (gamma, omega, h0/F(m**3)))
                    floor = mpc(gamma/F(m*m))
                    M=gg+oo; R=gg*gg+oo*oo; C=R*(2*gg+oo)
                    real_sqrt_second=((gg+M+mp.sqrt(2*gg/hh))**2+C*hh)/floor
                    real_width=((2/hh+gg)**2+C*hh+1/(hh*hh))/floor
                    ell=mp.log(40); n=mp.mpf(m)**12
                    linear=real_width*ell/(3*n)
                    real_radius=linear+mp.sqrt(2*real_sqrt_second**2*ell/n+linear**2)
                    self.assertGreaterEqual(mpc(p["sampling_radius_upper"]), real_radius)
                    self.assertGreaterEqual(mpc(p["second_moment_coefficient_upper"])*m**5, real_sqrt_second)

    def test_existing_exact_score_range_and_bias_fit(self):
        for m in (1, 2, 5, 20):
            p=co.design(m)
            old=fr.make_protocol(h=p["h"], q_abs_max=1,
                envelope_floor=p["envelope_floor"],
                generator_bounds=p["generator_bounds"],global_gamma_upper=1)
            self.assertEqual(old["corrected_score_bounds"], p["corrected_score_bounds"])
            self.assertLessEqual(old["corrected_score_bounds"][1]-old["corrected_score_bounds"][0],
                                 p["range_coefficient_upper"]*m**8)

    def test_deterministic_mean_error_costs_twice_for_accuracy(self):
        p=co.design(1024)
        a=co.certify_mean(1,p)
        b=co.certify_mean(1,p,numerical_mean_radius=F(1,100))
        self.assertEqual(a["entropy_production_rate_lower"]-b["entropy_production_rate_lower"], F(1,100))
        self.assertEqual(b["joint_accuracy_gap_upper"]-a["joint_accuracy_gap_upper"], F(1,50))
        self.assertEqual(b["confidence_failure_upper"], F(1,40))
        self.assertEqual(b["joint_validity_and_accuracy_failure_upper"], F(1,20))

    def test_null_summary_is_not_clipped_record_by_record(self):
        p=co.design(8)
        result=co.certify_mean(0,p)
        self.assertEqual(result["entropy_production_rate_lower"],0)
        self.assertLess(result["signed_rate_lower"],0)

    def test_tampering_and_unsupported_inputs_rejected(self):
        p=co.design(8)
        p["sampling_radius_upper"]=F(0)
        with self.assertRaises(ValueError):co.certify_mean(1,p)
        for bad in (0,-1,True,2.0):
            with self.assertRaises((ValueError,TypeError)):co.design(bad)
        for kwargs in ({"gamma":0},{"omega_abs_upper":-1},{"reference_time":0},{"alpha":1}):
            with self.assertRaises(ValueError):co.design(2,**kwargs)

    def test_deterministic_output_does_not_claim_record_evidence(self):
        data=co.deterministic_examples()
        self.assertIn("no_simulated_or_measured_records",data["kind"])
        self.assertTrue(data["fixed_precision_diagnostic"]["consistent_radius_below_fixed_grid"])
        self.assertEqual(len(data["plans"]),9)


if __name__=="__main__":
    unittest.main()
