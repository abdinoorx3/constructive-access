from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import joint_calibration as jc


class JointCalibrationTests(unittest.TestCase):
    def test_exact_support_and_driven_population_gap(self):
        example=jc.population(F(19,25),[F(3,4),F(69,100),F(59,100)])
        self.assertEqual(example["reversible_support"],F(-18,625))
        self.assertEqual(example["reversible_gap"],F(34,625))
        # Every single nonnegative spectral atom obeys the exact quadratic.
        for A in [F(0),F(1,4),F(1,2),F(1)]:
            for x in [F(0),F(1,5),F(19,25),F(1)]:
                p=[(1+A*x**k)/2 for k in range(3)]
                gap=jc.population(F(19,25),p)["reversible_gap"]
                self.assertEqual(gap,-A*(x-F(19,25))**2/2)

    def test_reversible_expm_operator_identity(self):
        Q=np.array([[-2.,2.,0.],[1.,-4.,3.],[0.,2.,-2.]])
        pi=np.array([2.,4.,6.])/12;f=np.array([-.8,.3,.9]);s=.76
        P=expm(Q*.7);C=[float(np.dot(pi*f,np.linalg.matrix_power(P,k)@f)) for k in range(3)]
        lhs=-s*s*C[0]+2*s*C[1]-C[2]
        residual=P@f-s*f
        self.assertAlmostEqual(lhs,-float(np.dot(pi,residual*residual)),places=13)
        self.assertLessEqual(lhs,1e-13)

    def test_finite_data_zero_lag_variance_and_zero_physical_time(self):
        result=jc.certify_counts([4000,8000,8000],[3000,5520,4720])
        self.assertGreater(result["entropy_production_rate_lower"],0)
        self.assertEqual(result["pair_records"],20000)
        self.assertEqual(result["endpoint_readouts"],40000)
        self.assertEqual(result["declared_evolution_time"],24000)
        a=result["support"]["coefficients"];region=result["confidence_region"]
        variance=sum(c*c/n for c,n in zip(a,[4000,8000,8000]))
        expected=jc.sqrt_upper(jc.ec.log_interval(20)[1]*variance/2)
        self.assertEqual(region["sampling_radius_upper"],expected)
        self.assertEqual(result["comparison_time_denominator_upper"],(F(38,25)+jc.sqrt_upper(F(2)))**2)

    def test_more_twins_reduce_variance_without_more_evolution_time(self):
        one=jc.certify_counts([4000,8000,8000],[3000,5520,4720])
        two=jc.certify_counts([8000,8000,8000],[6000,5520,4720])
        self.assertGreater(two["reversible_gap_lower"],one["reversible_gap_lower"])
        self.assertEqual(two["declared_evolution_time"],one["declared_evolution_time"])
        self.assertEqual(two["endpoint_readouts"]-one["endpoint_readouts"],8000)

    def test_zero_lag_uncertainty_negative_control(self):
        c=jc.lag_zero_uncertainty_control()
        self.assertEqual(c["true_population_gap"],0)
        self.assertEqual(c["one_lag_zero_record_failure_probability"],F(1,4))
        self.assertEqual(c["false_gap_with_exact_science_and_zero_lag_success_zero"],F(3,16))
        actual=jc.certify_counts([1,16000000,16000000],[0,10000000,9000000],parameter=F(1,2))
        self.assertEqual(actual["entropy_production_rate_lower"],0)

    def test_declared_whole_record_bias_and_failure(self):
        clean=jc.certify_counts([4000,8000,8000],[3000,5520,4720])
        changed=jc.certify_counts([4000,8000,8000],[3000,5520,4720],
                                 calibration_radii=[F(1,100),0,0],external_failure=F(1,100))
        self.assertEqual(changed["confidence_region"]["conditional_bias_penalty"],F(361,62500))
        self.assertEqual(changed["failure_upper"],F(3,50))
        self.assertLess(changed["entropy_production_rate_lower"],clean["entropy_production_rate_lower"])

    def test_invalid_contract_guards(self):
        base=dict(records=[4000,8000,8000],successes=[3000,5520,4720])
        for change in [dict(records=[4000,8000]),dict(parameter=F(11,10)),
                       dict(twin_contract="same_bit_squared"),dict(record_contract="raw_overlapping_pairs"),
                       dict(selection="postselected"),dict(lag_unit=0),dict(successes=[4001,5520,4720]),
                       dict(external_failure=F(99,100)),dict(alpha=0)]:
            with self.assertRaises((ValueError,TypeError)):
                jc.certify_counts(**(base|change))

    def test_parallel_dataset_stream_matches_serial(self):
        from concurrent.futures import ProcessPoolExecutor
        spec=jc.benchmark.specification();case=spec["cases"][2]
        joint=dict(records=[4000,8000,8000],parameter_s="19/25",alpha="1/20")
        task=(80351,2,7,case,spec,joint)
        serial=jc._dataset(task)
        with ProcessPoolExecutor(max_workers=1) as pool:
            parallel=pool.submit(jc._dataset,task).result()
        np.testing.assert_array_equal(serial[0],parallel[0])
        np.testing.assert_array_equal(serial[1],parallel[1])
        self.assertEqual(serial[2],parallel[2])
        self.assertEqual(serial[3]["calibrated_pinsker_rate"],parallel[3]["calibrated_pinsker_rate"])
        self.assertEqual(serial[3]["alternative"],parallel[3]["alternative"])


if __name__=="__main__":unittest.main()
