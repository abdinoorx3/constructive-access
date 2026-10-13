from fractions import Fraction as F
import unittest
import numpy as np
import calibrated_benchmark as cb


class CalibratedBenchmarkTests(unittest.TestCase):
    def test_independent_expm_oracle_and_exact_countermodel(self):
        spec=cb.specification();truth=cb.oracle(spec['cases'][0])
        np.testing.assert_allclose(truth['probabilities'],[.69,.59,.485712],rtol=0,atol=2e-15)
        self.assertAlmostEqual(truth['A'],.5,places=14)
        witness=cb.equilibrium_countermodel()
        self.assertEqual(witness['probabilities'][:2],[F(69,100),F(59,100)])
        self.assertGreater(witness['A'],F(1,2))

    def test_twin_readout_sampler_does_not_square_one_bit(self):
        rng=np.random.default_rng(928451)
        counts,raw=cb.calibration_records(rng,40000,[-1.,1.],gain=.5)
        estimate=(counts[0]+counts[3]-counts[1]-counts[2])/counts.sum()
        self.assertLess(abs(estimate-.25),.02)
        self.assertTrue(np.any(raw['first']!=raw['second']))
        self.assertTrue(np.array_equal(raw['initial'],raw['final']))
        self.assertEqual(float(raw['elapsed'].sum()),0.)

    def test_matched_cost_contract_and_predeclared_alternative(self):
        spec=cb.specification()
        self.assertEqual(spec['calibrated_arm']['pairs'],spec['alternative_arm']['pairs'])
        self.assertEqual(spec['calibrated_arm']['readouts'],spec['alternative_arm']['readouts'])
        self.assertEqual(spec['calibrated_arm']['declared_evolution_time'],24000)
        self.assertEqual(spec['alternative_arm']['declared_evolution_time'],36000)
        self.assertEqual(spec['extra_lag_alternative']['parameter'],'17/50')
        self.assertEqual(F(spec['calibrated']['alpha_science'])+F(spec['calibrated']['alpha_calibration']),F(1,20))

    def test_illustrative_centered_counts_and_invalid_control(self):
        spec=cb.specification()
        # Deliberately deterministic test inputs, not held-out datasets.
        result=cb.analyze([5520,4720,1943],[1500,500,500,1500],spec)
        self.assertEqual(result['status'],'analyzed')
        self.assertGreater(result['calibrated_pinsker_rate'],0)
        self.assertGreater(result['nonlinear']['certificate']['record_kl_lower'],0)
        wrong=cb.analyze([5783,5590,2710],[1020,980,980,1020],spec)
        self.assertGreater(wrong['nonlinear']['certificate']['record_kl_lower'],0)

    def test_empty_calibration_does_not_disable_independent_alternative(self):
        out=cb.analyze([5520,4720,1943],[0,4000,0,0],cb.specification())
        self.assertEqual(out['status'],'inconsistent_calibration_region')
        self.assertGreater(out['alternative']['entropy_production_rate_lower'],0)


if __name__=='__main__':unittest.main()
