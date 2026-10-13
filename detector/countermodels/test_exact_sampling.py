from pathlib import Path
import importlib.util
import math
import unittest
import numpy as np
from scipy.integrate import quad

spec=importlib.util.spec_from_file_location("exact_sampling",Path(__file__).with_name("exact_sampling.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ExposureSamplingCountermodels(unittest.TestCase):
    def test_independent_exposure_integrals(self):
        for index in (0,1,2,5):
            omega=2*math.pi*index
            variance=2*quad(lambda t:(1-t)*math.exp(-t)*math.cos(omega*t),0,1,epsabs=1e-12)[0]
            v,f=m.exposed_oscillator_moments(index)
            self.assertAlmostEqual(variance,v,places=11)
            for lag in (1,2,4):
                covariance=quad(lambda u:quad(lambda z: math.exp(-abs(lag+u-z))*math.cos(omega*(lag+u-z)),0,1,epsabs=1e-12)[0],0,1,epsabs=1e-12)[0]
                self.assertAlmostEqual(covariance,f*math.exp(-lag),places=11)

    def test_positive_mixtures_match_all_covariance_parameters(self):
        for index in (2,3,10,50):
            a0,an=m.mixture_weights(index)
            self.assertGreater(a0,0)
            self.assertGreater(an,0)
            actual=a0*np.array(m.exposed_oscillator_moments(0))+an*np.array(m.exposed_oscillator_moments(index))
            np.testing.assert_allclose(actual,m.exposed_oscillator_moments(1),rtol=1e-13,atol=1e-15)
            self.assertGreater(abs(a0+an-1),1)

    def test_ratio_strictly_decreases(self):
        ratios=[f/v for v,f in [m.exposed_oscillator_moments(i) for i in range(20)]]
        self.assertTrue(all(a>b for a,b in zip(ratios[:-1],ratios[1:])))

    def test_entropy_grows_without_changing_sampled_law(self):
        a,b=m.report(2),m.report(20)
        self.assertEqual(a["exposed_target_variance"],b["exposed_target_variance"])
        self.assertAlmostEqual(b["mixture_ordinary_entropy"]/a["mixture_ordinary_entropy"],100)
        self.assertLess(b["largest_covariance_numeric_residual_lags_0_to_20"],1e-14)

    def test_invalid_inputs(self):
        for index in (-1,True,1.5):
            with self.assertRaises(ValueError):m.exposed_oscillator_moments(index)
        for index in (1,True,2.5):
            with self.assertRaises(ValueError):m.mixture_weights(index)

if __name__=="__main__":unittest.main()
