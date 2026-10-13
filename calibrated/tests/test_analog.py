import importlib.util
from pathlib import Path
from fractions import Fraction as F
import unittest
import sys
import numpy as np

CODE=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(CODE))
import analog as a
from constructive_access.virtual import contrast


class AnalogTests(unittest.TestCase):
    def test_exact_score_rounding_encloses_stored_float_and_fraction_inputs(self):
        left=[F(1,3),.1,-.75,F(-2,7)];right=[F(7,9),-.3,.625,F(5,6)]
        result=a.quantized_pair_summary(left,right,1001)
        exact=sum((1+F(x)*F(y))/2 for x,y in zip(left,right))/4
        self.assertLessEqual(abs(exact-result['mean']),result['mean_quantization_radius'])
        self.assertEqual(result['score_sum']/result['records'],result['mean'])
        self.assertGreaterEqual(result['quantized_empirical_variance'],0)

    def test_rational_means_do_not_require_integer_successes(self):
        args=dict(lags=[1,2],coefficients=[1,-1],records=[10000,10000],durations=[F(1),F(2)],calibration_vertices=[(F(0),F(1,2))])
        means=[F(690001,1000000),F(590003,1000000)]
        mean=a.certify_means(means=means,**args)
        summed=a.certify_sums(score_sums=[m*n for m,n in zip(means,args['records'])],**args)
        self.assertEqual(mean,summed);self.assertEqual(mean['status'],'positive_certificate')
        self.assertNotEqual((means[0]*10000).denominator,1)

    def test_binary_input_recovers_existing_fixed_contrast_certificate(self):
        old=contrast.certify_counts([1,2],[F(1),F(-1)],[10000,10000],[6900,5900],[F(1),F(2)],mean_interval=(0,0),reference_mean_calibrated=True)
        new=a.certify_means([1,2],[1,-1],[10000,10000],[F(69,100),F(59,100)],[1,2],[(0,1)])
        self.assertEqual(old['sampling_radius_upper'],new['sampling_radius_upper'])
        self.assertEqual(old['empirical_contrast'],new['empirical_contrast'])
        self.assertEqual(old['entropy_production_rate_lower'],new['entropy_production_rate_lower'])

    def test_every_error_source_is_subtracted_and_confidence_is_combined(self):
        kwargs=dict(lags=[1,2],coefficients=[1,-1],records=[10000,10000],means=[F(69,100),F(59,100)],durations=[1,2],calibration_vertices=[(0,F(1,2))],alpha=F(1,40),calibration_failure=F(1,40))
        clean=a.certify_means(**kwargs)
        noisy=a.certify_means(**kwargs,mean_errors=[F(1,1000)]*2,conditional_radii=[F(1,500)]*2)
        self.assertEqual(noisy['signed_contrast_lower'],clean['signed_contrast_lower']-F(3,500))
        self.assertEqual(noisy['confidence_failure_upper'],F(1,20))
        self.assertLess(noisy['entropy_production_rate_lower'],clean['entropy_production_rate_lower'])

    def test_raoblackwell_variance_and_conditional_convex_order(self):
        means=[F(3,5),F(2,5)];seconds=[F(2,5),F(1,5)]
        result=a.conditional_coin_variance(means,seconds,[1,-1],[1000,2000])
        self.assertGreater(result['removed_conditional_coin_variance'],0)
        # Exact conditional Jensen for a signed contrast of two independent virtual coins.
        w1,w2=F(2,7),F(4,5);direct=2*w1-3*w2
        second=F(0);fourth=F(0)
        for b1 in [0,1]:
            for b2 in [0,1]:
                p=(w1 if b1 else 1-w1)*(w2 if b2 else 1-w2)
                value=2*b1-3*b2;second+=p*value**2;fourth+=p*value**4
        self.assertGreaterEqual(second,direct**2);self.assertGreaterEqual(fourth,direct**4)

    def test_declared_range_improves_concentration_only_when_valid(self):
        kwargs=dict(lags=[1,2],coefficients=[1,-1],records=[10000,10000],means=[F(3,5),F(1,2)],durations=[1,2],calibration_vertices=[(0,F(1,2))])
        base=a.certify_means(**kwargs);narrow=a.certify_means(**kwargs,score_bounds=[(F(1,4),F(3,4))]*2)
        self.assertEqual(narrow['concentration_variance_proxy'],base['concentration_variance_proxy']/4)
        self.assertLess(narrow['sampling_radius_upper'],base['sampling_radius_upper'])
        with self.assertRaises(ValueError):a.certify_means(**kwargs,score_bounds=[(F(0),F(1,4))]*2)

    def test_validation_and_descriptive_scope(self):
        with self.assertRaises(ValueError):a.quantized_pair_summary([1],[1,2])
        with self.assertRaises(ValueError):a.quantized_pair_summary([float('nan')],[0])
        with self.assertRaises(ValueError):a.quantized_pair_summary([1.1],[0])
        self.assertIn('descriptive_only',a.quantized_pair_summary([0],[0])['status'])
        with self.assertRaises(TypeError):a.certify_means([1],[1],[1000],[.5],[1],[(0,1)])


if __name__=='__main__':unittest.main()
