import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import robust_physics as rp


class RobustPhysicalTests(unittest.TestCase):
    def test_exact_exponential_enclosure(self):
        import decimal
        with decimal.localcontext() as c:
            c.prec = 100
            for x in (F(0), F(1,100), F(1), F(7)):
                low, high = rp.exp_negative_bounds(x)
                reference = (-decimal.Decimal(x.numerator)/x.denominator).exp()
                self.assertLessEqual(decimal.Decimal(low.numerator)/low.denominator, reference)
                self.assertGreaterEqual(decimal.Decimal(high.numerator)/high.denominator, reference)

    def test_generator_discrepancy_improvement(self):
        q = ((F(-3),F(2),F(1)),(F(0),F(0),F(0)),(F(0),F(0),F(0)))
        r = ((F(-3),F(1),F(2)),(F(0),F(0),F(0)),(F(0),F(0),F(0)))
        self.assertEqual(rp.generator_coupling_rate(q,r), 1)
        self.assertEqual(sum(abs(q[0][j]-r[0][j]) for j in (1,2)),2)

    def test_invalid_generators_rejected(self):
        with self.assertRaises(ValueError):
            rp.check_generator(((F(-2),F(1)),(F(1),F(-1))))
        with self.assertRaises(ValueError):
            rp.check_generator(((F(1),F(-1)),(F(1),F(-1))))
        with self.assertRaises(TypeError):
            rp.check_generator(((-1.0,1.0),(1.0,-1.0)))

    def test_matching_removes_inverse_duration_preparation_penalty(self):
        for duration in (F(1,10000), F(1,100), F(1,10)):
            bias = rp.matched_preparation_bias(F(1,100),duration,F(1,2),F(3),F(1))
            self.assertLessEqual(bias/(duration*F(1,2)),F(1,25))
        self.assertEqual(rp.matched_preparation_bias(F(0),F(1),F(1,2),F(3),F(1)),0)

    def test_preparation_slope_is_sharp(self):
        # rho-pi=(eta,-eta), L_j=(a,-b), giving exact derivative eta*s*(a+b).
        eta,s,a,b = F(1,100),F(1,2),F(3),F(1)
        derivative_error = s*abs(eta*a-(-eta)*b)
        self.assertEqual(derivative_error,eta*s*(a+b))

    def test_delay_zero_and_failure_status(self):
        self.assertEqual(rp.delay_oscillation_bound(F(0),F(4),F(4)),0)
        certificate = rp.robust_current_interval((F(1,100),F(1,50)),
            duration=(F(1),F(1)),retained_fraction=F(1,2),forward_rate_cap=F(3),
            reverse_rate_cap=F(1),pulse_exit_cap=F(4),delay=F(1),delay_exit_cap=F(4),delay_inflow_cap=F(4))
        self.assertEqual(certificate["status"],"delay_or_duration_contract_does_not_certify_sign")
        self.assertIsNone(certificate["current_interval"])

    def test_positive_exact_interval(self):
        certificate = rp.robust_current_interval((F(9,1000),F(11,1000)),
            duration=(F(1,10),F(1,10)),retained_fraction=F(1,2),forward_rate_cap=F(3),
            reverse_rate_cap=F(1),pulse_exit_cap=F(4),preparation_tv=F(1,1000),
            delay=F(1,100),delay_exit_cap=F(4),delay_inflow_cap=F(4))
        self.assertEqual(certificate["status"],"positive_current_certified")
        lo,hi = map(F,certificate["current_interval"])
        self.assertGreater(lo,0); self.assertGreater(hi,lo)

    def test_clock_jitter_lower_bound(self):
        import math
        for omega in (F(0),F(1,10),F(3,2)):
            low,high = rp.delayed_kernel_interval((F(1,100),F(2)),F(3),omega)
            for k in range(101):
                t=.01+1.99*k/100
                actual=max(-t,t*(math.exp(-3*t)-float(omega)))
                self.assertGreaterEqual(actual+1e-12,float(low))
            self.assertEqual(high,2)

    def test_shared_detector_offset_absent(self):
        result = rp.robust_current_interval((F(1,100),F(1,50)),
            duration=(F(1,10),F(1,10)),retained_fraction=F(1,2),forward_rate_cap=F(3),
            reverse_rate_cap=F(1),pulse_exit_cap=F(4),detector_gain=(F(4,5),F(9,10)))
        self.assertEqual(result["status"],"positive_current_certified")
        with self.assertRaises(ValueError):
            rp.robust_current_interval((F(0),F(0)),duration=(F(1),F(1)),
                retained_fraction=F(1,2),forward_rate_cap=F(3),reverse_rate_cap=F(1),
                pulse_exit_cap=F(4),detector_gain=(F(0),F(1)))

    def test_no_floating_contracts(self):
        with self.assertRaises(TypeError):
            rp.matched_preparation_bias(.1,F(1),F(1,2),F(3),F(1))
        with self.assertRaises(ValueError):
            rp.nonstationary_current_bias(F(-1),F(3),F(1),F(4),F(4),F(4),F(4),F(3))

    def test_stationarity_free_bias_exact(self):
        self.assertEqual(rp.nonstationary_current_bias(F(1,100),F(3),F(1),F(4),F(4),F(4),F(4),F(3)),F(1,5))

    def test_shared_generator_drift_is_quadratic(self):
        for t in (F(1,1000),F(1,100),F(1,10),F(1)):
            bound=rp.shared_generator_drift_bias(t,F(1,2),F(3),F(1),F(1,10))
            self.assertLessEqual(bound,F(1,4)*t*t)
            self.assertGreaterEqual(bound,0)
        self.assertEqual(rp.shared_generator_drift_bias(F(1),F(1,2),F(3),F(1),F(0)),0)

    def test_shared_drift_bound_integral(self):
        import math
        t,kappa=F(2),F(3,10)
        bound=rp.shared_generator_drift_bias(t,F(1,2),F(3),F(1),kappa)
        exact_float=5*(float(t)-(1-math.exp(-float(kappa*t)))/float(kappa))
        self.assertGreaterEqual(float(bound)+1e-12,exact_float)

    def test_synthetic_bound_trials(self):
        results = rp.make_results()
        self.assertEqual(results["random_network_checks"],180)
        self.assertLess(max(results["largest_observed_error_over_bound"].values()),1.000000001)


if __name__ == "__main__":
    unittest.main()
