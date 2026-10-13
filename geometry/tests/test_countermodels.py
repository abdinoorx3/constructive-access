"""Independent algebra, quadrature and semigroup checks of scope countermodels."""
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

_MODULE = Path(__file__).resolve().parents[1] / "code" / "countermodels.py"
_spec = importlib.util.spec_from_file_location("geometry_countermodels", _MODULE)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


class ConditionalMeanCountermodel(unittest.TestCase):
    def test_independent_phase_quadrature_and_entropy(self):
        # Integrate the original positive part, independently of its atan formula.
        cutoff = math.atan(1 / 24)
        invalid = 2 / math.pi * quad(
            lambda x: 72 - (math.cos(x) / math.sin(x))**2 / 8,
            cutoff, math.pi / 2, epsabs=1e-11)[0]
        truth = quad(lambda z: (8*(1+.5*math.cos(z)))**2 /
                     (1+.5*math.cos(z)), 0, 2*math.pi)[0] / (2*math.pi)
        result = m.averaged_gamma_counterexample()
        self.assertAlmostEqual(invalid, result["invalid_conditional_mean_substitution"], places=10)
        self.assertAlmostEqual(truth, 64, places=11)
        self.assertGreater(invalid, truth)

    def test_rational_rejection_certificate(self):
        cert = m.averaged_gamma_certificate()
        bound = Fraction(cert["invalid_bound_strictly_exceeds"])
        self.assertEqual(bound, Fraction(19619, 288))
        self.assertGreater(bound, 64)
        self.assertGreater(m.averaged_gamma_counterexample()["invalid_conditional_mean_substitution"], float(bound))

    def test_retained_volatility_removes_conditional_jensen_loss(self):
        rng = np.random.default_rng(1904)
        x, z = rng.uniform(0, 2*math.pi, size=(2,1000))
        d = 1+.5*np.cos(z)
        y = np.cos(x)
        v = d*(1-y*y)
        recovered_d = v/(1-y*y)
        g = -recovered_d*y
        k = recovered_d**2*y*y-64*recovered_d**2*(1-y*y)
        ratio = np.maximum(g*g-k,0)/v
        np.testing.assert_allclose(ratio,64*d,rtol=2e-10,atol=1e-10)

    def test_no_variance_or_no_current_limits(self):
        self.assertEqual(m.averaged_gamma_counterexample(c=3,amplitude=0)["invalid_conditional_mean_substitution"],9)
        self.assertEqual(m.averaged_gamma_counterexample(c=0)["invalid_conditional_mean_substitution"],0)


class FiniteLagCountermodel(unittest.TestCase):
    def test_conditioning_pair_resolves_scalar_branch(self):
        for c in np.linspace(-.99,-.01,57):
            other=-1-c
            self.assertAlmostEqual(c*c+c-.5,other*other+other-.5)
            self.assertAlmostEqual(m.local_volatility(c)-m.local_volatility(other),(1+2*c)**3,places=13)

    def test_independent_matrix_exponential(self):
        # Generator on cos(x),sin(x),cos(2x),sin(2x); no closed-form ratio used.
        generator=np.zeros((4,4))
        generator[:2,:2]=[[-1,1],[-1,-1]]
        generator[2:,2:]=[[-4,2],[-2,-4]]
        initial=np.array([1,0,.5,0])
        for h in (.03,.2,1):
            plus=expm(h*generator)@initial
            minus=expm(h*generator.T)@initial
            for x in (.3,1.1,1.9,2.3):
                basis=np.array([math.cos(x),math.sin(x),math.cos(2*x),math.sin(2*x)])
                f=initial@basis
                forward=plus@basis-f
                backward=minus@basis-f
                g=(forward+backward)/(2*h)
                k=forward*backward/h**2
                ratio=(g*g-k)/m.local_volatility(math.cos(x))
                self.assertAlmostEqual(ratio,m.finite_h_naive_ratio(math.cos(x),h),places=10)

    def test_finite_positive_cutoff_has_rational_violation(self):
        cert=m.finite_h_certificate()
        lower=Fraction(cert["naive_truncated_value_strictly_exceeds"])
        self.assertEqual(lower,Fraction(6517,2750))
        self.assertGreater(lower,1)
        result=m.finite_h_truncated_quadrature()
        self.assertGreater(result["naive_truncated_value"],float(lower))
        self.assertLess(result["quadrature_estimated_error"],1e-7)

    def test_certificate_integration_annuli_pass_cutoff(self):
        eps=Fraction(1,15625)
        for sign in (-1,1):
            for delta in np.linspace(1/200,1/10,100):
                c=-.5+sign*delta
                self.assertGreaterEqual(m.local_volatility(c)+1e-15,float(eps))
                self.assertGreater(math.exp(-1)*math.sin(1)+math.exp(-4)*math.sin(2)*c,7/25)

    def test_small_volatility_limit_diverges_at_fixed_lag(self):
        values=[m.finite_h_truncated_quadrature(epsilon=e)["naive_truncated_value"]
                for e in (1e-3,1e-4,1e-5)]
        self.assertTrue(values[0]<values[1]<values[2])
        self.assertGreater(values[2],8)
        self.assertTrue(math.isinf(m.finite_h_naive_ratio(-.5)))

    def test_invertible_recalibration_independent_quadrature(self):
        for kappa in (0,.2,.49,-.4):
            h=1
            value=quad(lambda x: ((math.exp(-h)*math.sin(h)+
                kappa*math.exp(-4*h)*math.sin(2*h)*math.cos(x))/
                (h*(1+2*kappa*math.cos(x))))**2/math.pi,0,math.pi,
                epsabs=1e-11)[0]
            result=m.invertible_recalibration_counterexample(kappa=kappa)
            self.assertAlmostEqual(value,result["invalid_finite_h_functional"],places=10)
            self.assertGreater(result["minimum_recalibration_derivative"],0)

    def test_invertible_recalibration_rational_certificate(self):
        cert=m.recalibration_certificate()
        lower=Fraction(cert["invalid_finite_h_value_strictly_exceeds"])
        self.assertEqual(lower,Fraction(4205,396))
        self.assertGreater(lower,10)
        self.assertGreater(m.invertible_recalibration_counterexample()["invalid_finite_h_functional"],float(lower))

    def test_invertible_recalibration_short_time_limit(self):
        self.assertAlmostEqual(m.invertible_recalibration_counterexample(h=1e-9)["invalid_finite_h_functional"],1,places=6)
        for kwargs in ({"kappa":.5},{"kappa":-.5},{"h":0}):
            with self.assertRaises(ValueError):m.invertible_recalibration_counterexample(**kwargs)

    def test_short_lag_limit_away_from_critical_points(self):
        for c in (-.9,-.3,.2,.8):
            self.assertAlmostEqual(m.finite_h_naive_ratio(c,h=1e-7),1,places=5)

    def test_exact_equilibrium_conditional_variance_control(self):
        # Arbitrary mixtures of hidden states with reversible semigroup increments.
        rng=np.random.default_rng(772)
        for _ in range(50):
            weights=rng.dirichlet(np.ones(10))
            increments=rng.normal(size=10)
            drift=weights@increments
            product=weights@(increments**2)
            defect=drift*drift-product
            self.assertLessEqual(defect,1e-14)
            self.assertAlmostEqual(defect,-weights@((increments-drift)**2),places=12)


class PhysicalAccessControls(unittest.TestCase):
    def test_generalized_parity_has_positive_ordinary_energy(self):
        normalization=quad(lambda p:math.exp(math.cos(p)),-math.pi,math.pi)[0]
        moment=quad(lambda p:math.sin(p)**2*math.exp(math.cos(p)),-math.pi,math.pi)[0]/normalization
        result=m.parity_counterexample()
        self.assertAlmostEqual(moment,result["ordinary_entropy_rate"],places=13)
        self.assertEqual(result["generalized_entropy_rate"],0)
        self.assertGreater(moment,.44)

    def test_generator_generalized_detailed_balance_on_fourier_basis(self):
        # Evaluate L^dagger f and Theta L Theta f independently for real Fourier modes.
        for n,k in ((1,0),(0,1),(2,3),(1,2)):
            for q,p in ((.2,.9),(2,-.5),(-.3,1.4)):
                angle=n*q+k*p
                f=math.cos(angle)
                fp=-k*math.sin(angle)
                fq=-n*math.sin(angle)
                adjoint=-(n*n+k*k)*f-math.sin(p)*fp-math.sin(p)*fq
                # Theta f(q,p)=cos(nq-kp); evaluate L Theta f at (q,-p).
                angle_theta=n*q+(-k)*(-p)
                theta_f=math.cos(angle_theta)
                theta_fp=k*math.sin(angle_theta)
                theta_fq=-n*math.sin(angle_theta)
                conjugate=-(n*n+k*k)*theta_f-math.sin(-p)*theta_fp+math.sin(-p)*theta_fq
                self.assertAlmostEqual(adjoint,conjugate,places=12)

    def test_noisy_quadratic_variation_separates_clean_and_sensor_terms(self):
        low=m.noisy_realized_qv(100)
        high=m.noisy_realized_qv(10000)
        self.assertLess(abs(high["clean_expected_realized_qv"]-1),abs(low["clean_expected_realized_qv"]-1))
        self.assertAlmostEqual(high["sensor_contribution"],100*low["sensor_contribution"])
        self.assertGreater(high["expected_realized_qv"],200)
        self.assertEqual(m.noisy_realized_qv(100,noise_variance=0)["sensor_contribution"],0)

    def test_invalid_numeric_inputs_rejected(self):
        for kwargs in ({"amplitude":1},{"amplitude":float("nan")},{"c":float("inf")}):
            with self.assertRaises(ValueError):m.averaged_gamma_counterexample(**kwargs)
        for kwargs in ({"h":0},{"epsilon":0},{"epsilon":float("nan")}):
            with self.assertRaises(ValueError):m.finite_h_truncated_quadrature(**kwargs)
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError):m.noisy_realized_qv(n)


if __name__ == "__main__":
    unittest.main()
