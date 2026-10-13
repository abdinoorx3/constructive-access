import math
from fractions import Fraction as F
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'code'))
import preparation as p


class PreparationTests(unittest.TestCase):
    def test_exact_nonclassical_certificate(self):
        lo,hi=p.depolarizing_contrast_bounds(F(1,10),F(2),F(9,4),F(9,10))
        self.assertLess(hi,-F(18,125))
        self.assertLess(hi-lo,F(1,10**50))

    def test_independent_lindblad_integration(self):
        import numpy as np
        from scipy.linalg import expm
        gamma=.1;omega=2.;horizon=2.25
        x=np.array([[0,1],[1,0]],complex);eye=np.eye(2)
        # Independently vectorize the explicitly specified superoperator.
        columns=[]
        for k in range(4):
            b=np.zeros((2,2),complex);b.reshape(-1)[k]=1
            columns.append((-1j*omega/2*(x@b-b@x)+gamma*(np.trace(b)*eye/2-b)).reshape(-1))
        l=np.column_stack(columns)
        block=np.zeros((8,8),complex);block[:4,:4]=l;block[:4,4:]=np.eye(4)
        integrated=expm(horizon*block)[:4,4:]/horizon
        delta=np.diag([1.,-1.]).reshape(-1)
        value=(integrated@delta).reshape(2,2)[0,0].real*.9
        lo,hi=p.depolarizing_contrast_bounds(F(1,10),F(2),F(9,4),F(9,10))
        self.assertAlmostEqual(value,float(lo),places=13)
        self.assertAlmostEqual(value,float(hi),places=13)
        # Directly verify the fixed quadrature preparation and its sign.
        quadrature_horizon=1.75
        integrated=expm(quadrature_horizon*block)[:4,4:]/quadrature_horizon
        y=np.array([[0,-1j],[1j,0]])
        rho_y=(eye+y)/2
        rho_j=np.diag([1.,0.])
        difference=(integrated@(rho_y-rho_j).reshape(-1)).reshape(2,2)[0,0].real*.9
        lo,hi=p.quadrature_preparation_gap_bounds(F(1,10),F(2),F(7,4),F(9,10))
        self.assertAlmostEqual(difference,float(lo),places=13)
        self.assertAlmostEqual(difference,float(hi),places=13)

    def test_unitary_limit(self):
        lo,hi=p.depolarizing_contrast_bounds(F(0),F(2),F(9,4))
        self.assertLess(lo,hi)
        self.assertAlmostEqual(float(lo),math.sin(4.5)/4.5,places=14)

    def test_exponential_clock_does_not_detect(self):
        for gamma in (F(0),F(1,10),F(3)):
            for omega in (F(1,10),F(2),F(100)):
                self.assertGreater(p.exponential_clock_contrast(gamma,omega,F(1)),0)

    def test_fixed_quadrature_and_optimized_preparation(self):
        qlo,qhi=p.quadrature_preparation_gap_bounds(F(1,10),F(2),F(7,4),F(9,10))
        olo,ohi=p.optimized_preparation_gap_bounds(F(1,10),F(2),F(7,4),F(9,10))
        self.assertGreater(qlo,F(257,1000))
        self.assertGreater(olo,qhi)
        # Strong damping removes orthogonal reversal but need not remove
        # the spectral-preparation effect; this is not a power guarantee.
        lo,hi=p.optimized_preparation_gap_bounds(F(10),F(2),F(1))
        self.assertGreater(lo,0)

    def test_classical_baseline_beats_simple_uniform(self):
        result=p.comparison_benchmark()
        scores={(r['method'],r['objective']):r['hoeffding_proxy'] for r in result['rows']}
        for objective in ('physical_records','expected_evolution_time'):
            self.assertLess(scores['survival_nonlinear',objective],scores['uniform_orthogonal',objective])
            self.assertLess(scores['uniform_quadrature',objective],scores['survival_nonlinear',objective])

    def test_exact_baseline_subtracted_witness(self):
        lo,hi=p.baseline_subtracted_witness_bounds(F(1,10),F(2),F(2),F(-12,37),F(35,37),F(9,10))
        self.assertGreater(lo,F(27,25))
        self.assertLess(hi-lo,F(1,10**50))
        import numpy as np
        from scipy.linalg import expm
        x=np.array([[0,1],[1,0]],complex)
        y=np.array([[0,-1j],[1j,0]],complex)
        z=np.diag([1.,-1.]);eye=np.eye(2)
        columns=[]
        for k in range(4):
            b=np.zeros((2,2),complex);b.reshape(-1)[k]=1
            columns.append((-1j*(x@b-b@x)+.1*(np.trace(b)*eye/2-b)).reshape(-1))
        l=np.column_stack(columns)
        block=np.zeros((8,8),complex);block[:4,:4]=l;block[:4,4:]=np.eye(4)
        average=expm(2*block)[:4,4:]/2
        delta=-12/37*y+35/37*z
        c0=delta[0,0].real
        ct=(average@delta.reshape(-1)).reshape(2,2)[0,0].real
        self.assertAlmostEqual(.9*(c0-ct),float(lo),places=13)

    def test_trig_enclosures(self):
        import mpmath as mp
        mp.mp.dps=100
        for x in (F(-9,2),F(0),F(1,7),F(9,2)):
            for kind in ('sin','cos'):
                lo,hi=p.trig_bounds(x,kind)
                actual=getattr(mp,kind)(mp.mpf(x.numerator)/x.denominator)
                self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,actual)
                self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,actual)

    def test_error_accounting(self):
        self.assertEqual(p.error_envelope(F(1,100),F(1,200),F(1,1000),F(0),F(0)),F(17,1000))

    def test_reject_inexact_or_invalid_input(self):
        for gamma in (.1,True,'1/10'):
            with self.assertRaises(TypeError):
                p.depolarizing_contrast_bounds(gamma,F(2),F(9,4))
        with self.assertRaises(ValueError):
            p.depolarizing_contrast_bounds(F(-1),F(2),F(9,4))
        with self.assertRaises(ValueError):
            p.error_envelope(F(-1),F(0),F(0),F(0),F(0))

    def test_seeded_models_and_failure_controls(self):
        result=p.make_results()
        self.assertEqual(result['classical_checks'],60)
        self.assertGreaterEqual(result['classical_min_margin'],0)
        self.assertLess(result['negative_controls']['classical_coarse_uniform_contrast_at_1'],0)
        self.assertLess(result['negative_controls']['classical_fixed_time_contrast_at_2'],0)

if __name__=='__main__': unittest.main()
