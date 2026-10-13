from pathlib import Path
import importlib.util
import math
import unittest
import numpy as np
from scipy.integrate import quad
import sympy as sp

spec=importlib.util.spec_from_file_location("rare_bump",Path(__file__).with_name("rare_bump.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class RareBumpChecks(unittest.TestCase):
    def test_cutoff_curvature_exact_maximum(self):
        t=sp.symbols('t',real=True)
        q=1-10*t**3+15*t**4-6*t**5
        for k in (1,2):
            self.assertEqual(sp.diff(q,t,k).subs(t,0),0)
            self.assertEqual(sp.diff(q,t,k).subs(t,1),0)
        critical=sp.solve(sp.diff(q,t,3),t)
        maxima=[sp.simplify(abs(sp.diff(q,t,2).subs(t,c))) for c in critical]
        self.assertTrue(all(sp.simplify(v-10*sp.sqrt(3)/3)==0 for v in maxima))
        self.assertLess(float(maxima[0]),6)

    def test_radial_laplacian_cancellation(self):
        s=sp.symbols('s',positive=True)
        chi=sp.Function('chi')(s)
        radial=chi/s
        laplacian=sp.diff(radial,s,2)+2*sp.diff(radial,s)/s
        self.assertEqual(sp.simplify(laplacian-sp.diff(chi,s,2)/s),0)

    def test_generator_applied_twice(self):
        x,z=sp.symbols('x z',real=True)
        d=sp.Function('d')(z)
        f=sp.cos(x)
        L=lambda u:d*sp.diff(u,x,2)+sp.diff(u,z,2)
        self.assertEqual(sp.simplify(L(f)+d*f),0)
        self.assertEqual(sp.simplify(L(L(f))-(d*d-sp.diff(d,z,2))*f),0)

    def test_bump_is_bounded_with_exact_plateau(self):
        vals=[m.smooth_plateau(s) for s in np.linspace(0,1.2,101)]
        self.assertTrue(all(0<=v<=1 for v in vals))
        self.assertTrue(all(a>=b for a,b in zip(vals[:-1],vals[1:])))
        self.assertEqual(m.diffusivity(.01,100,.1),100)
        self.assertEqual(m.diffusivity(.2,100,.1),1)
        self.assertAlmostEqual(m.smooth_plateau(.75),.5)

    def test_stationary_mean_bump_within_volume_bound(self):
        profile_integral=quad(lambda s:s*s*m.smooth_plateau(s),0,1,points=[.5],epsabs=1e-12)[0]
        self.assertGreater(profile_integral,1/24)  # exact inner plateau volume
        self.assertLess(profile_integral,1/3)     # enclosing ball volume
        for amplitude,radius in ((10,.01),(100,.0001)):
            actual=.5+(amplitude-1)*radius**3*profile_integral/(4*math.pi**2)
            bounds=m.model_summary(amplitude,radius)
            self.assertGreater(actual,bounds['stationary_mean_Gamma_lower'])
            self.assertLessEqual(actual,bounds['stationary_mean_Gamma_upper'])

    def test_coupling_bound_contracts_while_suprema_grow(self):
        low=m.model_summary(10,.01)
        high=m.model_summary(1000,1e-6)
        self.assertLess(high['full_continuous_record_tv_upper'],low['full_continuous_record_tv_upper'])
        self.assertGreater(high['global_L2f_supremum_at_least'],low['global_L2f_supremum_at_least'])
        self.assertEqual(high['entropy_rate'],0)
        self.assertEqual(m.record_tv_upper(.5,100),1)
        self.assertAlmostEqual(m.record_tv_upper(.001,2,records=2),2*m.record_tv_upper(.001,1))

    def test_potential_initial_expectation_constant(self):
        for r in (.001,.1,.5):
            exact_geometric_bound=r*(2+r/2)**2/(4*math.pi**2)
            self.assertLess(exact_geometric_bound,9*r/64)

    def test_exact_transition_profile_derivatives(self):
        s=sp.symbols('s',positive=True)
        profile=1/(1+sp.exp(1/(1-s)-1/(s-sp.Rational(1,2))))
        point=sp.Rational(3,4)
        self.assertEqual(profile.subs(s,point),sp.Rational(1,2))
        self.assertEqual(sp.diff(profile,s).subs(s,point),-8)
        self.assertEqual(sp.diff(profile,s,2).subs(s,point),0)
        radial=sp.diff(profile,s,2)+2*sp.diff(profile,s)/s
        self.assertEqual(sp.simplify(radial.subs(s,point)),-sp.Rational(64,3))

    def test_bounded_diffusion_large_first_generator(self):
        x=sp.symbols('x',real=True)
        k=sp.symbols('k',positive=True,integer=True)
        n=4*k
        d=1+(1+sp.sin(n*x))/2  # z in plateau
        Lf=sp.diff(d*sp.diff(sp.cos(x),x),x)
        self.assertEqual(sp.simplify(Lf.subs(x,sp.pi/2)),-n/2)
        result=m.bounded_diffusion_first_generator(400,1/400**2)
        self.assertEqual(result['diffusion_upper_bound'],2)
        self.assertEqual(result['global_Lf_supremum_at_least'],200)
        self.assertLess(result['full_continuous_record_tv_upper_formula_value'],.001)
        with self.assertRaises(ValueError):m.bounded_diffusion_first_generator(6,.1)

    def test_bounded_diffusion_and_drift_large_second_generator(self):
        result=m.bounded_diffusion_second_generator(.001)
        self.assertEqual(result['diffusion_upper_bound'],2)
        self.assertEqual(result['global_Lf_supremum'],2)
        self.assertGreater(result['global_L2f_supremum_at_least'],20_000_000)
        self.assertLess(result['full_continuous_record_tv_upper_formula_value'],.01)

    def test_invalid_contract_inputs(self):
        for kwargs in ({'amplitude':.5,'radius':.1},{'amplitude':2,'radius':0},{'amplitude':2,'radius':.6}):
            with self.assertRaises(ValueError):m.model_summary(**kwargs)
        for records in (0,True,1.5):
            with self.assertRaises(ValueError):m.record_tv_upper(.1,1,records)
        with self.assertRaises(ValueError):m.record_tv_upper(.1,float('nan'))
        with self.assertRaises(ValueError):m.smooth_plateau(-1)

if __name__=="__main__":unittest.main()
