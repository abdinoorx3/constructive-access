import sys
import unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from detector_boundary import (bounded_hazard_response, reset_defect_lower_bound,
    gate_dominance, contamination_envelope, _exp_negative, make_results, classical_hazard_lower_bound, mixed_preparation_bound, three_state_exclusion)


def generator(n, edges):
    q=np.zeros((n,n))
    for x,y,r in edges:q[x,y]+=r;q[x,x]-=r
    return q


def integrated(q,t,B):
    n=len(q);matrix=np.zeros((n+1,n+1));matrix[:n,:n]=q
    for x in B:matrix[x,n]=1
    return expm(t*matrix)[:n,n]

class DetectorBoundaryTests(unittest.TestCase):
    def test_exact_input_contract(self):
        for bad in (True,.5,None):
            with self.assertRaises(TypeError):bounded_hazard_response(bad,1,1)
        for args in ((-1,1,1),(1,1,0),(1,1,1,2)):
            with self.assertRaises(ValueError):bounded_hazard_response(*args)

    def test_exponential_rational_enclosure(self):
        import mpmath as mp
        mp.mp.dps=100
        for x in (F(0),F(1,10),F(7,3),F(10)):
            lo,hi=_exp_negative(x)
            exact=mp.exp(-mp.mpf(x.numerator)/x.denominator)
            self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,exact)
            self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,exact)

    def test_equal_hazard_phase_exact_bracket(self):
        pos=bounded_hazard_response(1,1,'1593/1000')
        neg=bounded_hazard_response(1,1,'797/500')
        self.assertGreater(pos['lower_bound_interval'][0],0)
        self.assertLess(neg['lower_bound_interval'][1],0)

    def test_zero_hazard_no_defect_penalty(self):
        for a,b in ((0,1),(1,0),(0,0)):
            r=bounded_hazard_response(a,b,1,F(3,4))
            self.assertGreater(r['lower_bound_interval'][0],0)
            self.assertEqual(r['K'],(F(0),F(0)))

    def test_reset_penalty_monotonic(self):
        previous=None
        for eta in (F(0),F(1,4),F(1,2),F(1)):
            lower=bounded_hazard_response(1,2,3,eta)['lower_bound_interval']
            if previous:self.assertLess(lower[1],previous[0])
            previous=lower

    def test_random_unrestricted_classical_models(self):
        rng=np.random.default_rng(84721)
        for _ in range(180):
            q=rng.uniform(0,3,(4,4));np.fill_diagonal(q,0);np.fill_diagonal(q,-q.sum(axis=1))
            a=max(q[0,1]+q[0,2],q[3,1]+q[3,2]);b=max(q[1,0]+q[1,3],q[2,0]+q[2,3])
            t=rng.uniform(.01,4)
            u=integrated(q,t,{1,2});h=u[1]-u[0]
            numeric=-np.expm1(-a*t)/a-np.expm1(-b*t)/b-t
            self.assertGreaterEqual(h+2e-13,numeric)

    def test_partial_reset_random_models(self):
        rng=np.random.default_rng(289113)
        for _ in range(180):
            a,b,c,d,e,f,g,eta=rng.uniform(.01,3,8);eta=eta/3
            q=generator(4,[(0,1,a*(1-eta)),(0,2,a*eta),(1,0,b),(1,2,c),(1,3,d),
                           (2,0,e),(2,1,f),(2,3,g),(3,0,1)])
            exitcap=max(b+d,e+g);t=rng.uniform(.01,3)
            aa=-np.expm1(-a*t)/a;ab=-np.expm1(-exitcap*t)/exitcap
            G=(np.exp(-a*t)-np.exp(-exitcap*t))/(exitcap-a)
            lower=(1-eta)*G+eta*(aa+ab-t)
            u=integrated(q,t,{1,2});self.assertGreaterEqual(u[1]-u[0]+2e-13,lower)

    def test_sharp_and_stationary_approximations(self):
        r=make_results()
        self.assertLess(r['maximum_numerical_residual'],1e-12)
        rows=r['irreducible_stationary_channel_approximation']
        errors=[abs(row['H']-row['limit']) for row in rows]
        self.assertTrue(all(x>y for x,y in zip(errors,errors[1:])))
        self.assertTrue(all(row['native_current']>0 for row in rows))
        self.assertTrue(all(row['response_identity_error']<1e-12 for row in rows))

    def test_defect_inference(self):
        r=reset_defect_lower_bound('-1/10',1,1,2)
        self.assertGreater(r['minimum_reset_defect'],0)
        self.assertLess(r['minimum_reset_defect'],1)
        bad=reset_defect_lower_bound(-10,1,1,2)
        self.assertEqual(bad['status'],'incompatible_hazard_contract')
        protected=reset_defect_lower_bound(1,1,1,1)
        self.assertEqual(protected['minimum_reset_defect'],0)

    def test_inverse_activity_exact_certificate(self):
        from scipy.optimize import brentq
        for c in (F(-1,10),F(0),F(1,2),F(-9,10)):
            r=classical_hazard_lower_bound(c,2)
            f=lambda z:2*(-np.expm1(-z))/z-1-float(c)
            root=brentq(f,1e-10,float(2/(1+c)),xtol=1e-14)/2
            self.assertLessEqual(float(r['hazard_lower_bound']),root+1e-14)
            self.assertGreaterEqual(float(r['root_hazard_upper_bound']),root-1e-14)
            self.assertGreater(r['lower_endpoint_response_interval'][0],c)
        self.assertEqual(classical_hazard_lower_bound(1,1)['hazard_lower_bound'],0)
        self.assertEqual(classical_hazard_lower_bound(-1,1)['status'],'incompatible_finite_hazard_contract')
        large=classical_hazard_lower_bound('-999999/1000000',1)
        self.assertGreater(large['hazard_lower_bound'],1000000)

    def test_partial_initial_occupancy_sharpness(self):
        import mpmath as mp
        mp.mp.dps=100
        t=F(2);p=F(1,2);eta0=F(1,3);eta1=F(2,3)
        result=mixed_preparation_bound(1,1,t,p,eta0,eta1)
        G=mp.mpf(2)*mp.exp(-2);A=1-mp.exp(-2);K=2-2*A+G
        expected=(G-mp.mpf(2)/3*K)/2-mp.mpf(1)/6*(2-A)
        lo,hi=result['lower_bound_interval']
        self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,expected)
        self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,expected)
        unchanged=mixed_preparation_bound(1,1,1,0,1,F(1,4))
        self.assertEqual(unchanged['lower_bound_interval'],bounded_hazard_response(1,1,1,F(1,4))['lower_bound_interval'])
        no_protection=mixed_preparation_bound(1,1,F(1,10),1,1,1)
        self.assertLess(no_protection['lower_bound_interval'][1],0)

    def test_three_state_pure_and_imperfect_preparations(self):
        rng=np.random.default_rng(91028)
        for n in (2,3):
            for _ in range(100):
                q=rng.uniform(0,3,(n,n));np.fill_diagonal(q,0);np.fill_diagonal(q,-q.sum(axis=1))
                t=rng.uniform(.01,5);B={1}
                if n==3 and rng.integers(2):B.add(2)
                u=integrated(q,t,B)
                inside=np.array([float(i in B) for i in range(n)]);inside/=inside.sum()
                outside=np.array([float(i not in B) for i in range(n)]);outside/=outside.sum()
                self.assertGreater((inside-outside)@u,-1e-12)
                f=rng.uniform(0,1,n);mb=rng.dirichlet(np.ones(n));ma=rng.dirichlet(np.ones(n))
                block=np.zeros((n+1,n+1));block[:n,:n]=q;block[:n,n]=f
                uf=expm(t*block)[:n,n]
                contrast=(mb-ma)@uf/t;lower=mb@f-ma@f-1
                self.assertGreaterEqual(contrast+1e-12,lower)
                self.assertLessEqual(abs(mb@f-ma@f-contrast),1+1e-12)

    def test_four_state_dimension_certificate_and_time_dependent_failure(self):
        result=three_state_exclusion('-1/10','49/50','1/50')
        self.assertEqual(result['minimum_classical_states'],4)
        self.assertGreater(result['contrast_drop_lower_bound'],1)
        self.assertEqual(three_state_exclusion('-1/100','49/50','1/50')['status'],'inconclusive')
        self.assertEqual(three_state_exclusion(-2)['status'],'incompatible_binary_probability_contract')
        # Three-state time-dependent counterexample: j->k pulse, then i->j,
        # then hold. Each pulse has rate10 and duration1/10; final hold lasts1.
        q1=generator(3,[(1,2,10)]);q2=generator(3,[(0,1,10)])
        row=np.array([-1.,1.,0.]);h=row@integrated(q1,.1,{1})
        row=row@expm(.1*q1);h+=row@integrated(q2,.1,{1})
        row=row@expm(.1*q2);h+=row[1]
        self.assertLess(h,0)
        self.assertAlmostEqual(h,1.9/np.e-.9,places=13)
        # Detector reversal, even with Q=0, gives score2 on two states.
        initial_difference=np.array([-1.,1.]);f0=np.array([0.,1.]);f1=1-f0
        self.assertEqual(initial_difference@(f0-f1),2)

    def test_composed_gateway_strict_extension(self):
        edges=[(0,2),(2,3),(1,2),(3,2)]
        result=gate_dominance(4,edges,{1,2})
        relation={(r['higher'],r['lower']):r['path'] for r in result['relations']}
        primitives={(r['higher'],r['lower']) for r in result['primitive']}
        self.assertNotIn((1,0),primitives)
        self.assertIn((1,0),relation)
        self.assertGreater(len(relation[1,0]),2)
        rng=np.random.default_rng(973)
        for _ in range(40):
            q=generator(4,[(x,y,rng.uniform(.01,5)) for x,y in edges])
            u=integrated(q,float(rng.uniform(.01,5)),{1,2})
            self.assertGreaterEqual(u[1]-u[0],-1e-13)

    def test_closed_detector_certificate(self):
        r=gate_dominance(4,[(0,2),(1,2),(2,1),(3,0)],{1,2})
        self.assertTrue(any(x['higher']==1 and x['lower']==0 for x in r['relations']))

    def test_finite_calibration_cannot_detect_arbitrarily_slow_bypass(self):
        for delta in (F(1,10),F(1,100),F(1,1000)):
            envelope=contamination_envelope(delta,[F(1)]*10)
            self.assertEqual(envelope,min(1,10*delta))
            t=2/float(delta)
            h=-np.expm1(-float(delta)*t)/float(delta)-t-np.expm1(-t)
            self.assertLess(h,0)
        self.assertEqual(contamination_envelope(0,[1,2,3]),0)
        with self.assertRaises(ValueError):contamination_envelope(-1,[1])

if __name__=='__main__':unittest.main()
