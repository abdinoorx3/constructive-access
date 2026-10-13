from fractions import Fraction as F
from pathlib import Path
import copy,sys,unittest
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import contrast as c

class ContrastTests(unittest.TestCase):
    def contract(self):
        return dict(lags=[1,2,3],coefficients=[-1,2,-1],records=[50000]*3,
                    successes=[35849,32265,27572],durations=[F(3,20),F(3,10),F(9,20)])
    def test_continuous_support(self):
        self.assertEqual(c.support([1,2,3],[-1,2,-1])['reversible_support_upper'],0)
        # Discrete positive moments on an independent grid validate signs.
        for x in np.linspace(0,1,301):self.assertLessEqual(-x+2*x*x-x**3,1e-14)
    def test_equilibrium_generators_obey(self):
        rng=np.random.default_rng(781)
        for n in [2,4,7]:
            pi=rng.dirichlet(np.ones(n));flow=rng.uniform(.01,1,(n,n));flow=(flow+flow.T)/2
            q=flow/pi[:,None];np.fill_diagonal(q,0);np.fill_diagonal(q,-q.sum(axis=1))
            g=rng.uniform(-1,1,n)
            p=np.array([(1+(pi*g)@expm(q*t)@g)/2 for t in [.15,.3,.45]])
            self.assertLessEqual(np.dot([-1,2,-1],p),1e-13)
    def test_positive_certificate_and_tampering(self):
        contract=self.contract();cert=c.certify_counts(**contract)
        self.assertGreater(cert['entropy_production_rate_lower'],0)
        self.assertTrue(c.verify(cert,**contract))
        altered=copy.deepcopy(cert);altered['entropy_production_rate_lower']+=1
        self.assertFalse(c.verify(altered,**contract))
    def test_conditional_bias_can_close_gap(self):
        contract=self.contract();contract['conditional_radii']=[F(1,100)]*3
        self.assertEqual(c.certify_counts(**contract)['entropy_production_rate_lower'],0)
    def test_invalid_postselection_and_counts(self):
        contract=self.contract()
        with self.assertRaises(ValueError):c.certify_counts(**contract,selection='selected_on_test_data')
        contract['successes'][0]=50001
        with self.assertRaises(ValueError):c.certify_counts(**contract)
    def test_stationary_reference_mean_guard(self):
        contract=self.contract()
        with self.assertRaises(ValueError):c.certify_counts(**contract,mean_interval=(0,0))
        cert=c.certify_counts(**contract,mean_interval=(0,0),reference_mean_calibrated=True)
        self.assertTrue(cert['reference_mean_calibrated'])
    def test_exact_equilibrium_box_point(self):
        boxes=[(F(695,1000),F(739,1000)),(F(623,1000),F(668,1000)),(F(529,1000),F(574,1000))]
        witness=c.reversible_box_witness([1,2,3],boxes)
        self.assertTrue(c.verify_box_witness(witness,[1,2,3],boxes))
        witness['probabilities'][1]+=F(1,10)
        self.assertFalse(c.verify_box_witness(witness,[1,2,3],boxes))
    def test_box_search_failure_is_only_inconclusive(self):
        w=c.reversible_box_witness([1,2],[(0,0),(1,1)])
        self.assertEqual(w['status'],'search_inconclusive')
    def test_path_duration_cost(self):
        d=self.contract();a=c.certify_counts(**d)
        d['durations']=[t+1 for t in d['durations']];b=c.certify_counts(**d)
        self.assertEqual(a['contrast_lower'],b['contrast_lower'])
        self.assertLess(b['entropy_production_rate_lower'],a['entropy_production_rate_lower'])
    def test_square_root_directed(self):
        for x in [F(0),F(1),F(2),F(1,1000000),F(7,11)]:
            b=c.sqrt_upper(x);self.assertGreaterEqual(b*b,x)
            if b:self.assertLess((b-F(1,1<<90))**2,x)
    def test_allocation_fixed_cost(self):
        plan=c.allocation([-1,2,-1],[1,1,1],150000)
        self.assertEqual(plan['records'],[37500,75000,37500])
        self.assertLess(plan['variance_proxy'],F(6,50000))
        varying=c.allocation([-1,2,-1],[F(3,20),F(3,10),F(9,20)],45000)
        self.assertLessEqual(varying['actual_cost'],45000)
    def test_weight_optimization_inequality(self):
        rng=np.random.default_rng(2341);a=np.array([1,2,1]);t=np.array([.15,.3,.45])
        best=np.sum(a*np.sqrt(t))**2
        for _ in range(100):
            w=rng.dirichlet(np.ones(3));self.assertGreaterEqual(np.dot(w,t)*np.sum(a*a/w)+1e-12,best)
        w=a/np.sqrt(t);w/=w.sum()
        self.assertAlmostEqual(np.dot(w,t)*np.sum(a*a/w),best)

if __name__=='__main__':unittest.main(verbosity=2)
