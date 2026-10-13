from pathlib import Path
import sys,unittest,copy,math
from fractions import Fraction as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import energy_certificate as e
import numpy as np
from scipy.linalg import expm


class EnergyCertificateTests(unittest.TestCase):
    def test_log_enclosures_and_exact_input(self):
        import mpmath as mp
        mp.mp.dps=90
        for value in [F(1,1000),F(7,9),F(1),F(123),F(10**10)]:
            lo,hi=e.log_interval(value);truth=mp.log(mp.mpf(value.numerator)/value.denominator)
            self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,truth)
            self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,truth)
        e.log_interval(F(1))
        for value in [True,1.0]:
            with self.assertRaises(TypeError):e.log_interval(value)

    def test_polynomial_bounds_against_stationary_points(self):
        import sympy as sp
        x=sp.Symbol('x')
        cases=[[0,1,-1],[1,-2,0,3,-2],[2,0,-6,4]]
        for powers in cases:
            poly=sum(c*x**j for j,c in enumerate(powers));roots=sp.solve(sp.diff(poly,x),x)
            points=[0,1]+[r for r in roots if r.is_real and 0<=r<=1]
            true_max=max(float(poly.subs(x,r)) for r in points)
            bounds=[e.polynomial_upper(powers,d) for d in [0,2,5]]
            self.assertTrue(all(float(v)+1e-14>=true_max for v in bounds))
            self.assertGreaterEqual(bounds[0],bounds[1]);self.assertGreaterEqual(bounds[1],bounds[2])

    def test_reversible_models_cannot_produce_false_bound(self):
        rng=np.random.default_rng(765931)
        for _ in range(35):
            n=4;pi=rng.dirichlet(np.ones(n));a=rng.uniform(.1,2,(n,n));a=(a+a.T)/2;np.fill_diagonal(a,0)
            Q=a/pi[:,None];Q-=np.diag(Q.sum(axis=1));g=rng.uniform(-1,1,n)
            probs=[.5+.5*(pi*g)@expm(Q*.1*k)@g for k in [1,2,3]]
            boxes=[(F(math.floor(p*10**10),10**10),F(math.ceil(p*10**10),10**10)) for p in probs]
            scores=[(F(int(rng.integers(1,100)),25),F(int(rng.integers(1,100)),25)) for _ in probs]
            cert=e.certify([1,2,3],F(1,10),[F(1,3)]*3,boxes,(-1,1),scores)
            self.assertEqual(cert['entropy_production_rate_lower'],0)

    def test_joint_box_revival_certificate_and_binding(self):
        cert=e.example()['certificate'];self.assertGreater(cert['entropy_production_rate_lower'],F(4,100))
        data={k:cert[k] for k in ['lags','step','weights','intervals','mean_interval','scores','depth']}
        self.assertTrue(e.verify(cert,**data))
        fake=copy.deepcopy(cert);fake['normalizer_upper']/=2
        self.assertFalse(e.verify(fake,**data))

    def test_curvature_detects_beyond_monotonicity(self):
        row=e.curvature_example();self.assertGreater(row['entropy_rate_lower_float'],.0007)
        boxes=row['exact_probability_intervals'];self.assertTrue(all(lo>F(1,2) for lo,hi in boxes))
        self.assertGreater(boxes[0][0],boxes[1][1]);self.assertGreater(boxes[1][0],boxes[2][1])
        self.assertLess(boxes[0][1]-2*boxes[1][0]+boxes[2][1],0)
        self.assertEqual(row['negative_correlation_bound'],0)
        self.assertEqual(row['monotonic_revival_bound'],0)

    def test_transient_mean_cannot_calibrate_reference(self):
        cert=e.example()['certificate'];data={k:cert[k] for k in ['lags','step','weights','intervals','mean_interval','scores','depth']}
        data['mean_interval']=(F(-1,100),F(1,100));narrow=e.certify(**data)
        with self.assertRaises(ValueError):e.resource_frontier(narrow)
        out=e.resource_frontier(narrow,F(0),reference_mean_calibrated=True)
        self.assertEqual(out['integrated_housekeeping_lower'],2*narrow['record_kl_lower'])
        broad=e.resource_frontier(cert,cert['record_kl_lower']);self.assertEqual(broad['integrated_housekeeping_lower'],0)

    def test_fixed_counts_joint_coverage(self):
        cert=e.example()['certificate']
        out=e.from_counts(cert['lags'],cert['step'],cert['weights'],[5000,5000],[2560,3680],F(11,500),F(1,20),cert['scores'])
        self.assertGreater(out['certificate']['entropy_production_rate_lower'],0)
        with self.assertRaises(ValueError):e.from_counts([1],1,[1],[100],[50],F(1,100),F(1,20),[(1,1)])

    def test_numerical_proposal_is_checked_on_continuous_domain(self):
        row=e.curvature_example();c=row['certificate']
        design=e.design(c['lags'],c['step'],c['weights'],c['intervals'],grid=17)
        candidate=design['certificate']
        self.assertGreater(candidate['record_kl_lower'],0)
        contract={k:candidate[k] for k in ['lags','step','weights','intervals','mean_interval','scores','depth']}
        self.assertTrue(e.verify(candidate,**contract))
        self.assertIn('exact continuous-domain',design['validity'])

    def test_validation_and_no_signal(self):
        cert=e.certify([1],1,[1],[(F(1,2),F(1,2))],(-1,1),[(1,1)])
        self.assertEqual(cert['record_kl_lower'],0)
        with self.assertRaises(ValueError):e.certify([0],1,[1],[(0,1)],(-1,1),[(1,1)])
        with self.assertRaises(ValueError):e.certify([1],1,[1],[(0,1)],(-1,1),[(0,1)])
        with self.assertRaises(ValueError):e.polynomial_upper([1,2],13)

if __name__=='__main__':unittest.main()
