import math
import unittest
import numpy as np
from scipy.linalg import expm
from physical_minimum import (ring_minimum, circulant_statistics, ring_generator,
                              sampled_mode, circle_covariance, phase_record_tv_bound,
                              pell_aliases, example, gated_ring_countermodel)

class PhysicalMinimumTests(unittest.TestCase):
    def test_matrix_realization_and_stationarity(self):
        q=ring_generator(20,.11458081181988637,.5504934894452805)
        np.testing.assert_allclose(q.sum(axis=0),0,atol=1e-14)
        np.testing.assert_allclose(q.sum(axis=1),0,atol=1e-14)
        f=np.cos(2*math.pi*np.arange(20)/20)
        for t in (.01,1,2,3):
            self.assertAlmostEqual(float(f@expm(t*q)@f/20),
                                   float(circle_covariance(t,.11458081181988637,.5504934894452805)),places=13)

    def test_nearest_neighbor_attains(self):
        for n in (3,4,10,30):
            r=ring_minimum(n,2,.1)
            rates=np.zeros(n-1);rates[0]=r['forward'];rates[-1]=r['backward']
            st=circulant_statistics(rates)
            for key in ('gamma','omega','entropy'):
                self.assertAlmostEqual(st[key],r[key],places=11)

    def test_random_nonlocal_rates_obey_minimum(self):
        rng=np.random.default_rng(48273)
        for n in range(3,25):
            for _ in range(8):
                st=circulant_statistics(np.exp(rng.uniform(-3,3,n-1)))
                lo=ring_minimum(n,st['gamma'],st['omega'])['entropy']
                self.assertGreaterEqual(st['entropy']+1e-10,lo)

    def test_antipodal_reversible_rate_is_not_free_damping_improvement(self):
        st=circulant_statistics([3.,5.,1.])
        lo=ring_minimum(4,st['gamma'],st['omega'])['entropy']
        self.assertGreater(st['entropy'],lo)

    def test_continuum_limit_and_monotonicity(self):
        bounds=[ring_minimum(n,1,2)['entropy'] for n in (8,16,32,128,10000)]
        self.assertTrue(all(a>b for a,b in zip(bounds,bounds[1:])))
        self.assertGreater(bounds[-1],4)
        self.assertLess(bounds[-1]-4,1e-5)

    def test_equilibrium_is_zero(self):
        self.assertEqual(ring_minimum(10,2,0)['entropy'],0)
        self.assertAlmostEqual(circulant_statistics([1,2,3,2,1])['entropy'],0)

    def test_invalid_or_infinite_entropy(self):
        for args in [(2,1,1),(4,1,2),(4,0,0),(3,1,float('nan'))]:
            with self.assertRaises(ValueError):ring_minimum(*args)
        self.assertTrue(math.isinf(circulant_statistics([1,0,0])['entropy']))
        with self.assertRaises(ValueError):circulant_statistics([0,1,0])

    def test_moment_inverse_and_fixed_calibration(self):
        z=sampled_mode(.5,.38,.18)
        self.assertAlmostEqual(z['circle_entropy_minimum'],2.644797825293866,places=12)
        np.testing.assert_allclose(circle_covariance([0,1,2],z['gamma'],z['principal_omega']),[.5,.38,.18])
        for args in [(1,.3,.1),(.5,.8,.1),(.5,0,0),(.5,0,float('nan'))]:
            with self.assertRaises(ValueError):sampled_mode(*args)

    def test_noncircular_same_correlation_has_lower_entropy(self):
        model=gated_ring_countermodel()
        q,pi,f=model['generator'],model['stationary'],model['detector']
        np.testing.assert_allclose(pi@q,0,atol=1e-15)
        self.assertAlmostEqual(float(pi@f),0)
        self.assertAlmostEqual(float(pi@(f*f)),.1)
        entropy=0.
        for i in range(5):
            for j in range(i+1,5):
                a,b=pi[i]*q[i,j],pi[j]*q[j,i]
                if a and b:entropy+=(a-b)*math.log(a/b)
        self.assertAlmostEqual(entropy,model['entropy'],places=14)
        self.assertLess(entropy,.25)
        for t in (0,.2,1,2,5):
            self.assertAlmostEqual(float((pi*f)@expm(t*q)@f),
                                   float(circle_covariance(t,1,.5,.1)),places=13)
        with self.assertRaises(ValueError):gated_ring_countermodel(amplitude=1)

    def test_exact_grid_alias_not_an_upper_bound(self):
        z=sampled_mode(.5,.38,.18)
        t=np.arange(10)
        first=circle_covariance(t,z['gamma'],z['principal_omega'])
        for k in (1,10,100):
            w=z['principal_omega']+2*math.pi*k
            np.testing.assert_allclose(circle_covariance(t,z['gamma'],w),first,atol=1e-12)
            self.assertGreater(w*w/z['gamma'],z['circle_entropy_minimum'])

    def test_two_irrational_cosine_lags_do_not_resolve_alias(self):
        t=np.array([1,math.sqrt(2)])
        w1,w2=math.pi*(math.sqrt(2)-1),math.pi*(math.sqrt(2)+1)
        np.testing.assert_allclose(np.cos(w1*t),np.cos(w2*t),atol=2e-15)
        self.assertNotAlmostEqual(abs(w1),abs(w2))

    def test_pell_sequence_is_arbitrarily_costly_and_close(self):
        rows=pell_aliases(count=9)
        for row in rows:
            self.assertEqual(row['p']**2-2*row['q']**2,1)
        self.assertGreater(rows[-1]['entropy'],1e13)
        self.assertLess(rows[-1]['two_interval_state_record_tv_upper'],1e-6)
        self.assertTrue(all(a['entropy']<b['entropy'] for a,b in zip(rows,rows[1:])))
        self.assertTrue(all(a['two_interval_state_record_tv_upper']>b['two_interval_state_record_tv_upper'] for a,b in zip(rows,rows[1:])))

    def test_tv_bound_resource_counting(self):
        self.assertEqual(phase_record_tv_bound([1],1,0),0)
        one=phase_record_tv_bound([1,math.sqrt(2)],1,.01)
        self.assertAlmostEqual(phase_record_tv_bound([1,math.sqrt(2)],1,.01,repeats=3),3*one)
        self.assertEqual(phase_record_tv_bound([1],1,math.pi,repeats=100),1)
        with self.assertRaises(ValueError):phase_record_tv_bound([0],1,.1)

    def test_reproducible_example(self):
        a=example(trials=50);b=example(trials=50)
        self.assertEqual(a,b)
        self.assertLess(a['max_matrix_covariance_error'],1e-13)
        self.assertGreaterEqual(a['random_circulant_checks']['min_entropy_margin'],-1e-10)

if __name__=='__main__':unittest.main()
