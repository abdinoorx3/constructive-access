import importlib.util
from pathlib import Path
from fractions import Fraction as F
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.stats import gamma as gamma_distribution

MODULE=Path(__file__).resolve().parents[1]/'code'/'spectral.py'
spec=importlib.util.spec_from_file_location('frontier_spectral',MODULE)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


class SpectralCertificates(unittest.TestCase):
    def test_exact_arithmetic_bounds_against_independent_numeric(self):
        for n in [3,4,5,8,16,32,62,100]:
            self.assertLessEqual(float(s.tangent_lower(n)),np.tan(np.pi/n)+1e-14)
        for x in [F(0),F(1,7),F(2),F(10000,3)]:
            lo,hi=s.sqrt_bounds(x,40)
            self.assertLessEqual(lo*lo,x); self.assertGreaterEqual(hi*hi,x)
            self.assertLessEqual(hi-lo,F(1,2**40))

    def test_resolvent_represents_physical_erlang_readout(self):
        Q=np.array([[-2.,2.,0.],[0.,-3.,3.],[1.,0.,-1.]])
        rho=np.array([.7,.3,0.]); f=np.array([.1,.9,.4]); rate=1.7
        R=rate*np.linalg.inv(rate*np.eye(3)-Q)
        for k in [1,2,4]:
            direct=quad(lambda t: (rho@expm(Q*t)@f)*gamma_distribution.pdf(t,k,scale=1/rate),0,np.inf,epsabs=1e-11)[0]
            self.assertAlmostEqual(direct,rho@np.linalg.matrix_power(R,k)@f,places=10)

    def test_quantum_target_erlang_integrals(self):
        u,v,moments=s.target_moments(5,F(1,10),F(2),F(17,10))
        for k in [1,2,6]:
            direct=quad(lambda t:.5*(1+np.exp(-.1*t)*np.cos(2*t))*gamma_distribution.pdf(t,k,scale=1/1.7),0,np.inf,epsabs=1e-10)[0]
            self.assertAlmostEqual(direct,float(moments[k]),places=9)

    def test_random_generators_satisfy_lens_and_root_product(self):
        rng=np.random.default_rng(310127)
        for n in [3,4,5,6,8]:
            cert=s.certificate(n,F(1,10),F(2),F(7,4))
            mu=complex(float(cert['mu_real']),float(cert['mu_imag']))
            t=float(cert['tangent_lower']); d=float(cert['spectral_distance'])
            v=mu.imag; pairs=cert['pair_count']
            for rep in range(24):
                rates=rng.exponential(3,size=(n,n)); np.fill_diagonal(rates,0)
                if rep%4==0: rates[rng.random((n,n))<.5]=0
                Q=rates-np.diag(rates.sum(axis=1)); R=1.75*np.linalg.inv(1.75*np.eye(n)-Q)
                roots=np.linalg.eigvals(R)
                self.assertLessEqual(max(abs(z)**2-z.real+t*abs(z.imag) for z in roots),2e-12)
                p=np.poly(R)
                lower=abs(1-mu)*d**pairs*v**(n-1-pairs)
                self.assertGreaterEqual(abs(np.polyval(p,mu))+1e-10,lower)
                self.assertLessEqual(np.abs(p).sum(),2**n+1e-9)
                self.assertAlmostEqual(np.abs(p).sum(),abs(np.polyval(p,-1)),places=8)
                normalized_lower=float(cert['one_distance'])/2*float(cert['real_factor'])**(n-1-2*pairs)*float(cert['pair_factor'])**pairs
                self.assertGreaterEqual(abs(np.polyval(p,mu))/np.abs(p).sum()+1e-12,normalized_lower)
            self.assertGreaterEqual(cert['gap_normalized'],cert['gap_simple'])

    def test_random_generator_record_never_in_forbidden_ball(self):
        rng=np.random.default_rng(921337)
        for n in [3,4,5,6]:
            cert=s.certificate(n,F(1,10),F(2),F(7,4))
            target=np.array(list(map(float,cert['moments'])))
            for rep in range(30):
                rates=rng.lognormal(0,2,size=(n,n)); np.fill_diagonal(rates,0)
                Q=rates-np.diag(rates.sum(axis=1)); R=1.75*np.linalg.inv(1.75*np.eye(n)-Q)
                rho=rng.dirichlet(np.ones(n)); f=rng.random(n)
                values=[]
                for k in range(n+2):
                    values.append(rho@np.linalg.matrix_power(R,k)@f)
                self.assertGreaterEqual(max(abs(target-np.array(values))),float(cert['gap']))

    def test_cyclic_realization_is_negative_control(self):
        # A four-state one-way cycle exactly produces damped cosine with gamma=omega.
        Q=np.zeros((4,4))
        for j in range(4): Q[j,(j+1)%4]=2; Q[j,j]=-2
        f=np.array([1.,.5,0.,.5]); rate=2.
        R=rate*np.linalg.inv(rate*np.eye(4)-Q)
        _,_,moments=s.target_moments(4,F(2),F(2),F(2))
        for k,m in enumerate(moments):
            self.assertAlmostEqual(np.linalg.matrix_power(R,k)[0]@f,float(m),places=13)
        self.assertEqual(s.certificate(4,F(2),F(2),F(2))['status'],'not_separated')

    def test_truncation_and_familywise_design_are_certified(self):
        cert=s.certificate(4,F(1,10),F(2),F(13,10)); plan=s.bounded_design(cert)
        self.assertGreater(cert['gap'],F(16816,10_000_000))
        for k in range(1,6):
            actual=gamma_distribution.sf(float(plan['maximum_duration']),k,scale=10/13)
            upper=s.erlang_tail_upper(k,F(13,10),plan['maximum_duration'])
            self.assertGreaterEqual(float(upper)+1e-20,actual)
            self.assertLessEqual(upper,cert['gap']/8)
        B=plan['hoeffding_exponent_lower']
        # Bound failure using exact e^-B enclosure, independent of the design's 2^-B test.
        self.assertLessEqual(2*plan['settings']*s.exp_minus_upper(B),plan['error_probability'])
        self.assertLess(2*plan['statistical_radius']+2*plan['tail_upper'],cert['gap'])

    def test_observation_box_needs_all_error_sources(self):
        cert=s.certificate(4,F(1,10),F(2),F(13,10)); gap=cert['gap']
        self.assertTrue(s.compatible_excluded(cert,cert['moments'],gap/4,gap/4,gap/4))
        self.assertFalse(s.compatible_excluded(cert,cert['moments'],gap/2,gap/4,gap/4))
        bad=list(cert['moments']); bad[2]+=2*gap
        self.assertFalse(s.compatible_excluded(cert,bad,gap/4))

    def test_inputs_enforce_exactness_and_scope(self):
        with self.assertRaises(TypeError): s.certificate(4,.1,F(2),F(2))
        with self.assertRaises(ValueError): s.certificate(2,F(1,10),F(2),F(2))
        with self.assertRaises(ValueError): s.certificate(4,F(-1),F(2),F(2))
        with self.assertRaises(ValueError): s.bounded_design(s.certificate(4,F(2),F(2),F(2)))


if __name__=='__main__': unittest.main()
