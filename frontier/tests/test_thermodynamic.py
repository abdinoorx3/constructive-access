"""Independent numerical cross-checks of the exact certificate implementation."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util
import math
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.integrate import quad

spec = importlib.util.spec_from_file_location('thermodynamic',Path(__file__).resolve().parents[1]/'code'/'thermodynamic.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def stationary(Q):
    n=len(Q); A=Q.T.copy(); A[-1,:]=1
    b=np.zeros(n);b[-1]=1
    return np.linalg.solve(A,b)


class ThermodynamicTests(unittest.TestCase):
    def test_exact_log_enclosures(self):
        import mpmath as mp
        mp.mp.dps=100
        for x in [F(1,40),F(1,3),F(1),F(5,4),F(3),F(20)]:
            lo,hi=m.log_bounds(x,200)
            exact=mp.log(mp.mpf(x.numerator)/x.denominator)
            self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,exact+mp.mpf('1e-95'))
            self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,exact-mp.mpf('1e-95'))

    def test_sharp_factor_four(self):
        for x in [1e-3,.01,.1,.5,.9,.999]:
            a,b=1+x,1-x
            rate=a*math.log(2*a/(a+b))+b*math.log(2*b/(a+b))
            sigma=(a-b)*math.log(a/b)
            self.assertLessEqual(rate,sigma/4+1e-14)
        self.assertLess(rate/sigma,.1)
        x=1e-3;a=1+x;b=1-x
        rate=a*math.log(a)+b*math.log(b)
        sigma=(a-b)*math.log(a/b)
        self.assertAlmostEqual(rate/sigma,.25,places=6)

    def test_random_network_path_and_endpoint_bounds(self):
        rng=np.random.default_rng(78231)
        for n in range(3,9):
            for _ in range(10):
                Q=rng.uniform(.01,3,(n,n));np.fill_diagonal(Q,0);Q-=np.diag(Q.sum(1))
                pi=stationary(Q);star=Q.T*pi[None,:]/pi[:,None];S=(Q+star)/2
                sigma=0.;divergence=0.
                for i in range(n):
                    for j in range(i):
                        a,b=pi[i]*Q[i,j],pi[j]*Q[j,i]
                        sigma+=(a-b)*math.log(a/b)
                        divergence+=a*math.log(2*a/(a+b))+b*math.log(2*b/(a+b))
                self.assertLessEqual(divergence,sigma/4+1e-11)
                self.assertLess(np.max(abs(pi@S)),1e-12)
                f=rng.uniform(-1,1,n)
                old=math.inf
                for t in [.03,.1,.7,2.]:
                    c=f@(pi[:,None]*expm(t*S))@f
                    self.assertGreaterEqual(c,(pi@f)**2-1e-12)
                    self.assertLessEqual(c,old+1e-12);old=c
                    p=(1+f@(pi[:,None]*expm(t*Q))@f)/2
                    q=(1+c)/2
                    kl=p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))
                    self.assertLessEqual(kl,t*divergence+1e-11)

    def test_ring_certificate_vs_expm(self):
        n=40;Q=np.zeros((n,n))
        for i in range(n):Q[i,(i+1)%n]=20;Q[i,(i-1)%n]=1;Q[i,i]=-21
        f=np.r_[np.ones(n//2),-np.ones(n//2)]
        for t in [F(8,5),F(21,10)]:
            lo,hi=m.outward_decimal(m.ring_same_bounds(n,F(20),F(1),t))
            independent=(1+f@expm(float(t)*Q)@f/n)/2
            self.assertLessEqual(float(lo)-1e-14,independent)
            self.assertGreaterEqual(float(hi)+1e-14,independent)

    def test_stochastic_detector_ring_certificate(self):
        n=40;Q=np.zeros((n,n))
        for i in range(n):Q[i,(i+1)%n]=20;Q[i,(i-1)%n]=1;Q[i,i]=-21
        f=[F(round(10**6*math.cos(2*math.pi*i/n)),10**6) for i in range(n)]
        fd=np.array([float(z) for z in f])
        p=[]
        for t in [F(3,20),F(3,10),F(9,20)]:
            lo,hi=m.outward_decimal(m.ring_same_bounds(n,F(20),F(1),t,detector=f))
            independent=(1+fd@expm(float(t)*Q)@fd/n)/2
            self.assertLessEqual(float(lo)-1e-14,independent)
            self.assertGreaterEqual(float(hi)+1e-14,independent)
            p.append((lo+hi)/2)
        self.assertGreater(p[-1],F(1,2))
        self.assertGreater(p[0],p[1]);self.assertGreater(p[1],p[2])
        self.assertLess(p[0]-2*p[1]+p[2],0)

    def test_multilag_certificate_detects_positive_revival(self):
        p=[F(512,1000),F(736,1000)];t=[F(8,5),F(21,10)];w=[F(1,2)]*2
        self.assertEqual(m.single_lag_entropy_lower(p[0],t[0]),0)
        self.assertEqual(m.single_lag_entropy_lower(p[1],t[1]),0)
        result=m.order_entropy_lower(p,t,w)
        self.assertEqual(result['projection'],[F(624,1000)]*2)
        self.assertGreater(float(result['rate_lower']),.0584)

    def test_isotonic_against_convex_numerical_optimizer(self):
        rng=np.random.default_rng(182)
        for _ in range(30):
            p=[F(int(i),100) for i in rng.integers(1,100,5)]
            w=[F(1,5)]*5; exact=m.decreasing_kl_projection(p,w)
            pf=np.array([float(x) for x in p]); q0=np.full(5,.7)
            fun=lambda q:np.mean(pf*np.log(pf/q)+(1-pf)*np.log((1-pf)/(1-q)))
            result=minimize(fun,q0,method='SLSQP',bounds=[(.5,.999999)]*5,
                constraints=[{'type':'ineq','fun':lambda q:q[:-1]-q[1:]}],options={'ftol':1e-12,'maxiter':200})
            self.assertTrue(result.success)
            self.assertAlmostEqual(fun(np.array([float(x) for x in exact])),result.fun,places=8)

    def test_stationary_binary_pair_has_zero_reversal_kl(self):
        Q=np.array([[-3.,2.,1.],[1.,-4.,3.],[4.,1.,-5.]])
        pi=stationary(Q);f=np.array([1.,0.,0.]);P=expm(.4*Q)
        joint=np.zeros((2,2))
        for i in range(3):
            for j in range(3):joint[int(f[i]),int(f[j])]+=pi[i]*P[i,j]
        self.assertLess(np.max(abs(joint-joint.T)),1e-14)

    def test_reversible_transient_counterexample(self):
        # j=0, i=1, l=2, k=3; B={j,k}; all stationary currents vanish.
        Q=np.array([[-1.01,.01,1.,0.],[.01,-1.01,0.,1.],[.1,0.,-.11,.01],[0.,.1,.01,-.11]])
        pi=stationary(Q)
        self.assertLess(np.max(abs(pi[:,None]*Q-(pi[:,None]*Q).T)),1e-14)
        f=np.array([1.,0.,0.,1.]);contrast=(expm(5*Q)[0]-expm(5*Q)[1])@f
        self.assertLess(contrast,-.7)

    def test_finite_count_certificate_and_rejections(self):
        r=m.from_counts(5000,2560,5000,3680,F(1,50),F(1,20),F(8,5),F(21,10))
        self.assertGreater(float(r['rate_lower']),.039)
        self.assertEqual(m.two_lag_entropy_lower(F(7,10),F(6,10),F(1),F(2)),0)
        with self.assertRaises(ValueError):
            m.from_counts(100,50,100,70,F(1,50),F(1,20),F(1),F(2))
        with self.assertRaises(TypeError):m.single_lag_entropy_lower(.4,F(1))
        with self.assertRaises(ValueError):m.binary_kl_bounds(F(1,2),F(0))

    def test_odd_variable_equilibrium_control(self):
        # Underdamped equilibrium position autocorrelation at half a period.
        gamma,omega=0.1,2.0;t=math.pi/omega
        covariance=math.exp(-gamma*t)*(math.cos(omega*t)+gamma/omega*math.sin(omega*t))
        self.assertLess(covariance,0)
        # Physical entropy production is zero at equilibrium; ordinary even
        # state Markov reversal is inapplicable when momentum is odd.

    def test_preparation_housekeeping_path_bound(self):
        Q=np.array([[-3.,2.,1.],[1.,-4.,3.],[4.,1.,-5.]])
        pi=stationary(Q);star=Q.T*pi[None,:]/pi[:,None];S=(Q+star)/2
        nu=np.array([.8,.1,.1]);row_hk=np.zeros(3);row_div=np.zeros(3)
        for i in range(3):
            for j in range(3):
                if i!=j:
                    row_hk[i]+=Q[i,j]*math.log(Q[i,j]/star[i,j])
                    row_div[i]+=Q[i,j]*math.log(Q[i,j]/S[i,j])
        self.assertTrue(np.all(row_hk>=-1e-12))
        self.assertTrue(np.all(row_div<=row_hk/2+1e-12))
        initial=float(np.sum(nu*np.log(nu/pi)))
        t=.8
        true_path=initial+quad(lambda s:nu@expm(s*Q)@row_div,0,t)[0]
        upper_path=initial+quad(lambda s:nu@expm(s*Q)@row_hk,0,t)[0]/2
        self.assertLessEqual(true_path,upper_path)
        f=np.array([1.,-1.,1.])
        p=(1+f@(nu[:,None]*expm(t*Q))@f)/2
        q=(1+f@(pi[:,None]*expm(t*S))@f)/2
        endpoint=p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))
        self.assertLessEqual(endpoint,true_path)

    def test_preparation_factor_half_sharpness(self):
        ratios=[]
        for epsilon in [.1,.01,.001,.0001]:
            delta=epsilon**2
            Q=np.array([[0.,1-delta,delta],[1-epsilon,0.,1.],[epsilon,1-epsilon+delta,0.]])
            Q-=np.diag(Q.sum(1));pi=stationary(Q)
            self.assertLess(np.max(abs(pi-np.ones(3)/3)),1e-12)
            p=np.array([1-delta,delta]);q=np.array([1-epsilon,epsilon])
            numerator=np.sum(p*np.log(2*p/(p+q)))
            denominator=np.sum(p*np.log(p/q))
            ratios.append(numerator/denominator)
        self.assertTrue(all(a<b for a,b in zip(ratios,ratios[1:])))
        self.assertLess(ratios[-1],.5)
        self.assertGreater(ratios[-1],.499)

    def test_preparation_can_pay_for_record_separation(self):
        Q=np.array([[-9.,9.],[1.,-1.]])
        pi=stationary(Q);nu=np.array([1.,0.]);f=np.array([1.,-1.])
        p=(1+f@(nu[:,None]*expm(Q))@f)/2
        self.assertLess(p,.11)
        J=p*math.log(2*p)+(1-p)*math.log(2*(1-p))
        self.assertGreater(J,.36)
        self.assertLess(J,math.log(1/pi[0]))
        out=m.two_lag_resource_frontier(F(512,1000),F(736,1000),F(8,5),F(21,10),F(1))
        self.assertGreater(out['record_information_lower'],0)
        self.assertEqual(out['integrated_housekeeping_lower'],0)


if __name__ == '__main__':unittest.main()
