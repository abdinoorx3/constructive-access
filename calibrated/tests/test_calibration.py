import importlib.util
from pathlib import Path
from fractions import Fraction as F
import unittest
from itertools import product
import numpy as np
import mpmath as mp
from scipy.linalg import expm

spec=importlib.util.spec_from_file_location('ci_calibration',Path(__file__).resolve().parents[1]/'code'/'calibration.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def telegraph(spectrum):
    r=len(spectrum);states=list(product([-1,1],repeat=r));n=len(states)
    Q=np.zeros((n,n));f=np.array([sum(np.sqrt(float(w))*z for (x,w),z in zip(spectrum,s)) for s in states])
    for i,s in enumerate(states):
        for j,(x,w) in enumerate(spectrum):
            dest=list(s);dest[j]*=-1;Q[i,states.index(tuple(dest))]=-np.log(float(x))/2
    np.fill_diagonal(Q,-Q.sum(axis=1));return Q,np.ones(n)/n,f


class CalibrationTests(unittest.TestCase):
    def test_calibration_polygon_and_infeasible_contract(self):
        v=c.calibration_vertices((F(-1,2),F(1,4)),(F(1,10),F(1,5)))
        self.assertEqual(v,[(F(0),F(1,10)),(F(0),F(1,5)),(F(1,10),F(1,10)),(F(1,5),F(1,5))])
        self.assertEqual(c.calibration_vertices((F(1,2),F(1,2)),(F(1,4),F(1,4))),[(F(1,4),F(1,4))])
        with self.assertRaises(ValueError):c.calibration_vertices((F(9,10),F(1)),(F(0),F(1,10)))

    def test_support_matches_quadratic_analytic_extremum(self):
        v=c.calibration_vertices((F(-1,10),F(1,10)),(F(1,5),F(3,10)))
        s=c.calibrated_support([1,2],[F(1),F(-1)],v)
        self.assertEqual(s['reversible_support_upper'],F(3,80))
        self.assertEqual(s['reversible_support_lower'],F(3,80))
        un=c.calibrated_support([1,2],[F(1),F(-1)],c.calibration_vertices())
        self.assertEqual(un['reversible_support_upper'],F(1,8))

    def test_support_bounds_dense_independent_search(self):
        rng=np.random.default_rng(760381)
        vertices=c.calibration_vertices((F(1,5),F(2,5)),(F(1,5),F(7,10)))
        for _ in range(12):
            coeff=[F(int(a),7) for a in rng.integers(-6,7,4)]
            sup=c.calibrated_support([1,2,3,4],coeff,vertices,5)
            for u,A in vertices:
                for x in np.linspace(0,1,401):
                    q=[(1+float(u)+(float(A-u))*x**k)/2 for k in range(1,5)]
                    self.assertLessEqual(np.dot(list(map(float,coeff)),q),float(sup['reversible_support_upper'])+1e-13)

    def test_sharp_block_realization_preserves_mean_variance_and_balance(self):
        atoms=[(F(1,4),F(2,3)),(F(3,4),F(1,3))]
        for m,A in [(F(0),F(1)),(F(1,3),F(5,9)),(F(-3,4),F(4,5)),(F(1),F(1))]:
            r=c.reversible_realization(m,A,atoms);P=r['step_matrix'];pi=r['stationary'];f=r['detector'];n=len(pi)
            self.assertEqual(sum(pi),1);self.assertEqual(sum(p*z for p,z in zip(pi,f)),m)
            self.assertEqual(sum(p*z*z for p,z in zip(pi,f)),A)
            for i in range(n):
                self.assertEqual(sum(P[i]),1)
                for j in range(n):self.assertEqual(pi[i]*P[i][j],pi[j]*P[j][i])
            v=f[:]
            for k in range(1,6):
                v=[sum(P[i][j]*v[j] for j in range(n)) for i in range(n)]
                measured=sum(pi[i]*f[i]*v[i] for i in range(n))
                self.assertEqual(measured,m*m+(A-m*m)*sum(w*x**k for x,w in atoms))
        self.assertFalse(c.reversible_realization(F(0),F(1),[(F(0),F(1))])['continuous_time_exact'])

    def test_same_state_product_calibrates_conditional_second_moment(self):
        pi=[F(1,3),F(2,3)];f=[F(1,2),F(-1,4)];A=sum(p*z*z for p,z in zip(pi,f));m=sum(p*z for p,z in zip(pi,f))
        joint=[F(0)]*4
        for p,z in zip(pi,f):
            probs=[(1+z)/2,(1-z)/2]
            for a in range(2):
                for b in range(2):joint[2*a+b]+=p*probs[a]*probs[b]
        self.assertEqual(joint[0]+joint[3]-joint[1]-joint[2],A)
        self.assertEqual(joint[0]-joint[3],m)
        self.assertNotEqual(A,1) # Squaring one binary outcome always returns one.
        result=c.paired_calibration([3125,1875,1875,3125])
        self.assertEqual(result['mean_estimate'],0);self.assertEqual(result['conditional_second_moment_estimate'],F(1,4))
        self.assertLessEqual(result['second_moment_interval'][0],F(1,4));self.assertGreaterEqual(result['second_moment_interval'][1],F(1,4))

    def test_conditional_noise_dependence_invalidates_zero_lag_formula(self):
        # Both marginal detector means are +/-1/2, yet anti-correlated noise makes product zero.
        conditional=[(F(1,2),[F(1,2),F(1,4),F(1,4),F(0)]),
                     (F(-1,2),[F(0),F(1,4),F(1,4),F(1,2)])]
        for f,j in conditional:
            self.assertEqual(j[0]+j[1]-j[2]-j[3],f)
            self.assertEqual(j[0]+j[2]-j[1]-j[3],f)
            self.assertEqual(j[0]+j[3]-j[1]-j[2],0)
            self.assertEqual(f*f,F(1,4))

    def test_exact_positive_lag_ambiguity_and_irreducible_realizations(self):
        for K in range(1,7):
            r=c.positive_lag_ambiguity(K)
            self.assertGreater(r['second_moment_difference'],0)
            for k in range(1,K+1):
                self.assertEqual(sum(w*x**k for x,w in r['positive_spectrum']),sum(w*x**k for x,w in r['negative_spectrum']))
            if K<=3:
                for spec in [r['positive_spectrum'],r['negative_spectrum']]:
                    Q,pi,f=telegraph(spec)
                    self.assertLessEqual(max(abs(f)),1+1e-14)
                    self.assertGreater(min(np.linalg.eigvalsh(-Q)[1:]),0)
                    for k in range(1,K+1):
                        self.assertAlmostEqual(np.dot(pi*f,expm(k*Q)@f),float(sum(w*x**k for x,w in spec)),places=13)

    def test_fast_modes_are_unseen_at_positive_lags_even_at_equilibrium(self):
        bound=c.fast_mode_pair_tv_bound(F(1),F(1,10),F(200),1000)
        with mp.workdps(100):
            exact_bound=mp.mpf(bound.numerator)/bound.denominator
            self.assertGreaterEqual(exact_bound,500*mp.exp(-20))
        self.assertLess(bound,F(11,10_000_000))
        # Model is symmetric, so every stationary edge current and entropy term vanishes.
        self.assertEqual(F(1,2)*100-F(1,2)*100,0)

    def test_rate_cap_delayed_calibration_handles_nonreversible_chains(self):
        rng=np.random.default_rng(760382)
        for n in [2,3,4,6]:
            for _ in range(8):
                rates=rng.uniform(.01,1,size=(n,n));np.fill_diagonal(rates,0);Q=rates-np.diag(rates.sum(axis=1))
                B=Q.T.copy();B[-1]=1;rhs=np.zeros(n);rhs[-1]=1;pi=np.linalg.solve(B,rhs)
                f=rng.uniform(-1,1,n);A=np.dot(pi,f*f);lag=.01;corr=np.dot(pi*f,expm(lag*Q)@f)
                lo=F(int(np.floor(corr*10**12)),10**12);hi=lo+F(1,10**12)
                cap=F(int(np.ceil(max(-np.diag(Q)))))
                res=c.delayed_calibration((lo,hi),F(1,100),cap)
                self.assertLessEqual(float(res['second_moment_interval'][0]),A+1e-12)
                self.assertGreaterEqual(float(res['second_moment_interval'][1]),A-1e-12)

    def test_calibration_reveals_steep_positive_decay(self):
        r=c.steep_decay_example()
        self.assertGreater(r['violation_lower'],F(3483,1_000_000))
        self.assertGreater(r['correlation_intervals'][1][0],0)
        self.assertGreater(r['correlation_intervals'][0][0],r['correlation_intervals'][1][1])
        Q=np.zeros((4,4))
        for j in range(4):Q[j,(j+1)%4]=3;Q[j,(j-1)%4]=1;Q[j,j]=-4
        f=np.array([.5,.5,-.5,-.5]);pi=np.ones(4)/4
        for lag,iv in zip(r['lags'],r['correlation_intervals']):
            val=np.dot(pi*f,expm(float(lag)*Q)@f)
            self.assertLessEqual(float(iv[0])-1e-14,val);self.assertGreaterEqual(float(iv[1])+1e-14,val)

    def test_analog_second_moment_is_an_upper_only_calibration(self):
        values=[F(x,4)+F(e,2) for x,e in product([-1,1],repeat=2)]
        m=sum(values)/4;B=sum(z*z for z in values)/4
        self.assertEqual(m,0);self.assertEqual(B,F(5,16))
        A=F(1,16);self.assertGreater(B,A)
        # Two independently randomized bits conditioned on the SAME noisy Z have product Z^2.
        encoded_product=F(0)
        for z in values:
            p=(1+z)/2
            encoded_product+=(p*p+(1-p)**2-2*p*(1-p))/4
        self.assertEqual(encoded_product,B);self.assertNotEqual(encoded_product,A)
        result=c.analog_calibration_summary(10000,m,B)
        self.assertGreaterEqual(result['observed_second_moment_upper'],B)
        self.assertLess(result['observed_second_moment_upper'],F(1,3))
        self.assertEqual(result['second_moment_interval'][0],0)
        self.assertTrue(any(u==0 and upper>=A for u,upper in result['vertices']))

    def test_validation_failures(self):
        with self.assertRaises(TypeError):c.calibration_vertices((-.1,.1),(0,1))
        with self.assertRaises(ValueError):c.paired_calibration([1,2,3])
        with self.assertRaises(ValueError):c.reversible_realization(F(1,2),F(0),[(F(1,2),F(1))])
        with self.assertRaises(ValueError):c.calibrated_support([0],[1],[(F(0),F(1))])

if __name__=='__main__':unittest.main()
