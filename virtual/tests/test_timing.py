import importlib.util
from pathlib import Path
from fractions import Fraction as F
import unittest
import numpy as np
from scipy.linalg import expm
from itertools import product

spec=importlib.util.spec_from_file_location('virtual_timing',Path(__file__).resolve().parents[1]/'code'/'timing.py')
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def reversible(rng,n):
    pi=rng.dirichlet(np.ones(n)); a=rng.uniform(.01,.2,size=(n,n)); a=(a+a.T)/2
    np.fill_diagonal(a,0); Q=a/pi[:,None]; np.fill_diagonal(Q,-Q.sum(axis=1))
    return Q,pi


class TimingTests(unittest.TestCase):
    def test_iid_increment_common_channel(self):
        rng=np.random.default_rng(560219)
        for n in range(2,8):
            Q,pi=reversible(rng,n);f=rng.uniform(-1,1,n)
            P=.75*expm(Q)+.25*expm(3*Q)
            for k in [1,2,3]:
                direct=np.zeros_like(P)
                for steps in product([(1,.75),(3,.25)],repeat=k):
                    direct+=np.prod([w for _,w in steps])*expm(sum(v for v,_ in steps)*Q)
                np.testing.assert_allclose(direct,np.linalg.matrix_power(P,k),atol=2e-13)
            vals=[np.dot(pi*f,np.linalg.matrix_power(P,k)@f) for k in range(1,7)]
            for _ in range(5):
                self.assertGreaterEqual(min(vals),-2e-12)
                vals=[vals[i]-vals[i+1] for i in range(len(vals)-1)]

    def test_random_scale_positive_spectral_mixture(self):
        rng=np.random.default_rng(560220)
        for n in range(2,8):
            Q,pi=reversible(rng,n);f=rng.uniform(-1,1,n)
            root=np.sqrt(pi);S=root[:,None]*Q/root[None,:]
            eig,V=np.linalg.eigh(S);mass=(V.T@(root*f))**2
            for k in range(1,7):
                direct=np.dot(pi*f,(.5*expm(k*Q)+.5*expm(2*k*Q))@f)
                spectral=np.dot(mass,.5*np.exp(k*eig)+.5*np.exp(2*k*eig))
                self.assertAlmostEqual(direct,spectral,places=12)

    def test_two_state_random_scale_can_increase_common_channel_dimension(self):
        x=t.make_results()['unsafe_random_scale_for_same_dimension']
        self.assertEqual(x['hankel_determinant'],F(9,32768))
        self.assertNotEqual(x['hankel_determinant'],0)
        r=x['three_state_realization'];P=np.array(r['P'],dtype=float)
        rho=np.array(r['initial'],dtype=float);f=np.array(r['detector'],dtype=float)
        for k,p in enumerate(x['occupation']):
            self.assertEqual(rho@np.linalg.matrix_power(P,k)@f,float(p))

    def test_ordered_mean_does_not_preserve_monotonicity(self):
        x=t.make_results()['unsafe_settingwise_ordered_means']
        self.assertEqual(x['mean_lags'],[F(1),F(2)])
        self.assertEqual(x['increase'],F(1537,5120))
        self.assertEqual(x['physical_entropy_rate'],0)

    def test_stochastic_order_does_not_preserve_convexity(self):
        x=t.make_results()['unsafe_stochastically_ordered_clocks_for_full_cone']
        self.assertEqual(x['mean_lags'],[F(1),F(2),F(3)])
        self.assertGreater(x['correlations'][0],x['correlations'][1])
        self.assertGreater(x['correlations'][1],x['correlations'][2])
        self.assertEqual(x['second_difference'],F(-7171,40960))
        # A pointwise coupling exists: (1,1,2) with .9, (1,11,12) with .1.

    def test_disjoint_block_formula_by_full_path_enumeration(self):
        rng=np.random.default_rng(560221)
        for n in [2,3,4]:
            Q,pi=reversible(rng,n);f=rng.uniform(-1,1,n)
            L=.8;gap=.4;A=expm(Q*L);B=expm(Q*gap)
            direct=0.
            for a,b,c,d in product(range(n),repeat=4):
                direct+=pi[a]*A[a,b]*B[b,c]*A[c,d]*(f[a]+f[b])*(f[c]+f[d])/4
            g=(f+A@f)/2
            self.assertAlmostEqual(direct,np.dot(pi*g,B@g),places=12)

    def test_overlap_and_antisymmetric_detector_false_witnesses(self):
        r=t.make_results()
        self.assertEqual(r['unsafe_overlapping_symmetric_blocks']['increase'],F(93,1024))
        self.assertEqual(r['unsafe_antisymmetric_block']['correlation'],F(-9,128))
        self.assertEqual(r['unsafe_center_spacing_index']['normalized_second_moment_gap'],F(-225,4096))
        self.assertEqual(r['unsafe_center_spacing_index']['actual_gaps'],[0,2])
        self.assertLess(r['unsafe_antisymmetric_block']['same_probability'],F(1,2))
        # Full path enumeration for the overlap example, with independent randomized encodings.
        rate=np.log(2)/2;Q=np.array([[-rate,rate],[rate,-rate]])
        for D in [2,4]:
            times=sorted(set([0,D,4,D+4])); mats=[expm(Q*(b-a)) for a,b in zip(times,times[1:])]
            direct=0.
            for states in product([-1,1],repeat=len(times)):
                p=.5
                for j,P in enumerate(mats):p*=P[(states[j]+1)//2,(states[j+1]+1)//2]
                bytime=dict(zip(times,states))
                direct+=p*(bytime[0]+bytime[4])*(bytime[D]+bytime[D+4])/4
            self.assertAlmostEqual(direct,float(t.endpoint_block_correlation(F(1,2),4,D)),places=13)

    def test_full_duration_controls_entropy_rate_conversion(self):
        r=t.stationary_rate_bound(F(1,100),[F(1,2),F(1,2)],[F(1),F(2)],F(2))
        self.assertEqual(r['mean_full_duration'],F(11,2))
        self.assertEqual(r['stationary_entropy_rate_lower'],F(2,275))
        without=t.stationary_rate_bound(F(1,100),[F(1,2),F(1,2)],[F(1),F(2)])
        self.assertGreater(without['stationary_entropy_rate_lower'],r['stationary_entropy_rate_lower'])

    def test_invalid_clock_inputs(self):
        with self.assertRaises(TypeError):t.laplace_value([(1,F(1))],.5)
        with self.assertRaises(ValueError):t.laplace_value([(-1,F(1))],F(1,2))
        with self.assertRaises(ValueError):t.mean_clock([(1,F(1,2))])
        with self.assertRaises(ValueError):t.stationary_rate_bound(F(1),[F(1)],[F(0)])

if __name__=='__main__':unittest.main()
