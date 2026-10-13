import importlib.util
from fractions import Fraction as F
from math import comb
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hierarchy',ROOT/'code'/'hierarchy.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class HierarchyTests(unittest.TestCase):
    def test_sperner_staircase(self):
        self.assertEqual([h.minimum_states(N) for N in [2,3,4,6,7,10,11,20,21,70,71]],
                         [2,3,4,4,5,5,6,6,7,8,9])

    def test_exact_and_noisy_certificates(self):
        self.assertTrue(h.certify_matrix_bounds(F(0),F(1,10),7)['certified'])
        out=h.certify_matrix_bounds(F(3,500),F(3,20),7,F(2,25))
        self.assertEqual(out['minimum_states'],5)
        self.assertEqual(out['margin'],F(1,1000))
        self.assertFalse(h.certify_matrix_bounds(F(1,100),F(1,10),7)['certified'])

    def test_boundary_is_not_strict_witness(self):
        self.assertFalse(h.certify_matrix_bounds(F(1,100),F(19,100),7,F(1,10))['certified'])
        self.assertFalse(h.certify_matrix_bounds(F(0),F(0),7)['certified'])

    def test_pi_and_qubit_intervals(self):
        import mpmath as mp
        mp.mp.dps=70
        lo,hi=h.pi_interval()
        to_mp=lambda f:mp.mpf(f.numerator)/mp.mpf(f.denominator)
        self.assertLess(to_mp(lo),mp.pi);self.assertGreater(to_mp(hi),mp.pi)
        for N,r in [(4,1000),(7,1000),(8,10000),(11,10000),(21,100000)]:
            for k,(lo,hi) in enumerate(h.qubit_probabilities(N,F(r)),1):
                val=(1-mp.exp(-2*mp.pi*k/(r*N))*mp.cos(2*mp.pi*k/N))/2
                self.assertLessEqual(to_mp(lo),val);self.assertGreaterEqual(to_mp(hi),val)
            self.assertTrue(h.certify_grid(h.qubit_probabilities(N,F(r)))['certified'])

    def test_exact_binomial_geometric_bounds(self):
        for n in [1,3,8,20]:
            for p in [F(0),F(1,7),F(1,2),F(6,7),F(1)]:
                masses=[F(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1)]
                for k in range(-1,n+2):
                    low=sum((m for j,m in enumerate(masses) if j<=k),F(0))
                    high=sum((m for j,m in enumerate(masses) if j>=k),F(0))
                    self.assertGreaterEqual(h.binomial_tail_upper(n,k,p,'lower'),low)
                    self.assertGreaterEqual(h.binomial_tail_upper(n,k,p,'upper'),high)

    def test_antichain_bound_sharp_for_abstract_factors(self):
        import itertools
        import numpy as np
        for n in range(2,8):
            sets=list(itertools.combinations(range(n),n//2))
            A=np.array([[int(k in s)/len(s) for k in range(n)] for s in sets])
            B=np.array([[int(k not in s) for s in sets] for k in range(n)])
            M=A@B
            self.assertTrue((M.diagonal()==0).all())
            self.assertGreater(min(M[i,j] for i in range(len(sets)) for j in range(len(sets)) if i!=j),0)
            self.assertEqual(h.minimum_states(len(sets)),n)

    def test_random_classical_models_do_not_violate(self):
        import numpy as np
        from scipy.linalg import expm
        rng=np.random.default_rng(618271)
        for n in range(1,6):
            N=comb(n,n//2)+1
            for rep in range(15):
                Q=rng.exponential(size=(n,n));np.fill_diagonal(Q,0)
                Q-=np.diag(Q.sum(axis=1))
                P=expm(Q*rng.uniform(.01,2))
                a=rng.dirichlet(np.ones(n));b=rng.random(n)
                ps=[]
                v=a.copy()
                for k in range(1,2*N):
                    v=v@P;ps.append(v@b)
                eps=ps[N-1];beta=min(p for k,p in enumerate(ps) if k!=N-1)
                self.assertLessEqual(beta,2*eps**.5-eps+1e-12)

    def test_ideal_discrete_classical_cycle_is_valid_larger_model(self):
        import numpy as np
        N=7;P=np.roll(np.eye(N),1,axis=1)
        a=np.eye(N)[0];b=np.sin(np.pi*np.arange(N)/N)**2
        vals=[]
        for k in range(1,2*N):
            a=a@P;vals.append(float(a@b))
        self.assertAlmostEqual(vals[N-1],0)
        self.assertTrue(h.certify_grid([(F(str(v)),F(str(v))) for v in vals])['certified'])
        self.assertLessEqual(h.minimum_states(N),N)

    def test_common_transition_contract_is_essential(self):
        # A single frozen hidden state with a detector chosen by step can emit
        # any desired p_k. Such changing readout is outside the factorization.
        fake=[F(1,2)]*13;fake[6]=F(0)
        self.assertTrue(h.certify_grid([(x,x) for x in fake])['certified'])

    def test_probability_input_validation(self):
        with self.assertRaises(TypeError):h.certify_matrix_bounds(.01,F(1,2),7)
        with self.assertRaises(ValueError):h.certify_grid([(F(0),F(1))]*12)
        with self.assertRaises(ValueError):h.certify_matrix_bounds(F(1,10),F(1,2),7,F(0))

    def test_frozen_sample_plan_exact_errors(self):
        ps=h.qubit_probabilities(7,F(1000))
        # Fixed independently inspectable acceptance region from the search.
        settings=[(1900,313),(100,22),(100,22),(100,22),(100,22),(1800,297),
                  (10300,48),(1800,297),(100,22),(100,22),(100,22),(100,22),(1700,282)]
        errors=[];miss=[]
        for k,((n,c),(lo,hi)) in enumerate(zip(settings,ps)):
            if k==6:
                errors.append(h.binomial_tail_upper(n,c,F(3,500),'lower'))
                miss.append(h.binomial_tail_upper(n,c+1,hi,'upper'))
            else:
                errors.append(h.binomial_tail_upper(n,c,F(3,20),'upper'))
                miss.append(h.binomial_tail_upper(n,c-1,lo,'lower'))
        self.assertLess(max(errors),F(1,20))
        self.assertLess(sum(miss),F(1,20))
        self.assertEqual(sum(n for n,c in settings),18300)


if __name__=='__main__': unittest.main()
