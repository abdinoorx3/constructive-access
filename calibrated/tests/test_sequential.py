from fractions import Fraction as F
from itertools import product
from pathlib import Path
import math
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import sequential as seq


class SequentialTests(unittest.TestCase):
    def test_predictable_decisions_ignore_future(self):
        first=[[1,1,0],[1,0,0],[0,1,0]]
        a=seq.certify(first+[[0,0,0]]*4)
        b=seq.certify(first+[[1,1,1]]*4)
        self.assertEqual(a['decisions'][:4],b['decisions'][:4])
        self.assertEqual(a['decisions'][0][1],0)

    def test_exact_support_on_continuous_interval(self):
        for r in range(49):
            s=F(r,48);a=[-s*s,2*s,F(-1)];b=-(1-s)**2/2
            for x in [F(0),s,F(1,7),F(1)]:
                for baseline in [F(1,2),F(3,4),F(1)]:
                    q=[baseline+(1-baseline)*x**k for k in (1,2,3)]
                    self.assertLessEqual(sum(c*y for c,y in zip(a,q)),b)

    def test_conditional_e_factor_on_reversible_models(self):
        pasts=[([0,0,0],0),([11,15,2],20),([100,180,20],200)]
        for counts,n in pasts:
            d=seq.policy(counts,n);a=d['coefficients'];lam=float(d['bet'])
            for x in [.05,.4,.8,1]:
                for amp in [.1,.5,1]:
                    p=[(1+amp*x**k)/2 for k in (1,2,3)]
                    factor=0.
                    for y in product((0,1),repeat=3):
                        prob=math.prod(z if bit else 1-z for bit,z in zip(y,p))
                        exponent=lam*float(sum(c*bit for c,bit in zip(a,y))-d['support'])-lam*lam*float(sum(c*c for c in a))/8
                        factor+=prob*math.exp(exponent)
                    self.assertLessEqual(factor,1+2e-14)

    def test_exact_export_binding(self):
        rows=[[1,1,0]]*100;c=seq.certify(rows)
        self.assertGreater(c['entropy_production_rate_lower'],0)
        self.assertTrue(seq.verify(c,rows))
        bad=dict(c);bad['entropy_production_rate_lower']+=1
        self.assertFalse(seq.verify(bad,rows))
        self.assertFalse(seq.verify(c,rows[:-1]))
        self.assertFalse(seq.verify(c,rows,durations=(1,2,3)))

    def test_bias_can_remove_signal(self):
        rows=[[1,1,0]]*100
        self.assertEqual(seq.certify(rows,conditional_radii=(1,1,1))['entropy_production_rate_lower'],0)

    def test_full_duration_penalty(self):
        rows=[[1,1,0]]*100
        a=seq.certify(rows,durations=(1,4,9))['entropy_production_rate_lower']
        b=seq.certify(rows,durations=(4,16,36))['entropy_production_rate_lower']
        self.assertEqual(a,4*b)

    def test_empty_and_zero_bets(self):
        self.assertEqual(seq.certify([])['entropy_production_rate_lower'],0)
        self.assertEqual(seq.certify([[1,1,0]])['entropy_production_rate_lower'],0)

    def test_invalid_protocols(self):
        for kwargs in [dict(mode='postselected'),dict(grid=7),dict(bet_grid=0),dict(alpha=0),dict(durations=(1,2)),dict(conditional_radii=(0,0,2))]:
            with self.assertRaises(ValueError):seq.certify([],**kwargs)
        for rows in [[[0,1]],[[0,1,2]],[[True,0,1]],[[.0,1,0]]]:
            with self.assertRaises(ValueError):seq.certify(rows)

    def test_vectorized_scores_equal_exact_replay(self):
        import numpy as np
        import sequential_benchmark as bench
        bits=np.random.default_rng(191).integers(0,2,(8,60,3),dtype=np.uint8)
        for mode in ['learned','fixed']:
            result=bench.batch_policies(bits,mode)
            for i,row in enumerate(bits):
                exact=seq.certify(row.astype(int).tolist(),mode=mode)
                self.assertEqual(exact['predictable_log_score'],F(int(result['final_score'][i]),result['score_denominator']))

    def test_exact_first_crossing(self):
        import numpy as np
        import sequential_benchmark as bench
        bits=np.tile(np.array([1,1,0],dtype=np.uint8),(2,100,1))
        result=bench.batch_policies(bits,'learned')
        k=int(result['first_crossing'][0]);self.assertLessEqual(k,100)
        self.assertEqual(seq.certify(bits[0,:k-1].astype(int).tolist())['entropy_production_rate_lower'],0)
        self.assertGreater(seq.certify(bits[0,:k].astype(int).tolist())['entropy_production_rate_lower'],0)

    def test_fixed_horizon_matches_existing_api(self):
        import numpy as np
        import sequential_benchmark as bench
        from constructive_access.virtual import contrast
        bits=np.random.default_rng(881).integers(0,2,(8,40,3),dtype=np.uint8)
        actual=bench.fixed_horizon(bits)
        for i,counts in enumerate(bits.sum(axis=1)):
            cert=contrast.certify_counts([1,2,3],[F(-4,9),F(4,3),F(-1)],[40]*3,[int(s) for s in counts],[F(3,20),F(3,10),F(9,20)])
            self.assertEqual(bool(actual[i]),cert['contrast_lower']>0)


if __name__=='__main__':unittest.main()
