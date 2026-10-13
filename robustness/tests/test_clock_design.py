from fractions import Fraction as F
from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import clock_design as c


class ClockDesign(unittest.TestCase):
    def test_known_pair_and_independent_certificate(self):
        menu=[dict(label='a',horizon=1,cost=1,contrast_lower=0,contrast_upper=0),
              dict(label='b',horizon=3,cost=3,contrast_lower=F(-1,2),contrast_upper=F(-1,2))]
        cert=c.design(menu,2)
        self.assertEqual(cert['guaranteed']['value'],F(1,4))
        self.assertEqual(cert['guaranteed']['weights'],[F(1,2),F(1,2)])
        self.assertTrue(c.verify(menu,2,cert))
        altered=copy.deepcopy(cert);altered['guaranteed']['dual_intercept']=1
        self.assertFalse(c.verify(menu,2,altered))
        self.assertFalse(c.verify(menu,F(3,2),cert))

    def test_mixture_density_and_moment(self):
        d=c.density_steps([1,3],[F(1,2),F(1,2)])
        self.assertEqual(d['mass'],1);self.assertEqual(d['mean_time'],1)
        self.assertEqual([r['density'] for r in d['steps']],[F(2,3),F(1,6)])

    def test_input_and_zero_cost_cases(self):
        menu=[dict(label='a',horizon=1,cost=0,contrast_lower=F(-1,3),contrast_upper=F(-1,4)),
              dict(label='b',horizon=2,cost=1,contrast_lower=F(-1,5),contrast_upper=F(-1,5))]
        cert=c.design(menu,0)
        self.assertEqual(cert['guaranteed']['value'],F(1,4))
        with self.assertRaises(TypeError):c.design(menu,.2)
        with self.assertRaises(ValueError):c.density_steps([1],[F(1,2)])

    def test_quantum_menu(self):
        result=c.make_results()
        for row in result['designs']:
            cert=row['certificate'];self.assertTrue(c.verify(result['menu'],cert['budget'],cert))
            self.assertLessEqual(sum(w>0 for w in cert['guaranteed']['weights']),2)
            self.assertEqual(row['clock']['mass'],1)


if __name__=='__main__':unittest.main()
