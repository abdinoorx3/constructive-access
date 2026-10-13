import sys
from pathlib import Path
from fractions import Fraction as F
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import design
import joint_analog

class DesignTests(unittest.TestCase):
    def test_singleton_agrees(self):
        n=[4000,3000,3000];m=[F(3,4),F(69,100),F(59,100)]
        a=design.certify_library(n,m,[F(3,4)])
        b=joint_analog.certify_means(n,m,parameter=F(3,4))
        self.assertEqual(a['simultaneous_gap_lower'],b['reversible_gap_lower'])
        self.assertEqual(a['entropy_production_rate_lower'],4*b['reversible_gap_lower'])
    def test_duplicate_candidates_do_not_inflate_penalty(self):
        a=design.certify_library([100]*3,[F(3,4),F(3,4),F(1,2)],[F(1,2),F(1,2)],external_failure=F(1,100))
        self.assertEqual(a['candidate_count'],1)
        self.assertEqual(a['confidence_failure_upper'],F(3,50))
        self.assertEqual(a['detector_readouts'],500)
    def test_simultaneous_allocation_and_winner(self):
        a=design.certify_library([10000]*3,[F(3,4),F(4,5),F(1,2)],[F(1,4),F(1,2),F(3,4)])
        self.assertEqual(a['candidate_statistical_failure'],F(1,60))
        self.assertEqual(a['simultaneous_gap_lower'],max(c['reversible_gap_lower'] for c in a['candidates']))
        self.assertEqual(a['declared_evolution_time'],30000)
    def test_reversible_population_cannot_give_positive_gap(self):
        for decay in (F(0),F(1,4),F(1,2),F(1)):
            means=[F(3,4),(1+decay/2)/2,(1+decay**2/2)/2]
            a=design.certify_library([10**8]*3,means,[F(k,16) for k in range(17)])
            self.assertEqual(a['entropy_production_rate_lower'],0)
    def test_same_data_library_rejected(self):
        with self.assertRaises(ValueError):design.certify_library([10]*3,[F(3,4)]*3,[F(1,2)],library_selection='same_data')
    def test_empty_library_rejected(self):
        with self.assertRaises(ValueError):design.certify_library([10]*3,[F(3,4)]*3,[])
    def test_penalty_never_improves_same_fixed_candidate(self):
        means=[F(3,4),F(3,4),F(11,20)]
        a=design.certify_library([3000]*3,means,[F(k,8) for k in range(9)])
        for c in a['candidates']:
            b=joint_analog.certify_means([3000]*3,means,parameter=c['support']['parameter'])
            self.assertLessEqual(c['reversible_gap_lower'],b['reversible_gap_lower'])

if __name__=='__main__':unittest.main()
