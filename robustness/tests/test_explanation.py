from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import explanation


class Explanation(unittest.TestCase):
    def test_count_to_sharp_explanation_boundary(self):
        result=explanation.example()['examples']
        self.assertEqual(result[0]['reset_explanation']['status'],'incompatible_hazard_contract')
        self.assertTrue(F(1,2)<result[1]['reset_explanation']['minimum_reset_defect']<1)
        self.assertLess(result[2]['reset_explanation']['minimum_reset_defect'],result[1]['reset_explanation']['minimum_reset_defect'])
        self.assertTrue(F(7,10)<result[1]['required_peak_crossing_intensity']['hazard_lower_bound']<1)

    def test_recalibration_changes_conclusion(self):
        tight=explanation.from_counts(600,343,1,1,F(9,4),F(9,10),F(1,1000))
        broad=explanation.from_counts(600,343,1,1,F(9,4),F(9,10),F(1,5))
        self.assertLess(broad['reset_explanation']['minimum_reset_defect'],tight['reset_explanation']['minimum_reset_defect'])
        with self.assertRaises(TypeError):explanation.from_counts(600,343,1,1,2.25)
        with self.assertRaises(ValueError):explanation.from_counts(600,343,1,1,2,gain=0)


if __name__=='__main__':unittest.main()
