from fractions import Fraction as F
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import pulse_test as p


class PulseTest(unittest.TestCase):
    def setUp(self):
        self.contract=dict(gap_early=F(3,100),gap_late=F(13,1000),
                           eta_baseline=F(1,2000),eta_early=F(1,2000),eta_late=F(1,2000))
        self.design=p.plan(**self.contract)

    def test_exact_alternative_and_counted_energy(self):
        r=p.example()
        self.assertEqual(r["synthetic_decision"]["conclusion"],"reject_declared_classical_contract")
        self.assertEqual(r["active_pulse_time"],F(11,4)*r["design"]["records_per_arm"])

    def test_external_contract_and_tampering(self):
        self.assertTrue(p.verify_plan(self.design,**self.contract))
        for key,val in [("records_per_arm",1),("early_threshold",0),
                        ("log_alpha_upper",0),("margin",1),
                        ("orientation","either orientation")]:
            d=copy.deepcopy(self.design);d[key]=val
            self.assertFalse(p.verify_plan(d,**self.contract),key)
        changed=dict(self.contract,gap_late=F(1,100))
        self.assertFalse(p.verify_plan(self.design,**changed))

    def test_inconclusive_and_strict_threshold(self):
        n=self.design["records_per_arm"]
        for counts in [(n//2,)*3,(n//2,n//3,n//3),(n//2,2*n//3,2*n//3)]:
            self.assertEqual(p.decide(self.design,*counts)["conclusion"],"inconclusive")

    def test_calibration_floor_and_validation(self):
        self.assertEqual(p.plan(F(1,100),F(1,100),F(1,100))["status"],"calibration_limited_design")
        for x in [True,0.03]:
            with self.assertRaises(TypeError):p.plan(x,F(1,10))
        with self.assertRaises(ValueError):p.decide(self.design,-1,0,0)
        with self.assertRaises(ValueError):p.plan(F(1,10),F(1,10),alpha=0)
        with self.assertRaises(ValueError):p.plan(F(4,5),F(4,5))


if __name__=="__main__":unittest.main()
