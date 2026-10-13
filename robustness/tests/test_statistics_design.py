import copy
from fractions import Fraction as F
from itertools import product
from math import comb, log
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import statistics_design as s


class StatisticsTests(unittest.TestCase):
    def test_log_enclosures(self):
        import mpmath as mp
        mp.mp.dps=90
        for p in (F(1,100),F(1,2),F(1),F(7,5),F(2),F(10000),F(100001,100000)):
            lo,hi=s.log_bounds(p)
            x=mp.log(mp.mpf(p.numerator)/p.denominator)
            self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,x)
            self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,x)

    def test_binomial_exact_enclosure(self):
        for n in range(0,15):
            for p in (F(1,3),F(1,2),F(4,5)):
                for k in range(-1,n+3):
                    v=sum((F(comb(n,j))*p**j*(1-p)**(n-j) for j in range(max(k,0),n+1)),F(0))
                    lo,hi=s.binomial_upper_bounds(n,k,p)
                    self.assertLessEqual(lo,v)
                    self.assertGreaterEqual(hi,v)

    def test_encoding(self):
        self.assertEqual([s.encode(a,y) for a,y in product((0,1),repeat=2)],[0,1,1,0])
        pi,pj=F(3,5),F(2,5)
        mean=F(1,2)*pi+F(1,2)*(1-pj)
        self.assertEqual(mean,(1+pi-pj)/2)
        with self.assertRaises(ValueError):s.encode(True,1)

    def test_fixed_exact_main(self):
        p=s.exact_fixed_plan(F(18,125),F(1,1000))
        self.assertTrue(s.verify_exact_fixed(p,F(18,125),F(1,1000)))
        self.assertEqual((p['records'],p['successes_threshold']),(600,321))
        self.assertLess(p['false_rejection_interval'][1],F(1,20))
        self.assertLess(p['miss_interval'][1],F(1,20))
        bad=copy.deepcopy(p);bad['successes_threshold']-=1
        self.assertFalse(s.verify_exact_fixed(bad,F(18,125),F(1,1000)))
        self.assertFalse(s.verify_exact_fixed(p,F(1,10),F(1,1000)))

    def test_insufficient_power_is_reported(self):
        p=s.exact_fixed_plan(F(18,125),F(1,1000),records=100)
        self.assertEqual(p['status'],'power_not_certified')
        self.assertFalse(s.verify_exact_fixed(p,F(18,125),F(1,1000)))

    def test_chernoff_certificate_and_tampering(self):
        p=s.fixed_plan(F(18,125),F(1,1000))
        self.assertTrue(s.verify_fixed(p,F(18,125),F(1,1000)))
        self.assertEqual(p['records'],1181)
        bad=copy.deepcopy(p);bad['null_kl_lower']*=2
        self.assertFalse(s.verify_fixed(bad,F(18,125),F(1,1000)))
        self.assertEqual(s.fixed_decision(p,p['records']),'reject_declared_classical_contract')
        self.assertEqual(s.fixed_decision(p,0),'inconclusive')

    def test_anytime_component_supermartingale(self):
        for b in (F(0),F(1,10),F(2,3)):
            p0=(1+b)/2
            for p in (F(0),p0/2,p0):
                for f in (F(1,10),F(1,3),F(9,10)):
                    q=p0+f*(1-p0)
                    self.assertLessEqual(p*q/p0+(1-p)*(1-q)/(1-p0),1)

    def test_anytime_mixture_crossing(self):
        # Enumerate all 2^10 paths at the boundary null, including optional
        # stopping. This is a finite negative control, not the Ville proof.
        rejected=F(0)
        expectation=F(0)
        for path in product((0,1),repeat=10):
            t=s.AnytimeContrast(alpha=F(1,5),fractions=(F(1,3),F(2,3)))
            for bit in path:t.update(bit)
            expectation+=t.e_value/F(2**10)
            if t.crossed:rejected+=F(1,2**10)
        self.assertEqual(expectation,1)
        self.assertLessEqual(rejected,F(1,5))

    def test_predictable_radius_null_bound(self):
        # A radius selected from completed history is safe. Sum exact path
        # probabilities and capital for this adapted boundary-null policy.
        weighted=F(0)
        for path in product((0,1),repeat=6):
            t=s.AnytimeContrast(fractions=(F(1,4),))
            probability=F(1)
            previous=0
            for bit in path:
                radius=F(previous,3)
                p=(1+radius)/2
                probability*=p if bit else 1-p
                t.update(bit,radius)
                previous=bit
            weighted+=probability*t.e_value
        self.assertEqual(weighted,1)

    def test_current_clock_selection_can_invalidate_test(self):
        # Equal-probability clock strata: contrasts +1 and -1, so averaged
        # encoded success is 1/2. Betting only in the +1 current stratum has
        # expected multiplier (3/2+1)/2=5/4 >1 and is therefore invalid.
        self.assertEqual(F(1,2)*F(3,2)+F(1,2),F(5,4))
        self.assertGreater(F(5,4),1)

    def test_anytime_power_certificates(self):
        p=s.anytime_power_plan(F(18,125),F(1,1000))
        self.assertEqual(p['records'],1739)
        self.assertTrue(s.verify_anytime(p,F(18,125),F(1,1000)))
        bad=copy.deepcopy(p);bad['records']=1
        self.assertFalse(s.verify_anytime(bad,F(18,125),F(1,1000)))
        exact=s.exact_anytime_power_plan(F(18,125),F(1,1000))
        self.assertTrue(s.verify_exact_anytime(exact,F(18,125),F(1,1000)))
        self.assertEqual((exact['records'],exact['terminal_successes_threshold']),(1100,601))
        self.assertLess(exact['miss_interval'][1],F(1,20))
        bad=copy.deepcopy(exact);bad['component_weight']=F(1,2)
        self.assertFalse(s.verify_exact_anytime(bad,F(18,125),F(1,1000)))

    def test_adapted_bernoulli_domination(self):
        # The conditional probability may depend on the complete past;
        # fixed-horizon upper-tail domination still holds.
        n=7;p0=F(3,5)
        masses=[F(0)]*(n+1)
        for path in product((0,1),repeat=n):
            probability=F(1)
            successes=0
            for k,bit in enumerate(path):
                p=p0 if successes%2 else p0/2
                probability*=p if bit else 1-p
                successes+=bit
            masses[successes]+=probability
        self.assertEqual(sum(masses),1)
        for k in range(n+1):
            self.assertLessEqual(sum(masses[k:]),s.binomial_upper_bounds(n,k,p0)[1])

    def test_strict_inputs_and_calibration_floor(self):
        for f in (s.fixed_plan,s.exact_fixed_plan,s.anytime_power_plan,s.exact_anytime_power_plan):
            self.assertEqual(f(F(1,10),F(1,10))['status'],'calibration_limited_design')
        with self.assertRaises(TypeError):s.fixed_plan(.144)
        with self.assertRaises(TypeError):s.log_bounds(True)
        with self.assertRaises(ValueError):s.AnytimeContrast(weights=(F(1),))
        with self.assertRaises(ValueError):s.AnytimeContrast().update(True)
        with self.assertRaises(ValueError):s.exact_fixed_plan(F(1,10),records=True)

    def test_confidence_bound_inversion(self):
        for p in (F(1,5),F(1,2),F(4,5)):
            n=12;violation=F(0)
            for k in range(n+1):
                bound=s.fixed_lower_bound(n,k,alpha=F(1,10),bits=14)
                if bound['success_probability_lower']>p:
                    violation+=F(comb(n,k))*p**k*(1-p)**(n-k)
            self.assertLessEqual(violation,F(1,10))
        self.assertEqual(s.fixed_lower_bound(5,0)['success_probability_lower'],0)
        fixed=s.fixed_lower_bound(600,343)
        self.assertGreater(fixed['encoded_contrast_lower'],F(7,100))
        self.assertLess(fixed['encoded_contrast_lower'],F(8,100))

    def test_anytime_confidence_inversion(self):
        for n in range(1,15):
            for k in range(n+1):
                bound=s.anytime_lower_bound(n,k,alpha=F(1,10),fraction=F(1,3),bits=14)
                lo=bound['success_probability_lower']
                if lo>0:
                    capital=(1+F(1,3)*(1-lo)/lo)**k*F(2,3)**(n-k)
                    self.assertGreaterEqual(capital,10)
                self.assertEqual(bound['encoded_contrast_lower'],2*lo-1)

    def test_four_arm_encoding(self):
        probabilities=(F(4,5),F(1,5),F(3,10),F(7,10))
        encoded=sum((p if arm in(0,3) else 1-p for arm,p in enumerate(probabilities)),F(0))/4
        w=probabilities[0]-probabilities[1]-probabilities[2]+probabilities[3]
        self.assertEqual(encoded,(2+w)/4)
        self.assertEqual([s.encode_dimension(arm,1) for arm in range(4)],[1,0,0,1])
        with self.assertRaises(ValueError):s.encode_dimension(True,1)

    def test_four_arm_exact_designs(self):
        p=s.dimension_plan()
        self.assertTrue(s.verify_dimension(p))
        self.assertLess(p['false_rejection_interval'][1],F(1,20))
        self.assertLess(p['miss_interval'][1],F(1,20))
        q=s.dimension_plan(F(27,25),F(1,500),6000,4560)
        self.assertTrue(s.verify_dimension(q,F(27,25),F(1,500)))
        self.assertLess(q['false_rejection_interval'][1],F(1,20))
        self.assertLess(q['miss_interval'][1],F(1,20))
        bad=copy.deepcopy(q);bad['witness_radius']=F(0)
        self.assertFalse(s.verify_dimension(bad,F(27,25),F(1,500)))
        generic=s.bernoulli_exact_plan(F(1501,2000),F(1539,2000),6000,4560)
        self.assertTrue(s.verify_bernoulli_exact(generic,F(1501,2000),F(1539,2000)))


if __name__=='__main__':unittest.main()
