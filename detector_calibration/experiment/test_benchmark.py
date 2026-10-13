"""Frozen-design algebra, aggregation and replay checks without a large bank."""
from fractions import Fraction as F
import copy
import itertools
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark as b


class Benchmark(unittest.TestCase):
    def test_seed_streams_are_distinct(self):
        c=b.design()
        seeds=[c['seed_base']+100000*ci+10*r+s for ci in range(4)
               for r in range(c['repetitions']) for s in range(3)]
        self.assertEqual(len(seeds),len(set(seeds)))

    def test_reference_rounding_distance_exact(self):
        for p in (F(19,20),F(59,100),F(9,10),F(83,150),F(0),F(1)):
            law=b.binary_law(p);den=1<<law['dyadic_bits']
            self.assertEqual(sum(law['dyadic_numerators']),den)
            self.assertEqual(law['total_variation_upper'],abs(F(law['dyadic_numerators'][1],den)-p))
            self.assertLess(law['total_variation_upper'],F(1,den))

    def test_exact_resources_and_bound_contract(self):
        c=b.design();self.assertEqual(c['science_records']*(c['center_replicas']+2),79999998)
        for case in c['cases']:
            gamma,omega,tau=map(F,(case['gamma'],case['omega'],case['exposure']))
            self.assertLessEqual(gamma*gamma+omega*omega,13)
            self.assertLessEqual(gamma*gamma+omega*omega,16)
            self.assertTrue(F(c['width_prior'][0])<=tau<=F(c['width_prior'][1]))
            self.assertLess(F(c['step_time']),tau)

    def test_endpoint_formula_and_calibration_abstention(self):
        h=F(7,200);gain=(F(9,10),F(9,10));tau=F(1,100000)
        conf={'h':h,'lambda':F(1),'statistics':{'S0':{'interval':(F(4),F(5))}}}
        cal={'status':'covered_region','gain':gain,'width':(tau,tau)}
        blur=b.cert.bias_bounds(0,h,1,tau,(4,4,13,13))['delta_N']
        expected=2*h**3*(F(4)/gain[0]**2-blur)**2/(2*(1-h)+b.cert.sqrt_upper(2))**2
        self.assertEqual(b.endpoint_lower(conf,cal,(4,4,13,13)),expected)
        cal['status']='gain_not_separated_from_zero'
        self.assertEqual(b.endpoint_lower(conf,cal,(4,4,13,13)),0)

    def test_count_evaluation_is_exact_and_preserves_all_prefixes(self):
        c=b.design();n=c['science_records'];categories=b.cert.category_order(c['order'])
        counts=[n//len(categories)]*len(categories);counts[0]+=n-sum(counts)
        result=b.evaluate_counts(c,c['cases'][0],counts,95000,59000)
        self.assertIsInstance(result['entropy_lower'],F)
        self.assertEqual(result['entropy_lower'],0)
        self.assertEqual([row['order'] for row in result['prefixes']],[0,1])
        self.assertTrue(result['calibration_covered'])
        self.assertFalse(result['sigma_overshoot'])

    def test_aggregate_summary_retains_coverage_and_costs(self):
        c=b.design();rows=[]
        for ci,case in enumerate(c['cases']):
            for rep in range(2):
                result=dict(entropy_lower=F(rep),endpoint_lower=F(rep,10),gamma_upper=F(2),
                    termwise_max=F(rep,2),projected_max=F(rep),order_selected=rep,
                    prefixes=[dict(order=j,entropy_lower=F(rep),termwise_lower=F(rep,2),projected_lower=F(rep)) for j in range(2)],
                    calibration_covered=True,gamma_covered=True,sigma_overshoot=False,endpoint_overshoot=False)
                rows.append(dict(case_index=ci,certificate=result,simulation_product_law_tv_upper=F(1,10**12)))
        report=b.aggregate(c,rows,[],'fixturehash')
        self.assertEqual(report['datasets'],8)
        self.assertEqual(report['whole_bank_product_tv_upper'],F(8,10**12))
        for case in report['cases']:
            self.assertEqual(case['positive'],1)
            self.assertEqual(case['selected_prefix_counts'],{'0':1,'1':1})
            self.assertEqual(case['metrics']['entropy_lower']['mean'],.5)
            self.assertEqual(case['resources_per_dataset']['total_detector_readouts'],80199998)


if __name__=='__main__':unittest.main()
