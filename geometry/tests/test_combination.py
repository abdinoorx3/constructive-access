"""Shared-record composition: budgets, exact replay, coverage and resources."""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import combination as co
import finite_resolution as fr


class CombinationTests(unittest.TestCase):
    def arguments(self):
        p=fr.make_protocol(F(1,25),F(4,5),F(9,25),(F(19,6),F(19,6),10,10),global_gamma_upper=1)
        return dict(records=200000,geometry_total=0,geometry_square_total=0,
                    geometry_scale=256,endpoint_total=0,endpoint_square_total=0,
                    endpoint_scale=1<<24,protocol=p,geometry_score_error=F(20483,1179648),
                    endpoint_score_error=F(77457,10485760000),endpoint_parameter=F(24,25))

    def test_shared_acquisition_cost_and_exact_budget(self):
        x=co.certify_shared_records(**self.arguments(),external_failure=F(1,100))
        self.assertEqual(x['failure_allocation'],{'geometry':F(1,40),'endpoint':F(1,40)})
        self.assertEqual(x['confidence_failure_upper'],F(3,50))
        self.assertEqual(x['resources']['detector_readings'],600000)
        self.assertEqual(x['resources']['modeled_evolution_time'],16000)
        self.assertEqual(x['resources']['independent_stationary_preparations'],200000)
        self.assertEqual(x['selected_component'],'tie')

    def test_extra_geometry_coordinate_reads_count_once(self):
        a=self.arguments()
        a['protocol']=fr.make_protocol(F(1,25),F(4,5),F(9,25),
                                       (F(19,6),F(19,6),10,10),
                                       coordinate_readouts_per_record=2)
        x=co.certify_shared_records(**a)
        self.assertEqual(x['resources']['coordinate_detector_readings'],400000)
        self.assertEqual(x['resources']['detector_readings'],1000000)
        self.assertEqual(x['resources']['triple_detector_readings'],600000)

    def test_endpoint_exact_empirical_bernstein_and_bridge(self):
        n=200000
        # A constant positive endpoint defect gives zero sample variance.
        x=co.endpoint_certificate(n,n, n,100,F(24,25),F(1,25),alpha=F(1,40))
        self.assertEqual(x['empirical_variance'],0)
        expected=7*(1+(1+F(24,25))**2)*fr.log_interval(80)[1]/(3*(n-1))
        self.assertEqual(x['sampling_radius_upper'],expected)
        direct=co.from_gap(x['defect_lower']/2,lag_unit=F(1,25),range_bound=2,terms=24)
        self.assertEqual(x['entropy_production_rate_lower'],direct['entropy_production_rate_lower'])

    def test_zero_and_invalid_allocations_are_rejected(self):
        for kwargs in (dict(geometry_failure_share=0),dict(geometry_failure_share=1),
                       dict(alpha_total=1),dict(external_failure=F(99,100))):
            with self.assertRaises(ValueError):
                co.certify_shared_records(**self.arguments(),**kwargs)
        a=self.arguments()
        a['protocol']=fr.make_protocol(1,1,1,(1,1,1,1),readout_bounds=(-2,2))
        with self.assertRaises(ValueError):
            co.certify_shared_records(**a)

    def test_endpoint_invalid_summary_is_rejected(self):
        for args in ((1,0,0,1,0,1),(10,0,-1,1,0,1),
                     (10,100,1000,1,0,1),(10,0,0,0,0,1),(10,0,0,1,2,1)):
            with self.assertRaises(ValueError):
                co.endpoint_certificate(*args)

    def test_same_alpha_split_is_a_genuine_penalty(self):
        path=Path(__file__).resolve().parents[1]/'results'/'experiment.json'
        bank=json.loads(path.read_text())
        row=next(r for r in bank['datasets'] if r['scenario']=='driven')
        a=self.arguments()
        a.update(geometry_total=row['weighted_total'],geometry_square_total=row['weighted_square_total'],
                 endpoint_total=row['endpoint_total'],endpoint_square_total=row['endpoint_square_total'])
        x=co.certify_shared_records(**a)
        self.assertLess(float(x['geometry']['entropy_production_rate_lower']),row['weighted_rate_lower'])
        self.assertLess(float(x['endpoint']['entropy_production_rate_lower']),row['endpoint_rate_lower'])
        self.assertEqual(x['entropy_production_rate_lower'],max(
            x['geometry']['entropy_production_rate_lower'],x['endpoint']['entropy_production_rate_lower']))

    def test_replay_preserves_inputs_and_is_byte_reproducible(self):
        base=Path(__file__).resolve().parents[1]
        bank=base/'results'/'experiment.json'; contract=base/'contracts'/'experiment.json'
        before=(co.sha(bank),co.sha(contract))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'combination.json'
            report=co.replay(bank,contract,out)
            self.assertEqual(before,(co.sha(bank),co.sha(contract)))
            self.assertEqual(len(report['datasets']),1200)
            s=report['scenario_summary']
            self.assertEqual(s['driven']['methods']['combined']['positive_certificates'],400)
            self.assertEqual(s['weak_driven']['methods']['combined']['positive_certificates'],383)
            self.assertEqual(s['equilibrium']['methods']['combined']['positive_certificates'],0)
            saved=base/'results'/'combination.json'
            self.assertEqual(out.read_bytes(),saved.read_bytes())


if __name__=='__main__':
    unittest.main()
