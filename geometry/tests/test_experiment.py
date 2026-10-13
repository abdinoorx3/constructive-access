"""Checks of integer experiment arithmetic, model sampling and frozen design."""
from fractions import Fraction as F
from pathlib import Path
import sys
import json
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import experiment as e
import finite_resolution as fr

class ExperimentTests(unittest.TestCase):
    def test_exact_score_flooring(self):
        rng=np.random.default_rng(91)
        j=rng.integers(-e.D,e.D+1,(150,3),dtype=np.int64)
        j=np.concatenate((j,np.array([[e.D,0,-e.D],[0,0,0],[-e.D,e.D,-e.D]],dtype=np.int64)))
        w,b,retain=e.quantized_scores(j)
        p=e.protocol()
        for row,code,base,kept in zip(j,w,b,retain):
            values=tuple(F(int(x),e.D) for x in row)
            center=values[1]
            a=1-max(F(0),abs(center)-e.DELTA)**2
            score=fr.corrected_score(values,-center,a,p,retained=bool(kept))
            self.assertLessEqual(F(int(code),e.SCORE_D),score)
            self.assertLess(score-F(int(code),e.SCORE_D),F(1,e.SCORE_D))
            exactbase=-(values[0]-e.S*center)*(values[2]-e.S*center)
            self.assertLessEqual(F(int(base),e.BASELINE_D),exactbase)
            self.assertLess(exactbase-F(int(base),e.BASELINE_D),F(1,e.BASELINE_D))

    def test_large_square_aggregation(self):
        # Sum exceeds int64 although every square fits. Chunking must remain exact.
        v=np.full(200000,70000000,dtype=np.int64)
        total,squares=e.totals(v)
        self.assertEqual(total,200000*70000000)
        self.assertEqual(squares,200000*70000000**2)
        self.assertGreater(squares,np.iinfo(np.int64).max)

    def test_envelope_rounding_contract(self):
        for j in (-e.D,-e.D//2,-1,0,1,e.D//2,e.D):
            z=F(j,e.D)
            a=1-max(F(0),abs(z)-e.DELTA)**2
            for y in (max(F(-1),z-e.DELTA),z,min(F(1),z+e.DELTA)):
                self.assertLessEqual(1-y*y,a)

    def test_uniform_perturbation_allowance(self):
        rng=np.random.default_rng(212)
        j=rng.integers(-e.D,e.D+1,(100,3),dtype=np.int64)
        w,b,retain=e.quantized_scores(j)
        for row,wcode,bcode,kept in zip(j,w,b,retain):
            rounded=tuple(F(int(x),e.D) for x in row)
            actual=tuple(min(F(1),max(F(-1),x+e.DELTA*(1 if k%2 else -1))) for k,x in enumerate(rounded))
            c=rounded[1];a=1-max(F(0),abs(c)-e.DELTA)**2
            target=fr.corrected_score(actual,-c,a,e.protocol(),retained=bool(kept))
            self.assertLessEqual(abs(target-F(int(wcode),e.SCORE_D)),e.SCORE_ERROR)
            baseline=-(actual[0]-e.S*actual[1])*(actual[2]-e.S*actual[1])
            self.assertLessEqual(abs(baseline-F(int(bcode),e.BASELINE_D)),e.BASELINE_ERROR)

    def test_source_design_and_resources(self):
        d=e.frozen_design()
        self.assertEqual(d['independent_stationary_triples'],200000)
        self.assertEqual(d['resources_per_dataset']['detector_readings'],600000)
        self.assertEqual(d['resources_per_dataset']['modeled_evolution_time'],16000)
        self.assertEqual(d['common_generator_bounds'],(F(19,6),F(19,6),F(10),F(10)))
        self.assertNotEqual(d['master_seed'],82653)
        for omega in (0,1,3):
            self.assertGreaterEqual(F(19,6)**2,1+omega*omega)
            self.assertGreaterEqual(10,1+omega*omega)

    def test_seed_replay_and_independent_cases(self):
        a=e.simulate(0,0,n=100)
        self.assertTrue(np.array_equal(a,e.simulate(0,0,n=100)))
        self.assertFalse(np.array_equal(a,e.simulate(0,1,n=100)))
        self.assertFalse(np.array_equal(a,e.simulate(1,0,n=100)))
        self.assertEqual(a.dtype,np.int64)
        self.assertTrue(np.all(np.abs(a)<=e.D))

    def test_rounded_exact_score_confidence_runs(self):
        j=e.simulate(2,22,n=1000)
        w,b,_=e.quantized_scores(j)
        wt,ws=e.totals(w);bt,bs=e.totals(b)
        cert=e.infer(1000,wt,ws,bt,bs)
        self.assertEqual(cert[0]['entropy_production_rate_lower'],0)
        self.assertEqual(cert[2]['entropy_production_rate_lower'],0)
        self.assertEqual(cert[0]['independent_preparations'],1000)
        self.assertEqual(cert[0]['declared_evolution_time'],80)

    def test_frozen_bank_bindings_and_all_summary_counts(self):
        contract=e.PHASE/'contracts/experiment.json'
        path=e.PHASE/'results/experiment.json'
        data=json.loads(path.read_text())
        e.check_contract(contract)
        self.assertEqual(data['contract_sha256'],e.sha(contract))
        self.assertEqual(data['raw_sha256'],e.sha(e.PHASE/'results/experiment_raw.npz'))
        self.assertEqual(len(data['datasets']),1200)
        for name,omega in e.SCENARIOS:
            rows=[r for r in data['datasets'] if r['scenario']==name]
            self.assertEqual(len(rows),400)
            self.assertEqual([r['replicate'] for r in rows],list(range(400)))
            for method in ('weighted','endpoint','hoeffding'):
                values=[r[method+'_rate_lower'] for r in rows]
                summary=data['scenario_summary'][name]['methods'][method]
                self.assertEqual(summary['positive_certificates'],sum(v>0 for v in values))
                self.assertEqual(summary['over_truth'],sum(v>omega*omega for v in values))
                self.assertAlmostEqual(summary['mean_rate_lower'],sum(values)/400,places=14)

    def test_all_raw_exemplars_seed_and_exact_scores(self):
        data=json.loads((e.PHASE/'results/experiment.json').read_text())
        with np.load(e.PHASE/'results/experiment_raw.npz') as raw:
            self.assertEqual(len(raw.files),9)
            for idx,(name,_) in enumerate(e.SCENARIOS):
                j=raw[name+'_readings']
                self.assertTrue(np.array_equal(j,e.simulate(idx,0)))
                w,b,r=e.quantized_scores(j)
                self.assertTrue(np.array_equal(w,raw[name+'_weighted_codes']))
                self.assertTrue(np.array_equal(b,raw[name+'_endpoint_codes']))
                row=next(x for x in data['datasets'] if x['scenario']==name and x['replicate']==0)
                self.assertEqual(e.totals(w),(row['weighted_total'],row['weighted_square_total']))
                self.assertEqual(e.totals(b),(row['endpoint_total'],row['endpoint_square_total']))
                self.assertEqual(int(r.sum()),row['retained'])

    def test_prespecified_bank_certificate_replay(self):
        data=json.loads((e.PHASE/'results/experiment.json').read_text())
        for row in data['datasets']:
            if row['replicate'] not in (0,199,399):continue
            certificates=e.infer(row['records'],row['weighted_total'],row['weighted_square_total'],
                                 row['endpoint_total'],row['endpoint_square_total'])
            for method,cert in zip(('weighted','endpoint','hoeffding'),certificates):
                self.assertEqual(float(cert['entropy_production_rate_lower']),row[method+'_rate_lower'])
                self.assertEqual(float(cert['sampling_radius_upper']),row[method+'_sampling_radius_upper'])

    def test_input_guards(self):
        for bad in (np.zeros((3,3)),np.zeros((3,2),dtype=np.int64),np.array([[e.D+1,0,0]],dtype=np.int64)):
            with self.assertRaises(ValueError):e.quantized_scores(bad)

if __name__=='__main__':unittest.main()
