"""Check complete-record replay and population moments independently of inference."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import analog
import analog_benchmark as bench
import calibration


class AnalogBenchmarkTests(unittest.TestCase):
    def test_population_formula_matches_full_noisy_record_enumeration(self):
        n=4;sensor=np.array([4,0,-4,0]);scale=4;t=F(1,5)
        target=bench.population(n,3,1,sensor,scale,[t,2*t],'noisy_driven')
        Q=np.array([[-4,3,0,1],[1,-4,3,0],[0,1,-4,3],[3,0,1,-4]],dtype=float)
        P=expm(float(t)*Q);first=second=0.
        for x in range(n):
            for y in range(n):
                for a in [-1,1]:
                    for b in [-1,1]:
                        z0=(sensor[x]/scale+a)/2;z1=(sensor[y]/scale+b)/2
                        w=(1+z0*z1)/2;weight=P[x,y]/(n*4)
                        first+=weight*w;second+=weight*w*w
        self.assertAlmostEqual(first,target['agreement_means'][0],places=14)
        self.assertAlmostEqual(second,target['score_second_moments'][0],places=14)

    def test_saved_first_bank_sufficient_statistics_and_certificates_replay(self):
        path=ROOT/'results'/'analog_benchmark.json'
        if not path.exists():self.skipTest('production artifact not installed')
        result=json.loads(path.read_text());design=result['design']
        self.assertEqual(result['design_sha256'],hashlib.sha256((ROOT/'contracts'/'analog_benchmark.json').read_bytes()).hexdigest())
        for bank in result['first_raw_bank_per_case']:
            row=next(r for r in result['replicate_results'] if r['case']==bank['case'] and r['replicate']==0)
            stats=row['exact_statistics'];zc=bank['calibration_readings'];Nc=len(zc);den=bank['calibration_denominator']
            self.assertEqual(stats['calibration_sum'],sum(zc))
            self.assertEqual(stats['calibration_sum_squares'],sum(z*z for z in zc))
            cal=calibration.analog_calibration_summary(Nc,F(sum(zc),Nc*den),F(sum(z*z for z in zc),Nc*den*den),F(design['confidence_calibration']))
            means=[];coins=[]
            for raw,s in zip(bank['science'],stats['science']):
                products=[a*b for a,b in zip(raw['left'],raw['right'])]
                D=raw['left_denominator']*raw['right_denominator'];N=len(products)
                self.assertEqual(sum(products),s['product_sum']);self.assertEqual(sum(raw['encoded_bits']),s['encoded_successes'])
                self.assertEqual(D,s['product_denominator'])
                means.append(F(N*D+sum(products),2*N*D));coins.append(F(sum(raw['encoded_bits']),N))
            args=dict(lags=[1,2],coefficients=[1,-1],records=[N,N],durations=list(map(F,design['science_lags'])),calibration_vertices=cal['vertices'],alpha=F(design['confidence_science']),calibration_failure=F(design['confidence_calibration']))
            direct=analog.certify_means(means=means,**args);coin=analog.certify_means(means=coins,**args)
            self.assertEqual(row['direct_reject'],direct['contrast_lower']>0)
            self.assertEqual(row['coin_reject'],coin['contrast_lower']>0)
            self.assertEqual(row['direct_rate_lower'],float(direct['entropy_production_rate_lower']))
            self.assertEqual(row['coin_rate_lower'],float(coin['entropy_production_rate_lower']))


if __name__=='__main__':unittest.main()
