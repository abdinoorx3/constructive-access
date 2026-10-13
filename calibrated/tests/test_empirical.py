"""Checks for empirical diagnostics; none supplies experimental calibration."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
import empirical_analysis as ea


class EmpiricalChecks(unittest.TestCase):
    def test_ordered_covariance_matches_explicit_pairs(self):
        x=np.array([1.,2.,5.,7.,9.]); y=np.array([4.,-1.,8.,3.,2.])
        for k in [0,1,3]:
            expected=sum((x[i]-x.mean())*(y[i+k]-y.mean()) for i in range(len(x)-k))/(len(x)-k)
            self.assertAlmostEqual(ea.covariance(x,y,k),expected)
        self.assertNotEqual(ea.covariance(x,y,1),ea.covariance(y,x,1))

    def test_reversible_spectral_mixtures_satisfy_all_population_margins(self):
        # Includes a pure exponential, which detects a wrong midpoint formula
        # on the geometric grid k,2k,4k.
        for atoms,weights in [([.97],[.4]),([.1,.7,.99],[.1,.2,.25]),([0.,1.],[.2,.3])]:
            g={k:sum(w*x**k for x,w in zip(atoms,weights)) for k in ea.LAGS}
            for row in ea.moment_margins(g,sum(weights)):
                self.assertGreaterEqual(row['margin'],-1e-14)
        pure={k:.4*.97**k for k in ea.LAGS}
        for row in ea.moment_margins(pure,.4):
            if row['inequality']=='weighted_centered_log_convexity':
                self.assertAlmostEqual(row['margin'],0.,places=14)

    def test_nonmonotone_and_oversized_drop_are_exposed(self):
        rows=ea.moment_margins({1:.3,2:.01,4:.02},.4)
        self.assertLess(next(r['margin'] for r in rows if r['inequality']=='variance_upper_only_drop' and r['lags']==[1,2]),0)
        self.assertLess(next(r['margin'] for r in rows if r['inequality']=='monotone_decay' and r['lags']==[2,4]),0)

    def test_transform_has_no_evaluation_dependence(self):
        train=[np.array([-2.,-1.,1.,2.]),np.array([-3.,0.,1.,4.])]
        centers,scale=ea.transform_parameters(train)
        a=ea.bounded_transform(np.array([-1e6,0,1e6]),centers[0],scale)
        self.assertTrue(np.all(np.abs(a)<=1))
        self.assertEqual((centers,scale),ea.transform_parameters(train))
        with self.assertRaises(ValueError):ea.transform_parameters([np.ones(4)])

    def test_real_outputs_preserve_abstention_and_split(self):
        # Public raw records are intentionally optional in an installed package.
        data=BASE/'data/source_audit'
        if not (data/'harmwell_n1.h5').exists():self.skipTest('public raw records not installed')
        d=ea.analyze(data)
        self.assertEqual(d['training_samples'],10000)
        self.assertEqual(d['evaluation_samples'],39999)
        self.assertEqual(d['lag_grid_samples'],list(ea.LAGS))
        self.assertEqual(sum(d['eight_block_lengths']),39999)
        self.assertEqual(d['thermodynamic_certificate']['status'],'abstain')
        self.assertIsNone(d['thermodynamic_certificate']['alpha'])
        self.assertIsNone(d['thermodynamic_certificate']['entropy_rate_lower_bound'])
        self.assertIn('not confidence',d['block_interpretation'])
        self.assertTrue(all(r['pair_count']==39999-r['lag_samples'] for r in d['detector_comparison']['covariance_rows']))
        stored=json.loads((BASE/'results/empirical_analysis.json').read_text())
        self.assertEqual(d,stored)


if __name__=='__main__':unittest.main()
