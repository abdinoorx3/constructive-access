from fractions import Fraction as F
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm
from constructive_access.frontier.energy_certificate import certify

spec=importlib.util.spec_from_file_location('thermodynamic_lab',Path(__file__).resolve().parents[1]/'code'/'thermodynamic_lab.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ThermodynamicLabTests(unittest.TestCase):
    def test_poisson_path_means_and_endpoint_law(self):
        nstates=8;forward=3.;backward=.4;lag=.7;n=120000
        f=np.r_[np.ones(4),-np.ones(4)]
        record=m.ring_records(np.random.default_rng(783),n,lag,forward,backward,f)
        self.assertAlmostEqual(record['forward_jumps'].mean(),forward*lag,delta=.018)
        self.assertAlmostEqual(record['backward_jumps'].mean(),backward*lag,delta=.006)
        self.assertTrue(np.array_equal(record['final'],(record['initial']+record['forward_jumps']-record['backward_jumps'])%nstates))
        Q=np.zeros((nstates,nstates))
        for i in range(nstates):Q[i,(i+1)%nstates]=forward;Q[i,(i-1)%nstates]=backward;Q[i,i]=-forward-backward
        expected=(1+f@expm(lag*Q)@f/nstates)/2
        self.assertAlmostEqual(record['same'].mean(),expected,delta=.005)

    def test_detector_independent_noise_gain_squared(self):
        detector=np.r_[np.ones(4),-np.ones(4)]
        record=m.ring_records(np.random.default_rng(893),120000,.001,0.,0.,detector,gain=.2)
        self.assertAlmostEqual(record['same'].mean(),(1+.2**2)/2,delta=.005)
        self.assertAlmostEqual(record['first'].mean(),.5,delta=.005)

    def test_clock_jitter_exposure_and_equilibrium_control(self):
        record=m.ring_records(np.random.default_rng(903),100000,2.,1.,1.,[1]*4+[-1]*4,jitter_halfwidth=.3)
        self.assertGreaterEqual(record['elapsed'].min(),1.4)
        self.assertLessEqual(record['elapsed'].max(),2.6)
        self.assertAlmostEqual(record['elapsed'].mean(),2.,delta=.005)

    def test_prepared_gillespie_law(self):
        record=m.two_state_records(np.random.default_rng(113),80000,.3)
        expected=.1+.9*np.exp(-10*.3)
        self.assertAlmostEqual(record['same'].mean(),expected,delta=.005)
        self.assertTrue(np.array_equal(record['final'],(record['initial']+record['forward_jumps']-record['backward_jumps'])%2))

    def test_memory_control_is_explicitly_outside_channel(self):
        record=m.ring_records(np.random.default_rng(523),100000,.3,1.,1.,[1]*4+[-1]*4,memory_agreement=.73)
        self.assertAlmostEqual(record['same'].mean(),.73,delta=.005)

    def test_compiled_analysis_matches_exact_full_certificate(self):
        contract=m.specification()
        for name,counts in [('revival',[2560,3680]),('curvature',[35849,32265,27572])]:
            menu=contract['menus'][name];K=len(counts);n=menu['records_per_setting'];r=F(menu['box_radius'])
            compiled=m.compile_analysis(menu,contract['alpha'])
            analysis=m.analyze_counts(counts,compiled)
            boxes=[(max(F(0),F(s,n)-r),min(F(1),F(s,n)+r)) for s in counts]
            cert=certify(menu['lags'],F(menu['step']),[F(1,K)]*K,boxes,(-1,1),menu['scores'])
            self.assertEqual(analysis['full_cone_rate_lower'],float(cert['entropy_production_rate_lower']))
            self.assertEqual(analysis['full_cone'],cert['entropy_production_rate_lower']>0)

    def test_frozen_contract_and_deterministic_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=m.freeze(root)
            _,read_digest=m.load_contract(root);self.assertEqual(digest,read_digest)
            self.assertEqual(digest,m.freeze(root))
            path=root/'a.npz';arrays={'a':np.arange(8,dtype=np.int16),'b':np.linspace(0,1,8)}
            m.deterministic_npz(path,arrays);first=hashlib.sha256(path.read_bytes()).hexdigest()
            m.deterministic_npz(path,arrays);self.assertEqual(first,hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path) as loaded:self.assertTrue(np.array_equal(loaded['a'],arrays['a']))
            (root/'contracts'/'thermodynamic_lab.json').write_text('{}')
            with self.assertRaises(RuntimeError):m.freeze(root)

    def test_monte_carlo_intervals_and_invalid_inputs(self):
        lo,hi=m.clopper_pearson(0,200);self.assertEqual(lo,0);self.assertGreater(hi,0)
        lo,hi=m.clopper_pearson(200,200);self.assertEqual(hi,1);self.assertLess(lo,1)
        with self.assertRaises(ValueError):m.clopper_pearson(201,200)
        with self.assertRaises(ValueError):m.ring_records(np.random.default_rng(1),10,1,1,1,[1,-1],gain=2)

    def test_independent_oracle_matches_known_ring_values(self):
        spec=m.specification();scenario=spec['scenarios'][0]
        actual=m.oracle_probabilities(spec['menus']['revival'],scenario)
        np.testing.assert_allclose(actual,[.5119894663900273,.7360448459591435],atol=1e-12)
        equilibrium=spec['scenarios'][1]
        for menu in spec['menus'].values():
            p=m.oracle_probabilities(menu,equilibrium)
            self.assertTrue(np.all(p>=.5-1e-12));self.assertTrue(np.all(np.diff(p)<=1e-12))


if __name__=='__main__':unittest.main()
