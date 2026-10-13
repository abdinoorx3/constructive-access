import importlib.util
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dimension_lab',ROOT/'code'/'dimension_lab.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
C=json.loads((ROOT/'contracts'/'dimension_lab.json').read_text())
P={s['name']:s['parameters'] for s in C['scenarios']}
OLD={'epsilon':C['old_diagonal_probability_boundary'],'beta':C['old_offdiagonal_probability_boundary'],
     'tau':C['old_antichain_threshold'],'critical_counts':C['old_critical_counts']}


class DimensionLaboratoryTests(unittest.TestCase):
    def test_independent_bloch_lindblad_engines(self):
        for p in P.values():
            np.testing.assert_allclose(d.probability_engine(p),d.lindblad_probabilities(p),atol=3e-14,rtol=0)
        p=P['combined_design']|{'clock_scale':.99}
        np.testing.assert_allclose(d.probability_engine(p),d.lindblad_probabilities(p),atol=3e-14,rtol=0)

    def test_unitary_limit_from_density_dynamics(self):
        p=P['nominal']|{'depolarization':0.}
        expected=np.sin(np.pi*np.arange(1,14)/7)**2
        np.testing.assert_allclose(d.probability_engine(p),expected,atol=3e-14,rtol=0)

    def test_spam_affine_law(self):
        raw=d.probability_engine(P['nominal'])
        noisy=d.probability_engine(P['nominal']|{'false_positive':.02,'false_negative':.03})
        np.testing.assert_allclose(noisy,.02+.95*raw,atol=1e-15,rtol=0)
        flipped=d.probability_engine(P['nominal']|{'preparation_error':1.})
        np.testing.assert_allclose(flipped,1-raw,atol=1e-15,rtol=0)

    def test_noise_history_changes_channel_closure(self):
        p=P['nominal']|{'amplitude_jitter':.1,'timing_jitter':.05}
        self.assertGreater(np.max(abs(d.probability_engine(p)-d.quasi_static_probabilities(p))),.15)
        zero=P['nominal']
        np.testing.assert_allclose(d.probability_engine(zero),d.quasi_static_probabilities(zero),atol=2e-15,rtol=0)

    def test_old_exact_type_i_certificate(self):
        cert=d.certify_plan(OLD,C['records'])
        self.assertTrue(cert['valid']);self.assertLess(cert['alpha_bound'],.05)
        self.assertEqual(cert['minimum_states'],5)

    def test_incomplete_or_noninteger_settings_rejected(self):
        with self.assertRaises(ValueError):d.certify_plan(OLD|{'critical_counts':OLD['critical_counts'][:-1]},C['records'])
        with self.assertRaises(ValueError):d.certify_plan(OLD,C['records'][:-1])
        with self.assertRaises(ValueError):d.certify_plan(OLD,[1.5]*13)
        with self.assertRaises(ValueError):d.classify([1]*12,[1]*12)
        with self.assertRaises(ValueError):d.classify([1.2]*13,[1]*13)
        with self.assertRaises(ValueError):d.pass_probability([.5]*12,C['records'],OLD['critical_counts'])
        with self.assertRaises(ValueError):d.pass_probability([1.1]*13,C['records'],OLD['critical_counts'])

    def test_dimension_parameter_passed_to_classifier(self):
        self.assertTrue(d.classify(np.array([3,3,0,3,3]),[2,2,1,2,2],3))
        self.assertFalse(d.classify(np.array([3,3,2,3,3]),[2,2,1,2,2],3))

    def test_simple_classical_null(self):
        Pmat=np.eye(1);p=d.classical_probabilities(Pmat,[1],[.3])
        np.testing.assert_allclose(p,.3)
        power,_=d.pass_probability(p,C['records'],C['old_critical_counts'])
        self.assertLess(power,1e-30)

    def test_raw_shot_stream_reproduces_and_retains_sources(self):
        with tempfile.TemporaryDirectory() as td:
            a=d.retained_shots(P['combined_design'],[30]*13,1047,Path(td)/'a.npz')
            b=d.retained_shots(P['combined_design'],[30]*13,1047,Path(td)/'b.npz')
            self.assertEqual(a['sha256'],b['sha256'])
            with np.load(Path(td)/'a.npz') as data:
                self.assertEqual(len(data['observed']),390)
                self.assertTrue(np.all(np.isin(data['observed'],[0,1])))
                counts=[int(data['observed'][data['setting']==k].sum()) for k in range(1,14)]
                self.assertEqual(counts,a['counts'])
                self.assertIn('SIMULATED',str(data['source']))

    def test_common_clock_fit_repairs_known_phase_error(self):
        ps=d.probability_engine(P['coherent_amplitude_offset_0.01'])
        counts=np.rint(1000000*ps).astype(int)
        fit=d.fit_common_clock(counts,1000000,P['nominal'],[-.05,.05])
        self.assertAlmostEqual(fit['amplitude_offset_estimate'],.01,places=5)
        retimed=P['coherent_amplitude_offset_0.01']|{'clock_scale':fit['common_clock_scale']}
        old=d.pass_probability(ps,C['records'],C['old_critical_counts'])[0]
        new=d.pass_probability(d.probability_engine(retimed),C['records'],C['old_critical_counts'])[0]
        self.assertLess(old,.03);self.assertGreater(new,.98)

    def test_confidence_interval_endpoints(self):
        self.assertEqual(d.cp_interval(0,100)[0],0.)
        self.assertEqual(d.cp_interval(100,100)[1],1.)
        lo,hi=d.cp_interval(50,100);self.assertLess(lo,.5);self.assertGreater(hi,.5)

    def test_frozen_artifacts_consistent(self):
        r=json.loads((ROOT/'results'/'dimension_lab.json').read_text())
        self.assertEqual(r['contract_sha256'],hashlib.sha256((ROOT/'contracts'/'dimension_lab.json').read_bytes()).hexdigest())
        self.assertTrue(d.certify_plan(r['pilot_selected_plan'],C['records'])['valid'])
        self.assertEqual(r['pilot']['total_records'],260000)
        self.assertTrue(r['production_budget']['matched_between_plans'])
        for s in r['alternative_scenarios']:
            for name in ['old','pilot_selected']:
                h=s['holdout'][name];target=s['holdout']['numerical_binomial_power_crosscheck'][name]
                variance=max(target*(1-target)/h['repetitions'],1/h['repetitions']**2)
                self.assertLess(abs(h['frequency']-target),8*np.sqrt(variance))
        raw=r['retained_shot_stream']
        self.assertEqual(raw['sha256'],hashlib.sha256((ROOT/'results'/raw['file']).read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
