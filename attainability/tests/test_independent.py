"""Independent internal review: analytic counterexamples and artifact replay.

Floating checks challenge the proofs; they do not replace them or establish
experimental validity. This file does not call producer diagnostic functions
to establish the central inequalities or reproduce benchmark statistics.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
try:
    from constructive_access.attainability import spectral_dynamics as sd
    from constructive_access.attainability import joint_analog as ja
    from constructive_access.attainability import physical_minimum as pm
    from constructive_access.attainability import design
    from constructive_access.attainability import retiming
except ImportError:
    import spectral_dynamics as sd
    import joint_analog as ja
    import physical_minimum as pm
    import design
    import retiming


def phi(u):
    return 0.0 if u == 0 else 2*math.sqrt(u)*math.atanh(math.sqrt(u))


def stationary(q):
    a = q.T.copy()
    a[-1] = 1
    b = np.zeros(len(q)); b[-1] = 1
    return np.linalg.solve(a, b)


def entropy(q, pi):
    total = 0.
    for i in range(len(pi)):
        for j in range(i):
            a, b = pi[i]*q[i, j], pi[j]*q[j, i]
            if a != b:
                total += (a-b)*math.log(a/b)
    return total


def row_data(row):
    st = row['exact_statistics']
    n = [st['calibration_records']] + [r['records'] for r in st['science']]
    B = st['calibration_sum_squares']/(n[0]*st['calibration_denominator']**2)
    m = st['calibration_sum']/(n[0]*st['calibration_denominator'])
    means = [.5*(1+B)] + [.5*(1+r['product_sum']/(r['records']*r['product_denominator'])) for r in st['science']]
    coins = [r['encoded_successes']/r['records'] for r in st['science']]
    return n, B, m, means, coins


def scalar_gap(n, means, s, alpha=.05):
    radius = math.sqrt(math.log(1/alpha)/2*(s**4/(4*n[0])+4*s*s/n[1]+1/n[2]))
    signed = -s*s*means[0]+2*s*means[1]-means[2]+(1-s)**2/2-radius
    return max(0., signed), radius


class IndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = json.loads((ROOT/'results/joint_analog.json').read_text())

    def test_stationary_semigroup_chain_random_scalar_and_vector(self):
        rng = np.random.default_rng(871991)
        positive = 0
        for n in (3, 4, 5, 8):
            for trial in range(15):
                q = np.exp(rng.normal(0, 1.2, (n, n)))
                np.fill_diagonal(q, 0)
                np.fill_diagonal(q, -q.sum(axis=1))
                pi = stationary(q); t = .03 + .2*rng.random()
                p = expm(t*q); adj = p.T*pi[None, :]/pi[:, None]
                H, B = (p+adj)/2, (p-adj)/2
                tv = np.abs(p-adj).sum(axis=1)/2
                endpoint = float(np.sum(pi[:, None]*rel_entr(p, adj)))
                halfj = .5*np.sum(pi[:, None]*(rel_entr(p, adj)+rel_entr(adj, p)))
                self.assertAlmostEqual(endpoint, halfj, places=11)
                self.assertGreaterEqual(endpoint+1e-11, phi(float(pi@(tv*tv))))
                self.assertLessEqual(endpoint, t*entropy(q, pi)+1e-10)
                sym = np.sqrt(pi)[:, None]*H/np.sqrt(pi)[None, :]
                vals, vecs = np.linalg.eigh(sym)
                s = float(vals[-2])
                f = vecs[:, -2]/np.sqrt(pi)
                f /= np.max(np.abs(f))
                for obs in (f[:, None], rng.uniform(-1, 1, (n, 3))):
                    c = [float(np.sum(pi[:, None]*obs*(np.linalg.matrix_power(p, k)@obs))) for k in range(3)]
                    delta = -s*s*c[0]+2*s*c[1]-c[2]
                    skew = float(np.sum(pi[:, None]*(B@obs)**2))
                    residual = float(np.sum(pi[:, None]*((H-s*np.eye(n))@obs)**2))
                    diameter = np.linalg.norm(obs[:, None, :]-obs[None, :, :], axis=2).max()
                    self.assertAlmostEqual(delta, skew-residual, places=11)
                    self.assertLessEqual(skew, diameter**2*(pi@(tv*tv))/4+1e-11)
                    u = max(0., 4*delta/diameter**2)
                    self.assertLessEqual(phi(u), endpoint+1e-10)
                    positive += delta > 1e-12
        self.assertGreater(positive, 50)

    def test_scalar_divergence_equality_and_nonembeddability(self):
        for z in (.01, .3, .9):
            p = np.zeros((4, 4))
            for i in range(4):
                p[i, (i+1)%4] = (1+z)/2
                p[i, (i-1)%4] = (1-z)/2
            f = np.array([1., 1., -1., -1.])
            delta = -f@p@p@f/4
            d = np.sum(rel_entr(p, p.T))/4
            self.assertAlmostEqual(delta, z*z, places=14)
            self.assertAlmostEqual(d, phi(delta), places=13)
            self.assertAlmostEqual(np.linalg.det(p), -z*z, places=13)
            self.assertLess(np.linalg.det(p), 0)  # exp(tQ) has positive determinant.

    def test_phi_enclosure_against_high_precision_transcendental(self):
        import mpmath as mp
        with mp.workdps(100):
            for u in (F(0), F(1, 10000), F(1, 5), F(4, 5), F(999, 1000)):
                x = mp.mpf(u.numerator)/u.denominator
                actual = 2*mp.sqrt(x)*mp.atanh(mp.sqrt(x))
                old = F(0)
                for terms in (1, 3, 24, 60):
                    lo, hi = sd.phi_interval(u, terms)
                    ml = mp.mpf(lo.numerator)/lo.denominator
                    mh = mp.mpf(hi.numerator)/hi.denominator
                    self.assertLessEqual(ml, actual+mp.mpf('1e-98'))
                    self.assertGreaterEqual(mh+mp.mpf('1e-98'), actual)
                    self.assertGreaterEqual(lo, old); old = lo

    def test_bridge_units_domains_and_same_event(self):
        a = sd.from_gap(F(1, 20), lag_unit=F(1, 10), range_bound=2)
        self.assertEqual(a['normalized_defect'], F(1, 10))
        self.assertEqual(a['linear_rate_lower'], 2)
        self.assertLessEqual(float(a['entropy_production_rate_lower']), phi(.1)*10+1e-14)
        self.assertEqual(sd.from_gap(-1)['entropy_production_rate_lower'], 0)
        self.assertIn('incompatible', sd.from_gap(F(1, 2))['status'])
        for kwargs in ({'lag_unit': 0}, {'range_bound': 0}, {'terms': 0}):
            with self.assertRaises(ValueError): sd.from_gap(F(1, 20), **kwargs)
        for u in (-1, 1, 2):
            with self.assertRaises(ValueError): sd.phi_interval(u)

    def test_positive_part_is_essential_in_confidence_statement(self):
        # Constant f=1 has Delta=-(1-s)^2. A clipped gap 0 is not <=Delta/2.
        delta = sd.population_gap([1, 1, 1], F(1, 2))['defect']
        self.assertEqual(delta, F(-1, 4))
        self.assertGreater(0, delta/2)
        self.assertEqual(sd.from_gap(0)['entropy_production_rate_lower'], 0)

    def test_odd_parity_equilibrium_even_observable_counterexample(self):
        # Reflection T on a biased ring satisfies Q^dagger=TQT: physical
        # parity reversal gives zero EPR, although ordinary reversal does not.
        n = 12; q = np.zeros((n, n))
        for i in range(n):
            q[i, (i+1)%n] = 4; q[i, (i-1)%n] = 1; q[i, i] = -5
        parity = (-np.arange(n)) % n
        np.testing.assert_allclose(q.T, q[np.ix_(parity, parity)])
        f = np.cos(2*np.pi*np.arange(n)/n)
        np.testing.assert_allclose(f[parity], f, atol=1e-15)
        t = np.pi/(2*3*np.sin(2*np.pi/n)); p = expm(t*q)
        delta = -f@p@p@f/n
        self.assertGreater(delta, .1)
        self.assertGreater(phi(delta)/t, 0)

    def test_nonstationary_reversible_counterexample(self):
        q = np.array([[-1., 1.], [1., -1.]])
        mu = np.array([.9, .1]); f = np.array([1., 2.]); t = math.log(2)/2
        c = [np.dot(mu*f, expm(k*t*q)@f) for k in range(3)]
        np.testing.assert_allclose(c, [13/10, 59/40, 25/16], atol=1e-14)
        delta = -c[0]+2*c[1]-c[2]
        self.assertAlmostEqual(delta, 7/80)
        self.assertGreater(phi(4*delta)/t, .8*math.log(9))
        self.assertEqual(entropy(q, np.array([.5, .5])), 0)

    def test_stationary_reversible_detector_memory_counterexample(self):
        # Reversible telegraph C_X(v)=2^{-|v|}, lag unit ln2. The shifted
        # palindromic filter (1,-2,1) is reversible but not memoryless.
        weights = (1, -2, 1); offsets = (0, -2, -4)
        def c(k):
            return sum(F(a*b)*F(2)**(-abs(k+x-y)) for a, x in zip(weights, offsets) for b, y in zip(weights, offsets))
        self.assertEqual([c(k) for k in range(3)], [F(33, 8), F(21, 32), F(-159, 64)])
        delta = -c(2); width = 8
        self.assertGreater(phi(float(4*delta/width**2))/math.log(2), .47)

    def test_analog_noise_penalty_is_one_sided_not_latent_calibration(self):
        s = F(3, 4); A, B, c1, c2 = F(1, 4), F(1, 2), F(1, 5), F(1, 20)
        latent = (-s*s*A+2*s*c1-c2)/2
        observed = ja.population_gap(B, c1, c2, s)
        self.assertEqual(latent-observed, s*s*(B-A)/2)
        self.assertLess(observed, latent)

    def test_noisy_readout_population_by_direct_endpoint_enumeration(self):
        d = self.bank['design']; n = d['cycle_states']; den = d['sensor_grid_denominator']
        sensor = np.rint(den*np.cos(2*np.pi*np.arange(n)/n))/den
        summaries = {r['case']: r for r in self.bank['summaries']}
        for case in d['cases']:
            a, b = float(F(case['forward_rate'])), float(F(case['backward_rate']))
            q = np.zeros((n, n))
            for i in range(n): q[i, (i+1)%n], q[i, (i-1)%n], q[i, i] = a, b, -a-b
            noisy = case['detector'].startswith('half_')
            f = sensor/2 if noisy else sensor
            second = (sensor*sensor+1)/4 if noisy else sensor*sensor
            saved = summaries[case['name']]['population']
            self.assertAlmostEqual(float(np.mean(f*f)), saved['A'], places=14)
            self.assertAlmostEqual(float(np.mean(second)), saved['observed_second_moment'], places=14)
            for k in (1, 2):
                p = expm(k*float(F(d['lag_unit']))*q)
                c = f@p@f/n
                w2 = (1+2*c+second@p@second/n)/4
                self.assertAlmostEqual((1+c)/2, saved['agreement_means'][k-1], places=13)
                self.assertAlmostEqual(w2, saved['score_second_moments'][k-1], places=13)

    def test_noisy_population_obstruction_by_independent_high_precision_fourier(self):
        import mpmath as mp
        audit = json.loads((ROOT/'results/joint_analog_population_audit.json').read_text())
        def rational(x):
            v = F(x)
            return mp.mpf(v.numerator)/v.denominator
        with mp.workdps(100):
            f = [mp.mpf(z)/8192 for z in audit['detector_integers']]
            n = len(f); correlations = []
            for lag, bounds in zip((mp.mpf(1)/10, mp.mpf(1)/5), audit['correlation_intervals']):
                total = mp.mpc(0)
                for k in range(n):
                    angle = 2*mp.pi*k/n
                    fourier = sum(z*mp.exp(-1j*angle*j) for j, z in enumerate(f))
                    eigenvalue = 20*(mp.exp(1j*angle)-1)+(mp.exp(-1j*angle)-1)
                    total += abs(fourier)**2*mp.exp(lag*eigenvalue)/(n*n)
                self.assertLess(abs(total.imag), mp.mpf('1e-95'))
                self.assertLessEqual(rational(bounds[0]), total.real)
                self.assertGreaterEqual(rational(bounds[1]), total.real)
                correlations.append(total.real)
            exact_max = (correlations[0]**2/rational(audit['B'])-correlations[1])/2
            self.assertLessEqual(exact_max, rational(audit['max_gap_all_real_s_upper']))
            self.assertLess(exact_max, -mp.mpf('0.007'))
            self.assertTrue(audit['no_population_witness_in_entire_quadratic_family'])

    def test_all_joint_and_split_statistics_replayed_without_producer_api(self):
        t = float(F(self.bank['design']['lag_unit']))
        for row in self.bank['replicate_results']:
            n, B, m, means, coins = row_data(row)
            for name, s, science in [('joint_analog', .5, means[1:]), ('joint_analog_tuned', .75, means[1:]), ('joint_coin_tuned', .75, coins)]:
                gap, radius = scalar_gap(n, [means[0], *science], s)
                old = 8*gap*gap/(t*(2*s+math.sqrt(2))**2)
                self.assertAlmostEqual(old, row['rate_lower'][name], places=12)
                self.assertAlmostEqual(4*gap/t, row['direct_linear_rate_lower'][name], places=12)
                self.assertAlmostEqual(phi(2*gap)/t, row['direct_nonlinear_rate_lower'][name], places=12)
            bu = min(1., B+math.sqrt(math.log(80)/(2*n[0])))
            rm = math.sqrt(2*math.log(160)/n[0])
            umin = max(0., abs(m)-rm)**2
            for name, s, science in [('split_analog', .5, means[1:]), ('split_analog_tuned', .75, means[1:]), ('split_coin', .5, coins)]:
                sup = (2*s-1)/2+(s*s*bu-(1-s)**2*umin)/2
                radius = math.sqrt(math.log(40)/2*(4*s*s/n[1]+1/n[2]))
                gap = max(0., 2*s*science[0]-science[1]-sup-radius)
                old = 8*gap*gap/(t*(2*s+math.sqrt(2))**2)
                self.assertAlmostEqual(old, row['rate_lower'][name], places=12)
            repaired = max(0., scalar_gap(n, means, .75)[0]-.75**2/2)
            self.assertAlmostEqual(8*repaired**2/(t*(1.5+math.sqrt(2))**2), row['conservative_calibration_mismatch_repair_rate'], places=12)

    def test_benchmark_artifact_hashes_raw_statistics_and_resources(self):
        bank = self.bank; contract = ROOT/'contracts/joint_analog.json'
        self.assertEqual(hashlib.sha256(contract.read_bytes()).hexdigest(), bank['contract_sha256'])
        raw = ROOT/'results'/bank['raw_archive']
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), bank['raw_sha256'])
        arrays = np.load(raw, allow_pickle=False)
        for row in bank['replicate_results']:
            n, B, m, means, coins = row_data(row)
            self.assertEqual(sum(n), 10000)
            self.assertEqual(n[0]+2*n[1]+2*n[2], 16000)
            self.assertEqual(F(bank['design']['lag_unit'])*(n[1]+2*n[2]), 900)
            if row['replicate']: continue
            name = row['case']; st = row['exact_statistics']; z = arrays[name+'__calibration']
            self.assertEqual(int(z.sum()), st['calibration_sum'])
            self.assertEqual(int(z@z), st['calibration_sum_squares'])
            for k, s in enumerate(st['science'], 1):
                left, right = arrays[name+f'__lag{k}_left'], arrays[name+f'__lag{k}_right']
                self.assertEqual(int(left@right), s['product_sum'])
                self.assertEqual(int(arrays[name+f'__lag{k}_coins'].sum()), s['encoded_successes'])

    def test_benchmark_summaries_and_invalid_control_are_retained(self):
        rows = self.bank['replicate_results']
        self.assertEqual(len(rows), 1600)
        for summary in self.bank['summaries']:
            cases = [r for r in rows if r['case'] == summary['case']]
            self.assertEqual(len(cases), 400)
            for method, saved in summary['methods'].items():
                rates = np.array([r['rate_lower'][method] for r in cases])
                self.assertEqual(int((rates > 0).sum()), saved['positive_certificates'])
                self.assertAlmostEqual(float(rates.mean()), saved['mean_rate_lower'], places=14)
        bad = [r for r in rows if r['case'] == 'invalid_detector_gain_equilibrium']
        self.assertEqual(sum(r['rate_lower']['joint_analog_tuned'] > 0 for r in bad), 400)
        self.assertEqual(sum(r['conservative_calibration_mismatch_repair_rate'] > 0 for r in bad), 0)

    def test_library_union_budget_deduplication_and_independent_gap(self):
        records = [4000, 3000, 3000]; means = [F(3, 4), F(69, 100), F(59, 100)]
        candidates = [F(1, 2), F(3, 4), F(1, 2)]
        cert = design.certify_library(records, means, candidates, external_failure=F(1, 100), lag_unit=F(1, 10))
        self.assertEqual(cert['candidate_count'], 2)
        self.assertEqual(cert['candidate_statistical_failure'], F(1, 40))
        self.assertEqual(cert['confidence_failure_upper'], F(3, 50))
        gaps = [scalar_gap(records, list(map(float, means)), s, .025)[0] for s in (.5, .75)]
        self.assertAlmostEqual(float(cert['simultaneous_gap_lower']), max(gaps), places=14)
        self.assertEqual(cert['detector_readouts'], 16000)
        self.assertEqual(cert['declared_evolution_time'], 900)

    def test_library_refuses_same_data_enlargement_and_unsupported_contracts(self):
        base = dict(records=[10]*3, means=[F(3, 4), F(1, 2), F(1, 2)], parameters=[F(1, 2)])
        for extra in ({'library_selection': 'fit_same_data'}, {'parameter': F(3, 4)}, {'selection': 'predeclared'}, {'external_failure': 1}, {'record_contract': 'marginal_stationarity_only'}):
            with self.assertRaises(ValueError): design.certify_library(**base, **extra)
        with self.assertRaises(ValueError): design.certify_library([10]*3, [F(3, 4), F(1, 2), F(1, 2)], [])

    def test_all_finite_library_selections_replayed_independently(self):
        saved = json.loads((ROOT/'results/design.json').read_text())
        contract = ROOT/'contracts/design.json'
        self.assertEqual(hashlib.sha256(contract.read_bytes()).hexdigest(), saved['contract_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'results/joint_analog.json').read_bytes()).hexdigest(), saved['data_sha256'])
        parameters = sorted(set(map(F, saved['contract']['parameters'])))
        alpha = float(F(saved['contract']['alpha']))/len(parameters)
        source = {(r['case'], r['replicate']): r for r in self.bank['replicate_results']}
        t = float(F(self.bank['design']['lag_unit']))
        self.assertEqual(len(saved['replicate_results']), 1600)
        for row in saved['replicate_results']:
            n, B, m, means, coins = row_data(source[row['case'], row['replicate']])
            gaps = [scalar_gap(n, means, float(s), alpha)[0] for s in parameters]
            winner = max(range(len(gaps)), key=gaps.__getitem__)
            self.assertEqual(F(row['selected_parameter']), parameters[winner])
            self.assertAlmostEqual(row['library_rate'], 4*gaps[winner]/t, places=12)
            for s, rate in row['fixed_rates'].items():
                self.assertAlmostEqual(rate, 4*scalar_gap(n, means, float(F(s)))[0]/t, places=12)
        driven = next(s for s in saved['summaries'] if s['case'] == 'driven')
        self.assertEqual(driven['positive_certificates'], 362)
        self.assertEqual(driven['fixed']['3/4']['positive_certificates'], 399)

    def test_sixth_edition_reanalysis_keeps_counts_and_confidence_event(self):
        # Cumulative release uses the first location. The second permits this
        # phase's source tree to review its assembled release without any
        # required external workspace sibling or absolute path.
        candidates = (ROOT.parent/'calibrated/results', ROOT/'release/calibrated/results',
                      ROOT.parent/'calibrated_inference/results')
        prior = next((p for p in candidates if (p/'joint_calibration_counts.npz').is_file()), None)
        self.assertIsNotNone(prior, 'A declared sixth-edition result bank is required')
        replay = json.loads((ROOT/'results/spectral_dynamics_replay.json').read_text())
        for name, digest in replay['source_sha256'].items():
            self.assertEqual(hashlib.sha256((prior/name).read_bytes()).hexdigest(), digest)
        records = np.load(prior/'joint_calibration_counts.npz', allow_pickle=False)
        s = .76; rad = math.sqrt(math.log(20)/2*(s**4/4000+4*s*s/8000+1/8000))
        checked = 0
        for case in replay['cases']:
            name = case['case']; science = records[name+'__science_counts']; twins = records[name+'__calibration_counts']
            old = records[name+'__joint_calibration_pinsker_rates']
            for j, (ct, tw) in enumerate(zip(science, twins)):
                p0, p1, p2 = (tw[0]+tw[3])/4000, ct[0]/8000, ct[1]/8000
                gap = max(0., -s*s*p0+2*s*p1-p2+(1-s)**2/2-rad)
                self.assertAlmostEqual(8*gap*gap/(2*s+math.sqrt(2))**2, old[j], places=12)
                self.assertAlmostEqual(4*gap, case['linear_rates'][j], places=12)
                self.assertAlmostEqual(phi(2*gap), case['nonlinear_rates'][j], places=12)
                checked += 1
        self.assertEqual(checked, 1600)

    def test_finite_ring_minimum_against_arbitrary_nonlocal_generators(self):
        rng = np.random.default_rng(309471)
        for n in range(3, 18):
            for _ in range(12):
                rates = np.exp(rng.normal(0, 1.5, n-1))
                q = np.zeros((n, n))
                for i in range(n):
                    for j, rate in enumerate(rates, 1): q[i, (i+j)%n] = rate
                    q[i, i] = -rates.sum()
                v = np.exp(2j*np.pi*np.arange(n)/n)
                eigenvalue = (q@v)[0]
                gamma, omega = -eigenvalue.real, eigenvalue.imag
                pi = np.ones(n)/n
                self.assertLessEqual(pm.ring_minimum(n, gamma, omega)['entropy'], entropy(q, pi)+1e-9)
                np.testing.assert_allclose(q@v, eigenvalue*v, atol=1e-11)

    def test_ring_minimizer_matrix_entropy_and_continuum(self):
        gamma, omega = .7, 1.2; saved = []
        for n in (8, 12, 20, 80):
            x = pm.ring_minimum(n, gamma, omega)
            q = np.zeros((n, n))
            for i in range(n):
                q[i, (i+1)%n], q[i, (i-1)%n], q[i, i] = x['forward'], x['backward'], -x['forward']-x['backward']
            f = np.cos(2*np.pi*np.arange(n)/n)
            self.assertAlmostEqual(entropy(q, np.ones(n)/n), x['entropy'], places=11)
            for t in (.1, 1.):
                self.assertAlmostEqual(f@expm(t*q)@f/n, .5*math.exp(-gamma*t)*math.cos(omega*t), places=12)
            saved.append(x['entropy'])
        self.assertTrue(all(a>b for a, b in zip(saved, saved[1:])))
        self.assertGreater(saved[-1], omega*omega/gamma)

    def test_gated_hub_defeats_unrestricted_circle_minimum(self):
        # Rebuild the generator from the stated rates, then verify the API and
        # every covariance at a grid of times without using its diagnostics.
        A, gamma, omega, kappa = .1, 1., .5, .01
        a, b = (gamma-kappa+omega)/2, (gamma-kappa-omega)/2
        q = np.zeros((5, 5)); pi = np.array([A/4]*4+[1-A])
        for i in range(4):
            q[i, (i+1)%4] = a; q[i, (i-1)%4] = b
            q[i, 4] = kappa; q[i, i] = -a-b-kappa
        q[4, :4] = A*kappa/(4*(1-A)); q[4, 4] = -A*kappa/(1-A)
        f = np.array([1., 1., -1., -1., 0.])
        np.testing.assert_allclose(pi@q, 0, atol=1e-17)
        for t in np.linspace(0, 8, 33):
            self.assertAlmostEqual((pi*f)@expm(t*q)@f, A*math.exp(-gamma*t)*math.cos(omega*t), places=13)
        physical = entropy(q, pi)
        self.assertAlmostEqual(physical, A*(a-b)*math.log(a/b), places=14)
        self.assertLess(physical, omega*omega/gamma)
        producer = pm.gated_ring_countermodel()
        np.testing.assert_allclose(producer['generator'], q, atol=1e-15)
        self.assertAlmostEqual(producer['entropy'], physical, places=14)

    def test_wrapped_kernel_alias_and_total_variation_bound(self):
        # Integrate the actual heat kernel, independently of the Gaussian-TV
        # envelope used by the producer. Fourier series is exponentially fast.
        theta = np.linspace(-np.pi, np.pi, 32768, endpoint=False)
        gamma, t, shift = .8, .7, .2
        def density(angle):
            out = np.ones_like(theta)
            for k in range(1, 30): out += 2*np.exp(-gamma*t*k*k)*np.cos(k*(theta-angle))
            return out/(2*np.pi)
        tv = .5*np.mean(np.abs(density(0)-density(shift)))*2*np.pi
        upper = shift/(2*np.sqrt(np.pi*gamma*t))
        self.assertLessEqual(tv, upper+1e-12)
        np.testing.assert_allclose(density(0), density(2*np.pi*13), atol=2e-14)
        self.assertAlmostEqual(pm.phase_record_tv_bound([t], gamma, shift/t), upper, places=13)

    def test_pell_aliases_full_kernel_closeness_and_entropy_growth(self):
        rows = pm.pell_aliases(count=8)
        for row in rows:
            p, q = row['p'], row['q']
            self.assertEqual(p*p-2*q*q, 1)
            stable = 2*np.pi/(p+math.sqrt(2)*q)
            self.assertAlmostEqual(stable, row['second_gap_phase_error'], places=14)
            self.assertAlmostEqual(stable/(2*np.sqrt(np.pi*np.sqrt(2))), row['two_interval_state_record_tv_upper'], places=14)
            self.assertAlmostEqual((.5+2*np.pi*q)**2, row['entropy'], places=5)
        self.assertGreater(rows[-1]['entropy'], 1e12)
        self.assertLess(rows[-1]['two_interval_state_record_tv_upper'], 2e-6)

    def test_retiming_half_lag_bridge_and_costs_without_diagonal_acquisitions(self):
        n, mean, tau = 1800, F(2, 5), F(1, 2)
        result = retiming.certify_mean(n, mean, lag=tau)
        radius = math.sqrt(math.log(20)/(2*n))
        gap = .5-float(mean)-radius
        self.assertAlmostEqual(float(result['negative_correlation_gap_lower']), gap, places=14)
        self.assertAlmostEqual(float(result['entropy_production_rate_lower']), 2*phi(2*gap)/float(tau), places=12)
        self.assertEqual(result['direct_bridge']['lag_unit'], tau/2)
        self.assertEqual(result['declared_evolution_time'], 900)
        self.assertEqual(result['detector_readouts'], 3600)
        self.assertEqual(result['stationary_pair_preparations'], n)
        self.assertEqual(result['calibration_acquisitions'], 0)
        # A and C(tau/2) never enter a zero-coefficient contrast.
        for c0 in (F(1, 4), F(1, 2), F(1)):
            for c1 in (F(-1, 8), F(1, 8)):
                d = sd.population_gap([c0, c1, F(-1, 5)], 0)['defect']
                self.assertEqual(d, F(1, 5))
        self.assertEqual(retiming.certify_product_sum(n, -n, 5, lag=tau), result)

    def test_retiming_exact_null_rejection_probability_and_error_penalties(self):
        from scipy.stats import binom
        # A state-independent fresh fair +/-1 detector gives exactly Bernoulli
        # pair scores, with zero system EPR. Compute its null tail, no Monte Carlo.
        for n in (20, 100, 1800):
            result = retiming.certify_mean(n, F(1, 2))
            radius = result['sampling_radius_upper']
            largest_positive_count = math.ceil(n*(F(1, 2)-radius))-1
            self.assertLessEqual(float(binom.cdf(largest_positive_count, n, .5)), .05)
        clean = retiming.certify_mean(1800, F(2, 5))
        dirty = retiming.certify_mean(1800, F(2, 5), mean_error=F(1, 1000),
            conditional_radius=F(1, 500), external_failure=F(1, 100),
            record_contract='bounded_conditional_mean_error')
        self.assertEqual(clean['signed_gap_lower']-dirty['signed_gap_lower'], F(3, 1000))
        self.assertEqual(dirty['confidence_failure_upper'], F(3, 50))

    def test_retiming_refuses_unsupported_data_selection_and_sensor_contracts(self):
        for kw in ({'selection': 'optimized_on_same_records'}, {'lag': 0}, {'alpha': 1},
                   {'detector_contract': 'correlated_memory'}, {'record_contract': 'marginal_stationarity'},
                   {'mean_error': -1}, {'conditional_radius': -1}):
            with self.assertRaises(ValueError): retiming.certify_mean(1800, F(2, 5), **kw)
        for n, total, den in ((1800, 0, 0), (1800, .5, 1), (0, 0, 1), (1800, 1801, 1)):
            with self.assertRaises(ValueError): retiming.certify_product_sum(n, total, den)

    def test_retiming_population_interval_by_independent_fourier_diagonalization(self):
        import mpmath as mp
        saved = json.loads((ROOT/'results/retiming.json').read_text())
        with mp.workdps(100):
            def q(x):
                z = F(x)
                return mp.mpf(z.numerator)/z.denominator
            for case, summary in zip(saved['design']['cases'], saved['summaries']):
                f = [mp.mpf(z)/(8192 if case['noisy'] else 4096) for z in retiming.SENSOR]
                a, b, tau = q(case['forward_rate']), q(case['backward_rate']), q(saved['design']['new_lag'])
                total = mp.mpc(0)
                for k in range(20):
                    angle = 2*mp.pi*k/20
                    coeff = sum(z*mp.exp(-1j*angle*j) for j, z in enumerate(f))
                    eigenvalue = a*(mp.exp(1j*angle)-1)+b*(mp.exp(-1j*angle)-1)
                    total += abs(coeff)**2*mp.exp(tau*eigenvalue)/400
                lo, hi = map(q, summary['new_lag_correlation_interval'])
                self.assertLessEqual(lo, total.real)
                self.assertGreaterEqual(hi, total.real)
                self.assertLess(abs(total.imag), mp.mpf('1e-95'))
                if case['name'] == 'noisy_driven': self.assertLess(total.real, -mp.mpf('.07'))
                if 'equilibrium' in case['name']: self.assertGreater(total.real, 0)
            audit = saved['population_audit']
            self.assertLess(q(audit['old_two_lag_family_gap_upper']), -mp.mpf('.007'))
            self.assertLess(q(audit['new_correlation_interval'][1]), -mp.mpf('.07'))

    def test_all_fresh_retiming_and_baseline_rates_replayed_independently(self):
        saved = json.loads((ROOT/'results/retiming.json').read_text()); d = saved['design']
        self.assertNotEqual(d['seed'], self.bank['design']['seed'])
        self.assertEqual(len(saved['replicate_results']), 1600)
        tau = float(F(d['new_lag'])); dt = float(F(d['baseline_lag_unit']))
        alpha = float(F(d['alpha']))
        for row in saved['replicate_results']:
            st = row['exact_statistics']; cal = st['baseline_calibration']
            n = [cal['records']]+[p['records'] for p in st['baseline_pairs']]
            B = cal['sum_squares']/(n[0]*cal['denominator']**2)
            means = [(1+B)/2]+[(1+p['product_sum']/(p['records']*p['product_denominator']))/2 for p in st['baseline_pairs']]
            for s in d['baseline_parameters']:
                gap = scalar_gap(n, means, float(F(s)), alpha)[0]
                self.assertAlmostEqual(row['rate_lower']['baseline_s='+s], phi(2*gap)/dt, places=12)
            p = st['retimed_pairs']; N = p['records']
            mean = (1+p['product_sum']/(N*p['product_denominator']))/2
            gap = max(0., .5-mean-math.sqrt(math.log(1/alpha)/(2*N)))
            self.assertAlmostEqual(row['rate_lower']['retimed'], 2*phi(2*gap)/tau, places=12)
        for summary in saved['summaries']:
            rows = [r for r in saved['replicate_results'] if r['case'] == summary['case']]
            for method, report in summary['methods'].items():
                rates = [r['rate_lower'][method] for r in rows]
                self.assertEqual(sum(x > 0 for x in rates), report['positive_certificates'])
                self.assertAlmostEqual(sum(rates)/len(rates), report['mean_rate_lower'], places=13)
                self.assertEqual(sum(x > summary['actual_entropy_rate']+1e-12 for x in rates), report['empirical_upper_coverage_failures'])

    def test_retiming_frozen_raw_artifacts_and_both_resource_vectors(self):
        saved = json.loads((ROOT/'results/retiming.json').read_text()); d = saved['design']
        self.assertEqual(hashlib.sha256((ROOT/'contracts/retiming.json').read_bytes()).hexdigest(), saved['contract_sha256'])
        raw_path = ROOT/'results'/saved['raw_archive']
        self.assertEqual(hashlib.sha256(raw_path.read_bytes()).hexdigest(), saved['raw_sha256'])
        arrays = np.load(raw_path, allow_pickle=False)
        for row in saved['replicate_results']:
            st = row['exact_statistics']; cal = st['baseline_calibration']; pairs = st['baseline_pairs']; new = st['retimed_pairs']
            self.assertEqual(F(d['baseline_lag_unit'])*(pairs[0]['records']+2*pairs[1]['records']), 900)
            self.assertEqual(F(d['new_lag'])*new['records'], 900)
            self.assertEqual(cal['records']+2*sum(p['records'] for p in pairs), 16000)
            self.assertEqual(2*new['records'], 3600)
            self.assertEqual(cal['records']+sum(p['records'] for p in pairs), 10000)
            self.assertEqual(new['records'], 1800)
            if row['replicate']: continue
            name = row['case']; z = arrays[name+'__baseline_calibration']
            self.assertEqual(int(z@z), cal['sum_squares'])
            for k, pair in enumerate(pairs, 1):
                self.assertEqual(int(arrays[name+f'__baseline_lag{k}_left']@arrays[name+f'__baseline_lag{k}_right']), pair['product_sum'])
            self.assertEqual(int(arrays[name+'__retimed_left']@arrays[name+'__retimed_right']), new['product_sum'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReview)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  external_peer_review=False, python_version=platform.python_version(),
                  numpy_version=np.__version__,
                  reproduction='PYTHONPATH=src python attainability/tests/test_independent.py',
                  reproduction_workdir='release root',
                  scope='Independent proof challenges, model counterexamples, and replay of frozen simulated records; no new empirical physical measurements')
    sources = {'tests/test_independent.py': Path(__file__)}
    sources.update({'code/'+name+'.py': Path(module.__file__) for name, module in
                    [('spectral_dynamics', sd), ('physical_minimum', pm), ('joint_analog', ja), ('design', design), ('retiming', retiming)]})
    for name in ('contracts/joint_analog.json', 'contracts/design.json',
                 'results/joint_analog.json', 'results/joint_analog_raw.npz',
                 'results/design.json', 'results/physical_minimum.json',
                 'results/spectral_dynamics_replay.json',
                 'results/joint_analog_population_audit.json',
                 'contracts/retiming.json', 'results/retiming.json', 'results/retiming_raw.npz'):
        sources[name] = ROOT/name
    report['source_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    (ROOT/'results/independent_review.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    raise SystemExit(not result.wasSuccessful())
