import importlib.util,json,hashlib
from pathlib import Path
from fractions import Fraction as F
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dissipation_limits',ROOT/'code'/'dissipation_limits.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class DissipationLimitsTests(unittest.TestCase):
    def test_exact_stationarity_activity_and_flux(self):
        for rates in [[F(1)]*4,[F(51,50),F(26,25),F(49,50),F(24,25)]]:
            for q in [F(1,2),F(1,10),F(1,1000)]:
                m=d.model(rates,q);pi=m['stationary'];Q=m['generator']
                self.assertTrue(all(sum(pi[i]*Q[i][j] for i in range(4))==0 for j in range(4)))
                self.assertEqual(sum(pi[i]*rates[i] for i in range(4)),m['activity'])
                forward=[pi[i]*Q[i][(i+1)%4] for i in range(4)]
                reverse=[pi[(i+1)%4]*Q[(i+1)%4][i] for i in range(4)]
                self.assertEqual(len(set(forward)),1);self.assertEqual(len(set(reverse)),1)
                self.assertTrue(all(forward[i]/reverse[i]==(1-q)/q for i in range(4)))

    def test_complete_parity_path_exact_equivalence(self):
        intervals=[.13,.27,.04,.41,.15]
        expected=-sum(intervals)
        for q in [F(1,2),F(1,3),F(1,1000)]:
            self.assertAlmostEqual(d.parity_log_likelihood([1]*4,q,intervals),expected,places=13)
        self.assertEqual(d.coarse_tv_bound([1]*4,F(1000000)),0)

    def test_rate_tags_break_parity_likelihood_equivalence(self):
        rates=[F(6,5),F(7,5),F(4,5),F(3,5)]
        a=d.parity_log_likelihood(rates,F(1,2),[.1,.9,.2,.5,.8])
        b=d.parity_log_likelihood(rates,F(1,100),[.1,.9,.2,.5,.8])
        self.assertGreater(abs(a-b),1e-5)

    def test_entropy_against_stationary_edge_formula(self):
        rates=[F(3),F(2),F(4),F(5)]
        for q in [F(1,2),F(1,10),F(1,1000)]:
            m=d.model(rates,q);pi=m['stationary'];Q=m['generator']
            value=0.
            for i in range(4):
                a=float(pi[i]*Q[i][(i+1)%4]);b=float(pi[(i+1)%4]*Q[(i+1)%4][i])
                value+=(a-b)*np.log(a/b)
            lo,hi=map(float,m['entropy_production_interval'])
            self.assertLessEqual(lo-1e-12,value);self.assertGreaterEqual(hi+1e-12,value)
            A=float(m['stationary_flux_affinity_interval'][0])
            self.assertAlmostEqual(d.affinity_cap_upper(float(m['activity']),A),value,places=10)

    def test_finite_budget_gap_at_fixed_graph_and_rates(self):
        rates=[F(51,50),F(26,25),F(49,50),F(24,25)]
        result=d.finite_budget_pair(rates,F(10),F(1,100),F(100))
        self.assertGreater(result['entropy_gap_lower'],100)
        self.assertLessEqual(result['full_record_tv_upper'],F(1,100))
        self.assertTrue(0<result['high_entropy_reverse_probability']<result['reference_reverse_probability'])
        forced=d.unavoidable_infinite_upper_probability(rates,result['reference_reverse_probability'],F(10))
        self.assertEqual(forced,F(189,200))

    def test_pair_tv_by_independent_density_integral(self):
        rates=[F(6,5),F(7,5),F(4,5),F(3,5)]
        distances=[]
        for a,b in [(float(rates[0]),float(rates[2])),(float(rates[1]),float(rates[3]))]:
            crossing=np.log(a/b)/(a-b)
            f=lambda x:abs(a*np.exp(-a*x)-b*np.exp(-b*x))
            integral=quad(f,0,crossing,epsabs=1e-13)[0]+quad(f,crossing,np.inf,epsabs=1e-13)[0]
            distances.append(integral/2)
        actual=.4*distances[0]*distances[1]
        self.assertAlmostEqual(d.pair_tv(rates,F(1,2),F(1,10)),actual,places=12)

    def test_window_signal_against_transition_matrix(self):
        rates=[F(6,5),F(7,5),F(4,5),F(3,5)]
        values=[]
        for q in [F(1,2),F(1,10)]:
            m=d.model(rates,q);P=np.array([[float(x/rates[i]) for x in row] for i,row in enumerate(m['generator'])]);np.fill_diagonal(P,0)
            cdf=1-np.exp(-np.array(list(map(float,rates))))
            values.append(np.array([.5,0,.5,0])@np.diag(cdf)@P@cdf)
        coefficient=(values[1]-values[0])/.4
        lo,hi=map(float,d.window_signal(rates))
        self.assertLessEqual(lo-1e-14,coefficient);self.assertGreaterEqual(hi+1e-14,coefficient)

    def test_weak_tags_have_quadratic_signal(self):
        ratios=[]
        for eta in [F(1,100),F(1,200),F(1,400)]:
            rates=[1+eta,1+2*eta,1-eta,1-2*eta]
            lo,hi=d.window_signal(rates);ratios.append(float((lo+hi)/(2*eta*eta)))
        self.assertAlmostEqual(ratios[-1],4/np.e**2,places=5)

    def test_fixed_jump_interval_and_physical_duration(self):
        rates=[F(51,50),F(26,25),F(49,50),F(24,25)]
        result=d.direction_count_interval(rates,1000,10,F(1,1000))
        qlo,qhi=result['reverse_interval']
        self.assertLessEqual(float(qlo),beta.ppf(.025,10,991)+1e-12)
        self.assertGreaterEqual(float(qhi),beta.ppf(.975,11,990)-1e-12)
        self.assertGreater(result['entropy_production_interval'][0],0)
        self.assertLess(result['entropy_production_interval'][1],result['prior_entropy_interval'][1])
        self.assertEqual(result['optional_timeout_duration'],F(6250,3))
        self.assertLess(result['timeout_probability_upper'],F(1,10**100))

    def test_no_floor_distinguishes_missing_and_present_reverse_evidence(self):
        missing=d.direction_count_interval([1]*4,1000,0,F(0))
        present=d.direction_count_interval([1]*4,1000,10,F(0))
        self.assertEqual(missing['status'],'unbounded_upper')
        self.assertIsNone(missing['entropy_production_interval'][1])
        self.assertEqual(present['status'],'finite_interval')
        self.assertGreater(present['reverse_interval'][0],0)

    def test_coupled_equal_exit_records(self):
        result=d.coupled_records([1]*4,F(1,2),F(1,1000),F(5),300,441)
        self.assertEqual(result['parity_record_coupling']['mismatches'],0)
        self.assertGreater(result['full_record_coupling']['mismatches'],200)

    def test_input_contracts(self):
        with self.assertRaises(TypeError):d.model([1]*4,.1)
        with self.assertRaises(ValueError):d.model([1,1,1],F(1,10))
        with self.assertRaises(ValueError):d.model([1]*4,F(0))
        with self.assertRaises(ValueError):d.direction_count_interval([1]*4,100,1,F(-1))

    def test_frozen_result_provenance(self):
        result=json.loads((ROOT/'results'/'dissipation_limits.json').read_text())
        self.assertEqual(result['contract_sha256'],hashlib.sha256((ROOT/'contracts'/'dissipation_limits.json').read_bytes()).hexdigest())
        self.assertEqual(result['coupling_checks'][0]['parity_record_coupling']['mismatches'],0)
        self.assertEqual(result['finite_budget_pair']['state_count'],4)


if __name__=='__main__':unittest.main()
