from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import dependence
import numpy as np
from scipy.linalg import expm
from math import comb


def raw_count_moments(Q,detector,initial,groups,H):
    """Independent matrix derivative oracle for a weighted pair-count sum."""
    P=expm(Q*H)
    rows=[initial.copy()]+[np.zeros(len(Q)) for _ in range(4)]
    for records,lag,weight in groups:
        observed=expm(Q*lag)
        mask=detector[:,None]==detector[None,:]
        A=(observed*mask)@expm(Q*(H-lag))
        for _ in range(records):
            old=[row.copy() for row in rows]
            rows[0]=old[0]@P
            for order in range(1,5):
                rows[order]=old[order]@P+sum((comb(order,j)*weight**j*(old[order-j]@A)
                                                   for j in range(1,order+1)),np.zeros(len(Q)))
    raw=[float(row.sum()) for row in rows]
    mean=raw[1];variance=raw[2]-mean*mean
    fourth=raw[4]-4*mean*raw[3]+6*mean*mean*raw[2]-3*mean**4
    return mean,max(0.,variance),max(0.,fourth)


class DependenceTests(unittest.TestCase):
    def test_proxy_outward_and_not_iid_limit(self):
        for gap,H in [(F(1,500),F(2)),(F(1),F(1)),(F(10),F(10))]:
            proxy=dependence.block_proxy(gap,H)
            rho=np.exp(-float(gap*H));A=(1+rho)/(1-rho)
            self.assertGreaterEqual(float(proxy["contraction_upper"]),rho*(1-1e-15))
            self.assertGreaterEqual(float(proxy["whole_record_proxy_upper"]),(np.sqrt(A)+1)**2*(1-1e-13))
            self.assertGreaterEqual(proxy["whole_record_proxy_upper"],4)

    def test_initial_distance_is_added_once_to_failure(self):
        result=dependence.certify_counts([1,2],[-1,1],[100000]*2,[38000,53000],
                                         [1,2],2,F(1,500),F(49,1000),F(1,1000))
        self.assertEqual(result["total_failure_upper"],F(1,20))
        self.assertGreater(result["reversible_gap_lower"],0)
        self.assertEqual(result["initial_path_tv_upper"],F(1,1000))

    def test_box_and_calibration(self):
        box=dependence.confidence_box([100000]*2,[40000,60000],2,F(1,500),F(1,25),
                                      F(1,100),[F(1,100),F(1,100)])
        self.assertEqual(box["total_failure_upper"],F(1,20))
        self.assertEqual(box["intervals"][0][0],F(2,5)-box["radii"][0]-F(1,100))

    def test_optimized_old_baseline_counts_full_time(self):
        plan=dependence.optimize_old_gap(400000,2,F(1,500),F(1,8),[-1,1],F(1,20),
                                         [0,1000,2000,2500,3000])
        self.assertLessEqual(plan["physical_time"],400000)
        self.assertEqual(plan["physical_time"],plan["pair_records"]*(plan["blind_gap"]+2))
        self.assertEqual(plan["endpoint_readouts"],2*plan["pair_records"])
        for G in [0,1000,2000,2500,3000]:
            candidate=dependence.optimize_old_gap(400000,2,F(1,500),F(1,8),[-1,1],F(1,20),[G])
            self.assertLessEqual(plan["contrast_halfwidth"],candidate["contrast_halfwidth"])

    def test_sector_sampler_matches_independent_matrix_moments(self):
        spec=dict(kind="sector_bit",sector_flip_rate="3/10",bit_flip_rates=["1/5","7/5"])
        Q,detector=dependence._model(spec);initial=np.array([1.,0,0,0])
        groups=[(40,.9,-1),(40,1.7,1)];H=1.7
        mean,variance,fourth=raw_count_moments(Q,detector,initial,groups,H)
        rng=np.random.default_rng(590271);values=[];reps=4000
        for _ in range(reps):
            first,state=dependence.sector_group(rng,0,40,H,.9,.3,[.2,1.4])
            second,state=dependence.sector_group(rng,state,40,H,1.7,.3,[.2,1.4])
            values.append(second-first)
        values=np.array(values)
        self.assertLess(abs(float(values.mean())-mean),8*np.sqrt(variance/reps))
        self.assertLess(abs(float(values.var())-variance),8*np.sqrt(fourth/reps))

    def test_ring_sampler_matches_independent_matrix_moments(self):
        spec=dict(kind="ring",states=8,forward_rate="4",backward_rate="1/10",
                  detector=[0,0,0,0,1,1,1,1])
        Q,detector=dependence._model(spec);initial=np.zeros(8);initial[0]=1
        groups=[(40,1.,-1),(40,2.,1)];H=2.
        mean,variance,fourth=raw_count_moments(Q,detector,initial,groups,H)
        rng=np.random.default_rng(120479);values=[];reps=4000
        for _ in range(reps):
            first,state=dependence.ring_group(rng,0,40,H,1.,4.,.1,detector)
            second,state=dependence.ring_group(rng,state,40,H,2.,4.,.1,detector)
            values.append(second-first)
        values=np.array(values)
        self.assertLess(abs(float(values.mean())-mean),8*np.sqrt(variance/reps))
        self.assertLess(abs(float(values.var())-variance),8*np.sqrt(fourth/reps))

    def test_zero_switch_conditional_sampler(self):
        rng=np.random.default_rng(1180)
        values=[dependence.sector_group(rng,0,100,2.,1.,0.,[.2,1.4])[0]
                for _ in range(1000)]
        p=(1+np.exp(-.4))/2
        self.assertLess(abs(np.mean(values)-100*p),5*np.sqrt(100*p*(1-p)/1000))

    def test_invalid_contracts_rejected(self):
        base=dict(lags=[1,2],coefficients=[-1,1],records=[1000]*2,successes=[400,600],
                  durations=[1,2],block_duration=2,gap_lower=F(1,500))
        for change in [dict(schedule="adaptive"),dict(selection="postselected"),
                       dict(durations=[1,3]),dict(gap_lower=0),dict(coefficients=[0,0]),
                       dict(initial_tv=F(99,100)),dict(records=[True,1000])]:
            with self.assertRaises((TypeError,ValueError)):
                dependence.certify_counts(**(base|change))

    def test_normalized_resource_frontier_counts_readouts(self):
        kwargs=dict(budget=400000,burn=4000,gap_lower=F(1,500),
                    coefficients=[-1,1],alpha=F(49,1000),readout_cost_ratio=10)
        plan=dependence.optimize_block_cost(**kwargs,candidates=[2,10,20,50,100])
        self.assertEqual(plan["normalized_cost"],plan["physical_time"]+10*plan["endpoint_readouts"])
        self.assertLessEqual(plan["normalized_cost"],400000)
        self.assertEqual(plan["physical_time"],4000+plan["pair_records"]*plan["block_duration"])
        for H in [2,10,20,50,100]:
            row=dependence.optimize_block_cost(**kwargs,candidates=[H])
            self.assertLessEqual(plan["contrast_halfwidth"],row["contrast_halfwidth"])
        zero=dependence.optimize_block_cost(**(kwargs|dict(readout_cost_ratio=0)),candidates=[2,10,20,50,100])
        self.assertEqual(zero["block_duration"],2)

    def test_padded_sector_sampler_matches_matrix_oracle(self):
        spec=dict(kind="sector_bit",sector_flip_rate="1/100",bit_flip_rates=["1/5","7/5"])
        Q,detector=dependence._model(spec);initial=np.array([1.,0,0,0])
        groups=[(20,1.,-1),(20,2.,1)];H=50.
        mean,variance,fourth=raw_count_moments(Q,detector,initial,groups,H)
        rng=np.random.default_rng(89351);values=[];reps=4000
        for _ in range(reps):
            first,state=dependence.sector_group(rng,0,20,H,1.,.01,[.2,1.4])
            second,state=dependence.sector_group(rng,state,20,H,2.,.01,[.2,1.4])
            values.append(second-first)
        self.assertLess(abs(float(np.mean(values))-mean),8*np.sqrt(variance/reps))
        self.assertLess(abs(float(np.var(values))-variance),8*np.sqrt(fourth/reps))


if __name__=="__main__":unittest.main()
