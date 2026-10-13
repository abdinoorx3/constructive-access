from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import sampling
import numpy as np
from scipy.linalg import expm,eigvalsh
from constructive_access.frontier import energy_certificate as energy


class DependentSamplingTests(unittest.TestCase):
    def test_rational_confidence_radius_is_outward(self):
        for n,m,alpha in [(600,2,F(1,20)),(7,3,F(1,100)),(100000,12,F(1,1000))]:
            radius=sampling.confidence_radius(n,m,alpha)
            self.assertGreaterEqual(2*n*radius*radius,energy.log_interval(F(2*m)/alpha)[1])
            self.assertGreaterEqual(float(radius),np.sqrt(np.log(2*m/float(alpha))/(2*n)))

    def test_bias_is_added_once_not_multiplied_by_records(self):
        box=sampling.confidence_box([10000],[5000],F(1,50),F(1,20),[F(1,100)],
                                     [F(1,200)],F(1,100))
        self.assertEqual(box["intervals"],[(F(93,200),F(107,200))])
        self.assertEqual(box["total_failure_upper"],F(3,50))

    def test_zero_bias_matches_previous_independent_certificate(self):
        lags=[16,21]; step=F(1,10); weights=[F(1,2)]*2
        counts=[2560,3680]; records=[5000]*2; radius=F(1,40); alpha=F(1,20)
        scores=[(F(488,376),F(512,624)),(F(264,376),F(736,624))]
        old=energy.from_counts(lags,step,weights,records,counts,radius,alpha,scores)
        new=sampling.certify_counts(lags,step,weights,records,counts,radius,alpha,
                                    [0,0],scores)
        self.assertEqual(old["certificate"],new["certificate"])
        self.assertGreater(new["certificate"]["entropy_production_rate_lower"],0)

    def test_large_valid_bias_removes_unjustified_positive_bound(self):
        result=sampling.certify_counts([1,2],1,[F(1,2)]*2,[600]*2,[300,588],
                                      F(61,1000),F(1,20),[F(3,4)]*2,
                                      [(2,1),(1,2)])
        self.assertEqual(result["confidence"]["intervals"],[(0,1),(F(169,1000),1)])
        self.assertEqual(result["certificate"]["entropy_production_rate_lower"],0)

    def test_predictable_routing_conditional_mgf(self):
        # Enumerate an adaptive four-record experiment with quotas2,2 and
        # conditional means changing with the preceding outcome. No iid law.
        states=[(F(1),(),(0,0),F(0))]
        lam=.9; target=[F(1,3),F(2,3)]; delta=F(1,10)
        for _ in range(4):
            following=[]
            for mass,history,quotas,centered in states:
                preferred=(history[-1] if history else 0)
                k=preferred if quotas[preferred]<2 else 1-preferred
                mu=target[k]+(delta if history and history[-1] else -delta)
                q=list(quotas);q[k]+=1
                for y in (0,1):
                    following.append((mass*(mu if y else 1-mu),history+(y,),tuple(q),
                                      centered+(y-mu if k==0 else 0)))
            states=following
        mgf=sum(float(mass)*np.exp(lam*float(centered)-lam*lam*2/8)
                for mass,_,quotas,centered in states)
        self.assertLessEqual(mgf,1+1e-14)
        self.assertTrue(all(quotas==(2,2) for _,_,quotas,_ in states))

    def test_doeblin_and_exact_exponential_envelopes(self):
        self.assertEqual(sampling.doeblin_mix_bound(F(1,4),4),F(81,256))
        for x in [F(0),F(1,10),F(6),F(120)]:
            self.assertGreaterEqual(float(sampling.exp_negative_upper(x)),np.exp(-float(x))*(1-1e-15))
        envelope=sampling.spectral_mix_bound(F(1,500),F(1,4),3000)
        self.assertLess(envelope["conditional_bias_upper"],F(1,400))

    def test_mixing_bound_for_irreversible_generator(self):
        # A biased cycle is not reversible; the symmetric-part gap still
        # gives the asserted uniform total-variation bound.
        n=5;Q=np.zeros((n,n))
        for i in range(n):
            Q[i,(i+1)%n]=3;Q[i,(i-1)%n]=1;Q[i,i]=-4
        gap=float(eigvalsh(-(Q+Q.T)/2)[1])
        lower=F(str(gap)).limit_denominator(10000)-F(1,10000)
        for time in [F(1,10),F(1),F(3)]:
            P=expm(Q*float(time));actual=np.max(np.sum(abs(P-1/n),axis=1)/2)
            bound=sampling.spectral_mix_bound(lower,F(1,n),time)
            self.assertLessEqual(actual,float(bound["conditional_bias_upper"])+2e-14)

    def test_stationary_pair_mean_does_not_imply_conditional_mean(self):
        Q=np.zeros((4,4));rates=[.02,2.]
        for sector in range(2):
            for bit in range(2):
                i=2*sector+bit
                Q[i,2*sector+1-bit]=rates[sector]
                Q[i,2*(1-sector)+bit]=.001
                Q[i,i]=-rates[sector]-.001
        P=expm(Q);detector=np.array([0,1,0,1])
        conditional=np.array([sum(P[i,j] for j in range(4) if detector[i]==detector[j])
                              for i in range(4)])
        self.assertTrue(np.allclose(np.full(4,.25)@P,np.full(4,.25)))
        self.assertGreater(max(abs(conditional-np.mean(conditional))),.2)

    def test_invalid_contracts_are_rejected(self):
        valid=dict(records=[600],successes=[300],statistical_radii=F(61,1000),
                   alpha=F(1,20),conditional_bias_radii=[0])
        for change in [dict(statistical_radii=F(1,1000)),dict(successes=[601]),
                       dict(records=[True]),dict(conditional_bias_radii=[F(-1,10)]),
                       dict(sampling_design="stop_when_significant"),
                       dict(calibration_failure=F(99,100))]:
            with self.assertRaises((TypeError,ValueError)):
                sampling.confidence_box(**(valid|change))


if __name__=="__main__":
    unittest.main()
