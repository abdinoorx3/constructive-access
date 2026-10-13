from fractions import Fraction as F
import math
import numpy as np
import unittest
from scipy.optimize import linprog
from nonlinear import (linear_minimum, certify, verify, design, primal_bound,
                       confidence_region, kl_interval, calibrated_normalizer, activity_bridge,
                       diffusion_example)
from constructive_access.frontier import energy_certificate as ec


class NonlinearTests(unittest.TestCase):
    def test_linear_program_against_independent_scipy(self):
        rng=np.random.default_rng(43029)
        for _ in range(80):
            a=[F(int(x),7) for x in rng.integers(-7,8,5)]
            b=[F(int(x),11) for x in rng.integers(-11,12,5)]
            boxes=[(F(1,10),F(9,10))]*5
            maxval=sum(c*(hi if c>=0 else lo) for c,(lo,hi) in zip(a,boxes))
            minval=sum(c*(lo if c>=0 else hi) for c,(lo,hi) in zip(a,boxes))
            ell=(2*minval+3*maxval)/5
            out=linear_minimum(b,a,ell,boxes)
            ref=linprog(list(map(float,b)),A_ub=[[-float(x) for x in a]],b_ub=[-float(ell)],
                        bounds=[tuple(map(float,z)) for z in boxes],method='highs')
            assert ref.success
            assert abs(float(out['value'])-ref.fun)<1e-10
            assert sum(c*x for c,x in zip(a,out['minimizer']))>=ell


    def test_exact_bernoulli_projection_and_gain_beyond_pinsker(self):
        out=design([1],[1],[(0,0)],[1],F(99,100),[1],grid=3)
        lo=out['certificate']['record_kl_lower'];hi=out['primal']['record_kl_relaxation_upper']
        truth=.99*math.log(1.98)+.01*math.log(.02)
        assert abs(float(lo)-truth)<1e-12
        assert hi-lo<F(1,10**18)
        assert float(4*lo)>1.32*8*.49**2


    def test_equilibrium_feasible_point_forces_zero_lower(self):
        out=design([1,2],[F(1,2)]*2,[(0,1)],[1,-1],F(-1,10),[1,2],grid=9)
        assert out['certificate']['record_kl_lower']==0
        assert float(out['primal']['record_kl_relaxation_upper'])<1e-9


    def test_continuous_normalizer_dominates_offgrid_points(self):
        lags=[1,2,4];weights=[F(1,3)]*3;vertices=[(0,F(1,2)),(F(1,4),F(3,4))]
        scores=[(F(2),F(1,3)),(F(1,2),F(7,4)),(F(3,4),F(9,8))]
        out=calibrated_normalizer(lags,weights,vertices,scores,depth=5)
        for u,A in vertices:
            for j in range(1001):
                x=F(j,1000)
                q=[(1+u+(A-u)*x**k)/2 for k in lags]
                value=sum(w*((1-z)*h0+z*h1) for w,z,(h0,h1) in zip(weights,q,scores))
                assert value<=out['normalizer_upper']


    def test_dual_and_primal_bracket_nonlinear_projection(self):
        out=design([1,2,3],[F(1,3)]*3,[(0,0),(0,1),(1,1)],
                   [F(-4,9),F(4,3),-1],F(-3,100),[F(3,20),F(3,10),F(9,20)],grid=21)
        assert out['certificate']['record_kl_lower']>0
        assert 0<=out['certified_relaxation_gap_upper']<F(1,10**6)
        assert 'not an upper bound on physical entropy' in out['primal']['interpretation']


    def test_certificate_data_binding_and_postselected_scores(self):
        contract=dict(lags=[1],weights=[1],vertices=[(0,0)],coefficients=[1],
                      lower=F(4,5),durations=[2],scores=[(F(2,5),F(8,5))])
        cert=certify(**contract)
        assert verify(cert,**contract)
        changed=dict(contract,lower=F(3,4))
        assert not verify(cert,**changed)


    def test_reject_empty_regions_and_invalid_primal_points(self):
        with self.assertRaisesRegex(ValueError,'empty'):
            certify([1],[1],[(0,0)],[1],2,[1],[(1,1)])
        with self.assertRaisesRegex(ValueError,'halfspace'):
            primal_bound([1],[1],[(0,0)],[1],F(4,5),[F(1,2)],[(0,F(1,2),1)])
        with self.assertRaisesRegex(ValueError,'sum exactly'):
            primal_bound([1],[1],[(0,0)],[1],F(4,5),[F(4,5)],[(0,F(1,2),F(1,2))])


    def test_fixed_data_confidence_cost_and_bias(self):
        out=confidence_region([1,-1],[100,200],[70,100],conditional_radii=[F(1,100),F(1,50)])
        assert out['empirical_contrast']==F(1,5)
        assert out['conditional_bias_penalty']==F(3,100)
        expected=math.sqrt(math.log(20)*(1/100+1/200)/2)
        assert expected<=float(out['sampling_radius_upper'])+1e-15
        with self.assertRaisesRegex(ValueError,'predeclared'):
            confidence_region([1],[100],[70],selection='postselected')


    def test_endpoint_kl_and_degenerate_box(self):
        lo,hi=kl_interval(0,F(1,2))
        assert float(lo)<=math.log(2)+1e-15 and float(hi)>=math.log(2)-1e-15
        out=certify([1],[1],[(0,0)],[0],0,[1],[(F(2,5),F(8,5))],boxes=[(F(4,5),F(4,5))])
        truth=.8*math.log(1.6)+.2*math.log(.4)
        assert abs(float(out['record_kl_lower'])-truth)<1e-12


    def test_calibrated_negative_correlation_control(self):
        # A=0.2 caps reversible agreement by0.6. The rational signal0.1 is a
        # deterministic input vector, not a claimed physical realization.
        common=dict(lags=[1],weights=[1],coefficients=[-1],lower=F(-1,10),durations=[1])
        out=design(vertices=[(0,F(1,5))],grid=3,**common)
        assert abs(float(out['certificate']['record_kl_lower'])-(.1*math.log(.2)+.9*math.log(1.8)))<1e-10

    def test_second_moment_calibration_excludes_positive_decreasing_signal(self):
        common=dict(lags=[1,2],weights=[F(1,2)]*2,coefficients=[F(38,25),-1],
                    lower=F(1147,2500),durations=[1,2],grid=21)
        unknown=design(vertices=[(0,0),(0,1),(1,1)],**common)
        known=design(vertices=[(0,F(1,2))],**common)
        assert unknown['certificate']['record_kl_lower']==0
        assert known['certificate']['record_kl_lower']>F(1,1000)
        assert known['certified_relaxation_gap_upper']<F(1,10000)

    def test_activity_bridge_sharp_cycle_and_nonlinear_gain(self):
        x=F(3,4)
        jlo=((1+x)*ec.log_interval(1+x)[0]+(1-x)*ec.log_interval(1-x)[0])/2
        out=activity_bridge(15*jlo,3,5)
        exact=5*.75*math.log(7)
        assert out['status']=='certified_activity_bridge'
        assert abs(float(out['entropy_production_rate_lower'])-exact)<1e-12
        assert out['entropy_production_rate_lower']>out['linear_bridge_lower']
        weaker=activity_bridge(15*jlo,3,50)
        assert weaker['entropy_production_rate_lower']<out['entropy_production_rate_lower']

    def test_activity_bridge_domain_and_independent_inverse(self):
        from scipy.optimize import brentq
        for value in [F(1,10000),F(1,10),F(1,2),F(69,100)]:
            out=activity_bridge(value,1,1)
            root=brentq(lambda x:((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))/2-float(value),0,1-1e-14,xtol=1e-14)
            truth=root*math.log((1+root)/(1-root))
            assert abs(float(out['entropy_production_rate_lower'])-truth)<1e-10
        assert activity_bridge(F(7,10),1,1)['status']=='incompatible_activity_and_record_bounds'
        assert activity_bridge(0,1,1)['entropy_production_rate_lower']==0

    def test_diffusion_transfer_gaussian_calculation_and_control(self):
        out=diffusion_example();truth=math.exp(-1)*math.sinh(math.exp(-1)*math.cos(2))
        lo,hi=out['driven']['detector_correlation_interval']
        assert float(lo)<=truth<=float(hi)
        assert hi<0
        assert 0<out['driven']['certificate']['entropy_production_rate_lower']<8
        assert out['equilibrium']['certificate']['entropy_production_rate_lower']==0
        # Independent stationary covariance and quadratic current-energy check.
        rotation=np.array([[0.,-2.],[2.,0.]]);drift=-np.eye(2)+rotation
        np.testing.assert_array_equal(drift+drift.T+2*np.eye(2),np.zeros((2,2)))
        assert np.trace(rotation@rotation.T)==float(out['driven']['exact_stationary_entropy_rate'])
        assert out['driven']['exact_full_path_kl_to_reversible_reference']==2


if __name__=='__main__':
    unittest.main()
