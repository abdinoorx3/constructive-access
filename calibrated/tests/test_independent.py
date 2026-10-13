"""Independent mathematics and physical countermodels for calibrated inference."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util
import hashlib
import json
import math
import sys
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq,linprog


def producer(name):
    path=Path(__file__).resolve().parents[1]/"code"/(name+".py")
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("calibrated_review_"+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def telegraph(decay):
    rate=-math.log(decay)/2
    return np.array([[-rate,rate],[rate,-rate]])


def correlation(q,pi,g,t):
    return float((pi*g)@expm(q*t)@g)


def path_probability(q,pi,detector,bits,gaps):
    state=pi.copy()
    for index,bit in enumerate(bits):
        if index:state=state@expm(q*gaps[index-1])
        state=state*(detector if bit else 1-detector)
    return float(state.sum())


class IndependentCalibrationTests(unittest.TestCase):
    def test_same_bit_square_is_not_independent_zero_lag_calibration(self):
        pi=np.array([.3,.7]);g=np.array([-.4,.7])
        variance=float(pi@(g*g))
        same_readout_square=1.
        independent_agreement=float(pi@(((1+g)/2)**2+((1-g)/2)**2))
        self.assertAlmostEqual(2*independent_agreement-1,variance)
        self.assertNotAlmostEqual(same_readout_square,variance)

    def test_exact_positive_lag_pair_ambiguity_with_irreducible_equilibrium_models(self):
        high=np.kron(telegraph(.25),np.eye(2))+np.kron(np.eye(2),telegraph(.75))
        signs=np.array([1.,-1.])
        high_g=(math.sqrt(3)*np.kron(signs,np.ones(2))+np.kron(np.ones(2),signs))/4
        low=telegraph(.5);low_g=math.sqrt(3)*signs/4
        self.assertAlmostEqual(np.mean(high_g**2),.25)
        self.assertAlmostEqual(np.mean(low_g**2),3/16)
        for lag,truth in ((1,3/32),(2,3/64)):
            self.assertAlmostEqual(correlation(high,np.ones(4)/4,high_g,lag),truth)
            self.assertAlmostEqual(correlation(low,np.ones(2)/2,low_g,lag),truth)
            for bit0 in (0,1):
                for bit1 in (0,1):
                    p=path_probability(high,np.ones(4)/4,(1+high_g)/2,[bit0,bit1],[lag])
                    q=path_probability(low,np.ones(2)/2,(1+low_g)/2,[bit0,bit1],[lag])
                    self.assertAlmostEqual(p,q)

    def test_calibrated_second_moment_can_exclude_uncalibrated_reversible_fit(self):
        size=8;angle=2*math.pi/size
        activity=.2/(1-math.cos(angle));difference=.4/math.sin(angle)
        forward=(activity+difference)/2;reverse=(activity-difference)/2
        self.assertGreater(reverse,0)
        q=np.zeros((size,size))
        for i in range(size):
            q[i,(i+1)%size]=forward;q[i,(i-1)%size]=reverse;q[i,i]=-activity
        g=np.cos(angle*np.arange(size));pi=np.ones(size)/size
        amplitude=float(pi@(g*g))
        c1=correlation(q,pi,g,1);c2=correlation(q,pi,g,2)
        self.assertGreater(c1,0);self.assertGreater(c2,0)
        self.assertLess(c2,c1*c1/amplitude)
        fitted_amplitude=c1*c1/c2;fitted_decay=c2/c1
        self.assertLessEqual(fitted_amplitude,1)
        equilibrium=telegraph(fitted_decay)
        detector=math.sqrt(fitted_amplitude)*np.array([1.,-1.])
        for lag,c in ((1,c1),(2,c2)):
            self.assertAlmostEqual(correlation(equilibrium,np.ones(2)/2,detector,lag),c)
        self.assertGreater(fitted_amplitude,amplitude)

    def test_two_point_detector_realizes_all_valid_mean_second_moment_pairs(self):
        for mean in (-.9,-.4,0,.7,.95):
            for fraction in (0,.2,.7,1):
                second=mean**2+fraction*(1-mean**2)
                lower=(mean-second)/(1-mean)
                mass=(mean-lower)/(1-lower)
                self.assertGreaterEqual(lower,-1-1e-13)
                self.assertLessEqual(lower,mean+1e-13)
                pi=np.array([1-mass,mass]);g=np.array([lower,1.])
                self.assertAlmostEqual(float(pi@g),mean)
                self.assertAlmostEqual(float(pi@(g*g)),second)
                q=np.array([[-mass,mass],[1-mass,-(1-mass)]])*1.7
                for t in (.1,1,3):
                    self.assertAlmostEqual(correlation(q,pi,g,t),mean**2+(second-mean**2)*math.exp(-1.7*t))

    def test_parity_cycle_exactly_hides_unbounded_entropy_without_invisible_jumps(self):
        rate=1.7;detector=np.array([0.,1,0,1]);pi=np.ones(4)/4
        visible=np.array([[-rate,rate],[rate,-rate]])
        import itertools
        for bias in (.5,.8,.99,.99999):
            q=np.zeros((4,4))
            for i in range(4):
                q[i,(i+1)%4]=rate*bias;q[i,(i-1)%4]=rate*(1-bias);q[i,i]=-rate
            self.assertTrue(np.allclose(pi@q,0))
            for i in range(4):
                for j in range(4):
                    if i!=j and q[i,j]>0:self.assertNotEqual(detector[i],detector[j])
            entropy=sum((pi[i]*q[i,j]-pi[j]*q[j,i])*math.log(pi[i]*q[i,j]/(pi[j]*q[j,i]))
                        for i in range(4) for j in range(i+1,4) if q[i,j]>0)
            expected=rate*(2*bias-1)*math.log(bias/(1-bias))
            self.assertAlmostEqual(entropy,expected)
            for bits in itertools.product((0,1),repeat=4):
                p=path_probability(q,pi,detector,bits,[.13,.7,1.1])
                coarse=path_probability(visible,np.array([.5,.5]),np.array([0.,1]),bits,[.13,.7,1.1])
                self.assertAlmostEqual(p,coarse,places=13)

    def test_nonlumpable_near_irreversible_models_satisfy_exposure_coupling_bound(self):
        rates=np.array([.7,1.1,1.5,2.3]);pi=1/rates;pi/=pi.sum()
        matrices=[]
        for bias in (.93,1.):
            q=np.zeros((4,4))
            for i in range(4):
                q[i,(i+1)%4]=rates[i]*bias;q[i,(i-1)%4]=rates[i]*(1-bias);q[i,i]=-rates[i]
            self.assertTrue(np.allclose(pi@q,0));matrices.append(q)
        for time in (.02,.1,.4,1):
            for initial in np.eye(4):
                tv=.5*np.abs(initial@expm(matrices[0]*time)-initial@expm(matrices[1]*time)).sum()
                self.assertLessEqual(tv,.07*rates.max()*time+1e-14)

    def test_reversibilization_kl_can_saturate_while_physical_entropy_diverges(self):
        previous=0
        for bias in (.6,.9,.99,.999999):
            information=bias*math.log(2*bias)+(1-bias)*math.log(2*(1-bias))
            entropy=(2*bias-1)*math.log(bias/(1-bias))
            self.assertLess(information,math.log(2))
            self.assertGreater(entropy,4*information)
            self.assertGreater(entropy,previous);previous=entropy
        self.assertGreater(previous,13)

    def test_activity_calibrated_nonlinear_bridge_is_sharp_and_stronger(self):
        def information(x):
            return ((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))/2
        def entropy(x):return 2*x*math.atanh(x)
        def inverse(value):return brentq(lambda x:information(x)-value,0,1-1e-14)
        rng=np.random.default_rng(763417)
        for _ in range(100):
            asymmetry=rng.uniform(.01,.999,7)
            flux=rng.uniform(.02,3,7)
            activity=flux.sum()
            information_rate=sum(f*information(x) for f,x in zip(flux,asymmetry))
            physical_entropy=sum(f*entropy(x) for f,x in zip(flux,asymmetry))
            bridge=activity*entropy(inverse(information_rate/activity))
            self.assertGreater(bridge,4*information_rate)
            self.assertLessEqual(bridge,physical_entropy+1e-9)
            looser_cap=3*activity
            looser_bridge=looser_cap*entropy(inverse(information_rate/looser_cap))
            self.assertLess(looser_bridge,bridge)
        for asymmetry in (.05,.4,.9):
            activity=3.7
            inferred=activity*entropy(inverse(information(asymmetry)))
            self.assertAlmostEqual(inferred,activity*entropy(asymmetry),places=10)
        value=.2
        nearly_uncalibrated=100000*entropy(inverse(value/100000))
        self.assertAlmostEqual(nearly_uncalibrated,4*value,places=5)

    def test_calibration_module_realizations_and_general_finite_lag_ambiguity(self):
        module=producer("calibration")
        value=module.reversible_realization(F(1,3),F(5,9),[(F(1,4),F(2,3)),(F(3,4),F(1,3))])
        pi=np.array(list(map(float,value["stationary"])));g=np.array(list(map(float,value["detector"])))
        step=np.array([[float(x) for x in row] for row in value["step_matrix"]])
        self.assertTrue(np.allclose(pi@step,pi))
        self.assertTrue(np.allclose(pi[:,None]*step,(pi[:,None]*step).T))
        for k in range(1,7):
            expected=1/9+4/9*(2/3*.25**k+1/3*.75**k)
            self.assertAlmostEqual(float((pi*g)@np.linalg.matrix_power(step,k)@g),expected)
        for maximum in range(1,10):
            witness=module.positive_lag_ambiguity(maximum)
            high=witness["positive_spectrum"];low=witness["negative_spectrum"]
            for k in range(1,maximum+1):
                self.assertEqual(sum(w*x**k for x,w in high),sum(w*x**k for x,w in low))
            self.assertNotEqual(sum(w for x,w in high),sum(w for x,w in low))
            self.assertLessEqual(sum(math.sqrt(float(w)) for x,w in high),1)
            self.assertLessEqual(sum(math.sqrt(float(w)) for x,w in low),1)

    def test_calibrated_support_bounds_continuous_domain_and_delayed_access(self):
        module=producer("calibration")
        vertices=module.calibration_vertices((F(-1,4),F(1,3)),(F(1,5),F(3,5)))
        coefficients=[F(-3,4),F(4,3),F(-1)]
        certificate=module.calibrated_support([1,2,3],coefficients,vertices,depth=7)
        direct=max(sum(float(a)*(1+float(u)+(float(A)-float(u))*x**k)/2
                       for k,a in enumerate(coefficients,1))
                   for u,A in vertices for x in np.linspace(0,1,4001))
        self.assertGreaterEqual(float(certificate["reversible_support_upper"])+1e-13,direct)
        self.assertLessEqual(float(certificate["reversible_support_lower"]),direct+1e-6)
        q=np.array([[-1.,.9,.1],[.2,-.8,.6],[.7,.3,-1.]])
        system=q.T.copy();system[-1]=1;rhs=np.array([0.,0,1]);pi=np.linalg.solve(system,rhs)
        g=np.array([-.7,.2,.9]);second=float(pi@(g*g));delay=F(1,100)
        c=correlation(q,pi,g,float(delay))
        width=F(1,10**10);center=F(c).limit_denominator(10**12)
        result=module.delayed_calibration((center-width,center+width),delay,F(1))
        lo,hi=result["second_moment_interval"]
        self.assertLessEqual(float(lo),second);self.assertGreaterEqual(float(hi),second)

    def test_nonlinear_exact_halfspace_lp_matches_independent_solver(self):
        module=producer("nonlinear");rng=np.random.default_rng(493258)
        for _ in range(60):
            coefficients=[F(int(x),5) for x in rng.integers(-7,8,5)]
            objective=[F(int(x),7) for x in rng.integers(-9,10,5)]
            boxes=[(F(int(x),20),F(int(y),20)) for x,y in zip(rng.integers(0,8,5),rng.integers(12,21,5))]
            maximum=sum(c*(hi if c>=0 else lo) for c,(lo,hi) in zip(coefficients,boxes))
            minimum=sum(c*(lo if c>=0 else hi) for c,(lo,hi) in zip(coefficients,boxes))
            lower=(2*maximum+minimum)/3
            exact=module.linear_minimum(objective,coefficients,lower,boxes)
            numerical=linprog(np.array(list(map(float,objective))),A_ub=-np.array([list(map(float,coefficients))]),
                              b_ub=[-float(lower)],bounds=[tuple(map(float,x)) for x in boxes],method="highs")
            self.assertTrue(numerical.success)
            self.assertAlmostEqual(float(exact["value"]),numerical.fun,places=10)
            self.assertGreaterEqual(sum(a*p for a,p in zip(coefficients,exact["minimizer"])),lower)

    def test_activity_inversion_exact_certificate_matches_independent_root(self):
        module=producer("nonlinear")
        def j(x):return ((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))/2
        for value in (F(0),F(1,10000),F(1,20),F(1,4),F(3,5)):
            result=module.activity_bridge(value,F(3,2),F(2,3))
            self.assertEqual(result["status"],"certified_activity_bridge")
            root=0 if not value else brentq(lambda x:j(x)-float(value),0,1-1e-14)
            physical=2/3*2*root*math.atanh(root)
            bound=float(result["entropy_production_rate_lower"])
            self.assertLessEqual(bound,physical+1e-10)
            self.assertAlmostEqual(bound,physical,places=9)
            self.assertGreaterEqual(bound,4*float(value)/1.5-1e-14)
        self.assertEqual(module.activity_bridge(F(7,10),1,1)["status"],"incompatible_activity_and_record_bounds")
        with self.assertRaises(ValueError):module.activity_bridge(1,1,0)

    def test_nonlinear_dual_and_primal_bound_analytic_one_setting(self):
        module=producer("nonlinear")
        args=dict(lags=[1],weights=[F(1)],vertices=[(F(0),F(0))],coefficients=[F(1)],
                  lower=F(4,5),durations=[F(1)],scores=[(F(2,5),F(8,5))])
        certificate=module.certify(**args)
        expected=.8*math.log(1.6)+.2*math.log(.4)
        self.assertLessEqual(float(certificate["record_kl_lower"]),expected+1e-14)
        self.assertAlmostEqual(float(certificate["record_kl_lower"]),expected,places=12)
        self.assertTrue(module.verify(certificate,**args))
        modified=dict(certificate);modified["record_kl_lower"]+=F(1,1000)
        self.assertFalse(module.verify(modified,**args))
        primal=module.primal_bound(lags=[1],weights=[F(1)],vertices=[(F(0),F(0))],
                                  coefficients=[F(1)],lower=F(4,5),probabilities=[F(4,5)],atoms=[(0,F(1,2),F(1))])
        self.assertGreaterEqual(float(primal["record_kl_relaxation_upper"]),expected-1e-14)
        self.assertLess(float(primal["record_kl_relaxation_upper"]-certificate["record_kl_lower"]),1e-12)
        with self.assertRaises(ValueError):module.certify(**dict(args,lower=F(2)))

    def test_predictable_policy_eprocess_by_exact_history_enumeration(self):
        module=producer("sequential")
        import itertools
        probabilities=[F(1,2)+F(1,2)*F(3,5)**k for k in (1,2,3)]
        outcomes=list(itertools.product((0,1),repeat=3))
        masses=[math.prod(float(p if bit else 1-p) for p,bit in zip(probabilities,row)) for row in outcomes]
        # States carry exact scores and counts; only probability/exponential
        # evaluation uses floating point. The policy uses completed history.
        states=[((0,0,0),F(0),1.,False)]
        alpha=.9;threshold=math.log(1/alpha)
        for batch in range(4):
            updated=[]
            for counts,score,mass,crossed in states:
                decision=module.policy(list(counts),batch)
                a=decision["coefficients"];lam=decision["bet"];support=decision["support"]
                for row,prob in zip(outcomes,masses):
                    next_score=score+lam*(sum(c*y for c,y in zip(a,row))-support)-lam*lam*sum(c*c for c in a)/8
                    next_counts=tuple(x+y for x,y in zip(counts,row))
                    updated.append((next_counts,next_score,mass*prob,crossed or float(next_score)>threshold))
            states=updated
        self.assertAlmostEqual(sum(mass for _,_,mass,_ in states),1,places=12)
        self.assertLessEqual(sum(mass*math.exp(float(score)) for _,score,mass,_ in states),1+1e-12)
        self.assertLessEqual(sum(mass for _,_,mass,crossed in states if crossed),alpha+1e-12)

    def test_sequential_replay_binds_chronology_and_calibration(self):
        module=producer("sequential")
        records=[[1,1,0]]*60+[[0,1,0]]*15
        certificate=module.certify(records,alpha=F(1,20))
        self.assertTrue(module.verify(certificate,records,alpha=F(1,20)))
        self.assertFalse(module.verify(certificate,list(reversed(records)),alpha=F(1,20)))
        self.assertFalse(module.verify(certificate,records,alpha=F(1,10)))
        penalized=module.certify(records,conditional_radii=[F(1,100)]*3)
        self.assertLessEqual(penalized["entropy_production_rate_lower"],certificate["entropy_production_rate_lower"])
        counts=[0,0,0];score=F(0);weighted=[F(0)]*3
        for index,row in enumerate(records):
            choice=module.policy(counts,index);a=choice["coefficients"];lam=choice["bet"]
            score+=lam*(sum(c*y for c,y in zip(a,row))-choice["support"])-lam*lam*sum(c*c for c in a)/8
            weighted=[v+lam*abs(c) for v,c in zip(weighted,a)]
            counts=[v+y for v,y in zip(counts,row)]
        self.assertEqual(score,certificate["predictable_log_score"])
        self.assertEqual(weighted,certificate["weighted_absolute_coefficients"])
        self.assertEqual(module.certify([])["entropy_production_rate_lower"],0)
        with self.assertRaises(ValueError):module.certify([[1,0]])

    def test_dissipation_module_exact_path_likelihood_and_palm_density(self):
        module=producer("dissipation_limits")
        gaps=[.1,.3,.2,.7];exposure=sum(gaps);activity=F(7,5)
        # Conditional parity law: each observed jump has rate activity; final
        # interval contributes survival without a further jump factor.
        expected=(len(gaps)-1)*math.log(float(activity))-float(activity)*exposure
        for reverse in (F(1,2),F(1,4),F(1,1000)):
            logdensity=module.parity_log_likelihood([activity]*4,reverse,gaps)
            self.assertAlmostEqual(logdensity,expected,places=13)
        rates=[F(3),F(5),F(1),F(2)];reverse=F(1,4)
        for u,v in ((.1,.2),(.8,.3),(1.5,1.1)):
            f=[float(r)*math.exp(-float(r)*u) for r in rates]
            h=[float(r)*math.exp(-float(r)*v) for r in rates]
            marked=.5*(f[0]*((1-float(reverse))*h[1]+float(reverse)*h[3])
                         +f[2]*(float(reverse)*h[1]+(1-float(reverse))*h[3]))
            factorized=(f[0]+f[2])*(h[1]+h[3])/4+(1-2*float(reverse))*(f[0]-f[2])*(h[1]-h[3])/4
            self.assertAlmostEqual(marked,factorized,places=13)
        pair=module.finite_budget_pair(rates,F(2),F(1,100),F(50))
        self.assertGreater(pair["entropy_gap_lower"],50)
        self.assertLessEqual(pair["full_record_tv_upper"],F(1,100))

    def test_fixed_time_jump_count_does_not_supply_binomial_direction_marks(self):
        # Start at state0, restrict to exactly one jump in a fixed exposure.
        # Integrating its jump time weights forward and reverse marks by the
        # destination's no-further-jump probability. Unequal exits bias them.
        departure=1.;forward_destination=8.;reverse_destination=.2
        time=1.;p=.7
        def one_jump_mass(destination):
            return departure*(math.exp(-departure*time)-math.exp(-destination*time))/(destination-departure)
        forward=p*one_jump_mass(forward_destination)
        reverse=(1-p)*one_jump_mass(reverse_destination)
        conditioned=forward/(forward+reverse)
        self.assertGreater(abs(conditioned-p),.3)

    def test_common_block_dependent_record_mgf_by_transfer_matrix(self):
        q=np.array([[-.4,.4],[.7,-.7]]);pi=np.array([7/11,4/11])
        detector=np.array([-.8,.7]);coefficients=np.array([.7,-.2,.9,.4,-.6,.3])
        for block in (.1,.7,2):
            transition=expm(q*block)
            rho=math.exp(-1.1*block)
            inflation=(math.sqrt((1+rho)/(1-rho))+1)**2
            agreement=(1+np.outer(detector,detector))/2
            mean=float(np.sum(pi[:,None]*transition*agreement))
            for theta in (-3.,-1.,.4,2.):
                state=pi.copy()
                for weight in coefficients:
                    factor=(1-agreement)+agreement*np.exp(theta*weight)
                    state=state@(transition*factor)*math.exp(-theta*weight*mean)
                exact_mgf=float(state.sum())
                bound=math.exp(theta**2*inflation*float(coefficients@coefficients)/8)
                self.assertLessEqual(exact_mgf,bound*(1+1e-13))

    def test_stationary_dependence_can_violate_iid_hoeffding_mgf(self):
        stay=.995;transition=np.array([[stay,1-stay],[1-stay,stay]])
        state=np.array([.5,.5]);theta=1.;records=10
        factor=np.exp(theta*(np.array([0.,1.])-.5))
        for _ in range(records):state=(state*factor)@transition
        exact_mgf=float(state.sum())
        iid_bound=math.exp(theta**2*records/8)
        self.assertGreater(exact_mgf,10*iid_bound)

    def test_dependence_certificate_proxy_and_contract_restrictions(self):
        module=producer("dependence")
        result=module.block_proxy(F(1,5),F(3,2))
        rho=math.exp(-.3);proxy=(math.sqrt((1+rho)/(1-rho))+1)**2
        self.assertGreaterEqual(float(result["whole_record_proxy_upper"])+1e-12,proxy)
        contract=dict(lags=[1,2],coefficients=[-1,1],records=[1000,1000],successes=[400,650],
                      durations=[F(1),F(2)],block_duration=F(2),gap_lower=F(1,2),
                      alpha=F(3,100),initial_tv=F(1,100),external_failure=F(1,200))
        certificate=module.certify_counts(**contract)
        self.assertEqual(certificate["total_failure_upper"],F(9,200))
        with self.assertRaises(ValueError):module.certify_counts(**dict(contract,schedule="adaptive"))
        with self.assertRaises(ValueError):module.certify_counts(**dict(contract,durations=[1,3]))
        with self.assertRaises(ValueError):module.certify_counts(**dict(contract,mean_interval=(0,0)))

    def test_sector_sampler_checks_joint_readout_and_padding_against_expm(self):
        module=producer("dependence");rng=np.random.default_rng(578899)
        kappa=1.1;rates=[.2,2.3];duration=.8;lag=.3
        generator=np.zeros((4,4))
        for sector in (0,1):
            for bit in (0,1):
                i=2*sector+bit
                generator[i,2*sector+1-bit]=rates[sector]
                generator[i,2*(1-sector)+bit]=kappa
                generator[i,i]=-rates[sector]-kappa
        observed=expm(generator*lag);padding=expm(generator*(duration-lag))
        true_success=float(observed[0,[0,2]].sum())
        true_sector=float(expm(generator*duration)[0,2:].sum())
        true_joint=float((observed[0]*np.array([1.,0,1,0]))@padding@np.array([0.,0,1,1]))
        samples=6000;totals=np.zeros(3)
        for _ in range(samples):
            success,sector=module.sector_group(rng,0,1,duration,lag,kappa,rates)
            totals+=success,sector,success*sector
        for empirical,truth in zip(totals/samples,[true_success,true_sector,true_joint]):
            self.assertLess(abs(empirical-truth),6*.5/math.sqrt(samples))

    def test_analog_calibration_is_upper_only_and_shared_encoding_is_not_a_replicate(self):
        module=producer("calibration")
        readings=[F(sign,4)+F(noise,2) for sign in (-1,1) for noise in (-1,1)]
        observed_second=sum(z*z for z in readings)/4
        latent_second=F(1,16)
        encoded_product=F(0)
        for z in readings:
            plus=(1+z)/2
            encoded_product+=(plus*plus+(1-plus)**2-2*plus*(1-plus))/4
        self.assertEqual(observed_second,F(5,16))
        self.assertEqual(encoded_product,observed_second)
        self.assertGreater(encoded_product,latent_second)
        vertices=module.analog_vertices((F(0),F(0)),observed_second)
        self.assertIn((F(0),F(0)),vertices)
        self.assertIn((F(0),observed_second),vertices)
        summary=module.analog_calibration_summary(10000,F(0),observed_second,alpha=F(1,20))
        mean_radius=float(summary["mean_radius_upper"])
        second_radius=float(summary["observed_second_moment_radius_upper"])
        total_failure=2*math.exp(-10000*mean_radius**2/2)+math.exp(-2*10000*second_radius**2)
        self.assertLessEqual(total_failure,.05+1e-14)
        self.assertEqual(summary["second_moment_interval"][0],0)

    def test_direct_analog_score_has_same_physical_mean_and_less_coin_noise(self):
        module=producer("analog")
        q=np.array([[-.4,.4],[.7,-.7]]);pi=np.array([7/11,4/11])
        # Three distinct analog outcomes at each state, independently drawn
        # conditional on the state. Integrate the complete endpoint law.
        readings=np.array([-1.,-.25,.75])
        noise=np.array([[.6,.3,.1],[.05,.35,.6]])
        g=noise@readings
        for lag in (.1,.9,2.):
            joint=np.einsum("i,ij,ia,jb->ab",pi,expm(q*lag),noise,noise)
            scores=(1+np.outer(readings,readings))/2
            mean=float(np.sum(joint*scores));second=float(np.sum(joint*scores*scores))
            expected=(1+correlation(q,pi,g,lag))/2
            self.assertAlmostEqual(mean,expected,places=14)
            raw_variance=second-mean*mean
            coin_variance=mean*(1-mean)
            self.assertLess(raw_variance,coin_variance)
            # The random coin is a common observation kernel. Conditional
            # Jensen implies this MGF dominance for every tested real tilt.
            for tilt in (-4.,-1.,.7,3.):
                direct=float(np.sum(joint*np.exp(tilt*(scores-mean))))
                coin=(1-mean)*math.exp(-tilt*mean)+mean*math.exp(tilt*(1-mean))
                self.assertLessEqual(direct,coin+1e-14)
            check=module.conditional_coin_variance([F(mean)],[F(second)],[F(-3,2)],[73])
            self.assertAlmostEqual(float(check["direct_contrast_variance"]),2.25*raw_variance/73)
            self.assertAlmostEqual(float(check["coin_contrast_variance"]),2.25*coin_variance/73)

    def test_analog_quantization_encloses_stored_digital_pair_mean(self):
        module=producer("analog")
        values=[-1.,np.nextafter(-1.,0.),-.1,0.,np.nextafter(.5,1.),.9,1.]
        left=values*len(values);right=[x for x in values for _ in values]
        truth=sum((1+F(float(x))*F(float(y)))/2 for x,y in zip(left,right))/len(left)
        for denominator in (1,3,1000,2**24):
            summary=module.quantized_pair_summary(left,right,denominator)
            self.assertLessEqual(abs(summary["mean"]-truth),summary["mean_quantization_radius"])
            self.assertEqual(summary["score_sum"],summary["records"]*summary["mean"])
        with self.assertRaises(ValueError):module.quantized_pair_summary([0],[0,1])
        with self.assertRaises(ValueError):module.quantized_pair_summary([np.nan],[0])

    def test_sparse_padding_record_mgf_uses_complete_block_transition(self):
        module=producer("dependence")
        q=np.array([[-.3,.3],[.8,-.8]]);pi=np.array([8/11,3/11]);g=np.array([-.7,.9])
        agreement=(1+np.outer(g,g))/2
        weights=np.array([.7,-.4,.2,-.9,.6])
        lags=np.array([.1,.7,.3,.8,.2])
        for H in (.8,2.,6.):
            proxy=module.block_proxy(F(11,10),F(str(H)))
            K=float(proxy["whole_record_proxy_upper"])
            for tilt in (-2.,-.3,1.4):
                state=pi.copy()
                for weight,lag in zip(weights,lags):
                    P=expm(q*lag);padding=expm(q*(H-lag))
                    mean=float(np.sum(pi[:,None]*P*agreement))
                    weighted=P*((1-agreement)+agreement*math.exp(tilt*weight))
                    state=state@weighted@padding*math.exp(-tilt*weight*mean)
                self.assertLessEqual(float(state.sum()),math.exp(tilt*tilt*K*float(weights@weights)/8)*(1+1e-12))

    def test_sparse_cost_optimizer_matches_independent_grid_and_charges_padding(self):
        module=producer("dependence")
        candidates=[F(2),F(3),F(7),F(12)]
        for readout in (F(0),F(5),F(100)):
            result=module.optimize_block_cost(10000,100,F(1,20),[-1,1],F(49,1000),readout,candidates)
            rows=[]
            for H in candidates:
                n=int((10000-100)//(2*(H+2*readout)))
                rho=math.exp(-float(H)/20)
                K=(math.sqrt((1+rho)/(1-rho))+1)**2
                rows.append((K*2/n,H,n))
            _,H,n=min(rows)
            self.assertEqual(result["block_duration"],H)
            self.assertEqual(result["records_per_setting"],n)
            self.assertEqual(result["physical_time"],100+2*n*H)
            self.assertEqual(result["endpoint_readouts"],4*n)
            self.assertEqual(result["normalized_cost"],result["physical_time"]+readout*4*n)
            self.assertLessEqual(result["normalized_cost"],10000)

    def test_sparse_refinement_saved_counts_match_independent_physical_oracles(self):
        root=Path(__file__).resolve().parents[1]
        parent_path=root/"contracts/dependence.json"
        contract_path=root/"contracts/dependence_refinement.json"
        report=json.loads((root/"results/dependence_refinement.json").read_text())
        self.assertEqual(report["parent_contract_sha256"],hashlib.sha256(parent_path.read_bytes()).hexdigest())
        self.assertEqual(report["contract_sha256"],hashlib.sha256(contract_path.read_bytes()).hexdigest())
        contract=json.loads(parent_path.read_text());access=contract["access_contract"]
        targets={}
        for spec in contract["models"]:
            if spec["kind"]=="ring":
                size=spec["states"];q=np.zeros((size,size));detector=np.array(spec["detector"])
                for i in range(size):
                    q[i,(i+1)%size]=float(F(spec["forward_rate"]))
                    q[i,(i-1)%size]=float(F(spec["backward_rate"]))
            else:
                size=4;q=np.zeros((size,size));detector=np.array([0,1,0,1])
                for sector in (0,1):
                    for bit in (0,1):
                        i=2*sector+bit
                        q[i,2*(1-sector)+bit]=float(F(spec["sector_flip_rate"]))
                        q[i,2*sector+1-bit]=float(F(spec["bit_flip_rates"][sector]))
            q-=np.diag(q.sum(axis=1));pi=np.ones(size)/size
            same=(detector[:,None]==detector[None,:])
            probabilities=[float(np.sum(pi[:,None]*expm(q*float(F(t)))*same)) for t in access["readout_lags"]]
            targets[spec["name"]]=probabilities[1]-probabilities[0]
        lam=float(F(access["calibrated_additive_reversibilization_gap_lower"]))
        alpha=float(F(access["new_statistical_failure"]));burn=float(F(access["new_initial_blind_gap"]))
        for row in report["evaluations"]:
            counts=np.asarray(row["raw_counts"],dtype='<i8')
            self.assertEqual(row["counts_sha256"],hashlib.sha256(counts.tobytes()).hexdigest())
            H=row["block_duration"];n=int((row["budget"]-burn)//(2*H))
            rho=math.exp(-lam*H);K=(math.sqrt((1+rho)/(1-rho))+1)**2
            width=math.sqrt(K*math.log(1/alpha)/n)
            self.assertAlmostEqual(row["halfwidth"],width,places=10)
            self.assertAlmostEqual(row["true_contrast"],targets[row["model"]],places=12)
            lower=(counts[:,1]-counts[:,0])/n-width
            self.assertEqual(row["equilibrium_exclusion"]["events"],int(np.sum(lower>0)))
            self.assertEqual(row["one_sided_coverage"]["events"],int(np.sum(lower<=targets[row["model"]]+1e-12)))

    def test_fixed_jump_entropy_interval_encloses_independent_beta_quantiles(self):
        module=producer("dissipation_limits")
        from scipy.stats import beta
        rates=[F(7,10),F(6,5),F(8,5),F(21,10)]
        activity=4/sum(1/float(r) for r in rates)
        for n,k in ((13,0),(13,1),(13,5),(100,12),(1000,10)):
            result=module.direction_count_interval(rates,n,k,F(0),bits=24)
            lo=0 if k==0 else float(beta.ppf(.025,k,n-k+1))
            hi=min(.5,1 if k==n else float(beta.ppf(.975,k+1,n-k)))
            self.assertLessEqual(float(result["reverse_interval"][0]),lo+1e-12)
            self.assertGreaterEqual(float(result["reverse_interval"][1]),hi-1e-12)
            truth_low=activity*(1-2*hi)*math.log((1-hi)/hi)
            self.assertLessEqual(float(result["entropy_production_interval"][0]),truth_low+1e-12)
            if not lo:self.assertIsNone(result["entropy_production_interval"][1])
            else:
                truth_high=activity*(1-2*lo)*math.log((1-lo)/lo)
                self.assertGreaterEqual(float(result["entropy_production_interval"][1]),truth_high-1e-12)

    def test_frozen_sequential_bank_hashes_stopping_and_costs_independent_replay(self):
        root=Path(__file__).resolve().parents[1]
        path=root/"results/sequential_records.npz"
        report=json.loads((root/"results/sequential_benchmark.json").read_text())
        contract_path=root/"contracts/sequential_benchmark.json"
        contract=json.loads(contract_path.read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),report["raw_bank_sha256"])
        self.assertEqual(hashlib.sha256(contract_path.read_bytes()).hexdigest(),report["contract_sha256"])
        alpha=float(F(contract["alpha"]));threshold=math.log(1/alpha)
        grid=contract["grid"];bet_grid=contract["bet_grid"]
        with np.load(path,allow_pickle=False) as bank:
            for case in report["cases"]:
                name=case["case"];bits=bank[name+"__bits"]
                rep,horizon,_=bits.shape
                for mode in ("learned","fixed"):
                    counts=np.zeros((rep,3));score=np.zeros(rep);first=np.full(rep,horizon+1)
                    for index in range(horizon):
                        s=np.full(rep,2/3)
                        if mode=="learned" and index:
                            centered=2*counts/index-1
                            valid=(centered[:,0]>0)&(centered[:,1]>0)
                            s[:]=0
                            s[valid]=np.minimum(1,np.floor(grid*centered[valid,1]/centered[valid,0]+.5+1e-10)/grid)
                        a=np.column_stack((-s*s,2*s,-np.ones(rep)))
                        support=-(1-s)**2/2;norm=np.sum(a*a,axis=1)
                        smoothed=(counts+1)/(index+2)
                        estimated_gap=np.maximum(0,np.sum(a*smoothed,axis=1)-support)
                        bet=np.minimum(1,np.floor(bet_grid*4*estimated_gap/norm+.5+1e-10)/bet_grid)
                        score+=bet*(np.sum(a*bits[:,index],axis=1)-support)-bet*bet*norm/8
                        first[(first==horizon+1)&(score>threshold)]=index+1
                        counts+=bits[:,index]
                    np.testing.assert_array_equal(first,bank[name+"__"+mode+"__first_crossing"])
                    saved=case["methods"][mode];capped=np.minimum(first,horizon)
                    self.assertEqual(saved["detections"],int(np.sum(first<=horizon)))
                    self.assertAlmostEqual(saved["mean_capped_pairs"],3*capped.mean())
                    self.assertAlmostEqual(saved["mean_capped_endpoint_readouts"],6*capped.mean())
                    self.assertAlmostEqual(saved["mean_capped_evolution_time"],float(sum(map(F,contract["lags"])))*capped.mean())
                a=np.array([-4/9,4/3,-1.]);support=-1/18
                radius=math.sqrt(math.log(1/alpha)*float(a@a)/(2*horizon))
                detected=np.sum(bits.mean(axis=1)@a>support+radius)
                self.assertEqual(int(detected),case["methods"]["fixed_horizon"]["detections"])

    def test_empirical_loader_checks_hashes_and_remains_descriptive(self):
        module=producer("empirical_loader")
        root=Path(__file__).resolve().parents[1];directory=root/"data/source_audit"
        metadata=json.loads((directory/"zenodo_feedbacktrap.json").read_text())
        published={x["key"]:x for x in metadata["files"]}
        import tempfile
        import h5py
        for name,digest in module.FILES.items():
            raw=(directory/name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)
            self.assertEqual("md5:"+hashlib.md5(raw).hexdigest(),published[name]["checksum"])
            self.assertEqual(len(raw),published[name]["size"])
            measured,meta=module.load_measurement(directory,name)
            with h5py.File(directory/name,"r") as source:
                independent=np.asarray(source["x_data" if "groundtruth" in name else "data"][()])
            np.testing.assert_array_equal(measured,independent)
            self.assertEqual(measured.size,49999)
            self.assertFalse(meta["actual_timestamps_present"])
        with tempfile.TemporaryDirectory() as temp:
            name="harmwell_n1.h5";raw=(directory/name).read_bytes()
            (Path(temp)/name).write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                module.load_measurement(Path(temp),name)
        audit=module.audit(directory)
        self.assertFalse(audit["simulation"]);self.assertFalse(audit["new_measurement"])
        self.assertEqual(audit["certificate_eligibility"]["decision"],"abstain")

    def test_empirical_cross_detector_decomposition_and_no_entropy_claim(self):
        module=producer("empirical_analysis");loader=producer("empirical_loader")
        root=Path(__file__).resolve().parents[1];directory=root/"data/source_audit"
        x=loader.load_measurement(directory,"harmwell_n1.h5")[0][10000:]
        y=loader.load_measurement(directory,"harmwell_n1_groundtruth.h5")[0][10000:]
        rows=module.detector_rows(x,y)
        for row in rows:
            lag=row["lag_samples"];a=x-x.mean();b=y-y.mean()
            left=slice(None,-lag) if lag else slice(None)
            right=slice(lag,None) if lag else slice(None)
            residual=float(np.mean((a-b)[left]*(a-b)[right]))
            self.assertAlmostEqual(row["residual_covariance_nm2"],residual,places=11)
            expanded=row["auto_n1_nm2"]+row["auto_reference_nm2"]-row["cross_n1_to_reference_nm2"]-row["cross_reference_to_n1_nm2"]
            self.assertAlmostEqual(expanded,residual,places=11)
        report=json.loads((root/"results/empirical_analysis.json").read_text())
        self.assertEqual(report["evaluation_samples"],39999)
        self.assertEqual(sum(report["eight_block_lengths"]),39999)
        self.assertIsNone(report["thermodynamic_certificate"]["entropy_rate_lower_bound"])
        self.assertIsNone(report["thermodynamic_certificate"]["alpha"])

    def test_analog_frozen_benchmark_replays_from_exact_retained_statistics(self):
        root=Path(__file__).resolve().parents[1]
        path=root/"contracts/analog_benchmark.json";design=json.loads(path.read_text())
        report=json.loads((root/"results/analog_benchmark.json").read_text())
        self.assertEqual(report["design_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        N=design["science_records_per_lag"];Nc=design["independent_calibration_records"]
        alpha=float(F(design["confidence_science"]));calalpha=float(F(design["confidence_calibration"]))
        radius=math.sqrt(math.log(1/alpha)/N)
        mean_radius=math.sqrt(2*math.log(4/calalpha)/Nc)
        second_radius=math.sqrt(math.log(2/calalpha)/(2*Nc))
        denominator=sum(math.sqrt(float(F(t))) for t in design["science_lags"])**2
        by_case={}
        for row in report["replicate_results"]:
            exact=row["exact_statistics"];den=exact["calibration_denominator"]
            m=exact["calibration_sum"]/(Nc*den)
            upper=min(1,exact["calibration_sum_squares"]/(Nc*den*den)+second_radius)
            min_mean_square=max(0,abs(m)-mean_radius)**2
            support=(upper-min_mean_square)/8
            direct=[.5+x["product_sum"]/(2*N*x["product_denominator"]) for x in exact["science"]]
            coin=[x["encoded_successes"]/N for x in exact["science"]]
            self.assertAlmostEqual(row["calibration_B_upper"],upper,places=12)
            self.assertAlmostEqual(row["support_upper"],support,places=12)
            self.assertAlmostEqual(row["sampling_radius"],radius,places=12)
            for label,means in (("direct",direct),("coin",coin)):
                contrast=means[0]-means[1];gap=max(0,contrast-support-radius)
                self.assertAlmostEqual(row[label+"_contrast"],contrast,places=12)
                self.assertEqual(row[label+"_reject"],gap>0)
                self.assertAlmostEqual(row[label+"_rate_lower"],8*gap*gap/denominator,places=11)
            by_case.setdefault(row["case"],[]).append(row)
        for summary in report["summaries"]:
            rows=by_case[summary["case"]]
            self.assertEqual(len(rows),design["replicates_per_case"])
            for label in ("direct","coin"):
                self.assertEqual(summary[label+"_rejections"],sum(x[label+"_reject"] for x in rows))
                mse=np.mean([(x[label+"_contrast"]-x["target_contrast"])**2 for x in rows])
                self.assertAlmostEqual(summary[label+"_contrast_mse"],mse,places=15)
            self.assertEqual(summary["detector_readouts_per_replicate"],4*N+Nc)
            self.assertEqual(F(summary["science_evolution_time_per_replicate"]),N*sum(map(F,design["science_lags"])))
        for raw in report["first_raw_bank_per_case"]:
            exact=by_case[raw["case"]][0]["exact_statistics"]
            self.assertEqual(exact["calibration_sum"],sum(raw["calibration_readings"]))
            self.assertEqual(exact["calibration_sum_squares"],sum(x*x for x in raw["calibration_readings"]))
            for pair,summary in zip(raw["science"],exact["science"]):
                self.assertEqual(summary["product_sum"],sum(x*y for x,y in zip(pair["left"],pair["right"],strict=True)))
                self.assertEqual(summary["product_denominator"],pair["left_denominator"]*pair["right_denominator"])
                self.assertEqual(summary["encoded_successes"],sum(pair["encoded_bits"]))

    def test_joint_zero_lag_has_no_physical_divergence_but_has_sampling_noise(self):
        size=8;angle=2*math.pi/size
        activity=.2/(1-math.cos(angle));difference=.4/math.sin(angle)
        forward=(activity+difference)/2;reverse=(activity-difference)/2
        q=np.zeros((size,size))
        for i in range(size):
            q[i,(i+1)%size]=forward;q[i,(i-1)%size]=reverse;q[i,i]=-activity
        pi=np.ones(size)/size;g=np.cos(angle*np.arange(size));reversible=(q+q.T)/2
        p=[(1+correlation(q,pi,g,t))/2 for t in (0,1,2)]
        ref=[(1+correlation(reversible,pi,g,t))/2 for t in (0,1,2)]
        self.assertEqual(p[0],ref[0])
        self.assertGreater(p[0]*(1-p[0]),0)
        s=.75;a=np.array([-s*s,2*s,-1]);support=-(1-s)**2/2
        gap=float(a@p)-support
        self.assertGreater(gap,0)
        sigma=(forward-reverse)*math.log(forward/reverse)
        # The positive-lag terms alone bound the physical contrast difference.
        duration_denominator=(2*s+math.sqrt(2))**2
        self.assertLessEqual(8*gap*gap/duration_denominator,sigma)
        self.assertAlmostEqual(float(a@(np.array(p)-ref)),float(a[1:]@(np.array(p[1:])-ref[1:])),places=14)
        # At an equilibrium support point, an uncertain lag0 result still
        # carries sampling noise even if the positive-lag means are known.
        s=F(1,2);a=[-s*s,2*s,F(-1)];null=[F(3,4),F(5,8),F(9,16)]
        support=-(1-s)**2/2
        self.assertEqual(sum(c*p for c,p in zip(a,null)),support)
        wrong_rejection_probability=F(0)
        complete_rejection_probability=F(0)
        radius=abs(float(a[0]))*math.sqrt(math.log(20)/2)
        for y in (0,1):
            mass=null[0] if y else 1-null[0]
            observed=a[0]*y+a[1]*null[1]+a[2]*null[2]
            if observed>support:wrong_rejection_probability+=mass
            if float(observed-support)>radius:complete_rejection_probability+=mass
        self.assertEqual(wrong_rejection_probability,F(1,4))
        self.assertLessEqual(complete_rejection_probability,F(1,20))

    def test_calibrated_frozen_bank_and_dual_witnesses_independent_reconstruction(self):
        root=Path(__file__).resolve().parents[1]
        contract_path=root/"contracts/calibrated_benchmark.json"
        contract=json.loads(contract_path.read_text())
        report=json.loads((root/"results/calibrated_benchmark.json").read_text())
        self.assertEqual(report["contract_sha256"],hashlib.sha256(contract_path.read_bytes()).hexdigest())
        for kind in ("counts","raw"):
            path=root/"results"/report[kind+"_archive"]
            self.assertEqual(report[kind+"_archive_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        # Independent low-degree Bernstein evaluation via derivatives on
        # each dyadic interval, not the producer's recursive polynomial code.
        def polynomial_upper(coefficients,depth):
            p=np.polynomial.Polynomial(coefficients);degree=len(coefficients)-1
            step=2.**(-depth);upper=-math.inf
            for left in np.arange(2**depth)*step:
                local=[float(p.deriv(k)(left))*step**k/math.factorial(k) for k in range(degree+1)]
                for j in range(degree+1):
                    upper=max(upper,sum(math.comb(j,k)/math.comb(degree,k)*local[k] for k in range(j+1)))
            return upper
        calibrated_shape=polynomial_upper([0,1.52,-1],7)
        a=np.array([1.52,-1]);n=np.array(contract["science_records"][:2])
        radius=math.sqrt(math.log(40)*float(np.sum(a*a/n))/2)
        cal_radius=math.sqrt(2*math.log(160)/contract["calibration_records"])
        denominator=(1.52+math.sqrt(2))**2
        with np.load(root/"results"/report["counts_archive"],allow_pickle=False) as bank:
            for case in report["cases"]:
                name=case["case"];science=bank[name+"__science_counts"];twins=bank[name+"__calibration_counts"]
                self.assertTrue(np.all(twins.sum(axis=1)==contract["calibration_records"]))
                mhat=(twins[:,0]-twins[:,3])/contract["calibration_records"]
                Ahat=(twins[:,0]+twins[:,3]-twins[:,1]-twins[:,2])/contract["calibration_records"]
                umin=np.maximum(0,np.abs(mhat)-cal_radius)**2
                Amax=np.minimum(1,Ahat+cal_radius)
                support=.26*(1+umin)+(Amax-umin)*calibrated_shape/2
                gap=np.maximum(0,science[:,:2]/n@a-radius-support)
                expected=8*gap*gap/denominator
                np.testing.assert_allclose(bank[name+"__calibrated_pinsker_rate"],expected,atol=1e-12,rtol=1e-9)
                for method,s in (("uncalibrated_extra_lag",.34),("legacy_extra_lag",2/3)):
                    coeff=np.array([-s*s,2*s,-1]);N=np.array(contract["science_records"])
                    bound=max(polynomial_upper([-(1-s)**2/2,-s*s/2,s,-.5],6),-(1-s)**2)
                    r=math.sqrt(math.log(20)*float(np.sum(coeff*coeff/N))/2)
                    d=sum(abs(c)*math.sqrt(t) for c,t in zip(coeff,(1,2,3)))**2
                    lower=np.maximum(0,science/N@coeff-bound-r)
                    np.testing.assert_allclose(bank[name+"__"+method+"_rate"],8*lower*lower/d,atol=1e-12,rtol=1e-9)
                for method,summary in case["inference"].items():
                    values=bank[name+"__"+method+"_rate"]
                    self.assertEqual(summary["detections"],int(np.sum(values>0)))
                    self.assertAlmostEqual(case["rate_statistics"][method]["mean"],float(values.mean()),places=14)
                # Saved representative nonlinear scores are evaluated with an
                # independent HiGHS minimizer and exact quadratic extrema.
                for example in case["illustrative_exact_certificates"]:
                    cert=example["nonlinear"]["certificate"]
                    h=np.array([[float(F(x)) for x in pair] for pair in cert["scores"]])
                    weights=np.array([float(F(x)) for x in cert["weights"]])
                    coefficient=np.array([float(F(x)) for x in cert["coefficients"]])
                    lower=float(F(cert["lower"]));bounds=[tuple(float(F(x)) for x in pair) for pair in cert["boxes"]]
                    objective=weights*np.log(h[:,1]/h[:,0])
                    result=linprog(objective,A_ub=[-coefficient],b_ub=[-lower],bounds=bounds,method="highs")
                    self.assertTrue(result.success)
                    minimum=float(weights@np.log(h[:,0])+result.fun)
                    normalizer=0.
                    for u,A in cert["vertices"]:
                        u,A=float(F(u)),float(F(A))
                        p=np.polynomial.Polynomial([float(weights@(h[:,0]+(h[:,1]-h[:,0])*(1+u)/2)),
                            weights[0]*(h[0,1]-h[0,0])*(A-u)/2,
                            weights[1]*(h[1,1]-h[1,0])*(A-u)/2])
                        points=[0.,1.]+[float(x.real) for x in p.deriv().roots() if abs(x.imag)<1e-12 and 0<x.real<1]
                        normalizer=max(normalizer,max(float(p(x)) for x in points))
                    enclosed=float(F(cert["normalizer"]["normalizer_upper"]))
                    self.assertGreaterEqual(enclosed+1e-12,normalizer)
                    self.assertLessEqual(float(F(cert["log_expectation_lower"])),minimum+1e-12)
                    self.assertLessEqual(float(F(cert["record_kl_lower"])),max(0,minimum-math.log(enclosed))+1e-11)
            audit=json.loads((root/"results/calibrated_benchmark_audit.json").read_text())
            self.assertEqual(audit["original_contract_sha256"],report["contract_sha256"])
            self.assertEqual(audit["original_counts_sha256"],report["counts_archive_sha256"])
            self.assertEqual(len(audit["records"]),len(contract["cases"])*contract["datasets_per_case"])
            safe_by_case={case["name"]:[] for case in contract["cases"]}
            for row in audit["records"]:
                name=contract["cases"][row["case_index"]]["name"];idx=row["dataset"]
                self.assertEqual(row["status"],"analyzed")
                linear=F(row["pinsker_rate"]);nonlinear=F(row["raw_nonlinear_rate"])
                safe=F(row["safe_same_event_maximum"])
                self.assertEqual(safe,max(linear,nonlinear))
                self.assertEqual(float(linear),float(bank[name+"__calibrated_pinsker_rate"][idx]))
                self.assertEqual(float(nonlinear),float(bank[name+"__calibrated_nonlinear_rate"][idx]))
                safe_by_case[name].append(float(safe))
            for row in audit["cases"]:
                safe=safe_by_case[row["case"]]
                self.assertEqual(row["safe_maximum_detection"]["detections"],sum(x>0 for x in safe))
                self.assertAlmostEqual(row["safe_maximum_mean"],float(np.mean(safe)),places=14)
            with np.load(root/"results"/report["raw_archive"],allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["same"],raw["first"]==raw["second"])
                np.testing.assert_array_equal(raw["final"],(raw["initial"]+raw["forward_jumps"]-raw["backward_jumps"])%20)
                for setting in range(3):
                    mask=raw["setting"]==setting
                    self.assertEqual(int(raw["same"][mask].sum()),int(bank["driven__science_counts"][0,setting]))
                mask=raw["setting"]==3;first=raw["first"][mask];second=raw["second"][mask]
                observed=[int(np.sum((first==x)&(second==y))) for x,y in ((1,1),(1,0),(0,1),(0,0))]
                np.testing.assert_array_equal(observed,bank["driven__calibration_counts"][0])

    def test_joint_frozen_comparison_replays_all_counts_and_physical_costs(self):
        root=Path(__file__).resolve().parents[1]
        contract_path=root/"contracts/joint_calibration.json";contract=json.loads(contract_path.read_text())
        report=json.loads((root/"results/joint_calibration.json").read_text())
        self.assertEqual(report["contract_sha256"],hashlib.sha256(contract_path.read_bytes()).hexdigest())
        self.assertEqual(report["baseline_contract_sha256"],hashlib.sha256((root/"contracts/calibrated_benchmark.json").read_bytes()).hexdigest())
        for kind in ("counts","raw"):
            path=root/"results"/report[kind+"_archive"]
            self.assertEqual(report[kind+"_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        joint=contract["joint"];s=float(F(joint["parameter_s"]));a=np.array([-s*s,2*s,-1.])
        n=np.array(joint["records"]);alpha=float(F(joint["alpha"]))
        support=-(1-s)**2/2;radius=math.sqrt(math.log(1/alpha)*float(np.sum(a*a/n))/2)
        denominator=(2*s+math.sqrt(2))**2
        gamma=-math.log(math.sqrt(497)/25);omega=math.acos(19/math.sqrt(497))
        with np.load(root/"results"/report["counts_archive"],allow_pickle=False) as bank:
            for case,spec in zip(report["cases"],contract["cases"],strict=True):
                name=case["case"];self.assertEqual(name,spec["name"])
                science=bank[name+"__science_counts"];twins=bank[name+"__calibration_counts"]
                counts=np.column_stack((twins[:,0]+twins[:,3],science[:,:2]))
                self.assertTrue(np.all(twins.sum(axis=1)==n[0]))
                lower=counts/n@a-radius;gap=np.maximum(0,lower-support)
                expected=8*gap*gap/denominator
                np.testing.assert_allclose(bank[name+"__joint_calibration_pinsker_rates"],expected,atol=1e-12,rtol=1e-9)
                gain=spec["gain"];rotation=omega if spec["rates"]=="driven" else 0
                probabilities=np.array([.5+.25*gain*gain*math.exp(-gamma*k)*math.cos(rotation*k) for k in (0,1,2)])
                self.assertAlmostEqual(case["target_zero_lag_probability"],probabilities[0],places=14)
                self.assertEqual(case["joint_contrast_coverage"]["detections"],int(np.sum(lower<=float(a@probabilities))))
                for method,summary in case["inference"].items():
                    values=bank[name+"__"+method+"_rates"]
                    self.assertEqual(summary["detections"],int(np.sum(values>0)))
                    self.assertAlmostEqual(case["rate_statistics"][method]["mean"],float(np.mean(values)),places=14)
                for method,summary in case["joint_minus_baseline"].items():
                    diff=expected-bank[name+"__"+method+"_rates"]
                    self.assertAlmostEqual(summary["mean"],float(np.mean(diff)),places=13)
                    self.assertEqual(summary["strictly_positive"],int(np.sum(diff>1e-15)))
                    self.assertEqual(summary["strictly_negative"],int(np.sum(diff < -1e-15)))
            with np.load(root/"results"/report["raw_archive"],allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["same"],raw["first"]==raw["second"])
                np.testing.assert_array_equal(raw["final"],(raw["initial"]+raw["forward_jumps"]-raw["backward_jumps"])%20)
                self.assertEqual(raw["same"].size,contract["costs"]["all_menus_simulated_for_paired_comparison"]["pairs"])
                self.assertEqual(float(raw["elapsed"].sum()),contract["costs"]["all_menus_simulated_for_paired_comparison"]["declared_evolution_time"])
                for setting in range(3):
                    self.assertEqual(int(raw["same"][raw["setting"]==setting].sum()),int(bank["driven__science_counts"][0,setting]))
                mask=raw["setting"]==3;first=raw["first"][mask];second=raw["second"][mask]
                observed=[int(np.sum((first==x)&(second==y))) for x,y in ((1,1),(1,0),(0,1),(0,0))]
                np.testing.assert_array_equal(observed,bank["driven__calibration_counts"][0])
        cost=contract["costs"]["joint_and_two_stage"]
        self.assertEqual(cost["pairs"],sum(joint["records"]))
        self.assertEqual(cost["endpoint_readouts"],2*sum(joint["records"]))
        self.assertEqual(cost["declared_evolution_time"],joint["records"][1]+2*joint["records"][2])
        module=producer("joint_calibration")
        certificate=module.certify_counts([4000,8000,8000],[3000,5520,4720],parameter=F(19,25))
        self.assertAlmostEqual(float(certificate["comparison_time_denominator_upper"]),denominator,places=12)
        self.assertAlmostEqual(float(certificate["entropy_production_rate_lower"]),8*(.0544-radius)**2/denominator,places=12)
        with self.assertRaises(ValueError):module.certify_counts([4000,8000,8000],[3000,5520,4720],twin_contract="reused_noisy_bit")

    def test_diffusion_transfer_matches_gaussian_grid_kl_and_bounded_quadrature(self):
        module=producer("nonlinear");example=module.diffusion_example()
        nodes,weights=np.polynomial.hermite.hermgauss(40)
        x=math.sqrt(2)*nodes[:,None];noise=math.sqrt(2)*nodes[None,:]
        for name,omega in (("driven",2.),("equilibrium",0.)):
            rho=math.exp(-1)*math.cos(omega)
            y=rho*x+math.sqrt(1-rho*rho)*noise
            quadrature=float(np.sum(weights[:,None]*weights[None,:]*np.sin(x)*np.sin(y))/math.pi)
            analytic=math.exp(-1)*math.sinh(rho)
            self.assertAlmostEqual(quadrature,analytic,places=13)
            lower,upper=map(float,example[name]["detector_correlation_interval"])
            self.assertLessEqual(lower,quadrature+1e-14)
            self.assertGreaterEqual(upper,quadrature-1e-14)
            rlo,rhi=map(float,example[name]["underlying_gaussian_correlation_interval"])
            self.assertLessEqual(rlo,rho);self.assertGreaterEqual(rhi,rho)
        second=(1-math.exp(-2))/2
        lo,hi=map(float,example["second_moment_interval"])
        self.assertLessEqual(lo,second);self.assertGreaterEqual(hi,second)
        # A completely different route: exact4D Gaussian KL for one adjacent
        # stationary state pair, summed over a Markov observation grid.
        # Refining the grid recovers the full continuous-path Girsanov limit.
        omega=2.;duration=1.;sigma=2*omega*omega;previous=-1
        for intervals in (1,2,4,8,16,32,64,128,256):
            step=duration/intervals;angle=omega*step
            rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            transition=math.exp(-step)*rotation;reference=math.exp(-step)*np.eye(2)
            forward_covariance=np.block([[np.eye(2),transition.T],[transition,np.eye(2)]])
            reverse_covariance=np.block([[np.eye(2),reference.T],[reference,np.eye(2)]])
            gaussian_kl=.5*(np.trace(np.linalg.solve(reverse_covariance,forward_covariance))-4+
                np.linalg.slogdet(reverse_covariance)[1]-np.linalg.slogdet(forward_covariance)[1])
            grid_kl=intervals*gaussian_kl
            self.assertGreaterEqual(grid_kl,previous-1e-10)
            self.assertLessEqual(grid_kl,duration*sigma/4+1e-10)
            previous=grid_kl
        self.assertLess(duration*sigma/4-previous,.01)
        self.assertEqual(example["driven"]["exact_stationary_entropy_rate"],F(8))
        self.assertEqual(example["driven"]["exact_full_path_kl_to_reversible_reference"],F(2))
        p=(1+math.exp(-1)*math.sinh(math.exp(-1)*math.cos(2)))/2
        bernoulli_relaxation=p*math.log(2*p)+(1-p)*math.log(2*(1-p))
        certificate=example["driven"]["certificate"]
        certified=float(certificate["record_kl_lower"])
        self.assertGreater(certified,0)
        self.assertLessEqual(certified,bernoulli_relaxation+1e-12)
        self.assertAlmostEqual(certified,bernoulli_relaxation,places=10)
        self.assertLessEqual(float(certificate["entropy_production_rate_lower"]),sigma)
        self.assertEqual(example["equilibrium"]["certificate"]["entropy_production_rate_lower"],0)


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentCalibrationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={"scope":"independent mathematics, simulated benchmark replay, and descriptive integrity checks of previously published measurements; no empirical entropy certificate",
            "tests_run":result.testsRun,"failures":len(result.failures),
            "errors":len(result.errors),"passed":result.wasSuccessful(),
            "reproduction":"PYTHONPATH=src python calibrated/tests/test_independent.py",
            "reproduction_workdir":"release root",
            "external_peer_review":False,
            "float_precision":"NumPy float64 for independent numerical checks; Fraction for certificate comparisons",
            "python_version":sys.version.split()[0],"numpy_version":np.__version__}
    path=Path(__file__).resolve().parents[1]/"results"/"independent.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    sys.exit(not result.wasSuccessful())
