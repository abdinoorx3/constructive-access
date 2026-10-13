"""Independent virtual-laboratory mathematics and adversarial controls."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util
import json
import math
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import binom


def stationary(q):
    matrix=q.T.copy();matrix[-1]=1
    rhs=np.zeros(len(q));rhs[-1]=1
    return np.linalg.solve(matrix,rhs)


def producer(name):
    path=Path(__file__).resolve().parents[1]/"code"/(name+".py")
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("virtual_review_"+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class IndependentLaboratoryTests(unittest.TestCase):
    def test_common_iid_increments_give_one_channel(self):
        q=np.array([[-3.,2,1],[1,-2,1],[2,1,-3.]])
        steps=[.1,.3];weights=[.4,.6]
        average=sum(w*expm(q*t) for t,w in zip(steps,weights))
        for k in range(1,7):
            direct=sum(binom.pmf(j,k,.6)*expm(q*(.1*(k-j)+.3*j))
                       for j in range(k+1))
            self.assertTrue(np.allclose(direct,np.linalg.matrix_power(average,k)))

    def test_random_scale_preserves_reversible_moments_but_inflates_linear_dimension(self):
        import sympy as sp
        # A two-state reversible trace, clock scale drawn once per entire wait.
        p=[F(1,2)+F(1,4)*F(1,2)**k+F(1,4)*F(1,4)**k for k in range(5)]
        hankel=sp.Matrix([[sp.Rational(p[i+j].numerator,p[i+j].denominator)
                           for j in range(3)] for i in range(3)])
        self.assertGreater(hankel.det(),0)
        c=np.array([float(2*x-1) for x in p])
        for order in range(4):
            self.assertGreaterEqual(min((-1)**order*np.diff(c,n=order)),-1e-14)
        iid=[F(1,2)+F(1,2)*F(3,8)**k for k in range(5)]
        iid_hankel=sp.Matrix([[sp.Rational(iid[i+j].numerator,iid[i+j].denominator)
                               for j in range(3)] for i in range(3)])
        self.assertEqual(iid_hankel.det(),0)

    def test_lag_specific_equal_mean_jitter_fakes_equilibrium_revival(self):
        # Equilibrium symmetric two-state correlation exp(-t). Mean times1,2.
        first=math.exp(-1)
        second=(math.exp(-.1)+math.exp(-3.9))/2
        self.assertAlmostEqual((.1+3.9)/2,2)
        self.assertGreater(second,first)
        # A false revival conclusion results if this clock family is untreated.
        self.assertGreater((1+second)/2,(1+first)/2)

    def test_identical_symmetric_exposure_blocks_preserve_gap_moments(self):
        q=np.array([[-1.,1],[1,-1.]])
        pi=np.array([.5,.5]);f=np.array([1.,-1.]);length=.3
        # F=(Y_start+Y_end)/2 is bounded and invariant under block reversal.
        g=(f+expm(q*length)@f)/2
        for gap in (.1,.4,1):
            explicit=(pi@(f*(expm(q*gap)@f))
                      +2*pi@(f*(expm(q*(gap+length))@f))
                      +pi@(f*(expm(q*(gap+2*length))@f)))/4
            reduced=pi@(g*(expm(q*gap)@g))
            self.assertAlmostEqual(explicit,reduced,places=14)

    def test_center_lag_exposure_can_break_normalized_moment_cone(self):
        # Same blocks, but use center separation rather than inter-block gap.
        decay=2.;length=.4;step=.6
        amplitude=math.cosh(decay*length/2)**2
        x=math.exp(-decay*step)
        c1,c2=amplitude*x,amplitude*x*x
        self.assertGreater(amplitude,1)
        self.assertLess(c1,1)
        # Any probability moment measure on[0,1] obeys C2>=C1^2.
        self.assertLess(c2,c1*c1)

    def test_stationary_marginals_do_not_supply_conditional_mean_coverage(self):
        n=100;radius=.15
        # All outcomes equal one fair latent bit: every marginal is Bernoulli(.5).
        probability_of_coverage=sum(.5 for value in (0,1) if abs(value-.5)<=radius)
        self.assertEqual(probability_of_coverage,0)
        nominal_failure=2*math.exp(-2*n*radius*radius)
        self.assertLess(nominal_failure,.05)

    def test_uniform_tv_reset_controls_entire_pair_not_just_first_readout(self):
        q=np.array([[-2.,2],[1,-1.]])
        pi=stationary(q);gap=.7;lag=.4;f=np.array([.9,.2])
        agreement=(1-f)*(expm(q*lag)@(1-f))+f*(expm(q*lag)@f)
        true=pi@agreement
        reset=expm(q*gap)
        for row in reset:
            total_variation=np.sum(abs(row-pi))/2
            self.assertLessEqual(abs(row@agreement-true),total_variation+1e-14)

    def test_fixed_quota_conditional_mgf_bound_under_feedback(self):
        # Enumerate all adaptive histories; means vary in a declared bias band.
        settings=2;quota=3;p=[.3,.7];delta=.1;theta=.8
        states=[((0,0),(0,0),1.,0.)]
        for index in range(settings*quota):
            new=[]
            for used,success,prob,centered in states:
                preferred=int(sum(success)%2)
                arm=preferred if used[preferred]<quota else 1-preferred
                mean=p[arm]+(delta if sum(success)%2 else -delta)
                for outcome,chance in ((0,1-mean),(1,mean)):
                    use=list(used);use[arm]+=1
                    succ=list(success);succ[arm]+=outcome
                    increment=(outcome-mean) if arm==0 else 0
                    new.append((tuple(use),tuple(succ),prob*chance,centered+increment))
            states=new
        mgf=sum(prob*math.exp(theta*centered) for _,_,prob,centered in states)
        self.assertLessEqual(mgf,math.exp(quota*theta*theta/8)+1e-12)

    def test_linear_contrast_entropy_bound_on_stationary_networks(self):
        rng=np.random.default_rng(681137)
        a=np.array([-1.,2.,-1.]);times=np.array([.15,.30,.45])
        for n in range(3,10):
            for _ in range(20):
                q=rng.lognormal(0,1.5,(n,n));np.fill_diagonal(q,0)
                q-=np.diag(q.sum(axis=1));pi=stationary(q);f=rng.uniform(-1,1,n)
                p=np.array([.5+.5*pi@(f*(expm(q*t)@f)) for t in times])
                contrast=max(0.,a@p)  # reversible polynomial support is zero.
                flow=pi[:,None]*q
                sigma=sum((flow[i,j]-flow[j,i])*math.log(flow[i,j]/flow[j,i])
                          for i in range(n) for j in range(i+1,n))
                lower=8*contrast*contrast/sum(abs(a)*np.sqrt(times))**2
                self.assertLessEqual(lower,sigma+1e-10)

    def test_optimal_virtual_weights_saturate_time_information_cauchy(self):
        rng=np.random.default_rng(572138)
        for _ in range(100):
            a=rng.normal(size=5);times=rng.uniform(.01,3,5)
            w=abs(a)/np.sqrt(times);w/=sum(w)
            optimum=sum(abs(a)*np.sqrt(times))**2
            self.assertAlmostEqual(sum(w*times)*sum(a*a/w),optimum,places=11)
            random=rng.dirichlet(np.ones(5))
            self.assertGreaterEqual(sum(random*times)*sum(a*a/random)+1e-12,optimum)

    def test_linear_contrast_hoeffding_tail_exact_enumeration(self):
        # Independent fixed strata, same formula also follows from conditionalMGF.
        n1,n2=20,30;p1,p2=.2,.6;a1,a2=-1.,2.;alpha=.05
        radius=math.sqrt(.5*math.log(1/alpha)*(a1*a1/n1+a2*a2/n2))
        probability=0.
        for s1 in range(n1+1):
            for s2 in range(n2+1):
                error=a1*(s1/n1-p1)+a2*(s2/n2-p2)
                if error>=radius:
                    probability+=binom.pmf(s1,n1,p1)*binom.pmf(s2,n2,p2)
        self.assertLessEqual(probability,alpha)

    def test_contrast_exact_certificate_and_reference_mean_guard(self):
        module=producer("contrast")
        args=dict(lags=[1,2,3],coefficients=[-1,2,-1],records=[50000]*3,
                  successes=[35849,32265,27572],durations=[F(3,20),F(3,10),F(9,20)])
        cert=module.certify_counts(**args)
        self.assertTrue(module.verify(cert,**args))
        self.assertGreater(cert["entropy_production_rate_lower"],0)
        self.assertEqual(cert["resource_mixture_weights"],[F(1,4),F(1,2),F(1,4)])
        self.assertEqual(cert["record_kl_lower"],cert["contrast_lower"]**2/8)
        with self.assertRaises(ValueError):module.certify_counts(**args,mean_interval=(F(-1,10),F(1,10)))
        narrow=module.certify_counts(**args,mean_interval=(F(-1,10),F(1,10)),reference_mean_calibrated=True)
        self.assertGreaterEqual(narrow["contrast_lower"],cert["contrast_lower"])
        calibrated=module.certify_counts(**args,conditional_radii=[F(1,100)]*3)
        self.assertLessEqual(calibrated["contrast_lower"],cert["contrast_lower"])

    def test_contrast_exact_support_and_time_bound_against_numeric(self):
        module=producer("contrast")
        rng=np.random.default_rng(814493)
        for _ in range(50):
            a=[F(int(z),7) for z in rng.integers(-5,6,4)]
            if not any(a):a[0]=F(1)
            result=module.support([1,2,3,5],a,depth=3)
            bound=float(result["reversible_support_upper"])
            for r in (.5,.8,1):
                for x in np.linspace(0,1,301):
                    value=sum(float(c)*(r+(1-r)*x**k) for c,k in zip(a,[1,2,3,5]))
                    self.assertLessEqual(value,bound+1e-12)
        cert=module.example()["certificate"]
        C=float(cert["contrast_lower"])
        denominator=sum(abs(float(a))*math.sqrt(float(t)) for a,t in zip(cert["coefficients"],cert["durations"]))**2
        self.assertLessEqual(float(cert["entropy_production_rate_lower"]),8*C*C/denominator+1e-15)

    def test_timing_implementation_agrees_with_independent_controls(self):
        module=producer("timing");report=module.make_results()
        self.assertGreater(report["unsafe_random_scale_for_same_dimension"]["hankel_determinant"],0)
        self.assertLess(report["unsafe_center_spacing_index"]["normalized_second_moment_gap"],0)
        self.assertLess(report["unsafe_antisymmetric_block"]["correlation"],0)
        self.assertGreater(report["unsafe_overlapping_symmetric_blocks"]["increase"],0)
        for length in (1,2,4):
            for separation in (0,1,3,6):
                actual=module.endpoint_block_correlation(F(1,2),length,separation)
                expected=(2*.5**separation+.5**(separation+length)+.5**abs(separation-length))/4
                self.assertAlmostEqual(float(actual),expected,places=14)

    def test_ring_trajectory_sampler_against_independent_matrix_exponential(self):
        module=producer("thermodynamic_lab")
        rng=np.random.default_rng(554109);n=120000;states=10;lag=.73;forward=3.2;backward=.7
        detector=np.linspace(-.9,.8,states);gain=.72
        record=module.ring_records(rng,n,lag,forward,backward,detector,gain)
        q=np.zeros((states,states))
        for i in range(states):
            q[i,(i+1)%states]=forward;q[i,(i-1)%states]=backward;q[i,i]=-forward-backward
        g=gain*detector
        probability=.5+.5*g@expm(q*lag)@g/states
        error=abs(record["same"].mean()-probability)
        self.assertLess(error,6*math.sqrt(probability*(1-probability)/n))
        self.assertLess(abs(record["forward_jumps"].mean()-forward*lag),6*math.sqrt(forward*lag/n))
        self.assertLess(abs(record["backward_jumps"].mean()-backward*lag),6*math.sqrt(backward*lag/n))
        self.assertTrue(np.all(record["final"]==(record["initial"]+record["forward_jumps"]-record["backward_jumps"])%states))
        self.assertTrue(np.array_equal(record["same"],record["first"]==record["second"]))

    def test_prepared_equilibrium_gillespie_sampler_is_transient(self):
        module=producer("thermodynamic_lab")
        n=120000;t=.23;rng=np.random.default_rng(87135)
        record=module.two_state_records(rng,n,t,forward=9.,backward=1.,initial_state=0)
        q=np.array([[-9.,9.],[1.,-1.]])
        true=expm(q*t)[0,0]
        self.assertLess(abs(record["same"].mean()-true),6*math.sqrt(true*(1-true)/n))
        self.assertTrue(np.all(record["initial"]==0))
        self.assertLess(true,.5)
        self.assertTrue(np.all(record["forward_jumps"]-record["backward_jumps"]==record["final"]))

    def test_monte_carlo_intervals_do_not_call_zero_rejections_zero_risk(self):
        module=producer("thermodynamic_lab")
        lower,upper=module.clopper_pearson(0,200)
        self.assertEqual(lower,0)
        self.assertAlmostEqual(upper,1-.025**(1/200),places=13)
        self.assertGreater(upper,.01)
        lo,hi=module.clopper_pearson(100,200)
        self.assertLess(lo,.5);self.assertGreater(hi,.5)

    def test_dimension_channel_against_independent_rodrigues_rotation(self):
        module=producer("dimension_lab");rng=np.random.default_rng(537102)
        for _ in range(20):
            p=dict(omega=1.,amplitude_offset=rng.uniform(-.05,.05),
                   amplitude_jitter=rng.uniform(0,.04),timing_jitter=rng.uniform(0,.03),
                   detuning=rng.uniform(-.2,.2),depolarization=rng.uniform(.001,.03),
                   preparation_error=rng.uniform(0,.01),false_positive=rng.uniform(0,.01),
                   false_negative=rng.uniform(0,.02))
            rotations=[]
            for amp,clock in ((-1,-1),(-1,1),(1,-1),(1,1)):
                field=np.array([p["omega"]*(1+p["amplitude_offset"]+amp*p["amplitude_jitter"]),0,p["detuning"]])
                axis=field/np.linalg.norm(field);time=2*math.pi/7*(1+clock*p["timing_jitter"])
                angle=np.linalg.norm(field)*time
                cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                rotation=math.cos(angle)*np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*np.outer(axis,axis)
                rotations.append(math.exp(-p["depolarization"]*time)*rotation)
            channel=sum(rotations)/4;state=np.array([0,0,1-2*p["preparation_error"]]);truth=[]
            for k in range(1,14):
                state=channel@state
                truth.append(p["false_positive"]+(1-p["false_positive"]-p["false_negative"])*(1-state[2])/2)
            self.assertTrue(np.allclose(module.probability_engine(p),truth,atol=2e-13,rtol=0))
            self.assertTrue(np.allclose(module.lindblad_probabilities(p),truth,atol=2e-13,rtol=0))

    def test_dimension_raw_stream_replays_counts_and_step_memory(self):
        module=producer("dimension_lab")
        p=dict(omega=1.,amplitude_offset=.003,amplitude_jitter=.01,timing_jitter=.02,
               detuning=.02,depolarization=.002,preparation_error=.004,
               false_positive=.003,false_negative=.006)
        records=[500]*13
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"shots.npz"
            report=module.retained_shots(p,records,281355,file)
            with np.load(file,allow_pickle=False) as raw:
                reconstructed=[int(np.sum(raw["observed"][raw["setting"]==k],dtype=np.int64)) for k in range(1,14)]
                self.assertEqual(reconstructed,report["counts"])
                self.assertEqual(len(np.unique(raw["group_to_acquisition_permutation"])),sum(records))
                _,durations=module.step_channels(p)
                duration=sum(float(durations[raw[f"noise_choices_step_{k}"]].sum()) for k in range(1,14))
                self.assertAlmostEqual(duration,report["realized_evolution_duration"],places=10)
                self.assertEqual(report["channel_applications"],sum(k*n for k,n in enumerate(records,1)))
        common=module.probability_engine(p)
        static=module.quasi_static_probabilities(p)
        self.assertGreater(max(abs(common-static)),1e-4)

    def test_dimension_plan_rejects_missing_settings(self):
        module=producer("dimension_lab")
        records=[100]*13;critical=[20]*13
        plan=dict(epsilon="3/500",beta="3/20",tau="2/25",critical_counts=critical)
        with self.assertRaises(ValueError):module.certify_plan(plan,records[:-1])
        with self.assertRaises(ValueError):module.pass_probability([.3]*12,records,critical)
        with self.assertRaises(ValueError):module.classify(np.zeros(12,dtype=int),critical)
        with self.assertRaises(ValueError):module.pass_probability([.3]*13,records,critical[:-1])

    def test_sampling_conditional_box_and_mixing_envelope(self):
        module=producer("sampling")
        radius=module.confidence_radius(600,2,F(1,20))
        box=module.confidence_box([600,600],[420,400],radius,F(1,20),F(1,400),
                                  calibration_failure=F(1,100))
        self.assertEqual(box["total_failure_upper"],F(3,50))
        with self.assertRaises(ValueError):
            module.confidence_box([600,600],[420,400],radius,F(1,20),0,sampling_design="stop_when_significant")
        for rate in (F(1,10),F(1),F(3)):
            for gap in (F(0),F(1),F(5)):
                # Symmetric two-state chain: spectral gap2rate, pi_min=.5.
                result=module.spectral_mix_bound(2*rate,F(1,2),gap)
                true=.5*math.exp(-2*float(rate*gap))
                self.assertGreaterEqual(float(result["conditional_bias_upper"])+1e-14,true)
        self.assertEqual(module.doeblin_mix_bound(F(1,3),4),F(16,81))

    def test_shifted_cubic_contrast_is_hankel_witness(self):
        module=producer("contrast")
        for s in (F(0),F(1,4),F(2,3),F(1)):
            coefficients=[-s*s,2*s,F(-1)]
            exact_support=-(1-s)**2/2
            bound=module.support([1,2,3],coefficients,depth=6)["reversible_support_upper"]
            self.assertGreaterEqual(bound,exact_support)
            for x in np.linspace(0,1,501):
                q=[(1+x**k)/2 for k in (1,2,3)]
                self.assertLessEqual(sum(float(a)*p for a,p in zip(coefficients,q)),float(exact_support)+1e-13)
        correlations=np.array([.434,.2906,.1029])
        optimum=correlations[1]/correlations[0]
        contrast=(-optimum**2*correlations[0]+2*optimum*correlations[1]-correlations[2])/2
        self.assertAlmostEqual(contrast,(correlations[1]**2/correlations[0]-correlations[2])/2,places=14)
        self.assertGreater(contrast,0)

    def test_equilibrium_box_witness_has_independent_physical_realization(self):
        module=producer("contrast")
        lags=[1,2,3];decay=F(3,5);amplitude=F(4,5)
        truth=[(1+amplitude*decay**k)/2 for k in lags]
        boxes=[(p-F(1,1000),p+F(1,1000)) for p in truth]
        witness=module.reversible_box_witness(lags,boxes,grid=100)
        self.assertTrue(module.verify_box_witness(witness,lags,boxes))
        # Independently build a stationary continuous-time equilibrium process,
        # then sum the two independent detector-output agreement probabilities.
        step=.15;x=float(witness["decay_per_step"])
        rate=-math.log(x)/(2*step)
        generator=np.array([[-rate,rate],[rate,-rate]])
        plus=(1+np.array([1.,-1.])*math.sqrt(float(witness["squared_detector_mean"])))/2
        agreement=np.outer(plus,plus)+np.outer(1-plus,1-plus)
        for k,p,(lo,hi) in zip(lags,witness["probabilities"],boxes):
            transitions=expm(generator*(k*step))
            physical=float(np.sum(.5*transitions*agreement))
            self.assertAlmostEqual(physical,float(p),places=13)
            self.assertLessEqual(float(lo),physical)
            self.assertLessEqual(physical,float(hi))
        self.assertTrue(np.allclose(.5*generator,(.5*generator).T))

    def test_equilibrium_box_witness_binds_parameters_and_observation_contract(self):
        module=producer("contrast")
        lags=[1,2,3];boxes=[(F(1,2),F(9,10))]*3
        witness=module.reversible_box_witness(lags,boxes)
        self.assertTrue(module.verify_box_witness(witness,lags,boxes))
        changed=dict(witness);changed["probabilities"]=list(witness["probabilities"])
        changed["probabilities"][0]+=F(1,100)
        self.assertFalse(module.verify_box_witness(changed,lags,boxes))
        changed=dict(witness);changed["squared_detector_mean"]+=F(1,100)
        self.assertFalse(module.verify_box_witness(changed,lags,boxes))
        changed=dict(witness);changed["decay_per_step"]=F(1)
        self.assertFalse(module.verify_box_witness(changed,lags,boxes))
        other_boxes=list(boxes);other_boxes[0]=(F(51,100),F(9,10))
        self.assertFalse(module.verify_box_witness(witness,lags,other_boxes))
        self.assertFalse(module.verify_box_witness(witness,lags[:-1],boxes[:-1]))
        self.assertFalse(module.verify_box_witness({},lags,boxes))
        # An unsuccessful finite rational search must not be an exclusion proof.
        inconclusive=module.reversible_box_witness(lags,[(F(0),F(1,4))]*3)
        self.assertEqual(inconclusive["status"],"search_inconclusive")
        self.assertFalse(module.verify_box_witness(inconclusive,lags,[(F(0),F(1,4))]*3))


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentLaboratoryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={"scope":"independent virtual-laboratory checks; no measured data",
            "tests_run":result.testsRun,"failures":len(result.failures),
            "errors":len(result.errors),"passed":result.wasSuccessful(),
            "reproduction":"python tests/test_independent.py"}
    path=Path(__file__).resolve().parents[1]/"results"/"independent.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    sys.exit(not result.wasSuccessful())
