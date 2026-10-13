"""Independent internal review of observed diffusion geometry.

Analytic fixture calculations, numerical adversarial checks, and artifact
replay. No test is a substitute for the stated proofs or physical validation.
Central identities and replay calculations do not call producer diagnostics.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import itertools
import json
import math
import platform
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
try:
    from constructive_access.observed_geometry import finite_resolution as fr
    from constructive_access.observed_geometry import countermodels as cm
    from constructive_access.observed_geometry import combination as cb
    from constructive_access.observed_geometry import consistency as cs
except ImportError:
    import finite_resolution as fr
    import countermodels as cm
    import combination as cb
    import consistency as cs


def conditional_defect(a, c, probability, labels):
    """Finite-fiber conditional calculus, independently of producer modules."""
    out = []
    for key in sorted(set(labels)):
        mask = labels == key
        mass = probability[mask].sum()
        weights = probability[mask] / mass
        g = weights @ a[mask]
        k = weights @ (a[mask] ** 2 - c[mask] ** 2)
        m = weights @ c[mask] ** 2
        variance = weights @ (a[mask] - g) ** 2
        out.append((mask, mass, g, k, m, variance))
    return out


class IndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=json.loads((ROOT/'contracts/experiment.json').read_text())
        cls.bank=json.loads((ROOT/'results/experiment.json').read_text())

    def test_conditional_identity_slack_and_refinement_random_fibers(self):
        rng = np.random.default_rng(8507221)
        for _ in range(80):
            n = 48
            coarse = np.repeat(np.arange(4), 12)
            fine = np.repeat(np.arange(12), 4)
            probability = rng.random(n); probability /= probability.sum()
            V = np.repeat(rng.uniform(.1, 3, 4), 12)
            a, c = rng.normal(size=(2, n))
            e = c*c/V + rng.random(n)
            totals = []
            for labels in (coarse, fine, np.arange(n)):
                bound, slack = 0., 0.
                for mask, mass, g, k, m, variance in conditional_defect(a,c,probability,labels):
                    v = V[mask][0]
                    self.assertAlmostEqual(g*g-k, m-variance, places=12)
                    bound += mass*max(0,g*g-k)/v
                    slack += mass*min(m,variance)/v
                self.assertAlmostEqual(probability@e-bound,
                    probability@(e-c*c/V)+slack, places=12)
                self.assertLessEqual(bound, probability@e+1e-12)
                totals.append(bound)
            self.assertLessEqual(totals[0],totals[1]+1e-12)
            self.assertLessEqual(totals[1],totals[2]+1e-12)

    def test_invertible_scalar_recalibration_cancels_ito_correction(self):
        # psi(y)=y+y^3/5 is a global smooth diffeomorphism.
        rng = np.random.default_rng(94484)
        for _ in range(100):
            y, V = rng.uniform(-1,1),rng.uniform(.1,3)
            a,c = rng.normal(size=(2,9)); p = rng.random(9); p /= p.sum()
            first, second = 1+3*y*y/5, 6*y/5
            g,k = p@a,p@(a*a-c*c)
            az,cz = first*a+second*V,first*c
            gz,kz,Vz = p@az,p@(az*az-cz*cz),first*first*V
            self.assertAlmostEqual((gz*gz-kz)/Vz,(g*g-k)/V,places=12)
            self.assertAlmostEqual(gz,first*g+second*V,places=13)

    def test_circle_all_local_quantities_and_exact_attainment(self):
        for gamma,omega in ((1.,0.),(.3,2.),(2.,-.75)):
            phase = np.linspace(.003,2*np.pi-.003,1001)
            phase = phase[np.abs(np.sin(phase))>.002]
            y = np.cos(phase); V = gamma*np.sin(phase)**2
            a = -gamma*y; c = -omega*np.sin(phase)
            K = a*a-c*c
            ratio = (a*a-K)/V
            np.testing.assert_allclose(ratio,omega*omega/gamma,rtol=1e-9,atol=1e-10)
            for h in (.001,.05,.3):
                forward = (np.exp(-gamma*h)*np.cos(phase+omega*h)-y)/h
                backward = (np.exp(-gamma*h)*np.cos(phase-omega*h)-y)/h
                gh = (forward+backward)/2
                Kh = forward*backward
                actual = (gh*gh-Kh)/V
                target = np.exp(-2*gamma*h)*np.sin(omega*h)**2/(gamma*h*h)
                np.testing.assert_allclose(actual,target,rtol=1e-7,atol=2e-8)
                self.assertLessEqual(target,omega*omega/gamma+1e-12)

    def test_reversible_scalar_circle_transcripts_by_wrapped_kernel(self):
        # The kernel reflection proof is also checked by explicit integration:
        # sum the two hidden preimages of three prescribed cosine observations.
        def kernel(theta,t,omega,gamma):
            k = np.arange(1,80)
            return (1+2*np.sum(np.exp(-gamma*t*k*k)*np.cos(k*(theta-omega*t))))/(2*np.pi)
        gamma,omega = .7,1.9
        phase = (.4,1.1,2.4); gaps=(.3,.8)
        def density(phases,times):
            total=0.
            for signs in itertools.product((-1,1),repeat=3):
                x=np.array(phases)*signs
                total += kernel(x[1]-x[0],times[0],omega,gamma)*kernel(x[2]-x[1],times[1],omega,gamma)/(2*np.pi)
            return total/np.prod(np.sin(phases))
        self.assertAlmostEqual(density(phase,gaps),density(phase[::-1],gaps[::-1]),places=13)
        self.assertGreater(omega*omega/gamma,0)

    def test_reflection_symmetric_nonuniform_diffusion_family_is_sharp(self):
        phase=(np.arange(8192)+.37)*2*np.pi/8192
        density=(1+.3*np.cos(phase))/(2*np.pi)
        D=1+.2*np.cos(2*phase)
        derivative_D=-.4*np.sin(2*phase)
        derivative_log_density=-.3*np.sin(phase)/(1+.3*np.cos(phase))
        flux=.12
        a=-D*np.cos(phase)-(derivative_D+D*derivative_log_density)*np.sin(phase)
        c=-flux*np.sin(phase)/density
        V=D*np.sin(phase)**2
        # Reflection leaves a,V and e unchanged, so each conditional fiber's
        # reversible drift has zero variance; c itself changes sign.
        reflected=-phase
        ar=-(1+.2*np.cos(2*reflected))*np.cos(reflected)-(
            -.4*np.sin(2*reflected)+(1+.2*np.cos(2*reflected))*(-.3*np.sin(reflected)/(1+.3*np.cos(reflected))))*np.sin(reflected)
        np.testing.assert_allclose(ar,a,atol=1e-14)
        e=flux**2/(density*density*D)
        np.testing.assert_allclose(c*c/V,e,rtol=1e-13)
        mean_bound=np.mean(density*(c*c/V))*2*np.pi
        direct=flux**2*np.mean(1/(density*D))*2*np.pi
        self.assertAlmostEqual(mean_bound,direct,places=13)

    def test_quantized_center_rule_is_pointwise_and_reading_error_is_uniform(self):
        # A coarse grid exaggerates quantization so that a wrong envelope or
        # accidental q re-selection in the score comparison is exposed.
        denominator=128;delta=1/(2*denominator);h=.1;Q=.8;floor=.36
        rng=np.random.default_rng(140873)
        max_error=4*(2/h+Q)*delta/(h*floor)
        for _ in range(5000):
            true=rng.uniform(-1,1,3);z=np.rint(true*denominator)/denominator
            if abs(z[1])>.8:continue
            q=-z[1];envelope=1-max(0,abs(z[1])-delta)**2
            self.assertGreaterEqual(envelope+1e-15,1-true[1]**2)
            self.assertGreaterEqual(envelope,floor)
            def score(x):return -(q-(x[0]-x[1])/h)*(q-(x[2]-x[1])/h)/envelope
            self.assertLessEqual(abs(score(true)-score(z)),max_error+1e-12)

    def test_known_variance_score_bound_by_conditional_circle_moments(self):
        gamma,omega,h=1.,1.,.1
        p=fr.make_protocol('1/10','4/5','9/25',[2,2,2,2],global_gamma_upper=1)
        x=(np.arange(65536)+.5)*2*np.pi/65536;y=np.cos(x)
        mask=np.abs(y)<=.8;q=-y;a=1-y*y
        conditional=[]
        for direction in (-1,1):
            first=np.exp(-gamma*h)*np.cos(x+direction*omega*h)
            second=.5*(1+np.exp(-4*gamma*h)*np.cos(2*x+2*direction*omega*h))
            increment=(first-y)/h
            increment_square=(second-2*y*first+y*y)/(h*h)
            conditional.append((q-increment,q*q-2*q*increment+increment_square))
        mean_raw=-conditional[0][0]*conditional[1][0]
        second_raw=conditional[0][1]*conditional[1][1]
        correction=2*np.abs(q)*float(p['g_bias_upper'])+float(p['K_bias_upper'])
        corrected_second=np.where(mask,(second_raw-2*correction*mean_raw+correction*correction)/(a*a),0)
        corrected_mean=np.where(mask,(mean_raw-correction)/a,0)
        self.assertLessEqual(corrected_second.mean(),float(p['second_moment_upper']))
        self.assertLessEqual(corrected_mean.mean(),omega*omega/gamma)

    def test_matrix_positive_part_bound_with_singular_detector(self):
        rng=np.random.default_rng(88110)
        for n,k,rank in ((2,3,1),(3,2,2),(3,5,3),(4,4,2)):
            for _ in range(20):
                J=rng.normal(size=(k,rank))@rng.normal(size=(rank,n))
                T=rng.normal(size=(n,n)); D=T@T.T+np.eye(n)
                G=J@D@J.T
                ev,U=np.linalg.eigh(G)
                inverse=np.where(ev>max(ev)*1e-11,1/np.maximum(ev,1e-300),0)
                Q=(U*inverse)@U.T; root=(U*np.sqrt(inverse))@U.T
                a=rng.normal(size=(15,k)); v=rng.normal(size=(15,n)); c=v@J.T
                p=rng.random(15);p/=p.sum()
                g=p@a; K=np.einsum('i,ij,ik->jk',p,a,a)-np.einsum('i,ij,ik->jk',p,c,c)
                defect=root@(np.outer(g,g)-K)@root
                bound=np.maximum(0,np.linalg.eigvalsh(defect)).sum()
                entropy=np.einsum('i,ij,jk,ik',p,v,np.linalg.inv(D),v)
                self.assertLessEqual(bound,entropy+1e-8)
                np.testing.assert_allclose(G@Q@c.T,c.T,atol=3e-9)

    def test_zero_diffusivity_convention_and_dual(self):
        # At V=0, c=0 and Jensen makes the positive numerator exactly zero.
        a=np.array([-2.,1.,4.]);p=np.array([.2,.3,.5]);c=np.zeros(3)
        self.assertLessEqual((p@a)**2-p@(a*a-c*c),0)
        # For any selected q,w, the completed square bounds the exact dual.
        rng=np.random.default_rng(74811)
        for _ in range(100):
            V=rng.uniform(.001,2);g,K,q=rng.normal(size=3);w=rng.random()/V
            self.assertLessEqual(w*(2*q*g-q*q-K),max(0,g*g-K)/V+1e-12)

    def test_semigroup_bias_on_multimode_circle(self):
        phase=np.linspace(0,2*np.pi,4096,endpoint=False)
        coefficients={1:1.0,2:.35,4:-.1}
        gamma,omega=.8,1.3
        f=sum(a*np.cos(k*phase) for k,a in coefficients.items())
        def gen(order,sign):
            return sum((a*((-gamma*k*k+1j*sign*omega*k)**order)*np.exp(1j*k*phase)).real for k,a in coefficients.items())
        L,Lr=gen(1,1),gen(1,-1)
        M1=sum(abs(a)*math.sqrt(gamma**2*k**4+omega**2*k*k) for k,a in coefficients.items())
        M2=sum(abs(a)*(gamma**2*k**4+omega**2*k*k) for k,a in coefficients.items())
        bounds=[F(str(M1+1e-12))]*2+[F(str(M2+1e-12))]*2
        for h in (.001,.03,.4):
            plus=sum(a*np.exp(-gamma*k*k*h)*np.cos(k*(phase+omega*h)) for k,a in coefficients.items())
            minus=sum(a*np.exp(-gamma*k*k*h)*np.cos(k*(phase-omega*h)) for k,a in coefficients.items())
            u,v=(plus-f)/h,(minus-f)/h
            bias=fr.semigroup_bias(F(str(h)),bounds)
            self.assertLessEqual(np.max(np.abs((u+v)/2-(L+Lr)/2)),float(bias['g_bias_upper'])+1e-11)
            self.assertLessEqual(np.max(np.abs(u*v-L*Lr)),float(bias['K_bias_upper'])+1e-11)

    def test_score_range_independently_from_coupled_triple_geometry(self):
        p=fr.make_protocol('1/5','3/2','1/7',[3,4,11,13],readout_bounds=(-2,3))
        lo,hi=map(float,p['corrected_score_bounds']);h=.2
        rng=np.random.default_rng(284199)
        for _ in range(1000):
            x=rng.uniform(-2,3,3);q=rng.uniform(-1.5,1.5);a=rng.uniform(1/7,3)
            beta=2*abs(q)*float(p['g_bias_upper'])+float(p['K_bias_upper'])
            score=(-(q-(x[0]-x[1])/h)*(q-(x[2]-x[1])/h)-beta)/a
            self.assertGreaterEqual(score,lo-1e-10);self.assertLessEqual(score,hi+1e-10)
        # Producer rational score matches independent expansion exactly.
        x=(F(-7,5),F(2,7),F(4,5));q=F(3,8);a=F(2,3)
        raw=q*(x[0]+x[2]-2*x[1])/p['h']-q*q-(x[0]-x[1])*(x[2]-x[1])/p['h']**2
        correction=2*abs(q)*p['g_bias_upper']+p['K_bias_upper']
        self.assertEqual(fr.corrected_score(x,q,a,p),(raw-correction)/a)

    def test_shared_center_independent_noise_only_reduces_score(self):
        latent=np.array([-.2,.1,.3]);q=.12;h=.3;variance=.04
        clean=-(q-(latent[0]-latent[1])/h)*(q-(latent[2]-latent[1])/h)
        noisy=[]
        for signs in itertools.product((-1,1),repeat=3):
            x=latent+.2*np.array(signs)
            noisy.append(-(q-(x[0]-x[1])/h)*(q-(x[2]-x[1])/h))
        self.assertAlmostEqual(np.mean(noisy),clean-variance/(h*h),places=13)
        # Anticorrelated endpoints can instead raise a raw score; unbiasedness
        # alone does not establish the detector contract.
        dependent=[]
        for sign in (-1,1):
            x=latent+np.array([.2*sign,0,-.2*sign])
            dependent.append(-(q-(x[0]-x[1])/h)*(q-(x[2]-x[1])/h))
        self.assertGreater(np.mean(dependent),clean)

    def test_rational_log_and_square_root_outward_bounds(self):
        import mpmath as mp
        with mp.workdps(100):
            def q(x):return mp.mpf(x.numerator)/x.denominator
            for x in (F(1,1000),F(1),F(2),F(20),F(2)**90+F(1,17)):
                lo,hi=fr.log_interval(x);truth=mp.log(q(x))
                self.assertLessEqual(q(lo),truth+mp.mpf('1e-95'))
                self.assertGreaterEqual(q(hi),truth-mp.mpf('1e-95'))
                root=fr.sqrt_upper(x)
                self.assertGreaterEqual(root*root,x)
                self.assertLessEqual(q(root)-mp.sqrt(q(x)),mp.mpf(2)**-90)

    def test_empirical_bernstein_constants_and_exact_quantization(self):
        p=fr.make_protocol('1/2',1,1,[1,1,2,2])
        values=[F(-1),F(0),F(2),F(3)]*100
        n=len(values);mean=sum(values,F(0))/n;second=sum((x*x for x in values),F(0))/n
        out=fr.certify_summary(n,mean,p,concentration='empirical_bernstein',score_second_moment=second)
        lo,hi=p['corrected_score_bounds'];variance=F(n,n-1)*(second-mean*mean)
        radius=math.sqrt(2*float(variance)*math.log(40)/n)+7*float(hi-lo)*math.log(40)/(3*(n-1))
        self.assertEqual(out['empirical_variance'],variance)
        self.assertAlmostEqual(float(out['sampling_radius_upper']),radius,places=13)
        quant=F(1,1000)
        changed=fr.certify_summary(n,mean,p,concentration='empirical_bernstein',score_second_moment=second,quantization_radius=quant)
        self.assertEqual(changed['corrected_score_bounds'],(lo-quant,hi+quant))
        self.assertEqual(changed['uniform_score_quantization_penalty'],quant)
        self.assertLess(changed['signed_rate_lower'],out['signed_rate_lower'])

    def test_empirical_bernstein_exact_two_point_null_tail(self):
        p=fr.make_protocol(1,0,1,[0,0,0,0])
        lo,hi=map(float,p['corrected_score_bounds']);prob=-lo/(hi-lo)
        alpha=.05
        for n in (20,100,1000):
            k=np.arange(n+1);means=lo+(hi-lo)*k/n
            variance=(hi-lo)**2*k*(n-k)/(n*(n-1))
            radius=np.sqrt(2*variance*np.log(2/alpha)/n)+7*(hi-lo)*np.log(2/alpha)/(3*(n-1))
            rejected=means-radius>0
            self.assertLessEqual(float(binom.pmf(k,n,prob)[rejected].sum()),alpha)

    def test_protocol_refuses_selection_dependence_and_tampering(self):
        base=dict(h='1/5',q_abs_max=1,envelope_floor='1/10',generator_bounds=[2,2,4,4])
        for change in ({'selection':'same_data'},{'detector_contract':'memoryful'},
                       {'envelope_floor':0},{'h':0},{'coordinate_readouts_per_record':True}):
            with self.assertRaises(ValueError):fr.make_protocol(**(base|change))
        p=fr.make_protocol(**base);p['g_bias_upper']=F(0)
        with self.assertRaises(ValueError):fr.certify_summary(10,0,p)
        p=fr.make_protocol(**base)
        for kw in ({'record_contract':'stationary_only'},
                   {'concentration':'empirical_bernstein','score_second_moment':0,'mean_error':'1/100'},
                   {'concentration':'empirical_bernstein','score_second_moment':0,'record_contract':'bounded_conditional_mean_error'},
                   {'concentration':'bernstein'}):
            with self.assertRaises(ValueError):fr.certify_summary(10,0,p,**kw)

    def test_connected_torus_average_diffusivity_counterexample(self):
        # Integrate the invalid substitution itself, without its closed formula.
        c=8.;variance=.125
        value,error=quad(lambda x:max(0,c*c*(1+variance)-variance/math.tan(x)**2),
                         0,math.pi/2,epsabs=1e-9,points=[math.atan(1/24)])
        value*=2/math.pi
        self.assertGreater(value,64)
        self.assertAlmostEqual(value,cm.averaged_gamma_counterexample()['invalid_conditional_mean_substitution'],places=8)
        exact_lower=F(577,8)-F(1153,288)
        self.assertGreater(exact_lower,64)
        self.assertLess(float(exact_lower),value)
        # Retaining V resolves d(z)=V/(1-y^2), and attains E[c^2 d]=64.
        z=np.linspace(0,2*np.pi,8192,endpoint=False)
        diffusivity=1+.5*np.cos(z)
        self.assertAlmostEqual(np.mean(c*c*diffusivity),64,places=13)

    def test_naive_finite_h_bound_can_diverge_at_internal_critical_point(self):
        # Independently derive f(c)=c^2+c-1/2 and Gamma(c).
        # The alternate f-fiber c'=-1-c has different Gamma except at -1/2.
        for c in (-.9,-.7,-.4,-.1):
            alt=-1-c
            self.assertAlmostEqual(c*c+c-.5,alt*alt+alt-.5,places=14)
            G=lambda x:(1-x*x)*(1+2*x)**2
            self.assertAlmostEqual(G(c)-G(alt),(1+2*c)**3,places=14)
        numerator=math.exp(-1)*math.sin(1)-.5*math.exp(-4)*math.sin(2)
        self.assertNotEqual(numerator,0)
        ratios=[cm.finite_h_naive_ratio(-.5+x) for x in (.001,.0001)]
        self.assertGreater(ratios[1]/ratios[0],99)
        self.assertGreater(F(6517,2750),1)
        epsilon=1/15625
        for u in (-.1,-.005,.005,.1):
            self.assertGreaterEqual((1-(-.5+u)**2)*(2*u)**2,epsilon)
        # A rigorous integration over these subintervals already exceeds sigma.
        lower=F(7,25)**2*F(190,2)*F(7,22)
        self.assertEqual(lower,F(6517,2750))

    def test_generalized_parity_and_detector_noise_scope(self):
        norm=quad(lambda p:math.exp(math.cos(p)),0,2*math.pi,epsabs=1e-12)[0]
        ordinary=quad(lambda p:math.sin(p)**2*math.exp(math.cos(p)),0,2*math.pi,epsabs=1e-12)[0]/norm
        out=cm.parity_counterexample()
        self.assertAlmostEqual(ordinary,out['ordinary_entropy_rate'],places=13)
        self.assertEqual(out['generalized_entropy_rate'],0)
        for n in (100,1000,10000):
            out=cm.noisy_realized_qv(n)
            self.assertAlmostEqual(out['sensor_contribution'],.02*n,places=13)
            self.assertAlmostEqual(out['clean_expected_realized_qv'],n*(1-math.exp(-1/n)*math.cos(1/n)),places=13)
        self.assertGreater(cm.noisy_realized_qv(10000)['expected_realized_qv'],200)

    def test_finite_h_failure_survives_invertible_scalar_recalibration(self):
        kappa=.49;h=1.
        def ratio(x):
            c=math.cos(x)
            return ((math.exp(-h)*math.sin(h)+kappa*math.exp(-4*h)*math.sin(2*h)*c)/(h*(1+2*kappa*c)))**2
        direct=quad(ratio,0,2*math.pi,epsabs=1e-11,epsrel=1e-11,points=[math.pi])[0]/(2*math.pi)
        self.assertAlmostEqual(direct,cm.invertible_recalibration_counterexample()['invalid_finite_h_functional'],places=10)
        self.assertGreater(direct,11)
        self.assertGreater(1-2*kappa,0)
        self.assertGreater(F(4205,396),1)
        self.assertLess(float(F(4205,396)),direct)

    def test_all_frozen_experiment_certificates_replayed_independently(self):
        d=self.contract;p=d['protocol'];n=d['independent_stationary_triples']
        alpha=float(F(d['alpha_per_method']));h=float(F(d['lag']))
        wd,bd=d['score_denominator'],d['endpoint_denominator']
        we,be=float(F(d['score_uniform_error'])),float(F(d['endpoint_uniform_error']))
        low,high=map(lambda x:float(F(x)),p['corrected_score_bounds'])
        width=high-low+2*we
        s=float(F(d['endpoint_parameter']));bwidth=1+(1+s)**2+2*be
        log=math.log(2/alpha)
        self.assertEqual(len(self.bank['datasets']),1200)
        checked=0
        for row in self.bank['datasets']:
            self.assertEqual(row['records'],n)
            mean=row['weighted_total']/(n*wd)
            second=row['weighted_square_total']/(n*wd*wd)
            variance=n/(n-1)*(second-mean*mean)
            radius=math.sqrt(2*variance*log/n)+7*width*log/(3*(n-1))
            lower=max(0,mean-radius-we)
            self.assertAlmostEqual(mean,row['weighted_mean'],places=12)
            self.assertAlmostEqual(variance,row['weighted_variance'],places=9)
            self.assertAlmostEqual(radius,row['weighted_sampling_radius_upper'],places=12)
            self.assertAlmostEqual(lower,row['weighted_rate_lower'],places=12)
            hoeffding=width*math.sqrt(math.log(1/alpha)/(2*n))
            self.assertAlmostEqual(hoeffding,row['hoeffding_sampling_radius_upper'],places=11)
            self.assertAlmostEqual(max(0,mean-hoeffding-we),row['hoeffding_rate_lower'],places=12)
            bmean=row['endpoint_total']/(n*bd)
            bsecond=row['endpoint_square_total']/(n*bd*bd)
            bvariance=n/(n-1)*(bsecond-bmean*bmean)
            bradius=math.sqrt(2*bvariance*log/n)+7*bwidth*log/(3*(n-1))
            defect=max(0,bmean-bradius-be)
            phi=0 if defect==0 else 2*math.sqrt(defect)*math.atanh(math.sqrt(defect))
            self.assertAlmostEqual(bradius,row['endpoint_sampling_radius_upper'],places=14)
            self.assertAlmostEqual(phi/h,row['endpoint_rate_lower'],places=12)
            checked+=1
        self.assertEqual(checked,1200)

    def test_all_raw_exemplar_codes_and_aggregates_with_python_integers(self):
        d=self.contract;D=d['rounding_denominator'];wd=d['score_denominator'];bd=d['endpoint_denominator']
        self.assertEqual(D,2**20);self.assertEqual(wd,256);self.assertEqual(bd,2**24)
        records=np.load(ROOT/'results/experiment_raw.npz',allow_pickle=False)
        rows={r['scenario']:r for r in self.bank['datasets'] if r['replicate']==0}
        for scenario,row in rows.items():
            j=records[scenario+'_readings'];w=records[scenario+'_weighted_codes'];b=records[scenario+'_endpoint_codes']
            self.assertEqual(j.shape,(200000,3))
            totalw,totalw2,totalb,totalb2,retained=0,0,0,0,0
            for index,(triple,wi,bi) in enumerate(zip(j,w,b)):
                past,center,future=map(int,triple);wi,bi=int(wi),int(bi)
                left=25*past-24*center;right=25*future-24*center
                if 5*abs(center)<=4*D:
                    retained+=1
                    num=-15*left*right-6*abs(center)*D-19*D*D
                    den=15*(D*D-max(abs(center)-1,0)**2)
                    self.assertTrue(wi*den<=wd*num<(wi+1)*den)
                else:self.assertEqual(wi,0)
                denb=625*D*D
                self.assertTrue(bi*denb<=-bd*left*right<(bi+1)*denb)
                totalw+=wi;totalw2+=wi*wi;totalb+=bi;totalb2+=bi*bi
                # A sparse second path uses the unexpanded Fraction score,
                # independently checking the expanded integer coefficients.
                if index%997==0 and 5*abs(center)<=4*D:
                    x=list(map(lambda v:F(int(v),D),triple));q=-x[1]
                    envelope=1-max(F(0),abs(x[1])-F(1,D))**2
                    raw=-(q-25*(x[0]-x[1]))*(q-25*(x[2]-x[1]))
                    ideal=(raw-F(2,5)*abs(q)-F(19,15))/envelope
                    self.assertLessEqual(F(wi,wd),ideal)
                    self.assertLess(ideal,F(wi+1,wd))
            self.assertEqual(retained,row['retained'])
            self.assertEqual((totalw,totalw2,totalb,totalb2),
                (row['weighted_total'],row['weighted_square_total'],row['endpoint_total'],row['endpoint_square_total']))

    def test_frozen_resources_calibration_and_hash_bindings(self):
        d=self.contract;r=self.bank
        digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest(ROOT/'contracts/experiment.json'),r['contract_sha256'])
        self.assertEqual(digest(ROOT/'results/experiment_raw.npz'),r['raw_sha256'])
        self.assertEqual(d['source_hashes'],r['source_hashes'])
        for name in ('experiment.py','finite_resolution.py'):
            self.assertEqual(digest(ROOT/'code'/name),d['source_hashes'][name])
        self.assertEqual(d['resources_per_dataset'],r['resources_per_dataset'])
        n=d['independent_stationary_triples'];h=F(d['lag']);s=F(d['endpoint_parameter'])
        cost=d['resources_per_dataset']
        self.assertEqual(cost['detector_readings'],3*n)
        self.assertEqual(cost['independent_stationary_preparations'],n)
        self.assertEqual(F(cost['modeled_evolution_time']),2*h*n)
        self.assertEqual(F(cost['modeled_evolution_time']),16000)
        delta=F(d['reading_error_upper']);Q=F(d['protocol']['q_abs_max']);floor=F(d['protocol']['envelope_floor'])
        self.assertEqual(F(d['score_uniform_error']),4*(2/h+Q)*delta/(h*floor)+F(1,d['score_denominator']))
        self.assertEqual(F(d['endpoint_uniform_error']),2*(1+s)**2*delta+F(1,d['endpoint_denominator']))
        self.assertGreaterEqual(F(d['common_generator_bounds'][0])**2,1+d['declared_frequency_cap']**2)
        self.assertEqual(F(d['common_generator_bounds'][2]),1+d['declared_frequency_cap']**2)

    def test_raw_bank_seed_and_stationary_transition_reconstruction(self):
        d=self.contract
        raw=np.load(ROOT/'results/experiment_raw.npz',allow_pickle=False)
        h=float(F(d['lag']));D=d['rounding_denominator'];n=d['independent_stationary_triples']
        for row in self.bank['datasets']:
            if row['replicate']!=0:continue
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(row['seed'])))
            phase=rng.uniform(0,2*np.pi,n)
            innovations=rng.standard_normal((n,2))
            omega=d['scenarios'][row['scenario']]
            before=np.cos(phase-omega*h+math.sqrt(2*h)*innovations[:,0])
            center=np.cos(phase)
            after=np.cos(phase+omega*h+math.sqrt(2*h)*innovations[:,1])
            independently_rounded=np.rint(np.column_stack((before,center,after))*D).astype(np.int64)
            np.testing.assert_array_equal(independently_rounded,raw[row['scenario']+'_readings'])
            self.assertLess(abs(center.mean()),.01)
            self.assertLess(abs(np.mean(center*center)-.5),.01)

    def test_summary_retains_weak_signal_failure_and_all_controls(self):
        rows=self.bank['datasets']
        for scenario,summary in self.bank['scenario_summary'].items():
            selected=[r for r in rows if r['scenario']==scenario]
            self.assertEqual(len(selected),400)
            for method,claimed in summary['methods'].items():
                values=np.array([r[method+'_rate_lower'] for r in selected])
                self.assertEqual(np.count_nonzero(values),claimed['positive_certificates'])
                self.assertAlmostEqual(values.mean(),claimed['mean_rate_lower'],places=14)
                self.assertAlmostEqual(values.min(),claimed['minimum_rate_lower'],places=14)
                self.assertAlmostEqual(values.max(),claimed['maximum_rate_lower'],places=14)
                self.assertEqual(np.count_nonzero(values>summary['true_entropy_rate']),claimed['over_truth'])
        summary=self.bank['scenario_summary']
        self.assertEqual(summary['driven']['methods']['weighted']['positive_certificates'],400)
        self.assertEqual(summary['driven']['methods']['endpoint']['positive_certificates'],400)
        self.assertGreater(summary['driven']['methods']['weighted']['mean_rate_lower']/summary['driven']['methods']['endpoint']['mean_rate_lower'],9.8)
        self.assertEqual(summary['weak_driven']['methods']['weighted']['positive_certificates'],0)
        self.assertEqual(summary['weak_driven']['methods']['endpoint']['positive_certificates'],396)
        self.assertTrue(all(x['positive_certificates']==0 for x in summary['equilibrium']['methods'].values()))

    def test_all_rebalanced_combination_rates_replayed_independently(self):
        saved=json.loads((ROOT/'results/combination.json').read_text())
        d=self.contract;n=d['independent_stationary_triples'];h=float(F(d['lag']))
        alpha=float(F(saved['allocation']['geometry']))
        self.assertEqual(alpha,.025)
        self.assertEqual(saved['allocation']['geometry'],saved['allocation']['endpoint'])
        wd,bd=d['score_denominator'],d['endpoint_denominator']
        we,be=float(F(d['score_uniform_error'])),float(F(d['endpoint_uniform_error']))
        lo,hi=map(lambda x:float(F(x)),d['protocol']['corrected_score_bounds'])
        width=hi-lo+2*we;s=float(F(d['endpoint_parameter']));bw=1+(1+s)**2+2*be
        log=math.log(2/alpha)
        source={(r['scenario'],r['replicate']):r for r in self.bank['datasets']}
        self.assertEqual(len(saved['datasets']),1200)
        for row in saved['datasets']:
            original=source[row['scenario'],row['replicate']]
            m=original['weighted_total']/(n*wd)
            v=n/(n-1)*(original['weighted_square_total']/(n*wd*wd)-m*m)
            wrad=math.sqrt(2*v*log/n)+7*width*log/(3*(n-1))
            w=max(0,m-wrad-we)
            m=original['endpoint_total']/(n*bd)
            v=n/(n-1)*(original['endpoint_square_total']/(n*bd*bd)-m*m)
            erad=math.sqrt(2*v*log/n)+7*bw*log/(3*(n-1))
            defect=max(0,m-erad-be)
            e=2*math.sqrt(defect)*math.atanh(math.sqrt(defect))/h if defect else 0
            self.assertAlmostEqual(w,row['geometry_rate_lower'],places=12)
            self.assertAlmostEqual(e,row['endpoint_rate_lower'],places=12)
            self.assertAlmostEqual(wrad,row['geometry_radius_upper'],places=12)
            self.assertAlmostEqual(erad,row['endpoint_radius_upper'],places=14)
            self.assertAlmostEqual(max(w,e),row['combined_rate_lower'],places=12)
            winner='tie' if w==e else 'geometry' if w>e else 'endpoint'
            self.assertEqual(winner,row['selected_component'])
            self.assertLessEqual(row['geometry_rate_lower'],original['weighted_rate_lower']+1e-12)
            self.assertLessEqual(row['endpoint_rate_lower'],original['endpoint_rate_lower']+1e-12)

    def test_combination_api_union_budget_costs_and_validation(self):
        d=self.contract;frozen=d['protocol'];row=self.bank['datasets'][0]
        keys=('h','q_abs_max','envelope_floor','generator_bounds','readout_bounds',
              'global_gamma_upper','selection','detector_contract','coordinate_readouts_per_record')
        protocol=fr.make_protocol(**{k:frozen[k] for k in keys})
        args=dict(records=row['records'],geometry_total=row['weighted_total'],
            geometry_square_total=row['weighted_square_total'],geometry_scale=d['score_denominator'],
            endpoint_total=row['endpoint_total'],endpoint_square_total=row['endpoint_square_total'],
            endpoint_scale=d['endpoint_denominator'],protocol=protocol,
            geometry_score_error=d['score_uniform_error'],endpoint_score_error=d['endpoint_uniform_error'],
            endpoint_parameter=d['endpoint_parameter'])
        result=cb.certify_shared_records(**args,external_failure=F(1,100))
        self.assertEqual(result['failure_allocation'],{'geometry':F(1,40),'endpoint':F(1,40)})
        self.assertEqual(result['confidence_failure_upper'],F(3,50))
        self.assertEqual(result['joint_calibration_failure_upper'],F(1,100))
        self.assertEqual(result['entropy_production_rate_lower'],max(result['geometry']['entropy_production_rate_lower'],result['endpoint']['entropy_production_rate_lower']))
        self.assertEqual(result['resources']['detector_readings'],600000)
        self.assertEqual(result['resources']['independent_stationary_preparations'],200000)
        self.assertEqual(result['resources']['modeled_evolution_time'],16000)
        for change in ({'geometry_failure_share':0},{'geometry_failure_share':1},
                       {'alpha_total':F(99,100),'external_failure':F(1,100)},
                       {'endpoint_parameter':2},{'geometry_scale':0}):
            with self.assertRaises(ValueError):cb.certify_shared_records(**(args|change))
        changed=fr.make_protocol(**({k:frozen[k] for k in keys}|{'readout_bounds':(-2,2)}))
        with self.assertRaises(ValueError):cb.certify_shared_records(**(args|{'protocol':changed}))

    def test_combination_preserves_bank_and_reports_power_penalty(self):
        r=json.loads((ROOT/'results/combination.json').read_text())
        digest=lambda name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        self.assertEqual(r['inputs']['experiment_sha256'],digest('results/experiment.json'))
        self.assertEqual(r['inputs']['contract_sha256'],digest('contracts/experiment.json'))
        self.assertEqual(r['inputs']['raw_sha256'],digest('results/experiment_raw.npz'))
        self.assertEqual(r['source_hashes']['combination.py'],digest('code/combination.py'))
        self.assertEqual(r['source_hashes']['finite_resolution.py'],digest('code/finite_resolution.py'))
        self.assertEqual(sum(map(F,r['allocation'].values())),F(r['statistical_failure_upper']))
        self.assertIn('Retrospective',r['analysis_status'])
        for name,summary in r['scenario_summary'].items():
            rows=[x for x in r['datasets'] if x['scenario']==name]
            self.assertEqual(len(rows),400)
            for method,claimed in summary['methods'].items():
                values=[x[method+'_rate_lower'] for x in rows]
                self.assertEqual(sum(x>0 for x in values),claimed['positive_certificates'])
                self.assertAlmostEqual(math.fsum(values)/len(values),claimed['mean_rate_lower'],places=14)
                self.assertEqual(sum(x>summary['true_entropy_rate'] for x in values),claimed['over_truth'])
        s=r['scenario_summary']
        self.assertEqual(s['driven']['methods']['combined']['positive_certificates'],400)
        self.assertEqual(s['weak_driven']['methods']['combined']['positive_certificates'],383)
        self.assertEqual(s['weak_driven']['methods']['original_endpoint']['positive_certificates'],396)
        self.assertEqual(s['equilibrium']['methods']['combined']['positive_certificates'],0)
        self.assertLess(s['driven']['methods']['combined']['mean_rate_lower'],s['driven']['methods']['original_geometry']['mean_rate_lower'])

    def test_consistency_envelopes_against_actual_protocol_exactly(self):
        # These square-root-rational fixtures avoid confusing a mathematical
        # envelope with the fixed absolute rounding of a numerical square root.
        for gamma,Omega,h0,root in ((F(1),F(3),F(2),F(1)),
                                    (F(1,2),F(3,4),F(1,4),F(2)),
                                    (F(3),F(0),F(6),F(1))):
            self.assertEqual(root*root,2*gamma/h0)
            M=gamma+Omega;R=gamma*gamma+Omega*Omega;C=R*(2*gamma+Omega)
            H=((2*gamma+Omega+root)**2+C*h0)/gamma
            B0=(1/(h0*h0)+(2/h0+gamma)**2+C*h0)/gamma
            for m in (1,4,9,25):
                h=h0/(m**3);eps=gamma/(m*m)
                p=fr.make_protocol(h,gamma,eps,[M,M,R,R],global_gamma_upper=gamma)
                lo,hi=p['corrected_score_bounds']
                self.assertEqual(p['maximum_weighted_bias'],C*h/eps)
                self.assertLessEqual(hi-lo,B0*m**8)
                self.assertLessEqual(p['second_moment_upper'],H*H*m**10)
                self.assertEqual(2*C*h/eps,2*C*h0/(gamma*m))

    def test_consistency_wrapper_encloses_radius_and_removes_fixed_floor(self):
        import mpmath as mp
        with mp.workdps(160):
            def number(x):
                x=F(x)
                return mp.mpf(x.numerator)/x.denominator
            for gamma,Omega,h0 in ((F(1),F(1),F(1)),
                                   (F(1,10**40),F(0),F(10**40)),
                                   (F(13,7),F(19,11),F(3,17))):
                g,o,t=map(number,(gamma,Omega,h0));M=g+o;R=g*g+o*o;C=R*(2*g+o)
                H=((2*g+o+mp.sqrt(2*g/t))**2+C*t)/g
                B=(1/(t*t)+(2/t+g)**2+C*t)/g
                for m in (1,3,20,2**120):
                    plan=cs.design(m,gamma,Omega,h0)
                    self.assertGreaterEqual(number(plan['second_moment_coefficient_upper']),H)
                    self.assertEqual(plan['range_coefficient_upper'],F(str(gamma))**-1*(h0**-2+(2/h0+gamma)**2+(gamma*gamma+Omega*Omega)*(2*gamma+Omega)*h0))
                    ell=mp.log(40)
                    exact_linear=B*m**8*ell/(3*m**12)
                    exact_radius=exact_linear+mp.sqrt(2*H*H*m**10*ell/(m**12)+exact_linear**2)
                    self.assertGreaterEqual(number(plan['sampling_radius_upper']),exact_radius)
                    self.assertEqual(plan['sampling_radius_upper'],plan['radius_first_order_coefficient']/m+plan['radius_fourth_order_coefficient']/m**4)
        floor=fr.sqrt_upper(F(1,2**400))
        self.assertEqual(floor,F(1,2**90))
        small=cs.design(2**120)
        self.assertLess(small['sampling_radius_upper'],floor)
        doubled=cs.design(2**121)
        self.assertLess(doubled['sampling_radius_upper'],small['sampling_radius_upper']/2)
        # This is an algebraic API check on a synthetic summary, not a claim
        # that the astronomical design quota has been acquired or simulated.
        out=cs.certify_mean(F(1,2),small,numerical_mean_radius=F(1,1000))
        self.assertEqual(out['confidence_failure_upper'],F(1,40))
        self.assertEqual(out['joint_validity_and_accuracy_failure_upper'],F(1,20))
        self.assertEqual(out['joint_accuracy_gap_upper'],small['joint_accuracy_gap_upper']+F(1,500))
        self.assertEqual(out['signed_rate_lower'],F(1,2)-small['sampling_radius_upper']-F(1,1000))
        saved=json.loads((ROOT/'results/consistency.json').read_text())
        self.assertEqual(saved['kind'],'deterministic_design_calculation_no_simulated_or_measured_records')
        for p in saved['plans']:
            m=p['m'];self.assertEqual(p['records'],m**12)
            self.assertEqual(F(p['declared_evolution_time']),2*m**9)
            self.assertEqual(p['triple_detector_readouts'],3*m**12)
            radius=F(p['radius_first_order_coefficient'])/m+F(p['radius_fourth_order_coefficient'])/m**4
            self.assertEqual(F(p['sampling_radius_upper']),radius)


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReview)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths={'tests/test_independent.py':Path(__file__),
           'code/finite_resolution.py':Path(fr.__file__),
           'code/countermodels.py':Path(cm.__file__),
           'code/combination.py':Path(cb.__file__),
           'code/consistency.py':Path(cs.__file__)}
    for name in ('manuscript/population_theory.tex','manuscript/finite_resolution.tex',
                 'manuscript/countermodels.tex','manuscript/literature.tex','manuscript/experiment.tex',
                 'research/population_theory.txt','research/countermodels.txt','research/finite_resolution.txt',
                 'research/literature.txt','research/sources.json','research/experiment.txt',
                 'research/independent_review.txt',
                 'results/countermodels.json','code/experiment.py','contracts/experiment.json',
                 'results/experiment.json','results/experiment_raw.npz','results/combination.json',
                 'research/combination.txt','manuscript/combination.tex',
                 'research/consistency.txt','manuscript/consistency.tex','results/consistency.json'):
        if (ROOT/name).is_file():paths[name]=ROOT/name
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        external_peer_review=False,physical_experiments=False,
        python_version=platform.python_version(),numpy_version=np.__version__,
        scope='Independent analytic identity checks, diffusion countermodels, finite-error contracts and frozen simulation replay; no historical-priority claim',
        source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()})
    (ROOT/'results/independent_review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(not result.wasSuccessful())
