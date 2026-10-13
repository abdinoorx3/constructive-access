"""Independent analytic fixtures for the detector-access proofs.

These are deterministic calculations, not newly simulated laboratory records.
Nothing in observed_geometry or its frozen release is read as writable input.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import unittest
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'countermodels'))
import exact_sampling as producer


class DetectorReview(unittest.TestCase):
    def test_triangular_inverse_telescopes_and_periodic_kernel_remains(self):
        tau=.7;t=.24;frequency=1.3
        signal=lambda x:1.2+math.cos(frequency*x)+.3*math.sin(2.1*x)
        derivative=lambda x:(signal(x)-signal(x-tau))/tau
        for n in (1,2,7,40):
            reconstructed=1.2+tau*sum((1-k/n)*derivative(t-k*tau) for k in range(n))
            exact=signal(t)-sum(signal(t-j*tau)-1.2 for j in range(1,n+1))/n
            self.assertAlmostEqual(reconstructed,exact,places=13)
        for phase in (.2,1.1,2.7):
            exposed=quad(lambda s:math.cos(2*math.pi*s/tau+phase),t-tau,t,epsabs=1e-13)[0]/tau
            self.assertAlmostEqual(exposed,0,places=13)

    def test_covariance_reconstruction_and_derivative_free_norm_bounds(self):
        gamma,tau,omega=.6,.7,2.3
        for n in (1,2,8,50,500):
            variance=(n+2*sum((n-j)*math.exp(-gamma*j*tau)*math.cos(omega*j*tau) for j in range(1,n)))/(n*n)
            bound=1/(n*math.tanh(gamma*tau/2))
            self.assertGreaterEqual(variance,-1e-13)
            self.assertLessEqual(variance,bound+1e-13)
            x=np.geomspace(1e-15,1,1000)
            gain=-np.expm1(n*np.log1p(-np.minimum(x,1-1e-16)))/np.sqrt(x)
            self.assertLessEqual(gain.max(),math.sqrt(n)+1e-12)

    def test_exposed_cosine_variance_and_gaussian_tv_upper_bound(self):
        gamma,tau,nu,n=.8,.7,.4,20
        upper_coefficient=(tau*gamma+1+math.exp(-gamma*tau))/(tau*tau)
        for omega in (0.,1.,7.,21.,80.):
            direct=quad(lambda t:(tau-t)*math.exp(-gamma*t)*math.cos(omega*t),0,tau,epsabs=1e-12,limit=300)[0]/(tau*tau)
            z=gamma-1j*omega
            exact=(tau/z-(1-np.exp(-z*tau))/(z*z)).real/(tau*tau)
            self.assertAlmostEqual(exact,direct,places=12)
            self.assertGreaterEqual(exact,0)
            self.assertLessEqual(exact,upper_coefficient/(gamma*gamma+omega*omega)+1e-13)
            kl=n*exact/(2*nu*nu)
            self.assertAlmostEqual(math.sqrt(kl/2),math.sqrt(n*exact)/(2*nu),places=13)

    def test_gaussian_mixture_chisquare_and_quartic_independent_trial_order(self):
        # A symmetric two-point bounded signal has exact chi-square
        # cosh(a²/nu²)-1, derived without a small-amplitude expansion.
        nu=.8
        generic=math.expm1(1/(nu*nu))-1/(nu*nu)
        symmetric=math.cosh(1/(nu*nu))-1
        for amplitude in (.01,.1,.4,1.):
            ratio=lambda x:math.exp(-amplitude*amplitude/(2*nu*nu))*math.cosh(amplitude*x/(nu*nu))
            density=lambda x:math.exp(-x*x/(2*nu*nu))/(math.sqrt(2*math.pi)*nu)
            numeric=quad(lambda x:(ratio(x)-1)**2*density(x),-10*nu,10*nu,epsabs=1e-13)[0]
            exact=2*math.sinh(amplitude*amplitude/(2*nu*nu))**2
            self.assertAlmostEqual(numeric,exact,places=11)
            self.assertLessEqual(exact,amplitude**4*symmetric+1e-13)
            self.assertLessEqual(symmetric,generic)
        values=[]
        for omega in (10.,20.,40.,80.):
            # Variance surrogate a²=omega^-2; fixed n makes TV upper O(omega^-2).
            variance=1/(omega*omega)
            values.append(math.sqrt(100*symmetric/2)*variance)
        for first,second in zip(values,values[1:]):self.assertAlmostEqual(first/second,4)

    def test_oriented_phase_mean_exception_and_prepared_second_moment(self):
        gamma,tau,nu=.8,.7,.4
        c=-math.expm1(-gamma*tau)/tau
        for index in (1,2,5):
            omega=2*math.pi*index/tau
            direct=quad(lambda t:-math.exp(-gamma*t)*math.sin(omega*t),0,tau,epsabs=1e-12)[0]/tau
            mean=-c*omega/(gamma*gamma+omega*omega)
            self.assertAlmostEqual(direct,mean,places=12)
            mean2=-c*2*omega/(gamma*gamma+4*omega*omega)
            gap=c*omega*abs(2*omega*omega-gamma*gamma)/((gamma*gamma+omega*omega)*(gamma*gamma+4*omega*omega))
            self.assertAlmostEqual(abs(mean-mean2),gap,places=13)
            # Conditional Gaussian phase increments give this independent
            # two-time moment; no simulated trajectory is used.
            def outer(s):
                return quad(lambda lag:.5*math.exp(-gamma*lag)*(math.cos(omega*lag)
                    -math.exp(-4*gamma*s)*math.cos(omega*(2*s+lag))),0,tau-s,epsabs=1e-11)[0]
            second=2*quad(outer,0,tau,epsabs=1e-10)[0]/tau**2
            ito_upper=(2+gamma*tau+math.sqrt(2*gamma*tau))**2/(omega*tau)**2
            self.assertGreaterEqual(second,mean*mean-1e-12)
            self.assertLessEqual(second,ito_upper+1e-12)
            epsilon=.05
            n=math.ceil(8*(1+nu*nu)*math.log(1/epsilon)/(gap*gap))
            self.assertLessEqual(math.exp(-n*gap*gap/(8*(1+nu*nu))),epsilon)
        # The resonance formula has a real exception; choose gamma so that
        # a resonant omega is precisely gamma/sqrt(2).
        omega=2*math.pi/tau;exception_gamma=math.sqrt(2)*omega
        exceptional_c=-math.expm1(-exception_gamma*tau)/tau
        first=-exceptional_c*omega/(exception_gamma**2+omega**2)
        second=-exceptional_c*2*omega/(exception_gamma**2+4*omega**2)
        self.assertAlmostEqual(first,second,places=14)

    def test_matching_rate_constants_for_independent_exposure_trials(self):
        gamma,tau,nu=.8,.7,.4
        b=-math.expm1(-gamma*tau)
        resonance_variance=lambda w:((gamma*tau-b)*gamma*gamma+(gamma*tau+b)*w*w)/(tau*tau*(gamma*gamma+w*w)**2)
        c=b/tau
        fourth_bound=1+6*nu*nu+3*nu**4
        target_v=3*(gamma*tau+b)/(4*tau*tau)
        target_mu=c/2
        previous=None
        for index in (10,100,1000):
            omega=2*math.pi*index/tau
            dv=resonance_variance(omega)-resonance_variance(2*omega)
            dmu=c*omega*(2*omega*omega-gamma*gamma)/((gamma*gamma+omega*omega)*(gamma*gamma+4*omega*omega))
            self.assertGreater(dv,0);self.assertGreater(dmu,0)
            err=abs(dv*omega**2-target_v)+abs(dmu*omega-target_mu)
            if previous is not None:self.assertLess(err,previous)
            previous=err
            epsilon=.1
            n=math.ceil(4*fourth_bound/(epsilon*dv*dv))
            self.assertLessEqual(4*fourth_bound/(n*dv*dv),epsilon*(1+1e-15))
            # Pinsker applied to independent Gaussian location-mixture trials
            # makes a lower-bound threshold proportional to omega^2.
            K=(2+gamma*tau+math.sqrt(2*gamma*tau))/tau
            required=16*nu*nu*(1-2*epsilon)**2*omega*omega/(9*K*K)
            self.assertAlmostEqual(3*K*math.sqrt(required)/(4*nu*omega),1-2*epsilon)
        self.assertLess(previous,1e-7)

    def test_exact_fourier_moments_and_declared_readout_energy_bounds(self):
        # Ito's finite Fourier/moment hierarchy computes moments through order
        # four exactly by a finite matrix exponential. Modes outside [-4,4]
        # cannot reach the requested zero-mode fourth moment in four steps.
        gamma,tau,kappa,c=.8,.7,.6,.9
        modes=range(-4,5);width=9
        index=lambda degree,mode:degree*width+mode+4
        def moments(omega,phase):
            matrix=np.zeros((45,45),dtype=complex)
            initial=np.zeros(45,dtype=complex)
            for degree in range(5):
                for mode in modes:
                    row=index(degree,mode)
                    matrix[row,row]=-gamma*mode*mode+1j*omega*mode
                    if degree:
                        for neighbor in (mode-1,mode+1):
                            if neighbor in modes:matrix[row,index(degree-1,neighbor)]=degree/2
                    else:initial[row]=(1 if mode==0 else 0) if phase is None else np.exp(1j*mode*phase)
            output=expm(tau*matrix)@initial
            return tuple(float(output[index(degree,0)].real/tau**degree) for degree in (1,2,4))
        H2=(2+gamma*tau+math.sqrt(2*gamma*tau))**2/tau**2
        H4=8*((2+gamma*tau)**4+12*gamma*gamma*tau*tau)/tau**4
        for multiple in (1,3,10):
            omega=multiple*2*math.pi/tau
            for phase in (None,math.pi/2):
                mean,second,fourth=moments(omega,phase)
                expected=0 if phase is None else math.expm1(-gamma*tau)*omega/(tau*(gamma*gamma+omega*omega))
                self.assertAlmostEqual(mean,expected,places=12)
                self.assertGreaterEqual(fourth,second*second-1e-12)
                self.assertLessEqual(second,H2/omega**2+1e-12)
                self.assertLessEqual(fourth,H4/omega**4+1e-12)
                noise_variance=kappa/(c*omega**2)
                read_fourth=fourth+6*noise_variance*second+3*noise_variance**2
                uniform_coefficient=H4+6*kappa*H2/c+3*kappa*kappa/c**2
                self.assertLessEqual(read_fourth,uniform_coefficient/omega**4+1e-12)
            target_error=.1
            necessary_energy=16*kappa*(1-2*target_error)**2*omega**2/(9*H2)
            pair_tv_upper=3*math.sqrt(H2*necessary_energy)/(4*math.sqrt(kappa)*omega)
            self.assertAlmostEqual(pair_tv_upper,1-2*target_error,places=13)

    def test_analytical_design_table_by_independent_complex_moments(self):
        report=json.loads((ROOT/'design/results.json').read_text())
        self.assertEqual(report['source_sha256'],hashlib.sha256((ROOT/'design/counts.py').read_bytes()).hexdigest())
        self.assertEqual([r['index'] for r in report['rows']],[1,2,4,8,16,32])
        with mp.workdps(85):
            gamma=tau=mp.mpf(1);nu=mp.mpf(1)/2;epsilon=mp.mpf(1)/10
            def complex_variance(w):
                z=gamma-1j*w
                return mp.re(tau/z-(1-mp.exp(-z*tau))/z**2)/tau**2
            for row in report['rows']:
                w=2*mp.pi*row['index']/tau
                v1,v2=complex_variance(w),complex_variance(2*w)
                mean=lambda omega:mp.re(1j*(mp.exp((-gamma+1j*omega)*tau)-1)/(-gamma+1j*omega))/tau
                mu1,mu2=mean(w),mean(2*w)
                S=lambda v:(1+4*nu**2)*v-v**2+2*nu**4
                expected=dict(variance_omega=v1,variance_double=v2,
                    prepared_mean_omega=mu1,prepared_mean_double=mu2,
                    passive_necessary_threshold=2*(1-2*epsilon)**2/((mp.cosh(1/nu**2)-1)*(v1+v2)**2),
                    passive_sufficient_threshold=4*max(S(v1),S(v2))/(epsilon*(v1-v2)**2),
                    prepared_necessary_threshold=16*nu**2*(1-2*epsilon)**2*w**2/(9*(2+gamma*tau+mp.sqrt(2*gamma*tau))**2/tau**2),
                    prepared_sufficient_threshold=8*(1+nu**2)*mp.log(1/epsilon)/(mu1-mu2)**2)
                for key,value in expected.items():
                    self.assertLessEqual(abs(mp.mpf(row[key])-value),max(1,abs(value))*mp.mpf('1e-40'))
                    if key.endswith('_threshold'):
                        self.assertEqual(row[key.replace('_threshold','_numerical_ceiling')],int(mp.ceil(value)))

    def test_exact_rational_finite_count_separation(self):
        # This one comparison uses exact Fraction arithmetic, not decimal
        # rounding. Alternating arctan series and Machin's identity certify
        # the deliberately coarse pi bracket.
        F=Fraction
        def atan_bracket(x):
            lower=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(4)),F())
            upper=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(3)),F())
            return lower,upper
        l5,u5=atan_bracket(F(1,5));l239,u239=atan_bracket(F(1,239))
        self.assertGreater(16*l5-4*u239,F(157,50))
        self.assertLess(16*u5-4*l239,F(22,7))
        # e>8/3 by its positive Taylor tail; the geometric tail after k=3
        # bounds e<8/3+5/96<11/4. exp(7/3)>10 proves log(10)<7/3.
        self.assertLess(F(8,3)+F(5,96),F(11,4))
        self.assertGreater(sum((F(7,3)**j/F(math.factorial(j)) for j in range(7)),F()),10)
        cosh_constant_upper=(F(11,4)**4+F(3,8)**4)/2-1
        self.assertLess(cosh_constant_upper,28)
        wlo=64*F(157,50);whi=64*F(22,7)
        # a0=1-e^-1>5/8. The numerator increases and both denominator
        # factors increase on this positive interval, giving a safe gap.
        gap_lower=F(5,8)*wlo*(2*wlo*wlo-1)/((1+whi*whi)*(1+4*whi*whi))
        prepared_sufficient_upper=10*F(7,3)/gap_lower**2
        # V(w)<(2-e^-1)/w²<(18/11)/w², so V(w)+V(2w)
        # is at most (5/4)(18/11)/wlo².
        passive_necessary_lower=2*F(4,5)**2*wlo**4/(28*(F(5,4)*F(18,11))**2)
        self.assertLess(prepared_sufficient_upper,10_000_000)
        self.assertGreater(passive_necessary_lower,17_000_000)

    def test_ou_exposure_integrals_and_all_grid_covariances(self):
        def independent_moments(index):
            w=2*math.pi*index
            V=2*quad(lambda s:(1-s)*math.exp(-s)*math.cos(w*s),0,1,epsabs=1e-12)[0]
            # For k=1, s = u-v yields the triangular double-integral weight.
            lag1=quad(lambda s:(1-abs(s))*math.exp(-(1+s))*math.cos(w*(1+s)),-1,1,points=[0],epsabs=1e-12)[0]
            return V,math.e*lag1
        target=np.array(independent_moments(1));v0,f0=independent_moments(0)
        for index in (0,1,2,5,20):
            np.testing.assert_allclose(independent_moments(index),producer.exposed_oscillator_moments(index),atol=5e-13)
        for index in (2,5,20):
            vn,fn=independent_moments(index)
            weights=np.linalg.solve(np.array([[v0,vn],[f0,fn]]),target)
            self.assertTrue(np.all(weights>0))
            np.testing.assert_allclose(weights,producer.mixture_weights(index),rtol=2e-10,atol=1e-12)
            for lag in range(30):
                a=target[0] if lag==0 else target[1]*math.exp(-lag)
                b=weights@[v0,vn] if lag==0 else math.exp(-lag)*(weights@[f0,fn])
                self.assertAlmostEqual(a,b,places=12)
            self.assertGreater(8*math.pi**2*index**2,8*math.pi**2)

    def test_ou_ratio_monotonicity_and_visible_component_entropy(self):
        e=math.exp(-1);C=2*(math.cosh(1)-1)
        ratio=lambda x:C*(1-x)/(2*(e+(2-e)*x))
        for x in (0.,1.,10.,100.):
            step=1e-6*max(1,x)
            derivative=(ratio(x+step)-ratio(x-step))/(2*step)
            self.assertAlmostEqual(derivative,-C/(e+(2-e)*x)**2,delta=1e-8)
        for n in (2,10,100):
            a0,an=producer.mixture_weights(n)
            omega=2*math.pi*n
            # Local algebra for the *ideal* noncompact Gaussian record:
            # V=a0+an, Sf=-Y and E[(Af)^2|Y]=omega² an.
            lower=omega*omega*an/(a0+an)
            self.assertLess(lower,2*omega*omega)
            self.assertGreater(lower,.99*omega*omega)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DetectorReview))
    names=('review/test_review.py','review/REVIEW.txt',
           'theory/detector_recovery.tex','theory/detector_recovery.txt',
           'theory/finite_access.tex','theory/finite_access.txt',
           'countermodels/COUNTERMODELS.txt','countermodels/exact_sampling.py','countermodels/results.json',
           'design/counts.py','design/results.json','design/test_counts.py',
           'design/design.tex','design/DESIGN.txt','design/figure.py',
           'manuscript/entry.tex','manuscript/countermodels.tex',
           'manuscript/literature.tex','manuscript/bibitems.tex',
           'literature/audit.txt','literature/sources.json')
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        status='Independent internal proof and deterministic computation review of the detector addition; no physical experiment, historical priority certification, or external peer review',
        finite_count_certificate=dict(verified=result.wasSuccessful(),
            parameters='gamma=tau=1, noise standard deviation=1/2, equal-prior error=1/10, omega=64*pi versus 128*pi',
            prepared_sufficient_trials=10_000_000,passive_insufficient_trials=17_000_000,
            arithmetic='Exact Fraction inequalities with analytic bounds on pi, exp and log; separate from floating-point design displays',
            scope='Independent exposure trials, known hypotheses and calibration, stationary versus oriented preparation; not total resource cost'),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
    (ROOT/'review/results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(not result.wasSuccessful())
