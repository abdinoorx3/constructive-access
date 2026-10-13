"""Independent analytic/numerical cross-checks of detector-calibration claims.

No laboratory experiment or coverage simulation is asserted. Fractions are
exact; mpmath fixtures are high-precision numerical checks of analytic formulas.
Run directly from either the research tree or an unchanged packaged directory.
"""
from fractions import Fraction as F
from pathlib import Path
from math import comb, factorial
import importlib.util
import sys
import unittest
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
em = load('review_exposure_moments', 'experiment/exposure_moments.py')
cert = load('review_certification', 'code/certification.py')

def real(q):
    q = F(q)
    return mp.mpf(q.numerator)/q.denominator

def complex_fourth_exposure(gamma, omega, tau):
    """Separate signed-complex Fourier implementation; mpmath matrix exponential."""
    states = [(a,k) for a in range(5) for k in range(-(4-a),5-a)]
    index = {state:i for i,state in enumerate(states)}
    G = mp.matrix(len(states)); initial = mp.matrix(len(states),1)
    initial[index[(0,0)]] = 1
    for i,(a,k) in enumerate(states):
        G[i,i] = -real(gamma)*k*k + 1j*real(omega)*k
        if a:
            for shift in (-1,1):
                G[i,index[(a-1,k+shift)]] += mp.mpf(a)/2
    vector = mp.expm(real(tau)*G)*initial
    return vector[index[(4,0)]]/real(tau)**4

def pair_moments(gamma, omega, h, tau):
    z = real(gamma)-1j*real(omega)
    t = real(tau)
    variance = mp.re(t/z-(1-mp.exp(-z*t))/z**2)/t**2
    def covariance(gap):
        return mp.re(mp.exp(-z*real(gap))*(mp.cosh(z*t)-1)/z**2)/t**2
    return variance,covariance(h),covariance(2*h)

class IndependentCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = em.Model(F(7,5), F(-9,4), F(1,8), F(1,50), 4, F(4,5))
        cls.table,cls.diagnostics = em.moments(cls.model)

    def test_complex_fourier_fourth_moment_enclosed(self):
        m = self.model
        with mp.workdps(100):
            expected = complex_fourth_exposure(m.gamma,m.omega,m.exposure)
            point,radius = self.table[(0,4,0)]
            self.assertLess(abs(mp.im(expected)), mp.mpf('1e-90'))
            self.assertLessEqual(abs(real(point)-mp.re(expected)), real(radius))
            self.assertGreater(mp.re(expected), 0)
            self.assertLess(mp.re(expected), mp.mpf(3)/8)

    def test_nontrivial_exposure_covariances_enclosed(self):
        m = self.model
        with mp.workdps(100):
            variance,c1,c2 = pair_moments(m.gamma,m.omega,m.lag,m.exposure)
            for alpha,value in (((0,2,0),variance),((1,1,0),c1),((0,1,1),c1),((1,0,1),c2)):
                point,radius = self.table[alpha]
                self.assertLessEqual(abs(real(point)-value),real(radius))

    def test_signed_nonreversible_word_derivative_bound(self):
        # f=(cos(x)+sin(2x)/2)/(3/2), with constant drift 3 and diffusion 1.
        # Its nontrivial harmonics test signed prefix words, without assuming
        # the special Gamma geometry needed by the separate entropy theorem.
        with mp.workdps(70):
            f={1:mp.mpf(1)/3,-1:mp.mpf(1)/3,2:-1j*mp.mpf(1)/6,-2:1j*mp.mpf(1)/6}
            gaps=[mp.mpf(1)/5,mp.mpf(1)/10,mp.mpf(3)/10,mp.mpf(2)/5]
            def word(derivative=None):
                g=dict(f)
                for j in range(len(gaps)-1,-1,-1):
                    evolved={k:v*mp.exp((-k*k+3j*k)*gaps[j])*
                             ((-k*k+3j*k) if derivative==j else 1) for k,v in g.items()}
                    product={}
                    for k,v in evolved.items():
                        for a,c in f.items():product[k+a]=product.get(k+a,0)+v*c
                    g=product
                return g.get(0,0)
            # Triangle bound ||Lf||infinity <= (2 sqrt10 + sqrt52)/3.
            M=(2*mp.sqrt(10)+mp.sqrt(52))/3
            for j in range(4):
                derivative=word(j)
                self.assertLess(abs(mp.im(derivative)),mp.mpf('1e-65'))
                self.assertLessEqual(abs(derivative),min((j+1)**2,(4-j)**2)*M)
            self.assertLess(abs(word()),1)

    def test_uniform_cross_gap_coefficient_independent_integral(self):
        # U=max of p uniforms, V=min of q uniforms. Integrate |V-U|
        # against their independent beta densities, split at V=U.
        with mp.workdps(60):
            for p,q in ((1,1),(1,4),(3,2),(6,1)):
                expected = F(1)-F(1,p+1)-F(1,q+1)+F(2*factorial(p)*factorial(q),factorial(p+q+1))
                integral = mp.quad(lambda u: p*u**(p-1)*(
                    mp.quad(lambda v:(u-v)*q*(1-v)**(q-1),[0,u])+
                    mp.quad(lambda v:(v-u)*q*(1-v)**(q-1),[u,1])),[0,1])
                self.assertLess(abs(integral-real(expected)),mp.mpf('1e-55'))

    def test_multilinear_score_direct_conditional_expectation(self):
        # Independent product law with unequal latent values: verifies signs,
        # replica degrees and symmetric score without using a latent simulator.
        ys = (F(-2,3),F(1,4),F(3,5)); gain = F(7,9)
        m=2; k=2*m+2; h=F(1,7); lam=F(5,4); b=1-lam*h
        means={key:F(0) for key in cert.score_ranges(m,h,lam)}
        for left,plus,right in cert.category_order(m):
            prob=(1+left*gain*ys[0])/2*(1+right*gain*ys[2])/2
            p=(1+gain*ys[1])/2
            prob*=comb(k,plus)*p**plus*(1-p)**(k-plus)
            for key,value in cert.scores((left,plus,right),m,h,lam).items():
                means[key]+=prob*value
        for j in range(m+1):
            expected=-(ys[0]-b*ys[1])*(ys[2]-b*ys[1])*ys[1]**(2*j)/h**2
            self.assertEqual(means[f'S{j}'],gain**(2*j+2)*expected)
        self.assertEqual(means['V'],gain**2*(ys[1]**2-(ys[0]+ys[2])*ys[1]/2)/h)

    def test_reversible_blur_false_positive_and_valid_correction(self):
        h=tau=F(1,10); lam=F(1); b=1-lam*h
        with mp.workdps(100):
            d0,dh,d2=pair_moments(1,0,h,tau)
            raw=-(d2-2*real(b)*dh+real(b*b)*d0)/real(h*h)
            ideal=-mp.mpf('0.5')*(mp.exp(-real(h))-real(b))**2/real(h*h)
            self.assertGreater(raw,mp.mpf('0.1'))
            self.assertLess(ideal,0)
            bounds=cert.bias_bounds(0,h,lam,tau,(1,1,1,1))
            self.assertLessEqual(raw-real(bounds['beta'])-real(bounds['delta_N']),0)

    def test_signed_gain_extrema_and_degenerate_reference(self):
        gain=(F(2,5),F(4,5))
        for interval in ((F(-3),F(-1)),(F(-1),F(2)),(F(1),F(3))):
            result=cert.normalized_interval(interval,4,gain)
            corners=[x/g**4 for x in interval for g in gain]
            self.assertEqual(result,(min(corners),max(corners)))
        calibration=cert.calibrate_references(2,1,2,1,F(1,1000),(F(1,500),F(1,100)),F(1,20))
        self.assertEqual(calibration['status'],'gain_not_separated_from_zero')

    def test_nominal_gain_requires_residual_and_preserves_true_sum(self):
        # Exact population algebra fixture: treating a nominal gain as known
        # would falsely raise a positive contrast when actual gain is one.
        nominal=F(4,5); score=(F(1),F(1)); projection=(F(25,16),F(25,16))
        corrected=cert.projection_lower(projection,[score],nominal,(F(1),F(1)))
        self.assertGreater(projection[0],1)
        self.assertEqual(corrected,1)
        intervals=[(F(1),F(2)),(F(-3),F(-2)),(F(-1),F(4))]
        nominal_means=[F(3,2),F(-5,2),F(1)]
        fixed_projection=sum((x/nominal**(2*j+2) for j,x in enumerate(nominal_means)),F(0))
        lower=cert.projection_lower((fixed_projection,fixed_projection),intervals,nominal,(F(3,4),F(19,20)))
        for gain in (F(3,4),F(4,5),F(9,10),F(19,20)):
            actual=sum((x/gain**(2*j+2) for j,x in enumerate(nominal_means)),F(0))
            self.assertLessEqual(lower,actual)

    def test_prefix_projection_uses_existing_replicas_and_budget(self):
        m=2; gain=F(9,10)
        counts={category:100+index for index,category in enumerate(cert.category_order(m))}
        summary=cert.summarize_counts(counts,m,F(1,25),1,nominal_gain=gain)
        self.assertEqual(summary['readouts_per_record'],8)
        confidence=cert.confidence_statistics(summary,F(1,40))
        self.assertEqual(len(confidence['statistics']),9)
        self.assertEqual(confidence['confidence_event_count'],8)
        self.assertEqual(confidence['statistics']['R0']['alpha'],0)
        self.assertEqual(confidence['statistics']['R0']['interval'],
                         tuple(x/gain**2 for x in confidence['statistics']['S0']['interval']))
        self.assertEqual(sum(s['alpha'] for s in confidence['statistics'].values()),F(1,40))
        for order in range(m+1):
            expected=sum((summary['statistics'][f'S{j}']['sum']/gain**(2*j+2)
                          for j in range(order+1)),F(0))
            self.assertEqual(summary['statistics'][f'R{order}']['sum'],expected)

    def test_fixed_absolute_enclosure_floor_requires_precision_schedule(self):
        # Valid finite intervals are not automatically an asymptotic numerical
        # algorithm. This fixture protects the exact distinction in the review.
        floor = cert.sqrt_upper(F(1, 2**400))
        self.assertEqual(floor, F(1, 2**90))
        amplified = cert.normalized_interval((-floor,floor), 180, (F(1,2),F(1,2)))
        self.assertEqual(amplified, (-F(2**90),F(2**90)))

    def test_tutorial_shared_event_max_uses_stricter_scope(self):
        import json
        tutorial=load('review_calibration_tutorial','research/calibration_tutorial.py')
        for geometry,endpoint,selected in ((F(1,3),F(2,3),'endpoint'),
                                           (F(2,3),F(1,3),'geometry'),
                                           (F(0),F(0),'geometry')):
            result=tutorial.combined_certificate({'entropy_lower':geometry,'alpha':F(1,20)},
                                                  {'entropy_lower':endpoint})
            self.assertEqual(result['entropy_lower'],max(geometry,endpoint))
            self.assertEqual(result['selected_component'],selected)
            self.assertEqual(result['failure_probability_upper'],F(1,20))
            self.assertEqual(result['additional_failure_allocation'],0)
            self.assertIn('stricter geometry contract',result['scope_note'])
        fixture=json.loads((ROOT/'experiment/calibration_counts.json').read_text())
        replay=tutorial.replay(fixture,cert)
        self.assertEqual(replay['combined']['entropy_lower'],
                         max(replay['geometry']['entropy_lower'],replay['endpoint_baseline']['entropy_lower']))
        self.assertTrue(replay['recorded_results_match_exactly'])

    def test_hidden_modulation_geometry_and_generator_identity(self):
        import sympy as sp
        x,z=sp.symbols('x z',real=True)
        gamma,eta,epsilon=sp.symbols('gamma eta epsilon',positive=True)
        c=epsilon*sp.cos(z); f=sp.cos(x)
        def L(g,sign=1):
            return gamma*sp.diff(g,x,2)+sign*c*sp.diff(g,x)+eta*sp.diff(g,z,2)
        target=(gamma**2-c**2)*sp.cos(x)+(2*gamma+eta)*c*sp.sin(x)
        self.assertEqual(sp.trigsimp(L(L(f))-target),0)
        self.assertEqual(sp.trigsimp(L(L(f,-1),-1)-target.subs(x,-x)),0)
        squared_drive=sp.integrate(c*c,(z,0,2*sp.pi))/(2*sp.pi)
        self.assertEqual(squared_drive,epsilon**2/2)
        self.assertEqual(sp.integrate(c,(z,0,2*sp.pi)),0)
        symmetric=sp.simplify((L(f)+L(f,-1))/2)
        k=sp.integrate(L(f)*L(f,-1),(z,0,2*sp.pi))/(2*sp.pi)
        ratio=sp.trigsimp((symmetric**2-k)/(gamma*sp.sin(x)**2))
        self.assertEqual(ratio,epsilon**2/(2*gamma))

    def test_rational_radial_cutoff_and_uniform_hitting_constant(self):
        # C2 joins and independent symbolic maximum bound for q''.
        import sympy as sp
        t=sp.symbols('t',real=True)
        q=1-10*t**3+15*t**4-6*t**5
        for endpoint,value in ((0,1),(1,0)):
            self.assertEqual(q.subs(t,endpoint),value)
            self.assertEqual(sp.diff(q,t).subs(t,endpoint),0)
            self.assertEqual(sp.diff(q,t,2).subs(t,endpoint),0)
        critical=sp.solve(sp.diff(q,t,3),t)
        extrema=[sp.simplify(abs(sp.diff(q,t,2).subs(t,x))) for x in critical]
        self.assertEqual(extrema,[10*sp.sqrt(3)/3]*2)
        self.assertTrue(all(x<6 for x in extrema))
        # pi>3 suffices for the producer's last expectation inequality.
        self.assertEqual(F(9,4)**2/F(4*9),F(9,64))

if __name__=='__main__':
    unittest.main()
