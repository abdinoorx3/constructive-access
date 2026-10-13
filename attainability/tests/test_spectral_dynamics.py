from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
import mpmath as mp
import numpy as np
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import spectral_dynamics as sd


def ring(n,forward,backward):
    Q=np.zeros((n,n))
    for i in range(n):
        Q[i,(i+1)%n]+=forward;Q[i,(i-1)%n]+=backward
        Q[i,i]-=forward+backward
    return Q,np.ones(n)/n,np.cos(2*np.pi*np.arange(n)/n)


class SpectralDynamicsTests(unittest.TestCase):
    def test_exact_population_and_arbitrary_parameter(self):
        out=sd.population_gap([F(1,2),F(19,50),F(9,50)],F(19,25))
        self.assertEqual(out["defect"],F(68,625))
        self.assertEqual(out["probability_score_gap"],F(34,625))
        neg=sd.population_gap([1,-F(1,2),0],-F(1,2))
        self.assertEqual(neg["defect"],F(1,4))

    def test_nonlinear_series_high_precision_enclosure(self):
        mp.mp.dps=100
        for u in [F(0),F(1,10000),F(1,8),F(3,4),F(99,100)]:
            lo,hi=sd.phi_interval(u,32)
            x=mp.mpf(u.numerator)/u.denominator
            target=0 if x==0 else 2*mp.sqrt(x)*mp.atanh(mp.sqrt(x))
            self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator,target+mp.mpf('1e-95'))
            self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator,target-mp.mpf('1e-95'))
            self.assertGreaterEqual(lo,2*u)
            if u:self.assertLess(sd.phi_interval(u,33)[1]-sd.phi_interval(u,33)[0],hi-lo)

    def test_bridge_constants_and_scaling(self):
        b=sd.from_gap(F(34,625),range_bound=2)
        self.assertEqual(b["linear_rate_lower"],F(136,625))
        self.assertGreater(b["entropy_production_rate_lower"],b["linear_rate_lower"])
        self.assertEqual(sd.from_gap(F(34,625)/4,range_bound=1)["entropy_production_rate_lower"],b["entropy_production_rate_lower"])
        self.assertEqual(sd.from_gap(F(34,625),lag_unit=2)["entropy_production_rate_lower"],b["entropy_production_rate_lower"]/2)

    def test_zero_and_incompatible_defects_are_not_silently_clipped(self):
        self.assertEqual(sd.from_gap(-1)["entropy_production_rate_lower"],0)
        for g in [F(1,2),F(3,4)]:
            b=sd.from_gap(g)
            self.assertEqual(b["status"],"incompatible_with_finite_entropy_on_coverage_event")
            self.assertEqual(b["normalized_defect"],2*g)
            self.assertIsNone(b["nonlinear_formula_interval"])
        for args in [dict(gap_lower=1,lag_unit=0),dict(gap_lower=1,range_bound=0),dict(gap_lower=.1),dict(gap_lower=1,terms=0)]:
            with self.assertRaises((ValueError,TypeError)):sd.from_gap(**args)

    def test_random_stationary_models_proof_chain(self):
        rng=np.random.default_rng(772119)
        positives=0
        for n in [3,4,5,8,12]:
          for rep in range(12):
            # Symmetric conductances plus a conserved directed cycle ensure
            # stationarity even with nonuniform pi and inhomogeneous rates.
            pi=rng.dirichlet(np.ones(n)*8)
            symmetric=rng.uniform(.001,.004,(n,n))
            flux=(symmetric+symmetric.T)/2;np.fill_diagonal(flux,0)
            current=rng.uniform(.1,.8)
            for i in range(n):flux[i,(i+1)%n]+=current
            Q=flux/pi[:,None];np.fill_diagonal(Q,-Q.sum(axis=1))
            f=np.cos(2*np.pi*np.arange(n)/n);t=rng.uniform(.01,.2)
            P=expm(t*Q);s=np.dot(pi*f,P@f)/np.dot(pi,f*f)
            out=sd.inspect_model(Q,pi,f,t,s)
            self.assertAlmostEqual(out["identity_residual"],0,places=12)
            self.assertLessEqual(out["skew_norm"],out["detector_range"]**2*out["average_row_tv_squared"]/4+1e-13)
            self.assertLessEqual(out["nonlinear_rate_lower"]*t,out["endpoint_forward_reverse_kl"]+1e-11)
            self.assertLessEqual(out["endpoint_forward_reverse_kl"],out["path_forward_reverse_kl"]+1e-10)
            positives+=out["defect"]>0
        self.assertGreater(positives,40)

    def test_reversible_models_have_no_positive_defect(self):
        rng=np.random.default_rng(8194)
        for n in [2,5,11]:
            pi=rng.dirichlet(np.ones(n));flow=rng.uniform(.1,1,(n,n));flow=flow+flow.T
            np.fill_diagonal(flow,0);Q=flow/pi[:,None];np.fill_diagonal(Q,-Q.sum(axis=1))
            f=rng.uniform(-1,1,n)
            for s in [-2,-.5,.3,1,3]:
                out=sd.inspect_model(Q,pi,f,.037,s)
                self.assertLessEqual(out["defect"],1e-12)
                self.assertAlmostEqual(out["skew_norm"],0,places=20)

    def test_binary_jeffreys_ingredient_is_sharp(self):
        for v in [.01,.2,.7,.99]:
            p=np.array([1+v,1-v])/2;q=p[::-1]
            halfJ=.5*(np.sum(p*np.log(p/q))+np.sum(q*np.log(q/p)))
            self.assertAlmostEqual(halfJ,2*v*np.arctanh(v),places=12)

    def test_sharp_kernel_is_not_a_continuous_time_kernel(self):
        f=np.array([1.,1.,-1.,-1.]);pi=np.ones(4)/4
        for z in [.01,.2,.75,.99]:
            P=np.zeros((4,4))
            for i in range(4):
                P[i,(i+1)%4]=(1+z)/2;P[i,(i-1)%4]=(1-z)/2
            C=[np.dot(pi*f,np.linalg.matrix_power(P,k)@f) for k in range(3)]
            self.assertAlmostEqual(C[1],0,places=14)
            self.assertAlmostEqual(-C[2],z*z,places=14)
            active=P>0
            endpoint=np.sum(pi[:,None]*np.where(active,P*np.log(np.divide(P,P.T,out=np.ones_like(P),where=active)),0))
            self.assertAlmostEqual(endpoint,2*z*np.arctanh(z),places=13)
            self.assertAlmostEqual(np.linalg.det(P),-z*z,places=13)

    def test_vector_detector_uses_joint_diameter_not_sum_of_ranges(self):
        Q,pi,f=ring(20,2.06125923,.279824877);t=1.
        P=expm(Q*t);s=np.dot(pi*f,P@f)/np.dot(pi,f*f)
        vector=np.column_stack([f,np.sin(2*np.pi*np.arange(20)/20)])
        single=sd.inspect_model(Q,pi,f,t,s)
        joint=sd.inspect_model(Q,pi,vector,t,s)
        self.assertAlmostEqual(single["detector_range"],joint["detector_range"],places=14)
        self.assertAlmostEqual(joint["defect"],2*single["defect"],places=13)
        self.assertAlmostEqual(joint["linear_rate_lower"],2*single["linear_rate_lower"],places=13)
        self.assertLess(joint["nonlinear_rate_lower"],joint["entropy_rate"])

    def test_ring_identity_and_linear_bound(self):
        Q,pi,f=ring(20,2.06125923,.279824877);t=1
        theta=2*np.pi/20;gamma=(2.06125923+.279824877)*(1-np.cos(theta))
        omega=(2.06125923-.279824877)*np.sin(theta)
        s=np.exp(-gamma*t)*np.cos(omega*t)
        out=sd.inspect_model(Q,pi,f,t,s)
        self.assertAlmostEqual(out["symmetric_residual"],0,places=24)
        self.assertAlmostEqual(out["defect"],.5*np.exp(-2*gamma*t)*np.sin(omega*t)**2,places=12)
        self.assertLess(out["nonlinear_rate_lower"],out["entropy_rate"])
        self.assertGreater(out["nonlinear_rate_lower"],.22)

    def test_joint_coverage_event_reused_without_extra_alpha(self):
        from constructive_access.calibrated.joint_calibration import certify_counts
        old=certify_counts([4000,8000,8000],[3000,5520,4720])
        bridge=sd.from_gap(old["reversible_gap_lower"])
        self.assertGreater(float(bridge["entropy_production_rate_lower"]),.1)
        self.assertGreater(bridge["entropy_production_rate_lower"],100*old["entropy_production_rate_lower"])
        self.assertEqual(old["failure_upper"],F(1,20))

    def test_noisy_analog_diagonal_is_conservative(self):
        c0,c1,c2=F(1,2),F(19,50),F(9,50);s=F(19,25)
        ideal=sd.population_gap([c0,c1,c2],s)["defect"]
        for noise in [F(0),F(1,10),F(1,2)]:
            observed=sd.population_gap([c0+noise,c1,c2],s)["defect"]
            self.assertEqual(ideal-observed,s*s*noise)
            self.assertLessEqual(observed,ideal)

    def test_stationarity_is_essential_negative_control(self):
        # The generator itself is reversible, but the record starts in state0.
        Q=np.array([[-1.,1.],[1.,-1.]])
        f=np.array([.5,1.]);initial=np.array([1.,0.]);P=expm(Q*.4)
        C=[np.dot(initial*f,np.linalg.matrix_power(P,k)@f) for k in range(3)]
        defect=-C[0]+2*C[1]-C[2]
        self.assertGreater(defect,.03)
        self.assertAlmostEqual(defect,.125*(1-np.exp(-.8))**2,places=14)
        self.assertEqual(sd.inspect_model(Q,np.array([.5,.5]),f,.4,1)["entropy_rate"],0)

    def test_odd_variable_equilibrium_is_outside_contract(self):
        # Equilibrium inertial OU: position is even, velocity is odd. Applying
        # ordinary all-even reversal to its oscillatory position record would
        # incorrectly call its physical equilibrium dissipation positive.
        zeta=.2;M=np.array([[0.,1.],[-1.,-zeta]]);t=1.
        corr=lambda gap:np.exp(-1)*np.sinh(expm(M*gap)[0,0])
        C=[corr(k*t) for k in range(3)];s=C[1]/C[0]
        defect=-s*s*C[0]+2*s*C[1]-C[2]
        self.assertGreater(defect,.1)


if __name__=="__main__":unittest.main()
