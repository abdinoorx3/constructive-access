"""Independent adversarial tests; no implementation helpers imported.

Exact fractions independently validate signs; public APIs are checked against
separately derived reference calculations. SciPy augmented semigroups validate random
finite CTMC positivity and separately derived robustness inequalities.
"""
from fractions import Fraction as F
import json
from itertools import product
from math import factorial, comb
from pathlib import Path
import unittest
import sys
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def mm(A, B):
    return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]


def identity(n):
    return [[F(int(i==j)) for j in range(n)] for i in range(n)]


def as_fraction_matrix(A):
    return [[F(x) for x in row] for row in A]


def norm_inf(A):
    return max(sum(abs(x) for x in row) for row in A)


def weighted_series(A, T, degree=0, terms=96):
    """Integral of ((degree+1)t^degree/T^(degree+1)) exp(tA).

    Exact polynomial sum and rigorous max-row-sum remainder bound. Remainder
    uses weight <= probability measure, then bounds each exp-series term.
    """
    A=as_fraction_matrix(A); T=F(T); n=len(A)
    power=identity(n); result=[[F(0) for _ in range(n)] for _ in range(n)]
    for k in range(terms+1):
        coefficient=T**k*F(degree+1, (degree+k+1)*factorial(k))
        result=[[result[i][j]+coefficient*power[i][j] for j in range(n)] for i in range(n)]
        power=mm(power,A)
    x=T*norm_inf(A)
    ratio=x/F(terms+2)
    if ratio>=1: raise ValueError('Increase series length')
    radius=x**(terms+1)/factorial(terms+1)/(1-ratio)
    return result,radius


def exp_series(A, T, terms=96):
    A=as_fraction_matrix(A); T=F(T); n=len(A)
    power=identity(n); result=[[F(0) for _ in range(n)] for _ in range(n)]
    for k in range(terms+1):
        coefficient=T**k/factorial(k)
        result=[[result[i][j]+coefficient*power[i][j] for j in range(n)] for i in range(n)]
        power=mm(power,A)
    x=T*norm_inf(A); ratio=x/F(terms+2)
    if ratio>=1: raise ValueError('Increase series length')
    return result,x**(terms+1)/factorial(terms+1)/(1-ratio)


def exact_contrast(A,T,j,i,degree=0):
    M,r=weighted_series(A,T,degree)
    c=M[j][j]-M[i][j]
    return c-2*r,c+2*r


def generator(rates):
    Q=np.array(rates,dtype=float); np.fill_diagonal(Q,0)
    np.fill_diagonal(Q,-Q.sum(axis=1)); return Q


def integrated(Q,T):
    n=len(Q); A=np.zeros((2*n,2*n)); A[:n,:n]=Q; A[:n,n:]=np.eye(n)
    return expm(T*A)[:n,n:]


def report():
    cycle=[[-1,1,0],[0,-1,1],[1,0,-1]]
    lo,hi=exact_contrast(cycle,F(5),0,2,degree=1)
    v1,r1=weighted_series(cycle,F(3,2)); v2,r2=weighted_series(cycle,F(5,2))
    a,b=F(3,2),F(5,2)
    shifted=(b*(v2[0][0]-v2[2][0])-a*(v1[0][0]-v1[2][0]))/(b-a)
    shifted_radius=2*(b*r2+a*r1)/(b-a)
    A=[[F(-1,10),-2],[2,F(-1,10)]]
    K,r=weighted_series(A,F(9,4)); q=K[0][0]
    return {
      'status':'independent exact and numerical model checks; no experimental data',
      'quantum_uniform':{'bounds':[str(q-r),str(q+r)],'float':float(q)},
      'increasing_clock_classical':{'bounds':[str(lo),str(hi)],'float':float((lo+hi)/2)},
      'shifted_uniform_classical':{'bounds':[str(shifted-shifted_radius),str(shifted+shifted_radius)],'float':float(shifted)},
      'time_inhomogeneous_classical':{'analytic':'(2*(1-exp(-4))/4 + exp(-4)-1)/2','float':float((2*(1-np.exp(-4))/4+np.exp(-4)-1)/2)},
      'random_validation':{'seed':81017,'finite_generators':120,'positive_horizons_per_generator':5,'matched_prep_delay_nonstationary_and_shared_drift_trials':120,'general_generator_coupling_seed':11903,'general_generator_coupling_trials':100,'coarse_hazard_seed':817211,'coarse_hazard_trials':120,'coarse_hazard_horizons':3,'clock_lp_seed':432117,'clock_lp_menus':240,'independent_linprog_objective_comparisons':480},
      'independent_tests':26,
      'dimension_validation':{'three_state_generators':729,'off_diagonal_rate_grid':[0,1,2],'horizons':[.01,.5,2.],'separated_binary_detector_comparisons':26244,'row_pair_operator_checks':6561,'shared_latency_cases':60,'shared_latency_seed':671244,'orthogonal_quantum_score_lower':'261/250','tilted_quantum_score_lower':'27/25','tilted_bloch_axis':['0','-12/37','35/37'],'tilted_dimension_records':6000,'tilted_dimension_successes_threshold':4560,'tilted_dimension_score_bias':'1/500','scope':'arbitrary fixed preparations and stochastic detector; common homogeneous classical generator; averaged endpoint records'},
      'statistics_review':{'fixed_example_records':600,'fixed_example_threshold':321,'binomial_reference':'independent combinatorial integer polynomial','adaptive_null_path_horizon':10,'fixed_confidence_coverage_records':20,'fixed_confidence_true_probability_grid':'1/10,...,9/10','anytime_confidence_path_horizon':12,'cross_layer_example':'343/600 synthetic counts; independently checked beta quantile, calibration signs, reset inverse and hazard root'},
      'additional_certificates':['seven exact sharp detector extremizers','full robust current interval against exact record-law enclosure','coherent +Y comparison gap >257/1000 by exact Taylor and independent Liouville propagation'],
      'exact_method':'96-order rational matrix Taylor; exponential norm geometric remainder; no shared implementation helpers'
    }


class IndependentPhysicalTests(unittest.TestCase):
    def test_quantum_negative_exact_and_independent_semigroup(self):
        A=[[F(-1,10),-2],[2,F(-1,10)]]
        M,r=weighted_series(A,F(9,4)); self.assertLess(M[0][0]+r,F(-4,25))
        v=integrated(np.array(A,dtype=float),2.25)[0,0]/2.25
        self.assertAlmostEqual(float(M[0][0]),v,places=13)

    def test_quantum_full_lindblad_matches_bloch(self):
        I=np.eye(2); X=np.array([[0,1],[1,0]],complex)
        Z=np.diag([1,-1]).astype(complex); H=X
        # Direct column-vectorization depolarizing Lindbladian, not Bloch code.
        L=-1j*(np.kron(I,H)-np.kron(H.T,I))+.1*(np.outer((I/2).reshape(4,order='F'),I.reshape(4,order='F'))-np.eye(4))
        A=np.zeros((8,8),complex); A[:4,:4]=L; A[:4,4:]=np.eye(4)
        integral=expm(A*2.25)[:4,4:]/2.25
        evolved=(integral@Z.reshape(4,order='F')).reshape((2,2),order='F')
        self.assertLess(float(evolved[0,0].real),-.16)
        self.assertLess(abs(evolved[0,0].imag),1e-13)

    def test_positive_singletons_across_random_reducible_networks(self):
        rng=np.random.default_rng(81017)
        for trial in range(120):
            n=2+trial%7; rates=rng.integers(0,5,(n,n))/4
            if trial%3==0: rates[-1,:]=0 # include absorbing state
            Q=generator(rates); i,j=rng.choice(n,2,replace=False)
            for T in [.001,.1,.7,2,8]:
                H=integrated(Q,T)[j,j]-integrated(Q,T)[i,j]
                lower=quad(lambda s:np.exp(Q[j,j]*s+Q[i,i]*(T-s)),0,T)[0]
                self.assertGreater(H,0)
                self.assertGreaterEqual(H+2e-12,lower)
                self.assertLessEqual(H,T+2e-12)

    def test_fixed_time_counterexample_exact(self):
        cycle=[[-1,1,0],[0,-1,1],[1,0,-1]]
        M,r=exp_series(cycle,2)
        self.assertLess(M[0][0]-M[2][0]+2*r,0)
        lo,hi=exact_contrast(cycle,2,0,2)
        self.assertGreater(lo,0)

    def test_increasing_clock_counterexample_exact(self):
        lo,hi=exact_contrast([[-1,1,0],[0,-1,1],[1,0,-1]],5,0,2,degree=1)
        self.assertLess(hi,0)
        self.assertLess(hi,F(-1,10000))

    def test_shifted_uniform_counterexample_exact(self):
        cycle=[[-1,1,0],[0,-1,1],[1,0,-1]]
        a,b=F(3,2),F(5,2); A,ra=weighted_series(cycle,a); B,rb=weighted_series(cycle,b)
        c=(b*(B[0][0]-B[2][0])-a*(A[0][0]-A[2][0]))/(b-a)
        r=2*(b*rb+a*ra)/(b-a)
        self.assertLess(c+r,F(-3,100))

    def test_coarse_detector_counterexample_exact(self):
        # j=0, i=1, B={0,2}: state1 enters B at state2 rather than reset0.
        Q=[[-2,0,0,2],[0,-2,2,0],[0,0,0,0],[0,0,0,0]]
        M,r=weighted_series(Q,1)
        c=M[0][0]+M[0][2]-M[1][0]-M[1][2]
        self.assertLess(c+4*r,0)

    def test_time_inhomogeneous_counterexample(self):
        # Same scheduled generator for both preparations; singleton detector.
        Q1=generator([[0,0,4],[0,0,0],[0,0,0]])
        Q2=generator([[0,0,0],[4,0,0],[0,0,0]])
        M=(integrated(Q1,1)+expm(Q1)@integrated(Q2,1))/2
        self.assertLess(M[0,0]-M[1,0],-.24)

    def test_matched_preparation_and_delay_bounds_random(self):
        rng=np.random.default_rng(81017)
        for trial in range(120):
            n=3+trial%5; Q=generator(rng.uniform(.05,2,(n,n)))
            A=Q.T.copy(); A[-1,:]=1; rhs=np.zeros(n);rhs[-1]=1
            pi=np.linalg.solve(A,rhs); i,j=rng.choice(n,2,replace=False)
            a=Q[i,j]*.7; b=Q[j,i]*.6; r=.1+.8*rng.random(); s=1-r
            L=np.zeros_like(Q);L[i,j]=a;L[i,i]=-a;L[j,i]=b;L[j,j]=-b
            R=Q-s*L; T=.01+2*rng.random(); delay=.8*rng.random()
            rho=.75*pi+.25*np.eye(n)[rng.integers(n)]; eta=np.abs(rho-pi).sum()/2
            diff=expm(R*T)-expm(Q*T); endpoint=np.eye(n)[:,j]
            C=(rho-pi)@diff@endpoint
            cap=min(eta*s*(a+b)*T,2*eta*(-np.expm1(-s*max(a,b)*T)))
            self.assertLessEqual(abs(C),cap+2e-12)
            delayed=expm(Q*delay)@endpoint; err=abs(pi@diff@(delayed-endpoint))
            current=pi[i]*a-pi[j]*b
            omega=np.ptp(delayed-endpoint)
            cap1=s*abs(current)*T*omega
            cap2=s*abs(current)*T*((-np.expm1(Q[j,j]*delay))+(-np.expm1(-max(Q[x,j] for x in range(n) if x!=j)*delay)))
            self.assertLessEqual(err,cap1+2e-12);self.assertLessEqual(cap1,cap2+2e-12)
            # Arbitrary prepared-ensemble instantaneous current, no stationarity.
            jr=rho[i]*a-rho[j]*b
            ki=max(-R[i,i],max(R[x,i] for x in range(n) if x!=i))
            kj=max(-R[j,j],max(R[x,j] for x in range(n) if x!=j))
            beta=max(Q[x,j] for x in range(n) if x!=j)
            current_error=abs(-(rho@diff@endpoint)/(s*T)-jr)
            current_bound=T/2*(a*ki+b*kj+max(a,b)*(-Q[j,j]+beta))
            self.assertLessEqual(current_error,current_bound+2e-12)
            # Same additive native/control generator drift: difference is O(T²).
            E=generator(rng.uniform(0,.04,(n,n)))
            drift_error=abs(rho@(expm((Q+E)*T)-expm((R+E)*T)+diff)@endpoint)
            kappa=np.max(np.abs(E).sum(axis=1))/2
            drift_bound=s*(a+b+2*max(a,b))*kappa*T*T/2
            self.assertLessEqual(drift_error,drift_bound+3e-12)

    def test_general_generator_coupling_bound_random(self):
        rng=np.random.default_rng(11903)
        for trial in range(100):
            n=2+trial%6; Q=generator(rng.uniform(0,2,(n,n))); R=generator(rng.uniform(0,2,(n,n)))
            kappa=np.max(np.abs(Q-R).sum(axis=1))/2
            t=.01+rng.random(); distance=np.abs(expm(Q*t)-expm(R*t)).sum(axis=1)/2
            self.assertLessEqual(max(distance),-np.expm1(-kappa*t)+2e-12)


    def test_sharp_detector_bound_independent_exact_matrix(self):
        import detector_boundary as detector
        cases=[(F(a),F(b),F(t),F(eta)) for a,b,t,eta in
               [(0,2,1,1),(2,0,1,1),(1,1,2,F(1,2)),(1,1,2,1),
                (2,F(7,10),3,F(3,10)),(1,1,1,0),(3,2,F(1,10),F(9,10))]]
        for a,b,t,eta in cases:
            Q=[[-a,a*(1-eta),a*eta,0],[0,-b,0,b],[0,0,0,0],[0,0,0,0]]
            M,r=weighted_series(Q,t)
            h=t*(M[1][1]+M[1][2]-M[0][1]-M[0][2]); radius=4*t*r
            bound=detector.bounded_hazard_response(a,b,t,eta,terms=96)['lower_bound_interval']
            self.assertLessEqual(h-radius,bound[1])
            self.assertGreaterEqual(h+radius,bound[0])
            self.assertLess(float(bound[1]-bound[0]),1e-25)

    def test_arbitrary_coarse_bound_random_networks(self):
        rng=np.random.default_rng(817211)
        for trial in range(120):
            n=3+trial%7; Q=generator(rng.uniform(0,3,(n,n)))
            B=np.zeros(n);B[:1+trial%(n-1)]=1
            inside=np.flatnonzero(B);outside=np.flatnonzero(1-B)
            b=max(Q[x,outside].sum() for x in inside)
            a=max(Q[x,inside].sum() for x in outside)
            for t in [.01,.2,1.1]:
                U=integrated(Q,t)@B
                actual=U[inside[0]]-U[outside[0]]
                lower=(-np.expm1(-a*t))/a+(-np.expm1(-b*t))/b-t
                self.assertGreaterEqual(actual+3e-12,lower)

    def test_statistics_integer_tails_against_combinatorial_polynomial(self):
        import statistics_design as stats
        n=600; k=321
        for p in (F(1001,2000),F(1143,2000)):
            a,b=p.numerator,p.denominator
            exact=F(sum(comb(n,s)*a**s*(b-a)**(n-s) for s in range(k,n+1)),b**n)
            lo,hi=stats.binomial_upper_bounds(n,k,p)
            self.assertLessEqual(lo,exact);self.assertGreaterEqual(hi,exact)
        plan=stats.exact_fixed_plan(F(18,125),F(1,1000),records=n)
        self.assertEqual(plan['successes_threshold'],321)
        self.assertLess(plan['false_rejection_interval'][1],F(1,20))
        self.assertLess(plan['miss_interval'][1],F(1,20))
        tampered=dict(plan);tampered['successes_threshold']=320
        self.assertFalse(stats.verify_exact_fixed(tampered,F(18,125),F(1,1000)))

    def test_statistics_adaptive_null_optional_stopping_exact(self):
        # Enumerate a history-dependent non-iid null. Radius chosen from the
        # PREVIOUS bit, so this checks conditional validity rather than iid.
        import statistics_design as stats
        from copy import deepcopy
        alpha=F(1,4); horizon=10
        initial=stats.AnytimeContrast(alpha=alpha,fractions=(F(1,4),F(3,4)))
        active=[(F(1),initial,0)]; crossed=F(0)
        for t in range(horizon):
            new=[]
            for mass,process,last in active:
                radius=F(last,5); theta0=(1+radius)/2
                # Alternate boundary and strict null probabilities.
                p=theta0 if (t+last)%2 else theta0-F(1,10)
                previous=process.e_value; expected=F(0)
                for bit,prob in [(0,1-p),(1,p)]:
                    child=deepcopy(process); child.update(bit,radius)
                    expected+=prob*child.e_value
                    if child.crossed: crossed+=mass*prob
                    else:new.append((mass*prob,child,bit))
                self.assertLessEqual(expected,previous)
            active=new
        self.assertLessEqual(crossed,alpha)

    def test_statistics_invalid_clock_conditioning_breaks_null(self):
        # In the directed cycle, averaging the whole initial window is valid,
        # but choosing an arm-clock-dependent bet only at late times is not.
        Q=np.array([[-1,1,0],[0,-1,1],[1,0,-1]],float)
        p0,p1=expm(2*Q)[0,0],expm(2*Q)[2,0]
        encoded=(1+p1-p0)/2
        self.assertGreater(encoded,.5)



    def test_clock_design_against_independent_lp(self):
        import clock_design as clocks
        from scipy.optimize import linprog
        from copy import deepcopy
        rng=np.random.default_rng(432117)
        for trial in range(240):
            n=2+trial%9
            costs=[F(int(v),3) for v in rng.integers(0,22,n)]
            values=[F(int(v),20) for v in rng.integers(-17,18,n)]
            widths=[F(int(v),100) for v in rng.integers(0,5,n)]
            budget=min(costs)+F(int(rng.integers(0,40)),7)
            menu=[dict(label=str(i),horizon=F(i+1),cost=costs[i],
                       contrast_lower=-values[i]-widths[i],contrast_upper=-values[i]) for i in range(n)]
            result=clocks.design(menu,budget)
            self.assertTrue(clocks.verify(menu,budget,result))
            for name,coeff in [('guaranteed',values),('optimistic',[v+w for v,w in zip(values,widths)])]:
                external=linprog(-np.array(coeff,dtype=float),A_ub=[np.array(costs,dtype=float)],
                                 b_ub=[float(budget)],A_eq=[np.ones(n)],b_eq=[1.],bounds=[(0,None)]*n,method='highs')
                self.assertTrue(external.success)
                self.assertAlmostEqual(float(result[name]['value']),-external.fun,places=10)
                self.assertLessEqual(sum(w>0 for w in result[name]['weights']),2)
            tampered=deepcopy(result);tampered['guaranteed']['dual_intercept']+=F(1,100)
            self.assertFalse(clocks.verify(menu,budget,tampered))



    def test_coherent_comparison_exact_and_full_liouville(self):
        A=[[F(-1,10),2],[-2,F(-1,10)]]
        M,r=weighted_series(A,F(7,4))
        gap=F(9,20)*(M[0][1]-M[0][0])
        self.assertGreater(gap-F(9,10)*r,F(257,1000))
        I=np.eye(2); X=np.array([[0,1],[1,0]],complex)
        Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);H=X
        L=-1j*(np.kron(I,H)-np.kron(H.T,I))+.1*(np.outer((I/2).reshape(4,order='F'),I.reshape(4,order='F'))-np.eye(4))
        B=np.zeros((8,8),complex);B[:4,:4]=L;B[:4,4:]=np.eye(4)
        integrated_channel=expm(B*1.75)[:4,4:]/1.75
        difference=(integrated_channel@((Y-Z)/2).reshape(4,order='F')).reshape((2,2),order='F')
        self.assertAlmostEqual(float(gap),.9*difference[0,0].real,places=13)

    def test_uniform_marginal_clock_is_not_enough(self):
        # Classical symmetric unit-rate 2-state chain. Both arms have uniform
        # [0,2] clock marginals, coupled differently to their first jump time.
        # Shared latent random seeds can implement this coupling at time0.
        from scipy.optimize import brentq
        T=2.
        jroot=brentq(lambda s:T*(-np.expm1(-s))-s,.01,3.)
        iroot=brentq(lambda s:T*np.exp(-s)-s,.01,3.)
        pj=np.exp(-jroot)+quad(lambda s:.5*np.exp(-s)*(1-np.exp(-2*(T*(-np.expm1(-s))-s))),0,jroot,epsabs=1e-12)[0]
        pi=quad(lambda s:.5*np.exp(-s)*(1+np.exp(-2*(T*np.exp(-s)-s))),0,iroot,epsabs=1e-12)[0]
        self.assertLess(pj-pi,-.018)



    def test_complete_current_certificate_against_exact_record_law(self):
        import robust_physics as physics
        native=[[-F(401,100),F(401,100)],[F(4),-F(4)]]
        controlled=[[-F(5,2),F(5,2)],[F(88,25),-F(88,25)]]
        post=[[-4,4],[4,-4]]
        P,rp=exp_series(native,F(1,10));R,rr=exp_series(controlled,F(1,10))
        D,rd=exp_series(post,F(1,100))
        rho_n=[[F(51,100),F(49,100)]];rho_c=[[F(64,125),F(61,125)]]
        center=F(4,5)*(mm(mm(rho_n,P),D)[0][1]-mm(mm(rho_c,R),D)[0][1])
        radius=F(4,5)*(rp+rr+2*rd+(rp+rr)*rd)
        certificate=physics.robust_current_interval((center-radius,center+radius),
            duration=(F(1,10),F(1,10)),retained_fraction=F(1,2),
            forward_rate_cap=F(3),reverse_rate_cap=F(1),pulse_exit_cap=F(4),
            preparation_tv=F(1,100),preparation_mismatch_tv=F(1,500),
            native_generator_error_integral=F(1,1000),controlled_generator_error_integral=F(1,500),
            delay=F(1,100),delay_exit_cap=F(4),delay_inflow_cap=F(4),
            detector_gain=(F(4,5),F(4,5)))
        low,high=map(F,certificate['current_interval'])
        self.assertLessEqual(low,F(1));self.assertGreaterEqual(high,F(1))
        self.assertEqual(certificate['status'],'positive_current_certified')



    def test_fixed_confidence_coverage_and_cross_layer_signs(self):
        import statistics_design as stats
        import explanation
        from scipy.stats import beta
        from scipy.optimize import brentq
        alpha=F(1,20);n=20
        bounds=[stats.fixed_lower_bound(n,k,alpha,bits=12)['success_probability_lower'] for k in range(n+1)]
        for p in [F(k,10) for k in range(1,10)]:
            error=sum((F(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1) if bounds[k]>p),F(0))
            self.assertLessEqual(error,alpha)
        output=explanation.from_counts(600,343,1,1,F(9,4),F(9,10),F(1,1000))
        lower=output['statistics']['success_probability_lower']
        exact_cp=beta.ppf(.05,343,600-343+1)
        self.assertLessEqual(float(lower),exact_cp+1e-13)
        self.assertLess(exact_cp-float(lower),2**-24+1e-13)
        c=(1-2*lower+F(1,1000))/F(9,10)
        self.assertEqual(output['physical_contrast_upper'],c)
        t=2.25;g=t*np.exp(-t);k=t-2*(-np.expm1(-t))+g
        expected_eta=(g-t*float(c))/k
        certified_eta=output['reset_explanation']['minimum_reset_defect']
        self.assertLessEqual(float(certified_eta),expected_eta+1e-13)
        self.assertAlmostEqual(float(certified_eta),expected_eta,places=12)
        root=brentq(lambda z:2*(-np.expm1(-z))/z-1-float(c),1e-7,10.)/t
        hazard=output['required_peak_crossing_intensity']['hazard_lower_bound']
        self.assertLessEqual(float(hazard),root+1e-13)
        self.assertAlmostEqual(float(hazard),root,places=12)
        incompatible=explanation.from_counts(600,343,F(1,2),F(1,2),F(9,4),F(9,10),F(1,1000))
        self.assertEqual(incompatible['reset_explanation']['status'],'incompatible_hazard_contract')

    def test_anytime_confidence_coverage_exact_path_recursion(self):
        import statistics_design as stats
        alpha=F(1,4);f=F(1,3);horizon=12
        for p in [F(1,3),F(1,2),F(2,3)]:
            active={0:F(1)};crossed=F(0)
            for n in range(1,horizon+1):
                new={}
                for s,mass in active.items():
                    for bit,prob in [(0,1-p),(1,p)]:
                        count=s+bit
                        lower=stats.anytime_lower_bound(n,count,alpha,f,bits=12)['success_probability_lower']
                        if lower>p:crossed+=mass*prob
                        else:new[count]=new.get(count,F(0))+mass*prob
                active=new
            self.assertLessEqual(crossed,alpha)



    def test_three_state_dimension_bound_exhaustive_rate_grid(self):
        offdiagonal=[(i,j) for i in range(3) for j in range(3) if i!=j]
        cases=0
        for rates in product(range(3),repeat=6):
            Q=np.zeros((3,3))
            for (i,j),rate in zip(offdiagonal,rates):Q[i,j]=rate
            np.fill_diagonal(Q,-Q.sum(axis=1))
            for T in [.01,.5,2.]:
                R=integrated(Q,T)/T
                # Every pair of rows of I-R has TV norm<=1. This is the
                # supremum over ALL fixed f∈[0,1], and prep mixtures contract.
                B=np.eye(3)-R
                for i in range(3):
                    for j in range(i):
                        self.assertLessEqual(np.abs(B[i]-B[j]).sum()/2,1+2e-12)
                # Exhaust all nonempty proper binary detectors and separated
                # pure preps; mixtures follow by convexity.
                for bits in product([0,1],repeat=3):
                    if not 0<sum(bits)<3:continue
                    f=np.array(bits)
                    for i in range(3):
                        for j in range(3):
                            if bits[i] and not bits[j]:
                                contrast=(R[i]-R[j])@f
                                self.assertGreater(contrast,0)
                                cases+=1
        self.assertEqual(cases,26244)

    def test_dimension_witness_quantum_and_sharp_classical_models(self):
        # Observed four-arm score S=C0-C_uniform exceeds1 with a noisy qubit.
        A=[[F(-1,10),-2],[2,F(-1,10)]]
        M,r=weighted_series(A,F(9,4))
        score_lower=F(9,10)*(1-M[0][0]-r)
        self.assertGreater(score_lower,F(261,250))
        # Four-state classical model has separated initial preparations and
        # the same negative integrated contrast shape; exact Taylor certificate.
        Q=[[-2,0,0,2],[0,-2,2,0],[0,0,0,0],[0,0,0,0]]
        R,rr=weighted_series(Q,1);f=[1,0,1,0]
        C=sum((R[0][k]-R[1][k])*f[k] for k in range(4))
        self.assertGreater(1-C-4*rr,1)
        # A three-state model approaches the score1 bound from below, with
        # arbitrary fixed stochastic detector and no extremal-prep contract.
        q=generator([[0,0,0],[0,0,200],[0,0,0]])
        detector=np.array([.37,0,1]);reference=np.array([1,0,0]);comparison=np.array([0,1,0])
        score=(reference-comparison)@(np.eye(3)-integrated(q,1))@detector
        self.assertAlmostEqual(score,1-(-np.expm1(-200))/200,places=13)
        self.assertGreater(score,.99);self.assertLess(score,1)

    def test_dimension_witness_shared_latency(self):
        rng=np.random.default_rng(671244)
        for _ in range(60):
            Q=generator(rng.uniform(0,4,(3,3)))
            mu=rng.dirichlet(np.ones(3));nu=rng.dirichlet(np.ones(3));f=rng.uniform(0,1,3)
            delay=.01+rng.random();T=.01+2*rng.random()
            common=expm(Q*delay)
            score=(mu-nu)@common@(np.eye(3)-integrated(Q,T)/T)@f
            self.assertLessEqual(abs(score),1+2e-12)



    def test_tilted_quantum_dimension_score_exact_and_full_gksl(self):
        import preparation
        A=[[F(-1,10),2],[-2,F(-1,10)]]
        M,r=weighted_series(A,F(2))
        y,z,g=F(-12,37),F(35,37),F(9,10)
        score=g*(z*(1-M[0][0])-y*M[0][1])
        error=g*(abs(y)+abs(z))*r
        self.assertEqual(y*y+z*z,1)
        self.assertGreater(score-error,F(27,25))
        lower,upper=preparation.baseline_subtracted_witness_bounds(F(1,10),F(2),F(2),y,z,g)
        self.assertLessEqual(lower,score+error);self.assertGreaterEqual(upper,score-error)
        I=np.eye(2);X=np.array([[0,1],[1,0]],complex)
        Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);H=X
        L=-1j*(np.kron(I,H)-np.kron(H.T,I))+.1*(np.outer((I/2).reshape(4,order='F'),I.reshape(4,order='F'))-np.eye(4))
        B=np.zeros((8,8),complex);B[:4,:4]=L;B[:4,4:]=np.eye(4)
        delta=float(y)*Y+float(z)*Z
        evolved=((expm(B*2)[:4,4:]/2)@delta.reshape(4,order='F')).reshape((2,2),order='F')
        direct=.9*(float(z)-evolved[0,0].real)
        self.assertAlmostEqual(float(score),direct,places=13)

    def test_tilted_four_arm_dimension_statistics_and_encoding(self):
        import statistics_design as stats
        from scipy.stats import binom
        for outcomes in product([0,1],repeat=4):
            witness=outcomes[0]-outcomes[1]-outcomes[2]+outcomes[3]
            encoded=sum(stats.encode_dimension(a,y) for a,y in enumerate(outcomes))/4
            self.assertEqual(encoded,.5+witness/4)
        plan=stats.dimension_plan(F(27,25),F(1,500),6000,4560)
        self.assertTrue(stats.verify_dimension(plan,F(27,25),F(1,500)))
        self.assertEqual(plan['contract']['null_success_upper'],F(1501,2000))
        self.assertEqual(plan['contract']['alternative_success_lower'],F(1539,2000))
        reference_type1=binom.sf(4559,6000,1501/2000)
        reference_miss=binom.cdf(4559,6000,1539/2000)
        for interval,reference in [(plan['false_rejection_interval'],reference_type1),(plan['miss_interval'],reference_miss)]:
            self.assertAlmostEqual(float(interval[0]),reference,places=12)
            self.assertAlmostEqual(float(interval[1]),reference,places=12)
            self.assertLess(interval[1],F(1,20))
        tampered=dict(plan);tampered['witness_radius']=F(0)
        self.assertFalse(stats.verify_dimension(tampered,F(27,25),F(1,500)))


if __name__=='__main__':
    import sys
    if '--report' in sys.argv:
        (ROOT/'results'/'independent.json').write_text(json.dumps(report(),indent=2)+'\n')
    else:
        unittest.main()
