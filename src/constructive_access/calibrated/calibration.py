"""Exact detector-calibrated reversible moment supports and access controls.

The zero-lag calibration observable is a product of conditionally independent
same-state readings; squaring one binary reading is not equivalent.
"""
from fractions import Fraction as F
from math import comb, isqrt
from pathlib import Path
import json
from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper
from constructive_access.robust.preparation import exp_negative_bounds, trig_bounds


def exact(x):
    if type(x) not in (int,F): raise TypeError('Use integers or Fractions')
    return F(x)


def calibration_vertices(mean_interval=(-1,1),second_moment_interval=(0,1)):
    """Exact (u=m^2,A) polygon vertices from simultaneous interval bounds."""
    if len(mean_interval)!=2 or len(second_moment_interval)!=2: raise ValueError('intervals need two endpoints')
    ml,mh=map(exact,mean_interval); al,ah=map(exact,second_moment_interval)
    if ml>mh or al>ah: raise ValueError('ordered interval endpoints required')
    ml,mh=max(F(-1),ml),min(F(1),mh)
    al,ah=max(F(0),al),min(F(1),ah)
    if ml>mh or al>ah: raise ValueError('infeasible calibration interval')
    ul=F(0) if ml<=0<=mh else min(ml*ml,mh*mh)
    uh=max(ml*ml,mh*mh)
    points={(u,a) for u in [ul,uh] for a in [al,ah] if u<=a}
    for x in [ul,uh,al,ah]:
        if ul<=x<=uh and al<=x<=ah:points.add((x,x))
    if not points:raise ValueError('calibration contradicts m^2 <= A')
    return sorted(points)


def calibrated_support(lags,coefficients,vertices,depth=6):
    """Rational support enclosure for q_k=(1+u+(A-u) integral x^k)/2."""
    lags=list(lags); a=list(map(exact,coefficients)); vertices=[tuple(map(exact,v)) for v in vertices]
    if not lags or len(lags)!=len(a) or any(type(k) is not int or k<=0 for k in lags):raise ValueError('matching positive integer lags required')
    if len(set(lags))!=len(lags):raise ValueError('distinct lag indices required')
    if not vertices or any(len(v)!=2 or not 0<=v[0]<=v[1]<=1 for v in vertices):raise ValueError('valid calibration vertices required')
    if type(depth) is not int or not 0<=depth<=14:raise ValueError('depth outside 0..14')
    polys=[]; upper=[]; lower=[]; S=sum(a)
    for u,A in vertices:
        power=[F(0)]*(max(lags)+1); power[0]=(1+u)*S/2
        for k,c in zip(lags,a):power[k]+=(A-u)*c/2
        polys.append(power);upper.append(ec.polynomial_upper(power,depth))
        # A finite grid is a lower bound on the support, not its certificate upper bound.
        lower.append(max(sum((c*F(j,2**depth)**k for k,c in enumerate(power)),F(0)) for j in range(2**depth+1)))
    return {'lags':lags,'coefficients':a,'calibration_vertices':vertices,'depth':depth,
            'polynomials':polys,'polynomial_upper_bounds':upper,
            'reversible_support_upper':max(upper),'reversible_support_lower':max(lower),
            'support_enclosure_width':max(upper)-max(lower)}


def paired_calibration(counts,alpha=F(1,20)):
    """Simultaneous mean/A intervals from iid records (++,+-,-+,--).

    The two readings within a record may be marginally dependent through X;
    their noise must be independent conditional on the same unchanged X.
    """
    counts=list(counts);alpha=exact(alpha)
    if len(counts)!=4 or any(type(x) is not int or x<0 for x in counts) or sum(counts)<=0:raise ValueError('four nonnegative counts with positive total required')
    if not 0<alpha<1:raise ValueError('alpha must lie in (0,1)')
    pp,pm,mp,mm=counts;n=sum(counts)
    mhat=F(pp-mm,n);Ahat=F(pp+mm-pm-mp,n)
    radius=sqrt_upper(2*ec.log_interval(4/alpha)[1]/n)
    mi=(max(F(-1),mhat-radius),min(F(1),mhat+radius))
    ai=(max(F(0),Ahat-radius),min(F(1),Ahat+radius))
    result={'counts':counts,'records':n,'mean_estimate':mhat,'conditional_second_moment_estimate':Ahat,
            'same_probability_estimate':F(pp+mm,n),'radius_upper':radius,
            'mean_interval':mi,'second_moment_interval':ai,'confidence_failure_upper':alpha,
            'contract':'iid stationary records; two conditionally independent noninvasive readings at the same unchanged hidden state'}
    try: vertices=calibration_vertices(mi,ai)
    except ValueError:return dict(result,status='calibration_confidence_region_empty',vertices=[])
    return dict(result,status='valid_calibration_region',vertices=vertices)


def analog_vertices(mean_interval,observed_second_moment_upper):
    """Upper-only A calibration for a bounded memoryless analog detector.

    The caller supplies valid simultaneous calibration bounds, which may come
    from a dependence-aware procedure. E[f(X)^2] <= E[Z^2]; no lower transfer.
    """
    upper=exact(observed_second_moment_upper)
    if upper<0:raise ValueError('nonnegative observed second-moment bound required')
    return calibration_vertices(mean_interval,(F(0),min(F(1),upper)))


def analog_calibration_summary(records,mean_estimate,observed_second_moment,alpha=F(1,20)):
    """IID bounded-analog summary confidence region; not valid for raw dependent streams.

    Exact summary statistics are supplied by the caller. Calibration records
    use the same fixed stationary memoryless channel as subsequent science data.
    """
    m,B,alpha=map(exact,(mean_estimate,observed_second_moment,alpha))
    if type(records) is not int or records<=0 or not 0<alpha<1 or not -1<=m<=1 or not m*m<=B<=1:raise ValueError('invalid iid analog summaries')
    rm=sqrt_upper(2*ec.log_interval(4/alpha)[1]/records)
    rb=sqrt_upper(ec.log_interval(2/alpha)[1]/(2*records))
    mi=(max(F(-1),m-rm),min(F(1),m+rm));upper=min(F(1),B+rb)
    return {'records':records,'mean_estimate':m,'observed_second_moment_estimate':B,
            'mean_interval':mi,'observed_second_moment_upper':upper,
            'second_moment_interval':(F(0),upper),'vertices':analog_vertices(mi,upper),
            'confidence_failure_upper':alpha,'mean_radius_upper':rm,'observed_second_moment_radius_upper':rb,
            'contract':'iid stationary bounded analog records; fixed noninvasive memoryless detector; second moment transfers only as an upper bound on A'}


def reversible_realization(mean,second_moment,atoms):
    """Exact stochastic step realization of an atomic calibrated moment law.

    Atom zero denotes a closure point and is not an exact finite-rate CTMC.
    Positive atoms yield Q_i=(-log x_i)(1 pi_i-I) at unit time.
    """
    m,A=map(exact,(mean,second_moment));atoms=[(exact(x),exact(w)) for x,w in atoms]
    if not -1<=m<=1 or not m*m<=A<=1 or not atoms or any(not 0<=x<=1 or w<=0 for x,w in atoms) or sum(w for x,w in atoms)!=1:raise ValueError('invalid moments or atoms')
    if A==m*m:
        return {'stationary':[F(1)],'detector':[m],'step_matrix':[[F(1)]],
                'continuous_time_exact':True,'mean':m,'second_moment':A,'atoms':atoms}
    low=(m-A)/(1-m);r=(m-low)/(1-low);n=2*len(atoms)
    P=[[F(0)]*n for _ in range(n)];pi=[];f=[]
    for j,(x,w) in enumerate(atoms):
        pi.extend([w*r,w*(1-r)]);f.extend([F(1),low])
        for a in range(2):
            for b in range(2):P[2*j+a][2*j+b]=(x if a==b else 0)+(1-x)*([r,1-r][b])
    return {'stationary':pi,'detector':f,'step_matrix':P,'continuous_time_exact':all(x>0 for x,w in atoms),
            'mean':m,'second_moment':A,'atoms':atoms}


def positive_lag_ambiguity(K):
    """Exact two spectra with different mass and identical moments 1,...,K.

    Both are realizable by irreducible products of independent telegraph chains
    with bounded stochastic detectors. This certifies pair laws, not all
    multi-time output distributions or time-tagged trajectories.
    """
    if type(K) is not int or not 1<=K<=20:raise ValueError('K outside 1..20')
    scale=F(1,(2**(K+1)-1)**2)
    positive=[];negative=[]
    for j in range(1,K+2):
        x=F(j,K+2);coef=(-1)**j*comb(K+1,j)
        (positive if coef>0 else negative).append((x,scale*abs(coef)))
    moment=lambda atoms,k:sum((w*x**k for x,w in atoms),F(0))
    p=[moment(positive,k) for k in range(1,K+1)];q=[moment(negative,k) for k in range(1,K+1)]
    assert p==q
    return {'lags':list(range(1,K+1)),'positive_spectrum':positive,'negative_spectrum':negative,
            'positive_A':moment(positive,0),'negative_A':moment(negative,0),'mean':F(0),
            'matched_correlations':p,'second_moment_difference':abs(moment(positive,0)-moment(negative,0)),
            'irreducible_product_state_counts':[2**len(positive),2**len(negative)],
            'coefficient_scale':scale}


def delayed_calibration(correlation_interval,delay,exit_rate_cap,terms=48):
    """A interval from a stationary delayed pair and a uniform exit-rate cap.

    Valid also for nonreversible stationary chains. A-C(t) <= 2 P(any jump).
    This is a controlled approximation, not exact finite-lag identification.
    """
    lo,hi=map(exact,correlation_interval);t,L=map(exact,(delay,exit_rate_cap))
    if not -1<=lo<=hi<=1 or t<0 or L<0 or type(terms) is not int or terms<1:raise ValueError('invalid correlation, time or rate cap')
    x=t*L
    # Odd Taylor polynomial is a global lower bound on exp(-x), x>=0.
    count=2*terms+1;term=total=F(1)
    for k in range(1,count+1):term*=-x/k;total+=term
    survival_lower=max(F(0),total)
    correction=2*(1-survival_lower)
    interval=(max(F(0),lo),min(F(1),hi+correction))
    return {'second_moment_interval':interval,'uniform_exit_rate_cap':L,'delay':t,
            'jump_probability_upper':1-survival_lower,'A_minus_correlation_upper':correction,
            'status':'valid_interval' if interval[0]<=interval[1] else 'empty_interval'}


def fast_mode_pair_tv_bound(A,minimum_lag,relaxation_rate,records=1,terms=96):
    """Upper TV for N independent pair records vs iid fair outputs.

    Alternative: symmetric reversible two-state chain, detector +/-sqrt(A).
    """
    A,t,rate=map(exact,(A,minimum_lag,relaxation_rate))
    if not 0<=A<=1 or t<0 or rate<0 or type(records) is not int or records<1:raise ValueError('invalid fast mode parameters')
    x=t*rate;term=total=F(1)
    for k in range(1,terms+1):term*=x/k;total+=term
    decay_upper=1/total
    return min(F(1),records*A*decay_upper/2)


def steep_decay_example():
    """Certified positive monotone correlations with a calibrated drop violation."""
    A=F(1,4);t=F(7,40);intervals=[]
    for lag in [t,2*t]:
        e=exp_negative_bounds(4*lag);c=trig_bounds(2*lag,'cos')
        products=[x*y*A for x in e for y in c]
        intervals.append((min(products),max(products)))
    lower=(intervals[0][0]-intervals[1][1])/2
    return {'forward_rate':F(3),'backward_rate':F(1),'detector':[F(1,2),F(1,2),F(-1,2),F(-1,2)],
            'mean':F(0),'A':A,'lags':[t,2*t],'correlation_intervals':intervals,
            'agreement_drop_lower':lower,'calibrated_reversible_upper':A/8,
            'uncalibrated_reversible_upper':F(1,8),'violation_lower':lower-A/8,
            'true_model_entropy_rate_lower':2*ec.log_interval(F(3))[0],
            'scope':'exact synthetic population certificate; no sampled data or optimality claim'}


def make_results():
    vertices=calibration_vertices((F(-1,10),F(1,10)),(F(1,5),F(3,10)))
    support=calibrated_support([1,2],[F(1),F(-1)],vertices)
    # Synthetic population-sized table, not a generated or measured data claim.
    calib=paired_calibration([3125,1875,1875,3125])
    sharp=reversible_realization(F(1,3),F(5,9),[(F(1,4),F(2,3)),(F(3,4),F(1,3))])
    high=[(F(1,4),F(3,16)),(F(3,4),F(1,16))];low=[(F(1,2),F(3,16))]
    return {'evidence':'exact synthetic constructions and illustrative calibration counts; no empirical data',
            'calibration_vertices':vertices,'support_example':support,'steep_decay_example':steep_decay_example(),'illustrative_calibration_counts':calib,
            'sharp_reversible_realization':sharp,'finite_menu_ambiguities':[positive_lag_ambiguity(K) for K in range(1,7)],
            'simple_two_lag_ambiguity':{'high_spectrum':high,'low_spectrum':low,'high_A':F(1,4),'low_A':F(3,16),
                'correlations':[F(3,32),F(3,64)],'mean':F(0),'irreducible_state_counts':[4,2]},
            'fast_mode':{'A_difference':F(1),'minimum_lag':F(1,10),'relaxation_rate':F(200),'records':1000,
                'record_tv_upper':fast_mode_pair_tv_bound(F(1),F(1,10),F(200),1000),'stationary_entropy_production':F(0)},
            'delayed_calibration':delayed_calibration((F(1,5),F(21,100)),F(1,100),F(2)),
            'analog_upper_only_example':{'signal_amplitude':F(1,4),'independent_noise_amplitude':F(1,2),'mean':F(0),'actual_A':F(1,16),'observed_second_moment':F(5,16),'two_encodings_of_one_observed_Z_product':F(5,16),'calibration':analog_calibration_summary(10000,F(0),F(5,16))},
            'shared_noise_failure':{'actual_mean':F(0),'actual_A':F(0),'same_bit_square':F(1),
                'perfectly_correlated_two_sensor_noise_product':F(1),'anticorrelated_control':{'actual_A':F(1,4),'apparent_product':F(0),'joint_counts_proportions':[F(1,4)]*4},'warning':'conditional detector-noise independence fails'}}


def _jsonify(x):
    if isinstance(x,F):return {'fraction':str(x),'decimal':float(x)}
    if isinstance(x,dict):return {k:_jsonify(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonify(v) for v in x]
    return x

if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'results'/'calibration.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(_jsonify(make_results()),indent=2)+'\n');print(out)
