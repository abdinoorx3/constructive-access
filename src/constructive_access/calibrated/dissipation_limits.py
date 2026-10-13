"""Four-state dissipation nonidentifiability and finite-budget repairs.

No invisible spectator, parallel microscopic channel, or unobserved jump is
needed: every transition of the simple four-cycle flips the parity detector.
All simulated records are synthetic. Exact bounds use rational arithmetic.
"""
from fractions import Fraction as F
from pathlib import Path
from math import ceil,exp,log
import argparse,hashlib,json
import numpy as np
from scipy.stats import beta
from constructive_access.frontier.energy_certificate import log_interval
from constructive_access.robust.statistics_design import fixed_lower_bound


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,F)):
        raise TypeError('Use exact integers, rational strings, or Fractions.')
    return F(x)


def check_rates(rates):
    rates=list(map(rational,rates))
    if len(rates)!=4 or min(rates)<=0:raise ValueError('Four strictly positive exit rates required.')
    return rates


def model(rates,reverse_probability):
    r=check_rates(rates);q=rational(reverse_probability)
    if not 0<q<=F(1,2):raise ValueError('Require0<reverse probability<=1/2.')
    Q=[[F(0)]*4 for _ in range(4)]
    for i in range(4):
        Q[i][(i+1)%4]=r[i]*(1-q);Q[i][(i-1)%4]=r[i]*q;Q[i][i]=-r[i]
    normalizer=sum((1/x for x in r),F(0))
    stationary=[1/(x*normalizer) for x in r]
    K=4/normalizer
    a,b=log_interval((1-q)/q)
    entropy=(K*(1-2*q)*a,K*(1-2*q)*b)
    return {'rates':r,'reverse_probability':q,'generator':Q,'stationary':stationary,
            'activity':K,'stationary_flux_affinity_interval':(a,b),
            'entropy_production_interval':entropy,'all_jumps_flip_detector':True,
            'simple_microscopic_graph':True}


def record_tv_bound(rates,q_a,q_b,exposure):
    r=check_rates(rates);qa,qb,L=map(rational,(q_a,q_b,exposure))
    if not 0<=qa<=1 or not 0<=qb<=1 or L<0:raise ValueError('Invalid bias or exposure.')
    return min(F(1),max(r)*abs(qa-qb)*L)


def coarse_tv_bound(rates,exposure):
    """Bias-independent parity-record bound using conditional intensity ranges."""
    r=check_rates(rates);L=rational(exposure)
    if L<0:raise ValueError('Nonnegative exposure required.')
    spread=max(abs(r[0]-r[2]),abs(r[1]-r[3]))
    return min(F(1),spread*L)


def finite_budget_pair(rates,exposure,tv_budget,entropy_gap):
    r=check_rates(rates);L,epsilon,gap=map(rational,(exposure,tv_budget,entropy_gap))
    if L<=0 or not 0<epsilon<1 or gap<=0:raise ValueError('Positive budget and entropy gap required.')
    q0=min(F(1,4),epsilon/(2*max(r)*L))
    first=model(r,q0);K=first['activity'];log2_lower=log_interval(F(2))[0]
    target=first['entropy_production_interval'][1]+gap
    bits=max(3,ceil(2*target/(K*log2_lower))+3)
    while F(1,2**bits)>=q0:bits+=1
    q1=F(1,2**bits);second=model(r,q1)
    certified_gap=second['entropy_production_interval'][0]-first['entropy_production_interval'][1]
    variation=record_tv_bound(r,q0,q1,L)
    assert certified_gap>gap and variation<=epsilon
    return {'reference_reverse_probability':q0,'high_entropy_reverse_probability':q1,
            'reference_entropy_interval':first['entropy_production_interval'],
            'high_entropy_interval':second['entropy_production_interval'],
            'entropy_gap_lower':certified_gap,'full_record_tv_upper':variation,
            'minimax_large_error_probability_lower':(1-variation)/2,
            'both_models_have_positive_reverse_rates':True,'exposure':L,
            'state_count':4,'rates_are_identical':r}


def unavoidable_infinite_upper_probability(rates,q,exposure,alpha=F(1,20)):
    """For every uniformly valid upper bound over all positive reverse rates."""
    r=check_rates(rates);q,L,a=map(rational,(q,exposure,alpha))
    if not 0<q<=F(1,2) or L<0 or not 0<a<1:raise ValueError('Invalid model or error level.')
    return max(F(0),1-a-max(r)*L*q)


def affinity_cap_upper(activity,affinity):
    """Numerical display of K*A*tanh(A/2), not an exact float certificate."""
    K,A=float(activity),float(affinity)
    if K<0 or A<0:raise ValueError('Nonnegative activity and affinity required.')
    return K*A*np.tanh(A/2)


def direction_count_interval(rates,records,reverse_count,reverse_floor,alpha=F(1,20),bits=22):
    """Fixed-jump quota; completed marks are IID. A fixed-time N is not interchangeable.

    At a timeout return the complete prior interval instead of conditioning the
    binomial calculation on successful completion. All displayed entropy is in
    natural entropy units per declared time, not automatically heat.
    """
    r=check_rates(rates);floor,alpha=map(rational,(reverse_floor,alpha))
    if not 0<=floor<=F(1,2) or not 0<alpha<1:raise ValueError('Nonnegative calibrated floor and valid alpha required.')
    lower=fixed_lower_bound(records,reverse_count,alpha/2,bits)['success_probability_lower']
    upper=1-fixed_lower_bound(records,records-reverse_count,alpha/2,bits)['success_probability_lower']
    qlo=max(floor,lower);qhi=min(F(1,2),upper)
    if qlo>qhi:return {'status':'empty_model_intersection','reverse_interval':(qlo,qhi),'confidence_failure_upper':alpha}
    low=model(r,qhi)['entropy_production_interval'][0]
    high=None if qlo==0 else model(r,qlo)['entropy_production_interval'][1]
    return {'status':'unbounded_upper' if high is None else 'finite_interval','records':records,'reverse_count':reverse_count,
            'reverse_interval':(qlo,qhi),'entropy_production_interval':(low,high),
            'calibrated_reverse_floor':floor,'confidence_failure_upper':alpha,
            'expected_duration_upper':F(records)/min(r),
            'optional_timeout_duration':F(2*records)/min(r),
            'timeout_probability_upper':F(3,4)**records,
            'timeout_rule':'Return the prior entropy interval; never treat completed-only quotas as an unbiased conditional sample.',
            'prior_entropy_interval':(F(0),None if floor==0 else model(r,floor)['entropy_production_interval'][1])}


def exp_negative_interval(x,terms=80):
    x=rational(x)
    if x<0 or x>=terms+2:raise ValueError('Increase the Taylor degree for this nonnegative exponent.')
    term=partial=F(1)
    for k in range(1,terms+1):term*=x/k;partial+=term
    tail=term*x/(terms+1)/(1-x/F(terms+2))
    return 1/(partial+tail),1/partial


def window_signal(rates,threshold_a=F(1),threshold_b=F(1)):
    """Exact magnitude coefficient of q in one binary Palm-pair event.

    Assumes r0>r2 and r1>r3. This does NOT declare successive observed event
    pairs independent or supply their physical reset distribution.
    """
    r=check_rates(rates);a,b=map(rational,(threshold_a,threshold_b))
    if not r[0]>r[2] or not r[1]>r[3] or min(a,b)<=0:
        raise ValueError('Positive ordered kinetic tags and windows required.')
    def difference(fast,slow,t):
        x=exp_negative_interval(slow*t);y=exp_negative_interval(fast*t)
        return max(F(0),x[0]-y[1]),x[1]-y[0]
    u=difference(r[0],r[2],a);v=difference(r[1],r[3],b)
    return (u[0]*v[0]/2,u[1]*v[1]/2)


def exponential_tv(rate_a,rate_b):
    a,b=float(rate_a),float(rate_b)
    if min(a,b)<=0:raise ValueError('Positive exponential rates required.')
    if a==b:return 0.
    a,b=max(a,b),min(a,b)
    t=log(a/b)/(a-b)
    return exp(-b*t)-exp(-a*t)


def pair_tv(rates,q_a,q_b):
    r=check_rates(rates)
    return abs(float(rational(q_a)-rational(q_b)))*exponential_tv(r[0],r[2])*exponential_tv(r[1],r[3])


def parity_log_likelihood(rates,reverse_probability,intervals,initial_parity=0):
    """Conditional parity-path density, including the final no-jump interval."""
    m=model(rates,reverse_probability);r=np.array(list(map(float,m['rates'])))
    if initial_parity not in(0,1) or not intervals or min(intervals)<0:raise ValueError('Invalid parity record.')
    jump=np.array([[float(x) for x in row] for row in m['generator']]);np.fill_diagonal(jump,0.)
    v=np.array(list(map(float,m['stationary'])));v[np.arange(4)%2!=initial_parity]=0.;v/=v.sum()
    result=0.
    for index,t in enumerate(intervals):
        v*=np.exp(-r*t)
        if index+1<len(intervals):v=v@jump
        mass=v.sum()
        if mass<=0:return float('-inf')
        result+=log(mass);v/=mass
    return result


def coupled_records(rates,q_a,q_b,exposure,replications,seed):
    """Uniformization coupling; mismatch frequency is an upper bound on TV.

    It is not a TV estimator: identical laws can disagree under a bad coupling.
    The chosen coupling makes equal-exit parity records exactly identical.
    """
    r=np.array([float(x) for x in check_rates(rates)]);R=max(r);qa,qb=map(float,(q_a,q_b))
    stationary=(1/r)/np.sum(1/r);rng=np.random.default_rng(seed)
    full=coarse=0
    for repetition in range(replications):
        ia=ib=int(rng.choice(4,p=stationary));t=0.;same_full=True;same_coarse=True
        while True:
            t+=rng.exponential(1/R)
            if t>float(exposure):break
            u,v=rng.random(2);ja=u<r[ia]/R;jb=u<r[ib]/R
            if ja!=jb:same_full=False;same_coarse=False
            if ja:
                da=-1 if v<qa else 1;ia=(ia+da)%4
            if jb:
                db=-1 if v<qb else 1;ib=(ib+db)%4
            if ja and jb and da!=db:same_full=False
        full+=not same_full;coarse+=not same_coarse
    def summary(count):
        lo=0. if count==0 else float(beta.ppf(.025,count,replications-count+1))
        hi=1. if count==replications else float(beta.ppf(.975,count+1,replications-count))
        return {'mismatches':count,'replications':replications,'frequency':count/replications,
                'clopper_pearson_interval':[lo,hi]}
    return {'full_record_coupling':summary(full),'parity_record_coupling':summary(coarse),
            'coupling_is_not_a_total_variation_estimator':True}


def compact(x):
    if isinstance(x,F):
        d=10**16;lo=F((x*d).numerator//(x*d).denominator,d);hi=F(-((-x*d).numerator//(-x*d).denominator),d)
        result={'lower':str(lo),'upper':str(hi),'decimal':float(x)}
        if x and lo==0:result['exact_small_value']=str(x)
        return result
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [compact(v) for v in x]
    return x


def run(contract,contract_bytes):
    equal=list(map(F,contract['equal_rates']));tagged=list(map(F,contract['tagged_rates']));L=F(contract['exposure'])
    target=contract['finite_budget_target']
    near=finite_budget_pair(tagged,L,F(target['record_tv_upper']),F(target['entropy_gap_lower']))
    path_intervals=[.12,.23,.07,.33,.25]
    equal_rows=[]
    for qstring in contract['equivalent_reverse_probabilities']:
        q=F(qstring);m=model(equal,q)
        equal_rows.append({'reverse_probability':q,'activity':m['activity'],'stationary':m['stationary'],
            'entropy_interval':m['entropy_production_interval'],
            'parity_log_likelihood':parity_log_likelihood(equal,q,path_intervals),
            'same_probability_at_unit_lag':(1+exp(-2))/2})
    coupled=[]
    for j,control in enumerate(contract['coupled_controls']):
        rates=equal if control['rates']=='equal' else tagged
        if control['name']=='near_irreversible':qa,qb=near['reference_reverse_probability'],near['high_entropy_reverse_probability']
        else:qa,qb=F(control['q_a']),F(control['q_b'])
        row=coupled_records(rates,qa,qb,L,contract['replications'],contract['seed']+j)
        row.update(name=control['name'],rates=rates,q_a=qa,q_b=qb,
                   full_record_tv_upper=record_tv_bound(rates,qa,qb,L),
                   parity_record_tv_upper=coarse_tv_bound(rates,L))
        coupled.append(row)
    tags=[]
    for text in contract['weak_tag_amplitudes']:
        eta=F(text);rates=[1+eta,1+2*eta,1-eta,1-2*eta]
        tags.append({'tag_amplitude':eta,'rates':rates,
                     'window_response_coefficient':window_signal(rates),
                     'pair_tv_for_bias_half_vs_one_tenth':pair_tv(rates,F(1,2),F(1,10)),
                     'finite_exposure_parity_tv_upper':coarse_tv_bound(rates,L)})
    direction=contract['resolved_direction_example']
    interval=direction_count_interval(tagged,direction['fixed_jumps'],direction['reverse_count'],
                    F(direction['reverse_probability_floor']),F(direction['alpha']),direction['bits'])
    return {'evidence':'Exact constructions and synthetic coupling checks; no measured physical data',
            'contract_sha256':hashlib.sha256(contract_bytes).hexdigest(),
            'equal_exit_exact_equivalence':equal_rows,'finite_budget_pair':near,
            'unavoidable_infinite_upper_probability':unavoidable_infinite_upper_probability(tagged,near['reference_reverse_probability'],L),
            'coupling_checks':coupled,'kinetic_tag_repair':tags,'resolved_direction_repair':interval,
            'without_floor_no_reverse_evidence':direction_count_interval(tagged,direction['fixed_jumps'],0,F(0)),
            'without_floor_positive_reverse_evidence':direction_count_interval(tagged,direction['fixed_jumps'],direction['reverse_count'],F(0)),
            'assumptions':['Fixed simple four-state cycle with physically resolved reverse transitions; no hidden parallel channels.',
                'All jumps flip the parity detector; observations include the entire timed parity path.',
                'Complete-record no-go uses pathwise bounded total physical exposure and bounded exit rates.',
                'A fixed jump quota, not a fixed elapsed-time jump count, supplies IID direction marks.',
                'A uniformly finite guaranteed upper requires a reverse floor or affinity cap; positive resolved reverse evidence can also yield a finite data-dependent upper.'],
            'limitations':['Established lumpability and coarse-graining obstructions are not relabeled as discoveries.',
                'Kinetic-tag identification is within the specified calibrated four-state family.',
                'Palm waiting-time pairs from one trajectory are not silently treated as independent.',
                'Natural entropy units become heat only under the appropriate physical thermodynamic assumptions.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,default=Path(__file__).resolve().parents[1]/'contracts'/'dissipation_limits.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'dissipation_limits.json')
    args=parser.parse_args();raw=args.contract.read_bytes();result=run(json.loads(raw),raw)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(compact(result),indent=2,sort_keys=True)+'\n')
    print(json.dumps({'entropy_gap_lower':float(result['finite_budget_pair']['entropy_gap_lower']),
        'full_record_tv_upper':float(result['finite_budget_pair']['full_record_tv_upper']),
        'forced_infinite_upper_probability':float(result['unavoidable_infinite_upper_probability'])}))
