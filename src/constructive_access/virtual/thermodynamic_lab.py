#!/usr/bin/env python3
"""Independent stochastic trajectory laboratory for temporal entropy inference.

Synthetic trajectories, not laboratory measurements. Ring trajectories use
independent Poisson jump counts; binary readouts are generated from physical
endpoint states. Inference never receives those hidden states.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.linalg import expm
from scipy.stats import beta
from constructive_access.frontier.energy_certificate import certify, log_interval


def specification():
    # Values are fixed before any production dataset is generated.
    cosine=['1','123461/125000','951057/1000000','891007/1000000','809017/1000000',
        '707107/1000000','117557/200000','45399/100000','309017/1000000','78217/500000',
        '0','-78217/500000','-309017/1000000','-45399/100000','-117557/200000',
        '-707107/1000000','-809017/1000000','-891007/1000000','-951057/1000000',
        '-123461/125000','-1','-123461/125000','-951057/1000000','-891007/1000000',
        '-809017/1000000','-707107/1000000','-117557/200000','-45399/100000',
        '-309017/1000000','-78217/500000','0','78217/500000','309017/1000000',
        '45399/100000','117557/200000','707107/1000000','809017/1000000',
        '891007/1000000','951057/1000000','123461/125000']
    menus={
        'revival':{'lags':[16,21],'step':'1/10','records_per_setting':5000,
            'box_radius':'21/1000','detector':['1']*20+['-1']*20,
            'scores':[['61/47','32/39'],['33/47','46/39']],
            'memory_agreement':[.512,.736]},
        'curvature':{'lags':[1,2,3],'step':'3/20','records_per_setting':50000,
            'box_radius':'7/1000','detector':cosine,
            'scores':[['101/100','199/200'],['49/50','101/100'],['101/100','199/200']],
            'memory_agreement':[.7169821595,.64530792105,.55144537275]}}
    scenarios=[
        {'name':'driven','forward':20.,'backward':1.,'gain':1.,'jitter_halfwidth':0.,'contract':'within'},
        {'name':'equilibrium','forward':10.5,'backward':10.5,'gain':1.,'jitter_halfwidth':0.,'contract':'within'},
        {'name':'high_readout_noise','forward':20.,'backward':1.,'gain':.2,'jitter_halfwidth':0.,'contract':'within'},
        {'name':'independent_clock_jitter','forward':20.,'backward':1.,'gain':1.,'jitter_halfwidth':.3,'contract':'within_random_scale_clock'},
        {'name':'detector_memory','forward':10.5,'backward':10.5,'gain':1.,'jitter_halfwidth':0.,'contract':'outside_detector_memory'}]
    return {'version':1,'status':'frozen before production; synthetic stochastic trajectories',
        'pilot_seed':94170011,'holdout_seed':94170029,'pilot_datasets':20,'holdout_datasets':200,
        'alpha':'1/20','states':40,'menus':menus,'scenarios':scenarios,
        'extra_case':{'menu':'revival','name':'prepared_equilibrium','forward':9.,'backward':1.,
            'states':2,'initial_state':0,'contract':'outside_stationarity; valid preparation-resource interpretation'},
        'raw_dataset':'curvature/driven, first held-out dataset, every simulated record',
        'matched_access':'All three inference methods use identical counts, settings and physical exposure within each case.',
        'root_predeclared_extension':'curvature contrast(-1,2,-1), alpha.05; independently implemented by root; same held-out counts',
        'units':'arbitrary declared model time; rates are illustrative, not empirical calibration',
        'clock_law':'actual lag = nominal lag * V, independent V uniform[1-jitter,1+jitter] with common law at every setting',
        'count_interval':'simultaneous two-sided Hoeffding box, 2K exp(-2nr²)<=alpha, checked with exact rational log enclosure',
        'monte_carlo_interval':'two-sided95% Clopper-Pearson interval across independent held-out datasets',
        'no_claims':['empirical physical validation','universal heat interpretation','free stationary reset','inference validity under detector memory']}


def canonical_json(data):
    return json.dumps(data,sort_keys=True,indent=2)+'\n'


def freeze(root):
    path=root/'contracts'/'thermodynamic_lab.json';path.parent.mkdir(parents=True,exist_ok=True)
    encoded=canonical_json(specification())
    if path.exists():
        if path.read_text()!=encoded:raise RuntimeError('Frozen contract differs; no overwrite allowed')
    else:
        with path.open('x') as stream:stream.write(encoded)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(root):
    path=root/'contracts'/'thermodynamic_lab.json'
    if not path.exists():raise RuntimeError('Freeze the contract before simulation')
    contract=json.loads(path.read_text())
    if contract!=specification():raise RuntimeError('Contract and code specification disagree')
    return contract,hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer=io.BytesIO();np.lib.format.write_array(buffer,np.asarray(arrays[name]),allow_pickle=False)
            member=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0))
            member.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(member,buffer.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)


def ring_records(rng,n,lag,forward,backward,detector,gain=1.,jitter_halfwidth=0.,memory_agreement=None):
    """Sample exact ring path increments and two independent physical readouts.

    Poisson count representation is exact for a translation-invariant ring;
    it is not an approximation to a Bernoulli endpoint formula.
    """
    detector=np.asarray(detector,dtype=float);states=len(detector)
    if n<=0 or lag<=0 or min(forward,backward)<0 or not 0<=gain<=1 or not 0<=jitter_halfwidth<1:
        raise ValueError('Invalid physical simulation parameters')
    if np.any(abs(detector)>1):raise ValueError('Detector conditional means outside[-1,1]')
    initial=rng.integers(0,states,n,dtype=np.int16)
    scale=rng.uniform(1-jitter_halfwidth,1+jitter_halfwidth,n) if jitter_halfwidth else np.ones(n)
    elapsed=lag*scale
    plus=rng.poisson(forward*elapsed);minus=rng.poisson(backward*elapsed)
    final=(initial+plus-minus)%states
    first=(rng.random(n)<(1+gain*detector[initial])/2).astype(np.int8)
    second=(rng.random(n)<(1+gain*detector[final])/2).astype(np.int8)
    if memory_agreement is not None:
        # Deliberate outside-contract detector: its second output depends on
        # its stored first output, rather than only the second physical state.
        agree=rng.random(n)<memory_agreement
        second=np.where(agree,first,1-first).astype(np.int8)
    return dict(initial=initial,final=final.astype(np.int16),forward_jumps=plus.astype(np.int32),
        backward_jumps=minus.astype(np.int32),elapsed=elapsed,first=first,second=second,
        same=(first==second).astype(np.uint8))


def two_state_records(rng,n,lag,forward=9.,backward=1.,initial_state=0):
    """Independent direct Gillespie paths for the prepared equilibrium control."""
    state=np.full(n,initial_state,dtype=np.int8);initial=state.copy();clock=np.zeros(n)
    plus=np.zeros(n,dtype=np.int32);minus=plus.copy();active=np.arange(n)
    while len(active):
        rates=np.where(state[active]==0,forward,backward)
        candidate=clock[active]+rng.exponential(1/rates)
        keep=candidate<lag
        jumping=active[keep];clock[jumping]=candidate[keep]
        was_zero=state[jumping]==0
        plus[jumping]+=was_zero;minus[jumping]+=~was_zero
        state[jumping]=1-state[jumping];active=jumping
    return dict(initial=initial,final=state,forward_jumps=plus,backward_jumps=minus,
        elapsed=np.full(n,lag),first=initial,second=state,same=(initial==state).astype(np.uint8))


def compile_analysis(menu,alpha):
    K=len(menu['lags']);n=menu['records_per_setting'];radius=F(menu['box_radius'])
    if 2*n*radius*radius<log_interval(F(2*K)/F(alpha))[1]:
        raise ValueError('Frozen probability box does not have advertised coverage')
    scores=[tuple(map(F,pair)) for pair in menu['scores']]
    base=certify(menu['lags'],F(menu['step']),[F(1,K)]*K,[(F(0),F(1))]*K,
                 (F(-1),F(1)),scores)
    logs=[tuple(log_interval(value)[0] for value in row) for row in scores]
    return {'n':n,'radius':radius,'K':K,'logs':logs,
        'normalizer_log_upper':log_interval(base['normalizer_upper'])[1],
        'mean_lag':F(menu['step'])*F(sum(menu['lags']),K)}


def analyze_counts(successes,compiled):
    n,r,K=compiled['n'],compiled['radius'],compiled['K']
    boxes=[(max(F(0),F(int(s),n)-r),min(F(1),F(int(s),n)+r)) for s in successes]
    negative=any(hi<F(1,2) for lo,hi in boxes)
    revival=any(boxes[i][1]<boxes[j][0] for i in range(K) for j in range(i+1,K))
    score=F(0)
    for (lo,hi),(log0,log1) in zip(boxes,compiled['logs']):
        score+=min((1-lo)*log0+lo*log1,(1-hi)*log0+hi*log1)/K
    J=max(F(0),score-compiled['normalizer_log_upper'])
    return {'negative':negative,'revival':revival,'full_cone':J>0,
            'full_cone_rate_lower':float(4*J/compiled['mean_lag'])}


def clopper_pearson(successes,trials,alpha=.05):
    if not 0<=successes<=trials or trials<=0:raise ValueError('Invalid binomial counts')
    lower=0. if successes==0 else float(beta.ppf(alpha/2,successes,trials-successes+1))
    upper=1. if successes==trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes))
    return [lower,upper]


def oracle_probabilities(menu,scenario):
    """Independent matrix-exponential diagnostic; never used for sampling.

    Jitter averaging is numerical Gauss-Legendre quadrature. This oracle is
    for simulator/coverage checks, not a physical calibration certificate.
    """
    times=[float(F(menu['step'])*k) for k in menu['lags']]
    if scenario['name']=='detector_memory':return np.asarray(menu['memory_agreement'])
    if scenario['name']=='prepared_equilibrium':return .1+.9*np.exp(-10*np.asarray(times))
    f=np.array([float(F(z))*scenario['gain'] for z in menu['detector']]);n=len(f)
    Q=np.zeros((n,n));a=scenario['forward'];b=scenario['backward']
    for i in range(n):Q[i,(i+1)%n]=a;Q[i,(i-1)%n]=b;Q[i,i]=-a-b
    jitter=scenario['jitter_halfwidth']
    def evaluate(order):
        nodes,weights=np.polynomial.legendre.leggauss(order) if jitter else (np.array([0.]),np.array([2.]))
        return np.array([sum(weight*(1+f@expm((1+jitter*node)*t*Q)@f/n)/4
                            for node,weight in zip(nodes,weights)) for t in times])
    values=evaluate(32)
    if jitter and np.max(abs(values-evaluate(64)))>1e-11:
        raise ArithmeticError('Independent jitter quadrature failed its convergence check')
    return values


def run(root,pilot_only=False):
    contract,contract_sha=load_contract(root);results=root/'results';results.mkdir(parents=True,exist_ok=True)
    period='pilot' if pilot_only else 'holdout'
    base_seed=contract[period+'_seed'];repetitions=contract[period+'_datasets']
    arrays={};summaries=[];case_index=0;raw_saved=False
    for menu_name,menu in contract['menus'].items():
        scenarios=list(contract['scenarios'])
        if menu_name=='revival':scenarios.append(contract['extra_case'])
        compiled=compile_analysis(menu,contract['alpha'])
        detector=[float(F(z)) for z in menu['detector']]
        for scenario in scenarios:
            case=scenario['name'];key=menu_name+'__'+case;K=len(menu['lags']);n=menu['records_per_setting']
            counts=np.zeros((repetitions,K),dtype=np.int32);exposures=np.zeros((repetitions,K))
            rejections={name:[] for name in ('negative','revival','full_cone')};bounds=[]
            for repetition in range(repetitions):
                raw_parts=[]
                for setting,k in enumerate(menu['lags']):
                    # Independent streams bind each dataset/setting to the
                    # frozen seed, without reuse across pilot or holdout.
                    rng=np.random.default_rng(np.random.SeedSequence([base_seed,case_index,repetition,setting]))
                    lag=float(F(menu['step'])*k)
                    if case=='prepared_equilibrium':
                        record=two_state_records(rng,n,lag)
                    else:
                        memory=menu['memory_agreement'][setting] if case=='detector_memory' else None
                        record=ring_records(rng,n,lag,scenario['forward'],scenario['backward'],detector,
                            scenario['gain'],scenario['jitter_halfwidth'],memory)
                    counts[repetition,setting]=int(record['same'].sum())
                    exposures[repetition,setting]=float(record['elapsed'].sum())
                    if not pilot_only and menu_name=='curvature' and case=='driven' and repetition==0:
                        record['setting']=np.full(n,setting,dtype=np.uint8);raw_parts.append(record)
                if raw_parts:
                    raw={name:np.concatenate([part[name] for part in raw_parts]) for name in raw_parts[0]}
                    deterministic_npz(results/'thermodynamic_lab_raw.npz',raw);raw_saved=True
                analysis=analyze_counts(counts[repetition],compiled)
                for method in rejections:rejections[method].append(analysis[method])
                bounds.append(analysis['full_cone_rate_lower'])
            arrays[key+'__counts']=counts;arrays[key+'__exposure']=exposures
            oracle=oracle_probabilities(menu,scenario)
            box_covers=np.all(abs(counts/n-oracle[None,:])<=float(compiled['radius'])+1e-12,axis=1)
            true_sigma=0. if scenario['forward']==scenario['backward'] or case=='prepared_equilibrium' else (
                scenario['forward']-scenario['backward'])*float(np.log(scenario['forward']/scenario['backward']))
            summary={'case':key,'contract':scenario['contract'],'datasets':repetitions,
                'records_per_setting':n,'settings':K,'records_per_dataset':n*K,
                'mean_total_observation_exposure':float(exposures.sum(axis=1).mean()),
                'reset_and_detector_energy':'not included or empirically calibrated',
                'simulated_system_stationary_entropy_rate':true_sigma,
                'mean_same_probabilities':(counts.mean(axis=0)/n).tolist(),
                'independent_expm_oracle_probabilities':oracle.tolist(),
                'joint_box_coverage':{'covered_datasets':int(box_covers.sum()),
                    'rate':float(box_covers.mean()),
                    'monte_carlo_95_interval':clopper_pearson(int(box_covers.sum()),repetitions),
                    'oracle_status':'numerical matrix exponential; jitter quadrature independently converged'},
                'mean_full_cone_rate_lower':float(np.mean(bounds)),
                'inference':{method:{'rejections':int(sum(flags)),'rate':float(np.mean(flags)),
                    'monte_carlo_95_interval':clopper_pearson(int(sum(flags)),repetitions)}
                    for method,flags in rejections.items()}}
            if case=='prepared_equilibrium':summary['initial_preparation_divergence']=float(np.log(10))
            summaries.append(summary);case_index+=1
            print(f'{period}: {key}: '+', '.join(f'{method}={sum(flags)}/{repetitions}' for method,flags in rejections.items()),flush=True)
    suffix='_pilot' if pilot_only else ''
    archive=results/f'thermodynamic_lab_counts{suffix}.npz';deterministic_npz(archive,arrays)
    report={'evidence':'independent stochastic trajectories and noisy simulated readouts; not laboratory data',
        'period':period,'contract_sha256':contract_sha,'seed':base_seed,'datasets_per_case':repetitions,
        'cases':summaries,'counts_archive':archive.name,'counts_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'sampling':'independent physical trajectory pairs; exact Poisson ring counts or direct two-state Gillespie',
        'analysis':'same simultaneous confidence boxes and same records/exposure across all three methods',
        'outside_contract_cases_retained':True,'raw_dataset_retained':raw_saved}
    if raw_saved:
        raw=results/'thermodynamic_lab_raw.npz';report['raw_archive']=raw.name
        report['raw_archive_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
    (results/f'thermodynamic_lab{suffix}.json').write_text(canonical_json(report))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args()
    if args.freeze:print(freeze(args.root))
    else:run(args.root,args.pilot_only)
