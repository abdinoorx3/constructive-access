"""Frozen, matched virtual-laboratory study of detector calibration and KL bounds.

All records are simulated hidden-state paths plus noisy readouts. No result is
empirical physics. Full comparison metadata retains unequal evolution times,
unknown reset/detector overhead, and the invalid-calibration failure control.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.linalg import expm
from constructive_access.virtual.thermodynamic_lab import (
    ring_records, deterministic_npz, clopper_pearson)
from constructive_access.virtual.contrast import certify_counts, sqrt_upper
from constructive_access.frontier import energy_certificate as ec
try:
    from . import nonlinear, calibration
except ImportError:
    import nonlinear, calibration


def alternative_design():
    """Prior-model design over a fixed grid, before any stochastic sampling."""
    truth=[F(69,100),F(59,100),F(30357,62500)];counts=[8000,8000,4000]
    candidates=[]
    for j in range(1,100):
        s=F(j,100);a=[-s*s,2*s,F(-1)];support=-(1-s)**2/2
        radius=sqrt_upper(ec.log_interval(20)[1]*sum(c*c/n for c,n in zip(a,counts))/2)
        lower=sum(c*p for c,p in zip(a,truth))-support-radius
        denominator=sum(abs(c)*sqrt_upper(F(t)) for c,t in zip(a,[1,2,3]))**2
        objective=8*max(F(0),lower)**2/denominator
        candidates.append((objective,s,a))
    value,s,a=max(candidates,key=lambda row:row[0])
    return dict(parameter=str(s),coefficients=list(map(str,a)),
        grid='s=j/100, j=1,...,99',population_probabilities=list(map(str,truth)),
        criterion='maximize model-centered finite-data Pinsker rate lower after fixed one-sided Hoeffding penalty; no generated data',
        predicted_objective=float(value),legacy_coefficients=['-4/9','4/3','-1'])


def specification():
    return dict(version=1, evidence='simulated stochastic trajectories; no hardware measurements',
        seed=73024119,datasets_per_case=400,states=20,
        model_definition={'r':'sqrt(497)/25','omega':'acos(19/sqrt(497))','gamma':'-log(r)',
            'phi':'2*pi/20','forward':'(gamma/(1-cos(phi))+omega/sin(phi))/2',
            'backward':'(gamma/(1-cos(phi))-omega/sin(phi))/2',
            'detector':'cos(2*pi*j/20)','stationary_preparation':'uniform independent reset each pair'},
        cases=[{'name':'driven','rates':'driven','gain':1.,'calibration_gain':1.,'contract':'within'},
               {'name':'equilibrium','rates':'equal_same_activity','gain':1.,'calibration_gain':1.,'contract':'within'},
               {'name':'high_readout_noise','rates':'driven','gain':.2,'calibration_gain':.2,'contract':'within'},
               {'name':'invalid_calibration','rates':'equal_same_activity','gain':1.,'calibration_gain':.2,
                'contract':'outside: detector gain changes between calibration and science'}],
        science_lags=[1,2,3],science_records=[8000,8000,4000],calibration_records=4000,
        calibrated={'coefficients':['38/25','-1'],'alpha_science':'1/40','alpha_calibration':'1/40',
                    'mixture_weights':['291409/426973','135564/426973'],
                    'proposal_grid':33,'polynomial_depth':7,
                    'confidence_region':'same covered contrast halfspace and same calibrated polygon for nonlinear and Pinsker'},
        extra_lag_alternative=dict(alternative_design(),alpha='1/20',
                    inference='predeclared prior-model-designed uncalibrated shifted-cubic contrast with optimized Pinsker time weights'),
        calibration_method='two independently randomized binary readings conditioned on one unchanged uniformly sampled hidden state',
        calibrated_arm={'pairs':20000,'readouts':40000,'declared_evolution_time':24000,'settings':[0,1,2]},
        alternative_arm={'pairs':20000,'readouts':40000,'declared_evolution_time':36000,'settings':[1,2,3]},
        cost_limits=['pair/readout counts matched, evolution times differ',
            'zero-lag calibration means ideal same-state readings; finite detector exposure not modeled',
            'reset, sensor hardware, measurement latency and energy uncalibrated, not zero',
            'joint simulation retains shared lag1/2 records, plus both independent extra menus'],
        numerical_method='hidden dynamics evaluated in floating point; inference uses rational outward bounds; no exact physical calibration claim',
        declared_outputs=['matched detection counts and pointwise95% Clopper-Pearson intervals',
            'all-dataset bound ratios and paired differences, true simulated entropy, calibration and joint confidence coverage',
            'relaxation primal-dual gaps for every nonlinear result',
            'exact equilibrium countermodel for uncalibrated two-lag law',
            'first driven dataset raw records plus all held-out sufficient counts'],
        no_pilot='model and coefficients chosen analytically before production; no generated pilot or data-based redesign',
        seed_stream='SeedSequence([seed,case_index,dataset_index,setting_index]); calibration setting_index3')


def freeze(root):
    path=root/'contracts'/'calibrated_benchmark.json';path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(specification(),sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=content:raise RuntimeError('Frozen contract differs; will not overwrite')
    else:
        with path.open('x') as stream:stream.write(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(case):
    r=math.sqrt(497)/25;omega=math.acos(19/math.sqrt(497));gamma=-math.log(r);phi=2*math.pi/20
    forward=(gamma/(1-math.cos(phi))+omega/math.sin(phi))/2
    backward=(gamma/(1-math.cos(phi))-omega/math.sin(phi))/2
    if case['rates']=='equal_same_activity':forward=backward=(forward+backward)/2
    detector=np.cos(2*math.pi*np.arange(20)/20)
    return forward,backward,detector


def calibration_records(rng,n,detector,gain):
    state=rng.integers(0,len(detector),n,dtype=np.int16)
    prob=(1+gain*np.asarray(detector)[state])/2
    first=(rng.random(n)<prob).astype(np.int8)
    second=(rng.random(n)<prob).astype(np.int8)
    counts=np.array([np.sum((first==1)&(second==1)),np.sum((first==1)&(second==0)),
                     np.sum((first==0)&(second==1)),np.sum((first==0)&(second==0))],dtype=np.int32)
    return counts,dict(initial=state,final=state.copy(),forward_jumps=np.zeros(n,dtype=np.int32),
        backward_jumps=np.zeros(n,dtype=np.int32),elapsed=np.zeros(n),first=first,second=second,
        same=(first==second).astype(np.uint8))


def oracle(case):
    a,b,f=model(case);f=case['gain']*f;n=len(f);Q=np.zeros((n,n))
    for i in range(n):Q[i,(i+1)%n]=a;Q[i,(i-1)%n]=b;Q[i,i]=-a-b
    p=[(1+f@expm(t*Q)@f/n)/2 for t in [1,2,3]]
    return dict(probabilities=p,mean=float(np.mean(f)),A=float(np.mean(f*f)),
                stationary_entropy_rate=(a-b)*math.log(a/b),activity=a+b)


def analyze(science_counts,calibration_counts,contract):
    alternatives={}
    for name,key in [('alternative','coefficients'),('legacy_alternative','legacy_coefficients')]:
        alternatives[name]=certify_counts([1,2,3],list(map(F,contract['extra_lag_alternative'][key])),
            contract['science_records'],[int(x) for x in science_counts],[1,2,3],
            alpha=F(contract['extra_lag_alternative']['alpha']))
    cal=calibration.paired_calibration([int(x) for x in calibration_counts],F(contract['calibrated']['alpha_calibration']))
    a=list(map(F,contract['calibrated']['coefficients']))
    region=nonlinear.confidence_region(a,contract['science_records'][:2],
        [int(x) for x in science_counts[:2]],F(contract['calibrated']['alpha_science']))
    if cal['status']!='valid_calibration_region':
        return dict(status='inconsistent_calibration_region',calibration=cal,region=region,**alternatives)
    vertices=cal['vertices'];sup=calibration.calibrated_support([1,2],a,vertices,depth=7)
    gap=max(F(0),region['lower']-sup['reversible_support_upper'])
    denominator=sum(abs(c)*sqrt_upper(F(t)) for c,t in zip(a,[1,2]))**2
    pinsker=8*gap*gap/denominator
    w=list(map(F,contract['calibrated']['mixture_weights']))
    if not gap:
        # Conservative shortcut: no positive certificate from this support
        # enclosure. This is NOT proof that the exact projection equals zero;
        # a tighter support computation or different method could improve it.
        null_scores=[(F(1),F(1))]*2
        cert=nonlinear.certify([1,2],w,vertices,a,region['lower'],[1,2],null_scores,depth=7)
        projected=dict(certificate=cert,primal=None,certified_relaxation_gap_upper=None,
                       proposal_success=None,proposal_status='not run: nonpositive contrast')
    else:
        projected=nonlinear.design([1,2],w,vertices,a,region['lower'],[1,2],
            grid=contract['calibrated']['proposal_grid'],depth=contract['calibrated']['polynomial_depth'])
    # Zero witness follows from the exact same-lag equilibrium countermodel;
    # this is not a declaration that ALL uncalibrated interventions are blind.
    return dict(status='analyzed',calibration=cal,region=region,support=sup,
        calibrated_pinsker_rate=pinsker,nonlinear=projected,
        **alternatives)


def equilibrium_countermodel():
    x=F(9,19);A=F(361,450)
    return dict(mean=F(0),A=A,decay_per_unit=x,
        detector='conditional means +/-sqrt(361/450)',jump_rate='-log(9/19)/2',
        probabilities=[(1+A*x**k)/2 for k in [1,2,3]],
        stationary_entropy_production=F(0),
        exact_scope='finite irreducible symmetric two-state CTMC; matches the driven model at lags1,2 only; differing A requires genuine additional calibration access')


def summary(flags):
    n=len(flags);count=int(sum(flags))
    return dict(detections=count,datasets=n,rate=count/n,
                monte_carlo_95_interval=clopper_pearson(count,n))


def _audit_one(task):
    case_index,rep,science,calcounts,contract=task
    result=analyze(science,calcounts,contract)
    if result['status']!='analyzed':
        return dict(case_index=case_index,dataset=rep,status=result['status'])
    linear=result['calibrated_pinsker_rate'];projected=result['nonlinear']
    raw=projected['certificate']['entropy_production_rate_lower']
    record=dict(case_index=case_index,dataset=rep,status='analyzed',
        pinsker_rate=linear,raw_nonlinear_rate=raw,safe_same_event_maximum=max(linear,raw),
        relaxation_gap=projected['certified_relaxation_gap_upper'],
        proposal_success=projected['proposal_success'])
    if linear>0 and raw==0:
        # Explicit post-production numerical convergence diagnostic; this does
        # NOT replace the frozen grid33/depth7 result reported in the study.
        refined=nonlinear.design([1,2],list(map(F,contract['calibrated']['mixture_weights'])),
            result['calibration']['vertices'],result['region']['coefficients'],
            result['region']['lower'],[1,2],grid=129,depth=9)
        record['posthoc_refinement']=dict(grid=129,depth=9,
            rate=refined['certificate']['entropy_production_rate_lower'],
            relaxation_gap=refined['certified_relaxation_gap_upper'],
            interpretation='numerical convergence diagnostic on the same covered region; frozen raw result retained')
    return record


def audit(root,workers=4):
    """Counts-only independent replay; no new stochastic data are generated."""
    from concurrent.futures import ProcessPoolExecutor
    contract=json.loads((root/'contracts'/'calibrated_benchmark.json').read_text())
    report=json.loads((root/'results'/'calibrated_benchmark.json').read_text())
    source=root/'results'/report['counts_archive'];arrays=np.load(source)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['counts_archive_sha256']:
        raise RuntimeError('Held-out count archive hash mismatch')
    tasks=[]
    for ci,case in enumerate(contract['cases']):
        for rep in range(contract['datasets_per_case']):
            tasks.append((ci,rep,arrays[case['name']+'__science_counts'][rep].tolist(),
                          arrays[case['name']+'__calibration_counts'][rep].tolist(),contract))
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(_audit_one,tasks,chunksize=10):
            if record['status']=='analyzed':
                name=contract['cases'][record['case_index']]['name'];rep=record['dataset']
                for field,method in [('pinsker_rate','calibrated_pinsker'),('raw_nonlinear_rate','calibrated_nonlinear')]:
                    previous=float(arrays[name+'__'+method+'_rate'][rep])
                    if abs(float(record[field])-previous)>1e-12:
                        raise ArithmeticError('Replayed inference differs from frozen result')
            records.append(record)
            if len(records)%200==0:print(f'counts-only audit: {len(records)}/{len(tasks)}',flush=True)
    summaries=[]
    for ci,case in enumerate(contract['cases']):
        selected=[x for x in records if x['case_index']==ci and x['status']=='analyzed']
        safe=[float(x['safe_same_event_maximum']) for x in selected]
        raw=[float(x['raw_nonlinear_rate']) for x in selected]
        differences=[float(x['safe_same_event_maximum']-x['pinsker_rate']) for x in selected]
        summaries.append(dict(case=case['name'],safe_maximum_detection=summary([x>0 for x in safe]),
            safe_maximum_mean=float(np.mean(safe)),raw_nonlinear_mean=float(np.mean(raw)),
            safe_minus_pinsker_mean=float(np.mean(differences)),
            safe_minus_pinsker_monte_carlo_standard_error=float(np.std(differences,ddof=1)/math.sqrt(len(differences))),
            marginal_raw_nonlinear_sign_losses=sum('posthoc_refinement' in x for x in selected)))
    result=dict(evidence='deterministic replay of frozen held-out counts; no resampling',
        original_contract_sha256=report['contract_sha256'],original_counts_sha256=report['counts_archive_sha256'],
        recomputation_tolerance=1e-12,workers=workers,records=records,cases=summaries,
        safe_maximum_interpretation='maximum of two simultaneously valid lower bounds on the SAME covered event; no additional alpha; supplement preserves raw frozen comparison',
        refinement_interpretation='higher-grid/depth scores only diagnose marginal numerical sign loss after production; not substituted into frozen results')
    (root/'results'/'calibrated_benchmark_audit.json').write_text(json.dumps(nonlinear.jsonable(result),sort_keys=True,indent=2)+'\n')
    return result


def _production_one(task):
    """Independent indexed stream; execution order cannot change its records."""
    ci,rep,case,contract=task;a,b,f=model(case);science=np.zeros(3,dtype=np.int32);raw=[]
    for setting,(lag,n) in enumerate(zip(contract['science_lags'],contract['science_records'])):
        rng=np.random.default_rng(np.random.SeedSequence([contract['seed'],ci,rep,setting]))
        record=ring_records(rng,n,lag,a,b,f,gain=case['gain'])
        science[setting]=int(record['same'].sum())
        if ci==0 and rep==0:record['setting']=np.full(n,setting,dtype=np.uint8);raw.append(record)
    rng=np.random.default_rng(np.random.SeedSequence([contract['seed'],ci,rep,3]))
    calcounts,crecord=calibration_records(rng,contract['calibration_records'],f,case['calibration_gain'])
    if ci==0 and rep==0:
        crecord['setting']=np.full(contract['calibration_records'],3,dtype=np.uint8);raw.append(crecord)
    full_raw={key:np.concatenate([r[key] for r in raw]) for key in raw[0]} if raw else None
    return science,calcounts,full_raw,analyze(science,calcounts,contract)


def run(root,workers=4):
    from concurrent.futures import ProcessPoolExecutor
    path=root/'contracts'/'calibrated_benchmark.json'
    if not path.exists():raise RuntimeError('Freeze and review contract before production')
    contract=json.loads(path.read_text())
    if contract!=specification():raise RuntimeError('Code specification changed after freeze')
    sha=hashlib.sha256(path.read_bytes()).hexdigest();out=root/'results';out.mkdir(parents=True,exist_ok=True)
    all_counts={};cases=[];nsets=contract['datasets_per_case']
    for ci,case in enumerate(contract['cases']):
        a,b,f=model(case);truth=oracle(case);science=np.zeros((nsets,3),dtype=np.int32);calcounts=np.zeros((nsets,4),dtype=np.int32)
        flags={name:[] for name in ['calibrated_pinsker','calibrated_nonlinear','uncalibrated_extra_lag','legacy_extra_lag']}
        rates={name:[] for name in flags};gaps=[];calcover=[];regioncover=[];jointcover=[];solver=[];invalid=0
        details=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes=list(pool.map(_production_one,[(ci,rep,case,contract) for rep in range(nsets)],chunksize=10))
        for rep,(science_row,cal_row,raw,result) in enumerate(outcomes):
            science[rep]=science_row;calcounts[rep]=cal_row
            if raw is not None:deterministic_npz(out/'calibrated_benchmark_raw.npz',raw)
            if result['status']!='analyzed':
                invalid+=1
                for name in flags:
                    key={'uncalibrated_extra_lag':'alternative','legacy_extra_lag':'legacy_alternative'}.get(name)
                    value=result[key]['entropy_production_rate_lower'] if key else F(0)
                    flags[name].append(value>0);rates[name].append(float(value))
                calcover.append(False);regioncover.append(False);jointcover.append(False)
                continue
            cr=result['calibration'];ml,mh=map(float,cr['mean_interval']);al,ah=map(float,cr['second_moment_interval'])
            covered=ml<=truth['mean']<=mh and al<=truth['A']<=ah
            truecontrast=sum(float(F(c))*p for c,p in zip(contract['calibrated']['coefficients'],truth['probabilities']))
            rc=float(result['region']['lower'])<=truecontrast
            calcover.append(covered);regioncover.append(rc);jointcover.append(covered and rc)
            values=dict(calibrated_pinsker=result['calibrated_pinsker_rate'],
                        calibrated_nonlinear=result['nonlinear']['certificate']['entropy_production_rate_lower'],
                        uncalibrated_extra_lag=result['alternative']['entropy_production_rate_lower'],
                        legacy_extra_lag=result['legacy_alternative']['entropy_production_rate_lower'])
            for name,value in values.items():flags[name].append(value>0);rates[name].append(float(value))
            if result['nonlinear']['certified_relaxation_gap_upper'] is not None:
                gaps.append(float(result['nonlinear']['certified_relaxation_gap_upper']))
                solver.append(bool(result['nonlinear']['proposal_success']))
            if rep in [0,1]:details.append(result)
            if (rep+1)%100==0:print(f"{case['name']}: {rep+1}/{nsets}",flush=True)
        all_counts[case['name']+'__science_counts']=science;all_counts[case['name']+'__calibration_counts']=calcounts
        for name,values in rates.items():all_counts[case['name']+'__'+name+'_rate']=np.asarray(values)
        nonlinear_values=np.asarray(rates['calibrated_nonlinear']);linear_values=np.asarray(rates['calibrated_pinsker'])
        both=linear_values>0;ratios=nonlinear_values[both]/linear_values[both]
        cases.append(dict(case=case['name'],contract=case['contract'],oracle=truth,
            inference={name:summary(values) for name,values in flags.items()},
            rate_statistics={name:dict(mean=float(np.mean(v)),median=float(np.median(v)),
                minimum=float(np.min(v)),maximum=float(np.max(v)),
                mean_fraction_of_true_entropy=None if truth['stationary_entropy_rate']==0 else float(np.mean(v)/truth['stationary_entropy_rate'])) for name,v in rates.items()},
            nonlinear_to_pinsker_ratio_on_positive_pinsker=dict(datasets=int(both.sum()),
                mean=None if not len(ratios) else float(ratios.mean()),minimum=None if not len(ratios) else float(ratios.min()),maximum=None if not len(ratios) else float(ratios.max())),
            nonlinear_minus_pinsker_mean=float(np.mean(nonlinear_values-linear_values)),
            calibration_coverage=summary(calcover),science_halfspace_coverage=summary(regioncover),joint_coverage=summary(jointcover),
            relaxation_gap=dict(optimized_datasets=len(gaps),mean=None if not gaps else float(np.mean(gaps)),max=None if not gaps else float(np.max(gaps))),
            numerical_proposal_successes=int(sum(solver)),inconsistent_calibration_regions=invalid,
            illustrative_exact_certificates=details))
        print(case['name']+': '+', '.join(f'{name}={sum(v)}/{nsets}' for name,v in flags.items()),flush=True)
    archive=out/'calibrated_benchmark_counts.npz';deterministic_npz(archive,all_counts)
    report=dict(evidence=contract['evidence'],contract_sha256=sha,seed=contract['seed'],cases=cases,
        costs={'calibrated_arm':contract['calibrated_arm'],'extra_lag_arm':contract['alternative_arm'],'limitations':contract['cost_limits']},
        exact_two_lag_equilibrium_countermodel=equilibrium_countermodel(),
        counts_archive=archive.name,counts_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        raw_archive='calibrated_benchmark_raw.npz',raw_archive_sha256=hashlib.sha256((out/'calibrated_benchmark_raw.npz').read_bytes()).hexdigest(),
        scope='finite stated model/menu/seed; pointwise Monte Carlo intervals; no universal ranking, physical measurement, calibrated energy, or total-dissipation reconstruction')
    (out/'calibrated_benchmark.json').write_text(json.dumps(nonlinear.jsonable(report),sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--freeze',action='store_true');parser.add_argument('--audit',action='store_true');parser.add_argument('--with-audit',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.freeze:print(freeze(args.root))
    elif args.audit:audit(args.root,args.workers)
    else:
        run(args.root,args.workers)
        if args.with_audit:audit(args.root,args.workers)
