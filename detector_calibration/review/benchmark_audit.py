"""Independent sufficient-count and confidence-event replay of the frozen bank.

No new random records are drawn. Population truth is enclosed by the rational
law solver; empirical event coverage is a diagnostic, not the coverage proof.
"""
from fractions import Fraction as F
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiment'))
sys.path.insert(0,str(ROOT/'code'))
import certification as cert
import benchmark as bank

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_scores(category,h,b,kappa0):
    """K=4 Newton/product identities, independent of elementary() implementation."""
    left,plus,right=category
    spin=F(2*plus-4)
    e1=spin/4
    e2=(spin*spin-4)/12
    e4=F((-1)**(4-plus))
    e3=e4*e1
    s0=(-left*right+b*(left+right)*e1-b*b*e2)/(h*h)
    s1=(-left*right*e2+b*(left+right)*e3-b*b*e4)/(h*h)
    return {'S0':s0,'S1':s1,'U1':e2,'V':(e2-F(left+right,2)*e1)/h,
            'R0':s0/(kappa0*kappa0),'R1':s0/kappa0**2+s1/kappa0**4}

def classify(interval,population):
    lo,hi=interval;point,radius=population
    if lo<=point-radius and point+radius<=hi:return 'covered'
    if point+radius<lo or point-radius>hi:return 'missed'
    return 'boundary_unresolved'

def independent_sample(law,records,seed):
    # A separately batched integer implementation; no floating thresholds.
    import numpy as np
    generator=np.random.PCG64(seed)
    cumulative=np.cumsum(np.asarray(law['dyadic_numerators'],dtype=np.uint64),dtype=np.uint64)
    totals=np.zeros(len(cumulative),dtype=np.int64)
    for start in range(0,records,262139):
        count=min(262139,records-start)
        raw=generator.random_raw(count)>>np.uint64(64-law['dyadic_bits'])
        index=np.searchsorted(cumulative,raw,side='right')
        totals+=np.bincount(index,minlength=len(cumulative))
    return totals.tolist()

def audit():
    contract_path=ROOT/'experiment/contract.json'
    contract=json.loads(contract_path.read_text())
    if contract['source_sha256']!=bank.source_bindings():raise AssertionError('Frozen source changed')
    if contract['order']!=1 or contract['center_replicas']!=4:raise AssertionError('Audit fixture is the frozen K=4 design')
    raw_path=ROOT/'experiment/benchmark_counts.json.gz'
    raw=json.loads(gzip.decompress(raw_path.read_bytes()))
    report=json.loads((ROOT/'experiment/benchmark.json').read_text())
    assert raw['contract_sha256']==report['contract_sha256']==sha(contract_path)
    assert report['counts_sha256']==sha(raw_path)
    h=F(contract['h']);b=1-F(contract['lambda_'])*h;k0=F(contract['nominal_gain'])
    assert 0<=b<=1
    categories=raw['category_order']
    assert categories==[list(c) for c in cert.category_order(1)]
    score_rows=[independent_scores(c,h,b,k0) for c in categories]
    law_population=[];law_charges=[];law_data=[]
    for case,binding in zip(contract['cases'],report['law_files']):
        path=ROOT/binding['path'];assert sha(path)==binding['sha256']
        law=json.loads(path.read_text());science=law['science'];law_data.append(law)
        assert science['outcomes']==categories
        probabilities=list(map(F,science['probability_centers']))
        radii=list(map(F,science['probability_radii']))
        law_population.append({key:(sum((p*row[key] for p,row in zip(probabilities,score_rows)),F(0)),
                                     sum((r*abs(row[key]) for r,row in zip(radii,score_rows)),F(0)))
                               for key in score_rows[0]})
        charge=contract['science_records']*F(science['total_variation_upper'])
        charge+=contract['reference_records']*sum((F(law[name]['total_variation_upper']) for name in ('gain','step')),F(0))
        law_charges.append(min(F(1),charge))
        gamma,omega=F(case['gamma']),F(case['omega'])
        mp,mm,rp,rm=map(F,contract['generator_bounds'])
        assert gamma*gamma+omega*omega<=min(mp,mm)**2
        assert gamma*gamma+omega*omega<=min(rp,rm)
    counters=[{'name':case['name'],'datasets':0,'joint_target_event_covered':0,
               'joint_target_event_missed':0,'joint_target_event_boundary_unresolved':0,
               'simultaneous_target_and_reference_inclusion':0,'calibration_region_inclusion':0,'calibration_usable':0,
               'finite_gamma_upper_bounds':0,'unbounded_gamma_abstentions':0,
               'gamma_upper_coverage':0,'positive_geometry':0,'positive_endpoint':0,
               'geometry_overshoots':0,'endpoint_overshoots':0,'combined_positive':0,'combined_overshoots':0}
              for case in contract['cases']]
    values=[{'geometry':[],'endpoint':[],'combined':[]} for _ in counters]
    seen=set();all_seeds=set();bank_tv=F(0)
    for row in raw['records']:
        ci,rep=row['case_index'],row['replicate'];case=contract['cases'][ci]
        key=(ci,rep);assert key not in seen;seen.add(key)
        assert 0<=rep<contract['repetitions'] and row['case']==case['name']
        base=contract['seed_base']+100000*ci+10*rep
        assert row['seeds']==dict(science=base,gain=base+1,step=base+2)
        for seed in row['seeds'].values():assert seed not in all_seeds;all_seeds.add(seed)
        counts=row['science_counts'];n=contract['science_records']
        assert len(counts)==len(categories) and all(type(x) is int and x>=0 for x in counts) and sum(counts)==n
        for key in ('gain_plus','step_plus'):assert 0<=row[key]<=contract['reference_records']
        summary=cert.summarize_counts(dict(zip(map(tuple,categories),counts)),1,h,F(contract['lambda_']),k0)
        for key,item in summary['statistics'].items():
            assert item['sum']==sum((c*s[key] for c,s in zip(counts,score_rows)),F(0))
            assert item['sum_squares']==sum((c*s[key]**2 for c,s in zip(counts,score_rows)),F(0))
        confidence=cert.confidence_statistics(summary,F(contract['alpha_target']))
        assert confidence['confidence_event_count']==5
        statuses=[classify(item['interval'],law_population[ci][key]) for key,item in confidence['statistics'].items()]
        event='missed' if 'missed' in statuses else ('boundary_unresolved' if 'boundary_unresolved' in statuses else 'covered')
        result=bank.evaluate_counts(contract,case,counts,row['gain_plus'],row['step_plus'])
        assert cert.jsonable(result)==row['certificate']
        assert F(row['simulation_product_law_tv_upper'])==law_charges[ci]
        bank_tv+=law_charges[ci]
        c=counters[ci];c['datasets']+=1;c['joint_target_event_'+event]+=1
        gi=result['gain_region'];ti=result['width_region']
        inclusion=gi is not None and ti is not None and gi[0]<=F(case['gain'])<=gi[1] and ti[0]<=F(case['exposure'])<=ti[1]
        c['simultaneous_target_and_reference_inclusion']+=int(event=='covered' and inclusion)
        c['calibration_region_inclusion']+=int(inclusion)
        c['calibration_usable']+=int(result['calibration_status']=='covered_region')
        upper=result['gamma_upper'];c['finite_gamma_upper_bounds']+=int(upper is not None)
        c['unbounded_gamma_abstentions']+=int(upper is None)
        c['gamma_upper_coverage']+=int(upper is None or upper>=F(case['gamma']))
        geom,endpoint=result['entropy_lower'],result['endpoint_lower']
        c['positive_geometry']+=int(geom>0);c['positive_endpoint']+=int(endpoint>0)
        sigma=F(case['omega'])**2/F(case['gamma'])
        c['geometry_overshoots']+=int(geom>sigma);c['endpoint_overshoots']+=int(endpoint>sigma)
        combined=max(geom,endpoint)
        c['combined_positive']+=int(combined>0);c['combined_overshoots']+=int(combined>sigma)
        values[ci]['geometry'].append(geom);values[ci]['endpoint'].append(endpoint);values[ci]['combined'].append(combined)
    assert len(seen)==len(contract['cases'])*contract['repetitions']
    assert min(F(1),bank_tv)==F(report['whole_bank_product_tv_upper'])
    for counter,reported,vals in zip(counters,report['cases'],values):
        assert counter['datasets']==contract['repetitions']==reported['datasets']
        assert counter['positive_geometry']==reported['positive']
        assert counter['positive_endpoint']==reported['endpoint_positive']
        assert counter['geometry_overshoots']==reported['sigma_overshoots']
        assert counter['endpoint_overshoots']==reported['endpoint_overshoots']
        for short,long in (('geometry','entropy_lower'),('endpoint','endpoint_lower')):
            exact_mean=sum(vals[short],F(0))/len(vals[short])
            assert math.isclose(float(exact_mean),reported['metrics'][long]['mean'],rel_tol=1e-12,abs_tol=1e-15)
            counter[short+'_mean_diagnostic']=float(exact_mean)
        counter['combined_mean_diagnostic']=float(sum(vals['combined'],F(0))/len(vals['combined']))
        n=contract['science_records'];nr=contract['reference_records']
        resources=reported['resources_per_dataset']
        assert resources['target_detector_readouts']==6*n
        assert resources['reference_detector_readouts']==2*nr
        assert resources['total_detector_readouts']==6*n+2*nr
        assert F(resources['modeled_target_evolution_time'])==n*(2*h+F(reported['exposure']))
    selected=[]
    for ci,replicate in ((0,0),(1,399),(3,399)):
        row=next(r for r in raw['records'] if r['case_index']==ci and r['replicate']==replicate)
        laws=law_data[ci]
        science=independent_sample(laws['science'],contract['science_records'],row['seeds']['science'])
        gain=independent_sample(laws['gain'],contract['reference_records'],row['seeds']['gain'])
        step=independent_sample(laws['step'],contract['reference_records'],row['seeds']['step'])
        assert science==row['science_counts']
        assert gain[1]==row['gain_plus'] and step[1]==row['step_plus']
        selected.append({'case':row['case'],'replicate':replicate,'all_counts_match':True})
    return {'status':'complete_sufficient_count_replay','datasets':len(seen),'cases':counters,
            'independently_batched_seed_replay':selected,
            'whole_bank_product_tv_upper':str(min(F(1),bank_tv)),
            'scope':'Population event inclusion uses rational law enclosures; no unresolved-boundary event is silently called covered. Gamma abstention is explicitly an infinite upper bound. Selected saved seeds were regenerated with a separately batched integer sampler; no new dataset or experiment was added.'}

if __name__=='__main__':
    result=audit()
    path=ROOT/'review/benchmark_audit.json'
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'datasets':result['datasets'],'output':str(path)}))
