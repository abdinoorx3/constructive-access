"""Frozen independent-record virtual detector experiment, using exact law enclosures.

The simulator is a circle diffusion; the certificate uses the separately stated
local-geometry class and reference-calibration contract. There are no physical
laboratory observations. Integer categorical sampling uses PCG64 random bits.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from fractions import Fraction as F
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys
import numpy as np
import exposure_moments as em
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import certification as cert

BASE=Path(__file__).resolve().parent
WORK={}


def timestamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(value):return json.dumps(value,indent=2,sort_keys=True,default=em.encode)+'\n'


def design():
    return dict(schema_version=1,scope='simulated_stationary_circle_with_declared_detector_contract',
        order=1,center_replicas=4,science_records=13333333,reference_records=100000,
        repetitions=400,h='7/200',lambda_='1',nominal_gain='9/10',
        alpha_target='1/25',alpha_calibration='1/100',
        width_prior=['1/200000','1/50000'],gain_prior=['0','1'],
        step_time='1/500000',generator_bounds=['4','4','13','13'],
        seed_base=1032026400000,
        seed_rule='seed_base+100000*case_index+10*replicate+stream; streams science0,gain1,step2',
        dyadic_bits=63,moment_tv_tolerance=str(F(1,1<<90)),
        cases=[dict(name='driven',gamma='1',omega='3',gain='9/10',exposure='1/100000'),
               dict(name='equilibrium',gamma='1',omega='0',gain='9/10',exposure='1/100000'),
               dict(name='weak_driven',gamma='1',omega='1',gain='9/10',exposure='1/100000'),
               dict(name='changed_calibration',gamma='1/2',omega='3',gain='4/5',exposure='3/200000')],
        simulation_assumption='Independent uniform random bits idealized; implementation is NumPy PCG64. No claim of true randomness or laboratory validation.',
        comparison='Geometry closure certificate and broader-class endpoint bridge use the same science and calibration confidence event.',
        design_selection='Selected using recorded exact population design diagnostics; no stochastic pilot science records were used.',
        uncalibrated_costs='Energy, preparation/reset durations, replica construction and shutter-control energy are unmeasured; readout counts and modeled durations are reported.')


def source_bindings():
    return {'experiment/benchmark.py':sha(__file__),
            'experiment/exposure_moments.py':sha(em.__file__),
            'code/certification.py':sha(cert.__file__)}


def freeze(path):
    if path.exists():raise FileExistsError('A frozen contract already exists; refusing to overwrite it')
    contract={**design(),'frozen_at_utc':timestamp(),'source_sha256':source_bindings()}
    path.write_text(dump(contract))
    print(f'FROZEN {path} sha256={sha(path)}',flush=True)


def binary_law(plus_probability,bits=63):
    p=F(plus_probability)
    if not 0<=p<=1:raise ValueError('Binary probability outside [0,1]')
    grid=1<<bits;q=p.numerator*grid//p.denominator
    return dict(dyadic_bits=bits,dyadic_numerators=[grid-q,q],
                total_variation_upper=abs(F(q,grid)-p),outcomes=[-1,1])


def endpoint_lower(confidence,calibration,bounds):
    """Broader-class endpoint bridge, on the same covered S0/reference event."""
    if calibration['status']!='covered_region':return F(0)
    h=F(confidence['h']);lam=F(confidence['lambda']);b=1-lam*h
    if b<0:raise ValueError('This frozen endpoint normalization requires b>=0')
    raw=cert.normalized_interval(confidence['statistics']['S0']['interval'],2,calibration['gain'])[0]
    blur=cert.bias_bounds(0,h,lam,F(calibration['width'][1]),bounds)['delta_N']
    contrast=max(F(0),raw-blur)
    return 2*h**3*contrast**2/(2*b+cert.sqrt_upper(2))**2


def initialize(contract,laws):
    WORK.update(contract=contract,laws=laws)


def evaluate_counts(contract,case,counts,gain_plus,step_plus):
    m=contract['order'];h=F(contract['h']);lam=F(contract['lambda_'])
    mapping=dict(zip(cert.category_order(m),counts))
    summary=cert.summarize_counts(mapping,m,h,lam,F(contract['nominal_gain']))
    confidence=cert.confidence_statistics(summary,F(contract['alpha_target']))
    calibration=cert.calibrate_references(contract['reference_records'],gain_plus,
        contract['reference_records'],step_plus,F(contract['step_time']),
        tuple(map(F,contract['width_prior'])),F(contract['alpha_calibration']),
        gain_prior=tuple(map(F,contract['gain_prior'])))
    bounds=tuple(map(F,contract['generator_bounds']))
    result=cert.certify(confidence,calibration,bounds)
    endpoint=endpoint_lower(confidence,calibration,bounds)
    prefix=[]
    if 'order_results' in result:
        factor=max(F(0),result['A_lower'])/result['H_upper'] if result['H_upper']>0 else F(0)
        for row in result['order_results']:
            prefix.append(dict(order=row['order'],entropy_lower=row['entropy_lower'],
                termwise_lower=max(F(0),row['termwise_contrast_lower']-row['penalty_upper'])*factor,
                projected_lower=max(F(0),row['projected_contrast_lower']-row['penalty_upper'])*factor,
                P_lower=row['P_lower'],penalty_upper=row['penalty_upper']))
    sigma=F(case['omega'])**2/F(case['gamma'])
    covered=(calibration.get('status')=='covered_region' and
             calibration['gain'][0]<=F(case['gain'])<=calibration['gain'][1] and
             calibration['width'][0]<=F(case['exposure'])<=calibration['width'][1])
    gamma_upper=result.get('gamma_upper')
    return dict(status=result['status'],entropy_lower=result['entropy_lower'],
        endpoint_lower=endpoint,gamma_upper=gamma_upper,
        order_selected=result.get('order_selected'),prefixes=prefix,
        termwise_max=max((row['termwise_lower'] for row in prefix),default=F(0)),
        projected_max=max((row['projected_lower'] for row in prefix),default=F(0)),
        gain_region=calibration.get('gain'),width_region=calibration.get('width'),
        calibration_status=calibration['status'],calibration_covered=covered,
        gamma_covered=(gamma_upper is None or gamma_upper>=F(case['gamma'])),
        sigma_overshoot=result['entropy_lower']>sigma,
        endpoint_overshoot=endpoint>sigma)


def one_dataset(job):
    ci,replicate=job;contract=WORK['contract'];case=contract['cases'][ci];laws=WORK['laws'][ci]
    base=contract['seed_base']+100000*ci+10*replicate
    seeds=dict(science=base,gain=base+1,step=base+2)
    n=contract['science_records'];nr=contract['reference_records']
    science=em.sample_counts(laws['science'],n,seeds['science'])
    gain=em.sample_counts(laws['gain'],nr,seeds['gain'])
    step=em.sample_counts(laws['step'],nr,seeds['step'])
    charge=min(F(1),science['simulation_product_law_tv_upper']+
               gain['simulation_product_law_tv_upper']+step['simulation_product_law_tv_upper'])
    result=evaluate_counts(contract,case,science['counts'],gain['counts'][1],step['counts'][1])
    return dict(case=case['name'],case_index=ci,replicate=replicate,seeds=seeds,
        science_counts=science['counts'],gain_plus=gain['counts'][1],step_plus=step['counts'][1],
        simulation_product_law_tv_upper=charge,certificate=result)


def distribution(values):
    a=np.asarray(list(map(float,values)),dtype=float)
    return dict(mean=float(a.mean()),minimum=float(a.min()),median=float(np.median(a)),
                maximum=float(a.max()),standard_deviation=float(a.std(ddof=1)),
                q025=float(np.quantile(a,.025)),q975=float(np.quantile(a,.975)))


def aggregate(contract,records,lawfiles,contract_hash):
    cases=[]
    for ci,case in enumerate(contract['cases']):
        rows=[r for r in records if r['case_index']==ci];c=[r['certificate'] for r in rows]
        metrics={name:distribution(r[name] for r in c) for name in
                 ('entropy_lower','endpoint_lower','termwise_max','projected_max')}
        gamma=[r['gamma_upper'] for r in c if r['gamma_upper'] is not None]
        if gamma:metrics['gamma_upper']=distribution(gamma)
        for order in range(contract['order']+1):
            for field in ('entropy_lower','termwise_lower','projected_lower'):
                metrics[f'prefix{order}_{field}']=distribution(next((p[field] for p in r['prefixes'] if p['order']==order),F(0)) for r in c)
        n=contract['science_records'];nr=contract['reference_records'];h=F(contract['h']);tau=F(case['exposure'])
        cases.append(dict(**case,simulator_entropy=F(case['omega'])**2/F(case['gamma']),datasets=len(rows),
            positive=sum(r['entropy_lower']>0 for r in c),endpoint_positive=sum(r['endpoint_lower']>0 for r in c),
            calibration_covered=sum(r['calibration_covered'] for r in c),gamma_covered=sum(r['gamma_covered'] for r in c),
            sigma_overshoots=sum(r['sigma_overshoot'] for r in c),endpoint_overshoots=sum(r['endpoint_overshoot'] for r in c),
            selected_prefix_counts={str(order):sum(r['order_selected']==order for r in c) for order in range(contract['order']+1)},
            metrics=metrics,per_dataset_simulation_tv_upper=max(r['simulation_product_law_tv_upper'] for r in rows),
            resources_per_dataset=dict(target_detector_readouts=n*(contract['center_replicas']+2),
                reference_detector_readouts=2*nr,total_detector_readouts=n*(contract['center_replicas']+2)+2*nr,
                independent_stationary_preparations=n,controlled_reference_preparations=2*nr,
                target_exposure_windows=3*n,reference_exposure_windows=2*nr,
                modeled_target_evolution_time=n*(2*h+tau),
                sum_target_window_durations=3*n*tau,sum_reference_window_durations=2*nr*tau)))
    return dict(status='completed_frozen_virtual_experiment',data_origin='simulated',
        contract_sha256=contract_hash,source_sha256=source_bindings(),
        numpy_version=np.__version__,datasets=len(records),cases=cases,law_files=lawfiles,
        whole_bank_product_tv_upper=min(F(1),sum((r['simulation_product_law_tv_upper'] for r in records),F(0))),
        conclusions_scope='Finite-data behavior of the declared certificate on an enclosed circle simulator; no laboratory validation and no universal advantage over all inference methods.',
        confidence_scope='Per-dataset target+calibration failure budget1/20; simulation law discrepancy is separately charged. Coverage frequencies are diagnostics, not a proof of the theorem.',
        comparison_scope='The endpoint bridge has broader structural scope. Larger geometry certificates do not establish domination under identical model assumptions.')


def run(contract_path,workers):
    contract=json.loads(contract_path.read_text())
    if any(contract.get(k)!=v for k,v in design().items()):raise ValueError('Frozen design differs from this benchmark source')
    if contract['source_sha256']!=source_bindings():raise ValueError('Frozen source binding changed')
    if workers<1 or workers>4:raise ValueError('Use1..4 process workers')
    lawdir=BASE/'benchmark_laws';lawdir.mkdir(exist_ok=True)
    laws=[];lawfiles=[]
    for case in contract['cases']:
        model=em.Model(case['gamma'],case['omega'],contract['h'],case['exposure'],contract['center_replicas'],case['gain'])
        science=em.outcome_law(model,F(contract['moment_tv_tolerance']),contract['dyadic_bits'],True)
        gain=binary_law(em.reference_law(case['gain'])[1],contract['dyadic_bits'])
        step=binary_law(em.reference_law(case['gain'],case['exposure'],contract['step_time'])[1],contract['dyadic_bits'])
        bundle=dict(science=science,gain=gain,step=step);laws.append(bundle)
        path=lawdir/(case['name']+'.json');path.write_text(dump(bundle))
        lawfiles.append(dict(path='experiment/benchmark_laws/'+path.name,sha256=sha(path)))
    started=timestamp();records=[]
    jobs=[(ci,r) for ci in range(len(laws)) for r in range(contract['repetitions'])]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(contract,laws)) as pool:
        futures=[pool.submit(one_dataset,job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%25==0:print(f'completed{len(records)}/{len(jobs)}',flush=True)
    records.sort(key=lambda r:(r['case_index'],r['replicate']))
    raw=dict(schema_version=1,contract_sha256=sha(contract_path),category_order=cert.category_order(contract['order']),records=records)
    raw_bytes=dump(raw).encode();compressed=gzip.compress(raw_bytes,compresslevel=9,mtime=0)
    (BASE/'benchmark_counts.json.gz').write_bytes(compressed)
    report=aggregate(contract,records,lawfiles,sha(contract_path))
    report['counts_sha256']=sha(BASE/'benchmark_counts.json.gz')
    (BASE/'benchmark.json').write_text(dump(report))
    fixture=dict(contract=contract,dataset=records[0],data_origin='simulated')
    (BASE/'calibration_counts.json').write_text(dump(fixture))
    if not (BASE/'benchmark_run.json').exists():
        (BASE/'benchmark_run.json').write_text(dump(dict(started_at_utc=started,completed_at_utc=timestamp(),
            frozen_at_utc=contract['frozen_at_utc'],contract_sha256=sha(contract_path),workers=workers,
            note='First-generation wall-clock provenance; later numerical reproductions retain this original record.')))
    for case in report['cases']:
        print(case['name'],case['positive'],case['metrics']['entropy_lower']['mean'],
              'endpoint',case['metrics']['endpoint_lower']['mean'],'overshoots',case['sigma_overshoots'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--contract',type=Path,default=BASE/'contract.json')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.freeze:freeze(args.contract)
    else:run(args.contract,args.workers)


if __name__=='__main__':main()
