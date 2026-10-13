"""Exact-law prospective diagnostics for fixed projections and covered prefixes.

No science/reference records are sampled. Values obtained by placing expected
sufficient statistics into a confidence formula are design diagnostics only;
they are neither confidence certificates nor estimates of success probability.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import exposure_moments as em
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import certification as cert


def population_statistics(law,m,h,k0):
    rows=[cert.scores(c,m,h,1,k0) for c in law['outcomes']]
    ranges=cert.score_ranges(m,h,1,k0)
    stats={}
    for key in rows[0]:
        mean=sum((p*r[key] for p,r in zip(law['probability_centers'],rows)),F(0))
        second=sum((p*r[key]**2 for p,r in zip(law['probability_centers'],rows)),F(0))
        error=sum((p*abs(r[key]) for p,r in zip(law['probability_radii'],rows)),F(0))
        stats[key]=dict(mean=mean,second=second,mean_enclosure_radius=error,range=ranges[key])
    return stats


def prospective(stats,m,h,tau,k0,bounds,n=10000000,nref=100000):
    summary={'records':n,'order':m,'h':h,'lambda':F(1),'nominal_gain':k0,
             'statistics':{key:{'sum':n*s['mean'],'sum_squares':n*s['second'],'range':s['range']}
                           for key,s in stats.items()}}
    confidence=cert.confidence_statistics(summary,F(1,40))
    calibration=cert.calibrate_references(nref,19*nref//20,nref,59*nref//100,
        tau/5,(tau/2,2*tau),F(1,40))
    result=cert.certify(confidence,calibration,bounds)
    return result,confidence


def numerical(value):
    if isinstance(value,F):return float(value)
    if isinstance(value,dict):return {k:numerical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [numerical(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refined-hardware',action='store_true')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    lawdir=base/'projected_laws';lawdir.mkdir(exist_ok=True)
    taus=(F(1,1000000),) if args.refined_hardware else (F(1,100000),)
    bounds=(F(16,5),F(16,5),F(10),F(10)) if args.refined_hardware else (F(4),F(4),F(13),F(13))
    cases=[];k0=F(9,10)
    for tau in taus:
        for h in (F(1,40),F(7,200),F(1,20)):
            for m in (0,1,2):
                model=em.Model(1,3,h,tau,2*m+2,k0)
                law=em.outcome_law(model,collapse_centers=True)
                name=f'law_h{h.numerator}_{h.denominator}_tau{tau.denominator}_m{m}.json'
                lawtext=json.dumps(law,indent=2,default=em.encode)+'\n'
                (lawdir/name).write_text(lawtext)
                stats=population_statistics(law,m,h,k0)
                result,confidence=prospective(stats,m,h,tau,k0,bounds)
                matched_records=80000000//(2*m+4)
                matched,_=prospective(stats,m,h,tau,k0,bounds,n=matched_records)
                joint_sd={str(l):math.sqrt(float(stats[f'R{l}']['second']-stats[f'R{l}']['mean']**2)) for l in range(m+1)}
                separate_sd={str(l):sum(math.sqrt(float(stats[f'S{j}']['second']-stats[f'S{j}']['mean']**2))/float(k0**(2*j+2)) for j in range(l+1)) for l in range(m+1)}
                case=dict(lag=str(h),exposure=str(tau),maximum_order=m,center_replicas=2*m+2,
                    law_file=name,law_sha256=hashlib.sha256(lawtext.encode()).hexdigest(),
                    law_tv_upper=float(law['total_variation_upper']),
                    population_statistics={key:dict(mean=float(s['mean']),sd=math.sqrt(float(s['second']-s['mean']**2)),
                        mean_enclosure_radius=float(s['mean_enclosure_radius']),range=list(map(float,s['range']))) for key,s in stats.items()},
                    projection_sd=joint_sd,sum_component_sd=separate_sd,
                    population_plugin=numerical(result),radii={key:float(item['radius']) for key,item in confidence['statistics'].items()},
                    matched_readout_diagnostic=dict(target_records=matched_records,
                        target_readouts=matched_records*(2*m+4),budget_ceiling=80000000,
                        population_plugin=numerical(matched)),
                    resources=law['resource_per_record'])
                cases.append(case)
                print(f"tau={tau} h={h} K={2*m+2}: "+', '.join(f"l{r['order']}={float(r['entropy_lower']):.5g}" for r in result['order_results'])+
                      f"; max={float(result['entropy_lower']):.5g}, Rsd={joint_sd}",flush=True)
    label='refined' if args.refined_hardware else 'initial'
    report=dict(status='unfrozen_exact_population_diagnostics_no_random_records',
        warning='Population plug-in numbers are not confidence certificates, coverage tests or detection probabilities.',
        hardware_contract=label,parameters=dict(gamma=1,omega=3,gain='9/10',nominal_gain='9/10',lambda_=1,
            caps=list(map(str,bounds)),target_records=10000000,reference_records_each=100000,
            width_prior='[actual_tau/2,2*actual_tau]',step_time='actual_tau/5',joint_failure_budget='1/20'),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(em.__file__),Path(cert.__file__))},cases=cases)
    (base/f'projected_pilot_{label}.json').write_text(json.dumps(report,indent=2,default=em.encode)+'\n')


if __name__=='__main__':main()
