"""Deterministic prospective-design calculations; these are not observed data.

Exact enclosed circle laws supply population means and second moments. A
population plug-in diagnostic evaluates the confidence formulas after replacing
random sufficient statistics by their expectations. It is neither a power
estimate nor a confidence certificate, and cannot establish finite-sample
coverage or success. No pseudorandom science or reference records are drawn.
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


def population_statistics(law,m,h,lam=F(1)):
    rows=[cert.scores(category,m,h,lam) for category in law['outcomes']]
    keys=tuple(rows[0]);p=law['probability_centers'];radii=law['probability_radii']
    means={key:sum((prob*row[key] for prob,row in zip(p,rows)),F(0)) for key in keys}
    seconds={key:sum((prob*row[key]**2 for prob,row in zip(p,rows)),F(0)) for key in keys}
    ranges=cert.score_ranges(m,h,lam)
    stats={key:dict(mean=means[key],second=seconds[key],
                   mean_error=sum((rad*abs(row[key]) for rad,row in zip(radii,rows)),F(0)),
                   second_error=sum((rad*row[key]**2 for rad,row in zip(radii,rows)),F(0)),
                   range=ranges[key]) for key in keys}
    covariance={a:{b:sum((prob*row[a]*row[b] for prob,row in zip(p,rows)),F(0))-means[a]*means[b]
                   for b in keys} for a in keys}
    return stats,covariance


def prospective_diagnostic(stats,m,h,n,nref=100000):
    # Deliberately not a count table: these totals are population expectations.
    # The variance/sample-size convention matches the eventual exact code,
    # but no inference or probability claim is attached to this diagnostic.
    summary=dict(records=n,order=m,h=h,lambda_=F(1))
    summary['lambda']=summary.pop('lambda_')
    summary['statistics']={key:dict(sum=n*s['mean'],sum_squares=n*s['second'],range=s['range'])
                           for key,s in stats.items()}
    interval=cert.confidence_statistics(summary,F(1,40))
    calibration=cert.calibrate_references(nref,19*nref//20,nref,59*nref//100,
        F(1,500000),(F(1,200000),F(1,50000)),F(1,40))
    result=cert.certify(interval,calibration,(4,4,13,13))
    # Preserve the original full-order-only exploratory design after the API
    # gained a covered prefix maximum. This replays the recorded failures.
    if 'order_results' in result:
        full=result['order_results'][-1]
        result={**result,'entropy_lower':full['entropy_lower'],
                'P_lower':full['P_lower'],'biases':full['biases']}
    raw_s0=cert.normalized_interval(interval['statistics']['S0']['interval'],2,calibration['gain'])[0]
    # S0's finite-exposure bias is delta_N at m=0; finite-lag beta does not
    # enter the separate endpoint time-reversal bridge supplied by root.
    blur0=cert.bias_bounds(0,h,1,calibration['width'][1],(4,4,13,13))['delta_N']
    baseline_s=max(F(0),raw_s0-blur0)
    baseline=2*float(h)**3*float(baseline_s)**2/(2*float(1-h)+math.sqrt(2))**2
    return dict(records=n,reference_records_each=nref,
        diagnostic_only_not_certificate=True,
        population_plugin_geometry_lower=float(result['entropy_lower']),
        population_plugin_endpoint_lower=baseline,
        P_lower=float(result['P_lower']),A_lower=float(result['A_lower']),
        gamma_upper=float(result['gamma_upper']) if result['gamma_upper'] is not None else None,
        gain_region=list(map(float,calibration['gain'])),width_region=list(map(float,calibration['width'])),
        bias_beta=float(result['biases']['beta']),bias_delta_N=float(result['biases']['delta_N']),
        radii={key:float(value['radius']) for key,value in interval['statistics'].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--law-dir',type=Path,required=True)
    args=parser.parse_args();args.law_dir.mkdir(parents=True,exist_ok=True)
    cases=[];h=F(1,50);tau=F(1,100000);gain=F(9,10)
    for gamma in (F(1),F(1,2),F(2)):
        for omega in (3,1,0):
            for m in (0,1,2):
                model=em.Model(gamma,omega,h,tau,2*m+2,gain)
                law=em.outcome_law(model,collapse_centers=True)
                filename=f'law_gamma{str(gamma).replace("/","_")}_omega{omega}_m{m}.json'
                lawtext=json.dumps(law,indent=2,default=em.encode)+'\n'
                (args.law_dir/filename).write_text(lawtext)
                stats,covariance=population_statistics(law,m,h)
                case=dict(gamma=str(gamma),omega=omega,order=m,
                    simulator_entropy=float(F(omega*omega)/gamma),
                    law_file=filename,law_sha256=hashlib.sha256(lawtext.encode()).hexdigest(),
                    law_tv_upper=float(law['total_variation_upper']),
                    statistics={key:dict(mean=float(s['mean']),standard_deviation=math.sqrt(float(s['second']-s['mean']**2)),
                        mean_enclosure_radius=float(s['mean_error']),second_enclosure_radius=float(s['second_error']),
                        range=list(map(float,s['range']))) for key,s in stats.items()},
                    covariance={a:{b:float(value) for b,value in row.items()} for a,row in covariance.items()},
                    design_diagnostics=[prospective_diagnostic(stats,m,h,n) for n in (2000000,10000000,50000000)],
                    resources=law['resource_per_record'])
                cases.append(case)
                print(f"gamma={gamma} omega={omega} m={m}: "+', '.join(
                    f"n={d['records']}: geometry={d['population_plugin_geometry_lower']:.5g}, endpoint={d['population_plugin_endpoint_lower']:.5g}"
                    for d in case['design_diagnostics']),flush=True)
    report=dict(status='unfrozen_population_design_diagnostic_no_random_records',
        assumptions='Exact enclosed circle simulator; broad-class inference assumptions are separate. Numbers are diagnostic floating displays of exact rational computations.',
        warning='Population plug-in values are not finite-sample confidence certificates or estimated detection probabilities.',
        calibration=dict(gain='9/10',exposure='1/100000',lag='1/50',step_time='1/500000',
                         prior_width=['1/200000','1/50000'],caps=[4,4,13,13],lambda_=1),
        source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                      (Path(__file__),Path(em.__file__),Path(cert.__file__))},cases=cases)
    args.output.write_text(json.dumps(report,indent=2,default=em.encode)+'\n')


if __name__=='__main__':main()
