"""Finite-library same-data contrast selection with simultaneous coverage."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
try:
    from . import joint_analog
except ImportError:
    import joint_analog
from constructive_access.frontier import energy_certificate as ec


def certify_library(records, means, parameters, alpha=F(1,20),
                    external_failure=0, library_selection='predeclared', **kwargs):
    """Select a contrast using these data after covering every library member.

    An external apparatus failure event is counted once, rather than once per
    candidate. Data may be shared across candidates; no independence between
    their estimates is used. Physical resources count each acquisition once.
    """
    if library_selection not in ('predeclared','independent_pilot'):
        raise ValueError('Declare the candidate library before these observations')
    alpha, eta=map(ec.exact,(alpha,external_failure))
    if not 0<alpha<1 or not 0<=eta<1 or alpha+eta>=1:
        raise ValueError('Invalid statistical or external failure budget')
    values=sorted(set(map(ec.exact,parameters)))
    if not values:raise ValueError('At least one candidate is required')
    if any(k in kwargs for k in ('parameter','selection')):
        raise ValueError('The simultaneous wrapper owns parameter selection')
    certificates=[joint_analog.certify_means(records,means,parameter=s,
        alpha=alpha/len(values),external_failure=0,selection='predeclared',**kwargs)
        for s in values]
    # Every candidate uses the same lag; both direct bridges increase with gap.
    best=max(range(len(values)),key=lambda i:certificates[i]['reversible_gap_lower'])
    winner=certificates[best]
    gap=winner['reversible_gap_lower'];delta=winner['lag_unit']
    return dict(status=winner['status'],selected_parameter=values[best],
        simultaneous_gap_lower=gap,entropy_production_rate_lower=4*gap/delta,
        bridge='direct linear semigroup bound, detector conditional mean range at most 2',
        confidence_failure_upper=alpha+eta,external_failure_upper=eta,
        candidate_statistical_failure=alpha/len(values),candidate_count=len(values),
        candidates=certificates,library_selection=library_selection,
        acquisitions=winner['acquisitions'],detector_readouts=winner['detector_readouts'],
        declared_evolution_time=winner['declared_evolution_time'],
        interpretation='All library gaps are simultaneous. No independence between candidates or extra acquisition cost is presumed. Preparation and apparatus overhead still require calibration.')


def run(root):
    import numpy as np
    from constructive_access.calibrated.analog_benchmark import wilson
    contract_path=root/'contracts/design.json';raw=contract_path.read_bytes()
    contract=json.loads(raw)
    if not contract['frozen_before_analysis']:raise ValueError('Freeze before analysis')
    source=root/'results/joint_analog.json';bank=json.loads(source.read_text())
    reports=[];summary=[]
    for row in bank['replicate_results']:
        stat=row['exact_statistics'];nc=stat['calibration_records'];den=stat['calibration_denominator']
        B=F(stat['calibration_sum_squares'],nc*den**2)
        science=stat['science'];n=[nc]+[r['records'] for r in science]
        means=[(1+B)/2]+[F(r['records']*r['product_denominator']+r['product_sum'],2*r['records']*r['product_denominator']) for r in science]
        result=certify_library(n,means,contract['parameters'],alpha=F(contract['alpha']),lag_unit=F(bank['design']['lag_unit']))
        # Fixed comparators receive their own alpha, not the divided library alpha.
        fixed={}
        for s in bank['design']['parameters']:
            c=joint_analog.certify_means(n,means,parameter=F(s),alpha=F(contract['alpha']),lag_unit=F(bank['design']['lag_unit']))
            fixed[s]=float(4*c['reversible_gap_lower']/c['lag_unit'])
        reports.append(dict(case=row['case'],replicate=row['replicate'],selected_parameter=str(result['selected_parameter']),
            library_rate=float(result['entropy_production_rate_lower']),fixed_rates=fixed))
    for case in bank['summaries']:
        rows=[r for r in reports if r['case']==case['case']];N=len(rows)
        rates=np.array([r['library_rate'] for r in rows]);positives=int((rates>0).sum())
        fixed={s:dict(positive_certificates=sum(r['fixed_rates'][s]>0 for r in rows),mean_rate_lower=float(np.mean([r['fixed_rates'][s] for r in rows]))) for s in bank['design']['parameters']}
        selected={s:sum(r['selected_parameter']==s for r in rows) for s in contract['parameters']}
        summary.append(dict(case=case['case'],contract_valid=case['contract_valid'],replicates=N,
            positive_certificates=positives,wilson95=wilson(positives,N),mean_rate_lower=float(rates.mean()),
            fixed=fixed,selected_counts=selected))
    result=dict(contract=contract,contract_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summaries=summary,replicate_results=reports,
        evidence='Replay of frozen synthetic records; no additional acquisition or empirical measurement')
    (root/'results/design.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);run(p.parse_args().root)
