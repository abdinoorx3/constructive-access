"""Reproduce three calibrated interface examples; no apparatus is measured."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path
from constructive_access import (
    identify_linear_target, minimax_certificate, verify_minimax_certificate,
    count_cap, verify_count_cap, declared_energy_cap,
    AdequacyMonitor, Interval,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    matrix=[[1,1],['1/2',1]];target=[1,0];errors=['1/100','1/100']
    certificate=minimax_certificate(matrix,target,errors)
    checked=verify_minimax_certificate(matrix,target,errors,certificate)
    if checked['minimax_absolute_error'] != '1/25':
        raise AssertionError('Incorrect calibrated target radius')
    observed=[F(4),F(5,2)]
    estimate=sum(F(w)*y for w,y in zip(certificate['weights'],observed))
    cap=count_cap(10,'1/100')
    if not verify_count_cap(cap):
        raise AssertionError('Energy count certificate failed')
    bound=declared_energy_cap(cap,'3/2',4)
    result={
        'evidence_type':'Exact analytic examples with declared inputs; no physical measurements.',
        'linear_target':{'fingerprint':matrix,'target':target,'measurement_error_box':errors,
                         'certificate':certificate,'independent_check':checked,
                         'example_observation':list(map(str,observed)),
                         'target_estimate':str(estimate),
                         'deterministic_interval':[str(estimate-F(1,25)),str(estimate+F(1,25))],
                         'condition':'The full positive parameter orthant, known response matrix, actual fingerprint access and full coordinatewise error box.'},
        'unidentifiable_before_selective_probe':identify_linear_target([[1,1]],target),
        'unidentified_parameters_but_identified_sum':identify_linear_target([[1,1]],[1,1]),
        'declared_resource':{'count_certificate':cap,'work_per_event_upper':'3/2','apparatus_upper':'4',
                             'gross_work_cap':str(bound),'unit':'one consistently calibrated physical work unit, supplied by user',
                             'condition':'Nonexplosive unit-jump counting process; actual predictable compensator <=10, gross work <=3/2 per event plus apparatus <=4. These premises are not inferred from the observed state law.'}}
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(json.dumps({'linear_target_estimate':str(estimate),'certified_error':certificate['absolute_error'],
                      'count_cap':cap['count_cap'],'declared_gross_work_cap':str(bound)}))


if __name__=='__main__':main()
