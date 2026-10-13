"""Replay exact finite-resolution inference on three frozen synthetic datasets.

The statistics are from experiment.json, replicate zero of each scenario.
They do not validate the physical/calibration assumptions stated in the reference.
No simulator, hidden drift parameter or physical apparatus is used by this example.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse,json
from constructive_access.observed_geometry import finite_resolution as fr
from constructive_access.observed_geometry import combination
from constructive_access.observed_geometry import consistency

FIXTURES = {'driven': {'records': 200000, 'weighted_total': 183044801, 'weighted_square_total': 22937690638535, 'endpoint_total': 21939647551, 'endpoint_square_total': 159952249205148379}, 'weak_driven': {'records': 200000, 'weighted_total': -35105791, 'weighted_square_total': 17566955124873, 'endpoint_total': 2546564112, 'endpoint_square_total': 121421502865611596}, 'equilibrium': {'records': 200000, 'weighted_total': -64118757, 'weighted_square_total': 17242573725021, 'endpoint_total': -296967669, 'endpoint_square_total': 118986911602110831}}
CONTRACT_SHA256 = "7aedd529884f112867dfb45e645e85760f5a33af2057fa8acae63115828cef0a"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    protocol=fr.make_protocol(h='1/25',q_abs_max='4/5',envelope_floor='9/25',
        generator_bounds=['19/6','19/6',10,10],global_gamma_upper=1)
    reading_error=F(1,2**20)
    score_error=4*(2/protocol['h']+protocol['q_abs_max'])*reading_error/(protocol['h']*protocol['envelope_floor'])+F(1,256)
    rows={}
    for name,data in FIXTURES.items():
        result=fr.certify_totals(records=data['records'],
            score_total=F(data['weighted_total'],256),
            score_square_total=F(data['weighted_square_total'],256**2),
            protocol=protocol,alpha='1/20',concentration='empirical_bernstein',
            quantization_radius=score_error)
        combined=combination.certify_shared_records(records=data['records'],
            geometry_total=data['weighted_total'],geometry_square_total=data['weighted_square_total'],
            geometry_scale=256,endpoint_total=data['endpoint_total'],
            endpoint_square_total=data['endpoint_square_total'],endpoint_scale=2**24,
            protocol=protocol,geometry_score_error=score_error,
            endpoint_score_error=2*(1+F(24,25))**2*reading_error+F(1,2**24),
            endpoint_parameter='24/25',alpha_total='1/20')
        rows[name]=dict(status=result['status'],
            covered_fallback_lower=str(combined['entropy_production_rate_lower']),
            covered_fallback_lower_decimal=float(combined['entropy_production_rate_lower']),
            covered_fallback_selected=combined['selected_component'],
            fallback_total_failure=str(combined['confidence_failure_upper']),
            entropy_rate_lower=str(result['entropy_production_rate_lower']),
            entropy_rate_lower_decimal=float(result['entropy_production_rate_lower']),
            sampling_radius_upper=str(result['sampling_radius_upper']),
            confidence_failure_upper=str(result['confidence_failure_upper']),
            independent_preparations=result['independent_preparations'],
            detector_readouts=result['triple_detector_readouts'],
            evolution_time=str(result['declared_evolution_time']))
    consistency_plan=consistency.design(1024)
    out=dict(evidence='Frozen synthetic sufficient statistics; conditional mathematical coverage, not laboratory validation.',
             consistency_design=dict(evidence='Deterministic design only; no corresponding records acquired.',
                 m=consistency_plan['m'],records=consistency_plan['records'],
                 sampling_radius_upper=str(consistency_plan['sampling_radius_upper']),
                 total_accuracy_gap_upper=str(consistency_plan['joint_accuracy_gap_upper']),
                 evolution_time=str(consistency_plan['declared_evolution_time'])),
             contract_sha256=CONTRACT_SHA256,
             common_calibration='Pointwise Gamma envelope and generator derivative caps are supplied assumptions; no hidden drift is fitted.',
             excluded_resources='Calibration, preparation/reset duration, exposure and energy remain unmeasured.',
             records=rows)
    content=json.dumps(out,sort_keys=True,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(content)
    else:print(content,end='')

if __name__=='__main__':main()
