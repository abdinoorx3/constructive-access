"""Installed APIs for calibrated thermodynamic inference and its limits.

Run from the release root after installing the package:
    python examples/calibrated_tutorial.py --output results/calibrated_tutorial.json

Every input below is an exact mathematical example or an explicitly synthetic
record. No empirical data, numerical optimization, or random sampling is used.
Certificates check mathematical contracts; they do not establish those physical
contracts for an apparatus.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

from constructive_access.calibrated import (
    analog, calibration, dependence, dissipation_limits, joint_calibration,
    nonlinear, sequential,
)


def compact(value):
    """Outward rational summaries; all assertions use exact values first."""
    if isinstance(value, F):
        scale = 10**16
        lower = F((value*scale).numerator//(value*scale).denominator, scale)
        upper = F(-((-value*scale).numerator//(-value*scale).denominator), scale)
        out = {'lower':str(lower), 'upper':str(upper)}
        if value and lower == 0:
            out['exact_small_value'] = str(value)
        return out
    if isinstance(value, dict):
        return {key:compact(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [compact(item) for item in value]
    return value


def examples():
    # A finite menu of positive-lag pair probabilities need not determine A.
    # This example concerns pair laws, not complete output trajectories.
    ambiguity = calibration.positive_lag_ambiguity(3)
    moment = lambda atoms,k: sum((w*x**k for x,w in atoms), F(0))
    assert ambiguity['second_moment_difference'] > 0
    for k in ambiguity['lags']:
        assert moment(ambiguity['positive_spectrum'],k) == moment(
            ambiguity['negative_spectrum'],k)
    steep = calibration.steep_decay_example()
    assert steep['violation_lower'] > F(3483,10**6)
    assert steep['agreement_drop_lower'] < steep['uncalibrated_reversible_upper']

    # Same-state paired calibration requires conditional detector independence.
    paired = calibration.paired_calibration([3125,1875,1875,3125])
    assert paired['status'] == 'valid_calibration_region'
    assert paired['conditional_second_moment_estimate'] == F(1,4)

    # A single contrast can retain the lag-zero calibration uncertainty rather
    # than paying for separate confidence regions. Zero lag contributes count
    # variance and detector operations even though its comparison duration is
    # zero under the ideal same-state, same-detector twin contract.
    joint = joint_calibration.certify_counts(
        [4000,8000,8000],[3000,5520,4720],parameter=F(19,25))
    assert joint['status'] == 'positive_certificate'
    assert joint['support']['reversible_support'] == -F(18,625)
    assert joint['pair_records'] == 20000 and joint['endpoint_readouts'] == 40000
    assert joint['declared_evolution_time'] == 24000
    assert F(6829,10**7) < joint['entropy_production_rate_lower'] < F(683,10**6)
    joint_control = joint_calibration.lag_zero_uncertainty_control()
    assert joint_control['true_population_gap'] == 0
    assert joint_control['one_lag_zero_record_failure_probability'] == F(1,4)

    # Explicit rational scores and a feasible primal point bracket a two-lag
    # calibrated record-KL relaxation. An upper bracket on that relaxation is
    # never an upper bound on the unknown physical entropy-production rate.
    common = dict(lags=[1,2], weights=[F(1,2)]*2,
                  vertices=[(0,F(1,2))], coefficients=[F(38,25),-1],
                  lower=F(1147,2500))
    scores = [(F(231,250),F(259,250)),(F(131,125),F(243,250))]
    dual_contract = dict(common, durations=[1,2], scores=scores)
    dual = nonlinear.certify(**dual_contract)
    assert nonlinear.verify(dual, **dual_contract)
    primal = nonlinear.primal_bound(
        **common, probabilities=[F(353,500),F(7679,12500)],
        atoms=[(0,F(29,40),1)])
    bracket_gap = primal['record_kl_relaxation_upper'] - dual['record_kl_lower']
    assert dual['record_kl_lower'] > F(1,1000)
    assert 0 <= bracket_gap < F(1,100000)

    # Separately knowing an upper bound on microscopic activity strengthens
    # the record-KL bridge. Detector changes alone do not calibrate activity.
    bridge = nonlinear.activity_bridge(F(1,2),1,1)
    assert bridge['status'] == 'certified_activity_bridge'
    assert bridge['entropy_production_rate_lower'] > F(269,100)
    assert bridge['linear_bridge_lower'] == 2

    # All jumps are visible through parity, yet their circulation can be hidden.
    equal_equilibrium = dissipation_limits.model([1]*4,F(1,2))
    equal_driven = dissipation_limits.model([1]*4,F(1,1000))
    path = [.1,.2,.3]  # This separate likelihood implementation uses floats.
    # The proof is exact. This equality is an independent floating likelihood
    # implementation check, not the certificate on which the proof rests.
    log_eq = dissipation_limits.parity_log_likelihood([1]*4,F(1,2),path)
    log_driven = dissipation_limits.parity_log_likelihood([1]*4,F(1,1000),path)
    assert abs(log_eq-log_driven) < 1e-12
    assert equal_equilibrium['entropy_production_interval'][1] == 0
    assert equal_driven['entropy_production_interval'][0] > 6
    rates = [F(51,50),F(26,25),F(49,50),F(24,25)]
    finite_pair = dissipation_limits.finite_budget_pair(rates,10,F(1,100),100)
    assert finite_pair['entropy_gap_lower'] > 200
    assert finite_pair['full_record_tv_upper'] <= F(1,200)
    forced_infinity = dissipation_limits.unavoidable_infinite_upper_probability(
        rates,finite_pair['reference_reverse_probability'],10)
    assert forced_infinity == F(189,200)
    repaired = dissipation_limits.direction_count_interval(rates,1000,10,F(1,1000))
    without_floor = dissipation_limits.direction_count_interval(rates,1000,10,0)
    no_reverse = dissipation_limits.direction_count_interval(rates,1000,0,0)
    assert repaired['status'] == without_floor['status'] == 'finite_interval'
    assert no_reverse['status'] == 'unbounded_upper'
    assert no_reverse['entropy_production_interval'][1] is None
    assert repaired['entropy_production_interval'][1] < F(528,100)

    # A manufactured record stream demonstrates exact predictable replay and
    # certification at several prefixes. It is not a power study or an output
    # claimed to have been sampled from a physical process.
    records = [[1,1,0]]*100
    prefixes = []
    for n in (10,30,100):
        certificate = sequential.certify(records[:n])
        assert sequential.verify(certificate,records[:n])
        prefixes.append({key:certificate[key] for key in (
            'batches','records','counts','records_sha256','status',
            'predictable_log_score','rejection_threshold_upper',
            'entropy_production_rate_lower','guarantee')})
    assert prefixes[-1]['status'] == 'positive_certificate'

    # Dependence is handled through a separately calibrated contraction bound.
    # Fixed common blocks include physical evolution between readouts; the
    # initial-law error is charged once to whole-record coverage.
    dependent = dependence.certify_counts(
        [1,2],[-1,1],[100000,100000],[38000,53000],[1,2],2,
        F(1,500),F(49,1000),F(1,1000))
    assert dependent['status'] == 'positive_certificate'
    assert dependent['total_failure_upper'] == F(1,20)
    assert dependent['proxy']['whole_record_proxy_upper'] > 4

    # Analog observations can retain their bounded scores. Squaring a noisy
    # reading supplies only an upper bound on A; it is not two independent
    # same-state readings. Quantization error is separately propagated.
    analog_cal = calibration.analog_calibration_summary(10000,0,F(5,16))
    assert analog_cal['observed_second_moment_upper'] >= F(5,16)
    summary = analog.quantized_pair_summary(
        [F(1,3),F(-1,2)],[F(3,4),F(1,5)],denominator=1000)
    assert summary['mean_quantization_radius'] == F(1,2000)
    analog_bound = analog.certify_means(
        [1,2],[1,-1],[10000,10000],[F(69,100),F(59,100)],
        [1,2],[(0,F(1,2))],alpha=F(1,40),
        calibration_failure=F(1,40),mean_errors=[F(1,10**6)]*2)
    assert analog_bound['status'] == 'positive_certificate'
    assert analog_bound['confidence_failure_upper'] == F(1,20)
    retained_variance = analog.conditional_coin_variance(
        [F(3,5)],[F(2,5)],[1],[1000])
    assert retained_variance['direct_contrast_variance'] == F(1,25000)
    assert retained_variance['coin_contrast_variance'] == F(3,12500)

    return {
        'evidence':'Exact mathematical certificates and synthetic records; no empirical data',
        'calibration':{'finite_positive_lag_ambiguity':ambiguity,
                       'calibrated_steep_decay':steep,'synthetic_paired_counts':paired},
        'joint_lag_zero_inference':{
            'synthetic_fixed_quota_certificate':joint,
            'why_lag_zero_sampling_uncertainty_cannot_be_dropped':joint_control},
        'nonlinear_relaxation':{'lower_certificate':dual,'feasible_upper':primal,
                                'certified_bracket_gap_upper':bracket_gap,
                                'upper_is_not_physical_entropy_upper':True},
        'activity_bridge':bridge,
        'dissipation_limits':{
            'same_parity_path_log_likelihood':log_eq,
            'equal_exit_entropy_intervals':[
                equal_equilibrium['entropy_production_interval'],
                equal_driven['entropy_production_interval']],
            'finite_budget_pair':finite_pair,
            'unavoidable_infinite_upper_probability':forced_infinity,
            'resolved_fixed_quota_repair':repaired,
            'positive_reverse_evidence_without_prior_floor':without_floor,
            'zero_reverse_evidence_without_prior_floor':no_reverse,
            'null_entropy_upper_endpoint_means':'positive infinity'},
        'predictable_prefix_replay':prefixes,
        'dependent_block_certificate':dependent,
        'bounded_analog':{'upper_only_calibration':analog_cal,
                          'quantized_summary':summary,'synthetic_mean_certificate':analog_bound,
                          'mathematical_variance_comparison':retained_variance},
        'limitations':[
            'Model certificates do not validate stationarity, memoryless detectors, noninvasiveness, clocks or mixing for an apparatus.',
            'Each sampling design has its own contract; fixed jump quotas, common blocks, independent pairs and predictable batches are not interchangeable.',
            'Entropy is in natural units per declared model time; no heat value follows without reservoir and unit calibration.',
            'Calibration, reset, preparation and readout overheads are additional resources unless explicitly measured.',
            'Historical novelty, external peer review and laboratory validation are not established by this tutorial.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/calibrated_tutorial.json'))
    args = parser.parse_args()
    result = examples()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(compact(result),sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'assertions_passed':True,
                      'exact_relaxation_gap_below':1e-5,
                      'finite_budget_entropy_gap_above':200,
                      'evidence':'mathematical and synthetic'}))


if __name__ == '__main__':
    main()
