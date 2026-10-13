"""Covered maximum of two fixed dissipation certificates on shared triples.

This is an application of the union bound, not a new statistical theorem.
Neither independence of the two scores nor an additional acquisition is needed.
Rules and failure allocation must be fixed before prospective test data.  The
CLI reanalysis of an existing bank is explicitly retrospective.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import sys

try:
    from . import finite_resolution as fr
except ImportError:
    import finite_resolution as fr
try:
    from constructive_access.attainability.spectral_dynamics import from_gap
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'physical_attainability'/'release'/'src'))
    from constructive_access.attainability.spectral_dynamics import from_gap


def endpoint_certificate(records, total, square_total, scale, parameter,
                         lag, uniform_score_error=0, alpha=F(1, 40)):
    """Exact same-triple endpoint defect with a predeclared s in [0,1].

    Integer sums summarize quantized scores -(sY0-Ypast)(sY0-Yfuture).
    Ideal readouts lie in [-1,1].  The independent stationary triple contract,
    ordinary physical reversal, calibrated clock and uniform score perturbation
    are physical premises; these aggregates do not test them.
    """
    if type(records) is not int or records < 2 or type(scale) is not int or scale <= 0:
        raise ValueError('At least two records and a positive integer score scale are required')
    total, squares, s, h, error, alpha = map(fr.exact, (
        total, square_total, parameter, lag, uniform_score_error, alpha))
    if not 0 <= s <= 1 or h <= 0 or error < 0 or not 0 < alpha < 1:
        raise ValueError('Use s in [0,1], a positive lag and valid error allocation')
    mean = total/(scale*records)
    second = squares/(scale*scale*records)
    low, high = -(1+s)**2-error, 1+error
    if not low <= mean <= high or not mean*mean <= second <= (low+high)*mean-low*high:
        raise ValueError('Inconsistent exact endpoint score moments')
    variance = F(records, records-1)*(second-mean*mean)
    logarithm = fr.log_interval(2/alpha)[1]
    radius = (fr.sqrt_upper(2*variance*logarithm/records)
              + 7*(high-low)*logarithm/(3*(records-1)))
    signed = mean-radius-error
    defect = max(F(0), signed)
    bridge = from_gap(defect/2, lag_unit=h, range_bound=2, terms=24)
    return dict(score_mean=mean, empirical_variance=variance,
                score_bounds=(low, high), parameter=s, lag=h,
                sampling_radius_upper=radius, uniform_score_error=error,
                signed_defect_lower=signed, defect_lower=defect,
                entropy_production_rate_lower=bridge['entropy_production_rate_lower'],
                confidence_failure_upper=alpha, bridge=bridge)


def certify_shared_records(records, geometry_total, geometry_square_total,
                           geometry_scale, endpoint_total, endpoint_square_total,
                           endpoint_scale, protocol, geometry_score_error,
                           endpoint_score_error, endpoint_parameter,
                           alpha_total=F(1, 20), geometry_failure_share=F(1, 2),
                           external_failure=0):
    """Bonferroni-covered max using the SAME fixed quota of physical triples.

    Protocol and allocation are predeclared or conditionally fixed by an
    independent pilot.  Both scores are computed on all the same triples;
    geometry-excluded centers contribute zero rather than removing records.
    external_failure is a joint failure budget for the shared pre-test physical
    calibration of both methods, counted once.  Separate calibration budgets
    may conservatively be summed by the caller before passing this argument.
    """
    p = fr.validate_protocol(protocol)
    if p['readout_bounds'] != (F(-1), F(1)):
        raise ValueError('Shared endpoint/geometry rules require the common normalized [-1,1] readout contract')
    if type(records) is not int or records < 2:
        raise ValueError('At least two independent stationary triples are required')
    if type(geometry_scale) is not int or geometry_scale <= 0:
        raise ValueError('A positive integer geometry score scale is required')
    alpha, share, eta = map(fr.exact, (alpha_total, geometry_failure_share, external_failure))
    if not 0 < alpha < 1 or not 0 < share < 1 or not 0 <= eta < 1 or alpha+eta >= 1:
        raise ValueError('Positive component failure allocations and a valid joint budget required')
    a_geometry, a_endpoint = alpha*share, alpha*(1-share)
    geometry = fr.certify_totals(
        records=records,
        score_total=fr.exact(geometry_total)/geometry_scale,
        score_square_total=fr.exact(geometry_square_total)/(geometry_scale*geometry_scale),
        protocol=p, alpha=a_geometry,
        concentration='empirical_bernstein', quantization_radius=geometry_score_error)
    endpoint = endpoint_certificate(
        records, endpoint_total, endpoint_square_total, endpoint_scale,
        endpoint_parameter, p['h'], uniform_score_error=endpoint_score_error,
        alpha=a_endpoint)
    rg, re = geometry['entropy_production_rate_lower'], endpoint['entropy_production_rate_lower']
    rate = max(rg, re)
    winner = ('tie' if rg == re else 'geometry' if rg > re else 'endpoint')
    coordinate_readings = p['coordinate_readouts_per_record']*records
    return dict(status='positive_certificate' if rate else 'no_positive_certificate',
                entropy_production_rate_lower=rate, selected_component=winner,
                confidence_failure_upper=alpha+eta,
                statistical_failure_upper=alpha, joint_calibration_failure_upper=eta,
                geometry=geometry, endpoint=endpoint,
                failure_allocation=dict(geometry=a_geometry, endpoint=a_endpoint),
                shared_records=records,
                resources=dict(detector_readings=3*records+coordinate_readings,
                    triple_detector_readings=3*records,
                    coordinate_detector_readings=coordinate_readings,
                    independent_stationary_preparations=records,
                    modeled_evolution_time=2*p['h']*records),
                resource_contract='The two scores use the same triples. Acquisitions, endpoint readouts and evolution time are counted once; coordinate calibration, preparation, reset, exposure, energy and pilot overhead require separate accounting.',
                coverage_contract='Fixed component rules and fixed failure allocation; independent stationary triples. Component scores may be arbitrarily dependent. Both physical premises hold on one shared pre-test calibration event, or a separately proved joint event. This certificate does not cover data-driven optimization over allocations.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _encode(value):
    if isinstance(value, F):
        return str(value)
    raise TypeError(type(value).__name__)


def replay(experiment_path, contract_path, output_path):
    """Retrospective equal-allocation reanalysis; never changes frozen inputs."""
    bank = json.loads(Path(experiment_path).read_text())
    contract = json.loads(Path(contract_path).read_text())
    if bank['contract_sha256'] != sha(contract_path):
        raise ValueError('The bank does not bind the supplied frozen contract')
    expected_count = contract['replicates']*len(contract['scenarios'])
    if len(bank['datasets']) != expected_count or any(
            r['records'] != contract['independent_stationary_triples'] for r in bank['datasets']):
        raise ValueError('Frozen bank count or acquisition quota differs from its contract')
    if len({(r['scenario'],r['replicate']) for r in bank['datasets']}) != expected_count:
        raise ValueError('Frozen bank has duplicate dataset identifiers')
    source_files = {'finite_resolution.py': Path(fr.__file__),
                    'spectral_dynamics.py': Path(importlib.import_module(from_gap.__module__).__file__)}
    for name, path in source_files.items():
        if sha(path) != contract['source_hashes'][name]:
            raise ValueError('A frozen inferential dependency has changed: '+name)
    frozen = contract['protocol']
    p = fr.make_protocol(**{key: frozen[key] for key in (
        'h', 'q_abs_max', 'envelope_floor', 'generator_bounds', 'readout_bounds',
        'global_gamma_upper', 'selection', 'detector_contract', 'coordinate_readouts_per_record')})
    rows = []; groups = {}
    for row in bank['datasets']:
        result = certify_shared_records(
            records=row['records'], geometry_total=row['weighted_total'],
            geometry_square_total=row['weighted_square_total'],
            geometry_scale=contract['score_denominator'],
            endpoint_total=row['endpoint_total'], endpoint_square_total=row['endpoint_square_total'],
            endpoint_scale=contract['endpoint_denominator'], protocol=p,
            geometry_score_error=contract['score_uniform_error'],
            endpoint_score_error=contract['endpoint_uniform_error'],
            endpoint_parameter=contract['endpoint_parameter'])
        out = dict(scenario=row['scenario'], replicate=row['replicate'], records=row['records'],
                   geometry_rate_lower=float(result['geometry']['entropy_production_rate_lower']),
                   endpoint_rate_lower=float(result['endpoint']['entropy_production_rate_lower']),
                   combined_rate_lower=float(result['entropy_production_rate_lower']),
                   selected_component=result['selected_component'],
                   original_geometry_rate_lower=row['weighted_rate_lower'],
                   original_endpoint_rate_lower=row['endpoint_rate_lower'],
                   geometry_radius_upper=float(result['geometry']['sampling_radius_upper']),
                   endpoint_radius_upper=float(result['endpoint']['sampling_radius_upper']))
        rows.append(out); groups.setdefault(row['scenario'], []).append(out)
    summaries = {}
    for name, values in sorted(groups.items()):
        truth = contract['scenarios'][name]**2/contract['gamma']
        methods = {}
        for method in ('geometry', 'endpoint', 'combined', 'original_geometry', 'original_endpoint'):
            rates = [r[method+'_rate_lower'] for r in values]
            methods[method] = dict(positive_certificates=sum(x > 0 for x in rates),
                                  mean_rate_lower=math.fsum(rates)/len(rates),
                                  minimum_rate_lower=min(rates), maximum_rate_lower=max(rates),
                                  over_truth=sum(x > truth for x in rates))
        summaries[name] = dict(records=len(values), true_entropy_rate=truth, methods=methods,
                              winners={who:sum(r['selected_component']==who for r in values)
                                       for who in ('geometry', 'endpoint', 'tie')})
    output = dict(version=1,
        analysis_status='Retrospective reanalysis of a frozen bank, with no new acquisition. The equal allocation was introduced after inspection of the primary results. Fixed-rule prospective coverage follows from the union bound; this is not a separately frozen confirmatory experiment.',
        statistical_failure_upper=F(1,20), allocation=dict(geometry=F(1,40), endpoint=F(1,40)),
        inputs=dict(experiment_sha256=sha(experiment_path),contract_sha256=sha(contract_path),
                    raw_sha256=bank['raw_sha256']),
        source_hashes={**{name:sha(path) for name,path in source_files.items()},
                       'combination.py':sha(__file__)},
        datasets=rows, scenario_summary=summaries,
        resources_per_dataset=result['resources'],
        resource_contract=result['resource_contract'], coverage_contract=result['coverage_contract'],
        original_methods='The frozen individual-alpha=.05 benchmark remains unchanged. The newly covered maximum recomputes both component bounds at alpha=.025; the original bounds cannot simply be maximized while retaining a .05 guarantee.')
    Path(output_path).write_text(json.dumps(output, indent=2, sort_keys=True, default=_encode)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--experiment', type=Path, default=base/'results'/'experiment.json')
    parser.add_argument('--contract', type=Path, default=base/'contracts'/'experiment.json')
    parser.add_argument('--output', type=Path, default=base/'results'/'combination.json')
    args = parser.parse_args()
    report = replay(args.experiment, args.contract, args.output)
    print(json.dumps(report['scenario_summary'], indent=2, sort_keys=True))
