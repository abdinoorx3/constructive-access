"""Installed APIs: classical realization size, approximation, and entropy.

All examples are mathematical models or explicitly synthetic counts. No file
loaded by this tutorial is a laboratory dataset. Exact certificates are checked
before compact outward-rounded summaries are written to the JSON output.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

from constructive_access.frontier import (
    hierarchy, spectral, limits, thermodynamic, energy_certificate,
)


def compact(value):
    """Deterministic outward rational summaries, without enormous fractions."""
    if isinstance(value, F):
        scale = 10**16
        lower = F((value*scale).numerator//(value*scale).denominator, scale)
        upper = F(-((-value*scale).numerator//(-value*scale).denominator), scale)
        result = {'lower':str(lower), 'upper':str(upper)}
        # A small nonzero bound must not disappear from the human-readable
        # summary. This decimal is explanatory; exact comparisons occur above.
        if value and lower == 0:
            result['nonzero_decimal_approximation'] = float(value)
        return result
    if isinstance(value, dict):
        return {key:compact(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [compact(item) for item in value]
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/frontier_tutorial.json'))
    args=parser.parse_args()

    # Finite-noise dimension witnesses need one common repeated channel,
    # preparation and detector. There is no exit-rate assumption here.
    hierarchy_rows=[]
    for grid,ratio,required in [(7,1000,5),(11,10000,6),(21,100000,7)]:
        intervals=hierarchy.qubit_probabilities(grid,F(ratio))
        witness=hierarchy.certify_grid(intervals)
        assert witness['certified'] and witness['minimum_states']==required
        hierarchy_rows.append(dict(grid_size=grid,observation_settings=2*grid-1,
                                   coherence_ratio=ratio,certificate=witness))

    # Erlang-clock certificates retain a strict finite-record separation from
    # every classical CTMC of the declared dimension, even with unknown rates.
    small=spectral.certificate(4,F(1,10),F(2),F(13,10))
    endpoint=spectral.certificate(62,F(1,10),F(2),F(141,100))
    assert small['status']==endpoint['status']=='separated'
    assert small['gap']>0 and endpoint['gap']>0
    finite_plan=spectral.bounded_design(small)
    assert spectral.compatible_excluded(
        small,small['moments'],small['gap']/4,
        truncation_radius=finite_plan['tail_upper'])

    # The larger-model counterpoint is constructive, not a claim that the
    # quantum curve excludes every finite classical model.
    cycle62=limits.cycle_admissibility_certificate(F(1,10),F(2),62)
    cycle63=limits.cycle_admissibility_certificate(F(1,10),F(2),63)
    assert cycle62['inadmissible'] and cycle63['admissible']
    approximation=limits.smooth_certificate(
        F(401,200),F(9,4),F(1,20),preparations=2,reversible=True)
    assert approximation['uniform_error_bound']<=F(1,20)
    balance=limits.detailed_balance_certificate(
        [[F(0),F(1,2),F(1)],[F(1),F(1,2),F(0)]],F(10),F(1,10))
    assert balance['valid']
    transcript=limits.transcript_bound(F(1,10000),100)
    assert transcript<F(1,100)

    # Counts are synthetic independent stationary-pair records. The physical
    # interpretation additionally needs ordinary time reversal, a fixed
    # noninvasive detector, and the stated thermodynamic model assumptions.
    counts=thermodynamic.from_counts(
        5000,2560,5000,3680,F(1,50),F(1,20),F(8,5),F(21,10))
    assert counts['rate_lower']>0
    tradeoff=thermodynamic.two_lag_resource_frontier(
        counts['first_upper'],counts['second_lower'],F(8,5),F(21,10),F(1,1000))
    assert tradeoff['integrated_housekeeping_lower']>0

    # A full reversible-mixture certificate can detect positive decreasing
    # but nonconvex correlations where the two simpler witnesses return zero.
    curvature=energy_certificate.curvature_example()
    certificate=curvature['certificate']
    fields=('lags','step','weights','intervals','mean_interval','scores','depth')
    assert energy_certificate.verify(certificate,**{key:certificate[key] for key in fields})
    assert certificate['entropy_production_rate_lower']>0
    assert curvature['negative_correlation_bound']==0
    assert curvature['monotonic_revival_bound']==0
    frontier=energy_certificate.resource_frontier(certificate,preparation_kl_cap=F(0))
    assert frontier['integrated_housekeeping_lower']>0

    output={
        'evidence':'Exact model certificates and explicitly synthetic counts; no laboratory data',
        'dimension_hierarchy':hierarchy_rows,
        'finite_clock_exclusion':{
            'n4_exclusion_gap':small['gap'],
            'n62_exclusion_gap':endpoint['gap'],
            'n62_gap_verified_positive':True,
            'required_states_at_endpoint':63,
            'n4_bounded_record_design':finite_plan,
            'note':'Finite guarantees can have very large record requirements; these are sufficient designs.'},
        'larger_classical_explanations':{
            'cycle62_admissibility':cycle62,'cycle63_admissibility':cycle63,
            'reversible_finite_window_approximation':approximation,
            'small_exact_detailed_balance_check':balance,
            '100_record_transcript_tv_upper':transcript,
            'note':'Reversible approximation may consume a far-from-equilibrium preparation resource.'},
        'synthetic_entropy_inference':{
            'counts':[2560,3680],'records_per_lag':5000,'confidence_result':counts,
            'preparation_housekeeping_frontier':tradeoff,
            'rate_units':'natural entropy per declared model time'},
        'curvature_certificate':{
            'evidence':curvature['evidence'],
            'negative_correlation_bound':0,'monotonic_revival_bound':0,
            'certificate':certificate,'resource_frontier':frontier},
        'limitations':[
            'Historical priority and external peer review are not established by this tutorial.',
            'Classical state-count conclusions require common homogeneous dynamics and interfaces.',
            'Entropy conclusions require the separate stationary or preparation-budget physical contract.',
            'No heat or energy value follows without physical units and the appropriate thermodynamic assumptions.'],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(compact(output),sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'hierarchy_states':[5,6,7],
                      'finite_clock_required_states':63,
                      'curvature_entropy_bound_positive':True}))


if __name__=='__main__':main()
