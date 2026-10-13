"""Run the final independent calibration audit and write deterministic reports.

Defaults relocate with the unchanged detector_calibration directory. This is
internal review, not external peer review or physical validation. The explicit
manifest binds reviewed scientific sources, frozen records and derived reports;
mutable progress metadata and generated figures are deliberately excluded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCES=(
    'AGENTS.md',
    'code/certification.py','code/API.txt','code/PROVENANCE.json','tests/test_certification.py',
    'theory/calibration_theory.txt','theory/calibration_theory.tex',
    'countermodels/RARE_BUMP_PROOF.txt','countermodels/rare_bump.py',
    'countermodels/test_rare_bump.py','countermodels/results.json',
    'experiment/exposure_moments.py','experiment/test_exposure_moments.py',
    'experiment/EXPERIMENT.txt','experiment/benchmark.py','experiment/test_benchmark.py',
    'experiment/contract.json','experiment/benchmark.json','experiment/benchmark_counts.json.gz',
    'experiment/benchmark_run.json','experiment/calibration_counts.json',
    'experiment/benchmark_laws/driven.json','experiment/benchmark_laws/equilibrium.json',
    'experiment/benchmark_laws/weak_driven.json','experiment/benchmark_laws/changed_calibration.json',
    'experiment/projected_pilot.py','experiment/projected_pilot_initial.json','experiment/projected_pilot_refined.json',
    'experiment/population_pilot.py','experiment/population_pilot.json',
    'experiment/population_lag_sweep.py','experiment/population_lag_sweep.json',
    'literature/audit.txt','literature/sources.json',
    'manuscript/calibration_entry.tex','manuscript/calibration_limits.tex',
    'manuscript/calibration_experiment.tex','manuscript/calibration_literature.tex',
    'manuscript/calibration_bibitems.tex',
    'research/calibration_tutorial.py','research/make_calibration_figure.py',
    'research/release_entry.txt','research/new_claims.json',
    'review/ASSESSMENT.txt','review/test_independent.py','review/benchmark_audit.py',
    'review/benchmark_audit.json','review/run_review.py',
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'review/results.json')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'review'),pattern='test_independent.py')
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    if not run.wasSuccessful():return 1
    import benchmark_audit
    benchmark=benchmark_audit.audit()
    (ROOT/'review/benchmark_audit.json').write_text(json.dumps(benchmark,indent=2,sort_keys=True)+'\n')
    result={
      'phase':'detector_calibration','status':'passed',
      'external_peer_review':False,'physical_experiment':False,
      'independent_checks':{'run':run.testsRun,'failures':len(run.failures),
          'errors':len(run.errors),'skipped':len(run.skipped)},
      'scope':'Scoped proof, executable arithmetic, final scientific presentation and complete frozen-bank internal audit. Historical priority, external acceptance and physical validation are not established.',
      'verified_components':[
          'Replica score expectation and fixed-step physical inequality under the pointwise Gamma identity',
          'Signed nonreversible semigroup-word bound and uniform-box moment coefficients',
          'Joint nuisance and entropy ratio on a simultaneous confidence event',
          'Fixed nominal-gain covariance preservation, residual correction and covered prefix selection',
          'Rational finite-stage API, shared-event endpoint fallback and common-gain references',
          'Finite Fourier exposure-law enclosure with independent signed-complex-basis fixture',
          'Exact hidden stochastic-drive compatible minimum and ordinary-parity scope',
          'Compact rare-region obstruction including bounded-diffusion generator variants',
          'Complete sufficient-count replay, population confidence-event checks and selected-seed regeneration',
          'Final theorem, benchmark, entry, claims, literature, tutorial and figure-source audit'],
      'benchmark_audit':benchmark,
      'limitations':[
          'Pointwise geometry, generator caps, detector independence, calibration response and ordinary reversal remain physical premises',
          'Fixed absolute arithmetic floors are finite-valid; the sharp consistency construction requires increasing precision and physical access',
          'The frozen bank is synthetic; its 80,199,998 readings per dataset do not measure preparation or detector energy',
          'Higher prefix did not win and geometry did not dominate the broader endpoint method in sensitivity',
          'The deterministic shared-event maximum has the stricter geometry scope and is a post-benchmark combination',
          'Independent internal review is not external peer review, proof of priority or empirical discovery'],
      'source_sha256':{relative:hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() for relative in SOURCES},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'tests':run.testsRun,'replayed_datasets':benchmark['datasets'],
                      'source_bindings':len(SOURCES),'output':str(args.output)}))
    return 0

if __name__=='__main__':sys.exit(main())
