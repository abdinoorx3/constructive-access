#!/usr/bin/env python3
"""Reproduce the unified reference and library without network or installation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CALIBRATION_WORKFLOW=json.loads((ROOT/'research/calibration_workflow.json').read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    files=[]
    for folder in ('src','tests','examples','manuscript','research','completion/code','completion/tests','completion/research','robustness/code','robustness/tests','robustness/research','frontier/code','frontier/tests','frontier/research','virtual/code','virtual/tests','virtual/research','virtual/contracts','calibrated/code','calibrated/tests','calibrated/research','calibrated/contracts','calibrated/data/source_audit','attainability/code','attainability/tests','attainability/research','attainability/contracts','geometry/code','geometry/tests','geometry/research','geometry/contracts','geometry/manuscript','detector/theory','detector/countermodels','detector/design','detector/review','detector/literature','detector/manuscript','detector/research'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.tex','.txt','.json','.h5') and '__pycache__' not in p.parts and not ('detector' in p.parts and p.name=='results.json'))
    files.extend(ROOT/name for name in ('README.txt','CLAIMS.json','pyproject.toml','requirements-reproduce.txt','verify.py','completion/run_tests.py','robustness/run_tests.py','robustness/run_independent.py','frontier/run_tests.py','virtual/run_tests.py','calibrated/run_tests.py','attainability/run_tests.py','geometry/run_tests.py','detector/run_tests.py'))
    for folder in CALIBRATION_WORKFLOW['source_folders']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py','.tex','.txt','.json','.md','.csv','.yaml','.yml','.npz','.npy','.h5','.gz','.zip')
                     and '__pycache__' not in p.parts)
    files.append(ROOT/'detector_calibration/run_tests.py')
    generated=set(CALIBRATION_WORKFLOW['generated_outputs'])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))
            if str(p.relative_to(ROOT)) not in generated}

def provenance():
    data=json.loads((ROOT/'src/constructive_access/_vendor/PROVENANCE.json').read_text())
    checks=[{'file':v['vendored_file'],'expected':v['vendored_sha256'],
             'actual':sha(ROOT/v['vendored_file'])} for v in data]
    new=json.loads((ROOT/'src/constructive_access/completion/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/robust/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/frontier/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/virtual/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/calibrated/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/attainability/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/observed_geometry/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/detector/PROVENANCE.json').read_text())
    new+=json.loads((ROOT/'src/constructive_access/detector_calibration/PROVENANCE.json').read_text())
    for row in new:
        checks.extend([{'file':row['original_file'],'expected':row['original_sha256'],'actual':sha(ROOT/row['original_file'])},
                       {'file':row['packaged_file'],'expected':row['packaged_sha256'],'actual':sha(ROOT/row['packaged_file'])}])
    return {'passed':all(x['expected']==x['actual'] for x in checks),'files':checks}

def run(name,argv,cwd=ROOT):
    env=dict(os.environ)
    env.pop('PYTHONOPTIMIZE',None)
    env['PYTHONPATH']=str(ROOT/'src')
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    env['MPLBACKEND']='Agg'
    env['MPLCONFIGDIR']=str(ROOT/'logs/matplotlib_cache')
    env['XDG_CACHE_HOME']=str(ROOT/'logs/xdg_cache')
    started=time.monotonic()
    done=subprocess.run(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
    log=ROOT/'logs'/f'verify_{name}.log'
    log.write_text(done.stdout)
    print(f"{name}: {'PASS' if done.returncode==0 else 'FAIL'}",flush=True)
    return {'name':name,'argv':argv,'exit_code':done.returncode,
            'elapsed_seconds':round(time.monotonic()-started,6),
            'log':str(log.relative_to(ROOT)),'log_sha256':sha(log)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-pdf',action='store_true')
    args=parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run verification without Python optimization flags.')
    for folder in ('logs','results','completion/results','robustness/results','frontier/results','virtual/results','calibrated/results','attainability/results','geometry/results'):(ROOT/folder).mkdir(exist_ok=True)
    for relative in CALIBRATION_WORKFLOW['generated_outputs']:
        (ROOT/relative).parent.mkdir(parents=True,exist_ok=True)
    before=snapshot();vp=provenance()
    if not vp['passed']:parser.error('Vendored-source integrity check failed.')
    python=sys.executable
    jobs=[('tests',[python,'-m','unittest','discover','-s','tests','-v']),
          ('suite_tutorial',[python,'examples/suite_tutorial.py','--output','results/suite_tutorial.json']),
          ('advanced_tutorial',[python,'examples/advanced_tutorial.py','--output','results/advanced_tutorial.json']),
          ('physics_consistency',[python,'research/physics_consistency.py'])]
    for name in ('active','passive','calibrated','calibration_gap_closed','grammar'):
        jobs.append((f'analyze_{name}',[python,'-m','constructive_access','analyze',f'examples/{name}.json','-o',f'results/{name}_report.json']))
    jobs.extend([('completion_tests',[python,'completion/run_tests.py']),
                 ('completion_independent',[python,'completion/tests/test_independent.py']),
                 ('completion_tutorial',[python,'examples/completion_tutorial.py','--output','results/completion_tutorial.json'])])
    for name in ('graph_access','robust_completion','composition','physical_access','quantum_completion','pulse_test'):
        jobs.append((name,[python,f'completion/code/{name}.py']))
    jobs.extend([('robust_tests',[python,'robustness/run_tests.py']),
                 ('robust_independent',[python,'robustness/run_independent.py']),
                 ('robust_tutorial',[python,'examples/robust_tutorial.py','--output','results/robust_tutorial.json'])])
    for name in ('clock_design','detector_boundary','explanation','preparation','robust_physics','statistics_design'):
        argv=[python,f'robustness/code/{name}.py']
        if name in ('detector_boundary','preparation'):argv+=['--output',f'robustness/results/{name}.json']
        jobs.append(('robust_'+name,argv))
    jobs.extend([('frontier_tests',[python,'frontier/run_tests.py']),
                 ('frontier_independent',[python,'frontier/tests/test_independent.py']),
                 ('frontier_tutorial',[python,'examples/frontier_tutorial.py','--output','results/frontier_tutorial.json'])])
    for name in ('hierarchy','spectral','limits','thermodynamic','energy_certificate'):
        jobs.append(('frontier_'+name,[python,f'frontier/code/{name}.py']))
    jobs.extend([('virtual_independent',[python,'virtual/tests/test_independent.py']),
                 ('virtual_tutorial',[python,'examples/virtual_tutorial.py','--output','results/virtual_tutorial.json'])])
    for name in ('contrast','timing','sampling','dimension_lab','thermodynamic_lab'):
        jobs.append(('virtual_'+name,[python,f'virtual/code/{name}.py']))
    jobs.append(('virtual_dimension_retiming',[python,'virtual/code/dimension_lab.py','--retiming-contract','virtual/contracts/dimension_lab_retiming.json']))
    jobs.append(('virtual_dimension_total_budget',[python,'virtual/code/dimension_lab.py','--total-budget-contract','virtual/contracts/dimension_lab_total_budget.json']))
    jobs.append(('virtual_thermodynamic_pilot',[python,'virtual/code/thermodynamic_lab.py','--pilot-only']))
    jobs.append(('virtual_allocation',[python,'virtual/research/allocation_benchmark.py']))
    for name in ('calibration','nonlinear','dissipation_limits','sequential','sequential_benchmark','dependence','analog','analog_benchmark','empirical_loader','empirical_analysis','calibrated_benchmark','joint_calibration'):
        argv=[python,f'calibrated/code/{name}.py']
        if name=='calibrated_benchmark':argv+=['--workers','4','--with-audit']
        jobs.append(('calibrated_'+name,argv))
    jobs.append(('calibrated_dependence_refinement',[python,'calibrated/code/dependence.py','--refinement']))
    jobs.append(('calibrated_tutorial',[python,'examples/calibrated_tutorial.py','--output','results/calibrated_tutorial.json']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        commands=list(pool.map(lambda job:run(*job),jobs))
    if all(c['exit_code']==0 for c in commands):
        commands.append(run('calibrated_tests',[python,'calibrated/run_tests.py']))
        commands.append(run('calibrated_independent',[python,'calibrated/tests/test_independent.py']))
    if all(c['exit_code']==0 for c in commands):
        for name,argv in [
            ('physical_minimum',[python,'attainability/code/physical_minimum.py','--output','attainability/results/physical_minimum.json']),
            ('spectral_replay',[python,'attainability/code/spectral_dynamics.py','--sixth-results','calibrated/results','--output','attainability/results/spectral_dynamics_replay.json']),
            ('joint_analog',[python,'attainability/code/joint_analog.py']),
            ('population_audit',[python,'attainability/code/joint_analog.py','--population-audit']),
            ('design',[python,'attainability/code/design.py']),
            ('retiming',[python,'attainability/code/retiming.py']),
            ('tests',[python,'attainability/run_tests.py']),
            ('independent',[python,'attainability/tests/test_independent.py']),
            ('tutorial',[python,'examples/attainability_tutorial.py','--output','results/attainability_tutorial.json'])]:
            commands.append(run('attainability_'+name,argv))
            if commands[-1]['exit_code']:break
    if all(c['exit_code']==0 for c in commands):
        for name,argv in [
            ('countermodels',[python,'geometry/code/countermodels.py','--output','geometry/results/countermodels.json']),
            ('experiment',[python,'geometry/code/experiment.py','--run']),
            ('combination',[python,'geometry/code/combination.py']),
            ('consistency',[python,'geometry/code/consistency.py','--output','geometry/results/consistency.json']),
            ('tests',[python,'geometry/run_tests.py']),
            ('independent',[python,'geometry/tests/test_independent.py']),
            ('tutorial',[python,'examples/geometry_tutorial.py','--output','results/geometry_tutorial.json'])]:
            commands.append(run('geometry_'+name,argv))
            if commands[-1]['exit_code']:break
    if all(c['exit_code']==0 for c in commands):
        for name,argv in [
            ('countermodels',[python,'detector/countermodels/exact_sampling.py','--output','detector/countermodels/results.json']),
            ('design',[python,'detector/design/counts.py','--output','detector/design/results.json']),
            ('tests',[python,'detector/run_tests.py']),
            ('independent',[python,'detector/review/test_review.py']),
            ('tutorial',[python,'examples/detector_tutorial.py','--output','results/detector_tutorial.json'])]:
            commands.append(run('detector_'+name,argv))
            if commands[-1]['exit_code']:break
    if all(c['exit_code']==0 for c in commands):
        for job in CALIBRATION_WORKFLOW['jobs']:
            commands.append(run(job['name'],[python]+job['argv']))
            if commands[-1]['exit_code']:break
    # Artifact-consistency tests depend on the generated dimension experiment.
    if all(c['exit_code']==0 for c in commands):
        commands.append(run('virtual_tests',[python,'virtual/run_tests.py']))
    if all(c['exit_code']==0 for c in commands):
        commands.append(run('virtual_contrast_benchmark',[python,'virtual/code/contrast_benchmark.py']))
    if all(c['exit_code']==0 for c in commands):
        commands.append(run('virtual_box_obstruction',[python,'virtual/research/box_obstruction.py']))
    if all(c['exit_code']==0 for c in commands):
        for name in ('active','passive','calibrated','calibration_gap_closed','grammar'):
            commands.append(run(f'check_{name}',[python,'-m','constructive_access','verify',f'examples/{name}.json',f'results/{name}_report.json']))
    if all(c['exit_code']==0 for c in commands):
        commands.append(run('figures',[python,'research/make_reference_figure.py']))
        commands.append(run('completion_figures',[python,'research/make_completion_figure.py']))
        commands.append(run('robust_figures',[python,'research/make_robust_figure.py']))
        commands.append(run('frontier_figures',[python,'research/make_frontier_figure.py']))
        commands.append(run('virtual_figures',[python,'research/make_virtual_figure.py']))
        commands.append(run('calibrated_figures',[python,'research/make_calibrated_figure.py','--output-dir','manuscript/figures']))
        commands.append(run('attainability_figures',[python,'research/make_attainability_figure.py','--output-dir','manuscript/figures']))
        commands.append(run('geometry_figures',[python,'research/make_geometry_figure.py','--output-dir','manuscript/figures']))
        commands.append(run('detector_figures',[python,'detector/design/figure.py','--output-prefix','manuscript/figures/detector_access']))
        for job in CALIBRATION_WORKFLOW['figure_jobs']:
            commands.append(run(job['name'],[python]+job['argv']))
            if commands[-1]['exit_code']:break
    if not args.skip_pdf and all(c['exit_code']==0 for c in commands):
        # Three passes support a fresh book's table of contents and all citations.
        for number in range(1,4):
            commands.append(run(f'pdflatex_{number}',['pdflatex','-interaction=nonstopmode','-halt-on-error','reference.tex'],ROOT/'manuscript'))
            if commands[-1]['exit_code']:break
    after=snapshot()
    problems=[]
    if not args.skip_pdf:
        log=ROOT/'manuscript/reference.log'
        problems=[line for line in log.read_text(errors='replace').splitlines()
                  if any(s in line for s in ('Overfull ','undefined references','undefined citations','LaTeX Error','multiply defined'))] if log.exists() else ['Missing LaTeX log']
    required=['suite_tutorial','advanced_tutorial','physics_consistency','completion_tutorial','robust_tutorial','frontier_tutorial','virtual_tutorial','calibrated_tutorial','attainability_tutorial','geometry_tutorial','detector_tutorial']+[f'{name}_report' for name in ('active','passive','calibrated','calibration_gap_closed','grammar')]
    required_paths=[f'results/{name}.json' for name in required]
    required_paths += [f'robustness/results/{name}.json' for name in
        ('clock_design','detector_boundary','explanation','preparation','robust_physics','statistics','independent')]
    required_paths += [f'completion/results/{name}.json' for name in
        ('graph_access','robust_completion','composition','physical_access','quantum_completion','pulse_test','redteam')]
    required_paths += [f'frontier/results/{name}.json' for name in
        ('hierarchy','spectral','limits','thermodynamic','energy_certificate','independent')]
    required_paths += [f'virtual/results/{name}.json' for name in
        ('contrast','timing','sampling','dimension_lab','dimension_lab_retiming','dimension_lab_total_budget','thermodynamic_lab','thermodynamic_lab_pilot','contrast_benchmark','contrast_allocation','box_obstruction','independent')]
    required_paths += [f'virtual/results/{name}.npz' for name in
        ('dimension_lab_shots','thermodynamic_lab_counts','thermodynamic_lab_counts_pilot','thermodynamic_lab_raw','contrast_refinement_counts','contrast_allocation_counts')]
    required_paths += [f'calibrated/results/{name}.json' for name in
        ('calibration','nonlinear','dissipation_limits','sequential','sequential_benchmark','dependence','dependence_refinement','analog','analog_benchmark','data_audit','empirical_analysis','calibrated_benchmark','calibrated_benchmark_audit','joint_calibration','independent')]
    required_paths += [f'calibrated/results/{name}.npz' for name in
        ('sequential_records','calibrated_benchmark_raw','calibrated_benchmark_counts','joint_calibration_raw','joint_calibration_counts')]
    required_paths += [f'manuscript/figures/{name}.{ext}' for name in ('calibrated_inference','calibrated_resources') for ext in ('pdf','png')]
    required_paths += [f'attainability/results/{name}.json' for name in ('physical_minimum','spectral_dynamics_replay','joint_analog','joint_analog_population_audit','design','retiming','independent_review')]
    required_paths += ['attainability/results/joint_analog_raw.npz','attainability/results/retiming_raw.npz']
    required_paths += [f'manuscript/figures/attainability.{ext}' for ext in ('pdf','png')]
    required_paths += [f'geometry/results/{name}.json' for name in ('countermodels','experiment','combination','consistency','independent_review')]
    required_paths += ['geometry/results/experiment_raw.npz']
    required_paths += [f'manuscript/figures/observed_geometry.{ext}' for ext in ('pdf','png')]
    required_paths += [f'detector/{name}/results.json' for name in ('countermodels','design','review')]
    required_paths += [f'manuscript/figures/detector_access.{ext}' for ext in ('pdf','png')]
    required_paths+=CALIBRATION_WORKFLOW['generated_outputs']
    required_paths=list(dict.fromkeys(required_paths))
    missing=[name for name in required_paths if not (ROOT/name).is_file()]
    if not args.skip_pdf and not (ROOT/'manuscript/reference.pdf').is_file():missing.append('manuscript/reference.pdf')
    expected_names=[name for name in CALIBRATION_WORKFLOW['expected_stage_names']
                    if not args.skip_pdf or not name.startswith('pdflatex_')]
    expected_commands=len(expected_names)
    passed=[c['name'] for c in commands]==expected_names and len(commands)==expected_commands and all(c['exit_code']==0 for c in commands) and before==after and provenance()['passed'] and not problems and not missing
    report={'recorded_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if passed else 'failed',
            'scope':'Internal mathematical/software reproduction and exploratory reanalysis of published measurements; no new physical measurement, empirical entropy validation, or external review.',
            'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ('sympy','mpmath','numpy','scipy','matplotlib','h5py')},
            'commands':commands,'expected_stage_names':expected_names,'expected_commands':expected_commands,
            'source_unchanged':before==after,'source_sha256':after,
            'vendored_provenance':vp,'typesetting_problems':problems,'missing_outputs':missing,
            'result_sha256':{name:sha(ROOT/name) for name in required_paths if (ROOT/name).is_file()}}
    (ROOT/'results/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'commands':len(commands),'source_unchanged':before==after,'typesetting_problems':problems}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
