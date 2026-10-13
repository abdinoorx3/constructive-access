"""Offline clean-tree and installed-wheel reproduction for the unified release."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    work=Path(tempfile.mkdtemp(prefix='constructive-access-clean-'))
    copy=work/'reference'
    shutil.copytree(ROOT,copy,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info','build','dist','logs','results','*.aux','*.log','*.out','*.toc','*.pdf','*.png'))
    for relative in ('detector/countermodels/results.json','detector/design/results.json','detector/review/results.json'):
        (copy/relative).unlink(missing_ok=True)
    calibration_workflow=json.loads((ROOT/'research/calibration_workflow.json').read_text())
    for relative in calibration_workflow['generated_outputs']:
        (copy/relative).unlink(missing_ok=True)
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None);env.pop('PYTHONOPTIMIZE',None)
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    env['PIP_DISABLE_PIP_VERSION_CHECK']='1';env['PIP_NO_INDEX']='1'
    commands=[]
    def run(name,argv,cwd=work):
        started=time.monotonic()
        done=subprocess.run(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
        log=logdir/f'clean_{name}.log';log.write_text(done.stdout)
        row={'name':name,'argv':list(map(str,argv)),'exit_code':done.returncode,'elapsed_seconds':round(time.monotonic()-started,6),'log':str(log.relative_to(ROOT)),'log_sha256':digest(log)}
        commands.append(row)
        print(f"{name}: {'PASS' if done.returncode==0 else 'FAIL'}",flush=True)
        if done.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
        return done
    run('reproduce',[sys.executable,'verify.py'],copy)
    clean_report=json.loads((copy/'results/verification.json').read_text())
    if clean_report['status']!='passed':raise AssertionError('Clean full reproduction failed')
    (ROOT/'results/clean_reproduction_verification.json').write_text(json.dumps(clean_report,indent=2)+'\n')
    run('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(work/'wheel'),str(copy)])
    wheels=list((work/'wheel').glob('*.whl'))
    if len(wheels)!=1:raise AssertionError('Expected one built distribution')
    venv=work/'environment'
    run('venv',[sys.executable,'-m','venv','--system-site-packages',str(venv)])
    python=venv/'bin/python'
    run('install',[str(python),'-m','pip','install','--no-index','--no-deps',str(wheels[0])])
    example_dir=work/'examples';shutil.copytree(copy/'examples',example_dir)
    run('installed_import',[str(python),'-I','-c',
        "import constructive_access, pathlib, sys; p=pathlib.Path(constructive_access.__file__).resolve(); assert str(p).startswith(str(pathlib.Path(sys.prefix).resolve())), p; print(constructive_access.__version__); print(p)"])
    run('installed_analyze',[str(venv/'bin/constructive-access'),'analyze',str(example_dir/'active.json'),'-o',str(work/'active_report.json')])
    run('installed_verify',[str(venv/'bin/constructive-access'),'verify',str(example_dir/'active.json'),str(work/'active_report.json')])
    run('installed_suite',[str(python),str(example_dir/'suite_tutorial.py'),'--output',str(work/'suite_tutorial.json')])
    run('installed_advanced',[str(python),str(example_dir/'advanced_tutorial.py'),'--output',str(work/'advanced_tutorial.json')])
    run('installed_completion',[str(python),str(example_dir/'completion_tutorial.py'),'--output',str(work/'completion_tutorial.json')])
    run('installed_robust',[str(python),str(example_dir/'robust_tutorial.py'),'--output',str(work/'robust_tutorial.json')])
    run('installed_frontier',[str(python),str(example_dir/'frontier_tutorial.py'),'--output',str(work/'frontier_tutorial.json')])
    run('installed_virtual',[str(python),str(example_dir/'virtual_tutorial.py'),'--output',str(work/'virtual_tutorial.json')])
    run('installed_calibrated',[str(python),str(example_dir/'calibrated_tutorial.py'),'--output',str(work/'calibrated_tutorial.json')])
    for module,outname in [('empirical_loader','data_audit'),('empirical_analysis','empirical_analysis')]:
        run('installed_'+module,[str(python),'-m','constructive_access.calibrated.'+module,'--data-dir',str(copy/'calibrated/data/source_audit'),'--output',str(work/(outname+'.json'))])
    run('installed_attainability',[str(python),str(example_dir/'attainability_tutorial.py'),'--output',str(work/'attainability_tutorial.json')])
    run('installed_geometry',[str(python),str(example_dir/'geometry_tutorial.py'),'--output',str(work/'geometry_tutorial.json')])
    run('installed_detector',[str(python),str(example_dir/'detector_tutorial.py'),'--output',str(work/'detector_tutorial.json')])
    run('installed_detector_calibration',[str(python),str(example_dir/'calibration_tutorial.py'),'--output',str(work/'calibration_tutorial.json')])
    installed_calibration_outputs={}
    for job in calibration_workflow['installed_module_jobs']:
        target=work/(job['name']+'.json')
        argv=[str(target) if arg==job['output'] else arg for arg in job['argv']]
        run(job['name'],[str(python),'-m',job['module']]+argv)
        installed_calibration_outputs[job['name']]=(job['output'],target)
    comparisons={}
    for name in ('active','passive','calibrated','calibration_gap_closed','grammar'):
        rel=f'results/{name}_report.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('suite_tutorial','advanced_tutorial','physics_consistency','completion_tutorial','robust_tutorial','frontier_tutorial','virtual_tutorial','calibrated_tutorial','attainability_tutorial','geometry_tutorial','detector_tutorial'):
        rel=f'results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('active_report','suite_tutorial','advanced_tutorial','completion_tutorial','robust_tutorial','frontier_tutorial','virtual_tutorial','calibrated_tutorial','attainability_tutorial','geometry_tutorial','detector_tutorial'):
        comparisons['installed:'+name]=(ROOT/'results'/f'{name}.json').read_bytes()==(work/f'{name}.json').read_bytes()
    for name in ('graph_access','robust_completion','composition','physical_access','quantum_completion','pulse_test','redteam'):
        rel=f'completion/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('clock_design','detector_boundary','explanation','preparation','robust_physics','statistics','independent'):
        rel=f'robustness/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('hierarchy','spectral','limits','thermodynamic','energy_certificate','independent'):
        rel=f'frontier/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('contrast','timing','sampling','dimension_lab','dimension_lab_retiming','dimension_lab_total_budget','thermodynamic_lab','thermodynamic_lab_pilot','contrast_benchmark','contrast_allocation','box_obstruction','independent'):
        rel=f'virtual/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('dimension_lab_shots','thermodynamic_lab_counts','thermodynamic_lab_counts_pilot','thermodynamic_lab_raw','contrast_refinement_counts','contrast_allocation_counts'):
        rel=f'virtual/results/{name}.npz'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('calibration','nonlinear','dissipation_limits','sequential','sequential_benchmark','dependence','dependence_refinement','analog','analog_benchmark','data_audit','empirical_analysis','calibrated_benchmark','calibrated_benchmark_audit','joint_calibration','independent'):
        rel=f'calibrated/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('sequential_records','calibrated_benchmark_raw','calibrated_benchmark_counts','joint_calibration_raw','joint_calibration_counts'):
        rel=f'calibrated/results/{name}.npz'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('harmwell_n1.h5','harmwell_n1_groundtruth.h5','harmwell_n2.h5'):
        rel=f'calibrated/data/source_audit/{name}'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('calibrated_inference','calibrated_resources'):
        for ext in ('pdf','png'):
            rel=f'manuscript/figures/{name}.{ext}'
            comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('data_audit','empirical_analysis'):
        comparisons['installed:'+name]=(ROOT/'calibrated/results'/f'{name}.json').read_bytes()==(work/f'{name}.json').read_bytes()
    for name in ('physical_minimum','spectral_dynamics_replay','joint_analog','joint_analog_population_audit','design','retiming','independent_review'):
        rel=f'attainability/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for rel in ('attainability/results/joint_analog_raw.npz','attainability/results/retiming_raw.npz','manuscript/figures/attainability.pdf','manuscript/figures/attainability.png'):
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('countermodels','experiment','combination','consistency','independent_review'):
        rel=f'geometry/results/{name}.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for rel in ('geometry/results/experiment_raw.npz','manuscript/figures/observed_geometry.pdf','manuscript/figures/observed_geometry.png'):
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for name in ('countermodels','design','review'):
        rel=f'detector/{name}/results.json'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for ext in ('pdf','png'):
        rel=f'manuscript/figures/detector_access.{ext}'
        comparisons[rel]=(ROOT/rel).read_bytes()==(copy/rel).read_bytes()
    for relative in calibration_workflow['generated_outputs']:
        comparisons[relative]=(ROOT/relative).read_bytes()==(copy/relative).read_bytes()
    comparisons['installed:calibration_tutorial']=(ROOT/'results/calibration_tutorial.json').read_bytes()==(work/'calibration_tutorial.json').read_bytes()
    for name,(relative,target) in installed_calibration_outputs.items():
        comparisons[name]=(ROOT/relative).read_bytes()==target.read_bytes()
    import fitz
    def text(path):
        with fitz.open(path) as pdf:return '\n'.join(page.get_text() for page in pdf),len(pdf)
    original_text,pages=text(ROOT/'manuscript/reference.pdf')
    clean_text,clean_pages=text(copy/'manuscript/reference.pdf')
    comparisons['reference_pdf_extracted_text']=original_text==clean_text
    comparisons['reference_page_count']=pages==clean_pages
    if not all(comparisons.values()):raise AssertionError(f'Reproduction mismatch: {comparisons}')
    dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    finalwheel=dist/wheels[0].name;shutil.copy2(wheels[0],finalwheel)
    report={'recorded_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','workspace':str(work),
            'network_used':False,'environment':'Clean copied source tree, then local wheel in a separate venv sharing only already-installed third-party dependencies. No PYTHONPATH for installed execution.',
            'prior_research_directories_required':False,'commands':commands,'comparisons':comparisons,
            'pdf_pages':pages,'wheel':str(finalwheel.relative_to(ROOT)),'wheel_sha256':digest(finalwheel),
            'independence_limit':'This validates portability and deterministic reproduction, not independent external scientific review.'}
    (ROOT/'results/portability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','comparisons':len(comparisons),'pdf_pages':pages,'wheel':str(finalwheel)}))

if __name__=='__main__':main()
