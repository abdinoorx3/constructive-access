"""Matched held-out simulation comparisons for fixed moment contrasts.

All records come from explicit ring paths and noisy endpoint readouts. The
legacy score is a baseline, not a claim to represent the best older method.
An optimized continuous-moment score baseline is included separately.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from constructive_access.frontier import energy_certificate as ec
try:
    from . import contrast,thermodynamic_lab as lab
except ImportError:
    import contrast,thermodynamic_lab as lab


def summary(flags):
    yes=int(np.sum(flags));n=len(flags)
    return {'detections':yes,'datasets':n,'rate':yes/n,'monte_carlo_95_interval':lab.clopper_pearson(yes,n)}


def optimized_box(counts,n,radius):
    boxes=[(max(F(0),F(int(s),n)-radius),min(F(1),F(int(s),n)+radius)) for s in counts]
    result=ec.design([1,2,3],F(3,20),[F(1,3)]*3,boxes,grid=33,depth=6)
    return result['certificate'],result['proposal_success']


def radius_for(n):
    return contrast.sqrt_upper(ec.log_interval(F(120))[1]/(2*n))


def evaluate_fixed(counts,n,coefficients,old_menu,optimized=True):
    radius=radius_for(n);old=dict(old_menu)
    old['records_per_setting']=n;old['box_radius']=str(radius)
    compiled=lab.compile_analysis(old,'1/20')
    direct=[];legacy=[];optimal=[];rates=[];box_rates=[];proposal_success=[]
    for row in counts:
        contract=dict(lags=[1,2,3],coefficients=coefficients,records=[n]*3,
                      successes=[int(s) for s in row],durations=[F(3,20),F(3,10),F(9,20)],alpha=F(1,20))
        result=contrast.certify_counts(**contract);direct.append(result['contrast_lower']>0)
        rates.append(float(result['entropy_production_rate_lower']))
        prior=lab.analyze_counts(row,compiled);legacy.append(prior['full_cone'])
        if optimized:
            improved,success=optimized_box(row,n,radius);optimal.append(improved['record_kl_lower']>0)
            box_rates.append(float(improved['entropy_production_rate_lower']));proposal_success.append(success)
    direct=np.array(direct);legacy=np.array(legacy);optimal=np.array(optimal)
    result={'fixed_contrast':summary(direct),'previous_frozen_score_box':summary(legacy),
            'mean_fixed_contrast_entropy_lower':float(np.mean(rates)),
            'box_radius':str(radius),'records_per_setting':n,'records_per_dataset':3*n,
            'nominal_evolution_time_per_dataset':float(n*F(9,10)),
            'confidence_level':.95,'contrast_coefficients':[str(a) for a in coefficients]}
    if optimized:
        result['optimized_continuous_cone_box']=summary(optimal)
        result['optimizer_successes']=int(np.sum(proposal_success))
        result['mean_optimized_box_entropy_lower']=float(np.mean(box_rates))
        wins=int(np.sum(direct&~optimal));losses=int(np.sum(~direct&optimal));discord=wins+losses
        result['paired_comparison']={'direct_only':wins,'box_only':losses,'both':int(np.sum(direct&optimal)),
            'neither':int(np.sum(~direct&~optimal)),
            'two_sided_exact_mcnemar_p':float(binomtest(wins,discord,.5).pvalue) if discord else 1.}
    return result


def main(root):
    first_contract=root/'contracts/contrast_benchmark.json'
    second_contract=root/'contracts/contrast_refinement.json'
    if not first_contract.exists() or not second_contract.exists():raise RuntimeError('Freeze benchmark contracts before execution')
    second=json.loads(second_contract.read_text())
    generator,_=lab.load_contract(root);menu=generator['menus']['curvature']
    data=np.load(root/'results/thermodynamic_lab_counts.npz',allow_pickle=False)
    first=[]
    for case in ['driven','equilibrium','high_readout_noise','independent_clock_jitter','detector_memory']:
        counts=data['curvature__'+case+'__counts']
        # Optimize the stronger baseline on every held-out row; the simultaneous
        # box justifies this score selection without a second confidence charge.
        row=evaluate_fixed(counts,50000,[F(-1),F(2),F(-1)],menu)
        row['case']=case;row['outside_physical_contract']=case=='detector_memory'
        row['actual_mean_evolution_time']=float(data['curvature__'+case+'__exposure'].sum(axis=1).mean())
        first.append(row);print('primary',case,row['fixed_contrast']['detections'],row['optimized_continuous_cone_box']['detections'],flush=True)
    refinements=[];saved={}
    detector=[float(F(z)) for z in menu['detector']]
    for case_index,case in enumerate(second['scenarios']):
        n=second['records_per_setting'];reps=second['datasets'];counts=np.zeros((reps,3),dtype=np.int32)
        exposure=np.zeros((reps,3))
        forward,backward=(20.,1.) if case!='equilibrium' else (10.5,10.5)
        gain=.2 if case=='high_readout_noise' else 1.
        jitter=.3 if case=='independent_clock_jitter' else 0.
        for repetition in range(reps):
            for setting,t in enumerate([.15,.3,.45]):
                rng=np.random.default_rng(np.random.SeedSequence([second['seed'],case_index,repetition,setting]))
                record=lab.ring_records(rng,n,t,forward,backward,detector,gain,jitter)
                counts[repetition,setting]=int(record['same'].sum());exposure[repetition,setting]=float(record['elapsed'].sum())
        saved[case+'__counts']=counts;saved[case+'__exposure']=exposure
        row=evaluate_fixed(counts,n,list(map(F,second['coefficients'])),menu)
        row['case']=case;row['actual_mean_evolution_time']=float(exposure.sum(axis=1).mean())
        row['outside_physical_contract']=False;refinements.append(row)
        print('refinement',case,row['fixed_contrast']['detections'],row['optimized_continuous_cone_box']['detections'],flush=True)
    archive=root/'results/contrast_refinement_counts.npz';lab.deterministic_npz(archive,saved)
    out={'evidence':'independent simulated trajectories; no measured laboratory data',
         'first_contract_sha256':hashlib.sha256(first_contract.read_bytes()).hexdigest(),
         'refinement_contract_sha256':hashlib.sha256(second_contract.read_bytes()).hexdigest(),
         'primary_existing_holdout':first,'new_refinement_holdout':refinements,
         'refinement_counts_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
         'confidence_note':'Each method has95% model-based confidence per dataset. MonteCarlo intervals describe repeated simulation performance; familywise simultaneous performance across scenarios is not claimed.',
         'baseline_note':'Optimized box scores have exact continuous-domain checking. An optimizer failure affects power only; a zero certificate does not prove that every score fails.'}
    (root/'results/contrast_benchmark.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':main(Path(__file__).resolve().parents[1])
