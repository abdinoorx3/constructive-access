"""Independent virtual laboratory for the finite-record dimension witness.

Physical records are simulated with Bloch-generator matrix exponentials, not
with the certificate's sinusoidal probability helper. This is synthetic data.
"""
from __future__ import annotations
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
from math import isqrt,ceil,sqrt,log
import argparse
from numbers import Integral
import hashlib
import json
import numpy as np
from scipy.linalg import expm
from scipy.stats import binom,beta
from scipy.optimize import minimize_scalar
from constructive_access.frontier import hierarchy


def step_channels(parameters,grid_size=7):
    p=parameters;delta=2*np.pi/(grid_size*p['omega'])
    matrices=[];durations=[]
    for amp_sign,time_sign in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        wx=p['omega']*(1+p['amplitude_offset']+amp_sign*p['amplitude_jitter'])
        wz=p['detuning'];duration=delta*p.get('clock_scale',1.)*(1+time_sign*p['timing_jitter'])
        if duration<=0 or not 0<=p['preparation_error']<=1:
            raise ValueError('Invalid physical preparation or duration.')
        skew=np.array([[0.,-wz,0.],[wz,0.,-wx],[0.,wx,0.]])
        matrices.append(expm((skew-p['depolarization']*np.eye(3))*duration))
        durations.append(duration)
    return np.array(matrices),np.array(durations)


def probability_engine(parameters,grid_size=7):
    matrices,durations=step_channels(parameters,grid_size)
    channel=np.mean(matrices,axis=0)
    state=np.array([0.,0.,1-2*parameters['preparation_error']])
    probabilities=[]
    for k in range(1,2*grid_size):
        state=channel@state
        ideal=(1-state[2])/2
        probabilities.append(parameters['false_positive']+
            (1-parameters['false_positive']-parameters['false_negative'])*ideal)
    return np.array(probabilities)


def lindblad_probabilities(parameters,grid_size=7):
    """Independent four-dimensional density-matrix calculation, column vec."""
    p=parameters;I=np.eye(2);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.])
    delta=2*np.pi/(grid_size*p['omega']);channels=[]
    reset=np.outer((I/2).reshape(4,order='F'),I.reshape(4,order='F'))
    for a,t in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        H=(p['omega']*(1+p['amplitude_offset']+a*p['amplitude_jitter'])*X+p['detuning']*Z)/2
        L=-1j*(np.kron(I,H)-np.kron(H.T,I))+p['depolarization']*(reset-np.eye(4))
        channels.append(expm(L*delta*p.get('clock_scale',1.)*(1+t*p['timing_jitter'])))
    channel=np.mean(channels,axis=0)
    rho=np.diag([1-p['preparation_error'],p['preparation_error']]).astype(complex).reshape(4,order='F')
    probabilities=[]
    for k in range(1,2*grid_size):
        rho=channel@rho
        population=float(rho.reshape((2,2),order='F')[1,1].real)
        probabilities.append(p['false_positive']+(1-p['false_positive']-p['false_negative'])*population)
    return np.array(probabilities)


def quasi_static_probabilities(parameters,grid_size=7):
    """Outside effective-channel closure: one noise choice held for the run."""
    matrices,_=step_channels(parameters,grid_size)
    state=np.array([0.,0.,1-2*parameters['preparation_error']])
    probabilities=[]
    for k in range(1,2*grid_size):
        z=np.mean([(np.linalg.matrix_power(M,k)@state)[2] for M in matrices])
        probabilities.append(parameters['false_positive']+
            (1-parameters['false_positive']-parameters['false_negative'])*(1-z)/2)
    return np.array(probabilities)


def classical_probabilities(P,initial,detector,grid_size=7):
    state=np.array(initial,dtype=float);values=[]
    for k in range(1,2*grid_size):
        state=state@P;values.append(state@detector)
    return np.array(values)


@lru_cache(maxsize=8192)
def exact_tail(n,k,p,tail):
    return hierarchy.binomial_tail_upper(n,k,p,tail)


def critical_count(n,p,direction,alpha):
    if direction=='low':
        k=int(binom.ppf(float(alpha),n,float(p)))
        while exact_tail(n,k,p,'lower')>alpha:k-=1
        return k,exact_tail(n,k,p,'lower')
    k=int(binom.isf(float(alpha),n,float(p)))+1
    while exact_tail(n,k,p,'upper')>alpha:k+=1
    return k,exact_tail(n,k,p,'upper')



def _validate_plan_shape(records,critical,grid_size):
    if type(grid_size) is not int or grid_size<2 or len(records)!=2*grid_size-1 or len(critical)!=len(records):
        raise ValueError('Exactly2N-1 matching settings and critical counts are required.')
    if any(isinstance(n,bool) or not isinstance(n,Integral) or n<=0 for n in records):
        raise ValueError('Record quotas must be positive integers.')
    if any(isinstance(k,bool) or not isinstance(k,Integral) or not 0<=k<=n for k,n in zip(critical,records)):
        raise ValueError('Critical counts must be integers within their quotas.')


def certify_plan(plan,records,grid_size=7,alpha=F(1,20)):
    _validate_plan_shape(records,plan['critical_counts'],grid_size)
    epsilon,beta0,tau=map(F,(plan['epsilon'],plan['beta'],plan['tau']))
    witness=hierarchy.certify_matrix_bounds(epsilon,beta0,grid_size,tau)
    errors=[]
    for j,(n,k) in enumerate(zip(records,plan['critical_counts'])):
        errors.append(exact_tail(n,k,epsilon if j==grid_size-1 else beta0,
                                 'lower' if j==grid_size-1 else 'upper'))
    return {'valid':witness['certified'] and max(errors)<=alpha,
            'minimum_states':witness['minimum_states'],
            'alpha_exactly_verified':max(errors)<=alpha,
            'alpha_bound':float(max(errors)),
            'witness_margin':str(witness['margin'])}


def pass_probability(probabilities,records,critical,grid_size=7):
    _validate_plan_shape(records,critical,grid_size)
    if len(probabilities)!=len(records) or any(not np.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('One probability in[0,1] is required per setting.')
    values=[]
    for j,(p,n,k) in enumerate(zip(probabilities,records,critical)):
        values.append(binom.logcdf(k,n,p) if j==grid_size-1 else binom.logsf(k-1,n,p))
    return float(np.exp(sum(values))),float(sum(values))


def select_pilot_plan(contract,pilot_counts):
    N=contract['grid_size'];records=contract['records'];alpha=F(contract['alpha'])
    estimates=np.array(pilot_counts)/contract['pilot']['records_per_setting']
    candidates=[]
    denominator=contract['pilot']['tau_denominator'];bden=contract['pilot']['beta_rounding_denominator']
    for value in contract['pilot']['epsilon_candidates']:
        epsilon=F(value)
        tau=F(isqrt(epsilon.numerator*denominator**2//epsilon.denominator),denominator)
        bound=tau+(1-tau)*epsilon/tau
        beta0=F(ceil(bound*bden)+1,bden)
        critical=[]
        for j,n in enumerate(records):
            k,_=critical_count(n,epsilon if j==N-1 else beta0,'low' if j==N-1 else 'high',alpha)
            critical.append(k)
        score,_=pass_probability(estimates,records,critical,N)
        candidates.append({'epsilon':str(epsilon),'beta':str(beta0),'tau':str(tau),
                           'critical_counts':critical,'pilot_estimated_joint_power':score})
    winner=max(candidates,key=lambda p:p['pilot_estimated_joint_power'])
    assert certify_plan(winner,records,N,alpha)['valid']
    return winner,candidates


def classify(counts,critical,grid_size=7):
    counts=np.asarray(counts);critical=np.asarray(critical)
    if type(grid_size) is not int or grid_size<2 or counts.ndim not in(1,2) or counts.shape[-1]!=2*grid_size-1 or critical.shape!=(2*grid_size-1,):
        raise ValueError('Complete matching2N-1 settings are required.')
    if not np.issubdtype(counts.dtype,np.integer) or not np.issubdtype(critical.dtype,np.integer) or np.any(counts<0) or np.any(critical<0):
        raise ValueError('Counts and critical counts must be nonnegative integers.')
    flags=counts>=critical
    flags[...,grid_size-1]=counts[...,grid_size-1]<=critical[grid_size-1]
    return np.all(flags,axis=-1)


def cp_interval(successes,trials,coverage=.95):
    a=(1-coverage)/2
    lo=0. if successes==0 else float(beta.ppf(a,successes,trials-successes+1))
    hi=1. if successes==trials else float(beta.ppf(1-a,successes+1,trials-successes))
    return [lo,hi]


def summarize_passes(passes,coverage=.95):
    successes=int(np.sum(passes));trials=len(passes)
    return {'rejects':successes,'repetitions':trials,'frequency':successes/trials,
            'clopper_pearson_interval':cp_interval(successes,trials,coverage)}


def evaluate(probabilities,contract,old,new,seed,stream=0):
    rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(stream,)))
    R=contract['monte_carlo_repetitions'];records=np.array(contract['records'])
    counts=rng.binomial(records,np.clip(probabilities,0,1),size=(R,len(records)))
    x=classify(counts,old['critical_counts'],contract['grid_size']);y=classify(counts,new['critical_counts'],contract['grid_size'])
    difference=float(np.mean(y.astype(int)-x.astype(int)))
    radius=sqrt(2*log(2/(1-contract['interval_coverage']))/R)
    exact_old,_=pass_probability(probabilities,records,old['critical_counts'],contract['grid_size'])
    exact_new,_=pass_probability(probabilities,records,new['critical_counts'],contract['grid_size'])
    return {'old':summarize_passes(x,contract['interval_coverage']),
            'pilot_selected':summarize_passes(y,contract['interval_coverage']),
            'numerical_binomial_power_crosscheck':{'old':exact_old,'pilot_selected':exact_new},
            'paired_improvement':{'mean':difference,'improved_only':int(np.sum(y&~x)),
                'worse_only':int(np.sum(x&~y)),
                'hoeffding_interval':[max(-1.,difference-radius),min(1.,difference+radius)]}}


def retained_shots(parameters,records,seed,destination,grid_size=7):
    """An independent record stream with explicit preparation/noise/Born draws."""
    rng=np.random.default_rng(seed);matrices,durations=step_channels(parameters,grid_size)
    settings=[];initial_bits=[];born=[];observed=[];data={};total_duration=0.
    for k,n in enumerate(records,1):
        prep=(rng.random(n)<parameters['preparation_error']).astype(np.uint8)
        state=np.zeros((n,3));state[:,2]=1-2*prep.astype(float)
        choices=rng.integers(0,4,size=(n,k),dtype=np.uint8)
        for j in range(k):state=np.einsum('nij,nj->ni',matrices[choices[:,j]],state)
        projective=(rng.random(n)<(1-state[:,2])/2).astype(np.uint8)
        flip_probability=np.where(projective,parameters['false_negative'],parameters['false_positive'])
        measured=projective^(rng.random(n)<flip_probability).astype(np.uint8)
        settings.extend([k]*n);initial_bits.extend(prep);born.extend(projective);observed.extend(measured)
        data[f'noise_choices_step_{k}']=choices
        total_duration+=float(durations[choices].sum())
    order=rng.permutation(sum(records))
    grouped_settings=np.array(settings,dtype=np.uint8);grouped_observed=np.array(observed,dtype=np.uint8)
    counts=[int(grouped_observed[grouped_settings==k].sum()) for k in range(1,2*grid_size)]
    data.update(setting=grouped_settings[order],observed=grouped_observed[order],
                initial_bit=np.array(initial_bits,dtype=np.uint8)[order],
                projective_bit=np.array(born,dtype=np.uint8)[order],
                group_to_acquisition_permutation=order,
                source=np.array('SIMULATED; no laboratory data'),seed=np.array(seed,dtype=np.int64))
    np.savez_compressed(destination,**data)
    return {'file':Path(destination).name,'records':sum(records),'counts':counts,
            'seed':seed,'channel_applications':sum((k+1)*n for k,n in enumerate(records)),
            'realized_evolution_duration':total_duration,
            'sha256':hashlib.sha256(Path(destination).read_bytes()).hexdigest(),
            'randomized_acquisition_order':True,'separate_from_pilot_and_power_holdout':True}


def classical_library(contract,old,new):
    rng=np.random.default_rng(contract['seeds']['classical_library']);selected=[]
    config=contract['classical_library'];records=contract['records']
    for n in config['dimensions']:
        rows=[]
        for j in range(config['models_per_dimension']):
            if j%2:
                P=np.array([rng.dirichlet(np.full(n,.15)) for _ in range(n)])
                model_class='homogeneous discrete stochastic channel'
            else:
                Q=rng.exponential(size=(n,n));np.fill_diagonal(Q,0.)
                Q-=np.diag(Q.sum(axis=1));P=expm(Q*10**rng.uniform(-2,1))
                model_class='homogeneous continuous-time chain sampled at a fixed step'
            initial=rng.dirichlet(np.full(n,.2))
            detector=rng.random(n) if j%3 else rng.integers(0,2,n).astype(float)
            probabilities=classical_probabilities(P,initial,detector)
            probabilities=np.clip(probabilities,0,1)
            _,a=pass_probability(probabilities,records,old['critical_counts'],contract['grid_size'])
            _,b=pass_probability(probabilities,records,new['critical_counts'],contract['grid_size'])
            rows.append({'dimension':n,'candidate_index':j,'class':model_class,
                'transition':P.tolist(),'preparation':initial.tolist(),'detector':detector.tolist(),
                'probabilities':probabilities.tolist(),'pilot_selection_log_rejection_probability':max(a,b)})
        selected.extend(sorted(rows,key=lambda z:z['pilot_selection_log_rejection_probability'],reverse=True)[:config['selected_per_dimension']])
    return selected


def run(contract,contract_bytes,output_directory):
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=True)
    N=contract['grid_size'];records=contract['records'];seeds=contract['seeds']
    scenarios={s['name']:s['parameters'] for s in contract['scenarios']}
    old={'epsilon':contract['old_diagonal_probability_boundary'],
         'beta':contract['old_offdiagonal_probability_boundary'],
         'tau':contract['old_antichain_threshold'],'critical_counts':contract['old_critical_counts']}
    old_certificate=certify_plan(old,records,N,F(contract['alpha']));assert old_certificate['valid']
    physical_probabilities={name:probability_engine(p,N) for name,p in scenarios.items()}
    solver_error=max(float(np.max(abs(physical_probabilities[name]-lindblad_probabilities(p,N)))) for name,p in scenarios.items())
    assert solver_error<1e-11
    pilot=contract['pilot'];rng=np.random.default_rng(pilot['seed'])
    pilot_counts=rng.binomial(pilot['records_per_setting'],physical_probabilities[pilot['scenario']]).tolist()
    new,candidates=select_pilot_plan(contract,pilot_counts)
    new_certificate=certify_plan(new,records,N,F(contract['alpha']));assert new_certificate['valid']
    results=[]
    for j,(name,parameters) in enumerate(scenarios.items()):
        ps=physical_probabilities[name]
        row={'name':name,'parameters':parameters,'mean_probabilities':ps.tolist(),
             'holdout':evaluate(ps,contract,old,new,seeds['heldout'],j)}
        results.append(row)
    nulls=classical_library(contract,old,new)
    for j,row in enumerate(nulls):
        row['holdout']=evaluate(row['probabilities'],contract,old,new,seeds['classical_holdout'],j)
        if not np.isfinite(row['pilot_selection_log_rejection_probability']):
            row['pilot_selection_log_rejection_probability']=None
    fake=physical_probabilities['nominal']
    drift={'hidden_states':1,'generator':[[0]],'fixed_detector':False,
           'detector_success_probability_by_setting':fake.tolist(),
           'outside_contract':True,
           'explanation':'A setting-dependent readout creates the entire pattern on one frozen hidden state.',
           'holdout':evaluate(fake,contract,old,new,seeds['drift_control'])}
    quasistatic=scenarios['nominal']|{'amplitude_jitter':.1,'timing_jitter':.05}
    iid=probability_engine(quasistatic,N);static=quasi_static_probabilities(quasistatic,N)
    memory={'parameters':quasistatic,'iid_step_mean_probabilities':iid.tolist(),
            'whole_run_quasistatic_probabilities':static.tolist(),
            'maximum_difference':float(np.max(abs(iid-static))),
            'contract':'A noise label held for the run is controller memory; it must be included in a classical state count.'}
    raw=retained_shots(scenarios[contract['raw_dataset_scenario']],records,seeds['raw_shots'],
                      output_directory/'dimension_lab_shots.npz',N)
    raw['scenario']=contract['raw_dataset_scenario']
    raw['old_rejects']=bool(classify(np.array(raw['counts']),old['critical_counts'],N))
    raw['pilot_selected_rejects']=bool(classify(np.array(raw['counts']),new['critical_counts'],N))
    expected=physical_probabilities[raw['scenario']]*np.array(records)
    variance=expected*(1-physical_probabilities[raw['scenario']])
    raw['maximum_marginal_standardized_count_residual']=float(np.max(abs(np.array(raw['counts'])-expected)/np.sqrt(variance)))
    return {'evidence':'Independent physical simulations and synthetic records; no empirical laboratory validation',
        'contract_sha256':hashlib.sha256(contract_bytes).hexdigest(),'monte_carlo_seed':seeds['heldout'],
        'bloch_lindblad_maximum_probability_difference':solver_error,
        'old_plan':old,'old_exact_certificate':old_certificate,
        'pilot_selected_plan':new,'pilot_selected_exact_certificate':new_certificate,
        'pilot':{'seed':pilot['seed'],'scenario':pilot['scenario'],'counts':pilot_counts,
                 'records_per_setting':pilot['records_per_setting'],'total_records':(2*N-1)*pilot['records_per_setting'],
                 'candidate_plans':candidates,'heldout_seed_distinct':True},
        'production_budget':{'records':sum(records),'nominal_channel_applications':sum((k+1)*n for k,n in enumerate(records)),
            'nominal_evolution_duration':sum((k+1)*n for k,n in enumerate(records))*2*np.pi/N,
            'matched_between_plans':True,'excludes_reset_readout_controller_overhead':True,
            'pilot_records_are_additional':True},
        'alternative_scenarios':results,'selected_classical_nulls':nulls,
        'outside_contract_readout_control':drift,'quasistatic_memory_control':memory,
        'retained_shot_stream':raw,
        'interpretation':['Both plans have target-independent type-I validity under the common-channel/preparation/detector contract.',
            'IID errors per repeated step preserve the effective channel; errors held for an entire run add hidden memory.',
            'Fixed preparation and detector error reduce power without invalidating the null theorem.',
            'The selected plan is optimized for one independent simulated pilot, not universally better.',
            'Power intervals are marginal95percent Clopper-Pearson Monte Carlo intervals.',
            'Parameter values are illustrative and have no claimed device calibration.']}



def fit_common_clock(pilot_counts,records_per_setting,assumed_parameters,bounds):
    """A model-dependent design proposal, never part of null certification."""
    counts=np.array(pilot_counts)
    def objective(offset):
        probabilities=np.clip(probability_engine(assumed_parameters|{'amplitude_offset':float(offset)}),1e-14,1-1e-14)
        return float(-np.sum(counts*np.log(probabilities)+(records_per_setting-counts)*np.log1p(-probabilities)))
    fit=minimize_scalar(objective,bounds=tuple(bounds),method='bounded',options={'xatol':1e-10})
    if not fit.success:raise RuntimeError('Clock-calibration fit did not converge.')
    return {'amplitude_offset_estimate':float(fit.x),'common_clock_scale':1/(1+float(fit.x)),
            'negative_log_likelihood':float(fit.fun),'fit_success':bool(fit.success),
            'model_dependent_power_calibration_not_null_assumption':True}


def run_retiming(parent,contract,contract_bytes,output_directory):
    scenarios={s['name']:s['parameters'] for s in parent['scenarios']}
    records=parent['records'];critical=parent['old_critical_counts'];N=parent['grid_size']
    rawrows=[];pilot=contract['pilot'];R=contract['holdout']['repetitions']
    for index,name in enumerate(contract['scenarios']):
        parameters=scenarios[name]
        old_probabilities=probability_engine(parameters,N)
        rng=np.random.default_rng(np.random.SeedSequence(pilot['seed'],spawn_key=(index,)))
        pilot_counts=rng.binomial(pilot['records_per_setting'],old_probabilities).tolist()
        fit=fit_common_clock(pilot_counts,pilot['records_per_setting'],scenarios['nominal'],pilot['offset_bounds'])
        retimed_parameters=parameters|{'clock_scale':fit['common_clock_scale']}
        probabilities=probability_engine(retimed_parameters,N)
        solver_error=float(np.max(abs(probabilities-lindblad_probabilities(retimed_parameters,N))))
        assert solver_error<1e-11
        summaries={}
        for arm,ps in enumerate([old_probabilities,probabilities]):
            rng=np.random.default_rng(np.random.SeedSequence(contract['holdout']['seed'],spawn_key=(index,arm)))
            counts=rng.binomial(np.array(records),np.clip(ps,0,1),size=(R,len(records)))
            summary=summarize_passes(classify(counts,critical,N))
            summary['numerical_binomial_power_crosscheck']=pass_probability(ps,records,critical,N)[0]
            summaries['original_clock' if arm==0 else 'retimed_clock']=summary
        difference=summaries['retimed_clock']['frequency']-summaries['original_clock']['frequency']
        radius=2*sqrt(log(4/.05)/(2*R))
        old_duration=sum((k+1)*n for k,n in enumerate(records))*2*np.pi/N
        pilot_steps=pilot['records_per_setting']*sum(pilot['settings'])
        rawrows.append({'name':name,'pilot_counts':pilot_counts,'fit':fit,
            'original_parameters':parameters,'retimed_parameters':retimed_parameters,
            'original_probabilities':old_probabilities.tolist(),'retimed_probabilities':probabilities.tolist(),
            'holdout':summaries,'power_improvement':{'difference':difference,
                'hoeffding_interval':[max(-1.,difference-radius),min(1.,difference+radius)]},
            'bloch_lindblad_maximum_probability_difference':solver_error,
            'production_records_each':sum(records),'production_steps_each':sum((k+1)*n for k,n in enumerate(records)),
            'original_nominal_evolution_duration':old_duration,
            'retimed_nominal_evolution_duration':old_duration*fit['common_clock_scale'],
            'pilot_resources':{'records':pilot['records_per_setting']*len(pilot['settings']),
                'channel_applications':pilot_steps,'nominal_evolution_duration':pilot_steps*2*np.pi/N},
            'amortized_pilot_overhead_if_calibration_remains_valid':[
                {'production_experiments':uses,'extra_records_per_experiment':pilot['records_per_setting']*len(pilot['settings'])/uses,
                 'extra_channel_steps_per_experiment':pilot_steps/uses,
                 'fraction_of_production_record_budget':pilot['records_per_setting']*len(pilot['settings'])/(uses*sum(records))}
                for uses in [1,10,100,1000]]})
    return {'evidence':'Second frozen simulated pilot/holdout loop; no physical calibration or measured data',
        'contract_sha256':hashlib.sha256(contract_bytes).hexdigest(),'pilot_seed':pilot['seed'],
        'heldout_seed':contract['holdout']['seed'],'old_critical_counts_unchanged':critical,
        'shared_channel_argument':'A pilot-selected common step defines one fixed P; conditioning on the independent pilot leaves the old dimension theorem and exact tail guarantees unchanged.',
        'settings_are_not_individually_retimed':True,'scenarios':rawrows,
        'limitations':['A model-based frequency fit proposes a common clock; it does not prove hardware stationarity or readout consistency.',
            'Pilot cost is260000records per calibration; production-only comparisons are not total-resource comparisons.',
            'Amortization assumes the calibrated drive remains sufficiently stable across subsequent experiments.',
            'A common timing repair cannot undo arbitrary detector error, depolarization or setting-dependent drift.',
            'All four declared cases and independent95percent Monte Carlo intervals are retained.']}



def run_total_budget(parent,contract,contract_bytes):
    weights=np.array(parent['records']);total=contract['total_records']
    exact=[F(int(n)*total,int(weights.sum())) for n in weights]
    records=[x.numerator//x.denominator for x in exact]
    remainder=total-sum(records)
    order=sorted(range(len(records)),key=lambda i:(exact[i]-records[i],-i),reverse=True)
    for i in order[:remainder]:records[i]+=1
    critical=[]
    for j,n in enumerate(records):
        k,_=critical_count(n,F(parent['old_diagonal_probability_boundary']) if j==6 else F(parent['old_offdiagonal_probability_boundary']),
                           'low' if j==6 else 'high',F(parent['alpha']))
        critical.append(k)
    plan={'epsilon':parent['old_diagonal_probability_boundary'],'beta':parent['old_offdiagonal_probability_boundary'],
          'tau':parent['old_antichain_threshold'],'critical_counts':critical}
    certificate=certify_plan(plan,records);assert certificate['valid']
    scenarios={s['name']:s['parameters'] for s in parent['scenarios']};rows=[]
    for index,name in enumerate(contract['scenarios']):
        probabilities=probability_engine(scenarios[name])
        rng=np.random.default_rng(np.random.SeedSequence(contract['seed'],spawn_key=(index,)))
        counts=rng.binomial(np.array(records),np.clip(probabilities,0,1),size=(contract['monte_carlo_repetitions'],len(records)))
        row=summarize_passes(classify(counts,critical))
        row['numerical_binomial_power_crosscheck']=pass_probability(probabilities,records,critical)[0]
        rows.append({'name':name,'holdout':row})
    steps=sum((k+1)*n for k,n in enumerate(records))
    return {'evidence':'Additional frozen synthetic comparison; no measured data',
            'contract_sha256':hashlib.sha256(contract_bytes).hexdigest(),'seed':contract['seed'],
            'records':records,'total_records':sum(records),'plan':plan,'exact_certificate':certificate,
            'total_steps':steps,'nominal_evolution_duration':steps*2*np.pi/7,'scenarios':rows,
            'limitations':['Matchespilotplusproductionrecordcount, butphysicaldurationsaredifferentandreported.',
                'Retainsoriginalclockandoriginalforbiddenprobabilityregion;notanoptimizedororaclebaseline.',
                'Conclusionsaboutallpossibleexperimentaldesignsoruniversalcostsuperioritydonotfollow.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,default=Path(__file__).resolve().parents[1]/'contracts'/'dimension_lab.json')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--retiming-contract',type=Path)
    parser.add_argument('--total-budget-contract',type=Path)
    args=parser.parse_args();raw=args.contract.read_bytes();contract=json.loads(raw)
    if args.total_budget_contract is not None:
        extra_bytes=args.total_budget_contract.read_bytes()
        result=run_total_budget(contract,json.loads(extra_bytes),extra_bytes)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'dimension_lab_total_budget.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({r['name']:r['holdout']['frequency'] for r in result['scenarios']}))
        return
    if args.retiming_contract is not None:
        retiming_bytes=args.retiming_contract.read_bytes()
        result=run_retiming(contract,json.loads(retiming_bytes),retiming_bytes,args.output_dir)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'dimension_lab_retiming.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({r['name']:{name:row['frequency'] for name,row in r['holdout'].items()} for r in result['scenarios']}))
        return
    result=run(contract,raw,args.output_dir)
    (args.output_dir/'dimension_lab.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    target=next(r for r in result['alternative_scenarios'] if r['name']=='combined_design')
    print(json.dumps({'pilot_selected_plan':result['pilot_selected_plan'],
                      'combined_holdout':target['holdout'],'solver_error':result['bloch_lindblad_maximum_probability_difference']}))


if __name__=='__main__':main()
