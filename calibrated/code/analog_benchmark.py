"""Frozen matched analog/virtual-coin benchmark on stationary CTMC pair records."""
from fractions import Fraction as F
from pathlib import Path
import json
import hashlib
import math
import numpy as np
from scipy.linalg import expm
try:
    from . import analog, calibration
except ImportError:
    import analog, calibration


def wilson(successes,total,z=1.959963984540054):
    p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total**2))/d
    return [max(0,center-half),min(1,center+half)]


def _reading(states,rng,case,sensor,scale):
    z=sensor[states]
    if case=='noisy_driven':return z+scale*rng.choice([-1,1],size=len(states)),2*scale
    return z,scale


def population(states,forward,backward,sensor,scale,lags,case):
    Q=np.zeros((states,states))
    for j in range(states):Q[j,(j+1)%states]=forward;Q[j,(j-1)%states]=backward;Q[j,j]=-forward-backward
    f=sensor.astype(float)/scale
    if case=='noisy_driven':
        square=(f*f+1)/4;f=f/2
    else:square=f*f
    p=[];r=[]
    for lag in lags:
        P=expm(float(lag)*Q);C=np.mean(f*(P@f));D=np.mean(square*(P@square))
        p.append((1+C)/2);r.append((1+2*C+D)/4)
    return {'agreement_means':p,'score_second_moments':r,'contrast':p[0]-p[1],
            'mean':float(np.mean(f)),'A':float(np.mean(f*f)),'observed_second_moment':float(np.mean(square)),
            'population_direct_contrast_variance':None,'population_coin_contrast_variance':None}


def run(design):
    root=Path(__file__).resolve().parents[1];rng=np.random.default_rng(design['seed'])
    reps=design['replicates_per_case'];N=design['science_records_per_lag'];Nc=design['independent_calibration_records']
    states=design['cycle_states'];scale=design['sensor_grid_denominator'];lags=list(map(F,design['science_lags']))
    sensor=np.rint(scale*np.cos(2*np.pi*np.arange(states)/states)).astype(np.int64)
    rows=[];summaries=[];raw_exemplars=[]
    for casecfg in design['cases']:
        name=casecfg['name'];forward=float(F(str(casecfg['forward_rate'])));backward=float(F(str(casecfg['backward_rate'])))
        target=population(states,forward,backward,sensor,scale,lags,name)
        target['population_direct_contrast_variance']=sum(r-p*p for p,r in zip(target['agreement_means'],target['score_second_moments']))/N
        target['population_coin_contrast_variance']=sum(p*(1-p) for p in target['agreement_means'])/N
        case_rows=[]
        for rep in range(reps):
            st=rng.integers(0,states,size=Nc);zc,den=_reading(st,rng,name,sensor,scale)
            mhat=F(int(np.sum(zc)),Nc*den);Bhat=F(int(np.sum(zc*zc)),Nc*den*den)
            cal=calibration.analog_calibration_summary(Nc,mhat,Bhat,F(design['confidence_calibration']))
            means=[];coinmeans=[];rawvars=[];coinvars=[];sufficient=[]
            exemplar={'case':name,'replicate':rep,'calibration_readings':zc.tolist(),
                      'calibration_denominator':den,'science':[]} if rep==0 else None
            for lag in lags:
                x=rng.integers(0,states,size=N)
                y=(x+rng.poisson(forward*float(lag),size=N)-rng.poisson(backward*float(lag),size=N))%states
                z0,den0=_reading(x,rng,name,sensor,scale);zt,dent=_reading(y,rng,name,sensor,scale)
                prod=z0*zt;D=den0*dent
                mean=F(N*D+int(np.sum(prod)),2*N*D);means.append(mean)
                W=(D+prod).astype(float)/(2*D)
                bits=rng.random(N)<W;coinmeans.append(F(int(np.sum(bits)),N))
                rawvars.append(float(np.var(W,ddof=1)));coinvars.append(float(np.var(bits,ddof=1)))
                sufficient.append({'records':N,'product_sum':int(np.sum(prod)),
                    'product_denominator':D,'encoded_successes':int(np.sum(bits))})
                if exemplar is not None:
                    exemplar['science'].append({'lag':str(lag),'left':z0.tolist(),'right':zt.tolist(),
                        'left_denominator':den0,'right_denominator':dent,
                        'encoded_bits':bits.astype(int).tolist()})
            if exemplar is not None:raw_exemplars.append(exemplar)
            kwargs=dict(lags=[1,2],coefficients=[1,-1],records=[N,N],durations=lags,
                        calibration_vertices=cal['vertices'],alpha=F(design['confidence_science']),
                        calibration_failure=F(design['confidence_calibration']))
            direct=analog.certify_means(means=means,**kwargs);coin=analog.certify_means(means=coinmeans,**kwargs)
            row={'case':name,'replicate':rep,'direct_contrast':float(means[0]-means[1]),
                 'coin_contrast':float(coinmeans[0]-coinmeans[1]),'target_contrast':target['contrast'],
                 'direct_reject':direct['contrast_lower']>0,'coin_reject':coin['contrast_lower']>0,
                 'direct_rate_lower':float(direct['entropy_production_rate_lower']),
                 'coin_rate_lower':float(coin['entropy_production_rate_lower']),
                 'support_upper':float(direct['support']['reversible_support_upper']),
                 'sampling_radius':float(direct['sampling_radius_upper']),
                 'calibration_B_upper':float(cal['observed_second_moment_upper']),
                 'direct_sample_variances':rawvars,'coin_sample_variances':coinvars,
                 'exact_statistics':{'calibration_records':Nc,'calibration_sum':int(np.sum(zc)),
                     'calibration_sum_squares':int(np.sum(zc*zc)),'calibration_denominator':den,
                     'science':sufficient}}
            case_rows.append(row);rows.append(row)
        d=sum(r['direct_reject'] for r in case_rows);b=sum(r['coin_reject'] for r in case_rows)
        direct_mse=float(np.mean([(r['direct_contrast']-target['contrast'])**2 for r in case_rows]))
        coin_mse=float(np.mean([(r['coin_contrast']-target['contrast'])**2 for r in case_rows]))
        difference=np.array([int(r['direct_reject'])-int(r['coin_reject']) for r in case_rows])
        mean=float(np.mean(difference));se=float(np.std(difference,ddof=1)/np.sqrt(reps))
        summaries.append({'case':name,'population':target,'replicates':reps,'direct_rejections':d,'coin_rejections':b,
                'direct_rejection_fraction':d/reps,'coin_rejection_fraction':b/reps,
                'direct_wilson95':wilson(d,reps),'coin_wilson95':wilson(b,reps),
                'paired_rejection_difference':mean,'paired_difference_normal95':[mean-1.96*se,mean+1.96*se],
                'direct_contrast_mse':direct_mse,'coin_contrast_mse':coin_mse,
                'mse_ratio_coin_over_direct':coin_mse/direct_mse,
                'physical_records_per_replicate':2*N+Nc,
                'detector_readouts_per_replicate':4*N+Nc,
                'science_evolution_time_per_replicate':str(N*sum(lags)),
                'unmeasured_overhead':'stationary preparation and detector/calibration overhead not claimed zero'})
        print(json.dumps({'case':name,'direct_rejections':d,'coin_rejections':b,'replicates':reps}),flush=True)
    return {'evidence':'held-out synthetic stationary pair records from frozen design; no laboratory data',
            'design':design,'summaries':summaries,'replicate_results':rows,'first_raw_bank_per_case':raw_exemplars,
            'logging_revision':'Added exact sufficient statistics, first raw bank per case, and detector-readout counts after independent review; frozen seed, data generation, certificates and evaluation choices unchanged.',
            'interval_scope':'Wilson intervals are nominal binomial 95% intervals; paired normal intervals are asymptotic summaries and degenerate [0,0] intervals do not establish equality.',
            'comparison_scope':'same analog bank and calibration; only virtual coin reduction differs; not a claim of universal power dominance'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];source=root/'contracts'/'analog_benchmark.json'
    design=json.loads(source.read_text());result=run(design)
    result['design_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=root/'results'/'analog_benchmark.json';out.write_text(json.dumps(result,indent=2)+'\n');print(out)
