"""Frozen independent-trajectory evaluation of predictable contrast learning."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.stats import beta
from constructive_access.virtual import contrast,thermodynamic_lab as lab
from constructive_access.frontier import energy_certificate as ec
try:
    from . import sequential
except ImportError:
    import sequential


def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def batch_policies(bits,mode,grid=48,bet_grid=128,alpha=F(1,20)):
    """Integer-score vectorization of sequential.policy, checked by exact replay."""
    sequential.policy([0,0,0],0,mode,grid,bet_grid)
    if bits.ndim!=3 or not np.all((bits==0)|(bits==1)):
        raise ValueError('a three-dimensional binary bank is required')
    rep,horizon,arms=bits.shape
    if arms!=3 or mode not in ('learned','fixed'):raise ValueError('invalid bank or policy')
    if horizon>100000 or horizon*42*grid**4*bet_grid**2>np.iinfo(np.int64).max:
        raise ValueError('outside checked int64 score bounds')
    denominator=8*grid**4*bet_grid**2
    threshold=ec.log_interval(1/alpha)[1]*denominator
    crossing_integer=threshold.numerator//threshold.denominator+1
    counts=np.zeros((rep,3),dtype=np.int64);score=np.zeros(rep,dtype=np.int64)
    first=np.full(rep,horizon+1,dtype=int)
    for n in range(horizon):
        center=2*counts-n
        r=np.full(rep,2*grid//3,dtype=np.int64)
        if mode=='learned' and n:
            valid=(center[:,0]>0)&(center[:,1]>0)
            r[:]=0
            numerator=grid*center[valid,1];den=center[valid,0]
            r[valid]=np.minimum(grid,(2*numerator+den)//(2*den))
        aa=np.column_stack((-r*r,2*r*grid,-np.full(rep,grid*grid,dtype=np.int64)))
        h=np.sum(aa*center,axis=1)
        norm=r**4+4*r*r*grid*grid+grid**4
        numerator=2*grid*grid*bet_grid*np.maximum(0,h);den=(n+2)*norm
        ell=np.minimum(bet_grid,(2*numerator+den)//(2*den))
        z=2*np.sum(aa*bits[:,n,:],axis=1)+(grid-r)**2
        score+=4*grid*grid*bet_grid*ell*z-ell*ell*norm
        hit=(score>=crossing_integer)&(first==horizon+1)
        first[hit]=n+1
        counts+=bits[:,n,:]
    capped=np.minimum(first,horizon)
    return dict(first_crossing=first,capped_batches=capped,final_score=score,
                score_denominator=denominator,detected=first<=horizon)


def fixed_horizon(bits,alpha=F(1,20)):
    n=bits.shape[1];a=[F(-4,9),F(4,3),F(-1)]
    support=contrast.support([1,2,3],a)['reversible_support_upper']
    radius=contrast.sqrt_upper(ec.log_interval(1/alpha)[1]*sum(c*c/n for c in a)/2)
    threshold=support+radius
    out=[]
    for counts in bits.sum(axis=1):
        empirical=sum(c*F(int(s),n) for c,s in zip(a,counts))
        out.append(empirical>threshold)
    return np.array(out,dtype=bool)


def summarize(result,horizon,batch_duration=F(9,10)):
    detected=result['detected'];n=len(detected);k=int(detected.sum())
    capped=result['capped_batches']
    return dict(detections=k,replicates=n,probability=k/n,mc_interval_95=interval(k,n),
        mean_capped_batches=float(capped.mean()),mean_capped_pairs=float(3*capped.mean()),
        mean_capped_endpoint_readouts=float(6*capped.mean()),
        mean_capped_evolution_time=float(float(batch_duration)*capped.mean()),
        median_capped_batches=float(np.median(capped)),maximum_batches=horizon,
        interpretation='Non-detections consume the full budget; means include them, not only successful runs.')


def run(root):
    contract_path=root/'contracts/sequential_benchmark.json'
    c=json.loads(contract_path.read_text());rng=np.random.default_rng(c['seed'])
    rep=c['replicates'];n=c['maximum_batches'];durations=list(map(F,c['lags']));alpha=F(c['alpha'])
    detector=np.cos(2*np.pi*np.arange(c['ring_states'])/c['ring_states'])
    arrays={};reports=[];replays=[]
    for scenario in c['cases']:
        bits=np.empty((rep,n,3),dtype=np.uint8)
        for k,t in enumerate(durations):
            raw=lab.ring_records(rng,rep*n,float(t),scenario['forward'],scenario['backward'],detector,scenario['gain'])
            bits[:,:,k]=raw['same'].reshape(rep,n)
        arrays[scenario['name']+'__bits']=bits
        report=dict(case=scenario['name'],simulation=True,
                    true_entropy_rate=(scenario['forward']-scenario['backward'])*math.log(scenario['forward']/scenario['backward']),methods={})
        for mode in ['learned','fixed']:
            result=batch_policies(bits,mode,c['grid'],c['bet_grid'],alpha)
            report['methods'][mode]=summarize(result,n,sum(durations))
            arrays[scenario['name']+'__'+mode+'__first_crossing']=result['first_crossing']
            for idx in [0,1]:
                stop=min(int(result['first_crossing'][idx]),n)
                cert=sequential.certify(bits[idx,:stop,:].astype(int).tolist(),durations=durations,alpha=alpha,mode=mode,grid=c['grid'],bet_grid=c['bet_grid'])
                assert (cert['entropy_production_rate_lower']>0)==bool(result['detected'][idx])
                if stop==n:assert cert['predictable_log_score']==F(int(result['final_score'][idx]),result['score_denominator'])
                replays.append(dict(case=scenario['name'],mode=mode,index=idx,batches=stop,certificate=cert))
        fixed=fixed_horizon(bits,alpha);k=int(fixed.sum())
        report['methods']['fixed_horizon']=dict(detections=k,replicates=rep,probability=k/rep,mc_interval_95=interval(k,rep),pairs=3*n,endpoint_readouts=6*n,evolution_time=float(sum(durations)*n))
        reports.append(report)
        print(json.dumps({'case':scenario['name'],'detections':{m:v['detections'] for m,v in report['methods'].items()}}),flush=True)
    output_dir=root/'results';output_dir.mkdir(exist_ok=True)
    path=output_dir/'sequential_records.npz';lab.deterministic_npz(path,arrays)
    output=dict(evidence='held-out synthetic trajectory experiments',
        contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        raw_bank_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=reports,
        exact_prefix_replays=replays,
        limitation='No empirical apparatus validation, no optimal-policy claim, no energy advantage asserted without reset/readout calibration. Policies share a simulated full bank; future bits are inaccessible to their decisions.')
    (output_dir/'sequential_benchmark.json').write_text(json.dumps(contrast.jsonable(output),sort_keys=True,indent=2)+'\n')


if __name__=='__main__':run(Path(__file__).resolve().parents[1])
