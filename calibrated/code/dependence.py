"""Whole-block Markov confidence without a fresh long gap before every record.

The stationary state-function bound is Fan--Jiang--Sun (2021).  A conditional
Hoeffding martingale term and Holder's inequality extend it to a statistic of
each noninvasively observed block.  A calibrated mixing envelope is required;
this module never estimates an unseen mixing rate from the same counts.
"""
from fractions import Fraction as F
from pathlib import Path
from functools import lru_cache
import hashlib
import json

from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual import contrast,sampling


@lru_cache(maxsize=2048)
def block_proxy(gap_lower,block_duration):
    rate,H=map(ec.exact,(gap_lower,block_duration))
    if rate<=0 or H<=0:
        raise ValueError("Positive calibrated gap and common block duration required")
    rho=sampling.exp_negative_upper(rate*H)
    if not 0<=rho<1:
        raise ArithmeticError("A strict contraction must be certified")
    A=(1+rho)/(1-rho)
    root=contrast.sqrt_upper(A)
    return dict(contraction_upper=rho,state_function_proxy=A,
                whole_record_proxy_upper=(root+1)**2,
                calibrated_gap_lower=rate,common_block_duration=H)


def _validate(records,successes,alpha,initial_tv,external_failure):
    alpha,beta,eta=map(ec.exact,(alpha,initial_tv,external_failure))
    if not 0<alpha<1 or not 0<=beta<1 or not 0<=eta<1 or alpha+beta+eta>=1:
        raise ValueError("Invalid global failure budget")
    if not records or len(records)!=len(successes):
        raise ValueError("Matching nonempty count arrays required")
    if any(type(n) is not int or n<=0 for n in records):
        raise ValueError("Positive fixed integer quotas required")
    if any(type(s) is not int or not 0<=s<=n for n,s in zip(records,successes)):
        raise ValueError("Invalid success counts")
    return alpha,beta,eta


def certify_counts(lags,coefficients,records,successes,durations,block_duration,
                   gap_lower,alpha=F(49,1000),initial_tv=F(1,1000),
                   calibration_radii=None,external_failure=0,
                   mean_interval=(-1,1),reference_mean_calibrated=False,
                   selection="predeclared",schedule="fixed_common_blocks",depth=6):
    """A fixed contrast with dependent block records and a known L2 envelope.

    Schedule and coefficients must be fixed independently of held-out outcomes.
    All hidden start states evolve for the same deterministic block duration H.
    Each noninvasive record lies within its block; endpoint detector draws are
    fresh given the hidden state, or their memory is included in the calibrated
    state. initial_tv is initial-law TV distance, transferred to the entire
    future law once; external_failure is a different global calibration event.
    """
    if selection not in ("predeclared","independent_pilot"):
        raise ValueError("Freeze coefficients before held-out outcomes")
    if schedule!="fixed_common_blocks":
        raise ValueError("Adaptive routing and irregular block times are unsupported")
    alpha,beta,eta=_validate(records,successes,alpha,initial_tv,external_failure)
    proxy=block_proxy(gap_lower,block_duration);H=proxy["common_block_duration"]
    durations=list(map(ec.exact,durations));m=len(records)
    if len(durations)!=m or any(not 0<t<=H for t in durations):
        raise ValueError("Each complete observation must fit within the common block")
    sup=contrast.support(lags,coefficients,mean_interval,depth)
    if len(lags)!=m:
        raise ValueError("One deterministic quota per setting required")
    if sup["mean_interval"]!=(F(-1),F(1)) and not reference_mean_calibrated:
        raise ValueError("A narrow reference mean needs independent calibration")
    a=sup["coefficients"]
    errors=[F(0)]*m if calibration_radii is None else list(map(ec.exact,calibration_radii))
    if len(errors)!=m or any(not 0<=x<=1 for x in errors):
        raise ValueError("Invalid whole-record calibration radii")
    V=sum((c*c/n for c,n in zip(a,records)),F(0))
    radius=contrast.sqrt_upper(proxy["whole_record_proxy_upper"]*V*
                               ec.log_interval(1/alpha)[1]/2)
    bias=sum((abs(c)*e for c,e in zip(a,errors)),F(0))
    empirical=sum((c*F(s,n) for c,s,n in zip(a,successes,records)),F(0))
    lower=empirical-radius-bias
    gap=max(F(0),lower-sup["reversible_support_upper"])
    denominator=sum((abs(c)*contrast.sqrt_upper(t) for c,t in zip(a,durations)),F(0))**2
    return dict(status="positive_certificate" if gap else "no_positive_certificate",
                records=list(records),successes=list(successes),lags=list(lags),
                coefficients=a,durations=durations,block_duration=H,proxy=proxy,
                support=sup,empirical_contrast=empirical,sampling_radius_upper=radius,
                whole_record_calibration_penalty=bias,contrast_lower=lower,
                reversible_gap_lower=gap,entropy_production_rate_lower=8*gap*gap/denominator,
                stationary_statistical_failure=alpha,initial_path_tv_upper=beta,
                external_contract_failure_upper=eta,total_failure_upper=alpha+beta+eta,
                selection=selection,schedule=schedule,
                contract="Stationary common-block hidden Markov reference; externally calibrated absolute L2 contraction; fixed schedule; bounded noninvasive records; initial TV transferred once; physical entropy assumptions remain separate")


def confidence_box(records,successes,block_duration,gap_lower,alpha,
                   initial_tv=0,calibration_radii=None,external_failure=0):
    alpha,beta,eta=_validate(records,successes,alpha,initial_tv,external_failure)
    proxy=block_proxy(gap_lower,block_duration);m=len(records)
    errors=[F(0)]*m if calibration_radii is None else list(map(ec.exact,calibration_radii))
    if len(errors)!=m or any(not 0<=e<=1 for e in errors):
        raise ValueError("Invalid calibration radii")
    log=ec.log_interval(F(2*m)/alpha)[1]
    radii=[contrast.sqrt_upper(proxy["whole_record_proxy_upper"]*log/(2*n)) for n in records]
    boxes=[(max(F(0),F(s,n)-r-e),min(F(1),F(s,n)+r+e))
           for n,s,r,e in zip(records,successes,radii,errors)]
    return dict(intervals=boxes,radii=radii,proxy=proxy,
                total_failure_upper=alpha+beta+eta,
                required_contract="Fixed independent-of-data schedule and equal-duration blocks; calibrated hidden L2 contraction, initial TV and detector assumptions")


def optimize_old_gap(budget,H,gap_lower,pmin,coefficients,alpha,candidates):
    """Strong prior baseline: exact radius minimization over a declared gap grid."""
    budget,H,gap_lower,pmin,alpha=map(ec.exact,(budget,H,gap_lower,pmin,alpha))
    a=list(map(ec.exact,coefficients));m=len(a)
    if not a or any(c==0 for c in a) or H<=0 or budget<=0 or not 0<alpha<1:
        raise ValueError("Invalid allocation contract")
    rows=[]
    for candidate in candidates:
        G=ec.exact(candidate)
        if G<0:raise ValueError("Negative blind gap")
        n=int(budget//(m*(G+H)))
        if n<1:continue
        delta=sampling.spectral_mix_bound(gap_lower,pmin,G)["conditional_bias_upper"]
        variance=sum((c*c/n for c in a),F(0))
        r=contrast.sqrt_upper(ec.log_interval(1/alpha)[1]*variance/2)
        halfwidth=r+sum(map(abs,a))*delta
        rows.append((halfwidth,G,n,r,delta))
    if not rows:raise ValueError("No feasible declared gap candidate")
    halfwidth,G,n,r,delta=min(rows)
    return dict(blind_gap=G,records_per_setting=n,sampling_radius_upper=r,
                conditional_bias_upper=delta,contrast_halfwidth=halfwidth,
                physical_time=m*n*(G+H),unused_time=budget-m*n*(G+H),
                pair_records=m*n,endpoint_readouts=2*m*n,
                search_scope="Optimum over the predeclared finite gap grid; not a global protocol optimum")


def sector_group(rng,sector,records,H,lag,kappa,bit_rates):
    """Exact conditional count simulator for metastable sector/bit blocks.

    Sector flips form an independent Poisson process. Conditional on its path,
    bit flips have independent inhomogeneous-Poisson increments. Record parity
    probabilities therefore aggregate by binomial draws over constant-sector
    blocks, with individually integrated hazards in switch-containing blocks.
    Padding remains physical evolution and can change the next start sector.
    """
    import numpy as np
    if records<1 or not 0<lag<=H or kappa<0:
        raise ValueError("Invalid sector simulation contract")
    total=records*H
    count=int(rng.poisson(kappa*total))
    if not count:
        q=(1+np.exp(-2*bit_rates[sector]*lag))/2
        return int(rng.binomial(records,q)),int(sector)
    times=np.sort(rng.random(count)*total)
    bins=np.floor(times/H).astype(np.int64)
    unique,starts,event_counts=np.unique(bins,return_index=True,return_counts=True)
    start_sectors=(sector+(starts%2))%2
    clean_lengths=np.diff(np.concatenate(([-1],unique)))-1
    clean=[int(clean_lengths[start_sectors==s].sum()) for s in (0,1)]
    final_sector=(sector+count)%2
    clean[final_sector]+=records-int(unique[-1])-1
    a=np.asarray(bit_rates,dtype=float)
    before=(sector+np.arange(count)%2)%2
    delta=a[1-before]-a[before]
    remaining=np.maximum(0.,bins*H+lag-times)
    hazards=a[start_sectors]*lag+np.add.reduceat(delta*remaining,starts)
    if np.min(hazards)<-1e-10:
        raise ArithmeticError("Integrated hazard became negative")
    probabilities=(1+np.exp(-2*np.maximum(hazards,0)))/2
    successes=int(np.sum(rng.random(len(unique))<probabilities))
    for s in (0,1):
        q=(1+np.exp(-2*a[s]*lag))/2
        successes+=int(rng.binomial(clean[s],q))
    return successes,int(final_sector)


def ring_group(rng,phase,records,H,lag,forward,backward,detector):
    import numpy as np
    detector=np.asarray(detector);n=len(detector)
    observed=rng.poisson(forward*lag,records)-rng.poisson(backward*lag,records)
    if H>lag:
        padding=rng.poisson(forward*(H-lag),records)-rng.poisson(backward*(H-lag),records)
    else:
        padding=np.zeros(records,dtype=np.int64)
    full=observed+padding
    starts=(phase+np.cumsum(full)-full)%n
    ends=(starts+observed)%n
    return int(np.sum(detector[starts]==detector[ends])),int((phase+np.sum(full))%n)


def _model(spec):
    import numpy as np
    if spec["kind"]=="sector_bit":
        Q=np.zeros((4,4));k=float(F(spec["sector_flip_rate"]))
        rates=list(map(lambda x:float(F(x)),spec["bit_flip_rates"]))
        for s in (0,1):
            for b in (0,1):
                i=2*s+b;Q[i,2*s+1-b]=rates[s];Q[i,2*(1-s)+b]=k
                Q[i,i]=-rates[s]-k
        detector=np.array([0,1,0,1])
    else:
        n=spec["states"];Q=np.zeros((n,n));a=float(F(spec["forward_rate"]));b=float(F(spec["backward_rate"]))
        for i in range(n):Q[i,(i+1)%n]=a;Q[i,(i-1)%n]=b;Q[i,i]=-a-b
        detector=np.array(spec["detector"])
    return Q,detector


def _cdf(P):
    import numpy as np
    P=np.maximum(P,0);P/=P.sum(axis=1,keepdims=True)
    cdf=P.cumsum(axis=1);cdf[:,-1]=1
    return cdf


def _summary(flags):
    import numpy as np
    yes=int(np.sum(flags));n=len(flags)
    return dict(events=yes,replicates=n,frequency=yes/n,
                approximate_wilson_mc95=sampling._wilson(yes,n))


def main(root):
    import numpy as np
    from scipy.linalg import expm,eigvalsh
    contract_path=root/"contracts/dependence.json"
    raw=contract_path.read_bytes();contract=json.loads(raw)
    if not contract["frozen_before_heldout_generation"]:
        raise ValueError("Freeze the evaluation before production")
    access=contract["access_contract"];design=contract["design"]
    H=F(access["common_block_duration"]);lags=list(map(F,access["readout_lags"]))
    a=list(map(F,access["contrast_coefficients"]))
    lam=F(access["calibrated_additive_reversibilization_gap_lower"])
    pmin=F(access["calibrated_stationary_mass_lower"])
    burn=F(access["new_initial_blind_gap"]);beta=F(access["new_initial_TV_failure_allocation"])
    alpha=F(access["new_statistical_failure"]);oldalpha=F(access["total_nominal_failure"])
    mixing=sampling.spectral_mix_bound(lam,pmin,burn)
    if mixing["conditional_bias_upper"]>beta:raise ValueError("Initial TV allocation is not certified")
    reps=design["replicates_per_model_budget_protocol"];seed=design["seed"]
    grid=design["baseline_gap_candidates"]
    candidates=list(range(grid["start"],grid["stop_inclusive"]+1,grid["step"]))
    plans=[]
    for B in design["physical_time_budgets"]:
        old=optimize_old_gap(B,H,lam,pmin,a,oldalpha,candidates)
        n=int((B-burn)//(len(lags)*H))
        proto=certify_counts([1,2],a,[n]*2,[0]*2,lags,H,lam,alpha,beta)
        plans.append(dict(budget=B,old=old,new=dict(records_per_setting=n,
                     pair_records=2*n,endpoint_readouts=4*n,physical_time=burn+2*n*H,
                     unused_time=B-burn-2*n*H,contrast_halfwidth=proto["sampling_radius_upper"])))
    output=[]
    for model_index,spec in enumerate(contract["models"]):
        Q,detector=_model(spec);pi=np.full(len(Q),1/len(Q));P=[expm(Q*float(t)) for t in lags]
        targets=np.array([sum(pi[i]*p[i,j] for i in range(len(Q)) for j in range(len(Q))
                              if detector[i]==detector[j]) for p in P])
        target=float(np.dot(list(map(float,a)),targets))
        actual_gap=float(eigvalsh(-(Q+Q.T)/2)[1])
        if spec["within_calibrated_contract"] and (
                actual_gap<float(lam)-1e-12 or min(pi)<float(pmin)-1e-12):
            raise ValueError("A declared valid simulator violates the calibrated envelope")
        for budget_index,plan in enumerate(plans):
            old=plan["old"];n=plan["new"]["records_per_setting"]
            counts=np.zeros((reps,2),dtype=np.int64)
            for repetition in range(reps):
                rng=np.random.default_rng(np.random.SeedSequence([seed,model_index,budget_index,0,repetition]))
                stationary=not spec["within_calibrated_contract"]
                if spec["kind"]=="sector_bit":
                    kappa=float(F(spec["sector_flip_rate"]))
                    rates=list(map(lambda x:float(F(x)),spec["bit_flip_rates"]))
                    state=int(rng.integers(0,2)) if stationary else 0
                    state=(state+int(rng.poisson(kappa*float(burn))))%2
                    for k,t in enumerate(lags):
                        counts[repetition,k],state=sector_group(rng,state,n,float(H),float(t),kappa,rates)
                else:
                    forward=float(F(spec["forward_rate"]));backward=float(F(spec["backward_rate"]))
                    state=(int(rng.poisson(forward*float(burn)))-int(rng.poisson(backward*float(burn))))%len(Q)
                    for k,t in enumerate(lags):
                        counts[repetition,k],state=ring_group(rng,state,n,float(H),float(t),forward,backward,detector)
            oldn=old["records_per_setting"]
            oldcounts=np.zeros((reps,2),dtype=np.int64)
            rng=np.random.default_rng(np.random.SeedSequence([seed,model_index,budget_index,1]))
            state=rng.integers(0,len(Q),size=reps) if not spec["within_calibrated_contract"] else np.zeros(reps,dtype=int)
            gapcdf=_cdf(expm(Q*float(old["blind_gap"])))
            for k,t in enumerate(lags):
                pcdf=_cdf(P[k]);padcdf=_cdf(expm(Q*float(H-t)))
                for _ in range(oldn):
                    state=(rng.random(reps)[:,None]>gapcdf[state]).sum(axis=1)
                    before=detector[state]
                    state=(rng.random(reps)[:,None]>pcdf[state]).sum(axis=1)
                    oldcounts[:,k]+=before==detector[state]
                    if H>t:state=(rng.random(reps)[:,None]>padcdf[state]).sum(axis=1)
            empirical=counts@np.asarray(a,dtype=float)/n
            oldemp=oldcounts@np.asarray(a,dtype=float)/oldn
            newwidth=float(plan["new"]["contrast_halfwidth"])
            oldwidth=float(old["contrast_halfwidth"])
            naivewidth=float(contrast.sqrt_upper(ec.log_interval(1/oldalpha)[1]*sum((c*c/n for c in a),F(0))/2))
            results=[]
            for label,estimates,width,rawcounts in [("continuous_markov",empirical,newwidth,counts),
                  ("optimized_old_blind_gap",oldemp,oldwidth,oldcounts),
                  ("unqualified_independent_analysis",empirical,naivewidth,counts)]:
                lower=estimates-width
                denominator=float(sum(abs(float(c))*np.sqrt(float(t)) for c,t in zip(a,lags))**2)
                rate=8*np.maximum(lower,0)**2/denominator
                results.append(dict(method=label,one_sided_coverage=_summary(lower<=target+1e-12),
                                    equilibrium_exclusion=_summary(lower>0),
                                    halfwidth=width,mean_entropy_rate_lower=float(np.mean(rate)),
                                    invalid_analysis=(label=="unqualified_independent_analysis"),
                                    raw_counts=rawcounts.tolist(),
                                    counts_sha256=hashlib.sha256(rawcounts.astype('<i8').tobytes()).hexdigest()))
            output.append(dict(model=spec["name"],within_calibrated_contract=spec["within_calibrated_contract"],
                               budget=plan["budget"],actual_additive_gap=actual_gap,
                               stationary_pair_probabilities=targets.tolist(),true_contrast=target,
                               methods=results))
            print(json.dumps(dict(model=spec["name"],budget=plan["budget"],
                  exclusions={r["method"]:r["equilibrium_exclusion"]["events"] for r in results})),flush=True)
    result=dict(contract_sha256=hashlib.sha256(raw).hexdigest(),seed=seed,
                initial_mixing_certificate=mixing,block_proxy=block_proxy(lam,H),
                plans=plans,evaluations=output,
                scope="Synthetic held-out trajectories; calibrated mixing assumed externally; time gains do not establish energy gains; no physical laboratory data")
    (root/"results/dependence.json").write_text(json.dumps(result,default=str,sort_keys=True,indent=2)+"\n")


def optimize_block_cost(budget,burn,gap_lower,coefficients,alpha,
                        readout_cost_ratio,candidates):
    """Certified finite-grid radius design for normalized cost T + zeta R.

    R counts individual endpoint readouts, two per pair. The conversion to
    energy requires separately measured constant waiting power and detector
    energy; neither quantity is asserted by this parametric calculation.
    """
    C,G,lam,alpha,zeta=map(ec.exact,(budget,burn,gap_lower,alpha,readout_cost_ratio))
    a=list(map(ec.exact,coefficients));m=len(a)
    if C<=G or G<0 or zeta<0 or not a or not any(a) or not 0<alpha<1:
        raise ValueError("Invalid normalized resource allocation")
    best=None
    for candidate in candidates:
        H=ec.exact(candidate)
        if H<=0:raise ValueError("Positive common blocks required")
        n=int((C-G)//(m*(H+2*zeta)))
        if not n:continue
        K=block_proxy(lam,H)["whole_record_proxy_upper"]
        objective=K*sum((c*c/n for c in a),F(0))
        row=(objective,H,n,K)
        if best is None or row<best:best=row
    if best is None:raise ValueError("No feasible candidate")
    objective,H,n,K=best
    T=G+m*n*H;R=2*m*n;used=T+zeta*R
    return dict(block_duration=H,records_per_setting=n,pair_records=m*n,
                endpoint_readouts=R,physical_time=T,normalized_cost=used,
                normalized_budget=C,unused_normalized_budget=C-used,
                readout_cost_ratio=zeta,whole_record_proxy_upper=K,
                contrast_halfwidth=contrast.sqrt_upper(objective*ec.log_interval(1/alpha)[1]/2),
                selection_scope="Exact minimizer of the certified halfwidth over the declared finite block-duration grid; hypothetical normalized energy accounting")


def refinement(root):
    import numpy as np
    from scipy.linalg import expm,eigvalsh
    raw=(root/"contracts/dependence_refinement.json").read_bytes();contract=json.loads(raw)
    parent_raw=(root/"contracts/dependence.json").read_bytes();parent=json.loads(parent_raw)
    if not contract["frozen_before_heldout_generation"] or hashlib.sha256(parent_raw).hexdigest()!=contract["parent_contract_sha256"]:
        raise ValueError("Frozen parent/refinement contract mismatch")
    access=parent["access_contract"];design=contract["design"]
    a=list(map(F,access["contrast_coefficients"]));lags=list(map(F,access["readout_lags"]))
    lam=F(access["calibrated_additive_reversibilization_gap_lower"])
    burn=F(access["new_initial_blind_gap"]);alpha=F(access["new_statistical_failure"])
    beta=F(access["new_initial_TV_failure_allocation"])
    reps=design["replicates_per_model_budget_block"];seed=design["seed"]
    pmin=F(access["calibrated_stationary_mass_lower"])
    if sampling.spectral_mix_bound(lam,pmin,burn)["conditional_bias_upper"]>beta:
        raise ValueError("Initial mixing contract failed")
    evaluations=[];plans=[]
    for menu in design["evaluation_menu"]:
        B=menu["physical_time_budget"]
        for H in menu["block_durations"]:
            if H<max(lags):raise ValueError("Record exceeds block")
            n=int((B-burn)//(2*H))
            plan=optimize_block_cost(B,burn,lam,a,alpha,0,[H])
            width=float(plan["contrast_halfwidth"]);plans.append(dict(budget=B,**plan))
            for model_index,spec in enumerate(parent["models"]):
                if spec["name"] not in design["models"]:continue
                Q,detector=_model(spec);pi=np.full(len(Q),1/len(Q))
                actual_gap=float(eigvalsh(-(Q+Q.T)/2)[1])
                if spec["within_calibrated_contract"] and (actual_gap<float(lam)-1e-12 or min(pi)<float(pmin)-1e-12):
                    raise ValueError("Declared valid simulation violates its mixing contract")
                probs=[sum(pi[i]*p[i,j] for i in range(len(Q)) for j in range(len(Q))
                           if detector[i]==detector[j]) for p in [expm(Q*float(t)) for t in lags]]
                target=float(np.dot(list(map(float,a)),probs));counts=np.zeros((reps,2),dtype=np.int64)
                for rep in range(reps):
                    rng=np.random.default_rng(np.random.SeedSequence([seed,B,H,model_index,rep]))
                    if spec["kind"]=="sector_bit":
                        kappa=float(F(spec["sector_flip_rate"]));rates=list(map(lambda x:float(F(x)),spec["bit_flip_rates"]))
                        state=0 if spec["within_calibrated_contract"] else int(rng.integers(0,2))
                        state=(state+int(rng.poisson(kappa*float(burn))))%2
                        for k,t in enumerate(lags):
                            counts[rep,k],state=sector_group(rng,state,n,float(H),float(t),kappa,rates)
                    else:
                        forward=float(F(spec["forward_rate"]));backward=float(F(spec["backward_rate"]))
                        state=(int(rng.poisson(forward*float(burn)))-int(rng.poisson(backward*float(burn))))%len(Q)
                        for k,t in enumerate(lags):
                            counts[rep,k],state=ring_group(rng,state,n,float(H),float(t),forward,backward,detector)
                empirical=counts@np.asarray(a,dtype=float)/n;lower=empirical-width
                result=dict(model=spec["name"],budget=B,block_duration=H,
                            within_calibrated_contract=spec["within_calibrated_contract"],
                            actual_additive_gap=actual_gap,true_contrast=target,
                            one_sided_coverage=_summary(lower<=target+1e-12),
                            equilibrium_exclusion=_summary(lower>0),
                            halfwidth=width,raw_counts=counts.tolist(),
                            counts_sha256=hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest())
                evaluations.append(result)
                print(json.dumps({k:result[k] for k in ("model","budget","block_duration","equilibrium_exclusion")}),flush=True)
    analytic=contract["analytic_design"];grid=analytic["block_duration_grid"]
    candidates=list(range(grid["start"],grid["stop_inclusive"]+1,grid["step"]))
    frontier=[]
    for B in analytic["normalized_total_budgets"]:
        for zeta in analytic["normalized_readout_cost_ratios"]:
            frontier.append(optimize_block_cost(B,burn,lam,a,alpha,F(zeta),candidates))
    result=dict(contract_sha256=hashlib.sha256(raw).hexdigest(),
                parent_contract_sha256=hashlib.sha256(parent_raw).hexdigest(),
                plans=plans,evaluations=evaluations,normalized_cost_frontier=frontier,
                scope=analytic["scope"],seed=seed)
    (root/"results/dependence_refinement.json").write_text(json.dumps(result,default=str,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    import sys
    run=refinement if "--refinement" in sys.argv else main
    run(Path(__file__).resolve().parents[1])
