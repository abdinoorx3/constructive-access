"""Joint confidence for lag-zero twins and positive-lag reversible moments.

The quadratic moment inequality and concentration ingredients are established.
This module integrates lag-zero calibration into one fixed contrast, including
its sampling variance while assigning it zero physical comparison duration.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper
from constructive_access.virtual.thermodynamic_lab import deterministic_npz,ring_records
try:
    from . import nonlinear,calibrated_benchmark as benchmark
except ImportError:
    import nonlinear
    import calibrated_benchmark as benchmark


def support(parameter):
    s=ec.exact(parameter)
    if not 0<=s<=1:
        raise ValueError("Choose a fixed rational parameter in [0,1]")
    return dict(parameter=s,coefficients=[-s*s,2*s,F(-1)],
                reversible_support=-(1-s)**2/2,
                squared_polynomial=[s*s,-2*s,F(1)],
                equality_example="Mean-zero reversible one-mode detector with decay x=s; fair state-independent detector also attains equality")


def certify_counts(records,successes,parameter=F(19,25),lag_unit=1,
                   alpha=F(1,20),calibration_radii=None,external_failure=0,
                   selection="predeclared",
                   twin_contract="independent_same_state_same_detector",
                   record_contract="independent_stationary_pairs"):
    """Exact one-sided certificate for fixed [lag0,lag1,lag2] pair quotas.

    Metadata declares assumptions rather than empirically verifying them.
    A zero-lag calibration radius bounds probability error relative to the
    ideal same-state same-channel twin law. It is not a free detector change.
    """
    if twin_contract!="independent_same_state_same_detector":
        raise ValueError("Same-state independent twin noise and unchanged detector required")
    if record_contract!="independent_stationary_pairs":
        raise ValueError("This API requires independently prepared stationary pairs")
    delta,eta,alpha=map(ec.exact,(lag_unit,external_failure,alpha))
    if delta<=0 or not 0<=eta<1 or not 0<alpha<1 or alpha+eta>=1:
        raise ValueError("Invalid positive time unit or failure budget")
    if len(records)!=3 or len(successes)!=3:
        raise ValueError("Three count groups ordered at lags0,1,2 required")
    sup=support(parameter);a=sup["coefficients"]
    region=nonlinear.confidence_region(a,records,successes,alpha,
                conditional_radii=calibration_radii,selection=selection)
    gap=max(F(0),region["lower"]-sup["reversible_support"])
    denominator=(2*sup["parameter"]*sqrt_upper(delta)+sqrt_upper(2*delta))**2
    return dict(status="positive_certificate" if gap else "no_positive_certificate",
                support=sup,confidence_region=region,
                reversible_gap_lower=gap,comparison_time_denominator_upper=denominator,
                entropy_production_rate_lower=8*gap*gap/denominator,
                failure_upper=alpha+eta,external_contract_failure_upper=eta,
                lags=[F(0),delta,2*delta],pair_records=sum(records),
                endpoint_readouts=2*sum(records),
                declared_evolution_time=delta*records[1]+2*delta*records[2],
                twin_contract=twin_contract,record_contract=record_contract,
                interpretation="Lag0 contributes finite-data variance and detector operations; its Q-versus-additive-reversible divergence is exactly zero only under the common ideal twin contract. Entropy interpretation requires the ordinary stationary Markov thermodynamic assumptions.")


def population(parameter,probabilities):
    sup=support(parameter)
    p=list(map(ec.exact,probabilities))
    if len(p)!=3 or any(not 0<=x<=1 for x in p):
        raise ValueError("Three valid exact probabilities required")
    contrast=sum((c*x for c,x in zip(sup["coefficients"],p)),F(0))
    return dict(**sup,probabilities=p,contrast=contrast,
                reversible_gap=contrast-sup["reversible_support"])


def lag_zero_uncertainty_control():
    """Exact negative control: excluding lag0 variance causes false evidence."""
    s=F(1,2);p=[F(3,4),F(5,8),F(9,16)];a=support(s)["coefficients"]
    fake=sum((c*x for c,x in zip(a,[F(0),p[1],p[2]])),F(0))
    return dict(parameter=s,true_reversible_probabilities=p,
                true_population_gap=population(s,p)["reversible_gap"],
                one_lag_zero_record_failure_probability=1-p[0],
                false_gap_with_exact_science_and_zero_lag_success_zero=fake-support(s)["reversible_support"],
                explanation="With one lag0 pair, agreement is zero with probability1/4. Treating that estimate as exact produces false gap3/16 despite a reversible model. Lag0 uncertainty must be retained.")


def _dataset(task):
    """Independent seeded dataset; parallel execution changes no random stream."""
    import numpy as np
    seed,case_index,rep,case,spec,joint=task
    forward,backward,detector=benchmark.model(case)
    counts=np.zeros(3,dtype=np.int32);rawrecords=[]
    for k,(lag,n) in enumerate(zip(spec["science_lags"],spec["science_records"])):
        rng=np.random.default_rng(np.random.SeedSequence([seed,case_index,rep,k]))
        record=ring_records(rng,n,lag,forward,backward,detector,gain=case["gain"])
        counts[k]=int(record["same"].sum())
        if case_index==0 and rep==0:
            record["setting"]=np.full(n,k,dtype=np.uint8);rawrecords.append(record)
    rng=np.random.default_rng(np.random.SeedSequence([seed,case_index,rep,3]))
    twins,rawtwin=benchmark.calibration_records(rng,4000,detector,case["calibration_gain"])
    if case_index==0 and rep==0:
        rawtwin["setting"]=np.full(4000,3,dtype=np.uint8);rawrecords.append(rawtwin)
    jc=certify_counts(joint["records"],[int(twins[0]+twins[3]),int(counts[0]),int(counts[1])],
                      parameter=F(joint["parameter_s"]),alpha=F(joint["alpha"]))
    old=benchmark.analyze(counts,twins,spec)
    return counts,twins,jc,old,rawrecords


def run(root,workers=4):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    if type(workers) is not int or not 1<=workers<=32:
        raise ValueError("Use1..32 deterministic dataset workers")
    path=root/"contracts/joint_calibration.json";raw=path.read_bytes();contract=json.loads(raw)
    if not contract["frozen_before_heldout_generation"]:
        raise ValueError("Freeze before held-out generation")
    spec=benchmark.specification();joint=contract["joint"]
    baseline_raw=(root/"contracts/calibrated_benchmark.json").read_bytes()
    if spec!=json.loads(baseline_raw):
        raise ValueError("Baseline implementation specification differs from its saved frozen contract")
    comparison_keys=("name","rates","gain","calibration_gain")
    if [{k:c[k] for k in comparison_keys} for c in contract["cases"]]!=[{k:c[k] for k in comparison_keys} for c in spec["cases"]]:
        raise ValueError("Baseline/model cases changed after the joint contract")
    nsets=contract["datasets_per_case"];seed=contract["seed"]
    names=["joint_calibration_pinsker","two_stage_pinsker","two_stage_nonlinear",
           "optimized_extra_lag","legacy_extra_lag"]
    archive={};cases=[]
    for case_index,case in enumerate(contract["cases"]):
        forward,backward,detector=benchmark.model(case);truth=benchmark.oracle(case)
        target_p0=(1+truth["A"])/2
        target_joint=sum(float(F(c))*p for c,p in zip(joint["coefficients"],[target_p0,*truth["probabilities"][:2]]))
        counts=np.zeros((nsets,3),dtype=np.int32);twins=np.zeros((nsets,4),dtype=np.int32)
        rates={name:[] for name in names};coverage=[];calcover=[];twostagecover=[];examples=[]
        optimized=[];relaxationgaps=[]
        tasks=[(seed,case_index,rep,case,spec,joint) for rep in range(nsets)]
        context=ProcessPoolExecutor(max_workers=workers) if workers>1 else nullcontext(None)
        with context as pool:
          stream=pool.map(_dataset,tasks,chunksize=1) if pool else map(_dataset,tasks)
          for rep,(science_row,twin_row,jc,old,rawrecords) in enumerate(stream):
            counts[rep]=science_row;twins[rep]=twin_row
            if rawrecords:
                deterministic_npz(root/"results/joint_calibration_raw.npz",
                    {key:np.concatenate([r[key] for r in rawrecords]) for key in rawrecords[0]})
            analyzed=old["status"]=="analyzed"
            values=dict(joint_calibration_pinsker=jc["entropy_production_rate_lower"],
                two_stage_pinsker=old["calibrated_pinsker_rate"] if analyzed else F(0),
                two_stage_nonlinear=old["nonlinear"]["certificate"]["entropy_production_rate_lower"] if analyzed else F(0),
                optimized_extra_lag=old["alternative"]["entropy_production_rate_lower"],
                legacy_extra_lag=old["legacy_alternative"]["entropy_production_rate_lower"])
            for name in names:rates[name].append(float(values[name]))
            coverage.append(float(jc["confidence_region"]["lower"])<=target_joint)
            cal=old["calibration"]
            ml,mh=map(float,cal["mean_interval"]);al,ah=map(float,cal["second_moment_interval"])
            cc=analyzed and ml<=truth["mean"]<=mh and al<=truth["A"]<=ah
            true_science=sum(float(F(c))*p for c,p in zip(spec["calibrated"]["coefficients"],truth["probabilities"]))
            calcover.append(cc);twostagecover.append(cc and float(old["region"]["lower"])<=true_science)
            if analyzed and old["nonlinear"]["certified_relaxation_gap_upper"] is not None:
                optimized.append(bool(old["nonlinear"]["proposal_success"]))
                relaxationgaps.append(float(old["nonlinear"]["certified_relaxation_gap_upper"]))
            if rep<2:examples.append(dict(joint=jc,two_stage_and_extra=old))
            if (rep+1)%100==0:print(f"{case['name']}: {rep+1}/{nsets}",flush=True)
        archive[case["name"]+"__science_counts"]=counts
        archive[case["name"]+"__calibration_counts"]=twins
        for name,values in rates.items():archive[case["name"]+"__"+name+"_rates"]=np.asarray(values)
        differences={}
        for name in names[1:]:
            diff=np.asarray(rates[names[0]])-np.asarray(rates[name])
            differences[name]=dict(mean=float(diff.mean()),minimum=float(diff.min()),maximum=float(diff.max()),
                strictly_positive=int(np.sum(diff>1e-15)),strictly_negative=int(np.sum(diff < -1e-15)))
        cases.append(dict(case=case["name"],contract=case["contract"],oracle=truth,
            target_zero_lag_probability=target_p0,
            inference={name:benchmark.summary(np.asarray(values)>0) for name,values in rates.items()},
            rate_statistics={name:dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=float(np.min(values)),maximum=float(np.max(values)),
                mean_fraction_of_true_entropy=None if truth["stationary_entropy_rate"]==0 else float(np.mean(values)/truth["stationary_entropy_rate"])) for name,values in rates.items()},
            joint_minus_baseline=differences,
            joint_contrast_coverage=benchmark.summary(coverage),two_stage_calibration_coverage=benchmark.summary(calcover),
            two_stage_joint_region_coverage=benchmark.summary(twostagecover),
            nonlinear_proposal_successes=int(sum(optimized)),nonlinear_optimized_datasets=len(optimized),
            nonlinear_max_certified_relaxation_gap=None if not relaxationgaps else max(relaxationgaps),
            illustrative_exact_certificates=examples))
        print(json.dumps(dict(case=case["name"],detections={name:sum(np.asarray(values)>0).item() for name,values in rates.items()})),flush=True)
    counts_path=root/"results/joint_calibration_counts.npz";deterministic_npz(counts_path,archive)
    report=dict(contract_sha256=hashlib.sha256(raw).hexdigest(),
        baseline_contract_sha256=hashlib.sha256(baseline_raw).hexdigest(),seed=seed,cases=cases,
        costs=contract["costs"],scope=contract["scope"],
        exact_population_example=population(F(19,25),[F(3,4),F(69,100),F(59,100)]),
        lag_zero_uncertainty_control=lag_zero_uncertainty_control(),
        counts_archive=counts_path.name,counts_sha256=hashlib.sha256(counts_path.read_bytes()).hexdigest(),
        raw_archive="joint_calibration_raw.npz",raw_sha256=hashlib.sha256((root/"results/joint_calibration_raw.npz").read_bytes()).hexdigest())
    (root/"results/joint_calibration.json").write_text(json.dumps(report,default=str,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();run(args.root,args.workers)
