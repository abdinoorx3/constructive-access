"""Finite-lag scalar correlation defects bound stationary dissipation.

The population theorem concerns a stationary, time-homogeneous Markov process
with ordinary (even-variable) time reversal. Its statistical use requires a
separately valid confidence bound on the supplied contrast. Exact rational
arithmetic certifies the numerical bridge; it does not validate an apparatus.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from constructive_access.frontier.energy_certificate import exact


def phi_interval(u, terms=24):
    """Enclose 2 sqrt(u) atanh(sqrt(u)) by a positive rational series.

    Valid for 0 <= u < 1. The finite sum is a certified lower bound, and
    bounding every remaining denominator by its first value gives the upper.
    """
    u = exact(u)
    if not 0 <= u < 1:
        raise ValueError("The finite-entropy nonlinear domain is 0 <= u < 1")
    if type(terms) is not int or not 1 <= terms <= 512:
        raise ValueError("Use 1..512 series terms")
    lower = 2 * sum((u**k / (2*k-1) for k in range(1, terms+1)), F(0))
    upper = lower + 2*u**(terms+1)/((2*terms+1)*(1-u))
    return lower, upper


def from_gap(gap_lower, lag_unit=1, range_bound=2, terms=24):
    """Apply the physical bridge to a covered joint probability-score gap.

    The clipped positive input must obey 2[g]+ <= [-s^2 C(0)+2s C(t)-C(2t)]+
    on an already established event. Such an event is provided by binary twins,
    or bounded analog squared-readout calibration, with a fixed/predictable
    admissible design. This function neither creates nor spends an alpha.

    If u >= 1, finite-entropy compatibility fails *on that coverage event*.
    The result flags this case, retains the universally valid linear bound,
    and does not silently clip u or infer that the data falsify the model.
    """
    g, t, width = map(exact, (gap_lower, lag_unit, range_bound))
    if t <= 0 or width <= 0:
        raise ValueError("Positive physical lag and valid positive range bound required")
    if type(terms) is not int or not 1 <= terms <= 512:
        raise ValueError("Use 1..512 series terms")
    g = max(F(0), g)
    u = 8*g/(width*width)
    linear = 2*u/t
    if u >= 1:
        return dict(status="incompatible_with_finite_entropy_on_coverage_event",
            gap_lower=g, normalized_defect=u, lag_unit=t, range_bound=width,
            linear_rate_lower=linear, entropy_production_rate_lower=linear,
            nonlinear_formula_interval=None, series_terms=terms,
            warning="Coverage can fail at its declared probability; the flag alone is not an empirical falsification. No clipping or numerical infinity is used.")
    lo, hi = phi_interval(u, terms)
    return dict(status="positive_certificate" if g else "no_positive_certificate",
        gap_lower=g, normalized_defect=u, lag_unit=t, range_bound=width,
        linear_rate_lower=linear, entropy_production_rate_lower=lo/t,
        nonlinear_formula_interval=[lo/t, hi/t], series_terms=terms,
        interpretation="The interval encloses the nonlinear lower-bound formula, not the actual entropy production rate. Statistical coverage and physical model assumptions are inherited from the supplied gap.")


def population_gap(correlations, parameter):
    """Exact Δ_s and the corresponding probability-score gap Δ_s/2.

    s can be any real rational in the population identity. This function does
    not authorize selecting s after seeing inference data without correction.
    """
    if len(correlations) != 3:
        raise ValueError("C(0), C(t), C(2t) are required")
    c0, c1, c2 = map(exact, correlations)
    s = exact(parameter)
    if c0 < 0:
        raise ValueError("The zero-lag second moment must be nonnegative")
    defect = -s*s*c0 + 2*s*c1-c2
    return dict(parameter=s, correlations=[c0,c1,c2], defect=defect,
                probability_score_gap=defect/2)


def inspect_model(generator, stationary, detector, lag, parameter):
    """Floating-point diagnostic of the proof chain, not an exact certificate."""
    import numpy as np
    from scipy.linalg import expm
    from scipy.special import xlogy
    Q=np.asarray(generator,dtype=float);pi=np.asarray(stationary,dtype=float)
    f=np.asarray(detector,dtype=float);n=len(pi)
    if f.ndim==1:f=f[:,None]
    if Q.shape!=(n,n) or f.ndim!=2 or f.shape[0]!=n or np.any(pi<=0) or not np.isclose(pi.sum(),1) or not all(np.all(np.isfinite(x)) for x in [Q,pi,f]):
        raise ValueError("Invalid finite state-space inputs")
    off=Q-np.diag(np.diag(Q))
    if np.min(off)<-1e-13 or not np.allclose(Q.sum(axis=1),0,atol=1e-11) or not np.allclose(pi@Q,0,atol=1e-11):
        raise ValueError("A valid stationary row generator is required")
    if lag<=0 or not np.isfinite(lag) or not np.isfinite(parameter):raise ValueError("Invalid lag or parameter")
    P=expm(lag*Q);reverse=P.T*pi[None,:]/pi[:,None]
    H=(P+reverse)/2;B=(P-reverse)/2
    C=np.array([np.sum(pi[:,None]*f*(np.linalg.matrix_power(P,k)@f)) for k in range(3)])
    defect=-parameter**2*C[0]+2*parameter*C[1]-C[2]
    skew_norm=float(np.sum(pi[:,None]*(B@f)**2))
    residual=float(np.sum(pi[:,None]*((H-parameter*np.eye(n))@f)**2))
    tv=np.abs(P-reverse).sum(axis=1)/2
    ratio=np.ones_like(P)
    np.divide(P,reverse,out=ratio,where=P>0)
    endpoint=float(np.sum(pi[:,None]*xlogy(P,ratio)))
    flux=pi[:,None]*off;reverse_flux=flux.T
    active=(flux+reverse_flux)>0
    if np.any((flux==0)&active):sigma=float("inf")
    else:sigma=float(np.sum(xlogy(flux[active],flux[active]/reverse_flux[active])))
    width=float(np.max(np.linalg.norm(f[:,None,:]-f[None,:,:],axis=2)))
    u=0. if width==0 else max(0.,4*defect/width**2)
    nonlinear=0. if u==0 else (float("inf") if u>=1 else 2*np.sqrt(u)*np.arctanh(np.sqrt(u))/lag)
    return dict(correlations=C.tolist(),defect=float(defect),skew_norm=skew_norm,
        symmetric_residual=residual,identity_residual=float(defect-skew_norm+residual),
        detector_range=width,average_row_tv_squared=float(np.dot(pi,tv*tv)),
        endpoint_forward_reverse_kl=endpoint,path_forward_reverse_kl=lag*sigma,
        entropy_rate=sigma,nonlinear_rate_lower=nonlinear,
        linear_rate_lower=0. if width==0 else max(0.,8*defect/(width*width*lag)))


def _json(value):
    if isinstance(value,F):return str(value)
    raise TypeError(type(value).__name__)


def replay_sixth_joint(data_dir, output=None):
    """Reapply a new proved bridge to all frozen sixth-edition joint records.

    This is a retrospective deterministic reanalysis, with no new samples,
    retuning, altered confidence region or additional measurement expenditure.
    """
    import numpy as np
    from constructive_access.calibrated import joint_calibration as old
    data_dir=Path(data_dir);npz=data_dir/"joint_calibration_counts.npz"
    saved=data_dir/"joint_calibration.json"
    original=json.loads(saved.read_text());records=np.load(npz,allow_pickle=False)
    results=[];largest_replay_error=0.;covered_violation=0
    for case in original["cases"]:
        name=case["case"];counts=records[name+"__science_counts"]
        twins=records[name+"__calibration_counts"]
        old_rates=records[name+"__joint_calibration_pinsker_rates"]
        linear=[];nonlinear=[];examples=[];positive=0;violations=0
        for i,(ct,tw) in enumerate(zip(counts,twins)):
            certificate=old.certify_counts([4000,8000,8000],
                [int(tw[0]+tw[3]),int(ct[0]),int(ct[1])])
            largest_replay_error=max(largest_replay_error,abs(float(certificate["entropy_production_rate_lower"])-old_rates[i]))
            bridge=from_gap(certificate["reversible_gap_lower"])
            linear.append(float(bridge["linear_rate_lower"]))
            nonlinear.append(float(bridge["entropy_production_rate_lower"]))
            positive+=int(bridge["entropy_production_rate_lower"]>0)
            sigma=case["oracle"]["stationary_entropy_rate"]
            violations+=int(nonlinear[-1]>sigma+1e-12)
            if i<2:examples.append(dict(record=i,old_certificate=certificate,bridge=bridge))
        oldmean=float(np.mean(old_rates));newmean=float(np.mean(nonlinear))
        results.append(dict(case=name,contract=case["contract"],datasets=len(counts),
            positive_certificates=positive,old_positive_certificates=int(np.sum(old_rates>0)),
            old_mean_rate_lower=oldmean,linear_mean_rate_lower=float(np.mean(linear)),
            nonlinear_mean_rate_lower=newmean,nonlinear_to_old_mean_ratio=None if oldmean==0 else newmean/oldmean,
            actual_model_entropy_rate=case["oracle"]["stationary_entropy_rate"],
            empirical_coverage_failures=violations,
            exact_examples=examples,linear_rates=linear,nonlinear_rates=nonlinear))
    result=dict(scope="Retrospective reanalysis of the frozen sixth-edition simulated counts; no physical measurements, samples or confidence events added. Invalid calibration control remains invalid.",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [npz,saved]},
        source_costs=original["costs"],max_old_rate_replay_error=largest_replay_error,cases=results)
    if output is not None:Path(output).write_text(json.dumps(result,default=_json,indent=2)+"\n")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sixth-results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=replay_sixth_joint(args.sixth_results,args.output)
    print(json.dumps({"cases":[{k:c[k] for k in ["case","positive_certificates","old_mean_rate_lower","linear_mean_rate_lower","nonlinear_mean_rate_lower","nonlinear_to_old_mean_ratio"]} for c in result["cases"]],"replay_error":result["max_old_rate_replay_error"]},indent=2))
