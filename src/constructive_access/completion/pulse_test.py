"""Predeclared three-arm test of the protected classical pulse-sign contract.

Uses independent Bernoulli records within/between arms; a shared baseline is
reused in the two contrasts. All design guarantees are exact rational checks.
No rejection identifies quantumness without the physical null assumptions.
"""
from fractions import Fraction as F
from math import ceil
import json
from pathlib import Path
try:
    from .quantum_completion import rational, log_interval, sqrt_interval, pulse_certificate
except ImportError:
    from quantum_completion import rational, log_interval, sqrt_interval, pulse_certificate


def _inputs(gap_early, gap_late, eta_baseline, eta_early, eta_late, alpha, beta):
    values = tuple(map(rational, (gap_early, gap_late, eta_baseline,
                                  eta_early, eta_late, alpha, beta)))
    g1, g2, e0, e1, e2, a, b = values
    if not 0 < g1 <= 1 or not 0 < g2 <= 1 or g1+g2>1:
        raise ValueError("strictly positive probability gaps with sum <= 1 required")
    if any(not 0 <= e <= 1 for e in (e0, e1, e2)):
        raise ValueError("calibration radii must lie in [0,1]")
    if not 0 < a < 1 or not 0 < b < 1:
        raise ValueError("alpha and beta must lie in (0,1)")
    return values


def plan(gap_early, gap_late, eta_baseline=0, eta_early=0, eta_late=0,
         alpha=F(1,20), beta=F(1,20)):
    """Sufficient records for early < baseline < late; orientation fixed in advance.

    eta's bound deviations from each arm's nominal Bernoulli law. Guarantees
    are conditional on a fixed valid calibration contract and independent records.
    Shared affine detector effects belong in the gaps, not automatically in eta.
    This is a conservative design, not a minimum-sample theorem.
    """
    g1,g2,e0,e1,e2,a,b = _inputs(gap_early,gap_late,eta_baseline,
                                eta_early,eta_late,alpha,beta)
    contract = dict(gap_early=g1,gap_late=g2,eta_baseline=e0,
                    eta_early=e1,eta_late=e2,alpha=a,beta=b)
    margin = min(g1-2*(e0+e1),g2-2*(e0+e2))
    if margin <= 0:
        return {"status":"calibration_limited_design", "contract":contract,
                "margin":margin}
    la=log_interval(1/a,256)[1]
    lb=log_interval(2/b,256)[1]
    sa=sqrt_interval(la,12)[1]
    sb=sqrt_interval(lb,12)[1]
    n=ceil((sa+sb)**2/margin**2)
    ha=margin*sa/(sa+sb)
    hb=margin*sb/(sa+sb)
    result={"status":"certified_sufficient_design", "contract":contract,
            "margin":margin, "records_per_arm":n, "total_records":3*n,
            "null_statistical_radius":ha,"power_statistical_radius":hb,
            "early_threshold":e0+e1+ha,"late_threshold":e0+e2+ha,
            "log_alpha_upper":la,"log_power_upper":lb,
            "orientation":"early < baseline < late"}
    if not verify_plan(result,**contract):
        raise ArithmeticError("internal design certificate failed")
    return result


def verify_plan(candidate, gap_early, gap_late, eta_baseline=0, eta_early=0,
                eta_late=0, alpha=F(1,20), beta=F(1,20)):
    """Check proof inequalities against a separately supplied exact contract."""
    try:
        g1,g2,e0,e1,e2,a,b = _inputs(gap_early,gap_late,eta_baseline,
                                    eta_early,eta_late,alpha,beta)
        expected=dict(gap_early=g1,gap_late=g2,eta_baseline=e0,
                      eta_early=e1,eta_late=e2,alpha=a,beta=b)
        if candidate.get("contract") != expected:
            return False
        if candidate.get("status") != "certified_sufficient_design":
            return False
        n=candidate["records_per_arm"]
        if isinstance(n,bool) or not isinstance(n,int) or n <= 0:
            return False
        ha=rational(candidate["null_statistical_radius"])
        hb=rational(candidate["power_statistical_radius"])
        margin=min(g1-2*(e0+e1),g2-2*(e0+e2))
        la=log_interval(1/a,256)[1]
        lb=log_interval(2/b,256)[1]
        return (ha>0 and hb>0 and ha+hb<=margin
                and n*ha*ha>=la and n*hb*hb>=lb
                and candidate["early_threshold"]==e0+e1+ha
                and candidate["late_threshold"]==e0+e2+ha
                and candidate["total_records"]==3*n
                and candidate["orientation"]=="early < baseline < late"
                and candidate["margin"]==margin
                and candidate["log_alpha_upper"]==la
                and candidate["log_power_upper"]==lb)
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return False


def decide(candidate, baseline_successes, early_successes, late_successes):
    """Return a fixed-design conclusion; synthetic data are not experimental data."""
    if not verify_plan(candidate,**candidate.get("contract",{})):
        raise ValueError("invalid design certificate")
    n=candidate["records_per_arm"]
    counts=(baseline_successes,early_successes,late_successes)
    if any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<=n for v in counts):
        raise ValueError("each arm requires an integer success count between 0 and n")
    c0,c1,c2=counts
    d1=F(c0-c1,n)
    d2=F(c2-c0,n)
    reject=d1>candidate["early_threshold"] and d2>candidate["late_threshold"]
    return {"conclusion":"reject_declared_classical_contract" if reject else "inconclusive",
            "early_contrast":d1,"late_contrast":d2,
            "false_rejection_bound":candidate["contract"]["alpha"],
            "physical_assumptions_required":True}


def example():
    early=pulse_certificate(F(1,10),1,F(1,10),F(1,2))
    late=pulse_certificate(F(1,10),1,F(1,10),F(9,4))
    gain=F(9,10)
    assert -gain*early["response_upper"]>F(3,100)
    assert gain*late["response_lower"]>F(13,1000)
    design=plan(F(3,100),F(13,1000),F(1,2000),F(1,2000),F(1,2000))
    n=design["records_per_arm"]
    # Explicit synthetic mean counts, useful only to demonstrate the decision API.
    p0=F(1,2)
    p1=p0+gain*(early["response_lower"]+early["response_upper"])/2
    p2=p0+gain*(late["response_lower"]+late["response_upper"])/2
    counts=tuple(round(n*p) for p in (p0,p1,p2))
    return {"evidence":"analytical design; synthetic mean-count demonstration only",
            "design":design,"detector_gain":gain,"detector_false_positive_offset":F(1,20),
            "times":[F(1,2),F(9,4)],
            "active_pulse_time":n*F(11,4),
            "energy_note":"add calibrated pulse power and preparation/readout energy for all 3*n records",
            "synthetic_counts":counts,"synthetic_decision":decide(design,*counts),
            "statistical_assumptions":"fixed predeclared orientation; independent Bernoulli records; fixed valid calibration envelopes",
            "calibration_failure_note":"if envelopes fail with probability delta, add delta once to the unconditional bound"}


if __name__=="__main__":
    target=Path(__file__).resolve().parents[1]/"results"/"pulse_test.json"
    target.write_text(json.dumps(example(),default=str,indent=2)+"\n")
    print("wrote",target)
