"""Analytic calibration impossibility on a compact uniformly elliptic diffusion.

The numerical output evaluates proved bounds. It does not simulate trajectories
or certify a discretized hitting experiment. See RARE_BUMP_PROOF.txt.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def _finite(value, name):
    x=float(value)
    if not math.isfinite(x):raise ValueError(f"{name} must be finite")
    return x


def smooth_plateau(s: float) -> float:
    """C-infinity radial profile, one on [0,1/2], zero on [1,infinity)."""
    s=_finite(s,"s")
    if s<0:raise ValueError("s must be nonnegative")
    if s<=.5:return 1.0
    if s>=1:return 0.0
    # h(1-s)/(h(1-s)+h(s-1/2)), computed as a stable logistic.
    difference=1/(1-s)-1/(s-.5)
    if difference>=0:
        e=math.exp(-difference)
        return e/(1+e)
    return 1/(1+math.exp(difference))


def diffusivity(radial_distance: float, amplitude: float, radius: float) -> float:
    amplitude=_finite(amplitude,"amplitude")
    radius=_finite(radius,"radius")
    distance=_finite(radial_distance,"radial_distance")
    if amplitude<1 or not 0<radius<=.5 or distance<0:
        raise ValueError("require amplitude>=1, 0<radius<=1/2, distance>=0")
    return 1+(amplitude-1)*smooth_plateau(distance/radius)


def record_tv_upper(radius: float, total_time: float, records: int=1) -> float:
    """Bound for stationary records with lengths summing to total_time.

    Fresh stationary initializations account for the term proportional to
    records. The coupling applies even to full continuous hidden-state records.
    """
    radius=_finite(radius,"radius")
    total_time=_finite(total_time,"total_time")
    if not 0<radius<=.5 or total_time<0 or not isinstance(records,int) or isinstance(records,bool) or records<1:
        raise ValueError("require valid radius/time and positive integer records")
    return min(1.0,radius*(9*records/64+6*total_time))


def model_summary(amplitude: float, radius: float, total_time: float=1.0, records: int=1) -> dict:
    amplitude=_finite(amplitude,"amplitude")
    radius=_finite(radius,"radius")
    diffusivity(0,amplitude,radius)  # contract validation
    return {
        "amplitude":amplitude,"radius":radius,"entropy_rate":0.0,
        "common_ellipticity_lower_bound":1.0,
        "global_Gamma_supremum":amplitude,
        "global_Lf_supremum":amplitude,
        "global_L2f_supremum_at_least":amplitude**2,
        "full_continuous_record_tv_upper":record_tv_upper(radius,total_time,records),
        "stationary_mean_Gamma_lower":.5,
        "stationary_mean_Gamma_upper":.5+(amplitude-1)*radius**3/(12*math.pi**2),
        "total_observation_time":float(total_time),"stationary_records":records,
    }


def bounded_diffusion_first_generator(frequency: int, radius: float, total_time: float=1.0) -> dict:
    """D<=2 but ||Lf|| grows through divergence-form drift partial_x d."""
    if not isinstance(frequency,int) or isinstance(frequency,bool) or frequency<4 or frequency%4:
        raise ValueError("frequency must be a positive integer multiple of four")
    return {
        "frequency":frequency,"radius":float(radius),"entropy_rate":0.0,
        "diffusion_lower_bound":1.0,"diffusion_upper_bound":2.0,
        "global_Gamma_upper_bound":2.0,
        "global_Lf_supremum_at_least":frequency/2,
        "full_continuous_record_tv_upper_formula_value":record_tv_upper(radius,total_time),
    }


def bounded_diffusion_second_generator(radius: float, total_time: float=1.0) -> dict:
    """D<=2 and ||Lf||=2, but ||L²f|| diverges as the bump narrows."""
    base=model_summary(2,radius,total_time)
    return {
        "radius":float(radius),"entropy_rate":0.0,
        "diffusion_lower_bound":1.0,"diffusion_upper_bound":2.0,
        "global_Gamma_supremum":2.0,"global_Lf_supremum":2.0,
        "global_L2f_supremum_at_least":9/4+64/(3*radius*radius),
        "full_continuous_record_tv_upper_formula_value":base["full_continuous_record_tv_upper"],
    }


def report() -> dict:
    return {
        "status":"non-outward-rounded floating diagnostics for proved analytic formulas; no trajectory simulation",
        "numerical_caveat":"Values labelled upper/lower evaluate analytic bounds but are not interval certificates; rounding may erase tiny positive margins.",
        "model_class":"smooth compact T4, uniform stationary law, D>=I, ordinary reversal",
        "detector":"f(x,z)=cos(x), range [-1,1]",
        "target":"global or stationary essential suprema; not visited-path values",
        "examples":[model_summary(float(m),float(m)**-2) for m in (10,100,1000)],
        "bounded_diffusion_first_generator":[bounded_diffusion_first_generator(n,n**-2) for n in (4,40,400)],
        "bounded_diffusion_second_generator":[bounded_diffusion_second_generator(r) for r in (.1,.01,.001)],
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    payload=json.dumps(report(),indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:args.output.write_text(payload)
    else:print(payload,end="")
