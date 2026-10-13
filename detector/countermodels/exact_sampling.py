"""Exploratory detector-access countermodels; not part of observed_geometry release.

Known unit-width boxcar exposure and unit-grid sampling do not identify a
mixing continuous diffusion law. Countermodels are Euclidean Gaussian OU
oscillators, not compact bounded-detector diffusions.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad


def exposed_oscillator_moments(index: int) -> tuple[float,float]:
    """Unit-variance oscillator: variance V_m and lag coefficient F_m.

    Cov(Z_0,Z_k)=F_m*exp(-|k|) for every nonzero integer k.
    """
    if not isinstance(index,int) or isinstance(index,bool) or index<0:
        raise ValueError("index must be a nonnegative integer")
    x=(2*math.pi*index)**2
    e=math.exp(-1)
    variance=2*(e+(2-e)*x)/(1+x)**2
    lag_coefficient=2*(math.cosh(1)-1)*(1-x)/(1+x)**2
    return variance,lag_coefficient


def mixture_weights(high_index: int=2) -> tuple[float,float]:
    if not isinstance(high_index,int) or isinstance(high_index,bool) or high_index<2:
        raise ValueError("high_index must be an integer at least two")
    v0,f0=exposed_oscillator_moments(0)
    v1,f1=exposed_oscillator_moments(1)
    vn,fn=exposed_oscillator_moments(high_index)
    r0,r1,rn=f0/v0,f1/v1,fn/vn
    a0=v1*(r1-rn)/((r0-rn)*v0)
    an=v1*(r0-r1)/((r0-rn)*vn)
    return a0,an


def report(high_index: int=2) -> dict:
    a0,an=mixture_weights(high_index)
    target=np.array(exposed_oscillator_moments(1))
    matched=a0*np.array(exposed_oscillator_moments(0))+an*np.array(exposed_oscillator_moments(high_index))
    errors=[]
    for lag in range(21):
        c1=target[0] if lag==0 else target[1]*math.exp(-lag)
        c2=matched[0] if lag==0 else matched[1]*math.exp(-lag)
        errors.append(abs(c1-c2))
    return {
        "status":"exact analytic equivalence; floating-point values are diagnostics",
        "model_class":"stationary smooth mixing uniformly elliptic Euclidean OU; not compact",
        "known_exposure":1,"sampling_interval":1,
        "target_frequency":2*math.pi,"target_latent_variance":1,
        "mixture_frequencies":[0,2*math.pi*high_index],
        "mixture_variance_weights":[a0,an],
        "mixture_latent_variance":a0+an,
        "exposed_target_variance":target[0],"exposed_mixture_variance":matched[0],
        "exposed_target_lag_coefficient":target[1],"exposed_mixture_lag_coefficient":matched[1],
        "largest_covariance_numeric_residual_lags_0_to_20":max(errors),
        "target_ordinary_entropy":8*math.pi**2,
        "mixture_ordinary_entropy":8*math.pi**2*high_index**2,
        "all_sampled_finite_dimensional_laws_equal":True,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    payload=json.dumps({"fixed_sampling":[report(n) for n in (2,5,20)]},indent=2,sort_keys=True)+"\n"
    if args.output:args.output.write_text(payload)
    else:print(payload,end="")
