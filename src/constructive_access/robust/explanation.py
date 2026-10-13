"""Turn randomized-preparation counts into conditional classical explanation bounds.

This module does not label a system quantum. It quantifies the reset mismatch
or peak macro-crossing intensity needed by a declared classical explanation.
The initial reference lies inside B and comparison lies outside B; a coherent
comparison preparation with nonzero initial B occupancy needs another contract.
"""
from fractions import Fraction as F
from pathlib import Path
import json
try:
    from . import detector_boundary as boundary
    from . import statistics_design as statistics
except ImportError:
    import detector_boundary as boundary
    import statistics_design as statistics


def from_counts(records,successes,entrance_cap,exit_cap,time,gain=1,radius=0,
                alpha=F(1,20),bits=24):
    """Fixed-horizon confidence bound, conditional on the supplied physical contract.

    Encoded success is Y in comparison arm and 1-Y in reference arm, with a
    fresh fair arm coin and a common independent Uniform[0,time] clock.
    radius bounds additional observed-contrast discrepancy after common gain.
    It is not a confidence sequence: choose record count in advance.
    """
    a,b,t,g,eta,alpha=map(boundary.rational,(entrance_cap,exit_cap,time,gain,radius,alpha))
    if a<0 or b<0 or t<=0 or not 0<g<=1 or not 0<=eta<=1:
        raise ValueError('Nonnegative caps, positive time/gain<=1, and radius in [0,1] required')
    inference=statistics.fixed_lower_bound(records,successes,alpha,bits)
    observed_upper=1-2*inference['success_probability_lower']
    physical_upper=(observed_upper+eta)/g
    reset=boundary.reset_defect_lower_bound(t*physical_upper,a,b,t)
    hazard=boundary.classical_hazard_lower_bound(physical_upper,t)
    return {'records':records,'successes':successes,'alpha':alpha,
            'contract':dict(entrance_cap=a,exit_cap=b,time=t,gain=g,radius=eta,bits=bits),
            'statistics':inference,'observed_contrast_upper':observed_upper,
            'physical_contrast_upper':physical_upper,
            'reset_explanation':reset,'required_peak_crossing_intensity':hazard,
            'scope':'Fixed record count; reference supported inside B, comparison outside B; independent common uniform clock and fair arm draw; calibrated common positive detector gain and contrast envelope. Reset-defect bound additionally needs the declared Markov/regenerative interface. Peak-intensity bound does not identify total dynamical activity or entropy production.'}


def example():
    # These counts are illustrative model-mean rounded counts, not measured data.
    outputs=[]
    for cap in (F(1,2),F(1),F(2)):
        outputs.append(from_counts(600,343,cap,cap,F(9,4),F(9,10),F(1,1000)))
    return {'evidence':'Explicitly synthetic 343 encoded successes out of 600; no empirical observations',
            'examples':outputs,
            'interpretation':'Confidence bounds are conditional on calibration and preparation/clock assumptions. Failing a classical explanation contract is not device-independent quantumness.'}


if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'results/explanation.json'
    p.write_text(json.dumps(example(),default=str,indent=2)+'\n')
    print('wrote',p)
