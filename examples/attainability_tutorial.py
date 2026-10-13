"""A complete declared-input example; illustrative counts are synthetic."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json
from constructive_access.attainability import joint_analog,spectral_dynamics,physical_minimum,design

def main(output):
    records=[4000,8000,8000];means=[F(3,4),F(69,100),F(59,100)]
    joint=joint_analog.certify_means(records,means,parameter=F(19,25),lag_unit=1)
    direct=spectral_dynamics.from_gap(joint['reversible_gap_lower'])
    chosen=design.certify_library(records,means,[F(k,16) for k in range(17)])
    mode=physical_minimum.sampled_mode(.5,.38,.18)
    ring=physical_minimum.ring_minimum(20,mode['gamma'],mode['principal_omega'])
    result=dict(evidence='Synthetic illustrative sufficient statistics; no physical experiment',
        fixed_joint=joint,direct_bridge=direct,simultaneous_library=chosen,
        single_mode_assumption=mode,physical_ring_realization=ring,
        interpretation='A confidence lower bound is not a physical upper bound. The circle value is a realizable value within a separately imposed architecture. Numerical model calculations are distinct from exact rational confidence certificates.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,default=str,indent=2,sort_keys=True)+'\n')
    print(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
