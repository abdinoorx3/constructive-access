"""Installed APIs: exact assay design and synthetic classical-explanation bounds."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
from constructive_access.robust import preparation,statistics_design,explanation,clock_design


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/robust_tutorial.json'))
    args=parser.parse_args()
    lo,hi=preparation.depolarizing_contrast_bounds(F(1,10),F(2),F(9,4),F(9,10))
    assert hi<-F(18,125)
    contract=dict(gap=F(18,125),radius=F(1,1000))
    fixed=statistics_design.exact_fixed_plan(**contract)
    assert statistics_design.verify_exact_fixed(fixed,**contract)
    sequential=statistics_design.exact_anytime_power_plan(**contract)
    assert statistics_design.verify_exact_anytime(sequential,**contract)
    witness=preparation.baseline_subtracted_witness_bounds(
        F(1,10),F(2),F(2),F(-12,37),F(35,37),F(9,10))
    assert witness[0]>F(27,25)
    dimension=statistics_design.dimension_plan(F(27,25),F(1,500),6000,4560)
    assert statistics_design.verify_dimension(dimension,F(27,25),F(1,500))
    inferred=explanation.from_counts(600,343,1,1,F(9,4),F(9,10),F(1,1000))
    output={'evidence':'Exact model and design certificates; 343/600 is explicitly synthetic',
            'certified_gap_lower':F(18,125),'fixed_plan':fixed,'anytime_plan':sequential,
            'dimension_witness_interval':witness,'dimension_plan':dimension,
            'synthetic_explanation':inferred,'clock_design':clock_design.make_results()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,default=str,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'fixed_records':fixed['records'],
                      'anytime_sufficient_records':sequential['records'],
                      'dimension_records':dimension['records'],
                      'synthetic_reset_defect_lower':float(inferred['reset_explanation']['minimum_reset_defect'])}))


if __name__=='__main__':main()
