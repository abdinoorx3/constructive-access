"""Small deterministic lag sensitivity study, without sampled observations."""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import exposure_moments as em
from population_pilot import population_statistics,prospective_diagnostic


def main():
    output=Path(__file__).resolve().parent/'population_lag_sweep.json'
    cases=[]
    for h in (F(1,40),F(3,100),F(7,200),F(1,25),F(1,20)):
        for m in (0,1,2):
            law=em.outcome_law(em.Model(1,3,h,F(1,100000),2*m+2,F(9,10)),collapse_centers=True)
            stats,_=population_statistics(law,m,h)
            diagnostics=[prospective_diagnostic(stats,m,h,n,nref) for nref in (100000,1000000)
                         for n in (2000000,10000000,50000000)]
            case=dict(lag=str(h),order=m,law_tv_upper=float(law['total_variation_upper']),
                      population_score_means={k:float(v['mean']) for k,v in stats.items()},
                      population_score_sd={k:math.sqrt(float(v['second']-v['mean']**2)) for k,v in stats.items()},
                      design_diagnostics=diagnostics)
            cases.append(case)
            print(f'h={h} m={m}: '+', '.join(f"n={d['records']},ref={d['reference_records_each']}: {d['population_plugin_geometry_lower']:.4g}" for d in diagnostics),flush=True)
    output.write_text(json.dumps(dict(status='deterministic_unfrozen_design_diagnostics',
        warning='Population plug-in values are neither confidence certificates nor power estimates; no science/reference records sampled.',
        fixed_parameters=dict(gamma=1,omega=3,tau='1/100000',gain='9/10',caps=[4,4,13,13],lambda_=1),
        cases=cases),indent=2)+'\n')


if __name__=='__main__':main()
