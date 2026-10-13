"""Exact feasible equilibrium explanations inside simulation confidence boxes."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import contrast
from constructive_access.frontier import energy_certificate as ec

def main():
    root=Path(__file__).resolve().parents[1];path=root/'results/contrast_refinement_counts.npz'
    arrays=np.load(path,allow_pickle=False);n=5000
    radius=contrast.sqrt_upper(ec.log_interval(F(120))[1]/(2*n))
    rows=[];witnesses=[]
    for case in ['driven','equilibrium','high_readout_noise','independent_clock_jitter']:
        count=0;simultaneous_direct=0;data=arrays[case+'__counts']
        for i,row in enumerate(data):
            boxes=[(max(F(0),F(int(s),n)-radius),min(F(1),F(int(s),n)+radius)) for s in row]
            w=contrast.reversible_box_witness([1,2,3],boxes)
            if w['status']=='feasible_equilibrium_point':
                assert contrast.verify_box_witness(w,[1,2,3],boxes);count+=1
                cert=contrast.certify_counts([1,2,3],[F(-4,9),F(4,3),F(-1)],[n]*3,[int(s) for s in row],[F(3,20),F(3,10),F(9,20)])
                if cert['contrast_lower']>0:simultaneous_direct+=1
                witnesses.append(dict(case=case,dataset=i,witness=w))
        rows.append(dict(case=case,datasets=len(data),verified_equilibrium_box_points=count,
                         direct_positive_despite_box_point=simultaneous_direct,
                         interpretation='Actual two-state equilibrium explanation at deterministic lags' if case!='independent_clock_jitter' else 'Point in the conservative moment comparison class; not a realization under the particular jitter law'))
    report={'evidence':'exact countermodels for supplied simultaneous boxes around simulated data',
            'counts_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'radius':str(radius),'cases':rows,
            'meaning':'A verified reversible point prevents any positive dissipation lower bound uniform over the entire box and the same comparison class. Direct-contrast confidence uses a different confidence region; there is no contradiction.',
            'search_inconclusive_is_not_infeasibility':True,'witnesses':witnesses}
    (root/'results/box_obstruction.json').write_text(json.dumps(contrast.jsonable(report),sort_keys=True,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
