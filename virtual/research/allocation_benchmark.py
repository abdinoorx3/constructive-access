"""Frozen equal-record allocation comparison with independent trajectory data."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
import numpy as np
from constructive_access.virtual import contrast,thermodynamic_lab as lab

def main():
    root=Path(__file__).resolve().parents[1]
    contract_path=root/'contracts/contrast_benchmark.json'
    contract=json.loads(contract_path.read_text())
    assert '330173' in contract['additional_prespecified_comparison']
    physical,_=lab.load_contract(root)
    detector=[float(F(x)) for x in physical['menus']['curvature']['detector']]
    quotas=[37500,75000,37500];times=[F(3,20),F(3,10),F(9,20)]
    counts=np.zeros((200,3),dtype=np.int32);flags=[];rates=[]
    for repetition in range(200):
        for k,(n,t) in enumerate(zip(quotas,times)):
            rng=np.random.default_rng(np.random.SeedSequence([330173,repetition,k]))
            rec=lab.ring_records(rng,n,float(t),20,1,detector)
            counts[repetition,k]=int(rec['same'].sum())
        cert=contrast.certify_counts([1,2,3],[-1,2,-1],quotas,[int(s) for s in counts[repetition]],times)
        flags.append(cert['contrast_lower']>0);rates.append(float(cert['entropy_production_rate_lower']))
    archive=root/'results/contrast_allocation_counts.npz';lab.deterministic_npz(archive,{'counts':counts})
    report={'evidence':'independent simulated ring paths with equal total records and expected evolution time',
            'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),'seed':330173,
            'records_per_setting':quotas,'records_per_dataset':sum(quotas),'datasets':200,
            'total_evolution_time':str(sum(n*t for n,t in zip(quotas,times))),
            'detections':sum(flags),'rate':sum(flags)/len(flags),
            'monte_carlo_95_interval':lab.clopper_pearson(sum(flags),len(flags)),
            'mean_entropy_rate_lower':float(np.mean(rates)),
            'counts_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'comparison_note':'Independent cohorts, not paired outcomes; fixed contrast(-1,2,-1) in both. Acquisition cost excludes uncalibrated reset/apparatus overhead. Integer allocation is exact optimum for this fixed-record continuous proxy and chosen divisible quotas, not a general optimal experiment.'}
    (root/'results/contrast_allocation.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
