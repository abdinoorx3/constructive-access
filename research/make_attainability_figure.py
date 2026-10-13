"""Deterministic standalone figure from retained results, never invented data."""
from pathlib import Path
import argparse,json
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root,output):
    # Script is located in release/research or development/research.
    phase=root/'attainability' if (root/'attainability').exists() else root
    replay=json.loads((phase/'results/spectral_dynamics_replay.json').read_text())
    a=json.loads((phase/'results/joint_analog.json').read_text())
    d=json.loads((phase/'results/design.json').read_text())
    r=next(c for c in replay['cases'] if c['case']=='driven')
    c=next(c for c in a['summaries'] if c['case']=='driven')
    grid=next(c for c in d['summaries'] if c['case']=='driven')
    # Physical construction is evaluated from the included analytical formula.
    from constructive_access.attainability.physical_minimum import sampled_mode,ring_minimum
    mode=sampled_mode(.5,.38,.18)
    ns=np.arange(16,101)
    e=[ring_minimum(int(n),mode['gamma'],mode['principal_omega'])['entropy'] for n in ns]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    vals=[r['old_mean_rate_lower'],r['nonlinear_mean_rate_lower']]
    ax[0].bar(['Previous','Direct'],vals,color=['#8b9aa5','#167d9a'],width=.55)
    ax[0].set_yscale('log');ax[0].axhline(r['actual_model_entropy_rate'],color='#bd6843',ls='--',label='Simulated true rate')
    ax[0].set_ylim(.0003,10);ax[0].set_ylabel('Mean lower bound (nats / time)')
    ax[0].set_title('Same data: 140× stronger bound');ax[0].legend(fontsize=7,loc='upper left')
    for i,v in enumerate(vals):ax[0].text(i,v*1.18,f'{v:.5g}',ha='center',fontsize=8)
    names=['Encoded\nsplit','Analog\nsplit','Analog\njoint','Analog\nlibrary']
    counts=[c['methods'][k]['positive_certificates'] for k in ['split_coin','split_analog','joint_analog']]+[grid['positive_certificates']]
    ax[1].bar(names,counts,color=['#bbc3c7','#8b9aa5','#167d9a','#41705c'],width=.7)
    ax[1].set_ylim(0,435);ax[1].set_ylabel('Positive certificates / 400')
    ax[1].set_title('Same acquisition resources')
    for i,v in enumerate(counts):ax[1].text(i,v+8,str(v),ha='center')
    ax[2].plot(ns,e,color='#167d9a',lw=2,label='Finite ring minimum')
    ax[2].axhline(mode['circle_entropy_minimum'],color='#bd6843',ls='--',label='Circle diffusion')
    ax[2].set_xlabel('Number of ring states');ax[2].set_ylabel('Entropy rate (nats / time)')
    ax[2].set_title('Physical minimum within an architecture');ax[2].legend(fontsize=7)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'attainability.png',dpi=180,metadata={'Software':'Constructive Access'})
    stamp=datetime(2026,10,12,tzinfo=timezone.utc)
    fig.savefig(output/'attainability.pdf',metadata={'Title':'Physical attainability and direct inference','Author':'Abdinoor, 2D Labs','CreationDate':stamp,'ModDate':stamp})
    plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.root,a.output_dir)
