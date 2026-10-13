"""Deterministic, standalone figure for the frozen observed-geometry study."""
from pathlib import Path
import argparse, json, os
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('manuscript/figures'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    phase=root/'geometry' if (root/'geometry').is_dir() else root
    report=json.loads((phase/'results/experiment.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    colors={'weighted':'#176b87','endpoint':'#b85c38'}
    names=['driven','weak_driven','equilibrium']
    labels=['Stronger drive','Weaker drive','Equilibrium']
    positions=np.arange(3)
    for k,method in enumerate(('weighted','endpoint')):
        offset=(k-.5)*.34
        rows=[report['scenario_summary'][name]['methods'][method] for name in names]
        values=[x['mean_rate_lower'] for x in rows]
        bars=axes[0].bar(positions+offset,values,width=.32,color=colors[method],
                        label='Local geometry' if method=='weighted' else 'Endpoint')
        for bar,value in zip(bars,values):
            axes[0].annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,value),
                             xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        counts=[x['positive_certificates'] for x in rows]
        bars=axes[1].bar(positions+offset,counts,width=.32,color=colors[method])
        for bar,value in zip(bars,counts):
            axes[1].annotate(str(value),(bar.get_x()+bar.get_width()/2,value),
                             xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set(ylabel='Mean certified entropy rate',ylim=(0,3.65),
                title='Stronger bound under stronger driving')
    axes[1].set(ylabel='Positive certificates out of 400',ylim=(0,455),
                title='No universal detection advantage')
    for ax in axes:
        ax.set_xticks(positions,labels)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',labelsize=9)
    axes[0].legend(frameon=False,loc='upper right')
    fig.suptitle('Same synthetic triples, distinct physical calibration requirements',fontsize=12)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'observed_geometry.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(args.output_dir/'observed_geometry.png',dpi=180,metadata={'Software':'Constructive Access 0.8.0'})
    plt.close(fig)

if __name__=='__main__':main()
