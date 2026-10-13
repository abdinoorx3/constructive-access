"""Plot deterministic analytical count bounds and an exact finite comparison.

Usage: python design/figure.py
Optional --input, --review, and --output-prefix relocate all I/O.  No simulation
is performed. The left panel displays high-precision numerical evaluations;
the right panel uses the independent review's exact-rational certificate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PASSIVE='#2B659C'
PREPARED='#19816C'


def make_figure(input_path,review_path,output_prefix):
    data=json.loads(Path(input_path).read_text())
    review=json.loads(Path(review_path).read_text())
    expected_parameters={'gamma':1,'exposure_width':1,
        'noise_standard_deviation':'1/2','equal_prior_Bayes_error':'1/10'}
    if data.get('parameters') != expected_parameters:
        raise ValueError('The plotted labels require gamma=tau=1, nu=1/2 and error=1/10')
    expected_hypotheses=('Known omega=2*pi*m versus 2*omega. Each trial independently resets '
        'to stationarity (passive) or known phase pi/2 (prepared).')
    if data.get('hypotheses') != expected_hypotheses:
        raise ValueError('The figure requires the declared independent omega-versus-2omega trial hypotheses')
    certificate=review['finite_count_certificate']
    expected_certificate_parameters=('gamma=tau=1, noise standard deviation=1/2, '
        'equal-prior error=1/10, omega=64*pi versus 128*pi')
    if certificate.get('parameters') != expected_certificate_parameters:
        raise ValueError('The finite-count certificate has a different calibration or hypothesis pair')
    if not review['passed'] or not certificate['verified']:
        raise ValueError('The exact finite-count certificate must pass independent review')
    if certificate['passive_insufficient_trials'] != 17_000_000 or certificate['prepared_sufficient_trials'] != 10_000_000:
        raise ValueError('The figure is scoped to the reviewed 17M/10M comparison')
    rows=data['rows'];indices=np.array([r['index'] for r in rows])
    if list(indices) != [1,2,4,8,16,32]:
        raise ValueError('Unexpected analytical frequency grid')
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'dejavusans',
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'xtick.direction':'out','ytick.direction':'out',
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
    })
    fig,(ax,comparison)=plt.subplots(1,2,figsize=(11.4,5.6),
        gridspec_kw={'width_ratios':[1.28,1]},layout=None)
    fig.subplots_adjust(left=.085,right=.985,bottom=.25,top=.78,wspace=.39)
    for method,color,label in (('passive',PASSIVE,'Passive'),('prepared',PREPARED,'Phase prepared')):
        lower=np.array([float(r[f'{method}_necessary_threshold']) for r in rows])
        upper=np.array([float(r[f'{method}_sufficient_threshold']) for r in rows])
        ax.fill_between(indices,lower,upper,color=color,alpha=.08,zorder=1)
        ax.plot(indices,lower,'--',marker='v',markersize=4.5,color=color,lw=1.6,
                label=f'{label}: necessary',zorder=3)
        ax.plot(indices,upper,'-',marker='o',markersize=4.3,color=color,lw=1.8,
                label=f'{label}: sufficient',zorder=4)
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    ax.set_xlim(.9,35.5);ax.set_ylim(.35,1.2e10)
    ax.set_xticks(indices,labels=[str(x) for x in indices])
    ax.set_yticks([1,1e2,1e4,1e6,1e8,1e10])
    ax.set_xlabel(r'Resonance index $m$  ($\omega=2\pi m$)')
    ax.set_ylabel('Independent exposure trials')
    ax.set_title('Analytical count brackets',fontweight='bold',fontsize=11,pad=15)
    ax.grid(axis='y',which='major',color='#DCE1E5',lw=.6,zorder=0)
    ax.legend(loc='upper left',frameon=True,framealpha=.96,edgecolor='#E0E4E7',
              fontsize=8.3,handlelength=2.5,labelspacing=.48,borderpad=.7)
    ax.text(0,-.21,'Shading bounds the required count; it is not a confidence band.',
            transform=ax.transAxes,fontsize=8.2,color='#48545D',ha='left')

    comparison.set_title(r'Exact comparison at $m=32$',fontweight='bold',fontsize=11,pad=15)
    comparison.barh([1,0],[17,10],height=.32,color=[PASSIVE,PREPARED],zorder=3)
    comparison.text(8.5,1,'17M insufficient',ha='center',va='center',color='white',fontweight='bold',fontsize=9.5)
    comparison.text(5,0,'10M sufficient',ha='center',va='center',color='white',fontweight='bold',fontsize=9.5)
    comparison.annotate('',xy=(19.5,1),xytext=(17,1),
                        arrowprops=dict(arrowstyle='->',color=PASSIVE,lw=1.8))
    comparison.text(.25,.69,'Every passive decision rule',ha='left',va='top',fontsize=9,color='#48545D')
    comparison.text(.25,-.31,'Constructive mean test',ha='left',va='top',fontsize=9,color='#48545D')
    comparison.set_yticks([1,0],labels=['Passive','Phase\nprepared'])
    comparison.tick_params(axis='y',length=0,pad=7)
    comparison.set_xlim(0,20);comparison.set_ylim(-.68,1.55)
    comparison.set_xticks([0,5,10,15,20])
    comparison.set_xlabel('Exposed readouts (millions)')
    comparison.grid(axis='x',color='#DCE1E5',lw=.6,zorder=0)
    comparison.spines['left'].set_visible(False)
    comparison.text(0,-.21,'This finite separation uses exact rational inequalities.',
                    transform=comparison.transAxes,fontsize=8.2,color='#48545D')

    fig.suptitle('Detector access changes the exposure count',x=.085,y=.965,ha='left',
                 fontsize=16,fontweight='bold',color='#162D3D')
    fig.text(.085,.888,r'Known alternatives $\omega$ and $2\omega$; $\gamma=\tau=1$, Gaussian noise $\nu=1/2$, target binary error $\leq0.1$.',
             ha='left',fontsize=10.1,color='#374B59')
    fig.text(.085,.083,'Fixed read noise. Each trial requires an independent preparation; preparation time and energy are not counted.',
             ha='left',fontsize=8.9,color='#374B59')
    fig.text(.085,.045,r'Under the separately assumed sensor law $\nu^2=\kappa/E$, both optimized read-energy orders are $\Theta(\omega^2)$.',
             ha='left',fontsize=8.9,color='#374B59')
    output_prefix=Path(output_prefix);output_prefix.parent.mkdir(parents=True,exist_ok=True)
    metadata={'Title':'Analytical detector-access exposure counts',
              'Author':'Abdinoor, 2D Labs','Subject':'Analytical bounds, not simulated detection data',
              'Creator':'Matplotlib','CreationDate':None,'ModDate':None}
    fig.savefig(output_prefix.with_suffix('.pdf'),metadata=metadata)
    fig.savefig(output_prefix.with_suffix('.png'),dpi=180,
                metadata={'Software':'Matplotlib','Title':'Analytical detector-access exposure counts'})
    plt.close(fig)
    return output_prefix.with_suffix('.pdf'),output_prefix.with_suffix('.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'design/results.json')
    parser.add_argument('--review',type=Path,default=ROOT/'review/results.json')
    parser.add_argument('--output-prefix',type=Path,default=ROOT/'design/detector_counts')
    args=parser.parse_args()
    for path in make_figure(args.input,args.review,args.output_prefix):print(path)

if __name__=='__main__':main()
