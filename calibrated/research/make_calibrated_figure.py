"""Rebuild calibrated-inference figures from saved scientific reports.

This script performs no new Monte Carlo study or optimization. Curves marked
mathematical use closed-form theorem expressions. Points marked simulated read
the frozen report summaries, including unsuccessful and outside-contract cases.
Evolution time and detector reads are separate resources; neither is a calibrated
measurement of joules. Default paths work in the phase and cumulative release.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BLUE = '#176b91'
ORANGE = '#c47724'
GREEN = '#33816b'
RED = '#ae4747'
GRAY = '#606878'
COLORS = [BLUE,ORANGE,GREEN,RED]


def number(value):
    """Read the release's exact-rational and compact JSON encodings."""
    if isinstance(value,dict):
        for key in ('decimal','fraction','lower'):
            if key in value:
                return number(value[key])
        raise ValueError('Unrecognized numerical summary')
    return float(Fraction(value)) if isinstance(value,str) else float(value)


def locate_results(explicit=None):
    if explicit is not None:
        return explicit.resolve()
    base = Path(__file__).resolve().parents[1]
    for directory in (base/'calibrated/results',base/'results',
                      Path.cwd()/'calibrated/results',Path.cwd()/'results'):
        if (directory/'sequential_benchmark.json').is_file():
            return directory.resolve()
    raise FileNotFoundError('Pass --results-dir containing the calibrated reports.')


def load(directory,name):
    path = directory/name
    if not path.is_file():
        raise FileNotFoundError(f'Missing finalized report: {path}')
    return json.loads(path.read_text())


def setup_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlelocation':'left','axes.titlepad':10,'axes.grid':True,
        'grid.alpha':.18,'grid.linewidth':.6,'legend.frameon':False,
        'legend.fontsize':8.8,'xtick.labelsize':9,'ytick.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180,
        'svg.hashsalt':'calibrated-inference-figures-v1',
    })


def save(fig,directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    pdf = directory/(name+'.pdf')
    png = directory/(name+'.png')
    fig.savefig(pdf,bbox_inches='tight',metadata={
        'Title':name.replace('_',' '),'Author':'Abdinoor / 2D Labs',
        'Subject':'Mathematical and simulated results; not laboratory validation',
        'Creator':'make_calibrated_figure.py','CreationDate':None,'ModDate':None})
    fig.savefig(png,bbox_inches='tight',metadata={'Software':'make_calibrated_figure.py'})
    plt.close(fig)
    return [pdf,png]


def errorbar(ax,x,p,interval,**kwargs):
    lo,hi = map(number,interval)
    ax.errorbar(x,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],
                capsize=3,elinewidth=1,**kwargs)


def footer(fig,paragraphs):
    text = '\n'.join(textwrap.fill(p,width=108) for p in paragraphs)
    fig.text(.075,.025,text,fontsize=8.3,color=GRAY,linespacing=1.35)


def inference_figure(directory,output):
    report = load(directory,'calibrated_benchmark.json')
    cases = {row['case']:row for row in report['cases']}
    driven = cases['driven']
    methods = ['calibrated_pinsker','calibrated_nonlinear','uncalibrated_extra_lag']
    calibrated_cost = report['costs']['calibrated_arm']
    alternative_cost = report['costs']['extra_lag_arm']
    assert calibrated_cost['pairs'] == alternative_cost['pairs']
    assert calibrated_cost['readouts'] == alternative_cost['readouts']
    fig,axs = plt.subplots(2,2,figsize=(7.8,9.1))
    fig.subplots_adjust(hspace=.61,wspace=.39,bottom=.20,top=.83,left=.115,right=.97)
    fig.suptitle('Calibrated inference: stronger bounds, explicit limits',
                 fontsize=13,x=.075,ha='left',y=.985)
    fig.text(.075,.947,'Synthetic models and mathematical bounds\nNo laboratory validation',
             fontsize=9.5,color=GRAY,linespacing=1.5,va='top')

    ax = axs[0,0]
    values = [number(driven['rate_statistics'][method]['mean']) for method in methods]
    ax.bar(range(3),values,color=[ORANGE,BLUE,GREEN],width=.61)
    times = [calibrated_cost['declared_evolution_time']]*2+[
        alternative_cost['declared_evolution_time']]
    names = ['Pinsker\ncalibrated','Nonlinear\ncalibrated','Extra lag\nuncalibrated']
    ax.set_xticks(range(3),[f'{name}\nT={time:,}' for name,time in zip(names,times)])
    ax.tick_params(axis='x',labelsize=8.5)
    ax.set_ylabel('Mean certified entropy-rate\nlower bound')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    ax.set_title('A  Stronger lower bounds\n(simulated)')
    top = max(values)
    if top:
        ax.set_ylim(0,top*1.32)
        for j,value in enumerate(values):
            ax.text(j,value+top*.03,f'{value:.3g}',ha='center',fontsize=8)
    ratio = driven['nonlinear_to_pinsker_ratio_on_positive_pinsker']
    ax.text(.98,.96,f'Mean ratio: {number(ratio["mean"]):.2f} (positive Pinsker)\n'
            f'True model rate: {number(driven["oracle"]["stationary_entropy_rate"]):.3g}',
            ha='right',va='top',transform=ax.transAxes,fontsize=8.5,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.85})

    ax = axs[0,1]
    order = ['driven','equilibrium','high_readout_noise','invalid_calibration']
    labels = ['Driven','Equilibrium','Weak\nreadout','Drift\n(invalid)']
    ax.axvspan(2.57,3.47,color=RED,alpha=.065,zorder=0)
    for j,(method,color,label) in enumerate(zip(methods,[ORANGE,BLUE,GREEN],
        ['Calibrated Pinsker','Nonlinear (raw scores)','Extra lag'])):
        for i,name in enumerate(order):
            row = cases[name]['inference'][method]
            errorbar(ax,i+(j-1)*.19,number(row['rate']),row['monte_carlo_95_interval'],
                     marker='o',linestyle='none',color=color,markersize=4,
                     label=label if i == 0 else None)
    ax.axhline(.05,color=GRAY,linestyle=':',linewidth=.8)
    ax.set_xticks(range(4),labels)
    ax.set_ylim(-.04,1.06)
    ax.set_ylabel('Positive certificate frequency')
    ax.set_title('B  Detection and calibration failure\n(simulated)')
    ax.tick_params(axis='x',labelsize=8.5)
    ax.legend(loc='center left',fontsize=8)
    ax.text(.97,.12,'95% pointwise\nMonte Carlo intervals',transform=ax.transAxes,
            ha='right',fontsize=8.5,color=GRAY)

    ax = axs[1,0]
    asymmetry = np.linspace(0,.9997,700)
    record = ((1+asymmetry)*np.log1p(asymmetry)+
              (1-asymmetry)*np.log1p(-asymmetry))/2
    entropy = asymmetry*np.log((1+asymmetry)/(1-asymmetry))
    ax.plot(record,entropy,color=BLUE,linewidth=2,label='With microscopic activity cap')
    ax.plot(record,4*record,color=ORANGE,linestyle='--',linewidth=1.6,label='Linear bridge')
    ax.set_xlim(0,.7)
    ax.set_ylim(0,8)
    ax.set_xlabel('Record KL /\n(activity cap × expected duration)')
    ax.set_ylabel('Entropy-rate lower bound /\nactivity cap')
    ax.set_title('C  Nonlinear activity bridge\n(mathematical)')
    ax.legend(loc='upper left',fontsize=8)
    ax.text(.035,.50,'The cap concerns all microscopic\njumps. Detector flips do not\nestablish that cap.',
            transform=ax.transAxes,color=GRAY,fontsize=8.5)

    ax = axs[1,1]
    q = np.geomspace(.5,1e-10,500)
    sigma = (1-2*q)*np.log((1-q)/q)
    ax.semilogx(q,sigma,color=RED,linewidth=2,label='Actual microscopic entropy rate')
    ax.semilogx(q,np.zeros_like(q),color=GRAY,linestyle='--',label='Observed parity-path irreversibility')
    ax.invert_xaxis()
    ax.set_ylim(-1,25)
    ax.set_xlabel('Reverse-jump probability q\n(toward zero →)')
    ax.set_ylabel('Natural entropy per model time')
    ax.set_title('D  Hidden dissipation, visible jumps\n(mathematical)')
    ax.legend(loc='upper left',fontsize=7.7)

    footer(fig,[
        'A–B: two-stage calibration benchmark, 400 simulated datasets per case; 20,000 pairs and 40,000 reads per arm. Means include zero bounds; nonlinear scores are raw.',
        'Counts include calibration. Evolution times differ; reset, detector latency, hardware and energy overhead remain unmeasured.',
        'C–D: conditional mathematical statements. D fixes the four-state graph and unit activity; every jump is visible, yet the entire parity path law agrees.'])
    return save(fig,output,'calibrated_inference')


def resources_figure(directory,output):
    sequential = load(directory,'sequential_benchmark.json')
    dependence = load(directory,'dependence.json')
    refinement = load(directory,'dependence_refinement.json')
    fig,axs = plt.subplots(2,2,figsize=(7.8,9.1))
    fig.subplots_adjust(hspace=.63,wspace=.39,bottom=.20,top=.83,left=.115,right=.97)
    fig.suptitle('Physical access has several resource costs',fontsize=13,x=.075,ha='left',y=.985)
    fig.text(.075,.947,'Simulated power, evolution time and detector reads\nThese quantities are not calibrated energy savings.',
             fontsize=9.5,color=GRAY,linespacing=1.5,va='top')

    ax = axs[0,0]
    by_case = {row['case']:row for row in sequential['cases']}
    methods = [('fixed_horizon','s',GRAY),('fixed','o',ORANGE),('learned','^',BLUE)]
    for ci,case in enumerate(['driven','changed_drive']):
        for name,marker,color in methods:
            row = by_case[case]['methods'][name]
            time = row.get('mean_capped_evolution_time',row.get('evolution_time'))
            errorbar(ax,time,number(row['probability']),row['mc_interval_95'],
                     marker=marker,linestyle='none',color=color,markersize=6,
                     markerfacecolor=color if ci == 0 else 'white')
    handles = [Line2D([0],[0],marker=m,color=c,linestyle='none',label=l)
               for (_,m,c),l in zip(methods,['Fixed horizon','Fixed anytime','Learned anytime'])]
    ax.legend(handles=handles,loc='center left')
    ax.set_xlim(0,4750)
    ax.set_ylim(.68,1.035)
    ax.set_xlabel('Mean evolution time\n(including non-detections)')
    ax.set_ylabel('Detection probability\nby the same maximum budget')
    ax.set_title('A  Sequential stopping tradeoff\n(simulated)')

    ax = axs[0,1]
    for budget,color in [(400000,BLUE),(800000,GREEN)]:
        oldplan = next(p for p in dependence['plans'] if p['budget'] == budget)
        rows = [dict(oldplan['new'],block_duration='2')]+[
            p for p in refinement['plans'] if p['budget'] == budget]
        rows.sort(key=lambda r:r['endpoint_readouts'])
        ax.semilogx([p['endpoint_readouts'] for p in rows],
                    [number(p['contrast_halfwidth']) for p in rows],
                    'o-',color=color,markersize=4,label=f'T={budget:,}, common blocks')
        old = oldplan['old']
        ax.semilogx(old['endpoint_readouts'],number(old['contrast_halfwidth']),
                    '*',color=color,markersize=11)
        for p in rows:
            h = number(p['block_duration'])
            if h in (2,50,500):
                ax.annotate(f'H={h:g}',(p['endpoint_readouts'],number(p['contrast_halfwidth'])),
                            xytext=(3,5),textcoords='offset points',fontsize=8.5,color=color)
    ax.set_xlabel('Endpoint readouts (log scale)')
    ax.set_ylabel('Certified contrast halfwidth')
    ax.set_title('B  Time versus readout demand\n(mathematical design)')
    handles,labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0],[0],marker='*',color=GRAY,linestyle='none',markersize=8))
    labels.append('Blind-gap baseline')
    ax.legend(handles,labels,loc='upper right',fontsize=8)

    ax = axs[1,0]
    for budget,color in [(400000,BLUE),(800000,GREEN)]:
        denseplan = next(p for p in dependence['plans'] if p['budget'] == budget)
        dense_eval = next(e for e in dependence['evaluations']
                          if e['budget'] == budget and e['model'] == 'driven_ring')
        dense = next(m for m in dense_eval['methods'] if m['method'] == 'continuous_markov')
        rows = [(denseplan['new']['endpoint_readouts'],dense['equilibrium_exclusion'])]
        for row in refinement['evaluations']:
            if row['budget'] != budget or row['model'] != 'driven_ring':
                continue
            plan = next(p for p in refinement['plans']
                        if p['budget'] == budget and number(p['block_duration']) == row['block_duration'])
            rows.append((plan['endpoint_readouts'],row['equilibrium_exclusion']))
        rows.sort(key=lambda r:r[0])
        ax.plot([x for x,_ in rows],[number(r['frequency']) for _,r in rows],color=color,linewidth=1)
        for j,(x,row) in enumerate(rows):
            errorbar(ax,x,number(row['frequency']),row['approximate_wilson_mc95'],
                     marker='o',linestyle='none',color=color,markersize=4,
                     label=f'T={budget:,}' if j == 0 else None)
    ax.set_xscale('log')
    ax.set_ylim(0,1.06)
    ax.set_xlabel('Endpoint readouts (log scale)')
    ax.set_ylabel('Detection probability')
    ax.set_title('C  Sparse observation preserves power\n(in some simulated regimes)')
    ax.legend(loc='lower right')

    ax = axs[1,1]
    settings = [('metastable_equilibrium','continuous_markov','Block\ncertificate',BLUE),
                ('metastable_equilibrium','optimized_old_blind_gap','Blind-gap\ncertificate',GREEN),
                ('metastable_equilibrium','unqualified_independent_analysis','IID\n(invalid)',RED),
                ('wrong_mixing_calibration','continuous_markov','Mixing bound\n(invalid)',RED)]
    for i,(model,method,label,color) in enumerate(settings):
        case = next(e for e in dependence['evaluations'] if e['budget'] == 400000 and e['model'] == model)
        row = next(m for m in case['methods'] if m['method'] == method)['equilibrium_exclusion']
        errorbar(ax,i,number(row['frequency']),row['approximate_wilson_mc95'],
                 marker='o',linestyle='none',color=color,markersize=6)
    ax.axhline(.05,color=GRAY,linestyle=':',label='Nominal total error budget')
    ax.axvspan(1.5,3.5,color=RED,alpha=.06,zorder=0)
    ax.set_xticks(range(4),[r[2] for r in settings])
    ax.set_xlim(-.4,3.5)
    ax.set_ylim(-.01,.34)
    ax.set_ylabel('False positive frequency\nunder equilibrium')
    ax.set_title('D  Dependence assumptions matter\n(simulated equilibrium)')
    ax.tick_params(axis='x',labelsize=8.5)
    ax.legend(loc='upper left',fontsize=8)
    ax.text(.98,.36,'Shaded: invalid premises',transform=ax.transAxes,
            ha='right',fontsize=8,color=RED)

    footer(fig,[
        'A: 400 runs; filled markers show the original drive, open markers the changed drive. Means include failed runs at their full budget.',
        'B–C match evolution-time budgets, not readout counts. H is block duration. C: 1,000 runs per point. All intervals are pointwise 95% Monte Carlo intervals.',
        'Mixing and initial-law control require separate calibration. Reset, latency and control overhead are additional; no value here is a calibrated joule cost.'])
    return save(fig,output,'calibrated_resources')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--only',choices=['all','inference','resources'],default='all')
    args = parser.parse_args()
    results = locate_results(args.results_dir)
    output = args.output_dir.resolve() if args.output_dir else results
    setup_style()
    paths = []
    if args.only in ('all','inference'):
        paths += inference_figure(results,output)
    if args.only in ('all','resources'):
        paths += resources_figure(results,output)
    print(json.dumps({'figures':[{'path':str(path),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'evidence':'Saved synthetic benchmarks and mathematical curves; no empirical validation'},indent=2))


if __name__ == '__main__':
    main()
