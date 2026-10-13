#!/usr/bin/env python3
"""Native-width overview of the frozen virtual-laboratory results.

Default production rendering requires all three final result files. An explicit
--preview-pending flag is available solely for layout inspection while the
contrast benchmark runs; it never invents benchmark points.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def errors(rate,interval):
    low,high=interval
    if not 0<=low<=rate<=high<=1:raise ValueError('Invalid Monte Carlo interval')
    return np.array([[100*(rate-low)],[100*(high-rate)]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--preview-pending',action='store_true')
    args=parser.parse_args()
    root=args.release_root.resolve();results=root/'virtual'/'results'
    if not results.is_dir():results=root/'results'
    sampling=json.loads((results/'sampling.json').read_text())
    retiming=json.loads((results/'dimension_lab_retiming.json').read_text())
    benchmark_path=results/'contrast_benchmark.json'
    if benchmark_path.exists():benchmark=json.loads(benchmark_path.read_text())
    elif args.preview_pending:benchmark=None
    else:raise FileNotFoundError('Production figure requires final contrast_benchmark.json')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180,
        'figure.facecolor':'white','axes.facecolor':'white'})
    blue='#176B96';orange='#B75622';green='#36755B';grey='#68727B'
    fig=plt.figure(figsize=(6.5,5.9))
    grid=fig.add_gridspec(2,2,left=.105,right=.98,bottom=.225,top=.865,
                         hspace=.94,wspace=.43,height_ratios=[1,1])
    a,b,c=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]

    scenarios={row['name']:row for row in sampling['scenarios']}
    reuse=scenarios['shared_endpoint_pairs'];gap=scenarios['blind_mixing_gap']
    naive=next(row for row in reuse['analyses'] if row['label']=='unqualified_iid')
    corrected=next(row for row in gap['analyses'] if row['label']=='declared_corrected')
    for x,row,color in [(0,naive,orange),(1,corrected,blue)]:
        rate=row['false_reversible_monotonicity_exclusions']
        a.bar(x,100*rate,width=.52,color=color,alpha=.85,zorder=2)
        a.errorbar(x,100*rate,yerr=errors(rate,row['false_exclusion_mc95']),
                   fmt='none',ecolor='black',capsize=3,linewidth=.8,zorder=4)
        a.text(x,100*rate+2,f'{100*rate:.1f}%',ha='center',fontsize=8.5,color=color)
    a.axhline(5,color=grey,linestyle='--',linewidth=.8)
    a.text(1.42,5.5,'Nominal 5%',ha='right',fontsize=7.8,color=grey)
    a.set_xlim(-.5,1.5);a.set_ylim(-1.5,32)
    a.set_xticks([0,1],['Naive endpoint\nreuse','Mixing gap\n+ correction'])
    a.set_yticks([0,10,20,30]);a.set_ylabel('False exclusion (%)')
    a.set_title('A  Dependence correction',loc='left',fontweight='bold',pad=10)

    selected=[]
    for offset in [.005,.01,.02]:
        row=next(row for row in retiming['scenarios']
                 if row['name']==f'coherent_amplitude_offset_{offset}')
        selected.append(row)
    x=np.arange(3)
    for key,label,color,marker in [('original_clock','Original timing',orange,'o'),
                                  ('retimed_clock','Retimed',blue,'s')]:
        y=np.array([row['holdout'][key]['frequency'] for row in selected])
        bounds=np.array([row['holdout'][key]['clopper_pearson_interval'] for row in selected])
        b.errorbar(x,100*y,yerr=100*np.vstack([y-bounds[:,0],bounds[:,1]-y]),
                   fmt=marker+'-',label=label,color=color,markersize=4,capsize=3,linewidth=1)
    b.set_xticks(x,['0.5','1','2']);b.set_xlabel('Coherent offset (%)')
    b.set_ylim(-5,109);b.set_yticks([0,50,100]);b.set_ylabel('Detection probability (%)')
    b.set_title('B  Common clock retiming',loc='left',fontweight='bold',pad=10)
    b.legend(loc='center right',frameon=False,fontsize=7.8,handlelength=1.6)

    c.set_title('C  Refined contrast versus optimized box inference',loc='left',fontweight='bold',pad=10)
    if benchmark is None:
        c.axis('off');c.text(.5,.5,'Benchmark pending\nNo comparison points plotted',
                             transform=c.transAxes,ha='center',va='center',fontsize=10,color=grey)
        records=15000;exposure=4500
    else:
        refinements={row['case']:row for row in benchmark['new_refinement_holdout']}
        cases=['driven','independent_clock_jitter','high_readout_noise','equilibrium']
        labels=['Driven','Scale jitter','High readout\nnoise','Equilibrium']
        x=np.arange(len(cases));width=.30
        for position,(key,label,color) in enumerate([
                ('fixed_contrast','Predeclared contrast',blue),
                ('optimized_continuous_cone_box','Optimized box witness',orange)]):
            delta=(-.5 if position==0 else .5)*width
            for i,case in enumerate(cases):
                item=refinements[case][key];rate=item['rate']
                c.bar(i+delta,100*rate,width=width,color=color,alpha=.85,
                      label=label if i==0 else None,zorder=2)
                c.errorbar(i+delta,100*rate,yerr=errors(rate,item['monte_carlo_95_interval']),
                           fmt='none',ecolor='black',capsize=3,linewidth=.8,zorder=4)
        rows=[refinements[case] for case in cases]
        records=rows[0]['records_per_dataset'];exposure=rows[0]['nominal_evolution_time_per_dataset']
        if any(row['records_per_dataset']!=records or
               row['nominal_evolution_time_per_dataset']!=exposure for row in rows):
            raise ValueError('Figure requires matched nominal budgets in refinement cases')
        c.set_xticks(x,labels);c.set_ylabel('Detection probability (%)')
        c.set_ylim(-4,111);c.set_yticks([0,50,100])
        c.legend(loc='upper right',frameon=False,fontsize=8,ncol=2)
        c.grid(axis='y',color='#E6E8EA',linewidth=.5,zorder=0)

    multiplier=gap['physical_model_time_per_replicate']/reuse['physical_model_time_per_replicate']
    pilot=selected[0]['pilot_resources']['records']
    fig.text(.055,.969,'Virtual laboratory: failures, corrections, and matched tests',
             fontsize=10.3,fontweight='bold',ha='left',va='top')
    notes=[
        f'A: mixing-gap correction uses {multiplier:,.0f} times the observation time.',
        f'B: production records are matched; retiming adds {pilot:,} pilot records.',
        f'C: {records:,} records and {exposure:,.0f} expected model-time exposure per dataset.',
        'Error bars: pointwise 95% Monte Carlo intervals. Simulations, not hardware measurements.']
    for i,note in enumerate(notes):
        fig.text(.055,.145-.028*i,note,ha='left',va='top',fontsize=7.6,color=grey)
    destination=args.output_dir or root/'manuscript'/'figures'
    destination.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png'):
        path=destination/f'virtual_laboratory.{extension}'
        metadata={'Creator':'2D Labs reproducible research','CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(path,metadata=metadata);print(path)
    plt.close(fig)


if __name__=='__main__':main()
