#!/usr/bin/env python3
"""Render the model-based realization/resource overview from release JSON.

Run from a release root, or pass --release-root explicitly. The figure adds
no simulations or inferred experimental data; all scientific values are read
from the corresponding reproducible results. Matplotlib is the only plotting
dependency. PDF timestamps are omitted for deterministic metadata.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np


def rational(value):
    return float(Fraction(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.release_root.resolve()
    results=root/'frontier'/'results'
    if not results.is_dir():
        results=root/'results'  # Source-phase preview uses the same JSON.
    hierarchy=json.loads((results/'hierarchy.json').read_text())
    energy=json.loads((results/'energy_certificate.json').read_text())
    destination=args.output_dir or root/'manuscript'/'figures'
    destination.mkdir(parents=True,exist_ok=True)

    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.dpi':180,'figure.facecolor':'white','axes.facecolor':'white',
    })
    blue='#176B96'; orange='#B75622'; grey='#68727B'; green='#36755B'
    # Native book width avoids shrinking labels when included at text width.
    fig=plt.figure(figsize=(6.5,5.8))
    grid=fig.add_gridspec(2,2,left=.11,right=.98,bottom=.13,top=.86,
                         wspace=.38,hspace=.72,height_ratios=[1,1])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),
          fig.add_subplot(grid[1,:])]

    ax=axes[0]
    entries=hierarchy['hierarchy']
    coherence=[row['coherence_ratio'] for row in entries]
    lower=[row['certificate']['minimum_states'] for row in entries]
    ax.scatter(coherence,lower,s=58,color=blue,zorder=3)
    for index,(x,y,row) in enumerate(zip(coherence,lower,entries)):
        alignment='left' if index==0 else ('right' if index==len(entries)-1 else 'center')
        ax.annotate(f"{row['settings']} time settings",(x,y),xytext=(0,12),
                    textcoords='offset points',ha=alignment,fontsize=8,color=grey)
    ax.set_xscale('log')
    ax.set_xlim(min(coherence)/1.8,max(coherence)*1.8)
    ax.set_ylim(min(lower)-.75,max(lower)+.55)
    ax.set_yticks(lower)
    ax.xaxis.set_major_locator(FixedLocator(coherence))
    ax.xaxis.set_major_formatter(FixedFormatter([r'$10^3$',r'$10^4$',r'$10^5$']))
    ax.minorticks_off()
    ax.set_xlabel(r'Specified qubit coherence $\omega/\gamma$')
    ax.set_ylabel('Minimum classical states')
    ax.set_title('A  Classical state-count bounds',loc='left',fontweight='bold',pad=11)
    ax.grid(axis='y',color='#E4E8EA',linewidth=.6,zorder=0)
    ax.text(.04,.05,'Specified qubit models\nHilbert-space dimension: 2',
            transform=ax.transAxes,fontsize=8,color=grey,va='bottom')

    ax=axes[1]
    curvature=energy['curvature'];certificate=curvature['certificate']
    step=rational(certificate['step'])
    times=np.array(certificate['lags'],dtype=float)*step
    exact=np.array([[rational(z) for z in interval]
                    for interval in curvature['exact_probability_intervals']])
    boxes=np.array([[rational(z) for z in interval]
                    for interval in certificate['intervals']])
    means=exact.mean(axis=1)
    errors=np.vstack([means-boxes[:,0],boxes[:,1]-means])
    ax.plot(times,means,color=blue,lw=1,alpha=.6,zorder=2)
    ax.plot(times[[0,2]],means[[0,2]],'--',color=grey,lw=1,zorder=1)
    ax.errorbar(times,means,yerr=errors,fmt='o',color=blue,ms=5,
                capsize=4,elinewidth=1.4,zorder=4)
    midpoint=(means[0]+means[2])/2
    ax.annotate('',xy=(times[1],means[1]),xytext=(times[1],midpoint),
                arrowprops={'arrowstyle':'<->','color':orange,'lw':1.4})
    ax.annotate('Above the\nequilibrium chord',xy=(times[1],(means[1]+midpoint)/2),
                xytext=(.17,.565),fontsize=8,color=orange,
                arrowprops={'arrowstyle':'-','color':orange,'lw':.7})
    ax.set_xlim(times[0]-.025,times[-1]+.025)
    ax.set_ylim(.525,.767)
    ax.set_xticks(times)
    ax.set_yticks([.55,.60,.65,.70])
    ax.set_xlabel('Lag (model time)')
    ax.set_ylabel('Matching-endpoint probability')
    ax.set_title('B  Stationary correlation shape',loc='left',fontweight='bold',pad=11)
    ax.text(.04,.97,'Positive and decreasing;\nnot convex',
            transform=ax.transAxes,fontsize=8,color=grey,va='top')
    ax.text(.04,.025,r'Rate $>7.31\times10^{-4}$ / time',
            transform=ax.transAxes,fontsize=8,color=blue,va='bottom')

    ax=axes[2]
    J=rational(certificate['record_kl_lower'])
    prep=np.linspace(0,1.28,257)
    housekeeping=2*np.maximum(0,1-prep)
    ax.fill_between(prep,0,housekeeping,color=orange,alpha=.14,zorder=0)
    ax.plot(prep,housekeeping,color=orange,lw=2,zorder=2)
    ax.set_xlim(0,1.28);ax.set_ylim(0,2.65)
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,1,2])
    ax.set_xlabel(r'Preparation, $D(\nu\Vert\pi)/J$')
    ax.set_ylabel(r'Housekeeping budget, $K/J$')
    ax.set_title('C  Necessary preparation or housekeeping budget',loc='left',fontweight='bold',pad=11)
    ax.text(.42,2.18,r'$D(\nu\Vert\pi)+K/2\;\geq\;J$',fontsize=10,ha='center',color=green)
    ax.text(.77,1.23,'Not excluded\nby this bound',ha='center',fontsize=9,color=green)
    ax.text(.24,.40,'Excluded\nby the record',ha='center',fontsize=9,color=orange)
    ax.text(.98,.94,f'J = {J:.3g} nats',transform=ax.transAxes,
            ha='right',va='top',fontsize=8.5,color=grey)

    fig.text(.06,.967,'Classical explanation: states, timing, and resources',
             fontsize=10.5,fontweight='bold',ha='left',va='top')
    fig.text(.06,.025,
        'Specified models and synthetic uncertainty; no laboratory measurements.',
        fontsize=8,color=grey,ha='left')
    for extension in ('pdf','png'):
        path=destination/f'realization_frontier.{extension}'
        metadata={'Creator':'2D Labs reproducible research', 'CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(path,metadata=metadata)
        print(path)
    plt.close(fig)


if __name__=='__main__':
    main()
