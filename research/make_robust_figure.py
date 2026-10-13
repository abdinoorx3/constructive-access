"""Numerical illustrations of exact sharp bounds and declared statistical plans."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
t=np.linspace(.001,5,300);a=(1-np.exp(-t))/t
g=np.exp(-t);k=1-2*a+g
for eta,color in zip([0,.25,.5,1],['#245681','#679a9c','#bba563','#8b658b']):
    axes[0,0].plot(t,g-eta*k,label=f'Reset defect {eta:g}',color=color)
c=(.1+np.exp(-.1*t)*(-.1*np.cos(2*t)+2*np.sin(2*t)))/(4.01*t)
axes[0,0].plot(t,c,'--',color='#b54b34',label='Depolarizing qubit')
axes[0,0].axhline(0,color='0.5',lw=.7)
axes[0,0].set(title='A  Sharp classical explanation boundary',xlabel='Observation horizon (caps a = b = 1)',ylabel='Normalized occupation contrast')
axes[0,0].legend(fontsize=8)

example=json.loads((ROOT/'robustness/results/explanation.json').read_text())['examples'][1]
cu=float(Fraction(example['physical_contrast_upper']))
ls=np.linspace(.2,2,300);z=2.25*ls
aa=(1-np.exp(-z))/z;gg=np.exp(-z);kk=1-2*aa+gg
eta=np.maximum(0,(gg-cu)/kk)
axes[0,1].plot(ls,eta,color='#245681')
axes[0,1].axhline(1,color='0.5',lw=.8,ls='--')
axes[0,1].axvspan(.2,.8127481484,color='#b54b34',alpha=.1)
axes[0,1].scatter([1],[.6040934688],color='#245681')
axes[0,1].set(xlim=(.2,2),ylim=(0,1.6),title='B  Conditional inference from synthetic counts',xlabel='Supplied common crossing cap',ylabel='Required reset defect (95% lower bound)')
axes[0,1].text(.25,1.4,'No compatible defect ≤ 1',fontsize=8,color='#b54b34')
axes[0,1].text(1.04,.64,'343 / 600 encoded successes',fontsize=8)

ts=np.geomspace(.01,1,200)
axes[1,0].loglog(ts,.04/ts,label='Unmatched preparation bound',color='#8b658b')
axes[1,0].loglog(ts,np.full_like(ts,.04),label='Matched control bound',color='#245681')
axes[1,0].set(title='C  Matched control removes a short-pulse penalty',xlabel='Pulse duration (declared units)',ylabel='Certified preparation contribution to current error')
axes[1,0].legend(fontsize=8)

axes[1,1].bar(['Exact fixed','Anytime horizon','Chernoff fixed'],[600,1100,1181],color=['#245681','#679a9c','#bba563'])
axes[1,1].set(title='D  Same contrast and calibration contract',ylabel='Certified sufficient endpoint records')
for n,v in enumerate([600,1100,1181]):axes[1,1].text(n,v+18,str(v),ha='center',fontsize=9)
axes[1,1].set_ylim(0,1350)
fig.savefig(ROOT/'manuscript/robust_results.pdf')
fig.savefig(ROOT/'manuscript/robust_results.png',dpi=160)
print('Wrote robust_results.pdf and .png; all data are model-based illustrations.')

# The state-count plot compares model laws, not measured data.
fig2,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
x=np.linspace(-1,1,200)
ax[0].fill_between(x,np.maximum(-1,x-1),np.minimum(1,x+1),color='#679a9c',alpha=.2,label='Allowed bound: ≤3 classical states')
ax[0].plot(x,np.maximum(-1,x-1),color='#679a9c',lw=1)
ax[0].plot(x,np.minimum(1,x+1),color='#679a9c',lw=1)
t=np.linspace(.001,5,400);dec=np.exp(-.1*t)
c=(.1+dec*(-.1*np.cos(2*t)+2*np.sin(2*t)))/(4.01*t)
s=(2-dec*(.1*np.sin(2*t)+2*np.cos(2*t)))/(4.01*t)
gain=.9;ny=-12/37;nz=35/37
ct=gain*(nz*c+ny*s);c0=gain*nz
c2=(.1+np.exp(-.2)*(-.1*np.cos(4)+2*np.sin(4)))/8.02
s2=(2-np.exp(-.2)*(.1*np.sin(4)+2*np.cos(4)))/8.02
ct2=gain*(nz*c2+ny*s2)
ax[0].plot(np.full_like(ct,c0),ct,color='#b54b34',lw=2,label='Specified noisy qubit, varying horizon')
ax[0].scatter([c0],[ct2],color='#b54b34',zorder=5)
ax[0].annotate('Certified example: T = 2',(c0,ct2),xytext=(-.6,-.6),arrowprops={'arrowstyle':'->','color':'#b54b34'},fontsize=9)
ax[0].set(xlim=(-1,1),ylim=(-1,1),xlabel='Reference-time contrast C₀',ylabel='Random-time contrast Cₜ',title='A  An observable classical-state bound')
ax[0].legend(loc='upper left',fontsize=7)
ax[1].plot(t,.9*(1-c),label='Detector eigenpreparations',color='#245681')
ax[1].plot(t,c0-ct,label='Fixed tilted preparations',color='#b54b34')
ax[1].axhline(1,color='0.3',ls='--',lw=1,label='Three-state classical limit')
ax[1].scatter([2],[c0-ct2],color='#b54b34')
ax[1].annotate('W > 1.08',(2,c0-ct2),xytext=(2.4,1.16),arrowprops={'arrowstyle':'->'},fontsize=9)
ax[1].set(xlim=(0,5),ylim=(0,1.25),xlabel='Uniform observation horizon',ylabel='Witness W = C₀ − Cₜ',title='B  Quantum prediction with fixed readout gain 0.9')
ax[1].legend(loc='lower right',fontsize=8)
fig2.savefig(ROOT/'manuscript/dimension_witness.pdf')
fig2.savefig(ROOT/'manuscript/dimension_witness.png',dpi=160)
print('Wrote dimension_witness.pdf and .png; model predictions only.')
