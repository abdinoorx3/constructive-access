"""Numerical illustration of separately proved and exactly certified results."""
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10,7.4),layout='constrained')
times=np.linspace(0,5,251)

def quantum(t,g=1):
    gamma=.11
    a=np.array([[-4*gamma,2*g,.18],[-2*g,-2*gamma,0],[0,0,0.]])
    return -.5*expm(a*t)[0,2]

axes[0,0].plot(times,[quantum(t,0) for t in times],color='#245681',label='Classical two-state control')
axes[0,0].plot(times,[quantum(t) for t in times],color='#b54b34',label='Driven qubit')
for t in [.5,2.25]:axes[0,0].scatter([t],[quantum(t)],color='#b54b34',s=30,zorder=4)
axes[0,0].axhline(0,color='0.5',lw=.8)
axes[0,0].set(title='A  Resolved endpoint, selective pulse',xlabel='Pulse duration (declared units)',ylabel='Population shift from native baseline')
axes[0,0].legend(fontsize=9)

qs=np.linspace(.51,14,251)
minimum=[]
for q in qs:
    w=np.sqrt(4*q*q-1)
    minimum.append((2-2*q*np.exp(-3*(np.pi+np.arctan(w))/w))/(4*q*q+8))
axes[0,1].plot(qs,minimum,color='#b54b34')
axes[0,1].axhline(0,color='0.5',lw=.8)
axes[0,1].axvline(4.570664532533903,color='#245681',ls='--',lw=1)
axes[0,1].annotate('Certified threshold\n4.57066453253 < q < 4.57066453254',xy=(4.57,0),xytext=(6,.046),fontsize=9,arrowprops={'arrowstyle':'->','color':'0.4'})
axes[0,1].set(title='B  Exact response boundary (no extra dephasing)',xlabel='Coupling / damping, q = g / gamma',ylabel='First local minimum of gamma I(t)')

channels=[(0,1,2,1),(0,2,10,1),(1,3,10,1),(2,3,1,3)]
qr=np.zeros((4,4))
for k,(i,j,a,b) in enumerate(channels):
    factor=.5 if k==0 else 1
    qr[i,j]+=factor*a;qr[j,i]+=factor*b
qr-=np.diag(qr.sum(axis=1))
pi=np.array([7,4,80,30])/121
ts=np.linspace(0,1.6,201)
pop=np.array([pi@expm(t*qr)-pi for t in ts])
axes[1,0].plot(ts,pop[:,1],color='#245681',label='Resolved endpoint {1}')
axes[1,0].plot(ts,pop[:,1]+pop[:,2],color='#8b658b',label='Coarse region {1,2}')
axes[1,0].axhline(0,color='0.5',lw=.8)
axes[1,0].set(title='C  A classical false-positive control',xlabel='Pulse duration (declared units)',ylabel='Population shift')
axes[1,0].legend(fontsize=9)

x=np.arange(2);width=.34
axes[1,1].bar(x-width/2,[2700,67500],width,label='Two-sensor cut',color='#245681')
axes[1,1].bar(x+width/2,[6400,6400],width,label='Single-sensor cut',color='#ba9455')
axes[1,1].set_yscale('log')
axes[1,1].set_xticks(x,['Zero floors','Calibration floors'])
axes[1,1].set(title='D  Calibration can reverse the best access design',ylabel='Certified acquisition energy (model units)')
axes[1,1].legend(fontsize=9)
fig.savefig(ROOT/'manuscript/completion_results.pdf')
fig.savefig(ROOT/'manuscript/completion_results.png',dpi=160)
print('Wrote completion_results.pdf and .png; curves are numerical illustrations.')
