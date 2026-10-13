"""Generate the reference's conceptual map and exact-law tutorial figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(11,3.3))
ax.set_xlim(0,11);ax.set_ylim(0,3.3);ax.axis('off')
labels=[('Specify','Target, candidate models,\npreparation and access'),('Distinguish','Equivalence certificate\nor executable witness'),('Calibrate','Robust gap, target interval\nand finite-error design'),('Infer','Retained target set,\ndecision or abstention')]
for i,(title,body) in enumerate(labels):
    x=.12+i*2.75
    box=FancyBboxPatch((x,1.24),2.5,1.3,boxstyle='round,pad=0.04',linewidth=1,edgecolor='#3a667d',facecolor='#edf4f7')
    ax.add_patch(box);ax.text(x+1.25,2.2,title,ha='center',weight='bold',fontsize=13,color='#20475d')
    ax.text(x+1.25,1.7,body,ha='center',va='center',fontsize=10)
    if i<3:ax.add_patch(FancyArrowPatch((x+2.54,1.9),(x+2.70,1.9),arrowstyle='-|>',mutation_scale=11,color='#3a667d'))
ax.text(5.5,.78,'Physical ledger: preparation + control + acquisition + reset + supporting apparatus',ha='center',fontsize=11)
ax.text(5.5,.35,'Every guarantee names its assumptions, error objective, resource unit and failure conditions.',ha='center',fontsize=10,color='#44545c')
fig.tight_layout(pad=.3)
fig.savefig(ROOT/'manuscript/access_map.pdf',bbox_inches='tight')
fig.savefig(ROOT/'manuscript/access_map.png',dpi=180,bbox_inches='tight');plt.close(fig)
result=json.loads((ROOT/'results/suite_tutorial.json').read_text())
models=list(result['targets']);menu=result['menu']
from fractions import Fraction
values=np.array([[float(Fraction(e['laws'][h]['1'])) for e in menu] for h in models])
fig,ax=plt.subplots(figsize=(6.6,3.4))
ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
ax.set_xticks(range(3),['Left · charge 1','Right · charge 1','Direct · charge 3'])
ax.set_yticks(range(5),[h+' / '+result['targets'][h] for h in models])
for i in range(5):
    for j in range(3):ax.text(j,i,str(Fraction(menu[j]['laws'][models[i]]['1'])),ha='center',va='center',color='white' if values[i,j]>.5 else '#173f55')
ax.set_title('Complete binary-record law: probability of outcome 1',pad=12)
ax.set_xlabel('Left and right together identify the target; A and A_copy remain indistinguishable.',fontsize=9,labelpad=12)
fig.tight_layout()
fig.savefig(ROOT/'manuscript/suite_laws.pdf',bbox_inches='tight')
fig.savefig(ROOT/'manuscript/suite_laws.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps({'figures':['access_map.pdf','suite_laws.pdf'],'source':'Exact finite probabilities from suite_tutorial.json; conceptual map contains no measured data.'}))
