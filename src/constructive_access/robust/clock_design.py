"""Exact finite-menu design for decreasing-density random readout clocks.

A mixture of Uniform[0,T] clocks preserves integrated classical return order.
The optimizer is a two-constraint linear program, not new generic LP theory.
Cost means calibrated expected acquisition cost; no tail-energy guarantee follows.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,F)):
        raise TypeError('Use exact integers, rational strings, or Fraction')
    return F(x)


def normalize(menu,budget):
    budget=rational(budget)
    if budget<0 or not menu:raise ValueError('Nonempty menu and nonnegative budget required')
    rows=[]
    for row in menu:
        label=row['label']
        if not isinstance(label,str) or not label:raise ValueError('Nonempty labels required')
        h,c,lo,hi=map(rational,(row['horizon'],row['cost'],row['contrast_lower'],row['contrast_upper']))
        if h<=0 or c<0 or not -1<=lo<=hi<=1:raise ValueError('Invalid clock/contrast contract')
        rows.append(dict(label=label,horizon=h,cost=c,contrast_lower=lo,contrast_upper=hi))
    if len({r['label'] for r in rows})!=len(rows):raise ValueError('Duplicate label')
    if min(r['cost'] for r in rows)>budget:raise ValueError('No clock mixture fits budget')
    return rows,budget


def _linear_optimum(costs,values,budget):
    n=len(costs); candidates=[]
    for i in range(n):
        if costs[i]<=budget:
            w=[F(0)]*n;w[i]=F(1)
            candidates.append((values[i],-costs[i],tuple(w)))
    for i,j in combinations(range(n),2):
        if costs[i]==costs[j]:continue
        wi=(budget-costs[j])/(costs[i]-costs[j])
        if 0<wi<1:
            w=[F(0)]*n;w[i]=wi;w[j]=1-wi
            candidates.append((wi*values[i]+(1-wi)*values[j],-budget,tuple(w)))
    value,negcost,weights=max(candidates)
    used=-negcost
    if used<budget:lam=F(0)
    else:
        lam=max([F(0)]+[(v-value)/(c-budget) for c,v in zip(costs,values) if c>budget])
    intercept=value-lam*budget
    if any(intercept+lam*c<v for c,v in zip(costs,values)):
        raise ArithmeticError('Failed supporting-line certificate')
    return dict(value=value,weights=list(weights),expected_cost=used,
                dual_slope=lam,dual_intercept=intercept)


def _check_linear(costs,values,budget,cert):
    try:
        weights=list(map(rational,cert['weights']))
        lam=rational(cert['dual_slope']);intercept=rational(cert['dual_intercept'])
        value=rational(cert['value']);cost=rational(cert['expected_cost'])
        return (len(weights)==len(costs) and all(w>=0 for w in weights)
                and sum(weights)==1 and cost==sum(w*c for w,c in zip(weights,costs))
                and cost<=budget and lam>=0
                and value==sum(w*v for w,v in zip(weights,values))
                and value==intercept+lam*budget
                and all(intercept+lam*c>=v for c,v in zip(costs,values)))
    except (KeyError,TypeError,ValueError):return False


def design(menu,budget):
    """Maximize guaranteed negative contrast under a finite expected-cost budget."""
    rows,budget=normalize(menu,budget)
    costs=[r['cost'] for r in rows]
    guaranteed=_linear_optimum(costs,[-r['contrast_upper'] for r in rows],budget)
    optimistic=_linear_optimum(costs,[-r['contrast_lower'] for r in rows],budget)
    result={'budget':budget,'labels':[r['label'] for r in rows],
            'guaranteed':guaranteed,'optimistic':optimistic,
            'optimal_value_gap':optimistic['value']-guaranteed['value']}
    if not verify(menu,budget,result):raise ArithmeticError('Certificate failed')
    return result


def verify(menu,budget,cert):
    """Check exact primal/dual identities against a separately supplied menu."""
    try:
        rows,budget=normalize(menu,budget)
        costs=[r['cost'] for r in rows]
        return (rational(cert['budget'])==budget and cert['labels']==[r['label'] for r in rows]
                and _check_linear(costs,[-r['contrast_upper'] for r in rows],budget,cert['guaranteed'])
                and _check_linear(costs,[-r['contrast_lower'] for r in rows],budget,cert['optimistic'])
                and rational(cert['optimal_value_gap'])==rational(cert['optimistic']['value'])-rational(cert['guaranteed']['value']))
    except (KeyError,TypeError,ValueError):return False


def density_steps(horizons,weights):
    hs=list(map(rational,horizons));ws=list(map(rational,weights))
    if len(hs)!=len(ws) or not hs or any(h<=0 for h in hs) or any(w<0 for w in ws) or sum(ws)!=1:
        raise ValueError('Valid horizons and probability weights required')
    knots=[F(0)]+sorted(set(h for h,w in zip(hs,ws) if w>0))
    rows=[]
    for lo,hi in zip(knots,knots[1:]):
        density=sum(w/h for w,h in zip(ws,hs) if h>=hi)
        rows.append({'left':lo,'right':hi,'density':density})
    return {'steps':rows,'mass':sum((r['right']-r['left'])*r['density'] for r in rows),
            'mean_time':sum((r['right']**2-r['left']**2)*r['density']/2 for r in rows)}


def make_results():
    from .preparation import depolarizing_contrast_bounds
    menu=[]
    for k in range(1,17):
        h=F(k,4)
        lo,hi=depolarizing_contrast_bounds(F(1,10),F(2),h,F(9,10))
        # Outward rounding keeps exported certificates compact.
        scale=10**12
        lo=F(lo.numerator*scale//lo.denominator,scale)
        hi=F(-((-hi.numerator*scale)//hi.denominator),scale)
        menu.append(dict(label='T='+str(h),horizon=h,cost=h/2,contrast_lower=lo,contrast_upper=hi))
    designs=[]
    for budget in (F(3,4),F(9,10),F(1),F(9,8),F(3,2)):
        cert=design(menu,budget)
        rows=density_steps([r['horizon'] for r in menu],cert['guaranteed']['weights'])
        designs.append({'certificate':cert,'clock':rows})
    best=max((max(F(0),-r['contrast_upper'])**2/r['cost'],r['label']) for r in menu)
    return {'scope':'Exact finite-menu guaranteed-contrast design under expected active-time budget; synthetic physical model',
            'menu':menu,'designs':designs,'best_single_clock_gap_squared_per_mean_time':{'value':best[0],'label':best[1]},
            'resource_limit':'Expected pulse duration only. Add calibrated preparation, readout, control power and physical overhead; realized tail expenditure is not certified here.',
            'source':'Nonincreasing densities as uniform scale mixtures and finite-design LP are established mathematical ingredients.'}


if __name__=='__main__':
    output=Path(__file__).resolve().parents[1]/'results/clock_design.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(make_results(),default=str,indent=2)+'\n')
    print('wrote',output)
