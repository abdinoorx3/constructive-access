"""Exploratory exact-enclosure model for replicated binary exposure readouts.

Stationary circle dphi=omega dt+sqrt(2gamma)dW; three windows start at0,h,2h
and have width tau<=h. Each binary detector replica has conditional mean equal
to a declared gain times its exposure average and is independent of other replicas given the whole
latent path. Endpoints have one replica; the middle window has an even quota.

Finite Fourier/additive-functional moments give an EXACT finite ODE system for
all event probabilities. Rational Taylor actions enclose its solution; there is
no Brownian path mesh. Dyadic sampling approximates the enclosed law with an
explicit total-variation radius. No benchmark or physical calibration is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import ceil, comb
import argparse
import json
from pathlib import Path


def exact(value):
    if isinstance(value,bool) or not isinstance(value,(int,str,F)):
        raise TypeError('Use integers, Fractions, or rational strings')
    return F(value)


@dataclass(frozen=True)
class Model:
    gamma:F
    omega:F
    lag:F
    exposure:F
    center_replicas:int=2
    gain:F=F(1)

    def __post_init__(self):
        for name in ('gamma','omega','lag','exposure','gain'):
            object.__setattr__(self,name,exact(getattr(self,name)))
        if not 0<=self.gain<=1:
            raise ValueError('The declared binary gain must lie in [0,1]')
        if self.gamma<=0 or not 0<self.exposure<=self.lag:
            raise ValueError('gamma>0 and0<exposure<=lag are required')
        if type(self.center_replicas) is not int or self.center_replicas not in (2,4,6):
            raise ValueError('Use2,4 or6 independent center replicas')

    @property
    def degree(self):return self.center_replicas+2
    @property
    def quotas(self):return (1,self.center_replicas,1)


def state_space(model):
    # To obtain moments of total degree<=D at mode zero, only |k|+degree<=D
    # is needed: every coupling lowers polynomial degree and shifts mode by1.
    states=[]
    for alpha in product(*(range(n+1) for n in model.quotas)):
        for mode in range(model.degree-sum(alpha)+1):
            states.append((alpha,mode,0)) # cosine
            if mode:states.append((alpha,mode,1)) # sine
    return states,{state:i for i,state in enumerate(states)}


def generator_rows(model,states,index,active):
    rows=[]
    for alpha,k,part in states:
        row={}
        def add(a,mode,component,value):
            if value==0 or (component==1 and mode==0):return
            if mode<0:
                mode=-mode
                if component==1:value=-value
            key=(a,mode,component)
            if key not in index:raise AssertionError('Moment cone does not close')
            i=index[key];row[i]=row.get(i,F(0))+value
        add(alpha,k,part,-model.gamma*k*k)
        add(alpha,k,1-part,(1 if part else -1)*model.omega*k)
        if active is not None and alpha[active]:
            beta=list(alpha);beta[active]-=1;beta=tuple(beta)
            coeff=F(alpha[active],2)
            add(beta,k+1,part,coeff);add(beta,k-1,part,coeff)
        rows.append(tuple((i,c) for i,c in row.items() if c))
    return tuple(rows)


def matrix_norm(rows):
    return max((sum((abs(c) for _,c in row),F(0)) for row in rows),default=F(0))


def apply(rows,vector,scale):
    return [scale*sum((c*vector[i] for i,c in row),F(0)) for row in rows]


def taylor_action(rows,time,vector,tolerance):
    """Rational approximation to exp(time*G)v and rigorous infinity error.

    exp(x)<=3^ceil(x) forx>=0. Taylor's remainder is at most
    ||v||inf exp(x) x^(N+1)/(N+1)!, x=time*||G||inf.
    """
    x=time*matrix_norm(rows)
    growth=F(3**ceil(x)) if x else F(1)
    norm=max(map(abs,vector),default=F(0))
    order=0;tail=norm*growth*x
    while tail>tolerance:
        order+=1;tail*=x/F(order+1)
        if order>512:raise ValueError('Requested model/precision exceeds exploratory512-term cap')
    current=vector[:];result=vector[:]
    for degree in range(1,order+1):
        current=apply(rows,current,time/F(degree))
        result=[a+b for a,b in zip(result,current)]
    return result,tail,growth,order


def moments(model,tv_tolerance=F(1,1<<90)):
    """Enclose all normalized exposure moments needed for the finite record."""
    tolerance=exact(tv_tolerance)
    if not 0<tolerance<1:raise ValueError('Use a TV tolerance in(0,1)')
    states,index=state_space(model)
    stages=[]
    for j in range(3):
        stages.append((generator_rows(model,states,index,j),model.exposure,j))
        if j<2 and model.lag>model.exposure:
            stages.append((generator_rows(model,states,index,None),model.lag-model.exposure,None))
    growths=[F(3**ceil(time*matrix_norm(rows))) for rows,time,_ in stages]
    growth_total=F(1)
    for g in growths:growth_total*=g
    normalization=max(F(1),model.exposure**(-model.degree))
    stage_tolerance=tolerance/(len(stages)*growth_total*(1<<(model.degree-1))*normalization)
    vector=[F(0)]*len(states);vector[index[((0,0,0),0,0)]]=F(1)
    radius=F(0);stage_reports=[]
    for rows,time,active in stages:
        vector,tail,growth,order=taylor_action(rows,time,vector,stage_tolerance)
        radius=growth*radius+tail
        stage_reports.append(dict(active_window=active,time=time,taylor_order=order,
                                  matrix_norm=matrix_norm(rows),remainder_radius=tail))
    output={}
    for alpha in product(*(range(n+1) for n in model.quotas)):
        scale=model.exposure**sum(alpha)
        output[alpha]=(vector[index[(alpha,0,0)]]/scale,radius/scale)
    return output,dict(real_ode_states=len(states),max_polynomial_degree=model.degree,
        raw_moment_radius=radius,requested_tv_tolerance=tolerance,stages=stage_reports)


def polynomial(signs):
    result=[1]
    for sign in signs:
        new=[0]*(len(result)+1)
        for i,c in enumerate(result):new[i]+=c;new[i+1]+=sign*c
        result=new
    return result


def outcome_law(model,tv_tolerance=F(1,1<<90),dyadic_bits=63,collapse_centers=False):
    """Binary outcome law with probability radii and an exact dyadic alternative.

    If collapse_centers=True, an outcome is(left sign,center plus count,right
    sign), representing all permutations of exchangeable center replicas.
    Otherwise each outcome retains all ordered detector signs.
    """
    if type(dyadic_bits) is not int or not 1<=dyadic_bits<=63:
        raise ValueError('The integer sampler supports1..63 bits')
    table,diagnostics=moments(model,tv_tolerance)
    if type(collapse_centers) is not bool:
        raise TypeError('collapse_centers must be Boolean')
    outcomes=[];nominals=[];radii=[]
    denominator=1<<model.degree
    if collapse_centers:
        record_specs=[]
        for left,plus,right in product((-1,1),range(model.center_replicas+1),(-1,1)):
            record=(left,)+(1,)*plus+(-1,)*(model.center_replicas-plus)+(right,)
            record_specs.append((record,(left,plus,right),comb(model.center_replicas,plus)))
    else:
        record_specs=((r,r,1) for r in product((-1,1),repeat=model.degree))
    for record,label,multiplicity in record_specs:
        coefficients=(polynomial(record[:1]),polynomial(record[1:-1]),polynomial(record[-1:]))
        point=F(0);radius=F(0)
        for alpha in product(*(range(n+1) for n in model.quotas)):
            coeff=coefficients[0][alpha[0]]*coefficients[1][alpha[1]]*coefficients[2][alpha[2]]*model.gain**sum(alpha)
            mu,error=table[alpha];point+=coeff*mu;radius+=abs(coeff)*error
        point*=F(multiplicity,denominator);radius*=F(multiplicity,denominator)
        outcomes.append(label);nominals.append(point);radii.append(radius)
    if sum(nominals,F(0))!=1:raise AssertionError('Event probabilities must sum exactly toone')
    positive=[max(F(0),p) for p in nominals];mass=sum(positive,F(0))
    grid=1<<dyadic_bits
    counts=[int((p/mass)*grid) for p in positive]
    counts[max(range(len(counts)),key=lambda i:positive[i])]+=grid-sum(counts)
    tv=min(F(1),sum((abs(F(q,grid)-p)+e for q,p,e in zip(counts,nominals,radii)),F(0))/2)
    return dict(model=dict(gamma=model.gamma,omega=model.omega,lag=model.lag,
                          exposure=model.exposure,center_replicas=model.center_replicas,gain=model.gain),
        outcomes=outcomes,collapsed_centers=collapse_centers,probability_centers=nominals,probability_radii=radii,
        dyadic_bits=dyadic_bits,dyadic_numerators=counts,total_variation_upper=tv,
        diagnostics=diagnostics,
        resource_per_record=dict(independent_stationary_preparations=1,
            binary_detector_readouts=model.degree,exposure_windows=3,
            modeled_evolution_time=2*model.lag+model.exposure,
            uncounted='Preparation, resets, shutter timing, replica construction, energy and geometry calibration are additional.'),
        scope='Circle simulator only; inference validity for a broader Gamma-closure class needs its own theorem. Binary replicas are independent given the latent path and have declared conditional mean gain times exposure. Nonunit gain requires separate inference calibration; the simulator does not infer it from target records.')


def elementary_symmetric(plus,replicas,degree):
    """Average product over all degree-sized subsets of exchangeable binary reads."""
    if any(type(x) is not int for x in (plus,replicas,degree)) or not 0<=plus<=replicas or not 0<=degree<=replicas:
        raise ValueError('Valid integer center counts and polynomial degree required')
    numerator=sum((-1)**(degree-j)*comb(plus,j)*comb(replicas-plus,degree-j)
                  for j in range(max(0,degree-(replicas-plus)),min(degree,plus)+1))
    return F(numerator,comb(replicas,degree))


def record_statistics(record,center_replicas,lag,drift_rule=1):
    """Raw symmetric score vector on a collapsed(left,plus,right) record.

    No gain normalization, exposure correction or confidence calculation occurs
    here. S_j has conditional gain degree2j+2; U_j has degree2j; V has degree2.
    Those corrections belong to the separately covered inference theorem.
    """
    if len(record)!=3 or record[0] not in (-1,1) or record[2] not in (-1,1):
        raise ValueError('A collapsed binary record is required')
    h,lam=map(exact,(lag,drift_rule))
    if h<=0 or type(center_replicas) is not int or center_replicas not in (2,4,6):
        raise ValueError('Positive lag and2,4,6 center replicas required')
    left,plus,right=record;b=1-lam*h
    e=[elementary_symmetric(plus,center_replicas,j) for j in range(center_replicas+1)]
    S=[(-left*right*e[2*j]+b*(left+right)*e[2*j+1]-b*b*e[2*j+2])/(h*h)
       for j in range(center_replicas//2)]
    U=[e[2*j] for j in range(center_replicas//2)]
    V=(e[2]-F(left+right,2)*e[1])/h
    return dict(S=S,U=U,V=V)


def reference_law(gain,exposure=None,step_after=None):
    """Exact Bernoulli calibration laws for the declared controlled references.

    Constant reference f=1 has mean gain. If exposure and step_after are
    supplied, a calibrated step f:0->1 leaves step_after units at1 inside the
    window and its mean is gain*step_after/exposure. This controlled reference
    is an additional apparatus operation, not a stationary circle record.
    """
    gain=exact(gain)
    if not 0<=gain<=1:raise ValueError('Reference gain must lie in [0,1]')
    if exposure is None and step_after is None:mean=gain
    elif exposure is not None and step_after is not None:
        exposure,step_after=map(exact,(exposure,step_after))
        if not 0<step_after<exposure:raise ValueError('Require0<known step duration<exposure')
        mean=gain*step_after/exposure
    else:raise ValueError('Supply both exposure and step duration, or neither')
    return {-1:(1-mean)/2,1:(1+mean)/2}


def sample_counts(law,records,seed,batch_size=1000000):
    """Integer-only categorical draws from the exact dyadic approximation.

    Uniform random bits are supplied by NumPy PCG64. As with every numerical
    simulator, pseudorandomness is an implementation assumption. Conditional on
    ideal independent uniform bits, no floating-probability sampling error occurs.
    The product-law distance from the physical record model is<=n*the statedTV.
    """
    import numpy as np
    if type(records) is not int or records<0:raise ValueError('Nonnegative integer record quota required')
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size required')
    bits=law['dyadic_bits'];numerators=law['dyadic_numerators']
    if sum(numerators)!=1<<bits or min(numerators)<0:raise ValueError('Invalid dyadic law')
    cdf=np.cumsum(np.array(numerators,dtype=np.uint64),dtype=np.uint64)
    rng=np.random.PCG64(seed);counts=np.zeros(len(numerators),dtype=np.int64)
    for start in range(0,records,batch_size):
        n=min(batch_size,records-start)
        raw=rng.random_raw(n)>>np.uint64(64-bits)
        indices=np.searchsorted(cdf,raw,side='right')
        counts+=np.bincount(indices,minlength=len(counts))
    return dict(counts=counts.tolist(),records=records,seed=seed,
                simulation_product_law_tv_upper=min(F(1),records*law['total_variation_upper']),
                sampling='PCG64 uniform bits and exact integer CDF; no floating categorical threshold')


def encode(value):
    if isinstance(value,F):return str(value)
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gamma',default='1');p.add_argument('--omega',default='3')
    p.add_argument('--lag',default='1/10');p.add_argument('--exposure',default='1/1000')
    p.add_argument('--center-replicas',type=int,default=2)
    p.add_argument('--gain',default='1')
    p.add_argument('--collapse-centers',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    model=Model(args.gamma,args.omega,args.lag,args.exposure,args.center_replicas,args.gain)
    report=outcome_law(model,collapse_centers=args.collapse_centers)
    text=json.dumps(report,indent=2,default=encode)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')

if __name__=='__main__':main()
