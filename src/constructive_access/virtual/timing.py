"""Exact operational timing and exposure checks for stationary correlation tests.

This module contains finite rational clock examples and error-free algebraic
cross-checks. It does not infer a clock contract from a small measured sample.
"""
from fractions import Fraction as F
from pathlib import Path
import json


def _f(x):
    if not isinstance(x,(int,F)):
        raise TypeError('Use rational inputs')
    return F(x)


def clock_law(law):
    law=[(t,_f(w)) for t,w in law]
    if not law or any(type(t) is not int or t<0 or w<=0 for t,w in law):
        raise ValueError('Integer nonnegative durations and positive weights required')
    if sum(w for _,w in law)!=1:
        raise ValueError('Clock probabilities must sum to one')
    return law


def mean_clock(law):
    return sum((t*w for t,w in clock_law(law)),F(0))


def laplace_value(law,base):
    """E exp(-lambda T) for integer T and rational base=exp(-lambda)."""
    base=_f(base)
    if not 0<=base<=1: raise ValueError('base must lie in [0,1]')
    return sum((w*base**t for t,w in clock_law(law)),F(0))


def iid_increment_moments(law,base,count):
    if type(count) is not int or count<1: raise ValueError('positive count required')
    phi=laplace_value(law,base)
    return [phi**k for k in range(1,count+1)]


def random_scale_moments(law,base,count):
    if type(count) is not int or count<1: raise ValueError('positive count required')
    law=clock_law(law); base=_f(base)
    if not 0<=base<=1: raise ValueError('base must lie in [0,1]')
    return [sum((w*base**(k*t) for t,w in law),F(0)) for k in range(1,count+1)]


def finite_differences(values):
    """Return signed forward differences: every entry must be >=0 for moments.

    These are necessary checks, not a complete finite moment-membership test."""
    values=list(map(_f,values)); rows=[values]
    while len(values)>1:
        values=[values[j]-values[j+1] for j in range(len(values)-1)]
        rows.append(values)
    return rows


def determinant(matrix):
    a=[list(map(_f,row)) for row in matrix]; n=len(a)
    if any(len(row)!=n for row in a): raise ValueError('square matrix required')
    det=F(1)
    for k in range(n):
        pivot=next((j for j in range(k,n) if a[j][k]),None)
        if pivot is None: return F(0)
        if pivot!=k: a[k],a[pivot]=a[pivot],a[k];det=-det
        value=a[k][k];det*=value
        for i in range(k+1,n):
            ratio=a[i][k]/value
            for j in range(k+1,n): a[i][j]-=ratio*a[k][j]
    return det


def endpoint_block_correlation(base,length,start_separation,antisymmetric=False):
    """Exact symmetric two-state correlations for (X0 +/- XL)/2 block channels.

    Outputs of each block are independent randomized binary encodings of that
    block functional. X takes values +/-1. Overlap is allowed in this evaluator
    deliberately, so it can demonstrate failure of the disjoint-block theorem.
    """
    base=_f(base)
    if not 0<base<=1 or type(length) is not int or length<0 or type(start_separation) is not int or start_separation<0:
        raise ValueError('positive base and integer nonnegative times required')
    L,D=length,start_separation
    sign=-1 if antisymmetric else 1
    return (2*base**D+sign*(base**(D+L)+base**abs(D-L)))/4


def stationary_rate_bound(information_lower,weights,mean_gaps,block_length=F(0)):
    """4 times a certified record KL separation / full mean path duration."""
    info,L=map(_f,(information_lower,block_length))
    weights=list(map(_f,weights)); gaps=list(map(_f,mean_gaps))
    if info<0 or L<0 or not weights or len(weights)!=len(gaps) or any(w<=0 for w in weights) or sum(weights)!=1 or any(t<0 for t in gaps):
        raise ValueError('invalid information, weights, means or exposure length')
    span=2*L+sum((w*t for w,t in zip(weights,gaps)),F(0))
    if span<=0: raise ValueError('positive mean full duration required')
    return {'mean_full_duration':span,'stationary_entropy_rate_lower':4*info/span}


def make_results():
    base=F(1,2)
    increments=[(1,F(3,4)),(3,F(1,4))]
    scale=[(1,F(1,2)),(2,F(1,2))]
    iid=iid_increment_moments(increments,base,6)
    scaled=random_scale_moments(scale,base,6)
    # Underlying two-state process with fixed + preparation and + occupation.
    p=[(1+sum((w*base**(k*t) for t,w in scale),F(0)))/2 for k in range(5)]
    hankel=[[p[i+j] for j in range(3)] for i in range(3)]
    false_revival=[[(1,F(1))],[(0,F(4,5)),(10,F(1,5))]]
    false_convex=[[(1,F(1))],[(1,F(9,10)),(11,F(1,10))],[(2,F(9,10)),(12,F(1,10))]]
    revival=[laplace_value(law,base) for law in false_revival]
    convex=[laplace_value(law,base) for law in false_convex]
    disjoint=[endpoint_block_correlation(base,4,4+k) for k in range(1,5)]
    overlap=[endpoint_block_correlation(base,4,D) for D in [2,4]]
    anti=endpoint_block_correlation(base,2,3,True)
    center=[endpoint_block_correlation(base,2,D) for D in [2,4]]
    return {
      'evidence':'exact synthetic population examples; no sampled or measured physical data',
      'safe_iid_increment':{'law':increments,'mean_increment':mean_clock(increments),'correlations':iid,'signed_differences':finite_differences(iid)},
      'safe_random_scale_thermodynamics':{'law':scale,'correlations':scaled,'signed_differences':finite_differences(scaled)},
      'unsafe_random_scale_for_same_dimension':{'occupation':p,'hankel':hankel,'hankel_determinant':determinant(hankel),'physical_states':2,'common_channel_minimum_dimension':3,'three_state_realization':{'P':[[F(1,2),F(0),F(1,2)],[F(0),F(1,4),F(3,4)],[F(0),F(0),F(1)]],'initial':[F(1,2),F(1,2),F(0)],'detector':[F(1),F(1),F(1,2)]}},
      'unsafe_settingwise_ordered_means':{'laws':false_revival,'mean_lags':[mean_clock(law) for law in false_revival],'correlations':revival,'increase':revival[1]-revival[0],'physical_entropy_rate':F(0)},
      'unsafe_stochastically_ordered_clocks_for_full_cone':{'laws':false_convex,'mean_lags':[mean_clock(law) for law in false_convex],'correlations':convex,'second_difference':convex[0]-2*convex[1]+convex[2],'physical_entropy_rate':F(0)},
      'safe_disjoint_symmetric_blocks':{'length':4,'gaps':[1,2,3,4],'correlations':disjoint,'signed_differences':finite_differences(disjoint)},
      'unsafe_overlapping_symmetric_blocks':{'length':4,'start_separations':[2,4],'correlations':overlap,'increase':overlap[1]-overlap[0],'physical_entropy_rate':F(0)},
      'unsafe_center_spacing_index':{'length':2,'start_separations':[2,4],'actual_gaps':[0,2],'correlations':center,'normalized_second_moment_gap':center[1]-center[0]**2,'physical_entropy_rate':F(0)},
      'unsafe_antisymmetric_block':{'length':2,'gap':1,'correlation':anti,'same_probability':(1+anti)/2,'physical_entropy_rate':F(0)},
      'duration_accounting_example':stationary_rate_bound(F(1,100),[F(1,2),F(1,2)],[F(1),F(2)],F(2))}


def _jsonify(x):
    if isinstance(x,F): return {'fraction':str(x),'decimal':float(x)}
    if isinstance(x,dict): return {k:_jsonify(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_jsonify(v) for v in x]
    return x

if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'results'/'timing.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(_jsonify(make_results()),indent=2)+'\n')
    print(out)
