"""Exact finite-clock exclusion certificates for small classical CTMC realizations.

All theorem certificates use fractions. Numerical examples are labelled separately.
The classical generator spectral sector is established prior mathematics; the
finite Erlang moment robustness argument is derived in the accompanying proof.
"""
from fractions import Fraction as F
from math import isqrt, comb
import json
from pathlib import Path


def _fraction(x):
    if not isinstance(x, (int, F)):
        raise TypeError('Exact certificate inputs must be integers or Fractions')
    return F(x)


def sqrt_bounds(x, bits=96):
    x = _fraction(x)
    if x < 0 or bits < 1:
        raise ValueError('nonnegative radicand and positive precision required')
    scale = 1 << bits
    q = isqrt(x.numerator * scale * scale // x.denominator)
    lo = F(q, scale)
    return lo, lo if lo * lo == x else F(q + 1, scale)


def atan_bounds(x, terms=32):
    x = _fraction(x)
    if not 0 <= x <= 1 or terms < 1:
        raise ValueError('alternating arctangent requires 0 <= x <= 1')
    a = sum(((-1)**j * x**(2*j+1) / (2*j+1) for j in range(terms)), F(0))
    b = a + (-1)**terms * x**(2*terms+1) / (2*terms+1)
    return min(a,b), max(a,b)


def tangent_lower(n):
    """Lower bound tan(pi/n), n>=3, from Machin and tan x>=x+x^3/3."""
    if not isinstance(n, int) or n < 3:
        raise ValueError('n must be an integer at least 3')
    if n == 4:
        return F(1)
    a, A = atan_bounds(F(1,5))
    b, B = atan_bounds(F(1,239))
    x = (16*a-4*B) / n
    value=x+x**3/3
    scale=1<<96
    return F(value.numerator*scale//value.denominator,scale)


def target_moments(n, gamma, omega, rate):
    """P(+Z at Erlang(k,rate)) for +Z-prepared depolarizing X-rotation."""
    gamma, omega, rate = map(_fraction, (gamma,omega,rate))
    if gamma < 0 or omega <= 0 or rate <= 0:
        raise ValueError('gamma>=0, omega>0, rate>0 required')
    den = (rate+gamma)**2+omega**2
    u,v = rate*(rate+gamma)/den, rate*omega/den
    re,im = F(1),F(0)
    moments=[]
    for k in range(n+2):
        moments.append((1+re)/2)
        re,im = re*u-im*v,re*v+im*u
    return u,v,moments


def certificate(n, gamma, omega, rate, bits=96):
    """Uniform l-infinity distance lower bound to every <=n-state CTMC record.

    The settings are k=0,...,n+1 independent Erlang clocks. A fixed preparation
    and fixed probabilistic binary detector are permitted in the classical null.
    No exit-rate calibration, reversibility or stationary preparation is assumed.
    """
    if not isinstance(n,int) or n < 3:
        raise ValueError('n must be an integer at least 3')
    u,v,moments=target_moments(n,gamma,omega,rate)
    t=tangent_lower(n)
    # Resolvent spectrum lies in disk |z-(1/2-it/2)| <= sqrt(1+t*t)/2.
    radius_sq=(1+t*t)/4
    offset_sq=(u-F(1,2))**2+(v+t/2)**2
    violation=offset_sq-radius_sq
    result={'n':n,'gamma':_fraction(gamma),'omega':_fraction(omega),
            'rate':_fraction(rate),'tangent_lower':t,'mu_real':u,'mu_imag':v,
            'moments':moments,'disk_violation':violation}
    if violation <= 0:
        return dict(result,status='not_separated',gap=F(0))
    _,radius_hi=sqrt_bounds(radius_sq,bits)
    _,offset_hi=sqrt_bounds(offset_sq,bits)
    distance=min(v,violation/(radius_hi+offset_hi))
    scale=1<<bits
    distance=F(distance.numerator*scale//distance.denominator,scale)
    if distance<=0:
        return dict(result,status='insufficient_arithmetic_precision',gap=F(0))
    one_distance,_=sqrt_bounds((1-u)**2+v*v,bits)
    _,residual_factor=sqrt_bounds(1+(1+u)**2/(v*v),bits)
    pairs=(n-1)//2
    numerator=one_distance*distance**pairs*v**(n-1-pairs)
    gap_simple=numerator/(2**(n+1)*residual_factor)
    # Normalize by P(-1), then Cayley-transform every characteristic root.
    scale_sq=(1+u)**2+v*v
    nu_real=(1-u*u-v*v)/scale_sq
    nu_imag_abs=2*v/scale_sq
    normalized_offset_sq=(nu_real-F(1,2))**2+(nu_imag_abs+t/2)**2
    normalized_violation=normalized_offset_sq-radius_sq
    _,normalized_offset_hi=sqrt_bounds(normalized_offset_sq,bits)
    normalized_distance=min(nu_imag_abs,normalized_violation/(normalized_offset_hi+radius_hi))
    normalized_distance=F(normalized_distance.numerator*scale//normalized_distance.denominator,scale)
    _,scale_hi=sqrt_bounds(scale_sq,bits)
    real_factor=v/scale_hi
    pair_factor=scale_sq/4*normalized_distance*(2*nu_imag_abs-normalized_distance)
    gap_normalized=one_distance*real_factor**(n-1-2*pairs)*pair_factor**pairs/(4*residual_factor)
    gap=max(gap_simple,gap_normalized)
    return dict(result,status='separated',spectral_distance=distance,
                pair_count=pairs,one_distance=one_distance,
                residual_factor=residual_factor,gap_simple=gap_simple,
                cayley_real=nu_real,cayley_imag_abs=nu_imag_abs,
                cayley_distance=normalized_distance,real_factor=real_factor,
                pair_factor=pair_factor,gap_normalized=gap_normalized,gap=gap)


def exp_minus_upper(x,terms=None):
    """e^-x <= reciprocal of the positive partial series of e^x."""
    x=_fraction(x)
    if x < 0:
        raise ValueError('nonnegative x required')
    if terms is None:
        terms=max(32,4*((x.numerator+x.denominator-1)//x.denominator)+16)
    term=total=F(1)
    for j in range(1,terms+1):
        term*=x/j
        total+=term
    return 1/total


def erlang_tail_upper(k,rate,horizon):
    rate,horizon=map(_fraction,(rate,horizon))
    if not isinstance(k,int) or k<0 or rate<=0 or horizon<0:
        raise ValueError('invalid Erlang clock')
    if k==0:
        return F(0)
    x=rate*horizon
    term=total=F(1)
    for j in range(1,k):
        term*=x/j
        total+=term
    return min(F(1),exp_minus_upper(x)*total)


def bounded_design(cert,alpha=F(1,20)):
    """Conservative finite-horizon finite-record plan for the nominal target.

    At T>L record 0 without running the system longer. That changes each moment
    by at most the Erlang tail. N independent Bernoulli trials are used per setting.
    No calibration allowance is included: add it explicitly to the error budget.
    """
    if cert['status']!='separated':
        raise ValueError('a positive separation certificate is required')
    alpha=_fraction(alpha)
    if not 0<alpha<1:
        raise ValueError('alpha must lie strictly between zero and one')
    gap=cert['gap']; kmax=cert['n']+1; rate=cert['rate']
    x=8
    while erlang_tail_upper(kmax,rate,F(x)/rate)>gap/8:
        x*=2
    horizon=F(x)/rate
    tail=erlang_tail_upper(kmax,rate,horizon)
    statistical_radius=gap/4
    settings=cert['n']+2
    # e^-B < 2^-B and a union bound give confidence >=1-alpha.
    B=1
    while F(2*settings,2**B)>alpha:
        B+=1
    raw=F(B,2*statistical_radius**2)
    records=(raw.numerator+raw.denominator-1)//raw.denominator
    return {'settings':settings,'records_per_setting':records,
            'total_records':records*settings,'maximum_duration':horizon,
            'tail_upper':tail,'statistical_radius':statistical_radius,
            'error_probability':alpha,'hoeffding_exponent_lower':
            2*records*statistical_radius**2,
            'power_target': 'nominal target; compare confidence box with target moments',
            'nominal_acceptance_radius':statistical_radius+tail,
            'symmetric_calibration_margin_for_nominal_power':gap/4-tail}


def compatible_excluded(cert,estimates,statistical_radius,
                        calibration_radius=F(0),truncation_radius=F(0)):
    """Whether a simultaneous confidence box lies within the forbidden target ball."""
    estimates=list(map(_fraction,estimates))
    if len(estimates)!=len(cert['moments']):
        raise ValueError('one estimate per declared clock setting required')
    radii=list(map(_fraction,(statistical_radius,calibration_radius,truncation_radius)))
    if min(radii)<0:
        raise ValueError('error radii must be nonnegative')
    distance=max(abs(x-y) for x,y in zip(estimates,cert['moments']))
    return cert['status']=='separated' and distance+sum(radii)<cert['gap']


def _jsonify(obj):
    if isinstance(obj,F):
        return {'fraction':str(obj),'decimal':float(obj)}
    if isinstance(obj,dict):
        return {k:_jsonify(v) for k,v in obj.items()}
    if isinstance(obj,list):
        return [_jsonify(x) for x in obj]
    return obj


def make_results():
    rows=[]
    for n,rate in [(4,F(13,10)),(5,F(133,100)),(6,F(67,50)),
                   (8,F(34,25)),(10,F(69,50)),(16,F(139,100)),
                   (32,F(7,5)),(60,F(141,100)),(62,F(141,100))]:
        cert=certificate(n,F(1,10),F(2),rate)
        row={'n_excluded':n,'required_classical_states':n+1,
             'rate':rate,'guaranteed_uniform_gap':cert['gap'],
             'tangent_lower':cert['tangent_lower'],
             'clock_settings':n+2}
        if n<=8:
            row['design']=bounded_design(cert)
        rows.append(row)
    return {'scope':'mathematical certificates and synthetic qubit targets; no empirical data',
            'rows':rows,'n4_certificate':certificate(4,F(1,10),F(2),F(13,10)),
            'boundary_control':certificate(4,F(2),F(2),F(2)),
            'novelty':'spectral wedge and exact spectral dimension separation are prior art; finite Erlang robustness proof is a candidate contribution, not an established priority claim'}


if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'results'/'spectral.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(_jsonify(make_results()),indent=2)+'\n')
    print(out)
