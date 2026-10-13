"""Deterministic design diagnostics for exposed cosine simple hypotheses.

Exact analytical bounds are stated in the manuscript. mpmath evaluates them
at high precision here; these values are NOT certified interval enclosures,
Monte Carlo records, measured performance, or proofs of optimal constants.
The comparison is omega versus 2*omega, NOT signal versus an absent signal.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import hashlib
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
INDICES=(1,2,4,8,16,32)
DPS=100


def variance(omega,gamma,tau):
    """Stationary exposure variance at resonance omega*tau=2*pi*integer ONLY.

    This closed form omits nonresonant oscillatory terms; use the general
    complex response formula in the manuscript away from resonance.
    """
    b=-mp.expm1(-gamma*tau)
    return ((gamma*tau-b)*gamma**2+(gamma*tau+b)*omega**2)/(tau**2*(gamma**2+omega**2)**2)


def prepared_mean(omega,gamma,tau):
    """Phase-pi/2 mean at resonance omega*tau=2*pi*integer ONLY.

    This closed form omits the nonresonant exposure endpoint phase term.
    """
    return mp.expm1(-gamma*tau)*omega/(tau*(gamma**2+omega**2))


def bounds(index,gamma=1,tau=1,noise='0.5',error='0.1'):
    if isinstance(index,bool) or not isinstance(index,int) or index<1:
        raise ValueError('A positive integer resonance index is required')
    g,t,nu,eps=map(mp.mpf,(gamma,tau,noise,error))
    if not g>0 or not t>0 or not nu>0 or not 0<eps<mp.mpf('.5'):
        raise ValueError('Positive gamma, exposure width and noise; error in (0,1/2) required')
    omega=2*mp.pi*index/t
    v1=variance(omega,g,t);v2=variance(2*omega,g,t)
    mu1=prepared_mean(omega,g,t);mu2=prepared_mean(2*omega,g,t)
    gap_v=abs(v1-v2);gap_mu=abs(mu1-mu2)
    c=mp.cosh(1/nu**2)-1
    k=(2+g*t+mp.sqrt(2*g*t))/t
    square_variance=lambda v: (1+4*nu**2)*v-v*v+2*nu**4
    worst_square_variance=max(square_variance(v1),square_variance(v2))
    passive_lower=2*(1-2*eps)**2/(c*(v1+v2)**2)
    passive_upper=4*worst_square_variance/(eps*gap_v**2) if gap_v else mp.inf
    prepared_lower=16*nu**2*(1-2*eps)**2*omega**2/(9*k**2)
    prepared_upper=8*(1+nu**2)*mp.log(1/eps)/gap_mu**2 if gap_mu else mp.inf
    return dict(index=index,omega=omega,variance_omega=v1,variance_double=v2,
                variance_gap=gap_v,square_variance_upper=worst_square_variance,prepared_mean_omega=mu1,
                prepared_mean_double=mu2,prepared_mean_gap=gap_mu,
                passive_necessary_threshold=passive_lower,
                passive_sufficient_threshold=passive_upper,
                prepared_necessary_threshold=prepared_lower,
                prepared_sufficient_threshold=prepared_upper)


def render(value):
    if isinstance(value,int):return value
    return mp.nstr(value,42)


def report():
    with mp.workdps(DPS):
        rows=[]
        for m in INDICES:
            row=bounds(m)
            displayed={key:render(value) for key,value in row.items()}
            for method in ('passive','prepared'):
                for kind in ('necessary','sufficient'):
                    threshold=row[f'{method}_{kind}_threshold']
                    displayed[f'{method}_{kind}_numerical_ceiling']=int(mp.ceil(threshold))
            rows.append(displayed)
        return dict(status='Analytical design bounds; high-precision numerical displays, not certified rounding or experimental data.',
            hypotheses='Known omega=2*pi*m versus 2*omega. Each trial independently resets to stationarity (passive) or known phase pi/2 (prepared).',
            parameters=dict(gamma=1,exposure_width=1,noise_standard_deviation='1/2',equal_prior_Bayes_error='1/10'),
            precision_decimal_digits=DPS,mpmath_version=mp.__version__,rows=rows,
            exact_formulas=dict(passive_necessary='2(1-2eps)^2 / [(cosh(1/nu^2)-1)(v(w)+v(2w))^2]',
                passive_sufficient='4 max[Snu(v(w)),Snu(v(2w))]/(eps (v(w)-v(2w))^2); Snu(v)=(1+4nu^2)v-v^2+2nu^4',
                prepared_necessary='16nu^2(1-2eps)^2 w^2/(9K^2), K=(2+gamma*tau+sqrt(2gamma*tau))/tau',
                prepared_sufficient='8(1+nu^2)log(1/eps)/(mu(w)-mu(2w))^2'),
            count_scope='Necessary bounds concern any decision rule in the declared independent-trial experiment. Sufficient bounds concern specific second-moment or sample-mean threshold tests. The intervals do not assert exact minimax counts.',
            calibration_scope='Known clock, exposure, gamma, noise law, simple hypotheses and, for preparation, phase pi/2. Calibration and preparation are additional access, not inferred from these tests.',
            energy_scope='At fixed positive read-noise, passive count is Theta(omega^4) and prepared count Theta(omega^2). If a separately declared sensor obeys nu^2=kappa/E, optimized total read-energy exponents can both be Theta(omega^2). Neither statement includes preparation, reset, driving, clock, exposure or dissipation calibration costs.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'design/results.json')
    args=p.parse_args();r=report()
    args.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    for row in r['rows']:
        print(row['index'],*(row[f'{method}_{kind}_numerical_ceiling']
            for method in ('passive','prepared') for kind in ('necessary','sufficient')))

if __name__=='__main__':main()
