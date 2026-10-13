"""Predictable contrast learning with an anytime entropy-rate certificate.

Only the declared statistical and mathematical contract is checked. Physical
stationarity, ordinary reversal, detector noninvasiveness, reset/dependence and
duration calibration remain external premises. No arbitrary postselection.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual import contrast


def rounded_ratio(numerator, denominator):
    """Nearest integer with ties rounded upward, for a nonnegative ratio."""
    if numerator < 0 or denominator <= 0:
        raise ValueError('nonnegative numerator and positive denominator required')
    return (2*numerator+denominator)//(2*denominator)


def policy(counts, batches, mode='learned', grid=48, bet_grid=128):
    """A reproducible decision using only completed past batches.

    The three gap indices are1,2,3. A shifted cubic has a=(-s²,2s,-1)
    and exact reversible support b=-(1-s)²/2. A rational grid and a clipped
    plug-in Hoeffding bet give deterministic, replayable decisions. Neither
    the plug-in estimate nor the policy is asserted globally optimal.
    """
    if type(batches) is not int or batches < 0:
        raise ValueError('nonnegative number of completed batches required')
    if len(counts) != 3 or any(type(x) is not int or not 0 <= x <= batches for x in counts):
        raise ValueError('three valid past counts required')
    if type(grid) is not int or not 3 <= grid <= 96 or grid % 3:
        raise ValueError('contrast grid must be a multiple of3 in3..96')
    if type(bet_grid) is not int or not 1 <= bet_grid <= 256:
        raise ValueError('bet grid must lie in1..256')
    if mode not in ('learned','fixed'):
        raise ValueError('unknown policy')
    centered = [2*x-batches for x in counts]
    if mode == 'fixed' or batches == 0:
        r = 2*grid//3
    elif centered[0] > 0 and centered[1] > 0:
        r = min(grid, rounded_ratio(grid*centered[1], centered[0]))
    else:
        r = 0
    numerator = -r*r*centered[0] + 2*r*grid*centered[1] - grid*grid*centered[2]
    norm = r**4 + 4*r*r*grid*grid + grid**4
    ell = min(bet_grid, rounded_ratio(2*grid*grid*bet_grid*max(0,numerator), (batches+2)*norm))
    s=F(r,grid)
    return dict(s=s, coefficients=[-s*s,2*s,F(-1)], support=-(1-s)**2/2,
                bet=F(ell,bet_grid), grid_index=r, bet_index=ell)


def certify(records, durations=(F(3,20),F(3,10),F(9,20)),
            alpha=F(1,20), conditional_radii=(0,0,0), mode='learned',
            grid=48, bet_grid=128):
    """Replay a predictable policy and certify its current anytime lower bound.

    Each row is one complete chronological batch of three agreement bits.
    Conditional on all previous readouts, each bit's mean must be within its
    declared radius of the corresponding fixed target probability. This is
    stronger than stationarity of the marginal means. No independence beyond
    that conditional requirement is used. Durations bound each full path.
    """
    times=list(map(ec.exact,durations));delta=list(map(ec.exact,conditional_radii))
    alpha=ec.exact(alpha)
    if len(times)!=3 or any(t<=0 for t in times):
        raise ValueError('three positive full durations required')
    if len(delta)!=3 or any(not 0<=d<=1 for d in delta):
        raise ValueError('three conditional bias radii in[0,1] required')
    if not 0<alpha<1:raise ValueError('failure probability must lie in(0,1)')
    policy([0,0,0],0,mode,grid,bet_grid)
    rows=[list(row) for row in records]
    if any(len(row)!=3 or any(type(x) is not int or x not in (0,1) for x in row) for row in rows):
        raise ValueError('records must be complete three-bit batches')
    counts=[0,0,0];score=F(0);weighted_coefficients=[F(0)]*3
    choices=[]
    for j,row in enumerate(rows):
        decision=policy(counts,j,mode,grid,bet_grid)
        a=decision['coefficients'];lam=decision['bet'];b=decision['support']
        # All coefficients and bets are chosen before this batch is read.
        score += lam*(sum(c*y for c,y in zip(a,row))-b-sum(abs(c)*d for c,d in zip(a,delta)))
        score -= lam*lam*sum(c*c for c in a)/8
        weighted_coefficients=[h+lam*abs(c) for h,c in zip(weighted_coefficients,a)]
        counts=[x+y for x,y in zip(counts,row)]
        choices.append([decision['grid_index'],decision['bet_index']])
    threshold=ec.log_interval(1/alpha)[1]
    denominator=sum(h*contrast.sqrt_upper(t) for h,t in zip(weighted_coefficients,times))
    gap=max(F(0),score-threshold)
    rate=8*(gap/denominator)**2 if denominator else F(0)
    encoded=json.dumps(rows,separators=(',',':')).encode()
    return dict(schema='predictable-contrast-v1', mode=mode, grid=grid, bet_grid=bet_grid,
        batches=len(rows), records=3*len(rows), counts=counts,
        records_sha256=hashlib.sha256(encoded).hexdigest(), decisions=choices,
        durations=times, conditional_radii=delta, alpha=alpha,
        predictable_log_score=score, rejection_threshold_upper=threshold,
        weighted_absolute_coefficients=weighted_coefficients,
        duration_denominator_upper=denominator,
        entropy_production_rate_lower=rate,
        status='positive_certificate' if rate else 'no_positive_certificate',
        guarantee='anytime over completed batches under the declared conditional-mean and stationary physical contract; no assertion of optimal policy')


def verify(candidate,records,**contract):
    try:return candidate==certify(records,**contract)
    except (ValueError,TypeError,KeyError,ZeroDivisionError):return False


def main():
    rows=[[1,1,0]]*100
    cert=certify(rows)
    assert verify(cert,rows)
    root=Path(__file__).resolve().parents[1]
    (root/'results').mkdir(exist_ok=True)
    output=dict(evidence='exact checks and explicitly synthetic example',
        learned=cert,fixed=certify(rows,mode='fixed'),
        empty=certify([]),all_zero_bet_abstains=certify([[0,0,0]])['status']=='no_positive_certificate')
    (root/'results/sequential.json').write_text(json.dumps(contrast.jsonable(output),sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':cert['status'],'example_records':cert['records']}))


if __name__=='__main__':main()
